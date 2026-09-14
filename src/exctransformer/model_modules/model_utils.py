import h5py
import os
import zipfile
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from exctransformer.model_modules.metric_tracker import Scalar, Accuracy
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
)

'''
Metric tracking and schedulers were obtained from the original moftransformer implementation 
    credists to https://github.com/hspark1212/MOFTransformer
'''

def set_metrics(pl_module):
    for split in ["train", "val"]:
        for k, v in pl_module.config["loss_names"].items():
            if v < 1:
                continue
            if k == "regression":
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_mae", Scalar())
            else:
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())


def set_task(pl_module):
    pl_module.current_tasks = [
        k for k, v in pl_module.config["loss_names"].items() if v >= 1
    ]
    return


def epoch_wrapup(pl_module):
    phase = "train" if pl_module.training else "val"

    the_metric = 0

    for loss_name, v in pl_module.config["loss_names"].items():
        if v < 1:
            continue

        if loss_name == "regression":
            # mse loss
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
                batch_size=pl_module.config["per_gpu_batchsize"],
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            # mae loss
            value = getattr(pl_module, f"{phase}_{loss_name}_mae").compute()
            pl_module.log(
                f"{loss_name}/{phase}/mae_epoch",
                value,
                batch_size=pl_module.config["per_gpu_batchsize"],
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{loss_name}_mae").reset()

            value = -value
        else:
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(
                f"{loss_name}/{phase}/accuracy_epoch",
                value,
                batch_size=pl_module.config["per_gpu_batchsize"],
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
                batch_size=pl_module.config["per_gpu_batchsize"],
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        the_metric += value

    pl_module.log(f"{phase}/the_metric", the_metric, sync_dist=True)



def init_weights(module):
    '''
    Weight initialization
    Based on the original moftransformer implementation 
    credists to https://github.com/hspark1212/MOFTransformer
    '''
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def set_schedule(pl_module):
    """
    Sets up an optimizer and learning rate scheduler for the 
    Crystal_GraphSAGE_Transformer_encoder architecture.
    Uses different learning rates and weight decay settings for:
      - Regular parameters
      - Biases and normalization parameters (no weight decay)
      - Head parameters (ie, the final output layer ()'fc_out') with a learning rate multiplier
    """
    # Retrieve configuration values
    lr = pl_module.config["learning_rate"]
    wd = pl_module.config["weight_decay"]

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    head_names = ["regression_head", "classification_head"]
    lr_mult = pl_module.config["lr_mult"]
    end_lr = pl_module.config["end_lr"]
    decay_power = pl_module.config["decay_power"]
    optim_type = pl_module.config["optim_type"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)  # not within no_decay
                and not any(bb in n for bb in head_names)  # not within head_names
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)  # within no_decay
                and not any(bb in n for bb in head_names)  # not within head_names
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)  # not within no_decay
                and any(bb in n for bb in head_names)  # within head_names
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
                # within no_decay and head_names
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult,
        },
    ]

    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

    if pl_module.trainer.max_steps == -1:
        max_steps = pl_module.trainer.estimated_stepping_batches
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = pl_module.config["warmup_steps"]
    if isinstance(pl_module.config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)

    print(
        f"max_epochs: {pl_module.trainer.max_epochs} | max_steps: {max_steps} | warmup_steps : {warmup_steps} "
        f"| weight_decay : {wd} | decay_power : {decay_power}"
    )

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    elif decay_power == "constant":
        scheduler = get_constant_schedule(
            optimizer,
        )
    elif decay_power == "constant_with_warmup":
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )


def get_feature_lengths_from_dataset(config):
    """
    Helper function to obtain the atom feature length and neighbor feature length
    from the dataset specified in the config dictionary.

    Parameters:
    -----------
    config : dict
        Configuration dictionary containing the dataset_path key.

    Returns:
    --------
    atom_fea_len : int
        Length of the atom features.
    nbr_fea_len : int
        Length of the neighbor features.

    Raises:
    -------
    FileNotFoundError:
        If the dataset_path is not found or no valid HDF5 files are present.
    ValueError:
        If the required datasets are not found in the HDF5 file.
    """
    dataset_path = config.get("dataset_path", "")
    if not dataset_path:
        raise ValueError("The 'dataset_path' key is missing or empty in the config dictionary.")

    # Check if the dataset path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist.")

    # Look for the first valid HDF5 file in the dataset path
    h5_files = [f for f in os.listdir(dataset_path) if f.endswith('.h5')]
    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found in the dataset path '{dataset_path}'.")

    # Open the first HDF5 file and extract feature lengths
    first_h5_file = os.path.join(dataset_path, h5_files[0])
    try:
        with h5py.File(first_h5_file, 'r') as f:
            if 'graph' not in f:
                raise ValueError(f"Group 'graph' not found in the HDF5 file '{first_h5_file}'.")
            grp = f['graph']
            if 'atom_fea' not in grp or 'edge_fea' not in grp:
                raise ValueError(f"Required datasets 'atom_fea' or 'nbr_fea' not found in the HDF5 file '{first_h5_file}'.")
            
            # Get the feature lengths
            atom_fea_len = grp['atom_fea'].shape[1]  # Assuming shape is (N, atom_fea_len)
            nbr_fea_len = grp['edge_fea'].shape[2]  # Assuming shape is (N, M, bond_fea_len)
            graph_only_mode = bool(config.get("graph_only_mode", False))
            if (not graph_only_mode) and 'additional_features' in grp:
                if not torch.is_tensor(grp['additional_features']):
                    add_fea = torch.tensor(grp['additional_features'], dtype=torch.float32)
                else:
                    add_fea = grp['additional_features']
            else:
                add_fea = torch.tensor([])  # Optional additional features
    except KeyError as e:
        raise ValueError(f"Missing required dataset in the HDF5 file: {e}")
    except Exception as e:
        raise RuntimeError(f"Error reading HDF5 file '{first_h5_file}': {e}")

    return atom_fea_len, nbr_fea_len, add_fea


class Normalizer(object):
    """
    Normalize for regression.
    """

    def __init__(self, mean=None, std=None, device='cpu'):
        self.device = device
        self.mean = self._to_tensor(mean, device) if mean is not None else None
        self.std = self._to_tensor(std, device) if std is not None else None

    def _to_tensor(self, data, device):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(data, torch.Tensor):
            return data.to(device)
        if isinstance(data, list):
            return torch.tensor(data, dtype=torch.float32, device=device)
        # Support numeric scalars (float/int/np.number).
        return torch.tensor(data, dtype=torch.float32, device=device)

    def encode(self, tensor):
        if self.mean is not None and self.std is not None:
            return (tensor - self.mean) / self.std
        return tensor

    def decode(self, tensor):
        if self.mean is not None and self.std is not None:
            return tensor * self.std + self.mean
        return tensor



def load_pretrained_weights(path, model, zip=False, strict=False):
    """
    Load pretrained weights with comprehensive layer matching.
    
    Args:
        path: Path to .ckpt file
        model: Your custom model
        zip: Whether the ckpt is zipped
        strict: Whether to require exact parameter matching
    """
    # Load checkpoint
    if zip:
        with zipfile.ZipFile(path, 'r') as zip_ref:
            ckpt_filename = next((name for name in ['moftransformer.ckpt', 'pmtransformer.ckpt'] 
                               if name in zip_ref.namelist()), None)
            if not ckpt_filename:
                raise FileNotFoundError("No valid .ckpt found in zip")
            
            with zip_ref.open(ckpt_filename) as ckpt_file:
                buffer = io.BytesIO(ckpt_file.read())
                checkpoint = torch.load(buffer, map_location='cpu')
    else:
        checkpoint = torch.load(path, map_location='cpu')

    state_dict = checkpoint.get("state_dict", checkpoint)
    
    # 1. Load transformer blocks
    for i, block in enumerate(model.blocks):
        # Attention weights
        attn_prefix = f"transformer.blocks.{i}.attn."
        block.attn.qkv.weight.data.copy_(state_dict[attn_prefix + "qkv.weight"])
        block.attn.qkv.bias.data.copy_(state_dict[attn_prefix + "qkv.bias"])
        block.attn.proj.weight.data.copy_(state_dict[attn_prefix + "proj.weight"])
        block.attn.proj.bias.data.copy_(state_dict[attn_prefix + "proj.bias"])

        # Normalization layers
        block_prefix = f"transformer.blocks.{i}."
        for norm in ["norm1", "norm2"]:  # Per-block norms
            if f"{block_prefix}{norm}.weight" in state_dict:
                getattr(block, norm).weight.data.copy_(state_dict[f"{block_prefix}{norm}.weight"])
                getattr(block, norm).bias.data.copy_(state_dict[f"{block_prefix}{norm}.bias"])

        # MLP layers
        mlp_prefix = f"{block_prefix}mlp."
        block.mlp.fc1.weight.data.copy_(state_dict[mlp_prefix + "fc1.weight"])
        block.mlp.fc1.bias.data.copy_(state_dict[mlp_prefix + "fc1.bias"])
        block.mlp.fc2.weight.data.copy_(state_dict[mlp_prefix + "fc2.weight"])
        block.mlp.fc2.bias.data.copy_(state_dict[mlp_prefix + "fc2.bias"])

    # 2. Load final norm layer if exists
    if "transformer.norm.weight" in state_dict and hasattr(model, "norm"):
        model.norm.weight.data.copy_(state_dict["transformer.norm.weight"])
        model.norm.bias.data.copy_(state_dict["transformer.norm.bias"])
        print("Final norm layer loaded")

    # 3. Optional: Load other components (embeddings, heads, etc.)
    if not strict:
        print("Non-strict mode: Some weights may not be loaded")
    else:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")

    print("Weights successfully loaded")
