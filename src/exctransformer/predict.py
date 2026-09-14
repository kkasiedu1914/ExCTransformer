# exctransformer version 1.0.0
import os
import copy
import pytorch_lightning as pl
from pathlib import Path
import warnings
import torch

from exctransformer.config import ex
from exctransformer.config import config as config_
from exctransformer.model_modules.module import Module
from exctransformer.datamodule.data_module import DataModule
from exctransformer.utils.configchecks import (
    ConfigurationError, 
    get_num_devices)


def _load_checkpoint_hparams(model_path):
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")
    hparams = checkpoint.get("hyper_parameters", {})
    return hparams if isinstance(hparams, dict) else {}

def predict(dataset_path, model_path,**kwargs):
    """
    Credits to https://github.com/hspark1212/MOFTransformer
     """
    config = copy.deepcopy(config_())
    config.update(_load_checkpoint_hparams(model_path))
    for key in kwargs.keys():
        if key not in config:
            raise ConfigurationError(f'{key} is not in configuration.')

    config.update(kwargs)

    config["dataset_path"] = dataset_path
    config['load_path'] = model_path
    config['predict_set'] = True
    
    main(config)


@ex.automain
def main(_config):
    config = copy.deepcopy(_config)


    os.makedirs(config["log_dir"], exist_ok=True)
    pl.seed_everything(config['seed'])

    num_device = get_num_devices(config)
    num_nodes = config['num_nodes']
    if num_nodes > 1:
        warnings.warn(f"function <predict> only support 1 devices. change num_nodes {num_nodes} -> 1")
        config['num_nodes'] = 1
    if num_device > 1:
        warnings.warn(f"function <predict> only support 1 devices. change num_devices {num_device} -> 1")
        config['devices'] = 1
    
 # TODO: Include some function to to test that all major configurations are properly set before initializing the model  
    model = Module(config)
    datamodule = DataModule(config)
    model.eval()

  # TODO: In future version include a pl strategy for distributed training

    trainer = pl.Trainer(
        accelerator=config['accelerator'],
        devices=config["device"],
        num_nodes=config["num_nodes"],
        precision=config["precision"],
        benchmark=True,
        max_epochs=1,
        log_every_n_steps=0,
        deterministic=True,
        logger=False,
    )

    trainer.predict(model, datamodule)
 
