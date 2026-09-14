import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import mean_absolute_error

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def eval_regression(pl_module, batch, normalizer):
    """
    Evaluate regression or gather predictions when in predict mode.
    In predict mode, only returns ids, cls_feats, and decoded logits.
    """
    # Perform inference
    inference = pl_module.infer(batch)
    logits = pl_module.regression_head(inference["cls_feats"])  # [B, n_targets]

    predict_mode = getattr(getattr(pl_module, 'trainer', None), 'predicting', False)
    
    logits = logits.squeeze(-1)
    logits_decoded = normalizer.decode(logits)

    if predict_mode and pl_module.config["no_labels"]:
        # Only return ids, features, and predictions
        return {
            "cif_id": inference["cif_id"],
            "cls_feats": inference["cls_feats"],
            "regression_logits": logits_decoded,
        }

    # Otherwise, compute loss and metrics as usual
    raw_t = batch["target"]
    if isinstance(raw_t, torch.Tensor):
        labels = raw_t.clone().detach().to(logits.device).float()
    else:
        labels = torch.tensor(raw_t, dtype=torch.float32, device=logits.device)
    if labels.numel() == logits.numel():
        labels = labels.reshape_as(logits)

    labels_encoded = normalizer.encode(labels)
    loss = F.mse_loss(logits, labels_encoded)

    labels_decoded = normalizer.decode(labels_encoded)

    # Determine phase
    phase = "train" if pl_module.training else "val"
    loss_metric = getattr(pl_module, f"{phase}_regression_loss")(loss)
    mae_metric = getattr(pl_module, f"{phase}_regression_mae")(mean_absolute_error(logits_decoded, labels_decoded))

    
    if pl_module.write_log:
        pl_module.log(f"regression/{phase}/loss", loss_metric, sync_dist=True)
        pl_module.log(f"regression/{phase}/mae", mae_metric, sync_dist=True)

    return {
        "cif_id": inference["cif_id"],
        "cls_feats": inference["cls_feats"],
        "regression_loss": loss,
        "regression_logits": logits_decoded,
        "regression_labels": labels_decoded,
    }


def eval_classification(pl_module, batch):
 
    inference = pl_module.infer(batch)
    logits, is_binary = pl_module.classification_head(inference["cls_feats"])

    # Detect predict stage
    predict_mode = getattr(getattr(pl_module, 'trainer', None), 'predicting', False)

    if predict_mode and pl_module.config["no_labels"]:
        if is_binary:
            logits = logits.squeeze(-1)
        return {
            "cif_id": inference["cif_id"],
            "cls_feats": inference["cls_feats"],
            "classification_logits": logits,
        }

    raw_t = batch["target"]
    if isinstance(raw_t, torch.Tensor):
        labels = raw_t.clone().detach().to(logits.device).float()
    else:
        labels = torch.tensor(raw_t, dtype=torch.float32, device=logits.device)

    # Compute loss
    if is_binary:
        logits = logits.squeeze(-1)
        if labels.numel() == logits.numel():
            labels = labels.reshape_as(logits)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
    else:
        labels = labels.reshape(-1).long()
        loss = F.cross_entropy(logits, labels)

    # Determine phase
    phase = "train" if pl_module.training else "val"
    # Update metrics
    loss_metric = getattr(pl_module, f"{phase}_classification_loss")(loss)
    acc_metric = getattr(pl_module, f"{phase}_classification_accuracy")(logits, labels)

    if pl_module.write_log:
        pl_module.log(f"classification/{phase}/loss", loss_metric, sync_dist=True)
        pl_module.log(f"classification/{phase}/accuracy", acc_metric, sync_dist=True)

    return {
        "cif_id": inference["cif_id"],
        "cls_feats": inference["cls_feats"],
        "classification_loss": loss,
        "classification_logits": logits,
        "classification_labels": labels,
    }
