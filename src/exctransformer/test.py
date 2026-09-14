# exctransformer version 1.0.0
import os
import copy
import pytorch_lightning as pl
import warnings
from typing import Any, Dict, Mapping, Optional
import torch

from exctransformer.config import ex
from exctransformer.config import config as build_base_config
from exctransformer.model_modules.module import Module
from exctransformer.datamodule.data_module import DataModule
from exctransformer.utils.configchecks import ConfigurationError

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)


def _load_checkpoint_hparams(model_path: str) -> Dict[str, Any]:
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")
    hparams = checkpoint.get("hyper_parameters", {})
    return hparams if isinstance(hparams, dict) else {}


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            return None
    return None


def _log_metrics(metrics: Mapping[str, Any]) -> None:
    run = ex.current_run
    if run is None:
        return
    for key, value in metrics.items():
        scalar = _as_float(value)
        if scalar is not None:
            run.log_scalar(key, scalar)



def test(dataset_path, model_path, test_only=False, **kwargs):
    '''
    Credits to  https://github.com/hspark1212/MOFTransformer
    '''
    
    config = copy.deepcopy(build_base_config())
    config.update(_load_checkpoint_hparams(model_path))
    for key in kwargs.keys():
        if key not in config:
            raise ConfigurationError(f"{key} is not in configuration.")

    config.update(kwargs)
    config["dataset_path"] = dataset_path
    config["load_path"] = model_path
    config["test_only"] = test_only  # Add test_only to config

    return main(config)

@ex.automain
def main(_config):
    config = copy.deepcopy(_config)

    config['test_only'] = True

    pl.seed_everything(config["seed"])

 # TODO: Include some function to to test that all major configurations are properly set before initializing the model  

    model = Module(config)
    datamodule = DataModule(config)

    model.eval()

    # TODO: In future version include a pl strategy for distributed training

    trainer = pl.Trainer(
        accelerator=config["accelerator"],
        devices=config["device"],
        num_nodes=config["num_nodes"],
        precision=config["precision"],
        benchmark=True,
        max_epochs=1,
        logger=False,
        log_every_n_steps=0,
        deterministic=True,
    )

    results = trainer.test(model, datamodule=datamodule)
    metrics = results[0] if results else {}
    serialisable: Dict[str, float] = {}
    for key, value in metrics.items():
        scalar = _as_float(value)
        if scalar is not None:
            serialisable[key] = scalar
    _log_metrics(serialisable)
    return serialisable


    
