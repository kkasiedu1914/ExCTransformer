# exctransformer version 1.0.0
import os
import copy
from pathlib import Path
import shutil
import warnings
import torch

from typing import Any, Dict, Mapping, Optional

import pytorch_lightning as pl

from exctransformer.config import ex
from exctransformer.config import config as build_base_config
from exctransformer.model_modules.module import Module
from exctransformer.datamodule.data_module import DataModule
from exctransformer.utils.configchecks import (
    ConfigurationError, 
    get_num_devices)
from exctransformer.utils.configchecks import ConfigurationError


warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)


def _as_float(value: Any) -> Optional[float]:
    """Safely convert scalar-like objects (e.g. tensors) to Python floats."""

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
    """Log final metrics to Sacred, if a run is active."""

    run = ex.current_run
    if run is None:
        return
    for key, value in metrics.items():
        scalar = _as_float(value)
        if scalar is not None:
            run.log_scalar(key, scalar)


def _copy_checkpoint(src: Optional[Path], dst: Path) -> None:
    """Copy a checkpoint to ``dst`` when available."""

    if src is None:
        return
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if src.resolve() == dst.resolve():
            return
    except FileNotFoundError:
        # ``resolve`` can fail on some network filesystems; fall back to copy.
        pass
    shutil.copy(src, dst)


def train(dataset_path, log_dir="logs/", test_only=False, **kwargs):
    '''
    Credits to  https://github.com/hspark1212/MOFTransformer
    '''
    
    config = copy.deepcopy(build_base_config())
    for key in kwargs.keys():
        if key not in config:
            raise ConfigurationError(f"{key} is not in configuration.")

    config.update(kwargs)
    config["dataset_path"] = dataset_path
    config["log_dir"] = log_dir
    config["test_only"] = test_only  # Add test_only to config

    return main(config)


@ex.automain
def main(_config):
    config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    exp_name = f"{config['exp_name']}"

    # -- Enable Tensor Core matmul precision if requested --
    precision_mode = config.get("set_float32_matmul_precision")
    if precision_mode in ("high", "medium", "highest"):
        torch.set_float32_matmul_precision(precision_mode)

    model = Module(config)
    datamodule = DataModule(config)

    # TODO: Include some function to to test that all major configurations are properly set before initializing the model
   
    os.makedirs(config["log_dir"], exist_ok=True)

    # Update experiment name for test_only
    if config["test_only"]:
        name = (
            f"test_{exp_name}_seed{config['seed']}_from_"
            f"{str(config['load_path']).split('/')[-1][:-5]}"
        )
    else:
        name = (
            f"{exp_name}_seed{config['seed']}_from_"
            f"{str(config['load_path']).split('/')[-1][:-5]}"
        )

    logger = pl.loggers.TensorBoardLogger(
        config["log_dir"],
        name=name,
         version=config["load_version"] if config["load_version"] is not None else None,
    )

    checkpoint_dir = Path(logger.log_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_device = get_num_devices(config)
    print("num_device", num_device)

    # Gradient accumulation logic
    if num_device == 0:
        accumulate_grad_batches = config["batch_size_grad"] // (
            config["per_gpu_batchsize"] * config["num_nodes"]
        )
    else:
        accumulate_grad_batches = config["batch_size_grad"] // (
            config["per_gpu_batchsize"] * num_device * config["num_nodes"]
        )

    max_steps = config["max_steps"] if config["max_steps"] is not None else None


    # TODO: In future version include a pl strategy for distributed training

    trainer = pl.Trainer(
        accelerator=config["accelerator"],
        devices=config["device"],
        num_nodes=config["num_nodes"],
        precision=config["precision"],
        benchmark=True,
        max_epochs=config["max_epochs"],
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=config["log_every_n_steps"],
        val_check_interval=config["val_check_interval"],
        deterministic=True,
    )

    metrics: Dict[str, Any] = {}

    if not config["test_only"]:
        interrupted = False
        try:
            trainer.fit(model, datamodule, ckpt_path=config["resume_from"])
        except SystemExit as exc:  # Lightning uses ``sys.exit`` on interruption
            if getattr(exc, "code", 1) not in (0, 1):
                raise
            interrupted = True
            print(f"Trainer exited early (code {exc.code}); treating as interrupted run.")
        except KeyboardInterrupt:
            interrupted = True
            print("Training interrupted by user; attempting to save last checkpoint.")

        best_ckpt = Path(checkpoint_callback.best_model_path) if checkpoint_callback.best_model_path else None
        last_ckpt = Path(checkpoint_callback.last_model_path) if checkpoint_callback.last_model_path else None

        if interrupted and last_ckpt is None:
            # ``ModelCheckpoint`` did not manage to persist the last checkpoint; save manually.
            manual_last = checkpoint_dir / "last.ckpt"
            trainer.save_checkpoint(str(manual_last))
            last_ckpt = manual_last

        if best_ckpt is None:
            print(
                "No best checkpoint produced. Ensure that the validation metric "
                "'val/the_metric' is logged during training."
            )
        _copy_checkpoint(best_ckpt, checkpoint_dir / "best.ckpt")
        _copy_checkpoint(last_ckpt, checkpoint_dir / "last.ckpt")

        if not interrupted:
            if checkpoint_callback.best_model_path:
                ckpt_identifier: Optional[str] = "best"
            elif checkpoint_callback.last_model_path:
                ckpt_identifier = "last"
            else:
                ckpt_identifier = None

            if ckpt_identifier is None:
                print("No checkpoint available for evaluation; skipping test step.")
            else:
                test_results = trainer.test(model, datamodule, ckpt_path=ckpt_identifier)
                if test_results:
                    metrics.update(test_results[0])
        else:
            print("Skipping evaluation because training was interrupted before completion.")

        callback_metrics = {k: _as_float(v) for k, v in trainer.callback_metrics.items()}
        metrics.update({k: v for k, v in callback_metrics.items() if v is not None})
    else:
        test_results = trainer.test(model, datamodule)
        if test_results:
            metrics.update(test_results[0])

    serialisable: Dict[str, float] = {}
    for key, value in metrics.items():
        scalar = _as_float(value)
        if scalar is not None:
            serialisable[key] = scalar
    _log_metrics(serialisable)
    return serialisable
