# exctransformer version 1.0

from exctransformer.pretrained_models import ( 
                                 DEFAULT_EXCTRANSFORMER_PATH,
                                 DEFAULT_GSCGCNNTRANSFORMER_PATH,
                                 DEFAULT_MOFTRANSFORMER_PATH,
                                 DEFAULT_PMTRANSFORMER_PATH 
                                 )
from pytorch_lightning.trainer.connectors.accelerator_connector import _AcceleratorConnector as AC


class ConfigurationError(Exception):
    pass

def _loss_names(d):
    ret = {
        "regression": 0,  # regression
        "classification": 0,  # classification
    }
    ret.update(d)
    return ret


def _get_auto_device(_config):
    accelerator = AC(accelerator=_config["accelerator"]).accelerator
    devices = accelerator.auto_device_count()

    return devices

def get_num_devices(_config):
    if isinstance(devices := _config["device"], list):
        devices = len(devices)
    elif isinstance(devices, int):
        pass
    elif devices == "auto" or devices is None:
        devices = _get_auto_device(_config)
    else:
        raise ConfigurationError(
            f'devices must be int, list, and "auto", not {devices}'
        )
    return devices

def set_model_load_path(path):
    if path == "exctransformer":
        return DEFAULT_EXCTRANSFORMER_PATH
    if path == "gscgcnntransformer":
        return DEFAULT_GSCGCNNTRANSFORMER_PATH
    if path == "moftransformer":
        return DEFAULT_MOFTRANSFORMER_PATH
    if path == 'pmtransformer':
        return DEFAULT_PMTRANSFORMER_PATH
    elif not path:
        return ""
    elif str(path)[-4:] == "ckpt":
        return path
    else:
        raise ConfigurationError(
            f"path must be 'exctransformer', 'gscgcnntransformer' (legacy alias), 'moftransformer', pmtransformer, None, or *.ckpt, not {path}"
        )
    
