import os
from exctransformer import __root_dir__

def _default_model_path(*filenames):
    for filename in filenames:
        candidate = os.path.join(__root_dir__, "pretrained_models", filename)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(__root_dir__, "pretrained_models", filenames[0])


DEFAULT_EXCTRANSFORMER_PATH = _default_model_path("exctransformer.ckpt", "gscgcnntransformer.ckpt")
DEFAULT_GSCGCNNTRANSFORMER_PATH = DEFAULT_EXCTRANSFORMER_PATH
DEFAULT_MOFTRANSFORMER_PATH = _default_model_path("moftransformer.ckpt")
DEFAULT_PMTRANSFORMER_PATH = _default_model_path("pmtransformer.ckpt")
