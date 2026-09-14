"""Top-level package exports for ExCTransformer."""

from __future__ import annotations

import importlib
import os

from exctransformer._version import __version__

__root_dir__ = os.path.dirname(__file__)

__all__ = [
    "__version__",
    "__root_dir__",
    "train",
    "test",
    "predict",
    "explain",
    "explain_batch",
    "visualize_explanation",
    "compare_faithfulness",
    "rebuild_explanations_summary",
    "datamodule",
    "model_modules",
    "utils",
]

_LAZY_EXPORTS = {
    "train": ("exctransformer.train", "train"),
    "test": ("exctransformer.test", "test"),
    "predict": ("exctransformer.predict", "predict"),
    "explain": ("exctransformer.explain", "explain"),
    "explain_batch": ("exctransformer.explain", "explain_batch"),
    "visualize_explanation": ("exctransformer.explain", "visualize_explanation"),
    "compare_faithfulness": ("exctransformer.explain", "compare_faithfulness"),
    "rebuild_explanations_summary": ("exctransformer.explain", "rebuild_explanations_summary"),
    "datamodule": ("exctransformer.datamodule", None),
    "model_modules": ("exctransformer.model_modules", None),
    "utils": ("exctransformer.utils", None),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = importlib.import_module(module_name)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value
