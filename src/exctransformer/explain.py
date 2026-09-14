"""Public explanation-facing API for the current package."""

from __future__ import annotations

from typing import Any

from exctransformer.SHAP_visualization.workflows import (
    rebuild_batch_summary_workflow,
    run_batch_explanations_workflow,
    run_faithfulness_comparison_workflow,
    run_single_explanation_workflow,
    visualize_saved_explanation_workflow,
)


def explain(*args: Any, **kwargs: Any):
    """Run a single explanation workflow."""

    return run_single_explanation_workflow(*args, **kwargs)


def explain_batch(*args: Any, **kwargs: Any):
    """Run batch explanations over a collection of samples."""

    return run_batch_explanations_workflow(*args, **kwargs)


def visualize_explanation(*args: Any, **kwargs: Any):
    """Visualize a saved explanation payload."""

    return visualize_saved_explanation_workflow(*args, **kwargs)


def compare_faithfulness(*args: Any, **kwargs: Any):
    """Compare SHAP and attention ranking faithfulness on a sample."""

    return run_faithfulness_comparison_workflow(*args, **kwargs)


def rebuild_explanations_summary(*args: Any, **kwargs: Any):
    """Rebuild a batch explanation summary from saved explanation outputs."""

    return rebuild_batch_summary_workflow(*args, **kwargs)


__all__ = [
    "explain",
    "explain_batch",
    "visualize_explanation",
    "compare_faithfulness",
    "rebuild_explanations_summary",
]
