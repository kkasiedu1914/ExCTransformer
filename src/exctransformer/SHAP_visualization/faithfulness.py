"""
Faithfulness tests for GraphSVX explanations on MOF predictions.

These utilities perform progressive occlusion (embedding neutralization or
mask-only ablation) on top-ranked atoms/features and compare the prediction
drop against random occlusion. This is a functional faithfulness check under
the GraphSVX perturbation operator, not a physical vacancy simulation.
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from exctransformer.SHAP_visualization import utils

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib optional for headless usage
    plt = None


def _to_1d_numpy(values: torch.Tensor | np.ndarray | Sequence[float]) -> np.ndarray:
    if torch.is_tensor(values):
        arr = values.detach().cpu().float().view(-1).numpy()
    else:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return arr


def _resolve_shap_vector(
    shap_values: torch.Tensor | np.ndarray | Sequence[float],
    num_atoms: int,
    num_add_tokens: int,
) -> np.ndarray:
    """
    Resolve SHAP values into a 1D numpy vector aligned with [atoms + add_tokens].
    """
    shap_vec = _to_1d_numpy(shap_values)
    expected = num_atoms + num_add_tokens
    if shap_vec.size != expected:
        raise ValueError(
            f"SHAP length mismatch: got {shap_vec.size}, expected {expected} "
            f"(num_atoms={num_atoms}, num_add_tokens={num_add_tokens})."
        )
    return shap_vec


def _neutral_baseline(
    x: torch.Tensor,
    num_atoms: int,
    mode: str = "mean",
) -> torch.Tensor:
    """
    Compute the neutralization baseline used by GraphSVX.
    - mean: mean over atom embeddings (Expectation)
    - zero: zeros (Null)
    """
    mode = str(mode or "mean").lower()
    if mode in {"zero", "zeros", "null", "absolute"}:
        return torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
    if mode in {"mean", "expectation", "avg", "average"}:
        return x[:num_atoms].mean(dim=0)
    raise ValueError(f"Unknown neutralize mode: {mode}")


def _apply_occlusion(
    base_data: Dict,
    excluded_nodes: Sequence[int],
    num_atoms: int,
    neutralize_mode: str = "mean",
    use_mask_only: bool = False,
    zero_adj: bool = True,
) -> Dict:
    """
    Create a perturbed data dict using the GraphSVX perturbation operator.
    This zeroes adjacency rows/cols (if dense), and either neutralizes embeddings
    or passes excluded nodes for mask-only ablation.
    """
    data = {}
    for key, value in base_data.items():
        data[key] = value

    x = data["x"].clone()
    edge_index = data.get("edge_index", None)

    # Support attention masking as a neutralize mode.
    # If neutralize_mode is "mask"/"attention", force mask-only ablation.
    mode = str(neutralize_mode or "mean").lower()
    if mode in {"mask", "mask_only", "attention", "attention_mask"}:
        use_mask_only = True

    excluded_nodes = [int(i) for i in excluded_nodes]
    if excluded_nodes:
        if zero_adj and torch.is_tensor(edge_index) and edge_index.ndim == 2:
            if edge_index.shape[0] == edge_index.shape[1]:
                edge_index = edge_index.clone()
                edge_index[excluded_nodes, :] = 0
                edge_index[:, excluded_nodes] = 0
        if not use_mask_only:
            baseline = _neutral_baseline(x, num_atoms=num_atoms, mode=neutralize_mode)
            x[excluded_nodes] = baseline

    data["x"] = x
    if edge_index is not None:
        data["edge_index"] = edge_index
    if use_mask_only:
        data["excluded_nodes"] = torch.as_tensor(
            excluded_nodes, dtype=torch.long, device=x.device
        )
        data["use_mask_only"] = True
    return data


def _predict(
    model,
    data: Dict,
    pred_idx: int,
    configs: Dict,
) -> float:
    with torch.no_grad():
        pred = model.infer_out(data, pred_idx, configs=configs)
    if torch.is_tensor(pred):
        pred = pred.detach().cpu()
        if pred.numel() == 1:
            return float(pred.view(-1)[0])
        return float(pred.mean().item())
    return float(pred)


def _build_ranked_indices(
    shap_vec: np.ndarray,
    num_atoms: int,
    num_add_tokens: int,
    include_additional_tokens: bool,
    exclude_void_fraction: bool,
    void_fraction_index: Optional[int],
    exclude_indices: Optional[Sequence[int]] = None,
) -> List[int]:
    total = num_atoms + num_add_tokens
    candidates = list(range(total))
    if not include_additional_tokens and num_add_tokens > 0:
        candidates = list(range(num_atoms))
    if exclude_void_fraction and num_add_tokens > 0:
        vf_index = void_fraction_index
        if vf_index is None:
            vf_index = num_atoms  # default: first additional token
        candidates = [i for i in candidates if i != vf_index]
    if exclude_indices:
        exclude_set = {int(i) for i in exclude_indices}
        candidates = [i for i in candidates if i not in exclude_set]
    ranked = sorted(candidates, key=lambda i: abs(shap_vec[i]), reverse=True)
    return ranked


def progressive_occlusion_curve(
    model,
    base_data: Dict,
    shap_values: torch.Tensor | np.ndarray | Sequence[float],
    num_atoms: int,
    num_add_tokens: int,
    pred_idx: int,
    configs: Dict,
    k_values: Sequence[int],
    num_random: int = 10,
    random_seed: Optional[int] = 7,
    neutralize_mode: str = "zero",
    use_mask_only: bool = False,
    include_additional_tokens: bool = True,
    exclude_void_fraction: bool = False,
    void_fraction_index: Optional[int] = None,
    exclude_indices: Optional[Sequence[int]] = None,
    zero_adj: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute progressive occlusion curves for top-k and random-k perturbations.
    Returns a dict with curves and metadata.
    """
    shap_vec = _resolve_shap_vector(shap_values, num_atoms, num_add_tokens)
    ranked = _build_ranked_indices(
        shap_vec,
        num_atoms=num_atoms,
        num_add_tokens=num_add_tokens,
        include_additional_tokens=include_additional_tokens,
        exclude_void_fraction=exclude_void_fraction,
        void_fraction_index=void_fraction_index,
        exclude_indices=exclude_indices,
    )
    if not ranked:
        raise ValueError("No candidate indices remain after filtering.")

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    k_values = [int(k) for k in k_values if int(k) > 0]
    max_k = min(len(ranked), max(k_values))
    k_values = [k for k in k_values if k <= max_k]
    if not k_values:
        raise ValueError("No valid k_values after clipping to candidate size.")

    base_pred = _predict(model, base_data, pred_idx, configs)

    top_drops = []
    rand_drops_mean = []
    rand_drops_std = []
    per_k_random = []

    for k in k_values:
        top_nodes = ranked[:k]
        data_top = _apply_occlusion(
            base_data,
            top_nodes,
            num_atoms=num_atoms,
            neutralize_mode=neutralize_mode,
            use_mask_only=use_mask_only,
            zero_adj=zero_adj,
        )
        pred_top = _predict(model, data_top, pred_idx, configs)
        top_drops.append(base_pred - pred_top)

        random_drops = []
        for _ in range(num_random):
            rand_nodes = random.sample(ranked, k=k)
            data_rand = _apply_occlusion(
                base_data,
                rand_nodes,
                num_atoms=num_atoms,
                neutralize_mode=neutralize_mode,
                use_mask_only=use_mask_only,
                zero_adj=zero_adj,
            )
            pred_rand = _predict(model, data_rand, pred_idx, configs)
            random_drops.append(base_pred - pred_rand)
        per_k_random.append(random_drops)
        rand_drops_mean.append(float(np.mean(random_drops)))
        rand_drops_std.append(float(np.std(random_drops)))

    return {
        "k": np.asarray(k_values, dtype=int),
        "base_pred": float(base_pred),
        "top_drop": np.asarray(top_drops, dtype=np.float32),
        "random_drop_mean": np.asarray(rand_drops_mean, dtype=np.float32),
        "random_drop_std": np.asarray(rand_drops_std, dtype=np.float32),
        "random_drop_samples": per_k_random,
        "ranked_indices": ranked,
        "shap_values": shap_vec,
        "pred_idx": int(pred_idx),
        "neutralize_mode": neutralize_mode,
        "use_mask_only": bool(use_mask_only),
        "include_additional_tokens": bool(include_additional_tokens),
        "exclude_void_fraction": bool(exclude_void_fraction),
        "void_fraction_index": void_fraction_index,
    }


def _attention_rollout_scores(
    model,
    base_data: Dict,
    configs: Dict,
    num_atoms: int,
    num_add_tokens: int,
    batch_idx: int = 0,
    sep_tokens: int = 1,
) -> np.ndarray:
    """
    Compute attention rollout scores aligned to [atoms + add_tokens].
    """
    _, attn_weights = model.infer_out(
        base_data,
        base_data["pred_index"],
        configs=configs,
        return_attn_weights=True,
    )
    attn_vec = utils.get_heatmap(
        attns=attn_weights,
        batch_idx=batch_idx,
        graph_len=num_atoms,
        skip_cls=True,
        sep_tokens=sep_tokens,
        add_feat_tokens=num_add_tokens,
    )
    return _to_1d_numpy(attn_vec)


def compare_topk_shap_vs_attention(
    model,
    base_data: Dict,
    shap_values: torch.Tensor | np.ndarray | Sequence[float],
    num_atoms: int,
    num_add_tokens: int,
    pred_idx: int,
    configs: Dict,
    k: int = 5,
    neutralize_mode: str = "zero",
    use_mask_only: bool = False,
    include_additional_tokens: bool = True,
    zero_adj: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Side-by-side occlusion: top-k by SHAP vs top-k by attention rollout.

    This function first removes top-k features by |SHAP|, then removes top-k
    features by attention rollout scores. It reports prediction drops for both.
    """
    shap_vec = _resolve_shap_vector(shap_values, num_atoms, num_add_tokens)
    shap_ranked = _build_ranked_indices(
        shap_vec,
        num_atoms=num_atoms,
        num_add_tokens=num_add_tokens,
        include_additional_tokens=include_additional_tokens,
        exclude_void_fraction=False,
        void_fraction_index=None,
    )
    attn_scores = _attention_rollout_scores(
        model=model,
        base_data=base_data,
        configs=configs,
        num_atoms=num_atoms,
        num_add_tokens=num_add_tokens,
    )
    attn_ranked = _build_ranked_indices(
        attn_scores,
        num_atoms=num_atoms,
        num_add_tokens=num_add_tokens,
        include_additional_tokens=include_additional_tokens,
        exclude_void_fraction=False,
        void_fraction_index=None,
    )

    k = min(k, len(shap_ranked), len(attn_ranked))
    shap_topk = shap_ranked[:k]
    attn_topk = attn_ranked[:k]

    base_pred = _predict(model, base_data, pred_idx, configs)

    data_shap = _apply_occlusion(
        base_data,
        shap_topk,
        num_atoms=num_atoms,
        neutralize_mode=neutralize_mode,
        use_mask_only=use_mask_only,
        zero_adj=zero_adj,
    )
    pred_shap = _predict(model, data_shap, pred_idx, configs)

    data_attn = _apply_occlusion(
        base_data,
        attn_topk,
        num_atoms=num_atoms,
        neutralize_mode=neutralize_mode,
        use_mask_only=use_mask_only,
        zero_adj=zero_adj,
    )
    pred_attn = _predict(model, data_attn, pred_idx, configs)

    return {
        "k": k,
        "base_pred": float(base_pred),
        "shap_drop": float(base_pred - pred_shap),
        "attn_drop": float(base_pred - pred_attn),
        "shap_topk": shap_topk,
        "attn_topk": attn_topk,
        "attn_scores": attn_scores,
    }


def plot_faithfulness_curve(
    result: Dict[str, np.ndarray],
    title: Optional[str] = None,
    label_prefix: Optional[str] = None,
    ax=None,
    show: bool = True,
    save_path: Optional[str] = None,
    label_fontsize: Optional[float] = None,
    title_fontsize: Optional[float] = None,
    tick_fontsize: Optional[float] = None,
    legend_fontsize: Optional[float] = None,
):
    """
    Plot faithfulness curves: top-k vs random-k occlusion.
    Use label_prefix to distinguish multiple gases/targets on a shared axis.
    """
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
    else:
        fig = ax.figure

    k = result["k"]
    prefix = f"{label_prefix} " if label_prefix else ""
    ax.plot(k, result["top_drop"], marker="o", label=f"{prefix}Top-k (|phi|)")
    ax.plot(k, result["random_drop_mean"], marker="o", label=f"{prefix}Random-k")
    ax.fill_between(
        k,
        result["random_drop_mean"] - result["random_drop_std"],
        result["random_drop_mean"] + result["random_drop_std"],
        alpha=0.2,
    )
    ax.set_xlabel("Number of neutralized atoms (k)", fontsize=label_fontsize)
    ax.set_ylabel("Prediction drop", fontsize=label_fontsize)
    ax.grid(True, linestyle="--", alpha=0.4)
    if title:
        ax.set_title(title, fontsize=title_fontsize)
    if tick_fontsize is not None:
        ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize)

    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
    if show:
        plt.show()
    return ax


def run_faithfulness_tests(
    visualizer,
    shap_output: Optional[Dict] = None,
    shap_output_path: Optional[str] = None,
    shap_values: Optional[torch.Tensor | np.ndarray | Sequence[float]] = None,
    pred_idx: Optional[int] = None,
    k_values: Optional[Sequence[int]] = None,
    top_k_max: int = 20,
    num_random: int = 10,
    random_seed: Optional[int] = 7,
    neutralize_mode: str = "zero",
    use_mask_only: Optional[bool] = None,
    include_additional_tokens: bool = True,
    exclude_void_fraction: bool = False,
    void_fraction_index: Optional[int] = None,
    exclude_indices: Optional[Sequence[int]] = None,
    zero_adj: bool = True,
    plot: bool = True,
    plot_title: Optional[str] = None,
    save_plot_path: Optional[str] = None,
    run_void_fraction_check: bool = False,
    compare_attention_topk: bool = True,
    attention_topk: int = 5,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Run progressive occlusion faithfulness tests for a SHAPVisualizer instance.

    This uses the GraphSVX perturbation operator (embedding neutralization or
    mask-only ablation) and compares top-k attribution occlusion against
    random-k occlusion. Optionally runs a second curve that excludes the
    void-fraction feature (first additional token).
    """
    payload = visualizer._resolve_shap_output(shap_output, shap_output_path)
    if shap_values is None:
        shap_values = payload["shap_values"] if payload is not None else visualizer.shap_values
    if shap_values is None:
        raise ValueError("No shap_values provided or available on the visualizer.")

    if pred_idx is None:
        pred_idx = int(getattr(visualizer, "pred_idx", 0))

    if use_mask_only is None:
        use_mask_only = False
        if payload is not None:
            use_mask_only = bool(payload.get("explain_config", {}).get("use_mask_only", False))
        elif getattr(visualizer, "explain_config", None):
            use_mask_only = bool(visualizer.explain_config.get("use_mask_only", False))

    num_atoms = int(visualizer.data.get("num_atoms", 0))
    num_add_tokens = int(visualizer.data.get("num_add_tokens", 0))
    if k_values is None:
        max_k = min(top_k_max, max(1, num_atoms + num_add_tokens))
        k_values = list(range(1, max_k + 1))

    result = progressive_occlusion_curve(
        model=visualizer.model,
        base_data=visualizer.data,
        shap_values=shap_values,
        num_atoms=num_atoms,
        num_add_tokens=num_add_tokens,
        pred_idx=pred_idx,
        configs=visualizer.data["config"],
        k_values=k_values,
        num_random=num_random,
        random_seed=random_seed,
        neutralize_mode=neutralize_mode,
        use_mask_only=use_mask_only,
        include_additional_tokens=include_additional_tokens,
        exclude_void_fraction=exclude_void_fraction,
        void_fraction_index=void_fraction_index,
        exclude_indices=exclude_indices,
        zero_adj=zero_adj,
    )

    results = {"primary": result}

    if plot:
        title = plot_title or f"Faithfulness (pred_idx={pred_idx})"
        plot_faithfulness_curve(result, title=title, save_path=save_plot_path)

    if run_void_fraction_check:
        vf_result = progressive_occlusion_curve(
            model=visualizer.model,
            base_data=visualizer.data,
            shap_values=shap_values,
            num_atoms=num_atoms,
            num_add_tokens=num_add_tokens,
            pred_idx=pred_idx,
            configs=visualizer.data["config"],
            k_values=k_values,
            num_random=num_random,
            random_seed=random_seed,
            neutralize_mode=neutralize_mode,
            use_mask_only=use_mask_only,
            include_additional_tokens=include_additional_tokens,
            exclude_void_fraction=True,
            void_fraction_index=void_fraction_index,
            exclude_indices=exclude_indices,
            zero_adj=zero_adj,
        )
        results["void_fraction_excluded"] = vf_result

    # Attention rollout comparison block:
    # This performs the same occlusion test but using attention rollout scores
    # instead of SHAP values. It removes top-k (default=5) by attention and
    # reports the prediction drop side-by-side with top-k SHAP removal.
    if compare_attention_topk:
        results["shap_vs_attention_topk"] = compare_topk_shap_vs_attention(
            model=visualizer.model,
            base_data=visualizer.data,
            shap_values=shap_values,
            num_atoms=num_atoms,
            num_add_tokens=num_add_tokens,
            pred_idx=pred_idx,
            configs=visualizer.data["config"],
            k=attention_topk,
            neutralize_mode=neutralize_mode,
            use_mask_only=use_mask_only,
            include_additional_tokens=include_additional_tokens,
            zero_adj=zero_adj,
        )

    return results
