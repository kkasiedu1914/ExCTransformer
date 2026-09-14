"""Higher-level explanation workflows built on top of SHAP primitives.

These helpers provide reusable entry points for common explanation tasks so
scripts and future CLI commands do not need to duplicate orchestration logic.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from exctransformer import __root_dir__
from exctransformer.SHAP_visualization.compute_shap import (
    SHAPVisualizer,
    build_samples_from_dir,
    explain_batch,
    rebuild_batch_explanations_summary,
)
from exctransformer.SHAP_visualization import faithfulness as faith


def _load_predict_stems(path: str | None) -> set[str] | None:
    if not path:
        return None
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            stems = {Path(row["cif_id"]).stem for row in reader if row.get("cif_id")}
    except OSError:
        return None
    return stems or None


def resolve_single_h5_input(
    *,
    data_dir: str,
    single_h5: str | None = None,
    single_id: str | None = None,
    root_dir: str | None = None,
) -> str:
    """Resolve a user-specified H5 sample path from explicit path or sample id."""

    root_dir = root_dir or __root_dir__
    resolved_h5 = None
    if single_h5:
        candidates = [
            single_h5,
            os.path.join(data_dir, single_h5),
            os.path.join(root_dir, data_dir, single_h5),
            os.path.join(root_dir, single_h5),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                resolved_h5 = candidate
                break
        if resolved_h5 is None:
            raise FileNotFoundError(f"SHAP_SINGLE_H5 not found: {single_h5}")
    elif single_id:
        stem = single_id if str(single_id).endswith(".h5") else f"{single_id}.h5"
        candidates = [
            os.path.join(data_dir, stem),
            os.path.join(root_dir, data_dir, stem),
            os.path.join(root_dir, stem),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                resolved_h5 = candidate
                break
        if resolved_h5 is None:
            raise FileNotFoundError(f"SHAP_SINGLE_ID not found: {single_id}")
    else:
        raise ValueError("Set SHAP_SINGLE_H5 or SHAP_SINGLE_ID for single explanations.")
    return resolved_h5


def rebuild_batch_summary_workflow(
    *,
    explanations_dir: str,
    log_path: str | None = None,
    expected_total_samples: int | None = None,
    glob_pattern: str = "*_pred*_shap.pt",
    recursive: bool = False,
    overwrite: bool = True,
    sort_by: str = "mtime",
) -> dict[str, Any]:
    """Rebuild a batch explanation summary/log from saved SHAP outputs."""

    return rebuild_batch_explanations_summary(
        explanations_dir=explanations_dir,
        log_path=log_path,
        expected_total_samples=expected_total_samples,
        glob_pattern=glob_pattern,
        recursive=recursive,
        overwrite=overwrite,
        sort_by=sort_by,
    )


def run_single_explanation_workflow(
    *,
    data_dir: str,
    model_path: str,
    pred_idx: int,
    single_h5: str | None = None,
    single_id: str | None = None,
    cif_dir: str | None = None,
    zip_path: str | None = None,
    include_additional_tokens: bool = False,
    args_g: str = "WLS",
    args_hv: str = "auto",
    args_coal: str = "NewSmarterSeparate",
    use_mask_only: bool = False,
    compare_models: bool = False,
    compare_model_list: list[str] | None = None,
    coalition_size: str | int = "small",
    graphsvx_info: bool = True,
    seed: int | None = None,
    s_override: int | None = None,
    out_dir: str = "shap_outputs_single",
    output_file_stem: str | None = None,
    gpu: str | bool = "auto",
    model_kwargs: dict[str, Any] | None = None,
    root_dir: str | None = None,
) -> SHAPVisualizer:
    """Run a single explanation workflow and save the SHAP output."""

    resolved_h5 = resolve_single_h5_input(
        data_dir=data_dir,
        single_h5=single_h5,
        single_id=single_id,
        root_dir=root_dir,
    )
    resolved_cif = None
    if cif_dir is not None:
        resolved_cif = str(Path(cif_dir).expanduser().resolve() / f"{Path(resolved_h5).stem}.cif")
    return SHAPVisualizer(
        cif_id_h5=resolved_h5,
        cif_file=resolved_cif,
        model_path=model_path,
        pred_idx=pred_idx,
        zip_path=zip_path,
        include_additional_tokens=include_additional_tokens,
        args_g=args_g,
        args_hv=args_hv,
        args_coal=args_coal,
        use_mask_only=use_mask_only,
        compare_models=compare_models,
        compare_model_list=compare_model_list,
        coalition_size=coalition_size,
        graphsvx_info=graphsvx_info,
        seed=seed,
        S_override=s_override,
        compute_shap=True,
        save_output=True,
        output_dir=out_dir,
        output_file_stem=output_file_stem,
        gpu=gpu,
        **dict(model_kwargs or {}),
    )


def run_batch_explanations_workflow(
    *,
    data_dir: str,
    model_path: str,
    pred_idx: int,
    cif_dir: str | None = None,
    zip_path: str | None = None,
    build_max_samples: int | None = 1000,
    model_kwargs: dict[str, Any] | None = None,
    out_dir: str = "shap_outputs",
    log_path: str | bool | None = True,
    include_additional_tokens: bool = False,
    args_g: str = "WLS",
    args_coal: str = "NewSmarterSeparate",
    use_mask_only: bool = False,
    compare_models: bool = False,
    compare_model_list: list[str] | None = None,
    batch_predict_csv: str | None = None,
    batch_max_samples: int | None = None,
    batch_criteria: dict[str, Any] | None = None,
    return_valid_sample_count: bool = True,
    coalition_size: str | int = "small",
    s_override: int | None = None,
    gpu: str | bool = "auto",
    graphsvx_info: bool = True,
    seed: int | None = None,
    sample_ids: list[str] | None = None,
    sample_indices: list[int] | None = None,
    use_predict_stems: bool = True,
) -> Any:
    """Build a sample set from a folder and run batch SHAP explanations."""

    filter_stems = None
    if batch_predict_csv and use_predict_stems:
        filter_stems = _load_predict_stems(batch_predict_csv)

    samples = build_samples_from_dir(
        data_dir=data_dir,
        cif_dir=cif_dir,
        zip_path=zip_path,
        max_samples=build_max_samples,
        model_kwargs=dict(model_kwargs or {}),
        filter_stems=filter_stems,
    )
    return explain_batch(
        samples=samples,
        model_path=model_path,
        pred_idx=pred_idx,
        out_dir=out_dir,
        log_path=log_path,
        include_additional_tokens=include_additional_tokens,
        args_g=args_g,
        args_coal=args_coal,
        use_mask_only=use_mask_only,
        compare_models=compare_models,
        compare_model_list=compare_model_list,
        predict_csv_path=batch_predict_csv,
        max_explain_samples=batch_max_samples,
        prediction_criteria=batch_criteria,
        return_valid_sample_count=return_valid_sample_count,
        coalition_size=coalition_size,
        S_override=s_override,
        gpu=gpu,
        graphsvx_info=graphsvx_info,
        seed=seed,
        sample_ids=sample_ids,
        sample_indices=sample_indices,
    )


def rebuild_batch_summary_to_json(**kwargs: Any) -> str:
    """Convenience wrapper for scripts that want a pretty JSON string."""

    return json.dumps(rebuild_batch_summary_workflow(**kwargs), indent=2)


def visualize_saved_explanation_workflow(
    *,
    shap_path: str,
    zip_path_override: str | None = None,
    graph_len: int | None = None,
    add_feat_tokens: int | None = None,
    sep_tokens: int = 1,
    vis_modes: list[str] | tuple[str, ...] | None = None,
    animate: bool = False,
    show: bool = True,
    close: bool | None = None,
    dpi: int = 600,
    show_axis: bool = False,
    show_colorbar: bool = True,
    figsize_scale: float = 1.0,
    shap_unit_label: str | None = None,
    annotate: str = "auto",
    save_formats: list[str] | tuple[str, ...] = ("png", "pdf"),
    save_dir: str | None = None,
    filename_prefix: str | None = None,
    highlight_positive: bool | None = None,
    highlight_radius: float = 4.0,
    highlight_context_alpha: float = 0.5,
    model_path: str | None = None,
    dataset_path: str | None = None,
    num_targets: int | None = None,
    no_labels: bool | None = None,
    model_seed: int | None = None,
    gpu: str | bool = "auto",
    use_checkpoint_hparams: bool = True,
    cleanup_extracted: bool = False,
) -> dict[str, Any]:
    """Visualize a saved SHAP payload and optionally add attention-derived views."""

    import numpy as np
    from matplotlib import pyplot as plt

    from exctransformer.SHAP_visualization.visualizer import PatchVisualizer

    shap_path = str(Path(shap_path).expanduser().resolve())
    if not os.path.exists(shap_path):
        raise FileNotFoundError(f"SHAP_VIS_SHAP_PATH not found: {shap_path}")

    if zip_path_override is not None:
        zip_path_override = str(Path(zip_path_override).expanduser().resolve())
    if save_dir is not None:
        save_dir = str(Path(save_dir).expanduser().resolve())
        os.makedirs(save_dir, exist_ok=True)

    vis = SHAPVisualizer.from_shap_output(shap_path, zip_path=zip_path_override)
    explain_config = vis.explain_config or {}
    explain_model_kwargs = dict(explain_config.get("model_kwargs") or {})

    if graph_len is None:
        graph_len = explain_config.get("num_atoms")
    if graph_len is None:
        graph_len = vis.num_atoms
    if graph_len is None:
        raise ValueError("Unable to infer graph length; set SHAP_VIS_GRAPH_LEN.")
    graph_len = int(graph_len)

    if add_feat_tokens is None:
        add_feat_tokens = explain_config.get("num_add_tokens")
    if add_feat_tokens is None:
        add_feat_tokens = getattr(vis, "add_feat_num", 0) or 0
    add_feat_tokens = int(add_feat_tokens)
    sep_tokens = int(sep_tokens)

    vis_modes = [str(mode).lower() for mode in (vis_modes or ["dual"])]
    if "all" in vis_modes:
        vis_modes = [
            "atom-shap",
            "shap-only",
            "dual",
            "attn-shap-2d",
            "attn-shap-3d",
        ]

    close = (not show) if close is None else close
    filename_prefix = filename_prefix or Path(shap_path).stem
    save_formats = tuple(save_formats)

    attention_modes = {"attn-heatmap", "attn-shap-2d", "attn-shap-3d"}
    vis_with_attention = vis
    if any(mode in attention_modes for mode in vis_modes):
        model_path = model_path or vis.model_path
        if model_path is None:
            raise ValueError(
                "Attention visualizations require a model. "
                "Set SHAP_VIS_MODEL_PATH (or save payloads with attention included)."
            )
        model_path = str(Path(model_path).expanduser().resolve())
        if dataset_path:
            explain_model_kwargs["dataset_path"] = str(Path(dataset_path).expanduser().resolve())
        if num_targets is not None:
            explain_model_kwargs["num_targets"] = int(num_targets)
        elif "num_targets" not in explain_model_kwargs:
            explain_model_kwargs["num_targets"] = 3
        if no_labels is not None:
            explain_model_kwargs["no_labels"] = bool(no_labels)
        elif "no_labels" not in explain_model_kwargs:
            explain_model_kwargs["no_labels"] = False
        if model_seed is not None:
            explain_model_kwargs["seed"] = model_seed

        vis_with_attention = SHAPVisualizer(
            cif_id_h5=vis.cif_id,
            model_path=model_path,
            pred_idx=vis.pred_idx,
            zip_path=zip_path_override or vis.zip_path,
            compute_shap=False,
            shap_output_path=shap_path,
            gpu=gpu,
            use_checkpoint_hparams=use_checkpoint_hparams,
            **explain_model_kwargs,
        )

    payload = vis._resolve_shap_output(shap_output_path=shap_path)
    shap_values = payload["shap_values"] if payload is not None else vis.shap_values
    cif_file = payload.get("cif_file", vis.cif_file) if payload is not None else vis.cif_file
    zip_path = zip_path_override or (payload.get("zip_path") if payload is not None else vis.zip_path)
    add_feat_num = (
        int(payload.get("num_add_tokens", getattr(vis, "add_feat_num", 0)))
        if payload is not None
        else int(getattr(vis, "add_feat_num", 0))
    )
    num_atoms = payload.get("num_atoms") if payload is not None else getattr(vis, "num_atoms", None)

    if shap_values is None:
        raise ValueError(f"No SHAP values found in payload: {shap_path}")
    if hasattr(shap_values, "detach"):
        shap_values = shap_values.detach().cpu()
        if shap_values.ndim == 1:
            shap_values = shap_values.unsqueeze(0)
        shap_len = int(shap_values.numel())
    else:
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)
        shap_len = int(shap_values.size)

    if num_atoms is not None:
        num_atoms = int(num_atoms)
        if shap_len == num_atoms:
            add_feat_num = 0
        elif shap_len > num_atoms and add_feat_num == 0:
            add_feat_num = shap_len - num_atoms

    if cif_file is not None and not os.path.exists(cif_file) and zip_path:
        extracted_path, extracted_new = vis.extract_cif_file_from_zip(zip_path, cif_file)
        if extracted_path is not None:
            cif_file = extracted_path
            if extracted_new:
                vis._cleanup_paths.append(extracted_path)

    if cif_file is None or not os.path.exists(cif_file):
        raise FileNotFoundError(f"Could not resolve CIF file for visualization: {cif_file}")

    executed: list[dict[str, Any]] = []
    base_figsize = (15 * figsize_scale, 15 * figsize_scale)
    dual_figsize = (20 * figsize_scale, 10 * figsize_scale)

    for mode in vis_modes:
        if mode in {"atom-shap", "shap-only", "dual"}:
            patch_visualizer = PatchVisualizer(
                path_cif=cif_file,
                heatmap_graph=shap_values,
                uni_idx=[[i] for i in range(max(0, shap_len - 1))],
                figsize=dual_figsize if mode == "dual" else base_figsize,
                view_init=(25, 25),
                show_axis=show_axis,
                show_colorbar=show_colorbar,
            )
            if mode == "dual":
                fig, _ = patch_visualizer.draw_graph_dual(
                    minatt=float(shap_values.min()),
                    maxatt=float(shap_values.max()),
                    alpha=0.7,
                    atomic_scale_factor=4,
                    grid_scale_factor=1,
                    att_scale_factor=3,
                    add_feat_num=add_feat_num,
                    return_fig=True,
                    SHAP_unit_label=shap_unit_label or " ",
                    highlight_positive=highlight_positive,
                    highlight_radius=highlight_radius,
                    highlight_context_alpha=highlight_context_alpha,
                    show_colorbar=show_colorbar,
                )
            else:
                fig, _ = patch_visualizer.draw_graph(
                    minatt=float(shap_values.min()),
                    maxatt=float(shap_values.max()),
                    alpha=0.7,
                    atomic_scale_factor=4,
                    grid_scale_factor=1,
                    att_scale_factor=3,
                    add_feat_num=add_feat_num,
                    return_fig=True,
                    mode=mode,
                    SHAP_unit_label=shap_unit_label or " ",
                    highlight_positive=highlight_positive,
                    highlight_radius=highlight_radius,
                    highlight_context_alpha=highlight_context_alpha,
                    show_colorbar=show_colorbar,
                )

            save_paths = []
            if save_dir:
                for ext in save_formats:
                    save_path = os.path.join(save_dir, f"{filename_prefix}_{mode.replace('-', '_')}.{ext}")
                    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02, facecolor="white")
                    save_paths.append(save_path)
            if show:
                plt.show()
            if close or not show:
                plt.close(fig)

            if animate:
                anim_path = None
                if save_dir:
                    anim_path = os.path.join(save_dir, f"{filename_prefix}_{mode.replace('-', '_')}.mp4")
                if mode == "dual":
                    anim_draw_graph = patch_visualizer.animate_dual(
                        patch_visualizer.draw_graph_dual,
                        frame=360,
                        interval=20,
                        savefile=anim_path,
                        fps=30,
                    )
                    anim_draw_graph(
                        minatt=float(shap_values.min()),
                        maxatt=float(shap_values.max()),
                        alpha=0.7,
                        add_feat_num=add_feat_num,
                        highlight_positive=highlight_positive,
                        highlight_radius=highlight_radius,
                        highlight_context_alpha=highlight_context_alpha,
                        show_colorbar=show_colorbar,
                        SHAP_unit_label=shap_unit_label or " ",
                    )
                else:
                    animated_draw_graph = patch_visualizer.animate(
                        patch_visualizer.draw_graph,
                        frame=360,
                        interval=20,
                        savefile=anim_path,
                        fps=30,
                    )
                    animated_draw_graph(
                        minatt=float(shap_values.min()),
                        maxatt=float(shap_values.max()),
                        alpha=0.7,
                        add_feat_num=add_feat_num,
                        mode=mode,
                        highlight_positive=highlight_positive,
                        highlight_radius=highlight_radius,
                        highlight_context_alpha=highlight_context_alpha,
                        show_colorbar=show_colorbar,
                        SHAP_unit_label=shap_unit_label or " ",
                    )
                if anim_path:
                    save_paths.append(anim_path)

            executed.append({"mode": mode, "save_paths": save_paths})
        elif mode == "attn-heatmap":
            heatmap = vis_with_attention.att_heatmap_vs_shap_vis(
                graph_len=graph_len,
                sep_tokens=sep_tokens,
                add_feat_tokens=add_feat_tokens,
            )
            heatmap_np = (
                heatmap.detach().cpu().numpy()
                if hasattr(heatmap, "detach")
                else np.asarray(heatmap)
            )
            fig, ax = plt.subplots(figsize=(max(8.0, min(24.0, 0.16 * len(heatmap_np))), 2.8))
            image = ax.imshow(heatmap_np.reshape(1, -1), aspect="auto", cmap="viridis")
            ax.set_title(f"Attention rollout | {Path(shap_path).stem}")
            ax.set_xlabel("Token index (atoms first, feature tokens last)")
            ax.set_yticks([])
            fig.colorbar(image, ax=ax, label="Attention rollout")

            save_paths = []
            if save_dir:
                for ext in save_formats:
                    save_path = os.path.join(save_dir, f"{filename_prefix}_attention_rollout.{ext}")
                    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02, facecolor="white")
                    save_paths.append(save_path)
            if show:
                plt.show()
            if close or not show:
                plt.close(fig)

            executed.append(
                {"mode": mode, "token_count": int(len(heatmap_np)), "save_paths": save_paths}
            )
        elif mode == "attn-shap-2d":
            _, _, meta = vis_with_attention.attn_shap_heatmaps_figure(
                graph_len=graph_len,
                sep_tokens=sep_tokens,
                add_feat_tokens=add_feat_tokens,
                save=bool(save_dir),
                show=show,
                close=close or (not show),
                out_dir=save_dir or os.getcwd(),
                filename=f"{filename_prefix}_attn_shap_2d",
                dpi=dpi,
                annotate=annotate,
                shap_output_path=shap_path,
            )
            executed.append({"mode": mode, "save_paths": meta.get("save_paths", [])})
        elif mode == "attn-shap-3d":
            _, _, meta = vis_with_attention.attn_shap_triptych_figure(
                graph_len=graph_len,
                sep_tokens=sep_tokens,
                add_feat_tokens=add_feat_tokens,
                save=bool(save_dir),
                show=show,
                close=close or (not show),
                out_dir=save_dir or os.getcwd(),
                filename=f"{filename_prefix}_attn_shap_3d",
                dpi=dpi,
                shap_output_path=shap_path,
            )
            executed.append({"mode": mode, "save_paths": meta.get("save_paths", [])})
        else:
            raise ValueError(
                f"Unsupported SHAP_VIS_MODES entry: {mode}. "
                "Use atom-shap, shap-only, dual, attn-shap-2d, attn-shap-3d, or all."
            )

    if cleanup_extracted:
        vis.cleanup_extracted_files()
        if vis_with_attention is not vis:
            vis_with_attention.cleanup_extracted_files()

    return {
        "shap_path": shap_path,
        "cif_file": cif_file,
        "graph_len": graph_len,
        "add_feat_tokens": add_feat_tokens,
        "modes": executed,
    }


def run_faithfulness_comparison_workflow(
    *,
    h5_path: str,
    shap_output_path: str,
    model_path: str,
    dataset_path: str,
    zip_path: str,
    pred_idx: int = 0,
    include_additional_tokens: bool = True,
    gpu: str | bool = "auto",
    top_k_max: int = 20,
    k_step: int = 1,
    num_random: int = 10,
    num_random_attn: int = 10,
    neutralize_mode: str = "mask",
    zero_adj: bool = False,
    exclude_void_fraction: bool = False,
    void_fraction_index: int | None = None,
    exclude_add_tokens: list[int] | tuple[int, ...] | None = None,
    show_plot: bool = False,
    save_plot_path: str | None = "faithfulness_plot.png",
    show_random_band: bool = False,
) -> dict[str, Any]:
    """Compare SHAP-ranked and attention-ranked faithfulness on a single sample."""

    h5_path = str(Path(h5_path).expanduser().resolve())
    shap_output_path = str(Path(shap_output_path).expanduser().resolve())
    dataset_path = str(Path(dataset_path).expanduser().resolve())
    zip_path = str(Path(zip_path).expanduser().resolve())
    model_path = str(Path(model_path).expanduser().resolve())

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 not found: {h5_path}")

    vis = SHAPVisualizer(
        cif_id_h5=h5_path,
        model_path=model_path,
        pred_idx=pred_idx,
        compute_shap=False,
        shap_output_path=shap_output_path,
        include_additional_tokens=include_additional_tokens,
        dataset_path=dataset_path,
        zip_path=zip_path,
        gpu=gpu,
    )

    num_atoms = int(vis.data["num_atoms"])
    num_add_tokens = int(vis.data["num_add_tokens"])

    exclude_indices: list[int] = []
    if exclude_void_fraction:
        if void_fraction_index is None:
            exclude_indices.append(num_atoms)
        else:
            exclude_indices.append(int(void_fraction_index))
    if exclude_add_tokens and num_add_tokens > 0:
        for add_idx in exclude_add_tokens:
            if add_idx == 0:
                continue
            pos = (num_add_tokens + add_idx + 1) if add_idx < 0 else add_idx
            if 1 <= pos <= num_add_tokens:
                exclude_indices.append(num_atoms + (pos - 1))
    if not exclude_indices:
        exclude_indices = None

    results = vis.faithfulness_tests(
        top_k_max=top_k_max,
        num_random=num_random,
        neutralize_mode=neutralize_mode,
        zero_adj=zero_adj,
        exclude_void_fraction=False,
        void_fraction_index=None,
        exclude_indices=exclude_indices,
        plot=False,
    )

    shap_curve = results["primary"]
    k_values = shap_curve["k"]
    if k_step > 1:
        idx = slice(None, None, k_step)
        k_values = k_values[idx]
        shap_curve = dict(shap_curve)
        shap_curve["k"] = shap_curve["k"][idx]
        shap_curve["top_drop"] = shap_curve["top_drop"][idx]
        shap_curve["random_drop_mean"] = shap_curve["random_drop_mean"][idx]
        shap_curve["random_drop_std"] = shap_curve["random_drop_std"][idx]

    attn_scores = faith._attention_rollout_scores(
        model=vis.model,
        base_data=vis.data,
        configs=vis.data["config"],
        num_atoms=num_atoms,
        num_add_tokens=num_add_tokens,
    )
    attn_curve = faith.progressive_occlusion_curve(
        model=vis.model,
        base_data=vis.data,
        shap_values=attn_scores,
        num_atoms=num_atoms,
        num_add_tokens=num_add_tokens,
        pred_idx=vis.pred_idx,
        configs=vis.data["config"],
        k_values=k_values,
        num_random=num_random_attn,
        neutralize_mode=neutralize_mode,
        include_additional_tokens=True,
        exclude_void_fraction=False,
        void_fraction_index=None,
        exclude_indices=exclude_indices,
        zero_adj=zero_adj,
    )

    if not show_plot:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(shap_curve["k"], shap_curve["top_drop"], marker="o", label="Top-k SHAP")
    ax.plot(attn_curve["k"], attn_curve["top_drop"], marker="o", label="Top-k Attention")
    ax.plot(shap_curve["k"], shap_curve["random_drop_mean"], marker="o", label="Random-k")
    if show_random_band:
        ax.fill_between(
            shap_curve["k"],
            shap_curve["random_drop_mean"] - shap_curve["random_drop_std"],
            shap_curve["random_drop_mean"] + shap_curve["random_drop_std"],
            alpha=0.2,
        )
    ax.set_xlabel("Number of neutralized atoms (k)")
    ax.set_ylabel("Prediction drop")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    if save_plot_path:
        fig.savefig(save_plot_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {
        "h5_path": h5_path,
        "shap_output_path": shap_output_path,
        "num_atoms": num_atoms,
        "num_add_tokens": num_add_tokens,
        "exclude_indices": exclude_indices,
        "shap_curve": shap_curve,
        "attention_curve": attn_curve,
        "save_plot_path": save_plot_path,
    }
