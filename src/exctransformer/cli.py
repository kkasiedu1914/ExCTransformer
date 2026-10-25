"""Command-line entry points for ExCTransformer workflows."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

from exctransformer._version import __version__
from exctransformer.assets import AssetError, default_assets_dir, download_all_assets, download_asset, list_manifest_assets


def _coerce_value(raw: str) -> Any:
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        lowered = raw.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "none":
            return None
        return raw


def _parse_overrides(items: list[str] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE override, got: {item}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Override key cannot be empty: {item}")
        overrides[key] = _coerce_value(raw_value.strip())
    return overrides


def _normalise_result(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _normalise_result(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalise_result(v) for v in value]
    if hasattr(value, "shap_output_path") or hasattr(value, "cif_id"):
        return {
            "cif_id": getattr(value, "cif_id", None),
            "pred_idx": getattr(value, "pred_idx", None),
            "shap_output_path": getattr(value, "shap_output_path", None),
        }
    return str(value)


def _emit_result(result: Any) -> int:
    if result is None:
        return 0
    print(json.dumps(_normalise_result(result), indent=2, sort_keys=True))
    return 0


def _add_common_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override configuration values passed through to the package API.",
    )


def _handle_train(args: argparse.Namespace) -> int:
    from exctransformer.train import train

    train(
        dataset_path=args.dataset_path,
        log_dir=args.log_dir,
        test_only=args.test_only,
        **_parse_overrides(args.overrides),
    )
    return 0


def _handle_test(args: argparse.Namespace) -> int:
    from exctransformer.test import test

    test(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        **_parse_overrides(args.overrides),
    )
    return 0


def _handle_predict(args: argparse.Namespace) -> int:
    from exctransformer.predict import predict

    predict(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        **_parse_overrides(args.overrides),
    )
    return 0


def _handle_explain_single(args: argparse.Namespace) -> int:
    from exctransformer.explain import explain

    model_kwargs = _parse_overrides(args.overrides)
    result = explain(
        data_dir=args.data_dir,
        model_path=args.model_path,
        pred_idx=args.pred_idx,
        single_h5=args.single_h5,
        single_id=args.single_id,
        cif_dir=args.cif_dir,
        zip_path=args.zip_path,
        include_additional_tokens=args.include_additional_tokens,
        args_g=args.args_g,
        args_hv=args.args_hv,
        args_coal=args.args_coal,
        use_mask_only=args.use_mask_only,
        coalition_size=args.coalition_size,
        s_override=args.s_override,
        out_dir=args.out_dir,
        gpu=args.gpu,
        model_kwargs=model_kwargs,
    )
    return _emit_result(result)


def _handle_explain_batch(args: argparse.Namespace) -> int:
    from exctransformer.explain import explain_batch

    model_kwargs = _parse_overrides(args.overrides)
    result = explain_batch(
        data_dir=args.data_dir,
        model_path=args.model_path,
        pred_idx=args.pred_idx,
        cif_dir=args.cif_dir,
        zip_path=args.zip_path,
        build_max_samples=args.build_max_samples,
        out_dir=args.out_dir,
        batch_predict_csv=args.batch_predict_csv,
        batch_max_samples=args.batch_max_samples,
        include_additional_tokens=args.include_additional_tokens,
        args_g=args.args_g,
        args_coal=args.args_coal,
        use_mask_only=args.use_mask_only,
        coalition_size=args.coalition_size,
        s_override=args.s_override,
        gpu=args.gpu,
        sample_ids=args.sample_ids,
        sample_indices=args.sample_indices,
        model_kwargs=model_kwargs,
    )
    return _emit_result(result)


def _handle_explain_visualize(args: argparse.Namespace) -> int:
    from exctransformer.explain import visualize_explanation

    result = visualize_explanation(
        shap_path=args.shap_path,
        zip_path_override=args.zip_path_override,
        vis_modes=args.vis_modes,
        animate=args.animate,
        show=args.show,
        dpi=args.dpi,
        show_axis=args.show_axis,
        show_colorbar=args.show_colorbar,
        figsize_scale=args.figsize_scale,
        save_dir=args.save_dir,
        filename_prefix=args.filename_prefix,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        gpu=args.gpu,
        **_parse_overrides(args.overrides),
    )
    return _emit_result(result)


def _handle_explain_faithfulness(args: argparse.Namespace) -> int:
    from exctransformer.explain import compare_faithfulness

    result = compare_faithfulness(
        h5_path=args.h5_path,
        shap_output_path=args.shap_output_path,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        zip_path=args.zip_path,
        pred_idx=args.pred_idx,
        include_additional_tokens=args.include_additional_tokens,
        gpu=args.gpu,
        top_k_max=args.top_k_max,
        k_step=args.k_step,
        num_random=args.num_random,
        num_random_attn=args.num_random_attn,
        neutralize_mode=args.neutralize_mode,
        zero_adj=args.zero_adj,
        exclude_void_fraction=args.exclude_void_fraction,
        void_fraction_index=args.void_fraction_index,
        exclude_add_tokens=args.exclude_add_tokens,
        show_plot=args.show_plot,
        save_plot_path=args.save_plot_path,
        show_random_band=args.show_random_band,
    )
    return _emit_result(result)


def _handle_explain_rebuild(args: argparse.Namespace) -> int:
    from exctransformer.explain import rebuild_explanations_summary

    result = rebuild_explanations_summary(
        explanations_dir=args.explanations_dir,
        log_path=args.log_path,
        expected_total_samples=args.expected_total_samples,
        glob_pattern=args.glob_pattern,
        recursive=args.recursive,
        overwrite=not args.no_overwrite,
        sort_by=args.sort_by,
    )
    return _emit_result(result)


def _handle_build_dataset(args: argparse.Namespace) -> int:
    from exctransformer.datamodule.workflows import build_h5_dataset_workflow

    result = build_h5_dataset_workflow(
        data_dir=args.data_dir,
        cif_id_prop=args.cif_id_prop,
        cif_path_z=args.cif_path_z,
        dataset_size=args.dataset_size,
        random_seed=args.random_seed,
        radius=args.radius,
        dmin=args.dmin,
        dmax=args.dmax,
        step=args.step,
        output_h5=args.output_h5,
        zip=args.zip,
        resume_load=args.resume_load,
        additional_cols=args.additional_cols,
        target_cols=args.target_cols,
        predict_set=args.predict_set,
    )
    return _emit_result(result)


def _handle_download_list(args: argparse.Namespace) -> int:
    assets = list_manifest_assets(manifest=args.manifest, version=args.version)
    return _emit_result(
        {
            "manifest": args.manifest,
            "version": args.version or __version__,
            "default_cache_dir": str(default_assets_dir()),
            "assets": assets,
        }
    )


def _handle_download_fetch(args: argparse.Namespace) -> int:
    result = download_asset(
        args.name,
        manifest=args.manifest,
        version=args.version,
        destination=args.destination,
        force=args.force,
        extract=(False if args.no_extract else None),
    )
    return _emit_result(result)


def _handle_download_all(args: argparse.Namespace) -> int:
    result = download_all_assets(
        manifest=args.manifest,
        version=args.version,
        destination=args.destination,
        force=args.force,
        extract=(False if args.no_extract else None),
        kind=args.kind,
    )
    return _emit_result(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="exct", description="ExCTransformer command-line interface.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train an ExCTransformer model.")
    train_parser.add_argument("dataset_path")
    train_parser.add_argument("--log-dir", default="logs/")
    train_parser.add_argument("--test-only", action="store_true")
    _add_common_overrides(train_parser)
    train_parser.set_defaults(handler=_handle_train)

    test_parser = subparsers.add_parser("test", help="Evaluate a trained ExCTransformer model.")
    test_parser.add_argument("dataset_path")
    test_parser.add_argument("model_path")
    _add_common_overrides(test_parser)
    test_parser.set_defaults(handler=_handle_test)

    predict_parser = subparsers.add_parser("predict", help="Run prediction with a trained model.")
    predict_parser.add_argument("dataset_path")
    predict_parser.add_argument("model_path")
    _add_common_overrides(predict_parser)
    predict_parser.set_defaults(handler=_handle_predict)

    explain_parser = subparsers.add_parser("explain", help="Explanation and visualization workflows.")
    explain_subparsers = explain_parser.add_subparsers(dest="explain_command", required=True)

    explain_single = explain_subparsers.add_parser("single", help="Explain a single H5 sample.")
    explain_single.add_argument("data_dir")
    explain_single.add_argument("model_path")
    explain_single.add_argument("pred_idx", type=int)
    explain_single.add_argument("--single-h5")
    explain_single.add_argument("--single-id")
    explain_single.add_argument("--cif-dir")
    explain_single.add_argument("--zip-path")
    explain_single.add_argument("--out-dir", default="shap_outputs_single")
    explain_single.add_argument("--gpu", default="auto")
    explain_single.add_argument("--args-g", default="WLS")
    explain_single.add_argument("--args-hv", default="auto")
    explain_single.add_argument("--args-coal", default="NewSmarterSeparate")
    explain_single.add_argument("--coalition-size", default="small")
    explain_single.add_argument("--s-override", type=int)
    explain_single.add_argument("--include-additional-tokens", action="store_true")
    explain_single.add_argument("--use-mask-only", action="store_true")
    _add_common_overrides(explain_single)
    explain_single.set_defaults(handler=_handle_explain_single)

    explain_batch_parser = explain_subparsers.add_parser("batch", help="Explain a batch of samples.")
    explain_batch_parser.add_argument("data_dir")
    explain_batch_parser.add_argument("model_path")
    explain_batch_parser.add_argument("pred_idx", type=int)
    explain_batch_parser.add_argument("--cif-dir")
    explain_batch_parser.add_argument("--zip-path")
    explain_batch_parser.add_argument("--out-dir", default="shap_outputs")
    explain_batch_parser.add_argument("--gpu", default="auto")
    explain_batch_parser.add_argument("--args-g", default="WLS")
    explain_batch_parser.add_argument("--args-coal", default="NewSmarterSeparate")
    explain_batch_parser.add_argument("--coalition-size", default="small")
    explain_batch_parser.add_argument("--build-max-samples", type=int, default=1000)
    explain_batch_parser.add_argument("--batch-max-samples", type=int)
    explain_batch_parser.add_argument("--batch-predict-csv")
    explain_batch_parser.add_argument("--sample-ids", nargs="*")
    explain_batch_parser.add_argument("--sample-indices", nargs="*", type=int)
    explain_batch_parser.add_argument("--include-additional-tokens", action="store_true")
    explain_batch_parser.add_argument("--use-mask-only", action="store_true")
    explain_batch_parser.add_argument("--s-override", type=int)
    _add_common_overrides(explain_batch_parser)
    explain_batch_parser.set_defaults(handler=_handle_explain_batch)

    visualize_parser = explain_subparsers.add_parser("visualize", help="Visualize a saved explanation payload.")
    visualize_parser.add_argument("shap_path")
    visualize_parser.add_argument("--zip-path-override")
    visualize_parser.add_argument("--save-dir")
    visualize_parser.add_argument("--filename-prefix")
    visualize_parser.add_argument("--model-path")
    visualize_parser.add_argument("--dataset-path")
    visualize_parser.add_argument("--gpu", default="auto")
    visualize_parser.add_argument("--dpi", type=int, default=600)
    visualize_parser.add_argument("--figsize-scale", type=float, default=1.0)
    visualize_parser.add_argument("--vis-modes", nargs="*", default=["dual"])
    visualize_parser.add_argument("--animate", action="store_true")
    visualize_parser.add_argument("--show", action="store_true")
    visualize_parser.add_argument("--show-axis", action="store_true")
    visualize_parser.add_argument("--show-colorbar", action="store_true", default=True)
    _add_common_overrides(visualize_parser)
    visualize_parser.set_defaults(handler=_handle_explain_visualize)

    faithfulness_parser = explain_subparsers.add_parser("faithfulness", help="Compare SHAP and attention faithfulness.")
    faithfulness_parser.add_argument("h5_path")
    faithfulness_parser.add_argument("shap_output_path")
    faithfulness_parser.add_argument("model_path")
    faithfulness_parser.add_argument("dataset_path")
    faithfulness_parser.add_argument("zip_path")
    faithfulness_parser.add_argument("--pred-idx", type=int, default=0)
    faithfulness_parser.add_argument("--gpu", default="auto")
    faithfulness_parser.add_argument("--top-k-max", type=int, default=20)
    faithfulness_parser.add_argument("--k-step", type=int, default=1)
    faithfulness_parser.add_argument("--num-random", type=int, default=10)
    faithfulness_parser.add_argument("--num-random-attn", type=int, default=10)
    faithfulness_parser.add_argument("--neutralize-mode", default="mask")
    faithfulness_parser.add_argument("--void-fraction-index", type=int)
    faithfulness_parser.add_argument("--exclude-add-tokens", nargs="*", type=int)
    faithfulness_parser.add_argument("--include-additional-tokens", action="store_true", default=True)
    faithfulness_parser.add_argument("--zero-adj", action="store_true")
    faithfulness_parser.add_argument("--exclude-void-fraction", action="store_true")
    faithfulness_parser.add_argument("--show-plot", action="store_true")
    faithfulness_parser.add_argument("--show-random-band", action="store_true")
    faithfulness_parser.add_argument("--save-plot-path", default="faithfulness_plot.png")
    faithfulness_parser.set_defaults(handler=_handle_explain_faithfulness)

    rebuild_parser = explain_subparsers.add_parser("rebuild-summary", help="Rebuild a batch explanation log/summary.")
    rebuild_parser.add_argument("explanations_dir")
    rebuild_parser.add_argument("--log-path")
    rebuild_parser.add_argument("--expected-total-samples", type=int)
    rebuild_parser.add_argument("--glob-pattern", default="*_pred*_shap.pt")
    rebuild_parser.add_argument("--sort-by", default="mtime")
    rebuild_parser.add_argument("--recursive", action="store_true")
    rebuild_parser.add_argument("--no-overwrite", action="store_true")
    rebuild_parser.set_defaults(handler=_handle_explain_rebuild)

    build_dataset_parser = subparsers.add_parser("build-dataset", help="Build an HDF5 dataset from CIF inputs.")
    build_dataset_parser.add_argument("data_dir")
    build_dataset_parser.add_argument("cif_id_prop")
    build_dataset_parser.add_argument("cif_path_z")
    build_dataset_parser.add_argument("output_h5")
    build_dataset_parser.add_argument("--dataset-size", type=int, default=1000)
    build_dataset_parser.add_argument("--random-seed", type=int, default=123)
    build_dataset_parser.add_argument("--radius", type=float, default=8.0)
    build_dataset_parser.add_argument("--dmin", type=float, default=0.0)
    build_dataset_parser.add_argument("--dmax", type=float, default=8.0)
    build_dataset_parser.add_argument("--step", type=float, default=0.2)
    build_dataset_parser.add_argument("--additional-cols", nargs="*", type=int)
    build_dataset_parser.add_argument("--target-cols", nargs="*", type=int)
    build_dataset_parser.add_argument("--zip", action="store_true", default=False)
    build_dataset_parser.add_argument("--resume-load", action="store_true")
    build_dataset_parser.add_argument("--predict-set", action="store_true")
    build_dataset_parser.set_defaults(handler=_handle_build_dataset)

    download_parser = subparsers.add_parser("download", help="List or fetch versioned ExCTransformer assets.")
    download_subparsers = download_parser.add_subparsers(dest="download_command", required=True)

    download_list = download_subparsers.add_parser("list", help="List assets in the selected manifest.")
    download_list.add_argument("--manifest")
    download_list.add_argument("--version")
    download_list.set_defaults(handler=_handle_download_list)

    download_fetch = download_subparsers.add_parser("fetch", help="Download one named asset.")
    download_fetch.add_argument("name")
    download_fetch.add_argument("--manifest")
    download_fetch.add_argument("--version")
    download_fetch.add_argument("--destination")
    download_fetch.add_argument("--force", action="store_true")
    download_fetch.add_argument("--no-extract", action="store_true")
    download_fetch.set_defaults(handler=_handle_download_fetch)

    download_all = download_subparsers.add_parser("all", help="Download every asset in the selected manifest.")
    download_all.add_argument("--manifest")
    download_all.add_argument("--version")
    download_all.add_argument("--destination")
    download_all.add_argument("--kind", choices=["data", "model", "metadata", "explanation"])
    download_all.add_argument("--force", action="store_true")
    download_all.add_argument("--no-extract", action="store_true")
    download_all.set_defaults(handler=_handle_download_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except AssetError as exc:
        parser.exit(2, f"error: {exc}\n")


def train_main(argv: list[str] | None = None) -> int:
    return main(["train", *(argv or [])])


def test_main(argv: list[str] | None = None) -> int:
    return main(["test", *(argv or [])])


def predict_main(argv: list[str] | None = None) -> int:
    return main(["predict", *(argv or [])])


def explain_main(argv: list[str] | None = None) -> int:
    return main(["explain", *(argv or [])])


def build_dataset_main(argv: list[str] | None = None) -> int:
    return main(["build-dataset", *(argv or [])])


def download_main(argv: list[str] | None = None) -> int:
    return main(["download", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
