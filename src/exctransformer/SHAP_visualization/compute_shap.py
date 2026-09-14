import torch
import os
import numpy as np
import copy
import zipfile
import h5py
import json
import time
import csv
from pathlib import Path
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.io.cif import CifParser
from tempfile import NamedTemporaryFile
from typing import Optional, Tuple

from exctransformer.model_modules.gscgcnn import rebuild_and_mask_batch
from exctransformer.config import ex
from exctransformer.SHAP_visualization.utils import load_model, load_h5_file
from exctransformer.model_modules import model_utils
from exctransformer.SHAP_visualization import utils
from exctransformer.SHAP_visualization import faithfulness
from exctransformer.SHAP_visualization.visualizer import PatchVisualizer
from exctransformer.SHAP_visualization.GraphSVX import GraphSVX_explainer
from exctransformer.datamodule.batch_data import safe_load_flat_sample
import warnings

warnings.filterwarnings("ignore", message="crystal system 'triclinic' is not interpreted")


def _resolve_gpu_setting(gpu):
    if gpu is None:
        return torch.cuda.is_available()
    if isinstance(gpu, str):
        value = gpu.strip().lower()
        if value in {"", "auto"}:
            return torch.cuda.is_available()
        if value in {"1", "true", "yes", "y", "on"}:
            if torch.cuda.is_available():
                return True
            warnings.warn("GPU requested but not available; falling back to CPU.")
            return False
        if value in {"0", "false", "no", "n", "off"}:
            return False
        return torch.cuda.is_available()
    requested = bool(gpu)
    if requested and not torch.cuda.is_available():
        warnings.warn("GPU requested but not available; falling back to CPU.")
        return False
    return requested


def _resolve_seed_setting(seed):
    if seed is None:
        return None
    if isinstance(seed, str):
        value = seed.strip().lower()
        if value in {"", "auto", "none", "off", "false"}:
            return None
        try:
            return int(seed)
        except ValueError:
            return None
    if isinstance(seed, bool):
        return None
    if isinstance(seed, (int, float)):
        return int(seed)
    return None


def _coerce_jsonable(value):
    if isinstance(value, dict):
        return {k: _coerce_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_jsonable(v) for v in value]
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _to_float(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    if isinstance(value, np.generic):
        return float(value.item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _accumulate_compare_metrics(compare_metrics, compare_sums, compare_counts):
    if not isinstance(compare_metrics, dict):
        return
    for model_name, metrics in compare_metrics.items():
        if not isinstance(metrics, dict):
            continue
        model_sums = compare_sums.setdefault(model_name, {})
        model_counts = compare_counts.setdefault(model_name, {})
        for metric_name, metric_val in metrics.items():
            metric_float = _to_float(metric_val)
            if metric_float is None:
                continue
            model_sums[metric_name] = model_sums.get(metric_name, 0.0) + metric_float
            model_counts[metric_name] = model_counts.get(metric_name, 0) + 1


def _build_batch_summary_record(
    total_samples,
    sum_r2,
    count_r2,
    sum_weighted_r2,
    count_weighted_r2,
    compare_sums,
    compare_counts,
    batch_elapsed_s=None,
):
    primary_avg_r2 = (sum_r2 / count_r2) if count_r2 else None
    primary_avg_weighted_r2 = (
        (sum_weighted_r2 / count_weighted_r2) if count_weighted_r2 else None
    )
    batch_summary = {
        "record_type": "batch_summary",
        "total_samples": total_samples,
        "avg_explanation_r2": primary_avg_r2,
        "avg_explanation_weighted_r2": primary_avg_weighted_r2,
        "compare_model_averages": None,
        "batch_elapsed_s": batch_elapsed_s,
    }
    compare_avgs = {
        "primary": {
            "explanation_r2": primary_avg_r2,
            "explanation_weighted_r2": primary_avg_weighted_r2,
        }
    }
    if compare_sums:
        for model_name, metrics in compare_sums.items():
            counts = compare_counts.get(model_name, {})
            model_avgs = {}
            for metric_name, total_val in metrics.items():
                denom = counts.get(metric_name, 0)
                model_avgs[metric_name] = (total_val / denom) if denom else None
            compare_avgs[model_name] = model_avgs
    batch_summary["compare_model_averages"] = compare_avgs
    return batch_summary


def _normalize_log_match_path(path_value, base_dir=None):
    if not path_value:
        return None
    path_obj = Path(path_value)
    if not path_obj.is_absolute() and base_dir is not None:
        path_obj = Path(base_dir) / path_obj
    return str(path_obj.resolve())


def _extract_log_match_keys(record, base_dir=None):
    shap_path = _normalize_log_match_path(record.get("shap_output_path"), base_dir=base_dir)
    pred_idx = record.get("pred_idx")
    if pred_idx is None:
        pred_idx = record.get("pred_index")
    stem = record.get("file_stem")
    cif_id = record.get("cif_id_h5")
    if cif_id is None:
        cif_id = record.get("cif_id")
    if not stem and cif_id:
        stem = Path(str(cif_id)).stem
    return {
        "shap_output_path": shap_path,
        "pred_idx": pred_idx,
        "file_stem": stem,
        "cif_id_h5": cif_id,
    }


def _read_existing_log_metadata(log_path):
    path = Path(log_path)
    if not path.exists():
        return None

    inferred_total_samples = None
    ordered_entries = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total_samples = entry.get("total_samples")
                if inferred_total_samples is None and isinstance(total_samples, int):
                    inferred_total_samples = total_samples
                if entry.get("record_type") == "batch_summary":
                    continue
                match_keys = _extract_log_match_keys(entry, base_dir=path.parent)
                if any(match_keys.values()):
                    ordered_entries.append(match_keys)
    except OSError:
        return None

    return {
        "total_samples": inferred_total_samples,
        "ordered_entries": ordered_entries,
    }


def rebuild_batch_explanations_log(
    out_dir,
    log_path=None,
    *,
    expected_total_samples=None,
    glob_pattern="*_pred*_shap.pt",
    recursive=False,
    overwrite=False,
    sort_by="mtime",
):
    """
    Rebuild a batch explanation JSONL log from saved per-sample SHAP payloads.

    This is intended for recovery scenarios where `batch_explanations_log.jsonl`
    is incomplete or missing, but the per-sample `*_shap.pt` files still exist.

    Args:
        out_dir (str | Path): Directory containing saved SHAP payloads.
        log_path (str | Path | None): Output JSONL path. Defaults to
            `<out_dir>/batch_explanations_log.jsonl`.
        expected_total_samples (int | None): Optional intended batch size. Use
            this when only a subset of the batch completed but you want the
            rebuilt summary to keep the original total (for example, 4 of 50).
        glob_pattern (str): Filename pattern used to discover SHAP payloads.
        recursive (bool): Search recursively under `out_dir`.
        overwrite (bool): Overwrite an existing log file when True.
        sort_by (str): `"mtime"` (default) or `"name"` for recovered sample order.

    Returns:
        dict: Summary of the rebuilt log, including counts and output path.
    """
    out_dir = Path(out_dir)
    if not out_dir.exists():
        raise FileNotFoundError(f"Explanation directory does not exist: {out_dir}")
    if not out_dir.is_dir():
        raise NotADirectoryError(f"Explanation path is not a directory: {out_dir}")

    if log_path is None:
        log_path = out_dir / "batch_explanations_log.jsonl"
    else:
        log_path = Path(log_path)

    existing_meta = _read_existing_log_metadata(log_path)

    if log_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing log: {log_path}. "
            "Pass overwrite=True to rebuild in place."
        )

    iterator = out_dir.rglob(glob_pattern) if recursive else out_dir.glob(glob_pattern)
    shap_paths = [path for path in iterator if path.is_file()]
    if not shap_paths:
        raise FileNotFoundError(
            f"No SHAP payloads matching '{glob_pattern}' were found in {out_dir}"
        )

    order_lookup = {}
    if existing_meta:
        for idx, entry in enumerate(existing_meta.get("ordered_entries", [])):
            keys = [
                entry.get("shap_output_path"),
                (
                    entry.get("file_stem"),
                    entry.get("pred_idx"),
                ),
                (
                    entry.get("cif_id_h5"),
                    entry.get("pred_idx"),
                ),
            ]
            for key in keys:
                if key and key not in order_lookup:
                    order_lookup[key] = idx

    recovered = []
    for path in shap_paths:
        payload = utils.load_shap_output(path)
        if not payload:
            continue
        pred_idx = payload.get("pred_index")
        cif_id_h5 = payload.get("cif_id")
        file_stem = Path(str(cif_id_h5)).stem if cif_id_h5 else path.name.split("_pred", 1)[0]
        path_abs = str(path.resolve())
        order_idx = None
        for key in (
            path_abs,
            (file_stem, pred_idx),
            (cif_id_h5, pred_idx),
        ):
            if key in order_lookup:
                order_idx = order_lookup[key]
                break
        recovered.append(
            {
                "path": path,
                "path_abs": path_abs,
                "mtime": path.stat().st_mtime,
                "payload": payload,
                "pred_idx": pred_idx,
                "cif_id_h5": cif_id_h5,
                "file_stem": file_stem,
                "order_idx": order_idx,
            }
        )

    if not recovered:
        raise ValueError(f"Failed to load any SHAP payloads from {out_dir}")

    if str(sort_by).lower() == "name":
        fallback_key = lambda item: (item["path"].name.lower(), item["path_abs"])
    else:
        fallback_key = lambda item: (item["mtime"], item["path"].name.lower(), item["path_abs"])

    recovered.sort(
        key=lambda item: (
            item["order_idx"] is None,
            item["order_idx"] if item["order_idx"] is not None else 0,
            *fallback_key(item),
        )
    )

    total_samples = expected_total_samples
    if total_samples is None and existing_meta:
        total_samples = existing_meta.get("total_samples")
    if total_samples is None:
        total_samples = len(recovered)
    total_samples = max(int(total_samples), len(recovered))

    log_path.parent.mkdir(parents=True, exist_ok=True)

    sum_r2 = 0.0
    sum_weighted_r2 = 0.0
    count_r2 = 0
    count_weighted_r2 = 0
    compare_sums = {}
    compare_counts = {}
    record_count = 0

    with log_path.open("w", encoding="utf-8") as handle:
        for idx, item in enumerate(recovered, start=1):
            payload = item["payload"]
            r2_val = _to_float(payload.get("explanation_r2"))
            weighted_r2_val = _to_float(payload.get("explanation_weighted_r2"))
            compare_metrics = _coerce_jsonable(payload.get("compare_model_metrics"))

            if r2_val is not None:
                sum_r2 += r2_val
                count_r2 += 1
            if weighted_r2_val is not None:
                sum_weighted_r2 += weighted_r2_val
                count_weighted_r2 += 1
            _accumulate_compare_metrics(compare_metrics, compare_sums, compare_counts)

            avg_r2 = (sum_r2 / count_r2) if count_r2 else None
            avg_weighted_r2 = (
                (sum_weighted_r2 / count_weighted_r2) if count_weighted_r2 else None
            )
            compare_model_list = payload.get("compare_model_list")
            explain_config = _coerce_jsonable(payload.get("explain_config"))
            compare_models = False
            if isinstance(explain_config, dict):
                compare_models = bool(explain_config.get("compare_models"))
            if not compare_models:
                compare_models = bool(compare_metrics or compare_model_list)

            log_entry = {
                "sample_index": idx,
                "total_samples": total_samples,
                "cif_id_h5": payload.get("cif_id"),
                "cif_file": payload.get("cif_file"),
                "zip_path": payload.get("zip_path"),
                "file_stem": item["file_stem"],
                "pred_idx": item["pred_idx"],
                "shap_output_path": item["path_abs"],
                "compare_models": compare_models,
                "compare_model_list": _coerce_jsonable(compare_model_list),
                "compare_model_metrics": compare_metrics,
                "explanation_r2": r2_val,
                "explanation_weighted_r2": weighted_r2_val,
                "avg_explanation_r2": avg_r2,
                "avg_explanation_weighted_r2": avg_weighted_r2,
                "explain_config": explain_config,
                "sample_started_at": None,
                "sample_finished_at": None,
                "sample_elapsed_s": None,
                "batch_elapsed_s": None,
                "recovered_from_shap_output": True,
            }
            handle.write(json.dumps(log_entry, separators=(",", ":")) + "\n")
            record_count += 1

        batch_summary = _build_batch_summary_record(
            total_samples=total_samples,
            sum_r2=sum_r2,
            count_r2=count_r2,
            sum_weighted_r2=sum_weighted_r2,
            count_weighted_r2=count_weighted_r2,
            compare_sums=compare_sums,
            compare_counts=compare_counts,
            batch_elapsed_s=None,
        )
        batch_summary["completed_samples"] = record_count
        batch_summary["recovered_from_shap_output"] = True
        handle.write(json.dumps(batch_summary, separators=(",", ":")) + "\n")

    return {
        "log_path": str(log_path),
        "completed_samples": record_count,
        "total_samples": total_samples,
        "avg_explanation_r2": batch_summary.get("avg_explanation_r2"),
        "avg_explanation_weighted_r2": batch_summary.get(
            "avg_explanation_weighted_r2"
        ),
    }


def rebuild_batch_explanations_summary(
    explanations_dir,
    log_path=None,
    *,
    expected_total_samples=None,
    glob_pattern="*_pred*_shap.pt",
    recursive=False,
    overwrite=False,
    sort_by="mtime",
):
    """
    Rebuild the batch explanation JSONL summary from a directory of saved explanations.

    This is a naming-focused wrapper around `rebuild_batch_explanations_log` for
    archive recovery workflows where the main need is to restore the trailing
    `batch_summary` record (and, when needed, the sample JSONL entries preceding it).
    """
    return rebuild_batch_explanations_log(
        out_dir=explanations_dir,
        log_path=log_path,
        expected_total_samples=expected_total_samples,
        glob_pattern=glob_pattern,
        recursive=recursive,
        overwrite=overwrite,
        sort_by=sort_by,
    )


def _clone_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _clone_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_device(v, device) for v in value)
    return value


class SHAPVisualizer(object):
    @classmethod
    def from_shap_output(cls, shap_output_path, zip_path=None):
        """
        Create a lightweight visualizer for plotting from a saved SHAP payload.
        This skips model loading and SHAP computation.
        """
        payload = utils.load_shap_output(shap_output_path)
        if payload is None:
            raise ValueError(f"Failed to load SHAP payload: {shap_output_path}")
        obj = cls.__new__(cls)
        obj.kwargs = {}
        obj.cif_id = payload.get("cif_id")
        obj.cif_file = payload.get("cif_file")
        obj.zip_path = zip_path or payload.get("zip_path")
        obj.model_path = payload.get("model_path")
        obj.pred_idx = payload.get("pred_index")
        obj.gpu = False
        obj._cleanup_paths = []
        obj.model = None
        obj.configs = payload.get("configs")
        obj.cif_ = None
        obj.graph_embeds = None
        obj.added_feats = None
        obj.add_feat_num = int(payload.get("num_add_tokens") or 0)
        obj.node_embeddings = payload.get("node_embeddings")
        obj.adj_edge = payload.get("adj_edge")
        obj.data = {}
        obj.explainer = None
        obj.shap_values = payload.get("shap_values")
        obj.base_value = payload.get("base_value")
        obj.num_atoms = payload.get("num_atoms")
        obj.explain_config = payload.get("explain_config")
        obj.compare_model_list = payload.get("compare_model_list")
        obj.compare_results = payload.get("compare_model_metrics")
        obj.explanation_r2 = payload.get("explanation_r2")
        obj.explanation_weighted_r2 = payload.get("explanation_weighted_r2")
        return obj

    def __init__(
        self,
        cif_id_h5,
        model_path,
        pred_idx,
        cif_file=None,
        zip_path=None,
        include_additional_tokens=False,
        gpu="auto",
        args_g="WLS",
        args_hv="auto",
        args_coal="NewSmarterSeparate",
        use_mask_only=False,
        compare_models=False,
        compare_model_list=None,
        seed=None,
        compute_shap=True,
        shap_output_path=None,
        save_output=False,
        output_dir="shap_outputs",
        output_file_stem=None,
        coalition_size="small",
        S_override=None,
        verbose=True,
        graphsvx_info=True,
        use_checkpoint_hparams=True,
        **kwargs,
    ):
        """
        Initialize a SHAP visualizer for a single crystal sample.

        This helper loads the model, builds graph/feature embeddings for the sample,
        constructs the adjacency matrix from the CIF, and optionally computes GraphSVX
        SHAP values. It can also load precomputed SHAP outputs from disk and skip
        computation.

        Args:
            cif_id_h5 (str): Path to the per-sample H5 file.
            cif_file (str | None): Optional CIF path or filename to pair with the H5.
            model_path (str): Path to the trained model checkpoint.
            pred_idx (int): Target index to explain.
            zip_path (str | None): Optional path to zip file containing CIFs.
            include_additional_tokens (bool): Treat additional feature tokens as players.
            gpu (bool | str | None): Use GPU for model inference. Set True/False or
                "auto" to enable when CUDA is available (default).
            args_g (str): Surrogate model type for SHAP (e.g., "WLS").
            args_hv (str): Strategy used to map simplified inputs to model outputs
                ("graph_regression", "graph_classification", or "auto").
            args_coal (str): Coalition sampling strategy.
            use_mask_only (bool): Use attention-mask ablation instead of feature replacement.
            compare_models (bool): Compare surrogate models on the sampled coalitions.
            compare_model_list (list[str] | None): Models to compare if compare_models is True.
            seed (int | str | None): RNG seed for reproducible coalition sampling. When
                None/"none"/"auto", sampling remains stochastic.
            compute_shap (bool): Whether to compute SHAP values on init.
            shap_output_path (str | None): Load SHAP values from a saved payload.
            save_output (bool): Save SHAP outputs after computation.
            output_dir (str): Output directory for saved SHAP payloads.
            output_file_stem (str | None): Optional filename stem for saved payloads.
            coalition_size (str | int): "small", "medium", "large", or an int for num_samples.
            S_override (int | None): Optional override for GraphSVX S (max coalition size).
            verbose (bool): Print run configuration for single-sample explanations.
            graphsvx_info (bool): Print GraphSVX surrogate diagnostics (r2, etc.).
            use_checkpoint_hparams (bool): Load the Module via `load_from_checkpoint`
                so saved hparams (e.g., mean/std) are restored.
            **kwargs: Extra model configuration overrides (e.g., dataset_path, num_targets).
        """
        self.kwargs = copy.deepcopy(kwargs)
        self.verbose = verbose
        self.cif_id = cif_id_h5
        self._cleanup_paths = []
        requested_cif = cif_file or (cif_id_h5.rsplit('.', 1)[0] + ".cif")
        requested_cif_name = Path(requested_cif).name
        h5_path = Path(cif_id_h5).expanduser().resolve()
        if zip_path is not None:
            file_name = requested_cif_name
            extracted_path, extracted_new = self.extract_cif_file_from_zip(zip_path, file_name)
            if extracted_path is not None:
                if extracted_new:
                    self._cleanup_paths.append(extracted_path)
                self.cif_file = extracted_path
            else:
                self.cif_file = requested_cif
        else:
            cif_candidates = [
                Path(requested_cif).expanduser(),
                h5_path.parent / requested_cif_name,
                h5_path.parent / "cifs" / requested_cif_name,
                h5_path.parent.parent / "cifs" / requested_cif_name,
            ]
            resolved_cif = next((str(path) for path in cif_candidates if path.exists()), None)
            self.cif_file = resolved_cif or requested_cif
        
        self.model_path = model_path
        self.pred_idx = pred_idx
        self.gpu = _resolve_gpu_setting(gpu)
        self.zip_path = zip_path
        if "num_targets" not in self.kwargs or self.kwargs.get("num_targets") in (None, 0):
            inferred = _infer_num_targets_from_h5(cif_id_h5)
            if inferred is not None:
                self.kwargs["num_targets"] = inferred

        self.model, self.configs = load_model(
            self.model_path,
            use_checkpoint_hparams=use_checkpoint_hparams,
            **self.kwargs,
        )
        self.seed = _resolve_seed_setting(seed)
        self.device = torch.device(
            "cuda" if self.gpu and torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        raw_cif = load_h5_file(self.cif_id)
        self.cif_ = _clone_to_device(raw_cif, self.device)
        self.model.write_log = False
        model_utils.set_task(self.model)
        self.h5_add_tokens = _infer_add_tokens_from_h5(self.cif_)

        # Bind SHAPVisualizer helpers to the model instance (not to this visualizer).
        self.model.init_embed_gen = SHAPVisualizer.init_embed_gen.__get__(
            self.model, self.model.__class__
        )
        self.model.infer_out = SHAPVisualizer.infer_out.__get__(
            self.model, self.model.__class__
        )
        self.model.reverse_rebuild_and_mask_batch = (
            SHAPVisualizer.reverse_rebuild_and_mask_batch.__get__(
                self.model, self.model.__class__
            )
        )
        
        self.graph_embeds, self.added_feats = self.model.init_embed_gen(self.cif_)
        if self.added_feats is not None:
            self.add_feat_num = self.added_feats.shape[0]
            self.node_embeddings =  torch.cat([self.graph_embeds, self.added_feats], dim=0)
        else:
            self.add_feat_num = 0
            self.node_embeddings = self.graph_embeds

        # Expose how many *token embeddings* are appended after graph atoms (NOT how many raw
        # scalar additional features exist) so `infer_out` can split `data["x"]` correctly.
        self.model._shap_num_add_tokens = self.add_feat_num


        self.adj_edge = self.build_adjacency_matrix(self.cif_file)
        if self.added_feats is not None:
            self.adj_edge = self._pad_adjacency_for_tokens(
                self.adj_edge, self.added_feats.shape[0]
            )
        if self.adj_edge.shape[0] != self.node_embeddings.shape[0]:
            warnings.warn(
                "Adjacency/embedding size mismatch; skipping sample. "
                "Inspect the H5 and CIF (or CIF zip) sources to ensure they describe the same structure "
                "(e.g., primitive cell vs. supercell mismatch)."
            )
            raise ValueError(
                f"Adjacency/embedding size mismatch for sample {self.cif_id}."
            )

        self.data = {"x": self.node_embeddings,
                    "edge_index": self.adj_edge,
                    "pred_index": self.pred_idx,
                    "config": self.configs,
                    "num_atoms": int(self.graph_embeds.shape[0]),
                    "num_add_tokens": int(self.add_feat_num)}
        
        self.explainer = GraphSVX_explainer(self.data, self.model, gpu=self.gpu)
        self.shap_values = None
        self.base_value = None
        self.explain_config = None
        self.compare_model_list = compare_model_list
        self.compare_results = None
        self.explanation_r2 = None
        self.explanation_weighted_r2 = None
        self.shap_output_path = None
        if shap_output_path is not None:
            payload = utils.load_shap_output(shap_output_path)
            if payload is not None:
                self.shap_values = payload.get("shap_values")
                self.base_value = payload.get("base_value")
                self.explain_config = payload.get("explain_config")
                self.compare_model_list = payload.get("compare_model_list")
                self.compare_results = payload.get("compare_model_metrics")
                self.explanation_r2 = payload.get("explanation_r2")
                self.explanation_weighted_r2 = payload.get("explanation_weighted_r2")
                self.shap_output_path = shap_output_path

        if compute_shap:
            self.compute_shap(
                include_additional_tokens=include_additional_tokens,
                args_g=args_g,
                args_hv=args_hv,
                args_coal=args_coal,
                use_mask_only=use_mask_only,
                compare_models=compare_models,
                compare_model_list=compare_model_list,
                coalition_size=coalition_size,
                S_override=S_override,
                graphsvx_info=graphsvx_info,
                seed=self.seed,
            )
            if save_output:
                self.shap_output_path = self.save_shap_output(
                    out_dir=output_dir,
                    file_stem=output_file_stem,
                )

    def compute_shap(
        self,
        include_additional_tokens=False,
        args_g="WLS",
        args_hv="auto",
        args_coal="NewSmarterSeparate",
        use_mask_only=False,
        compare_models=False,
        compare_model_list=None,
        coalition_size="small",
        S_override=None,
        graphsvx_info=True,
        seed=None,
    ):
        resolved_seed = _resolve_seed_setting(seed)
        if resolved_seed is None:
            resolved_seed = self.seed
        num_atoms = int(self.graph_embeds.shape[0])
        add_players_used = int(self.add_feat_num) if include_additional_tokens else 0
        add_players_h5 = int(self.h5_add_tokens) if include_additional_tokens else 0
        player_count = num_atoms + add_players_used
        num_samples, S = self._resolve_coalition_size(
            coalition_size, player_count, default_policy="small"
        )
        if str(args_coal).strip() == "All":
            all_cap = int(getattr(self.explainer, "all_coalition_cap", 20000))
            num_samples = min(all_cap, 2**player_count)
        if S_override is not None:
            try:
                S_override = int(S_override)
            except (TypeError, ValueError):
                S_override = None
        if S_override is not None:
            if S_override < 1:
                S_override = 1
            S = min(S_override, max(1, player_count - 1))
        self.last_num_samples = num_samples
        self.last_S = S
        if graphsvx_info:
            self.explainer.progress_log_every_s = 1800
            self.explainer.coalition_log_every_s = 1800
        else:
            self.explainer.progress_log_every_s = None
            self.explainer.coalition_log_every_s = None
        if self.verbose:
            if include_additional_tokens and add_players_h5 != add_players_used:
                print(
                    "Warning: H5 additional_features count does not match embedded "
                    f"token count (h5={add_players_h5}, embedded={add_players_used})."
                )
            print(
                "SHAP run config | "
                f"num_atoms={num_atoms} add_tokens={add_players_used} "
                f"h5_add_tokens={add_players_h5} players={player_count} "
                f"coalition_builder={args_coal} coalition_size={coalition_size} "
                f"num_samples={num_samples} S={S}"
            )
        self.shap_values, self.base_value = self.explainer.explain_crystal(
            num_samples=num_samples,
            S=S,
            info=graphsvx_info,
            include_additional_tokens=include_additional_tokens,
            use_mask_only=use_mask_only,
            args_hv=args_hv,
            args_coal=args_coal,  # 'Smarter' NewSmarterSeparate, All, Random
            args_g=args_g,  # WLS (Ridge), WLR_sklearn, WLR_Lasso, WTR_XGB
            seed=resolved_seed,
        )
        requested_samples = getattr(self.explainer, "last_requested_num_samples", None)
        effective_samples = getattr(self.explainer, "last_effective_num_samples", None)
        if self.verbose:
            print(
                "Coalition sampling summary | "
                f"coalition_builder={args_coal} "
                f"num_samples_requested={requested_samples} "
                f"num_samples_effective={effective_samples}"
            )
        self.explain_config = {
            "coalition_size": coalition_size,
            "num_samples": num_samples,
            "num_samples_requested": requested_samples,
            "num_samples_effective": effective_samples,
            "S": S,
            "S_override": S_override,
            "args_g": args_g,
            "args_hv": args_hv,
            "args_coal": args_coal,
            "use_mask_only": use_mask_only,
            "include_additional_tokens": include_additional_tokens,
            "compare_models": compare_models,
            "compare_model_list": compare_model_list,
            "seed": resolved_seed,
            "num_atoms": num_atoms,
            "num_add_tokens": add_players_used,
            "num_add_tokens_h5": add_players_h5,
            "player_count": player_count,
            "model_kwargs": copy.deepcopy(self.kwargs),
        }
        self.compare_model_list = compare_model_list
        metrics = getattr(self.explainer, "surrogate_metrics", None)
        if isinstance(metrics, list) and metrics:
            metrics = metrics[0]
        if isinstance(metrics, dict):
            self.explanation_r2 = metrics.get("r2")
            self.explanation_weighted_r2 = metrics.get("weighted_r2")

        if compare_models:
            self.compare_results = self.explainer.compare_explanation_models(
                z_=self.explainer.z_,
                weights=self.explainer.weights,
                fz=self.explainer.fz,
                models=compare_model_list,
                info=graphsvx_info,
            )
        return self.shap_values, self.base_value
        
    def faithfulness_tests(
        self,
        shap_output=None,
        shap_output_path=None,
        shap_values=None,
        pred_idx=None,
        k_values=None,
        top_k_max=20,
        num_random=10,
        random_seed=7,
        neutralize_mode="mean",
        use_mask_only=None,
        include_additional_tokens=True,
        exclude_void_fraction=False,
        void_fraction_index=None,
        exclude_indices=None,
        zero_adj=True,
        plot=True,
        plot_title=None,
        save_plot_path=None,
        run_void_fraction_check=False,
    ):
        """
        Run faithfulness tests (progressive occlusion) for this sample.

        This wraps exctransformer.SHAP_visualization.faithfulness.run_faithfulness_tests
        and uses the current model, data, and SHAP values.
        """
        return faithfulness.run_faithfulness_tests(
            visualizer=self,
            shap_output=shap_output,
            shap_output_path=shap_output_path,
            shap_values=shap_values,
            pred_idx=pred_idx,
            k_values=k_values,
            top_k_max=top_k_max,
            num_random=num_random,
            random_seed=random_seed,
            neutralize_mode=neutralize_mode,
            use_mask_only=use_mask_only,
            include_additional_tokens=include_additional_tokens,
            exclude_void_fraction=exclude_void_fraction,
            void_fraction_index=void_fraction_index,
            exclude_indices=exclude_indices,
            zero_adj=zero_adj,
            plot=plot,
            plot_title=plot_title,
            save_plot_path=save_plot_path,
            run_void_fraction_check=run_void_fraction_check,
        )


    def _to_cpu(self, value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        return value

    def _get_optional_sample_field(self, key):
        if isinstance(self.cif_, dict) and key in self.cif_:
            return self.cif_[key]
        return getattr(self.model, key, None)

    def build_shap_output(self, include_surrogate=True, include_attention=False):
        """
        Build a serializable payload that captures the full sample context plus SHAP outputs.
        """
        pred, features = self.model.infer_out(
            self.data,
            self.data["pred_index"],
            configs=self.data["config"],
            return_features=True,
        )
        payload = {
            "cif_id": self.cif_id,
            "cif_file": self.cif_file,
            "zip_path": self.zip_path,
            "pred_index": self.pred_idx,
            "model_path": self.model_path,
            "task_types": getattr(self.model, "current_tasks", None),
            "configs": copy.deepcopy(self.configs),
            "num_atoms": int(self.graph_embeds.shape[0]),
            "num_add_tokens": int(self.add_feat_num),
            "node_embeddings": self._to_cpu(self.node_embeddings),
            "adj_edge": self._to_cpu(self.adj_edge),
            "shap_values": self._to_cpu(self.shap_values),
            "base_value": self._to_cpu(self.base_value),
            "prediction": self._to_cpu(pred),
            "num_atoms_per_crystal": self._get_optional_sample_field("num_atoms_per_crystal"),
            "crystal_atom_idx": self._get_optional_sample_field("crystal_atom_idx"),
            "explain_config": copy.deepcopy(self.explain_config),
            "compare_model_list": copy.deepcopy(self.compare_model_list),
            "compare_model_metrics": copy.deepcopy(self.compare_results),
            "explanation_r2": self.explanation_r2,
            "explanation_weighted_r2": self.explanation_weighted_r2,
        }
        metrics = getattr(self.explainer, "surrogate_metrics", None)
        if metrics is not None:
            payload["surrogate_metrics"] = copy.deepcopy(metrics)
        if self.explanation_r2 is None or self.explanation_weighted_r2 is None:
            metrics = getattr(self.explainer, "surrogate_metrics", None)
            if isinstance(metrics, list) and metrics:
                metrics = metrics[0]
            if isinstance(metrics, dict):
                payload["explanation_r2"] = metrics.get("r2")
                payload["explanation_weighted_r2"] = metrics.get("weighted_r2")

        if features is not None:
            payload.update(
                {
                    "graph_feats": self._to_cpu(features.get("graph_feats")),
                    "added_feats": self._to_cpu(features.get("added_feats")),
                    "cls_feats": self._to_cpu(features.get("cls_feats")),
                    "raw_cls_feats": self._to_cpu(features.get("raw_cls_feats")),
                    "graph_mask": self._to_cpu(features.get("graph_mask")),
                }
            )

        if include_surrogate:
            # These can be large; keep them grouped to make opt-in storage obvious.
            payload["surrogate"] = {
                "z": self._to_cpu(getattr(self.explainer, "z_", None)),
                "weights": self._to_cpu(getattr(self.explainer, "weights", None)),
                "fz": self._to_cpu(getattr(self.explainer, "fz", None)),
            }

        if include_attention:
            _, attn_weights = self.model.infer_out(
                self.data,
                self.data["pred_index"],
                configs=self.data["config"],
                return_attn_weights=True,
            )
            payload["attn_weights"] = [self._to_cpu(w) for w in attn_weights]

        return payload

    @staticmethod
    def _resolve_coalition_size(coalition_size, player_count, default_policy="small"):
        if coalition_size is None:
            coalition_size = default_policy
        if isinstance(coalition_size, int):
            num_samples = max(1, coalition_size)
            S = min(8, max(1, player_count - 1))
            return num_samples, S
        size = str(coalition_size).lower().strip()
        if size == "small":
            num_samples = max(512, min(2048, 4 * player_count))
        elif size == "medium":
            num_samples = max(1024, min(4096, 8 * player_count))
        elif size == "large":
            num_samples = max(2048, min(8192, 16 * player_count))
        else:
            raise ValueError(
                "coalition_size must be 'small', 'medium', 'large', or an int."
            )
        S = min(8, max(1, player_count - 1))
        return num_samples, S

    def save_shap_output(
        self,
        out_dir="shap_outputs",
        file_stem=None,
        include_surrogate=True,
        include_attention=False,
        overwrite=True,
    ):
        """
        Save a per-sample SHAP output file containing inputs, features, and explanations.
        """
        os.makedirs(out_dir, exist_ok=True)
        stem = file_stem or Path(self.cif_id).stem or Path(self.cif_file).stem
        out_path = os.path.join(out_dir, f"{stem}_pred{self.pred_idx}_shap.pt")
        if os.path.exists(out_path) and not overwrite:
            raise FileExistsError(f"SHAP output already exists: {out_path}")
        payload = self.build_shap_output(
            include_surrogate=include_surrogate,
            include_attention=include_attention,
        )
        torch.save(payload, out_path)
        return out_path

    def _resolve_shap_output(self, shap_output=None, shap_output_path=None):
        if shap_output is not None:
            return shap_output
        if shap_output_path:
            return utils.load_shap_output(shap_output_path)
        return None

    def init_embed_gen(self, cif_file):
        self.atom_fea = cif_file["atom_fea"]  # [B, F]
        self.nbr_fea = cif_file["nbr_fea"]  # [B, M, F]
        self.nbr_fea_idx = cif_file["nbr_fea_idx"]  # [B, M]
        self.crystal_atom_idx = cif_file["crystal_atom_idx"]  # [B]
        self.num_atoms_per_crystal = cif_file["num_atoms_per_crystal"]  # [B]
        self.cif_id = cif_file["cif_id"]

        if self.additional_feat_embedding is not None:
            self.additional_features = cif_file["additional_features"]  # [B, A]
            self.additional_feat_num = self.additional_features.shape[1] 

        
        graph_embeds, self.graph_mask = self.graphsage(
            self.atom_fea, self.nbr_fea, self.nbr_fea_idx, self.num_atoms_per_crystal
        )  # [B, N, hid_dim], [B, N]

        # Obtain atom embeddings only
        graph_embeds = self.reverse_rebuild_and_mask_batch(
            graph_embeds, self.graph_mask
        ) # [B, total_atoms, hid_dim]
        

        if self.additional_feat_embedding is not None:
            self.additional_features = self.normalizer_add_feat.encode(self.additional_features)
            self.additional_fea_embeds, self.additional_fea_mask = self.additional_feat_embedding(
                self.additional_features
            )  # [B, 2, hid_dim], [B, 2]
            additional_feats = self.additional_fea_embeds[:, 1, :]  \
                if  self.additional_feat_num==1 else \
                    self.additional_fea_embeds[:, 1:self.additional_feat_num, :]# [B, hid_dim]

        return graph_embeds, additional_feats if self.additional_feat_embedding is not None else None

          

    def infer_out(self, data, pred_idx, configs=None, return_attn_weights=False, return_features=False):
        max_atoms = configs["max_crystal_atoms"]
        out_dim = configs["out_dim"]
        num_add_tokens = int(getattr(self, "_shap_num_add_tokens", 0) or 0)
        if num_add_tokens < 0:
            raise ValueError(f"_shap_num_add_tokens must be >= 0, got {num_add_tokens}.")

        atom_x = data["x"][:-num_add_tokens, :] if num_add_tokens else data["x"]
        graph_embeds, graph_mask = rebuild_and_mask_batch(
            (None, atom_x),
            self.num_atoms_per_crystal,
            max_atoms,
            out_dim,
        )
    
        device = data["x"].device if torch.is_tensor(data.get("x")) else self.device
        additional_fea_mask = self.additional_fea_mask if self.additional_feat_embedding is not None else None
        if additional_fea_mask is not None and additional_fea_mask.device != device:
            additional_fea_mask = additional_fea_mask.to(device)
        additional_fea_embeds = None

        if self.additional_feat_embedding is not None:
            if num_add_tokens != 1:
                raise ValueError(
                    f"SHAP expects exactly 1 additional feature token, got {num_add_tokens}."
                )
            additional_feat_token = data["x"][-1:, :]  # [1, hid_dim]
            additional_fea_embeds = self.additional_fea_embeds
            if additional_fea_embeds.device != device:
                additional_fea_embeds = additional_fea_embeds.to(device)
            additional_fea_embeds = additional_fea_embeds.clone()
            additional_fea_embeds[:, 1, :] = additional_feat_token
        
        # If GraphSVX excluded nodes and we are in mask-only mode, remove from attention.
        excluded_nodes = data.get("excluded_nodes", None)
        use_mask_only = bool(data.get("use_mask_only", False))
        if use_mask_only and excluded_nodes is not None:
            excluded_nodes = torch.as_tensor(excluded_nodes, dtype=torch.long, device=graph_mask.device)
            if excluded_nodes.numel() > 0:
                num_atoms = atom_x.shape[0]
                atom_excluded = excluded_nodes[(excluded_nodes >= 0) & (excluded_nodes < num_atoms)]
                if atom_excluded.numel() > 0:
                    graph_mask[:, atom_excluded] = 0

                if self.additional_feat_embedding is not None:
                    add_excluded = excluded_nodes[(excluded_nodes >= num_atoms) & (excluded_nodes < num_atoms + num_add_tokens)] - num_atoms
                    if add_excluded.numel() > 0:
                        # additional_fea_mask = [additional_cls, additional_feature_token]
                        additional_fea_mask = additional_fea_mask.clone()
                        additional_fea_mask[:, 1] = 0


        # Add class token to graph embeddings
        cls_tokens = torch.zeros(len(self.num_atoms_per_crystal)).to(graph_embeds)  # [B]
        cls_embeds = self.cls_embeddings(cls_tokens[:, None, None])  # [B, 1, hid_dim]
        cls_mask = torch.ones(len(self.num_atoms_per_crystal), 1).to(graph_mask)  # [B, 1]
        
        graph_embeds = torch.cat([cls_embeds, graph_embeds], dim=1)  # [B, N+1, hid_dim]
        graph_mask = torch.cat([cls_mask, graph_mask], dim=1)  # [B, N+1]
        
        if self.additional_feat_embedding is not None:
          # Add token type embeddings (type=1 for additional features)
            additional_fea_embeds = additional_fea_embeds + self.token_type_embeddings(
                torch.ones_like(additional_fea_mask, device=device).long()
            )

        # Add token type embeddings (type=0 for graph tokens)
        graph_embeds = graph_embeds + self.token_type_embeddings(
            torch.zeros_like(graph_mask, device=device).long()
        )

        # Combine all embeddings
        if self.additional_feat_embedding is not None:
            combined_embeds = torch.cat(
                [graph_embeds, additional_fea_embeds], dim=1
            )  # [B, N+1+2, hid_dim]
            combined_mask = torch.cat(
                [graph_mask, additional_fea_mask], dim=1
            )  # [B, N+1+2]
        else:
            combined_embeds = graph_embeds  # [B, N+1, hid_dim]
            combined_mask = graph_mask  # [B, N+1]

        # Process through transformer
        x = combined_embeds
        attn_weights = []
        for blk in self.transformer.blocks:
            x = blk(x, mask=combined_mask)
            attn_weights.append(blk.attn_out)  # Store attention weights
        x = self.transformer.norm(x)
        
        # Split features
        if self.additional_feat_embedding is not None:
            graph_feats = x[:, :graph_embeds.shape[1]]  # [B, N+1, hid_dim]
            added_feats = x[:, graph_embeds.shape[1]:]  # [B, 2, hid_dim]
        else:
            graph_feats = x  # [B, N+1, hid_dim]
            added_feats = None

        # Pool CLS token (first token)
        cls_feats = self.pooler(x)  # [B, hid_dim]

        ret = {
            "cif_id": self.cif_id,
            "num_atoms_per_crystal": self.num_atoms_per_crystal,
            "crystal_atom_idx": self.crystal_atom_idx,
            "graph_feats": graph_feats,
            "added_feats": added_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],  # Unpooled CLS token
            "graph_mask": graph_mask,
        }

        if "regression" in self.current_tasks:
            logits = self.regression_head(ret["cls_feats"]) 
            # Decode the logits
            logits = logits.squeeze(-1)
            logits_decoded = self.normalizer.decode(logits)
            self.logits = logits_decoded

        if "classification" in self.current_tasks:
            logits_out = self.classification_head(ret["cls_feats"])
            if isinstance(logits_out, tuple):
                logits, _ = logits_out
            else:
                logits = logits_out
            self.logits = logits

        if "classification" in self.current_tasks and torch.is_tensor(self.logits):
            if self.logits.ndim == 1 or (self.logits.ndim == 2 and self.logits.shape[-1] == 1):
                logits = self.logits.squeeze(-1)
                probs_pos = torch.sigmoid(logits)
                if pred_idx == 0:
                    outputs = 1.0 - probs_pos
                elif pred_idx == 1:
                    outputs = probs_pos
                else:
                    raise ValueError(
                        f"Binary classification expects pred_idx 0 or 1, got {pred_idx}."
                    )
            else:
                probs = torch.softmax(self.logits, dim=-1)
                outputs = probs[:, pred_idx]
        elif torch.is_tensor(self.logits) and self.logits.ndim == 1:
            outputs = self.logits
        else:
            outputs = self.logits[:, pred_idx]
        if return_attn_weights and return_features:
            return outputs, attn_weights, ret
        if return_attn_weights:
            return outputs, attn_weights
        if return_features:
            return outputs, ret
        return outputs

    def build_adjacency_matrix(self, cif_file, add_reverse=True):
        """
        Build an adjacency matrix from a CIF file.
        Parameters:
            cif_file (str): Path to the CIF file.
            add_reverse (bool): Whether to add reverse edges (i.e., make the graph undirected).
        Returns:
            adj_matrix (ndarray): Adjacency matrix of the crystal structure.
        """
        parser = CifParser(cif_file)
        structure = parser.parse_structures(primitive=True)[0] 

        # Initialize neighbor finder
        nn_finder = CrystalNN()

        n_atoms = len(structure.sites)

        adj_matrix = torch.zeros((n_atoms, n_atoms), dtype=torch.float32)

        for idx in range(n_atoms):
            neighbors = nn_finder.get_nn_info(structure, idx)
            for neighbor in neighbors:
                neighbor_idx = neighbor['site_index']
                adj_matrix[idx, neighbor_idx] = 1.0
                adj_matrix[neighbor_idx, idx] = 1.0  # Ensure symmetry

        return adj_matrix

    def _pad_adjacency_for_tokens(self, adj_matrix, pad_tokens):
        if pad_tokens <= 0:
            return adj_matrix
        pad_rows = torch.zeros(
            pad_tokens,
            adj_matrix.shape[1],
            dtype=adj_matrix.dtype,
            device=adj_matrix.device,
        )
        adj_matrix = torch.cat([adj_matrix, pad_rows], dim=0)
        pad_cols = torch.zeros(
            adj_matrix.shape[0],
            pad_tokens,
            dtype=adj_matrix.dtype,
            device=adj_matrix.device,
        )
        adj_matrix = torch.cat([adj_matrix, pad_cols], dim=1)
        return adj_matrix

    def reverse_rebuild_and_mask_batch(self, reshaped_atom_fea, mask):
        """
        Reconstructs the original h_prime (node embeddings) by concatenating valid embeddings
        from each crystal in a batch.

        Parameters:
            reshaped_atom_fea (Tensor): Shape [B, max_crystal_atoms, out_dim].
            mask (Tensor): Boolean tensor of shape [B, max_crystal_atoms] indicating valid embeddings.

        Returns:
            h_prime (Tensor): Concatenated node embeddings of shape [total_atoms, out_dim],
                            where total_atoms = sum(valid tokens per crystal).
        """
        valid_embeddings = []
        for i in range(reshaped_atom_fea.shape[0]):
            valid = reshaped_atom_fea[i][mask[i]]
            valid_embeddings.append(valid)
        return torch.cat(valid_embeddings, dim=0)    
    
  
    def shap_vis(self,
                mode='dual',
                figsize=(15,15), 
                view_init=(25, 25),
                show_axis=True,
                show_colorbar=True,
                alpha=0.7,
                atomic_scale_factor=4,
                grid_scale_factor=1, 
                att_scale_factor=3,
                animate=True,
                shap_output=None,
                shap_output_path=None,
                zip_path_override=None,
                cleanup_extracted=False,
                 **kwargs):
        """
        Visualize SHAP values using the modified PatchVisualizer.
        """
        payload = self._resolve_shap_output(shap_output, shap_output_path)
        shap_values = payload["shap_values"] if payload is not None else self.shap_values
        cif_file = payload.get("cif_file", self.cif_file) if payload is not None else self.cif_file
        add_feat_num = int(payload.get("num_add_tokens", self.add_feat_num)) if payload is not None else self.add_feat_num
        num_atoms = payload.get("num_atoms") if payload is not None else getattr(self, "num_atoms", None)
        zip_path = zip_path_override
        if zip_path is None:
            zip_path = payload.get("zip_path") if payload is not None else getattr(self, "zip_path", None)

        if shap_values is None:
            raise ValueError("No SHAP values available to plot.")
        if torch.is_tensor(shap_values):
            if shap_values.ndim == 1:
                shap_values = shap_values.unsqueeze(0)
            shap_len = shap_values.numel()
        else:
            shap_values = np.asarray(shap_values)
            if shap_values.ndim == 1:
                shap_values = shap_values.reshape(1, -1)
            shap_len = shap_values.size

        if num_atoms is not None:
            num_atoms = int(num_atoms)
            if shap_len == num_atoms:
                add_feat_num = 0
            elif shap_len == num_atoms + add_feat_num:
                pass
            elif shap_len > num_atoms and add_feat_num == 0:
                add_feat_num = shap_len - num_atoms
            elif shap_len != num_atoms + add_feat_num:
                warnings.warn(
                    f"SHAP length {shap_len} does not match num_atoms={num_atoms} "
                    f"+ num_add_tokens={add_feat_num}. Plotting may be misaligned."
                )

        if cif_file is not None and not os.path.exists(cif_file) and zip_path:
            extracted_path, extracted_new = self.extract_cif_file_from_zip(zip_path, cif_file)
            if extracted_path is not None:
                cif_file = extracted_path
                if extracted_new:
                    self._cleanup_paths.append(extracted_path)

        patch_visualizer = PatchVisualizer(
            path_cif=cif_file,
            heatmap_graph=shap_values,
            uni_idx=[[i] for i in range(len(shap_values)-1)],
            figsize=figsize,
            view_init=view_init,
            show_axis=show_axis,
            show_colorbar=show_colorbar
        )
        if not mode == 'dual':
            patch_visualizer.draw_graph(
                                        minatt=shap_values.min(), 
                                        maxatt=shap_values.max(),
                                        alpha=alpha, 
                                        atomic_scale_factor=atomic_scale_factor, 
                                        grid_scale_factor=grid_scale_factor, 
                                        att_scale_factor=att_scale_factor,
                                        add_feat_num=add_feat_num,
                                        mode=mode,
                                        **kwargs)
        else:
            patch_visualizer.draw_graph_dual(
                                        minatt=shap_values.min(), 
                                        maxatt=shap_values.max(),
                                        alpha=alpha, 
                                        atomic_scale_factor=atomic_scale_factor, 
                                        grid_scale_factor=grid_scale_factor, 
                                        att_scale_factor=att_scale_factor,
                                        add_feat_num=add_feat_num,
                                        mode=mode,
                                        **kwargs)

        if animate and mode !='dual':
            animated_draw_graph = patch_visualizer.animate(
                patch_visualizer.draw_graph_dual,
                frame=360,    
                interval=20,  
                savefile="Crystal animation.mp4" 
            )

            animated_draw_graph(minatt=shap_values.min(), 
                                maxatt=shap_values.max(), 
                                alpha=0.7, 
                                add_feat_num=add_feat_num,mode=mode)

        elif animate and mode =='dual':
            anim_draw_graph = patch_visualizer.animate_dual(
            patch_visualizer.draw_graph_dual,
            frame=360,
            interval=20,
            savefile="Crystal_animation.mp4",
            fps=30
        )
            anim_draw_graph(
                minatt=shap_values.min(),
                maxatt=shap_values.max(),
                alpha=0.7,
                add_feat_num=add_feat_num
)
        if cleanup_extracted:
            self.cleanup_extracted_files()

    def att_heatmap_vs_shap_vis(self, **kwargs):
        """
        Visualize attention scores with SHAP values as a heatmap.
        """
        _,attn_weights = self.model.infer_out(self.data, self.data["pred_index"], configs=self.data["config"], return_attn_weights=True)
        heatmap = utils.get_heatmap(attn_weights, **kwargs)
        return heatmap

    def attn_shap_triptych_figure(
        self,
        *,
        graph_len: int = 500,
        sep_tokens: int = 1,
        add_feat_tokens: int = 1,
        add_feat_names: Optional[list[str]] = None,
        batch_idx: int = 0,
        save: bool = True,
        show: bool = True,
        close: bool = False,
        out_dir: str = "attn_shap_figures",
        filename: Optional[str] = None,
        dpi: int = 600,
        view_init: Tuple[float, float] = (25, 25),
        figsize: Tuple[float, float] = (14.0, 5.2),
        alpha: float = 1.0,
        show_element_legend: bool = True,
        atomic_scale_factor: float = 4.0,
        att_scale_factor: float = 1.0,
        grid_scale_factor: float = 1.0,
        panel_titles: Optional[Tuple[str, str, str, str]] = (
            "Original crystal",
            "Atom SHAPs",
            "Attention rollout",
            "Mismatch (impact ignored)",
        ),
        shap_output=None,
        shap_output_path=None,
    ):
        """
        Create a publication-style 3D triptych comparing SHAP vs attention vs mismatch.

        This is a convenience wrapper around `exctransformer.SHAP_visualization.utils.plot_attention_shap_triptych_3d`.
        It:
          1) Computes the attention rollout vector from transformer attention weights.
          2) Aligns SHAP and attention vectors to the CIF atom count for rendering.
          3) Saves high-DPI outputs by default.
        """
        payload = self._resolve_shap_output(shap_output, shap_output_path)
        if payload is not None and payload.get("attn_weights") is not None:
            attn_weights = payload["attn_weights"]
        else:
            if self.model is None:
                raise ValueError("No model loaded to compute attention weights.")
            _, attn_weights = self.model.infer_out(
                self.data,
                self.data["pred_index"],
                configs=self.data["config"],
                return_attn_weights=True,
            )
        attn_rollout = utils.get_heatmap(
            attn_weights,
            batch_idx=batch_idx,
            graph_len=graph_len,
            sep_tokens=sep_tokens,
            add_feat_tokens=add_feat_tokens,
            skip_cls=True,
        )

        # SHAP values can be returned as [1, N] or [N]; convert to a 1D vector.
        shap_source = payload["shap_values"] if payload is not None else self.shap_values
        shap_vec = shap_source.detach().cpu().flatten() if torch.is_tensor(shap_source) else torch.tensor(shap_source).flatten()
        cif_file = payload.get("cif_file", self.cif_file) if payload is not None else self.cif_file

        return utils.plot_attention_shap_triptych_3d(
            path_cif=cif_file,
            shap_values=shap_vec,
            attention_values=attn_rollout,
            save=save,
            show=show,
            close=close,
            out_dir=out_dir,
            filename=filename,
            dpi=dpi,
            view_init=view_init,
            figsize=figsize,
            alpha=alpha,
            show_element_legend=show_element_legend,
            atomic_scale_factor=atomic_scale_factor,
            att_scale_factor=att_scale_factor,
            grid_scale_factor=grid_scale_factor,
            panel_titles=panel_titles,
            add_feat_label="ADF",
            add_feat_names=add_feat_names,
            add_feat_tokens=add_feat_tokens,
            sep_tokens=sep_tokens,
        )

    def attn_shap_heatmaps_figure(
        self,
        *,
        graph_len: int = 500,
        sep_tokens: int = 1,
        add_feat_tokens: int = 1,
        add_feat_names: Optional[list[str]] = None,
        batch_idx: int = 0,
        save: bool = True,
        show: bool = True,
        close: bool = False,
        out_dir: str = "attn_shap_figures",
        filename: Optional[str] = None,
        dpi: int = 600,
        figsize: Tuple[float, float] = (12.8, 4.6),
        annotate: str = "auto",
        max_annotate_tokens: int = 120,
        panel_titles: Optional[Tuple[str, str, str]] = None,
        suptitle: Optional[str] = None,
        shap_output=None,
        shap_output_path=None,
    ):
        """
        Create a publication-style 2D heatmap triptych (SHAP / attention / mismatch).

        This is often easier to read than 3D for token-level alignment because each token
        is a single square cell with an element label and optional value annotations.
        """
        payload = self._resolve_shap_output(shap_output, shap_output_path)
        if payload is not None and payload.get("attn_weights") is not None:
            attn_weights = payload["attn_weights"]
        else:
            if self.model is None:
                raise ValueError("No model loaded to compute attention weights.")
            _, attn_weights = self.model.infer_out(
                self.data,
                self.data["pred_index"],
                configs=self.data["config"],
                return_attn_weights=True,
            )
        attn_rollout = utils.get_heatmap(
            attn_weights,
            batch_idx=batch_idx,
            graph_len=graph_len,
            sep_tokens=sep_tokens,
            add_feat_tokens=add_feat_tokens,
            skip_cls=True,
        )

        shap_source = payload["shap_values"] if payload is not None else self.shap_values
        shap_vec = (
            shap_source.detach().cpu().flatten()
            if torch.is_tensor(shap_source)
            else torch.tensor(shap_source).flatten()
        )
        cif_file = payload.get("cif_file", self.cif_file) if payload is not None else self.cif_file

        return utils.plot_attention_shap_triptych_2d(
            path_cif=cif_file,
            shap_values=shap_vec,
            attention_values=attn_rollout,
            save=save,
            show=show,
            close=close,
            out_dir=out_dir,
            filename=filename,
            dpi=dpi,
            figsize=figsize,
            annotate=annotate,
            max_annotate_tokens=max_annotate_tokens,
            add_feat_label="ADF",
            add_feat_names=add_feat_names,
            add_feat_tokens=add_feat_tokens,
            panel_titles=panel_titles,
            suptitle=suptitle,
        )

    def extract_cif_file_from_zip(self, zip_file, file_name, extract_to=None):
        if extract_to is None:
            extract_to = os.getcwd()
        
        extract_name = os.path.basename(file_name)
        extract_path = os.path.join(extract_to, extract_name)
        if os.path.exists(extract_path):
            return extract_path, False
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            members = zip_ref.namelist()
            target = None
            if extract_name in members:
                target = extract_name
            else:
                for name in members:
                    if os.path.basename(name) == extract_name:
                        target = name
                        break
            if target is None:
                warnings.warn(f"CIF {extract_name} not found in zip {zip_file}.")
                return None, False
            zip_ref.extract(target, extract_to)
            extracted = os.path.join(extract_to, target)
            return extracted, True

    def cleanup_extracted_files(self):
        """
        Remove any CIFs extracted from zip archives during this run.
        """
        for path in self._cleanup_paths:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
        self._cleanup_paths = []


def build_samples_from_dir(
    data_dir,
    cif_dir=None,
    zip_path=None,
    max_samples=None,
    model_kwargs=None,
    filter_stems=None,
    h5_glob="*.h5",
    cif_ext=".cif",
):
    """
    Build the sample dictionaries needed by `explain_batch` from a data folder.

    Args:
        data_dir (str): Folder containing per-sample H5 files.
        cif_dir (str | None): Folder containing CIF files (if not using a zip).
        zip_path (str | None): Path to a zip file containing CIFs.
        max_samples (int | None): Limit the number of samples returned.
        model_kwargs (dict | None): Extra model kwargs (e.g., num_targets) to attach per sample.
        filter_stems (set[str] | None): Optional stem whitelist for selecting H5/CIF pairs.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    h5_files = sorted([p for p in data_path.rglob(h5_glob) if p.is_file()])
    if not h5_files:
        warnings.warn(f"No H5 files found in {data_dir} with glob {h5_glob}.")
        return []

    zip_names = None
    if zip_path is not None:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_names = {
                Path(name).name: name
                for name in zip_ref.namelist()
                if name.lower().endswith(cif_ext.lower())
            }
        if not zip_names:
            warnings.warn(f"No CIF files found in zip {zip_path}.")

    if cif_dir is not None:
        cif_root = Path(cif_dir)
    else:
        sibling_cif_root = data_path.parent / "cifs"
        cif_root = sibling_cif_root if sibling_cif_root.exists() else data_path
    stems_filter = set(filter_stems) if filter_stems else None
    samples = []
    model_kwargs = dict(model_kwargs or {})
    for h5_path in h5_files:
        stem = h5_path.stem
        if stems_filter is not None and stem not in stems_filter:
            continue
        cif_name = f"{stem}{cif_ext}"
        if zip_path is not None:
            if zip_names is not None and cif_name not in zip_names:
                warnings.warn(f"CIF {cif_name} not found in zip {zip_path}; skipping.")
                continue
            cif_file = cif_name
        else:
            cif_file = cif_root / cif_name
            if not cif_file.exists():
                warnings.warn(f"CIF file not found: {cif_file}; skipping.")
                continue
            cif_file = str(cif_file)

        sample = {
            "cif_id_h5": str(h5_path),
            "cif_file": cif_file,
            "zip_path": zip_path,
            "file_stem": stem,
        }
        if model_kwargs:
            sample["model_kwargs"] = dict(model_kwargs)
        samples.append(sample)

        if max_samples is not None and len(samples) >= max_samples:
            break

    return samples


def filter_samples_by_predict_csv(
    samples,
    predict_csv_path,
    pred_idx=None,
    max_samples=None,
    criteria=None,
    criteria_fn=None,
):
    """
    Filter samples using a predict_results_*.csv file and optional criteria.

    Args:
        samples (list[dict]): Batch samples with cif_id_h5 or file_stem set.
        predict_csv_path (str): Path to predict_results_*.csv.
        pred_idx (int | None): Target index to align with pred_#/target_# columns.
        max_samples (int | None): Max number of samples to return.
        criteria (dict | None): Optional filter rules for predictions/targets.
            Supported keys: pred_min, pred_max, target_min, target_max,
            abs_error_max, error_min, error_max, pred_range, target_range.
        criteria_fn (callable | None): Custom predicate (row, pred, target) -> bool.
    """
    if not predict_csv_path:
        return samples, len(samples)

    criteria = dict(criteria or {})
    sample_map = {}
    for sample in samples:
        stem = sample.get("file_stem")
        if not stem:
            stem = Path(sample["cif_id_h5"]).stem
        sample_map.setdefault(stem, []).append(sample)

    def _to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _pick_cols(row):
        if "prediction" in row:
            return "prediction", ("target" if "target" in row else None)
        pred_candidates = []
        target_candidates = []
        if isinstance(pred_idx, int):
            pred_candidates.extend([f"pred_{pred_idx}", f"pred_{pred_idx + 1}"])
            target_candidates.extend([f"target_{pred_idx}", f"target_{pred_idx + 1}"])
        pred_candidates.extend([k for k in row.keys() if k.startswith("pred_")])
        target_candidates.extend([k for k in row.keys() if k.startswith("target_")])
        pred_col = next((c for c in pred_candidates if c in row), None)
        target_col = next((c for c in target_candidates if c in row), None)
        return pred_col, target_col

    def _passes(pred_val, target_val, row):
        if criteria_fn is not None:
            return bool(criteria_fn(row, pred_val, target_val))
        if criteria is None:
            return True
        pred_range = criteria.get("pred_range")
        target_range = criteria.get("target_range")
        if pred_range is not None:
            criteria.setdefault("pred_min", pred_range[0])
            criteria.setdefault("pred_max", pred_range[1])
        if target_range is not None:
            criteria.setdefault("target_min", target_range[0])
            criteria.setdefault("target_max", target_range[1])
        pred_min = criteria.get("pred_min")
        pred_max = criteria.get("pred_max")
        target_min = criteria.get("target_min")
        target_max = criteria.get("target_max")
        abs_error_max = criteria.get("abs_error_max")
        error_min = criteria.get("error_min")
        error_max = criteria.get("error_max")
        if pred_min is not None and (pred_val is None or pred_val < pred_min):
            return False
        if pred_max is not None and (pred_val is None or pred_val > pred_max):
            return False
        if target_min is not None and (target_val is None or target_val < target_min):
            return False
        if target_max is not None and (target_val is None or target_val > target_max):
            return False
        if target_val is not None and pred_val is not None:
            err = pred_val - target_val
            if abs_error_max is not None and abs(err) > abs_error_max:
                return False
            if error_min is not None and err < error_min:
                return False
            if error_max is not None and err > error_max:
                return False
        return True

    matched = []
    matched_count = 0
    with open(predict_csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cif_id = row.get("cif_id")
            if not cif_id:
                continue
            stem = Path(cif_id).stem
            sample_list = sample_map.get(stem)
            if not sample_list:
                continue
            pred_col, target_col = _pick_cols(row)
            pred_val = _to_float(row.get(pred_col)) if pred_col else None
            target_val = _to_float(row.get(target_col)) if target_col else None
            if _passes(pred_val, target_val, row):
                matched_count += 1
                matched.append(sample_list.pop(0))
                if max_samples is not None and len(matched) >= max_samples:
                    break
    return matched, matched_count


def explain_batch(
    samples,
    model_path,
    pred_idx,
    out_dir="shap_outputs",
    gpu="auto",
    include_additional_tokens=False,
    args_g="WLS",
    args_hv="auto",
    args_coal="NewSmarterSeparate",
    use_mask_only=False,
    compare_models=False,
    compare_model_list=None,
    include_surrogate=True,
    include_attention=False,
    coalition_size="small",
    S_override=None,
    verbose=True,
    graphsvx_info=True,
    log_path=None,
    predict_csv_path=None,
    max_explain_samples=None,
    prediction_criteria=None,
    criteria_fn=None,
    return_valid_sample_count=True,
    sample_ids=None,
    sample_indices=None,
    continue_from_previous=True,
    **kwargs,
):
    """
    Compute and save SHAP explanations for a batch of samples.

    Args:
        samples (list[dict]): Each dict should include "cif_id_h5" and "cif_file".
            Optional keys: "pred_idx", "zip_path", "file_stem".
        gpu (bool | str | None): Use GPU for model inference. Set True/False or
            "auto" to enable when CUDA is available (default).
        args_hv (str): Strategy for task head selection ("graph_regression",
            "graph_classification", or "auto").
        graphsvx_info (bool): Print GraphSVX surrogate diagnostics (r2, etc.).
        S_override (int | None): Optional override for GraphSVX S (max coalition size).
        log_path (str | bool | None): Optional JSONL log file path. Use True to
            write to out_dir/batch_explanations_log.jsonl.
        return_valid_sample_count (bool): When True, return (outputs, matched_count)
            where matched_count is the number of samples meeting criteria.
        sample_ids (list[str] | None): Optional list of H5 stems/filenames to explain.
        sample_indices (list[int] | int | None): Optional indices into the provided
            samples list (supports negatives) to explain.
        continue_from_previous (bool): When True, skip samples already explained
            based on existing output files or log entries.
    """
    def _select_samples(samples_list, ids=None, indices=None):
        if ids:
            ids = {str(v).lower() for v in ids}
        else:
            ids = set()
        if indices is None:
            norm_indices = []
        elif isinstance(indices, (list, tuple, set)):
            norm_indices = list(indices)
        else:
            norm_indices = [indices]
        selected = []
        seen = set()
        total_len = len(samples_list)

        for raw_idx in norm_indices:
            try:
                idx = int(raw_idx)
            except (TypeError, ValueError):
                continue
            if idx < 0:
                idx = total_len + idx
            if 0 <= idx < total_len:
                sample = samples_list[idx]
                key = (sample.get("cif_id_h5"), sample.get("cif_file"))
                if key not in seen:
                    selected.append(sample)
                    seen.add(key)

        if ids:
            for sample in samples_list:
                ident_candidates = [
                    str(sample.get("file_stem") or ""),
                    Path(sample.get("cif_id_h5", "")).stem,
                    Path(sample.get("cif_id_h5", "")).name,
                    str(sample.get("cif_id_h5", "")),
                ]
                if any(c and c.lower() in ids for c in ident_candidates):
                    key = (sample.get("cif_id_h5"), sample.get("cif_file"))
                    if key not in seen:
                        selected.append(sample)
                        seen.add(key)

        return selected if ids or norm_indices else samples_list

    samples = _select_samples(samples, sample_ids, sample_indices)
    matched_count = len(samples)
    if (sample_ids or sample_indices) and matched_count == 0:
        if verbose:
            print("No samples matched requested ids/indices; skipping batch explanation.")
        return ([], 0) if return_valid_sample_count else []

    if predict_csv_path:
        samples, matched_count = filter_samples_by_predict_csv(
            samples=samples,
            predict_csv_path=predict_csv_path,
            pred_idx=pred_idx,
            max_samples=max_explain_samples,
            criteria=prediction_criteria,
            criteria_fn=criteria_fn,
        )
        if verbose and matched_count == 0:
            print("No samples matched prediction criteria; skipping batch explanation.")
        elif verbose and not samples:
            print("No samples selected for explanation after filtering.")
        elif verbose:
            print(f"Matched {matched_count} sample(s); explaining {len(samples)}.")
    use_gpu = _resolve_gpu_setting(gpu)
    if len(samples) > 1 and not use_gpu:
        warnings.warn(
            "Batch SHAP explanations on CPU can be very slow. "
            "Consider enabling GPU with gpu=True."
        )
    os.makedirs(out_dir, exist_ok=True)
    if log_path is True:
        log_path = os.path.join(out_dir, "batch_explanations_log.jsonl")
    if log_path:
        log_path = str(log_path)
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    log_handle = None

    def _load_explained_entries(log_file):
        explained_paths = set()
        explained_keys = set()
        if not log_file or not os.path.exists(log_file):
            return explained_paths, explained_keys
        try:
            with open(log_file, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("record_type") == "batch_summary":
                        continue
                    out_path = entry.get("shap_output_path")
                    if out_path:
                        explained_paths.add(os.path.abspath(out_path))
                    key = (
                        entry.get("cif_id_h5"),
                        entry.get("file_stem"),
                        entry.get("pred_idx"),
                    )
                    explained_keys.add(key)
        except OSError:
            return explained_paths, explained_keys
        return explained_paths, explained_keys

    def _expected_output_path(sample, default_pred_idx, output_dir):
        stem = sample.get("file_stem") or Path(sample.get("cif_id_h5", "")).stem
        sample_pred_idx = sample.get("pred_idx", default_pred_idx)
        return os.path.join(output_dir, f"{stem}_pred{sample_pred_idx}_shap.pt"), stem, sample_pred_idx

    if continue_from_previous and out_dir:
        explained_paths, explained_keys = _load_explained_entries(log_path)
        remaining = []
        skipped = 0
        for sample in samples:
            out_path, stem, sample_pred_idx = _expected_output_path(sample, pred_idx, out_dir)
            key = (sample.get("cif_id_h5"), sample.get("file_stem"), sample_pred_idx)
            if os.path.exists(out_path) or os.path.abspath(out_path) in explained_paths or key in explained_keys:
                skipped += 1
                continue
            remaining.append(sample)
        if verbose and skipped:
            print(f"Skipping {skipped} sample(s) already explained; {len(remaining)} remaining.")
        samples = remaining
        matched_count = len(samples)

    start_time = time.time()
    sum_r2 = 0.0
    sum_weighted_r2 = 0.0
    count_r2 = 0
    count_weighted_r2 = 0
    compare_sums = {}
    compare_counts = {}
    outputs = []
    total = len(samples)
    if log_path:
        log_handle = open(log_path, "a", encoding="utf-8")
    try:
        for idx, sample in enumerate(samples, start=1):
            sample_start = time.time()
            cif_id_h5 = sample["cif_id_h5"]
            cif_file = sample["cif_file"]
            sample_pred_idx = sample.get("pred_idx", pred_idx)
            zip_path = sample.get("zip_path", None)
            file_stem = sample.get("file_stem", None)
            local_kwargs = dict(kwargs)
            local_kwargs["verbose"] = False
            sample_model_kwargs = sample.get("model_kwargs", None)
            if sample_model_kwargs:
                local_kwargs.update(sample_model_kwargs)
            if "num_targets" not in local_kwargs or local_kwargs.get("num_targets") in (None, 0):
                inferred = _infer_num_targets_from_h5(cif_id_h5)
                if inferred is not None:
                    local_kwargs["num_targets"] = inferred
            dataset_path = sample.get("dataset_path", local_kwargs.get("dataset_path"))
            if not dataset_path:
                dataset_path = str(Path(cif_id_h5).parent)
            local_kwargs["dataset_path"] = dataset_path
            visualizer = None
            stem_label = file_stem or Path(cif_id_h5).stem
            try:
                visualizer = SHAPVisualizer(
                    cif_id_h5=cif_id_h5,
                    cif_file=cif_file,
                    model_path=model_path,
                    pred_idx=sample_pred_idx,
                    zip_path=zip_path,
                    include_additional_tokens=include_additional_tokens,
                    gpu=use_gpu,
                    args_g=args_g,
                    args_hv=args_hv,
                    args_coal=args_coal,
                    use_mask_only=use_mask_only,
                    compare_models=compare_models,
                    compare_model_list=compare_model_list,
                    coalition_size=coalition_size,
                    S_override=S_override,
                    graphsvx_info=graphsvx_info,
                    compute_shap=False,
                    **local_kwargs,
                )
                num_samples = None
                S = None
                if verbose:
                    add_players = int(visualizer.add_feat_num) if include_additional_tokens else 0
                    add_players_h5 = int(getattr(visualizer, "h5_add_tokens", 0)) if include_additional_tokens else 0
                    num_atoms = int(visualizer.graph_embeds.shape[0])
                    player_count = num_atoms + add_players
                    num_samples, S = visualizer._resolve_coalition_size(
                        coalition_size, player_count, default_policy="small"
                    )
                    if str(args_coal).strip() == "All":
                        all_cap = int(
                            getattr(visualizer.explainer, "all_coalition_cap", 20000)
                        )
                        num_samples = min(all_cap, 2**player_count)
                    if S_override is not None:
                        try:
                            s_override_val = int(S_override)
                        except (TypeError, ValueError):
                            s_override_val = None
                        if s_override_val is not None:
                            if s_override_val < 1:
                                s_override_val = 1
                            S = min(s_override_val, max(1, player_count - 1))
                    line = (
                        f"{_format_progress(idx, total)} current={stem_label} "
                        f"scanned={idx} remaining={total - idx} "
                        f"num_atoms={num_atoms} add_tokens={add_players} "
                        f"h5_add_tokens={add_players_h5} players={player_count} "
                        f"coalition_builder={args_coal} coalition_size={coalition_size} "
                        f"num_samples={num_samples} S={S} "
                        f"save_output={True}"
                    )
                    print(line, end="\r", flush=True)
                    print(
                        f"{_format_progress(idx, total)} GraphSVX Explanation initialized ".ljust(60),
                        flush=True,
                    )
                shap_start = time.time()
                visualizer.compute_shap(
                    include_additional_tokens=include_additional_tokens,
                    args_g=args_g,
                    args_hv=args_hv,
                    args_coal=args_coal,
                    use_mask_only=use_mask_only,
                    compare_models=compare_models,
                    compare_model_list=compare_model_list,
                    coalition_size=coalition_size,
                    S_override=S_override,
                    graphsvx_info=graphsvx_info,
                    seed=local_kwargs.get("seed"),
                )
                shap_end = time.time()
                if verbose:
                    elapsed = shap_end - shap_start
                    req_samples = getattr(
                        visualizer.explainer, "last_requested_num_samples", num_samples
                    )
                    eff_samples = getattr(
                        visualizer.explainer, "last_effective_num_samples", num_samples
                    )
                    print(
                        (
                            f"{_format_progress(idx, total)} GraphSVX completed in {elapsed:.1f}s "
                            f"num_samples_requested={req_samples} "
                            f"num_samples_effective={eff_samples}"
                        ).ljust(60)
                    )
                out_path = visualizer.save_shap_output(
                    out_dir=out_dir,
                    file_stem=file_stem,
                    include_surrogate=include_surrogate,
                    include_attention=include_attention,
                )
                outputs.append(out_path)
                r2_val = _to_float(getattr(visualizer, "explanation_r2", None))
                weighted_r2_val = _to_float(getattr(visualizer, "explanation_weighted_r2", None))
                compare_metrics = getattr(visualizer, "compare_results", None)
                _accumulate_compare_metrics(compare_metrics, compare_sums, compare_counts)
                if r2_val is not None:
                    sum_r2 += r2_val
                    count_r2 += 1
                if weighted_r2_val is not None:
                    sum_weighted_r2 += weighted_r2_val
                    count_weighted_r2 += 1
                avg_r2 = (sum_r2 / count_r2) if count_r2 else None
                avg_weighted_r2 = (sum_weighted_r2 / count_weighted_r2) if count_weighted_r2 else None
                sample_end = time.time()
                if verbose:
                    avg_line = (
                        f"{_format_progress(idx, total)} "
                        f"avg_explanation_r2={avg_r2} "
                        f"avg_explanation_weighted_r2={avg_weighted_r2}"
                    )
                    print(avg_line.ljust(60))
                if log_handle:
                    log_entry = {
                        "sample_index": idx,
                        "total_samples": total,
                        "cif_id_h5": cif_id_h5,
                        "cif_file": cif_file,
                        "zip_path": zip_path,
                        "file_stem": file_stem,
                        "pred_idx": sample_pred_idx,
                        "shap_output_path": out_path,
                        "compare_models": compare_models,
                        "compare_model_list": compare_model_list,
                        "compare_model_metrics": _coerce_jsonable(getattr(visualizer, "compare_results", None)),
                        "explanation_r2": r2_val,
                        "explanation_weighted_r2": weighted_r2_val,
                        "avg_explanation_r2": avg_r2,
                        "avg_explanation_weighted_r2": avg_weighted_r2,
                        "explain_config": _coerce_jsonable(getattr(visualizer, "explain_config", None)),
                        "sample_started_at": sample_start,
                        "sample_finished_at": sample_end,
                        "sample_elapsed_s": sample_end - sample_start,
                        "batch_elapsed_s": sample_end - start_time,
                    }
                    log_handle.write(json.dumps(log_entry, separators=(",", ":")) + "\n")
                    log_handle.flush()
            except Exception as exc:
                sample_end = time.time()
                if verbose:
                    print(
                        f"{_format_progress(idx, total)} skipped {stem_label}: {exc}".ljust(60),
                        flush=True,
                    )
                if log_handle:
                    log_entry = {
                        "record_type": "sample_error",
                        "sample_index": idx,
                        "total_samples": total,
                        "cif_id_h5": cif_id_h5,
                        "cif_file": cif_file,
                        "zip_path": zip_path,
                        "file_stem": file_stem,
                        "pred_idx": sample_pred_idx,
                        "error": str(exc),
                        "sample_started_at": sample_start,
                        "sample_finished_at": sample_end,
                        "sample_elapsed_s": sample_end - sample_start,
                        "batch_elapsed_s": sample_end - start_time,
                    }
                    log_handle.write(json.dumps(log_entry, separators=(",", ":")) + "\n")
                    log_handle.flush()
                continue
            finally:
                if visualizer is not None:
                    visualizer.cleanup_extracted_files()
    finally:
        if log_handle:
            batch_summary = _build_batch_summary_record(
                total_samples=total,
                sum_r2=sum_r2,
                count_r2=count_r2,
                sum_weighted_r2=sum_weighted_r2,
                count_weighted_r2=count_weighted_r2,
                compare_sums=compare_sums,
                compare_counts=compare_counts,
                batch_elapsed_s=time.time() - start_time,
            )
            log_handle.write(json.dumps(batch_summary, separators=(",", ":")) + "\n")
            log_handle.flush()
            log_handle.close()
    if total and verbose:
        avg_r2 = (sum_r2 / count_r2) if count_r2 else None
        avg_weighted_r2 = (sum_weighted_r2 / count_weighted_r2) if count_weighted_r2 else None
        print(f"Batch averages: explanation_r2={avg_r2} weighted_r2={avg_weighted_r2} samples={total}")
        print(f"{_format_progress(total, total)} complete".ljust(60))
    if return_valid_sample_count:
        return outputs, matched_count
    return outputs


def _infer_num_targets_from_h5(h5_path):
    try:
        with h5py.File(h5_path, "r") as f:
            sample = safe_load_flat_sample(f, file_identifier=h5_path)
    except Exception:
        return None

    if not sample or "target" not in sample:
        return None
    target = sample["target"]
    if torch.is_tensor(target):
        shape = tuple(target.shape)
    else:
        target = np.asarray(target)
        shape = tuple(target.shape)

    if len(shape) == 0:
        return 1
    if len(shape) == 1:
        return int(shape[0])
    if len(shape) >= 2:
        return int(shape[-1]) if shape[-1] != 1 else int(shape[0])
    return None


def _format_progress(current, total, width=12):
    total = max(total, 1)
    current = max(0, min(current, total))
    filled = int(round(width * (current / total)))
    filled = max(0, min(filled, width))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total}"


def _infer_add_tokens_from_h5(sample):
    if not isinstance(sample, dict):
        return 0
    add_feats = sample.get("additional_features")
    if add_feats is None:
        return 0
    if torch.is_tensor(add_feats):
        shape = add_feats.shape
    else:
        shape = np.asarray(add_feats).shape
    if len(shape) == 0:
        return 0
    if len(shape) == 1:
        return int(shape[0])
    return int(shape[1])
