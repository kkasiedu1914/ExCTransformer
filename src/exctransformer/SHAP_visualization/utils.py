import numpy as np
import h5py
import copy
import warnings
import os
from itertools import product

from contextlib import nullcontext
from functools import lru_cache
import pytorch_lightning as pl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pymatgen.io.cif import CifParser
from pymatgen.io.ase import AseAtomsAdaptor
from ase.build import make_supercell
from collections.abc import Iterable
import math
import torch

import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
from matplotlib.text import Text
from ase.data import chemical_symbols

from exctransformer import __root_dir__
from exctransformer.config import config
from exctransformer.SHAP_visualization import cpk_colors
from exctransformer.model_modules.module import Module
from exctransformer.datamodule.batch_data import safe_load_flat_sample, collate_graphs
from exctransformer.utils.configchecks import ConfigurationError, set_model_load_path

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union


def _freeze_cache_value(value):
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze_cache_value(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_cache_value(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_cache_value(v) for v in value))
    return value


def _thaw_cache_value(value):
    if isinstance(value, tuple):
        if all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            for item in value
        ):
            return {key: _thaw_cache_value(val) for key, val in value}
        return [_thaw_cache_value(item) for item in value]
    return value


def _torch_load_checkpoint(path):
    if not path:
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return None


def _safe_globals_context():
    # Allow known-safe globals during weights_only loads (PyTorch 2.6+).
    try:
        from sacred.config.custom_containers import ReadOnlyDict
    except Exception:
        return nullcontext()
    try:
        from torch.serialization import safe_globals
    except Exception:
        safe_globals = None
    try:
        from torch.serialization import add_safe_globals
    except Exception:
        add_safe_globals = None
    if safe_globals is not None:
        return safe_globals([ReadOnlyDict])
    if add_safe_globals is not None:
        try:
            add_safe_globals([ReadOnlyDict])
        except Exception:
            return nullcontext()
    return nullcontext()


def _extract_hparams_from_checkpoint(checkpoint):
    for key in ("hyper_parameters", "hparams", "config", "cfg"):
        payload = checkpoint.get(key)
        if payload is None:
            continue
        if isinstance(payload, dict):
            return dict(payload)
        payload_dict = getattr(payload, "__dict__", None)
        if isinstance(payload_dict, dict):
            return dict(payload_dict)
    return None


def _extract_seed_from_checkpoint(checkpoint):
    payload = _extract_hparams_from_checkpoint(checkpoint)
    if not payload:
        return None
    seed_val = payload.get("seed")
    if isinstance(seed_val, bool):
        return None
    if isinstance(seed_val, (int, float)):
        return int(seed_val)
    return None


def _detect_seed_from_checkpoint(model_path):
    if not model_path:
        return None
    resolved_path = set_model_load_path(model_path) or model_path
    if not os.path.isfile(resolved_path):
        return None
    checkpoint = _torch_load_checkpoint(resolved_path)
    if checkpoint is None:
        return None
    if not isinstance(checkpoint, dict):
        return None
    return _extract_seed_from_checkpoint(checkpoint)


def load_model(model_path, use_checkpoint_hparams=False, **kwargs):
    frozen_kwargs = tuple(sorted((key, _freeze_cache_value(val)) for key, val in kwargs.items()))
    return _load_model_cached(model_path, use_checkpoint_hparams, frozen_kwargs)


@lru_cache(maxsize=None)
def _load_model_cached(model_path, use_checkpoint_hparams=False, frozen_kwargs=()):
    """
    Loads and returns the model from the specified path using the provided configuration settings.
    
    Args:
        model_path (str): Path to the trained model.
        use_checkpoint_hparams (bool): If True, restore the Module via
            `load_from_checkpoint` so saved hparams (e.g., mean/std) are used.
    
    Returns:
        model: A PyTorch Lightning model set to evaluation mode and loaded on CPU.
    """
    kwargs = {key: _thaw_cache_value(val) for key, val in frozen_kwargs}

    # Retrieve default configuration and update parameters
    base_config = config()
    for key in kwargs.keys():
        if key not in base_config:
            raise ConfigurationError(f"{key} is not in configuration.")

    config_ = dict(base_config)
    config_.update(kwargs)
    resolved_path = None
    checkpoint_hparams = None
    if use_checkpoint_hparams:
        resolved_path = set_model_load_path(model_path)
        if not resolved_path:
            raise ConfigurationError(
                "model_path must be a checkpoint (.ckpt) or a known preset name."
            )
        checkpoint = _torch_load_checkpoint(resolved_path)
        if isinstance(checkpoint, dict):
            checkpoint_hparams = _extract_hparams_from_checkpoint(checkpoint)
        if isinstance(checkpoint_hparams, dict):
            merged = dict(base_config)
            merged.update({k: v for k, v in checkpoint_hparams.items() if k in merged})
            merged.update(kwargs)
            config_ = merged
    if "seed" not in kwargs or kwargs.get("seed") is None:
        detected_seed = _detect_seed_from_checkpoint(model_path)
        if detected_seed is not None:
            config_["seed"] = detected_seed
    config_["per_gpu_batchsize"] = 1
    config_["load_path"] = model_path
    config_["test_only"] = True
    config_["log_dir"] = "result_visualization"

    configs = { "max_crystal_atoms": config_["max_crystal_atoms"],
                "out_dim": config_["out_dim"],
    }

    # Set the random seed for reproducibility
    pl.seed_everything(config_["seed"])

    if use_checkpoint_hparams:
        # Restore saved hparams (mean/std, etc.) from checkpoint and
        # supply defaults for any missing keys (e.g., no_labels).
        try:
            with _safe_globals_context():
                model = Module.load_from_checkpoint(
                    resolved_path,
                    map_location="cpu",
                    **config_,
                )
        except Exception as exc:
            checkpoint = _torch_load_checkpoint(resolved_path)
            if not isinstance(checkpoint, dict):
                raise ConfigurationError(
                    f"Failed to load checkpoint: {resolved_path}"
                ) from exc
            model = Module(config_)
            state_dict = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=False)
        cfg = getattr(model, "config", None)
        if cfg is None:
            cfg = dict(getattr(model, "hparams", {}))
        configs = {
            "max_crystal_atoms": cfg.get("max_crystal_atoms", config_["max_crystal_atoms"]),
            "out_dim": cfg.get("out_dim", config_["out_dim"]),
        }
    else:
        # Instantiate, evaluate, and move the model to CPU
        model = Module(config_)
        configs = {
            "max_crystal_atoms": config_["max_crystal_atoms"],
            "out_dim": config_["out_dim"],
        }

    model.eval()
    model.to("cpu")
    return model, configs

@lru_cache(maxsize=None)
def load_h5_file(cif_id_file_path):
    """
    Loads an H5 file corresponding to the given cif_id_file_path and returns the collated graph sample.
    
    Args:
        cif_id_file_path (str): Relative path to the CIF file.
    
    Returns:
        The collated graph sample if successful; otherwise, None.
    """
    file_path = cif_id_file_path
    try:
        with h5py.File(file_path, 'r') as f:
            sample = safe_load_flat_sample(f, file_identifier=file_path)
        return collate_graphs([sample])
    except Exception as e:
        warnings.warn(f"Error loading file {file_path}: {e}")
        return None


def load_shap_output(path):
    """
    Load a serialized SHAP output payload saved by SHAPVisualizer.
    """
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        warnings.warn(f"Error loading SHAP output {path}: {exc}")
        return None
    if isinstance(payload, dict):
        payload = dict(payload)
        payload_dir = Path(path).expanduser().resolve().parent
        cif_file = payload.get("cif_file")
        if cif_file:
            cif_path = Path(cif_file).expanduser()
            if not cif_path.exists():
                relative_candidate = payload_dir / cif_file
                basename_candidate = payload_dir / cif_path.name
                if relative_candidate.exists():
                    payload["cif_file"] = str(relative_candidate.resolve())
                elif basename_candidate.exists():
                    payload["cif_file"] = str(basename_candidate.resolve())
    return payload
    

# remaining functions were obtained or based on : https://github.com/hspark1212/MOFTransformer

def cuboid_data(position, color=None, num_patches=(6, 6, 6), lattice=None):
    """
    Get cuboid plain data from position and size data
    parameters:
        :param position: <list/tuple> patch positions => [x, y, z]
        :param color: <list/tuple> colors => [r, g, b, w]
        :param num_patches: number of patches in each axis (default : (6, 6, 6))
        :param lattice: <np.ndarray> p_lattice vector for unit p_lattice
        :return: <tuple> (list of plain vector, list of color vector)
    """
    if isinstance(num_patches, (tuple, list)):
        num_patches = np.array(num_patches)
    elif not isinstance(num_patches, np.ndarray):
        raise TypeError(f"num_patches must be tuple or list, not {type(num_patches)}")

    bound = np.array([[0, 1] for _ in range(3)]) + np.array(position)[:, np.newaxis]
    vertex = np.array(list(product(*bound)))
    plane_ls = []
    for i, (dn, up) in enumerate(bound):
        plane1 = np.matmul(vertex[vertex[:, i] == dn], lattice / num_patches)
        plane2 = np.matmul(vertex[vertex[:, i] == up], lattice / num_patches)

        plane_ls.append(plane1)
        plane_ls.append(plane2)

    plane_ls = np.array(plane_ls).astype("float")
    plane_ls[:, [0, 1], :] = plane_ls[:, [1, 0], :]

    color_ls = np.repeat(color[np.newaxis, :], 6, axis=0)

    return plane_ls, color_ls
    
def get_shap_color(value, vmin, vmax):
    """
    Map a SHAP value to a color.
    Negative values interpolate between gray and blue.
    Positive values interpolate between gray and red.
    
    Parameters:
      value (float): The SHAP value.
      vmin (float): The minimum SHAP value (should be negative).
      vmax (float): The maximum SHAP value (should be positive).
      
    Returns:
      color (tuple): RGB color tuple.
    """
    # Define base colors in RGB
    gray = np.array([0.5, 0.5, 0.5])
    red = np.array([1.0, 0.0, 0.0])
    blue = np.array([0.0, 0.0, 1.0])
    
    if value >= 0:
        # Scale value to [0, 1] for positive values
        t = value / vmax if vmax != 0 else 0
        color = (1 - t) * gray + t * red
    else:
        # Scale value to [0, 1] for negative values; note vmin is negative
        t = value / vmin if vmin != 0 else 0  # t is positive
        color = (1 - t) * gray + t * blue
        
    # Ensure the color components remain within [0, 1]
    return tuple(np.clip(color, 0, 1))


def plot_cube(positions, colors, lattice, num_patches=(6, 6, 6), **kwargs):
    """
    help function for draw 3d cube plot
    :param positions: <list> list of patch position
    :param colors: <list -> list> list of color codes [r, g, b, w]
    :param lattice: <np.ndarray> p_lattice vector for unit p_lattice
    :param num_patches: number of patches in each axis (default : (6, 6, 6))
    :param kwargs: kwargs for <matplotlib.Poly3DCollection>
    :return: <matplotlib.Poly3DCollection> cuboid matplotlib object
    """

    data = [
        cuboid_data(pos, color, num_patches=num_patches, lattice=lattice)
        for pos, color in zip(positions, colors)
    ]
    plain_ls, color_ls = zip(*data)

    return Poly3DCollection(
        np.concatenate(plain_ls), facecolors=np.concatenate(color_ls), **kwargs
    )


@lru_cache
def get_primitive_structure(path_cif, tolerance=2.0):
    (st,) = CifParser(path_cif, occupancy_tolerance=tolerance).get_structures(
        primitive=True
    )
    return st


@lru_cache
def get_structure(
    path_cif, make_supercell=False, dtype="ase", *, max_length=60, min_length=30
):
    """
    get primitive structure from path_cif
    :param path_cif: <str> path for cif file
    :param make_supercell: <bool> if True,
    :param dtype: <str> -> ['ase', 'pymatgen'] return type for structure.
    :param max_length: <int/float> max p_lattice length of structure file (Å)
    :param min_length: <int/float> min p_lattice length of structure file (Å)
    :return: <pymatgen.Structure> structure file from path cif
    """
    try:
        primitive_structure = get_primitive_structure(path_cif)
    except ValueError as e:
        raise ValueError(f"{path_cif} failed : (read pymatgen) {e}")

    # Preserve the same atom ordering used when the graph/H5 sample was built.
    # The dataset builder uses pymatgen's CIF parsing order, while
    # ase.io.read/spglib can reorder atoms and misalign SHAP values with labels.
    atoms = AseAtomsAdaptor().get_atoms(primitive_structure)
    atoms.set_pbc(True)

    if make_supercell:
        atoms = get_supercell_structure(atoms, max_length, min_length)

    if dtype == "pymatgen":
        return AseAtomsAdaptor().get_structure(atoms)
    elif dtype == "ase":
        return atoms
    else:
        raise TypeError(f"type must be ase or pymatgen, not {dtype}")


def get_supercell_structure(atoms, max_length=60, min_length=30):
    """
    get supercell atoms from <ase.Atoms>
    :param atoms: <ase.Atoms> object
    :param max_length: <int/float> max p_lattice length of structure file (Å)
    :param min_length: <int/float> min p_lattice length of structure file (Å)
    :return: <ase.Atoms or pymatgen.Structure> structure type.
    """
    scale_abc = []
    for l in atoms.cell.cellpar()[:3]:
        if l > max_length:
            raise ValueError(
                f"primitive p_lattice is larger than max_length {max_length}"
            )
        elif l < min_length:
            scale_abc.append(math.ceil(min_length / l))
        else:
            scale_abc.append(1)

    m = np.zeros([3, 3])
    np.fill_diagonal(m, scale_abc)
    atoms = make_supercell(atoms, m)
    return atoms


def compact_plot_geometry(atoms, lattice=None, *, coordinate_scale=0.88):
    """
    Return a visualization-only copy of the structure with coordinates compressed
    around the unit-cell center so 3D renders feel tighter without changing the
    underlying chemistry or graph geometry.

    Parameters:
        atoms: ASE Atoms object.
        lattice: Optional 3x3 lattice array. Defaults to atoms.cell.array.
        coordinate_scale: Multiplicative factor applied about the cell center.
            Values < 1.0 pull coordinates inward for plotting only.

    Returns:
        (plot_atoms, plot_lattice, cell_start)
    """
    scale = float(coordinate_scale)
    if scale <= 0:
        raise ValueError(f"coordinate_scale must be > 0, got {coordinate_scale}")

    if lattice is None:
        lattice = atoms.cell.array
    lattice = np.asarray(lattice, dtype=float)

    plot_atoms = atoms.copy()
    if np.isclose(scale, 1.0):
        return plot_atoms, lattice, np.zeros(3, dtype=float)

    cell_center = np.sum(lattice, axis=0) / 2.0
    coords = plot_atoms.get_positions()
    plot_atoms.set_positions(cell_center + scale * (coords - cell_center))
    plot_lattice = lattice * scale
    cell_start = cell_center * (1.0 - scale)
    return plot_atoms, plot_lattice, cell_start


def cuboid_data(position, color=None, num_patches=(6, 6, 6), lattice=None):
    """
    Get cuboid plain data from position and size data
    :param position: <list/tuple> patch positions => [x, y, z]
    :param color: <list/tuple> colors => [r, g, b, w]
    :param num_patches: number of patches in each axis (default : (6, 6, 6))
    :param lattice: <np.ndarray> p_lattice vector for unit p_lattice
    :return: <tuple> (list of plain vector, list of color vector)
    """
    if isinstance(num_patches, (tuple, list)):
        num_patches = np.array(num_patches)
    elif not isinstance(num_patches, np.ndarray):
        raise TypeError(f"num_patches must be tuple or list, not {type(num_patches)}")

    bound = np.array([[0, 1] for _ in range(3)]) + np.array(position)[:, np.newaxis]
    vertex = np.array(list(product(*bound)))
    plane_ls = []
    for i, (dn, up) in enumerate(bound):
        plane1 = np.matmul(vertex[vertex[:, i] == dn], lattice / num_patches)
        plane2 = np.matmul(vertex[vertex[:, i] == up], lattice / num_patches)

        plane_ls.append(plane1)
        plane_ls.append(plane2)

    plane_ls = np.array(plane_ls).astype("float")
    plane_ls[:, [0, 1], :] = plane_ls[:, [1, 0], :]

    color_ls = np.repeat(color[np.newaxis, :], 6, axis=0)

    return plane_ls, color_ls


def plot_cube(positions, colors, lattice, num_patches=(6, 6, 6), **kwargs):
    """
    help function for draw 3d cube plot
    :param positions: <list> list of patch position
    :param colors: <list -> list> list of color codes [r, g, b, w]
    :param lattice: <np.ndarray> p_lattice vector for unit p_lattice
    :param num_patches: number of patches in each axis (default : (6, 6, 6))
    :param kwargs: kwargs for <matplotlib.Poly3DCollection>
    :return: <matplotlib.Poly3DCollection> cuboid matplotlib object
    """

    data = [
        cuboid_data(pos, color, num_patches=num_patches, lattice=lattice)
        for pos, color in zip(positions, colors)
    ]
    plain_ls, color_ls = zip(*data)

    return Poly3DCollection(
        np.concatenate(plain_ls), facecolors=np.concatenate(color_ls), **kwargs
    )


def get_heatmap(
    attns,
    batch_idx=0,
    graph_len=500,
    skip_cls=True,
    *,
    sep_tokens=1,
    add_feat_tokens=1,
):

    """
    Generates a heatmap based on attention rollout as described in the paper 
    "Quantifying Attention Flow in Transformers".

    This function computes the attention flow across transformer layers, 
    aggregates the attention weights, and generates a heatmap representation 
    of the attention distribution.

    Args:
        attns (list[torch.Tensor]): Per-layer attention weights. Each tensor has shape
            `[B, num_heads, T, T]`.
        batch_idx (int | None, optional): Batch index to process. If `None`, processes
            all items in the batch. Defaults to 0.
        graph_len (int, optional): Number of atom/graph tokens after CLS. Defaults to 500.
        skip_cls (bool, optional): If True, uses CLS-to-token flow and excludes CLS itself.
            Defaults to True.
        sep_tokens (int, optional): Number of special tokens between the padded graph
            token block and the appended additional feature tokens (e.g., an internal
            "CLS/SEP" for added features). Defaults to 1.
        add_feat_tokens (int, optional): Number of additional feature tokens to include.
            Defaults to 1.

    Returns:
        torch.Tensor: A 1D importance vector over `[graph_len] + [add_feat_tokens]`
        tokens. Shape is `[graph_len + add_feat_tokens]` if `batch_idx` is an int,
        else `[B, graph_len + add_feat_tokens]`. Additional feature tokens are read
        from the tail of the transformer sequence so padded graph slots do not shift
        their positions.
   """
    attn_weights = torch.stack(attns)  # [L, B, H, T, T]

    def _validate(token_count_excluding_cls: int) -> None:
        need = graph_len + sep_tokens + add_feat_tokens
        if need > token_count_excluding_cls:
            raise ValueError(
                "Token layout mismatch for attention rollout: "
                f"need graph_len({graph_len}) + sep_tokens({sep_tokens}) + "
                f"add_feat_tokens({add_feat_tokens}) = {need} tokens after CLS, "
                f"but only {token_count_excluding_cls} are available."
            )

    if batch_idx is None:
        att_mat = attn_weights.mean(dim=2)  # head-average -> [L, B, T, T]
        token_count = att_mat.size(-1)

        residual_att = torch.eye(
            token_count, device=att_mat.device, dtype=att_mat.dtype
        )[None, None, :, :]  # [1, 1, T, T]
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1, keepdim=True)

        joint_attentions = torch.zeros_like(aug_att_mat)
        joint_attentions[0] = aug_att_mat[0]
        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

        v = joint_attentions[-1]  # [B, T, T]

        if skip_cls:
            cls_flow = v[:, 0, 1:]  # [B, T-1]
            _validate(cls_flow.size(-1))
            atoms = cls_flow[:, :graph_len]
            add_feats = cls_flow[:, -add_feat_tokens:] if add_feat_tokens > 0 else cls_flow[:, :0]
            return torch.cat([atoms, add_feats], dim=-1)

        flow = v[:, 0, :]  # [B, T]
        atoms = flow[:, 1 : 1 + graph_len]
        add_feats = flow[:, -add_feat_tokens:] if add_feat_tokens > 0 else flow[:, :0]
        return torch.cat([atoms, add_feats], dim=-1)

    att_mat = attn_weights[:, batch_idx]  # [L, H, T, T]
    att_mat = att_mat.mean(dim=1)  # head-average -> [L, T, T]
    token_count = att_mat.size(-1)

    residual_att = torch.eye(
        token_count, device=att_mat.device, dtype=att_mat.dtype
    )[None, :, :]  # [1, T, T]
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1, keepdim=True)

    joint_attentions = torch.zeros_like(aug_att_mat)
    joint_attentions[0] = aug_att_mat[0]
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    v = joint_attentions[-1]  # [T, T]
    if skip_cls:
        cls_flow = v[0, 1:]  # [T-1]
        _validate(cls_flow.size(0))
        atoms = cls_flow[:graph_len]
        add_feats = cls_flow[-add_feat_tokens:] if add_feat_tokens > 0 else cls_flow[:0]
        return torch.cat([atoms, add_feats], dim=0)

    flow = v[0, :]  # [T]
    atoms = flow[1 : 1 + graph_len]
    add_feats = flow[-add_feat_tokens:] if add_feat_tokens > 0 else flow[:0]
    return torch.cat([atoms, add_feats], dim=0)


def scaler(value, min_att, max_att):
    if isinstance(value, float):
        if value > max_att:
            value = max_att
        elif value < min_att:
            value = min_att
        return float((value - min_att) / (max_att - min_att))

    elif isinstance(value, np.ndarray):
        value = copy.deepcopy(value)
        value[value > max_att] = max_att
        value[value < min_att] = min_att
        return (value - min_att) / (max_att - min_att)
    elif isinstance(value, Iterable):
        return scaler(np.array(list(value), dtype="float"), min_att, max_att)
    else:
        raise TypeError(f"value must be float, list, or np.array, not {type(value)}")
    
    
class HandlerCircleNoText(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        """
        Draw a circle patch at 200% the normal radius (no text inside).
        """
        # Base circle radius from the legend's allocated width/height.
        base_radius = min(width, height) / 2.0
        # Increase size by 200%.
        radius = 2.0 * base_radius

        center_x = xdescent + width / 2.0
        center_y = ydescent + height / 2.0
        
        # Create a circle patch (face color = CPK color, edge color = black)
        circle = mpatches.Circle(
            (center_x, center_y),
            radius,
            facecolor=orig_handle.get_facecolor(),
            edgecolor='black',
            lw=0.8
        )
        
        # Apply transforms so the circle moves with the legend
        circle.set_transform(trans)
        
        # Return a list of legend artists (in this case, only the circle)
        return [circle]


class HandlerCircleWithText(HandlerPatch):
    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        """
        Draw a colored circle with the element symbol centered inside.

        Uses a luminance heuristic to choose black/white text for contrast.
        The element symbol is taken from `orig_handle.get_label()`.
        """
        # Slightly enlarge the legend marker so 1–2 letter element symbols
        # fit comfortably inside without feeling cramped.
        radius = 1.25 * (min(width, height) / 2.0)
        center_x = xdescent + width / 2.0
        center_y = ydescent + height / 2.0

        face = orig_handle.get_facecolor()
        # `face` can be shape (4,) or (1,4); normalize to (4,).
        face_rgba = np.array(face).reshape(-1)
        if face_rgba.size >= 4:
            r, g, b = face_rgba[:3]
        else:
            r, g, b = 1.0, 1.0, 1.0
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = "white" if luminance < 0.5 else "black"

        circle = mpatches.Circle(
            (center_x, center_y),
            radius,
            facecolor=orig_handle.get_facecolor(),
            edgecolor="black",
            lw=0.8,
        )
        circle.set_transform(trans)

        label = orig_handle.get_label()
        text = Text(
            x=center_x,
            y=center_y,
            text=label,
            color=text_color,
            ha="center",
            va="center",
            fontsize=max(7, int(fontsize * 0.9)),
            weight="bold",
        )
        text.set_transform(trans)
        return [circle, text]


def create_circle_legend_handles(atoms):
    atomic_numbers = atoms.get_atomic_numbers()
    unique_nums = sorted(set(atomic_numbers))
    
    legend_handles = []
    legend_labels = []
    
    for num in unique_nums:
        color = cpk_colors.get_cpk_color(num)
        label = chemical_symbols[num]
        
        # Circle patch (the position and radius here won't matter,
        # as the legend handler overrides them, but the facecolor does).
        circle = mpatches.Circle((0, 0), 0.5, fc=color, ec='black', lw=0.8)
        
        # We set the label so that Matplotlib knows how to name this entry in the legend.
        circle.set_label(label)
        
        legend_handles.append(circle)
        legend_labels.append(label)
    
    return legend_handles, legend_labels


def plot_attention_shap_triptych_3d(
    *,
    path_cif: str,
    shap_values: Union[torch.Tensor, np.ndarray, Sequence[float]],
    attention_values: Union[torch.Tensor, np.ndarray, Sequence[float]],
    save: bool = True,
    show: bool = True,
    close: bool = False,
    out_dir: Optional[Union[str, Path]] = None,
    filename: Optional[str] = None,
    formats: Tuple[str, ...] = ("png", "pdf"),
    dpi: int = 600,
    figsize: Tuple[float, float] = (14.0, 5.2),
    view_init: Tuple[float, float] = (25, 25),
    alpha: float = 1.0,
    atomic_scale_factor: float = 4.0,
    att_scale_factor: float = 1.0,
    grid_scale_factor: float = 1.0,
    show_element_legend: bool = True,
    panel_titles: Optional[Tuple[str, str, str, str]] = (
        "Original crystal",
        "Atom SHAPs",
        "Attention rollout",
        "Mismatch (impact ignored)",
    ),
    add_feat_label: str = "ADF",
    add_feat_names: Optional[Sequence[str]] = None,
    add_feat_tokens: int = 1,
    sep_tokens: int = 1,
    suptitle: Optional[str] = None,
):
    """
    Create a publication-style 3D triptych comparing SHAP vs attention rollout vs mismatch.

    This is designed for the token layout used in this repo:
      `[CLS] + [max_atoms tokens] + [added_feature_internal_CLS/SEP] + [added_feature_value]`

    The plot focuses on atom tokens (colored on the crystal structure) and additionally
    reports the added-feature token values in the figure header.

    Panels:
      1) SHAP values (signed, diverging colormap)
      2) Attention rollout (non-negative, sequential colormap)
      3) Mismatch score (high-impact-but-ignored): `abs(shap_norm) * (1 - attn_norm)`

    Args:
        path_cif: Path to the CIF file to render.
        shap_values: SHAP values per token. Expected to include atom tokens and
            `add_feat_tokens` appended at the end.
        attention_values: Attention rollout importance per token. Expected to include
            atom tokens and `add_feat_tokens` appended at the end.
        save: If True, saves the figure to disk.
        show: If True, displays the figure via `matplotlib.pyplot.show()`.
        close: If True, closes the figure after saving/showing (helps avoid memory buildup
            when generating many figures).
        out_dir: Output directory (defaults to current working directory).
        filename: Base filename (without extension). Defaults to `<cif_stem>_attn_shap_triptych`.
        formats: File extensions to save (e.g., ("png","pdf")).
        dpi: Raster DPI for formats like PNG.
        figsize: Figure size in inches.
        view_init: (elev, azim) camera view shared across subplots.
        alpha: Alpha used for the overlay scatter.
        atomic_scale_factor: Base scatter size scaling for the CPK background atoms.
        overlay_scale_factor: Scatter size scaling for the colored overlay.
        panel_titles: Optional titles for (atoms, shap, attention, mismatch). If None,
            no subplot titles are shown.
        add_feat_label: Label used for the additional feature token in the header.
        add_feat_tokens: Number of additional feature *value* tokens included at the end.
        sep_tokens: Number of special tokens between atoms and added feature tokens.
        suptitle: Optional figure-level title (defaults to CIF stem).

    Returns:
        (fig, axes, meta) where:
          - fig is a matplotlib Figure
          - axes is a tuple (ax_shap, ax_attn, ax_mismatch)
          - meta is a dict with aligned vectors and save paths (if saved)
    """
    import os
    from matplotlib import pyplot as plt
    from matplotlib.colors import TwoSlopeNorm, Normalize

    from exctransformer.SHAP_visualization.drawer import (
        draw_cell,
        draw_atoms,
        draw_colorbar,
    )
    from exctransformer.SHAP_visualization.setting import set_axes_equal

    def _to_1d_tensor(x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().flatten().to(torch.float32)
        return torch.tensor(np.asarray(x), dtype=torch.float32).flatten()

    def _minmax_norm(x: torch.Tensor) -> torch.Tensor:
        x_min = torch.min(x)
        x_max = torch.max(x)
        if torch.isclose(x_min, x_max):
            return torch.zeros_like(x)
        return (x - x_min) / (x_max - x_min)

    # ---- Load structure (primitive by default in get_structure) ----
    atoms = get_structure(path_cif, make_supercell=False, dtype="ase")
    lattice = atoms.get_cell().array
    plot_atoms, plot_lattice, cell_start = compact_plot_geometry(
        atoms, lattice=lattice, coordinate_scale=0.88
    )
    symbols = atoms.get_chemical_symbols()
    atomic_numbers = atoms.get_atomic_numbers()
    n_atoms = len(symbols)

    # ---- Align input vectors to [n_atoms + add_feat_tokens] ----
    shap = _to_1d_tensor(shap_values)
    attn = _to_1d_tensor(attention_values)

    # Both vectors may come from padded token spaces (e.g., max_atoms=500).
    # For the 3D plot we map only the first n_atoms to coordinates and keep the last
    # `add_feat_tokens` values as "global feature tokens".
    expected_len = n_atoms + add_feat_tokens

    def _align(vec: torch.Tensor, name: str) -> Tuple[torch.Tensor, int]:
        if vec.numel() < n_atoms:
            raise ValueError(
                f"{name} length {vec.numel()} is smaller than required n_atoms={n_atoms}."
            )
        if vec.numel() == n_atoms:
            return vec[:n_atoms], 0
        if vec.numel() == expected_len:
            return vec, add_feat_tokens
        # If we have a padded vector, assume layout:
        #   [atoms padded ...] + [sep tokens] + [add_feat_tokens]
        if add_feat_tokens > 0 and vec.numel() > expected_len:
            add_tail = vec[-add_feat_tokens:]
            atom_head = vec[:n_atoms]
            return torch.cat([atom_head, add_tail], dim=0), add_feat_tokens
        # If add_feat_tokens is 0 (or vec is shorter than expected), infer tail length.
        inferred_add = vec.numel() - n_atoms
        return vec[: n_atoms + inferred_add], inferred_add

    shap, shap_add_tokens = _align(shap, "shap_values")
    attn, attn_add_tokens = _align(attn, "attention_values")
    effective_add_tokens = min(shap_add_tokens, attn_add_tokens)
    if shap_add_tokens != attn_add_tokens:
        warnings.warn(
            f"shap/attention add token mismatch (shap={shap_add_tokens}, attn={attn_add_tokens}); "
            f"using add_feat_tokens={effective_add_tokens}."
        )
        if effective_add_tokens == 0:
            shap = shap[:n_atoms]
            attn = attn[:n_atoms]
        else:
            shap = torch.cat([shap[:n_atoms], shap[-effective_add_tokens:]], dim=0)
            attn = torch.cat([attn[:n_atoms], attn[-effective_add_tokens:]], dim=0)

    add_feat_tokens = effective_add_tokens
    expected_len = n_atoms + add_feat_tokens

    shap_atoms = shap[:n_atoms]
    attn_atoms = attn[:n_atoms]
    shap_add = shap[-add_feat_tokens:] if add_feat_tokens > 0 else None
    attn_add = attn[-add_feat_tokens:] if add_feat_tokens > 0 else None

    # ---- Derived mismatch metric (high impact but ignored) ----
    shap_abs_norm = _minmax_norm(torch.abs(shap_atoms))
    attn_norm = _minmax_norm(attn_atoms)
    mismatch = shap_abs_norm * (1.0 - attn_norm)

    # ---- Colors / norms ----
    shap_vmax = float(torch.max(torch.abs(shap_atoms)).item()) if n_atoms else 1.0
    shap_norm = TwoSlopeNorm(vmin=-shap_vmax, vcenter=0.0, vmax=shap_vmax)
    attn_min = float(torch.min(attn_atoms).item()) if n_atoms else 0.0
    attn_max = float(torch.max(attn_atoms).item()) if n_atoms else 1.0
    if attn_min == attn_max:
        attn_max = attn_min + 1e-12
    attn_norm_plot = Normalize(vmin=attn_min, vmax=attn_max)

    mismatch_min = float(torch.min(mismatch).item()) if n_atoms else 0.0
    mismatch_max = float(torch.max(mismatch).item()) if n_atoms else 1.0
    if mismatch_min == mismatch_max:
        mismatch_max = mismatch_min + 1e-12
    mismatch_norm_plot = Normalize(vmin=mismatch_min, vmax=mismatch_max)

    shap_cmap = plt.get_cmap("coolwarm")
    attn_cmap = plt.get_cmap("viridis")
    mismatch_cmap = plt.get_cmap("magma")

    # Match PatchVisualizer sizing conventions.
    atomic_scale = float(figsize[0] * figsize[1])
    base_atom_scale = atomic_scale * atomic_scale_factor * grid_scale_factor
    # Keep overlays crisp by matching the base atom sizing; increase only if explicitly requested.
    overlay_atom_scale = atomic_scale * atomic_scale_factor * att_scale_factor * grid_scale_factor

    from ase.data import covalent_radii

    radii = np.array([covalent_radii[z] for z in atomic_numbers], dtype=float)

    def _draw_colored_atoms(ax, colors_rgba):
        coords_ = plot_atoms.get_positions()
        ax.scatter(
            xs=coords_[:, 0],
            ys=coords_[:, 1],
            zs=coords_[:, 2],
            c=colors_rgba,
            # Match `draw_atoms` behavior: per-element sizes based on covalent radii.
            s=radii * overlay_atom_scale,
            marker="o",
            edgecolor="black",
            linewidths=0.65,
            alpha=alpha,
        )

    # ---- Figure ----
    with plt.rc_context(
        {
            "figure.dpi": 150,
            "savefig.dpi": dpi,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    ):
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(1, 4, wspace=0.04)
        ax_atoms = fig.add_subplot(gs[0, 0], projection="3d")
        ax_shap = fig.add_subplot(gs[0, 1], projection="3d")
        ax_attn = fig.add_subplot(gs[0, 2], projection="3d")
        ax_mis = fig.add_subplot(gs[0, 3], projection="3d")

        axes = (ax_atoms, ax_shap, ax_attn, ax_mis)
        for ax in axes:
            ax.view_init(elev=view_init[0], azim=view_init[1])
            ax.set_axis_off()
            draw_cell(ax, plot_lattice, s_point=cell_start, color="black")

        # Panel 1: normal atoms (CPK) only.
        draw_atoms(ax_atoms, plot_atoms, base_atom_scale)

        # Panels 2-4: overlays only (no base atoms) for clearer comparison.
        shap_colors = shap_cmap(shap_norm(shap_atoms.numpy()))
        _draw_colored_atoms(ax_shap, shap_colors)

        attn_colors = attn_cmap(attn_norm_plot(attn_atoms.numpy()))
        _draw_colored_atoms(ax_attn, attn_colors)

        mismatch_colors = mismatch_cmap(mismatch_norm_plot(mismatch.numpy()))
        _draw_colored_atoms(ax_mis, mismatch_colors)
        
        if panel_titles is not None:
            if len(panel_titles) != 4:
                raise ValueError("panel_titles must have exactly 4 strings.")
            ax_atoms.set_title(panel_titles[0])
            ax_shap.set_title(panel_titles[1])
            ax_attn.set_title(panel_titles[2])
            ax_mis.set_title(panel_titles[3])

        # Keep consistent scaling.
        for ax in axes:
            set_axes_equal(ax, scale_factor=grid_scale_factor)

        # Optional element legend (off by default to keep panels clean).
        if show_element_legend:
            legend_handles, legend_labels = create_circle_legend_handles(atoms)
            # Place a horizontal legend below the atom-only panel.
            bbox = ax_atoms.get_position()
            # Tight packing directly under the panel.
            y = max(0.0, bbox.y0 - 0.012)
            x_text = bbox.x0
            x_leg = bbox.x0
            fig.text(x_text, y + 0.002, "Elements", ha="left", va="top", fontsize=9)
            fig.legend(
                legend_handles,
                [""] * len(legend_labels),
                handler_map={mpatches.Circle: HandlerCircleWithText()},
                loc="upper left",
                bbox_to_anchor=(x_leg + 0.055, y + 0.002),
                borderaxespad=0.0,
                # Typical MOFs have ~5–10 unique elements; allow wrapping for larger cases.
                ncol=min(len(legend_labels), 10),
                labelspacing=0.15,
                handletextpad=0.0,
                columnspacing=0.35,
                borderpad=0.0,
                handlelength=1.0,
                handleheight=1.0,
                frameon=False,
            )

        # Colorbars.
        draw_colorbar(fig, ax_shap, shap_cmap, -shap_vmax, shap_vmax, norm=shap_norm, label="SHAP")
        draw_colorbar(fig, ax_attn, attn_cmap, attn_min, attn_max, norm=attn_norm_plot, label="Attention")
        draw_colorbar(fig, ax_mis, mismatch_cmap, mismatch_min, mismatch_max, norm=mismatch_norm_plot, label="Mismatch")

        # Figure title + added-feature token values (if present).
        cif_stem = Path(path_cif).stem
        if suptitle is None:
            suptitle = None
        header = suptitle
        if shap_add is not None and attn_add is not None:
            if add_feat_names is not None and len(add_feat_names) != add_feat_tokens:
                raise ValueError(
                    f"add_feat_names must have length {add_feat_tokens}, got {len(add_feat_names)}"
                )
            token_labels = []
            for i in range(add_feat_tokens):
                token_labels.append(f"{add_feat_label}{i+1}")

            if add_feat_names:
                token_labels = [
                    f"{tok}({name})" for tok, name in zip(token_labels, add_feat_names)
                ]

            shap_add_str = ", ".join(
                [f"{k}={v:+.4f}" for k, v in zip(token_labels, shap_add.tolist())]
            )
            attn_add_str = ", ".join(
                [f"{k}={v:.4f}" for k, v in zip(token_labels, attn_add.tolist())]
            )
            header = (
                f"{suptitle}  |  Added feature token(s): SHAP[{shap_add_str}], "
                f"Attention[{attn_add_str}]"
            )
        if header:
            fig.suptitle(header, y=1.02, fontsize=12)

    # ---- Saving ----
    save_paths = []
    if save:
        out_dir = Path(out_dir) if out_dir is not None else Path(os.getcwd())
        out_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"{Path(path_cif).stem}_attn_shap_triptych"

        for ext in formats:
            path = out_dir / f"{filename}.{ext}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
            save_paths.append(str(path))

    if show:
        plt.show()

    if close:
        plt.close(fig)

    meta = {
        "n_atoms": n_atoms,
        "shap_atoms": shap_atoms,
        "attn_atoms": attn_atoms,
        "mismatch_atoms": mismatch,
        "shap_add": shap_add,
        "attn_add": attn_add,
        "save_paths": save_paths,
    }
    return fig, (ax_shap, ax_attn, ax_mis), meta


def plot_attention_shap_triptych_2d(
    *,
    path_cif: str,
    shap_values: Union[torch.Tensor, np.ndarray, Sequence[float]],
    attention_values: Union[torch.Tensor, np.ndarray, Sequence[float]],
    save: bool = True,
    show: bool = True,
    close: bool = False,
    out_dir: Optional[Union[str, Path]] = None,
    filename: Optional[str] = None,
    formats: Tuple[str, ...] = ("png", "pdf"),
    dpi: int = 600,
    figsize: Tuple[float, float] = (12.8, 4.6),
    annotate: str = "auto",
    max_annotate_tokens: int = 120,
    value_fmt_shap: str = "{:+.3f}",
    value_fmt_pos: str = "{:.3f}",
    value_fmt_mismatch: str = "{:.3f}",
    add_feat_label: str = "ADF",
    add_feat_names: Optional[Sequence[str]] = None,
    add_feat_tokens: int = 1,
    panel_titles: Optional[Tuple[str, str, str]] = (
        "Atom SHAPs",
        "Attention rollout",
        "Mismatch (impact ignored)",
    ),
    suptitle: Optional[str] = None,
):
    """
    Create a publication-style 2D heatmap triptych comparing SHAP vs attention vs mismatch.

    This is meant to be a *readable* alternative to 3D renders when you want to inspect
    token-level alignment directly. Each token is placed into a square grid (row-major)
    and colored by its scalar value.

    Token assumptions:
      - The first `n_atoms` values correspond to atom tokens (in CIF order).
      - The last `add_feat_tokens` values correspond to additional feature token(s).
      - If vectors are longer (e.g., padded to max_atoms), this function truncates to
        the first `n_atoms` and the last `add_feat_tokens`.

    Panels:
      1) SHAP values (signed, diverging colormap)
      2) Attention rollout (non-negative, sequential colormap)
      3) Mismatch score (high-impact-but-ignored): `abs(shap_norm) * (1 - attn_norm)`

    Each square can be annotated with the element symbol; value annotations are optional
    and are automatically disabled for large token counts unless `annotate="values"`.

    Args:
        path_cif: Path to CIF (used to get element symbols and atom count).
        shap_values: Token SHAP values (atoms first; may include padding).
        attention_values: Token attention rollout values (atoms first; may include padding).
        save/show/close: Saving and display controls.
        out_dir/filename/formats/dpi: Saving controls.
        figsize: Figure size in inches.
        annotate: One of {"auto","none","symbols","values"}.
        max_annotate_tokens: If `annotate="auto"`, value annotations are enabled only if
            `n_tokens <= max_annotate_tokens`.
        value_fmt_*: Format strings for cell value text.
        add_feat_label: Label printed for the additional feature token(s).
        add_feat_tokens: Number of feature tokens appended at the end to include.
        suptitle: Optional title (defaults to CIF stem).

    Returns:
        (fig, axes, meta) where axes is (ax_shap, ax_attn, ax_mismatch).
    """
    import os
    import math
    from matplotlib import pyplot as plt
    from matplotlib.colors import TwoSlopeNorm, Normalize

    def _to_1d_tensor(x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().flatten().to(torch.float32)
        return torch.tensor(np.asarray(x), dtype=torch.float32).flatten()

    def _minmax_norm(x: torch.Tensor) -> torch.Tensor:
        x_min = torch.min(x)
        x_max = torch.max(x)
        if torch.isclose(x_min, x_max):
            return torch.zeros_like(x)
        return (x - x_min) / (x_max - x_min)

    atoms = get_structure(path_cif, make_supercell=False, dtype="ase")
    symbols = atoms.get_chemical_symbols()
    n_atoms = len(symbols)

    shap = _to_1d_tensor(shap_values)
    attn = _to_1d_tensor(attention_values)

    expected_len = n_atoms + add_feat_tokens

    def _align(vec: torch.Tensor, name: str) -> Tuple[torch.Tensor, int]:
        if vec.numel() < n_atoms:
            raise ValueError(
                f"{name} length {vec.numel()} is smaller than required n_atoms={n_atoms}."
            )
        if vec.numel() == n_atoms:
            return vec[:n_atoms], 0
        if vec.numel() == expected_len:
            return vec, add_feat_tokens
        if add_feat_tokens > 0 and vec.numel() > expected_len:
            add_tail = vec[-add_feat_tokens:]
            atom_head = vec[:n_atoms]
            return torch.cat([atom_head, add_tail], dim=0), add_feat_tokens
        inferred_add = vec.numel() - n_atoms
        return vec[: n_atoms + inferred_add], inferred_add

    shap, shap_add_tokens = _align(shap, "shap_values")
    attn, attn_add_tokens = _align(attn, "attention_values")
    effective_add_tokens = min(shap_add_tokens, attn_add_tokens)
    if shap_add_tokens != attn_add_tokens:
        warnings.warn(
            f"shap/attention add token mismatch (shap={shap_add_tokens}, attn={attn_add_tokens}); "
            f"using add_feat_tokens={effective_add_tokens}."
        )
        if effective_add_tokens == 0:
            shap = shap[:n_atoms]
            attn = attn[:n_atoms]
        else:
            shap = torch.cat([shap[:n_atoms], shap[-effective_add_tokens:]], dim=0)
            attn = torch.cat([attn[:n_atoms], attn[-effective_add_tokens:]], dim=0)

    add_feat_tokens = effective_add_tokens
    expected_len = n_atoms + add_feat_tokens

    # Labels: element symbols for atoms, then a short label for each feature token.
    labels = list(symbols)
    if add_feat_tokens > 0:
        if add_feat_names is not None and len(add_feat_names) != add_feat_tokens:
            raise ValueError(
                f"add_feat_names must have length {add_feat_tokens}, got {len(add_feat_names)}"
            )
        for i in range(add_feat_tokens):
            labels.append(f"{add_feat_label}{i+1}")

    shap_atoms = shap[:n_atoms]
    attn_atoms = attn[:n_atoms]
    shap_abs_norm = _minmax_norm(torch.abs(shap_atoms))
    attn_norm = _minmax_norm(attn_atoms)
    mismatch_atoms = shap_abs_norm * (1.0 - attn_norm)

    # Append feature tokens to mismatch as 0 by default (they do not have a spatial atom mapping).
    # If you later want a different treatment, you can pass feature-specific mismatch.
    if add_feat_tokens > 0:
        mismatch = torch.cat([mismatch_atoms, torch.zeros(add_feat_tokens)], dim=0)
    else:
        mismatch = mismatch_atoms

    n_tokens = expected_len
    n_cols = int(math.ceil(math.sqrt(n_tokens)))
    n_rows = int(math.ceil(n_tokens / n_cols))

    def _pack_grid(vec: torch.Tensor) -> np.ndarray:
        grid = torch.full((n_rows * n_cols,), float("nan"))
        grid[:n_tokens] = vec.to(torch.float32)
        return grid.view(n_rows, n_cols).numpy()

    shap_grid = _pack_grid(shap)
    attn_grid = _pack_grid(attn)
    mismatch_grid = _pack_grid(mismatch)

    # Decide annotation mode.
    annotate = annotate.lower()
    if annotate not in {"auto", "none", "symbols", "values"}:
        raise ValueError(f"annotate must be one of auto/none/symbols/values, got {annotate!r}")
    if annotate == "auto":
        annotate = "values" if n_tokens <= max_annotate_tokens else "symbols"

    def _safe_vmin_vmax(vmin: float, vmax: float, eps: float = 1e-12) -> tuple[float, float]:
        # Avoid vmin==vmax which can trigger Matplotlib cursor-formatting overflows.
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return 0.0, 1.0
        if vmin == vmax:
            return vmin - eps, vmax + eps
        return vmin, vmax

    # Color normalization (ensure non-degenerate norms for interactive backends).
    shap_vmax = float(np.nanmax(np.abs(shap_grid))) if np.isfinite(shap_grid).any() else 1.0
    if shap_vmax == 0.0:
        shap_vmax = 1e-12
    shap_norm = TwoSlopeNorm(vmin=-shap_vmax, vcenter=0.0, vmax=shap_vmax)

    attn_vmin = float(np.nanmin(attn_grid)) if np.isfinite(attn_grid).any() else 0.0
    attn_vmax = float(np.nanmax(attn_grid)) if np.isfinite(attn_grid).any() else 1.0
    attn_vmin, attn_vmax = _safe_vmin_vmax(attn_vmin, attn_vmax)
    attn_plot_norm = Normalize(vmin=attn_vmin, vmax=attn_vmax)

    mis_vmin = float(np.nanmin(mismatch_grid)) if np.isfinite(mismatch_grid).any() else 0.0
    mis_vmax = float(np.nanmax(mismatch_grid)) if np.isfinite(mismatch_grid).any() else 1.0
    mis_vmin, mis_vmax = _safe_vmin_vmax(mis_vmin, mis_vmax)
    mis_plot_norm = Normalize(vmin=mis_vmin, vmax=mis_vmax)

    with plt.rc_context(
        {
            "figure.dpi": 150,
            "savefig.dpi": dpi,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        }
    ):
        fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
        ax_shap, ax_attn, ax_mis = axes

        cmap_shap = plt.get_cmap("coolwarm").copy()
        cmap_attn = plt.get_cmap("viridis").copy()
        cmap_mis = plt.get_cmap("magma").copy()
        for cmap in (cmap_shap, cmap_attn, cmap_mis):
            cmap.set_bad(color="#f0f0f0")

        im0 = ax_shap.imshow(
            shap_grid, cmap=cmap_shap, norm=shap_norm, aspect="equal", interpolation="nearest"
        )
        im1 = ax_attn.imshow(
            attn_grid, cmap=cmap_attn, norm=attn_plot_norm, aspect="equal", interpolation="nearest"
        )
        im2 = ax_mis.imshow(
            mismatch_grid, cmap=cmap_mis, norm=mis_plot_norm, aspect="equal", interpolation="nearest"
        )
        
        if panel_titles is not None:
            if len(panel_titles) != 3:
                raise ValueError("panel_titles must have exactly 3 strings.")
            ax_shap.set_title(panel_titles[0])
            ax_attn.set_title(panel_titles[1])
            ax_mis.set_title(panel_titles[2])

        # Matplotlib's default hover formatting can throw OverflowError when the
        # underlying normalization is degenerate (vmin == vmax) or when values are NaN.
        # Use a safe, constant-time formatter to avoid noisy interactive-backend errors.
        def _safe_cursor(data):
            try:
                if data is None:
                    return ""
                x = float(data)
                if not np.isfinite(x):
                    return ""
                return f"{x:.4g}"
            except Exception:
                return ""

        im0.format_cursor_data = _safe_cursor
        im1.format_cursor_data = _safe_cursor
        im2.format_cursor_data = _safe_cursor

        # Remove ticks for a clean, publication-style panel.
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            # Cell borders for clarity (publication-style).
            ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8, alpha=0.9)
            ax.tick_params(which="minor", bottom=False, left=False)

        def _text_color_from_rgba(rgba) -> str:
            r, g, b = rgba[:3]
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            return "white" if luminance < 0.5 else "black"

        # Annotate grid cells.
        def _iter_cells():
            for idx in range(n_tokens):
                r = idx // n_cols
                c = idx % n_cols
                yield idx, r, c

        for idx, r, c in _iter_cells():
            label = labels[idx]
            shap_rgba = cmap_shap(shap_norm(shap_grid[r, c])) if np.isfinite(shap_grid[r, c]) else (1, 1, 1, 1)
            attn_rgba = cmap_attn(attn_plot_norm(attn_grid[r, c])) if np.isfinite(attn_grid[r, c]) else (1, 1, 1, 1)
            mis_rgba = cmap_mis(mis_plot_norm(mismatch_grid[r, c])) if np.isfinite(mismatch_grid[r, c]) else (1, 1, 1, 1)

            if annotate in {"symbols", "values"}:
                ax_shap.text(
                    c,
                    r,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=_text_color_from_rgba(shap_rgba),
                )
                ax_attn.text(
                    c,
                    r,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=_text_color_from_rgba(attn_rgba),
                )
                ax_mis.text(
                    c,
                    r,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=_text_color_from_rgba(mis_rgba),
                )

            if annotate == "values":
                # Add a smaller value on a second line (slightly below center).
                ax_shap.text(
                    c,
                    r + 0.33,
                    value_fmt_shap.format(float(shap_grid[r, c])),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=_text_color_from_rgba(shap_rgba),
                )
                ax_attn.text(
                    c,
                    r + 0.33,
                    value_fmt_pos.format(float(attn_grid[r, c])),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=_text_color_from_rgba(attn_rgba),
                )
                ax_mis.text(
                    c,
                    r + 0.33,
                    value_fmt_mismatch.format(float(mismatch_grid[r, c])),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=_text_color_from_rgba(mis_rgba),
                )

        # Colorbars.
        cbar0 = fig.colorbar(im0, ax=ax_shap, fraction=0.046, pad=0.02, shrink=0.85)
        cbar0.ax.set_ylabel("SHAP", rotation=270, labelpad=14)
        cbar1 = fig.colorbar(im1, ax=ax_attn, fraction=0.046, pad=0.02, shrink=0.85)
        cbar1.ax.set_ylabel("Attention", rotation=270, labelpad=14)
        cbar2 = fig.colorbar(im2, ax=ax_mis, fraction=0.046, pad=0.02, shrink=0.85)
        cbar2.ax.set_ylabel("Mismatch", rotation=270, labelpad=14)

        cif_stem = Path(path_cif).stem
        if suptitle is None:
            suptitle = None
        if suptitle:
            fig.suptitle(suptitle, y=1.06, fontsize=12)

        # Optional legend mapping ADF tokens -> user-provided names (kept compact).
        if add_feat_tokens > 0 and add_feat_names:
            mapping = "  ".join(
                [f"{add_feat_label}{i+1}={name}" for i, name in enumerate(add_feat_names)]
            )
            fig.text(
                0.5,
                0.01,
                mapping,
                ha="center",
                va="bottom",
                fontsize=9,
            )

    save_paths = []
    if save:
        out_dir = Path(out_dir) if out_dir is not None else Path(os.getcwd())
        out_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"{Path(path_cif).stem}_attn_shap_heatmaps"
        for ext in formats:
            path = out_dir / f"{filename}.{ext}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
            save_paths.append(str(path))

    if show:
        plt.show()
    if close:
        plt.close(fig)

    meta = {
        "n_atoms": n_atoms,
        "n_tokens": n_tokens,
        "grid_shape": (n_rows, n_cols),
        "labels": labels,
        "shap": shap,
        "attention": attn,
        "mismatch": mismatch,
        "save_paths": save_paths,
    }
    return fig, (ax_shap, ax_attn, ax_mis), meta
