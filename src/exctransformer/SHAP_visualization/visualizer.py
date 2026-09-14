from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
from functools import wraps, partial
from matplotlib import animation
import matplotlib.patches as mpatches
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch


from exctransformer.SHAP_visualization.setting import (
    get_cmap, 
    get_fig_ax, 
    set_fig_ax, 
    set_axes_equal,
    DEFAULT_FIGSIZE,
    DEFAULT_VIEW_INIT,
    get_default_cbar_kwargs)

from exctransformer.SHAP_visualization.drawer import (
    draw_cell, 
    draw_atoms, 
    draw_atoms_highlight, 
    draw_colorbar,
    draw_heatmap_graph)

from exctransformer.SHAP_visualization.utils import (
    get_structure,
    scaler,
    HandlerCircleNoText,
    create_circle_legend_handles,
    compact_plot_geometry,
)


from itertools import product
import numpy as np
import os
from pathlib import Path
from collections.abc import Iterable, Sequence


class PatchVisualizer(object):
    def __init__(self, path_cif, heatmap_graph, uni_idx, **kwargs):
        """
        SHAP Visualizer from "MOFTransformer model"
        :param path_cif: <str> path for original cif file.
        :param heatmap_graph: <np.array> graph attention score for cif
        :param uni_idx: <list-> list> uni index for cif. (from model.utils.prepare_data)
        :param kwargs:
            figsize : (float, float) figure size
            view_init : (float, float) view init from matplotlib
            show_axis : <bool> If True, axis are visible. (default : False)
            show_colorbar : <bool> If True, colorbar are visible. (default : True)
            cmap : (str or matplotlib.colors.ListedColormap) color map used in figure. (default : None)
            num_patches : (int, int, int) number of patches (default : (6, 6, 6))
            max_length : <float> max p_lattice length of structure file (Å)
            min_length: <float> min p_lattice length of structure file (Å)
        """
        self.path_cif = path_cif
        self.heatmap_graph = heatmap_graph
        self.uni_idx = uni_idx
        self.kwargs = kwargs
        self.cbar_kwargs = get_default_cbar_kwargs(self.figsize)

        # get primitive structure from cif
        p_atoms = get_structure(self.path_cif, make_supercell=False, dtype="ase")
        self.p_atoms = p_atoms
        self.p_lattice = p_atoms.cell.array

        # get supercell atoms from cif
        max_length, min_length = kwargs.get("max_length", 60), kwargs.get(
            "min_length", 30
        )
        super_atoms = get_structure(
            self.path_cif,
            make_supercell=True,
            dtype="ase",
            max_length=max_length,
            min_length=min_length,
        )
        self.s_atoms = super_atoms
        self.s_lattice = super_atoms.cell.array

    @classmethod
    def from_batch(cls, batch, batch_idx, model, cif_root, **kwargs):
        """
        Attention visualizer from "MOFTransformer" model and "dataloader batch"
        :param batch: Dataloader -> batch
        :param batch_idx: index for batch index
        :param model: <torch.model> fine-tuned MOFTransformer model
        :param cif_root: <str> root dir for cif files
        :param kwargs:
            figsize : (float, float) figure size
            view_init : (float, float) view init from matplotlib
            show_axis : <bool> If True, axis are visible. (default : False)
            show_colorbar : <bool> If True, colorbar are visible. (default : True)
            cmap : (str or matplotlib.colors.ListedColormap) color map used in figure. (default : None)
            num_patches : (int, int, int) number of patches (default : (6, 6, 6))
            max_length : <float> max p_lattice length of structure file (Å)
            min_length: <float> min p_lattice length of structure file (Å)
        :return: <PatchVisualizer> patch visualizer object
        """
        output = model.infer(batch)
        heatmap_graph, heatmap_grid = get_heatmap(output, batch_idx)
        uni_idx = batch["uni_idx"][batch_idx]
        cif_id = batch["cif_id"][batch_idx]
        path_cif = os.path.join(cif_root, cif_id + ".cif")
        return cls(
            path_cif, heatmap_graph, heatmap_grid, uni_idx, cif_id=cif_id, **kwargs
        )

    @classmethod
    def from_index(
        cls, index, model_path, data_root, downstream="", cif_root=None, **kwargs
    ):
        """
        Create PatchVisualizer from index. The index corresponds 1:1 to the MOF in the json file in dataset folder.
        The index matches the cif order of json.
        :param index: (int) index of dataset.
        :param model_path: (str) path of model from fine-tuned MOFTransformer with format '.ckpt'
        :param data_root: (str) path of dataset directory obtained from 'prepared_data.py. (see Dataset Preparation)
                MOFs to be visualized must exist in {dataset_folder}/test.json or {dataset_folder}/test_{downstream}.json,
                and {dataset_folder}/test folder. *.graphdata, *.grid, *.griddata16 files should be existed in {dataset_folder}/test folder.
        :param downstream: (str, optional) Use if data are existed in {dataset_folder}/test_{downstream}.json (default:'')
        :param cif_root: (str, optional) path of directory including cif file. The cif lists in the dataset folder should be included.
                If not specified, it is automatically specified as a {dataset_folder}/test folder.
        :param kwargs:
            figsize : (float, float) figure size
            view_init : (float, float) view init from matplotlib
            show_axis : <bool> If True, axis are visible. (default : False)
            show_colorbar : <bool> If True, colorbar are visible. (default : True)
            cmap : (str or matplotlib.colors.ListedColormap) color map used in figure. (default : None)
            num_patches : (int, int, int) number of patches (default : (6, 6, 6))
            max_length : <float> max p_lattice length of structure file (Å)
            min_length: <float> min p_lattice length of structure file (Å)
        :return: PatchVisualizer class for index
        """
        model, data_iter = get_model_and_datamodule(model_path, data_root, downstream)
        batch = get_batch_from_index(data_iter, index)
        if cif_root is None:
            cif_root = os.path.join(data_root, "test")

        return cls.from_batch(batch, 0, model, cif_root, **kwargs)

    @classmethod
    def from_cifname(
        cls, cifname, model_path, data_root, downstream="", cif_root=None, **kwargs
    ):
        """
        Create PatchVisualizer from cif name. cif must be in test.json or test_{downstream}.json.

        :param cifname : (str) name or path of cif. Data matching the corresponding cif name is retrieved from the dataset.
        :param model_path: (str) path of model from fine-tuned MOFTransformer with format '.ckpt'
        :param data_root: (str) path of dataset directory obtained from 'prepared_data.py. (see Dataset Preparation)
                MOFs to be visualized must exist in {dataset_folder}/test.json or {dataset_folder}/test_{downstream}.json,
                and {dataset_folder}/test folder. *.graphdata, *.grid, *.griddata16 files should be existed in {dataset_folder}/test folder.
        :param downstream: (str, optional) Use if data are existed in {dataset_folder}/test_{downstream}.json (default:'')
        :param cif_root: (str, optional) path of directory including cif file. The cif lists in the dataset folder should be included.
                If not specified, it is automatically specified as a {dataset_folder}/test folder.
        :param kwargs:
            figsize : (float, float) figure size
            view_init : (float, float) view init from matplotlib
            show_axis : <bool> If True, axis are visible. (default : False)
            show_colorbar : <bool> If True, colorbar are visible. (default : True)
            cmap : (str or matplotlib.colors.ListedColormap) color map used in figure. (default : None)
            num_patches : (int, int, int) number of patches (default : (6, 6, 6))
            max_length : <float> max p_lattice length of structure file (Å)
            min_length: <float> min p_lattice length of structure file (Å)
        :return: PatchVisualizer class for index
        """
        model, data_iter = get_model_and_datamodule(model_path, data_root, downstream)
        batch = get_batch_from_cif_id(data_iter, cifname)
        if cif_root is None:
            cif_root = os.path.join(data_root, "test")
        return cls.from_batch(batch, 0, model, cif_root, **kwargs)

    def __repr__(self):
        return f"class <PatchVisualizer> from {self.cif_id}"

    @property
    def cif_id(self):
        if cif_id := self.kwargs.get("cifname"):
            return cif_id
        else:
            return Path(self.path_cif).stem

    @property
    def cmap(self):
        return self.kwargs.get("cmap", None)

    @cmap.setter
    def cmap(self, cmap):
        if isinstance(cmap, (str, ListedColormap)) or cmap is None:
            self.kwargs["cmap"] = cmap
        else:
            raise TypeError(
                f"cmap must be str, ListedColormap, or None, not {type(cmap)}"
            )

    @property
    def num_patches(self):
        return self.kwargs.get("num_patches", (6, 6, 6))

    @property
    def figsize(self):
        return self.kwargs.get("figsize", DEFAULT_FIGSIZE)

    @figsize.setter
    def figsize(self, figsize):
        if not isinstance(figsize, Sequence):
            raise TypeError(f"figsize must be tuple or list, not {type(figsize)}")
        elif len(figsize) != 2:
            raise ValueError(f"figsize must be (float, float) not {figsize}")
        self.kwargs["figsize"] = figsize
        self._sync_cbar_kwargs()

    @property
    def view_init(self):
        return self.kwargs.get("view_init", DEFAULT_VIEW_INIT)

    @view_init.setter
    def view_init(self, view_init):
        if not isinstance(view_init, Sequence):
            raise TypeError(f"view_init must be tuple or list, not {type(view_init)}")
        elif len(view_init) != 2:
            raise ValueError(f"view_init must be (float, float) not {view_init}")
        self.kwargs["view_init"] = view_init

    @property
    def show_axis(self):
        return self.kwargs.get("show_colorbar", False)

    @show_axis.setter
    def show_axis(self, show_axis):
        if not isinstance(show_axis, bool):
            raise TypeError(f"show_axis must be bool, not {type(show_axis)}")
        self.kwargs["show_axis"] = show_axis

    @property
    def show_colorbar(self):
        return self.kwargs.get("show_colorbar", True)

    @show_colorbar.setter
    def show_colorbar(self, show_colorbar):
        if not isinstance(show_colorbar, bool):
            raise TypeError(f"show_colorbar must be bool, not {type(show_colorbar)}")
        self.kwargs["show_colorbar"] = show_colorbar

    @property
    def atomic_scale(self):
        return self.figsize[0] * self.figsize[1]

    def set_default(self):
        self.kwargs = {}
        self._sync_cbar_kwargs()

    def _sync_cbar_kwargs(self):
        figsize = self.figsize
        self.cbar_kwargs["labelpad"] = figsize[0] * 2
        self.cbar_kwargs["labelsize"] = figsize[0] * 1.5
        self.cbar_kwargs["fontsize"] = figsize[0]

    def _get_indice_inside_patch(self, patch_position, ep=0.0):
        def is_inside_patch(r_pos, pos_patch, num_patch):
            lower_bound = (pos_patch - ep) / num_patch < r_pos
            upper_bound = r_pos < (pos_patch + 1 + ep) / num_patch
            return np.logical_and(lower_bound, upper_bound)

        relative_position = self.s_atoms.get_scaled_positions()
        position_bools = [
            is_inside_patch(r_pos, pos_patch, num_patch)
            for r_pos, pos_patch, num_patch in zip(
                relative_position.T, patch_position, self.num_patches
            )
        ]  # x, y, z

        position_bool = np.logical_and.reduce(position_bools)  # total
        (indice,) = np.where(position_bool)
        return indice

    def _grid_attention_rank(self, rank):
        heatmap_grid = self.heatmap_grid
        sort = np.flip(
            np.unravel_index(np.argsort(heatmap_grid, axis=None), heatmap_grid.shape),
            axis=-1,
        )
        if isinstance(rank, int):
            return np.array(sort)[:, rank][np.newaxis, :]
        elif isinstance(rank, Iterable):
            return np.array(sort)[:, tuple(rank)].T
        else:
            raise TypeError(f"rank must be int or iterable, not {type(rank)}")

    def set_colorbar_options(self, default=False, **cbar_kwargs):
        if default:
            self.cbar_kwargs = get_default_cbar_kwargs(self.figsize)
        else:
            for key, value in cbar_kwargs.items():
                if key in self.cbar_kwargs:
                    self.cbar_kwargs[key] = value


    def draw_graph(self,
                minatt=-0.01, 
                maxatt=0.01, *, 
                alpha=0.7, 
                atomic_scale_factor=4, 
                grid_scale_factor=1, 
                att_scale_factor=3,
                add_feat_num, 
                return_fig=False, **kwargs):
                
        heatmap_graph = self.heatmap_graph
        lattice = self.p_lattice
        atoms = self.p_atoms
        coordinate_scale = float(kwargs.get("coordinate_scale", 0.88))
        plot_atoms, plot_lattice, cell_start = compact_plot_geometry(
            atoms, lattice=lattice, coordinate_scale=coordinate_scale
        )
        mode = kwargs.get("mode","atom-shap")
        highlight_positive = kwargs.get("highlight_positive", None)
        highlight_radius = kwargs.get("highlight_radius", 4.0)
        highlight_context_alpha = kwargs.get("highlight_context_alpha", 0.5)

        fig, ax = get_fig_ax(**self.kwargs)

        cmap = get_cmap(kwargs.get("cmap", "bwr"))
        
        set_fig_ax(ax, **kwargs)

        # Ensure we only use SHAP values for atoms. Remove geometric feature additions.
        if add_feat_num > 0:
            atom_shap = heatmap_graph[:, :-add_feat_num].flatten()
        else:
            atom_shap = heatmap_graph.flatten()
        if hasattr(atom_shap, "detach"):
            atom_shap = atom_shap.detach().cpu().numpy()
        else:
            atom_shap = np.asarray(atom_shap)
        # The 3D atom plots do not render additional feature tokens, so the
        # color scale should be derived from atom SHAPs only and remain
        # symmetric around zero: negative -> blue, positive -> red.
        max_abs_att = float(np.max(np.abs(atom_shap))) if atom_shap.size else 1.0
        if max_abs_att == 0.0:
            max_abs_att = 1e-12
        minatt = -max_abs_att
        maxatt = max_abs_att
        try:
            norm = TwoSlopeNorm(vmin=minatt, vcenter=0, vmax=maxatt)
        except ValueError:
            # If the values are not suitable for TwoSlopeNorm, use a simple normalization
            norm = TwoSlopeNorm(vmin=minatt, vcenter=(minatt+maxatt)/2, vmax=maxatt)
            
        if mode=="atom-shap":
            draw_cell(ax, plot_lattice, s_point=cell_start, color="black")
            if highlight_positive is not None:
                # Highlight positive SHAP values
                draw_atoms_highlight(ax, plot_atoms, self.atomic_scale * atomic_scale_factor * grid_scale_factor,
                                     shap_values=atom_shap, highlight_positive=highlight_positive,
                                     radius=highlight_radius, context_alpha=highlight_context_alpha)
            else:
                draw_atoms(ax, plot_atoms, self.atomic_scale * atomic_scale_factor * grid_scale_factor)
            
            colors = cmap(norm(atom_shap))
            atomic_scale = self.atomic_scale * att_scale_factor * grid_scale_factor * atomic_scale_factor
            if highlight_positive is None:
                draw_heatmap_graph(ax, plot_atoms, colors, atomic_scale, alpha)
            draw_cell(ax, plot_lattice, s_point=cell_start, color="black")
            if highlight_positive is not None:
                # Highlight positive SHAP values
                draw_atoms_highlight(ax, plot_atoms, self.atomic_scale * atomic_scale_factor * grid_scale_factor,
                                     shap_values=atom_shap, highlight_positive=highlight_positive,
                                     radius=highlight_radius, context_alpha=highlight_context_alpha)
            else:
                draw_atoms(ax, plot_atoms, self.atomic_scale * atomic_scale_factor * grid_scale_factor)      

        elif mode=="shap-only":
            draw_cell(ax, plot_lattice, s_point=cell_start, color="black")
            if highlight_positive is not None:
                # Highlight positive SHAP values
                draw_atoms_highlight(ax, plot_atoms, self.atomic_scale * atomic_scale_factor * grid_scale_factor,
                                        shap_values=atom_shap, cmap=cmap, minatt=minatt, maxatt=maxatt,
                                        highlight_positive=highlight_positive,
                                        radius=highlight_radius, shap_only=True,
                                        context_alpha=highlight_context_alpha)
            else:
                draw_atoms(ax, plot_atoms, self.atomic_scale * atomic_scale_factor * grid_scale_factor,
                        shap_values=atom_shap, cmap=cmap, minatt=minatt, maxatt=maxatt)

        if kwargs.get("show_colorbar", self.show_colorbar):
            draw_colorbar(fig, ax, cmap, minatt, maxatt, norm=norm, label="SHAP Value"+ " ("+ kwargs.get("SHAP_unit_label", " ") + ")" if kwargs.get("SHAP_unit_label", " ") != " " else "SHAP Value", 
                          **self.cbar_kwargs)
            if mode != "shap-only":
                # Create circle legend handles (no text inside circles).
                legend_handles, legend_labels = create_circle_legend_handles(atoms)
                
                # Add the legend with our custom circle handler.
                ax.legend(
                legend_handles,
                legend_labels,
                handler_map={mpatches.Circle: HandlerCircleNoText()},
                loc='upper right',
                bbox_to_anchor=(1.0, 1.0),
                borderaxespad=0.0,
                labelspacing=0.4,
                handletextpad=0.8,
                borderpad=0.5,
                title="Elements",
                frameon=False
                )
            set_axes_equal(ax, scale_factor=grid_scale_factor)

        if return_fig:
            return fig, ax
        else:
            plt.show()

    def draw_graph_dual(self,
                    minatt=-0.01, 
                    maxatt=0.01, *,
                    alpha=0.7, 
                    atomic_scale_factor=4, 
                    grid_scale_factor=1, 
                    att_scale_factor=3,
                    add_feat_num, 
                    return_fig=False, **kwargs):
        """
        Draws a dual visualization: one subplot showing the standard "atom-shap" mode 
        and the other showing "shap-only" mode side by side.
        Both subplots are interactive.
        """

        class HandlerCircleNoText(HandlerPatch):
            def create_artists(self, legend, artist, xdescent, ydescent,
                            width, height, fontsize, trans):
                center = 0.5 * width - xdescent, 0.5 * height - ydescent
                p = mpatches.Circle(xy=center, radius=height,
                                    facecolor=artist.get_facecolor())
                self.update_prop(p, artist, legend)
                p.set_transform(trans)
                return [p]

        heatmap_graph = self.heatmap_graph
        lattice = self.p_lattice
        atoms = self.p_atoms
        coordinate_scale = float(kwargs.get("coordinate_scale", 0.88))
        plot_atoms, plot_lattice, cell_start = compact_plot_geometry(
            atoms, lattice=lattice, coordinate_scale=coordinate_scale
        )
        highlight_positive = kwargs.get("highlight_positive", None)
        highlight_radius = kwargs.get("highlight_radius", 4.0)
        highlight_context_alpha = kwargs.get("highlight_context_alpha", 0.5)

        # Create a figure with 1 row and 2 columns for two 3D subplots.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10),
                                    subplot_kw={'projection': '3d'})
        fig.subplots_adjust(left=0.03, right=0.97, bottom=0.04, top=0.96, wspace=0.06)

        # Turn off the axis lines and grid for both subplots.
        ax1.set_axis_off()
        ax1.grid(False)
        ax2.set_axis_off()
        ax2.grid(False)

        # Diverging colormap and normalization.
        cmap = get_cmap(kwargs.get("cmap", "bwr"))
        # Slice out any added features from SHAP values
        if add_feat_num > 0:
            atom_shap = heatmap_graph[:, :-add_feat_num].flatten()
        else:
            atom_shap = heatmap_graph.flatten()
        if hasattr(atom_shap, "detach"):
            atom_shap = atom_shap.detach().cpu().numpy()
        else:
            atom_shap = np.asarray(atom_shap)
        # Match the standalone atom plot: ignore additional-feature tokens and
        # keep the SHAP color scale symmetric around zero.
        max_abs_att = float(np.max(np.abs(atom_shap))) if atom_shap.size else 1.0
        if max_abs_att == 0.0:
            max_abs_att = 1e-12
        minatt = -max_abs_att
        maxatt = max_abs_att
        try:
            norm = TwoSlopeNorm(vmin=minatt, vcenter=0, vmax=maxatt)
        except ValueError:
            # Keep the dual plot on the same normalization path as draw_graph.
            norm = TwoSlopeNorm(vmin=minatt, vcenter=(minatt + maxatt) / 2, vmax=maxatt)

        # ============ Left Subplot: "atom-shap" Mode ============
        # Draw the cell, default-colored atoms, and SHAP overlay
        draw_cell(ax1, plot_lattice, s_point=cell_start, color="black")
        if highlight_positive is not None:
                # Highlight positive SHAP values
            draw_atoms_highlight(ax1, plot_atoms, self.atomic_scale * atomic_scale_factor * grid_scale_factor,
                                     shap_values=atom_shap, highlight_positive=highlight_positive,
                                     radius=highlight_radius, context_alpha=highlight_context_alpha)
        else:
                draw_atoms(ax1, plot_atoms, self.atomic_scale * atomic_scale_factor * grid_scale_factor)
        
        colors = cmap(norm(atom_shap))
        atomic_scale_overlay = (self.atomic_scale * att_scale_factor *
                                grid_scale_factor * atomic_scale_factor)
        
        if highlight_positive is None:
            draw_heatmap_graph(ax1, plot_atoms, colors, atomic_scale_overlay, alpha)

        draw_cell(ax1, plot_lattice, s_point=cell_start, color="black")
        if highlight_positive is not None:
                # Highlight positive SHAP values
            draw_atoms_highlight(ax1, plot_atoms, self.atomic_scale * atomic_scale_factor * grid_scale_factor,
                                     shap_values=atom_shap, highlight_positive=highlight_positive,
                                     radius=highlight_radius, context_alpha=highlight_context_alpha)
        else:
                draw_atoms(ax1, plot_atoms, self.atomic_scale * atomic_scale_factor * grid_scale_factor)

        # Add the legend on the left subplot
        legend_handles, legend_labels = create_circle_legend_handles(atoms)
        ax1.legend(
            legend_handles,
            legend_labels,
            handler_map={mpatches.Circle: HandlerCircleNoText()},
            loc='upper left',
            bbox_to_anchor=(0.02, 0.98),
            borderaxespad=0.0,
            labelspacing=0.4,
            handletextpad=0.8,
            borderpad=0.5,
            title="Elements",
            frameon=False
        )

        # ============ Right Subplot: "shap-only" Mode ============
        draw_cell(ax2, plot_lattice, s_point=cell_start, color="black")
        if highlight_positive is not None:
            # Highlight positive SHAP values
            draw_atoms_highlight(ax2, plot_atoms, self.atomic_scale * atomic_scale_factor * grid_scale_factor,
                                    shap_values=atom_shap, cmap=cmap, minatt=minatt, maxatt=maxatt,
                                    highlight_positive=highlight_positive,
                                    radius=highlight_radius, shap_only=True,
                                    context_alpha=highlight_context_alpha)
        else:
            draw_atoms(ax2, plot_atoms, self.atomic_scale * atomic_scale_factor * grid_scale_factor,
                    shap_values=atom_shap, cmap=cmap, minatt=minatt, maxatt=maxatt)

        # Optionally add a shared colorbar on one of the subplots
        if kwargs.get("show_colorbar", self.show_colorbar):
            draw_colorbar(fig, ax2, cmap, minatt, maxatt, norm=norm, label="SHAP Value"+ " ("+ kwargs.get("SHAP_unit_label", " ") + ")" if kwargs.get("SHAP_unit_label", " ") != " " else "SHAP Value", 
                          **self.cbar_kwargs)

        # Keep objects to scale but hide axes
        set_axes_equal(ax1, scale_factor=grid_scale_factor)
        set_axes_equal(ax2, scale_factor=grid_scale_factor)

        if return_fig:
            return fig, (ax1, ax2)
        else:
            plt.show()
    


    def animate(self, func, frame=360, interval=20, savefile=None, fps=30):
        def turn(i, ax, fig, **kwargs):
            view_init = kwargs.get("view_init", self.view_init)
            ax.view_init(elev=view_init[0], azim=i)
            return []  # Return an empty list of artists
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            kwargs["return_fig"] = True
            fig, ax = func(*args, **kwargs)
            anim = animation.FuncAnimation(
                fig,
                partial(turn, ax=ax, fig=fig, **kwargs),
                init_func=lambda: [],  # Return an empty list instead of the figure
                frames=frame,
                interval=interval,
                blit=True,
            )
            if savefile:
                anim.save(savefile, fps=fps, dpi=100)

            plt.show()
            return anim

        return wrapper


    def animate_dual(self, func, frame=360, interval=20, savefile=None, fps=30):
        """
        Animates a dual-subplot 3D visualization by rotating both subplots in sync.
        Assumes that `func` returns (fig, (ax1, ax2)) when called with return_fig=True.
        """
        def turn(i, ax1, ax2, fig, **kwargs):
            view_init = kwargs.get("view_init", (30, 0))  # (elev, initial_azim)
            elev = view_init[0]
            ax1.view_init(elev=elev, azim=i)
            ax2.view_init(elev=elev, azim=i)
            return []  # Return an empty list of artists

        @wraps(func)
        def wrapper(*args, **kwargs):
            kwargs["return_fig"] = True
            # Unpack the axes from the function's return value:
            fig, axes = func(*args, **kwargs)
            # Ensure axes is a tuple of two axes:
            if isinstance(axes, tuple) and len(axes) == 2:
                ax1, ax2 = axes
            else:
                raise ValueError("The drawing function must return (fig, (ax1, ax2)).")
            
            anim = animation.FuncAnimation(
                fig,
                partial(turn, ax1=ax1, ax2=ax2, fig=fig, **kwargs),
                init_func=lambda: [],
                frames=frame,
                interval=interval,
                blit=True,
            )
            if savefile:
                anim.save(savefile, fps=fps, dpi=100)
            plt.show()
            return anim

        return wrapper
