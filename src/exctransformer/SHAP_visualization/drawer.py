#cgstransformer version 1.0
import numpy as np
from itertools import combinations
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.ticker as mticker
from ase.data import covalent_radii
from exctransformer.SHAP_visualization.utils import plot_cube, get_shap_color
from exctransformer.SHAP_visualization import cpk_colors
from matplotlib.colors import TwoSlopeNorm

# Adapted from  https://github.com/hspark1212/MOFTransformer

def draw_colorbar(fig, ax, cmap, minatt, maxatt, norm=None, label="SHAP Value", **cbar_kwargs):
    """
    Draws a colorbar that maps SHAP values to colors.
    If norm is not provided, a default normalization is used.
    """
    from matplotlib.colors import Normalize
    if norm is None:
        norm = Normalize(vmin=minatt, vmax=maxatt)
    smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(smap, ax=ax,
                        fraction=cbar_kwargs.get("fraction", 0.1),
                        shrink=cbar_kwargs.get("shrink", 0.5))
    cbar.ax.tick_params(labelsize=cbar_kwargs.get("fontsize", 10))
    num_ticks = cbar_kwargs.get("num_ticks", 6)
    # Create ticks linearly spaced in the normalized space
    ticks = np.linspace(norm.vmin, norm.vmax, num_ticks)
    ticks_label = np.round(ticks, decimals=cbar_kwargs.get("decimals", 4))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticks_label)
    cbar.ax.set_ylabel(label, rotation=270,
                       labelpad=cbar_kwargs.get("labelpad", 10),
                       fontdict={"size": cbar_kwargs.get("labelsize", 12)})


def draw_line(ax, pos1, pos2, **kwargs):
    """
    Draw line from position 1 to position 2
    Parameters:
        :param ax: <matplotlib.axes> figure axis
        :param pos1: <np.array> starting point position
        :param pos2: <np.array> end point position
        :param kwargs: matplotlib plot3D kwargs
        :return:
    """
    ax.plot3D(*zip(pos1, pos2), **kwargs)

def draw_cell(ax, lattice, s_point=None, **kwargs):
    """
    Draw unit-p_lattice p_lattice using matplotlib
    :param ax: <matplotlib.axes> figure axis
    :param lattice: <np.array> p_lattice vectors (3 X 3 matrix)
    :param s_point: <np.array> start point of p_lattice
    :param kwargs: matplotlib plot3D kwargs
    """
    vec1, vec2, vec3 = lattice
    if s_point is None:
        s_point = np.zeros(3)

    opp_vec = vec1 + vec2 + vec3 + s_point

    for v1, v2 in combinations([vec1, vec2, vec3], 2):
        draw_line(ax, s_point, s_point + v1, **kwargs)
        draw_line(ax, s_point, s_point + v2, **kwargs)
        draw_line(ax, s_point + v1, s_point + v1 + v2, **kwargs)
        draw_line(ax, s_point + v2, s_point + v1 + v2, **kwargs)
        draw_line(ax, s_point + v1 + v2, opp_vec, **kwargs)


def draw_atoms(ax, atoms, atomic_scale, shap_values=None, cmap="bwr", minatt=-0.01, maxatt=0.01):
    """
    Draw atoms using matplotlib.
    
    If shap_values is provided, each atom is colored based on its SHAP value using a diverging
    colormap such that negative values are bluish and positive values are reddish.
    Otherwise, standard CPK colors are used.
    
    Parameters:
      ax          : matplotlib.axes figure axis.
      atoms       : ASE Atoms object.
      atomic_scale: Scaling factor for atom sizes.
      shap_values : Optional array of SHAP values (one per atom).
      cmap        : Colormap name or object for SHAP-based coloring (default "bwr").
      minatt      : Minimum SHAP value for normalization.
      maxatt      : Maximum SHAP value for normalization.
    """
    coords = atoms.get_positions()
    atomic_numbers = atoms.get_atomic_numbers()
    atomic_sizes = np.array([covalent_radii[i] for i in atomic_numbers])
    
    if shap_values is not None:
        # Ensure that the number of SHAP values matches the number of atoms.
        # If there is an extra token (e.g., a class token), slice it off.
        if len(shap_values) != len(atomic_numbers):
            shap_values = shap_values[1:len(atomic_numbers)+1]
        try:
            norm = TwoSlopeNorm(vmin=minatt, vcenter=0, vmax=maxatt)
        except ValueError:
            norm = TwoSlopeNorm(vmin=minatt, vcenter=(minatt + maxatt) / 2, vmax=maxatt)
        # Obtain the colormap (if a string is provided, convert it)
        colormap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        atom_colors = [colormap(norm(val)) for val in shap_values]
    else:
        atom_colors = np.array([cpk_colors.get_cpk_color(i) for i in atomic_numbers])
    
    ax.scatter(
        xs=coords[:, 0],
        ys=coords[:, 1],
        zs=coords[:, 2],
        c=atom_colors,
        s=atomic_sizes * atomic_scale,
        marker="o",
        edgecolor="black",
        linewidths=0.8,
        alpha=1.0,
    )


def draw_heatmap_grid(ax, positions, colors, lattice, num_patches, alpha=0.5, **kwargs):
    cubes = plot_cube(
        positions,
        colors,
        lattice=lattice,
        num_patches=num_patches,
        edgecolor=None,
        alpha=alpha,
    )
    ax.add_collection3d(cubes, **kwargs)


def draw_heatmap_graph(ax, atoms, colors, atomic_scale, alpha):
    """
    Draws a 3D heatmap graph on the given matplotlib axis,
    where each atom is colored according to its SHAP value.
    
    Parameters:
      ax           : matplotlib 3D axis.
      atoms        : ASE Atoms object.
      colors       : Array of RGBA colors (one per atom).
      atomic_scale : Scale for the atom markers.
      alpha        : Transparency level.
    """
    coords = atoms.get_positions()
    ax.scatter(
         xs=coords[:, 0],
         ys=coords[:, 1],
         zs=coords[:, 2],
         c=colors,   # Use the full colors array (one color per atom)
         s=atomic_scale,
         marker="o",
         linewidth=0,
         alpha=alpha,
    )
    

def draw_atoms_highlight(ax, atoms, atomic_scale, shap_values, cmap="bwr",
                         minatt=-0.01, maxatt=0.01, highlight_positive=True,
                         radius=4.0, shap_only=False, context_alpha=0.5):
    """
    Draws the highlighted atom and its neighbors (within a given radius) using matplotlib,
    following the same coloring protocol as draw_atoms.
    The atoms of interest are drawn in color, and all other atoms are plotted in light grey.
    
    Parameters:
      ax                : matplotlib.axes figure axis.
      atoms             : ASE Atoms object.
      atomic_scale      : Scaling factor for atom sizes.
      shap_values       : Array of SHAP values (one per atom).
      cmap              : Colormap name or object for SHAP-based coloring (default "bwr").
      minatt            : Minimum SHAP value for normalization.
      maxatt            : Maximum SHAP value for normalization.
      highlight_positive: If True, highlight atom with highest positive SHAP value;
                          if False, highlight atom with lowest (most negative) SHAP value.
      radius            : Neighborhood radius (default 4.0 Å).
      shap_only         : If True, use the SHAP-based coloring for highlighted atoms; otherwise use CPK colors.
      context_alpha     : Transparency for non-highlighted context atoms.
    """
 

    # Get atom positions, atomic numbers, and sizes.
    coords = atoms.get_positions()
    atomic_numbers = atoms.get_atomic_numbers()
    atomic_sizes = np.array([covalent_radii[i] for i in atomic_numbers])
    
    # Prepare colors following the protocol of draw_atoms.
    if shap_only:
        # Adjust SHAP values if necessary.
        if len(shap_values) != len(atomic_numbers):
            shap_values = shap_values[1:len(atomic_numbers)+1]
        try:
            norm = TwoSlopeNorm(vmin=minatt, vcenter=0, vmax=maxatt)
        except ValueError:
            norm = TwoSlopeNorm(vmin=minatt, vcenter=(minatt + maxatt) / 2, vmax=maxatt)
        colormap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        atom_colors = [colormap(norm(val)) for val in shap_values]
    else:
        atom_colors = np.array([cpk_colors.get_cpk_color(i) for i in atomic_numbers])
    
    # Determine the central atom index based on SHAP value.
    shap_arr = np.array(shap_values)
    central_idx = np.argmax(shap_arr) if highlight_positive else np.argmin(shap_arr)
    central_coord = coords[central_idx]
    
    # Compute Euclidean distances from the central atom.
    distances = np.linalg.norm(coords - central_coord, axis=1)
    # Find neighbor indices within the given radius.
    neighbor_indices = np.where(distances <= radius)[0]
    
    # Determine indices for context (atoms not of interest).
    all_indices = np.arange(len(coords))
    context_indices = np.setdiff1d(all_indices, neighbor_indices)
    
    # Plot context atoms in light grey.
    ax.scatter(
        xs=coords[context_indices, 0],
        ys=coords[context_indices, 1],
        zs=coords[context_indices, 2],
        c="lightgrey",
        s=atomic_sizes[context_indices] * atomic_scale,
        marker="o",
        edgecolor="black",
        linewidths=0.8,
        alpha=context_alpha,
    )
    
    # Plot the atoms of interest.
    ax.scatter(
        xs=coords[neighbor_indices, 0],
        ys=coords[neighbor_indices, 1],
        zs=coords[neighbor_indices, 2],
        c=np.array(atom_colors)[neighbor_indices],
        s=atomic_sizes[neighbor_indices] * atomic_scale,
        marker="o",
        edgecolor="black",
        linewidths=1.2,
        alpha=1.0,
    )
