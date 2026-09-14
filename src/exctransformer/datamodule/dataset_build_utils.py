import torch 
import numpy as np
import os, sys
import h5py
import zipfile
import io
import logging
import re
import copy
from exctransformer.config import config as config_

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

config = copy.deepcopy(config_())



def merge_graphs(atom_in_fea_list, nbr_fea_list, nbr_fea_idx_list):
    """
    Merges multiple graphs into a single batched graph.
    Adjusts neighbor indices to reference the concatenated node feature tensor.
    
    Parameters:
        atom_in_fea_list: List of Tensors [(N_i, atom_fea_len), ...]
        nbr_fea_list: List of Tensors [(N_i, M, nbr_fea_len), ...]
        nbr_fea_idx_list: List of LongTensors [(N_i, M), ...]
    
    Returns:
        merged_atom_in_fea: Tensor of shape (N_total, atom_fea_len)
        merged_nbr_fea: Tensor of shape (N_total, M, nbr_fea_len)
        merged_nbr_fea_idx: Tensor of shape (N_total, M)
    """
    merged_atom_in_fea = torch.cat(atom_in_fea_list, dim=0)
    merged_nbr_fea = torch.cat(nbr_fea_list, dim=0)
    
    offset = 0
    adjusted_idx = []
    for idx in nbr_fea_idx_list:
        # Adjust indices based on the cumulative number of nodes seen so far.
        adjusted_idx.append(idx + offset)
        offset += idx.size(0)
    
    merged_nbr_fea_idx = torch.cat(adjusted_idx, dim=0)
    return merged_atom_in_fea, merged_nbr_fea, merged_nbr_fea_idx


def load_h5_object(h5_obj):
    """Recursively load data from an h5py object."""
    if isinstance(h5_obj, h5py.Dataset):
        return h5_obj[()]  # Load the dataset's entire content.
    elif isinstance(h5_obj, h5py.Group):
        # Recursively load each item in the group.
        return {key: load_h5_object(h5_obj[key]) for key in h5_obj.keys()}
    else:
        raise ValueError("Unknown h5py object type.")

def load_flat_sample(h5_file):
    # Load the entire HDF5 content.
    sample = {k: load_h5_object(h5_file[k]) for k in h5_file.keys()}
    # If there's a single top-level key, return its contents.
    if len(sample) == 1:
        return list(sample.values())[0]
    return sample



def open_or_create_h5_file(filepath: str):
    # Ensure the parent directory exists.
    parent_dir = os.path.dirname(filepath)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    
    # Check if the file exists and is a valid HDF5 file.
    if os.path.exists(filepath):
        try:
            with h5py.File(filepath, 'r') as f:
                pass  # File is valid.
            file_mode = 'a'  # Append mode for an existing valid file.
        except (OSError, IOError):
            # File exists but is not a valid HDF5 file.
            os.remove(filepath)
            file_mode = 'w'
    else:
        file_mode = 'w'  # Create a new file.
    
    return h5py.File(filepath, file_mode)


def create_compressed_dataset(group, name, data, dtype=None):
    arr = np.array(data) if not isinstance(data, np.ndarray) else data
    if arr.shape == ():
        return group.create_dataset(name, data=data, dtype=dtype)
    else:
        return group.create_dataset(name, data=data, dtype=dtype,
                                    compression="gzip", compression_opts=9)


def zip_h5_file(h5_file_path, zip_output_path, arcname=None):
    """
    Zip an HDF5 file into a zip archive.
    
    Parameters:
        h5_file_path (str): Path to the HDF5 file to be zipped.
        zip_output_path (str): Path to the output zip archive.
        arcname (str): Name for the file inside the archive. Defaults to the base name of h5_file_path.
    """
    if arcname is None:
        arcname = os.path.basename(h5_file_path)
    with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(h5_file_path, arcname=arcname)


def read_dataset_from_h5(file_path, dataset_path, zipped=False, h5_filename=None):
    """    
    Parameters:
        file_path (str): Path to the HDF5 file or to the zip archive containing it.
        dataset_path (str): Full path of the dataset within the HDF5 file.
        zipped (bool): If True, 'file_path' is treated as a zip archive.
        h5_filename (str): Name of the HDF5 file inside the zip archive (required if zipped is True).
    
    Returns:
        The dataset loaded into memory.
    """
    if zipped:
        if h5_filename is None:
            raise ValueError("h5_filename must be provided when zipped is True.")
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            with zip_file.open(h5_filename) as h5_file_bytes:
                h5_data = h5_file_bytes.read()
                h5_buffer = io.BytesIO(h5_data)
                with h5py.File(h5_buffer, 'r') as f:
                    return f[dataset_path][()]
    else:
        with h5py.File(file_path, 'r') as f:
            return f[dataset_path][()]


def read_group_from_h5(file_path, group_path, zipped=False, h5_filename=None):
    """
    Recursively read an entire group (with its subgroups/datasets) from an HDF5 file.
    Supports both plain files and files stored within a zip archive.
    
    Parameters:
        file_path (str): Path to the HDF5 file or to the zip archive containing it.
        group_path (str): Full path of the group within the HDF5 file.
        zipped (bool): If True, 'file_path' is treated as a zip archive.
        h5_filename (str): Name of the HDF5 file inside the zip archive (required if zipped is True).
    
    Returns:
        A nested dictionary representing the group's structure and data.
    """
    def recursive_load(h5_obj):
        group_dict = {}
        for key, item in h5_obj.items():
            if isinstance(item, h5py.Group):
                group_dict[key] = recursive_load(item)
            else:
                group_dict[key] = item[()]
        return group_dict

    if zipped:
        if h5_filename is None:
            raise ValueError("h5_filename must be provided when zipped is True.")
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            with zip_file.open(h5_filename) as h5_file_bytes:
                h5_data = h5_file_bytes.read()
                h5_buffer = io.BytesIO(h5_data)
                with h5py.File(h5_buffer, 'r') as f:
                    return recursive_load(f[group_path])
    else:
        with h5py.File(file_path, 'r') as f:
            return recursive_load(f[group_path])
        


def rename_output_folder(output_folder: str, num_files: int):
    """
    Renames the output folder by removing any previous set of numbers and appending the new number of successfully built .h5 files.

    Args:
        output_folder (str): The original output folder path.
        num_files (int): The number of successfully built .h5 files.
    """
    new_folder_name = re.sub(r'\d+', str(num_files), output_folder)
    if new_folder_name == output_folder:
        new_folder_name = f"{output_folder}_{num_files}"
    os.rename(output_folder, new_folder_name)
    logger.info(f"Output folder renamed to: {new_folder_name}")




def update_progress(current_graph, total_graphs, bar_length=40, use_tqdm=False):
    """
    Dynamically updates the console with progress information.
    
    Args:
        current_graph (int): The number of graphs built so far.
        total_graphs (int): The total number of graphs to build.
        bar_length (int): Length of the progress bar (if used).
        use_tqdm (bool): Whether to use tqdm for the progress bar.
    """
    if use_tqdm:
        from tqdm import tqdm
        # Initialize the tqdm progress bar only once with the current progress.
        if not hasattr(update_progress, 'tqdm_bar'):
            update_progress.tqdm_bar = tqdm(total=total_graphs, initial=current_graph)
        # Increment the progress bar by 1 for each call.
        update_progress.tqdm_bar.update(1)
        # Close the bar if we have reached the total.
        if current_graph + 1 == total_graphs:
            update_progress.tqdm_bar.close()
    else:
        percent = float(current_graph) / total_graphs
        hashes = '#' * int(round(percent * bar_length))
        spaces = ' ' * (bar_length - len(hashes))
        sys.stdout.write(f"\r[{hashes}{spaces}] {int(percent * 100)}% ({current_graph}/{total_graphs})")
        sys.stdout.flush()
        if current_graph == total_graphs:
            print()  # Add a newline at the end


def count_h5_dataset(output_folder: str, dataset_length: int):
    """
    Counts the number of .h5 files in the output folder and logs the result.

    Args:
        output_folder (str): Path to the folder containing .h5 files.
        dataset_length (int): Expected number of .h5 files.
    """
    # Count the number of .h5 files in the output folder
    h5_files = [f for f in os.listdir(output_folder) if f.endswith('.h5')]
    num_h5_files = len(h5_files)

    logger.info("-" * 48)
    logger.info(f"Total number of graph.h5 files built successfully: {num_h5_files}/{dataset_length}")
    logger.info("-" * 48)
    return num_h5_files
