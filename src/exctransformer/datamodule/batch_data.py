import io
import os
import warnings
import h5py
import zipfile
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


from exctransformer.datamodule.dataset_build_utils import (
    load_flat_sample,
    merge_graphs
)


def safe_load_flat_sample(f, file_identifier=""):
    """
    Safely load a sample from an h5py file object.
    If an error occurs (e.g., file corruption), a warning is issued and None is returned.
    """
    try:
        sample = load_flat_sample(f)
        return sample
    except Exception as e:
        warnings.warn(f"Failed to load sample from {file_identifier}: {e}")
        return None

class GraphDataset(Dataset):
    def __init__(self, dir_path, zipped=False, lazy=True, transform=None):
        """
        Custom dataset for graph structures stored in HDF5 files within a directory or a zip file.
        
        When lazy=True, __getitem__ returns a key so that the collate function loads
        the required batch data all at once.
        """
        self.dir_path = dir_path
        self.zipped = zipped
        self.lazy = lazy
        self.transform = transform

        # Cache the graph keys from the HDF5 files.
        self.graph_keys = []
        if self.zipped:
            with zipfile.ZipFile(self.dir_path, 'r') as zip_file:
                for file_name in zip_file.namelist():
                    if file_name.endswith('.h5'):
                        self.graph_keys.append(file_name)
        else:
            for file_name in os.listdir(self.dir_path):
                if file_name.endswith('.h5'):
                    self.graph_keys.append(file_name)
        self.graph_keys.sort()

        # If not lazy, preload all data into memory with file integrity checks.
        if not self.lazy:
            self.data = []
            valid_keys = []
            if self.zipped:
                with zipfile.ZipFile(self.dir_path, 'r') as zip_file:
                    for key in self.graph_keys:
                        try:
                            with zip_file.open(key) as h5_file_bytes:
                                h5_data = h5_file_bytes.read()
                                h5_buffer = io.BytesIO(h5_data)
                                with h5py.File(h5_buffer, 'r') as f:
                                    sample = safe_load_flat_sample(f, file_identifier=key)
                        except Exception as e:
                            warnings.warn(f"Error loading file {key} in zip: {e}")
                            sample = None

                        if sample is not None:
                            if self.transform:
                                sample = self.transform(sample)
                            self.data.append(sample)
                            valid_keys.append(key)
            else:
                for key in self.graph_keys:
                    file_path = os.path.join(self.dir_path, key)
                    try:
                        with h5py.File(file_path, 'r') as f:
                            sample = safe_load_flat_sample(f, file_identifier=file_path)
                    except Exception as e:
                        warnings.warn(f"Error loading file {file_path}: {e}")
                        sample = None

                    if sample is not None:
                        if self.transform:
                            sample = self.transform(sample)
                        self.data.append(sample)
                        valid_keys.append(key)
            # Update keys and number of graphs based on successfully loaded files.
            self.graph_keys = valid_keys
            self.num_graphs = len(self.data)
        else:
            self.num_graphs = len(self.graph_keys)

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        # When lazy, simply return the key; actual data is loaded in the collate function.
        if self.lazy:
            return self.graph_keys[idx]
        else:
            return self.data[idx]
        

def load_batch_samples(keys, dir_path, zipped, transform):
    """
    Given a list of keys, load all corresponding samples from the HDF5 files.
    Each file is safely loaded; if a file is corrupted, it is skipped with a warning.
    """
    samples = []
    if zipped:
        with zipfile.ZipFile(dir_path, 'r') as zip_file:
            for key in keys:
                try:
                    with zip_file.open(key) as h5_file_bytes:
                        h5_data = h5_file_bytes.read()
                        h5_buffer = io.BytesIO(h5_data)
                        with h5py.File(h5_buffer, 'r') as f:
                            sample = safe_load_flat_sample(f, file_identifier=key)
                except Exception as e:
                    warnings.warn(f"Error loading file {key} in zip: {e}")
                    sample = None

                if sample is not None:
                    if transform:
                        sample = transform(sample)
                    samples.append(sample)
    else:
        for key in keys:
            file_path = os.path.join(dir_path, key)
            try:
                with h5py.File(file_path, 'r') as f:
                    sample = safe_load_flat_sample(f, file_identifier=file_path)
            except Exception as e:
                warnings.warn(f"Error loading file {file_path}: {e}")
                sample = None

            if sample is not None:
                if transform:
                    sample = transform(sample)
                samples.append(sample)
    return samples


def check_tensor_validity(tensor, name, sample_id):
    """
    Checks if the tensor contains any NaN or Inf values.
    Issues a warning if invalid values are found and returns False.
    Returns True if the tensor is valid.
    """
    if torch.isnan(tensor).any():
        warnings.warn(f"Sample {sample_id}: Tensor '{name}' contains NaN values. Skipping sample.", UserWarning)
        return False
    if torch.isinf(tensor).any():
        warnings.warn(f"Sample {sample_id}: Tensor '{name}' contains Inf values. Skipping sample.", UserWarning)
        return False
    return True

def collate_graphs(samples, include_additional_features=True):
    """
    Collates a batch of graph data samples into a single batch dictionary.
    """
    atom_fea_list = []
    nbr_fea_list = []
    nbr_fea_idx_list = []

    additional_features_list = []
    target_list = []
    cif_ids = []
    crystal_atom_idx = []
    num_atoms_per_crystal = []  # To store the number of atoms in each crystal

    atom_offset = 0  # Tracks the cumulative number of atoms processed

    for idx, sample in enumerate(samples):
        skip_sample = False

        # --- Atom Features ---
        atom_fea = sample['atom_fea']
        if not torch.is_tensor(atom_fea):
            atom_fea = torch.tensor(atom_fea, dtype=torch.float32)
        if not check_tensor_validity(atom_fea, "atom_fea", idx):
            skip_sample = True

        # --- Neighbor Features ---
        nbr_fea = sample['edge_fea']
        if not torch.is_tensor(nbr_fea):
            nbr_fea = torch.tensor(nbr_fea, dtype=torch.float32)
        if not check_tensor_validity(nbr_fea, "nbr_fea", idx):
            skip_sample = True

        # --- Neighbor Indices ---
        nbr_fea_idx = sample['nbr_fea_idx']
        if not torch.is_tensor(nbr_fea_idx):
            nbr_fea_idx = torch.tensor(nbr_fea_idx, dtype=torch.long)
        if not check_tensor_validity(nbr_fea_idx, "nbr_fea_idx", idx):
            skip_sample = True

        # --- Optional: Additional Features ---
        if include_additional_features and 'additional_features' in sample:
            add_feat = sample['additional_features']
            if not torch.is_tensor(add_feat):
                add_feat = torch.tensor(add_feat, dtype=torch.float32)
            if not check_tensor_validity(add_feat, "additional_features", idx):
                skip_sample = True
            else:
                additional_features_list.append(add_feat)

        # --- Targets ---
        if 'target' in sample:
            target_feat = sample['target']
            if not torch.is_tensor(target_feat):
                target_feat = torch.tensor(target_feat, dtype=torch.float32)
            if target_feat.ndim == 1:
                target_feat = target_feat.unsqueeze(0)
            if not check_tensor_validity(target_feat, "target", idx):
                skip_sample = True
            else:
                target_list.append(target_feat)

        # --- CIF ID ---
        if 'cif_id' in sample:
            cif = sample['cif_id']
            if isinstance(cif, bytes):
                cif = cif.decode('utf-8')
            cif_ids.append(cif)

        if skip_sample:
            continue

        # --- Track Crystal Atom Indices ---
        num_atoms = atom_fea.size(0)
        crystal_atom_idx.append(torch.arange(atom_offset, atom_offset + num_atoms, dtype=torch.long))
        atom_offset += num_atoms

        num_atoms_per_crystal.append(num_atoms)

        atom_fea_list.append(atom_fea)
        nbr_fea_list.append(nbr_fea)
        nbr_fea_idx_list.append(nbr_fea_idx)

    if len(atom_fea_list) == 0:
        raise ValueError("An entire Batch was invalid. Check your data for NaN/Inf values.")

    # --- Merge using the provided utility function ---
    merged_atom_fea, merged_nbr_fea, merged_nbr_fea_idx = merge_graphs(
        atom_fea_list, nbr_fea_list, nbr_fea_idx_list
    )

    batch_dict = {
        "atom_fea": merged_atom_fea,
        "nbr_fea": merged_nbr_fea,
        "nbr_fea_idx": merged_nbr_fea_idx,
    }

    if len(additional_features_list) > 0:
        batch_dict["additional_features"] = torch.stack(additional_features_list)

    if len(target_list) > 0:
        batch_dict["target"] = torch.cat(target_list, dim=0)

    if len(cif_ids) > 0:
        batch_dict["cif_id"] = cif_ids

    batch_dict["crystal_atom_idx"] = torch.cat(crystal_atom_idx)

    batch_dict["num_atoms_per_crystal"] = num_atoms_per_crystal

    return batch_dict



def make_graph_collate_fn(dir_path, zipped, transform, include_additional_features=True):
    """
    Creates a collate function that checks if the batch items are keys.
    If so, it loads all samples for the batch in one go.
    """
    def collate_fn(batch):
        # If lazy mode is active, batch elements are keys (strings)
        if batch and isinstance(batch[0], str):
            samples = load_batch_samples(batch, dir_path, zipped, transform)
        else:
            samples = batch
        return collate_graphs(samples, include_additional_features=include_additional_features)
    return collate_fn


    
def create_data_loaders(dataset, 
                        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                        batch_size=256, shuffle=True, num_workers=0,
                        train_size=None, val_size=None, test_size=None,
                        predict_set=False, sub_data=None,
                        include_additional_features=True):
    """
    Splits the dataset into train, validation, and test subsets.

    If any of train_size, val_size, or test_size is provided, those values take priority.
    Missing values are computed using the given ratios.
    Leftover samples (i.e. if the sum of the provided sizes is less than the total)
    are ignored. An error is raised only if any specified size (or their sum) exceeds
    the total number of samples.
    """
    dataset_total = len(dataset.graph_keys)
    indices = list(range(dataset_total))
    if shuffle:
        np.random.shuffle(indices)
    if sub_data is not None:
        if sub_data > dataset_total:
            raise ValueError(
                f"sub_data ({sub_data}) cannot be greater than total number of samples ({dataset_total})."
            )
        indices = indices[:sub_data]
    total = len(indices)

    # When predicting, only a single loader is created.
    if predict_set:
        subset = Subset(dataset, indices) if sub_data is not None else dataset
        collate_fn = (make_graph_collate_fn(dataset.dir_path, dataset.zipped, 
                                             dataset.transform, include_additional_features)
                      if dataset.lazy else lambda samples: collate_graphs(
                          samples, include_additional_features=include_additional_features
                      ))
        predict_loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, collate_fn=collate_fn)
        return predict_loader

    # If any explicit size is provided, use it; otherwise, fill in with default ratios.
    if (train_size is not None) or (val_size is not None) or (test_size is not None):
        train_size_set = train_size is not None
        val_size_set = val_size is not None
        test_size_set = test_size is not None
        if train_size is None:
            train_size = int(total * train_ratio)
        if val_size is None:
            val_size = int(total * val_ratio)
        if test_size is None:
            test_size = int(total * test_ratio)
        
        # Check that no individual size exceeds the total.
        if train_size > total:
            raise ValueError(f"train_size ({train_size}) cannot be greater than total number of samples ({total}).")
        if val_size > total:
            raise ValueError(f"val_size ({val_size}) cannot be greater than total number of samples ({total}).")
        if test_size > total:
            raise ValueError(f"test_size ({test_size}) cannot be greater than total number of samples ({total}).")
        
        # Check that the sum of sizes does not exceed the total.
        if train_size + val_size + test_size > total:
            if train_size_set and val_size_set and test_size_set:
                raise ValueError(
                    f"Sum of specified sizes ({train_size+val_size+test_size}) exceeds total number of samples ({total})."
                )
            overflow = train_size + val_size + test_size - total
            if not test_size_set and overflow > 0:
                shrink = min(test_size, overflow)
                test_size -= shrink
                overflow -= shrink
            if not val_size_set and overflow > 0:
                shrink = min(val_size, overflow)
                val_size -= shrink
                overflow -= shrink
            if not train_size_set and overflow > 0:
                shrink = min(train_size, overflow)
                train_size -= shrink
                overflow -= shrink
            if overflow > 0:
                raise ValueError(
                    f"Sum of specified sizes ({train_size+val_size+test_size}) exceeds total number of samples ({total})."
                )
    else:
        # Use ratio-based splitting that covers the entire dataset.
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        test_size = total - train_size - val_size  # All remaining samples are used for testing.

    # Create indices for each subset.
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:train_size + val_size + test_size]

    # Create the subsets.
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    # Use lazy mode collate function if needed.
    collate_fn = (make_graph_collate_fn(dataset.dir_path, dataset.zipped, 
                                          dataset.transform, include_additional_features)
                  if dataset.lazy else lambda samples: collate_graphs(
                      samples, include_additional_features=include_additional_features
                  ))
    
    # Create DataLoaders.
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader





