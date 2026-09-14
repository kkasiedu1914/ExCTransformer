import os
import zipfile  # for handling zipped CIF files
import random
import json
import warnings
import numpy as np
import pyarrow.csv as pv
import torch
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from tempfile import NamedTemporaryFile
import h5py
from exctransformer.datamodule.dataset_build_utils import (
    open_or_create_h5_file,
    create_compressed_dataset,
    update_progress,
    rename_output_folder,
    count_h5_dataset,
    )

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


DEFAULT_ATOM_INIT_FILE = Path(__file__).resolve().parent / "data" / "atom_init.json"


warnings.filterwarnings("ignore", module="pymatgen")
warnings.simplefilter(action='ignore', category=FutureWarning)


# TODO: Make adding additional features optional in the future

class GaussianDistance:
    """
    (https://github.com/txie-93/cgcnn)
    Expands interatomic distances using a set of Gaussian basis functions.

    This transformation converts scalar distances into a higher-dimensional representation,
    where each dimension corresponds to a Gaussian filter centered at different values.
    """
    def __init__(self, dmin: float, dmax: float, step: float, var: float = None):
        """
        (https://github.com/txie-93/cgcnn)
        Args:
            dmin (float): Minimum interatomic distance.
            dmax (float): Maximum interatomic distance.
            step (float): Step size for determining Gaussian centers.
            var (float, optional): Variance for the Gaussian filters (defaults to step if None).
        """
        assert dmin < dmax, "dmin must be less than dmax."
        assert (dmax - dmin) > step, "The range (dmax - dmin) must be greater than step."
        self.filter = np.arange(dmin, dmax + step, step)
        self.var = var if var is not None else step

    def expand(self, distances: np.ndarray) -> np.ndarray:
        """
        (https://github.com/txie-93/cgcnn)
        Applies Gaussian filters to the input distances.

        Args:
            distances (np.ndarray): Array of distances (can be multi-dimensional).

        Returns:
            np.ndarray: Expanded representation with an extra dimension for the Gaussian filters.
        """
        return np.exp(-((distances[..., np.newaxis] - self.filter) ** 2) / (self.var ** 2))


class AtomInitializer:
    """
    (https://github.com/txie-93/cgcnn)
    Base class for initializing atom feature vectors.
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types, f"Atom type {atom_type} not found in initializer."
        return self._embedding[atom_type]




class AtomCustomJSONInitializer(AtomInitializer):
    """
    (https://github.com/txie-93/cgcnn)
    Initializes atom feature vectors from a JSON configuration file.

    The JSON file should map atomic numbers (as strings) to a list representing the atom's feature vector.
    """
    def __init__(self, elem_embedding_file: str):
        with open(elem_embedding_file, 'r') as f:
            elem_embedding = json.load(f)
        # Convert keys to integers.
        elem_embedding = {int(key): value for key, value in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)



class CrystalGraphDataset:
    def __init__(
        self,
        data_dir: str,
        cif_id_prop: str,
        cif_path_z: str,
        dataset_size: int,
        random_seed: int = 123,
        radius: float = 10.0,
        dmin: float = 0.0,
        dmax: float = 6.0,
        step: float = 0.2,
        zip: bool = True,
        predict_set: bool = False,
    ):
        """
        (https://github.com/txie-93/cgcnn)
        Dataset class for processing crystal graph data.

        Args:
            data_dir (str): Directory containing all necessary files for building the crystal graph dataset.
            cif_id_prop (str): File name (without extension) for the CSV file containing CIF IDs and properties.
            cif_path_z (str): Directory OR path to a zip file containing all CIF files.
            dataset_size (int): Number of CIF files to process.
            random_seed (int): Seed for shuffling the CIF ID list.
            dmin (float): Minimum distance for Gaussian expansion.
            dmax (float): Maximum distance (also used as cutoff radius).
            step (float): Step size for the Gaussian filter.
            zip (bool): If True, load CIF files from a zipped archive.
            predict_set (bool): If True, do not include target features in the .h5 file.
        """
        self.data_dir = data_dir
        self.cif_id_prop = cif_id_prop
        self.cif_path_z = cif_path_z  # If zip==True, this should be the path to the zip archive.
        self.dataset_size = dataset_size
        self.random_seed = random_seed
        self.radius = radius
        self.dmin = dmin
        self.dmax = dmax  # Also used as the neighbor cutoff radius.
        self.step = step
        self.zip = zip
        self.predict_set = predict_set
        self.cif_source = (
            self.cif_path_z
            if os.path.isabs(self.cif_path_z)
            else os.path.join(self.data_dir, self.cif_path_z)
        )

        # Check if data directory exists.
        assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist."

        # Load the CIF ID and property CSV.
        cif_id_prop_file = os.path.join(self.data_dir, self.cif_id_prop + '.csv')
        assert os.path.exists(cif_id_prop_file), f"CIF ID property file {cif_id_prop_file} does not exist."
        self.id_prop_data = pv.read_csv(cif_id_prop_file).to_pandas().values.tolist()

        random.seed(random_seed)
        random.shuffle(self.id_prop_data)

        # Prefer a user-provided atom_init.json in the data directory, but fall back
        # to the packaged default shipped with ExCTransformer.
        atom_init_file = os.path.join(self.data_dir, 'atom_init.json')
        if not os.path.exists(atom_init_file):
            atom_init_file = str(DEFAULT_ATOM_INIT_FILE)
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)

        # Initialize Gaussian distance filter (dmax acts as cutoff).
        self.gdf = GaussianDistance(dmin=dmin, dmax=dmax, step=step)


    def build_crystal_graph(self, cif_path: str, radius: float = None, max_num_nbr: int = 12, zip_handle=None):
        """
        Constructs a crystal graph from a given CIF file.

        Args:
            cif_path (str): If self.zip is True, this is the filename (or CIF id) inside the zip.
                            Otherwise, this is the relative path (from data_dir) to the CIF file.
            radius (float): Radius within which neighbors are considered (defaults to self.dmax if None).
            max_num_nbr (int): Maximum number of neighbors to consider for each atom.
            zip_handle: Optional zipfile.ZipFile handle (used if self.zip is True).

        Returns:
            tuple: (node_feat, edge_index) where:
                   - node_feat is a tensor combining node features and processed neighbor information.
                   - edge_index is a tensor in PyTorch Geometric format.
        """
        if radius is None:
            radius = self.dmax


        # Load the CIF file either from zip or from disk.
        if self.zip:
            # Use the provided zip_handle if available; otherwise open a new one.
            if zip_handle is None:
                with zipfile.ZipFile(self.cif_source, "r") as z:
                    cif_filename = cif_path if cif_path.endswith('.cif') else cif_path + '.cif'
                    cif_id = z.read(cif_filename).decode('utf-8')
            else:
                cif_filename = cif_path if cif_path.endswith('.cif') else cif_path + '.cif'
                cif_id = zip_handle.read(cif_filename).decode('utf-8')
            #parser = CifParser.from_string(cif_id)
            structure = self.parse_cif_string(cif_id)
        else:
            # Load from the file system.
            cif_filename = cif_path if cif_path.endswith('.cif') else cif_path + '.cif'
            cif_file = os.path.join(self.cif_source, cif_filename)
            parser = CifParser(cif_file)
            structure = parser.parse_structures(primitive=True)[0] 

        # Build atom feature matrix.
        atom_fea = np.vstack([self.ari.get_atom_fea(structure[i].specie.number) for i in range(len(structure))])
        atom_fea = torch.Tensor(atom_fea)

        # Retrieve neighbors within the specified radius.
        all_nbrs = structure.get_all_neighbors(radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, edge_fea = [], []

        for nbr in all_nbrs:
            if len(nbr) < max_num_nbr:
                warnings.warn(f"Failed to build crystal: Atom does not have enough neighbors. Consider increasing the radius.")
                return None, None, None  # Skip this crystal
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:max_num_nbr])))
                edge_fea.append(list(map(lambda x: x[1], nbr[:max_num_nbr])))

        nbr_fea_idx, edge_fea = np.array(nbr_fea_idx), np.array(edge_fea)

        # Apply Gaussian distance filter to the neighbor features
        gdf = GaussianDistance(dmin=0.0, dmax=radius, step=0.2)
        edge_fea = gdf.expand(edge_fea)

        # Convert to torch tensors
        atom_fea = torch.Tensor(atom_fea)
        edge_fea = torch.Tensor(edge_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)

        return atom_fea, edge_fea, nbr_fea_idx



    def parse_cif_string(self, cif_string):
      with NamedTemporaryFile('w') as tmp:
          tmp.write(cif_string)
          tmp.flush()
          parser = CifParser(tmp.name)
          return parser.get_structures()[0]
      

    def build_h5_dataset(self, output_folder: str,
                        resume_load: bool = False, 
                        additional_cols: list = None,
                        target_cols: list = None):
        """
        Processes CIF files, builds their corresponding crystal graphs,
        and saves them along with associated additional features and target properties into individual HDF5 files.
        """
        self.additional_cols = additional_cols
        self.target_cols = target_cols
        # Modify output folder name if predict_set is True.
        if self.predict_set:
            output_folder = output_folder + '_predict'
            logger.info("Preparing prediction set")

        # Ensure output folder exists.
        os.makedirs(output_folder, exist_ok=True)

        # If initializing from scratch, ensure no leftover checkpoint exists.
        if not resume_load:
            logger.info("-" * 40)
            logger.info("Initializing graph loading from scratch.")
            logger.info("-" * 40)

        # Checkpointing logic.
        ckpt_file = os.path.join(output_folder, "checkpoint.ckpt")
        if resume_load:
            if os.path.exists(ckpt_file):
                with open(ckpt_file, 'r') as f:
                    graph_counter = int(f.read().strip())
                logger.info("-" * 44)
                logger.info(f"Resuming graph loading from graph index: {graph_counter}")
                logger.info("-" * 44)
            else:
                graph_counter = 0
                warnings.warn("No resume file found. Loading data from scratch!")
        else:
            graph_counter = 0

        num_graphs = min(self.dataset_size, len(self.id_prop_data))

        # Determine default target_cols if not provided.
        if self.id_prop_data:
            n_cols = len(self.id_prop_data[0])
            if target_cols is None:
                if additional_cols is None:
                    target_cols = list(range(1, n_cols))
                else:
                    target_cols = [i for i in range(1, n_cols) if i not in additional_cols]

        if self.zip:
            with zipfile.ZipFile(self.cif_source, "r") as z:
                for idx in range(graph_counter, num_graphs):
                    row = self.id_prop_data[idx]
                    cif_id = row[0]  # Assumes first column is the CIF ID.
                    cif_filename = cif_id  # Using CIF ID as filename base.
                    try:
                        atom_fea, edge_fea, nbr_fea_idx = self.build_crystal_graph(cif_filename, zip_handle=z, radius=self.radius)
                    except Exception as e:
                        warnings.warn(f"Failed to build graph for CIF {cif_id}: {e}")
                        continue

                    additional_features = [row[col] for col in additional_cols] if additional_cols is not None else []
                    target = [row[col] for col in target_cols] if target_cols is not None else []
                    try:
                        additional_features = list(map(float, additional_features))
                    except Exception:
                        pass
                    try:
                        target = list(map(float, target))
                    except Exception:
                        pass

                    try:
                        output_file = os.path.join(output_folder, f"{cif_id}.h5")
                        with open_or_create_h5_file(output_file) as h5file:
                            grp = h5file.create_group('graph')
                            create_compressed_dataset(grp, 'atom_fea', atom_fea.numpy())
                            create_compressed_dataset(grp, 'edge_fea', edge_fea.numpy())
                            create_compressed_dataset(grp, 'nbr_fea_idx', nbr_fea_idx.numpy())
                            

                            grp.create_dataset('cif_id', data=cif_filename, dtype=h5py.string_dtype(encoding='utf-8'))
                            if additional_features:
                                create_compressed_dataset(grp, 'additional_features', np.array(additional_features, dtype=np.float32))
                            if not self.predict_set and target:
                                create_compressed_dataset(grp, 'target', np.array(target, dtype=np.float32))
                        graph_counter = idx + 1
                        with open(ckpt_file, 'w') as f:
                            f.write(str(graph_counter))
                    except Exception as e:
                        warnings.warn(f"Error saving graph for CIF {cif_id}: {e}")
                        graph_counter = idx + 1
                        with open(ckpt_file, 'w') as f:
                            f.write(str(graph_counter))
                        continue
                    update_progress(idx+1, self.dataset_size, use_tqdm=True)
        else:
            # Similar modifications for the non-zipped branch.
            for idx in range(graph_counter, num_graphs):
                row = self.id_prop_data[idx]
                cif_id = row[0]
                cif_filename = cif_id
                try:
                     atom_fea, edge_fea, nbr_fea_idx  = self.build_crystal_graph(cif_filename, radius=self.radius)  # zip_handle not used here
                except Exception as e:
                    warnings.warn(f"Failed to build graph for CIF {cif_id}: {e}")
                    continue

                additional_features = [row[col] for col in additional_cols] if additional_cols is not None else []
                target = [row[col] for col in target_cols] if target_cols is not None else []
                try:
                    additional_features = list(map(float, additional_features))
                except Exception:
                    pass
                try:
                    target = list(map(float, target))
                except Exception:
                    pass

                try:
                    output_file = os.path.join(output_folder, f"{cif_id}.h5")
                    with open_or_create_h5_file(output_file) as h5file:
                        grp = h5file.create_group('graph')
                        create_compressed_dataset(grp, 'atom_fea', atom_fea.numpy())
                        create_compressed_dataset(grp, 'edge_fea', edge_fea.numpy())
                        create_compressed_dataset(grp, 'nbr_fea_idx', nbr_fea_idx.numpy())
                        grp.create_dataset('cif_id', data=cif_filename, dtype=h5py.string_dtype(encoding='utf-8'))
                        if additional_features:
                            create_compressed_dataset(grp, 'additional_features', np.array(additional_features, dtype=np.float32))
                        if not self.predict_set and target:
                            create_compressed_dataset(grp, 'target', np.array(target, dtype=np.float32))
                    graph_counter = idx + 1
                    with open(ckpt_file, 'w') as f:
                        f.write(str(graph_counter))
                except Exception as e:
                    warnings.warn(f"Error saving graph for CIF {cif_id}: {e}")
                    graph_counter = idx + 1
                    with open(ckpt_file, 'w') as f:
                        f.write(str(graph_counter))
                    continue
                update_progress(idx+1, self.dataset_size, use_tqdm=True)

        file_counts = count_h5_dataset(output_folder, self.dataset_size)

        logger.info(f"Checking integrity of all files created...")
        self.check_all_file_integrity(output_folder)

        if not output_folder[-1].isdigit():
            rename_output_folder(output_folder, file_counts)


    def check_all_file_integrity(self, output_folder: str):
        """
        Checks the integrity of all files in the output folder, deletes corrupted ones, and rebuilds them.
        """
        # Check if the output folder exists
        if not os.path.exists(output_folder):
            raise FileNotFoundError(f"Output folder {output_folder} does not exist.")

        # Get all HDF5 files in the output folder
        h5_files = [f for f in os.listdir(output_folder) if f.endswith('.h5')]
        logger.info(f"Found {len(h5_files)} HDF5 files in {output_folder}.")

        # Check each file for integrity
        for h5_file in h5_files:
            file_path = os.path.join(output_folder, h5_file)
            try:
                with h5py.File(file_path, 'r') as f:
                    # Check if the required datasets exist
                    if 'graph' not in f:
                        raise ValueError(f"Group 'graph' not found in {h5_file}.")
                    grp = f['graph']
                    required_datasets = ['atom_fea', 'edge_fea', 'nbr_fea_idx', 'cif_id']
                    if self.additional_cols:
                        required_datasets.append('additional_features')
                    if not self.predict_set:
                        required_datasets= required_datasets + ['target']
                    for dataset in required_datasets:
                        if dataset not in grp:
                            raise ValueError(f"Dataset '{dataset}' not found in {h5_file}.")
            except Exception as e:
                logger.warning(f"File {h5_file} is corrupted or incomplete: {e}")
                logger.info(f"Deleting corrupted file {h5_file}...")
                
                # Delete the corrupted file
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted corrupted file {h5_file}.")
                except Exception as delete_error:
                    logger.error(f"Failed to delete corrupted file {h5_file}: {delete_error}")
                    continue

                logger.info(f"Attempting to rebuild {h5_file}...")

                # Extract CIF ID from the HDF5 file name
                cif_id = os.path.splitext(h5_file)[0]

                try:
                    # Rebuild the HDF5 file
                    if self.zip:
                        with zipfile.ZipFile(self.cif_source, "r") as z:
                            atom_fea, edge_fea, nbr_fea_idx = self.build_crystal_graph(cif_id, zip_handle=z, radius=self.radius)
                    else:
                        atom_fea, edge_fea, nbr_fea_idx = self.build_crystal_graph(cif_id, radius=self.radius)

                    # Retrieve additional features and target properties
                    row = next((r for r in self.id_prop_data if r[0] == cif_id), None)
                    if row is None:
                        raise ValueError(f"Row with CIF ID {cif_id} not found in id_prop_data.")

                    additional_features = [row[col] for col in self.additional_cols] if self.additional_cols is not None else []
                    target = [row[col] for col in self.target_cols] if self.target_cols is not None else []

                    try:
                        additional_features = list(map(float, additional_features))
                    except Exception:
                        pass
                    try:
                        target = list(map(float, target))
                    except Exception:
                        pass

                    # Save the rebuilt HDF5 file
                    with open_or_create_h5_file(file_path) as h5file:
                        grp = h5file.create_group('graph')
                        create_compressed_dataset(grp, 'atom_fea', atom_fea.numpy())
                        create_compressed_dataset(grp, 'edge_fea', edge_fea.numpy())
                        create_compressed_dataset(grp, 'nbr_fea_idx', nbr_fea_idx.numpy())
                        grp.create_dataset('cif_id', data=cif_id, dtype=h5py.string_dtype(encoding='utf-8'))
                        if additional_features:
                            create_compressed_dataset(grp, 'additional_features', np.array(additional_features, dtype=np.float32))
                        if not self.predict_set and target:
                            create_compressed_dataset(grp, 'target', np.array(target, dtype=np.float32))
                    logger.info(f"Successfully rebuilt {h5_file}.")
                except Exception as rebuild_error:
                    logger.error(f"Failed to rebuild {h5_file}: {rebuild_error}")
                    continue

        logger.info("All files checked.")
        

    
