"""Sacred-compatible entry point for building ExCTransformer HDF5 graph datasets."""

from exctransformer.config import ex
from exctransformer.datamodule.workflows import build_h5_dataset_workflow

@ex.automain
def main(_config):
    """
    Main function to build the dataset from CIF files and save to HDF5.
    """
    config = _config

    data_dir = config["data_dir"]               
    cif_id_prop = config["cif_id_prop"]
    cif_path_z = config["cif_path_z"]        
    dataset_size = config["dataset_size"]
    random_seed = config["random_seed"]
    radius = config["radius"]
    dmin = config["dmin"]
    dmax = config["dmax"]                    
    step = config["step"]
    output_h5 = config["output_h5"]
    zip = config["zip"]
    resume_load = config["resume_load"]
    additional_cols = config.get("additional_cols", [1])  
    target_cols = config.get("target_cols", [6, 7])
    predict_set= config["predict_set"]

    return build_h5_dataset_workflow(
        data_dir=data_dir,
        cif_id_prop=cif_id_prop,
        cif_path_z=cif_path_z,
        dataset_size=dataset_size,
        random_seed=random_seed,
        radius=radius,
        dmin=dmin,
        dmax=dmax,
        step=step,
        output_h5=output_h5,
        zip=zip,
        resume_load=resume_load,
        additional_cols=additional_cols,
        target_cols=target_cols,
        predict_set=predict_set,
    )
