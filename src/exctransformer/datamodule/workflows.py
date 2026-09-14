"""Higher-level dataset build workflows."""

from __future__ import annotations

from typing import Any

from exctransformer.datamodule import dataset_build


def build_h5_dataset_workflow(
    *,
    data_dir: str,
    cif_id_prop: str,
    cif_path_z: str,
    dataset_size: int = 1000,
    random_seed: int = 123,
    radius: float = 8.0,
    dmin: float = 0.0,
    dmax: float = 8.0,
    step: float = 0.2,
    output_h5: str,
    zip: bool = False,
    resume_load: bool = False,
    additional_cols: list[int] | None = None,
    target_cols: list[int] | None = None,
    predict_set: bool = False,
) -> Any:
    """Build an HDF5 dataset from CIF inputs using the existing graph builder."""

    dataset_processor = dataset_build.CrystalGraphDataset(
        data_dir=data_dir,
        cif_id_prop=cif_id_prop,
        cif_path_z=cif_path_z,
        dataset_size=dataset_size,
        random_seed=random_seed,
        radius=radius,
        dmin=dmin,
        dmax=dmax,
        step=step,
        zip=zip,
        predict_set=predict_set,
    )
    return dataset_processor.build_h5_dataset(
        output_folder=output_h5,
        additional_cols=additional_cols,
        target_cols=target_cols,
        resume_load=resume_load,
    )
