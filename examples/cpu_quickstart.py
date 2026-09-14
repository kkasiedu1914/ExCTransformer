"""Build toy crystals, train a small ExCT, and predict on CPU without downloads."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch
from pymatgen.core import Lattice, Structure

from exctransformer.datamodule.workflows import build_h5_dataset_workflow
from exctransformer.predict import predict
from exctransformer.train import train


def run(output: Path, with_explanation: bool = False) -> Path:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cifs = output / "cifs"
    cifs.mkdir()
    # Artificial targets demonstrate software behavior; they are not adsorption data.
    with (output / "targets.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cif_id", "toy_target"])
        for index in range(12):
            side = 4.5 + 0.05 * index
            structure = Structure(
                Lattice.cubic(side), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
            )
            structure.to(filename=str(cifs / f"toy-{index}.cif"))
            writer.writerow([f"toy-{index}", side - 4.5])

    graphs = output / "graphs12"
    build_h5_dataset_workflow(
        data_dir=str(output), cif_id_prop="targets", cif_path_z=str(cifs),
        output_h5=str(graphs), dataset_size=12,
    )
    if len(list(graphs.glob("*.h5"))) != 12:
        raise RuntimeError("Expected twelve graph files from the toy structures")

    torch.set_num_threads(2)
    previous_dir = Path.cwd()
    os.chdir(output)  # Keep prediction CSVs and logs with this run's other outputs.
    try:
        train(
            dataset_path=str(graphs), log_dir=str(output / "logs"),
            accelerator="cpu", device="auto", precision="32-true",
            num_targets=1, num_layers=2, hidden_dims=16, graph_attn_heads=2,
            out_dim=32, hid_dim=32, transformer_num_layers=1, num_heads=2,
            max_crystal_atoms=16, batch_size=4, per_gpu_batchsize=4,
            batch_size_grad=4, max_epochs=1, train_size=8, val_size=2,
            test_size=2, num_workers=0, log_every_n_steps=1,
        )
        checkpoint = next((output / "logs").rglob("best.ckpt"))
        predict(str(graphs), str(checkpoint), accelerator="cpu", precision="32-true")
        if with_explanation:
            from exctransformer.explain import explain, visualize_explanation

            result = explain(
                data_dir=str(graphs), model_path=str(checkpoint), pred_idx=0,
                single_id="toy-0", cif_dir=str(cifs),
                out_dir=str(output / "explanations"), args_g="WLS_Ridge",
                gpu=False, seed=0, s_override=1, graphsvx_info=False,
            )
            visualize_explanation(
                shap_path=result.shap_output_path, vis_modes=["shap-only"],
                save_dir=str(output / "figures"), show=False, dpi=100,
            )
    finally:
        os.chdir(previous_dir)
    print(f"Tutorial complete. Checkpoint: {checkpoint}")
    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("example-output"))
    parser.add_argument("--explain", action="store_true", help="Also compute and plot GraphSVX attributions")
    args = parser.parse_args()
    run(args.output, with_explanation=args.explain)
