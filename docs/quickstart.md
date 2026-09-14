# CPU quickstart

Install from the repository using the Python 3.11 instructions in the
[README](../README.md). From the repository root, run:

```bash
python examples/cpu_quickstart.py --output example-output
```

The tutorial creates twelve two-atom cubic structures with artificial scalar
targets. It converts CIFs into HDF5 crystal graphs, trains a small ExCT with an
8/2/2 train/validation/test split for one epoch, and runs checkpoint-based
prediction. Everything runs on CPU without network access after installation.
It tests software operation; these synthetic targets have no adsorption meaning.

To also compute WLS_Ridge GraphSVX attributions and save a PNG/PDF plot:

```bash
python examples/cpu_quickstart.py --output example-with-explanation --explain
```

The explanation and plots appear under `explanations/` and `figures/` inside the
output directory. Their values describe this small tutorial model only.

The output directory contains:

- `cifs/` and `targets.csv`: generated inputs with a header and one row per crystal.
- `graphs12/`: one HDF5 graph per crystal.
- `logs/`: TensorBoard events, training configuration, and `best.ckpt`/`last.ckpt`.
- Prediction/evaluation CSVs written by the model workflows.

Use a new output directory for a new run. The reduced model dimensions are for
quick testing; the defaults use the larger research architecture.

## Your own dataset

Provide a CIF directory and a CSV with a header. Column zero holds the CIF stem
(for example, `sample-1` for `sample-1.cif`); subsequent columns hold targets.
Pass zero-based `target_cols` and `additional_cols` explicitly when the CSV also
contains scalar descriptors. Keep target order consistent across training,
prediction, and explanations. Target indices do not automatically identify gases.

```python
from exctransformer.datamodule.workflows import build_h5_dataset_workflow

build_h5_dataset_workflow(
    data_dir="/absolute/path/to/inputs",
    cif_id_prop="targets",  # targets.csv
    cif_path_z="cifs",      # relative to data_dir, or an absolute path
    output_h5="/absolute/path/to/graphs",
    target_cols=[1],
    dataset_size=1000,
)
```

The legacy graph builder appends the successful graph count to output directory
names that do not already end in a digit. Use the path printed by the builder for
training. Check skipped-structure warnings and the number of valid graphs before
starting a research run. Atom features are bundled in the installed package.

```bash
exct train /absolute/path/to/graphs_1000 \
  --set num_targets=1 --set accelerator=cpu --set precision=32-true \
  --set max_epochs=30
exct predict /absolute/path/to/graphs_1000 /absolute/path/to/best.ckpt \
  --set accelerator=cpu --set precision=32-true
```

For research reproduction, use the matching [asset manifests](assets.md), dataset
partitions, target scaling, configuration, and checkpoint. A one-epoch tutorial
does not reproduce manuscript metrics. Load checkpoints and saved `.pt` explanation
files only from trusted sources: the current research formats use Python pickle.

## Explanations

Use `exct explain single --help` and the examples in the README to supply a graph,
matching checkpoint, CIF, and target index. Explanations neutralize contextualized
atom tokens after CSAN; they do not remove atoms from a crystal or recompute its
local graph for each coalition. Inspect surrogate fidelity before interpreting
attributions. See [visualization](visualization.md) and [faithfulness](faithfulness.md).
