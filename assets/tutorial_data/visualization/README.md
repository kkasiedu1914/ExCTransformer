## Visualization Tutorial Data

This directory contains a compact, paper-used visualization payload for the `hMOF-20208` hydrogen case.

Files:

- `hmof_20208/hMOF-20208.cif`: CIF structure used for atom-level rendering.
- `hmof_20208/hMOF-20208_pred0_shap.pt`: saved SHAP payload for the H2 prediction target.

The payload is small enough to ship with the source repository and is intended for visualization-only tutorials and smoke checks. It is not a training dataset or benchmark dataset.

For the README visualization tutorial, first fetch a compatible checkpoint:

```bash
exct download fetch exctransformer-base --destination ./release-assets
```

Then render the 2D attention/SHAP heatmap from the repository root:

```bash
exct explain visualize assets/tutorial_data/visualization/hmof_20208/hMOF-20208_pred0_shap.pt \
  --model-path ./release-assets/1.0.0/models/exctransformer-base/exctransformer.ckpt \
  --save-dir docs/assets/visualization/generated \
  --filename-prefix hmof_20208_pred0 \
  --vis-modes attn-shap-2d \
  --dpi 180
```
