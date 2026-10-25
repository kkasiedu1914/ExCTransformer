# Visualization Workflows

ExCTransformer can render saved explanation payloads without re-running GraphSVX. This repository includes a compact paper-used visualization payload for `hMOF-20208`:

```text
assets/tutorial_data/visualization/hmof_20208/hMOF-20208_pred0_shap.pt
assets/tutorial_data/visualization/hmof_20208/hMOF-20208.cif
```

Attention comparison views require a compatible checkpoint when the saved payload does not already include attention weights:

```bash
exct download fetch exctransformer-base --destination ./release-assets
```

Render a 2D attention/SHAP heatmap:

```bash
exct explain visualize assets/tutorial_data/visualization/hmof_20208/hMOF-20208_pred0_shap.pt \
  --model-path ./release-assets/1.0.2/models/exctransformer-base/exctransformer.ckpt \
  --save-dir docs/assets/visualization/generated \
  --filename-prefix hmof_20208_pred0 \
  --vis-modes attn-shap-2d \
  --dpi 180
```

Output:

![hMOF-20208 attention and SHAP heatmap visualization](assets/visualization/hmof_20208_attn_shap_heatmaps.png)

Render a 3D attention/SHAP comparison:

```bash
exct download fetch synthesized-structure-explanations --destination ./release-assets

exct explain visualize \
  './release-assets/1.0.2/explanations/synthesized-structures/extracted/explanations/by_structure/2020[Cu][pts]3[ASR]1/ch4/2020[Cu][pts]3[ASR]1_pred2_shap.pt' \
  --model-path ./release-assets/1.0.2/models/exctransformer-base/exctransformer.ckpt \
  --save-dir docs/assets/visualization/generated \
  --filename-prefix 2020_cu_pts_ch4_pred2 \
  --vis-modes attn-shap-3d \
  --dpi 180
```

Output:

![2020 Cu pts CH4 3D attention and SHAP visualization](assets/visualization/2020_cu_pts_ch4_attn_shap_3d.png)

The displayed images are paper-used PNG outputs copied from the manuscript figure directory. They are documentation figures, not training assets.

## New Structures

For a new structure, the normal workflow is:

1. Run `exct explain single` or `exct explain batch` to create a saved SHAP payload.
2. Run `exct explain visualize` on the saved payload.
3. Add `--model-path` when attention comparison views are needed and the payload does not already include attention weights.
