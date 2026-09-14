# Faithfulness Analysis

ExCTransformer includes a faithfulness workflow for saved explanation payloads. The comparison is built around progressive occlusion: atoms are neutralized or masked in ranked order, and the resulting prediction drop is tracked as the perturbed set grows. The same perturbation schedule is then repeated for attention-ranked atoms and for random selections, which makes it possible to compare attribution quality against a baseline under the same operator.

This is a model-behavior check rather than a structural simulation. The perturbations are applied in embedding space or through node masking, depending on the selected mode.

## What The Workflow Uses

The faithfulness path is exposed in two places:

- Python API: `exctransformer.explain.compare_faithfulness(...)`
- CLI: `exct explain faithfulness ...`

The core implementation lives in:

- [faithfulness.py](../src/exctransformer/SHAP_visualization/faithfulness.py)
- [workflows.py](../src/exctransformer/SHAP_visualization/workflows.py)

The main curve computation is `progressive_occlusion_curve(...)`, which is also the part of the codebase that handles the top-k progression used in the earlier development plots.

## Inputs

The faithfulness workflow consumes:

- an `.h5` graph sample
- a saved SHAP payload such as `*_pred0_shap.pt`
- a trained checkpoint
- the graph dataset directory
- a CIF ZIP archive path used to resolve the structure when needed

The current CLI signature is:

```bash
exct explain faithfulness H5_PATH SHAP_OUTPUT_PATH MODEL_PATH DATASET_PATH ZIP_PATH
```

## Python Surface

The Python-facing entry point is:

```python
from exctransformer.explain import compare_faithfulness

result = compare_faithfulness(
    h5_path="/path/to/sample.h5",
    shap_output_path="/path/to/sample_pred0_shap.pt",
    model_path="/path/to/best.ckpt",
    dataset_path="/path/to/graph_data",
    zip_path="/path/to/case_bundle.zip",
    pred_idx=0,
    save_plot_path="faithfulness_plot.png",
)
```

The returned object contains the SHAP-ranked curve, the attention-ranked curve, random baselines, and the metadata needed to reconstruct the comparison.

## CLI Surface

The same workflow is available from the CLI:

```bash
exct explain faithfulness \
  /path/to/sample.h5 \
  /path/to/sample_pred0_shap.pt \
  /path/to/best.ckpt \
  /path/to/graph_data \
  /path/to/case_bundle.zip \
  --pred-idx 0 \
  --top-k-max 20 \
  --k-step 1 \
  --num-random 10 \
  --num-random-attn 10 \
  --neutralize-mode mask \
  --show-random-band \
  --save-plot-path faithfulness_plot.png
```

## Output Shape

The comparison produces a progressive-occlusion plot and a structured result payload. The plot typically contains:

- a SHAP-ranked drop curve
- an attention-ranked drop curve
- a random baseline
- an optional random uncertainty band

The default saved figure path is `faithfulness_plot.png` unless another path is provided.

## Interpretation

The comparison is most informative when the SHAP-ranked curve drops earlier or more sharply than the random baseline under the same perturbation budget. The attention-ranked curve provides a second reference point, showing whether the model’s internal attention ordering is concentrating on the same decisive regions as the explanation ranking.

The curve therefore answers a narrow but useful question: whether the atoms receiving high attribution are also the atoms whose removal or neutralization changes the model output most strongly.

## Main Options

The most relevant exposed options are:

- `--pred-idx`
  Selects the target index for multi-target runs.
- `--top-k-max`
  Sets the largest perturbation set size included in the curve.
- `--k-step`
  Controls the spacing between successive k-values.
- `--num-random`
  Sets how many random SHAP-basis baselines are averaged per k.
- `--num-random-attn`
  Sets how many random baselines are used on the attention side.
- `--neutralize-mode`
  Chooses how excluded nodes are perturbed. The current default is `mask`.
- `--zero-adj`
  Zeroes adjacency entries for excluded nodes when the representation supports it.
- `--exclude-void-fraction`
  Removes the void-fraction token from ranking when additional tokens are present.
- `--void-fraction-index`
  Overrides which token index is treated as void fraction.
- `--exclude-add-tokens`
  Removes selected additional-token indices from the candidate set.
- `--show-random-band`
  Adds the random uncertainty band to the rendered plot.

## Relation To Visualization

Faithfulness analysis is separate from the saved-payload rendering views described in [visualization.md](visualization.md). The visualization pages render attribution patterns on the structure; the faithfulness workflow tests whether those ranked patterns remain behaviorally meaningful under progressive perturbation.
