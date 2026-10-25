# Release Assets

**Research asset access:** the Figshare dataset is public as of 2026-10-25. The download commands below work without an account or API token. The CPU quickstart and use with your own data also remain self-contained.

ExCTransformer source distributions do not include full datasets, checkpoints, generated SHAP outputs, logs, or media files. Those assets are distributed through the official Figshare record and resolved by the package manifest.

Figshare record:

```text
https://doi.org/10.6084/m9.figshare.31941117.v1
```

Local cache:

```bash
~/.cache/exctransformer/assets/
```

Override the cache:

```bash
export EXCTRANSFORMER_ASSETS_DIR=/path/to/exctransformer-assets
```

List assets:

```bash
exct download list
```

Fetch a checkpoint:

```bash
exct download fetch exctransformer-base --destination ./release-assets
```

Fetch all models:

```bash
exct download all --kind model --destination ./release-assets
```

Fetch all assets:

```bash
exct download all --destination ./release-assets
```

The v1.0.2 manifest is stored at `assets/manifests/v1.0.2.json` and packaged at `src/exctransformer/manifests/v1.0.2.json`. It references the same published research files as v1.0.0; the v1.0.0 manifests remain available for reproducibility. Each entry records the Figshare URL, file size, checksum, and extraction behavior used by the downloader.

Public downloads require no account, API key, or credentials. Run either `exct download` or the `exct-download` shortcut directly; both use the same versioned manifest, follow the public file redirects automatically, and verify checksums.
