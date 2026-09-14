# Release Assets

**Research asset access:** the Figshare record was not publicly accessible when checked on 2026-09-14. The commands below require access until its owner publishes the record. Making this GitHub repository public does not publish Figshare assets. The CPU quickstart and use with your own data do not depend on Figshare.

ExCTransformer source distributions do not include full datasets, checkpoints, generated SHAP outputs, logs, or media files. Those assets are distributed through the official Figshare record and resolved by the package manifest.

Figshare record:

```text
https://figshare.com/articles/dataset/_/31941117
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

The v1.0.0 manifest is stored at `assets/manifests/v1.0.0.json` and packaged at `src/exctransformer/manifests/v1.0.0.json`. Each entry records the Figshare URL, file size, checksum, and extraction behavior used by the downloader.

While the Figshare article is private (**Will be changed**), set `FIGSHARE_TOKEN` or `FIGSHARE_TOKEN_FILE` before running `exct download`. The downloader sends that token only to Figshare and follows the signed storage redirect without forwarding the token. After the article is public, users should not need a token for the same manifest URLs.
