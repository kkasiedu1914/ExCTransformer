## Asset Manifests

This manifest files define the official downloadable ExCTransformer assets for each package release.

`v1.0.0.json` points to the Figshare-hosted v1.0.0 asset record:

```text
https://figshare.com/articles/dataset/_/31941117
```

The manifest records:

- asset name used by `exct download`
- asset category: `data`, `model`, `metadata`, or `explanation`
- direct Figshare file URL
- file name and byte size
- checksum
- extraction behavior
- destination folder under the local asset cache
- citation or usage notes where needed

Figshare currently exposes MD5 checksums for these files, so the v1.0.0 manifest records `md5` values. The downloader also supports `sha256` for future mirrors or locally produced asset packs.

By default, files are cached under:

```bash
~/.cache/exctransformer/assets/
```

Set `EXCTRANSFORMER_ASSETS_DIR` to use another location.

Private pre-publication checks can use `FIGSHARE_TOKEN` or `FIGSHARE_TOKEN_FILE`. Public users should not need either variable once the Figshare article is made public.
