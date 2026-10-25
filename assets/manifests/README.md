## Asset Manifests

These manifest files define the official downloadable ExCTransformer assets for each package release.

The versioned manifests point to the published Figshare asset record:

```text
https://doi.org/10.6084/m9.figshare.31941117.v1
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

Figshare currently exposes MD5 checksums for these files, so the release manifests record `md5` values. The downloader also supports `sha256` for future mirrors or locally produced asset packs.

By default, files are cached under:

```bash
~/.cache/exctransformer/assets/
```

Set `EXCTRANSFORMER_ASSETS_DIR` to use another location.

The published Figshare assets are public. The downloader follows their file links automatically and requires no account, API key, or credentials.
