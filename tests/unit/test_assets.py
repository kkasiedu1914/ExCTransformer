from __future__ import annotations

import json
from pathlib import Path

from exctransformer.assets import download_asset, list_manifest_assets, resolve_manifest_path, load_manifest
from exctransformer._version import __version__


ROOT = Path(__file__).resolve().parents[2]


def test_manifest_resolves_current_version() -> None:
    manifest_path = resolve_manifest_path(version="1.0.0")
    assert manifest_path.name == "v1.0.0.json"


def test_manifest_lists_expected_assets() -> None:
    assets = list_manifest_assets(version="1.0.0")
    names = {asset["name"] for asset in assets}
    assert "exctransformer-base" in names
    assert "hmof-graph-data" in names
    assert "hmof-metadata" in names
    assert "id-prop-reg-h2" in names


def test_local_asset_download_writes_metadata(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": "test",
                "assets": [
                    {
                        "name": "hmof-metadata-fixture",
                        "kind": "metadata",
                        "source_path": str(ROOT / "tests" / "fixtures" / "data" / "hmof_data.csv"),
                        "file_name": "hmof_data.csv",
                        "sha256": "a88d9f5a6927905212ea5d64df1e0b5ae9d79674c2e46e84edcbd4a677f62b92",
                        "extract": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = download_asset("hmof-metadata-fixture", manifest=str(manifest), destination=str(tmp_path))
    downloaded = Path(result["downloaded_to"])
    metadata = Path(result["cached_under"]) / "asset.json"

    assert downloaded.exists()
    assert metadata.exists()
    payload = json.loads(metadata.read_text())
    assert payload["name"] == "hmof-metadata-fixture"


def test_default_manifest_matches_package_version() -> None:
    manifest = load_manifest()
    assert manifest["version"] == __version__
    assert Path(manifest["_manifest_path"]).name == f"v{__version__}.json"
    assert len(manifest["assets"]) == 9


def test_current_source_and_packaged_manifests_match() -> None:
    name = f"v{__version__}.json"
    source = json.loads((ROOT / "assets" / "manifests" / name).read_text())
    packaged = json.loads((ROOT / "src" / "exctransformer" / "manifests" / name).read_text())
    assert source == packaged
    previous = json.loads((ROOT / "assets" / "manifests" / "v1.0.0.json").read_text())
    keys = ["name", "url", "file_name", "size_bytes", "md5", "extract", "destination"]
    assert [[entry.get(key) for key in keys] for entry in source["assets"]] == [
        [entry.get(key) for key in keys] for entry in previous["assets"]
    ]
