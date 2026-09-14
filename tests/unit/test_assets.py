from __future__ import annotations

import json
from pathlib import Path

from exctransformer.assets import download_asset, list_manifest_assets, resolve_manifest_path


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
                        "sha256": "095474f860f642ce017b339641c131ab7aa54f042f98b413ba03ce23e4a4ce97",
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
