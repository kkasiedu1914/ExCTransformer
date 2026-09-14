from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "exctransformer.cli", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def test_download_list_smoke() -> None:
    result = _run_cli("download", "list")
    payload = json.loads(result.stdout)
    names = {asset["name"] for asset in payload["assets"]}
    assert "exctransformer-base" in names
    assert "hmof-metadata" in names


def test_download_fetch_local_asset_smoke(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": "test",
                "assets": [
                    {
                        "name": "id-prop-reg-h2-fixture",
                        "kind": "metadata",
                        "source_path": str(ROOT / "tests" / "fixtures" / "data" / "id_prop_reg_h2.csv"),
                        "file_name": "id_prop_reg_h2.csv",
                        "sha256": "a6766da03b021edbff833de63e9a10c21854d6b030db82ac586d19e0d47ddc33",
                        "extract": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = _run_cli(
        "download",
        "fetch",
        "id-prop-reg-h2-fixture",
        "--manifest",
        str(manifest),
        "--destination",
        str(tmp_path),
    )
    payload = json.loads(result.stdout)
    assert Path(payload["downloaded_to"]).exists()
