from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


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


@pytest.mark.parametrize("explicit_arguments", [False, True])
def test_download_shortcut_list_smoke(explicit_arguments: bool) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    call = "download_main(['list'])" if explicit_arguments else "download_main()"
    result = subprocess.run(
        [sys.executable, "-c",
         f"from exctransformer.cli import download_main; raise SystemExit({call})",
         "invalid-process-argument" if explicit_arguments else "list"],
        cwd=ROOT, env=env, text=True, capture_output=True, check=True,
    )
    names = {asset["name"] for asset in json.loads(result.stdout)["assets"]}
    assert "exctransformer-base" in names


def test_download_shortcut_explicit_empty_arguments_do_not_use_process_arguments() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c",
         "from exctransformer.cli import download_main; raise SystemExit(download_main([]))",
         "list"],
        cwd=ROOT, env=env, text=True, capture_output=True,
    )
    assert result.returncode == 2
    assert "download_command" in result.stderr


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
                        "sha256": "836d33e995909a7acf98ec42a32a94958054a59ba87e47437c7a256b79b6fdd9",
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
