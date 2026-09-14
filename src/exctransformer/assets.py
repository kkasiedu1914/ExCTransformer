"""Manifest-driven asset listing and download helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tarfile
import urllib.error
import urllib.request
from urllib.parse import urlparse
import zipfile
from pathlib import Path
from typing import Any

from exctransformer._version import __version__


class AssetError(RuntimeError):
    """Raised when asset manifests or downloads are invalid."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def package_manifest_dir() -> Path:
    return Path(__file__).resolve().parent / "manifests"


def repo_manifest_dir() -> Path:
    return repo_root() / "assets" / "manifests"


def default_assets_dir() -> Path:
    base = os.getenv("EXCTRANSFORMER_ASSETS_DIR", "~/.cache/exctransformer/assets")
    return Path(base).expanduser().resolve()


def resolve_manifest_path(
    *,
    manifest: str | None = None,
    version: str | None = None,
) -> Path:
    if manifest:
        path = Path(manifest).expanduser().resolve()
        if not path.exists():
            raise AssetError(f"Manifest not found: {path}")
        return path

    version = version or __version__
    candidate_names = [f"{version}.json"]
    if not str(version).startswith("v"):
        candidate_names.append(f"v{version}.json")

    for candidate_name in candidate_names:
        repo_candidate = repo_manifest_dir() / candidate_name
        if repo_candidate.exists():
            return repo_candidate

        package_candidate = package_manifest_dir() / candidate_name
        if package_candidate.exists():
            return package_candidate

    raise AssetError(f"No manifest found for version {version}.")


def load_manifest(*, manifest: str | None = None, version: str | None = None) -> dict[str, Any]:
    manifest_path = resolve_manifest_path(manifest=manifest, version=version)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["_manifest_path"] = str(manifest_path)
    return payload


def list_manifest_assets(*, manifest: str | None = None, version: str | None = None) -> list[dict[str, Any]]:
    payload = load_manifest(manifest=manifest, version=version)
    assets = payload.get("assets", [])
    result = []
    for asset in assets:
        result.append(
            {
                "name": asset["name"],
                "kind": asset.get("kind", "data"),
                "description": asset.get("description", ""),
                "version": payload.get("version", version or __version__),
                "extract": bool(asset.get("extract", False)),
                "file_name": asset.get("file_name"),
                "size_bytes": asset.get("size_bytes"),
                "url": asset.get("url"),
            }
        )
    return result


def _file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    return _file_hash(path, "sha256")


def _md5(path: Path) -> str:
    return _file_hash(path, "md5")


def _verify_sha256(path: Path, expected: str | None) -> None:
    if not expected:
        return
    observed = _sha256(path)
    if observed != expected:
        raise AssetError(f"Checksum mismatch for {path.name}: expected {expected}, got {observed}")


def _verify_md5(path: Path, expected: str | None) -> None:
    if not expected:
        return
    observed = _md5(path)
    if observed != expected:
        raise AssetError(f"MD5 mismatch for {path.name}: expected {expected}, got {observed}")


def _verify_checksums(path: Path, entry: dict[str, Any]) -> None:
    _verify_sha256(path, entry.get("sha256"))
    _verify_md5(path, entry.get("md5"))


def _resolve_source_path(asset: dict[str, Any], manifest_path: Path) -> Path:
    source = asset.get("source_path")
    if not source:
        raise AssetError(f"Asset {asset['name']} is missing both url and source_path.")
    candidate = Path(source).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()
    repo_candidate = repo_root() / source
    if repo_candidate.exists():
        return repo_candidate.resolve()
    manifest_candidate = manifest_path.parent / source
    if manifest_candidate.exists():
        return manifest_candidate.resolve()
    raise AssetError(f"Local source_path not found for asset {asset['name']}: {source}")


def _figshare_token() -> str | None:
    token = os.getenv("FIGSHARE_TOKEN")
    if token:
        return token.strip()
    token_file = os.getenv("FIGSHARE_TOKEN_FILE")
    if token_file:
        path = Path(token_file).expanduser()
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    return None


def _download_url(url: str, destination: Path) -> None:
    headers = {}
    parsed = urlparse(url)
    if parsed.scheme == "https" and parsed.hostname in {"figshare.com", "api.figshare.com", "ndownloader.figshare.com"}:
        token = _figshare_token()
        if token:
            headers["Authorization"] = f"token {token}"
    request = urllib.request.Request(url, headers=headers)

    if headers:
        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
                return None

        opener = urllib.request.build_opener(_NoRedirect)
        try:
            response = opener.open(request)
        except urllib.error.HTTPError as exc:
            if exc.code not in {301, 302, 303, 307, 308}:
                raise
            redirected_url = exc.headers.get("Location")
            if not redirected_url:
                raise
            request = urllib.request.Request(redirected_url)
        else:
            with response, open(destination, "wb") as handle:
                shutil.copyfileobj(response, handle)
            return

    with urllib.request.urlopen(request) as response, open(destination, "wb") as handle:
        shutil.copyfileobj(response, handle)


def _contained_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if Path(relative).is_absolute() or not candidate.is_relative_to(root.resolve()):
        raise AssetError(f"Path escapes asset destination: {relative}")
    return candidate


def _extract_archive(path: Path, destination: Path) -> Path:
    extracted_dir = destination / "extracted"
    _contained_path(destination, "extracted")
    extracted_dir.mkdir(parents=True, exist_ok=True)

    suffixes = "".join(path.suffixes[-2:]).lower()
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                _contained_path(extracted_dir, member.filename)
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise AssetError(f"Archive links are not supported: {member.filename}")
            archive.extractall(extracted_dir)
    elif suffixes in {".tar.gz", ".tgz"} or path.suffix.lower() == ".tar":
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                _contained_path(extracted_dir, member.name)
                if not (member.isfile() or member.isdir()):
                    raise AssetError(f"Unsupported archive member: {member.name}")
            archive.extractall(extracted_dir)
    else:
        raise AssetError(f"Unsupported archive format for extraction: {path.name}")
    return extracted_dir


def download_asset(
    name: str,
    *,
    manifest: str | None = None,
    version: str | None = None,
    destination: str | None = None,
    force: bool = False,
    extract: bool | None = None,
) -> dict[str, Any]:
    payload = load_manifest(manifest=manifest, version=version)
    manifest_path = Path(payload["_manifest_path"])
    manifest_version = payload.get("version", version or __version__)
    assets = payload.get("assets", [])

    entry = next((asset for asset in assets if asset["name"] == name), None)
    if entry is None:
        raise AssetError(f"Unknown asset '{name}'.")

    asset_root = Path(destination).expanduser().resolve() if destination else default_assets_dir()
    target_subdir = entry.get("destination") or name
    version_dir = _contained_path(asset_root, str(manifest_version))
    target_dir = _contained_path(version_dir, target_subdir)
    target_dir.mkdir(parents=True, exist_ok=True)

    file_name = entry.get("file_name")
    if not file_name:
        if entry.get("url"):
            file_name = Path(entry["url"]).name
        elif entry.get("source_path"):
            file_name = Path(entry["source_path"]).name
        else:
            file_name = name

    downloaded_path = _contained_path(target_dir, file_name)
    should_extract = entry.get("extract", False) if extract is None else extract

    if downloaded_path.exists() and not force:
        _verify_checksums(downloaded_path, entry)
    else:
        if downloaded_path.exists():
            downloaded_path.unlink()

        if entry.get("url"):
            _download_url(entry["url"], downloaded_path)
        else:
            shutil.copy2(_resolve_source_path(entry, manifest_path), downloaded_path)

        _verify_checksums(downloaded_path, entry)

    extracted_to = None
    if should_extract:
        extracted_to = _extract_archive(downloaded_path, target_dir)

    metadata = {
        "name": name,
        "kind": entry.get("kind", "data"),
        "version": manifest_version,
        "downloaded_to": str(downloaded_path),
        "cached_under": str(target_dir),
        "source": entry.get("url") or entry.get("source_path"),
        "sha256": entry.get("sha256"),
        "md5": entry.get("md5"),
        "extract": bool(should_extract),
        "extracted_to": str(extracted_to) if extracted_to else None,
    }
    metadata_path = target_dir / "asset.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return metadata


def download_all_assets(
    *,
    manifest: str | None = None,
    version: str | None = None,
    destination: str | None = None,
    force: bool = False,
    extract: bool | None = None,
    kind: str | None = None,
) -> list[dict[str, Any]]:
    payload = load_manifest(manifest=manifest, version=version)
    results = []
    for entry in payload.get("assets", []):
        if kind and entry.get("kind", "data") != kind:
            continue
        results.append(
            download_asset(
                entry["name"],
                manifest=manifest,
                version=payload.get("version", version),
                destination=destination,
                force=force,
                extract=extract,
            )
        )
    return results
