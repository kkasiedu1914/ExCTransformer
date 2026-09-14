import io
import tarfile
import zipfile

import pytest

from exctransformer.assets import AssetError, _extract_archive


@pytest.mark.parametrize("kind", ["zip", "tar"])
def test_archive_rejects_parent_traversal(tmp_path, kind):
    path = tmp_path / f"bad.{kind}"
    if kind == "zip":
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("../../escaped.txt", "bad")
    else:
        with tarfile.open(path, "w") as archive:
            member = tarfile.TarInfo("../../escaped.txt")
            member.size = 3
            archive.addfile(member, io.BytesIO(b"bad"))
    with pytest.raises(AssetError):
        _extract_archive(path, tmp_path / "cache")
    assert not (tmp_path / "escaped.txt").exists()


def test_tar_rejects_symbolic_links(tmp_path):
    path = tmp_path / "bad.tar"
    with tarfile.open(path, "w") as archive:
        member = tarfile.TarInfo("escape")
        member.type = tarfile.SYMTYPE
        member.linkname = "../.."
        archive.addfile(member)
    with pytest.raises(AssetError):
        _extract_archive(path, tmp_path / "cache")


@pytest.mark.parametrize("kind", ["zip", "tar"])
def test_valid_archive_extracts(tmp_path, kind):
    path = tmp_path / f"good.{kind}"
    if kind == "zip":
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("data/example.txt", "ok")
    else:
        with tarfile.open(path, "w") as archive:
            member = tarfile.TarInfo("data/example.txt")
            member.size = 2
            archive.addfile(member, io.BytesIO(b"ok"))
    target = _extract_archive(path, tmp_path / "cache")
    assert (target / "data/example.txt").read_text() == "ok"
