"""Session transfer roundtrip tests (export .pianoscribe -> import)."""

import json

from piano_scribe.project import Project
from piano_scribe.transfer import export_session, import_session, TransferError
from piano_scribe.types import NoteEvent


def _make_project(root, name="my song", notes=2):
    proj = Project.create(root / name, name)
    proj.notes = [
        NoteEvent(pitch=60, onset=0.0, end=0.8, confidence=0.9, velocity=90),
        NoteEvent(pitch=64, onset=0.5, end=1.2, confidence=0.4, velocity=80),
    ][:notes]
    proj._save_corrected()
    return proj


def test_export_import_roundtrip(tmp_path):
    proj = _make_project(tmp_path)
    archive = tmp_path / "out.pianoscribe"
    export_session(proj.root, archive)
    assert archive.exists() and archive.stat().st_size > 0

    target = tmp_path / "incoming"
    folder = import_session(archive, target)
    assert folder.name == "my song"
    assert (folder / "project.json").exists()
    imported = Project.open(folder)
    assert [n.pitch for n in imported.notes] == [60, 64]
    meta = json.loads((folder / "project.json").read_text(encoding="utf-8"))
    assert meta["name"] == "my song"


def test_import_existing_name_gets_suffix(tmp_path):
    _make_project(tmp_path)
    archive = tmp_path / "out.pianoscribe"
    export_session(tmp_path / "my song", archive)
    incoming = tmp_path / "incoming"
    f1 = import_session(archive, incoming)           # first import
    f2 = import_session(archive, incoming)           # second -> suffix
    assert f1.name == "my song"
    assert f2.name == "my song-2"


def test_import_rejects_non_sessions(tmp_path):
    bogus = tmp_path / "bogus.pianoscribe"
    import zipfile
    with zipfile.ZipFile(bogus, "w") as zf:
        zf.writestr("random/file.txt", "hi")
    try:
        import_session(bogus, tmp_path)
        raise AssertionError("should have raised")
    except TransferError:
        pass


def test_import_rejects_path_traversal(tmp_path):
    evil = tmp_path / "evil.pianoscribe"
    import zipfile
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("evil/project.json", "{}")
        zf.writestr("evil/../../outside.txt", "pwn")
    try:
        import_session(evil, tmp_path)
        raise AssertionError("should have raised")
    except TransferError:
        pass
    assert not (tmp_path / "outside.txt").exists()