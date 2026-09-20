"""Session transfer: export a session to a .pianoscribe archive and import
it back — the "easy to transfer in and out" path for the front end.

Archive layout = the project directory verbatim (project.json, recording/,
corrected/, score/, exports/). Import validates and safely extracts:
no absolute paths, no ".." escapes, exactly one project.json.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


class TransferError(Exception):
    pass


def export_session(project_root: Path, out_zip: Path, include_processing: bool = True) -> Path:
    """Zip a session folder into a portable .pianoscribe archive."""
    project_root = Path(project_root)
    out_zip = Path(out_zip)
    if not (project_root / "project.json").exists():
        raise TransferError(f"not a piano-scribe session: {project_root}")
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(project_root.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(project_root).as_posix()
            if not include_processing and rel.startswith("processing/"):
                continue
            zf.write(f, arcname=f"{project_root.name}/{rel}")
    return out_zip


def import_session(archive: Path, target_dir: Path) -> Path:
    """Extract a .pianoscribe archive into ``target_dir``; returns the
    new session folder (existing name gets a -2 suffix)."""
    archive = Path(archive)
    target_dir = Path(target_dir)
    if not archive.exists():
        raise TransferError(f"file not found: {archive}")
    try:
        zf = zipfile.ZipFile(archive)
    except zipfile.BadZipFile as e:
        raise TransferError(f"not a valid session archive: {archive.name}") from e

    with zf:
        members = zf.infolist()
        if not members:
            raise TransferError("archive is empty")
        # safety: only relative paths within a single root folder
        roots = set()
        for m in members:
            p = Path(m.filename)
            if p.is_absolute() or ".." in p.parts:
                raise TransferError(f"unsafe path in archive: {m.filename}")
            roots.add(p.parts[0] if len(p.parts) > 1 else p.name)
        if len(roots) != 1:
            raise TransferError("archive must contain exactly one session folder")
        root_name = next(iter(roots))
        session_folder = target_dir / root_name
        i = 2
        while session_folder.exists():
            session_folder = target_dir / f"{root_name}-{i}"
            i += 1
        extract_root = session_folder.parent / f"__extract_{root_name}"
        if extract_root.exists():
            shutil.rmtree(extract_root)
        session_folder.parent.mkdir(parents=True, exist_ok=True)
        zf.extractall(extract_root)
        src = extract_root / root_name
        if not (src / "project.json").exists():
            shutil.rmtree(extract_root, ignore_errors=True)
            raise TransferError("no project.json inside — not a piano-scribe session")
        shutil.move(str(src), str(session_folder))
        shutil.rmtree(extract_root, ignore_errors=True)
    return session_folder