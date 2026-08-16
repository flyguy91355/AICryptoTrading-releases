"""Applying a downloaded release to a live install: what's safe to touch,
and (Task 4) how the archive gets extracted and copied. Deny-list always
wins over allow-list, per the spec's 'never touch instance data' rule.
See docs/superpowers/specs/2026-08-11-update-available-feature-design.md."""

import shutil
import tarfile
from pathlib import Path

DENIED_PATH_PREFIXES = (
    ".env",
    "data/",
    "config/settings.yaml",
    "certs/",
)

ALLOWED_PATH_PREFIXES = (
    "src/",
    "web/",
    "scripts/",
    "requirements.txt",
    "requirements-lock.txt",
)


def is_path_updatable(relative_path: str) -> bool:
    """True iff an update is allowed to overwrite this path (relative to
    the install root). Checked in order: deny-list first (always wins),
    then the explicit allow-list, then a fallback rule for a bare
    top-level *.py file (e.g. start.py)."""
    normalized = relative_path.replace("\\", "/").lstrip("/")

    for denied in DENIED_PATH_PREFIXES:
        if normalized == denied or normalized.startswith(denied):
            return False

    for allowed in ALLOWED_PATH_PREFIXES:
        if normalized == allowed or normalized.startswith(allowed):
            return True

    if "/" not in normalized and normalized.endswith(".py"):
        return True

    return False


def requirements_changed(old_content: str, new_content: str) -> bool:
    """True iff the two requirements.txt contents differ, ignoring leading/
    trailing whitespace (a trailing-newline-only diff shouldn't trigger a
    real pip install)."""
    return old_content.strip() != new_content.strip()


def extract_release_archive(archive_path: str, dest_dir: str) -> str:
    """Extracts a .tar.gz release archive into dest_dir and returns the
    path to its single top-level directory (GitHub's auto-generated
    release tarballs always have exactly one)."""
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(dest_dir, filter="data")
    dest_path = Path(dest_dir)
    top_level_dirs = [entry for entry in dest_path.iterdir() if entry.is_dir()]
    if len(top_level_dirs) != 1:
        raise ValueError(
            f"Expected exactly one top-level directory in the release archive, "
            f"found {len(top_level_dirs)}"
        )
    return str(top_level_dirs[0])


def copy_updatable_files(source_dir: str, target_dir: str) -> list[str]:
    """Walks source_dir, copies every file whose path (relative to
    source_dir) passes is_path_updatable() into the same relative path
    under target_dir, creating parent directories as needed. Returns the
    sorted list of relative paths actually copied. Never touches anything
    outside that allow-list, even if the source tree contains a denied
    path (e.g. a stray data/ directory in a malformed archive) — the deny
    check in is_path_updatable() is authoritative regardless of what's on
    disk in target_dir already."""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    copied: list[str] = []

    for file_path in source_path.rglob("*"):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(source_path).as_posix()
        if not is_path_updatable(relative):
            continue
        destination = target_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, destination)
        copied.append(relative)

    return sorted(copied)
