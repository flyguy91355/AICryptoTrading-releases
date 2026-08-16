"""Pure version comparison and local VERSION-file I/O for the update-available
feature. No network, no DashboardState dependency — fully unit-testable.
See docs/superpowers/specs/2026-08-11-update-available-feature-design.md."""

from pathlib import Path


def parse_version(tag: str) -> tuple[int, ...]:
    """Parses a tag like 'v1.4.0' or '1.4.0' into (1, 4, 0). Raises ValueError
    on any non-numeric part."""
    cleaned = tag.strip()
    if cleaned.startswith("v") or cleaned.startswith("V"):
        cleaned = cleaned[1:]
    parts = cleaned.split(".")
    return tuple(int(part) for part in parts)


def is_newer(current: str, latest: str) -> bool:
    """True iff latest's parsed version is strictly greater than current's.
    Shorter tuples are padded with zeros so 'v1.4' vs 'v1.4.1' compares as
    (1, 4, 0) < (1, 4, 1) rather than raising."""
    current_parts = parse_version(current)
    latest_parts = parse_version(latest)
    length = max(len(current_parts), len(latest_parts))
    current_padded = current_parts + (0,) * (length - len(current_parts))
    latest_padded = latest_parts + (0,) * (length - len(latest_parts))
    return latest_padded > current_padded


def read_local_version(path: str) -> str | None:
    """Returns the stripped contents of the local VERSION file, or None if
    it doesn't exist yet (a fresh install that hasn't been bootstrapped)."""
    file_path = Path(path)
    if not file_path.exists():
        return None
    return file_path.read_text().strip()


def write_local_version(path: str, version: str) -> None:
    """Writes version (stripped) to the local VERSION file, creating it if
    needed."""
    Path(path).write_text(version.strip() + "\n")
