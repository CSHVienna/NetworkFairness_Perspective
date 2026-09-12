"""Path and filesystem helpers for config/scenario I/O."""

from __future__ import annotations

from pathlib import Path


def path_exists(path: str | Path) -> bool:
    """Return True if *path* exists on the filesystem."""
    return Path(path).exists()


def validate_dir(path: str | Path) -> Path:
    """Ensure *path* exists as a directory; create it (and parents) if not. Return the path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def path_join(*parts: str | Path) -> Path:
    """Join path components and return a Path."""
    return Path(*parts)
