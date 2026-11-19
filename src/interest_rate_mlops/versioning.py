"""Dataset / code version helpers.

A model is only meaningful together with the exact data and code that produced
it. Three identifiers are recorded on every run:

* ``dataset_version`` - content hash of the raw file (``sha256:<12 hex>``)
* ``dvc_hash``        - the DVC object hash from ``<file>.dvc`` when present
* ``git_commit``      - current commit (``dirty`` marked when applicable)
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from interest_rate_mlops.config import PROJECT_ROOT
from interest_rate_mlops.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class VersionRecord:
    """Reproducibility identifiers for one dataset snapshot."""

    dataset_path: str
    dataset_version: str
    file_size_bytes: int
    dvc_hash: str | None
    git_commit: str | None
    git_dirty: bool | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_version(path: str | Path) -> str:
    """Short, content-addressed dataset version string."""
    return f"sha256:{file_sha256(path)[:12]}"


def git_commit(root: Path | str = PROJECT_ROOT, short: bool = True) -> str | None:
    """Current git commit, or ``None`` when git is unavailable/not a repo."""
    args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            args,
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return result.stdout.strip() or None


def git_is_dirty(root: Path | str = PROJECT_ROOT) -> bool | None:
    """Whether the working tree has uncommitted changes (``None`` if unknown)."""
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "status", "--porcelain"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return bool(result.stdout.strip())


def dvc_hash(path: str | Path) -> str | None:
    """Read the DVC object hash from the sibling ``<file>.dvc`` pointer."""
    dvc_file = Path(f"{path}.dvc")
    if not dvc_file.exists():
        return None
    try:
        payload = yaml.safe_load(dvc_file.read_text(encoding="utf-8")) or {}
        for output in payload.get("outs", []):
            if output.get("md5"):
                return f"md5:{output['md5']}"
    except (yaml.YAMLError, OSError) as exc:  # pragma: no cover - defensive
        logger.warning(
            "could not read DVC pointer", extra={"event": "dvc_hash_failed"}, exc_info=exc
        )
    return None


def build_version_record(path: str | Path) -> VersionRecord:
    """Collect all reproducibility identifiers for a dataset file."""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"dataset file not found: {resolved}")
    return VersionRecord(
        dataset_path=str(resolved),
        dataset_version=dataset_version(resolved),
        file_size_bytes=resolved.stat().st_size,
        dvc_hash=dvc_hash(resolved),
        git_commit=git_commit(),
        git_dirty=git_is_dirty(),
    )
