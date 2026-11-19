"""Small storage helpers used by pipelines and the API.

Writes are atomic (temp file + ``os.replace``) so a consumer never reads a
half-written JSON summary or parquet file.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from interest_rate_mlops.exceptions import MLOpsError


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory of ``path`` and return the path."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def write_json(path: str | Path, payload: dict[str, Any] | list[Any]) -> Path:
    """Atomically write JSON with a stable key order."""
    resolved = ensure_parent(path)
    handle, temp_name = tempfile.mkstemp(dir=str(resolved.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as temp:
            json.dump(payload, temp, indent=2, sort_keys=True, default=str)
            temp.write("\n")
        Path(temp_name).replace(resolved)
    finally:
        temp_path = Path(temp_name)
        if temp_path.exists():
            temp_path.unlink()
    return resolved


def read_json(path: str | Path, default: Any | None = None) -> Any:
    """Read JSON, returning ``default`` when the file does not exist."""
    resolved = Path(path)
    if not resolved.exists():
        return default
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise MLOpsError(f"corrupt JSON at {resolved}: {exc}") from exc


def write_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    """Atomically write a parquet file (requires pyarrow)."""
    resolved = ensure_parent(path)
    handle, temp_name = tempfile.mkstemp(dir=str(resolved.parent), suffix=".parquet.tmp")
    os.close(handle)
    try:
        frame.to_parquet(temp_name, index=True)
        Path(temp_name).replace(resolved)
    finally:
        temp_path = Path(temp_name)
        if temp_path.exists():
            temp_path.unlink()
    return resolved


def read_parquet(path: str | Path) -> pd.DataFrame:
    """Read parquet, raising a domain error when the artifact is missing."""
    resolved = Path(path)
    if not resolved.exists():
        raise MLOpsError(
            f"artifact not found: {resolved}. Run the ingestion/training pipeline first."
        )
    return pd.read_parquet(resolved)
