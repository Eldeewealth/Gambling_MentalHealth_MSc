from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _ensure_parent(path: Path | str) -> Path:
    p = Path(path)
    if p.parent:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


def safe_write_parquet(df: pd.DataFrame, path: Path | str, *, index: bool = False) -> Path:
    """Atomically write a DataFrame to Parquet, ensuring parent dirs exist.

    Writes to a temporary file and moves into place to reduce risk of partial files.
    """
    out = _ensure_parent(path)
    tmp = out.with_suffix(out.suffix + ".tmp")
    df.to_parquet(tmp, index=index)
    tmp.replace(out)
    return out


def safe_write_csv(df: pd.DataFrame, path: Path | str, *, index: bool = False) -> Path:
    """Atomically write a DataFrame to CSV, ensuring parent dirs exist."""
    out = _ensure_parent(path)
    tmp = out.with_suffix(out.suffix + ".tmp")
    df.to_csv(tmp, index=index)
    tmp.replace(out)
    return out


def read_parquet(path: Path | str, columns: Optional[list[str]] = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def read_csv(path: Path | str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def with_suffix(path: Path | str, suffix: str) -> Path:
    p = Path(path)
    return p.with_suffix(suffix)

