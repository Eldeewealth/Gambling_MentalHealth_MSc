from __future__ import annotations

from typing import Iterable

import pandas as pd

# Common missing/null label variants encountered in surveys
MISS_LABELS: set[str] = {
    "", "na", "n/a", "-", "--", "none", "null", "nan",
    "don't know", "dont know", "refused", "dk",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with lower_snake_case column names."""
    def to_snake(s: str) -> str:
        return (
            s.strip()
            .replace("/", "_")
            .replace("-", "_")
            .replace(" ", "_")
            .lower()
        )

    out = df.copy()
    out.columns = [to_snake(c) for c in out.columns]
    return out


def normalize_missing(df: pd.DataFrame, cols: Iterable[str] | None = None) -> pd.DataFrame:
    """Standardize survey-style missing labels to real NaN.

    If cols is None, applies to all object/string columns.
    """
    out = df.copy()
    target_cols = cols or [c for c in out.columns if pd.api.types.is_string_dtype(out[c])]
    for c in target_cols:
        out[c] = out[c].astype("string").str.strip().str.lower().replace(list(MISS_LABELS), pd.NA)
    return out


def promote_columns(df: pd.DataFrame, preferred_order: list[str]) -> pd.DataFrame:
    """Reorder columns so important ones appear first; keep others afterward."""
    rest = [c for c in df.columns if c not in preferred_order]
    return df[[*(c for c in preferred_order if c in df.columns), *rest]]

