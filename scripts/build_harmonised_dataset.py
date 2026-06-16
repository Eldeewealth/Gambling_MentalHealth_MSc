from __future__ import annotations

from pathlib import Path

import pandas as pd

HARMONISED_PATH = Path("data/processed/harmonised/combined_individual_level.parquet")


def build_harmonised_dataset() -> pd.DataFrame:
    """Load the harmonised combined dataset from local processed parquet."""
    if not HARMONISED_PATH.exists():
        raise FileNotFoundError(f"Harmonised dataset not found: {HARMONISED_PATH}")

    df = pd.read_parquet(HARMONISED_PATH)
    return df.copy()
