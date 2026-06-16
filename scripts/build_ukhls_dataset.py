from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

UKHLS_RAW_PATHS: Dict[str, Path] = {
    "K": Path("data/interim/ukhls_k_columns_of_interest.parquet"),
    "L": Path("data/interim/ukhls_l_columns_of_interest.parquet"),
    "N": Path("data/interim/ukhls_n_columns_of_interest.parquet"),
}

WAVE_NUMBER: Dict[str, int] = {"K": 11, "L": 12, "N": 13}
SEX_MAP: Dict[int, str] = {1: "Male", 2: "Female"}

EXPECTED_COLUMNS: List[str] = [
    "participant_id",
    "survey_wave",
    "wave_number",
    "sex",
    "age_years",
    "region",
    "net_monthly_income",
    "mental_health_score",
    "anxiety_level",
]


def _load_wave(path: Path, wave: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"UKHLS raw dataset not found: {path}")

    df = pd.read_parquet(path)
    prefix = wave.lower()

    rename_map = {
        "pidp": "participant_id",
        f"{prefix}_sex_dv": "sex",
        f"{prefix}_age_dv": "age_years",
        f"{prefix}_gor_dv": "region",
        f"{prefix}_fimnnet_dv": "net_monthly_income",
        f"{prefix}_scghq1_dv": "mental_health_score",
    }

    if wave == "N":
        rename_map["n_mhgad"] = "anxiety_level"

    df = df.rename(columns=rename_map)
    if "participant_id" in df.columns:
        df["participant_id"] = pd.to_numeric(df["participant_id"], errors="coerce").astype("Int64")

    if "sex" in df.columns:
        df["sex"] = pd.to_numeric(df["sex"], errors="coerce").map(SEX_MAP).astype("string")

    df = df.assign(
        survey_wave=wave,
        wave_number=WAVE_NUMBER[wave],
    )

    selected = [c for c in EXPECTED_COLUMNS if c in df.columns]
    return df.loc[:, selected]


def build_ukhls_dataset() -> pd.DataFrame:
    """Build the UKHLS dataset from local interim UKHLS parquet sources."""
    frames: List[pd.DataFrame] = []
    for wave, path in UKHLS_RAW_PATHS.items():
        frames.append(_load_wave(path, wave))

    if not frames:
        raise RuntimeError("No UKHLS datasets could be loaded.")

    combined = pd.concat(frames, ignore_index=True, copy=False)
    combined = combined.drop_duplicates(subset=["participant_id", "survey_wave"], keep="first")
    return combined.loc[:, [c for c in EXPECTED_COLUMNS if c in combined.columns]]
