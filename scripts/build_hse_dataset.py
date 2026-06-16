from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

HSE_RAW_PATH = Path("data/interim/hse_2018.csv")

GHQ_ITEM_COLUMNS = [
    "ghqconc",
    "ghqsleep",
    "ghquse",
    "ghqdecis",
    "ghqstrai",
    "ghqover",
    "ghqenjoy",
    "ghqface",
    "ghqunhap",
    "ghqconfi",
    "ghqworth",
    "ghqhappy",
]

PGSI_ITEM_COLUMNS = [f"pgsi{i}" for i in range(1, 10)]


def _lower_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    return df


def _numeric_sum(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    numeric_cols = [c for c in cols if c in df.columns]
    if not numeric_cols:
        return pd.Series(dtype="float64")
    return df[numeric_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)


def _pgsi_category(score_series: pd.Series) -> pd.Series:
    cats = pd.Series(index=score_series.index, dtype="object")
    cats = cats.where(score_series.notna())

    cats = cats.mask(score_series <= 0, "Non-problem gambler")
    cats = cats.mask((score_series >= 1) & (score_series <= 2), "Low-risk gambler")
    cats = cats.mask((score_series >= 3) & (score_series <= 7), "Moderate-risk gambler")
    cats = cats.mask(score_series >= 8, "Problem gambler")
    return cats


def build_hse_dataset() -> pd.DataFrame:
    """Build the HSE dataset from the raw interim CSV.

    This function loads the raw HSE 2018 CSV from the repository, applies
    minimal cleaning, computes GHQ-12 and PGSI scores, and derives a few
    standard labels used in the analysis dashboard.
    """
    if not HSE_RAW_PATH.exists():
        raise FileNotFoundError(f"HSE raw dataset not found: {HSE_RAW_PATH}")

    df = pd.read_csv(HSE_RAW_PATH)
    df = _lower_columns(df)

    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

    ghq_cols = [c for c in GHQ_ITEM_COLUMNS if c in df.columns]
    if ghq_cols:
        df["ghq12_score"] = _numeric_sum(df, ghq_cols)
    elif "ghq12scr" in df.columns:
        df["ghq12_score"] = pd.to_numeric(df["ghq12scr"], errors="coerce")
    else:
        df["ghq12_score"] = pd.NA

    pgsi_cols = [c for c in PGSI_ITEM_COLUMNS if c in df.columns]
    if pgsi_cols:
        df["pgsi_score"] = _numeric_sum(df, pgsi_cols)
    elif "pgsisc" in df.columns:
        df["pgsi_score"] = pd.to_numeric(df["pgsisc"], errors="coerce")
    else:
        df["pgsi_score"] = pd.NA

    df["problem_gambling"] = df["pgsi_score"].ge(8).astype("Int64")
    df["ghq12_distress"] = df["ghq12_score"].ge(4).astype("Int64")
    df["pgsi_category"] = _pgsi_category(df["pgsi_score"])

    return df
