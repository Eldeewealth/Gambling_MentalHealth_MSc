from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

GSGB_A8_PATH = Path("data/interim/gsgb_a8_raw.parquet")
GSGB_A15_PATH = Path("data/interim/gsgb_a15_raw.parquet")
GSGB_A14_PATH = Path("data/interim/gsgb_a14_raw.parquet")


def _normalize_a8(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace("\n", " ") for c in df.columns]
    df = df.rename(columns={
        "sex and age group (years)": "group_label",
        "participation in the past four weeks (percentage)": "pct_all",
        "participation in the past four weeks excluding lottery draw only players i (percentage)": "pct_no_lottery",
        "unweighted bases (number) ii,iii": "base_unweighted",
        "weighted bases (number) ii,iii": "base_weighted",
    })

    if "group_label" in df.columns:
        df["group_label"] = df["group_label"].astype(str).str.strip()
    df["group_type"] = df["group_label"].apply(lambda x: "age" if any(tok.isdigit() for tok in str(x).split()) else ("sex" if str(x).strip().lower() in {"all participants", "all males", "all females"} else "other"))
    df["age_group"] = df["group_label"].where(df["group_type"] == "age")
    df["sex"] = df["group_label"].where(df["group_type"] == "sex")
    return df


def _normalize_a15(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace("\n", " ") for c in df.columns]
    df = df.rename(columns={
        "feelings towards gambling": "feeling_label",
        "all participants: gambled in the past 12 months (percentage)": "pct_all_gamblers",
        "all participants: gambled in the past 12 months excluding lottery draw only players ii (percentage)": "pct_all_gamblers_excluding_lottery",
    })

    return df


def _normalize_a14(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace("\n", " ") for c in df.columns]
    df = df.rename(columns={
        list(df.columns)[0]: "reason_label",
    })
    return df


def build_gsgb_dataset() -> Dict[str, pd.DataFrame]:
    """Return GSGB A8/A15/A14 datasets from local interim raw parquet sources."""
    output: Dict[str, pd.DataFrame] = {}

    if GSGB_A8_PATH.exists():
        output["a8"] = _normalize_a8(pd.read_parquet(GSGB_A8_PATH))
    if GSGB_A15_PATH.exists():
        output["a15"] = _normalize_a15(pd.read_parquet(GSGB_A15_PATH))
    if GSGB_A14_PATH.exists():
        output["a14"] = _normalize_a14(pd.read_parquet(GSGB_A14_PATH))

    return output
