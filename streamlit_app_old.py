# =========================
# streamlit_app.py (1/5) — Imports, constants, config
# =========================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

APP_TITLE = "Gambling and mental health in the UK — Dashboard of dissertation outputs"
APP_SUBTITLE = (
    "Examiner-facing dashboard built from analysis-ready outputs: HSE 2018, GSGB 2023, UKHLS (Understanding Society)."
)

DEFAULT_PATHS: Dict[str, str] = {
    # HSE + UKHLS (analysis-ready)
    "hse_analysis_ready": "data/processed/analysis-ready/hse_2018_analysis_ready.parquet",
    "ukhls_analysis_ready": "data/processed/analysis-ready/ukhls_kln_analysis_ready_commonpidp.parquet",
    # GSGB (model-ready summary tables)
    "gsgb_a8": "data/processed/model-ready/gsgb_a8_model_ready.parquet",
    "gsgb_a15": "data/processed/model-ready/gsgb_a15_model_ready.parquet",
    "gsgb_a14": "data/processed/model-ready/gsgb_a14_model_ready.parquet",  # optional

    "harmonised_combined": "data/processed/harmonised_combined_hse_ukhls.parquet",
}

st.set_page_config(page_title="MSc dashboard — Gambling & mental health", layout="wide")
# =========================
# streamlit_app.py (2/5) — Robust file IO + column detection utilities
# =========================

def _to_path(p: Union[str, Path]) -> Path:
    return p if isinstance(p, Path) else Path(p)


def read_parquet_safe(path: Union[str, Path]) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    p = _to_path(path)
    if not p.exists():
        return None, f"File not found: {p.as_posix()}"
    try:
        return pd.read_parquet(p), None
    except Exception as e:
        return None, f"Could not read parquet: {p.as_posix()}\n{type(e).__name__}: {e}"


def resolve_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Return the first matching column name from candidates (case-sensitive, then case-insensitive)."""
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    lower_map = {c.lower(): c for c in cols}
    for c in candidates:
        key = c.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def list_like_pct_columns(df: pd.DataFrame) -> List[str]:
    """
    Heuristic: return columns likely to be 'percentage/proportion' measures.
    Useful for GSGB A8 where the main measure may differ slightly.
    """
    pct_cols: List[str] = []
    for c in df.columns:
        cl = str(c).lower()
        if cl.startswith("pct") or "percent" in cl or cl.endswith("_pct") or cl.endswith("_percentage"):
            pct_cols.append(c)
    return pct_cols


def ensure_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(s, errors="coerce")


def looks_like_proportion(s: pd.Series) -> bool:
    """
    If values are mostly <= 1.5, treat as proportion (0–1). Otherwise treat as percent already.
    """
    x = ensure_numeric(s).dropna()
    if x.empty:
        return False
    return float(x.max()) <= 1.5


def to_percent_series(s: pd.Series) -> pd.Series:
    x = ensure_numeric(s)
    return x * 100.0 if looks_like_proportion(x) else x


def coerce_binary(s: pd.Series) -> pd.Series:
    """
    Coerce a series into 0/1 when possible. Works for numeric and common string encodings.
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    ss = s.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1, "0": 0,
        "yes": 1, "no": 0,
        "true": 1, "false": 0,
        "y": 1, "n": 0,
        "distressed": 1, "not distressed": 0,
    }
    out = ss.map(mapping)
    out2 = pd.to_numeric(ss, errors="coerce")
    out = out.where(out.notna(), out2)
    return out


def weighted_mean(x: pd.Series, w: pd.Series) -> float:
    x = ensure_numeric(x)
    w = ensure_numeric(w)

    m = x.notna() & w.notna()
    if int(m.sum()) == 0:
        return float("nan")

    ww = w[m].astype(float)
    xx = x[m].astype(float)

    # Drop non-positive weights (common source of zero-sum)
    keep = ww > 0
    ww = ww[keep]
    xx = xx[keep]

    if ww.empty:
        return float("nan")

    wsum = float(ww.sum())
    if wsum <= 0:
        return float("nan")

    return float((xx * ww).sum() / wsum)

def weighted_pct_binary(x: pd.Series, w: pd.Series) -> float:
    xb = coerce_binary(x)
    w = ensure_numeric(w)

    m = xb.notna() & w.notna()
    if int(m.sum()) == 0:
        return float("nan")

    ww = w[m].astype(float)
    xx = xb[m].astype(float)

    keep = ww > 0
    ww = ww[keep]
    xx = xx[keep]

    if ww.empty:
        return float("nan")

    wsum = float(ww.sum())
    if wsum <= 0:
        return float("nan")

    return float((xx * ww).sum() / wsum * 100.0)

def missing_cols_message(missing: List[str], file_key: str) -> str:
    return (
        "This section cannot be shown because required columns are missing.\n\n"
        f"- Missing columns: {', '.join(missing)}\n"
        f"- Source: {file_key}\n\n"
        "Fix: confirm the parquet has these columns, or edit the file paths in the sidebar."
    )


def unique_sorted_str(df: pd.DataFrame, col: str) -> List[str]:
    vals = df[col].dropna().astype(str)
    vals = vals[vals.str.strip() != ""]
    return sorted(vals.unique().tolist())
# =========================
# streamlit_app.py (3/5) — Loaders (cached) + filter logic + sidebar paths
# =========================

@st.cache_data(show_spinner=False)
def load_hse(path: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    df, err = read_parquet_safe(path)
    if df is None:
        return None, err

    # Derived variable per notebook: ghq12_distress = 1 if ghq12_score >= 4 else 0
    ghq_col = resolve_col(df, ["ghq12_score"])
    if ghq_col is not None and "ghq12_distress" not in df.columns:
        ghq = ensure_numeric(df[ghq_col])
        df["ghq12_distress"] = np.where(ghq >= 4, 1, np.where(ghq.notna(), 0, np.nan))

    return df, None


@st.cache_data(show_spinner=False)
def load_ukhls(path: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    return read_parquet_safe(path)


@st.cache_data(show_spinner=False)
def load_gsgb_a8(path: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    return read_parquet_safe(path)


@st.cache_data(show_spinner=False)
def load_gsgb_a15(path: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    return read_parquet_safe(path)

@st.cache_data(show_spinner=False)
def load_harmonised_combined(path: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    return read_parquet_safe(path)

@st.cache_data(show_spinner=False)
def load_gsgb_a14(path: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    return read_parquet_safe(path)


@dataclass
class FilterSpec:
    sex: Optional[List[str]] = None
    age_group: Optional[List[str]] = None
    country: Optional[List[str]] = None
    region: Optional[List[str]] = None
    pgsi: Optional[List[str]] = None
    distress: Optional[List[str]] = None
    wave: Optional[List[str]] = None


def apply_filters(df: pd.DataFrame, spec: FilterSpec) -> pd.DataFrame:
    out = df.copy()

    if spec.sex:
        c = resolve_col(out, ["sex", "sex_harmonised"])
        if c is not None:
            out = out[out[c].astype(str).isin(spec.sex)]

    if spec.age_group:
        c = resolve_col(out, ["age_group", "age_group_harmonised"])
        if c is not None:
            out = out[out[c].astype(str).isin(spec.age_group)]

    if spec.country:
        c = resolve_col(out, ["country"])
        if c is not None:
            out = out[out[c].astype(str).isin(spec.country)]

    if spec.region:
        c = resolve_col(out, ["region"])
        if c is not None:
            out = out[out[c].astype(str).isin(spec.region)]

    if spec.pgsi:
        c = resolve_col(out, ["pgsi_label"])
        if c is not None:
            out = out[out[c].astype(str).isin(spec.pgsi)]

    if spec.distress:
        c = resolve_col(out, ["ghq12_distress"])
        if c is not None:
            dd = coerce_binary(out[c])
            label_map = {1: "Distressed", 0: "Not distressed"}
            out = out.assign(_distress_label=dd.map(label_map))
            out = out[out["_distress_label"].astype(str).isin(spec.distress)]
            out = out.drop(columns=["_distress_label"], errors="ignore")

    if spec.wave:
        c = resolve_col(out, ["survey_wave", "wave_number"])
        if c is not None:
            out = out[out[c].astype(str).isin(spec.wave)]

    return out


def sidebar_paths() -> Dict[str, str]:
    if "paths" not in st.session_state:
        st.session_state["paths"] = dict(DEFAULT_PATHS)

    with st.sidebar.expander("Data file paths (parquet)", expanded=False):
        st.caption("Edit if your folder structure differs. Click Apply to reload.")
        updated: Dict[str, str] = {}
        for k, v in st.session_state["paths"].items():
            updated[k] = st.text_input(k, value=v, key=f"path_{k}")
        if st.button("Apply"):
            st.session_state["paths"] = updated
            st.cache_data.clear()
            st.rerun()

    return st.session_state["paths"]


def reset_filters_button() -> None:
    if st.sidebar.button("Reset filters"):
        for k in list(st.session_state.keys()):
            if k.startswith("flt_"):
                del st.session_state[k]
        st.rerun()
# =========================
# streamlit_app.py (4/5) — Plot helpers (Plotly Express)
# =========================

def plot_distribution_percent(df: pd.DataFrame, col: str, title: str, x_label: Optional[str] = None) -> None:
    s = df[col].dropna().astype(str)
    s = s[s.str.strip() != ""]
    if s.empty:
        st.info("No data available after filtering.")
        return

    tab = s.value_counts(dropna=False).rename_axis(col).reset_index(name="n")
    tab["Proportion (%)"] = tab["n"] / tab["n"].sum() * 100.0

    fig = px.bar(
        tab,
        x=col,
        y="Proportion (%)",
        title=title,
        labels={col: x_label or col.replace("_", " ").title(), "Proportion (%)": "Proportion (%)"},
        hover_data={"n": True, "Proportion (%)": ":.2f"},
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_binary_rate_by_group(
    df: pd.DataFrame,
    outcome_col: str,
    group_cols: List[str],
    title: str,
    y_label: str,
    weight_col: Optional[str] = None,
) -> None:
    missing = [c for c in [outcome_col] + group_cols if c not in df.columns]
    if missing:
        st.error(missing_cols_message(missing, file_key="(current dataframe)"))
        return

    cols = [outcome_col] + group_cols
    if weight_col and weight_col in df.columns:
        cols.append(weight_col)

    d = df[cols].copy()
    d[outcome_col] = coerce_binary(d[outcome_col])

    if weight_col and weight_col in d.columns:
        g = (
            d.dropna(subset=[outcome_col] + group_cols + [weight_col])
            .groupby(group_cols, dropna=False)
            .apply(lambda x: weighted_pct_binary(x[outcome_col], x[weight_col]))
            .reset_index(name=y_label)
        )
    else:
        g = (
            d.dropna(subset=[outcome_col] + group_cols)
            .groupby(group_cols, dropna=False)[outcome_col]
            .mean()
            .reset_index(name=y_label)
        )
        g[y_label] = g[y_label] * 100.0

    if g.empty:
        st.info("No data available after filtering.")
        return

    if len(group_cols) == 1:
        x = group_cols[0]
        fig = px.bar(g, x=x, y=y_label, title=title, labels={x: x.replace("_", " ").title(), y_label: y_label})
    else:
        x = group_cols[0]
        color = group_cols[1]
        fig = px.bar(
            g,
            x=x,
            y=y_label,
            color=color,
            barmode="group",
            title=title,
            labels={x: x.replace("_", " ").title(), color: color.replace("_", " ").title(), y_label: y_label},
        )

    st.plotly_chart(fig, use_container_width=True)


def plot_mean_by_group(
    df: pd.DataFrame,
    value_col: str,
    group_cols: List[str],
    title: str,
    y_label: str,
    weight_col: Optional[str] = None,
) -> None:
    missing = [c for c in [value_col] + group_cols if c not in df.columns]
    if missing:
        st.error(missing_cols_message(missing, file_key="(current dataframe)"))
        return

    cols = [value_col] + group_cols
    if weight_col and weight_col in df.columns:
        cols.append(weight_col)

    d = df[cols].copy()
    d[value_col] = ensure_numeric(d[value_col])

    # Always compute base n (unweighted)
    base = (
        d.dropna(subset=[value_col] + group_cols)
        .groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="n")
    )

    if weight_col and weight_col in d.columns:
        # Weighted mean
        g = (
            d.dropna(subset=[value_col] + group_cols + [weight_col])
            .groupby(group_cols, dropna=False)
            .apply(lambda x: weighted_mean(x[value_col], x[weight_col]))
            .reset_index(name=y_label)
        )
        # Also compute weight sum as a "weighted base" (useful for diagnostics)
        wsum = (
            d.dropna(subset=[value_col] + group_cols + [weight_col])
            .groupby(group_cols, dropna=False)[weight_col]
            .apply(lambda s: float(ensure_numeric(s).where(ensure_numeric(s) > 0).sum()))
            .reset_index(name="weight_sum")
        )
        g = g.merge(base, on=group_cols, how="left").merge(wsum, on=group_cols, how="left")
        hover_extra = {"n": True, "weight_sum": ":.2f", y_label: ":.2f"}
    else:
        # Unweighted mean
        g = (
            d.dropna(subset=[value_col] + group_cols)
            .groupby(group_cols, dropna=False)[value_col]
            .mean()
            .reset_index(name=y_label)
        )
        g = g.merge(base, on=group_cols, how="left")
        hover_extra = {"n": True, y_label: ":.2f"}

    if g.empty:
        st.info("No data available after filtering.")
        return

    if len(group_cols) == 1:
        x = group_cols[0]
        fig = px.bar(
            g,
            x=x,
            y=y_label,
            title=title,
            labels={x: x.replace("_", " ").title(), y_label: y_label},
            hover_data=hover_extra,
        )
    else:
        x = group_cols[0]
        color = group_cols[1]
        fig = px.bar(
            g,
            x=x,
            y=y_label,
            color=color,
            barmode="group",
            title=title,
            labels={x: x.replace("_", " ").title(), color: color.replace("_", " ").title(), y_label: y_label},
            hover_data=hover_extra,
        )

    st.plotly_chart(fig, use_container_width=True)
# =========================
# streamlit_app.py (5/5) — Pages + main()
# =========================

def page_harmonised(paths: Dict[str, str]) -> None:
    df, err = load_harmonised_combined(paths["harmonised_combined"])
    if df is None:
        st.error(err)
        st.info("Expected: data/processed/harmonised/combined_individual_level.parquet")
        return

    st.subheader("Harmonised view (HSE + UKHLS)")
    st.caption("Triangulation dataset created in harmonisation step (shared variables across surveys).")

    reset_filters_button()

    # Filters (use harmonised columns if present)
    spec = FilterSpec()
    if "sex_harmonised" in df.columns:
        vals = unique_sorted_str(df, "sex_harmonised")
        spec.sex = st.sidebar.multiselect("Sex (harmonised)", vals, default=vals, key="flt_harm_sex")
    if "age_group_harmonised" in df.columns:
        vals = unique_sorted_str(df, "age_group_harmonised")
        spec.age_group = st.sidebar.multiselect("Age group (harmonised)", vals, default=vals, key="flt_harm_age")
    if "survey_year_harmonised" in df.columns:
        vals = unique_sorted_str(df, "survey_year_harmonised")
        # reuse "wave" slot just as a generic filter holder
        spec.wave = st.sidebar.multiselect("Survey year (harmonised)", vals, default=vals, key="flt_harm_year")

    # apply filters manually for harmonised cols
    df_f = df.copy()
    if spec.sex and "sex_harmonised" in df_f.columns:
        df_f = df_f[df_f["sex_harmonised"].astype(str).isin(spec.sex)]
    if spec.age_group and "age_group_harmonised" in df_f.columns:
        df_f = df_f[df_f["age_group_harmonised"].astype(str).isin(spec.age_group)]
    if spec.wave and "survey_year_harmonised" in df_f.columns:
        df_f = df_f[df_f["survey_year_harmonised"].astype(str).isin(spec.wave)]

    tabs = st.tabs(["Composition", "Sex × age gradients", "Income gradients (if available)"])

    with tabs[0]:
        if "dataset" in df_f.columns:
            plot_distribution_percent(df_f, "dataset", "Composition by dataset", x_label="Dataset")
        if "sex_harmonised" in df_f.columns:
            plot_distribution_percent(df_f, "sex_harmonised", "Composition by sex (harmonised)", x_label="Sex")
        if "age_group_harmonised" in df_f.columns:
            plot_distribution_percent(df_f, "age_group_harmonised", "Composition by age group (harmonised)", x_label="Age group")

    with tabs[1]:
        # This tab focuses on distributional comparisons (since outcomes may not be fully aligned in the combined file)
        if all(c in df_f.columns for c in ["age_group_harmonised", "sex_harmonised"]):
            # show % within filtered dataset: use distribution percent of age group, coloured by sex via a grouped count
            tmp = df_f.dropna(subset=["age_group_harmonised", "sex_harmonised"]).copy()
            g = tmp.groupby(["age_group_harmonised", "sex_harmonised"]).size().reset_index(name="n")
            g["Proportion (%)"] = g["n"] / g.groupby("sex_harmonised")["n"].transform("sum") * 100.0
            fig = px.bar(
                g,
                x="age_group_harmonised",
                y="Proportion (%)",
                color="sex_harmonised",
                barmode="group",
                title="Age distribution by sex (harmonised) — Proportion within sex",
                labels={"age_group_harmonised": "Age group", "sex_harmonised": "Sex", "Proportion (%)": "Proportion (%)"},
                hover_data={"n": True, "Proportion (%)": ":.2f"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Required harmonised columns not found: age_group_harmonised and sex_harmonised.")

    with tabs[2]:
        if "income_quintile_harmonised" in df_f.columns:
            plot_distribution_percent(
                df_f,
                "income_quintile_harmonised",
                "Income quintile (harmonised) distribution",
                x_label="Income quintile",
            )
        else:
            st.info("income_quintile_harmonised not found in this combined file (expected for HSE+UKHLS only).")

    with st.expander("Data preview (first 50 rows)"):
        st.dataframe(df_f.head(50), use_container_width=True)

def page_hse(paths: Dict[str, str]) -> None:
    df, err = load_hse(paths["hse_analysis_ready"])
    if df is None:
        st.error(err)
        return

    st.subheader("HSE 2018")
    st.caption("Individual-level analysis-ready dataset (HSE 2018).")

    reset_filters_button()

    # Sidebar filters (only if columns exist)
    spec = FilterSpec()
    if resolve_col(df, ["sex"]) is not None:
        spec.sex = st.sidebar.multiselect(
            "Sex", options=unique_sorted_str(df, "sex"), default=unique_sorted_str(df, "sex"), key="flt_hse_sex"
        )
    if resolve_col(df, ["age_group"]) is not None:
        spec.age_group = st.sidebar.multiselect(
            "Age group",
            options=unique_sorted_str(df, "age_group"),
            default=unique_sorted_str(df, "age_group"),
            key="flt_hse_age",
        )
    if resolve_col(df, ["pgsi_label"]) is not None:
        spec.pgsi = st.sidebar.multiselect(
            "PGSI category",
            options=unique_sorted_str(df, "pgsi_label"),
            default=unique_sorted_str(df, "pgsi_label"),
            key="flt_hse_pgsi",
        )
    if "ghq12_distress" in df.columns:
        spec.distress = st.sidebar.multiselect(
            "GHQ-12 distress",
            options=["Not distressed", "Distressed"],
            default=["Not distressed", "Distressed"],
            key="flt_hse_distress",
        )

    df_f = apply_filters(df, spec)

    tabs = st.tabs(["Sample profile", "Gambling & harm", "Mental health (GHQ-12)"])

    with tabs[0]:
        missing = [c for c in ["sex", "age_group"] if c not in df_f.columns]
        if missing:
            st.error(missing_cols_message(missing, file_key="hse_analysis_ready"))
        else:
            c1, c2 = st.columns(2)
            with c1:
                plot_distribution_percent(df_f, "sex", "Sample profile: Sex distribution", x_label="Sex")
            with c2:
                plot_distribution_percent(df_f, "age_group", "Sample profile: Age group distribution", x_label="Age group")

        if "social_class" in df_f.columns:
            plot_distribution_percent(df_f, "social_class", "Sample profile: Social class distribution", x_label="Social class")
        if "income_group" in df_f.columns:
            plot_distribution_percent(df_f, "income_group", "Sample profile: Income group distribution", x_label="Income group")

        with st.expander("Data preview (first 50 rows)"):
            st.dataframe(df_f.head(50), use_container_width=True)

    with tabs[1]:
        pgsi_col = resolve_col(df_f, ["pgsi_label"])
        prob_col = resolve_col(df_f, ["problem_gambling"])

        if pgsi_col is None:
            st.error(missing_cols_message(["pgsi_label"], file_key="hse_analysis_ready"))
        else:
            plot_distribution_percent(df_f, pgsi_col, "PGSI distribution (HSE 2018)", x_label="PGSI category")

        if prob_col is None:
            st.error(missing_cols_message(["problem_gambling"], file_key="hse_analysis_ready"))
        else:
            if "sex" in df_f.columns:
                plot_binary_rate_by_group(df_f, prob_col, ["sex"], "Problem gambling by sex (HSE 2018)", "Problem gambling (%)")
            if "age_group" in df_f.columns:
                plot_binary_rate_by_group(
                    df_f, prob_col, ["age_group"], "Problem gambling by age group (HSE 2018)", "Problem gambling (%)"
                )
            if "sex" in df_f.columns and "age_group" in df_f.columns:
                plot_binary_rate_by_group(
                    df_f,
                    prob_col,
                    ["age_group", "sex"],
                    "Problem gambling by age group × sex (HSE 2018)",
                    "Problem gambling (%)",
                )

    with tabs[2]:
        ghq = resolve_col(df_f, ["ghq12_score"])
        pgsi = resolve_col(df_f, ["pgsi_label"])
        distress = resolve_col(df_f, ["ghq12_distress"])

        missing = []
        if ghq is None:
            missing.append("ghq12_score")
        if pgsi is None:
            missing.append("pgsi_label")

        if missing:
            st.error(missing_cols_message(missing, file_key="hse_analysis_ready"))
        else:
            plot_mean_by_group(
                df_f,
                value_col=ghq,
                group_cols=[pgsi],
                title="Mean GHQ-12 score by PGSI category (HSE 2018)",
                y_label="GHQ-12 mean score",
            )

        if distress is None:
            st.error(missing_cols_message(["ghq12_distress"], file_key="hse_analysis_ready (derived)"))
        else:
            if "sex" in df_f.columns and "age_group" in df_f.columns:
                plot_binary_rate_by_group(
                    df_f,
                    outcome_col=distress,
                    group_cols=["age_group", "sex"],
                    title="GHQ-12 distress (%) by age group × sex (HSE 2018)",
                    y_label="Distressed (%)",
                )
            else:
                st.info("Cannot plot GHQ-12 distress by age × sex (requires sex and age_group).")


def page_gsgb(paths: Dict[str, str]) -> None:
    df_a8, err_a8 = load_gsgb_a8(paths["gsgb_a8"])
    if df_a8 is None:
        st.error(err_a8)
        return

    df_a15, err_a15 = load_gsgb_a15(paths["gsgb_a15"])
    if df_a15 is None:
        st.error(err_a15)
        return

    st.subheader("GSGB 2023")
    st.caption("Summary-table outputs (A.8 participation and A.15 feelings towards gambling).")

    reset_filters_button()

    # Filters (A8)
    spec = FilterSpec()
    if "sex" in df_a8.columns:
        sex_vals = unique_sorted_str(df_a8, "sex")
        spec.sex = st.sidebar.multiselect("Sex", options=sex_vals, default=sex_vals, key="flt_gsgb_sex")
    if "age_group" in df_a8.columns:
        age_vals = unique_sorted_str(df_a8, "age_group")
        spec.age_group = st.sidebar.multiselect("Age group", options=age_vals, default=age_vals, key="flt_gsgb_age")

    df_a8_f = apply_filters(df_a8, spec)

    tabs = st.tabs(["Participation", "Attitudes"])

    with tabs[0]:
        # Required structural fields from your A8 processing
        req = ["group_type", "sex", "age_group"]
        miss = [c for c in req if c not in df_a8_f.columns]
        if miss:
            st.error(missing_cols_message(miss, file_key="gsgb_a8"))
            st.stop()

        # Let the app robustly select the participation measure
        preferred = ["pct_all", "pct_no_lottery", "pct_any_gambling", "pct"]
        pct_candidates = [c for c in preferred if c in df_a8_f.columns]
        if not pct_candidates:
            pct_candidates = list_like_pct_columns(df_a8_f)

        if not pct_candidates:
            st.error(
                missing_cols_message(
                    ["(a participation percentage column, e.g., pct_all)"], file_key="gsgb_a8"
                )
            )
            st.stop()

        pct_col = st.selectbox("Participation measure column", options=pct_candidates, index=0)

        # By age group: group_type == 'age'
        by_age = df_a8_f[df_a8_f["group_type"].astype(str) == "age"].copy()
        if by_age.empty:
            st.info("No age-group rows available after filtering (expected group_type == 'age').")
        else:
            by_age["Participation (%)"] = to_percent_series(by_age[pct_col])
            fig = px.bar(
                by_age,
                x="age_group",
                y="Participation (%)",
                title="Any gambling in the past 4 weeks by age group (GSGB 2023)",
                labels={"age_group": "Age group", "Participation (%)": "Participation (%)"},
            )
            st.plotly_chart(fig, use_container_width=True)

        # By sex: group_type == 'sex'
        by_sex = df_a8_f[df_a8_f["group_type"].astype(str) == "sex"].copy()
        if by_sex.empty:
            st.info("No sex rows available after filtering (expected group_type == 'sex').")
        else:
            by_sex["Participation (%)"] = to_percent_series(by_sex[pct_col])
            fig = px.bar(
                by_sex,
                x="sex",
                y="Participation (%)",
                title="Any gambling in the past 4 weeks by sex (GSGB 2023)",
                labels={"sex": "Sex", "Participation (%)": "Participation (%)"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("A.8 participation preview (first 50 rows)"):
            st.dataframe(df_a8_f.head(50), use_container_width=True)

    with tabs[1]:
        req = ["feeling_score", "feeling_label", "pct_all_gamblers", "pct_excl_lottery_only"]
        miss = [c for c in req if c not in df_a15.columns]
        if miss:
            st.error(missing_cols_message(miss, file_key="gsgb_a15"))
            st.stop()

        metric = st.radio(
            "Metric",
            options=["All gamblers", "Excluding lottery-draw-only players"],
            horizontal=True,
        )
        ycol = "pct_all_gamblers" if metric == "All gamblers" else "pct_excl_lottery_only"

        d = df_a15.copy()
        d["feeling_score"] = ensure_numeric(d["feeling_score"])
        d = d.sort_values("feeling_score", ascending=True)
        d["Proportion (%)"] = to_percent_series(d[ycol])

        fig = px.bar(
            d,
            x="feeling_label",
            y="Proportion (%)",
            title=f"Feelings towards gambling (GSGB 2023) — {metric}",
            labels={"feeling_label": "Feeling label", "Proportion (%)": "Proportion (%)"},
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Optional: Reasons for gambling (A.14)"):
            df_a14, err_a14 = load_gsgb_a14(paths["gsgb_a14"])
            if df_a14 is None:
                st.info("A.14 reasons file not found / not readable (optional panel).")
                st.caption(err_a14 or "")
                st.stop()

            # Robust column detection
            reason_col = resolve_col(df_a14, ["reason"])
            male_col = resolve_col(df_a14, ["pct_male"])
            female_col = resolve_col(df_a14, ["pct_female"])

            if reason_col is None or male_col is None or female_col is None:
                st.info("Could not detect (reason, pct_male, pct_female). Showing raw preview.")
                st.dataframe(df_a14.head(50), use_container_width=True)
                st.stop()

            top_n = st.slider("Number of reasons to display", min_value=5, max_value=25, value=12, step=1)
            dd = df_a14[[reason_col, male_col, female_col]].copy()
            dd.rename(columns={reason_col: "Reason", male_col: "Male", female_col: "Female"}, inplace=True)
            dd["Male"] = to_percent_series(dd["Male"])
            dd["Female"] = to_percent_series(dd["Female"])
            dd["Average (%)"] = dd[["Male", "Female"]].mean(axis=1, skipna=True)
            dd = dd.sort_values("Average (%)", ascending=False).head(top_n)

            long = dd.melt(id_vars=["Reason"], value_vars=["Male", "Female"], var_name="Sex", value_name="Proportion (%)")
            fig = px.bar(
                long,
                x="Reason",
                y="Proportion (%)",
                color="Sex",
                barmode="group",
                title="Reasons for gambling by sex (GSGB 2023; A.14)",
                labels={"Reason": "Reason", "Proportion (%)": "Proportion (%)", "Sex": "Sex"},
            )
            fig.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)


def page_ukhls(paths: Dict[str, str]) -> None:
    df, err = load_ukhls(paths["ukhls_analysis_ready"])
    if df is None:
        st.error(err)
        return

    st.subheader("UKHLS (Understanding Society)")
    st.caption("K+L+N analysis-ready panel (common pidp).")

    reset_filters_button()

    # --------------------
    # Sidebar filters
    # --------------------
    spec = FilterSpec()

    sex_col_raw = resolve_col(df, ["sex"])
    if sex_col_raw is not None:
        sex_vals = unique_sorted_str(df, sex_col_raw)
        spec.sex = st.sidebar.multiselect("Sex", options=sex_vals, default=sex_vals, key="flt_ukhls_sex")

    if "country" in df.columns:
        c_vals = unique_sorted_str(df, "country")
        spec.country = st.sidebar.multiselect("Country", options=c_vals, default=c_vals, key="flt_ukhls_country")

    if "region" in df.columns:
        r_vals = unique_sorted_str(df, "region")
        spec.region = st.sidebar.multiselect("Region", options=r_vals, default=r_vals, key="flt_ukhls_region")

    wave_col_raw = resolve_col(df, ["survey_wave", "wave_number"])
    if wave_col_raw is not None:
        w_vals = unique_sorted_str(df, wave_col_raw)
        spec.wave = st.sidebar.multiselect("Wave", options=w_vals, default=w_vals, key="flt_ukhls_wave")

    df_f = apply_filters(df, spec)

    # --------------------
    # Required columns
    # --------------------
    mh_col = resolve_col(df_f, ["mental_health_score"])
    anx_col = resolve_col(df_f, ["anxiety_level"])
    sex_col = resolve_col(df_f, ["sex"])
    has_country = "country" in df_f.columns
    has_region = "region" in df_f.columns

    if mh_col is None or sex_col is None or (not has_country and not has_region):
        miss = []
        if mh_col is None:
            miss.append("mental_health_score")
        if sex_col is None:
            miss.append("sex")
        if not has_country and not has_region:
            miss.append("country/region")
        st.error(missing_cols_message(miss, file_key="ukhls_analysis_ready"))
        return

    # --------------------
    # Main controls (geo)
    # --------------------
    geo_options = []
    if has_country:
        geo_options.append("Country")
    if has_region:
        geo_options.append("Region")

    geo_choice = st.radio("Geography field", options=geo_options, horizontal=True, key="ukhls_geo_choice")
    geo_col = "country" if geo_choice == "Country" else "region"

    # --------------------
    # Sidebar weights (ONLY ONE checkbox)
    # --------------------
    weight_candidates = ["individual_crosssectional_weight", "individual_longitudinal_weight"]
    available_weights = [c for c in weight_candidates if c in df_f.columns]

    use_w = False
    w_col = None

    if available_weights:
        use_w = st.sidebar.checkbox("Use weights for means", value=False, key="flt_ukhls_use_w")

        if use_w:
            w_col = st.sidebar.selectbox(
                "Weight column",
                options=available_weights,
                index=0,
                key="flt_ukhls_weight_col",
            )

            ww = ensure_numeric(df_f[w_col])
            bad = int((ww.isna() | (ww <= 0)).sum())
            if bad > 0:
                st.warning(
                    f"Some rows have missing or non-positive weights in '{w_col}' "
                    f"({bad} rows in the filtered data). Weighted means will ignore these."
                )
    else:
        st.sidebar.info("No weight columns found in this UKHLS file.")

    # --------------------
    # Plots
    # --------------------
    plot_mean_by_group(
        df_f,
        value_col=mh_col,
        group_cols=[geo_col, sex_col],
        title=f"Mean mental health score by {geo_choice.lower()} × sex (UKHLS)",
        y_label="Mean score",
        weight_col=w_col if use_w else None,
    )

    if anx_col is not None:
        st.divider()
        plot_mean_by_group(
            df_f,
            value_col=anx_col,
            group_cols=[geo_col, sex_col],
            title=f"Mean anxiety level by {geo_choice.lower()} × sex (UKHLS)",
            y_label="Mean anxiety level",
            weight_col=w_col if use_w else None,
        )

    with st.expander("Data preview (first 50 rows)"):
        st.dataframe(df_f.head(50), use_container_width=True)

def main() -> None:
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    paths = sidebar_paths()

    st.sidebar.header("Controls")
    dataset = st.sidebar.selectbox(
        "Dataset",
    options=["HSE 2018", "GSGB 2023", "UKHLS (Understanding Society)", "Harmonised (HSE + UKHLS)"],
    index=0,
    )

    st.sidebar.divider()
    if st.sidebar.checkbox("Show quick diagnostics", value=False):
        st.sidebar.caption("Loaded columns can help you spot naming mismatches.")
        if dataset == "HSE 2018":
            df, _ = load_hse(paths["hse_analysis_ready"])
            if df is not None:
                st.sidebar.write("HSE columns:", list(df.columns))
        elif dataset == "GSGB 2023":
            df, _ = load_gsgb_a8(paths["gsgb_a8"])
            if df is not None:
                st.sidebar.write("GSGB A8 columns:", list(df.columns))
            df2, _ = load_gsgb_a15(paths["gsgb_a15"])
            if df2 is not None:
                st.sidebar.write("GSGB A15 columns:", list(df2.columns))
        else:
            df, _ = load_ukhls(paths["ukhls_analysis_ready"])
            if df is not None:
                st.sidebar.write("UKHLS columns:", list(df.columns))

    if dataset == "HSE 2018":
        page_hse(paths)
    elif dataset == "GSGB 2023":
        page_gsgb(paths)
    elif dataset == "Harmonised (HSE + UKHLS)":
        page_harmonised(paths)

    else:
        page_ukhls(paths)


if __name__ == "__main__":
    main()
