# Gambling_MentalHealth_MSc

Overview of an MSc project exploring links between gambling behaviors and mental health using UK data sources (UKHLS, HSE, GSGB).

## Project Structure

See folders in this repo. Key paths:
- `notebooks/` — analysis in numbered order
- `scripts/` — reusable helpers and optional CLI
- `data/` — raw, interim, processed, and exports
- `output/` — auxiliary outputs (logs, converted notebooks, etc.)

## Environment

Using a local virtual environment in `.venv`.

PowerShell (Windows):
- Activate: `..\.venv\Scripts\Activate.ps1`
- Install deps: `pip install -r requirements.txt`
- Optional kernel: `python -m ipykernel install --user --name ukhls --display-name "UKHLS (.venv)"`

## Run Order

1. `notebooks/01_explore_UKHLS.ipynb`
   - Read UKHLS `.sav` files
   - Select 21 columns of interest
   - Save Parquet + CSV previews to `data/interim/`

2. `notebooks/02_harmonize_merge.ipynb`
   - Standardize and clean UKHLS subsets
   - Promote standardized fields
   - Save to `data/processed/` (standardized, model-ready, analysis-ready)

3. `notebooks/03_hse_preprocess.ipynb`
   - Read HSE SPSS
   - Map variables and derive PGSI & GHQ-12
   - Save to `data/interim/` and `data/processed/`

4. `notebooks/04_gsgb_preprocess.ipynb`
   - Read GSGB Excel tables
   - Rename variables and derive PGSI bins
   - Save to `data/interim/` and `data/processed/`

5. `notebooks/05_compare_cross_sources.ipynb`
   - Combine HSE + UKHLS + GSGB
   - EDA, regression, SHAP, fairness checks
   - Export figures/tables to `data/exports/`

## Scripts

- `scripts/merge_ukhls.py` — optional CLI mirroring step 02
- `scripts/utils_io.py` — safe save/load helpers
- `scripts/utils_clean.py` — cleaning helpers and constants

## Data Locations

- `data/raw/UKHLS/` — SPSS inputs (k/l/n waves)
- `data/raw/HSE/` — SPSS inputs and README
- `data/raw/GSGB/` — Excel inputs and README

This repo ignores large, derived, and environment files in `.gitignore`.

