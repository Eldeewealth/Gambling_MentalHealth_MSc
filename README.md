# Gambling_MentalHealth_MSc

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/pandas-%3E%3D1.5-brightgreen.svg)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/jupyter-notebook-orange.svg)](https://jupyter.org/)
[![Data](https://img.shields.io/badge/data-UKHLS%2CHSE%2CGSGB-lightgrey.svg)](#data-sources)
[![Status](https://img.shields.io/badge/status-in%20progress-yellow.svg)](#project-status)

---

## Project Summary

A reproducible MSc analysis pipeline for UK survey evidence on gambling and mental health.  
This repository covers raw ingestion, variable mapping, derived outcomes, harmonisation, and analysis-ready dataset preparation for UKHLS, HSE 2018, and GSGB.

---

## Visual Overview

![Pipeline diagram](docs/images/pipeline-overview.png)

> Replace with your own image showing raw source ingestion, preprocessing, harmonisation, and analysis-ready export.

![Data model](docs/images/data-schema.png)

> Replace with your own image showing derived variables like `ghq12_score`, `pgsi_category`, and `problem_gambling`.

---

## Data Sources

- `data/raw/UKHLS/`: UKHLS waves `k`, `l`, `n`
- `data/raw/HSE/`: HSE 2018 SPSS source
- `data/raw/GSGB/`: GSGB Excel source tables

---

## What’s Included

### Notebooks
- `notebooks/01_explore_UKHLS.ipynb`
  - raw UKHLS SPSS ingestion
  - wave-specific variable dictionaries
  - Parquet export

- `notebooks/02_hse_preprocess.ipynb`
  - HSE 2018 SPSS ingestion
  - GHQ-12 and PGSI derivation
  - raw/clean/label standardisation
  - model-ready and analysis-ready dataset creation

- `notebooks/03_gsgb_preprocess.ipynb`
  - GSGB Excel ingestion
  - table cleaning, percentage conversion
  - structured GSGB model-ready outputs

- `notebooks/04_harmonisation.ipynb`
  - cross-source harmonisation
  - sex, age, income, year, and PGSI standardisation
  - harmonised dataset export

### Scripts
- `scripts/utils_clean.py`
  - column standardisation
  - missing-value normalisation
  - column promotion helper

- `scripts/utils_io.py`
  - atomic Parquet/CSV writing
  - robust file read/write

- `scripts/merge_ukhls.py`
  - combined UKHLS wave merge
  - core variable selection and CSV export

### Prototype App
- `streamlit_app_old.py`
  - dashboard prototype
  - dataset loaders, filters, charts

---

## Output Artifacts

- `data/processed/analysis-ready/`
- `data/processed/harmonised/`
- `data/processed/model-ready/`
- `data/interim/`
- `data/exports/tables/`
- `outputs/hse_models/`

---

## Environment

Install dependencies:

```powershell
pip install -r requirements.txt
```
## Core dependencies:

- pandas
- pyreadstat
- numpy
- ipykernel
- Run Order
- 01_explore_UKHLS.ipynb
- 02_hse_preprocess.ipynb
- 03_gsgb_preprocess.ipynb
- 04_harmonisation.ipynb

### Key Deliverables
- Raw ingestion of UKHLS, HSE, and GSGB
#### Derived variables:
- ghq12_score
- ghq12_label
- pgsi_score
- pgsi_category
- problem_gambling
- Harmonised demographic variables
