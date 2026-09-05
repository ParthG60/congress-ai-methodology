# AGENTS.md — Instructions for AI Agents Fact-Checking This Repository

## Project Overview
This repository provides transparent, reproducible methodology for the article "AI Use on Capitol Hill." Every claim in the article can be traced to a specific data file, variable, and code line.

## Architecture
- **Data:** Clean analytical CSVs in `data/` (small enough for full-agent ingestion if needed)
- **Code:** Self-contained Python scripts in `scripts/` (no local path dependencies, no API keys)
- **Scoring Pipeline:** `pipeline/` contains the full upstream workflow (Tier 2) — from GPO XML to scored CSVs. See `pipeline/README.md`.
- **Output:** Terminal tables printed by each script; figures saved to `figures/`

## Verification Checklist

### Claim 1: "6.4% of bills likely AI-assisted"
- Data: `data/quarterly_prevalence_with_ci.csv`, row 14 (2026Q2)
- Verify: `pct_ai` column = 6.36
- Script: `scripts/01_replicate_prevalence.py` → `run_bills_prevalence()`

### Claim 2: "~15% of Extensions of Remarks"
- Data: `data/crec_prevalence_calibrated.csv`, filter quarter >= 2023Q1, last row
- Verify: `point * 100` ≈ 15
- Script: `scripts/01_replicate_prevalence.py` → `run_crec_prevalence()`

### Claim 3: "Pre-ChatGPT false positive rate < 0.1%"
- Data: `data/joint_calibration_tables_corrected.csv`, filter `corpus == "federal_bills"`
- Verify: first bin (`[0.00, 0.20)`) has `empirical_rate = 0.0`

### Claim 4: "AI use is bipartisan"
- Script: `scripts/01_replicate_prevalence.py` → latest `dem_pct_ai` vs `rep_pct_ai`
- Should show convergent rates (~6% each)

### Claim 5: "Institutional capacity drives AI use"
- Script: `scripts/02_replicate_regressions.py` → `is_house` coefficient
- Should be positive and p < 0.01

### Claim 6: "Finance AI correlates with committee membership"
- Script: `scripts/03_replicate_mechanisms.py` → Finance section
- Shows 11% prevalence and committee interaction

### Claim 7: "Labor AI is zero"
- Script: `scripts/03_replicate_mechanisms.py` → Labor section
- Shows 0 AI-positive bills, Pangram-confirmed 18/18 human

## Data Schema (for agent code-writing)
The main analysis file `data/federal_content_bills_119.csv` has 2,994 rows and ~70 columns. Key columns for analysis:
- `score_max`: raw EditLens 3B score (float, 0-1)
- `is_house`: dummy (1 or 0)
- `sponsor_party`: "D", "R", "I"
- `topic_crime_and_law_enforcement`, `topic_finance_and_financial_sector`, `topic_labor_and_employment`: topic dummies
- `member_age`: sponsor age at bill introduction

## Running Code
```python
import pandas as pd
import numpy as np

df = pd.read_csv("data/federal_content_bills_119.csv")
print(f"Sample size: {len(df):,}")

# Prevalence at calibrated threshold (P(AI) >= 0.50 = score >= 0.65)
df["ai_c65"] = (df["score_max"] >= 0.65).astype(float)
print(f"Calibrated binary prevalence: {df['ai_c65'].mean() * 100:.2f}%")
```