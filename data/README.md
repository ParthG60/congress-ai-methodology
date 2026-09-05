# Data Dictionary

## federal_content_bills_119.csv (N=2,994)
119th Congress (2025-2026) bills with staff-written political-language blocks.
One row per bill.

| Column | Type | Description |
|---|---|---|
| `package_id` | str | Unique govinfo identifier |
| `congress` | int | 119 |
| `bill_type` | str | hr/s/hres/sres/hjres/sjres/hconres/sconres |
| `bill_num` | int | Bill number |
| `score_max` | float | Maximum EditLens score across bill blocks |
| `is_likely_ai` | int | Binary: score_max >= 0.35 (lenient threshold) |
| `ai_c35/ai_c50/ai_c65` | int | Binary at each threshold (computed by scripts) |
| `p_ai` | float | Calibrated P(AI) via isotonic lookup (computed by scripts) |
| `sponsor_party` | str | D / R / I |
| `is_house` | int | 1 = House, 0 = Senate |
| `is_solo_sponsor` | int | 1 = no cosponsors |
| `member_age` | float | Sponsor age at introduction |
| `topic_*` | int | CRS policy area dummies (32 columns) |
| `n_cosponsors` | int | Total cosponsors |
| `ln_words` | float | Log word count |

## federal_content_bills_extended.csv (N=6,414)
Pooled 118th (2023-2024) + 119th (2025-2026) Congress. Additional columns:
- `is_119`: dummy for 119th Congress
- `month`: calendar month of introduction
- `is_first_appearance`: 1 if block text first appeared in this congress (no recycling)

## joint_calibration_tables_corrected.csv (25 rows)
Isotonic calibration bins for three corpora (federal_bills, floor_speeches, state_bills).

| Column | Type | Description |
|---|---|---|
| `corpus` | str | federal_bills / floor_speeches / state_bills |
| `score_bin` | str | EditLens score range (e.g. "[0.00, 0.20)") |
| `n_sampled` | int | Human-reviewed samples in bin |
| `pangram_ai_hits` | int | Confirmed AI by Pangram 3.3.2 |
| `empirical_rate` | float | pangram_ai_hits / n_sampled |
| `ci_95_low/high` | float | 95% bootstrap CI |

## quarterly_prevalence_with_ci.csv (14 rows)
Pre-computed quarterly prevalence with bootstrap CIs for 2023Q1-2026Q2.

## quarterly_control_baseline.csv (14 rows)
Quarterly prevalence for statutory text (control blocks) only.

## crec_prevalence_calibrated.csv (30 rows)
Quarterly CREC Extensions of Remarks calibrated prevalence with CIs (2019Q1-2026Q2).

| Column | Type | Description |
|---|---|---|
| `quarter` | str | e.g. "2023Q1" |
| `n_granules` | int | Number of CREC granules scored |
| `point` | float | Calibrated P(AI) mean |
| `lo`/`hi` | float | 95% bootstrap CI |

## pangram_outlier_audit.csv (47 rows)
Results of Pangram 3.3.2 commercial detector audit on 47 flagged bills.

| Column | Type | Description |
|---|---|---|
| `bill_id` | str | Package ID |
| `sector` | str | Finance / Crime / Labor |
| `confirmed_ai` | int | Pangram verdict: 0/1 |

All data derived from GPO govinfo public domain sources.