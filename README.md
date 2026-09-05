# Congress AI Prevalence — Methodology & Replication

**This repository is a technical replication package for the article "AI Use on Capitol Hill."** It contains the exact data, scripts, and methodology documentation needed to verify every statistical claim in the piece.

## Two Tiers of Replicability

### Tier 1: Instant Replication (you are here)

```bash
pip install -r requirements.txt
python replicate.py
```

Reproduces every article table and figure from pre-scored data. **<10 seconds, $0, no API keys.**

### Tier 2: Full Upstream Scoring (from raw XML)

See `pipeline/README.md` for the complete end-to-end workflow that regenerates the scored
analytical CSVs from scratch — GPO bulk XML → Modal GPU inference → Pangram API audit →
calibration → replication. Requires HuggingFace access, a Modal account, and a Pangram
API key. Total cloud cost: ~$5.30.

```
pipeline/
├── fetch_bills.py           # Download GPO bulk XML
├── extract_blocks.py        # Parse XML into preamble/findings/control blocks
├── make_payload.py          # Hash + chunk for GPU inference
├── score_modal.py           # Modal L4 GPU inference app
├── score_editlens_reference.py  # Local CPU/CUDA fallback
├── score_pangram_api.py     # Pangram Commercial API audit
├── assemble_dataset.py      # Scored chunks → analytical CSVs
├── requirements-pipeline.txt# Heavy dependencies
└── README.md                # Step-by-step full guide
```

## Claim-to-Code Mapping

| Claim in draft | Script / Data | Variable |
|---|---|---|
| 6.4% of bills in late 2025/2026 likely AI | `scripts/01_replicate_prevalence.py` | `pct_ai` for `2026Q2` |
| ~15% of Extensions of Remarks likely AI | `scripts/01_replicate_prevalence.py` → `data/crec_prevalence_calibrated.csv` | `point` for latest quarter |
| False positive rate under 0.1% pre-ChatGPT | `data/joint_calibration_tables_corrected.csv` (bin 1: 0 hits in 30 pre-2023 bills) | `empirical_rate` = 0.0 |
| AI adoption is bipartisan | `scripts/01_replicate_prevalence.py` | `dem_pct_ai` vs `rep_pct_ai` (6.14% vs 6.75%) |
| Institutional capacity drives AI use | `scripts/02_replicate_regressions.py` | `is_house` coefficient: +2.03 pp |
| Finance AI is driven by committee pipeline | `scripts/03_replicate_mechanisms.py` | Finance +11% AI rate; committee interaction |
| Labor AI is zero | `scripts/03_replicate_mechanisms.py` | Labor 0% AI, Pangram-confirmed 18/18 human |

## Repository Structure

```
├── replicate.py               # Master orchestrator — runs everything
├── requirements.txt           # Lightweight: pandas, numpy, matplotlib, scipy
├── README.md                  # This file
├── METHODOLOGY.md             # Deep technical whitepaper
├── AGENTS.md                  # Instructions for AI agents
├── llms.txt                   # Machine-readable project index
│
├── data/                      # Analytical datasets (see data/README.md for schema)
│   ├── federal_content_bills_119.csv       # Primary 119th Congress sample (N=2,994)
│   ├── federal_content_bills_extended.csv  # Pooled 118th+119th panel (N=6,414)
│   ├── joint_calibration_tables_corrected.csv  # Isotonic calibration lookup
│   ├── quarterly_prevalence_with_ci.csv    # Pre-computed quarterly bills + bootstrap CIs
│   ├── quarterly_control_baseline.csv     # Statutory text control series
│   ├── crec_prevalence_calibrated.csv     # CREC Extensions of Remarks series
│   ├── pangram_outlier_audit.csv          # 47-bill Pangram 3.3.2 audit
│   ├── committee_membership.csv           # Member-to-committee mapping (3,896 rows)
│   └── replicated_regression_results.csv  # Stored 6-predictor LPM coefficients
│
├── scripts/                   # Self-contained replication scripts
│   ├── 01_replicate_prevalence.py
│   ├── 02_replicate_regressions.py
│   ├── 03_replicate_mechanisms.py
│   └── 04_generate_figures.py
│
├── figures/                   # Generated Effort News-style charts (dark theme)
│   ├── fig1_prevalence_quarterly.png
│   ├── fig2_crec_extensions.png
│   └── fig3_predictor_forest.png
│
└── pipeline/                  # Reference: how raw text was scored (Tier 2 transparency)
    ├── README.md
    └── score_editlens_reference.py
```

## Data Sources

All raw data is from free, public U.S. government sources:

- **BILLS (govinfo):** Full bill XML for 116th–119th Congress (2019–2026)
- **BILLSTATUS (govinfo):** Enactment status, cosponsorship, policy area per bill
- **CREC (govinfo):** Congressional Record Extensions of Remarks (floor insertions)
- **Detector:** [Pangram EditLens 3B (Llama-3.2-3B)](https://huggingface.co/pangram/editlens_Llama-3.2-3B) (paper: [arXiv 2510.03154](https://arxiv.org/abs/2510.03154))

**Author:** Parth Goyal — [parthsdatastack](https://parthgoyal.uk) / [@parthsdatastack](https://substack.com/@parthsdatastack)

## License

All data is U.S. government public domain. Code is provided for replication and transparency.