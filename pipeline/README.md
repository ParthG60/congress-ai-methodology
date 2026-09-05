# Scoring Pipeline Reference (Tier 2)

This directory contains reference code for the raw scoring pipeline — how bills go from GPO govinfo XML to EditLens scores.

## Pipeline Overview

1. **Fetch raw XML** → `src/fetch_bills.py` downloads GPO bulkdata zips per congress/session/type
2. **Extract blocks** → `src/extract_blocks.py` parses XML into findings, preamble, and control blocks
3. **Tokenize and score** → `src/score_editlens.py` runs EditLens inference per block
4. **Calibrate** → Joint gold sample assembled and isotonic calibration applied

## Reference: `score_editlens_reference.py`

This script replicates the exact scoring logic from the Pangram EditLens paper. It:
- Loads the model from HuggingFace
- Applies the same `clean_text` normalization (lowercase, whitespace normalization)
- Computes the same bucket-weighted score

**Prerequisites (not included in scope):**
- HuggingFace token with license acceptance for `pangram/editlens_Llama-3.2-3B`
- ~16 CPU-hours or GPU with 8+ GB VRAM for the full corpus

## Data Flow

```
GPO Bulk Data ZIPs
  ↓ fetch_bills.py
Raw XML files (per bill)
  ↓ extract_blocks.py
JSONL of text blocks (clean, deduplicated by SHA)
  ↓ score_editlens.py (GPU or CPU)
JSONL of per-block scores
  ↓ Calibration assembly (Python scripts)
Scored analytical datasets in data/
```