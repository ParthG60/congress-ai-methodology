# Pipeline: End-to-End Replication (Tier 2)

This directory contains everything needed to **reproduce the upstream scoring from scratch**
— from raw GPO XML to the analytical datasets.

**Two tiers of replicability:**

| Tier | What | Time | Cost | Requirements |
|---|---|---|---|---|
| 1 | `python replicate.py` | <10s | $0 | None (uses pre-scored data) |
| 2 | Full pipeline below | ~4 hours | ~$5.30 | HF token, Modal, Pangram key |

---

## Prerequisites

### 1. HuggingFace Access (gated model)

Both `meta-llama/Llama-3.2-3B` and `pangram/editlens_Llama-3.2-3B` are gated.

1. Create account at [huggingface.co](https://huggingface.co)
2. Accept license at [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)
3. Accept license at [pangram/editlens_Llama-3.2-3B](https://huggingface.co/pangram/editlens_Llama-3.2-3B)
4. Generate a [User Access Token](https://huggingface.co/settings/tokens) (read)

```bash
export HF_TOKEN="hf_..."
```

### 2. Modal Account (cloud GPU)

[modal.com](https://modal.com) — free tier includes enough credits for a full run (~$3.50).

```bash
pip install modal
modal setup
modal secret create huggingface HF_TOKEN="$HF_TOKEN"
modal volume create editlens
```

### 3. Pangram API Key (commercial audit)

Apply at [pangram.com](https://pangram.com) for an API key. ~$1.80 for the full 47-bill audit.

```bash
export PANGRAM_API_KEY="..."
```

### 4. Python environment

```bash
pip install -r pipeline/requirements-pipeline.txt
```

Expect: `torch`, `transformers`, `peft`, `modal`, `requests`, `lxml`, `pandas`, `numpy`.

---

## End-to-End Workflow

### Stage 1: Raw Data — Download GPO Bulk XML

Downloads BILLS + BILLSTATUS zips for 116th–119th Congress (~825 MB, free, unauthenticated).

```bash
# Full download (all congresses + types)
python pipeline/fetch_bills.py

# Smoke test (smallest slice: just 119th House resolutions)
python pipeline/fetch_bills.py --congress 119 --types hres

# Just the metadata (for bills.csv)
python pipeline/fetch_bills.py --collection BILLSTATUS
```

Output: `pipeline/data/raw/bills/{congress}/{session}/{type}/*.xml` (~1.6 GB unzipped)

### Stage 2: Block Segmentation — Extract preambles + findings

Parses XML into three block types: preamble (Whereas clauses), findings (political staff
sections), and control (Legislative Counsel legalese baseline).

```bash
python pipeline/extract_blocks.py
```

Output: `pipeline/data/blocks.jsonl` (one row per document × block_type, ~194 MB)

### Stage 3: GPU Payload — Hash, chunk, dedupe

Chunks blocks into ~700-word windows (1024 tokens for Llama-3.2-3B), Blake2b-hashes to
deduplicate identical text, and writes:

```bash
python pipeline/make_payload.py
```

Outputs:
- `pipeline/data/gpu/in/all.jsonl.gz` — GPU payload (~83k unique chunks, ~11 MB)
- `pipeline/data/block_text_map_all-w700.csv` — Hash→block mapping (76k rows, 6.3 MB)

The GPU payload has NO bill metadata — the cloud worker knows nothing about the project.

### Stage 4: Cloud GPU Inference — Score chunks on Modal L4

Upload payload, score on an L4 GPU (24GB Ada, fp16), pull results.

```bash
# 1. Upload payload to Modal volume
modal volume put editlens pipeline/data/gpu/in/all.jsonl.gz /in/all.jsonl.gz

# 2. Validate (24 chunks, ~$0.02, 30 seconds)
modal run pipeline/score_modal.py

# 3. Full run (detached — survives laptop sleep, ~$3.50, ~3.5 hours)
modal run --detach pipeline/score_modal.py full

# 4. Pull results
mkdir -p pipeline/data/gpu/out
modal volume get editlens /out/scores_gpu_3b.jsonl pipeline/data/gpu/out/scores_gpu_3b.jsonl
```

Output: `pipeline/data/gpu/out/scores_gpu_3b.jsonl` (per-chunk scores with SHA keys)

**Resume:** The Modal script tracks (sha, chunk_ix) of already-scored chunks. A killed run
can be restarted — it will skip completed chunks automatically.

### Stage 5: Local Reference Scoring (CPU/CUDA fallback)

For validating single bills on local hardware (slower, no cloud cost):

```bash
# Single text
python pipeline/score_editlens_reference.py \
    --text "Whereas the Congress finds that artificial intelligence..."

# Batch from JSONL
python pipeline/score_editlens_reference.py \
    --input data/sample_bills.jsonl \
    --device cpu
```

**Memory:** ~6 GB RAM on CUDA (fp16), ~16 GB on CPU (fp32). CPU speed: ~8 hours for 1k chunks.

### Stage 6: Pangram Commercial API Audit

The outlier audit uses Pangram's independent AI detector (model 3.3.2) to verify a
stratified sample of 47 bills. Output matches `data/pangram_outlier_audit_scores.jsonl`.

```bash
# First, prepare the audit payload from the scored blocks
# (See scripts/03_replicate_mechanisms.py for selection logic)

# Dry-run (prints cost estimate)
python pipeline/score_pangram_api.py \
    --input data/sample_bills.jsonl

# Live run (~$1.80 for 47 bills)
python pipeline/score_pangram_api.py \
    --input data/sample_bills.jsonl --go
```

### Stage 7: Assemble Analytical Datasets

Fan chunk scores back to blocks, aggregate to bills, validate against reference CSVs.

```bash
python pipeline/assemble_dataset.py
```

This step:
1. Merges scored chunks with the text map → per-block scores
2. Aggregates to bill level (highest-scoring block per `package_id`)
3. Validates against `data/federal_content_bills_119.csv` (checks score reconstruction)

### Stage 8: Calibration & Replication

The `data/` directory already contains the isotonic calibration tables and the four
replication scripts. Once scoring is complete, run:

```bash
python replicate.py
```

This produces all article tables and figures in <10 seconds.

---

## Architecture Overview

```
pipeline/
├── fetch_bills.py           # Stage 1: Download GPO XML (~825 MB)
├── extract_blocks.py        # Stage 2: Parse XML → blocks.jsonl
├── make_payload.py          # Stage 3: Hash + chunk → GPU payload
├── score_modal.py           # Stage 4: Modal L4 GPU inference
├── score_editlens_reference.py  # Stage 5: Local CPU/CUDA fallback
├── score_pangram_api.py     # Stage 6: Pangram Commercial API
├── assemble_dataset.py      # Stage 7: Chunks → analytical CSVs
│
├── requirements-pipeline.txt# Heavy dependencies (torch, modal, peft, lxml)
└── README.md                # This file
```

## Data Flow

```
GPO Bulkdata ZIPs
    │ fetch_bills.py (Stage 1)
    ▼
Raw XML files (~1.6 GB)
    │ extract_blocks.py (Stage 2)
    ▼
blocks.jsonl (~194 MB)
    │ make_payload.py (Stage 3)
    ▼
all.jsonl.gz  ──┬──→ Modal GPU (Stage 4) ──→ scores_gpu_3b.jsonl
                │
                └── block_text_map_all-w700.csv (hash → block mapping)
                                             │
                                             ▼
                                    assemble_dataset.py (Stage 7)
                                             │
                                             ▼
                          federal_content_bills_119.csv  ←─ already in data/
                          federal_content_bills_extended.csv

blocks.jsonl ──→ score_pangram_api.py (Stage 6) ──→ pangram_outlier_audit_scores.jsonl

data/federal_content_bills_119.csv
    │ replicate.py (Stage 8, Tier 1)
    ▼
Figures + article tables
```

## Resource Summary

| Stage | Script | Wall Time | Disk | Cloud Cost |
|---|---|---|---|---|
| 1 | `fetch_bills.py` | ~10 min | 1.6 GB | $0 |
| 2 | `extract_blocks.py` | ~5 min | 194 MB | $0 |
| 3 | `make_payload.py` | ~2 min | 17 MB | $0 |
| 4 | `score_modal.py` full | ~3.5 hours | 11 MB | ~$3.50 (Modal L4) |
| 5 | `score_editlens_reference.py` | variable | — | $0 (local) |
| 6 | `score_pangram_api.py` | ~15 min | 20 KB | ~$1.80 (Pangram API) |
| 7 | `assemble_dataset.py` | ~10 sec | — | $0 |
| 8 | `python replicate.py` | <10 sec | — | $0 |

**Total cloud cost:** ~$5.30 per full run.

## FAQ

**Q: Do I need to run Stages 1–7 to verify the article?**
A: No. `python replicate.py` (Tier 1) reproduces all numbers from pre-scored data
in <10 seconds with zero setup. The pipeline is only needed if you want to reproduce
the *scoring itself* — from raw XML to EditLens scores to analytical CSVs.

**Q: Can I skip the Modal GPU step and use the existing scores?**
A: Yes. `data/federal_content_bills_119.csv` contains the final scored dataset.
All replication scripts use these pre-computed files.

**Q: Can I score my own bills?**
A: Yes. Put your bill texts in a JSONL file with `{"text": "...", "id": "my_bill_1"}`
format and run `score_pangram_api.py --input my_bills.jsonl --go`.

**Q: How do I regenerate `bills.csv` (bill metadata)?**
A: Run `pipeline/fetch_bills.py --collection BILLSTATUS`, then extract the status XML
into a CSV. This step is documented but `bills.csv` itself (19 MB) is excluded from
the repo. See `scripts/01_replicate_prevalence.py` for the join keys.

**Q: The Modal run failed halfway. What now?**
A: Just restart it. The script tracks (sha, chunk_ix) of completed chunks and skips
them. This is why the payload deduplicates: identical text across bills only gets
scored once.