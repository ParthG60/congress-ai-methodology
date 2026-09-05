#!/usr/bin/env python3
"""Assemble scored chunks into bill-level analytical datasets.

Pipeline: scored chunks (Modal/Pangram output) → per-block scores → bill-level aggregation.

Step 1: Fan chunk scores back to blocks via the text map CSV (from make_payload.py).
Step 2: Aggregate per bill: each bill gets its max block score, block type, action date, etc.
Step 3: Validate against the published analytical CSVs in data/.

This script does NOT regenerate bills.csv or legislators.csv (those come from BILLSTATUS
extraction, documented in the pipeline README). Instead, it shows the core fan-out + aggregation
logic and validates correctness.

Usage:
    python pipeline/assemble_dataset.py
"""
import csv
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
REPO_DATA = HERE.parent / "data"

SCORES = DATA / "gpu" / "out" / "scores_gpu_3b.jsonl"   # Modal output
TEXT_MAP = DATA / "block_text_map_all-w700.csv"           # make_payload.py output
BILLS_119 = REPO_DATA / "federal_content_bills_119.csv"   # reference for validation
BILLS_EXT = REPO_DATA / "federal_content_bills_extended.csv"


def fan_out_scores(scores_path, map_path):
    """Map each scored chunk back to every original block sharing that text hash."""
    # Load text map
    tmap = pd.read_csv(map_path, dtype=str)

    # Load chunk scores
    chunk_rows = []
    with open(scores_path, encoding="utf-8") as f:
        for line in f:
            chunk_rows.append(json.loads(line))
    scores_df = pd.DataFrame(chunk_rows)

    # LEFT join: preserve ALL text-map rows, report unmatched.
    # Audit: inner join silently drops blocks whose chunk was never scored
    # (API failure, truncation, resume gap), biasing upward.
    merged = tmap.merge(
        scores_df,
        left_on=["sha", "chunk_ix"],
        right_on=["text_sha", "chunk_ix"],
        how="left",
    )
    # Drop raw text field if present
    if "text" in merged.columns:
        merged.drop(columns=["text"], inplace=True)

    n_unscored = merged["score"].isna().sum()
    n_total = len(merged)
    if n_unscored:
        print(f"  WARNING: {n_unscored:,} / {n_total:,} blocks ({n_unscored/n_total*100:.1f}%)"
              f" have no score (dropped or unscored chunks).")
        # Drop unscored rows so downstream aggregation doesn't produce NaN scores
        merged = merged.dropna(subset=["score"])

    # Audit: verify dedup cardinality — each (sha, chunk_ix) should map to 1 score
    card = tmap.groupby(["sha", "chunk_ix"], observed=True).size().max()
    print(f"  Map cardinality (sha, chunk_ix): max {card} (1.0 = perfect dedup)")

    # Sanity: no zero-score chunks survived scoring
    n_zero = (merged["score"] == 0.0).sum()
    if n_zero:
        print(f"  NOTE: {n_zero:,} chunks with score=0.0 (short blocks, or lossy tokenization)")

    # Score is per chunk. Blocks with multiple chunks get their max score.
    block_scores = (
        merged.groupby(["package_id", "congress", "bill_type", "bill_num",
                        "block_type", "action_date"], observed=True)
        .agg(
            score_max=("score", "max"),
            score_mean=("score", "mean"),
            n_chunks=("score", "count"),
        )
        .reset_index()
    )

    print(f"Chunk scores:  {len(scores_df):,}")
    print(f"Text map:      {len(tmap):,}")
    print(f"Merged:        {len(merged):,} (chunk-to-block mappings after dropna)")
    print(f"Block scores:  {len(block_scores):,}")
    return block_scores


def aggregate_to_bills(block_scores):
    """Aggregate per package_id: use the highest-scoring block for each bill."""
    # One row per bill: highest-scoring block's info
    bill = (
        block_scores.sort_values("score_max", ascending=False)
        .groupby("package_id", as_index=False)
        .first()
        .reset_index(drop=True)
    )
    print(f"Bill-level rows: {len(bill):,}")
    return bill


def validate(bill_level, reference_path, label=""):
    """Compare bill-level scores against the published analytical CSV."""
    ref = pd.read_csv(reference_path)
    # Both should have package_id and score_max
    merged = bill_level.merge(
        ref[["package_id", "score_max"]],
        on="package_id",
        suffixes=("_rebuilt", "_ref"),
    )
    if len(merged) == 0:
        print(f"  [{label}] WARNING: Zero package_id matches with reference data.")
        return

    merged["diff"] = (merged["score_max_rebuilt"] - merged["score_max_ref"]).abs()
    mean_diff = merged["diff"].mean()
    pct_exact = (merged["diff"] < 1e-6).mean() * 100

    print(f"  [{label}] Matched {len(merged):,} / {len(bill_level):,} bills with reference")
    print(f"         Mean absolute score diff: {mean_diff:.5f}")
    print(f"         Exact matches (diff=0):   {pct_exact:.1f}%")
    print(f"         Pass: {mean_diff < 0.01 and pct_exact > 90}")

    # Audit: check that bill-level max scores come from preamble/findings, not control
    bt_dist = bill_level["block_type"].value_counts(normalize=True)
    print(f"         Block-type distribution at bill level:")
    for bt, pct in bt_dist.items():
        print(f"           {bt}: {pct*100:.1f}%")
    control_share = bt_dist.get("control", 0)
    if control_share > 0.1:
        print(f"         NOTE: {control_share*100:.1f}% of bills max-scored on control "
              f"blocks (statutory boilerplate rather than staff-written text)")


def main():
    print("=" * 60)
    print("  ASSEMBLY PIPELINE: Scored Chunks → Bill-Level Dataset")
    print("=" * 60)

    if not SCORES.exists():
        print(f"\nNo scores file found at {SCORES}")
        print("Run the Modal scoring pipeline first.")
        print("Or copy existing scores from a completed run:")
        print("  modal volume get editlens /out/scores_gpu_3b.jsonl",
              SCORES)
        sys.exit(1)

    if not TEXT_MAP.exists():
        print(f"\nNo text map found at {TEXT_MAP}")
        print("Run make_payload.py first to generate it.")
        sys.exit(1)

    block_scores = fan_out_scores(SCORES, TEXT_MAP)
    bill_level = aggregate_to_bills(block_scores)

    print(f"\n--- Validation against published analytical CSVs ---")
    if BILLS_119.exists():
        validate(bill_level, BILLS_119, label="119")
    if BILLS_EXT.exists():
        validate(bill_level, BILLS_EXT, label="extended")

    print(f"\n--- Bill-level summary (first 5) ---")
    for _, row in bill_level.head(5).iterrows():
        print(f"  {row['package_id']:45s}  score={row['score_max']:.4f}"
              f"  block={row['block_type']}  date={row['action_date']}")


if __name__ == "__main__":
    main()