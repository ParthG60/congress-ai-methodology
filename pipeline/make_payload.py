#!/usr/bin/env python3
"""Build GPU work payload: hashed + chunked blocks ready for Modal inference.

Reads blocks.jsonl → chunks text into ~700-word windows (1024 tokens for Llama-3.2-3B),
Blake2b-hashes deduplicated texts, and writes:
    pipeline/data/gpu/in/all.jsonl.gz           GPU payload: {sha, ix, n, text}
    pipeline/data/block_text_map_all-w700.csv   Hash → block metadata map

The text map is how chunk scores fan back out to individual bill blocks downstream.
Payload rows carry no bill metadata — the GPU side knows nothing about the project.

Usage:
    python pipeline/make_payload.py                         # whole corpus
    python pipeline/make_payload.py --block-type preamble   # preambles only (smoke test)
"""
import argparse
import csv
import gzip
import hashlib
import json
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = pathlib.Path(__file__).resolve().parent
BLOCKS = HERE / "data" / "blocks.jsonl"
OUT_DIR = HERE / "data" / "gpu" / "in"
MAP_DIR = HERE / "data"

CHUNK_WORDS = 700           # ~1024-token window for Llama-3.2-3B
MIN_CHUNK_WORDS = 50
BLOCK_ORDER = {"preamble": 0, "findings": 1, "control": 2}


def clean_text(text: str) -> str:
    """Text normalisation matching Pangram's inference pipeline."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunks(text, words_per_chunk):
    """Yield (cleaned_text, chunk_index, n_chunks) for a text."""
    cleaned = clean_text(text)
    tokens = cleaned.split()
    n = max(1, (len(tokens) + words_per_chunk - 1) // words_per_chunk)
    for i in range(n):
        chunk = " ".join(tokens[i * words_per_chunk:(i + 1) * words_per_chunk])
        n_words = len(chunk.split())
        if n_words >= MIN_CHUNK_WORDS:
            yield chunk, i, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-type", nargs="*", choices=list(BLOCK_ORDER))
    ap.add_argument("--chunk-words", type=int, default=CHUNK_WORDS,
                    help=f"Words per chunk (default {CHUNK_WORDS})")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(BLOCKS, encoding="utf-8")]
    if a.block_type:
        rows = [r for r in rows if r["block_type"] in a.block_type]

    rows.sort(key=lambda r: (BLOCK_ORDER.get(r["block_type"], 9), r["congress"],
                             r["bill_type"], r["bill_num"]))

    tag = "-".join(sorted(a.block_type)) if a.block_type else "all"
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    payload_path = out_dir / f"{tag}.jsonl.gz"
    map_path = MAP_DIR / f"block_text_map_{tag}-w{a.chunk_words}.csv"

    payload_rows = {}      # (sha, ix) → row
    text_map_rows = []     # one per original block
    seen_hashes = set()    # dedup within this run

    for r in rows:
        for chunk_text, chunk_ix, n_chunks in chunks(r["text"], a.chunk_words):
            sha = hashlib.blake2b(chunk_text.encode("utf-8"), digest_size=16).hexdigest()
            key = (sha, chunk_ix)
            if key not in seen_hashes:
                seen_hashes.add(key)
                payload_rows[key] = {"sha": sha, "ix": chunk_ix, "n": n_chunks,
                                     "text": chunk_text}
            text_map_rows.append({
                "sha": sha,
                "chunk_ix": chunk_ix,
                "n_chunks": n_chunks,
                "package_id": r["package_id"],
                "congress": r["congress"],
                "bill_type": r["bill_type"],
                "bill_num": r["bill_num"],
                "block_type": r["block_type"],
                "action_date": r.get("action_date", ""),
            })

    # Write payload (sorted for deterministic ordering)
    with gzip.open(payload_path, "wt", encoding="utf-8") as f:
        for key in sorted(payload_rows):
            f.write(json.dumps(payload_rows[key], ensure_ascii=False) + "\n")

    # Write text map
    with open(map_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sha", "chunk_ix", "n_chunks", "package_id", "congress",
            "bill_type", "bill_num", "block_type", "action_date"])
        w.writeheader()
        w.writerows(text_map_rows)

    print(f"Payload: {len(payload_rows):,} unique chunks → {payload_path}  "
          f"({payload_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Map:     {len(text_map_rows):,} rows → {map_path}")


if __name__ == "__main__":
    main()