#!/usr/bin/env python3
"""Score bill texts using Pangram Commercial API (pangram 3.3.2) via their async bulk endpoint.

Used for the outlier audit: verifies EditLens-positive bills against a second, independent
AI-writing detector.

Cost: ~$0.04 per document. A full 47-bill audit runs ~$1.80.

Setup:
    export PANGRAM_API_KEY="..."

Dry-run (validate payload, print cost estimate):
    python pipeline/score_pangram_api.py \\
        --input pipeline/data/audit_payload.jsonl

Live run:
    python pipeline/score_pangram_api.py \\
        --input pipeline/data/audit_payload.jsonl --go

Output matches the schema in data/pangram_outlier_audit_scores.jsonl.
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = pathlib.Path(__file__).resolve().parent
API_ROOT = "https://text.external-api.pangram.com"
MODEL = "default"


def build_payload(input_path):
    """Read JSONL items from file. Each line must have {id, text}."""
    items = []
    for line in open(input_path, encoding="utf-8"):
        obj = json.loads(line)
        items.append({
            "id": obj.get("id", obj.get("package_id", f"item_{len(items)}")),
            "text": obj.get("text", ""),
            "original": obj,
        })
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL input with {id, text} per line")
    ap.add_argument("--go", action="store_true", help="Submit to Pangram API (dry-run if omitted)")
    ap.add_argument("--out", default=str(HERE / "data" / "pangram_api_results.jsonl"),
                    help="Output JSONL path")
    a = ap.parse_args()

    key = os.environ.get("PANGRAM_API_KEY")
    if not key:
        sys.exit("ERROR: PANGRAM_API_KEY environment variable not set")

    items = build_payload(a.input)
    n = len(items)
    cost = n * 0.04
    print(f"Payload: {n} items, estimated cost: ${cost:.2f} USD")
    print()
    for item in items:
        print(f"  {item['id']}: {len(item['text'].split())} words")

    if not a.go:
        print("\n[Dry Run] Pass --go to submit to Pangram Bulk API.")
        return

    # Single batch
    headers = {"x-api-key": key, "Content-Type": "application/json"}
    id_hash = hashlib.sha256(
        ",".join(item["id"] for item in items).encode("utf-8")
    ).hexdigest()[:32]
    ikey = f"audit_{id_hash}"

    payload = {
        "items": [{"id": item["id"], "text": item["text"][:20000]} for item in items],
        "model": MODEL,
    }

    print(f"\nSubmitting {n} items (Idempotency-Key: {ikey})...", flush=True)
    rq = requests.post(
        f"{API_ROOT}/bulk",
        headers={**headers, "Idempotency-Key": ikey},
        json=payload,
        timeout=120,
    )
    if rq.status_code not in (200, 202):
        sys.exit(f"Submit failed HTTP {rq.status_code}: {rq.text[:300]}")

    bid = rq.json().get("bulk_id")
    print(f"Accepted! bulk_id: {bid}. Polling...", flush=True)

    t_start = time.time()
    while True:
        st = requests.get(f"{API_ROOT}/bulk/{bid}", headers=headers, timeout=60).json()
        status = st.get("status")
        if status in ("succeeded", "failed", "partial"):
            break
        elapsed = int(time.time() - t_start)
        print(f"  Status: {status} ({elapsed}s elapsed)...", flush=True)
        time.sleep(12)

    print(f"Finished with status: {status} in {int(time.time()-t_start)}s", flush=True)

    # Fetch results
    results = []
    offset = 0
    while True:
        d = requests.get(
            f"{API_ROOT}/bulk/{bid}/results?offset={offset}&limit=1000",
            headers=headers, timeout=120,
        ).json()
        results.extend(d.get("results", []))
        remaining = d.get("remaining", 0)
        offset += len(d.get("results", []))
        if remaining <= 0:
            break

    # Merge results with original payload metadata
    item_map = {item["id"]: item["original"] for item in items}
    with open(a.out, "w", encoding="utf-8") as f:
        for res in results:
            item_id = res.get("id", "")
            orig = item_map.get(item_id, {})
            row = {
                "id": item_id,
                "category": orig.get("_category", ""),
                "editlens_score_max": orig.get("score_max", orig.get("editlens_score_max")),
                "ai_c65": orig.get("ai_c65", False),
                "topic": orig.get("_category", ""),
                "stage": res.get("stage", "STAGE_SUCCESS"),
                "error": res.get("error"),
                "prediction_short": res.get("prediction", {}).get("short", ""),
                "fraction_ai": res.get("prediction", {}).get("fraction_ai", 0),
                "fraction_ai_assisted": res.get("prediction", {}).get("fraction_ai_assisted", 0),
                "fraction_human": res.get("prediction", {}).get("fraction_human", 0),
                "sentences": res.get("prediction", {}).get("sentences", []),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(results)} results → {a.out}")


if __name__ == "__main__":
    main()