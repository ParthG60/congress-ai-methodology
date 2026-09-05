#!/usr/bin/env python3
"""Download GPO bulkdata BILLS + BILLSTATUS zips for 116th–119th Congress.

Both collections are free and unauthenticated.

    BILLS      /{congress}/{session}/{type}/BILLS-{congress}-{session}-{type}.zip
    BILLSTATUS /{congress}/{type}/BILLSTATUS-{congress}-{type}.zip      <- no session

BILLS carries full bill XML (one per bill VERSION).
BILLSTATUS carries one XML per BILL with sponsors, cosponsors, actions, enactment.

Sizes (measured 2026-08-17): BILLS 116-119 = ~615 MB zip; BILLSTATUS = ~209 MB zip.
~825 MB total zip, ~1.6 GB unzipped.

Resumable at zip granularity: any zip whose extraction directory already holds XML is skipped.
Downloads land in a .part file and rename on success.

Usage:
    python pipeline/fetch_bills.py                          # everything, 116-119
    python pipeline/fetch_bills.py --congress 119 --types hres   # smallest slice, ~6 MB
    python pipeline/fetch_bills.py --collection BILLSTATUS       # metadata only
"""
import argparse
import pathlib
import shutil
import sys
import time
import zipfile

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"

BASE = "https://www.govinfo.gov/bulkdata"
CONGRESSES = [116, 117, 118, 119]
SESSIONS = [1, 2]
TYPES = ["hr", "s", "hres", "sres", "hjres", "sjres", "hconres", "sconres"]
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/126 Safari/537.36"}

S = requests.Session()
S.headers.update(UA)


def targets(collection, congresses, types):
    """Yield (url, extract_dir) pairs. BILLSTATUS has no session level."""
    for c in congresses:
        for t in types:
            if collection == "BILLS":
                for s in SESSIONS:
                    yield (f"{BASE}/BILLS/{c}/{s}/{t}/BILLS-{c}-{s}-{t}.zip",
                           RAW / "bills" / str(c) / str(s) / t)
            else:
                yield (f"{BASE}/BILLSTATUS/{c}/{t}/BILLSTATUS-{c}-{t}.zip",
                       RAW / "status" / str(c) / t)


def fetch_one(url, dest):
    """Download + extract one zip. Returns (n_files, mb) or None if skipped/absent."""
    if dest.exists() and any(dest.glob("*.xml")):
        return None
    dest.mkdir(parents=True, exist_ok=True)
    zpath = dest / "_dl.zip"
    part = dest / "_dl.zip.part"

    with S.get(url, stream=True, timeout=120) as r:
        if r.status_code == 404:
            return None
        r.raise_for_status()
        n_bytes = int(r.headers.get("Content-Length", 0))
        with open(part, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
    # Write succeeded → rename
    part.rename(zpath)

    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)
    n_xml = len(list(dest.glob("*.xml")))
    zpath.unlink()
    return n_xml, n_bytes / 1_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--congress", nargs="*", type=int, default=CONGRESSES)
    ap.add_argument("--types", nargs="*", default=TYPES)
    ap.add_argument("--collection", default="BILLS", choices=["BILLS", "BILLSTATUS"])
    a = ap.parse_args()

    items = list(targets(a.collection, a.congress, a.types))
    total_notes = 0
    total_mb = 0.0

    for i, (url, dest) in enumerate(items):
        label = f"  [{i+1}/{len(items)}] {dest.parent.name}/{dest.name}"
        print(label, end=" ", flush=True)
        t0 = time.time()
        result = fetch_one(url, dest)
        if result is None:
            print("(cached)")
        else:
            n, mb = result
            total_notes += n
            total_mb += mb
            print(f"{n} files, {mb:.1f} MB ({time.time()-t0:.0f}s)")
    print(f"\nDone. {total_notes} XML files, {total_mb:.0f} MB total.")


if __name__ == "__main__":
    main()