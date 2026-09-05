#!/usr/bin/env python3
"""Extract scoreable text blocks (preamble, findings, control) from govinfo BILLS XML.

A bill is not written by one person. Its operative provisions come from the Office of the
Legislative Counsel — nonpartisan career drafting attorneys — while its findings/preambles
are written by the member's own political staff. We separate them:

    preamble   //preamble/whereas/text     political staff, zero template  <- best target
    findings   Findings/Purpose/Sense      political staff
    control    everything else in body     Legislative Counsel legalese   <- baseline

INTRODUCED VERSIONS ONLY (ih/is). Later stages (rh, eh, enr) add committee lawyers and
repeat the same bill 3-5x, weighting long-lived bills more heavily.

Usage:
    python pipeline/extract_blocks.py                      # all downloaded congresses
    python pipeline/extract_blocks.py --congress 119       # single congress
Output: pipeline/data/blocks.jsonl  (one row per document × block_type)
"""
import argparse
import collections
import json
import pathlib
import re
import sys

from lxml import etree

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "data" / "raw" / "bills"
OUT = HERE / "data" / "blocks.jsonl"

MIN_WORDS = 50
CONTROL_CAP = 400

DC = {"dc": "http://purl.org/dc/elements/1.1/"}

FNAME = re.compile(r"^BILLS-(\d{3})([a-z]+)(\d+)(i[hs]\d*)$", re.I)

FINDINGS_RE = re.compile(
    r"^\s*(findings?|purposes?|findings? and purposes?|findings?[;,] purposes?|"
    r"purposes? and findings?|sense of (the )?congress|sense of (the )?(house|senate)|"
    r"statement of policy|policy|declaration of policy|findings? and declarations?)\s*[.:]?\s*$",
    re.I,
)

DROP_SECTION_RE = re.compile(r"^\s*(short title|table of contents)\s*[.:]?\s*$", re.I)
QUOTED_BLOCK = "quoted-block"


def iter_section_text(node):
    """Recursively extract text from section/subsection children."""
    for child in node.iter():
        if isinstance(child.tag, str) and "section" in child.tag.lower():
            yield child


def text_of(node):
    """Get all text content (tags stripped) under a node."""
    return re.sub(r"\s+", " ", (node.text or "") + "".join(
        etree.tostring(e, method="text", encoding="unicode") for e in node.iterchildren()
    )).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--congress", nargs="*", type=int, default=[116, 117, 118, 119])
    a = ap.parse_args()

    out_dir = OUT.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for congress in a.congress:
        for session in [1, 2]:
            sess_dir = RAW / str(congress) / str(session)
            if not sess_dir.exists():
                continue
            for bill_type_dir in sorted(sess_dir.iterdir()):
                for xml_path in sorted(bill_type_dir.glob("*.xml")):
                    match = FNAME.match(xml_path.stem)
                    if not match:
                        continue

                    congress_num, btype, bnum, version = match.groups()
                    congress_num = int(congress_num)
                    bnum = int(bnum)

                    try:
                        tree = etree.parse(str(xml_path))
                    except Exception:
                        continue
                    root = tree.getroot()

                    # DC metadata
                    ns = {"dc": "http://purl.org/dc/elements/1.1/"}
                    action_date = root.findtext("dc:date", "", ns)[:10]
                    package_id = f"BILLS-{congress_num}{btype}{bnum}{version}"

                    # ---- preamble ----
                    preamble_els = root.findall(".//preamble/whereas/text")
                    preamble_text = " ".join(
                        t.strip() for t in preamble_els if t.strip()
                    )
                    if len(preamble_text.split()) >= MIN_WORDS:
                        rows.append({
                            "package_id": package_id,
                            "congress": congress_num,
                            "bill_type": btype,
                            "bill_num": bnum,
                            "version": version,
                            "action_date": action_date,
                            "block_type": "preamble",
                            "text": preamble_text,
                        })

                    # ---- findings ----
                    target_texts = []
                    for sec in root.iter("{http://schemas.govinfo.gov/usc/2024}section"):
                        heading = sec.findtext("heading", "")
                        heading = re.sub(r"\s+", " ", heading).strip()
                        if not heading:
                            heading = sec.findtext("header", "")
                            heading = re.sub(r"\s+", " ", heading).strip() if heading else ""
                        if FINDINGS_RE.match(heading):
                            sec_text = text_of(sec)
                            # Remove quoted blocks if present
                            for qb in sec.iter(QUOTED_BLOCK):
                                sec_text = sec_text.replace(text_of(qb) + " ", "")
                                sec_text = sec_text.replace(text_of(qb), "")
                            if len(sec_text.split()) >= MIN_WORDS:
                                target_texts.append(sec_text)

                    if target_texts:
                        findings_text = " ".join(target_texts)
                        if len(findings_text.split()) >= MIN_WORDS:
                            rows.append({
                                "package_id": package_id,
                                "congress": congress_num,
                                "bill_type": btype,
                                "bill_num": bnum,
                                "version": version,
                                "action_date": action_date,
                                "block_type": "findings",
                                "text": findings_text,
                            })

                    # ---- control (everything else) ----
                    control_parts = []
                    for sec in root.iter("{http://schemas.govinfo.gov/usc/2024}section"):
                        heading = sec.findtext("heading", "")
                        heading = re.sub(r"\s+", " ", heading).strip() if heading else ""
                        heading2 = sec.findtext("header", "")
                        heading2 = re.sub(r"\s+", " ", heading2).strip() if heading2 else ""
                        h = heading or heading2

                        if FINDINGS_RE.match(h) or DROP_SECTION_RE.match(h):
                            continue
                        sec_text = text_of(sec)
                        # Include quoted blocks for control (legalese baseline)
                        if len(sec_text.split()) >= MIN_WORDS:
                            control_parts.append(sec_text)

                    if control_parts:
                        # Take enough from the start to fill CONTROL_CAP words
                        control_text = ""
                        for part in control_parts:
                            words_needed = CONTROL_CAP - len(control_text.split())
                            if words_needed <= 0:
                                break
                            part_words = part.split()[:words_needed]
                            control_text += " " + " ".join(part_words)
                        control_text = control_text.strip()
                        if len(control_text.split()) >= MIN_WORDS:
                            rows.append({
                                "package_id": package_id,
                                "congress": congress_num,
                                "bill_type": btype,
                                "bill_num": bnum,
                                "version": version,
                                "action_date": action_date,
                                "block_type": "control",
                                "text": control_text,
                            })

    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows):,} blocks → {OUT}")


if __name__ == "__main__":
    main()