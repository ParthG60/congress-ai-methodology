# Methodology: Measuring AI Writing in US Legislation

## Overview

This project measures the prevalence of AI-assisted writing in U.S. federal legislation and Congressional Record floor speeches since the release of ChatGPT (November 30, 2022). The central measurement problem is that **you cannot simply run a detector on whole bills** — legislative text has a complex authorship structure that requires separating political staff writing from nonpartisan professional drafting.

## Why Naive Bill Scoring Fails

A bill introduced in Congress is not written by one person. It has three layers:

1. **Operative statutory provisions** (the "shall" language that amends the U.S. Code) → written by the **nonpartisan Office of the Legislative Counsel**, not the member or their staff. This text is formulaic, often copied from existing statute, and would trigger AI-detection false positives on structure alone.

2. **Quoted or incorporated text** → bills amend existing law, which means they contain decades-old text that could not possibly be AI-generated.

3. **Political layer** → the **Findings section** (in bills) and the **Whereas clauses / preamble** (in resolutions) are written by political staff in the member's personal office. These sections argue why the bill matters, have no required legal form, and are the cleanest read on staff writing.

**Our approach:** Score only the political-layer blocks (findings and preambles), not the statutory text. This is the sole methodological innovation that makes the measurement credible.

## Detector Choice: EditLens vs Classification

We use **Pangram EditLens (RoBERTa-large)**, a 355M-parameter model fine-tuned to regress on the *edit magnitude* between a human original and an AI-edited version (paper: [arXiv 2510.03154](https://arxiv.org/abs/2510.03154)). It is *not* a binary AI-vs-human classifier:

- Training target: cosine similarity between human draft and AI-edited version
- Output: a score in [0, 1] measuring how far text was moved from a human baseline
- Training domains: Amazon/Google reviews, Reddit Writing Prompts, FineWeb-EDU, XSum, CNN/DailyMail
- **Congressional text is far outside all training domains**

### Why not binary classifiers (GPTZero, Originality.ai, etc.)?

Binary classifiers produce a single "probability of AI" that conflates editing magnitude with generation likelihood. They are trained to detect *full generation* (GPT-4 writes a paragraph) rather than *editing* (a staffer writes a draft, then asks Claude to polish it). EditLens's edit-magnitude regression is better suited to the real-world use case: staff writing with AI assistance rather than wholesale AI generation.

## The Score Is Not a Probability

A core finding of the EditLens paper is that the raw score is not interpretable as a probability. The published threshold analysis shows:

| Threshold | Meaning on paper | Flags on pre-ChatGPT legislative text |
|---|---|---|
| **0.039** | Human vs *any* AI involvement | 89% of controls, 91% of findings, 99% of preambles — unusable |
| **0.960** | Fully AI-generated vs edited-or-human | 0 of 36,168 blocks — measured 0.00% FPR |

The low threshold is useless on legislative prose. The high threshold gives a reliable binary but is extremely conservative (flags only text that reads as obviously AI-generated to the detector).

### Our Solution: Isotonic Calibration

We bridge the gap by calibrating raw EditLens scores against **Pangram 3.3.2**, a commercial binary AI detector used for financial and legal document screening. We assembled a gold sample of 1,200 legislative blocks and labeled each with Pangram's verdict, then built an isotonic (monotonically increasing) calibration curve:

| EditLens Score Bin | N Sampled | Pangram-AI Hits | Empirical P(AI) |
|---|---|---|---|
| [0.00, 0.20) | 30 | 0 | 0.0% |
| [0.20, 0.35) | 80 | 4 | 5.0% |
| [0.35, 0.50) | 146 | 18 | 12.3% |
| [0.50, 0.65) | 77 | 23 | 29.9% |
| [0.65, 0.80) | 30 | 21 | 70.0% |
| [0.80, 1.00] | 17 | 12 | 70.6% |

The calibration is isotonic monotonic (enforced non-decreasing) and uses bootstrap CIs (B=10,000). Throughout the analysis we report **calibrated P(AI)** — the expected fraction of AI-written bills in a given bin — rather than raw scores.

**Headline binary threshold:** P(AI) >= 0.50, which corresponds to EditLens score >= 0.65. This is the threshold used for the "likely AI" classification.

## Pre-ChatGPT Negative Control

We scored all legislative blocks from **January 2019 to October 2022** — a period where LLM authorship is zero by construction. This measures the detector's **false positive rate on legislative prose**:

- **36,168 control blocks (statutory text):** 0 flagged at the 0.960 threshold → **FPR = 0.00%**
- **At our headline threshold (0.65):** FPR = **0.08%**

This confirms that the rise in AI scores after November 2022 is not a domain artefact.

## The Authorship Split

### Bills (HR, S, HJRES, SJRES)
- **Findings section** (<section> with a Findings/Purpose/Sense-of-Congress header): written by political staff. Scored.
- **Control sections** (everything else): written by Legislative Counsel. Scored as a baseline but not reported as AI prevalence.

### Resolutions (HRES, SRES, HCONRES, SCONRES)
- **Preamble** (Whereas clauses): written by political staff. Free-form, no template, cleanest signal.
- **No operative text**: resolutions don't create law, so Legislative Counsel involvement is minimal.

### What we do not score
- Operative statutory text (controlled for as a baseline)
- Short titles, bill headers, sponsor metadata
- Text recycled verbatim from pre-ChatGPT congresses (measured at 18% of post-2022 findings — removed from the primary estimate via the `is_first_appearance` flag)

## The Econometric Model

### Primary Specification

**Sample:** 119th Congress (2025–2026) content bills only — N = 2,994. We use a single congress to eliminate the calendar-time adoption confound entirely.

**Method:** Linear Probability Model (OLS on binary outcome) with standard errors clustered at the sponsor level (`sponsor_bioguide`). Cluster-robust inference accounts for the fact that each member introduces multiple bills and their AI usage is correlated within sponsor.

**Outcome:** Binary flag = 1 if P(AI) >= 0.50 (EditLens score >= 0.65 after isotonic calibration).

**Predictors (6, in 4 tiers):**

| Tier | Predictor | Coding |
|---|---|---|
| Institutional Capacity | `is_house` | 1 = House, 0 = Senate (Senate has ~4x more staff per office) |
| Coalition Scrutiny | `is_solo_sponsor` | 1 = no cosponsors at introduction |
| Sponsor Cohort | `age_per_decade_younger` | (60 - member_age) / 10, centered |
| Policy Area Poles | `topic_crime` | CRS policy area = Crime and Law Enforcement |
| | `topic_finance` | CRS policy area = Finance and Financial Sector |
| | `topic_labor` | CRS policy area = Labor and Employment |

### Four Traps Handled

1. **Right-censoring:** BILLSTATUS is live through mid-2026; post-ChatGPT bills are younger and have less time to succeed. We measure outcomes in a 12-month window from introduction and report primary results on the 119th only (not a completed congress).

2. **Cross-chamber comparability:** A resolution can never become law, so `max_stage` must be normalised within `bill_type`. We report `stage_frac` (reached stage / maximum possible stage for that type).

3. **Recycled text:** 18% of post-ChatGPT findings blocks and 16% of controls are verbatim text first introduced pre-ChatGPT. These cannot contain AI by construction. We report primary estimates on first-appearance blocks only.

4. **Majority-party flips:** The House flipped to Republican control in January 2023 (2 months post-ChatGPT). Raw party adoption rates could be majority-control artefacts. We control for `sponsor_in_majority` in all specifications.

### Headline Results (119th Congress)

| Predictor | Coefficient (pp) | SE | p-value |
|---|---|---|---|
| House sponsor | +2.03 | 0.68 | 0.003 ** |
| Solo sponsor | +2.27 | 1.08 | 0.036 * |
| 10y younger sponsor | +1.15 | 0.32 | <0.001 *** |
| Crime & Law Enforcement | +7.92 | 2.54 | 0.002 ** |
| Finance & Financial Sector | +7.87 | 3.72 | 0.034 * |
| Labor & Employment | -2.89 | 0.42 | <0.001 *** |

R2 = 0.026, N = 2,994 (119th Congress). Outcome: P(AI) >= 0.50 (EditLens score >= 0.65). LPM with sponsor-clustered SEs.

A note on the low R2: binary LPM on rare outcomes (3.9% prevalence) is structurally capped. The logistic AUC-ROC of 0.72 confirms moderate but real discrimination (5-fold CV AUC-PR = 0.11 vs baseline prevalence of 0.04 — a 2.7x lift over random).

## Sector Mechanisms

### Finance (11% AI rate — Committee Pipeline)

Finance and financial-sector bills have the highest AI prevalence. The mechanism is **committee-industry pipeline**: the key interaction is Finance topic × Financial Services/Senate Banking committee membership. Adding this interaction collapses the Finance topic effect from +7.87 pp to +2.07 pp (interaction = +12.13 pp, p=0.10). Qualitative inspection of all 7 Finance AI-positive bills confirms all were sponsored by committee members except Rep. Castro (who publicly expenses AI writing software). Typical bills: Bitcoin reserve, Dodd-Frank 1071 repeal, Fed communication reform.

### Crime (11% AI rate — Rapid-Response Messaging)

Crime bills are different: no committee concentration (Judiciary × Crime interaction = +2.58 pp, p=0.65). The AI-positive Crime bills are news-cycle-driven rapid-response messaging: Trump Gold Medal, gender transition criminalization, ICE riots, nitazene opioid crisis, Epstein document release, AI chatbots harming minors. Mean length: approximately 188 words — effectively press releases with bill numbers. These are written by junior staff under 24-hour deadlines.

### Labor (0% AI rate — Institutional Veto Players)

Labor bills show zero AI prevalence. Two categories:
1. Recurring Equal Pay Day resolutions (~40%) drafted by outside advocacy coalitions.
2. Statutory amendments (FLSA, ERISA, union bills) reviewed by AFL-CIO legal teams. The presence of institutional veto players (union GCs, coalition partners) makes AI use costly and detectable.

The zero rate is confirmed by an independent Pangram 3.3.2 audit: all 18 Labor sample bills returned fraction_ai = 0.0000.

## Outlier Validation: Pangram 3.3.2 Commercial Audit

We submitted all 29 AI-positive bills from the Finance and Crime sectors plus 18 Labor controls to Pangram 3.3.2 Bulk API:

| Sector | N | Pangram-Confirmed AI | % Agreement |
|---|---|---|---|
| Finance | 8 | 7 | 87.5% |
| Crime | 21 | 14 | 66.7% |
| Labor | 18 | 0 | 100% |

The single Finance false positive (s2019 TRAPS Act) reads like a genuine human staff draft with high fact density. The Crime false positives tend to be bills with high numerical density (dollar amounts, dates, code citations). Overall, EditLens and Pangram agree on 39/47 bills (82.9%), with EditLens more permissive (more false positives) — consistent with its design as a lenient edit-magnitude detector.

## Known Limitations

1. **Detector domain mismatch:** EditLens was trained on reviews, fiction, and news — not legislative text. Calibration against Pangram and false-positive measurement on pre-2022 text partially but not completely address this.

2. **Calibration sample size:** Our gold calibration sample (N=1,200 federal bills) is adequate for the six-bin calibration curve but too small for fine-grained score-level calibration. Future work should expand the gold sample.

3. **EditLens vs Pangram asymmetry:** The commercial audit shows 82.9% agreement, which is high but not perfect. Cases of disagreement warrant qualitative inspection.

4. **No individual intent inference:** A high AI score indicates the text reads as edited by AI. It does not identify *who* used the tool (staff vs. member), whether the user had permission, or whether the use violated internal policy.

5. **Coverage gap:** We do not measure AI use in committee reports, conference reports, floor speeches with Extensions of Remarks (CREC extension texts are included but full CREC floor speeches are not fully analyzed in this release).

## Appendix: CREC Extensions of Remarks

We also analyzed Congressional Record Extensions of Remarks (floor insertions from the 119th Congress). These are short statements appended to the record, typically written by junior staff with minimal review. The prevalence rate is higher than for bills (~15% in Q2 2026), consistent with the thin-institutional-capacity mechanism: less formal review = more AI use.

## Data Sources

All raw data from free public U.S. government APIs:
- **GPO govinfo BILLS:** https://www.govinfo.gov/bulkdata/BILLS/
- **GPO govinfo BILLSTATUS:** https://www.govinfo.gov/bulkdata/BILLSTATUS/
- **GPO govinfo CREC:** https://www.govinfo.gov/bulkdata/CREC/
- **Detector:** https://huggingface.co/pangram/editlens_roberta-large