"""Replicate 119th Congress binary AI / not-AI predictor models.

Output: prints terminal tables, saves CSV to data/.
"""
import math
import pathlib
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"

BINS_6 = ["[0.00, 0.20)", "[0.20, 0.35)", "[0.35, 0.50)", "[0.50, 0.65)", "[0.65, 0.80)", "[0.80, 1.00]"]
MIDS = np.array([0.10, 0.275, 0.425, 0.575, 0.725, 0.90])

FEATURES = [
    "intercept",
    "is_house",
    "is_solo_sponsor",
    "age_per_decade_younger",
    "topic_crime_and_law_enforcement",
    "topic_finance_and_financial_sector",
    "topic_labor_and_employment",
]
THRESHOLDS = {"ai_c35": 0.35, "ai_c50": 0.50, "ai_c65": 0.65}


def load_fed_calib():
    calib = pd.read_csv(DATA / "joint_calibration_tables_corrected.csv")
    sub = calib[calib.corpus == "federal_bills"]
    emp = np.array([float(sub[sub.score_bin == b]["empirical_rate"].values[0]) for b in BINS_6])
    for i in range(1, len(emp)):
        if emp[i] < emp[i - 1]:
            emp[i] = emp[i - 1]
    return emp


def p_ai(score, rates):
    return np.interp(score, MIDS, rates, left=0.0, right=float(rates[-1]))


def ols_clustered(y, X, feature_names, clusters):
    X_mat = np.asarray(X, dtype=float)
    y_vec = np.asarray(y, dtype=float)
    clusters = np.asarray(clusters)
    beta, *_ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
    n, k = X_mat.shape
    resids = y_vec - X_mat @ beta
    rss = float(np.sum(resids ** 2))
    tss = float(np.sum((y_vec - np.mean(y_vec)) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if k > 1 and n > k else r2
    G = len(np.unique(clusters))
    XtX_inv = np.linalg.pinv(X_mat.T @ X_mat)
    meat = np.zeros((k, k))
    for g in np.unique(clusters):
        idx = clusters == g
        u_g = X_mat[idx].T @ resids[idx]
        meat += np.outer(u_g, u_g)
    dfc = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 else 1.0
    cov = XtX_inv @ meat @ XtX_inv * dfc
    se = np.sqrt(np.maximum(0, np.diag(cov)))
    t = beta / np.where(se > 0, se, 1e-12)
    p_vals = np.array([2 * (1 - 0.5 * (1 + math.erf(abs(v) / math.sqrt(2)))) for v in t])
    return {name: {"coeff": b, "se": s, "p": pp} for name, b, s, pp in zip(feature_names, beta, se, p_vals)}, r2, adj_r2


def run_lpm(d, outcome, cluster_on="bioguide"):
    req = [outcome] + FEATURES
    dd = d.dropna(subset=req).copy()
    y = dd[outcome].astype(float).values
    X = dd[FEATURES].values
    cl = dd[cluster_on].fillna("unknown").values
    res, r2, adj_r2 = ols_clustered(y, X, FEATURES, cl)
    rows = []
    for v in FEATURES:
        r = res[v]
        rows.append({"variable": v, "coeff": round(r["coeff"], 6), "se": round(r["se"], 6),
                      "p": round(r["p"], 4), "r2": round(r2, 4), "adj_r2": round(adj_r2, 4),
                      "n": len(dd), "n_pos": int(y.sum())})
    return rows


def main():
    rates = load_fed_calib()

    df = pd.read_csv(DATA / "federal_content_bills_119.csv", low_memory=False)
    df = df.copy()
    df["intercept"] = 1.0
    df["is_solo_sponsor"] = (df["n_cosponsors"].fillna(0) == 0).astype(float)
    df["age_per_decade_younger"] = (60.0 - df["member_age"]) / 10.0
    df["age_per_decade_younger"] -= df["age_per_decade_younger"].mean()
    for name, cut in THRESHOLDS.items():
        df[name] = (df["score_max"] >= cut).astype(float)
    df["p_ai"] = p_ai(df["score_max"].values, rates)

    print(f"\n119th Congress analysis sample: N = {len(df):,}")
    for name, cut in THRESHOLDS.items():
        print(f"  Prevalence {name} (>= {cut}): {df[name].mean() * 100:.2f}% ({int(df[name].sum())} bills)")
    print(f"  Calibrated P(AI) expected mean: {df['p_ai'].mean() * 100:.2f}%")
    print(f"  Calibrated binary P(AI) >= 0.50: {df['p_ai'].ge(0.50).mean() * 100:.2f}% ({int(df['p_ai'].ge(0.50).sum())} bills)")

    # Run LPM for headline outcome (ai_c65)
    print("\n=== HEADLINE: 119th Congress, calibrated binary P(AI) >= 0.50 (score >= 0.65), LPM ===")
    rows = run_lpm(df, "ai_c65")
    for r in rows:
        if r["variable"] == "intercept":
            continue
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
        print(f"  {r['variable']:42s} {r['coeff'] * 100:+6.2f} pp  (se={r['se'] * 100:.2f}, p={r['p']:.4f} {sig})")
    print(f"  R2 = {rows[0]['r2']:.4f}, N = {rows[0]['n']:,}")

    # Run LPM for the calibrated continuous outcome
    print("\n=== CALIBRATED P(AI) (continuous, OLS) ===")
    rows_p = run_lpm(df, "p_ai")
    for r in rows_p:
        if r["variable"] == "intercept":
            continue
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
        pp = r["coeff"] * 100
        print(f"  {r['variable']:42s} {pp:+6.2f} pp  (se={r['se'] * 100:.2f}, p={r['p']:.4f} {sig})")
    print(f"  R2 = {rows_p[0]['r2']:.4f}, N = {rows_p[0]['n']:,}")

    # Save full results
    all_rows = []
    for name in ["ai_c35", "ai_c50", "ai_c65"]:
        rr = run_lpm(df, name)
        for r in rr:
            r.update({"outcome": name, "model": "LPM"})
        all_rows += rr
    rr = run_lpm(df, "p_ai")
    for r in rr:
        r.update({"outcome": "p_ai", "model": "OLS"})
    all_rows += rr
    pd.DataFrame(all_rows).to_csv(DATA / "replicated_regression_results.csv", index=False)
    print(f"\nSaved full results -> data/replicated_regression_results.csv")


if __name__ == "__main__":
    main()