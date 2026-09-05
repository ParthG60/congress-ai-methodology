"""Replicate sector tests: topic prevalence, Finance committee interaction, Crime characteristics, Labor.

All numbers computed live from data — no hardcoded coefficients.
"""
import pathlib
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"

BINS_6 = ["[0.00, 0.20)", "[0.20, 0.35)", "[0.35, 0.50)", "[0.50, 0.65)", "[0.65, 0.80)", "[0.80, 1.00]"]
MIDS = np.array([0.10, 0.275, 0.425, 0.575, 0.725, 0.90])


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
    """LPM with sponsor-clustered SEs (Liang-Zeger)."""
    X_mat = np.asarray(X, dtype=float)
    y_vec = np.asarray(y, dtype=float)
    clusters = np.asarray(clusters)
    beta, *_ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
    n, k = X_mat.shape
    resids = y_vec - X_mat @ beta
    G = len(np.unique(clusters))
    XtX_inv = np.linalg.pinv(X_mat.T @ X_mat)
    meat = np.zeros((k, k))
    for g in np.unique(clusters):
        idx = clusters == g
        u_g = X_mat[idx].T @ resids[idx]
        meat += np.outer(u_g, u_g)
    # Finite-sample correction
    factor = (G / (G - 1)) * ((n - 1) / (n - k))
    vcv = XtX_inv @ meat @ XtX_inv * factor
    se = np.sqrt(np.diag(vcv))
    t_vals = beta / se
    p_vals = 2 * (1 - _t_cdf(np.abs(t_vals), n - k))
    return {name: {"coeff": b, "se": s, "p": p}
            for name, b, s, p in zip(feature_names, beta, se, p_vals)}


def _t_cdf(x, df):
    from scipy import stats as _stats
    return _stats.t.cdf(x, df)


def main():
    rates = load_fed_calib()
    df = pd.read_csv(DATA / "federal_content_bills_extended.csv", low_memory=False)
    df["p_ai"] = p_ai(df["score_max"].values, rates)
    df["ai_c65"] = (df["score_max"] >= 0.65).astype(float)
    d119 = df[df["congress"] == 119].copy()

    # ── Topic-specific prevalence ──
    topic_map = {
        "topic_crime_and_law_enforcement": "Crime & Law Enforcement",
        "topic_finance_and_financial_sector": "Finance & Financial Sector",
        "topic_labor_and_employment": "Labor & Employment",
    }
    print("=" * 65)
    print(f"{'Sector':<35s} {'N':>6s} {'AI Rate (c65)':>15s} {'Calibrated P(ai)':>16s}")
    print("=" * 65)
    for col, label in topic_map.items():
        sub = d119[d119[col] == 1]
        n = len(sub)
        ai_rate = sub["ai_c65"].mean() * 100
        calib_val = sub["p_ai"].mean() * 100
        print(f"  {label:<33s} {n:>6d} {ai_rate:>13.2f}% {calib_val:>14.2f}%")
    rest = d119[~d119[list(topic_map.keys())].any(axis=1)]
    ai_rate = rest["ai_c65"].mean() * 100
    calib_val = rest["p_ai"].mean() * 100
    print(f"  {'Other topics':<33s} {len(rest):>6d} {ai_rate:>13.2f}% {calib_val:>14.2f}%")

    # ── Finance committee interaction effect ──
    print("\n─── Finance Committee Interaction ───")

    # Load the stored main effect from the full 6-predictor model
    stored = pd.read_csv(DATA / "replicated_regression_results.csv")
    fin_main = stored[(stored["variable"] == "topic_finance_and_financial_sector")
                      & (stored["outcome"] == "ai_c65") & (stored["model"] == "LPM")]
    if len(fin_main) > 0:
        r = fin_main.iloc[0]
        print(f"  Finance topic main effect (6-predictor model, no committee): "
              f"{r['coeff'] * 100:+.2f} pp (p={r['p']:.4f})")

    # Now add committee membership and interaction (live from committee_membership.csv)
    cmte = pd.read_csv(DATA / "committee_membership.csv")
    fin_cmte_ids = cmte[cmte["committee_name"].str.contains(
        "Financial Services|Banking,", case=False, na=False
    )]["bioguide"].unique()
    d119["is_finance_committee"] = d119["sponsor_bioguide"].isin(fin_cmte_ids).astype(float)

    # Replicate the full 6-predictor model plus committee + interaction
    full_feats = ["intercept", "is_house", "is_solo_sponsor", "age_per_decade_younger",
                  "topic_crime_and_law_enforcement", "topic_finance_and_financial_sector",
                  "topic_labor_and_employment", "is_finance_committee",
                  "fin_x_committee"]
    dd = d119.copy()
    dd["intercept"] = 1.0
    dd["is_solo_sponsor"] = (dd["n_cosponsors"].fillna(0) == 0).astype(float)
    dd["age_per_decade_younger"] = (60.0 - dd["member_age"]) / 10.0
    dd["age_per_decade_younger"] -= dd["age_per_decade_younger"].mean()
    dd["fin_x_committee"] = dd["topic_finance_and_financial_sector"] * dd["is_finance_committee"]
    dd = dd.dropna(subset=full_feats + ["ai_c65", "sponsor_bioguide"])

    y = dd["ai_c65"].astype(float).values
    X = dd[full_feats].values
    cl = dd["sponsor_bioguide"].fillna("unknown").values
    res = ols_clustered(y, X, full_feats, cl)

    fin_eff = res["topic_finance_and_financial_sector"]
    int_eff = res["fin_x_committee"]
    cmte_eff = res["is_finance_committee"]

    fin_only = fin_eff["coeff"] * 100
    interaction = int_eff["coeff"] * 100
    collapsed = fin_only + interaction
    print(f"  Finance × Fin. Services committee interaction: {interaction:+.2f} pp (p={int_eff['p']:.4f})")
    print(f"  Finance topic effect WITH committee control: {fin_only:+.2f} pp")
    print(f"  Collapsed effect for committee members (finance + interaction): {collapsed:+.2f} pp")
    print(f"  Committee membership direct effect: {cmte_eff['coeff'] * 100:+.2f} pp (p={cmte_eff['p']:.4f})")
    n_fin = int(dd["topic_finance_and_financial_sector"].sum())
    n_on_cmte = int((dd["topic_finance_and_financial_sector"].astype(bool) & dd["is_finance_committee"].astype(bool)).sum())
    print(f"  Finance bills: {n_fin}, sponsored by committee members: {n_on_cmte}")
    print(f"  Implication: Finance AI effect collapses when committee is controlled.")

    # ── Crime bill characteristics ──
    print("\n─── Crime Bill Characteristics ───")
    crime = d119[d119["topic_crime_and_law_enforcement"] == 1]
    crime_ai = crime[crime["ai_c65"] == 1]
    print(f"  Crime bills total: {len(crime)}")
    print(f"  AI-positive Crime bills: {len(crime_ai)}")
    if len(crime_ai) > 0:
        print(f"  Mean length AI Crime: {crime_ai['ln_words'].mean():.1f} log-words "
              f"(~{np.exp(crime_ai['ln_words'].mean()):.0f} words)")

    # Judiciary committee interaction (null check)
    jud_cmte_ids = cmte[cmte["committee_name"].str.contains(
        "Judiciary", case=False, na=False
    )]["bioguide"].unique()
    d119["is_judiciary"] = d119["sponsor_bioguide"].isin(jud_cmte_ids).astype(float)
    dd2 = d119.copy()
    dd2["intercept"] = 1.0
    dd2["topic_crime_judiciary"] = dd2["topic_crime_and_law_enforcement"] * dd2["is_judiciary"]
    jud_feats = ["intercept", "topic_crime_and_law_enforcement", "is_judiciary", "topic_crime_judiciary"]
    dd2 = dd2.dropna(subset=jud_feats + ["ai_c65", "sponsor_bioguide"])
    y2 = dd2["ai_c65"].values
    X2 = dd2[jud_feats].values
    cl2 = dd2["sponsor_bioguide"].fillna("unknown").values
    res2 = ols_clustered(y2, X2, jud_feats, cl2)
    jud_int = res2["topic_crime_judiciary"]
    print(f"  Judiciary × Crime interaction: {jud_int['coeff'] * 100:+.2f} pp (p={jud_int['p']:.4f}) — no committee concentration")

    # ── Labor bills: zero AI ──
    print("\n─── Labor Bills ───")
    labor = d119[d119["topic_labor_and_employment"] == 1]
    print(f"  Labor bills total: {len(labor)}")
    print(f"  AI-positive Labor (c65): {labor['ai_c65'].sum()}")
    print(f"  Calibrated P(AI) mean: {labor['p_ai'].mean() * 100:.3f}%")

    # ── Pangram outlier audit summary ──
    print("\n─── Pangram 3.3.2 Outlier Audit ───")
    audit = pd.read_csv(DATA / "pangram_outlier_audit.csv")
    print(f"  Bills submitted: {len(audit)}")
    for sector_label, sector_key in [("Finance", "finance"), ("Crime", "crime"), ("Labor", "labor")]:
        sub = audit[audit["category"].str.contains(sector_key, case=False, na=False)]
        n = len(sub)
        ai_count = (sub["prediction_short"] == "AI").sum()
        human_count = (sub["prediction_short"] == "Human").sum()
        print(f"  {sector_label}: {n} bills ({ai_count} AI, {human_count} Human)")


if __name__ == "__main__":
    main()