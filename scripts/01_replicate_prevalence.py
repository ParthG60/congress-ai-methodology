"""Print quarterly prevalence tables for bills and CREC Extensions of Remarks.

Uses pre-computed summary data (no heavy block-level files needed).
"""
import pathlib
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"


def run_bills_prevalence():
    df = pd.read_csv(DATA / "quarterly_prevalence_with_ci.csv")
    print("=" * 95)
    print(f"{'Quarter':<10} {'N':>5} {'Overall':>8} {'95% CI':>18} | {'Dem':>8} {'95% CI':>18} | {'Rep':>8} {'95% CI':>18}")
    print("=" * 95)
    for _, r in df.iterrows():
        print(f"{r['quarter']:<10} {int(r['n_bills']):>5} {r['pct_ai']:>6.2f}%  "
              f"[{r['ci_lo']:>5.2f}%, {r['ci_hi']:>5.2f}%]   "
              f"{r['dem_pct_ai']:>6.2f}%  [{r['dem_ci_lo']:>5.2f}%, {r['dem_ci_hi']:>5.2f}%]   "
              f"{r['rep_pct_ai']:>6.2f}%  [{r['rep_ci_lo']:>5.2f}%, {r['rep_ci_hi']:>5.2f}%]")

    # Control baseline
    ctrl = pd.read_csv(DATA / "quarterly_control_baseline.csv")
    print(f"\nControl (statutory text) latest: {ctrl.iloc[-1]['pct_ai']:.2f}%")

    latest = df.iloc[-1]
    print(f"\nHeadline: {latest['pct_ai']:.1f}% of bills (N={int(latest['n_bills'])}) in {latest['quarter']}")
    return df


def run_crec_prevalence():
    cal = pd.read_csv(DATA / "crec_prevalence_calibrated.csv")
    cal = cal[(cal.quarter >= "2023Q1") & (cal.quarter <= "2026Q2")]

    print("\n" + "=" * 80)
    print(f"{'Quarter':<10} {'N':>8} {'Prevalence':>12} {'95% CI':>18}")
    print("=" * 80)
    for _, row in cal.iterrows():
        pt = row["point"] * 100
        lo = row["lo"] * 100
        hi = row["hi"] * 100
        print(f"{str(row['quarter']):<10} {int(row['n_granules']):>8} {pt:>8.2f}%  [{lo:>5.2f}%, {hi:>5.2f}%]")

    latest = cal.iloc[-1]
    print(f"\nCREC headline: {latest['point'] * 100:.1f}% (N={int(latest['n_granules'])}) in {latest['quarter']}")
    return cal


if __name__ == "__main__":
    run_bills_prevalence()
    run_crec_prevalence()