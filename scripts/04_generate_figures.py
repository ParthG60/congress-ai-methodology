"""Generate Effort News-style figures for the methodology repo.

Produces:
  figures/fig1_prevalence_quarterly.png — Aggregate + Party breakdown
  figures/fig2_crec_extensions.png     — CREC Extensions of Remarks
  figures/fig3_predictor_forest.png    — 119th Congress coefficient plot
"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

# ── Dark theme (Effort News style) ──
BG_COLOR = "#000000"
TEXT_MAIN = "#f8fafc"
TEXT_MUTED = "#94a3b8"
TEXT_DIM = "#64748b"
HAIRLINE = "#777777"
WHITE = "#ffffff"
ZERO_COLOR = "#888888"
DEM_BLUE = "#3b82f6"
REP_RED = "#ef4444"
POS_COLOR = "#22c55e"
NEG_COLOR = "#ef4444"

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Georgia", "DejaVu Serif", "Times New Roman"]
plt.rcParams["axes.edgecolor"] = HAIRLINE
plt.rcParams["axes.linewidth"] = 0.8


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: Quarterly Bills
# ──────────────────────────────────────────────────────────────────────────────
def fig_prevalence_quarterly():
    df = pd.read_csv(DATA / "quarterly_prevalence_with_ci.csv")
    ctrl = pd.read_csv(DATA / "quarterly_control_baseline.csv")
    q_str = list(df["quarter"])
    x = np.arange(len(df))
    comb_v = df["pct_ai"].rolling(2, min_periods=1).mean().values
    ctrl_v = ctrl["pct_ai"].rolling(2, min_periods=1).mean().values
    dem_v = df["dem_pct_ai"].rolling(2, min_periods=1).mean().values
    rep_v = df["rep_pct_ai"].rolling(2, min_periods=1).mean().values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8), facecolor=BG_COLOR,
                                   gridspec_kw={'width_ratios': [1.2, 1]})

    # Panel A
    ax1.set_facecolor(BG_COLOR)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.25, color=HAIRLINE, zorder=1)
    ax1.plot(x, comb_v, color=WHITE, linewidth=2.8, label="Findings & Preambles", zorder=5)
    ax1.plot(x, ctrl_v, color=HAIRLINE, linewidth=1.8, linestyle="--",
             label="Statutory text only", zorder=3)
    ax1.text(x[-1] + 0.15, comb_v[-1], f"{comb_v[-1]:.1f}%", color=WHITE,
             fontsize=9.5, fontweight="bold", va="center")
    ax1.set_title("A. Congress Aggregate", fontsize=12.5,
                  fontweight="bold", color=TEXT_MAIN, loc="left", pad=12)
    ax1.set_ylabel("Expected AI Prevalence (% of Bills)", fontsize=9.5,
                   color=TEXT_MUTED, labelpad=8)
    ax1.set_ylim(-0.2, 8.2)
    ax1.set_xlim(-0.3, len(x) + 0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(q_str, fontsize=8.5, color=TEXT_MUTED, rotation=30)
    ax1.tick_params(colors=TEXT_MUTED, labelsize=9)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)
    ax1.legend(loc="upper left", frameon=False, fontsize=8.0, labelcolor=TEXT_MAIN)

    # Panel B
    ax2.set_facecolor(BG_COLOR)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.25, color=HAIRLINE, zorder=1)
    ax2.plot(x, dem_v, color=DEM_BLUE, linewidth=2.4, label="Democratic Staff", zorder=4)
    ax2.plot(x, rep_v, color=REP_RED, linewidth=2.4, label="Republican Staff", zorder=5)
    gap = abs(dem_v[-1] - rep_v[-1])
    d_off = 0.6 if dem_v[-1] >= rep_v[-1] else -0.4
    r_off = -0.4 if rep_v[-1] >= dem_v[-1] else 0.6
    d_va = "bottom" if d_off > 0 else "top"
    r_va = "top" if r_off < 0 else "bottom"
    ax2.text(x[-1] + 0.15, dem_v[-1] + d_off, f"D: {dem_v[-1]:.1f}%", color=DEM_BLUE,
             fontsize=9, fontweight="bold", va=d_va)
    ax2.text(x[-1] + 0.15, rep_v[-1] + r_off, f"R: {rep_v[-1]:.1f}%", color=REP_RED,
             fontsize=9, fontweight="bold", va=r_va)
    ax2.set_title("B. Party Comparisons", fontsize=12.5,
                  fontweight="bold", color=TEXT_MAIN, loc="left", pad=12)
    ax2.set_ylabel("Within-Party Expected AI Prevalence (%)", fontsize=9.5,
                   color=TEXT_MUTED, labelpad=8)
    ax2.set_ylim(-0.2, 8.2)
    ax2.set_xlim(-0.3, len(x) + 1.2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(q_str, fontsize=8.5, color=TEXT_MUTED, rotation=30)
    ax2.tick_params(colors=TEXT_MUTED, labelsize=9)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.legend(loc="upper left", frameon=False, fontsize=8.2, labelcolor=TEXT_MAIN)

    fig.suptitle("AI in Congressional Bills", fontsize=15, fontweight="bold",
                 color=TEXT_MAIN, x=0.06, y=0.97, ha="left")
    fig.text(0.06, 0.02,
             "Data: GPO govinfo 116th-119th Congress. EditLens 3B calibrated using Pangram 3.3.2.",
             fontsize=7.5, color=TEXT_DIM, fontfamily="monospace")
    out = FIGURES / "fig1_prevalence_quarterly.png"
    fig.savefig(out, dpi=300, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: CREC Extensions
# ──────────────────────────────────────────────────────────────────────────────
def fig_crec_extensions():
    cal = pd.read_csv(DATA / "crec_prevalence_calibrated.csv")
    cal = cal[(cal.quarter >= "2023Q1") & (cal.quarter <= "2026Q2")]
    q_str = list(cal["quarter"])
    y = cal["point"].values * 100
    if len(y) > 1:
        y = pd.Series(y).rolling(2, min_periods=1).mean().values
    x = np.arange(len(q_str))

    fig, ax = plt.subplots(figsize=(10.5, 5.4), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25, color=HAIRLINE, zorder=1)
    ax.plot(x, y, color=WHITE, linewidth=2.8, zorder=4)
    ax.text(x[-1] + 0.15, y[-1], f"{y[-1]:.1f}%", color=WHITE,
            fontsize=9.5, fontweight="bold", va="center")
    ax.set_title("AI in Congressional Record Extensions of Remarks",
                 fontsize=15, fontweight="bold", color=TEXT_MAIN, loc="left", pad=16)
    ax.set_ylabel("Calibrated AI Prevalence (%)", fontsize=10, color=TEXT_MUTED, labelpad=10)
    ax.set_ylim(-0.5, 30.0)
    ax.set_xlim(-0.3, len(x) + 0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(q_str, fontsize=8.5, color=TEXT_MUTED, rotation=30)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.08, 0.02,
             "Data: GPO govinfo CREC (2023-2026). EditLens 3B calibrated using Pangram 3.3.2.",
             fontsize=7.8, color=TEXT_DIM, fontfamily="monospace")
    out = FIGURES / "fig2_crec_extensions.png"
    fig.savefig(out, dpi=300, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: Coefficient forest plot
# ──────────────────────────────────────────────────────────────────────────────
def fig_predictor_forest():
    results = pd.read_csv(DATA / "replicated_regression_results.csv")
    headline = results[(results["outcome"] == "ai_c65") & (results["model"] == "LPM")].copy()

    LABELS = {
        "is_house": "House sponsor",
        "is_solo_sponsor": "Solo sponsor",
        "age_per_decade_younger": "10y younger sponsor",
        "topic_crime_and_law_enforcement": "Crime & Law Enforcement",
        "topic_finance_and_financial_sector": "Finance & Financial Sector",
        "topic_labor_and_employment": "Labor & Employment",
    }
    TIERS = {
        "Institutional Capacity": ["is_house"],
        "Coalition Scrutiny": ["is_solo_sponsor"],
        "Sponsor Cohort": ["age_per_decade_younger"],
        "Policy Area Poles": [
            "topic_crime_and_law_enforcement",
            "topic_finance_and_financial_sector",
            "topic_labor_and_employment",
        ],
    }
    VAR_ORDER = [v for tier in TIERS.values() for v in tier]

    def dot_color(coeff, p):
        if p < 0.05:
            return POS_COLOR if coeff > 0 else NEG_COLOR, 1.0
        if p < 0.10:
            return POS_COLOR if coeff > 0 else NEG_COLOR, 0.55
        return "#555555", 0.45

    def sig_parts(p):
        if p < 0.001:
            return "p<0.001"
        if p < 0.01:
            return f"p={p:.3f}"
        if p < 0.05:
            return f"p={p:.3f}"
        return f"p={p:.3f}"

    res_dict = {}
    for _, r in headline.iterrows():
        res_dict[r["variable"]] = r
    meta = headline.iloc[0] if len(headline) > 0 else None

    n_vars = len([v for v in VAR_ORDER if v in res_dict])
    fig, ax = plt.subplots(figsize=(9.5, 2.8 + n_vars * 0.30))
    fig.patch.set_facecolor(BG_COLOR)
    fig.subplots_adjust(right=0.70, bottom=0.22, top=0.88)

    # Build y positions with tier spacing
    rows_list = []
    prev_tier = None
    spacing = 0.0
    for v in VAR_ORDER:
        if v not in res_dict:
            continue
        t = next((t for t, vs in TIERS.items() if v in vs), "")
        if prev_tier is not None and t != prev_tier:
            spacing += 0.7
        rows_list.append((spacing, v))
        prev_tier = t
        spacing += 1.0

    y_pos = np.array([r[0] for r in rows_list])
    names = [r[1] for r in rows_list]
    coeffs = np.array([res_dict[v]["coeff"] * 100 for v in names])
    ses = np.array([res_dict[v]["se"] * 100 for v in names])
    ps = np.array([res_dict[v]["p"] for v in names])
    ci = 1.645 * ses  # 90% CI

    ax.axvline(0, color=ZERO_COLOR, linewidth=0.6, zorder=1)

    for i in range(len(names)):
        c, a = dot_color(coeffs[i], ps[i])
        ax.errorbar(coeffs[i], y_pos[i], xerr=ci[i], fmt="none",
                    ecolor=c, capsize=3, capthick=1, linewidth=1.2, alpha=a, zorder=3)
        ax.scatter(coeffs[i], y_pos[i], color=c, s=38, alpha=a, zorder=4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([LABELS.get(n, n) for n in names], fontsize=8, color=TEXT_MAIN)

    for i in range(len(names)):
        lo = coeffs[i] - ci[i]
        hi = coeffs[i] + ci[i]
        sign = "+" if coeffs[i] >= 0 else ""
        ax.text(1.02, y_pos[i],
                f"{sign}{coeffs[i]:.2f} pp  [{lo:.1f}, {hi:.1f}]  ({sig_parts(ps[i])})",
                transform=ax.get_yaxis_transform(), fontsize=6.5, color=TEXT_MUTED,
                va="center", ha="left", fontfamily="monospace")

    x_max = max(np.abs(coeffs + ci).max(), np.abs(coeffs - ci).max(), 1.0)
    x_lim = x_max * 1.35
    ax.set_xlim(-x_lim, x_lim)
    step = 2 if x_lim <= 12 else (4 if x_lim <= 24 else 5)
    ticks = np.arange(-x_lim, x_lim + step, step)
    ticks = ticks[(ticks >= -x_lim * 0.8) & (ticks <= x_lim * 0.8)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:+.0f} pp" if t != 0 else "0" for t in ticks],
                       fontsize=7.5, color=TEXT_MUTED)
    ax.set_xlabel("Change in probability of AI text (pp)", fontsize=8,
                  color=TEXT_MUTED, labelpad=6)

    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color(ZERO_COLOR)
    ax.spines[["bottom", "left"]].set_linewidth(0.5)
    ax.grid(axis="x", color=ZERO_COLOR, linewidth=0.4, zorder=0)
    ax.tick_params(axis="y", length=0, colors=TEXT_MUTED)
    ax.tick_params(axis="x", length=3, colors=ZERO_COLOR)
    y_min = y_pos.min() - 1.0
    y_max = y_pos.max() + 1.0
    ax.set_ylim(y_min, y_max)

    # Tier labels
    seen = {}
    for name, y in zip(names, y_pos):
        for t, vs in TIERS.items():
            if name in vs:
                seen[t] = y
    for t in TIERS:
        if t not in seen:
            continue
        ax.text(-0.22, seen[t], t, transform=ax.get_yaxis_transform(),
                fontsize=7.5, fontweight="bold", color=TEXT_MUTED, va="center", ha="right")

    ax.set_title("What drives P(AI) in legislative drafting", loc="left",
                 fontsize=12, fontweight="bold", color=TEXT_MAIN, pad=8)

    n_val = meta["n"] if meta is not None else 0
    r2_val = meta["r2"] if meta is not None else 0
    fig.text(0.0, 0.02,
             f"119th Congress (N={n_val:,} bills) | R2 = {r2_val:.4f} | "
             f"P(AI) >= 0.50 via isotonic calibration, 90% sponsor-clustered CIs",
             fontsize=6.5, color=TEXT_DIM, ha="left", va="bottom")

    out = FIGURES / "fig3_predictor_forest.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Saved -> {out}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig_prevalence_quarterly()
    fig_crec_extensions()
    fig_predictor_forest()