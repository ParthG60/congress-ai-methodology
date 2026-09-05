"""Generate publication-ready figures for the methodology repo.

Uses raw unsmoothed data so every label matches the underlying CSV exactly.
Light theme per editorial charting standard: white background, #eee grid, despine.

Figures:
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

# ── Light theme (editorial standard) ──
ACCENT = "#2563eb"        # strong blue for main series
ACCENT_ALPHA = "#93c5fd"   # lighter for CIs / secondary
GREY_LINE = "#555555"      # axis / grid lines
GREY_GRID = "#e0e0e0"      # light grid
GREY_TEXT = "#444444"      # labels
GREY_MUTED = "#777777"     # footnotes / dim annotations
DEM_BLUE = "#3b82f6"
REP_RED = "#ef4444"
POS_COLOR = "#16a34a"
NEG_COLOR = "#dc2626"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]
plt.rcParams["axes.edgecolor"] = GREY_LINE
plt.rcParams["axes.linewidth"] = 0.6


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: Quarterly Bills (raw unsmoothed)
# ──────────────────────────────────────────────────────────────────────────────
def fig_prevalence_quarterly():
    df = pd.read_csv(DATA / "quarterly_prevalence_with_ci.csv")
    ctrl = pd.read_csv(DATA / "quarterly_control_baseline.csv")
    q_str = list(df["quarter"])
    x = np.arange(len(df))

    # Raw values — no rolling averages
    comb_v = df["pct_ai"].values
    ctrl_v = ctrl["pct_ai"].values
    dem_v = df["dem_pct_ai"].values
    rep_v = df["rep_pct_ai"].values

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 5.2),
        gridspec_kw={"width_ratios": [1.2, 1]},
        facecolor="white",
    )

    # ── Panel A: Aggregate ──
    ax1.set_facecolor("white")
    ax1.grid(True, axis="y", color=GREY_GRID, linewidth=0.5, zorder=0)
    ax1.plot(x, comb_v, color=ACCENT, linewidth=2.2, label="Findings & Preambles", zorder=4)
    ax1.plot(x, ctrl_v, color=GREY_LINE, linewidth=1.5, linestyle="--",
             label="Statutory text only", zorder=3)
    ax1.text(x[-1] + 0.15, comb_v[-1], f"{comb_v[-1]:.1f}%", color=ACCENT,
             fontsize=9.5, fontweight="bold", va="center")
    ax1.set_title("A. Congress Aggregate", fontsize=12, fontweight="bold",
                  color="#000000", loc="left", pad=10)
    ax1.set_ylabel("AI Prevalence (% of bills)", fontsize=9, color=GREY_MUTED, labelpad=6)
    ax1.set_ylim(-0.1, 7.5)
    ax1.set_xlim(-0.3, len(x) + 0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(q_str, fontsize=7.5, color=GREY_TEXT, rotation=30)
    ax1.tick_params(colors=GREY_TEXT, labelsize=8)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax1.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=GREY_TEXT)

    # ── Panel B: Party ──
    ax2.set_facecolor("white")
    ax2.grid(True, axis="y", color=GREY_GRID, linewidth=0.5, zorder=0)
    ax2.plot(x, dem_v, color=DEM_BLUE, linewidth=2, label="Democratic staff", zorder=4)
    ax2.plot(x, rep_v, color=REP_RED, linewidth=2, label="Republican staff", zorder=5)
    ax2.text(x[-1] + 0.15, dem_v[-1], f"D: {dem_v[-1]:.1f}%", color=DEM_BLUE,
             fontsize=8.5, fontweight="bold", va="bottom")
    ax2.text(x[-1] + 0.15, rep_v[-1], f"R: {rep_v[-1]:.1f}%", color=REP_RED,
             fontsize=8.5, fontweight="bold", va="top")
    ax2.set_title("B. Party Comparisons", fontsize=12, fontweight="bold",
                  color="#000000", loc="left", pad=10)
    ax2.set_ylabel("Within-party AI Prevalence (%)", fontsize=9, color=GREY_MUTED, labelpad=6)
    ax2.set_ylim(-0.1, 7.5)
    ax2.set_xlim(-0.3, len(x) + 1.2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(q_str, fontsize=7.5, color=GREY_TEXT, rotation=30)
    ax2.tick_params(colors=GREY_TEXT, labelsize=8)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax2.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=GREY_TEXT)

    fig.suptitle("AI in Congressional Bills", fontsize=14, fontweight="bold",
                 color="#000000", x=0.06, y=0.97, ha="left")
    fig.text(0.06, 0.02,
             "Data: GPO govinfo 116th–119th Congress. EditLens 3B calibrated via Pangram 3.3.2.",
             fontsize=7, color=GREY_MUTED)
    out = FIGURES / "fig1_prevalence_quarterly.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: CREC Extensions (raw unsmoothed)
# ──────────────────────────────────────────────────────────────────────────────
def fig_crec_extensions():
    cal = pd.read_csv(DATA / "crec_prevalence_calibrated.csv")
    cal = cal[(cal.quarter >= "2023Q1") & (cal.quarter <= "2026Q2")]
    q_str = list(cal["quarter"])
    y = (cal["point"].values * 100)  # raw, no smoothing
    x = np.arange(len(q_str))

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    ax.set_facecolor("white")
    ax.grid(True, axis="y", color=GREY_GRID, linewidth=0.5, zorder=0)
    ax.plot(x, y, color=ACCENT, linewidth=2.2, zorder=4)
    ax.text(x[-1] + 0.15, y[-1], f"{y[-1]:.1f}%", color=ACCENT,
            fontsize=10, fontweight="bold", va="center")
    ax.set_title("AI in Congressional Record Extensions of Remarks",
                 fontsize=14, fontweight="bold", color="#000000", loc="left", pad=14)
    ax.set_ylabel("Calibrated AI Prevalence (%)", fontsize=9.5, color=GREY_MUTED, labelpad=8)
    ax.set_ylim(-0.5, 22.0)
    ax.set_xlim(-0.3, len(x) + 0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(q_str, fontsize=7.5, color=GREY_TEXT, rotation=30)
    ax.tick_params(colors=GREY_TEXT, labelsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.1, 0.02,
             "Data: GPO govinfo CREC (2023–2026). EditLens 3B calibrated via Pangram 3.3.2.",
             fontsize=7, color=GREY_MUTED)
    out = FIGURES / "fig2_crec_extensions.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
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
        "Policy Area": [
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
            return POS_COLOR if coeff > 0 else NEG_COLOR, 0.6
        return "#888888", 0.4

    def sig_label(p):
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
    fig, ax = plt.subplots(figsize=(9, 2.6 + n_vars * 0.28))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(right=0.65, bottom=0.22, top=0.88)

    rows_list = []
    prev_tier = None
    spacing = 0.0
    for v in VAR_ORDER:
        if v not in res_dict:
            continue
        t = next((t for t, vs in TIERS.items() if v in vs), "")
        if prev_tier is not None and t != prev_tier:
            spacing += 0.6
        rows_list.append((spacing, v))
        prev_tier = t
        spacing += 1.0

    y_pos = np.array([r[0] for r in rows_list])
    names = [r[1] for r in rows_list]
    coeffs = np.array([res_dict[v]["coeff"] * 100 for v in names])
    ses = np.array([res_dict[v]["se"] * 100 for v in names])
    ps = np.array([res_dict[v]["p"] for v in names])
    ci = 1.645 * ses  # 90% CI

    ax.axvline(0, color=GREY_LINE, linewidth=0.5, zorder=1)

    for i in range(len(names)):
        c, a = dot_color(coeffs[i], ps[i])
        ax.errorbar(coeffs[i], y_pos[i], xerr=ci[i], fmt="none",
                    ecolor=c, capsize=3, capthick=1, linewidth=1.2, alpha=a, zorder=3)
        ax.scatter(coeffs[i], y_pos[i], color=c, s=35, alpha=a, zorder=4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([LABELS.get(n, n) for n in names], fontsize=8.5, color="#000000")

    for i in range(len(names)):
        lo = coeffs[i] - ci[i]
        hi = coeffs[i] + ci[i]
        sign = "+" if coeffs[i] >= 0 else ""
        ax.text(1.02, y_pos[i],
                f"{sign}{coeffs[i]:.1f} pp  [{lo:.1f}, {hi:.1f}]  ({sig_label(ps[i])})",
                transform=ax.get_yaxis_transform(), fontsize=6.5, color=GREY_TEXT,
                va="center", ha="left", fontfamily="monospace")

    x_max = max(np.abs(coeffs + ci).max(), np.abs(coeffs - ci).max(), 1.0)
    x_lim = x_max * 1.35
    ax.set_xlim(-x_lim, x_lim)
    step = 2 if x_lim <= 12 else (4 if x_lim <= 24 else 5)
    ticks = np.arange(-x_lim, x_lim + step, step)
    ticks = ticks[(ticks >= -x_lim * 0.8) & (ticks <= x_lim * 0.8)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:+.0f} pp" if t != 0 else "0" for t in ticks],
                       fontsize=7, color=GREY_TEXT)
    ax.set_xlabel("Change in probability of AI text (pp)", fontsize=8.5,
                  color=GREY_MUTED, labelpad=5)

    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color(GREY_GRID)
    ax.spines[["bottom", "left"]].set_linewidth(0.5)
    ax.grid(axis="x", color=GREY_GRID, linewidth=0.4, zorder=0)
    ax.tick_params(axis="y", length=0, colors=GREY_TEXT)
    ax.tick_params(axis="x", length=3, colors=GREY_TEXT)
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
                fontsize=7.5, fontweight="bold", color=GREY_MUTED, va="center", ha="right")

    ax.set_title("What drives P(AI) in legislative drafting", loc="left",
                 fontsize=11, fontweight="bold", color="#000000", pad=6)

    n_val = meta["n"] if meta is not None else 0
    r2_val = meta["r2"] if meta is not None else 0
    fig.text(0.0, 0.02,
             f"119th Congress (N={n_val:,}) | R² = {r2_val:.3f} | "
             f"90% sponsor-clustered CIs",
             fontsize=6.5, color=GREY_MUTED, ha="left", va="bottom")

    out = FIGURES / "fig3_predictor_forest.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {out}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig_prevalence_quarterly()
    fig_crec_extensions()
    fig_predictor_forest()