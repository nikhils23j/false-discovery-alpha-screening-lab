"""Render PNG figures from results/study.json (needs matplotlib).

    python -m alpha_screen.plots --input results/study.json --outdir results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABELS = {
    "naive_top10_in_sample": "Naive top-10 (in-sample Sharpe)",
    "uncorrected_p05": "Holdout p<0.05, no correction",
    "bonferroni": "Bonferroni", "holm": "Holm", "romano_wolf": "Romano-Wolf (max-stat)",
    "benjamini_yekutieli": "Benjamini-Yekutieli",
    "bh_block_perm": "BH + 20-day block-permutation p",
    "bh_block_perm_50d": "BH + 50-day block-permutation p",
    "bh_newey_west": "BH + Newey-West p", "bh_iid_t": "BH + iid t-test p",
}


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fdr_power(study: dict, out: Path) -> None:
    methods = study["scenarios"]["main"]["methods"]
    names = list(LABELS)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, key, color, title in ((axes[0], "fdr", ORANGE, "Empirical FDR (mean FDP)"),
                                  (axes[1], "power", BLUE, "Power (share of true candidates found)")):
        for i, n in enumerate(names):
            e = methods[n][key]
            lo, hi = e["ci95"]
            ax.plot([lo, hi], [i, i], color=color, linewidth=2, solid_capstyle="round")
            ax.plot(e["mean"], i, "o", color=color, markersize=6, markeredgecolor="white", markeredgewidth=1)
            ax.text(hi + 0.01, i, f"{e['mean']:.3f}", va="center", fontsize=8.5, color=INK)
        _style(ax)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xlim(0, 1.05)
    axes[0].axvline(study["q"], color=MUTED, linestyle="--", linewidth=1)
    axes[0].text(study["q"] + 0.01, -0.45, "q = 0.05", fontsize=8.5, color=MUTED)
    axes[0].set_yticks(range(len(names)), [LABELS[n] for n in names], fontsize=9, color=INK)
    n = study["scenarios"]["main"]["trials"]
    fig.suptitle(f"{n} simulated markets, {study['candidate_family']['m']} candidate formulas each (dots: mean, bars: 95% CI)",
                 fontsize=10, color=MUTED, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fdr_power_by_method.png", dpi=150)
    plt.close(fig)


def power_curve(study: dict, out: Path) -> None:
    sc = study["scenarios"]
    scales = sorted(float(k.split("_", 1)[1]) for k in sc if k.startswith("sweep_"))
    pop = study["population_ic_estimate"]["by_scale"]
    ticks = [f"{s:g}\n(IC {pop[str(s)]['x0']:.3f})" if s > 0 else "0\n(no signal)" for s in scales]
    series = [("bh_block_perm", BLUE), ("romano_wolf", ORANGE), ("benjamini_yekutieli", AQUA), ("bh_iid_t", YELLOW)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, color in series:
        pw = [sc[f"sweep_{s}"]["methods"][name].get("power", {}).get("mean", 0.0) for s in scales]
        fdr = [sc[f"sweep_{s}"]["methods"][name]["fdr"]["mean"] for s in scales]
        axes[0].plot(scales[1:], pw[1:], "-o", color=color, linewidth=2, markersize=6, label=LABELS[name])
        axes[1].plot(scales, fdr, "-o", color=color, linewidth=2, markersize=6, label=LABELS[name])
    axes[1].axhline(study["q"], color=MUTED, linestyle="--", linewidth=1)
    axes[0].set_title("Power vs injected signal strength", fontsize=11, color=INK, loc="left")
    axes[1].set_title("Empirical FDR vs injected signal strength", fontsize=11, color=INK, loc="left")
    for ax in axes:
        _style(ax)
        ax.set_xticks(scales, ticks, fontsize=7.5)
        ax.set_xlabel("signal scale (estimated population rank IC of raw x0)", fontsize=9, color=MUTED)
    axes[0].legend(fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "power_curve.png", dpi=150)
    plt.close(fig)


def calibration(study: dict, out: Path) -> None:
    sc = study["scenarios"]
    labels = {"iid_t": "iid t-test", "newey_west": "Newey-West (20 lags)", "block_perm": "20-day block permutation",
              "block_perm_50d": "50-day block permutation"}
    keys = list(labels)
    scen = [("null_only", "No signal anywhere", BLUE), ("main", "Signal present (decoys only)", ORANGE)]
    fig, ax = plt.subplots(figsize=(9.5, 3.9))
    width = 0.36
    for j, (s, lab, color) in enumerate(scen):
        for i, k in enumerate(keys):
            e = sc[s]["null_rejection_rate_at_05"][k]
            x = i + (j - 0.5) * width
            ax.bar(x, e["mean"], width - 0.04, color=color, label=lab if i == 0 else None)
            ax.plot([x, x], e["ci95"], color=INK, linewidth=1.2)
            ax.text(x, e["ci95"][1] + 0.004, f"{e['mean']:.3f}", ha="center", fontsize=8.5, color=INK)
    ax.axhline(0.05, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xticks(range(len(keys)), [labels[k] for k in keys], fontsize=9, color=INK)
    ax.set_ylabel("share of truly-null formulas with p <= 0.05", fontsize=9, color=MUTED)
    ax.set_title("Are the p-values honest? (should be 0.05)", fontsize=11, color=INK, loc="left")
    _style(ax)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "pvalue_calibration.png", dpi=150)
    plt.close(fig)


def showcase(study: dict, out: Path) -> None:
    cands = study["showcase_seed42"]["signal"]["candidates"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for flag, color, lab in ((False, MUTED, "decoy-only formula"), (True, BLUE, "uses a signal primitive")):
        pts = [c for c in cands if c["injected"] == flag]
        ax.scatter([abs(c["in_sample_annual_sharpe"]) for c in pts], [c["holdout_directional_ic"] for c in pts],
                   s=22, color=color, alpha=0.85, edgecolors="white", linewidths=0.6, label=lab)
    sel = [c for c in cands if "bh_block_perm" in c["selected_by"]]
    ax.scatter([abs(c["in_sample_annual_sharpe"]) for c in sel], [c["holdout_directional_ic"] for c in sel],
               s=70, facecolors="none", edgecolors=ORANGE, linewidths=1.5, label="selected by BH (q=0.05)")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("|in-sample annualised Sharpe| (training window)", fontsize=9, color=MUTED)
    ax.set_ylabel("holdout mean rank IC (sign fixed in training)", fontsize=9, color=MUTED)
    ax.set_title("Seed 42: training looks vs holdout reality", fontsize=11, color=INK, loc="left")
    _style(ax)
    ax.legend(fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "seed42_insample_vs_holdout.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("results/study.json"))
    ap.add_argument("--outdir", type=Path, default=Path("results"))
    args = ap.parse_args()
    study = json.loads(args.input.read_text())
    args.outdir.mkdir(parents=True, exist_ok=True)
    for fn in (fdr_power, power_curve, calibration, showcase):
        fn(study, args.outdir)
    print("wrote", ", ".join(sorted(p.name for p in args.outdir.glob("*.png"))))


if __name__ == "__main__":
    main()
