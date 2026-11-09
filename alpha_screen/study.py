"""Monte-Carlo study: empirical FDR / FWER / power of screening procedures.

    python -m alpha_screen.study --profile full --workers 8 --output results/study.json
    python -m alpha_screen.study --profile quick --output results/quick.json   # ~1 minute smoke run
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import stats
from .lab import METHODS, candidate_family, population_ic, run_trial
from .market import MarketConfig

PROFILES = {
    # name: (main seeds, null seeds, sweep seeds per scale, regime seeds, permutations, pop-IC seeds)
    "full": dict(main=200, null=100, sweep=30, regime=50, permutations=9999, pop_seeds=8),
    "quick": dict(main=8, null=6, sweep=4, regime=4, permutations=999, pop_seeds=2),
}
SWEEP_SCALES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
CFG = MarketConfig()


def _trial(args):
    seed, scale, mult, perms, detail = args
    return run_trial(seed, CFG, signal_scale=scale, holdout_multiplier=mult, permutations=perms, detail=detail)


def _pop(args):
    scale, seeds = args
    return population_ic(CFG, scale, seeds)


def aggregate(trials: list[dict]) -> dict:
    out = {"trials": len(trials), "m": trials[0]["m"], "m_true": trials[0]["m_true"], "methods": {}}
    for name in METHODS:
        rows = [t["methods"][name] for t in trials]
        fwe = sum(r["V"] > 0 for r in rows)
        entry = {
            "fdr": stats.mean_ci([r["fdp"] for r in rows]),
            "fwer": {"rate": fwe / len(rows), "wilson95": list(stats.wilson_interval(fwe, len(rows)))},
            "mean_discoveries": float(np.mean([r["R"] for r in rows])),
            "mean_false_discoveries": float(np.mean([r["V"] for r in rows])),
        }
        if rows[0]["power"] is not None:
            entry["power"] = stats.mean_ci([r["power"] for r in rows])
        out["methods"][name] = entry
    out["null_rejection_rate_at_05"] = {
        k: stats.mean_ci([t["null_rejection_rate_at_05"][k] for t in trials])
        for k in trials[0]["null_rejection_rate_at_05"]
    }
    top = [t["naive_top1"] for t in trials]
    flagged = [x for x in top if x["deflated_sharpe"] > 0.95]
    out["naive_top1"] = {
        "fraction_true": float(np.mean([x["is_true"] for x in top])),
        "median_in_sample_annual_sharpe": float(np.median([x["annual_sharpe_in_sample"] for x in top])),
        "median_deflated_sharpe": float(np.median([x["deflated_sharpe"] for x in top])),
        "count_dsr_over_0.95": len(flagged),
        "fraction_true_given_dsr_over_0.95": float(np.mean([x["is_true"] for x in flagged])) if flagged else None,
        "fraction_true_given_dsr_at_most_0.95": float(np.mean([x["is_true"] for x in top if x["deflated_sharpe"] <= 0.95])) if len(flagged) < len(top) else None,
    }
    tr = [t["trading"] for t in trials]
    disc = sum(x["bh_discoveries"] for x in tr)
    nets = [x["bh_mean_net_sharpe"] for x in tr if x["bh_mean_net_sharpe"] is not None]
    gross = [x["bh_mean_gross_sharpe"] for x in tr if x["bh_mean_gross_sharpe"] is not None]
    out["trading_bh_block_perm"] = {
        "cost_bps_per_unit_turnover": tr[0]["cost_bps"],
        "total_discoveries": disc,
        "fraction_net_sharpe_positive": sum(x["bh_net_sharpe_positive"] for x in tr) / disc if disc else None,
        "median_trial_mean_gross_sharpe": float(np.median(gross)) if gross else None,
        "median_trial_mean_net_sharpe": float(np.median(nets)) if nets else None,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="full")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--output", type=Path, default=Path("results/study.json"))
    args = ap.parse_args()
    p = PROFILES[args.profile]
    perms = p["permutations"]
    start = time.time()

    jobs = {
        "main": [(s, 1.0, 1.0, perms, False) for s in range(0, p["main"])],
        "null_only": [(s, 0.0, 1.0, perms, False) for s in range(10_000, 10_000 + p["null"])],
        "regime_half": [(s, 1.0, 0.5, perms, False) for s in range(30_000, 30_000 + p["regime"])],
        "regime_dead": [(s, 1.0, 0.0, perms, False) for s in range(40_000, 40_000 + p["regime"])],
    }
    for k, scale in enumerate(SWEEP_SCALES):
        base = 20_000 + 1_000 * k
        jobs[f"sweep_{scale}"] = [(s, scale, 1.0, perms, False) for s in range(base, base + p["sweep"])]
    flat = [(name, job) for name, js in jobs.items() for job in js]
    showcase_jobs = [(42, 1.0, 1.0, perms, True), (42, 0.0, 1.0, perms, True)]
    pop_seeds = list(range(900_000, 900_000 + p["pop_seeds"]))

    with Pool(args.workers) as pool:
        pop_async = pool.map_async(_pop, [(s, pop_seeds) for s in SWEEP_SCALES])
        showcase = pool.map(_trial, showcase_jobs)
        results = pool.map(_trial, [j for _, j in flat], chunksize=2)
        pops = pop_async.get()

    grouped: dict[str, list] = {name: [] for name in jobs}
    for (name, _), res in zip(flat, results):
        grouped[name].append(res)

    exprs, uses_signal = candidate_family(CFG)
    idx_x0, idx_x1 = [e.name for e in exprs].index("x0"), [e.name for e in exprs].index("x1")
    pop_scale1 = pops[SWEEP_SCALES.index(1.0)]
    report = {
        "description": "Monte-Carlo screening study on a seeded synthetic market (no real prices).",
        "command": " ".join([Path(sys.executable).name, "-m", "alpha_screen.study"] + sys.argv[1:]),
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "workers": args.workers,
            "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        },
        "profile": args.profile, "settings": p, "market_config": CFG.to_dict(),
        "q": 0.05, "naive_k": 10, "newey_west_lags": 20, "cost_bps": 5.0,
        "seeds": {name: [j[0] for j in js][:1] + [j[0] for j in js][-1:] for name, js in jobs.items()},
        "candidate_family": {
            "m": len(exprs), "uses_signal_primitive": int(uses_signal.sum()),
            "decoy_only": int((~uses_signal).sum()),
            "names": [e.name for e in exprs],
        },
        "population_ic_estimate": {
            "days_used": p["pop_seeds"] * CFG.days, "seeds": [pop_seeds[0], pop_seeds[-1]],
            "by_scale": {str(s): {"x0": float(pop[idx_x0]), "x1": float(pop[idx_x1]),
                                  "max_abs_true": float(np.abs(pop[uses_signal]).max()),
                                  "max_abs_decoy_only": float(np.abs(pop[~uses_signal]).max())}
                         for s, pop in zip(SWEEP_SCALES, pops)},
            "scale1_median_abs_true": float(np.median(np.abs(pop_scale1[uses_signal]))),
            "note": "decoy-only candidates have exactly zero population IC by construction; their estimated values show the sampling-noise floor",
        },
        "scenarios": {name: aggregate(rs) for name, rs in grouped.items()},
        "showcase_seed42": {"signal": showcase[0], "null_only": showcase[1]},
        "elapsed_seconds": round(time.time() - start, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    main_s = report["scenarios"]["main"]["methods"]
    print(f"wrote {args.output} in {report['elapsed_seconds']} s")
    for name in METHODS:
        e = main_s[name]
        pw = e.get("power", {}).get("mean")
        print(f"  {name:24s} FDR {e['fdr']['mean']:.3f}  FWER {e['fwer']['rate']:.3f}  power {pw:.3f}  R {e['mean_discoveries']:.1f}")


if __name__ == "__main__":
    main()
