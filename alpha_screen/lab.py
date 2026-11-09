"""One screening trial on the grammar-generated candidate family.

Pipeline: simulate market -> evaluate ~217 expressions -> choose each
candidate's sign on the training window only -> test on the untouched
holdout with three kinds of p-values -> apply several multiple-testing
procedures -> score against the known ground truth.
"""

from __future__ import annotations

import math

import numpy as np

from . import stats
from .grammar import build_grammar, evaluate_all
from .market import MarketConfig, simulate

METHODS = [
    "naive_top10_in_sample", "uncorrected_p05", "bonferroni", "holm",
    "romano_wolf", "benjamini_yekutieli", "bh_block_perm", "bh_block_perm_50d", "bh_newey_west", "bh_iid_t",
]


def candidate_family(cfg: MarketConfig = MarketConfig()):
    exprs = build_grammar(len(cfg.persistence), set(cfg.signal_primitives))
    uses_signal = np.array([bool(e.primitives() & set(cfg.signal_primitives)) for e in exprs])
    return exprs, uses_signal


def rank_standardize(array: np.ndarray) -> np.ndarray:
    """Cross-sectional ranks (axis 1) scaled to zero mean, unit mean square.

    Same result as core.rank_standardize for tie-free data, but sorts a
    contiguous last axis and scatters ranks once, which is ~5x faster.
    """
    moved = np.ascontiguousarray(np.moveaxis(array, 1, -1))
    order = np.argsort(moved, axis=-1)
    n = moved.shape[-1]
    ranks = np.empty(moved.shape)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(n, dtype=float), moved.shape), axis=-1)
    ranks -= (n - 1) / 2.0
    ranks /= np.sqrt((n * n - 1) / 12.0)
    return np.ascontiguousarray(np.moveaxis(ranks, -1, 1))


def build_panel(seed: int, cfg: MarketConfig, signal_scale: float, holdout_multiplier: float = 1.0):
    """Return rank-standardized candidates X (days, assets, m), raw returns y, ranked returns yr."""
    prims, returns, sectors = simulate(seed, cfg, signal_scale, holdout_multiplier)
    exprs, _ = candidate_family(cfg)
    values = evaluate_all(exprs, prims, sectors)[cfg.burn_in:]
    if np.isnan(values).any():
        raise RuntimeError("burn-in too short for the grammar's lookbacks")
    y = returns[cfg.burn_in:]
    return rank_standardize(values), y, rank_standardize(y)


def block_cross_matrix(x_ranks: np.ndarray, y_ranks: np.ndarray, block_days: int) -> np.ndarray:
    """S[b, c, f] = sum over aligned days of block b (factor) and block c (returns) / (days*assets).

    The holdout mean IC under a block permutation pi equals sum_b S[b, pi(b), f],
    so thousands of permutations cost only gathers instead of full recomputation.
    """
    days, assets, m = x_ranks.shape
    if days % block_days:
        raise ValueError("holdout length must be a multiple of block_days")
    nb = days // block_days
    xb = x_ranks.reshape(nb, block_days * assets, m)
    yb = y_ranks.reshape(nb, block_days * assets)
    return np.einsum("bkf,ck->bcf", xb, yb, optimize=True) / (days * assets)


def block_permutation_null(cross: np.ndarray, permutations: int, rng: np.random.Generator) -> np.ndarray:
    nb = cross.shape[0]
    perms = np.argsort(rng.random((permutations, nb)), axis=1)
    null = np.zeros((permutations, cross.shape[2]))
    for b in range(nb):
        null += cross[b][perms[:, b]]
    return null


def long_short(x_ranks: np.ndarray, returns: np.ndarray, direction: np.ndarray, cost_bps: float):
    """Daily gross/net pnl and one-way turnover of a gross-1 rank-weighted long/short book."""
    w = x_ranks * direction / np.abs(x_ranks).sum(axis=1, keepdims=True)
    gross = np.einsum("daf,da->df", w, returns)
    turnover = np.zeros_like(gross)
    turnover[1:] = np.abs(np.diff(w, axis=0)).sum(axis=1)
    net = gross - cost_bps * 1e-4 * turnover
    return gross, net, turnover


def annual_sharpe(pnl: np.ndarray) -> np.ndarray:
    return pnl.mean(axis=0) / (pnl.std(axis=0, ddof=1) + 1e-15) * math.sqrt(252)


def run_trial(seed: int, cfg: MarketConfig = MarketConfig(), signal_scale: float = 1.0,
              holdout_multiplier: float = 1.0, permutations: int = 9999, q: float = 0.05,
              naive_k: int = 10, cost_bps: float = 5.0, nw_lags: int = 20, detail: bool = False) -> dict:
    exprs, uses_signal = candidate_family(cfg)
    m = len(exprs)
    is_true = uses_signal & (signal_scale > 0) & (holdout_multiplier > 0)
    x, y, yr = build_panel(seed, cfg, signal_scale, holdout_multiplier)
    tr = cfg.train_days
    x_tr, x_ho, y_tr, y_ho, yr_tr, yr_ho = x[:tr], x[tr:], y[:tr], y[tr:], yr[:tr], yr[tr:]

    # ---- training window only: choose direction, build the naive baseline
    train_ic = np.einsum("daf,da->df", x_tr, yr_tr) / cfg.assets
    direction = np.where(train_ic.mean(axis=0) >= 0, 1.0, -1.0)
    w_tr = x_tr / np.abs(x_tr).sum(axis=1, keepdims=True)
    pnl_tr = np.einsum("daf,da->df", w_tr, y_tr)
    sr_daily = pnl_tr.mean(axis=0) / pnl_tr.std(axis=0, ddof=1)
    naive = np.argsort(-np.abs(sr_daily), kind="stable")[:naive_k]
    best = naive[0]
    dsr = stats.deflated_sharpe_ratio(pnl_tr[:, best] * np.sign(sr_daily[best]),
                                      np.concatenate([sr_daily, -sr_daily]), 2 * m)

    # ---- holdout: directional mean rank IC and three flavours of p-value
    hold_daily = np.einsum("daf,da->df", x_ho, yr_ho) / cfg.assets * direction
    observed = hold_daily.mean(axis=0)
    cross = block_cross_matrix(x_ho, yr_ho, cfg.block_days)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7_919]))
    null = block_permutation_null(cross, permutations, rng) * direction
    p_perm = stats.permutation_p_values(observed, null)
    # sensitivity check: longer 50-day blocks (fewer blocks, more serial dependence kept)
    cross50 = block_cross_matrix(x_ho, yr_ho, 50)
    p_perm50 = stats.permutation_p_values(observed, block_permutation_null(cross50, permutations, rng) * direction)
    p_iid = stats.iid_t_p_values(hold_daily)
    p_nw = stats.newey_west_p_values(hold_daily, nw_lags)

    sel = {
        "naive_top10_in_sample": np.isin(np.arange(m), naive),
        "uncorrected_p05": p_perm <= q,
        "bonferroni": stats.bonferroni(p_perm, q),
        "holm": stats.holm(p_perm, q),
        "romano_wolf": stats.romano_wolf(observed, null, q)[0],
        "benjamini_yekutieli": stats.benjamini_yekutieli(p_perm, q),
        "bh_block_perm": stats.benjamini_hochberg(p_perm, q),
        "bh_block_perm_50d": stats.benjamini_hochberg(p_perm50, q),
        "bh_newey_west": stats.benjamini_hochberg(p_nw, q),
        "bh_iid_t": stats.benjamini_hochberg(p_iid, q),
    }
    m1 = int(is_true.sum())
    methods = {}
    for name, s in sel.items():
        r, v = int(s.sum()), int((s & ~is_true).sum())
        methods[name] = {"R": r, "V": v, "S": r - v, "fdp": v / max(r, 1),
                         "power": (r - v) / m1 if m1 else None}

    nulls = ~is_true
    calibration = {k: float(np.mean(p[nulls] <= 0.05)) if nulls.any() else None
                   for k, p in (("block_perm", p_perm), ("block_perm_50d", p_perm50), ("newey_west", p_nw), ("iid_t", p_iid))}

    gross, net, turnover = long_short(x_ho, y_ho, direction, cost_bps)
    g_sr, n_sr, avg_to = annual_sharpe(gross), annual_sharpe(net), turnover[1:].mean(axis=0)
    bh = sel["bh_block_perm"]
    trading = {
        "cost_bps": cost_bps,
        "bh_discoveries": int(bh.sum()),
        "bh_net_sharpe_positive": int((bh & (n_sr > 0)).sum()),
        "bh_mean_gross_sharpe": float(g_sr[bh].mean()) if bh.any() else None,
        "bh_mean_net_sharpe": float(n_sr[bh].mean()) if bh.any() else None,
        "bh_mean_turnover": float(avg_to[bh].mean()) if bh.any() else None,
    }
    out = {
        "seed": seed, "signal_scale": signal_scale, "holdout_multiplier": holdout_multiplier,
        "m": m, "m_true": m1, "methods": methods, "null_rejection_rate_at_05": calibration,
        "naive_top1": {"is_true": bool(is_true[best]), "deflated_sharpe": dsr,
                       "annual_sharpe_in_sample": float(abs(sr_daily[best]) * math.sqrt(252)),
                       "holdout_directional_ic": float(observed[best]) * float(np.sign(sr_daily[best]) * direction[best])},
        "trading": trading,
    }
    if detail:
        flat = x_tr.reshape(-1, m)
        corr = flat.T @ flat / flat.shape[0]
        off = np.abs(corr[np.triu_indices(m, 1)])
        out["correlation"] = {
            "pairs_abs_corr_over_0.8": int(np.sum(off > 0.8)),
            "pairs_abs_corr_over_0.5": int(np.sum(off > 0.5)),
            "effective_number_of_tests_li_ji": stats.effective_number_of_tests(corr),
        }
        _, rw_adj = stats.romano_wolf(observed, null, q)
        out["candidates"] = [{
            "name": exprs[i].name, "injected": bool(is_true[i]),
            "train_ic": float(train_ic[:, i].mean()), "direction": int(direction[i]),
            "in_sample_annual_sharpe": float(sr_daily[i] * math.sqrt(252)),
            "holdout_directional_ic": float(observed[i]),
            "p_block_perm": float(p_perm[i]), "p_block_perm_50d": float(p_perm50[i]),
            "p_newey_west": float(p_nw[i]), "p_iid_t": float(p_iid[i]),
            "romano_wolf_adjusted_p": float(rw_adj[i]),
            "holdout_gross_sharpe": float(g_sr[i]), "holdout_net_sharpe": float(n_sr[i]),
            "holdout_turnover": float(avg_to[i]),
            "selected_by": [k for k, s in sel.items() if s[i]],
        } for i in range(m)]
    return out


def population_ic(cfg: MarketConfig = MarketConfig(), signal_scale: float = 1.0,
                  seeds=range(900_000, 900_008)) -> np.ndarray:
    """Long-run mean rank IC per candidate, estimated on independent seeds (len(seeds)*days days)."""
    total = None
    for s in seeds:
        x, _, yr = build_panel(s, cfg, signal_scale)
        ic = np.einsum("daf,da->f", x, yr) / (x.shape[0] * cfg.assets)
        total = ic if total is None else total + ic
    return total / len(seeds)
