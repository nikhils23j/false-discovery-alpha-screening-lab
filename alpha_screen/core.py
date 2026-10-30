"""A reproducible research exercise, not a live trading or profitability test."""

from __future__ import annotations

import math

import numpy as np


TRUE_COUNT = 5
FACTOR_COUNT = 50
ASSET_COUNT = 12
DAY_COUNT = 1200
TRAIN_DAYS = 700
BLOCK_DAYS = 20


def synthetic_market(seed: int, days: int = DAY_COUNT, assets: int = ASSET_COUNT, signal_scale: float = 1.0):
    """Return time-t factors and time-(t+1) returns; five factors carry signal.

    The candidate array has shape (days, assets, 50). Independent decoys and
    correlated noise in returns make this a controlled detection exercise.
    All inputs are generated before the train/test split, with no fitted
    transformation or future returns entering a candidate factor.
    """
    if days < TRAIN_DAYS + BLOCK_DAYS * 2 or assets < 4:
        raise ValueError("Need enough days and assets for held-out block testing")
    if signal_scale < 0:
        raise ValueError("signal_scale cannot be negative")
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(days, assets, FACTOR_COUNT))
    factors = np.tanh(raw)  # bounded, monotone candidate expressions
    common = rng.normal(scale=0.2, size=(days, 1))
    noise = rng.normal(scale=0.7, size=(days, assets)) + common
    coefficients = np.array([0.28, 0.25, 0.22, 0.20, 0.18])
    next_returns = signal_scale * np.einsum("daf,f->da", factors[:, :, :TRUE_COUNT], coefficients) + noise
    names = [f"tanh(signal_{i+1})" for i in range(TRUE_COUNT)]
    names += [f"tanh(decoy_{i+1})" for i in range(FACTOR_COUNT - TRUE_COUNT)]
    return factors, next_returns, names


def rank_standardize(array: np.ndarray) -> np.ndarray:
    """Cross-sectional ranks with zero mean and unit mean square each day."""
    ranks = np.argsort(np.argsort(array, axis=1), axis=1).astype(float)
    ranks -= ranks.mean(axis=1, keepdims=True)
    scale = np.sqrt(np.mean(ranks * ranks, axis=1, keepdims=True))
    return ranks / scale


def daily_rank_ic(factors: np.ndarray, next_returns: np.ndarray) -> np.ndarray:
    """Daily cross-sectional rank IC: rows are dates, columns are factors."""
    x = rank_standardize(factors)
    y = rank_standardize(next_returns)
    return np.einsum("daf,da->df", x, y) / factors.shape[1]


def block_permutation_indices(days: int, block_days: int, rng: np.random.Generator) -> np.ndarray:
    """Reorder complete day blocks while preserving within-block order."""
    if days % block_days:
        raise ValueError("Holdout length must be a multiple of block size")
    count = days // block_days
    if count < 2:
        raise ValueError("At least two blocks required")
    order = rng.permutation(count)
    if np.array_equal(order, np.arange(count)):
        order = np.roll(order, 1)
    return (order[:, None] * block_days + np.arange(block_days)).ravel()


def benjamini_hochberg(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    """BH step-up discoveries for a predeclared family of tests."""
    if not 0 < q < 1:
        raise ValueError("q must lie between zero and one")
    p_values = np.asarray(p_values, dtype=float)
    if np.any(~np.isfinite(p_values)) or np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("p-values must be finite and between zero and one")
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    allowed = sorted_p <= q * np.arange(1, len(p_values) + 1) / len(p_values)
    result = np.zeros(len(p_values), dtype=bool)
    if np.any(allowed):
        result[order[: np.flatnonzero(allowed)[-1] + 1]] = True
    return result


def evaluate(seed: int = 42, permutations: int = 499, signal_scale: float = 1.0) -> dict:
    if permutations < 199:
        raise ValueError("At least 199 permutations are required for the 50-test BH grid")
    factors, returns, names = synthetic_market(seed, signal_scale=signal_scale)
    train_x, hold_x = factors[:TRAIN_DAYS], factors[TRAIN_DAYS:]
    train_y, hold_y = returns[:TRAIN_DAYS], returns[TRAIN_DAYS:]
    train_daily_ic = daily_rank_ic(train_x, train_y)
    hold_daily_ic = daily_rank_ic(hold_x, hold_y)
    train_ic = train_daily_ic.mean(axis=0)
    direction = np.where(train_ic >= 0, 1.0, -1.0)
    observed = hold_daily_ic.mean(axis=0) * direction

    # In-sample long/short rank weights provide the deliberately naive baseline.
    train_ranks = rank_standardize(train_x)
    weights = train_ranks / np.abs(train_ranks).sum(axis=1, keepdims=True)
    daily_pnl = np.einsum("daf,da->df", weights, train_y)
    train_sharpe = daily_pnl.mean(axis=0) / daily_pnl.std(axis=0, ddof=1) * math.sqrt(252)
    naive = np.argsort(-np.abs(train_sharpe))[:10]

    hold_ranks = rank_standardize(hold_x)
    hold_return_ranks = rank_standardize(hold_y)
    rng = np.random.default_rng(seed + 100_003)
    null = np.empty((permutations, FACTOR_COUNT))
    for i in range(permutations):
        idx = block_permutation_indices(len(hold_y), BLOCK_DAYS, rng)
        null[i] = np.einsum("daf,da->f", hold_ranks, hold_return_ranks[idx]) / (len(hold_y) * ASSET_COUNT)
    p_values = (1 + np.sum(null * direction >= observed, axis=0)) / (permutations + 1)
    discoveries = benjamini_hochberg(p_values)
    selected = np.flatnonzero(discoveries)
    true_positives = int(np.sum(selected < TRUE_COUNT))
    false_positives = int(np.sum(selected >= TRUE_COUNT))
    return {
        "experiment": {
            "seed": seed, "signal_scale": signal_scale, "days": DAY_COUNT, "train_days": TRAIN_DAYS,
            "holdout_days": DAY_COUNT - TRAIN_DAYS, "assets": ASSET_COUNT,
            "candidates": FACTOR_COUNT, "injected_signals": TRUE_COUNT,
            "block_days": BLOCK_DAYS, "permutations": permutations,
            "false_discovery_rate_q": 0.05,
            "data": "seeded synthetic factors and next-day returns; no live prices",
        },
        "summary": {
            "detected_injected_signals": true_positives,
            "missed_injected_signals": TRUE_COUNT - true_positives,
            "false_discoveries": false_positives,
            "bh_selected_count": len(selected),
            "naive_top10_injected": int(np.sum(naive < TRUE_COUNT)),
            "naive_top10_decoys": int(np.sum(naive >= TRUE_COUNT)),
            "target_met": bool(signal_scale > 0 and true_positives >= 4 and false_positives <= 2),
        },
        "selected": [names[i] for i in selected],
        "candidates": [
            {
                "name": names[i], "injected": i < TRUE_COUNT,
                "train_ic": float(train_ic[i]),
                "holdout_directional_ic": float(observed[i]),
                "block_permutation_p": float(p_values[i]),
                "bh_selected": bool(discoveries[i]),
                "naive_top10": bool(i in naive),
            }
            for i in range(FACTOR_COUNT)
        ],
    }
