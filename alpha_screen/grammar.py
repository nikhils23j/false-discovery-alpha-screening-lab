"""A tiny alpha-expression grammar over (days, assets) arrays.

Every operator is causal: the value on day t only uses inputs from days <= t.
Rows without enough history are NaN; the market generator adds a burn-in so
the evaluated window never contains NaNs.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

EPS = 1e-12


def _rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    if window < 1:
        raise ValueError("window must be positive")
    padded = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)], axis=0)
    out = np.full(x.shape, np.nan)
    out[window - 1:] = padded[window:] - padded[:-window]
    return out


def ts_mean(x: np.ndarray, window: int) -> np.ndarray:
    return _rolling_sum(x, window) / window


def ts_std(x: np.ndarray, window: int) -> np.ndarray:
    mean = ts_mean(x, window)
    var = ts_mean(x * x, window) - mean * mean
    return np.sqrt(np.maximum(var, 0.0))


def delta(x: np.ndarray, lag: int) -> np.ndarray:
    if lag < 1:
        raise ValueError("lag must be positive")
    out = np.full(x.shape, np.nan)
    out[lag:] = x[lag:] - x[:-lag]
    return out


def ts_zscore(x: np.ndarray, window: int) -> np.ndarray:
    return (x - ts_mean(x, window)) / (ts_std(x, window) + EPS)


def ts_corr(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    mx, my = ts_mean(x, window), ts_mean(y, window)
    cov = ts_mean(x * y, window) - mx * my
    return cov / (ts_std(x, window) * ts_std(y, window) + EPS)


def cs_zscore(x: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score each day (axis 1 = assets)."""
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / (std + EPS)


def sector_neutral(x: np.ndarray, sectors: np.ndarray) -> np.ndarray:
    out = x.copy()
    for s in np.unique(sectors):
        cols = sectors == s
        out[:, cols] -= x[:, cols].mean(axis=1, keepdims=True)
    return out


@dataclass(frozen=True)
class Expression:
    name: str
    op: str
    inputs: tuple[int, ...]
    param: int = 0

    def primitives(self) -> set[int]:
        return set(self.inputs)


UNARY_SPECS = [
    ("raw", 0), ("ts_mean", 5), ("ts_mean", 10), ("ts_mean", 20),
    ("delta", 1), ("delta", 5), ("delta", 20),
    ("ts_zscore", 20), ("ts_zscore", 60), ("sector_neutral", 0),
]
MAX_LOOKBACK = 60


def build_grammar(n_primitives: int, signal_primitives: set[int]) -> list[Expression]:
    """Enumerate the candidate family.

    Excluded on purpose (ground truth would be ambiguous): sub() of two signal
    primitives (could cancel) and ts_corr() touching a signal primitive (its
    population IC is zero by symmetry even though it is not independent).
    """
    exprs: list[Expression] = []
    for p in range(n_primitives):
        for op, param in UNARY_SPECS:
            name = f"x{p}" if op == "raw" else (f"{op}(x{p})" if op == "sector_neutral" else f"{op}(x{p}, {param})")
            exprs.append(Expression(name, op, (p,), param))
    for a, b in combinations(range(n_primitives), 2):
        exprs.append(Expression(f"add(x{a}, x{b})", "add", (a, b)))
        if not (a in signal_primitives and b in signal_primitives):
            exprs.append(Expression(f"sub(x{a}, x{b})", "sub", (a, b)))
    for a, b in combinations(range(n_primitives), 2):
        if a not in signal_primitives and b not in signal_primitives:
            exprs.append(Expression(f"ts_corr(x{a}, x{b}, 20)", "ts_corr", (a, b), 20))
    return exprs


def evaluate_expression(expr: Expression, prims: list[np.ndarray], sectors: np.ndarray) -> np.ndarray:
    x = prims[expr.inputs[0]]
    if expr.op == "raw":
        return x
    if expr.op == "ts_mean":
        return ts_mean(x, expr.param)
    if expr.op == "delta":
        return delta(x, expr.param)
    if expr.op == "ts_zscore":
        return ts_zscore(x, expr.param)
    if expr.op == "sector_neutral":
        return sector_neutral(x, sectors)
    y = prims[expr.inputs[1]]
    if expr.op == "add":
        return cs_zscore(x) + cs_zscore(y)
    if expr.op == "sub":
        return cs_zscore(x) - cs_zscore(y)
    if expr.op == "ts_corr":
        return ts_corr(x, y, expr.param)
    raise ValueError(f"unknown op {expr.op}")


def evaluate_all(exprs: list[Expression], prims: list[np.ndarray], sectors: np.ndarray) -> np.ndarray:
    """Return array (days, assets, n_expressions)."""
    days, assets = prims[0].shape
    out = np.empty((days, assets, len(exprs)))
    for j, e in enumerate(exprs):
        out[:, :, j] = evaluate_expression(e, prims, sectors)
    return out
