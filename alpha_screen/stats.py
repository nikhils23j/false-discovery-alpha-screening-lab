"""Multiple-testing procedures and small statistical helpers (numpy + stdlib)."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

_N = NormalDist()


def _check_p(p_values) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be a 1-D array of finite numbers in [0, 1]")
    return p


def _check_q(q: float) -> None:
    if not 0 < q < 1:
        raise ValueError("q must lie between zero and one")


def _step_up(p: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    order = np.argsort(p, kind="stable")
    ok = p[order] <= thresholds
    out = np.zeros(len(p), dtype=bool)
    if np.any(ok):
        out[order[: np.flatnonzero(ok)[-1] + 1]] = True
    return out


def benjamini_hochberg(p_values, q: float = 0.05) -> np.ndarray:
    """BH step-up: reject the k smallest, k = max{i : p_(i) <= i q / m}."""
    _check_q(q)
    p = _check_p(p_values)
    m = len(p)
    return _step_up(p, q * np.arange(1, m + 1) / m)


def benjamini_yekutieli(p_values, q: float = 0.05) -> np.ndarray:
    """BY step-up: BH with q divided by the harmonic number H_m (valid under any dependence)."""
    _check_q(q)
    p = _check_p(p_values)
    m = len(p)
    harmonic = np.sum(1.0 / np.arange(1, m + 1))
    return _step_up(p, q * np.arange(1, m + 1) / (m * harmonic))


def bonferroni(p_values, alpha: float = 0.05) -> np.ndarray:
    _check_q(alpha)
    p = _check_p(p_values)
    return p <= alpha / len(p)


def holm(p_values, alpha: float = 0.05) -> np.ndarray:
    """Holm step-down: walk up sorted p-values while p_(i) <= alpha / (m - i + 1)."""
    _check_q(alpha)
    p = _check_p(p_values)
    m = len(p)
    order = np.argsort(p, kind="stable")
    ok = p[order] <= alpha / (m - np.arange(m))
    out = np.zeros(m, dtype=bool)
    k = m if np.all(ok) else int(np.argmin(ok))
    out[order[:k]] = True
    return out


def permutation_p_values(observed: np.ndarray, null: np.ndarray) -> np.ndarray:
    """One-sided p = (1 + #{null >= observed}) / (B + 1); null has shape (B, m)."""
    return (1 + np.sum(null >= observed[None, :], axis=0)) / (null.shape[0] + 1)


def romano_wolf(observed: np.ndarray, null: np.ndarray, alpha: float = 0.05):
    """Romano-Wolf step-down max-statistic adjusted p-values (FWER control).

    Statistics are studentized by their own permutation null mean/sd so that
    candidates with different null spreads compete fairly. Returns
    (rejected, adjusted_p).
    """
    _check_q(alpha)
    mu = null.mean(axis=0)
    sd = null.std(axis=0) + 1e-15
    z = (observed - mu) / sd
    zn = (null - mu) / sd
    order = np.argsort(-z, kind="stable")
    # running max over the hypotheses ranked at or below position k
    tail_max = np.maximum.accumulate(zn[:, order[::-1]], axis=1)[:, ::-1]
    raw = (1 + np.sum(tail_max >= z[order][None, :], axis=0)) / (null.shape[0] + 1)
    adjusted_sorted = np.maximum.accumulate(raw)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted <= alpha, adjusted


def normal_sf(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 0.5 * np.vectorize(math.erfc)(z / math.sqrt(2.0))


def iid_t_p_values(daily: np.ndarray) -> np.ndarray:
    """One-sided p for mean(daily) > 0 treating days as independent (normal approx)."""
    n = daily.shape[0]
    se = daily.std(axis=0, ddof=1) / math.sqrt(n)
    return normal_sf(daily.mean(axis=0) / (se + 1e-15))


def newey_west_se(daily: np.ndarray, lags: int) -> np.ndarray:
    """HAC standard error of the mean with Bartlett weights; lags=0 gives the iid SE (ddof=0)."""
    n = daily.shape[0]
    e = daily - daily.mean(axis=0)
    var = np.sum(e * e, axis=0) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        var = var + 2.0 * w * np.sum(e[k:] * e[:-k], axis=0) / n
    return np.sqrt(np.maximum(var, 0.0) / n)


def newey_west_p_values(daily: np.ndarray, lags: int) -> np.ndarray:
    return normal_sf(daily.mean(axis=0) / (newey_west_se(daily, lags) + 1e-15))


def deflated_sharpe_ratio(pnl: np.ndarray, trial_sharpes: np.ndarray, n_trials: float) -> float:
    """Bailey & Lopez de Prado (2014) DSR for the selected strategy's per-period pnl.

    trial_sharpes are the per-period Sharpe ratios of all strategies tried
    (their variance sets the expected maximum under the null).
    """
    pnl = np.asarray(pnl, dtype=float)
    t = len(pnl)
    sr = pnl.mean() / pnl.std(ddof=1)
    e = pnl - pnl.mean()
    s2 = np.mean(e ** 2)
    skew = np.mean(e ** 3) / s2 ** 1.5
    kurt = np.mean(e ** 4) / s2 ** 2
    gamma = 0.5772156649015329
    n = max(float(n_trials), 2.0)
    sr0 = math.sqrt(np.var(trial_sharpes, ddof=1)) * (
        (1 - gamma) * _N.inv_cdf(1 - 1 / n) + gamma * _N.inv_cdf(1 - 1 / (n * math.e))
    )
    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr * sr, 1e-12))
    return _N.cdf((sr - sr0) * math.sqrt(t - 1) / denom)


def effective_number_of_tests(corr: np.ndarray) -> float:
    """Li & Ji (2005) eigenvalue estimate of the number of independent tests."""
    eig = np.abs(np.linalg.eigvalsh(corr))
    return float(np.sum((eig >= 1).astype(float) + (eig - np.floor(eig))))


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    phat = successes / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean_ci(values) -> dict:
    v = np.asarray(values, dtype=float)
    n = len(v)
    mean = float(v.mean())
    half = 1.96 * float(v.std(ddof=1)) / math.sqrt(n) if n > 1 else float("nan")
    return {"mean": mean, "ci95": [mean - half, mean + half], "n": n}
