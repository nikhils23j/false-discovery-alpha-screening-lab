"""Synthetic market with known ground truth, serial and cross-sectional dependence.

Ten observable primitives x0..x9 per asset are AR(1) processes; some share a
sector component. Only x0 (slow) and x1 (fast) drive next-day returns. Returns
also contain a market factor, sector shocks, persistent sector and asset
drifts (these make daily ICs autocorrelated) and fat-tailed noise.

Random streams are split with SeedSequence so the primitives never depend on
the signal strength: decoys are independent of returns by construction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .grammar import MAX_LOOKBACK


@dataclass(frozen=True)
class MarketConfig:
    assets: int = 40
    sectors: int = 4
    days: int = 1200
    train_days: int = 700
    block_days: int = 20
    burn_in: int = MAX_LOOKBACK + 20
    persistence: tuple = (0.97, 0.30, 0.0, 0.30, 0.60, 0.90, 0.97, 0.99, 0.95, 0.80)
    sector_share: tuple = (0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.5, 0.0, 0.3, 0.0)
    signal_primitives: tuple = (0, 1)
    signal_loadings: tuple = (0.0004, 0.00045)  # return per 1 sd of the primitive
    market_vol: float = 0.010
    sector_vol: float = 0.007
    idio_vol: float = 0.015
    noise_df: int = 5
    asset_drift_vol: float = 0.003
    sector_drift_vol: float = 0.002
    drift_phi: float = 0.97

    def to_dict(self) -> dict:
        return asdict(self)


def _ar1(rng, phi: float, shape, steps: int, shocks=None) -> np.ndarray:
    out = np.empty((steps,) + shape)
    e = rng.normal(size=(steps,) + shape) if shocks is None else shocks
    out[0] = e[0]
    k = np.sqrt(1 - phi * phi)
    for t in range(1, steps):
        out[t] = phi * out[t - 1] + k * e[t]
    return out


def simulate(seed: int, cfg: MarketConfig = MarketConfig(), signal_scale: float = 1.0,
             holdout_multiplier: float = 1.0):
    """Return (primitives list of (T, N) arrays incl. burn-in, returns (T, N), sectors).

    returns[t, a] is the return from the close of day t to day t+1, so a
    feature computed with data up to day t is matched with returns[t].
    The first `burn_in` rows are history only and are dropped by the caller.
    """
    if signal_scale < 0 or holdout_multiplier < 0:
        raise ValueError("signal_scale and holdout_multiplier must be non-negative")
    prim_ss, ret_ss = np.random.SeedSequence(seed).spawn(2)
    prim_rng, ret_rng = np.random.default_rng(prim_ss), np.random.default_rng(ret_ss)
    n, steps = cfg.assets, cfg.burn_in + cfg.days
    sectors = np.arange(n) % cfg.sectors

    prims = []
    for phi, share in zip(cfg.persistence, cfg.sector_share):
        idio = prim_rng.normal(size=(steps, n))
        sect = prim_rng.normal(size=(steps, cfg.sectors))[:, sectors]
        shocks = np.sqrt(1 - share) * idio + np.sqrt(share) * sect
        prims.append(_ar1(None, phi, (n,), steps, shocks))

    market = ret_rng.normal(scale=cfg.market_vol, size=(steps, 1))
    sector_shock = ret_rng.normal(scale=cfg.sector_vol, size=(steps, cfg.sectors))[:, sectors]
    noise = ret_rng.standard_t(cfg.noise_df, size=(steps, n)) * cfg.idio_vol / np.sqrt(cfg.noise_df / (cfg.noise_df - 2))
    asset_drift = cfg.asset_drift_vol * _ar1(ret_rng, cfg.drift_phi, (n,), steps)
    sector_drift = cfg.sector_drift_vol * _ar1(ret_rng, cfg.drift_phi, (cfg.sectors,), steps)[:, sectors]

    alpha = sum(g * prims[p] for p, g in zip(cfg.signal_primitives, cfg.signal_loadings))
    strength = np.full((steps, 1), signal_scale)
    strength[cfg.burn_in + cfg.train_days:] *= holdout_multiplier
    returns = market + sector_shock + asset_drift + sector_drift + noise + strength * alpha
    return prims, returns, sectors
