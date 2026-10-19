"""Hidden true value V(t).

V(t+1) = V(t) + kappa * (anchor(t) - V(t)) + sigma(regime) * Z + Jump

- anchor(t): a slow random walk. V is pulled towards it, which is the only
  learnable pattern (a small, partly predictable drift).
- regime: two-state Markov chain, calm or stormy, with different sigma.
- Jump: rare news shock, normal with standard deviation jump_scale.
- Z: standard normal noise.

One step represents one second. V is in ticks (a float); quotes are integer ticks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CALM, STORMY = 0, 1


@dataclass(frozen=True)
class PriceParams:
    v0: float = 1000.0
    kappa: float = 0.03            # pull of V towards the anchor per step
    anchor_sigma: float = 0.3      # anchor random-walk step size
    sigma_calm: float = 1.0        # ticks per step
    sigma_stormy: float = 3.0
    p_calm_to_stormy: float = 0.002   # expected calm spell ~500 steps
    p_stormy_to_calm: float = 0.008   # expected stormy spell ~125 steps
    jump_prob: float = 0.001
    jump_scale: float = 15.0       # ticks


@dataclass(frozen=True)
class PricePath:
    params: PriceParams
    V: np.ndarray          # true value, length n
    anchor: np.ndarray     # anchor, length n
    regime: np.ndarray     # 0 calm / 1 stormy, length n
    jump: np.ndarray       # jump size added at each step (0 if none)

    @property
    def n(self) -> int:
        return len(self.V)


def simulate_price(n: int, params: PriceParams = PriceParams(),
                   rng: np.random.Generator | None = None) -> PricePath:
    """Generate n steps of the hidden value process."""
    if n < 2:
        raise ValueError("n must be at least 2")
    rng = np.random.default_rng() if rng is None else rng
    p = params

    # Regime chain: loop is cheap (n ~ 1e4-1e5) and easy to read.
    u = rng.random(n)
    regime = np.empty(n, dtype=np.int8)
    regime[0] = CALM
    for t in range(1, n):
        if regime[t - 1] == CALM:
            regime[t] = STORMY if u[t] < p.p_calm_to_stormy else CALM
        else:
            regime[t] = CALM if u[t] < p.p_stormy_to_calm else STORMY

    sigma = np.where(regime == STORMY, p.sigma_stormy, p.sigma_calm)
    z = rng.standard_normal(n)
    jump = np.where(rng.random(n) < p.jump_prob, rng.normal(0.0, p.jump_scale, n), 0.0)
    anchor = p.v0 + np.concatenate(([0.0], np.cumsum(rng.normal(0.0, p.anchor_sigma, n - 1))))

    V = np.empty(n)
    V[0] = p.v0
    for t in range(n - 1):
        V[t + 1] = V[t] + p.kappa * (anchor[t] - V[t]) + sigma[t + 1] * z[t + 1] + jump[t + 1]
    jump[0] = 0.0
    return PricePath(params=p, V=V, anchor=anchor, regime=regime, jump=jump)


def oracle_prediction(path: PricePath, horizon: int) -> np.ndarray:
    """Best forecast of V(t+H) - V(t) for someone who knows anchor(t) and V(t).

    The anchor is a martingale, so E[anchor(t+k)] = anchor(t), and the gap
    anchor - V shrinks by a factor (1 - kappa) per step in expectation.
    """
    k = path.params.kappa
    gap = path.anchor - path.V
    return (1.0 - (1.0 - k) ** horizon) * gap


def oracle_r2(path: PricePath, horizon: int) -> float:
    """Share of the variance of H-step changes in V that the oracle forecast explains.

    This is the ceiling for any predictor: real bots see only noisy public prices,
    so they can do no better.
    """
    pred = oracle_prediction(path, horizon)[:-horizon]
    actual = path.V[horizon:] - path.V[:-horizon]
    sse = np.sum((actual - pred) ** 2)
    sst = np.sum((actual - actual.mean()) ** 2)
    return float(1.0 - sse / sst)


def reference_feed(path: PricePath, lag: int = 5, noise: float = 1.0, persistence: float = 0.95,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """A public reference price, e.g. the same asset trading on a bigger venue.

    ref(t) = V(t - lag) + e(t), with e an AR(1) noise of standard deviation `noise`
    and autocorrelation `persistence`. Everyone sees it; it lags V and is noisy, so
    traders who see V directly (value traders, the insider) still have an edge.
    It anchors market makers' fair values, the way prices on other venues and
    related instruments anchor real market makers.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = path.n
    shocks = rng.standard_normal(n) * noise * np.sqrt(1.0 - persistence ** 2)
    e = np.empty(n)
    e[0] = rng.standard_normal() * noise
    for t in range(1, n):
        e[t] = persistence * e[t - 1] + shocks[t]
    lagged = np.concatenate((np.full(lag, path.V[0]), path.V[:n - lag])) if lag > 0 else path.V.copy()
    return lagged + e
