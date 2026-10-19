"""Avellaneda-Stoikov market maker (inventory-aware).

reservation price  r = s - q * gamma * sigma^2 * tau
total spread       gamma * sigma^2 * tau + (2 / gamma) * ln(1 + gamma / k)

s = public fair value (blend of this market's mid and the reference price),
q = inventory, sigma^2 = per-step variance of the mid (estimated
online from public data), tau = a fixed look-ahead horizon in steps. The paper
uses the time left to a terminal date (T - t); a fixed tau gives the stationary
behaviour needed for a long-running market.
"""
from __future__ import annotations

import math

from bots.base import MarketMaker, MarketState, anchored_mid
from sim.types import Quote


class EwmaVariance:
    """Exponentially weighted per-step variance of a price series."""

    def __init__(self, halflife: float = 200.0, initial: float = 1.0,
                 floor: float = 0.1, cap: float = 25.0) -> None:
        self.alpha = 1.0 - 0.5 ** (1.0 / halflife)
        self.var = initial
        self.floor, self.cap = floor, cap

    def update(self, change: float) -> float:
        # Clipping stops a feedback loop: wider quotes -> bigger mid jumps ->
        # higher estimated variance -> even wider quotes.
        self.var += self.alpha * (change * change - self.var)
        self.var = min(max(self.var, self.floor), self.cap)
        return self.var


class AvellanedaStoikovMM(MarketMaker):
    kind = "avellaneda"

    def __init__(self, gamma: float = 0.05, k: float = 1.5, tau: float = 20.0,
                 size: int = 5, max_inventory: int = 20, vol_halflife: float = 500.0,
                 ref_weight: float = 0.8, max_skew: float = 2.0,
                 name: str | None = None) -> None:
        super().__init__(name)
        self.gamma, self.k, self.tau = gamma, k, tau
        self.size = size
        self.max_inventory = max_inventory
        self.ref_weight = ref_weight
        self.max_skew = max_skew
        self.vol = EwmaVariance(vol_halflife)

    def fair_value(self, state: MarketState) -> float:
        """The price around which to quote. Subclasses override this."""
        return anchored_mid(state, self.ref_weight)

    def skew(self, inventory: int, var: float) -> float:
        """Inventory shift q * gamma * sigma^2 * tau, capped at max_skew ticks.

        The cap is a practical guard: uncapped, one maker at its limit can quote far
        through everyone else and drag the whole market with it."""
        raw = inventory * self.gamma * var * self.tau
        return max(-self.max_skew, min(self.max_skew, raw))

    def half_spread(self, var: float) -> float:
        g = self.gamma
        return 0.5 * (g * var * self.tau + (2.0 / g) * math.log(1.0 + g / self.k))

    def on_tick(self, state: MarketState) -> Quote:
        h = state.mid_history
        if len(h) >= 2:
            self.vol.update(float(h[-1] - h[-2]))
        var = self.vol.var
        r = self.fair_value(state) - self.skew(state.inventory, var)
        half = self.half_spread(var)
        bid = math.floor(r - half)
        ask = math.ceil(r + half)
        if bid >= ask:
            ask = bid + 1
        bid_size, ask_size = self.sizes(state.inventory, self.size)
        return Quote(bid, ask, bid_size, ask_size)
