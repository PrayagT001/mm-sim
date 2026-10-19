"""Insider market maker.

Every K steps, with probability p, it receives a noisy RANGE for V(t+H):
    centre = V(t+H) + noise,  range = [centre - width, centre + width]
While informed (until t+H) it quotes around the range centre with a spread that
grows with the width (less certain = wider), keeping the inventory skew.
Otherwise it behaves exactly like the Avellaneda-Stoikov market maker.

Quoting around the centre can cross other makers' quotes; those parts trade
immediately, which is how the insider takes profit from other market makers.
"""
from __future__ import annotations

import math

from bots.avellaneda import AvellanedaStoikovMM
from bots.base import MarketState
from sim.price import PricePath
from sim.types import Quote


class InsiderMM(AvellanedaStoikovMM):
    kind = "insider"

    def __init__(self, path: PricePath, every: int = 50, p: float = 0.3, horizon: int = 30,
                 noise: float = 1.0, width: float = 2.0, name: str | None = None,
                 **as_kwargs) -> None:
        super().__init__(name=name, **as_kwargs)
        self.path = path
        self.every, self.p, self.horizon = every, p, horizon
        self.noise, self.width = noise, width
        self.centre: float | None = None
        self.informed_until = -1
        self.signals = 0

    def is_informed(self, t: int) -> bool:
        return t < self.informed_until

    def _maybe_receive_signal(self, t: int) -> None:
        if t % self.every == 0 and self.rng.random() < self.p:
            target = min(t + self.horizon, self.path.n - 1)
            self.centre = self.path.V[target] + self.rng.normal(0.0, self.noise)
            self.informed_until = target
            self.signals += 1

    def on_tick(self, state: MarketState) -> Quote:
        self._maybe_receive_signal(state.t)
        if not self.is_informed(state.t):
            return super().on_tick(state)
        h = state.mid_history
        if len(h) >= 2:
            self.vol.update(float(h[-1] - h[-2]))
        r = self.centre - self.skew(state.inventory, self.vol.var)
        half = 0.5 + self.width / 4.0
        bid, ask = math.floor(r - half), math.ceil(r + half)
        if bid >= ask:
            ask = bid + 1
        bid_size, ask_size = self.sizes(state.inventory, self.size)
        # Not post-only while informed: it is willing to take mispriced quotes.
        return Quote(bid, ask, bid_size, ask_size, post_only=False)
