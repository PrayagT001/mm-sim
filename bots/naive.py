"""Naive market maker: fixed symmetric spread around the public fair value, no inventory skew."""
from __future__ import annotations

import math

from bots.base import MarketMaker, MarketState, anchored_mid
from sim.types import Quote


class NaiveMM(MarketMaker):
    kind = "naive"

    def __init__(self, half_spread: float = 2.0, size: int = 5, max_inventory: int = 20,
                 ref_weight: float = 0.8, name: str | None = None) -> None:
        super().__init__(name)
        self.half_spread = half_spread
        self.size = size
        self.max_inventory = max_inventory
        self.ref_weight = ref_weight

    def on_tick(self, state: MarketState) -> Quote:
        fair = anchored_mid(state, self.ref_weight)
        bid = math.floor(fair - self.half_spread)
        ask = math.ceil(fair + self.half_spread)
        if bid >= ask:
            ask = bid + 1
        # A risk limit is the only concession to reality: at the limit it stops
        # adding to the position. Without one, a naive maker with unlimited
        # capital pins the price at its own quotes and the market never learns.
        bid_size, ask_size = self.sizes(state.inventory, self.size)
        return Quote(bid, ask, bid_size, ask_size)
