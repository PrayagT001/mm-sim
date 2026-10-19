"""Customers: the order flow market makers trade against."""
from __future__ import annotations

import math

from bots.base import BookView, MarketState, Taker
from sim.price import PricePath
from sim.types import Order, Side


class NoiseTrader(Taker):
    """Trades for reasons unrelated to value (hedging, rebalancing).

    It works like an institution executing a large order in pieces: when idle it
    starts a new run with probability `start_prob` per step, on a random side.
    During a run it sends a market order with probability `trade_prob` each step,
    always on the same side, and the run ends with probability 1 / mean_run per
    step. These runs are what make real order flow persistent.
    """
    kind = "noise"

    def __init__(self, start_prob: float = 0.01, mean_run: float = 20.0,
                 trade_prob: float = 0.5, mean_size: float = 2.0, max_size: int = 10,
                 name: str | None = None) -> None:
        super().__init__(name)
        self.start_prob, self.end_prob = start_prob, 1.0 / mean_run
        self.trade_prob = trade_prob
        self.p_size = 1.0 / mean_size
        self.max_size = max_size
        self.side: Side | None = None      # None = idle

    def on_tick(self, state: MarketState, book: BookView) -> list[Order]:
        if self.side is None:
            if self.rng.random() >= self.start_prob:
                return []
            self.side = Side.BUY if self.rng.random() < 0.5 else Side.SELL
        elif self.rng.random() < self.end_prob:
            self.side = None
            return []
        if self.rng.random() >= self.trade_prob:
            return []
        qty = int(min(self.rng.geometric(self.p_size), self.max_size))
        return [Order(self.side, qty)]


class ValueTrader(Taker):
    """Sees V with noise; buys when the ask is at least `threshold` below its estimate,
    sells when the bid is at least `threshold` above it.

    Size grows with the edge: one unit per `threshold` ticks of mispricing, up to
    max_size. Like a real arbitrageur, it trades harder the more wrong the price is,
    which is what keeps the market price close to value.
    """
    kind = "value"

    def __init__(self, path: PricePath, activity: float = 0.15, noise: float = 2.0,
                 threshold: float = 2.0, max_size: int = 10, name: str | None = None) -> None:
        super().__init__(name)
        self.path = path
        self.activity, self.noise = activity, noise
        self.threshold, self.max_size = threshold, max_size

    def on_tick(self, state: MarketState, book: BookView) -> list[Order]:
        if self.rng.random() >= self.activity:
            return []
        estimate = self.path.V[state.t] + self.rng.normal(0.0, self.noise)
        ask, bid = book.best_ask(), book.best_bid()
        if ask is not None and ask <= estimate - self.threshold:
            return [Order(Side.BUY, self._size(estimate - ask),
                          limit=math.floor(estimate - self.threshold))]
        if bid is not None and bid >= estimate + self.threshold:
            return [Order(Side.SELL, self._size(bid - estimate),
                          limit=math.ceil(estimate + self.threshold))]
        return []

    def _size(self, edge: float) -> int:
        return int(max(1, min(self.max_size, edge // self.threshold)))


class MomentumTrader(Taker):
    """Trend follower: buys after the public mid rose by more than `threshold`
    over `lookback` steps, sells after it fell. Sees only public prices."""
    kind = "momentum"

    def __init__(self, activity: float = 0.05, lookback: int = 20, threshold: float = 2.0,
                 max_size: int = 3, name: str | None = None) -> None:
        super().__init__(name)
        self.activity, self.lookback = activity, lookback
        self.threshold, self.max_size = threshold, max_size

    def on_tick(self, state: MarketState, book: BookView) -> list[Order]:
        h = state.mid_history
        if len(h) <= self.lookback or self.rng.random() >= self.activity:
            return []
        move = h[-1] - h[-1 - self.lookback]
        qty = int(self.rng.integers(1, self.max_size + 1))
        if move > self.threshold:
            return [Order(Side.BUY, qty)]
        if move < -self.threshold:
            return [Order(Side.SELL, qty)]
        return []
