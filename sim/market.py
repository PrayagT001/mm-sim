"""Dealer market for one step.

Market makers post two-sided quotes. Customers then send IOC orders that trade
against the best quotes. Rules (fixed for every experiment):

1. Best price first. Buyers take the lowest ask, sellers the highest bid.
2. Ties at the same price are broken by a random priority drawn once per quote
   per step from the engine's seeded RNG. This spreads flow fairly between
   market makers at the same price without favouring whoever was listed first.
3. Every trade happens at the resting (maker) price.
4. Post-only quotes (the default) never trade on arrival: a price that would
   cross another maker's quote is moved to one tick behind the best opposite
   quote. A quote with post_only=False trades against crossing quotes as the
   taker, and any remainder rests.
5. An agent never trades with itself (its own quotes are skipped).
6. Unfilled remainder of a customer order is cancelled; quotes expire at the end
   of the step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.types import Fill, Order, Quote, Side


@dataclass(slots=True)
class RestingQuote:
    maker_id: int
    maker_kind: str
    side: Side
    price: int
    size: int          # remaining
    priority: float    # random tie-break key, lower goes first


class QuoteBook:
    """All resting quotes for the current step."""

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._bids: list[RestingQuote] = []
        self._asks: list[RestingQuote] = []

    # ------------------------------------------------------------------ queries
    def _side_list(self, side: Side) -> list[RestingQuote]:
        return self._bids if side is Side.BUY else self._asks

    def _sorted(self, side: Side) -> list[RestingQuote]:
        """Resting quotes on a side, best first (best price, then priority)."""
        quotes = [q for q in self._side_list(side) if q.size > 0]
        if side is Side.BUY:
            return sorted(quotes, key=lambda q: (-q.price, q.priority))
        return sorted(quotes, key=lambda q: (q.price, q.priority))

    def best(self, side: Side, exclude: int | None = None) -> int | None:
        """Best price on a side, optionally ignoring one agent's own quotes."""
        prices = [q.price for q in self._side_list(side)
                  if q.size > 0 and q.maker_id != exclude]
        if not prices:
            return None
        return max(prices) if side is Side.BUY else min(prices)

    def best_bid(self, exclude: int | None = None) -> int | None:
        return self.best(Side.BUY, exclude)

    def best_ask(self, exclude: int | None = None) -> int | None:
        return self.best(Side.SELL, exclude)

    def depth(self, side: Side) -> int:
        return sum(q.size for q in self._side_list(side))

    # ----------------------------------------------------------------- commands
    def post(self, t: int, maker_id: int, maker_kind: str, quote: Quote) -> list[Fill]:
        """Post a maker's quote. Crossing parts trade at once (maker becomes taker)."""
        _validate_quote(quote)
        fills: list[Fill] = []
        for side, price, size in ((Side.BUY, quote.bid, quote.bid_size),
                                  (Side.SELL, quote.ask, quote.ask_size)):
            if price is None or size == 0:
                continue
            if quote.post_only:
                opposite = self.best(side.opposite(), exclude=maker_id)
                if opposite is not None:
                    if side is Side.BUY and price >= opposite:
                        price = opposite - 1
                    elif side is Side.SELL and price <= opposite:
                        price = opposite + 1
            order = Order(side=side, qty=size, limit=price)
            new_fills, remaining = self.execute(t, maker_id, maker_kind, order)
            fills.extend(new_fills)
            if remaining > 0:
                self._side_list(side).append(RestingQuote(
                    maker_id, maker_kind, side, price, remaining, float(self._rng.random())))
        return fills

    def execute(self, t: int, taker_id: int, taker_kind: str,
                order: Order) -> tuple[list[Fill], int]:
        """Match an IOC order against the opposite side. Returns (fills, unfilled qty)."""
        if order.qty <= 0:
            raise ValueError("order qty must be positive")
        fills: list[Fill] = []
        remaining = order.qty
        for rq in self._sorted(order.side.opposite()):
            if remaining == 0:
                break
            if rq.maker_id == taker_id:
                continue                       # rule 5: no self-trades
            if order.limit is not None:
                if order.side is Side.BUY and rq.price > order.limit:
                    break
                if order.side is Side.SELL and rq.price < order.limit:
                    break
            qty = min(remaining, rq.size)
            rq.size -= qty
            remaining -= qty
            fills.append(Fill(t=t, price=rq.price, qty=qty, maker_id=rq.maker_id,
                              taker_id=taker_id, taker_side=order.side,
                              maker_kind=rq.maker_kind, taker_kind=taker_kind))
        return fills, remaining


def _validate_quote(q: Quote) -> None:
    for name, v in (("bid", q.bid), ("ask", q.ask)):
        if v is not None and not isinstance(v, (int, np.integer)):
            raise TypeError(f"{name} must be an integer tick, got {v!r}")
    if q.bid_size < 0 or q.ask_size < 0:
        raise ValueError("quote sizes must be non-negative")
    if q.bid is not None and q.ask is not None and q.bid >= q.ask:
        raise ValueError(f"own quote is crossed: bid {q.bid} >= ask {q.ask}")
