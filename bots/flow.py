"""Flow-reactive market maker (price impact from order flow).

Fair value = public fair value + impact * sign(net) * sqrt(|net|), where net is the
net taker volume in the last step, on top of the Avellaneda-Stoikov inventory
skew. After net buying it quotes higher; after net selling, lower. Its quotes feed
into the next public mid, so order flow moves the price, which is how a market
learns from informed traders.

Why the square root: Kyle's model has impact proportional to flow, but real
markets show impact growing roughly with the square root of size (Bouchaud et
al., Trades, Quotes and Prices). A linear rule made the simulated impact linear.

This bot is part of the default population as "the rest of the professional
market": without some flow-reactive liquidity, nothing moves the price towards
the hidden value except inventory limits.
"""
from __future__ import annotations

import math

from bots.avellaneda import AvellanedaStoikovMM
from bots.base import MarketState


class FlowMM(AvellanedaStoikovMM):
    kind = "flow"

    def __init__(self, impact: float = 0.25, name: str | None = None, **as_kwargs) -> None:
        super().__init__(name=name, **as_kwargs)
        self.impact = impact

    def fair_value(self, state: MarketState) -> float:
        net = sum(tr.taker_side.value * tr.qty for tr in state.last_trades)
        shift = math.copysign(math.sqrt(abs(net)), net) if net else 0.0
        return super().fair_value(state) + self.impact * shift
