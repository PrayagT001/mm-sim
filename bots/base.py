"""The interface every participant follows.

Market makers return one Quote per step; takers return a list of IOC orders.
Both get fills through on_fill. Each agent receives its own seeded random
generator from the engine, so results do not depend on the order agents run in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from sim.types import Fill, Order, PublicTrade, Quote


@dataclass(frozen=True, slots=True)
class MarketState:
    """Everything a participant may legally see at the start of step t."""
    t: int
    mid: float                         # public mid at the end of step t-1
    best_bid: int | None               # public best quotes at step t-1
    best_ask: int | None
    mid_history: np.ndarray            # public mids for steps 0..t-1 (read-only)
    last_trades: tuple[PublicTrade, ...]   # trades during step t-1, no identities
    inventory: int                     # this agent's own position
    cash: float                        # this agent's own cash
    reference: float | None = None     # public reference price (other venue), if any


class BookView(Protocol):
    """Read-only view of the live quotes, given to takers when they act."""
    def best_bid(self, exclude: int | None = None) -> int | None: ...
    def best_ask(self, exclude: int | None = None) -> int | None: ...


class Agent:
    kind = "agent"

    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.kind
        self.agent_id = -100
        self.rng: np.random.Generator = np.random.default_rng(0)

    def bind(self, agent_id: int, rng: np.random.Generator) -> None:
        """Called once by the engine before the run starts."""
        self.agent_id = agent_id
        self.rng = rng

    def on_fill(self, fill: Fill) -> None:
        """Called after each of this agent's fills. Default: ignore."""


class MarketMaker(Agent):
    kind = "mm"
    max_inventory = 20

    def sizes(self, inventory: int, size: int) -> tuple[int, int]:
        """Quote sizes that can never take the position beyond +/- max_inventory."""
        bid_size = max(0, min(size, self.max_inventory - inventory))
        ask_size = max(0, min(size, self.max_inventory + inventory))
        return bid_size, ask_size

    def on_tick(self, state: MarketState) -> Quote | None:
        raise NotImplementedError


def anchored_mid(state: MarketState, ref_weight: float) -> float:
    """Fair value from public data: a blend of this market's mid and the reference price."""
    if state.reference is None or ref_weight == 0.0:
        return state.mid
    return ref_weight * state.reference + (1.0 - ref_weight) * state.mid


class Taker(Agent):
    kind = "taker"

    def on_tick(self, state: MarketState, book: BookView) -> list[Order]:
        raise NotImplementedError
