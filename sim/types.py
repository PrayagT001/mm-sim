"""Shared data types for the dealer-market simulator.

Prices are integer ticks. Quantities are positive integers.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Side(Enum):
    BUY = 1
    SELL = -1

    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


@dataclass(frozen=True, slots=True)
class Quote:
    """A market maker's two-sided quote for one step. None = no quote on that side.

    post_only (the default, as most real market makers use): a price that would
    cross another maker's quote is moved to one tick behind it instead of trading.
    post_only=False lets the quote trade against crossing quotes as a taker.
    """
    bid: int | None
    ask: int | None
    bid_size: int = 5
    ask_size: int = 5
    post_only: bool = True


@dataclass(frozen=True, slots=True)
class Order:
    """A taker order. limit None = market order. Unfilled remainder is cancelled (IOC)."""
    side: Side
    qty: int
    limit: int | None = None


@dataclass(frozen=True, slots=True)
class Fill:
    """One trade. The maker is the resting quote; the taker is the agent that crossed it."""
    t: int
    price: int
    qty: int
    maker_id: int
    taker_id: int
    taker_side: Side          # aggressor side
    maker_kind: str
    taker_kind: str

    def side_for(self, agent_id: int) -> Side:
        """Which side of this trade the given agent was on."""
        if agent_id == self.taker_id:
            return self.taker_side
        if agent_id == self.maker_id:
            return self.taker_side.opposite()
        raise ValueError(f"agent {agent_id} is not part of this fill")


@dataclass(frozen=True, slots=True)
class PublicTrade:
    """What everyone sees about a trade: no identities."""
    price: int
    qty: int
    taker_side: Side
