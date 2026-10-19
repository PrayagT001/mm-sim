"""Cash and inventory bookkeeping.

Every fill moves cash and inventory between exactly two agents, plus optional
fees: the taker pays `taker_fee` per unit to the exchange and the maker receives
`maker_rebate` per unit from it. The exchange is an account like any other, so
total cash and total inventory across all accounts are always zero.
"""
from __future__ import annotations

from collections import defaultdict

from sim.types import Fill

EXCHANGE_ID = -1


class Ledger:
    def __init__(self, taker_fee: float = 0.0, maker_rebate: float = 0.0) -> None:
        self.taker_fee = taker_fee
        self.maker_rebate = maker_rebate
        self.cash: defaultdict[int, float] = defaultdict(float)
        self.inventory: defaultdict[int, int] = defaultdict(int)
        self.volume: defaultdict[int, int] = defaultdict(int)

    def apply(self, fill: Fill) -> None:
        sign = fill.taker_side.value            # +1 taker bought, -1 taker sold
        notional = fill.price * fill.qty
        self.inventory[fill.taker_id] += sign * fill.qty
        self.inventory[fill.maker_id] -= sign * fill.qty
        self.cash[fill.taker_id] -= sign * notional
        self.cash[fill.maker_id] += sign * notional

        fee = self.taker_fee * fill.qty
        rebate = self.maker_rebate * fill.qty
        self.cash[fill.taker_id] -= fee
        self.cash[fill.maker_id] += rebate
        self.cash[EXCHANGE_ID] += fee - rebate

        self.volume[fill.taker_id] += fill.qty
        self.volume[fill.maker_id] += fill.qty

    def mark_to_market(self, agent_id: int, price: float) -> float:
        """Cash plus inventory valued at `price`."""
        return self.cash[agent_id] + self.inventory[agent_id] * price

    def total_cash(self) -> float:
        return sum(self.cash.values())

    def total_inventory(self) -> int:
        return sum(self.inventory.values())
