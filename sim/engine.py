"""The simulation loop.

Each step t:
  1. Market makers see public information up to t-1 (and the public reference
     price at t) and post quotes, in a random (seeded) order. Post-only quotes
     slide behind crossing quotes; aggressive ones trade (see sim/market.py).
  2. Takers act in a random (seeded) order; each sees the live quotes as earlier
     takers have depleted them, and its orders execute immediately.
  3. The public mid for step t is the mid of the book after customer flow.
  4. Every fill updates the ledger and is reported to both sides.
best_bid / best_ask record the quotes as posted (before customer flow).

Only the engine and a few privileged bots (value traders, the insider) can see
the hidden value V; the engine uses it only for evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bots.base import MarketMaker, MarketState, Taker
from sim.accounting import EXCHANGE_ID, Ledger
from sim.market import QuoteBook
from sim.price import PricePath
from sim.types import Fill, PublicTrade, Side


@dataclass
class SimConfig:
    taker_fee: float = 0.0
    maker_rebate: float = 0.0


@dataclass
class SimResult:
    path: PricePath
    reference: np.ndarray | None
    mid: np.ndarray
    best_bid: np.ndarray          # NaN where a side was empty
    best_ask: np.ndarray
    inventory: np.ndarray         # shape (n, n_agents), end of each step
    cash: np.ndarray
    agents: pd.DataFrame          # id, name, kind, role
    fills: pd.DataFrame
    ledger: Ledger = field(repr=False)

    @property
    def spread(self) -> np.ndarray:
        return self.best_ask - self.best_bid

    def equity(self, mark: str = "value") -> np.ndarray:
        """Mark-to-market equity per agent per step, marked at V ('value') or mid ('mid')."""
        price = self.path.V if mark == "value" else self.mid
        return self.cash + self.inventory * price[:, None]

    def final_pnl(self) -> pd.DataFrame:
        eq = self.equity("value")[-1]
        df = self.agents.copy()
        df["pnl"] = eq
        df["inventory"] = self.inventory[-1]
        df["volume"] = [self.ledger.volume[i] for i in df["id"]]
        return df


def _post_trade_mid(book: QuoteBook, trades: list[PublicTrade], previous: float) -> float:
    """Public mid at the end of a step, after customer flow.

    Taking liquidity moves the price mechanically: if buyers exhaust the best ask,
    the next ask is higher and so is the mid. If a whole side was exhausted, its
    effective price is the worst price traded on it this step. With no quotes and
    no trades on a side, the previous mid is kept.
    """
    bid, ask = book.best_bid(), book.best_ask()
    if bid is None:
        sells = [tr.price for tr in trades if tr.taker_side is Side.SELL]
        bid = min(sells) if sells else None
    if ask is None:
        buys = [tr.price for tr in trades if tr.taker_side is Side.BUY]
        ask = max(buys) if buys else None
    if bid is None or ask is None:
        return previous
    return (bid + ask) / 2


class Simulation:
    def __init__(self, path: PricePath, makers: list[MarketMaker], takers: list[Taker],
                 seed: int = 0, config: SimConfig | None = None,
                 reference: np.ndarray | None = None) -> None:
        if not makers:
            raise ValueError("need at least one market maker")
        self.path = path
        self.makers = makers
        self.takers = takers
        self.config = config or SimConfig()
        if reference is not None and len(reference) != path.n:
            raise ValueError("reference must have one value per step")
        self.reference = reference
        ss = np.random.SeedSequence(seed)
        child = ss.spawn(len(makers) + len(takers) + 1)
        self.rng = np.random.default_rng(child[0])
        self.agents: list = [*makers, *takers]
        for i, (agent, s) in enumerate(zip(self.agents, child[1:])):
            agent.bind(i, np.random.default_rng(s))

    def run(self) -> SimResult:
        n, A = self.path.n, len(self.agents)
        ledger = Ledger(self.config.taker_fee, self.config.maker_rebate)
        mid = np.empty(n)
        bb = np.full(n, np.nan)
        ba = np.full(n, np.nan)
        inv = np.zeros((n, A), dtype=np.int64)
        cash = np.zeros((n, A))
        fills: list[Fill] = []

        mid[0] = round(self.path.V[0])          # opening reference price
        last_bid, last_ask = None, None
        last_trades: tuple[PublicTrade, ...] = ()
        by_id = {a.agent_id: a for a in self.agents}

        def handle(new_fills: list[Fill], step_trades: list[PublicTrade]) -> None:
            for f in new_fills:
                ledger.apply(f)
                fills.append(f)
                step_trades.append(PublicTrade(f.price, f.qty, f.taker_side))
                by_id[f.maker_id].on_fill(f)
                by_id[f.taker_id].on_fill(f)

        for t in range(1, n):
            history = mid[:t]
            history.flags.writeable = False

            def state_for(agent) -> MarketState:
                return MarketState(t=t, mid=float(mid[t - 1]), best_bid=last_bid,
                                   best_ask=last_ask, mid_history=history,
                                   last_trades=last_trades,
                                   inventory=ledger.inventory[agent.agent_id],
                                   cash=ledger.cash[agent.agent_id],
                                   reference=None if self.reference is None
                                   else float(self.reference[t]))

            book = QuoteBook(self.rng)
            step_trades: list[PublicTrade] = []

            for i in self.rng.permutation(len(self.makers)):
                mm = self.makers[i]
                quote = mm.on_tick(state_for(mm))
                if quote is not None:
                    handle(book.post(t, mm.agent_id, mm.name, quote), step_trades)

            b, a = book.best_bid(), book.best_ask()
            bb[t] = np.nan if b is None else b
            ba[t] = np.nan if a is None else a

            for i in self.rng.permutation(len(self.takers)):
                tk = self.takers[i]
                for order in tk.on_tick(state_for(tk), book):
                    new_fills, _ = book.execute(t, tk.agent_id, tk.name, order)
                    handle(new_fills, step_trades)

            mid[t] = _post_trade_mid(book, step_trades, mid[t - 1])

            for j, agent in enumerate(self.agents):
                inv[t, j] = ledger.inventory[agent.agent_id]
                cash[t, j] = ledger.cash[agent.agent_id]
            last_bid, last_ask = b, a
            last_trades = tuple(step_trades)
            history.flags.writeable = True

        agents = pd.DataFrame({
            "id": [a.agent_id for a in self.agents],
            "name": [a.name for a in self.agents],
            "kind": [a.kind for a in self.agents],
            "role": ["maker"] * len(self.makers) + ["taker"] * len(self.takers),
        })
        fills_df = pd.DataFrame([{
            "t": f.t, "price": f.price, "qty": f.qty, "maker_id": f.maker_id,
            "taker_id": f.taker_id, "taker_side": f.taker_side.value,
            "maker": f.maker_kind, "taker": f.taker_kind,
        } for f in fills], columns=["t", "price", "qty", "maker_id", "taker_id",
                                     "taker_side", "maker", "taker"])
        assert EXCHANGE_ID not in agents["id"].values
        return SimResult(self.path, self.reference, mid, bb, ba, inv, cash, agents, fills_df, ledger)
