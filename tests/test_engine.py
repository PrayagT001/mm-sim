import numpy as np
import pandas as pd
import pytest

from sim.accounting import EXCHANGE_ID, Ledger
from sim.engine import SimConfig
from sim.scenarios import Population, run
from sim.types import Fill, Side


@pytest.fixture(scope="module")
def result():
    return run(seed=11, n=3000)


def test_ledger_conserves_cash_and_inventory_with_fees():
    led = Ledger(taker_fee=0.3, maker_rebate=0.1)
    led.apply(Fill(t=1, price=100, qty=2, maker_id=1, taker_id=2, taker_side=Side.BUY,
                   maker_kind="m", taker_kind="t"))
    assert led.inventory[2] == 2 and led.inventory[1] == -2
    assert led.cash[2] == pytest.approx(-200 - 0.6)
    assert led.cash[1] == pytest.approx(200 + 0.2)
    assert led.cash[EXCHANGE_ID] == pytest.approx(0.4)
    assert led.total_cash() == pytest.approx(0.0)
    assert led.total_inventory() == 0


def test_simulation_conserves_cash_and_inventory(result):
    assert result.ledger.total_inventory() == 0
    assert result.ledger.total_cash() == pytest.approx(0.0, abs=1e-6)
    assert np.all(result.inventory.sum(axis=1) == 0)


def test_conservation_holds_with_fees():
    res = run(seed=3, n=1500, config=SimConfig(taker_fee=0.5, maker_rebate=0.2))
    assert res.ledger.total_cash() == pytest.approx(0.0, abs=1e-6)
    assert res.ledger.cash[EXCHANGE_ID] > 0


def test_shares_bought_equal_shares_sold(result):
    f = result.fills
    bought = f.loc[f.taker_side == 1, "qty"].sum() + f.loc[f.taker_side == -1, "qty"].sum()
    assert bought == f["qty"].sum()       # every unit has exactly one buyer and one seller
    per_agent = result.inventory[-1]
    assert per_agent[per_agent > 0].sum() == -per_agent[per_agent < 0].sum()


def test_same_seed_same_result():
    a, b = run(seed=5, n=1500), run(seed=5, n=1500)
    pd.testing.assert_frame_equal(a.fills, b.fills)
    assert np.array_equal(a.mid, b.mid)


def test_different_seed_different_result():
    a, b = run(seed=5, n=1500), run(seed=6, n=1500)
    assert not np.array_equal(a.mid, b.mid)


def test_quotes_never_crossed(result):
    both = ~np.isnan(result.best_bid) & ~np.isnan(result.best_ask)
    assert both.mean() > 0.95
    assert np.all(result.best_bid[both] < result.best_ask[both])


def test_market_makers_respect_inventory_limits(result):
    makers = result.agents.index[result.agents.role == "maker"]
    assert np.abs(result.inventory[:, makers]).max() <= 20


def test_trades_happen_at_the_makers_quoted_price(result):
    makers = set(result.agents.loc[result.agents.role == "maker", "id"])
    f = result.fills[~result.fills["taker_id"].isin(makers)]    # customer trades only
    t = f["t"].to_numpy()
    buys = f["taker_side"].to_numpy() == 1
    # Customers trade after all quotes are posted, so a customer buy can never be
    # cheaper than that step's best ask (nor a sell dearer than the best bid).
    assert np.all(f["price"].to_numpy()[buys] >= result.best_ask[t[buys]])
    assert np.all(f["price"].to_numpy()[~buys] <= result.best_bid[t[~buys]])


def test_insider_signal_rate_matches_p():
    from bots.insider import InsiderMM
    from sim.engine import Simulation
    from sim.price import simulate_price

    path = simulate_price(20_000, rng=np.random.default_rng(0))
    insider = InsiderMM(path, every=50, p=0.3)
    Simulation(path, [insider], [], seed=1).run()
    rate = insider.signals / (20_000 // 50)
    assert 0.24 < rate < 0.36


def test_removing_the_reference_still_runs():
    res = run(seed=2, n=1000, pop=Population(use_reference=False))
    assert res.reference is None
    assert np.isfinite(res.mid).all()
