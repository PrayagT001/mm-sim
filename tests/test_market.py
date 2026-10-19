import numpy as np
import pytest

from sim.market import QuoteBook
from sim.types import Order, Quote, Side


def book(seed: int = 0) -> QuoteBook:
    return QuoteBook(np.random.default_rng(seed))


def test_empty_book_has_no_prices_and_fills_nothing():
    b = book()
    assert b.best_bid() is None and b.best_ask() is None
    fills, left = b.execute(0, 99, "noise", Order(Side.BUY, 4))
    assert fills == [] and left == 4


def test_buyer_takes_lowest_ask_at_the_maker_price():
    b = book()
    b.post(0, 1, "a", Quote(98, 103, 5, 5))
    b.post(0, 2, "b", Quote(99, 101, 5, 5))
    fills, left = b.execute(0, 99, "noise", Order(Side.BUY, 2))
    assert left == 0
    assert [(f.maker_id, f.price, f.qty) for f in fills] == [(2, 101, 2)]


def test_order_walks_levels_and_reports_unfilled_remainder():
    b = book()
    b.post(0, 1, "a", Quote(None, 101, 0, 3))
    b.post(0, 2, "b", Quote(None, 102, 0, 4))
    fills, left = b.execute(0, 99, "noise", Order(Side.BUY, 10))
    assert [(f.price, f.qty) for f in fills] == [(101, 3), (102, 4)]
    assert left == 3
    assert b.best_ask() is None


def test_limit_price_is_respected():
    b = book()
    b.post(0, 1, "a", Quote(None, 101, 0, 3))
    b.post(0, 2, "b", Quote(None, 103, 0, 3))
    fills, left = b.execute(0, 99, "value", Order(Side.BUY, 6, limit=102))
    assert [(f.price, f.qty) for f in fills] == [(101, 3)]
    assert left == 3


def test_sell_side_mirrors_buy_side():
    b = book()
    b.post(0, 1, "a", Quote(100, None, 3, 0))
    b.post(0, 2, "b", Quote(99, None, 3, 0))
    fills, _ = b.execute(0, 99, "noise", Order(Side.SELL, 4))
    assert [(f.price, f.qty) for f in fills] == [(100, 3), (99, 1)]
    assert all(f.taker_side is Side.SELL for f in fills)


def test_ties_are_split_randomly_but_reproducibly():
    def who_fills_first(seed):
        b = book(seed)
        b.post(0, 1, "a", Quote(None, 101, 0, 5))
        b.post(0, 2, "b", Quote(None, 101, 0, 5))
        return b.execute(0, 99, "noise", Order(Side.BUY, 1))[0][0].maker_id

    firsts = [who_fills_first(s) for s in range(200)]
    assert firsts == [who_fills_first(s) for s in range(200)]      # deterministic per seed
    assert 0.35 < np.mean(np.array(firsts) == 1) < 0.65            # no maker favoured


def test_no_self_trade():
    b = book()
    b.post(0, 1, "a", Quote(None, 101, 0, 5))
    fills, left = b.execute(0, 1, "a", Order(Side.BUY, 2))
    assert fills == [] and left == 2


def test_post_only_quote_slides_behind_instead_of_trading():
    b = book()
    b.post(0, 1, "a", Quote(99, 101, 5, 5))
    fills = b.post(0, 2, "b", Quote(102, 104, 5, 5))      # bid 102 would cross ask 101
    assert fills == []
    assert b.best_bid() == 100                              # moved to one tick behind 101
    assert b.best_ask() == 101


def test_aggressive_quote_trades_at_resting_price_and_rests_remainder():
    b = book()
    b.post(0, 1, "a", Quote(99, 101, 5, 3))
    fills = b.post(0, 2, "insider", Quote(102, 105, 5, 5, post_only=False))
    assert [(f.maker_id, f.taker_id, f.price, f.qty) for f in fills] == [(1, 2, 101, 3)]
    assert b.best_bid() == 102                              # 2 units left resting


def test_book_never_crossed_after_post_only_quotes():
    rng = np.random.default_rng(0)
    for trial in range(200):
        b = book(trial)
        for maker in range(5):
            mid = 1000 + rng.integers(-5, 6)
            b.post(0, maker, "m", Quote(int(mid - 1), int(mid + 1), 5, 5))
        assert b.best_bid() < b.best_ask()


@pytest.mark.parametrize("quote", [
    Quote(101, 100),              # own quote crossed
    Quote(100, 100),              # own quote locked
    Quote(99.5, 101),             # float price
    Quote(99, 101, -1, 5),        # negative size
])
def test_invalid_quotes_rejected(quote):
    with pytest.raises((ValueError, TypeError)):
        book().post(0, 1, "a", quote)


def test_non_positive_order_rejected():
    with pytest.raises(ValueError):
        book().execute(0, 1, "a", Order(Side.BUY, 0))
