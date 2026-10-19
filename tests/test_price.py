import numpy as np
import pytest

from sim.price import CALM, STORMY, PriceParams, oracle_r2, reference_feed, simulate_price


def test_same_seed_same_path():
    a = simulate_price(2000, rng=np.random.default_rng(7))
    b = simulate_price(2000, rng=np.random.default_rng(7))
    assert np.array_equal(a.V, b.V)
    assert np.array_equal(a.regime, b.regime)


def test_both_regimes_occur_and_stormy_is_more_volatile():
    p = simulate_price(50_000, PriceParams(jump_prob=0.0), np.random.default_rng(1))
    assert set(np.unique(p.regime)) == {CALM, STORMY}
    dv = np.diff(p.V)
    regime_next = p.regime[1:]
    assert dv[regime_next == STORMY].std() > 2 * dv[regime_next == CALM].std()


def test_jumps_are_rare():
    p = simulate_price(50_000, rng=np.random.default_rng(2))
    rate = np.mean(p.jump != 0)
    assert 0.0002 < rate < 0.003


def test_oracle_r2_is_in_the_designed_band():
    """Partly learnable, mostly unpredictable: about 10-15% at a 10-step horizon."""
    r2 = np.mean([oracle_r2(simulate_price(50_000, rng=np.random.default_rng(s)), 10)
                  for s in range(3)])
    assert 0.08 < r2 < 0.20


def test_no_mean_reversion_means_no_oracle_edge():
    p = simulate_price(50_000, PriceParams(kappa=0.0), np.random.default_rng(3))
    assert abs(oracle_r2(p, 10)) < 0.01


def test_reference_is_lagged_value_when_noise_free():
    p = simulate_price(500, rng=np.random.default_rng(4))
    ref = reference_feed(p, lag=5, noise=0.0, rng=np.random.default_rng(0))
    assert np.allclose(ref[5:], p.V[:-5])


def test_too_short_path_rejected():
    with pytest.raises(ValueError):
        simulate_price(1)
