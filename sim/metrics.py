"""Measurements on a finished simulation: realism checks and evaluation metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sim.engine import SimResult


# --------------------------------------------------------------- price statistics
def returns(prices: np.ndarray, horizon: int = 1) -> np.ndarray:
    """Non-overlapping price changes over `horizon` steps (ticks, not percent)."""
    p = prices[::horizon]
    return np.diff(p)


def autocorr(x: np.ndarray, lags: int | list[int]) -> np.ndarray:
    """Sample autocorrelation at the given lag(s)."""
    lags = [lags] if isinstance(lags, int) else list(lags)
    x = np.asarray(x, dtype=float) - np.mean(x)
    denom = np.dot(x, x)
    return np.array([np.dot(x[:-k], x[k:]) / denom for k in lags])


def excess_kurtosis(x: np.ndarray) -> float:
    """0 for a normal distribution; positive means fat tails."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    return float(np.mean(x ** 4) / np.mean(x ** 2) ** 2 - 3.0)


# ----------------------------------------------------------------- flow statistics
def trade_signs(res: SimResult) -> np.ndarray:
    """+1 for each buyer-initiated trade, -1 for seller-initiated, in time order."""
    return res.fills["taker_side"].to_numpy()


def net_flow(res: SimResult) -> np.ndarray:
    """Net signed taker volume per step."""
    flow = np.zeros(res.path.n)
    np.add.at(flow, res.fills["t"].to_numpy(), (res.fills["taker_side"] * res.fills["qty"]).to_numpy())
    return flow


def impact_by_size(res: SimResult, horizon: int = 10,
                   bins: tuple[int, ...] = (1, 2, 3, 5, 8, 13)) -> pd.DataFrame:
    """Average mid move in the direction of net flow, by size of the net flow in a step.

    Real markets: impact grows with size but less than proportionally (concave).
    """
    flow = net_flow(res)
    t = np.nonzero(flow)[0]
    t = t[(t >= 1) & (t + horizon < res.path.n)]
    move = (res.mid[t + horizon] - res.mid[t - 1]) * np.sign(flow[t])
    size = np.abs(flow[t])
    labels = np.digitize(size, bins[1:])
    df = pd.DataFrame({"size": size, "move": move, "bin": labels})
    out = df.groupby("bin").agg(size=("size", "mean"), impact=("move", "mean"), n=("move", "size"))
    return out.reset_index(drop=True)


# ----------------------------------------------------------- market-maker metrics
def markouts(res: SimResult, horizons: tuple[int, ...] = (1, 5, 10, 30, 60),
             against: str = "value") -> pd.DataFrame:
    """Per-fill markout from the MAKER's side, in ticks per unit.

    markout(H) = maker_side * (price_ref[t+H] - trade_price), maker_side = -taker_side.
    against='value' uses the hidden V (truth); against='mid' uses the public mid,
    which is what a real firm can measure.
    """
    ref = res.path.V if against == "value" else res.mid
    f = res.fills
    kinds = res.agents.set_index("id")["kind"]
    rows = []
    n = res.path.n
    for h in horizons:
        idx = np.minimum(f["t"].to_numpy() + h, n - 1)
        m = -f["taker_side"].to_numpy() * (ref[idx] - f["price"].to_numpy())
        rows.append(pd.DataFrame({
            "h": h, "markout": m, "qty": f["qty"].to_numpy(),
            "maker": f["maker"].to_numpy(), "taker": f["taker"].to_numpy(),
            "maker_kind": kinds.loc[f["maker_id"]].to_numpy(),
            "taker_kind": kinds.loc[f["taker_id"]].to_numpy(),
        }))
    return pd.concat(rows, ignore_index=True)


def weighted_markout(mk: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Quantity-weighted mean markout grouped by the given columns."""
    g = mk.assign(w=mk["markout"] * mk["qty"]).groupby(by)
    return (g["w"].sum() / g["qty"].sum()).rename("markout").reset_index()


def pnl_by_kind(res: SimResult) -> pd.DataFrame:
    """Final PnL marked at V, total and per unit traded, by participant kind."""
    p = res.final_pnl().groupby("kind")[["pnl", "volume"]].sum()
    p["pnl_per_unit"] = p["pnl"] / p["volume"].where(p["volume"] > 0)
    return p


def spread_by_regime(res: SimResult) -> dict[str, float]:
    s = res.spread
    reg = res.path.regime
    return {"calm": float(np.nanmean(s[reg == 0])), "stormy": float(np.nanmean(s[reg == 1]))}


def tracking(res: SimResult) -> dict[str, float]:
    """How closely the public price follows the hidden value."""
    err = res.mid - res.path.V
    dv = returns(res.path.V, 30)
    dm = returns(res.mid, 30)
    return {
        "mean_abs_error": float(np.mean(np.abs(err))),
        "p99_abs_error": float(np.percentile(np.abs(err), 99)),
        "corr_30step_changes": float(np.corrcoef(dv, dm)[0, 1]),
    }
