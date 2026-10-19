"""Realism report: does the simulated market behave like a real one?

Runs the default population over many seeds and checks it against well-known
properties of real markets (stylised facts), price discovery, and who should win
or lose. Prints a pass/fail table and saves figures to figures/.

Usage:  python -m experiments.validate_market [--seeds 20] [--steps 20000]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from sim import metrics as M
from sim.price import oracle_r2
from sim.scenarios import run

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


def measure(seed: int, steps: int) -> dict:
    res = run(seed, n=steps)
    r1 = M.returns(res.mid, 1)
    r10 = M.returns(res.mid, 10)
    signs = M.trade_signs(res)
    pnl = M.pnl_by_kind(res)
    mk = M.weighted_markout(M.markouts(res, horizons=(30,)), ["taker_kind"]).set_index("taker_kind")
    spreads = M.spread_by_regime(res)
    track = M.tracking(res)
    impact = M.impact_by_size(res)
    impact = impact[impact["n"] >= 100]           # ignore thin size buckets
    volume = res.fills.groupby("taker")["qty"].sum()
    takers = res.agents.set_index("name")["kind"]
    vol_by_kind = volume.groupby(takers.reindex(volume.index).to_numpy()).sum()
    out = {
        "seed": seed,
        "ret1_ac1": M.autocorr(r1, 1)[0],
        "ret10_ac1": M.autocorr(r10, 1)[0],
        "v_ret10_ac1": M.autocorr(M.returns(res.path.V, 10), 1)[0],
        "v_sd30": M.returns(res.path.V, 30).std(),
        "ret1_kurt": M.excess_kurtosis(r1),
        "ret10_kurt": M.excess_kurtosis(r10),
        "absret_ac1": M.autocorr(np.abs(r1), 1)[0],
        "absret_ac50": M.autocorr(np.abs(r1), 50)[0],
        "sign_ac1": M.autocorr(signs, 1)[0],
        "sign_ac20": M.autocorr(signs, 20)[0],
        "spread_calm": spreads["calm"],
        "spread_stormy": spreads["stormy"],
        "track_mae": track["mean_abs_error"],
        "mid_err_sd": float(np.std(res.mid - res.path.V)),
        "ref_err_sd": float(np.std(res.reference - res.path.V)),
        "ref_corr30": float(np.corrcoef(M.returns(res.reference, 30), M.returns(res.path.V, 30))[0, 1]),
        "track_corr30": track["corr_30step_changes"],
        "impact_small": impact["impact"].iloc[0],
        "impact_large": impact["impact"].iloc[-1],
        "size_small": impact["size"].iloc[0],
        "size_large": impact["size"].iloc[-1],
        "oracle_r2_10": oracle_r2(res.path, 10),
        "informed_share": vol_by_kind.get("value", 0) / vol_by_kind.sum(),
        "cash_total": res.ledger.total_cash(),
        "inv_total": res.ledger.total_inventory(),
    }
    for kind in ("noise", "value", "naive", "avellaneda", "flow", "insider"):
        out[f"ppu_{kind}"] = pnl["pnl_per_unit"].get(kind, np.nan)
    for kind in ("noise", "value"):
        out[f"markout30_{kind}"] = mk["markout"].get(kind, np.nan)
    return out


CHECKS = [
    # (label, real-market expectation, test on the across-seed mean)
    ("Accounting: total cash and inventory are zero", "exactly 0",
     lambda m: abs(m.cash_total) < 1e-6 and m.inv_total == 0),
    ("1-step returns barely autocorrelated", "|ac| < 0.15 (slightly negative: bid-ask bounce)",
     lambda m: abs(m.ret1_ac1) < 0.15),
    ("Market adds no predictability beyond V's designed pattern",
     "10-step return ac of the mid within 0.05 of V's own",
     lambda m: abs(m.ret10_ac1 - m.v_ret10_ac1) < 0.05),
    ("Fat tails", "excess kurtosis > 1 at 1 and 10 steps",
     lambda m: m.ret1_kurt > 1 and m.ret10_kurt > 1),
    ("Volatility clustering", "ac of |returns| > 0.05 at lag 1 and still > 0 at lag 50",
     lambda m: m.absret_ac1 > 0.05 and m.absret_ac50 > 0),
    ("Persistent order flow", "trade-sign ac > 0.1 at lag 1, > 0 at lag 20",
     lambda m: m.sign_ac1 > 0.1 and m.sign_ac20 > 0),
    ("Spreads widen in stormy regimes", "stormy spread > calm spread",
     lambda m: m.spread_stormy > m.spread_calm),
    ("Price discovery: the market is as informative as the best public signal",
     "sd(mid - V) <= 1.2 x sd(ref - V); 30-step corr with V within 0.05 of the reference's",
     lambda m: m.mid_err_sd <= 1.2 * m.ref_err_sd and m.track_corr30 >= m.ref_corr30 - 0.05),
    ("Concave price impact", "bigger flow moves price more, but less than proportionally",
     lambda m: m.impact_large > m.impact_small
     and m.impact_large / m.impact_small < m.size_large / m.size_small),
    ("Noise traders lose, informed traders win", "PnL per unit: noise < 0 < value",
     lambda m: m.ppu_noise < 0 < m.ppu_value),
    ("Noise flow is benign, informed flow toxic", "30-step maker markout: vs noise > 0 > vs value",
     lambda m: m.markout30_noise > 0 > m.markout30_value),
    ("Insider beats the market maker it is built on", "PnL per unit: insider > avellaneda",
     lambda m: m.ppu_insider > m.ppu_avellaneda),
]


def report(df: pd.DataFrame) -> pd.DataFrame:
    m = df.mean(numeric_only=True)
    rows = [{"check": label, "expectation": expect, "pass": bool(test(m))}
            for label, expect, test in CHECKS]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- figures
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def figures(seed: int, steps: int) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(exist_ok=True)
    res = run(seed, n=steps)
    paths = []

    # 1. Price vs hidden value
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=SURFACE)
    lo, hi = 2000, 6000
    t = np.arange(lo, hi)
    storm = res.path.regime[lo:hi] == 1
    ax.fill_between(t, 0, 1, where=storm, transform=ax.get_xaxis_transform(),
                    color=GRID, alpha=0.7, linewidth=0, label="stormy regime")
    ax.plot(t, res.path.V[lo:hi], color=INK, linewidth=1.2, label="hidden value V")
    ax.plot(t, res.mid[lo:hi], color=BLUE, linewidth=1.2, label="market mid")
    ax.plot(t, res.reference[lo:hi], color=ORANGE, linewidth=0.8, alpha=0.8,
            label="reference price (other venue)")
    _style(ax, "The market price tracks the hidden value", "step (1 step = 1 second)", "price (ticks)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=4)
    fig.tight_layout()
    p = FIG_DIR / "realism_price_vs_value.png"
    fig.savefig(p, dpi=150)
    paths.append(p)
    plt.close(fig)

    # 2. Stylised facts
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), facecolor=SURFACE)
    r10 = M.returns(res.mid, 10)
    z = (r10 - r10.mean()) / r10.std()
    edges = np.linspace(-8, 8, 61)
    dens, _ = np.histogram(z, bins=edges, density=True)
    centres = (edges[:-1] + edges[1:]) / 2
    axes[0].semilogy(centres[dens > 0], dens[dens > 0], color=BLUE, linewidth=2, label="simulated")
    axes[0].semilogy(centres, np.exp(-centres ** 2 / 2) / np.sqrt(2 * np.pi), color=INK2,
                     linewidth=1, linestyle="--", label="normal")
    axes[0].set_ylim(1e-5, 1)
    _style(axes[0], "Fat tails (10-step returns)", "standardised return", "density (log)")
    axes[0].legend(frameon=False, fontsize=8)

    r1 = M.returns(res.mid, 1)
    lags = np.arange(1, 101)
    axes[1].plot(lags, M.autocorr(np.abs(r1), lags), color=BLUE, linewidth=2, label="|returns|")
    axes[1].plot(lags, M.autocorr(r1, lags), color=ORANGE, linewidth=2, label="returns")
    axes[1].axhline(0, color=INK2, linewidth=0.8)
    _style(axes[1], "Volatility clusters, returns don't", "lag (steps)", "autocorrelation")
    axes[1].legend(frameon=False, fontsize=8)

    signs = M.trade_signs(res)
    tl = np.arange(1, 51)
    axes[2].plot(tl, M.autocorr(signs, tl), color=AQUA, linewidth=2)
    axes[2].axhline(0, color=INK2, linewidth=0.8)
    _style(axes[2], "Order flow is persistent", "lag (trades)", "trade-sign autocorrelation")
    fig.tight_layout()
    p = FIG_DIR / "realism_stylised_facts.png"
    fig.savefig(p, dpi=150)
    paths.append(p)
    plt.close(fig)

    # 3. Markouts by counterparty
    hs = (1, 2, 5, 10, 20, 30, 60, 120)
    mk = M.weighted_markout(M.markouts(res, horizons=hs), ["taker_kind", "h"])
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    colors = {"noise": BLUE, "value": ORANGE, "insider": AQUA}
    labels = {"noise": "vs noise traders", "value": "vs value traders",
              "insider": "vs the insider (when it takes)"}
    for kind, color in colors.items():
        d = mk[mk.taker_kind == kind]
        if len(d):
            ax.plot(d["h"], d["markout"], color=color, linewidth=2, marker="o", markersize=4,
                    label=labels[kind])
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xscale("log")
    _style(ax, "Market makers' markouts by who they traded with",
           "horizon H (steps, log scale)", "maker markout (ticks per unit)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = FIG_DIR / "realism_markouts.png"
    fig.savefig(p, dpi=150)
    paths.append(p)
    plt.close(fig)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(measure, range(args.seeds), [args.steps] * args.seeds))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 70)
    print("Mean across seeds (std in brackets):")
    summary = pd.DataFrame({"mean": df.mean(numeric_only=True), "std": df.std(numeric_only=True)})
    print(summary.drop(index="seed").round(3).to_string())
    print()
    rep = report(df)
    print(rep.to_string(index=False))
    print(f"\n{rep['pass'].sum()} / {len(rep)} checks pass")
    if not args.no_figures:
        for p in figures(0, args.steps):
            print("saved", p)


if __name__ == "__main__":
    main()
