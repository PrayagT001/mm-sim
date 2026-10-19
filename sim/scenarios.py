"""Ready-made market populations, so experiments differ only in what they vary."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bots.avellaneda import AvellanedaStoikovMM
from bots.base import MarketMaker, Taker
from bots.flow import FlowMM
from bots.insider import InsiderMM
from bots.naive import NaiveMM
from bots.takers import MomentumTrader, NoiseTrader, ValueTrader
from sim.engine import SimConfig, SimResult, Simulation
from sim.price import PriceParams, PricePath, reference_feed, simulate_price


@dataclass
class Population:
    n_noise: int = 8
    noise_start: float = 0.01              # chance an idle noise trader starts a run
    value_noises: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)
    value_activity: float = 0.15
    value_threshold: float = 2.0
    n_momentum: int = 0
    n_flow: int = 2                         # background professional liquidity
    flow_impact: float = 0.25
    include_naive: bool = True
    include_avellaneda: bool = True
    include_insider: bool = True
    insider_p: float = 0.3
    insider_noise: float = 1.0
    insider_width: float = 2.0
    extra_makers: list[MarketMaker] = field(default_factory=list)
    reference_lag: int = 5
    reference_noise: float = 1.0
    use_reference: bool = True


def build(path: PricePath, pop: Population) -> tuple[list[MarketMaker], list[Taker]]:
    makers: list[MarketMaker] = [FlowMM(impact=pop.flow_impact, name=f"flow_{i}")
                                 for i in range(pop.n_flow)]
    if pop.include_naive:
        makers.append(NaiveMM(name="naive"))
    if pop.include_avellaneda:
        makers.append(AvellanedaStoikovMM(name="avellaneda"))
    if pop.include_insider:
        makers.append(InsiderMM(path, p=pop.insider_p, noise=pop.insider_noise,
                                width=pop.insider_width, name="insider"))
    makers.extend(pop.extra_makers)

    takers: list[Taker] = [NoiseTrader(start_prob=pop.noise_start, name=f"noise_{i}")
                           for i in range(pop.n_noise)]
    takers += [ValueTrader(path, activity=pop.value_activity, noise=s,
                           threshold=pop.value_threshold, name=f"value_{i}")
               for i, s in enumerate(pop.value_noises)]
    takers += [MomentumTrader(name=f"momentum_{i}") for i in range(pop.n_momentum)]
    return makers, takers


def run(seed: int, n: int = 20_000, pop: Population | None = None,
        price: PriceParams | None = None, config: SimConfig | None = None) -> SimResult:
    """One full simulation. The same seed always gives the same result."""
    pop = pop or Population()
    ss = np.random.SeedSequence(seed)
    price_seed, ref_seed, sim_seed = ss.spawn(3)
    path = simulate_price(n, price or PriceParams(), np.random.default_rng(price_seed))
    ref = (reference_feed(path, pop.reference_lag, pop.reference_noise,
                          rng=np.random.default_rng(ref_seed)) if pop.use_reference else None)
    makers, takers = build(path, pop)
    sim_seed_int = int(sim_seed.generate_state(1)[0])
    return Simulation(path, makers, takers, seed=sim_seed_int, config=config,
                      reference=ref).run()
