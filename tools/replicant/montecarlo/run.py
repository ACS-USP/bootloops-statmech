"""Runs of the cluster and local algorithms over named seeds, and their statistics (stats.py).

  chain_series(algorithm, q, L, beta, sweeps, burn_in, seed) -> dict of per-sweep arrays of one chain:
      "E" (energy, the state before the sweep), "abs_m", "m2", "m2_improved" (SW: sum |C|^2 / N^2, the
      Fortuin-Kasteleyn estimator <m^2> = <sum |C|^2> / N^2, Salas & Sokal 1997 Eq. (3.9)), "cluster"
      (SW: number of clusters; Wolff: the flipped cluster's size).  Wolff is measured after every
      cluster flip, sweeps x flips flips with flips = max(1, round(N / mean size)) from a pilot chain
      (wolff_flips_per_sweep; simulate fixes it once per run from seeds[0], so every seed shares it), and
      its tau_int (per flip) is converted to sweeps by "tau_scale" = <|C|> / N over the run (the time unit
      of LPS Eq. (27): one sweep = N flipped sites)
  simulate(algorithm, q, L, beta, sweeps, burn_in, seeds) -> MCRun
      per observable: the mean over seeds of each chain's time average, its stderr over seeds (99%
      Student-t band in .bands), tau_int per seed (stats.tau_int) and its mean, and the equilibration
      diagnostic (stats.equilibration on the energy series); status "equilibrated" or "undetermined"
      (a drift, or any unreliable tau_int)
Algorithms: "swendsen_wang" (cluster.py; cluster_fast.py for L >= 16), "wolff" (cluster.py),
"metropolis" (metropolis.py; q = 2, even L).  Specific heat per seed: beta^2 N Var(e) of the chain
(the chain's own variance, biased low by about 2 tau_int(e) / n; the entry prints that bias in its notes).
Starting state: iid uniform colours (numpy default_rng(seed)), then burn_in sweeps discarded.
"""
from dataclasses import dataclass, field

import numpy as np

from ..core import Refusal
from ..percolation.unionfind import UnionFind
from . import stats as ST
from .cluster import bond_lists, neighbours, sw_sweep, wolff_step
from .cluster_fast import FAST_FROM, sw_sweep_fast
from .exact_small import energy, order_parameters, torus_bonds
from .fk import beta_value, bond_probability, check_q

ALGORITHMS = ("swendsen_wang", "wolff", "metropolis")
OBS = ("e", "abs_m", "m2", "C")
L_MAX = 256


def _orders(counts, N, q):
    am, m2 = order_parameters(np.asarray(counts), N, q)
    return float(am), float(m2)


def _sw_chain(q, L, p, total, rng):
    N = L * L
    out = np.empty((total, 5))
    if L >= FAST_FROM:
        u, v = torus_bonds(L)
        s = rng.integers(q, size=N)
        for t in range(total):
            counts = np.bincount(s, minlength=q)
            n_sat, n_cl, ss, _ = sw_sweep_fast(s, q, p, u, v, rng)
            out[t] = (n_sat, *_orders(counts, N, q), ss / N ** 2, n_cl)
        return out
    u, v = bond_lists(L)
    s = rng.integers(q, size=N).tolist()
    uf = UnionFind(N)
    for t in range(total):
        counts = [s.count(c) for c in range(q)]
        n_sat, n_cl, ss = sw_sweep(s, q, p, u, v, rng, uf)
        out[t] = (n_sat, *_orders(counts, N, q), ss / N ** 2, n_cl)
    return out


def _n_sat(s, u, v):
    return sum(1 for a, b in zip(u, v) if s[a] == s[b])


PILOT = 0x5EED


def wolff_flips_per_sweep(q, L, p, burn_in, seed):
    """The Wolff sweep unit, max(1, round(N / mean cluster size)) over 4 max(burn_in, 1) cluster flips of a pilot
    chain with its own generator default_rng([seed, PILOT]) from iid colours: deterministic given (q, L, beta,
    burn_in, seed); simulate fixes it ONCE per run from seeds[0], so every seed has the same unit and length."""
    N = L * L
    nb = neighbours(L)
    rng = np.random.default_rng([int(seed), PILOT])
    s = rng.integers(q, size=N).tolist()
    sizes = [wolff_step(s, q, p, nb, rng)[0] for _ in range(max(burn_in, 1) * 4)]
    return max(1, round(N / (sum(sizes) / len(sizes))))


def _wolff_chain(q, L, p, total, flips, rng):
    """Measured after every cluster flip; n_sat and the colour counts are tracked through each flip and
    checked against the final spins (an exact invariant).  Returns rows per flip (total x flips of them)."""
    N = L * L
    nb = neighbours(L)
    u, v = bond_lists(L)
    s = rng.integers(q, size=N).tolist()
    n = total * flips
    out = np.empty((n, 5))
    n_sat, counts = _n_sat(s, u, v), [s.count(c) for c in range(q)]
    for t in range(n):
        out[t] = (n_sat, *_orders(counts, N, q), np.nan, 0)
        size, d, a, b = wolff_step(s, q, p, nb, rng)
        out[t, 4] = size
        n_sat += d
        counts[a] -= size
        counts[b] += size
    if n_sat != _n_sat(s, u, v) or counts != [s.count(c) for c in range(q)]:
        raise Refusal("no-solution:sampler-invariant", "Wolff: tracked n_sat or colour counts != the final spins'")
    return out


def chain_series(algorithm, q, L, beta, sweeps, burn_in, seed, flips=None):
    N = L * L
    b = float(beta_value(q, beta))
    rng = np.random.default_rng(seed)
    if algorithm == "metropolis":
        if q != 2:
            raise Refusal("precondition:algorithm", "metropolis is the Ising (q = 2) single-spin flip")
        from .metropolis import metropolis_run
        r = metropolis_run(L, b, burn_in + sweeps, (seed,))
        E, M = r["E"][0, burn_in:].astype(float), r["M"][0, burn_in:].astype(float)
        return {"E": E, "abs_m": np.abs(M) / N, "m2": (M / N) ** 2, "m2_improved": None, "cluster": None,
                "flips": None, "tau_scale": 1.0}
    p = bond_probability(q, beta)
    if algorithm == "swendsen_wang":
        out, flips = _sw_chain(q, L, p, burn_in + sweeps, rng), None
    elif algorithm == "wolff":
        if flips is None:
            flips = wolff_flips_per_sweep(q, L, p, burn_in, seed)
        out = _wolff_chain(q, L, p, burn_in + sweeps, flips, rng)
    else:
        raise Refusal("precondition:algorithm", f"algorithm = {algorithm!r}: one of {', '.join(ALGORITHMS)}")
    out = out[burn_in * (flips or 1):]
    E = energy(out[:, 0], N, q).astype(float)
    return {"E": E, "abs_m": out[:, 1], "m2": out[:, 2],
            "m2_improved": out[:, 3] if algorithm == "swendsen_wang" else None, "cluster": out[:, 4],
            "flips": flips, "tau_scale": float(out[:, 4].mean()) / N if flips else 1.0}


@dataclass
class MCRun:
    algorithm: str
    q: int
    L: int
    beta: float
    sweeps: int
    burn_in: int
    seeds: tuple
    mean: dict
    stderr: dict
    bands: dict
    tau: dict                  # observable -> mean tau_int over seeds (sweeps)
    tau_stderr: dict           # observable -> stderr of that mean over seeds
    tau_reliable: bool
    equilibration: ST.Equilibration
    status: str
    per_seed: dict = field(default_factory=dict)
    flips: int | None = None   # Wolff: cluster flips per sweep, one value for the whole run


def _int(name, x, lo, hi):
    if isinstance(x, bool) or not isinstance(x, (int, np.integer)) or not lo <= x <= hi:
        raise Refusal(f"precondition:{name}", f"{name} = {x!r}: need an int in {lo} .. {hi}")
    return int(x)


def check_params(algorithm, q, L, sweeps, burn_in, seeds):
    """Validated (algorithm, q, L, sweeps, burn_in, seeds); refusals precondition:<name>."""
    from ..percolation.lattice2d import check_seeds
    if not isinstance(algorithm, str) or algorithm not in ALGORITHMS:
        raise Refusal("precondition:algorithm", f"algorithm = {algorithm!r}: one of {', '.join(ALGORITHMS)}")
    check_q(q)
    L = _int("size", L, 2, L_MAX)
    if algorithm == "metropolis" and (q != 2 or L % 2 or L < 4):
        raise Refusal("precondition:algorithm", f"metropolis is the Ising (q = 2) checkerboard sweep: even L >= 4 "
                      f"(on the 2 x 2 torus it is not ergodic: tests/test_mc_cluster.py), got q = {q}, L = {L}")
    return (algorithm, q, L, _int("sweeps", sweeps, 8, 10 ** 6), _int("burn-in", burn_in, 0, 10 ** 6),
            check_seeds(seeds))


def simulate(algorithm, q, L, beta, sweeps, burn_in, seeds):
    algorithm, q, L, sweeps, burn_in, seeds = check_params(algorithm, q, L, sweeps, burn_in, seeds)
    N = L * L
    b = float(beta_value(q, beta))
    per = {k: [] for k in OBS + ("m2_improved",)}
    taus = {"e": [], "m2": []}
    traces, rel, flips = [], True, []
    if algorithm == "metropolis":          # all seeds in one vectorised run (row r = chain_series(seeds[r]))
        from .metropolis import metropolis_run
        r = metropolis_run(L, b, burn_in + sweeps, seeds)
        chains = [{"E": r["E"][k, burn_in:].astype(float), "abs_m": np.abs(r["M"][k, burn_in:]) / N,
                   "m2": (r["M"][k, burn_in:] / N) ** 2, "m2_improved": None, "flips": None,
                   "tau_scale": 1.0}
                  for k in range(len(seeds))]
    else:
        unit = (wolff_flips_per_sweep(q, L, bond_probability(q, beta), burn_in, seeds[0])
                if algorithm == "wolff" else None)               # one sweep unit for every seed of the run
        chains = (chain_series(algorithm, q, L, beta, sweeps, burn_in, sd, unit) for sd in seeds)
    for c in chains:
        e = c["E"] / N
        per["e"].append(e.mean())
        per["abs_m"].append(c["abs_m"].mean())
        per["m2"].append(c["m2"].mean())
        per["C"].append(b * b * N * e.var())
        if c["m2_improved"] is not None:
            per["m2_improved"].append(c["m2_improved"].mean())
        for k, x in (("e", e), ("m2", c["m2"])):
            ti = ST.tau_int(x)
            taus[k].append(ti.tau * c["tau_scale"])
            rel = rel and ti.reliable
        traces.append(e)
        flips.append(c["flips"])
    if len(set(flips)) != 1 or len({len(x) for x in traces}) != 1:
        raise Refusal("no-solution:sampler-invariant", f"chains of one run differ in sweep unit or length: {flips}")
    keys = [k for k in per if per[k]]
    bands = {k: ST.band(per[k]) for k in keys}
    eq = ST.equilibration(np.array(traces))
    status = "equilibrated" if eq.equilibrated and rel else "undetermined"
    return MCRun(algorithm, q, L, b, sweeps, burn_in, seeds, {k: bands[k].mean for k in keys},
                 {k: bands[k].stderr for k in keys}, bands, {k: float(np.mean(v)) for k, v in taus.items()},
                 {k: float(np.std(v, ddof=1) / np.sqrt(len(v))) for k, v in taus.items()}, rel, eq, status,
                 {k: tuple(per[k]) for k in keys} | {f"tau_{k}": tuple(v) for k, v in taus.items()},
                 flips[0])


__all__ = ["ALGORITHMS", "MCRun", "chain_series", "simulate", "wolff_flips_per_sweep"]
