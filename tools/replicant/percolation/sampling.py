"""Finite random graphs, percolated and labelled by union-find: the statistical route of the P1
entries.  It shares no code with the generating-function route (laws.py, cavity.py, clusters.py,
closed_forms.py are not imported; the laws are parsed here, in floats).

Graphs on N vertices (numpy default_rng(seed) per instance, so a seed names an instance):
  ("bethe", k)     the configuration model with every degree k: the N k stubs paired uniformly at
                   random (a multigraph: its O(1) self-loops and double edges are kept; they change
                   the cluster sizes by O(1/N)); N k must be even
  ("poisson", c)   G(N, p) with p = c/(N - 1): the number of edges M ~ Binomial(N(N - 1)/2, p), then M
                   distinct pairs uniformly (so the mean degree is exactly c in expectation)
  ("finite", {k: p_k})   i.i.d. degrees from p_k (one vertex redrawn until the stub count is even),
                   then the configuration model
Dilution: ("bond", t) keeps each edge independently with probability t, ("site", t) each vertex
(edges between occupied vertices kept).  Clusters: percolation/unionfind.py over the kept edges.
Per instance: largest / N (the giant's estimate of P_inf), the mean size of the cluster of an occupied
vertex outside the largest cluster (sum s^2 / sum s over the other clusters), and P_s for s <= s_max
(occupied vertices in other clusters of size s / occupied vertices).  Over the named seeds: the mean,
its stderr and the 99% Student-t half-width (df = seeds - 1).  Statistical, never final, never
certified; at finite N the giant fraction carries an O(1/N) bias and inside the critical window
|t G1'(1) - 1| <~ N^(-1/3) the largest cluster is not a giant at all (window.py).
Sampler self-checks (refusals; quantile 1 - 1e-4 two-sided, so a correct sampler trips them rarely):
  no-solution:sampler-degree      the mean degree 2M/N over the seeds against the law's mean (Student-t
                                  band; exact for "bethe"; the parity redraw allowance k_max/N for
                                  "finite")
  no-solution:sampler-occupation  the pooled fraction of kept edges (bond) or occupied sites (site)
                                  against t (exact Clopper-Pearson interval)
  sample_percolation(law, dilution, N, seeds, s_max=3) -> SampleResult
_HOOKS["edge_prob_scale"] (1): a planted wrong edge probability of G(N, p); "occupation_scale" (1): a
planted wrong occupation probability (tests).
"""
import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ..core import Refusal
from .unionfind import UnionFind

_HOOKS = {"edge_prob_scale": 1.0, "occupation_scale": 1.0}
N_MIN, N_MAX = 50, 2_000_000
Q_CHECK = 1e-4


def _float_law(law):
    if not (isinstance(law, tuple) and len(law) == 2):
        raise Refusal("unsupported:degree-law", f"{law!r}: ('bethe', k), ('poisson', c), ('finite', ...)")
    kind, par = law
    if kind == "bethe":
        if isinstance(par, bool) or not isinstance(par, int) or par < 2:
            raise Refusal("precondition:coordination", f"k = {par!r}: need an int >= 2")
        return kind, np.array([par]), np.array([1.0])
    if kind == "poisson":
        c = float(Fraction(par))
        if not c > 0:
            raise Refusal("precondition:mean-degree", f"c = {par!r}: need c > 0")
        return kind, c, None
    if kind == "finite":
        if isinstance(par, str):
            par = dict(item.split(":") for item in par.replace(" ", "").split(",") if item)
        items = sorted((int(k), float(Fraction(p))) for k, p in dict(par).items())
        ks, ps = np.array([k for k, _ in items]), np.array([p for _, p in items])
        if np.any(ps < 0) or abs(ps.sum() - 1) > 1e-12 or ks.min() < 0:
            raise Refusal("precondition:degree-law", f"{par!r}: need p_k >= 0 summing to 1, k >= 0")
        return kind, ks, ps / ps.sum()
    if kind == "powerlaw":
        raise Refusal("unsupported:infinite-second-moment" if float(Fraction(par)) <= 3 else
                      "unsupported:degree-law", f"power law tau = {par}: not sampled (the cavity route "
                      "refuses it too)")
    raise Refusal("unsupported:degree-law", f"{law!r}")


def _er_edges(N, c, rng):
    pairs = N * (N - 1) // 2
    p = min(1.0, c / (N - 1) * _HOOKS["edge_prob_scale"])
    M = int(rng.binomial(pairs, p))
    keys = np.empty(0, dtype=np.int64)
    while keys.size < M:
        need = M - keys.size
        i = rng.integers(0, N, size=need + need // 8 + 8)
        j = rng.integers(0, N, size=i.size)
        ok = i != j
        lo, hi = np.minimum(i[ok], j[ok]), np.maximum(i[ok], j[ok])
        keys = np.unique(np.concatenate([keys, lo.astype(np.int64) * N + hi]))
    if keys.size > M:
        keys = rng.choice(keys, size=M, replace=False)
    return np.stack([keys // N, keys % N], 1)


def _config_edges(deg, rng):
    stubs = np.repeat(np.arange(deg.size), deg)
    rng.shuffle(stubs)
    return stubs.reshape(-1, 2)


def _degrees(kind, ks, ps, N, rng):
    if kind == "bethe":
        if (N * int(ks[0])) % 2:
            raise Refusal("precondition:n", f"N k = {N} x {ks[0]} is odd: no k-regular configuration")
        return np.full(N, int(ks[0]))
    if np.all(ks[ps > 0] % 2) and N % 2:
        raise Refusal("precondition:n", f"every degree is odd and N = {N} is odd: the stub count is odd")
    deg = rng.choice(ks, size=N, p=ps)
    while deg.sum() % 2:            # terminates: some degree of the other parity, or N even
        deg[int(rng.integers(N))] = rng.choice(ks, p=ps)
    return deg


def _instance(model, dilution, N, s_max, seed):
    kind, ks, ps = model
    rng = np.random.default_rng(seed)
    edges = _er_edges(N, ks, rng) if kind == "poisson" else _config_edges(_degrees(kind, ks, ps, N, rng), rng)
    dkind, t = dilution
    t = t * _HOOKS["occupation_scale"]
    if dkind == "bond":
        keep = rng.random(len(edges)) < t
        kept, occ = edges[keep], np.ones(N, dtype=bool)
        trials, hits = len(edges), int(keep.sum())
    else:
        occ = rng.random(N) < t
        kept = edges[occ[edges[:, 0]] & occ[edges[:, 1]]] if len(edges) else edges
        trials, hits = N, int(occ.sum())
    uf = UnionFind(N)
    uf.add_edges(kept.tolist())
    lab = np.array(uf.labels())[occ]
    n_occ = int(occ.sum())
    out = {"mean_degree": 2 * len(edges) / N, "trials": trials, "hits": hits}
    if n_occ == 0:
        out.update(P_inf=0.0, mean_finite=math.nan, **{f"P_s{s}": 0.0 for s in range(1, s_max + 1)})
        return out
    sizes = np.bincount(lab, minlength=N)
    sizes = sizes[sizes > 0]
    big = int(sizes.argmax())
    rest = np.delete(sizes, big).astype(float)
    out["P_inf"] = float(sizes[big]) / N
    out["mean_finite"] = float((rest ** 2).sum() / rest.sum()) if rest.size else math.nan
    for s in range(1, s_max + 1):
        out[f"P_s{s}"] = float(s * np.count_nonzero(rest == s)) / n_occ
    return out


@dataclass(frozen=True)
class SampleResult:
    N: int
    seeds: tuple
    mean: dict          # key -> mean over seeds
    stderr: dict        # key -> stderr of the mean
    halfwidth99: dict   # key -> 99% Student-t half-width
    checks: dict        # the sampler self-checks, as reported


def _t(q, df):
    from scipy.special import stdtrit
    return float(stdtrit(df, q))


def _seeds(seeds):
    if isinstance(seeds, (str, bytes)) or not hasattr(seeds, "__iter__"):
        raise Refusal("precondition:named-seeds", f"seeds = {seeds!r}: need a list of ints >= 0")
    out = tuple(seeds)
    if len(out) < 2 or any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) or s < 0
                           for s in out) or len(set(out)) != len(out):
        raise Refusal("precondition:named-seeds", f"seeds = {seeds!r}: need >= 2 distinct ints >= 0")
    return tuple(int(s) for s in out)


def _law_mean(model, N):
    kind, ks, ps = model
    if kind == "bethe":
        return float(ks[0]), 0.0
    if kind == "poisson":
        return float(ks), 0.0
    return float((ks * ps).sum()), float(ks.max()) / N


def sample_percolation(law, dilution, N, seeds, s_max=3):
    from ..csp.xorsat_finite import clopper_pearson
    model = _float_law(law)
    if not (isinstance(dilution, tuple) and len(dilution) == 2 and dilution[0] in ("bond", "site")):
        raise Refusal("precondition:dilution", f"{dilution!r}: need ('bond' | 'site', t)")
    t = float(Fraction(dilution[1]))
    if not 0 <= t <= 1:
        raise Refusal("precondition:occupation", f"t = {dilution[1]!r}: need 0 <= t <= 1")
    if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or not N_MIN <= N <= N_MAX:
        raise Refusal("precondition:n", f"N = {N!r}: need an int in {N_MIN} .. {N_MAX}")
    seeds = _seeds(seeds)
    runs = [_instance(model, (dilution[0], t), int(N), s_max, s) for s in seeds]
    n, tq = len(runs), _t(0.995, len(runs) - 1)
    mean, se, hw = {}, {}, {}
    for key in runs[0]:
        if key in ("trials", "hits"):
            continue
        xs = np.array([r[key] for r in runs], dtype=float)
        xs = xs[~np.isnan(xs)]
        if xs.size < 2:
            continue
        mean[key], se[key] = float(xs.mean()), float(xs.std(ddof=1) / math.sqrt(xs.size))
        hw[key] = tq * se[key]
    want, allow = _law_mean(model, N)
    band = _t(1 - Q_CHECK / 2, n - 1) * se["mean_degree"] + allow
    checks = {"mean_degree": {"sampled": mean["mean_degree"], "law": want, "band": band}}
    if abs(mean["mean_degree"] - want) > band + 1e-12 * want:
        raise Refusal("no-solution:sampler-degree", f"mean degree {mean['mean_degree']:.6g} over {n} "
                      f"seeds vs the law's {want:.6g}: outside the band {band:.3g}")
    trials, hits = sum(r["trials"] for r in runs), sum(r["hits"] for r in runs)
    if trials:
        lo, hi = clopper_pearson(hits, trials, conf=1 - Q_CHECK)
        checks["occupation"] = {"fraction": hits / trials, "t": t, "interval": (lo, hi)}
        if not lo <= t <= hi:
            raise Refusal("no-solution:sampler-occupation", f"kept fraction {hits}/{trials} vs t = {t}: "
                          f"outside the exact interval [{lo:.5g}, {hi:.5g}]")
    return SampleResult(int(N), seeds, mean, se, hw, checks)


__all__ = ["SampleResult", "sample_percolation"]
