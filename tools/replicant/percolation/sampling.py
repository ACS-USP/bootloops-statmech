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
Exact invariants, checked on every instance (a correct sampler never violates them; a violation is
refused no-solution:sampler-invariant): "bethe" 2M/N == k; the configuration model's stub count ==
the degree sum == 2M; G(N, p): exactly the M edges drawn, all distinct, no self-loops.
Statistical diagnostics (reported in SampleResult.checks with two-sided p-values, never refused in
the run path, so a correct sampler is never refused): the mean degree 2M/N over the seeds against the
law's mean (Student-t; for "finite" the parity redraw allowance k_max/N is subtracted first) and the
pooled kept fraction of edges (bond) or sites (site) against t (exact binomial).
diagnostic_refusals(result, alpha) turns them into refusals (no-solution:sampler-degree,
no-solution:sampler-occupation) when a p-value is below alpha: used by the tests on fixed named seeds.
Laws are validated here (no generating-function code shared): precondition:coordination (Bethe k an
int in 2 .. 1000), :mean-degree (c rational > 0), :degree-law (finite laws "k:p,k:p" or {k: p} with
int 0 <= k <= 1000, rational p_k >= 0 summing exactly to 1, mean degree > 0), :tau, :dilution,
:occupation, :n, :named-seeds; unsupported:degree-law, :infinite-second-moment.
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
MAX_DEGREE = 1000


def _q(x, name):
    """A finite rational from an int / Fraction / decimal or fraction string (precondition:<name>)."""
    if isinstance(x, bool) or x is None:
        raise Refusal(f"precondition:{name}", f"{name} = {x!r}: need a number")
    try:
        return Fraction(x.strip()) if isinstance(x, str) else Fraction(x)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        raise Refusal(f"precondition:{name}", f"{name} = {x!r}: need a finite rational number")


def _finite(par):
    if isinstance(par, str):
        items = [it for it in par.replace(" ", "").split(",") if it]
        pairs = [it.split(":") for it in items]
        if not pairs or any(len(pr) != 2 for pr in pairs):
            raise Refusal("precondition:degree-law", f"{par!r}: write the finite law as 'k:p,k:p'")
    elif isinstance(par, dict) and par:
        pairs = list(par.items())
    else:
        raise Refusal("precondition:degree-law", f"{par!r}: a finite law is a nonempty 'k:p,k:p' or {{k: p}}")
    law = {}
    for k, p in pairs:
        if isinstance(k, bool) or not str(k).strip().isdigit() or not 0 <= int(k) <= MAX_DEGREE:
            raise Refusal("precondition:degree-law", f"degree {k!r}: need an int in 0 .. {MAX_DEGREE}")
        q = _q(p, "degree-law")
        if q < 0:
            raise Refusal("precondition:degree-law", f"p_{k} = {p!r} < 0")
        law[int(k)] = law.get(int(k), Fraction(0)) + q
    if sum(law.values()) != 1:
        raise Refusal("precondition:degree-law", f"the p_k sum to {sum(law.values())}, not 1 (exactly)")
    if not sum(k * p for k, p in law.items()):
        raise Refusal("precondition:degree-law", "mean degree 0: no edges")
    items = sorted((k, p) for k, p in law.items() if p)
    return np.array([k for k, _ in items]), np.array([float(p) for _, p in items])


def _float_law(law):
    if not (isinstance(law, tuple) and len(law) == 2):
        raise Refusal("unsupported:degree-law", f"{law!r}: ('bethe', k), ('poisson', c), ('finite', ...)")
    kind, par = law
    if kind == "bethe":
        if isinstance(par, bool) or not isinstance(par, int) or not 2 <= par <= MAX_DEGREE:
            raise Refusal("precondition:coordination", f"k = {par!r}: need an int in 2 .. {MAX_DEGREE}")
        return kind, np.array([par]), np.array([1.0])
    if kind == "poisson":
        c = _q(par, "mean-degree")
        if not c > 0:
            raise Refusal("precondition:mean-degree", f"c = {par!r}: need c > 0")
        return kind, float(c), None
    if kind == "finite":
        ks, ps = _finite(par)
        return kind, ks, ps / ps.sum()
    if kind == "powerlaw":
        tau = _q(par, "tau")
        raise Refusal("unsupported:infinite-second-moment" if tau <= 3 else "unsupported:degree-law",
                      f"power law tau = {par}: not sampled (the cavity route refuses it too)")
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
    edges = np.stack([keys // N, keys % N], 1)
    if not (len(edges) == M and np.unique(keys).size == M and np.all(edges[:, 0] < edges[:, 1])):
        raise Refusal("no-solution:sampler-invariant", f"G(N, p): {len(edges)} edges for M = {M} drawn, "
                      "or a repeated pair / self-loop")
    return edges


def _config_edges(deg, rng):
    stubs = np.repeat(np.arange(deg.size), deg)
    rng.shuffle(stubs)
    if stubs.size % 2:
        raise Refusal("no-solution:sampler-invariant", f"odd stub count {stubs.size}")
    edges = stubs.reshape(-1, 2)
    if not (stubs.size == int(deg.sum()) == 2 * len(edges)):
        raise Refusal("no-solution:sampler-invariant", f"stub count {stubs.size}, degree sum {deg.sum()}, "
                      f"edges {len(edges)}")
    return edges


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


def _occupy(n, t, rng):
    """The keep step: n independent Bernoulli(t) draws (bond: edges; site: vertices)."""
    return rng.random(n) < t * _HOOKS["occupation_scale"]


def _instance(model, dilution, N, s_max, seed):
    kind, ks, ps = model
    rng = np.random.default_rng(seed)
    edges = _er_edges(N, ks, rng) if kind == "poisson" else _config_edges(_degrees(kind, ks, ps, N, rng), rng)
    if kind == "bethe" and 2 * len(edges) != N * int(ks[0]):
        raise Refusal("no-solution:sampler-invariant", f"2M/N = {2 * len(edges) / N} != k = {ks[0]}")
    dkind, t = dilution
    if dkind == "bond":
        keep = _occupy(len(edges), t, rng)
        kept, occ = edges[keep], np.ones(N, dtype=bool)
        trials, hits = len(edges), int(keep.sum())
    else:
        occ = _occupy(N, t, rng)
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
    checks: dict        # the statistical sampler diagnostics, with their p-values (never refused here)


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


def _binom_p(hits, trials, t):
    """Two-sided exact binomial p-value: 2 min(P(X <= hits), P(X >= hits)), capped at 1."""
    from scipy.stats import binom
    return float(min(1.0, 2 * min(binom.cdf(hits, trials, t), binom.sf(hits - 1, trials, t))))


def _t_p(z, df):
    from scipy.special import stdtr
    return float(2 * stdtr(df, -abs(z)))


def diagnostic_refusals(result, alpha):
    """Refuse when a diagnostic p-value is below alpha (tests on fixed named seeds only)."""
    d = result.checks.get("mean_degree", {})
    if d.get("p_value", 1) < alpha:
        raise Refusal("no-solution:sampler-degree", f"mean degree {d['sampled']:.6g} vs the law's "
                      f"{d['law']:.6g}: p = {d['p_value']:.3g} < {alpha}")
    o = result.checks.get("occupation", {})
    if o.get("p_value", 1) < alpha:
        raise Refusal("no-solution:sampler-occupation", f"kept fraction {o['fraction']:.6g} vs t = "
                      f"{o['t']}: p = {o['p_value']:.3g} < {alpha}")


def sample_percolation(law, dilution, N, seeds, s_max=3):
    model = _float_law(law)
    if not (isinstance(dilution, tuple) and len(dilution) == 2 and dilution[0] in ("bond", "site")):
        raise Refusal("precondition:dilution", f"{dilution!r}: need ('bond' | 'site', t)")
    t = float(_q(dilution[1], "occupation"))
    if not 0 <= t <= 1:
        raise Refusal("precondition:occupation", f"t = {dilution[1]!r}: need 0 <= t <= 1")
    if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or not N_MIN <= N <= N_MAX:
        raise Refusal("precondition:n", f"N = {N!r}: need an int in {N_MIN} .. {N_MAX}")
    seeds = _seeds(seeds)
    runs = [_instance(model, (dilution[0], t), int(N), s_max, s) for s in seeds]
    n = len(runs)
    mean, se, hw = {}, {}, {}
    for key in runs[0]:
        if key in ("trials", "hits"):
            continue
        xs = np.array([r[key] for r in runs], dtype=float)
        xs = xs[~np.isnan(xs)]
        if xs.size < 2:
            continue
        mean[key], se[key] = float(xs.mean()), float(xs.std(ddof=1) / math.sqrt(xs.size))
        hw[key] = _t(0.995, xs.size - 1) * se[key]          # df of the seeds actually averaged
    want, allow = _law_mean(model, N)
    excess = max(0.0, abs(mean["mean_degree"] - want) - allow)
    pdeg = 1.0 if excess <= 1e-12 * want else (_t_p(excess / se["mean_degree"], n - 1)
                                               if se["mean_degree"] > 0 else 0.0)
    checks = {"mean_degree": {"sampled": mean["mean_degree"], "law": want, "allowance": allow,
                              "p_value": pdeg, "kind": "diagnostic (Student-t), not a refusal"}}
    trials, hits = sum(r["trials"] for r in runs), sum(r["hits"] for r in runs)
    if trials:
        checks["occupation"] = {"fraction": hits / trials, "t": t, "trials": trials,
                                "p_value": _binom_p(hits, trials, t),
                                "kind": "diagnostic (exact binomial), not a refusal"}
    return SampleResult(int(N), seeds, mean, se, hw, checks)


__all__ = ["SampleResult", "diagnostic_refusals", "sample_percolation"]
