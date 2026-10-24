"""Cavity route on sparse random graphs: BP, RS population dynamics, Viana-Bray T_c.

Model and conventions (Mezard & Parisi, Eur. Phys. J. B 20, 217 (2001),
arXiv:cond-mat/0009418, Secs. II-IV):
  E = -sum_<ij> J_ij s_i s_j - H sum_i s_i,  s_i = +-1,  beta = 1/T (beta = inf: T = 0).
  Degree laws (cavity/graphs.py): ("regular", k) = connectivity k + 1, excess degree k;
  ("poisson", c) = Erdos-Renyi with mean degree c, excess degree Poisson(c);
  ("tree", n) = random tree (BP only).  J laws: ("pm1", J), ("gauss", sigma).

Cavity recursion (MP 2001 Eqs. (3)-(6)): the cavity field h_{i->j} acts on s_i with
the link (ij) removed; adding the link transmits
  u(J, h) = (1/beta) atanh(tanh(beta J) tanh(beta h))
          = [lncosh(beta(J + h)) - lncosh(beta(J - h))] / (2 beta)     (stable form)
  -> (|J + h| - |J - h|) / 2 at T = 0 (MP, arXiv:cond-mat/0207121, Eq. (10)),
and h_{i->j} = H + sum_{l in di \\ j} u(J_il, h_{l->i}).  Full field H_i: all neighbours.
Observables per spin (MP 2001 Eqs. (12), (13), (16); cbar = mean degree):
  q = E tanh^2(beta H_i);  e = (cbar/2) E[-J (t + t1 t2)/(1 + t t1 t2)] - H E tanh(beta H_i),
  t = tanh(beta J), t_a = tanh(beta h_a);
  -beta f = (cbar/2) E ln Z_link(J, h1, h2) - E[(d - 1) ln 2cosh(beta H_d)],
  Z_link = sum_{s,s'} exp(beta(J s s' + h1 s + h2 s')).  At T = 0: e := f.
Transition (MP 2001 Eq. (36); Viana & Bray 1985 for Poisson): h = 0 is stable iff
  B E_J tanh^2(beta J) < 1, B = k (regular) or c (Poisson).

  bp_fixed_point(graph_or_law, beta, J_law, *, seed, ...)  -> BPResult
  population_dynamics(degree_law, J_law, beta, *, pop, sweeps, seed, ...) -> PDResult
  pd_growth_factor / pd_transition_beta   PD's own transition location (statistical)
  viana_bray_beta_c / viana_bray_tc(degree_law, J_law) -> mpf (closed form / mp root)
  rs_zero_temperature_pm1(k)              exact T = 0 RS integer-field solution (+-1)

Honest label: outputs carry CAVITY_CAVEAT; PD values are statistical (mean +- stderr
over independent runs, seeds from numpy SeedSequence(seed).spawn).
"""
import math
from dataclasses import dataclass, field as dfield

import mpmath as mp
import numpy as np
import sympy as sp

from ..core import Refusal, State, Step, Trace, run_pipeline
from ..families.common import CAVITY_CAVEAT
from . import graphs as G

STEP_NAMES = ("remove_site", "cavity_field_law", "add_site", "self_consistency")
_MP01 = "Mezard & Parisi, Eur. Phys. J. B 20, 217 (2001), arXiv:cond-mat/0009418"
_MP02 = "Mezard & Parisi, The cavity method at zero temperature, arXiv:cond-mat/0207121"


def _ln2cosh(x):
    """ln(2 cosh x), overflow-free."""
    a = np.abs(x)
    return a + np.log1p(np.exp(-2 * a))


def transfer(beta, J, h):
    """u(J, h); beta = inf gives the T = 0 limit."""
    if math.isinf(beta):
        return 0.5 * (np.abs(J + h) - np.abs(J - h))
    return (_ln2cosh(beta * (J + h)) - _ln2cosh(beta * (J - h))) / (2 * beta)


def _beta(beta, *, allow_inf):
    b = float(beta)
    if not b > 0 or math.isnan(b):
        raise Refusal("precondition:beta-positive", f"beta = 1/T must be > 0, got {beta!r}")
    if math.isinf(b) and not allow_inf:
        raise Refusal("precondition:beta-finite", "this routine needs finite beta (T > 0)")
    return b


# ---- belief propagation on one instance ------------------------------------------------
@dataclass(frozen=True)
class BPResult:
    """messages[2e] = u_{i->j}, messages[2e+1] = u_{j->i} for graph.edges[e] = (i, j):
    the field (in units of J's energy scale, not times beta) that i sends to j.
    marginals and log_z are None when converged is False: the last iterate is not a
    fixed point, so Bethe quantities computed from it would mean nothing."""
    messages: np.ndarray
    converged: bool
    iterations: int
    residual: float
    graph: G.Graph
    marginals: np.ndarray | None     # <s_i> = tanh(beta H_i); None if not converged
    log_z: float | None              # Bethe ln Z (exact on trees); None if not converged
    caveat: str = CAVITY_CAVEAT


def bp_fixed_point(graph_degree_law, beta, J_law, *, seed: int, n=None, field=0.0,
                   damping=0.0, max_iter=1000, tol=1e-13, require_converged=False,
                   _transfer=None):
    """Parallel BP with optional damping u <- d u + (1 - d) u_new, random start
    (seeded).  `graph_degree_law` is a graphs.Graph (then J_law must be None) or a
    degree law sampled at size n with couplings from J_law.  converged iff the max
    message change < tol; with require_converged a non-converged run is refused."""
    rng = np.random.default_rng(seed)
    if isinstance(graph_degree_law, G.Graph):
        if J_law is not None:
            raise Refusal("precondition:j-law", "an explicit Graph carries its couplings; "
                          "pass J_law=None")
        g = graph_degree_law.check()
    else:
        g = G.sample_graph(graph_degree_law, J_law, n, rng)
    b = _beta(beta, allow_inf=False)
    if not 0 <= damping < 1:
        raise Refusal("precondition:damping", f"damping must be in [0, 1), got {damping}")
    T = _transfer or transfer
    i, j = g.edges[:, 0], g.edges[:, 1]
    src = np.empty(2 * len(i), int)
    src[0::2], src[1::2] = i, j
    dst = np.empty_like(src)
    dst[0::2], dst[1::2] = j, i
    Jd, rev = np.repeat(g.J, 2), np.arange(len(src)) ^ 1
    u = 0.5 * rng.standard_normal(len(src))

    def cavity(u):
        tot = field + np.bincount(dst, weights=u, minlength=g.n)
        return tot, tot[src] - u[rev]          # full fields; h_{src -> dst}

    it, res = 0, (math.inf if len(src) else 0.0)
    while res >= tol and it < max_iter:
        it += 1
        new = damping * u + (1 - damping) * T(b, Jd, cavity(u)[1])
        res, u = float(np.max(np.abs(new - u))), new
    conv = bool(res < tol)
    if require_converged and not conv:
        raise Refusal("no-convergence", f"BP stopped after {it} iterations with residual "
                      f"{res:.3g} >= tol {tol:g}; try damping in (0, 1) or more iterations "
                      "(below T_c on a loopy graph BP may have no stable RS fixed point)")
    H, h = cavity(u)
    h1, h2 = h[0::2], h[1::2]
    J = g.J
    lz_link = np.logaddexp.reduce(b * np.stack([J + h1 + h2, J - h1 - h2, -J + h1 - h2,
                                                -J - h1 + h2]), axis=0)
    d = g.degrees()
    log_z = float(lz_link.sum() - ((d - 1) * _ln2cosh(b * H)).sum())
    if not (np.all(np.isfinite(u)) and math.isfinite(log_z)):
        raise Refusal("no-solution:audit", f"non-finite BP messages or ln Z at beta = {b}")
    if not conv:
        return BPResult(u, False, it, res, g, None, None)
    return BPResult(u, True, it, res, g, np.tanh(b * H), log_z)


# ---- population dynamics -------------------------------------------------------------
@dataclass(frozen=True)
class PDResult:
    """value / stderr: {"q", "e", "f"} per spin (stderr over independent runs)."""
    value: dict
    stderr: dict
    label: str = "statistical"
    # "paramagnet" (all runs collapsed to h = 0), "spin-glass" / "rs-in-field" (stationary
    # within the drift test), or "undetermined" (q still drifting: not equilibrated)
    phase: str = ""
    note: str = ""
    params: dict = dfield(default_factory=dict)
    trace: Trace = None
    caveat: str = CAVITY_CAVEAT


def _sum_u(rng, h, degs, jl, b, T, field):
    """field + sum of degs[a] transmitted fields u(J, h_random) for each slot a."""
    dmax = int(degs.max(initial=0))
    if dmax == 0:
        return np.full(len(degs), float(field))
    idx = rng.integers(len(h), size=(len(degs), dmax))
    u = T(b, G.sample_couplings(jl, idx.shape, rng), h[idx])
    if not np.all(degs == dmax):
        u = np.where(np.arange(dmax) < degs[:, None], u, 0.0)
    return field + u.sum(1)


def _measure(rng, h, law, jl, b, T, field, cbar):
    m = len(h)
    H = _sum_u(rng, h, G.sample_degrees(law, m, rng), jl, b, T, field)
    d = G.sample_degrees(law, m, rng)                      # an independent site for f
    Hd = _sum_u(rng, h, d, jl, b, T, field)
    J = G.sample_couplings(jl, m, rng)
    h1, h2 = h[rng.integers(m, size=m)], h[rng.integers(m, size=m)]
    combos = np.stack([J + h1 + h2, J - h1 - h2, -J + h1 - h2, -J - h1 + h2])
    if math.isinf(b):
        q = np.mean(H != 0)
        f = -(cbar / 2) * np.mean(combos.max(0)) + np.mean((d - 1) * np.abs(Hd))
        return q, f, f
    t, t1, t2 = np.tanh(b * J), np.tanh(b * h1), np.tanh(b * h2)
    m_i = np.tanh(b * H)
    e = (cbar / 2) * np.mean(-J * (t + t1 * t2) / (1 + t * t1 * t2)) - field * np.mean(m_i)
    lz = np.logaddexp.reduce(b * combos, axis=0)
    f = -((cbar / 2) * np.mean(lz) - np.mean((d - 1) * _ln2cosh(b * Hd))) / b
    return float(np.mean(m_i**2)), float(e), float(f)


def _pd_steps():
    def remove(s):
        p = s["params"]
        B, cbar = G.branching(p["degree_law"])
        return s.with_(route="cavity", cavity_system={
            "desc": f"remove one link: cavity spin keeps excess degree law "
                    f"{p['degree_law']} (B = {B}), mean degree {cbar}", "B": B, "cbar": cbar})

    def law(s):
        p = s["params"]
        return s.with_(field_law={
            "desc": f"population of {p['pop']} iid cavity fields, field-biased start "
                    f"h = {p['init_field']} (all members); h = 0 is the paramagnet"})

    def add(s):
        planted = s["params"]["transfer"] is not transfer
        return s.with_(site={"desc": "u = (1/beta) atanh(tanh(beta J) tanh(beta h)); "
                             "h_new = H + sum of excess-degree u's" +
                             (" [NON-DEFAULT TRANSFER]" if planted else "")})

    def close(s):
        return s.with_(solution=_pd_solve(s["params"], s["cavity_system"]["cbar"]))

    cite = f"{_MP01}, Secs. III-IV and VI (population dynamics)"
    note = lambda key: (lambda s: "route=cavity; " + s[key]["desc"])
    return [Step("remove_site", lambda s: [k for k in ("params",) if k not in s], remove,
                 cite, note("cavity_system")),
            Step("cavity_field_law", lambda s: [k for k in ("cavity_system",) if k not in s],
                 law, cite, note("field_law")),
            Step("add_site", lambda s: [k for k in ("field_law",) if k not in s], add,
                 f"{_MP01}, Eqs. (3)-(5)", note("site")),
            Step("self_consistency", lambda s: [k for k in ("site",) if k not in s], close,
                 f"{_MP01}, Eqs. (12), (13), (16)",
                 lambda s: f"route=cavity; phase {s['solution'][2]}")]


def t_quantile(prob, df):
    """Student-t quantile (mpmath: regularized incomplete beta, bracketed root)."""
    with mp.workdps(20):
        cdf = lambda t: 1 - mp.betainc(df / 2, mp.mpf(1) / 2, 0, df / (df + t * t),
                                       regularized=True) / 2
        hi = mp.mpf(1)
        while cdf(hi) < prob:
            hi *= 2
        return float(mp.findroot(lambda t: cdf(t) - prob, (hi / 2 if hi > 1 else 0, hi),
                                 solver="illinois"))


def _drift(qs_runs, runs):
    """Equilibration diagnostic from the dynamics only: per run, relative change of q
    between the first and second half of the measured sweeps; drifting iff the mean
    change across runs is nonzero by a two-sided Student-t test at 99%."""
    rel = []
    for qs in qs_runs:
        a, c = np.mean(qs[:len(qs) // 2]), np.mean(qs[len(qs) - len(qs) // 2:])
        scale = (abs(a) + abs(c)) / 2
        rel.append((c - a) / scale if scale > 0 else 0.0)
    rel = np.array(rel)
    m, se = float(rel.mean()), float(rel.std(ddof=1) / math.sqrt(runs))
    thr = t_quantile(0.995, runs - 1) * se
    return (abs(m) > thr if se > 0 else m != 0), m, thr


def _pd_solve(p, cbar):
    law, jl, b, T, fld = p["degree_law"], p["J_law"], p["beta"], p["transfer"], p["field"]
    per_run, qs_runs, collapsed = [], [], True
    for ss in np.random.SeedSequence(p["seed"]).spawn(p["runs"]):
        rng = np.random.default_rng(ss)
        h = np.full(p["pop"], float(p["init_field"]))
        acc = []
        for t in range(p["burn_in"] + p["sweeps"]):
            h = _sum_u(rng, h, G.sample_excess_degrees(law, p["pop"], rng), jl, b, T, fld)
            if not np.all(np.isfinite(h)):
                raise Refusal("no-solution:audit", f"population diverged at sweep {t}")
            if t >= p["burn_in"]:
                acc.append(_measure(rng, h, law, jl, b, T, fld, cbar))
        collapsed &= bool(np.max(np.abs(h)) < 1e-12)
        per_run.append(np.mean(acc, axis=0))
        qs_runs.append([a[0] for a in acc])
    per_run = np.array(per_run)
    mean, err = per_run.mean(0), per_run.std(0, ddof=1) / math.sqrt(len(per_run))
    if not (np.all(np.isfinite(mean)) and -1e-12 <= mean[0] <= 1 + 1e-12):
        raise Refusal("no-solution:audit", f"PD observables out of range: {mean}")
    keys = ("q", "e", "f")
    drifting, m, thr = _drift(qs_runs, p["runs"])
    if collapsed:
        phase, note = "paramagnet", "every run collapsed to h = 0 (max |h| < 1e-12)"
    elif drifting:
        phase = "undetermined"
        note = (f"not equilibrated: q changed by {m:+.3g} (relative) between the halves of "
                f"the measured sweeps, beyond the 99% Student-t band {thr:.2g}; the values "
                "carry burn-in bias not in stderr (near T_c: slow decay to h = 0 or slow "
                "growth); increase burn_in")
        if p["require_equilibrated"]:
            raise Refusal("no-convergence:equilibration", note)
    else:
        phase = "spin-glass" if fld == 0 else "rs-in-field"
        note = f"stationary: relative q drift {m:+.2g} within the 99% band {thr:.2g}"
    return dict(zip(keys, map(float, mean))), dict(zip(keys, map(float, err))), phase, note


def _pd_params(degree_law, J_law, beta, pop, sweeps, seed, runs, allow_inf):
    G.degree_law(degree_law)
    G.j_law(J_law)
    b = _beta(beta, allow_inf=allow_inf)
    if not (isinstance(pop, int) and pop >= 100):
        raise Refusal("precondition:population-size", f"pop must be an int >= 100, got {pop!r}")
    if not (isinstance(runs, int) and runs >= 2):
        raise Refusal("precondition:runs", f"runs must be >= 2 for a stderr, got {runs!r}")
    if not (isinstance(sweeps, int) and sweeps >= 4):
        raise Refusal("precondition:sweeps", f"sweeps must be an int >= 4 (the drift test "
                      f"compares halves of the measured sweeps), got {sweeps!r}")
    return b


def population_dynamics(degree_law, J_law, beta, *, pop: int, sweeps: int, seed: int,
                        burn_in=50, runs=8, field=0.0, init_field=1.0,
                        require_equilibrated=False, _transfer=None):
    """RS population dynamics (parallel update: each sweep replaces the whole
    population by `pop` fresh h = H + sum of u's from random members), `burn_in`
    sweeps then `sweeps` measured sweeps, in `runs` independent runs.  A population that
    is still drifting (see _drift) is labelled phase "undetermined", or refused with
    no-convergence:equilibration when require_equilibrated is set."""
    b = _pd_params(degree_law, J_law, beta, pop, sweeps, seed, runs, allow_inf=True)
    params = dict(degree_law=degree_law, J_law=J_law, beta=b, pop=pop, sweeps=sweeps,
                  seed=seed, burn_in=int(burn_in), runs=runs, field=float(field),
                  init_field=float(init_field), transfer=_transfer or transfer,
                  require_equilibrated=bool(require_equilibrated))
    state, trace = run_pipeline(_pd_steps(), State(params=params))
    value, err, phase, note = state["solution"]
    shown = {k: v for k, v in params.items() if k != "transfer"}
    return PDResult(value, err, "statistical", phase, note, shown, trace)


def pd_growth_factor(degree_law, J_law, beta, *, pop: int, sweeps: int, seed: int, runs=4,
                     amplitude=1e-6, _transfer=None) -> PDResult:
    """Linear stability of h = 0, measured: start from h ~ amplitude N(0, 1), apply the
    full transfer, record E h_new^2 / E h^2 per sweep, rescale to `amplitude`.
    growth > 1 means the paramagnet is unstable (beta > beta_c)."""
    b = _pd_params(degree_law, J_law, beta, pop, sweeps, seed, runs, allow_inf=False)
    T = _transfer or transfer
    ratios = []
    for ss in np.random.SeedSequence(seed).spawn(runs):
        rng = np.random.default_rng(ss)
        h, acc = amplitude * rng.standard_normal(pop), []
        for t in range(sweeps + 2):
            new = _sum_u(rng, h, G.sample_excess_degrees(degree_law, pop, rng), J_law, b, T, 0.0)
            if t >= 2:                        # two sweeps to forget the Gaussian start
                acc.append(np.mean(new**2) / np.mean(h**2))
            h = new * amplitude / math.sqrt(np.mean(new**2))
        ratios.append(np.mean(acc))
    r = np.array(ratios)
    if not np.all(np.isfinite(r)):
        raise Refusal("no-solution:audit", f"non-finite growth factors {r}")
    return PDResult({"growth": float(r.mean())},
                    {"growth": float(r.std(ddof=1) / math.sqrt(len(r)))},
                    note="linear growth of a tiny-amplitude population", params=dict(degree_law=degree_law, J_law=J_law, beta=b, pop=pop))


def pd_transition_beta(degree_law, J_law, *, lo, hi, pop, sweeps, seed, iters=12):
    """Bisection for growth = 1 with common random numbers (statistical)."""
    g = lambda b: pd_growth_factor(degree_law, J_law, b, pop=pop, sweeps=sweeps,
                                   seed=seed).value["growth"] - 1
    if not (g(lo) < 0 < g(hi)):
        raise Refusal("no-solution:bracket", f"growth - 1 does not change sign on [{lo}, {hi}]")
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if g(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


# ---- Viana-Bray / Bethe-lattice transition ------------------------------------------------
def viana_bray_beta_c(degree_law, J_law):
    """beta_c solving B E_J tanh^2(beta_c J) = 1 (B = k regular, c Poisson): closed form
    atanh(1/sqrt B)/J for +-J; mpmath quadrature + bracketed root for Gaussian J."""
    B, _ = G.branching(degree_law)
    kind, s = G.j_law(J_law)
    if not B > 1:
        raise Refusal("no-transition", f"branching B = {B} <= 1: B E tanh^2(beta J) < 1 "
                      "at every finite beta, the paramagnet is stable down to T = 0")
    with mp.workdps(30):
        Bm, sm = mp.mpf(B), mp.mpf(repr(s))
        if kind == "pm1":
            return +(mp.atanh(1 / mp.sqrt(Bm)) / sm)
        dens = lambda z: mp.exp(-z * z / 2) / mp.sqrt(2 * mp.pi)
        # even integrand; the dropped tail |z| > 16 weighs < 1e-56 (below the 30-digit target)
        g = lambda b: 2 * Bm * mp.quad(lambda z: dens(z) * mp.tanh(b * z)**2,
                                       [0, 2, 4, 8, 16], method="gauss-legendre") - 1
        lo, hi = 1 / mp.sqrt(Bm) / 2, 1 / mp.sqrt(Bm)    # E tanh^2(bz) < b^2: g(lo) < 0
        while g(hi) <= 0:
            lo, hi = hi, 2 * hi
        root = mp.findroot(g, (lo, hi), solver="illinois")
        if not abs(g(root)) < mp.mpf("1e-25"):
            raise Refusal("no-solution:audit", f"Gaussian beta_c residual {g(root)}")
        return +(root / sm)


def viana_bray_tc(degree_law, J_law):
    """T_c = 1 / beta_c (see viana_bray_beta_c), as an mpf."""
    with mp.workdps(30):
        return +(1 / viana_bray_beta_c(degree_law, J_law))


# ---- exact T = 0 RS solution for +-J on the regular graph -----------------------------------
def rs_zero_temperature_pm1(k):
    """At T = 0 with J = +-1, cavity fields are integers and u = J sign(h) in {-1, 0, 1}:
    with P(u = 0) = x, P(u = +-1) = (1 - x)/2, h = sum of k u's and the closure is
    x = P(h = 0).  The root x in (0, 1) (x = 1 is the paramagnet) gives the field law and
    e = f = -[(k+1)/2 E max_{s,s'}(J s s' + h1 s + h2 s') - k E |H|], H = sum of k+1 u's."""
    if not (isinstance(k, int) and k >= 2):
        raise Refusal("precondition:k", f"needs integer excess degree k >= 2, got {k!r}")
    x, z = sp.symbols("x z")
    law = lambda n: sp.Poly(sp.expand((x + (1 - x) / 2 * (z + 1 / z))**n * z**n), z)
    coeff = lambda n, r: law(n).coeff_monomial(z**(n + r))
    roots = [r for r in sp.Poly(coeff(k, 0) - x, x).all_roots() if r.is_real and 0 < r < 1]
    if len(roots) != 1:
        raise Refusal("no-solution:rs-t0", f"k = {k}: {len(roots)} roots of x = P(h=0) in (0,1)")
    x0 = roots[0]
    ph = {r: sp.simplify(coeff(k, r).subs(x, x0)) for r in range(-k, k + 1)}
    pH = {r: sp.simplify(coeff(k + 1, r).subs(x, x0)) for r in range(-k - 1, k + 2)}
    link = sum(sp.Rational(1, 2) * ph[a] * ph[c] * max(J + a + c, J - a - c, -J + a - c,
                                                       -J - a + c)
               for J in (-1, 1) for a in ph for c in ph)
    site = sum(p * abs(r) for r, p in pH.items())
    e = sp.simplify(-(sp.Rational(k + 1, 2) * link - k * site))
    with mp.workdps(30):
        q = sp.simplify(1 - pH[0])          # q = E tanh^2(beta H) -> P(H != 0) at T = 0
        return {"p0": mp.mpf(str(sp.N(x0, 40))), "e": mp.mpf(str(sp.N(e, 40))),
                "q": mp.mpf(str(sp.N(q, 40))), "q_exact": q,
                "p_exact": {r: ph[r] for r in range(0, k + 1)}, "e_exact": e,
                "cite": f"{_MP02}, Sec. III.B (recursion); computed here", "caveat": CAVITY_CAVEAT}
