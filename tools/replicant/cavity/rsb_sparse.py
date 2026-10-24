"""Sparse 1RSB cavity at T > 0: population of populations (Mezard-Parisi 2001, Sec. V-VI).

Model and laws as cavity/sparse.py: E = -sum_<ij> J_ij s_i s_j, ("regular", k) or
("poisson", c) graphs, ("pm1", J) or ("gauss", sigma) couplings, beta = 1/T finite.

Convention for m (stated once; Mezard-Parisi 2001 write it x):
  m in (0, 1] is the Parisi breakpoint.  Each edge carries M -> oo states alpha with
  cavity fields h^alpha; adding a vertex (or link) to state alpha shifts its free energy by
  DeltaF^alpha = -(1/beta) ln z^alpha and the states are reweighted by
      exp(-beta m DeltaF^alpha) = (z^alpha)^m          (MP 2001 Eqs. (44)-(45)),
  with the normalised site factor (k inputs, coupling J_i to cavity field h_i)
      z = sum_{s0} prod_i sum_{si} exp(beta J_i s0 si + beta h_i si) / prod_i 2cosh(beta h_i)
  and h_new = sum_i u(J_i, h_i) (u as cavity/sparse.transfer).  y = beta m is the T = 0
  "mu" of Mezard-Parisi 2003 (cavity/rsb_sparse_t0.py, reweighting exp(-y DeltaE)).
Reduction to RS (verified in tests): at m = 1 the reweighted barycentres E_w tanh(beta h)
  obey the RS recursion exactly, so f(m=1) = f_RS, e(m=1) = e_RS, q0(m=1) = q_RS
  (Mezard-Montanari 2009, ch. 19); and any population whose inner populations are deltas
  is the RS solution at every m (the trivial 1RSB solution).  At T = 0 the RS limit is
  y = 0 (no reweighting).  So the plan's "m = 1 reduces to RS" holds in this convention.
Recursion (population of populations): `pop` outer members, each an inner population of
  `inner` fields (one per state).  A sweep replaces every outer member: draw an excess
  degree d and couplings J_1..J_d (common to all states of the new edge), parents a_1..a_d;
  for each new inner slot draw one state from each parent, compute (h_new, ln z), then
  resample `inner` slots with probability proportional to z^m.
Observables (MP 2001 Eqs. (47)-(49), (52), (54)); cbar = mean degree:
  -beta m f = E ln E_states[z_site^m] - (cbar/2) E ln E_states[z_link^m],
  e = (cbar/2) E sum_alpha W_link^alpha (-J <s s'>_alpha),
  q1 = E sum_alpha W_site^alpha tanh^2(beta H^alpha),  q0 = E sum_{alpha != beta} W W' t t'.
Finite `inner` biases ln E_states (Jensen, O(1/inner)): MP 2001 extrapolate in M; we do
not.  Honest label: statistical, RSB_CAVEAT.

  pop_of_pops_1rsb(degree_law, J_law, beta, m, *, pop, inner, sweeps, seed, ...) -> RSBResult
  rsb_sparse_energy / rsb_ground_state_energy / factorized_1rsb_t0: rsb_sparse_t0.py
"""
import math
import numpy as np

from ..core import Refusal
from . import graphs as G
from ._rsb_common import (FACTORIZED_CAVEAT, MP01, MP03, RSB_CAVEAT,  # noqa: F401
                          STEP_NAMES, RSBResult,
                          check_sizes, pipeline, summarize)
from .sparse import _beta, _ln2cosh, _measure, transfer

# ---- finite temperature ----------------------------------------------------------------
LINK_PAIRS = 8   # link observables average over 8 * inner random state pairs (variance)


def _site(b, J, h):
    """Normalised site: J, h (..., d) -> (H = sum u, ln z); J = 0 pads (u = 0, ln z += 0)."""
    u = transfer(b, J, h)
    lc = 0.5 * (_ln2cosh(b * (J + h)) + _ln2cosh(b * (J - h))) - _ln2cosh(b * h)
    H = u.sum(-1)
    return H, lc.sum(-1) + _ln2cosh(b * H)


def _link(b, J, h1, h2):
    combos = np.stack([J + h1 + h2, J - h1 - h2, -J + h1 - h2, -J - h1 + h2])
    lz = np.logaddexp.reduce(b * combos, axis=0) - _ln2cosh(b * h1) - _ln2cosh(b * h2)
    t, t1, t2 = np.tanh(b * J), np.tanh(b * h1), np.tanh(b * h2)
    return lz, -J * (t + t1 * t2) / (1 + t * t1 * t2)


def _weights(lz, m):
    w = np.exp(m * (lz - lz.max(1, keepdims=True)))
    return w / w.sum(1, keepdims=True)


def _ln_mean_zm(lz, m):
    mx = lz.max(1)
    return m * mx + np.log(np.mean(np.exp(m * (lz - mx[:, None])), 1))


def _resample(rng, w):
    """Row-wise multinomial resampling: (P, I) normalised weights -> (P, I) indices."""
    P, I = w.shape
    cw = np.cumsum(w, 1)
    cw[:, -1] = 1.0
    off = np.arange(P)[:, None]
    idx = np.searchsorted((cw + off).ravel(), (rng.random((P, I)) + off).ravel(), "right")
    return np.minimum(idx.reshape(P, I) - off * I, I - 1)


def _inputs(rng, h, d, jl, inner):
    """For each of len(d) new members: d[a] parents (padded with J = 0), one random state
    per parent per inner slot -> (J (n, 1, dmax), fields (n, inner, dmax))."""
    P, I = h.shape
    n, dmax = len(d), max(int(d.max(initial=0)), 1)
    par = rng.integers(P, size=(n, 1, dmax))
    fields = h[par, rng.integers(I, size=(n, inner, dmax))]
    J = G.sample_couplings(jl, (n, 1, dmax), rng)
    J = np.where(np.arange(dmax) < d[:, None, None], J, 0.0)
    return J, np.where(J != 0, fields, 0.0)


def _spread(x):
    """q1 - q0, with rounding noise of delta inner populations (|x| < 1e-12) set to 0."""
    return 0.0 if abs(x) < 1e-12 else float(x)


def _ft_run(p, rng):
    law, jl, b, m, P, I = (p[k] for k in ("degree_law", "J_law", "beta", "m", "pop", "inner"))
    cbar = G.branching(law)[1]
    h = (p["init_field"] + rng.standard_normal((P, 1))
         + p["init_spread"] * rng.standard_normal((P, I)))
    rows, acc, q1t, q0t = np.arange(P)[:, None], [], [], []
    for t in range(p["burn_in"] + p["sweeps"]):
        J, hin = _inputs(rng, h, G.sample_excess_degrees(law, P, rng), jl, I)
        H, lz = _site(b, J, hin)
        h = H[rows, _resample(rng, _weights(lz, p["reweight_sign"] * m))]
        if not np.all(np.isfinite(h)):
            raise Refusal("no-solution:audit", f"population diverged at sweep {t}")
        th = np.tanh(b * h)
        q1t.append(float(np.mean(th**2)))
        q0t.append(_spread(q1t[-1] - float(np.mean(th.mean(1)**2))))   # inner spread
        if t < p["burn_in"]:
            continue
        J, hin = _inputs(rng, h, G.sample_degrees(law, P, rng), jl, I)
        Hs, lzs = _site(b, J, hin)
        a, c = rng.integers(P, size=(2, P))
        L = min(I * I, p["link_pairs"] * I)        # state pairs (alpha, beta) per link
        h1 = h[a[:, None], rng.integers(I, size=(P, L))]
        h2 = h[c[:, None], rng.integers(I, size=(P, L))]
        lzl, eps = _link(b, G.sample_couplings(jl, (P, 1), rng), h1, h2)
        f = -(np.mean(_ln_mean_zm(lzs, m)) - cbar / 2 * np.mean(_ln_mean_zm(lzl, m))) / (b * m)
        e = cbar / 2 * np.mean((_weights(lzl, m) * eps).sum(1))
        ws, mag = _weights(lzs, m), np.tanh(b * Hs)
        s1, s2 = (ws * mag).sum(1), (ws**2 * mag**2).sum(1)
        off = 1 - (ws**2).sum(1)
        q0 = np.where(off > 1e-9, (s1**2 - s2) / np.where(off > 1e-9, off, 1), s1**2)
        row = (f, e, float(np.mean(q0)), float(np.mean((ws * mag**2).sum(1))))
        if m == 1:                                 # exact m = 1 functional of barycentres
            mb = np.clip(np.tanh(b * h).mean(1), -1 + 1e-15, 1 - 1e-15)
            _, e_b, f_b = _measure(rng, np.arctanh(mb) / b, law, jl, b, transfer, 0.0, cbar)
            row += (f_b, e_b)
        acc.append(row)
    collapsed = bool(np.max(np.abs(h)) < 1e-12)
    delta = bool(np.max(np.ptp(h, axis=1)) < 1e-12)
    spread = [_spread(a[3] - a[2]) for a in acc]               # q1 - q0 per measured sweep
    return np.mean(acc, 0), [a[3] for a in acc], spread, q1t, q0t, collapsed, delta


def _ft_solve(p):
    per_run, ms, ts, coll, delt = [], ([], []), ([], []), True, True
    for ss in np.random.SeedSequence(p["seed"]).spawn(p["runs"]):
        mean, q1m, spm, q1t, spt, c, d = _ft_run(p, np.random.default_rng(ss))
        per_run.append(mean)
        ms[0].append(q1m), ms[1].append(spm), ts[0].append(q1t), ts[1].append(spt)
        coll, delt = coll and c, delt and d
    phase = "paramagnet" if coll else ("rs" if delt else "1rsb")
    keys = ("f", "e", "q0", "q1") + (("f_bary", "e_bary") if p["m"] == 1 else ())
    val, err, phase, note = summarize(per_run, keys, ms, ts, p["sweeps"],
                                      p["require_equilibrated"], phase)
    note += (f"; f carries a finite-inner (Jensen) bias O(1/inner), inner = {p['inner']}, "
             "not extrapolated")
    return val, err, phase, note


def pop_of_pops_1rsb(degree_law, J_law, beta, m, *, pop: int, inner: int, sweeps: int,
                     seed: int, burn_in=None, runs=4, init_field=0.0, init_spread=1.0,
                     require_equilibrated=False, _reweight_sign=1) -> RSBResult:
    """Finite-T 1RSB population of populations (see module docstring).  Initial state:
    outer centres init_field + N(0, 1), inner members centre + init_spread N(0, 1)
    (init_spread = 0: delta inner populations, the RS solution).  burn_in defaults to
    max(2 sweeps, 20).  Drift diagnostic as Task 12 (cavity/_stats.drift) on q1 and on
    the inner spread q1 - q0 (a growing or collapsing spread is drift): a drifting run is
    "undetermined" or refused (require_equilibrated).
    At m = 1 the result also carries f_bary, e_bary: the Bethe (RS) f and e of the
    reweighted barycentres tanh(beta hbar) = E_states tanh(beta h) of each inner population
    (Task 12's _measure on the barycentre population): the exact infinite-inner m = 1
    functional, which isolates the identity f(m=1) = f_RS from finite-inner estimator bias.
    Reliable window: NOT mapped in (m, T) (cost).  Measured: a biased start at 0.95 beta_c
    (k = 2) is flagged; m = 1 agrees with RS (pop 40000 reference) within the combined
    99% Student-t band for f, e, q0, f_bary, e_bary (slow test); small inner populations (<~ 20)
    collapse towards q1 = q0 within ~60 sweeps at k = 2, beta = 1.5.
    _reweight_sign: test hook for planted errors (0 drops z^m, -1 uses z^-m)."""
    G.degree_law(degree_law)
    G.j_law(J_law)
    b = _beta(beta, allow_inf=False)
    mm = float(m)
    if not 0 < mm <= 1:
        raise Refusal("precondition:parisi-m", f"Parisi m must be in (0, 1], got {m!r} (m -> 0 "
                      "at finite T is not this routine: use y = beta m at T = 0)")
    if not (isinstance(inner, int) and inner >= 2):
        raise Refusal("precondition:inner-size", f"inner (states per edge) must be an int "
                      f">= 2, got {inner!r}")
    burn_in = max(2 * sweeps, 20) if burn_in is None else burn_in
    check_sizes(pop, sweeps, runs, burn_in)
    params = dict(degree_law=degree_law, J_law=J_law, beta=b, m=mm, pop=pop, inner=inner,
                  sweeps=sweeps, seed=seed, burn_in=burn_in, runs=runs,
                  init_field=float(init_field), init_spread=float(init_spread),
                  require_equilibrated=bool(require_equilibrated),
                  reweight_sign=_reweight_sign, link_pairs=LINK_PAIRS)
    planted = " [NON-DEFAULT REWEIGHTING]" if _reweight_sign != 1 else ""
    descs = (f"remove one link: cavity spin keeps excess degree law {degree_law}",
             f"population of {pop} inner populations of {inner} states each (cavity fields "
             f"per state), spread {init_spread} around N({init_field}, 1) centres",
             "h_new = sum u(J, h) per state, states reweighted by z^m = exp(-beta m DeltaF), "
             f"m = {mm}" + planted)
    state, trace = pipeline(params, descs, _ft_solve,
                            (f"{MP01}, Sec. V", f"{MP01}, Sec. V, Eqs. (37)-(46)",
                             f"{MP01}, Eqs. (44)-(45)", f"{MP01}, Eqs. (47)-(54)"))
    val, err, phase, note = state["solution"]
    return RSBResult(val, err, "statistical", phase, note, params, trace)


from .rsb_sparse_t0 import (factorized_1rsb_t0, rsb_ground_state_energy,  # noqa: E402,F401
                            rsb_sparse_energy)
