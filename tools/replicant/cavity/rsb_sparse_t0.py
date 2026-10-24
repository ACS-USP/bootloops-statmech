"""Sparse 1RSB at T = 0: population of integer-field surveys, +-J on the (k+1)-regular graph.

Mezard & Parisi, The cavity method at zero temperature, J. Stat. Phys. 111, 1 (2003),
arXiv:cond-mat/0207121 (MP03), Secs. IV-V.  Re-exported by cavity/rsb_sparse.py.

Convention: y = beta m >= 0 held fixed as T -> 0 (MP03's mu; MP03 Fig. 4: "mu = beta m").
States (local ground states) are reweighted by exp(-y DeltaE); y = 0 is RS (no
reweighting).  With J = +-1 cavity fields are integers h in [-k, k] and a survey p_j(h)
is the law of h_j over states.  We use NORMALISED state energies (each cavity state's
energy minimised over its cavity spin), so the iteration shift is
    DeltaE = -(sum_i delta_{h_i, 0} + |sum_i u_i|),   u_i = J_i sign(h_i),
(a(J, h) - |h| = delta_{h,0} for integer h), and the recursion is
    p0(r) ~ sum_{h_1..h_k} prod_i p_i(h_i) delta(r - sum_i u_i) exp(-y DeltaE).
This is MP03 Eq. (57) after the change of variables p_MP(h) ~ p(h) exp(-y |h|) (MP03's
unnormalised energies); Phi is invariant; reported p0 is converted to MP03's convention.
Observables per spin (J scale s: E(y; s) = s E(y s; 1)):
  phi_var = -(1/y)[E ln Z_site(k+1) - (k+1)/2 E ln Z_link]          (MP03 Eq. (31) form)
  phi     = -(1/y)[(k+1)/2 E ln Z_site(k) - (k-1)/2 E ln Z_site(k+1)]  (MP01 Eq. (50) form,
            valid at the fixed point, lower variance; the two are compared as a check)
  e = d(y phi_var)/dy (MP03 Eq. (46), explicit derivative only: variational),
  sigma = y (e - phi_var) (MP03 Eq. (48)).  Ground state: max_y phi (Sigma = 0).
Z_link = E exp(y (max_{s,s'}(J s s' + h1 s + h2 s') - |h1| - |h2|)).  y = 0: the limits.
"""
import math

import mpmath as mp
import numpy as np

from ..core import Refusal
from . import graphs as G
from ._stats import t_quantile
from ._rsb_common import FACTORIZED_CAVEAT, MP01, MP03, RSBResult, check_sizes, pipeline, summarize

INITS = ("random", "symmetric", "near-factorized")


def _law(degree_law, J_law):
    kind, k = G.degree_law(degree_law)
    jk, s = G.j_law(J_law)
    if kind != "regular" or k < 2:
        raise Refusal("unsupported:degree-law", f"{degree_law!r}: the integer-field survey "
                      "route needs fixed connectivity ('regular', k >= 2); use "
                      "pop_of_pops_1rsb at finite beta for other graphs")
    if jk != "pm1":
        raise Refusal("unsupported:J-law", f"{J_law!r}: integer cavity fields need +-J "
                      "couplings; use pop_of_pops_1rsb at finite beta for Gaussian J")
    return k, s


def _y(y):
    v = float(y)
    if not (v >= 0 and math.isfinite(v)):
        raise Refusal("precondition:y", f"y = beta m must be finite and >= 0, got {y!r}")
    return v


def _uw(p, J, y, k):
    """Survey (n, 2k+1) -> (value, d/dy) of weights over u = -1, 0, +1 incl. exp(y delta_h0)."""
    w = np.stack([p[:, :k].sum(1), p[:, k] * math.exp(y), p[:, k + 1:].sum(1)], 1)
    dw = np.zeros_like(w)
    dw[:, 1] = w[:, 1]
    flip = J < 0
    w[flip], dw[flip] = w[flip][:, ::-1], dw[flip][:, ::-1]
    return w, dw


def _conv(pairs):
    """Distribution of sum u (value, d/dy) from per-input (n, 3) weight pairs."""
    c, dc = pairs[0]
    for w, dw in pairs[1:]:
        n, L = c.shape
        new, dnew = np.zeros((n, L + 2)), np.zeros((n, L + 2))
        for j in range(3):
            new[:, j:j + L] += c * w[:, j:j + 1]
            dnew[:, j:j + L] += dc * w[:, j:j + 1] + c * dw[:, j:j + 1]
        c, dc = new, dnew
    return c, dc


def _site(p, rng, n, d, y, k, sign=1.0):
    """n sites with d inputs: (ln Z, E_w[X], new survey over r in [-d, d]); X = -DeltaE."""
    idx, J = rng.integers(len(p), size=(n, d)), rng.choice((-1.0, 1.0), size=(n, d))
    c, dc = _conv([_uw(p[idx[:, i]], J[:, i], sign * y, k) for i in range(d)])
    ar = np.abs(np.arange(-d, d + 1))
    ex = np.exp(sign * y * ar)
    Z = (c * ex).sum(1)
    return np.log(Z), ((dc + c * ar) * ex).sum(1) / Z, c * ex / Z[:, None]


def _link_table(k):
    hs = np.arange(-k, k + 1)
    best = lambda J: np.max([[[h1 * a + h2 * b + J * a * b for a in (-1, 1) for b in (-1, 1)]
                              for h2 in hs] for h1 in hs], axis=2)
    return {J: best(J) - np.abs(hs)[:, None] - np.abs(hs)[None, :] for J in (-1, 1)}


def _link(p, rng, n, y, table):
    i1, i2 = rng.integers(len(p), size=(2, n))
    X = np.where(rng.random(n)[:, None, None] < 0.5, table[1], table[-1])
    w = p[i1][:, :, None] * p[i2][:, None, :] * np.exp(y * X)
    Z = w.sum((1, 2))
    return np.log(Z), (w * X).sum((1, 2)) / Z


def _phi(lz, mean_x, y):
    return np.mean(lz) / y if y > 0 else np.mean(mean_x)


def _t0_run(k, y, sign, pop, burn_in, sweeps, rng, init):
    K = 2 * k + 1
    p = {"random": lambda: rng.dirichlet(np.ones(K), pop),
         "symmetric": lambda: np.full((pop, K), 1.0 / K),
         "near-factorized": lambda: (1 - 1e-6) / K + 1e-6 * rng.dirichlet(np.ones(K), pop)
         }[init]()
    table, hs = _link_table(k), np.abs(np.arange(-k, k + 1))
    acc, asym_t, p0_t = [], [], []
    for t in range(burn_in + sweeps):
        p = _site(p, rng, pop, k, y, k, sign)[2]
        mp_conv = p * np.exp(-y * hs)                    # MP03's convention, for p0
        asym_t.append(float(np.mean(np.abs(p[:, k + 1:].sum(1) - p[:, :k].sum(1)))))
        p0_t.append(float(np.mean(mp_conv[:, k] / mp_conv.sum(1))))
        if t < burn_in:
            continue
        l1, x1, _ = _site(p, rng, pop, k + 1, y, k)
        l1p, x1p, _ = _site(p, rng, pop, k, y, k)
        l2, x2 = _link(p, rng, pop, y, table)
        phi_var = -(_phi(l1, x1, y) - (k + 1) / 2 * _phi(l2, x2, y))
        phi = -((k + 1) / 2 * _phi(l1p, x1p, y) - (k - 1) / 2 * _phi(l1, x1, y))
        e = -(np.mean(x1) - (k + 1) / 2 * np.mean(x2))
        acc.append((phi, phi_var, e, y * (e - phi_var), p0_t[-1], asym_t[-1]))
    ident = bool(np.max(np.ptp(p, axis=0)) < 1e-12)
    return np.mean(acc, 0), [a[5] for a in acc], [a[4] for a in acc], asym_t, p0_t, ident


KEYS = ("phi", "phi_var", "e", "sigma", "p0", "asym")


def _scale(val, err, s):
    for d in (val, err):
        for key in ("phi", "phi_var", "e"):
            d[key] *= s
    return val, err


def _consistency(per_run, runs, require):
    d = np.array([r[0] - r[1] for r in per_run])
    mean, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(runs))
    tq = t_quantile(0.995, runs - 1)
    text = (f"phi (Eq. (50) form) - phi_var (Eq. (31) form) = {mean:+.2g} (99% Student-t "
            f"band {tq * se:.2g})")
    if abs(mean) > tq * se + 2e-3 and require:
        raise Refusal("no-solution:estimator-mismatch", text + ": the two free-energy forms "
                      "agree only at a fixed point of the 1RSB recursion; the population is "
                      "not one (planted error or wrong reweighting?)")
    return text


def rsb_sparse_energy(degree_law, J_law, y, *, pop: int, sweeps: int, seed: int,
                      burn_in=None, runs=4, init="random", require_equilibrated=False,
                      require_consistent=False, _reweight_sign=1) -> RSBResult:
    """T = 0 1RSB at fixed y: population of `pop` surveys, parallel sweeps (each replaces
    every survey by the MP03 recursion from k random ones), `runs` independent runs
    (SeedSequence(seed).spawn).  init: "random" (Dirichlet surveys, finds the non-factorized
    solution), "symmetric" (identical symmetric surveys: stays exactly factorized),
    "near-factorized" (symmetric + 1e-6 asymmetry: slow escape, tests the drift diagnostic).
    Drift diagnostic (cavity/_stats.drift) on the mean asymmetry |p(h>0) - p(h<0)| (in the
    normalised convention) and the mean p0 (MP03 convention).  require_consistent refuses
    when phi and phi_var differ by more than the 99% Student-t band + 2e-3 (the floor
    absorbs the unrelaxed small-population spread, measured ~0.01; planted errors give
    ~0.09).  Reliable window (measured, k = 2, y = 0.7): random start, pop 1000, burn_in
    100, 8 runs is labelled "1rsb" (slow test); near-factorized start with burn_in 4 is
    flagged in 12/12 seeds; burn_in 30 at pop 200 can still be "undetermined".  Other
    (k, y): not mapped.
    _reweight_sign: planted-error hook (0 drops exp(-y DeltaE) in the recursion)."""
    k, s = _law(degree_law, J_law)
    yv = _y(y)
    if init not in INITS:
        raise Refusal("precondition:init", f"init must be one of {INITS}, got {init!r}")
    burn_in = max(2 * sweeps, 20) if burn_in is None else burn_in
    check_sizes(pop, sweeps, runs, burn_in)
    params = dict(degree_law=degree_law, J_law=J_law, y=yv, pop=pop, sweeps=sweeps, seed=seed,
                  burn_in=burn_in, runs=runs, init=init, reweight_sign=_reweight_sign)

    def solve(p):
        per, ms, ts, ident = [], ([], []), ([], []), True
        for ss in np.random.SeedSequence(seed).spawn(runs):
            mean, am, pm, at, pt, idn = _t0_run(k, yv * s, _reweight_sign, pop, burn_in,
                                                sweeps, np.random.default_rng(ss), init)
            per.append(mean)
            ms[0].append(am), ms[1].append(pm), ts[0].append(at), ts[1].append(pt)
            ident &= idn
        phase = "rs" if yv == 0 else ("factorized" if ident and max(r[5] for r in per) < 1e-12
                                      else "1rsb")
        val, err, phase, note = summarize(per, KEYS, ms, ts, sweeps, require_equilibrated, phase)
        note += "; " + _consistency(per, runs, require_consistent)
        return (*_scale(val, err, s), phase, note)

    planted = " [NON-DEFAULT REWEIGHTING]" if _reweight_sign != 1 else ""
    descs = (f"remove one link: cavity spin keeps k = {k} further neighbours",
             f"population of {pop} surveys over integer cavity fields in [-{k}, {k}], "
             f"init {init}",
             f"survey recursion with state reweighting exp(-y DeltaE), y = {yv}" + planted)
    state, trace = pipeline(params, descs, solve,
                            (f"{MP03}, Sec. II", f"{MP03}, Sec. V.F.2, Eqs. (55)-(56)",
                             f"{MP03}, Eqs. (26), (57)", f"{MP03}, Eqs. (31), (46), (48); "
                             f"{MP01}, Eq. (50)"))
    val, err, phase, note = state["solution"]
    return RSBResult(val, err, "statistical", phase, note, params, trace)


def rsb_ground_state_energy(degree_law, J_law, *, ys, pop: int, sweeps: int, seed: int,
                            burn_in=None, runs=8, init="random", require_equilibrated=False,
                            _reweight_sign=1) -> RSBResult:
    """GGS energy e0 = max_y phi(y) (Sigma = 0, MP03 Eq. (44)): in each run, phi on the
    grid `ys` (one population per y, common random numbers across y); the run-averaged
    phi(y) is fitted by a least-squares parabola whose vertex must be an interior maximum.
    e0, y_star: vertex; stderr: jackknife over the `runs` independent runs (phi(y) is so
    flat near y* that per-run fits often lack a maximum).  Drift diagnostic per grid point.
    A parabola on a wide grid can sit slightly below a flat-topped maximum (systematic, not
    in stderr; measured 1e-5 on the exact factorized phi(y), k = 2, grid (0.3, ..., 1.1))."""
    k, s = _law(degree_law, J_law)
    yv = [_y(v) for v in ys]
    if not (len(yv) >= 3 and all(a < b for a, b in zip(yv, yv[1:])) and yv[0] > 0):
        raise Refusal("precondition:y-grid", f"ys must be >= 3 strictly increasing values > 0, "
                      f"got {ys!r}")
    if init not in INITS:
        raise Refusal("precondition:init", f"init must be one of {INITS}, got {init!r}")
    burn_in = max(2 * sweeps, 20) if burn_in is None else burn_in
    check_sizes(pop, sweeps, runs, burn_in)
    params = dict(degree_law=degree_law, J_law=J_law, ys=tuple(yv), pop=pop, sweeps=sweeps,
                  burn_in=burn_in, seed=seed, runs=runs, init=init, reweight_sign=_reweight_sign)

    def solve(p):
        phis, ms, ts, fact = [], [[] for _ in yv], [[] for _ in yv], True
        for ss in np.random.SeedSequence(seed).spawn(runs):
            row = []
            for j, y in enumerate(yv):
                mean, am, _, at, _, ident = _t0_run(k, y * s, _reweight_sign, pop, burn_in,
                                                    sweeps, np.random.default_rng(ss), init)
                row.append(mean[0] * s)
                ms[j].append(am), ts[j].append(at)
                fact &= ident and mean[5] < 1e-12
            phis.append(row)
        phis = np.array(phis)

        def vertex(rows):
            a, b, c = np.polyfit(yv, phis[rows].mean(0), 2)
            ystar = -b / (2 * a) if a < 0 else math.nan
            return c - b * b / (4 * a), ystar

        e0, ystar = vertex(slice(None))
        if not yv[0] <= ystar <= yv[-1]:
            raise Refusal("no-solution:no-maximum", f"no interior maximum of the run-averaged "
                          f"phi(y) on {yv} (phi = {np.round(phis.mean(0), 5).tolist()}); "
                          "widen the grid or lengthen the runs")
        jack = np.array([vertex([i for i in range(runs) if i != r]) for r in range(runs)])
        jerr = np.sqrt((runs - 1) / runs * ((jack - jack.mean(0))**2).sum(0))
        # phase from the diagnostics: "factorized" if every survey at every grid point stayed
        # identical and symmetric, else "1rsb"; summarize() may override with "undetermined"
        _, _, phase, note = summarize(phis, [f"phi@{y}" for y in yv], ms, ts, sweeps,
                                      require_equilibrated, "factorized" if fact else "1rsb")
        note = (f"max over y of the run-averaged phi by one least-squares parabola on {yv}; "
                f"stderr: jackknife over the {runs} runs (use Student-t, df = runs - 1); "
                + note)
        return ({"e0": float(e0), "y_star": float(ystar)},
                dict(zip(("e0", "y_star"), map(float, jerr))), phase, note)

    planted = " [NON-DEFAULT REWEIGHTING]" if _reweight_sign != 1 else ""
    descs = (f"remove one link: cavity spin keeps k = {k} further neighbours",
             f"population of {pop} surveys per grid point y in {yv}, init {init}",
             "survey recursion with state reweighting exp(-y DeltaE)" + planted)
    state, trace = pipeline(params, descs, solve,
                            (f"{MP03}, Sec. II", f"{MP03}, Sec. V.F.2, Eqs. (55)-(56)",
                             f"{MP03}, Eqs. (26), (57)", f"{MP03}, Eqs. (43)-(44): GGS energy "
                             f"= max_y phi; {MP01}, Eq. (50)"))
    val, err, phase, note = state["solution"]
    return RSBResult(val, err, "statistical", phase, note, params, trace)


def factorized_1rsb_t0(k, y=None, *, tol=1e-13, max_iter=10000):
    """Factorized 1RSB at T = 0 (MP03 Sec. V.F.1), +-1, (k+1)-regular: one symmetric survey
    for every edge, iterated deterministically to its fixed point (the recursion of
    rsb_sparse_energy with every input equal; J's sign is irrelevant by symmetry), then
    phi(y) from the variational site/link form with exact averages.  y=None: golden-section
    maximum over y in (0, 3] -> e0 = phi(y*), y_star.  Exact enumeration, no sampling."""
    if not (isinstance(k, int) and k >= 2):
        raise Refusal("precondition:k", f"needs integer excess degree k >= 2, got {k!r}")
    table, hs = _link_table(k), np.abs(np.arange(-k, k + 1))

    def at(yy):
        p = np.full((1, 2 * k + 1), 1.0 / (2 * k + 1))
        for it in range(max_iter):
            c = _conv([_uw(p, np.ones(1), yy, k)] * k)[0] * np.exp(yy * hs)
            new = c / c.sum()
            if np.max(np.abs(new - p)) < tol:
                break
            p = new
        else:
            raise Refusal("no-convergence", f"factorized survey did not converge at y = {yy}")
        c1, dc1 = _conv([_uw(p, np.ones(1), yy, k)] * (k + 1))
        ar = np.abs(np.arange(-k - 1, k + 2))
        Z1 = float((c1 * np.exp(yy * ar)).sum())
        X1 = float(((dc1 + c1 * ar) * np.exp(yy * ar)).sum()) / Z1
        pp = p[0][:, None] * p[0][None, :]
        Z2 = [float((pp * np.exp(yy * table[J])).sum()) for J in (-1, 1)]
        X2 = [float((pp * table[J] * np.exp(yy * table[J])).sum()) / z
              for J, z in zip((-1, 1), Z2)]
        ln2 = sum(map(math.log, Z2)) / 2
        phi = (-(math.log(Z1) - (k + 1) / 2 * ln2) / yy if yy > 0
               else -(X1 - (k + 1) / 4 * sum(X2)))
        e = -(X1 - (k + 1) / 4 * sum(X2))
        q = p[0] * np.exp(-yy * hs)
        return phi, e, float(q[k] / q.sum())

    with mp.workdps(20):
        if y is not None:
            phi, e, p0 = at(_y(y))
            return {"phi": mp.mpf(phi), "e": mp.mpf(e), "p0": mp.mpf(p0),
                    "sigma": mp.mpf(_y(y) * (e - phi)), "caveat": FACTORIZED_CAVEAT}
        g = (math.sqrt(5) - 1) / 2
        lo, hi = 1e-3, 3.0
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        fa, fb = at(a)[0], at(b)[0]
        while hi - lo > 1e-9:
            if fa > fb:
                hi, b, fb = b, a, fa
                a = hi - g * (hi - lo)
                fa = at(a)[0]
            else:
                lo, a, fa = a, b, fb
                b = lo + g * (hi - lo)
                fb = at(b)[0]
        ys = (lo + hi) / 2
        phi, e, p0 = at(ys)
        return {"e0": mp.mpf(phi), "y_star": mp.mpf(ys), "p0": mp.mpf(p0),
                "sigma": mp.mpf(ys * (e - phi)), "caveat": FACTORIZED_CAVEAT,
                "cite": f"{MP03}, Sec. V.F.1 (factorized Ansatz); computed here"}
