"""Shared helpers of the held-out tests (test_heldout*.py): the two held-out specs, the
independent checks that more than one test module uses, and the registry / catalog guards.

Not a test module.  Split out of tests/test_heldout.py (T15) without changing any check;
the conventions and the list of independent routes are in test_heldout.py's docstring.
"""
import itertools
import math
import os
import sys

import mpmath as mp
import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant import derive as D
from replicant.agree import default_spec
from replicant.core import Refusal
from replicant.refdata import check, load
from replicant.spec import ModelSpec

LN2 = mp.log(2)
FINITE_SIZE_C = 1        # enumeration slack c/N (assumed scale of finite-size corrections)


def _pm_one_sk():
    return ModelSpec("sk", "ising", "pm_one", params={"beta": D.BETA, "h": D.H_FIELD})


def _ising_perceptron():
    s = default_spec("perceptron")
    return ModelSpec("perceptron", "ising", s.disorder, s.constraint, s.prior, dict(s.params))


def _refusal(fn):
    try:
        out = fn()
    except Refusal as r:
        return r
    raise AssertionError(f"expected Refusal, got {out!r}")


def _run(calc, **p):
    """The Result, or the Refusal (a check treats a refusal as a failed route)."""
    try:
        return calc.run(**p)
    except Refusal as r:
        return r


# ---- independent checks: each returns a list of failures (empty = the check passes) ---
def _sk_tc_failures(calc):
    """T_c = 1: q = 0 at beta = 0.95, q > 0 at beta = 1.05, and the root in beta of the
    composed paramagnetic (trivial-branch) replicon equals the banked T_c."""
    fails = []
    lo = _run(calc, beta="0.95", h=0, dps=12)
    if isinstance(lo, Refusal) or abs(lo.values["q"]) > 1e-12:
        fails.append(f"beta=0.95: expected q = 0, got {lo}")
    hi = _run(calc, beta="1.05", h=0, dps=12, branch="nontrivial")
    if isinstance(hi, Refusal) or not hi.values["q"] > 1e-3:
        fails.append(f"beta=1.05: expected a q > 0 branch, got {hi}")
    lam = lambda b: _run(calc, beta=mp.nstr(b, 20), h=0, dps=15, quad="gh", branch="trivial")
    with mp.workdps(25):
        b0, b1 = mp.mpf("0.9"), mp.mpf("1.2")
        r0, r1 = lam(b0), lam(b1)
        for _ in range(8):
            if isinstance(r0, Refusal) or isinstance(r1, Refusal) or r1.values["replicon"] is None:
                bad = r0 if isinstance(r0, Refusal) else r1
                fails.append(f"paramagnetic replicon not available: {bad}")
                return fails
            f0, f1 = r0.values["replicon"], r1.values["replicon"]
            if f1 == f0:
                break
            b0, r0, b1 = b1, r1, b1 - f1 * (b1 - b0) / (f1 - f0)
            r1 = lam(b1)
            if abs(b1 - b0) < mp.mpf("1e-14"):
                break
        tc = 1 / b1
    c = check(load("heldout")["sk_pm_one_tc"], mp.nstr(tc, 20), label="pm_one T_c")
    if not c.ok:
        fails.append(f"T_c from the paramagnetic replicon: {c}")
    return fails


def _universality_failures(res, beta):
    """q of the composed +-J solve against the Gaussian catalog family (comparison target
    only, called after the composed solve)."""
    if isinstance(res, Refusal):
        return [f"composed solve refused: {res}"]
    from replicant.families.sk import sk_rs
    want = sk_rs(beta).q
    got = res.values["q"]
    return [] if abs(got - want) < 1e-10 else [f"q {got} != Gaussian {want}"]


def _entropy(calc, alpha, dps=15):
    r = _run(calc, alpha=alpha, kappa=0, dps=dps)
    return r if isinstance(r, Refusal) else r.values["entropy"]


def _small_alpha_failures(calc):
    """S -> ln 2 as alpha -> 0, with 0 <= ln2 (1 - alpha) - S <= alpha^2 (q = O(alpha), so
    the quenched RS entropy leaves the annealed one at second order)."""
    fails = []
    for a in ("0.001", "0.01"):
        S = _entropy(calc, a)
        if isinstance(S, Refusal):
            fails.append(f"alpha={a}: refused {S.reason}")
            continue
        gap = LN2 * (1 - mp.mpf(a)) - S
        if not (0 <= gap <= mp.mpf(a)**2):
            fails.append(f"alpha={a}: S = {S}, ln2(1-alpha) - S = {gap}")
    return fails


def _annealed_failures(calc, alphas):
    """S(alpha) <= ln 2 (1 - alpha) (Jensen: E ln Z <= ln E Z), slack 1e-12 (solver)."""
    fails = []
    for a in alphas:
        S = _entropy(calc, a)
        if isinstance(S, Refusal):
            fails.append(f"alpha={a}: refused {S.reason}")
        elif S > LN2 * (1 - mp.mpf(a)) + mp.mpf("1e-12"):
            fails.append(f"alpha={a}: S = {S} > ln2(1-alpha) = {LN2 * (1 - mp.mpf(a))}")
    return fails


def _zero_failures(calc):
    """The zero of the composed S(alpha) by secant (from 0.83, 0.836), against the banked
    threshold and the banked q, qhat there (perceptron.json, Ding & Sun brackets).
    Returns (failures, root or None)."""
    ref = load("perceptron")
    run = lambda a: _run(calc, alpha=mp.nstr(a, 20), kappa=0, dps=15)
    with mp.workdps(25):
        a0, a1 = mp.mpf("0.83"), mp.mpf("0.836")
        r0, r1 = run(a0), run(a1)
        for _ in range(8):
            if isinstance(r0, Refusal) or isinstance(r1, Refusal):
                return [f"composed solve refused near the threshold: "
                        f"{r0 if isinstance(r0, Refusal) else r1}"], None
            f0, f1 = r0.values["entropy"], r1.values["entropy"]
            if f0 == f1:
                return [f"S flat near 0.833 ({f0})"], None
            a0, r0, a1 = a1, r1, a1 - f1 * (a1 - a0) / (f1 - f0)
            if not 0.5 < a1 < 1.2:
                return [f"secant left (0.5, 1.2): {a1}"], None
            r1 = run(a1)
            if abs(a1 - a0) < mp.mpf("1e-11"):
                break
    fails = []
    for key, got in (("ising_1rsb_capacity", a1), ("ising_1rsb_q", r1.values["q"]),
                     ("ising_1rsb_qhat", r1.values["qhat"])):
        c = check(ref[key], mp.nstr(got, 20), label=key)
        if not c.ok:
            fails.append(f"{key}: {c}")
    return fails, a1


def _counts(N, P, samples, seed):
    """#{J in {-1,+1}^N : J.xi^mu > 0 for mu = 1..P} for `samples` Gaussian pattern sets."""
    rng = np.random.default_rng(seed)
    J = np.array(list(itertools.product((-1.0, 1.0), repeat=N)))
    return np.array([np.all(J @ rng.standard_normal((N, P)) > 0, axis=1).sum()
                     for _ in range(samples)])


ENUM_N, ENUM_SAMPLES = 12, 300
# named seeds; P patterns at N = 12, alpha = P/N.  For the mean of ln #solutions only
# loads where an empty solution set is rare are used (P <= 4: none in 3000 trial draws
# at N = 12; at P = 6 about 0.4% of draws are empty and ln 0 makes the mean undefined)
ENUM_SEEDS = {3: 160251, 4: 160252}
MARKOV_SEEDS = {3: 160251, 4: 160252, 6: 160255}     # the median check tolerates ln 0


def _alpha(P):
    with mp.workdps(30):
        return mp.nstr(mp.mpf(P) / ENUM_N, 20)


def _enumeration_failures(S_of, alpha0):
    """(i) the mean of (1/N) ln #solutions at N = 12 (alpha = 1/4, 1/3) lies within a 99%
    Student-t band plus c/N of the composed S(alpha); (ii) the mean capacity M_N/N
    (perceptron_counting.ising_capacity_enumeration, N = 12, 14) lies within a 99% band
    plus c/N of the composed zero alpha0.  Statistical, finite-size."""
    from replicant.cavity._stats import t_quantile
    from replicant.families.perceptron_counting import ising_capacity_enumeration
    fails = []
    t = float(t_quantile(0.995, ENUM_SAMPLES - 1))
    for P, seed in ENUM_SEEDS.items():
        a = _alpha(P)
        c = _counts(ENUM_N, P, ENUM_SAMPLES, seed)
        if (c == 0).any():
            fails.append(f"alpha={a}: a sample has no solution (seed {seed}); mean ln undefined")
            continue
        logs = np.log(c) / ENUM_N
        mean, se = logs.mean(), logs.std(ddof=1) / math.sqrt(ENUM_SAMPLES)
        S = S_of(a)
        if isinstance(S, Refusal):
            fails.append(f"alpha={a}: composed refused {S.reason}")
        elif abs(float(S) - mean) > t * se + FINITE_SIZE_C / ENUM_N:
            fails.append(f"alpha={a}: composed S {float(S):.5f} vs enumeration "
                         f"{mean:.5f} +- {se:.5f} (N={ENUM_N})")
    if alpha0 is None:
        return fails + ["no composed zero of S"]
    for N, seed in ((12, 160253), (14, 160254)):
        m, se = ising_capacity_enumeration(N, 200, seed=seed)
        band = float(t_quantile(0.995, 199)) * se + FINITE_SIZE_C / N
        if abs(m - float(alpha0)) > band:
            fails.append(f"N={N}: capacity {m:.4f} +- {se:.4f} vs composed zero "
                         f"{float(alpha0):.6f} (band {band:.3f})")
    return fails


# ---- positive controls and guards ----------------------------------------------------
def _with(registry, key, value, fn):
    saved = registry[key]
    registry[key] = value
    try:
        return fn()
    finally:
        registry[key] = saved


def _sk_sign_flipped(avg, n, sum_pairs, _orig=D.FAMILIES["sk"].order_parameters):
    """SK's order-parameter hook with the sign of the q_ab^2 coupling term flipped:
    c N/2 (n + 2 sum q^2) -> c N/2 (n - 2 sum q^2); the q saddle equation becomes
    qhat = -beta^2 q."""
    import dataclasses
    red = _orig(avg, n, sum_pairs)
    cN = sp.simplify(avg.leading / avg.X**2 * avg.N)
    return dataclasses.replace(red, energy=red.energy - 2 * cN * sum_pairs(lambda q, qh: q**2))


_FAMILY_FUNCS = {
    "replicant.families.sk": ("sk_rs", "sk_replicon", "sk_1rsb", "sk_ground_energy",
                              "sk_at_temperature", "rs_equations_symbolic"),
    "replicant.families.perceptron": ("gardner_entropy", "gardner_entropy_point",
                                      "gardner_capacity", "gardner_rs_point",
                                      "gardner_capacity_curve",
                                      "rs_equations_symbolic"),
    "replicant.catalog_entries": ("sk_rs", "sk_replicon", "sk_1rsb", "sk_ground_energy",
                                  "gardner_entropy_point", "gardner_capacity"),
}


def _no_catalog(fn):
    """Run fn with every catalog family run function, CatalogEntry.execute and
    catalog.run_cached replaced by functions that raise; restored afterwards."""
    import importlib
    from replicant import catalog as C
    calls = []

    def boom(name):
        def f(*a, **k):
            calls.append(name)
            raise AssertionError(f"composer called the catalog: {name}")
        return f
    saved = []
    for mod_name, names in _FAMILY_FUNCS.items():
        mod = importlib.import_module(mod_name)
        for n in names:
            saved.append((mod, n, getattr(mod, n)))
            setattr(mod, n, boom(f"{mod_name}.{n}"))
    saved += [(C.CatalogEntry, "execute", C.CatalogEntry.execute), (C, "run_cached", C.run_cached)]
    C.CatalogEntry.execute = boom("CatalogEntry.execute")
    C.run_cached = boom("run_cached")
    try:
        return fn(), calls
    finally:
        for obj, n, v in saved:
            setattr(obj, n, v)
