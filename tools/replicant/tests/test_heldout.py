"""Held-out models (Task 16): the project's acceptance test of reuse.

Two models that are NOT catalog entries are assembled by compose() from the level-(a)
steps of their nearest entries, solved, and checked against routes that did not feed
them; a positive control shows each check can fail.

Conventions.
  +-J SK: J_ij = +-1/sqrt(N) with equal probability (i < j), H = -sum_{i<j} J_ij s_i s_j
    - h sum_i s_i, s_i = +-1, beta = 1/T.  phi = lim (1/(N n)) ln E Z^n = -beta f (RS);
    entropy S = phi - beta dphi/dbeta, energy per spin u = (S - phi)/beta.
  Ising perceptron: J in {-1,+1}^N, P = alpha N Gaussian patterns, kappa = 0; a pattern
    is stored if J.xi > 0.  Entropy per weight S(alpha) = lim (1/N) E ln #solutions (RS:
    phi of the composed calculation, no temperature).

Independent routes (none is computed by the code under test, none feeds it):
  SK (a)  ln E Z at n = 1 by brute force over coupling signs (N <= 6) and by the per-pair
          sign average (N = 7..12), against derive.annealed_log_Z and against the composed
          functional at n = 1 (whose deviation is the stated dropped fourth cumulant).
  SK (b)  T_c = 1, RS ground energy -sqrt(2/pi) (bank targets, reference/heldout.json and
          sk.json) and q at beta = 1.5 equal to the Gaussian catalog family's value
          (families.sk.sk_rs, called only as the comparison target, AFTER the composed
          solve): universality at leading order in 1/N.
  SK (c)  the de Almeida-Thouless eigenvalue 1 - beta^2 E sech^4(beta sqrt(q) z),
          recomputed here by mpmath quadrature at the composed q.
  Perceptron (a) S -> ln 2 as alpha -> 0 (counting: 2^N vectors); (b) the zero of the
          composed S(alpha) against the banked Krauth-Mezard / Ding-Sun threshold (the
          root of the RS entropy) and its q, qhat; (c) the Jensen / annealed bound
          S <= ln E Z / N = ln 2 (1 - alpha) (each J satisfies one pattern with
          probability exactly 1/2, independently over patterns: E Z = 2^N 2^-P);
          (d) exact enumeration at N <= 14 (statistical, named seeds, Student-t bands
          plus an explicit finite-size slack c/N, c = 1, an assumption stated here).
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
from replicant.catalog import catalog
from replicant.compose import LEVEL_A_STEPS, compose, nearest_entry
from replicant.core import Refusal, State, run_pipeline
from replicant.refdata import check, load
from replicant.spec import ModelSpec
from replicant.testing import slow

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


def _at_eigenvalue(beta, q):
    """1 - beta^2 E_z sech^4(beta sqrt(q) z), z ~ N(0, 1): the AT replicon at h = 0,
    computed here (mpmath quadrature), not by the package."""
    with mp.workdps(30):
        b, q = mp.mpf(beta), mp.mpf(q)
        f = lambda z: mp.exp(-z * z / 2) / mp.sqrt(2 * mp.pi) * mp.sech(b * mp.sqrt(q) * z)**4
        return 1 - b**2 * mp.quad(f, [-mp.inf, 0, mp.inf])


SK_BETAS = ("4", "5", "6", "8", "10")


def _sk_ground_energy(calc):
    """(e0 from 5 points, e0 from the 4 lowest-T points): polynomial in T without a linear
    term (de/dT = T dS/dT -> 0, the RS entropy being finite at T = 0)."""
    us = {}
    for b in SK_BETAS:
        r = calc.run(beta=b, h=0, dps=15, branch="nontrivial")
        us[b] = (r.values["entropy"] - r.values["phi"]) / mp.mpf(b)
    with mp.workdps(30):
        def fit(bs):
            T = [1 / mp.mpf(b) for b in bs]
            A = mp.matrix([[1] + [t**p for p in range(2, len(bs) + 1)] for t in T])
            return mp.lu_solve(A, mp.matrix([us[b] for b in bs]))[0]
        return fit(SK_BETAS), fit(SK_BETAS[1:])


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


# ---- 1. +-J SK ----------------------------------------------------------------------
def test_sk_pm_one_annealed_matches_enumeration():
    """(a), default tier.  ln E Z at n = 1, beta = 0.7, h = 0.  Brute force: N <= 6 over
    every coupling sign pattern and every spin state (no identity used); N = 7..12 the
    per-pair average over the two signs, state by state (exact since the pairs are
    independent).  Both against derive.annealed_log_Z (the pm_one Family hook) to 1e-12,
    and against the COMPOSED functional at n = 1 (the composer's own replicate /
    average-disorder / introduce-order-parameters steps), whose gap to the exact value is
    the dropped fourth cumulant: exact - (N e + c) = -beta^4/24 (1 - 1/N) + O(1/N^2)."""
    b = 0.7
    spec = ModelSpec("sk", "ising", "pm_one", params={"beta": sp.Rational(7, 10), "h": 0})
    exact = D.annealed_log_Z(spec, D.N_SYM)
    st, _ = run_pipeline([D.replicate(spec, 1), D.average_disorder(),
                          D.introduce_order_parameters()], State())
    F = st["functional"]
    composed_n1 = D.N_SYM * F.expr + F.correction
    assert F.missing == "O(1)", F.missing        # the stated order of what was dropped
    for N in range(2, 13):
        pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        S = np.array(list(itertools.product((1.0, -1.0), repeat=N)))
        X = np.stack([S[:, i] * S[:, j] for i, j in pairs], axis=1)      # states x pairs
        if N <= 6:
            G = np.array(list(itertools.product((1.0, -1.0), repeat=len(pairs))))
            avg = np.exp(b / math.sqrt(N) * (G @ X.T)).sum() / len(G)
        else:
            avg = np.prod(np.cosh(b / math.sqrt(N) * X), axis=1).sum()
        enum = math.log(avg)
        want = float(exact.subs(D.N_SYM, N))
        assert abs(enum - want) < 1e-12 * max(1, abs(want)), (N, enum, want)
        gap = enum - float(composed_n1.subs(D.N_SYM, N))
        assert abs(gap + b**4 / 24) <= b**4 / (12 * N), (N, gap)


@slow
def test_sk_pm_one_composed_matches_independent_routes():
    calc = compose(_pm_one_sk())
    r = calc.report
    assert r.kind == "composed" and r.nearest == "sk"
    assert [(a.step, a.hook) for a in r.adapted] == [("average-disorder", "disorder:pm_one")]
    assert r.reused == [s for s in LEVEL_A_STEPS if s != "average-disorder"]
    # (b) T_c = 1 by three readings of the composed calculation
    assert not _sk_tc_failures(calc), _sk_tc_failures(calc)
    # (b) universality of q at beta = 1.5 (target computed after the composed solve)
    res = calc.run(beta="1.5", h=0, dps=15)
    assert not _universality_failures(res, "1.5")
    assert "dropped fourth cumulant" in res.trace.render() and "O(1/N) per spin" in \
        res.trace.render()
    # (b) RS ground energy: T -> 0 extrapolation of the composed energies
    e5, e4 = _sk_ground_energy(calc)
    closed = -mp.sqrt(2 / mp.pi)
    assert check(load("heldout")["sk_pm_one_rs_ground_energy"], mp.nstr(e5, 15),
                 label="pm_one e0").ok, e5
    assert check(load("sk")["rs_ground_energy"], mp.nstr(e5, 15), label="e0").ok, e5
    assert abs(e5 - closed) <= 3 * abs(e5 - e4), (e5, e4, closed)
    # (c) AT-unstable, NOT final; the replicon equals the independently computed AT one
    assert res.stable is False and res.final is False
    assert res.values["replicon"] < 0
    assert abs(res.values["replicon"] - _at_eigenvalue("1.5", res.values["q"])) < 1e-10
    assert {st.state for st in res.finality.values()} == {"unstable"}
    assert any("NOT final" in n for n in res.notes)
    rf = _refusal(lambda: calc.run(beta="1.5", h=0, dps=15, require_final=True))
    assert rf.reason == "unstable:rs" and "replicon" in rf.detail, rf


# ---- 2. Ising perceptron ------------------------------------------------------------
@slow
def test_ising_perceptron_composed_matches_independent_routes():
    calc = compose(_ising_perceptron())
    r = calc.report
    assert r.kind == "composed" and r.nearest == "perceptron"
    assert [a.step for a in r.adapted] == ["replicate", "introduce-order-parameters",
                                           "rs-ansatz", "solve-derived"], r.adapted
    assert {a.hook for a in r.adapted} == {"site:ising"}
    assert r.reused == ["average-disorder", "replica-limit", "rs-stationarity"]
    assert not _small_alpha_failures(calc), _small_alpha_failures(calc)          # (a)
    fails, alpha0 = _zero_failures(calc)                                       # (b)
    assert not fails, fails
    grid = ("0.1", "0.25", "0.5", "0.75", "1.0")
    assert not _annealed_failures(calc, grid), _annealed_failures(calc, grid)  # (c)
    cache = {}
    S_of = lambda a: cache.setdefault(a, _entropy(calc, a))
    assert not _enumeration_failures(S_of, alpha0), _enumeration_failures(S_of, alpha0)
    # (d) enumeration sanity, rigorous: median ln Z <= ln(2 E Z) (Markov), i.e. the
    # median of (1/N) ln #solutions <= ln2 (1 - alpha) + ln2/N at N = 12; statistical
    # (sample median, named seeds)
    for P, seed in MARKOV_SEEDS.items():
        c = _counts(ENUM_N, P, ENUM_SAMPLES, seed)
        with np.errstate(divide="ignore"):
            med = float(np.median(np.log(c) / ENUM_N))
        assert med <= math.log(2) * (1 - P / ENUM_N + 1 / ENUM_N), (P, med)
    # (e) not final: stable UNKNOWN below the zero, False (negative entropy) beyond it
    below = calc.run(alpha="0.5", kappa=0, dps=15)
    assert below.stable is None and not below.final
    above = calc.run(alpha="1.0", kappa=0, dps=15)
    assert above.values["entropy"] < 0 and above.stable is False and not above.final
    assert _refusal(lambda: calc.run(alpha="1.0", kappa=0, dps=15,
                                     require_final=True)).reason == "unstable:rs"
    assert _refusal(lambda: calc.run(alpha="0.5", kappa=0, dps=15,
                                     require_final=True)).reason == "unknown-stability:rs"


# ---- 3. positive controls -----------------------------------------------------------
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


@slow
def test_wrong_ansatz_is_rejected():
    import dataclasses
    # (a) RS forced where the replicon is negative: +-J SK at beta = 2
    calc = compose(_pm_one_sk())
    r = _refusal(lambda: calc.run(beta="2", h=0, dps=12, require_final=True))
    assert r.reason == "unstable:rs" and "replicon" in r.detail, r
    # (b) a sign-flipped saddle equation (SK order-parameter hook) fails the T_c and
    # universality routes: the flipped functional has no stationary point in the domain
    flipped = dataclasses.replace(D.FAMILIES["sk"], order_parameters=_sk_sign_flipped)

    def flipped_checks():
        bad = compose(_pm_one_sk())
        res = _run(bad, beta="1.5", h=0, dps=12)
        return _sk_tc_failures(bad), _universality_failures(res, "1.5"), res
    tc_f, uni_f, res = _with(D.FAMILIES, "sk", flipped, flipped_checks)
    assert tc_f and uni_f, (tc_f, uni_f)
    assert isinstance(res, Refusal) and res.reason == "no-solution" and \
        res.step == "rs-stationarity", res
    assert D.FAMILIES["sk"].order_parameters is not _sk_sign_flipped      # restored
    assert not _universality_failures(calc.run(beta="1.5", h=0, dps=12), "1.5")
    # (b') a solvable wrong model, not a refusal: the +-J law with its variance doubled
    # (J = +-sqrt(2/N)) has T_c = sqrt(2); the T_c route must see the wrong number
    law = D.LAWS["pm_one"]
    doubled = dataclasses.replace(
        law, cgf=lambda t, N: sp.log(sp.cosh(sp.sqrt(2) * t / sp.sqrt(N))))
    tc_wrong = _with(D.LAWS, "pm_one", doubled, lambda: _sk_tc_failures(compose(_pm_one_sk())))
    assert any("beta=0.95" in f for f in tc_wrong) and \
        any("paramagnetic replicon" in f for f in tc_wrong), tc_wrong
    assert D.LAWS["pm_one"] is law
    # (c) the Ising entropic term replaced by the spherical one: the composed Ising spec
    # with the spherical site.  S -> ln 2, the zero of S and the enumeration detect it; the
    # annealed bound does NOT (a spherical log-volume is <= 0 < ln2 (1 - alpha), alpha < 1)
    def spherical_checks():
        bad = compose(_ising_perceptron())
        zero, a0 = _zero_failures(bad)
        S_of = lambda a: _entropy(bad, a, dps=12)
        return (_small_alpha_failures(bad), zero, _enumeration_failures(S_of, a0),
                _annealed_failures(bad, ("0.5",)))
    small, zero, enum, ann = _with(D.SITES, "ising", D.SITES["spherical"], spherical_checks)
    assert small and zero and enum, (small, zero, enum)
    assert not ann, ann                 # the limitation, stated: this check is blind here
    assert D.SITES["ising"] is not D.SITES["spherical"]


# ---- 4. not in the catalog, and not looked up ------------------------------------------
_FAMILY_FUNCS = {
    "replicant.families.sk": ("sk_rs", "sk_replicon", "sk_1rsb", "sk_ground_energy",
                              "sk_at_temperature", "rs_equations_symbolic"),
    "replicant.families.perceptron": ("gardner_entropy", "gardner_capacity",
                                      "gardner_rs_point", "gardner_capacity_curve",
                                      "rs_equations_symbolic"),
    "replicant.catalog_entries": ("sk_rs", "sk_replicon", "sk_1rsb", "sk_ground_energy",
                                  "gardner_entropy", "gardner_capacity"),
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


def test_heldout_models_not_in_catalog():
    """Neither spec is a catalog entry (structural key); the nearest entries are the
    Gaussian SK and the spherical perceptron; and the composed +-J SK solves with every
    catalog family function made to raise (the one cheap default-tier composed solve:
    float Gauss-Hermite, dps 12, then the solver's independent audit).  The comparison
    target is called only afterwards."""
    from replicant.compose import FIELDS
    keys = [tuple(getattr(e.spec, f) for f in FIELDS) for e in catalog().values()]
    for spec, near in ((_pm_one_sk(), "sk"), (_ising_perceptron(), "perceptron")):
        assert tuple(getattr(spec, f) for f in FIELDS) not in keys, spec
        assert all(spec != e.spec for e in catalog().values())
        assert nearest_entry(spec).name == near
        assert compose(spec).report.kind == "composed"
    assert nearest_entry(_ising_perceptron()).spec.variables == "spherical"
    res, calls = _no_catalog(lambda: compose(_pm_one_sk()).run(beta="1.5", h=0, dps=12,
                                                               quad="gh"))
    assert calls == [] and res.report.kind == "composed" and res.entry is None
    from replicant.families.sk import sk_rs
    assert abs(res.values["q"] - sk_rs("1.5").q) < 1e-9
    # the guard itself works: a catalog run under it raises
    from replicant import catalog as C
    try:
        _no_catalog(lambda: compose(catalog()["sk"].spec).run(beta="1.5", h=0))
    except AssertionError as e:
        assert "composer called the catalog" in str(e)
    else:
        raise AssertionError("the no-catalog guard did not fire on a catalog run")
    assert C.CatalogEntry.execute.__name__ == "execute"           # restored


@slow
def test_heldout_ising_perceptron_not_looked_up():
    """The composed Ising perceptron solves with the catalog made to raise.  Afterwards it
    is compared with Task 8's level-(b) gardner_entropy('ising') as a consistency check
    only: that is the code path R13 keeps out of the catalog, so it is NOT counted as an
    independent route (those are in test_ising_perceptron_composed_matches_...)."""
    res, calls = _no_catalog(lambda: compose(_ising_perceptron()).run(alpha="0.5", kappa=0,
                                                                      dps=15))
    assert calls == [] and res.entry is None
    from replicant.families.perceptron import gardner_entropy
    with mp.workdps(20):
        want = gardner_entropy("0.5", "ising")
    assert abs(res.values["entropy"] - mp.mpf(getattr(want, "value", want))) < 1e-10


# ---- 5. documents -------------------------------------------------------------------
@slow
def test_heldout_documents_render():
    import shutil
    import tempfile
    import warnings
    from replicant.latex import OverfullDisplay, escape, overfull, render_latex, write_latex
    docs = {"sk_pm_one": compose(_pm_one_sk()).run(beta="1.5", h=0, dps=15),
            "ising_perceptron": compose(_ising_perceptron()).run(alpha="0.5", kappa=0, dps=15)}
    for name, res in docs.items():
        doc = render_latex(res)
        assert "Composed model" in doc, name
        assert escape(res.report.relative_to) in doc, name
        for a in res.report.adapted:
            assert escape(a.hook) in doc and escape(a.step) in doc, (name, a)
        for s in res.report.reused:
            assert escape(s) in doc, (name, s)
        assert "NOT final" in doc, name
    if not (shutil.which("tectonic") or shutil.which("pdflatex")):
        print("SKIP test_heldout_documents_render (compile): no TeX engine on PATH")
        return
    with tempfile.TemporaryDirectory() as d:
        for name, res in docs.items():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", OverfullDisplay)
                try:
                    write_latex(res, os.path.join(d, f"{name}.tex"), pdf=True)
                except Refusal as r:
                    if r.reason == "tex-engine:unavailable":
                        print(f"SKIP test_heldout_documents_render (compile): {r.detail}")
                        return
                    raise
            log = os.path.join(d, f"{name}.log")
            assert os.path.exists(os.path.join(d, f"{name}.pdf")), name
            bad = [o for o in overfull(log, 1.0) if o[2] == "display"]
            assert not bad and not caught, (name, bad, [str(w.message) for w in caught])
