"""E1: the replicon at level (a) for Gaussian-integral energies (perceptron, Hopfield).

The family replicons (families/perceptron.py gardner_rs_point, families/hopfield.py
_replicon) are comparison TARGETS only: nothing from them feeds derive_replicon.  A
family saddle used as an evaluation point is first checked to be a root of the DERIVED
RS equations (derived_replicon's saddle audit), so the point is the derived saddle too.
Conventions: lambda_R > 0 means RS replicon-stable (SK: 1 - beta^2 E sech^4).
"""
import os
import sys
from functools import lru_cache

import mpmath as mp
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant import derive as D
from replicant.agree import default_spec, derive_rs
from replicant.core import Refusal
from replicant.derive_replicon import derived_replicon, replicon
from replicant.solve_derived import solve_derived_rs
from replicant.testing import slow

TOL = mp.mpf("1e-8")
HOP = default_spec("hopfield")
SPH, ISI = default_spec("perceptron"), default_spec("perceptron_ising")
# Hopfield cross-check points of v1 Task 9 (alpha, beta, branch) and the family replicon
HOPFIELD_POINTS = (("0.1", "20", "spinglass"), ("0.05", "3", "spinglass"),
                   ("0.13815", 1 / mp.mpf("0.023"), "retrieval"), ("0.1", "20", "retrieval"),
                   ("0.13", "20", "retrieval"))


def _refusal(fn):
    try:
        fn()
    except Refusal as r:
        return r
    raise AssertionError("expected Refusal")


def _perceptron_point(variables, alpha, kappa=0):
    """The family's RS saddle at alpha (target code) as a derived-name mapping."""
    from replicant.families.perceptron import _solve_alpha
    pt = _solve_alpha(variables, mp.mpf(alpha), mp.mpf(kappa))
    vals = {"alpha": mp.mpf(alpha), "kappa": mp.mpf(kappa), "q": pt.q}
    if variables == "ising":
        vals["qhat"] = pt.qhat
    return vals, pt.replicon


@lru_cache(maxsize=None)
def _hopfield_point(alpha, beta, branch):
    """Family saddle (m, q, r) -> derived names; qhat = alpha beta^2 r (v1 Task 10)."""
    from replicant.families.hopfield import hopfield_rs
    s = hopfield_rs(alpha, beta, branch=branch)
    a, b = mp.mpf(alpha), mp.mpf(beta)
    return {"alpha": a, "beta": b, "m": s.m, "q": s.q, "qhat": a * b**2 * s.r}, s.replicon


def _agree(spec, vals, target, **kw):
    rep = derived_replicon(spec, vals, **kw)
    assert rep.value is not None, rep.note
    assert abs(rep.value - target) < TOL, (spec.family, spec.variables, rep.value, target)
    return rep


# ---- default tier --------------------------------------------------------------------
def _order_point(variables, order):
    """The family saddle at a given order parameter (q spherical, qhat Ising): no root
    solve, alpha comes from the family's stationarity equation."""
    from replicant.families.perceptron import gardner_rs_point
    pt = gardner_rs_point(variables, order, 0)
    vals = {"alpha": pt.alpha, "kappa": mp.mpf(0), "q": pt.q}
    if variables == "ising":
        vals["qhat"] = pt.qhat
    return vals, pt.replicon


def test_derived_replicon_at_family_saddles_default_tier():
    # one point per family: the derived lambda_R at the family's saddle, which the
    # derived equations must accept (audit), equals the family's replicon
    vals, lam = _order_point("spherical", "0.6")
    rep = _agree(SPH, vals, lam, dps=10)
    assert "covariance" in rep.route and "explicit-n" in rep.route, rep.route
    # the saddle audit is a real check: a point off the saddle is refused
    bad = {**vals, "q": vals["q"] * mp.mpf("1.01")}
    r = _refusal(lambda: derived_replicon(SPH, bad, dps=10))
    assert r.reason == "precondition:rs-saddle" and "residual" in r.detail, r
    vals, lam = _order_point("ising", "10")
    rep = _agree(ISI, vals, lam, dps=10)
    assert vals["alpha"] > 1.05 and rep.value < 0 and "ising-site" in rep.route, rep
    vals, lam = _hopfield_point("0.05", "3", "spinglass")        # cached by test_cavity_dense
    assert abs(lam - mp.mpf("-1.033")) < 1e-3                     # the Task 9 table value
    _agree(HOP, vals, lam, dps=10)


def _with_hook(registry, key, value, fn):
    saved = registry[key]
    registry[key] = value
    try:
        return fn()
    finally:
        registry[key] = saved
        assert registry[key] is saved


def test_planted_hook_error_changes_the_replicon():
    # the Hopfield energy hook: weight exp(beta y^2/2) -> exp(1.1 beta y^2/2).  The
    # planted hook must move the derived replicon at the same point far beyond the
    # tolerance, and the restored registry gives the family value back
    vals, lam = _hopfield_point("0.05", "3", "spinglass")
    fam = D.FAMILIES["hopfield"]

    def bad_average(zn):
        avg = fam.average(zn)
        y = avg.weight.variables[0]
        return type(avg)(avg.spec, avg.n, avg.alpha,
                         sp.Lambda(y, avg.weight.expr.subs(D.BETA, sp.Rational(11, 10) * D.BETA)),
                         avg.note, avg.dropped)
    bad = D.Family(fam.formal, fam.annealed, fam.supports_disorder, bad_average,
                   fam.order_parameters, fam.variables)
    planted = _with_hook(D.FAMILIES, "hopfield", bad,
                         lambda: derived_replicon(HOP, vals, dps=10, saddle_tol=None).value)
    assert abs(planted - lam) > 1e-2, (planted, lam)
    assert abs(derived_replicon(HOP, vals, dps=10).value - lam) < TOL


@slow
def test_planted_site_hook_changes_the_replicon():
    # the spherical SITES hook: (1/2) ln det Q -> (3/5) ln det Q (explicit and symbolic n,
    # one function): the explicit-n site rule reads the registered hook
    svals, slam = _order_point("spherical", "0.6")
    site = D.SITES["spherical"]
    bad = D.Site(lambda n, q, qh, f=0: site.trace(n, q, qh, f) * sp.Rational(6, 5),
                 site.conjugate)
    planted = _with_hook(D.SITES, "spherical", bad,
                         lambda: derived_replicon(SPH, svals, saddle_tol=None).value)
    assert abs(planted - slam) > 1e-2, (planted, slam)
    assert abs(derived_replicon(SPH, svals).value - slam) < TOL


def test_unsupported_family_still_refuses():
    # an energy whose Gaussian covariance is q_ab^2 (not q_ab) has no replicon rule:
    # the replicon is UNKNOWN and said so (solve_derived then reports stable = None)
    fam = D.FAMILIES["perceptron"]

    def squared_cov(avg, n, sum_pairs):
        q, _, _ = D._overlap_symbols(n)
        return D.Reduction(avg.alpha * D.LnGaussProd(avg.weight, q[D._a, D._b]**2, n),
                           sp.S(0), None)
    bad = D.Family(fam.formal, fam.annealed, fam.supports_disorder, fam.average,
                   squared_cov, fam.variables, fam.constraints)
    rep = _with_hook(D.FAMILIES, "perceptron", bad, lambda: replicon(derive_rs(ISI)[1]))
    assert rep.expr is None and "replicon not derived for this family" in rep.note, rep
    assert "covariance" in rep.note, rep.note
    # the supported families no longer carry the refusal text
    for spec in (SPH, ISI, HOP, default_spec("sk")):
        r = replicon(derive_rs(spec)[1])
        assert r.expr is not None and "not derived" not in r.note, (spec, r.note)


# ---- slow tier -----------------------------------------------------------------------
@slow
def test_derived_replicon_equals_family_replicon():
    # spherical alpha in {0.5, 1.0, 1.5} (kappa = 0) and alpha = 1, kappa = -0.5 (negative
    # replicon); Ising {0.5, 0.8, 1.2}; Hopfield: the five Task 9 points.  Spherical
    # alpha = 1 and Ising alpha = 0.5 also through solve_derived_rs (derived saddle).
    for alpha, kappa in (("0.5", 0), ("1.0", 0), ("1.5", 0), ("1.0", "-0.5")):
        _agree(SPH, *_perceptron_point("spherical", alpha, kappa))
    for alpha in ("0.5", "0.8", "1.2"):
        _agree(ISI, *_perceptron_point("ising", alpha))
    targets = ("-11.22", "-1.033", "-0.0281", "0.969", "0.800")
    for (a, b, br), want in zip(HOPFIELD_POINTS, targets):
        vals, lam = _hopfield_point(a, b, br)
        assert abs(lam - mp.mpf(want)) < 10 ** -(len(want.split(".")[1])), (a, b, br, lam)
        _agree(HOP, vals, lam)
    s = solve_derived_rs(SPH, alpha=1, kappa=0)
    assert abs(s.replicon - _perceptron_point("spherical", "1.0")[1]) < TOL and s.stable
    s = solve_derived_rs(ISI, alpha="0.5", kappa=0)
    assert abs(s.replicon - _perceptron_point("ising", "0.5")[1]) < TOL
    assert s.stable is True and s.entropy > 0
    assert abs(derived_replicon(ISI, s).value - s.replicon) < 1e-15


@slow
def test_replicon_sign_flips_at_AT_line():
    # root of alpha -> lambda_R(derived saddle at alpha), Ising perceptron, kappa = 0;
    # target: the family's own root (families/perceptron.py), never an input
    from replicant.families.perceptron import _solve_alpha
    lam = lambda a: solve_derived_rs(ISI, alpha=a, kappa=0, dps=15, gh_nodes=300).replicon
    lo, hi = mp.mpf("1.0"), mp.mpf("1.03")
    assert lam(lo) > 0 > lam(hi)                       # positive control: a sign change
    with mp.workdps(15):
        root = mp.findroot(lam, (lo, hi), solver="anderson", tol=1e-18)
        target = mp.findroot(lambda a: _solve_alpha("ising", a, mp.mpf(0)).replicon, (lo, hi),
                             solver="anderson", tol=1e-18)
    assert abs(root - target) < 1e-7, (root, target)
    assert abs(root - mp.mpf("1.0147")) < 1e-3, root          # the number quoted in v1 docs
    assert lam(mp.mpf("0.5")) > 0 and lam(mp.mpf("1.2")) < 0


@slow
def test_composed_ising_perceptron_stability_is_computed():
    from replicant.compose import compose
    from test_compose import _ising_perceptron
    calc = compose(_ising_perceptron())
    below = calc.run(alpha="0.5", kappa=0, dps=15)
    assert below.stable is True and below.values["entropy"] > 0 and below.final
    assert below.values["replicon"] > 0
    assert {f.state for f in below.finality.values()} == {"stable"}
    assert any("derived replicon" in f.note for f in below.finality.values())
    above = calc.run(alpha="1.02", kappa=0, dps=15, gh_nodes=300)
    assert above.stable is False and not above.final and above.values["replicon"] < 0
    assert {f.state for f in above.finality.values()} == {"unstable"}
    r = _refusal(lambda: calc.run(alpha="1.02", kappa=0, dps=15, gh_nodes=300,
                                  require_final=True))
    assert r.reason == "unstable:rs", r
