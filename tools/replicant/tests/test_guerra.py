"""Guerra's replica-symmetry-breaking bounds on the SK pressure, certified (E10): bounds/.

Conventions (families/sk.py, Guerra, Comm. Math. Phys. 233, 1 (2003), arXiv:cond-mat/0205123): J_ij ~
N(0, 1/N), field h, pressure P = -beta f = lim (1/N) E ln Z_N.  Guerra's Theorem 3: P <= P_K(x) for
every admissible order parameter x, so each certified P_K at any admissible parameters is a certified
upper bound on the limit pressure.  The chain P_K(params_K) <= ... <= P_RS(q_RS) at the package's
own optimiser outputs is a consistency check of those optimisers (the theorem orders the infima).
Independent routes: mpmath tanh-sinh (no arb, no shared code) for the RS bound and its stationary q;
the package's float64 Gauss-Hermite functionals (families/sk.py, families/parisi.py) for 1RSB / K-RSB.
Planted errors go through bounds.krsb._HOOKS, restored in try/finally (identity asserted).
"""
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import mpmath as mp

from replicant.core import Refusal
from replicant.refdata import load
from replicant.testing import slow


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


def _pkg_rs(beta, h=0):
    from replicant.families.sk import sk_rs
    s = sk_rs(beta, h)
    return s.q, -mp.mpf(beta) * s.f


def test_rs_bound_certified_and_minimised_at_rs_q():
    from replicant.bounds import guerra_rs_best_bound, guerra_rs_bound, rs_bound_mp, rs_stationary_q_mp
    dps = mp.mp.dps
    best = guerra_rs_best_bound(2, 0)
    assert best.digits >= 40 and best.rad() < mp.mpf("1e-40")
    # all stationary points (lesson 6): q = 0 (a local maximum of the bound) and q* (the minimum)
    kinds = [(s.kind, s.q.contains(0)) for s in best.stationary]
    assert kinds == [("max", True), ("min", False)], kinds
    assert best.stable and best.second_variation.arb > 0
    q_pkg, p_pkg = _pkg_rs(2)
    assert abs(best.q.mid() - q_pkg) < mp.mpf("1e-12")            # sk_rs: q to ~1e-12 (dps 15)
    assert abs(best.mid() - p_pkg) < best.rad() + mp.mpf("1e-12")  # the package's RS pressure
    # independent: mpmath tanh-sinh (no arb), stationary q by the secant method; 25 digits here,
    # 40 in the slow tier (test_rs_bound_mpmath_forty_digits)
    q_mp = rs_stationary_q_mp(2, 0, dps=25)
    assert abs(best.q.mid() - q_mp) < mp.mpf("1e-22")
    assert abs(rs_bound_mp(2, 0, q_mp, dps=25) - best.mid()) < mp.mpf("1e-22")
    assert "G' < 0 certified" in best.certificate and "Lipschitz exclusion" in best.certificate
    # the bound at any other admissible q is larger (certified comparison)
    for q in ("0.3", "0.8", 1):
        other = guerra_rs_bound(2, 0, q)
        assert other.arb > best.arb, q
    assert mp.mp.dps == dps


@slow
def test_rs_bound_mpmath_forty_digits():
    from replicant.bounds import guerra_rs_best_bound, rs_bound_mp, rs_stationary_q_mp
    best = guerra_rs_best_bound(2, 0)
    q_mp = rs_stationary_q_mp(2, 0, dps=40)
    assert best.q.contains(q_mp) or abs(best.q.mid() - q_mp) < mp.mpf("1e-35")
    assert abs(rs_bound_mp(2, 0, q_mp, dps=40) - best.mid()) < mp.mpf("1e-35")


@slow
def test_rs_best_bound_with_a_field():
    """h != 0: stationary points by Lipschitz exclusion (|G'| <= 1 + beta^2) and a certified G' > 0
    on the surviving cluster; one minimum, equal to the package's RS solution."""
    from replicant.bounds import guerra_rs_best_bound, rs_bound_mp, rs_stationary_q_mp
    for beta, h, fh in (("3/2", "3/10", "0.3"), (2, "1/10", "0.1"), ("1/2", "1/2", "0.5")):
        hb = guerra_rs_best_bound(beta, h, digits=30)
        assert [s.kind for s in hb.stationary] == ["min"], (beta, h)
        bf = Fraction(beta)
        q_pkg, p_pkg = _pkg_rs(str(mp.mpf(bf.numerator) / bf.denominator), fh)
        assert abs(hb.q.mid() - q_pkg) < mp.mpf("1e-12") and abs(hb.mid() - p_pkg) < mp.mpf("1e-12")
        q_mp = rs_stationary_q_mp(beta, h, dps=30)
        assert abs(rs_bound_mp(beta, h, q_mp, dps=30) - hb.mid()) < mp.mpf("1e-27")


def test_high_temperature_bound_is_tight():
    from flint import arb
    from replicant.bounds import guerra_rs_best_bound
    from replicant.lattice.ball import baller
    for beta in ("1/2", "9/10", "99/100", 1):
        best = guerra_rs_best_bound(beta, 0)
        assert [s.kind for s in best.stationary] == ["min"]
        # beta < 1: G' > 0 certified on a cover of [0, 1]; beta = 1 only: the stated lemma
        assert ("lemma" in best.certificate) == (beta == 1), best.certificate
        assert ("G' > 0 certified" in best.certificate) == (beta != 1), best.certificate
        assert best.q.arb.is_zero()                      # the minimiser is q = 0, exactly
        b = Fraction(beta)
        with baller().hygiene.ctx_guard(prec=300):        # ln 2 + beta^2/4, built independently
            target = arb(2).log() + arb(b.numerator) ** 2 / (4 * arb(b.denominator) ** 2)
            assert best.arb.overlaps(target) and best.digits >= 40, beta
        assert best.stable == (b < 1)                    # P''(0) = (beta^2/2)(1 - beta^2)
    # the certification can fail: a planted shift of G' by -1 in the cover is refused, not passed
    from replicant.bounds import guerra
    reg = guerra._HOOKS
    old = reg["dG_shift"]
    reg["dG_shift"] = -1
    try:
        _refuses(lambda: guerra_rs_best_bound("1/2", 0), "undetermined:stationary-points")
        _refuses(lambda: guerra_rs_best_bound("9/10", 0), "undetermined:stationary-points")
    finally:
        reg["dG_shift"] = old
        assert guerra._HOOKS is reg and reg["dG_shift"] is old
    ref = load("guerra")["annealed_pressure_beta_half"]
    with mp.workdps(40):
        assert abs(guerra_rs_best_bound("1/2", 0).mid() - mp.mpf(ref.value)) < mp.mpf(ref.tol)


def test_chain_rs_1rsb_respected_by_package_values():
    from replicant.bounds import check_bound_chain
    rep = check_bound_chain(2, K=1, digits=8)
    assert rep.ok, rep.violations
    names = [lv.name for lv in rep.levels]
    assert names == ["rs", "1rsb"], names
    rs, one = rep.levels
    assert rs.certified and one.certified and one.functional == "ok" and rs.functional == "ok"
    assert one.bound.arb < rs.bound.arb                  # certified: P_1RSB(params) < P_RS(q_RS)
    assert [(l.upper, l.lower, l.verdict) for l in rep.links] == [("rs", "1rsb", "ok")]
    # the package's 1RSB pressure (float64 Gauss-Hermite) equals the certified functional
    assert abs(one.reported - one.bound.mid()) < one.bound.rad() + one.tol


def test_planted_violation_is_detected():
    from replicant.bounds import check_bound_chain, package_values
    from replicant.families.sk import _rsb_phi_e
    pv = package_values(2, K=1)
    rs, one = pv
    # a defective optimiser: q0 = 0, m = 1 (its own functional value; P = ln 2 + beta^2/4 > P_RS)
    q1 = one.qs[1]
    bad = one.replace(qs=(0, q1), ms=(1,), pressure=mp.mpf(_rsb_phi_e(2.0, 0.0, 0.0, float(q1), 1.0)[0]))
    rep = check_bound_chain(2, K=1, digits=8, package=(rs, bad))
    assert not rep.ok and rep.violations == ("ordering:1rsb>rs",), rep.violations
    assert rep.links[0].verdict == "violation"
    # second control: a package value that disagrees with its own functional by 1e-6
    off = one.replace(pressure=one.pressure + mp.mpf("1e-6"))
    rep = check_bound_chain(2, K=1, digits=8, package=(rs, off))
    assert rep.violations == ("functional:1rsb",), rep.violations
    assert rep.levels[1].functional == "mismatch"


def _p1rsb_q0_zero_mp(beta, q1, m, dps=30):
    """P_1RSB at q0 = 0 by mpmath tanh-sinh: ln 2 + (beta^2/4)(1 - 2 q1 + (1 - m) q1^2)
    + (1/m) ln E_y cosh^m(beta sqrt(q1) y) (no arb, no shared code)."""
    with mp.workdps(dps):
        b, q1, m = (mp.mpf(Fraction(x).numerator) / Fraction(x).denominator for x in (beta, q1, m))
        c = 1 / mp.sqrt(2 * mp.pi)
        inner = mp.quad(lambda y: c * mp.exp(-y * y / 2) * mp.cosh(b * mp.sqrt(q1) * y) ** m,
                        [-mp.inf, -5, 0, 5, mp.inf])
        return mp.log(2) + b * b / 4 * (1 - 2 * q1 + (1 - m) * q1 * q1) + mp.log(inner) / m


def test_planted_evaluator_error_is_detected():
    """The nested evaluator's innermost exponent planted (cosh^{1.01 m}): the chain's functional
    comparison catches it.  Cheap point q0 = 0 (the outer Gaussian is trivial); the reference is an
    independent mpmath evaluation (the package's float64 functional is only good to ~7e-10 at this
    off-optimum point: families/sk.py _rsb_phi_e gives 1.5962745995322627 vs 1.59627459887046...)."""
    from replicant.bounds import check_bound_chain, krsb, package_values
    rs = package_values(2, K=1)[0]
    pt = rs.replace(level=1, qs=(0, Fraction(3, 5)), ms=(Fraction(1, 2),),
                    pressure=_p1rsb_q0_zero_mp(2, "3/5", "1/2"), tol=mp.mpf("1e-25"))
    clean = check_bound_chain(2, K=1, digits=8, package=(rs, pt))
    assert clean.levels[1].functional == "ok", clean.levels[1].detail
    reg = krsb._HOOKS
    old = reg["inner_exponent_scale"]
    reg["inner_exponent_scale"] = Fraction(101, 100)
    try:
        rep = check_bound_chain(2, K=1, digits=8, package=(rs, pt))
    finally:
        reg["inner_exponent_scale"] = old
        assert krsb._HOOKS is reg and reg["inner_exponent_scale"] is old
    assert "functional:1rsb" in rep.violations, rep.violations


def test_guerra_quadratic_term_is_the_package_one():
    """Guerra Eq. (14) with identity (26): log 2 + f(0, h) - (beta^2/2) int q x(q) dq, x = m_k on
    [q_{k-1}, q_k), 1 on [q_K, 1]; the last segment's Gaussian closes to beta^2 (1 - q_K)/2."""
    import sympy as sp
    from replicant.bounds.krsb import quadratic_term
    from replicant.families.sk import quadratic_term as sk_quadratic
    q0, q1, m, q = sp.symbols("q0 q1 m q")
    assert sp.expand(quadratic_term((q0, q1), (m,)) - sk_quadratic("rsb")) == 0
    assert sp.expand(quadratic_term((q,), ()) - sk_quadratic("rs")) == 0


def test_bound_refusals():
    from replicant.bounds import check_bound_chain, guerra_1rsb_bound, guerra_krsb_bound, guerra_rs_bound
    _refuses(lambda: guerra_rs_bound(2, 0, "1.1"), "precondition:order-parameter")
    _refuses(lambda: guerra_rs_bound(2, 0, -1), "precondition:order-parameter")
    _refuses(lambda: guerra_rs_bound(0, 0, "0.5"), "precondition:beta")
    _refuses(lambda: guerra_rs_bound(True, 0, "0.5"), "precondition:beta")
    _refuses(lambda: guerra_1rsb_bound(2, "0.6", "0.5", "0.3"), "precondition:order-parameter")
    _refuses(lambda: guerra_1rsb_bound(2, "0.2", "0.5", 0), "precondition:order-parameter")
    _refuses(lambda: guerra_krsb_bound(2, ("0.1", "0.4", "0.6"), ("0.4", "0.2")),
             "precondition:order-parameter")                     # m must be nondecreasing
    _refuses(lambda: guerra_krsb_bound(2, ("0.1", "0.4"), ("0.2", "0.3")), "precondition:levels")
    _refuses(lambda: check_bound_chain(2, h="0.1", K=2), "unsupported:field")
    _refuses(lambda: check_bound_chain(2, K=0), "precondition:K")
    for d in (0, -5, "x", True, 1.5):
        _refuses(lambda: guerra_1rsb_bound(2, "0.2", "0.5", "0.3", digits=d), "precondition:digits")
        _refuses(lambda: guerra_rs_bound(2, 0, "0.5", digits=d), "precondition:digits")
        _refuses(lambda: check_bound_chain(2, digits=d), "precondition:digits")


def test_sk_bounds_entry_certified_and_labelled():
    from replicant.catalog import catalog
    from replicant.compose import compose
    e = catalog()["sk_bounds"]
    assert e.rigor == {"bound": "certified"} and e.default_route == "bound"
    res2 = compose(e.spec, "bound").run(beta=2, digits=8)
    assert res2.rigor == "certified"
    assert all(k in res2.errors for k in ("rs_bound", "rs_q", "rsb1_bound"))
    st = res2.finality["rs_bound"]
    assert st.state == "exact-by-literature" and "Guerra" in st.cite and "Theorem 3" in st.cite
    assert res2.values["chain_verdict"] == "consistent", res2.values
    assert res2.finality["rsb1_bound"].state == "exact-by-literature"
    assert res2.finality["rs_q"].state == "stable"
    _refuses(lambda: compose(e.spec, "bound").run(beta=2, K=3), "precondition:K")
    for d in (0, -5, "x"):
        _refuses(lambda: compose(e.spec, "bound").run(beta=2, digits=d), "precondition:digits")
    # rs-only (the package's sk_1rsb collapses to RS where RS is stable): a one-level chain
    from replicant.bounds import check_bound_chain, package_values
    rep = check_bound_chain(2, K=1, digits=8, package=package_values(2, K=1)[:1])
    assert rep.ok and [lv.name for lv in rep.levels] == ["rs"] and rep.links == () and rep.notes


@slow
def test_chain_respected_by_package_values():
    """beta = 2: the package's K-RSB staircases K = 1 .. 5 (families/parisi.py) and RS / 1RSB.
    Certified: RS, 1RSB and the 2RSB functional (nested acb.integral, ~1 min); K = 3 .. 5 are the
    package's numbers with its own error estimate (numerical, not certified)."""
    from replicant.bounds import check_bound_chain
    rep = check_bound_chain(2, K=5, digits=8)
    assert rep.ok, rep.violations
    assert [lv.name for lv in rep.levels] == ["rs", "1rsb", "2rsb", "3rsb", "4rsb", "5rsb"]
    assert [lv.certified for lv in rep.levels] == [True, True, True, False, False, False]
    assert all(l.verdict == "ok" for l in rep.links)
    assert [l.numerical for l in rep.links] == [False, False, True, True, True]
    two = rep.levels[2]
    assert two.functional == "ok" and abs(two.reported - two.bound.mid()) < two.bound.rad() + two.tol


@slow
def test_1rsb_bound_forty_digits_and_m_one_is_rs():
    from replicant.bounds import guerra_1rsb_bound, guerra_rs_bound
    from replicant.families.sk import sk_1rsb
    s = sk_1rsb(2)
    b = guerra_1rsb_bound(2, s.q0, s.q1, s.m, digits=40)
    assert b.digits >= 40 and abs(b.mid() + 2 * s.f) < mp.mpf("1e-10")
    # m = 1: (1/m) ln E_y cosh = ln cosh + beta^2 (q1 - q0)/2, so P_1RSB(q0, q1, 1) = P_RS(q0)
    one = guerra_1rsb_bound(2, "0.3", "0.7", 1, digits=25)
    assert one.overlaps(guerra_rs_bound(2, 0, "0.3"))
