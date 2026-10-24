"""Perceptron family: Gardner capacity (spherical, Ising) and teacher-student curves.

Conventions (stated once, used everywhere below):
  patterns xi^mu, mu = 1..P = alpha N, i.i.d. N(0, 1) components (+-1 patterns give
  the same N -> infinity limit); labels absorbed into the patterns; a coupling
  vector J stores the set if J.xi^mu / sqrt(N) >= kappa for every mu.
  spherical: J.J = N;  Ising: J in {-1, +1}^N.  H(x) = int_x^inf Dz.
  Entropies are per coupling: spherical S = ln(fraction of the sphere),
  Ising S = (1/N) ln #solutions (so S(alpha = 0) = ln 2).
  Teacher-student: spherical teacher and student, noiseless labels sign(T.xi),
  generalization error eps = arccos(R)/pi with R the teacher-student overlap.
"""
import os
import sys
from fractions import Fraction
from math import comb

import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.core import Refusal
from replicant.families import perceptron_counting as C
from replicant.families import teacher_student as TS
from replicant.families.perceptron import (gardner_capacity, gardner_capacity_curve,
                                           gardner_entropy, gardner_rs_point)
from replicant.families.teacher_student import asymptotic_constant, teacher_student_error
from replicant.refdata import check, load
from replicant.solve import gauss_expect
from replicant.testing import slow

REF = load("perceptron")


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, r.reason
        return r
    raise AssertionError(f"expected Refusal {reason}")


# ---- spherical capacity ----------------------------------------------------
def test_spherical_capacity_is_two():
    dps = mp.mp.dps
    with mp.workdps(30):
        c = gardner_capacity("spherical")
        assert abs(c.alpha_c - 2) < mp.mpf("1e-20")
    assert mp.mp.dps == dps                      # no leak of any working precision
    assert c.stable is True and c.ansatz == "rs" and "physics-level" in c.caveat
    assert check(REF["spherical_capacity_kappa0"], c.alpha_c, label="Gardner alpha_c").ok
    # independent of the closed form: the q -> 1 limit of our own RS saddle,
    # alpha(q) = q / ((1-q) E[(phi/H)^2]), parametrised by q (no root finding)
    pts = [gardner_rs_point("spherical", 1 - mp.mpf(10)**-k).alpha for k in (4, 8)]
    gaps = [abs(a - 2) for a in pts]
    assert gaps[0] > gaps[1] and gaps[1] < 1e-3, pts


def test_spherical_capacity_curve_monotone_and_matches_integral():
    ks = [mp.mpf(k) for k in ("-1", "-0.5", "0", "0.5", "1", "2")]
    curve = gardner_capacity_curve("spherical", ks)
    a = [c.alpha_c for c in curve]
    assert all(x > y for x, y in zip(a, a[1:])), a           # decreasing in kappa
    assert all(x > 2 for x, k in zip(a, ks) if k < 0)          # kappa < 0: alpha_c > 2
    assert all(x < 2 for x, k in zip(a, ks) if k > 0)
    for k, x in zip(ks, a):
        # Gardner 1988: alpha_c = [int_{-kappa}^inf Dt (t + kappa)^2]^-1.  The integrand
        # has a kink in its 2nd derivative at t = -kappa, off gauss_expect's breakpoints
        # unless kappa = 0, so tanh-sinh gives only ~1e-9 relative there (exact at 0)
        integral = gauss_expect(lambda t: (t + k)**2 if t > -k else mp.mpf(0))
        tol = mp.mpf("1e-25") if k == 0 else mp.mpf("1e-7")
        assert abs(x - 1 / integral) < tol * x, (k, x, 1 / integral)
    # RS is exact (stable) for kappa >= 0 and broken for kappa < 0 (Franz-Parisi 2015)
    assert [c.stable for c in curve] == [False, False, True, True, True, True]


def test_spherical_capacity_cover_counting():
    """Independent route (no replicas, no 2): Cover's (1965) count of linearly
    separable dichotomies of p points in general position in R^N, by the
    recurrence C(N, p+1) = C(N, p) + C(N-1, p), C(N>=1, 1) = 2, C(0, p) = 0, in
    exact integers.  The separable fraction C / 2^p is exactly 1/2 at p = 2N for
    every N, and is a sigmoid in alpha = p/N steepening around 2."""
    for N in (1, 2, 5, 17, 40):
        for p in (1, N, 2 * N, 3 * N + 1):
            assert C.cover_count(p, N) == 2 * sum(comb(p - 1, k) for k in range(N))
        assert C.cover_separable_fraction(2 * N, N) == Fraction(1, 2)
    lo = [float(C.cover_separable_fraction(int(1.8 * N), N)) for N in (50, 400)]
    hi = [float(C.cover_separable_fraction(int(2.2 * N), N)) for N in (50, 400)]
    assert lo[0] < lo[1] and lo[1] > 0.99 and hi[0] > hi[1] and hi[1] < 0.01, (lo, hi)
    _refuses(lambda: C.cover_count(0, 3), "precondition:positive-integers")


def test_spherical_rs_replicon_sign_follows_kappa():
    """Replicon lambda = 1 - alpha E[((ln H)''(x))^2] at the RS point: positive below
    capacity for kappa >= 0, -> kappa (kappa H(-kappa) + phi(kappa)) alpha_c at q -> 1."""
    for k in ("0", "0.5"):
        pt = gardner_rs_point("spherical", mp.mpf("0.9"), kappa=mp.mpf(k))
        assert pt.replicon > 0 and pt.alpha < gardner_capacity("spherical", mp.mpf(k)).alpha_c
    k = mp.mpf(-1)
    c = gardner_capacity("spherical", k)
    assert c.replicon < 0 and not c.stable and "kappa < 0" in c.note
    near = gardner_rs_point("spherical", 1 - mp.mpf("1e-6"), kappa=k)
    assert abs(near.replicon - c.replicon) < 1e-2, (near.replicon, c.replicon)
    _refuses(lambda: gardner_capacity("spherical", 0, ansatz="1rsb"),
             "not-implemented:spherical-rsb")


def test_capacity_is_where_entropy_vanishes():
    """Spherical version space: the plan's 'entropy vanishes' means q -> 1 and the
    RS entropy S = ln(volume fraction) -> -infinity at alpha_c; S is finite and
    decreasing below (the divergence law is the slow test below)."""
    s = gardner_entropy(mp.mpf("1.5"), "spherical")
    assert mp.isfinite(s) and s < 0
    assert gardner_entropy(2, "spherical") == mp.ninf
    assert gardner_entropy(mp.mpf("2.5"), "spherical") == mp.ninf
    lo, hi = (gardner_rs_point("spherical", mp.mpf(q)) for q in ("0.5", "0.999"))
    assert lo.alpha < 1.5 < hi.alpha and lo.S > s > hi.S


@slow
def test_spherical_entropy_divergence_law():
    """S(alpha) decreases monotonically and diverges like the RS law
    S ~ c ln(1 - alpha/alpha_c), c = (1 + alpha_c H(-kappa))/2 = 1 at kappa = 0
    (1/2 from ln(1-q), 1/2 from the -ln x tail of ln H on the violated half)."""
    s = [gardner_entropy(mp.mpf(a), "spherical") for a in ("0.5", "1", "1.5", "1.9")]
    assert all(x > y for x, y in zip(s, s[1:])) and s[0] < 0, s
    pts = [gardner_rs_point("spherical", 1 - mp.mpf(10)**-k) for k in (1, 2, 3, 4, 5, 6)]
    assert all(a.alpha < b.alpha and a.S > b.S for a, b in zip(pts, pts[1:]))
    rest = [p.S - mp.log(1 - p.alpha / 2) for p in pts]
    d = [abs(x - y) for x, y in zip(rest, rest[1:])]
    assert all(x > y for x, y in zip(d, d[1:])) and d[-1] < 1e-2, rest
    # a wrong exponent (1/2) leaves a residual drifting by ~ln(10)/2 per decade
    half = [p.S - mp.log(1 - p.alpha / 2) / 2 for p in pts]
    assert abs(half[-1] - half[-2]) > 1


# ---- Ising capacity ----------------------------------------------------------
def test_ising_capacity_rs_flagged_unstable_and_1rsb_matches_reference():
    rs = gardner_capacity("ising")
    assert rs.ansatz == "rs" and rs.stable is False
    assert "negative" in rs.note and "replicon" in rs.note
    assert check(REF["ising_rs_capacity"], rs.alpha_c, label="Ising RS alpha_c").ok
    assert rs.alpha_c > 1        # above the counting bound alpha <= 1: RS cannot be right
    one = gardner_capacity("ising", ansatz="1rsb")
    assert one.ansatz == "1rsb" and one.stable is True
    assert check(REF["ising_1rsb_capacity"], one.alpha_c, label="Krauth-Mezard alpha_c").ok
    assert one.alpha_c < 1 < rs.alpha_c


def test_ising_1rsb_point_matches_reference_overlaps():
    """The frozen-1RSB capacity is where the RS entropy vanishes; at that point our
    own RS saddle (q, qhat) must equal Ding-Sun's (153) bracket."""
    one = gardner_capacity("ising", ansatz="1rsb")
    pt = one.point
    assert abs(pt.S) < mp.mpf("1e-12") and abs(pt.alpha - one.alpha_c) < mp.mpf("1e-15")
    assert check(REF["ising_1rsb_q"], pt.q, label="KM q").ok
    assert check(REF["ising_1rsb_qhat"], pt.qhat, label="KM qhat").ok
    assert pt.replicon > 0           # AT-stable there (Ding-Sun Fig. 1a)


@slow
def test_ising_rs_capacity_limit_is_4_over_pi():
    """Our own RS saddle parametrised by qhat: alpha(qhat) climbs towards
    the banked RS capacity as qhat -> infinity (q -> 1)."""
    want = mp.mpf(REF["ising_rs_capacity"].value)
    pts = [gardner_rs_point("ising", mp.mpf(10)**k) for k in (2, 4)]
    a = [p.alpha for p in pts]
    assert a[0] < a[1] < want and want - a[1] < 2e-2 and want - a[1] < (want - a[0]) / 5, a
    assert all(p.q > 0.9 for p in pts)


def test_ising_rs_flags_instability():
    """RS Ising near its capacity: replicon 1 - alpha E[sech^4] E[(g'')^2] < 0
    (it diverges like -alpha/(3(1-q))), and S < 0; at Krauth-Mezard's 0.833 the
    replicon is still positive (Ding-Sun: AT condition holds there)."""
    far = gardner_rs_point("ising", mp.mpf(100))
    assert far.replicon < 0 and far.S < 0
    near = gardner_rs_point("ising", mp.mpf("2.6"))
    assert near.replicon > 0 and abs(near.alpha - mp.mpf("0.834")) < 2e-3


@slow
def test_ising_rs_refusals_and_at_point():
    """gardner_entropy refuses where RS is AT-unstable or its entropy is negative;
    replicant's own AT point of the Ising RS saddle (kappa = 0) is near 1.015,
    between Krauth-Mezard's 0.833 and the RS endpoint 4/pi."""
    r = _refuses(lambda: gardner_entropy(mp.mpf("1.1"), "ising"), "unstable:rs")
    assert "replicon" in r.detail
    r = _refuses(lambda: gardner_entropy(mp.mpf("0.9"), "ising"), "no-solution:negative-entropy")
    assert "1rsb" in r.detail
    lam = lambda u: gardner_rs_point("ising", mp.exp(u)).replicon
    u = mp.findroot(lam, (mp.log(5), mp.log(8)), solver="secant")
    a_at = gardner_rs_point("ising", mp.exp(u)).alpha
    assert 1.0 < a_at < 1.03, a_at


def test_ising_entropy_to_ln2_at_alpha_zero():
    with mp.workdps(30):
        assert abs(gardner_entropy(0, "ising") - mp.log(2)) < mp.mpf("1e-25")
    s = gardner_entropy(mp.mpf("0.3"), "ising")
    assert 0 < s < mp.log(2) * (1 - mp.mpf("0.3"))       # below annealed (Jensen)


@slow
def test_ising_entropy_small_alpha_is_annealed_to_second_order():
    # small alpha: annealed ln 2 (1 - alpha) up to O(alpha^2); along the saddle (by qhat)
    gaps = []
    for qh in (mp.mpf("0.01"), mp.mpf("0.02")):
        pt = gardner_rs_point("ising", qh)
        gaps.append((pt.alpha, pt.S - mp.log(2) * (1 - pt.alpha)))
    (a1, g1), (a2, g2) = gaps
    assert g1 < 0 and g2 < 0 and abs(g2 / g1 / (a2 / a1)**2 - 1) < 0.05, gaps


@slow
def test_rs_saddles_are_stationary_and_minimal_in_q():
    """Audit of the stationarity equations (which rest on the identity
    d/dq E_t ln H(x) = -E_t E(x)^2 / (2(1-q))): finite-difference gradients of the
    RS functionals vanish at gardner_rs_point's saddles, and S is a minimum in q
    there (the n -> 0 replica rule: maximum of f = minimum of the entropy)."""
    from replicant.families import perceptron as P
    with mp.workdps(20):
        for k in (mp.mpf(0), mp.mpf("0.7")):
            pt = gardner_rs_point("spherical", mp.mpf("0.6"), kappa=k)
            S = lambda q: (mp.log(1 - q) / 2 + q / (2 * (1 - q))
                           + pt.alpha * P._T(q, 1 - q, k).elnh())
            assert abs(mp.diff(S, pt.q)) < 1e-10 and mp.diff(S, pt.q, 2) > 0
        pt = gardner_rs_point("ising", mp.mpf("2.6"))
        def S2(q, qh):
            g = mp.sqrt(qh)
            return (-qh * (1 - q) / 2 + P._gauss(lambda z: mp.log(2 * mp.cosh(g * z)), P._zpts(g))
                    + pt.alpha * P._T(q, 1 - q, mp.mpf(0)).elnh())
        assert abs(mp.diff(lambda q: S2(q, pt.qhat), pt.q)) < 1e-10
        assert abs(mp.diff(lambda h: S2(pt.q, h), pt.qhat)) < 1e-10


def test_ising_enumeration_small_n():
    """Independent route, statistical and finite-size: exact enumeration of all
    2^N Ising couplings, M_N = largest M with a solution for nested Gaussian
    patterns.  Mean M_N/N at N = 11, 13 sits near 0.87 (finite-size drift above
    0.833), far below the RS value 4/pi = 1.27 and below the bound 1.  This
    distinguishes 1RSB from RS; it cannot certify 0.833's digits."""
    rs = gardner_capacity("ising").alpha_c
    km = mp.mpf(REF["ising_1rsb_capacity"].value)
    for N in (11, 13):
        mean, err = C.ising_capacity_enumeration(N, samples=120, seed=N)
        assert 0 < err < 0.03
        assert mean < 1 and rs - mean > 10 * err          # RS excluded
        assert abs(mean - float(km)) < 0.1                 # loose: finite size
    _refuses(lambda: C.ising_capacity_enumeration(30, samples=1), "precondition:small-N")


def test_perceptron_refusals():
    _refuses(lambda: gardner_capacity("potts"), "precondition:known-variables")
    _refuses(lambda: gardner_capacity("spherical", ansatz="frsb"), "precondition:known-ansatz")
    _refuses(lambda: gardner_entropy(-1, "spherical"), "precondition:nonnegative-alpha")
    _refuses(lambda: gardner_rs_point("spherical", 1), "precondition:order-in-range")


# ---- teacher-student ---------------------------------------------------------
def test_learning_curves_start_at_half_and_decrease():
    for learning in ("gibbs", "bayes"):
        with mp.workdps(30):
            assert abs(teacher_student_error(0, learning) - mp.mpf("0.5")) < mp.mpf("1e-25")
        e = [teacher_student_error(mp.mpf(a), learning) for a in ("0.5", "2", "8")]
        assert 0.5 > e[0] > e[1] > e[2] > 0, (learning, e)
    _refuses(lambda: teacher_student_error(1, "hebb"), "precondition:known-learning")
    _refuses(lambda: teacher_student_error(-1, "gibbs"), "precondition:nonnegative-alpha")


def test_bayes_better_than_gibbs():
    for a in ("0.5", "3", "20"):
        g, b = (teacher_student_error(mp.mpf(a), l) for l in ("gibbs", "bayes"))
        assert b < g, (a, b, g)


def test_gibbs_error_asymptotic_constant_matches_reference():
    for learning, key in (("gibbs", "gibbs_asymptotic_constant"),
                          ("bayes", "bayes_asymptotic_constant")):
        c = asymptotic_constant(learning)
        assert check(REF[key], c, label=f"{learning} constant").ok
        # independent of the closed form: alpha * eps(alpha) from the saddle
        # equation approaches it with O(1/alpha) corrections
        prods = [a * teacher_student_error(a, learning) for a in (mp.mpf(100), mp.mpf(1000))]
        gaps = [abs(p - c) for p in prods]
        assert gaps[1] < gaps[0] / 5 and gaps[1] < 2e-3, (learning, prods, c)
    with mp.workdps(30):
        assert abs(asymptotic_constant("gibbs") / asymptotic_constant("bayes") - mp.sqrt(2)) < 1e-20


@slow
def test_gibbs_saddle_is_stationary_in_q_and_r():
    """Audit: the Gibbs point q = R (Nishimori) is a genuine stationary point of
    the two-variable RS functional s(q, R) (teacher as an extra replica)."""
    a = mp.mpf(3)
    R = TS.gibbs_overlap(a)
    with mp.workdps(20):
        s = lambda q, r: TS.gibbs_functional(q, r, a)
        dq = mp.diff(lambda q: s(q, R), R)
        dr = mp.diff(lambda r: s(R, r), R)
    assert abs(dq) < 1e-10 and abs(dr) < 1e-10, (dq, dr)
