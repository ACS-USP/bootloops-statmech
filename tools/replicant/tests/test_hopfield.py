"""Hopfield model, replica-symmetric (Amit, Gutfreund & Sompolinsky 1985/1987).

Conventions (stated once, used everywhere below):
  H = -(1/2N) sum_{ij} sum_{mu=1}^p xi_i^mu xi_j^mu s_i s_j, p = alpha N random +-1
  patterns, one condensed pattern (overlap m), the rest as Gaussian noise of variance
  alpha r.  beta = 1/T.  f is the AGS free energy per spin, PRL 55, 1530, Eq. (5):
    f = alpha/2 + m^2/2 + alpha/(2 beta) [ln(1 - C) - beta q/(1 - C)]
        + alpha beta r (1 - q)/2 - (1/beta) E_z ln 2 cosh beta(m + sqrt(alpha r) z),
  C = beta(1 - q);  stationary at m = E tanh, q = E tanh^2, r = q/(1 - C)^2.
  Branches: "retrieval" (m > 0, the larger-m solution, locally stable in m),
  "spinglass" (m = 0, q > 0), "paramagnet" (m = q = r = 0).
"""
import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.core import Refusal
from replicant.families import hopfield as H
from replicant.families.hopfield import (hopfield_alpha_c, hopfield_free_energy,
                                         hopfield_gradient, hopfield_rs, hopfield_t0)
from replicant.refdata import check, load
from replicant.testing import slow

REF = load("hopfield")
BIG_BETA = 20


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


def _E(f, beta, m, s):
    """Test's own E_z f(beta (m + s z)) at 30 digits, with a breakpoint at the kink."""
    with mp.workdps(30):
        pts = [-mp.inf, -8, 0, 8, mp.inf]
        if s > 0 and abs(m / s) < 8 and m != 0:
            pts = sorted(pts[1:-1] + [-m / s])
            pts = [-mp.inf] + pts + [mp.inf]
        c = 1 / mp.sqrt(2 * mp.pi)
        return mp.quad(lambda z: f(beta * (m + s * z)) * c * mp.exp(-z * z / 2), pts)


# ---- the four tests from the brief -------------------------------------------
def test_retrieval_at_low_alpha_has_m_near_one():
    sol = hopfield_rs("0.01", BIG_BETA)
    assert sol.branch == "retrieval" and sol.stable and "physics-level" in sol.caveat
    assert 1 - mp.mpf("1e-9") < sol.m <= 1 and sol.q > sol.m**2 - mp.mpf("1e-9"), sol
    # AGS Eq. (12): 1 - m ~ (2 alpha/pi)^(1/2) exp(-1/(2 alpha)) at T = 0, alpha -> 0.
    # Our T = 0 erf solve at alpha = 0.01, 0.02 (exact equation, no asymptotics),
    # ratio extrapolated linearly to alpha = 0 (the correction is O(alpha)).
    with mp.workdps(40):
        ratio = {a: (1 - hopfield_t0(a, "retrieval").m) * mp.exp(1 / (2 * a)) / mp.sqrt(a)
                 for a in (mp.mpf("0.01"), mp.mpf("0.02"))}
        lim = 2 * ratio[mp.mpf("0.01")] - ratio[mp.mpf("0.02")]
    assert check(REF["low_alpha_error_prefactor"], lim, label="AGS Eq. (12) prefactor").ok, lim


def test_paramagnet_boundary():
    """T_g = 1 + sqrt(alpha): just below, a q > 0 spin-glass solution exists and the
    paramagnet is unstable; just above, only the paramagnet (stable)."""
    alpha = mp.mpf("0.1")
    tg = 1 + mp.sqrt(alpha)
    below, above = 1 / (tg - mp.mpf("0.03")), 1 / (tg + mp.mpf("0.03"))
    sg = hopfield_rs(alpha, below, branch="spinglass")
    assert sg.branch == "spinglass" and sg.m == 0 and 0 < sg.q < mp.mpf("0.2"), sg
    pm = hopfield_rs(alpha, below, branch="paramagnet")
    assert pm.m == pm.q == pm.r == 0 and pm.stable is False and "T_g" in pm.note
    r = _refuses(lambda: hopfield_rs(alpha, above, branch="spinglass"), "no-solution:spinglass")
    assert "T_g" in r.detail and "paramagnet" in r.detail, r.detail
    pm = hopfield_rs(alpha, above, branch="paramagnet")
    assert pm.stable is True


def test_alpha_c_matches_reference():
    dps = mp.mp.dps
    ac = hopfield_alpha_c()
    assert mp.mp.dps == dps
    assert check(REF["alpha_c_T0"], ac, label="AGS alpha_c, T = 0").ok, ac
    # Steffan & Kuehn quote 9 digits: 0.137 905 566
    assert abs(ac - mp.mpf(REF["alpha_c_T0"].value)) < mp.mpf("1e-9"), ac
    # independent of the double-root solve: count sign changes of
    # h(y) = erf(y) - y (sqrt(2 alpha) + (2/sqrt(pi)) e^{-y^2}) on a grid, y in (0, 4]
    def n_roots(a):
        with mp.workdps(30):
            h = lambda y: mp.erf(y) - y * (mp.sqrt(2 * a) + 2 / mp.sqrt(mp.pi) * mp.exp(-y * y))
            ys = [mp.mpf(k) / 400 for k in range(1, 1601)]
            v = [h(y) for y in ys]
            return sum(1 for x, y in zip(v, v[1:]) if x * y < 0)
    assert n_roots(ac - mp.mpf("1e-4")) == 2 and n_roots(ac + mp.mpf("1e-4")) == 0
    t0 = hopfield_t0(ac, "retrieval")
    assert check(REF["m_at_alpha_c_T0"], t0.m, label="m at alpha_c").ok, t0.m
    assert check(REF["energy_at_alpha_c_T0"], t0.E, label="E at alpha_c").ok, t0.E
    assert abs(t0.m - mp.mpf(REF["m_at_alpha_c_T0"].value)) < mp.mpf("1e-6")
    assert abs(t0.E - mp.mpf(REF["energy_at_alpha_c_T0"].value)) < mp.mpf("1e-8")
    _refuses(lambda: hopfield_t0(ac + mp.mpf("1e-4"), "retrieval"), "no-solution:retrieval")


def test_branch_is_named_and_selectable():
    r = hopfield_rs("0.1", BIG_BETA, branch="retrieval")
    s = hopfield_rs("0.1", BIG_BETA, branch="spinglass")
    assert (r.branch, s.branch) == ("retrieval", "spinglass")
    assert r.m > mp.mpf("0.9") and s.m == 0 and s.q > mp.mpf("0.9"), (r, s)
    assert r.stable and s.stable
    assert hopfield_rs("0.1", BIG_BETA) == r          # default branch is retrieval
    # below T = 1 the q = 0 point is outside the RS functional's domain (ln(1 - beta))
    d = _refuses(lambda: hopfield_rs("0.1", BIG_BETA, branch="paramagnet"),
                 "no-solution:paramagnet").detail
    assert "T" in d and "1" in d


# ---- refusals that say why ----------------------------------------------------
def test_unknown_branch_refuses():
    r = _refuses(lambda: hopfield_rs("0.1", 5, branch="mixture"), "precondition:known-branch")
    assert "retrieval" in r.detail and "spinglass" in r.detail and "paramagnet" in r.detail
    _refuses(lambda: hopfield_rs("-0.1", 5), "precondition:alpha-positive")
    _refuses(lambda: hopfield_rs("0.1", 0), "precondition:beta-positive")
    _refuses(lambda: hopfield_t0("0.1", "paramagnet"), "precondition:known-branch")


def test_retrieval_refuses_at_high_temperature():
    r = _refuses(lambda: hopfield_rs("0.05", "0.9"), "no-solution:retrieval")
    assert "T" in r.detail and "tanh" in r.detail, r.detail
    _refuses(lambda: hopfield_alpha_c("0.9"), "no-solution:retrieval")


@slow
def test_retrieval_above_alpha_c_refuses():
    ac = hopfield_alpha_c(BIG_BETA)
    assert mp.mpf("0.1375") < ac < mp.mpf("0.1383"), ac       # 0.137794 at T = 0.05
    r = _refuses(lambda: hopfield_rs("0.15", BIG_BETA), "no-solution:retrieval")
    assert "above alpha_c(T)" in r.detail, r.detail
    ok = hopfield_rs(ac - mp.mpf("1e-3"), BIG_BETA)      # just below: found and stable
    assert ok.branch == "retrieval" and ok.stable and ok.m > mp.mpf("0.95")


# ---- free-energy ordering (AGS 1985) --------------------------------------------
def test_free_energy_ordering():
    """AGS PRL (1985): 'the SG energy is lower for 0.051 < alpha < 0.138, whereas the
    FM state becomes the absolute minimum below 0.051' (T = 0).  Checked at T = 1/20
    with the finite-beta solver and at T = 0 with the erf reduction."""
    for a, retrieval_lower in (("0.1", False), ("0.03", True)):
        fr = hopfield_rs(a, BIG_BETA).f
        fs = hopfield_rs(a, BIG_BETA, branch="spinglass").f
        assert (fr < fs) == retrieval_lower, (a, fr, fs)
    with mp.workdps(30):
        d = lambda a: hopfield_t0(a, "retrieval").E - hopfield_t0(a, "spinglass").E
        assert d(mp.mpf("0.03")) < 0 < d(mp.mpf("0.13"))
        a_star = mp.findroot(d, (mp.mpf("0.045"), mp.mpf("0.055")), solver="secant")
    assert check(REF["alpha_global_T0"], a_star, label="FM global below alpha").ok, a_star


@slow
def test_free_energy_ordering_near_alpha_c():
    fr = hopfield_rs("0.135", BIG_BETA).f
    fs = hopfield_rs("0.135", BIG_BETA, branch="spinglass").f
    assert fs < fr, (fr, fs)


# ---- every returned point is a genuine stationary point -------------------------
def test_returned_points_satisfy_equations():
    a, b = mp.mpf("0.1"), mp.mpf(BIG_BETA)
    for br in ("retrieval", "spinglass"):
        s = hopfield_rs(a, b, branch=br)
        with mp.workdps(30):
            sig = mp.sqrt(a * s.r)
            em = _E(mp.tanh, b, s.m, sig) - s.m
            eq = _E(lambda x: mp.tanh(x)**2, b, s.m, sig) - s.q
            er = s.q / (1 - b * (1 - s.q))**2 - s.r
        assert max(abs(em), abs(eq), abs(er)) < mp.mpf("1e-12"), (br, em, eq, er)
        assert s.residual < mp.mpf("1e-12") and max(abs(g) for g in s.grad) < mp.mpf("1e-8")
        # the test's own audit: numerical gradient of the AGS functional at the point
        with mp.workdps(20):
            F = lambda m, q, r: hopfield_free_energy(m, q, r, a, b)
            h = mp.mpf("1e-7")
            g = [(F(s.m + h, s.q, s.r) - F(s.m - h, s.q, s.r)) / (2 * h),
                 (F(s.m, s.q + h, s.r) - F(s.m, s.q - h, s.r)) / (2 * h),
                 (F(s.m, s.q, s.r + h) - F(s.m, s.q, s.r - h)) / (2 * h)]
        assert max(abs(x) for x in g) < mp.mpf("1e-7"), (br, g)
        assert abs(F(s.m, s.q, s.r) - s.f) < mp.mpf("1e-14")


def test_gradient_formula_matches_numerical_derivative():
    """The stationarity equations come from the analytic gradient; check it against
    central differences of the functional at generic (non-stationary) points."""
    with mp.workdps(30):
        for (m, q, r, a, b) in (("0.7", "0.8", "1.3", "0.1", "3"), ("0.2", "0.4", "0.5", "0.2", "0.8")):
            m, q, r, a, b = map(mp.mpf, (m, q, r, a, b))
            F = lambda *v: hopfield_free_energy(*v, a, b)
            h = mp.mpf("1e-9")
            num = [(F(*[v + h * (i == k) for i, v in enumerate((m, q, r))])
                    - F(*[v - h * (i == k) for i, v in enumerate((m, q, r))])) / (2 * h)
                   for k in range(3)]
            ana = hopfield_gradient(m, q, r, a, b)
            assert all(abs(x - y) < mp.mpf("1e-12") for x, y in zip(num, ana)), (num, ana)
            assert max(abs(x) for x in ana) > mp.mpf("1e-3")      # really non-stationary


def test_quadrature_matches_mpmath_quad():
    """The module's composite Gauss-Legendre E_z against mp.quad at 30 digits, including
    a sharp step (beta = 100) and a near-zero width."""
    with mp.workdps(20):
        for b, m, s in (("100", "0.97", "0.45"), ("1.5", "0.3", "1e-6")):
            b, m, s = map(mp.mpf, (b, m, s))
            got = H._moments(b, m, s)
            for k in (1, 4):          # odd and even; the others share the same nodes
                g = got[k - 1]
                want = _E(lambda x, k=k: mp.tanh(x)**k, b, m, s)
                assert abs(g - want) < mp.mpf("1e-17"), (b, m, s, k, g - want)


def test_functional_reduces_to_ags_energy_at_low_temperature():
    """f(beta = 20) of each branch is within O(T S) of the T = 0 energies from the erf
    reduction (AGS Eq. (11), E = alpha (1 - r)/2 - m^2/2), and alpha -> 0 gives the
    Mattis value -T ln 2cosh(beta m) + m^2/2 at m = tanh(beta m)."""
    for br in ("retrieval", "spinglass"):
        f = hopfield_rs("0.1", BIG_BETA, branch=br).f
        e0 = hopfield_t0("0.1", br).E
        assert abs(f - e0) < mp.mpf("5e-3"), (br, f, e0)
    with mp.workdps(30):
        b = mp.mpf(3)
        m = mp.findroot(lambda m: m - mp.tanh(b * m), 0.9)
        mattis = m**2 / 2 - mp.log(2 * mp.cosh(b * m)) / b
        f0 = hopfield_free_energy(m, m**2, 0, 0, b)
    assert abs(f0 - mattis) < mp.mpf("1e-25")


def test_no_global_precision_leak():
    dps = mp.mp.dps
    hopfield_rs("0.1", BIG_BETA)
    hopfield_t0("0.05", "spinglass")
    assert mp.mp.dps == dps


# ---- finite-temperature retrieval boundary (slow) --------------------------------
@slow
def test_alpha_c_finite_T_is_reentrant():
    """Steffan & Kuehn (1994): the RS boundary alpha_c(T) backbends; its maximum
    0.138 188 5 sits at T = 0.024, above the T = 0 value 0.137 905 566.  We find the
    maximum value to 7 digits but its location at T = 0.0229 (see the bank entry)."""
    Ts = [mp.mpf("0.020"), mp.mpf("0.023"), mp.mpf("0.026")]
    acs = [hopfield_alpha_c(1 / T) for T in Ts]
    assert acs[1] > acs[0] and acs[1] > acs[2] and acs[0] > hopfield_alpha_c(), acs
    assert check(REF["alpha_c_rs_max_over_T"], acs[1], label="max alpha_c(T)").ok, acs
    with mp.workdps(30):    # vertex of the parabola through the three points
        (t0, t1, t2), (a0, a1, a2) = Ts, acs
        num = (t1 - t0)**2 * (a1 - a2) - (t1 - t2)**2 * (a1 - a0)
        den = (t1 - t0) * (a1 - a2) - (t1 - t2) * (a1 - a0)
        tmax = t1 - num / (2 * den)
    assert check(REF["T_at_alpha_c_rs_max"], tmax, label="T at max alpha_c").ok, tmax
    # inside the reentrant sliver (alpha above the T = 0 alpha_c) retrieval exists at
    # T = 0.023 but not at T = 0.  Second route to the same point: the m-parametrised
    # curve alpha(m) (the solver's fallback), bracketed on the upper branch.
    a, b = mp.mpf("0.13815"), 1 / Ts[1]
    sol = hopfield_rs(a, b)
    assert sol.branch == "retrieval" and sol.stable and sol.residual < mp.mpf("1e-14")
    with mp.workdps(20):
        _, mf = H._fold_finite(H._key(b))
        m_alt = H._root(lambda x: H._alpha_of_m(b, x)[0] - a, mf,
                        H._mattis(b) * (1 - mp.mpf("1e-12")), xtol="1e-14")
    assert m_alt > mf and abs(m_alt - sol.m) < mp.mpf("1e-10"), (m_alt, sol.m)
    _refuses(lambda: hopfield_t0(a, "retrieval"), "no-solution:retrieval")


@slow
def test_retrieval_at_moderate_temperature():
    """T = 1/3: seeds from the Mattis state; m-stable; m below the T = 0 value."""
    sol = hopfield_rs("0.05", 3)
    t0 = hopfield_t0("0.05", "retrieval")
    assert sol.stable and mp.mpf("0.8") < sol.m < t0.m and sol.residual < mp.mpf("1e-14"), sol


@slow
def test_tg_by_bisection_on_existence():
    """Locate T_g at alpha = 0.05, 0.2 by bisection on 'spin-glass branch exists',
    using only hopfield_rs; compare (T_g - 1)/sqrt(alpha) to the banked coefficient."""
    for a in (mp.mpf("0.05"), mp.mpf("0.2")):
        lo, hi = mp.mpf(1), mp.mpf(2)          # temperatures: SG exists at lo, not at hi
        for _ in range(26):
            mid = (lo + hi) / 2
            try:
                hopfield_rs(a, 1 / mid, branch="spinglass")
                lo = mid
            except Refusal:
                hi = mid
        coef = ((lo + hi) / 2 - 1) / mp.sqrt(a)
        q_near = hopfield_rs(a, 1 / lo, branch="spinglass").q      # continuous: q -> 0
        assert 0 < q_near < mp.mpf("1e-5"), q_near
        assert check(REF["tg_sqrt_alpha_coefficient"], coef, label="T_g").ok, (a, coef)
