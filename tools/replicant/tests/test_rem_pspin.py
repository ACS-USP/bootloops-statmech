"""REM (Derrida 1980/81) and spherical p-spin (Crisanti-Sommers 1992), family (b).

Normalisations (stated once, used everywhere below):
  REM:    2^N i.i.d. energies E_i ~ N(0, N_scale * N).  N_scale = 1 (default)
          gives beta_c = sqrt(2 ln 2); Derrida's own choice P(E) ~ exp(-E^2/(N J^2))
          is N_scale = J^2/2, i.e. N_scale = 1/2 at J = 1, beta_c = 2 sqrt(ln 2).
          Banked values are in Derrida's normalisation and converted here.
  p-spin: H = -sum_{i1<..<ip} J sigma_i1..sigma_ip, sum sigma^2 = N,
          J ~ N(0, p!/(2 N^(p-1))) (J = 1), so E H^2 = N/2: the REM with N_scale = 1/2.
"""
import os
import sys

import mpmath as mp
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.core import Refusal
from replicant.families import pspin as P
from replicant.families.pspin import (pspin_critical_temperature, pspin_functional,
                                      pspin_spherical_1rsb)
from replicant.families.rem import (rem_1rsb, rem_1rsb_functional, rem_critical_beta,
                                    rem_entropy, rem_free_energy, rem_quenched_finite_n)
from replicant.refdata import check, load
from replicant.testing import slow

REM, PS = load("rem"), load("pspin")
HALF = mp.mpf("0.5")


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, r.reason
        return r
    raise AssertionError(f"expected Refusal {reason}")


# ---- REM --------------------------------------------------------------------
def test_rem_critical_beta_closed_form():
    with mp.workdps(30):
        bc = rem_critical_beta()
        assert abs(bc - mp.sqrt(2 * mp.log(2))) < mp.mpf("1e-25")
        # independent route: the entropy s(e) = ln 2 - e^2/(2 N_scale) vanishes at
        # e0 < 0 and beta_c = s'(e0) is the slope there (Derrida's microcanonical argument)
        for s in (mp.mpf(1), HALF, mp.mpf(3)):
            e0 = mp.findroot(lambda e: rem_entropy(e, N_scale=s), -1)
            slope = mp.diff(lambda e: rem_entropy(e, N_scale=s), e0)
            assert abs(slope - rem_critical_beta(N_scale=s)) < mp.mpf("1e-20")
        # Derrida's normalisation (N_scale = 1/2): T_c = 1/(2 sqrt(ln 2))
        assert check(REM["tc_derrida"], 1 / rem_critical_beta(N_scale=HALF), label="REM Tc").ok
    _refuses(lambda: rem_critical_beta(N_scale=0), "precondition:positive-N_scale")


def test_rem_free_energy_continuous_at_critical_beta():
    with mp.workdps(30):
        for s in (mp.mpf(1), HALF):
            bc, d = rem_critical_beta(N_scale=s), mp.mpf("1e-12")
            lo, hi = rem_free_energy(bc - d, N_scale=s), rem_free_energy(bc + d, N_scale=s)
            assert abs(lo - hi) < mp.mpf("1e-11")
            # energy e = d(beta f)/d beta is continuous too (only C jumps)
            e = lambda b: mp.diff(lambda x: x * rem_free_energy(x, N_scale=s), b)
            assert abs(e(bc * mp.mpf("0.999999")) - e(bc * mp.mpf("1.000001"))) < mp.mpf("1e-5")
            # high-T branch is the annealed one: -beta f = ln 2 + N_scale beta^2 / 2
            b = bc / 2
            assert abs(-b * rem_free_energy(b, N_scale=s) - (mp.log(2) + s * b**2 / 2)) < mp.mpf("1e-25")
        # frozen phase (Derrida, J = 1): f = E_ground / N = -sqrt(ln 2), independent of T
        for b in (3, 5, 40):
            assert check(REM["ground_energy_derrida"], rem_free_energy(b, N_scale=HALF),
                         label="REM frozen f").ok
    _refuses(lambda: rem_free_energy(0), "precondition:positive-beta")


def test_rem_1rsb_m_is_beta_c_over_beta():
    m, beta, s = sp.symbols("m beta s", positive=True)
    phi = rem_1rsb_functional()          # -beta f(m), from ansatz.sum_offdiag_power + n -> 0
    assert sp.simplify(phi - (sp.log(2) / m + s * beta**2 * m / 2)) == 0
    (m_star,) = sp.solve(sp.diff(phi, m), m)
    assert sp.simplify(m_star - sp.sqrt(2 * sp.log(2) / s) / beta) == 0
    with mp.workdps(30):
        for s_val in (mp.mpf(1), HALF):
            bc = rem_critical_beta(N_scale=s_val)
            for x in ("1", "1.5", "4"):
                b = bc * mp.mpf(x)
                sol = rem_1rsb(b, N_scale=s_val)
                assert check(REM["m_times_beta_over_beta_c"], sol.m * b / bc, label="REM m").ok
                assert sol.q0 == 0 and sol.q1 == 1
                assert abs(sol.f - rem_free_energy(b, N_scale=s_val)) < mp.mpf("1e-25")
                # replica rule: f is maximal in m at the 1RSB point (f'' < 0)
                f_m = lambda mm: -(mp.log(2) / mm + s_val * b**2 * mm / 2) / b
                assert mp.diff(f_m, sol.m, 2) < 0
        r = _refuses(lambda: rem_1rsb(rem_critical_beta() / 2), "no-solution")
        assert "paramagnet" in r.detail and "m" in r.detail
    assert "physics-level" in rem_1rsb(3).caveat


def test_rem_finite_n_quenched_matches_limit_and_asymptotics():
    """Exact finite-N quenched (1/N) E ln Z by quadrature (no replicas), Derrida
    normalisation.  High T: equals annealed up to exponentially small terms.
    Frozen phase: below the limit, approaching it, and close to the
    Derrida-Mottishaw large-N expansion (their Eq. 4)."""
    bc = float(rem_critical_beta(N_scale=HALF))
    hot = rem_quenched_finite_n(0.5 * bc, 40, N_scale=0.5)
    assert abs(hot - float(rem_free_energy(0.5 * bc, N_scale=HALF)) * -0.5 * bc) < 1e-7
    b = 2 * bc
    lim = -b * float(rem_free_energy(b, N_scale=HALF))
    gaps = [rem_quenched_finite_n(b, N, N_scale=0.5) - lim for N in (20, 40, 80)]
    assert all(g < 0 for g in gaps) and gaps[0] < gaps[1] < gaps[2]
    # o(1/N) remainder: N * |gap - expansion| = 0.075 at N = 80, shrinking with N
    assert 80 * abs(gaps[2] - _dm_asymptotic_gap(b, bc, 80)) < 0.1


def _dm_asymptotic_gap(b, bc, N):
    """Derrida & Mottishaw, J. Phys. A 55 (2022), arXiv:2202.12584, Eq. (4), per spin, minus the limit."""
    import math
    return (-(b / (2 * bc)) * math.log(4 * N * math.pi * math.log(2))
            + (b / bc) * math.lgamma(1 - bc / b) + (1 - b / bc) * (-0.5772156649015329)) / N


@slow
def test_rem_finite_n_log_n_correction():
    bc = float(rem_critical_beta(N_scale=HALF))
    b = 2 * bc
    lim = -b * float(rem_free_energy(b, N_scale=HALF))
    Ns = (40, 80, 160, 320)
    gaps = [rem_quenched_finite_n(b, N, N_scale=0.5) - lim for N in Ns]
    rem = [N * abs(g - _dm_asymptotic_gap(b, bc, N)) for N, g in zip(Ns, gaps)]
    assert all(a > c for a, c in zip(rem, rem[1:])) and rem[-1] < 0.03, rem
    # N * gap grows like -(beta / (2 beta_c)) ln N: slope per doubling -> (b/2bc) ln 2
    steps = [N2 * g2 - N1 * g1 for (N1, g1), (N2, g2) in zip(zip(Ns, gaps), zip(Ns[1:], gaps[1:]))]
    assert abs(steps[-1] + (b / (2 * bc)) * mp.log(2)) < 0.05


# ---- spherical p-spin -------------------------------------------------------
def test_pspin_tc_matches_reference_p3():
    ts = pspin_critical_temperature(3)
    assert check(PS["ts_p3"], ts, label="p-spin T_s(p=3)").ok
    # independent route: at m = 1 the m-equation of OUR functional is g(q) = 0 and
    # the 1RSB branch is born where g touches zero, g'(q) = 0 (Castellani-Cavagna Eq. 84)
    q, m, b = sp.symbols("q m beta")
    g = sp.diff(pspin_functional(3).subs(sp.Symbol("q0"), 0), m).subs(m, 1)
    G = sp.lambdify((q, b), [g, sp.diff(g, q)], "mpmath")
    with mp.workdps(30):
        qs, bs = mp.findroot(lambda x, y: G(x, y), (mp.mpf("0.6"), mp.mpf("1.7")))
        assert abs(1 / bs - ts) < mp.mpf("1e-20")
        assert check(PS["qs_p3"], qs, label="p-spin q_s(p=3)").ok


def test_pspin_dynamic_temperature_p3():
    td = pspin_critical_temperature(3, kind="dynamic")
    with mp.workdps(30):
        assert abs(td - mp.sqrt(mp.mpf(3) / 8)) < mp.mpf("1e-25")
    assert check(PS["td_p3"], td, label="p-spin T_d(p=3)").ok
    assert td > pspin_critical_temperature(3)
    _refuses(lambda: pspin_critical_temperature(3, kind="tc"), "precondition:known-transition")


def test_pspin_ground_energy_p3():
    sol = pspin_spherical_1rsb(3, 10000)
    assert check(PS["e0_p3"], sol.e, label="p-spin e0(p=3)").ok
    assert abs(sol.q1 - 1) < 1e-3 and sol.m < 1e-3


def test_pspin_high_temperature_is_paramagnetic():
    ts = pspin_critical_temperature(3)
    for T in (ts * mp.mpf("1.01"), mp.mpf("0.62"), mp.mpf(1)):
        r = _refuses(lambda: pspin_spherical_1rsb(3, 1 / T), "no-solution")
        assert "paramagnet" in r.detail and "T_s" in r.detail
    _refuses(lambda: pspin_spherical_1rsb(2, 3), "precondition:p-at-least-3")
    _refuses(lambda: pspin_spherical_1rsb(3, -1), "precondition:positive-beta")


def test_pspin_logdet_n_to_zero_matches_integer_n():
    """The ln det Q piece of the functional: eigenvalues of the 1RSB matrix,
    checked against ansatz.rsb_matrix at integer n before n -> 0."""
    from replicant.ansatz import rsb_matrix
    q0, q1 = sp.Rational(1, 5), sp.Rational(3, 5)
    for n, m in ((4, 2), (6, 3), (6, 2)):
        lam1, lam2 = 1 - q1, m * (q1 - q0) + 1 - q1
        lam3 = n * q0 + lam2
        want = (n - sp.Rational(n, m)) * sp.log(lam1) + (sp.Rational(n, m) - 1) * sp.log(lam2) + sp.log(lam3)
        got = sp.log(rsb_matrix([m], [q0, q1], n).det())
        assert abs(sp.N(got - want, 30)) < 1e-25
    q, m, q0 = sp.symbols("q m q0")
    lam2 = 1 - q + m * (q - q0)         # Castellani-Cavagna Eq. (81), times 1
    want0 = (1 - 1 / m) * sp.log(1 - q) + sp.log(lam2) / m + q0 / lam2
    got0 = P.logdet_n_to_zero()
    R = sp.Rational
    for vals in ({q: R(3, 5), m: R(2, 5), q0: R(1, 10)}, {q: R(9, 10), m: R(1, 20), q0: 0}):
        assert abs(sp.N((got0 - want0).subs(vals), 30)) < 1e-25


def test_pspin_1rsb_below_ts_is_local_maximum():
    for T in ("0.3", "0.5", "0.58"):
        b = 1 / mp.mpf(T)
        sol = pspin_spherical_1rsb(3, b)
        assert 0 < sol.m < 1 and sol.q0 == 0 and 0 < sol.q1 < 1
        eigs = P._hessian_f_eigs(3, b, sol.q0, sol.q1, sol.m)     # in (q0, q1, m)
        assert len(eigs) == 3 and max(eigs) < 0, eigs
        assert -b * sol.f < b**2 / 4             # f above the paramagnet's (replica rule)
        with mp.workdps(30):
            assert abs(sol.e + b / 2 * (1 + (sol.m - 1) * sol.q1**3)) < mp.mpf("1e-20")
    assert "physics-level" in sol.caveat


def test_pspin_approaches_rem_as_p_grows():
    """At fixed reduced inverse temperature x = beta/beta_s(p), as p grows the
    1RSB solution tends to the REM's (both with E H^2 = N/2): m -> beta_s/beta
    and the free-energy drop below the paramagnet, in units of beta_s^2, tends
    to the REM's -(x-1)^2/4.  The approach is monotone and slow (~1/ln p)."""
    x = mp.mpf(2)
    bc = rem_critical_beta(N_scale=HALF)
    b_rem = x * bc
    gap_rem = (-b_rem * rem_free_energy(b_rem, N_scale=HALF)
               - (mp.log(2) + b_rem**2 / 4)) / bc**2
    assert abs(gap_rem + (x - 1)**2 / 4) < mp.mpf("1e-14")
    m_ratio, gap_ratio = [], []
    for p in (3, 5, 10, 20, 100, 1000):
        bs = 1 / pspin_critical_temperature(p)
        b = x * bs
        sol = pspin_spherical_1rsb(p, b)
        m_ratio.append(sol.m * x)
        gap_ratio.append(((-b * sol.f) - b**2 / 4) / bs**2 / gap_rem)
    assert all(a < c for a, c in zip(m_ratio, m_ratio[1:])), m_ratio
    assert all(a < c for a, c in zip(gap_ratio, gap_ratio[1:])), gap_ratio
    assert all(r < 1 for r in m_ratio + gap_ratio)
    assert 1 - m_ratio[-1] < 0.04 and 1 - gap_ratio[-1] < 0.04


def test_pspin_ts_decreases_like_rem_with_growing_entropy():
    """T_s(p) -> 0 as p grows: from 1/p = y(ln(1/y) - 1) + O(y^2), p y ~ 1/ln p and
    T_s^2 = (p y/2)(1-y)^(p-2) ~ 1/(2 ln p), a REM transition (E H^2 = N/2) with
    configurational entropy ~ (ln p)/2.  Corrections are ln ln p / ln p: the
    ratio T_s sqrt(2 ln p) climbs to 1 only from p ~ 10 on (0.87 at p = 3)."""
    ps = (3, 10, 100, 10**4, 10**8)
    ts = [pspin_critical_temperature(p) for p in ps]
    assert all(a > c for a, c in zip(ts, ts[1:]))
    ratio = [t * mp.sqrt(2 * mp.log(p)) for t, p in zip(ts, ps)]
    assert all(a < c < 1 for a, c in zip(ratio[1:], ratio[2:])), ratio
