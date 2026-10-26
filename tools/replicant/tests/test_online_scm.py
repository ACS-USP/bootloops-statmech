"""O5: online gradient descent in the soft committee machine (SCM): averages, fixed points, eigenvalues.

Convention (online/scm.py): Saad & Solla's (K student, M teacher hidden units, g = erf(x/sqrt 2), eta/N,
eps = <(sigma - tau)^2>/2, alpha = examples/N).  Biehl-Schwarze 1995 (their Q is |J|; beta = 1 is M = 1)
and Biehl-Riegler-Woehler 1996 (SS's convention) are mapped there.  Banked numbers are targets only:
each is compared with a value computed by the ODE (Newton fixed points, numerical Jacobians, roots).
"""
import math
import os
import sys
from contextlib import contextmanager

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.core import Refusal
from replicant.online import scm as S
from replicant.online import scm_flow as F
from replicant.refdata import check, load
from replicant.testing import slow

BANK = load("online_scm")


@contextmanager
def planted(module, attr="_PLANT", **changes):
    hook = getattr(module, attr)
    saved = dict(hook)
    try:
        hook.update(changes)
        yield
    finally:
        hook.clear()
        hook.update(saved)
        assert getattr(module, attr) is hook and hook == saved


def _ok(key, got):
    return check(BANK[key], got, label=key).ok


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


def _covs(rng, d, n):
    out = []
    for _ in range(n):
        A = rng.normal(size=(d, d))
        out.append(A @ A.T / d + 0.05 * np.eye(d))
    return out


def test_erf_closed_forms_vs_quadrature():
    """I2, I3, I4 (SS MANNA Eqs. (5)-(6); I2 from NIPS Eq. (2)) against tensor Gauss-Hermite at random
    covariances (3 per dimension); an erf(x) normalisation (the 1 of 1 + C_aa halved) and a 1/pi prefactor
    (2/pi halved) must both fail by more than 1e-3."""
    rng = np.random.default_rng(5)
    fns = {2: S.i2, 3: S.i3, 4: S.i4}
    for d, fn in fns.items():
        for C in _covs(rng, d, 3):
            q = S.quadrature(C, n=28 if d < 4 else 22)
            assert abs(float(fn(C)) - q) < 2e-7, (d, float(fn(C)), q)
            for plant in ({"unit": 0.5}, {"pref": 0.5}):
                with planted(S, **plant):
                    assert abs(float(fn(C)) - q) > 1e-3, (d, plant)
    # a singular covariance (two identical students): the symmetric square root handles it
    C = np.array([[0.4, 0.4, 0.3], [0.4, 0.4, 0.3], [0.3, 0.3, 1.0]])
    assert abs(float(S.i3(C)) - S.quadrature(C, n=28)) < 1e-9


def test_eps_matches_the_gaussian_average_of_the_error():
    """eps_g = (1/2) sum s_u s_v I2 equals the quadrature of (sigma - tau)^2 / 2 built from I2 entries."""
    f = S.scm_flow(2, 2, 1.0)
    Q = np.array([[0.5, 0.1], [0.1, 0.3]])
    R = np.array([[0.4, 0.05], [0.02, 0.3]])
    C = f.cov(Q, R)
    s = np.array([-1.0, -1.0, 1.0, 1.0])
    e = 0.5 * sum(s[u] * s[v] * S.quadrature(C[np.ix_([u, v], [u, v])], n=40) for u in range(4) for v in range(4))
    assert abs(f.eps(Q, R) - e) < 1e-10
    assert abs(f.eps(np.eye(2), np.eye(2))) < 1e-14


def test_bs95_single_unit_eigenvalues():
    """K = M = 1 (BS's graded-response perceptron): the Jacobian of the ODE at (R, Q) = (1, 1) has BS
    Eq. (12)'s eigenvalues lambda_1 = (4 sqrt5/3)(eta/eta_c)(eta/eta_c - 1), lambda_2 = -2 sqrt5 eta/(9 eta_c)
    (eta_c = pi sqrt(5/3); typed from the paper here, never in the code); without the eta^2 term they differ."""
    etac = math.pi * math.sqrt(5 / 3)
    for eta in (0.5, 2.0, 3.5):
        lam = F.optimum_modes(S.scm_flow(1, 1, eta))
        want = sorted([4 * math.sqrt(5) / 3 * eta / etac * (eta / etac - 1), -2 * math.sqrt(5) * eta / (9 * etac)])
        assert np.allclose(sorted([lam.lam1, lam.lam2]), want, atol=1e-7), (eta, lam, want)
    with planted(S, eta2=0.0):
        lam = F.optimum_modes(S.scm_flow(1, 1, 3.5))
    assert not np.allclose(sorted([lam.lam1, lam.lam2]), want, atol=1e-3)


@slow
def test_bs95_single_unit_critical_rates():
    """eta_c and BS's eta_opt = 2 eta_c/3 (fastest asymptotic decay at fixed eta: lambda_1 = 2 lambda_2) are
    roots of the computed eigenvalues (also RefChecks of the catalog entry)."""
    r = F.critical_rates(1)
    assert _ok("bs95_eta_c_single", r["eta_max"]) and _ok("bs95_eta_c_single_formula", r["eta_max"])
    assert _ok("bs95_eta_opt_single", r["eta_crossing"]) and _ok("bs95_eta_opt_single_formula", r["eta_crossing"])
    assert abs(r["eta_fastest"] - r["eta_crossing"]) < 1e-4       # BS's rule is the fastest decay here


@slow
def test_bs95_two_students_one_teacher():
    """BS Sec. 4, beta = 1 (= SS's K = 2, M = 1): eta_c (the specialised optimum J_1 = B, J_2 = 0 loses
    stability) and eta_s (the symmetric fixed point becomes stable), both from the computed Jacobians."""
    r = F.bs95_k2_rates()
    assert _ok("bs95_k2_eta_c", r["eta_c"]), r
    assert _ok("bs95_k2_eta_s", r["eta_s"]), r
    assert r["eta_c"] < r["eta_s"]


def test_ss_small_eta_symmetric_fixed_point_and_escape_time():
    """SS (NIPS Eqs. (6)-(7)): the eta^2-truncated ODE has the symmetric fixed point Q* = C* = 1/(2K - 1),
    R* = 1/sqrt(K(2K - 1)), eps* = (K/pi)(pi/6 - K arcsin(1/(2K))) (0.0203 at K = 3), and the escape mode
    of time constant tau = (pi/2K)(2K - 1)^(1/2)(2K + 1)^(3/2) in units of 1/eta (NIPS p. 307)."""
    for K in (2, 3, 4):
        fp = F.symmetric_fixed_point(S.scm_flow(K, K, 0.7, truncated=True), branch="identical")
        assert abs(fp.Q - 1 / (2 * K - 1)) < 1e-9 and abs(fp.C - fp.Q) < 1e-9, (K, fp)
        assert abs(fp.R - 1 / math.sqrt(K * (2 * K - 1))) < 1e-9
        assert abs(fp.eps - K / math.pi * (math.pi / 6 - K * math.asin(1 / (2 * K)))) < 1e-10
        assert abs(0.7 / fp.lam_esc - math.pi / (2 * K) * math.sqrt(2 * K - 1) * (2 * K + 1) ** 1.5) < 1e-5
    fp3 = F.symmetric_fixed_point(S.scm_flow(3, 3, 1.0, truncated=True), branch="identical")
    assert _ok("ss_trunc_eps_k3", fp3.eps) and _ok("ss_trunc_eps_k3_formula", fp3.eps)
    assert _ok("ss_tau_k3", 1 / fp3.lam_esc) and _ok("ss_tau_k2", 1 / F.symmetric_fixed_point(
        S.scm_flow(2, 2, 1.0, truncated=True), branch="identical").lam_esc)


def test_ss_fig1d_plateau_height_and_the_eta2_plant():
    """SS NIPS Fig. 1d (read off, K = M = 3): the plateau height at eta = 0.7 is the permutation-symmetric fixed
    point (R_in = R, Q_ii = Q > Q_ik = C) of the full ODE; with the eta^2 term dropped (scm._PLANT) that branch
    merges into Q = C and the height falls to the truncated 0.0203, outside the 0.001 band.  (eta = 0.1, 0.3,
    0.5: the catalog's RefChecks and the slow test.)"""
    fp = F.symmetric_fixed_point(S.scm_flow(3, 3, 0.7), branch="permutation")
    assert fp.Q > fp.C and _ok("ss_fig1d_height_eta0p7", fp.eps), fp
    with planted(S, eta2=0.0):
        _refuses(lambda: F.symmetric_fixed_point(S.scm_flow(3, 3, 0.7), branch="permutation"), "no-solution:fixed-point")
        bad = F.symmetric_fixed_point(S.scm_flow(3, 3, 0.7), branch="identical").eps
    assert not _ok("ss_fig1d_height_eta0p7", bad), bad


@slow
def test_ss_fig1d_all_heights():
    for eta, key in ((0.1, "ss_fig1d_height_eta0p1"), (0.3, "ss_fig1d_height_eta0p3"),
                     (0.5, "ss_fig1d_height_eta0p5"), (0.7, "ss_fig1d_height_eta0p7")):
        fp = F.symmetric_fixed_point(S.scm_flow(3, 3, eta), branch="permutation")
        assert fp.Q > fp.C and _ok(key, fp.eps), (eta, fp)


def test_ss_fig2_asymptotic_eigenvalues():
    """SS NIPS Fig. 2 (read off, K = M = 3): the two leading eigenvalues at the optimum, lambda_1 (the mode
    in the linear term of eps) and lambda_2 (the straight line); without the eta^2 term lambda_1 stays negative
    at eta = 1.7 (no eta_max there).  The optimality class is stated."""
    for eta, tag in ((0.4, "0p4"), (1.0, "1p0"), (1.5, "1p5")):
        m = F.optimum_modes(S.scm_flow(3, 3, eta))
        assert _ok(f"ss_fig2_lambda1_eta{tag}", m.lam1) and _ok(f"ss_fig2_lambda2_eta{tag}", m.lam2), (eta, m)
    assert F.optimum_modes(S.scm_flow(3, 3, 1.7)).lam1 > 0
    with planted(S, eta2=0.0):
        assert F.optimum_modes(S.scm_flow(3, 3, 1.7)).lam1 < 0
    assert "fixed learning rate" in F.OPTIMALITY_CLASS and "asymptotic" in F.OPTIMALITY_CLASS.lower()


@slow
def test_ss_eta_max_and_the_optimal_rate():
    """eta_max (Fig. 2, read off: 1.62) as a root of lambda_1; SS's condition lambda_1(eta_opt) = 2 lambda_2(eta_opt)
    (NIPS p. 307) is computed, not banked (no number is printed in the sources read); at K = 3 the fastest
    asymptotic decay max(lambda_1, 2 lambda_2) is NOT at that crossing but at lambda_1's minimum (a finding)."""
    r = F.critical_rates(3)
    assert _ok("ss_fig2_eta_max_k3", r["eta_max"]), r
    assert 0.85 < r["eta_crossing"] < 1.0 and 1.2 < r["eta_fastest"] < 1.35, r
    assert r["rate_fastest"] < r["rate_crossing"] < 0


def test_brw96_fixed_points():
    """BRW Eqs. (20)-(21) (K = M = 2, eta = 2): the completely symmetric fixed point (R = 0.487, Q = 0.606,
    eps = 0.124) and the less symmetric one (R = 0.520, Q_11 = 1.081, Q_12 = 0.133, eps = 0.0652); Fig. 1
    (eta = 1.5, read off): the less symmetric plateau; Fig. 2: the completely symmetric plateau."""
    f = S.scm_flow(2, 2, 2.0)
    a = F.symmetric_fixed_point(f, branch="identical")
    b = F.symmetric_fixed_point(f, branch="permutation")
    for key, got in (("brw96_eq20_R", a.R), ("brw96_eq20_Q", a.Q), ("brw96_eq20_eps", a.eps),
                     ("brw96_eq21_R", b.R), ("brw96_eq21_Q11", b.Q), ("brw96_eq21_Q12", b.C), ("brw96_eq21_eps", b.eps)):
        assert _ok(key, got), (key, got)
    f = S.scm_flow(2, 2, 1.5)
    a = F.symmetric_fixed_point(f, branch="identical")
    b = F.symmetric_fixed_point(f, branch="permutation")
    for key, got in (("brw96_fig1_eps", b.eps), ("brw96_fig1_R", b.R), ("brw96_fig1_Q11", b.Q),
                     ("brw96_fig1_Q12", b.C), ("brw96_fig2_height", a.eps)):
        assert _ok(key, got), (key, got)
    assert _ok("brw96_fig2_shift_per_100x", 0.5 * math.log(100) / a.lam_esc), 1 / a.lam_esc
    assert not _ok("brw96_fig2_shift_per_100x", math.log(100) / a.lam_esc)     # the naive Eq. (12) fails


def test_refusals():
    _refuses(lambda: S.scm_flow(0, 2, 1.0), "precondition:K")
    _refuses(lambda: S.scm_flow(2, 2.5, 1.0), "precondition:M")
    _refuses(lambda: S.scm_flow(2, 2, -1.0), "precondition:eta")
    _refuses(lambda: S.scm_flow(2, 2, 1.0, teacher_kind="noisy"), "unsupported:teacher")
    _refuses(lambda: F.symmetric_fixed_point(S.scm_flow(1, 1, 1.0)), "precondition:symmetric")
    _refuses(lambda: F.symmetric_fixed_point(S.scm_flow(2, 2, 1.0, teacher_kind="graded")), "precondition:symmetric")
    _refuses(lambda: F.symmetric_fixed_point(S.scm_flow(2, 2, 1.0), branch="odd"), "unsupported:branch")
    _refuses(lambda: F.optimum_modes(S.scm_flow(2, 2, 1.0, teacher_kind="graded")), "precondition:symmetric")
    _refuses(lambda: F.integrate_scm(S.scm_flow(2, 2, 1.0), 10.0, None), "precondition:state-point")
    _refuses(lambda: F.integrate_scm(S.scm_flow(2, 2, 1.0), 10.0, (np.eye(2), np.eye(2) * 2)),
             "precondition:initial-overlap")
    _refuses(lambda: F.integrate_scm(S.scm_flow(2, 2, 1.0), -1.0, F.brw_initial(1e-6)), "precondition:alpha")
    _refuses(lambda: F.brw_initial(-1.0), "precondition:initial-overlap")


@slow
def test_ss_eta_max_first_order_in_one_over_k():
    """NIPS Eq. (9): eta_max = (pi/K)(75 - 42 sqrt3)/(25 sqrt3 - 42) to first order in 1/K, i.e.
    K eta_max(K) -> 5.4414 with an O(1/K) gap: the computed gap shrinks like 1/K (K = 3, 6, 12)."""
    a = float(BANK["ss_eta_max_coefficient"].value)
    ke = {K: K * F.critical_rates(K, need=("eta_max",))["eta_max"] for K in (3, 6, 12)}
    gaps = [ke[K] - a for K in (3, 6, 12)]
    print(ke, gaps)
    assert abs(gaps[0]) > abs(gaps[1]) > abs(gaps[2])
    assert 1.5 < gaps[0] / gaps[1] < 2.6 and 1.5 < gaps[1] / gaps[2] < 2.6    # O(1/K)
    assert _ok("ss_eta_max_coefficient", 2 * ke[12] - ke[6])                  # Richardson to K -> oo
