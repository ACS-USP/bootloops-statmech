"""O5: plateau detection in SCM learning curves (online/plateau.py).

The plateau is the stretch a trajectory spends near an UNSTABLE fixed point: the fixed point is a Newton
root from the slowest point of the trajectory, the entry and exit follow BRW's definition (Eqs. (9)-(11):
the largest deviation of an order parameter from its fixed-point value is below / exceeds B), the height
is eps at the fixed point.  A trajectory that has not left by alpha_max is `undetermined` (no escape
alpha is extrapolated), and every escape alpha is recomputed at rtol x 1e-2 with the spread reported.
"""
import math
import os
import sys
from contextlib import contextmanager

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.core import Refusal
from replicant.online import plateau as P
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


_BRW = S.scm_flow(2, 2, 1.5)


def test_plateau_escaped_undetermined_and_none():
    """K = M = 2, eta = 1.5 from BRW Eq. (16) (X = 1e-6, N = 1000): the trajectory sits on the identical-student
    fixed point (BRW Fig. 2's plateau, eps = 0.0411), escapes inside alpha_max = 70, and the escape alpha moves
    by < 1e-4 at rtol x 1e-2.  Cut at alpha_max = 40 (inside the plateau) it is undetermined: no exit, no
    length, nothing extrapolated.  K = M = 1 (BS's single unit) converges to the optimum: no plateau."""
    rep = P.analyse(_BRW, F.brw_initial(1e-6), 70.0, points=140)
    assert rep.status == "escaped" and rep.branch == "identical", rep
    assert _ok("brw96_fig2_height", rep.eps_fp) and abs(rep.eps_mid - rep.eps_fp) < 1e-4, rep
    assert 0 < rep.alpha_entry < rep.alpha_exit < 70 and rep.length == rep.alpha_exit - rep.alpha_entry
    assert rep.exit_rtol_spread < 1e-4, rep
    assert abs(rep.tau_esc - 1 / F.symmetric_fixed_point(_BRW, "identical").lam_esc) < 1e-6
    rep = P.analyse(_BRW, F.brw_initial(1e-6), 40.0, points=100, rtol_check=False)
    assert rep.status == "undetermined" and rep.alpha_exit is None and rep.length is None, rep
    assert rep.alpha_entry < 40 and "no escape" in rep.reason
    rep = P.analyse(S.scm_flow(1, 1, 1.0), (np.array([[0.5]]), np.array([[0.0]])), 10.0, points=50)
    assert rep.status == "none", rep


def test_an_escape_seeded_by_round_off_is_undetermined():
    """An exactly permutation-symmetric start (Q = 0.5 I, R = 0; K = M = 2, eta = 1.5) leaves the permutation
    point at alpha ~ 451 only through floating-point error: the initial symmetry breaking is 0 < ROUNDOFF_FLOOR,
    so the report is undetermined with no exit; BRW's X = 1e-6 start (breaking 2e-6) is not affected."""
    rep = P.analyse(_BRW, (0.5 * np.eye(2), np.zeros((2, 2))), 800.0, rtol=1e-6, points=60, rtol_check=False)
    assert rep.status == "undetermined" and "round-off" in rep.reason and rep.alpha_exit is None, rep
    assert P.symmetry_breaking(*F.brw_initial(1e-6), "identical") > 1e-6


@slow
def test_round_off_escape_k3_and_runaway_norms():
    """The reviewer's case: K = M = 3, eta = 0.5 from (0.3 I, 0) 'escaped' at alpha = 1683 with an rtol spread
    (0.477) just under its tolerance (0.487); it is now undetermined (seeded below round-off).  K = 3 at
    eta = 1.5 from SS's start: the norms run away (max Q_ii 4.4 at alpha = 200, growing): undetermined."""
    rep = P.analyse(S.scm_flow(3, 3, 0.5), (0.3 * np.eye(3), np.zeros((3, 3))), 4000.0)
    assert rep.status == "undetermined" and "round-off" in rep.reason, rep
    rep = P.analyse(S.scm_flow(3, 3, 1.5), F.ss_initial(3, 3, 1), 200.0)
    assert rep.status == "undetermined" and rep.runaway and "without bound" in rep.reason, rep


@slow
def test_identical_students_never_escape():
    """Exactly identical students (Q_12 = Q_11, R = 0) stay on their invariant manifold: undetermined at
    alpha_max = 150, where the X = 1e-6 run has long escaped."""
    rep = P.analyse(_BRW, (np.full((2, 2), 0.5), np.zeros((2, 2))), 150.0, rtol_check=False)
    assert rep.status == "undetermined" and rep.alpha_exit is None and rep.branch == "identical", rep


@slow
def test_the_rtol_check_can_fail():
    """The rtol check can fail: on a coarse grid at rtol 1e-2 the escape alpha moves by ~0.016 under rtol x 1e-2
    (measured: even that loose tolerance does not fake this escape, which the explicit X_R asymmetry drives),
    below SPREAD_TOL tau_esc = 0.072, so it stays escaped; with SPREAD_TOL lowered to 1e-4 (7e-4 in alpha) the
    report becomes undetermined."""
    loose = dict(rtol=1e-2, points=50)
    rep = P.analyse(_BRW, F.brw_initial(1e-6), 120.0, **loose)
    assert rep.status == "escaped" and 1e-3 < rep.exit_rtol_spread < P.SPREAD_TOL * rep.tau_esc, rep
    saved = P.SPREAD_TOL
    try:
        P.SPREAD_TOL = 1e-4
        rep = P.analyse(_BRW, F.brw_initial(1e-6), 120.0, **loose)
    finally:
        P.SPREAD_TOL = saved
    assert P.SPREAD_TOL == saved == 0.01
    assert rep.status == "undetermined" and "rtol" in rep.reason and rep.alpha_exit is None, rep


@slow
def test_eta2_plant_moves_the_plateau():
    """The ODE without its eta^2 (I4) term: the plateau height moves from 0.0411 to below 0.02 and the escape
    by more than 10 (the rtol spread is < 0.05)."""
    rep = P.analyse(_BRW, F.brw_initial(1e-6), 120.0)
    with planted(S, eta2=0.0):
        bad = P.analyse(_BRW, F.brw_initial(1e-6), 120.0, rtol_check=False)
    assert bad.eps_fp < 0.02 and not _ok("brw96_fig2_height", bad.eps_fp), bad
    assert bad.alpha_exit is None or abs(bad.alpha_exit - rep.alpha_exit) > 10, (bad, rep)


def test_plateau_refusals():
    _refuses(lambda: P.analyse(_BRW, F.brw_initial(1e-6), 120.0, B=0.0), "precondition:threshold")
    _refuses(lambda: P.analyse(_BRW, F.brw_initial(1e-6), 120.0, points=10), "precondition:grid")


@slow
def test_brw96_eq17_half_log_slope_and_fig2_shift():
    """BRW Eq. (17): with Eq. (16) initial conditions the plateau end moves by -(tau_esc/2) ln X (half the
    naive Eq. (12)); X = 1e-6, 1e-8, 1e-10 at N = 1000, eta = 1.5; the shifts against Fig. 2 (read off).
    Near-structural: brw_initial builds BRW's X_R = sqrt(2X/N) in, so this checks that the R asymmetry (not X
    itself) sets the escape and that the ODE's escape mode is tau_esc; only differences of alpha_P are used
    (BRW's alpha_0 is arbitrary)."""
    reps = [P.analyse(_BRW, F.brw_initial(x), 200.0) for x in (1e-6, 1e-8, 1e-10)]
    assert all(r.status == "escaped" for r in reps), reps
    tau = reps[0].tau_esc
    ends = [r.alpha_exit for r in reps]
    slope = np.polyfit(np.log([1e-6, 1e-8, 1e-10]), ends, 1)[0] / tau
    print("ends", ends, "tau", tau, "slope / tau", slope, "spreads", [r.exit_rtol_spread for r in reps])
    assert _ok("brw96_eq17_log_slope", slope), slope
    assert _ok("brw96_fig2_shift_per_100x", ends[1] - ends[0]) and _ok("brw96_fig2_shift_per_100x", ends[2] - ends[1])
    with planted(S, eta2=0.0):
        bad = P.analyse(_BRW, F.brw_initial(1e-8), 200.0, rtol_check=False)
    assert bad.alpha_exit is None or abs(bad.alpha_exit - ends[1]) > 10


@slow
def test_ss_fig1_plateau_and_trapping_time():
    """SS NIPS Fig. 1 (K = M = 3 from SS's random initial overlaps, seed 1): the eta = 0.1 plateau height is
    Fig. 1d's (read off) and sits on the permutation-symmetric point; plateau lengths at eta = 0.05, 0.1, 0.2
    scale as eta^-1 (SS96 p. 306: 'trapping times inversely proportional to eta')."""
    f = lambda eta: S.scm_flow(3, 3, eta)
    init = F.ss_initial(3, 3, 1)
    reps = {eta: P.analyse(f(eta), init, 1000.0 / eta) for eta in (0.05, 0.1, 0.2)}
    for eta, r in reps.items():
        print(eta, r)
        assert r.status == "escaped" and r.branch == "permutation", r
    assert _ok("ss_fig1d_height_eta0p1", reps[0.1].eps_fp) and _ok("ss_fig1d_height_eta0p1", reps[0.1].eps_mid)
    etas = np.array(sorted(reps))
    slope = np.polyfit(np.log(etas), np.log([reps[e].length for e in etas]), 1)[0]
    print("trapping exponent", slope)
    assert _ok("ss_trapping_eta_exponent", slope), slope
