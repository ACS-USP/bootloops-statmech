"""Plateau detection in SCM learning curves (O5).

A plateau is the stretch a trajectory spends near an UNSTABLE fixed point of the flow (BRW 1996, Sec. 2;
SS 1995).  analyse() integrates the flow, then:
  1. the fixed point: Newton (online/scm_flow.newton) on the full rates from the slowest grid point of the
     trajectory (largest |dZ/dalpha| smallest) with eps above EPS_FLOOR; a stable root is an attractor, not a
     plateau, and a failed Newton run (e.g. near the optimum, on the boundary of the overlap cone) proves
     nothing: either neighbourhood is masked and the next slowest point is tried (at most 5 times);
  2. entry and exit by BRW's definition (their Eqs. (9)-(11)): with d(alpha) = max |Z(alpha) - Z_fix| over
     all order parameters Z = (Q_ik, R_in), the entry is the first alpha with d < B and the exit (BRW's
     alpha_P) the first alpha after it with d > B, linearly interpolated on the grid;
  3. the escape alpha is recomputed at rtol x 1e-2; |difference| is exit_rtol_spread.
Status:
  escaped        entered and left inside the window, with exit_rtol_spread <= SPREAD_TOL tau_esc;
  undetermined   entered but not left by alpha_max (alpha_exit and length are None: no escape time is
                 extrapolated), or the escape alpha moved by more than SPREAD_TOL tau_esc under rtol x 1e-2;
  none           no unstable fixed point within B of the trajectory.
The height is eps at the fixed point (eps_fp); eps_mid is the trajectory's eps half-way through the plateau;
eps_end is eps at alpha_max.
A cascade through several fixed points (BRW Sec. 3.3) is reported by its slowest plateau only.
The fixed point's branch is classified as identical (R_in all equal, Q_ik all equal), permutation (R_in all
equal, Q_ii = Q > Q_ik = C) or asymmetric.  lam_esc / tau_esc: the largest real eigenvalue of the Jacobian
there and its inverse (BRW's escape time).
"""
import math
from collections import namedtuple

import numpy as np

from ..core import Refusal
from . import scm_flow as F

PlateauReport = namedtuple("PlateauReport", "status reason branch eps_fp eps_mid alpha_entry alpha_exit length "
                                            "lam_esc tau_esc B rtol exit_rtol_spread Q_fp R_fp eps_end")
EPS_FLOOR = 1e-4
SPREAD_TOL = 0.01                # allowed |escape(rtol) - escape(rtol x 1e-2)|, in units of tau_esc


def _branch(Q, R, tol=1e-6):
    K = Q.shape[0]
    if np.ptp(R) > tol:
        return "asymmetric"
    off = Q[~np.eye(K, dtype=bool)]
    if K == 1 or np.ptp(np.diag(Q)) > tol or (off.size and np.ptp(off) > tol):
        return "asymmetric"
    return "identical" if abs(Q[0, 0] - off[0]) < tol else "permutation"


def _find_fp(flow, tr):
    ys = np.array([flow.pack(q, r) for q, r in zip(tr.Q, tr.R)])
    speed = np.array([np.max(np.abs(flow.vector_rate(y))) for y in ys])
    mask = tr.eps > EPS_FLOOR
    for _ in range(5):
        if not mask.any():
            return None, ys
        j = int(np.flatnonzero(mask)[np.argmin(speed[mask])])
        near = np.max(np.abs(ys - ys[j]), axis=1) < 0.05
        try:
            with np.errstate(invalid="ignore"):
                y = F.newton(flow.vector_rate, ys[j])
        except (Refusal, np.linalg.LinAlgError):
            mask &= ~near                                       # e.g. Newton leaving the overlap cone
            continue
        lam = float(np.max(np.linalg.eigvals(F.jacobian(flow.vector_rate, y)).real))
        if lam > 0:
            return (y, lam), ys
        mask &= ~near & (np.max(np.abs(ys - y), axis=1) > 0.05)     # an attractor: look elsewhere
    return None, ys


def _crossings(alpha, d, B):
    inside = np.flatnonzero(d < B)
    if not inside.size:
        return None, None
    i = int(inside[0])
    a_in = alpha[i] if i == 0 else alpha[i - 1] + (alpha[i] - alpha[i - 1]) * (d[i - 1] - B) / (d[i - 1] - d[i])
    out = np.flatnonzero(d[i:] > B)
    if not out.size:
        return float(a_in), None
    j = i + int(out[0])
    a_out = alpha[j - 1] + (alpha[j] - alpha[j - 1]) * (B - d[j - 1]) / (d[j] - d[j - 1])
    return float(a_in), float(a_out)


def analyse(flow, init, alpha_max, B=0.05, rtol=1e-8, points=200, rtol_check=True):
    """PlateauReport of the trajectory from init = (Q, R) on (0, alpha_max] (module docstring)."""
    if not (isinstance(B, (int, float)) and 0 < B < 1):
        raise Refusal("precondition:threshold", f"B={B!r}: need 0 < B < 1 (BRW's plateau-end deviation)")
    if not (isinstance(points, int) and points >= 50):
        raise Refusal("precondition:grid", f"points={points!r}: need an int >= 50")
    alphas = np.linspace(alpha_max / points, alpha_max, points) if alpha_max > 0 else None
    tr = F.integrate_scm(flow, alpha_max, init, alphas=alphas, rtol=rtol)
    fp, ys = _find_fp(flow, tr)
    if fp is None:
        return PlateauReport("none", "no unstable fixed point near the trajectory", None, None, None, None, None,
                             None, None, None, B, rtol, None, None, None, float(tr.eps[-1]))
    y, lam = fp
    Qf, Rf = flow.unpack(y)
    d = np.max(np.abs(ys - y), axis=1)
    a_in, a_out = _crossings(tr.alpha, d, B)
    common = dict(branch=_branch(Qf, Rf), eps_fp=flow.eps(Qf, Rf), lam_esc=lam, tau_esc=1 / lam, B=B, rtol=rtol,
                  Q_fp=Qf, R_fp=Rf, eps_end=float(tr.eps[-1]))
    if a_in is None:
        return PlateauReport("none", f"the trajectory never comes within B = {B} of the unstable fixed point",
                             eps_mid=None, alpha_entry=None, alpha_exit=None, length=None, exit_rtol_spread=None,
                             **common)
    a_end = a_out if a_out is not None else alpha_max
    eps_mid = float(np.interp((a_in + a_end) / 2, tr.alpha, tr.eps))
    if a_out is None:
        return PlateauReport("undetermined", f"no escape by alpha_max = {alpha_max:g} (entered at {a_in:.6g}; "
                             "nothing extrapolated)", eps_mid=eps_mid, alpha_entry=a_in, alpha_exit=None, length=None,
                             exit_rtol_spread=None, **common)
    spread, status, reason = None, "escaped", f"left the plateau at alpha = {a_out:.6g}"
    if rtol_check:
        tr2 = F.integrate_scm(flow, alpha_max, init, alphas=alphas, rtol=rtol * 1e-2)
        d2 = np.max(np.abs(np.array([flow.pack(q, r) for q, r in zip(tr2.Q, tr2.R)]) - y), axis=1)
        b_out = _crossings(tr2.alpha, d2, B)[1]
        spread = math.inf if b_out is None else abs(b_out - a_out)
        if spread > SPREAD_TOL / lam:
            status, reason = "undetermined", (f"the escape alpha moves by {spread:.3g} > {SPREAD_TOL} tau_esc under "
                                              f"rtol {rtol:g} -> {rtol * 1e-2:g}: integrator tolerance, not resolved")
    return PlateauReport(status, reason, eps_mid=eps_mid, alpha_entry=a_in,
                         alpha_exit=a_out if status == "escaped" else None,
                         length=a_out - a_in if status == "escaped" else None, exit_rtol_spread=spread, **common)


__all__ = ["EPS_FLOOR", "PlateauReport", "SPREAD_TOL", "analyse"]
