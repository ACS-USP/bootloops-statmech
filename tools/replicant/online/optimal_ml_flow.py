"""Integration of VC97's optimal soft-committee flow (O6) and the paper's initial overlaps.

The flow of optimal_multilayer.OptimalFlow is singular where det H = 0 (A = H^-1 G diverges; VC97 Fig. 3's jump
of the norms near alpha = 0.5): it is integrated in an arclength-like time s with dalpha/ds = c,
dy/ds = c f(y), c = 1 / sqrt(1 + |f|^2 / 100), so c = 1 wherever |f| << 10 and the crossing is passed with
finite steps (scipy's RK45, outputs located as events on alpha).  A trial stage outside the overlap cone
([[Q, R], [R^T, 1]] not positive semidefinite) returns a large rate, so the step is rejected.

Initial overlaps (VC97 Figs. 1-2 captions: 'random initial conditions Q11 in [0, .5], Q22 in [0, 1E-6] and
Q12 ~ 0'): Q22 = 5e-7 (the middle of the range), Q12 = 0, R = 0 (R is not given; at N = 5000 it is O(1/sqrt N)),
and Q11 fitted to the paper's own Hessian inset at alpha = 0, an INPUT read off the vector data of the arXiv
PostScript figures: K = 2, M = 1: the top eigenvalue 1.0564 (fig1.ps inset, first point); K = M = 2: the lower
eigenvalue -0.0716 (fig2.ps inset; its top is clipped at the frame).  These two readings are not banked.
With R = 0 and M = 2 the flow keeps R_i1 = R_i2 (the teachers stay exchangeable), so it reaches the plateau
and never leaves it: the escape needs an asymmetry the paper's finite N supplies.

Reliability (M = 2): near the identical-student manifold Sigma has the eigenvalue 1 along (1, -1) and a small
one along (1, 1); there yhat changes steeply with Sigma_B (the posterior jumps between the arms of the level
curve) and the outer Gauss-Hermite of info_matrix converges slowly.  Its error then fakes a fixed point: from
VC97's start the flow stalls at e_g = 0.0192 (n = 6) or 0.0140 (n = 12) with min eig Sigma = 0.0073, instead of
reaching the plateau e_g* = 0.011611 where Sigma is singular.  Trajectory values with min eig Sigma below
SIGMA_RELIABLE are therefore reported as undetermined by the catalog (sig_min is returned for that).
"""
import math
from collections import namedtuple

import numpy as np

from ..core import Refusal
from .optimal_multilayer import eps_vc, hessian, posterior

OptTrajectory = namedtuple("OptTrajectory", "alpha Q R eps eig_lo eig_hi sig_min")
SIGMA_RELIABLE = 0.05      # M = 2: below this smallest eigenvalue of Sigma the W quadrature is not trusted
_FIG_A0 = {(2, 1): (1, 1.0564), (2, 2): (0, -0.0716)}      # (eigenvalue index, reading) at alpha = 0


def _cone(Q, R):
    C = np.block([[Q, R], [R.T, np.eye(R.shape[1])]])
    return bool(np.all(np.isfinite(C))) and np.linalg.eigvalsh(C).min() > -1e-12 and np.diag(Q).min() > -1e-12


def _start(q11, M):
    return np.diag([q11, 5e-7]), np.zeros((2, M))


def vc97_initial(K, M, Q11=None):
    """(Q0, R0, Q11) of VC97's Fig. 1 (K = 2, M = 1) or Fig. 2 (K = M = 2) run (module docstring)."""
    from scipy.optimize import brentq
    if (K, M) not in _FIG_A0:
        raise Refusal("unsupported:init", f"(K, M) = ({K}, {M}): VC97's initial overlaps are given for "
                      "(2, 1) and (2, 2)")
    if Q11 is None:
        idx, val = _FIG_A0[(K, M)]
        f = lambda q: np.linalg.eigvalsh(hessian(*_start(q, M)))[idx] - val
        Q11 = brentq(f, 1e-3, 0.5, xtol=1e-12)
    if not (isinstance(Q11, (int, float)) and 0 < Q11 < 1):
        raise Refusal("precondition:initial-overlap", f"Q11={Q11!r}: need 0 < Q11 < 1 (VC97: Q11 in [0, 0.5])")
    return _start(float(Q11), M) + (float(Q11),)


def vc97_initial_eigs(K, M, Q11=None):
    Q0, R0, _ = vc97_initial(K, M, Q11)
    return tuple(np.linalg.eigvalsh(hessian(Q0, R0)))


def integrate_optimal(flow, alpha_max, init, alphas=None, rtol=1e-8, atol=1e-11):
    """OptTrajectory of the optimal flow from init = (Q, R) at `alphas` (default 100 points to alpha_max)."""
    from scipy.integrate import solve_ivp
    if not (isinstance(alpha_max, (int, float)) and math.isfinite(alpha_max) and alpha_max > 0):
        raise Refusal("precondition:alpha", f"alpha_max={alpha_max!r}: need a finite alpha > 0")
    Q0, R0 = (np.array(v, dtype=float) for v in init)
    if Q0.shape != (flow.K, flow.K) or R0.shape != (flow.K, flow.M) or not _cone(Q0, R0):
        raise Refusal("precondition:initial-overlap", f"need Q {flow.K} x {flow.K}, R {flow.K} x {flow.M} with "
                      "[[Q, R], [R^T, 1]] positive semidefinite")
    outs = sorted(float(a) for a in (alphas if alphas is not None else np.linspace(alpha_max / 100, alpha_max, 100)))
    if not outs or outs[0] <= 0 or outs[-1] > alpha_max * (1 + 1e-12):
        raise Refusal("precondition:alpha", f"alphas must lie in (0, alpha_max = {alpha_max}]")
    n = len(flow.pack(Q0, R0))

    def rhs(s, z):
        Q, R = flow.unpack(z[1:])
        if not _cone(Q, R):
            return np.full(n + 1, 1e8)
        try:
            v = flow.vector_rate(z[1:])
        except Refusal:
            return np.full(n + 1, 1e8)
        c = 1 / math.sqrt(1 + float(v @ v) / 100.0)
        return np.concatenate([[c], c * v])

    events = [(lambda s, z, a=a: z[0] - a) for a in outs]
    for e in events:
        e.direction = 1
    end = lambda s, z: z[0] - outs[-1] * (1 + 1e-9)
    end.terminal = True
    sol = solve_ivp(rhs, (0.0, 1e3 * outs[-1] + 1e3), np.concatenate([[0.0], flow.pack(Q0, R0)]), rtol=rtol,
                    atol=atol, events=events + [end], dense_output=True)
    if any(len(t) == 0 for t in sol.t_events[:-1]):
        raise Refusal("no-solution:step-size", f"the optimal flow stalled before alpha = {outs[-1]:g} ({sol.message})")
    st = [flow.unpack(sol.sol(t[0])[1:]) for t in sol.t_events[:-1]]
    eig = np.array([np.linalg.eigvalsh(hessian(Q, R)) for Q, R in st])
    return OptTrajectory(np.array(outs), np.array([s[0] for s in st]), np.array([s[1] for s in st]),
                         np.array([eps_vc(*s) for s in st]), eig[:, 0], eig[:, -1],
                         np.array([np.linalg.eigvalsh(posterior(*s)[1])[0] for s in st]))


__all__ = ["OptTrajectory", "SIGMA_RELIABLE", "integrate_optimal", "vc97_initial", "vc97_initial_eigs"]
