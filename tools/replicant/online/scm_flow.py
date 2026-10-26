"""SCM flow tools (O5): integration, initial conditions, fixed points, linearisation, critical rates.

All on the Saad-Solla ODE of online/scm.py (convention there).  Nothing here types a paper's result:
fixed points are Newton roots of the computed rates, eigenvalues come from numerical Jacobians
(central differences, h = 1e-7), and critical learning rates are roots of those eigenvalues.

Symmetric fixed points (K >= 2, isotropic teacher T = I, any M): the manifold R_in = R, Q_ii = Q,
Q_ik = C (i != k) is invariant.  Two branches are distinguished (Biehl, Riegler & Woehler 1996, Fig. 3):
  identical    Q = C (identical student vectors; BRW's "completely symmetric" point, Eq. (20));
  permutation  Q > C (BRW's "less symmetric" point, Eq. (21); the plateau of SS NIPS Fig. 1, where
               Q = Q* + Delta, Delta = |J_perp|^2).
Each is reached by integrating the flow restricted to its manifold (from (R, Q) = (0, 0.5), resp.
(R, Q, C) = (0, 0.5, 0), the BRW / SS initial norms) and polished by Newton; the two merge as eta -> 0,
where the truncated ODE gives SS's Q* = C* = 1/(2K - 1).  lam_esc is the largest real part of the
eigenvalues of the full Jacobian there (BRW Eq. (9)); tau_esc = 1 / lam_esc.

Asymptotics at the optimum (K = M, T = I; SS NIPS p. 307, BS Eq. (12)): the Jacobian of the flow restricted
to R_in = R delta_in + S (1 - delta_in), Q_ik = Q delta_ik + C (1 - delta_ik) at (1, 0, 1, 0) ((R, Q) at
(1, 1) when K = 1).  Of its two leading eigenvalues, lam1 is the mode that enters the linear term of eps
(the larger |grad eps . v|), lam2 the other; the asymptotic decay rate is max(lam1, 2 lam2).
"""
import math
from collections import namedtuple

import numpy as np

from ..core import Refusal
from .odes import _dopri

OPTIMALITY_CLASS = ("fixed learning rate eta (no schedule), plain online gradient descent on the SCM with "
                    "K = M, isotropic teacher, realisable task; 'optimal' = fastest ASYMPTOTIC exponential "
                    "decay of eps_g near the optimum (SS NIPS p. 307; BS Sec. 3 for K = M = 1), not the "
                    "fastest escape from the plateau nor the lowest eps at finite alpha")
ScmTrajectory = namedtuple("ScmTrajectory", "alpha Q R eps")
SymFP = namedtuple("SymFP", "branch R Q C eps lam_esc eigs")
Modes = namedtuple("Modes", "lam1 lam2 rate")
_H = 1e-7
_RATES = {}                     # critical_rates memo, keyed with the planted-error hook's state


def _isotropic(flow):
    return np.allclose(flow.T, np.eye(flow.M))


def _state(flow, init):
    if init is None:
        raise Refusal("precondition:state-point", "the SCM flow needs the initial overlaps (Q, R)")
    try:
        Q, R = (np.array(v, dtype=float) for v in init)
    except (TypeError, ValueError):
        raise Refusal("precondition:state-point", "init must be (Q (K x K), R (K x M))") from None
    if Q.shape != (flow.K, flow.K) or R.shape != (flow.K, flow.M) or not (np.all(np.isfinite(Q)) and np.all(np.isfinite(R))):
        raise Refusal("precondition:state-point", f"need Q {flow.K} x {flow.K} and R {flow.K} x {flow.M}, finite")
    if not np.allclose(Q, Q.T) or np.linalg.eigvalsh(flow.cov(Q, R)).min() < -1e-12 or Q[0, 0] <= 0:
        raise Refusal("precondition:initial-overlap", "the covariance [[Q, R], [R^T, T]] must be positive "
                      "semidefinite with Q_11 > 0 (overlaps of real weight vectors)")
    return Q, R


def brw_initial(X, N=1000, Qt=0.5, u=(0.2, 1.0)):
    """BRW Eq. (16) with the Fig. 5 choice (K = M = 2): Q_11 = Q_22 = Qt, Q_12 = Qt - X, R_11 = R_12 = 0,
    R_21 = u_1 X_R, R_22 = u_2 X_R with X_R = sqrt(2 X / N) (the R asymmetry a finite N imposes)."""
    if not (isinstance(X, (int, float)) and 0 < X < Qt) or not (isinstance(N, (int, float)) and N > 1):
        raise Refusal("precondition:initial-overlap", f"X={X!r}, N={N!r}: need 0 < X < Qt = {Qt} and N > 1")
    xr = math.sqrt(2 * X / N)
    return (np.array([[Qt, Qt - X], [Qt - X, Qt]]), np.array([[0.0, 0.0], [u[0] * xr, u[1] * xr]]))


def ss_initial(K, M, seed):
    """SS NIPS p. 304: Q_ii ~ U[0, 0.5], Q_ik and R_in ~ U[0, 1e-12] (i != k), from the named seed."""
    rng = np.random.default_rng(seed)
    off = np.triu(rng.uniform(0, 1e-12, (K, K)), 1)
    return np.diag(rng.uniform(0, 0.5, K)) + off + off.T, rng.uniform(0, 1e-12, (K, M))


def integrate_scm(flow, alpha_max, init, alphas=None, rtol=1e-9, atol=1e-12):
    """ScmTrajectory(alpha, Q, R, eps) of the flow from init = (Q, R) at `alphas` (default 400 points)."""
    if not (isinstance(alpha_max, (int, float)) and math.isfinite(alpha_max) and alpha_max > 0):
        raise Refusal("precondition:alpha", f"alpha_max={alpha_max!r}: need a finite alpha > 0")
    Q0, R0 = _state(flow, init)
    outs = sorted(float(a) for a in (alphas if alphas is not None else np.linspace(alpha_max / 400, alpha_max, 400)))
    if not outs or outs[0] <= 0 or outs[-1] > alpha_max * (1 + 1e-12):
        raise Refusal("precondition:alpha", f"alphas must lie in (0, alpha_max = {alpha_max}]")
    ys = _dopri(lambda a, y: flow.vector_rate(y), 0.0, flow.pack(Q0, R0), outs, rtol, atol, math.inf)
    st = [flow.unpack(y) for y in ys]
    return ScmTrajectory(np.array(outs), np.array([s[0] for s in st]), np.array([s[1] for s in st]),
                         np.array([flow.eps(*s) for s in st]))


def jacobian(fn, p, h=_H):
    p = np.asarray(p, dtype=float)
    return np.array([(fn(p + h * e) - fn(p - h * e)) / (2 * h) for e in np.eye(len(p))]).T


def newton(fn, p, tol=1e-13, it=60):
    p = np.asarray(p, dtype=float)
    for _ in range(it):
        dp = np.linalg.lstsq(jacobian(fn, p), -fn(p), rcond=None)[0]
        p = p + dp
        if not np.all(np.isfinite(p)):
            break
        if np.max(np.abs(dp)) < tol:
            return p
    res = np.max(np.abs(fn(p))) if np.all(np.isfinite(p)) else math.inf
    if res < 1e-11:
        return p
    raise Refusal("no-solution:fixed-point", f"Newton did not converge (residual {res:.3g})")


def _sym_maps(flow, branch):
    K, M = flow.K, flow.M
    if branch == "identical":
        to = lambda p: (np.full((K, K), p[1]), np.full((K, M), p[0]))
        fn = lambda p: (lambda d: np.array([d[1][0, 0], d[0][0, 0]]))(flow.rates(*to(p)))
        return to, fn, np.array([0.0, 0.5])
    if branch == "permutation":
        to = lambda p: (np.full((K, K), p[2]) + np.eye(K) * (p[1] - p[2]), np.full((K, M), p[0]))
        fn = lambda p: (lambda d: np.array([d[1][0, 0], d[0][0, 0], d[0][0, 1]]))(flow.rates(*to(p)))
        return to, fn, np.array([0.0, 0.5, 0.0])
    raise Refusal("unsupported:branch", f"branch {branch!r}: identical | permutation")


def symmetric_fixed_point(flow, branch="permutation", guess=None):
    """SymFP of the symmetric manifold's branch (module docstring); guess: a previous (R, Q[, C])."""
    if flow.K < 2 or not _isotropic(flow):
        raise Refusal("precondition:symmetric", "symmetric fixed points need K >= 2 and an isotropic teacher")
    to, fn, p0 = _sym_maps(flow, branch)
    if guess is None:
        ys = _dopri(lambda a, y: fn(y), 0.0, p0, [10.0 / flow.eta], 1e-3, 1e-8, math.inf)  # a start for Newton
        guess = ys[-1]
    p = newton(fn, guess)
    if branch == "permutation" and p[1] - p[2] < 1e-7:
        raise Refusal("no-solution:fixed-point", "the permutation branch merged with the identical one (Q = C)")
    Q, R = to(p)
    y = flow.pack(Q, R)
    eigs = np.linalg.eigvals(jacobian(flow.vector_rate, y))
    eigs = eigs[np.argsort(-eigs.real)]
    return SymFP(branch, float(p[0]), float(p[1]), float(p[-1]) if branch == "permutation" else float(p[1]),
                 flow.eps(Q, R), float(eigs[0].real), eigs)


def _opt_map(flow):
    K = flow.K
    if K == 1:
        return (lambda p: (np.array([[p[1]]]), np.array([[p[0]]]))), np.array([1.0, 1.0])
    to = lambda p: (np.full((K, K), p[3]) + np.eye(K) * (p[2] - p[3]), np.full((K, K), p[1]) + np.eye(K) * (p[0] - p[1]))
    return to, np.array([1.0, 0.0, 1.0, 0.0])


def optimum_modes(flow):
    """Modes(lam1, lam2, rate) at the optimum (module docstring)."""
    if flow.K != flow.M or not _isotropic(flow):
        raise Refusal("precondition:symmetric", "the asymptotic modes need K = M and an isotropic teacher")
    to, p = _opt_map(flow)
    if flow.K == 1:
        fn = lambda q: (lambda d: np.array([d[1][0, 0], d[0][0, 0]]))(flow.rates(*to(q)))
    else:
        fn = lambda q: (lambda d: np.array([d[1][0, 0], d[1][0, 1], d[0][0, 0], d[0][0, 1]]))(flow.rates(*to(q)))
    w, V = np.linalg.eig(jacobian(fn, p, 1e-6))
    g = np.array([(flow.eps(*to(p + 1e-6 * e)) - flow.eps(*to(p - 1e-6 * e))) / 2e-6 for e in np.eye(len(p))])
    lead = np.argsort(-w.real)[:2]
    c = np.abs(g @ V[:, lead]) / np.linalg.norm(V[:, lead], axis=0)
    l1, l2 = (float(w[lead[i]].real) for i in ((0, 1) if c[0] >= c[1] else (1, 0)))
    return Modes(l1, l2, max(l1, 2 * l2))


def _bracket(fn, lo, hi, n=16):
    x0, f0 = lo, fn(lo)
    for x in np.geomspace(lo, hi, n)[1:]:
        fx = fn(x)
        if f0 * fx < 0:
            return x0, float(x)
        x0, f0 = float(x), fx
    raise Refusal("no-solution:critical-rate", f"no sign change in eta in [{lo:g}, {hi:g}]")


def critical_rates(K, need=("eta_max", "eta_crossing", "eta_fastest"), eta_hi=12.0):
    """For K = M, T = I: eta_max (lam1 = 0, the optimum loses stability), eta_crossing (SS's / BS's condition
    lam1 = 2 lam2), eta_fastest (argmin of the asymptotic rate max(lam1, 2 lam2) on (0, eta_max))."""
    from scipy.optimize import brentq, minimize_scalar

    from . import scm
    key = (K, tuple(need), eta_hi, tuple(sorted(scm._PLANT.items())))
    if key in _RATES:
        return dict(_RATES[key])
    modes = lambda e: optimum_modes(scm.scm_flow(K, K, float(e)))
    out = _RATES[key] = {}
    out["eta_max"] = brentq(lambda e: modes(e).lam1, *_bracket(lambda e: modes(e).lam1, 0.02, eta_hi), xtol=1e-10)
    if "eta_crossing" in need or "eta_fastest" in need:
        d = lambda e: (lambda m: m.lam1 - 2 * m.lam2)(modes(e))
        out["eta_crossing"] = brentq(d, *_bracket(d, 0.02 * out["eta_max"], 0.999 * out["eta_max"]), xtol=1e-10)
        out["rate_crossing"] = modes(out["eta_crossing"]).rate
    if "eta_fastest" in need:
        r = minimize_scalar(lambda e: modes(e).rate, bounds=(0.05 * out["eta_max"], 0.999 * out["eta_max"]),
                            method="bounded", options={"xatol": 1e-6})
        out["eta_fastest"], out["rate_fastest"] = float(r.x), float(r.fun)
    return dict(out)


def bs95_k2_rates():
    """BS Sec. 4 with beta = 1 (K = 2, M = 1): eta_c, where the specialised optimum (J_1 = B, J_2 = 0)
    loses stability (forward differences at Q_22 = 1e-12, inside the overlap cone), and eta_s, where the
    permutation-symmetric fixed point becomes stable (warm-started Newton along eta)."""
    from scipy.optimize import brentq

    from .scm import scm_flow

    def opt(e):
        f = scm_flow(2, 1, float(e))
        y = f.pack(np.diag([1.0, 1e-12]), np.array([[1.0], [0.0]]))
        F0 = f.vector_rate(y)
        J = np.array([(f.vector_rate(y + 1e-9 * u) - F0) / 1e-9 for u in np.eye(len(y))]).T
        return float(np.max(np.linalg.eigvals(J).real))

    eta_c = brentq(opt, *_bracket(opt, 1.0, 3.0, 12), xtol=1e-9)
    memo = {"p": None}

    def sym(e):
        fp = symmetric_fixed_point(scm_flow(2, 1, float(e)), "permutation", memo["p"])
        memo["p"] = np.array([fp.R, fp.Q, fp.C])
        return fp.lam_esc

    xs = np.arange(1.0, 3.01, 0.25)
    vals = [sym(x) for x in xs]
    i = next((i for i in range(len(xs) - 1) if vals[i] > 0 >= vals[i + 1]), None)
    if i is None:
        raise Refusal("no-solution:critical-rate", "the symmetric point does not become stable for eta in [1, 3]")
    memo["p"] = None
    sym(xs[i])
    eta_s = brentq(sym, xs[i], xs[i + 1], xtol=1e-9)
    return {"eta_c": eta_c, "eta_s": eta_s}


__all__ = ["Modes", "OPTIMALITY_CLASS", "ScmTrajectory", "SymFP", "brw_initial", "bs95_k2_rates", "critical_rates",
           "integrate_scm", "jacobian", "newton", "optimum_modes", "ss_initial", "symmetric_fixed_point"]
