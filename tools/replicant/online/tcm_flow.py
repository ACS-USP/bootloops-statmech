"""Order-parameter ODEs of online learning in the tree committee machine (CC95 Eqs. (7)-(8)).

Convention (online/tcm.py): K odd branches of N/K inputs, |B_k| = 1, J_k <- J_k + (1/N) F_k xi_k,
alpha = examples / N, R = J_k . B_k, Q = J_k . J_k, branch-symmetric.  Per example R changes by
F_k y_k / N and Q by (2 F_k |J_k| x_k + F_k^2 |xi_k|^2 / N) / N with |xi_k|^2 = N / K, so as N -> oo
(self-averaging ASSUMED, as CC95 do: "taking the mean values ... as approximations", their p. 6)

  dR/dalpha = eta c <F y>,   dQ/dalpha = 2 eta c sqrt(Q) <F x> + eta^2 c^2 <F^2> / K        (rates)

with F per unit scale c (c = sqrt Q for the optimal rules, 1 for Hebb; eta = 1 for the optimal rules).
This is CC95 Eqs. (7)-(8) in (R, Q) (their J_k = sqrt Q, rho_k = R / sqrt Q, the 1/(2K) of their
Eq. (7)).  The averages are tcm.averages (the TCM route); the integrator is O1's Dormand-Prince
(odes._dopri, a generic integrator: nothing of O1's averages or noise model is used) in (theta, ln Q):
drho/dalpha = eta (c/sqrt Q) <F (y - rho x)> - rho eta^2 (c^2/Q) <F^2> / (2K).

Closed form (derived here): Hebb (F = sigma_B), R = R0 + eta a alpha with a = <sigma_B y_k> = P0 sqrt(2/pi),
Q - R^2 = Q0 - R0^2 + eta^2 alpha / K; from J = 0 rho = (1 + pi / (2 K P0^2 alpha))^(-1/2), at K = 3
CC95 Eq. (30); eps ~ sqrt(K / (2 pi)) alpha^(-1/2).
Asymptotes: asymptote_tcm integrates to alpha_big (log-slope over a factor 2, constant alpha^p eps);
rate_limit_constant (optimal rules) reads kappa = lim alpha eps from the rate at theta -> 0: if
dtheta/dalpha = -k theta^2 (1 + O(theta)) then theta ~ 1/(k alpha) and kappa = (lim eps/theta) / k,
both limits by a Richardson step from theta = 1e-3, 5e-4.

_PLANT: private planted-error hook (tests only): "inv_k" multiplies the 1/K of the eta^2 term (K drops it).
"""
import math
from collections import namedtuple
from dataclasses import dataclass

import numpy as np

from ..core import Refusal
from . import tcm as T
from .odes import _dopri

EXPONENT = {"hebb": 0.5, "local_optimal": 1.0, "optimal": 1.0}
_PLANT = {"inv_k": 1.0}
_OPT = ("local_optimal", "optimal")
TcmTrajectory = namedtuple("TcmTrajectory", "alpha R Q rho eps_branch eps")


def _real(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


@dataclass(frozen=True, eq=False)
class TcmFlow:
    rule: str
    K: int
    eta: float

    def rates(self, alpha, R, Q):
        """(dR/dalpha, dQ/dalpha) for one branch: the documented ODE."""
        rho = max(-1.0, min(1.0, R / math.sqrt(Q)))
        a = T.averages(self.rule, math.acos(rho), self.K)
        c = math.sqrt(Q) if self.rule in _OPT else 1.0
        return (self.eta * c * a.ay,
                2 * self.eta * c * math.sqrt(Q) * a.ax + _PLANT["inv_k"] * self.eta ** 2 * c * c * a.a2 / self.K)

    def _theta_lnq(self, alpha, theta, lnq):
        """(dtheta/dalpha, d ln Q/dalpha); the optimal rules are scale free (Q drops out)."""
        if self.rule in _OPT:
            g1 = g2 = 1.0
        else:
            if not lnq < 700:
                return float("nan"), float("nan")         # a trial stage overshooting: the step is rejected
            g1, g2 = math.exp(-lnq / 2), math.exp(-lnq)
        if not 0 < theta < math.pi:
            return float("nan"), float("nan")             # a rejected trial stage of the integrator
        a = T.averages(self.rule, theta, self.K)
        diff = _PLANT["inv_k"] * self.eta ** 2 * g2 * a.a2 / self.K
        drho = self.eta * g1 * a.drive - math.cos(theta) * diff / 2
        return -drho / math.sin(theta), 2 * self.eta * g1 * a.ax + diff

    def theta_rate(self, theta, Q=1.0):
        return self._theta_lnq(0.0, theta, math.log(Q))[0]


def tcm_flow(rule, K=3, eta=1.0):
    """The TCM flow.  rule: hebb | local_optimal | optimal (nonlocal, K = 3); K odd >= 1."""
    T._check_k(K)
    if rule not in T.RULES:
        raise Refusal("unsupported:rule", f"rule {rule!r}: {', '.join(T.RULES)}")
    if rule == "optimal" and K != 3:
        raise Refusal("unsupported:K", f"the nonlocal optimal rule is implemented for K = 3 (CC95), not "
                      f"K = {K}; local_optimal takes any odd K")
    if not (_real(eta) and eta > 0):
        raise Refusal("precondition:eta", f"eta={eta!r}: need a finite learning rate > 0")
    if rule in _OPT and eta != 1:
        raise Refusal("precondition:optimal-eta", "the optimal modulations are the maximisers at eta = 1 only")
    return TcmFlow(rule, K, float(eta))


def _init(flow, init):
    if init is None:
        raise Refusal("precondition:state-point", "integrate_tcm needs the branch state (R0, Q0)")
    try:
        R0, Q0 = (float(v) for v in init)
    except (TypeError, ValueError):
        raise Refusal("precondition:state-point", f"init={init!r}: need (R0, Q0)") from None
    if not (math.isfinite(R0) and math.isfinite(Q0) and Q0 > 0 and abs(R0) < math.sqrt(Q0)):
        raise Refusal("precondition:initial-overlap", f"(R0, Q0) = ({R0}, {Q0}): need Q0 > 0, |R0| < sqrt(Q0)")
    if flow.rule in _OPT and R0 <= 0:
        raise Refusal("precondition:initial-overlap", "the optimal rules carry 1/rho: start at R0 > 0 "
                      "(e.g. R0 = 0.01; CC95 start their curves at rho ~ 0)")
    return math.acos(R0 / math.sqrt(Q0)), math.log(Q0)


def integrate_tcm(flow, alpha_max, init=None, alphas=None, rtol=1e-9, atol=1e-11):
    """TcmTrajectory(alpha, R, Q, rho, eps_branch, eps) from init = (R0, Q0) at `alphas` (default 50
    points up to alpha_max); eps = tcm.eps_committee(eps_branch, K)."""
    if not (_real(alpha_max) and alpha_max > 0):
        raise Refusal("precondition:alpha", f"alpha_max={alpha_max!r}: need a finite alpha > 0")
    th0, l0 = _init(flow, init)
    outs = sorted(float(a) for a in (alphas if alphas is not None else np.linspace(alpha_max / 50, alpha_max, 50)))
    if not outs or outs[0] <= 0 or outs[-1] > alpha_max * (1 + 1e-12):
        raise Refusal("precondition:alpha", f"alphas must lie in (0, alpha_max = {alpha_max}]")
    f = lambda a, y: np.array(flow._theta_lnq(a, y[0], y[1]))
    ys = _dopri(f, 0.0, (th0, l0), outs, rtol, atol, math.pi / 2 if flow.rule in _OPT else math.pi)
    th = np.array([y[0] for y in ys])
    Q = np.exp([y[1] for y in ys])
    eb = th / math.pi
    return TcmTrajectory(np.array(outs), np.sqrt(Q) * np.cos(th), Q, np.cos(th), eb,
                         np.array([T.eps_committee(e, flow.K) for e in eb]))


def hebb_closed_form(alpha, K, eta=1.0, init=(0.0, 1.0)):
    """(R, Q, eps) of TCM Hebb learning (derived: see the module docstring)."""
    R0, Q0 = (float(v) for v in init)
    a = T.tie_probability(K) * math.sqrt(2 / math.pi)
    R = R0 + eta * a * alpha
    perp = Q0 - R0 * R0 + eta * eta * alpha / K
    return R, perp + R * R, T.eps_committee(math.atan2(math.sqrt(perp), R) / math.pi, K)


def asymptote_tcm(flow, init, alpha_big=1e4, slope_tol=0.01):
    """(measured exponent, alpha_big^p eps(alpha_big)); a log-slope further than slope_tol from the
    derived p is refused undetermined:asymptote."""
    tr = integrate_tcm(flow, alpha_big, init, alphas=(alpha_big / 2, alpha_big))
    slope = -math.log(tr.eps[1] / tr.eps[0]) / math.log(2)
    p = EXPONENT[flow.rule]
    if abs(slope - p) > slope_tol:
        raise Refusal("undetermined:asymptote", f"{flow.rule}, K = {flow.K}: log-slope {slope:.4f} at alpha = "
                      f"{alpha_big:g} is not the derived exponent {p:.4f} (+- {slope_tol})")
    return slope, alpha_big ** p * float(tr.eps[1])


def rate_limit_constant(flow, thetas=(1e-3, 5e-4)):
    """kappa = lim alpha eps for an optimal rule from the rate at small theta (module docstring)."""
    if flow.rule not in _OPT:
        raise Refusal("precondition:asymptote", "the rate-limit constant is for the 1/alpha rules "
                      "(local_optimal, optimal); Hebb has its closed form")
    t1, t2 = thetas
    k1, k2 = (-flow.theta_rate(t) / t ** 2 for t in (t1, t2))
    r1, r2 = (T.eps_committee(t / math.pi, flow.K) / t for t in (t1, t2))
    w = t2 / (t1 - t2)                                      # linear Richardson to theta = 0
    return (r2 + w * (r2 - r1)) / (k2 + w * (k2 - k1))


__all__ = ["EXPONENT", "TcmFlow", "TcmTrajectory", "asymptote_tcm", "hebb_closed_form", "integrate_tcm",
           "rate_limit_constant", "tcm_flow"]
