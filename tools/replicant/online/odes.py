"""Order-parameter ODEs of the online perceptron in alpha = examples / N.

Convention (Biehl & Riegler, Europhys. Lett. 28, 525 (1994), Eqs. (1)-(6), with a learning rate
eta and output noise): teacher |B| = 1, inputs xi_j i.i.d. N(0, 1), observed label b = sign(B . xi)
flipped with probability `noise`, update J <- J + (eta / N) F(h, b, Q) b xi, h = J . xi,
R = J . B, Q = J . J (BR's Q is |J|; here Q = |J|^2), eps = arccos(R / sqrt Q) / pi (the error
against the TRUE teacher).  As N -> oo (self-averaging ASSUMED, as BR do: no concentration proof is
fetched for this package), with x = h / sqrt Q, y = B . xi:

  dR/dalpha = eta <F b y>,   dQ/dalpha = 2 eta <F b h> + eta^2 <F^2>        (rates)

The named rules have F = c(Q) g(Delta), Delta = b x (fields.closed): hebb g = 1, perceptron
g = Theta(-Delta) (kappa = 0), adatron g = -Delta Theta(-Delta) with c = sqrt Q (BR Eq. (12)),
optimal g = W* with c = sqrt Q (optimal.py); c = 1 otherwise.  A callable F(h, b, Q) is averaged by
fields.quadrature.  The integrator uses the equivalent pair (theta = arccos rho, ln Q):

  drho/dalpha = eta (c / sqrt Q) <g b (y - rho x)> - rho eta^2 (c^2 / Q) <g^2> / 2,
  dtheta/dalpha = -(drho/dalpha) / sin theta,

(no cancellation as rho -> 1), by an adaptive Dormand-Prince 5(4) step written here.

Asymptotes eps ~ C alpha^-p (EXPONENT, derived from the rates with the ansatz theta = k / sqrt Q
for the perceptron, theta proportional to 1/alpha for AdaTron and the optimal rule, and the Hebb
closed form): asymptote() measures the log-slope between alpha_big / 2 and alpha_big and returns
C = alpha_big^p eps(alpha_big).

_PLANT: private planted-error hook (tests only): "eta2" multiplies the eta^2 <F^2> term.
"""
import math
from collections import namedtuple
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..core import Refusal
from . import fields
from .optimal import w_star

Rule = namedtuple("Rule", "g scale")
RULES = {
    "hebb": Rule(lambda d, th, est: np.ones_like(np.asarray(d, dtype=float)), 0),
    "perceptron": Rule(lambda d, th, est: (np.asarray(d) < 0).astype(float), 0),
    "adatron": Rule(lambda d, th, est: -np.asarray(d) * (np.asarray(d) < 0), 1),
    "optimal": Rule(lambda d, th, est: w_star(d, th, est), 1),
}
EXPONENT = {"hebb": 0.5, "perceptron": 1 / 3, "adatron": 1.0, "optimal": 1.0}
_PLANT = {"eta2": 1.0}
Trajectory = namedtuple("Trajectory", "alpha R Q eps")


def _real(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


@dataclass(frozen=True, eq=False)
class Flow:
    rule: object               # a name in RULES or a callable F(h, b, Q)
    eta: object                # a number > 0 or a schedule alpha -> eta(alpha) > 0
    noise: float
    noise_est: float
    averages: str              # "closed" | "quadrature"

    def eta_at(self, alpha):
        e = self.eta(alpha) if callable(self.eta) else self.eta
        if not (_real(e) and e > 0):
            raise Refusal("precondition:eta", f"learning rate {e!r} at alpha = {alpha:.4g}: need a "
                          "finite eta > 0")
        return float(e)

    def _avg(self, theta, Q):
        """(Averages of F per unit scale, scale c(Q))."""
        if callable(self.rule):
            j = math.sqrt(Q)
            return fields.quadrature(lambda x, b: self.rule(j * x, b, Q), theta, self.noise), 1.0
        r = RULES[self.rule]
        c = math.sqrt(Q) if r.scale else 1.0
        if self.averages == "closed":
            return fields.closed(self.rule, theta, self.noise, self.noise_est), c
        g = lambda x, b: r.g(b * x, theta, self.noise_est)
        return fields.quadrature(g, theta, self.noise), c

    def rates(self, alpha, R, Q):
        """(dR/dalpha, dQ/dalpha): the documented ODE."""
        rho = max(-1.0, min(1.0, R / math.sqrt(Q)))
        a, c = self._avg(math.acos(rho), Q)
        eta = self.eta_at(alpha)
        return (eta * c * a.ay,
                2 * eta * c * math.sqrt(Q) * a.ax + _PLANT["eta2"] * eta * eta * c * c * a.a2)

    def _theta_lnq(self, alpha, theta, lnq):
        """(dtheta/dalpha, d ln Q/dalpha).  For the scale-free rules (c = sqrt Q) Q drops out, so
        an exponentially growing Q (imperfect learning) never overflows."""
        named = not callable(self.rule)
        if named and RULES[self.rule].scale:
            g1 = g2 = 1.0                               # c / sqrt Q, c^2 / Q
            a, _ = self._avg(theta, 1.0)
        else:
            if lnq > 700:
                raise Refusal("no-solution:overflow", f"Q = e^{lnq:.4g} overflows at alpha = {alpha:.6g}")
            q = math.exp(lnq)
            a, _ = self._avg(theta, q)
            g1, g2 = math.exp(-lnq / 2), math.exp(-lnq)
        eta = self.eta_at(alpha)
        diff = _PLANT["eta2"] * eta * eta * g2 * a.a2
        drho = eta * g1 * a.drive - math.cos(theta) * diff / 2
        return -drho / math.sin(theta), 2 * eta * g1 * a.ax + diff

    def theta_rate(self, alpha, theta, Q):
        """dtheta/dalpha (theta = arccos rho)."""
        return self._theta_lnq(alpha, theta, math.log(Q))[0]


def perceptron_flow(rule, eta=1.0, noise=0.0, noise_estimate=None, averages="closed"):
    """The online perceptron flow.  rule: hebb | perceptron | adatron | optimal, or a callable
    F(h, b, Q) (numpy-vectorised; averaged by quadrature).  eta: a number > 0 or a schedule
    eta(alpha).  noise: label flip probability in [0, 1/2).  noise_estimate: the flip probability
    the optimal rule assumes (default: the true one)."""
    if not callable(rule) and rule not in RULES:
        raise Refusal("unsupported:rule", f"rule {rule!r}: known rules {', '.join(RULES)}, or a "
                      "callable F(h, b, Q)")
    if not callable(eta) and not (_real(eta) and eta > 0):
        raise Refusal("precondition:eta", f"eta={eta!r}: need a finite learning rate > 0 (or a "
                      "schedule eta(alpha))")
    if not (_real(noise) and 0 <= noise < 0.5):
        raise Refusal("precondition:noise", f"noise={noise!r}: a flip probability in [0, 1/2)")
    if noise_estimate is not None and rule != "optimal":
        raise Refusal("precondition:known-parameter", "noise_estimate is read by the optimal rule "
                      "only (the other rules do not use the noise level)")
    if noise_estimate is not None and not (_real(noise_estimate) and 0 <= noise_estimate < 0.5):
        raise Refusal("precondition:noise", f"noise_estimate={noise_estimate!r}: in [0, 1/2)")
    if rule == "optimal" and (callable(eta) or eta != 1):
        raise Refusal("precondition:optimal-eta", "the optimal modulation maximises drho/dalpha at "
                      "eta = 1; with another eta (or a schedule) it is not the optimum (fold any "
                      "rescaling into a callable rule instead)")
    if averages not in ("closed", "quadrature"):
        raise Refusal("precondition:averages", f"averages={averages!r}: closed | quadrature")
    est = float(noise if noise_estimate is None else noise_estimate)
    return Flow(rule, eta, float(noise), est, "quadrature" if callable(rule) else averages)


# ---- integrator -------------------------------------------------------------------------------
_C = (0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1)
_A = ((), (1 / 5,), (3 / 40, 9 / 40), (44 / 45, -56 / 15, 32 / 9),
      (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
      (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
      (35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84))
_E = (71 / 57600, 0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40)


def _dopri(f, t0, y0, outs, rtol, atol, hi=math.pi):
    """Adaptive Dormand-Prince 5(4) from t0 to each t in `outs` (increasing): [y(t)], keeping
    0 < y[0] < hi (theta; hi = pi/2 for the optimal rule, which is singular at rho = 0)."""
    t, y = t0, np.array(y0, dtype=float)
    k0 = f(t, y)
    h = min(1e-3, outs[-1] / 100)
    res = []
    for target in outs:
        while t < target:
            h = min(h, target - t)
            ks = [k0]
            for i in range(1, 7):
                yi = y + h * sum(a * k for a, k in zip(_A[i], ks))
                ks.append(f(t + _C[i] * h, yi))
            ynew = yi
            err = h * sum(e * k for e, k in zip(_E, ks))
            sc = atol + rtol * np.maximum(np.abs(y), np.abs(ynew))
            en = float(np.max(np.abs(err) / sc))
            if en <= 1 and 0 < ynew[0] < hi:
                t, y, k0 = t + h, ynew, ks[6]
            if not (0 < ynew[0] < hi):
                en = max(en, 32.0)
            h *= min(5.0, max(0.2, 0.9 * (en or 1e-10) ** -0.2))
            if h < 1e-14 * max(1.0, t):
                if hi < math.pi and y[0] > hi - 1e-2:
                    raise Refusal("no-solution:rho-zero", f"the flow reaches rho = 0 at alpha = {t:.6g} "
                                  f"(rho = {math.cos(y[0]):.3g}): rho = 0 attracts and the optimal "
                                  "modulation, ~ 1/rho, is singular there (no learning)")
                raise Refusal("no-solution:step-size", f"the ODE step underflowed at alpha = {t:.6g}")
        res.append(y.copy())
    return res


def _init(flow, init):
    if init is None:
        raise Refusal("precondition:state-point", "integrate needs the initial state (R0, Q0) "
                      "(Biehl-Riegler Fig. 1 use (0, 1))")
    try:
        R0, Q0 = (float(v) for v in init)
    except (TypeError, ValueError):
        raise Refusal("precondition:state-point", f"init={init!r}: need (R0, Q0)") from None
    if not (math.isfinite(R0) and math.isfinite(Q0) and Q0 > 0 and abs(R0) < math.sqrt(Q0)):
        raise Refusal("precondition:initial-overlap", f"(R0, Q0) = ({R0}, {Q0}): need Q0 > 0 and "
                      "|R0| < sqrt(Q0) (rho0 < 1)")
    if flow.rule == "optimal" and R0 <= 0:
        raise Refusal("precondition:initial-overlap", "the optimal rule carries 1/rho: start at "
                      "R0 > 0 (e.g. R0 = 0.01; its rho dynamics is independent of Q)")
    return math.acos(R0 / math.sqrt(Q0)), math.log(Q0)


def integrate(flow, alpha_max, init=None, alphas=None, rtol=1e-9, atol=1e-11):
    """Trajectory(alpha, R, Q, eps) of `flow` from init = (R0, Q0) at the points `alphas`
    (default: 50 equally spaced points up to alpha_max)."""
    if not (_real(alpha_max) and alpha_max > 0):
        raise Refusal("precondition:alpha", f"alpha_max={alpha_max!r}: need a finite alpha > 0")
    th0, l0 = _init(flow, init)
    outs = sorted(float(a) for a in (alphas if alphas is not None else
                                     np.linspace(alpha_max / 50, alpha_max, 50)))
    if not outs or outs[0] <= 0 or outs[-1] > alpha_max * (1 + 1e-12):
        raise Refusal("precondition:alpha", f"alphas must lie in (0, alpha_max = {alpha_max}]")

    def f(a, y):
        return np.array(flow._theta_lnq(a, y[0], y[1]))
    ys = _dopri(f, 0.0, (th0, l0), outs, rtol, atol, math.pi / 2 if flow.rule == "optimal" else math.pi)
    th = np.array([y[0] for y in ys])
    with np.errstate(over="ignore"):
        Q = np.exp([y[1] for y in ys])                  # inf where Q itself overflows
    return Trajectory(np.array(outs), np.sqrt(Q) * np.cos(th), Q, th / math.pi)


def hebb_closed_form(alpha, eta=1.0, noise=0.0, init=(0.0, 1.0)):
    """(R, Q, eps) of Hebb learning at constant eta: R = R0 + eta a alpha, a = (1 - 2 noise)
    sqrt(2/pi), Q - R^2 = Q0 - R0^2 + eta^2 alpha (so eps = atan2(sqrt(Q - R^2), R) / pi has no
    cancellation)."""
    R0, Q0 = (float(v) for v in init)
    a = (1 - 2 * noise) * math.sqrt(2 / math.pi)
    R = R0 + eta * a * alpha
    perp = Q0 - R0 * R0 + eta * eta * alpha
    return R, perp + R * R, math.atan2(math.sqrt(perp), R) / math.pi


def asymptote(flow, init, alpha_big=1e4, slope_tol=0.01):
    """(measured exponent, constant alpha_big^p eps(alpha_big)) for a named rule at constant eta;
    the exponent is the log-slope of eps between alpha_big / 2 and alpha_big.  A slope further than
    slope_tol from the derived p is refused undetermined:asymptote (no constant claimed)."""
    if callable(flow.rule):
        raise Refusal("precondition:asymptote", "asymptotes are derived for the named rules only")
    tr = integrate(flow, alpha_big, init, alphas=(alpha_big / 2, alpha_big))
    slope = -math.log(tr.eps[1] / tr.eps[0]) / math.log(2)
    p = EXPONENT[flow.rule]
    if abs(slope - p) > slope_tol:
        raise Refusal("undetermined:asymptote", f"{flow.rule}: log-slope {slope:.4f} at alpha = "
                      f"{alpha_big:g} is not the derived exponent {p:.4f} (+- {slope_tol})")
    return slope, alpha_big ** p * float(tr.eps[1])


__all__ = ["EXPONENT", "Flow", "RULES", "Trajectory", "asymptote", "hebb_closed_form", "integrate",
           "perceptron_flow"]
