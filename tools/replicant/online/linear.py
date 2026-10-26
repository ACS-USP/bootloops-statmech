"""Online learning of a linear teacher by a linear perceptron with quadratic error (O2).

Kinouchi & Caticha, "On-line versus off-line learning in the linear perceptron: a comparative
study", Phys. Rev. E 52, 2878 (1995) -- KC95, read from the preprint IFUSP/P-1157 (May 1995;
journal version not checked).  Convention, mapped from KC Eqs. (1)-(8) with I = S . S / N = 1 into
odes.py's (Biehl-Riegler) one: teacher B with |B|^2 = M (KC's M = B . B), inputs xi with i.i.d.
N(0, 1) components, linear outputs b = B . xi (variance M) and h = J . xi, update

    J <- (1 - Omega / N) J + (eta / N) F(b, h) xi,     R = J . B,  Q = J . J,  rho = R / sqrt(Q M),
    e_g = <(b - h)^2> / 2 = (M + Q - 2 R) / 2          (KC Eq. (4); the quadratic error)

alpha = examples / N.  constraint "none" is Omega = 0 (KC's unconstrained learning); "sphere"
chooses Omega so that dQ/dalpha = 0 (KC Eq. (8): "the decay factor ... can be used to make
dQ/dalpha = 0"), i.e. the spherical perceptron Q = Q0.  As N -> oo (self-averaging ASSUMED, as KC
do), with (b, h) jointly Gaussian, <b^2> = M, <h^2> = Q, <b h> = R:

  dR/dalpha = eta <b F> - Omega R,   dQ/dalpha = 2 eta <h F> + eta^2 <F^2> - 2 Omega Q    (rates)
  drho/dalpha = [eta (Q <b F> - R <h F>) - eta^2 R <F^2> / 2] / (Q sqrt(Q M))  (Omega cancels)

Rules: "standard" F = b - h (KC Eq. (22), gradient of (b - h)^2 / 2), at a learning rate eta (a
number, a schedule eta(alpha), or "greedy": greedy_eta below); "optimal" F = sqrt(Q/M) b / rho - h =
(Q / R) b - h (KC Eq. (9) with I = 1), eta = 1, class in OPTIMALITY_CLASS; or a callable F(b, h, Q, R)
(averaged by a Gauss-Hermite rule).  For F = c b - d h the moments are bilinear:
<b F> = c M - d R, <h F> = c R - d Q, <F^2> = c^2 M - 2 c d R + d^2 Q.

Closed forms (derived here from the rates; the ODE is checked against them):
  standard, constant eta, unconstrained: D = M + Q - 2R = 2 e_g obeys dD/dalpha = (eta^2 - 2 eta) D,
      so D = D0 e^{(eta^2 - 2 eta) alpha}, R = M - (M - R0) e^{-eta alpha}, Q = D + 2R - M;
  standard, eta = 1, sphere: drho/dalpha = sqrt(m) - rho (1 + m) / 2, m = M / Q (KC Eq. (23)), so
      rho = rho_max + (rho0 - rho_max) e^{-(1 + m) alpha / 2}, rho_max = 2 sqrt(m) / (1 + m) (Eq. (24));
  optimal: drho/dalpha = (1 - rho^2) / (2 rho) (KC Eq. (17)), so rho^2 = 1 - (1 - rho0^2) e^{-alpha};
      unconstrained dQ/dalpha = Q (1 - rho^2) / rho^2 (Eq. (18)), so Q = Q0 rho^2 / rho0^2.
greedy_eta(R, Q, M) = (Q M - R^2) / (R (M + Q - 2R)) maximises drho/dalpha over eta for F = eta (b - h)
(the rate is eta A - eta^2 B).  KC's learning-rate parameter is Gamma = (1 - rho^2) / rho^2 (Eq. (13)).

The integrator is odes._dopri on (D, Q): D = |J - B|^2 > 0 has no cancellation as J -> B.
_PLANT: private planted-error hook (tests only): "eta2" multiplies eta^2 <F^2>, "cross" the cross
term of the closed <F^2>.
"""
import math
from collections import namedtuple
from dataclasses import dataclass

import numpy as np

from ..core import Refusal
from .odes import _dopri

RULES = ("standard", "optimal")
CONSTRAINTS = ("none", "sphere")
OPTIMALITY_CLASS = (
    "KC95 Sec. 3.2: among single-example online updates J <- (1 - Omega/N) J + (1/N) F xi whose "
    "modulation F may use the teacher output b, the student field h, Q, M and the TRUE overlap rho "
    "(any decay Omega, any learning rate folded into F), F_opt = sqrt(Q/M) b / rho - h (Eq. (9), I = 1) "
    "maximises drho/dalpha at every state, where it equals (1 - rho^2)/(2 rho), a function of rho "
    "alone; so rho_opt(alpha) bounds rho(alpha) of every member from the same rho0 (ODE comparison; "
    "KC p. 9, 'a theoretical upper bound for rho(alpha) for any algorithm used in on-line mode'). It "
    "reads rho and M, so it is not a practical algorithm (KC Sec. 3.5), and it is the optimum at eta "
    "= 1 only. It maximises rho, not e_g: unconstrained, e_g = (M + Q - 2R)/2 also depends on |J|, "
    "and the rule's Q = Q0 rho^2 / rho0^2 makes e_g large unless sqrt(Q0) = rho0 sqrt(M) (KC p. 9, Fig. 1; "
    "e.g. e_g = 7.6 at alpha = 3 from (R0, Q0) = (0.1, 0.5)). greedy_eta is optimal only in the subclass F = eta (b - h): greedy per example; on the "
    "sphere, where rho alone evolves, that makes it optimal for the whole curve within the subclass; "
    "unconstrained it is greedy only")
_PLANT = {"eta2": 1.0, "cross": 1.0}
LinTrajectory = namedtuple("LinTrajectory", "alpha R Q rho eg")
Asymptote = namedtuple("Asymptote", "alpha rho eg tail")
_GH = np.polynomial.hermite_e.hermegauss(12)


def _real(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def greedy_eta(R, Q, M=1.0):
    """The eta maximising drho/dalpha for F = eta (b - h) at (R, Q): (Q M - R^2) / (R (M + Q - 2R))."""
    if R <= 0:
        raise Refusal("no-solution:rho-zero", f"R = {R:.3g}: the greedy eta carries 1/R")
    return (Q * M - R * R) / (R * (M + Q - 2 * R))


def hypothesis_temperature(rho):
    """KC's Gamma = (1 - rho^2) / rho^2 = tan^2 theta (Eq. (13))."""
    rho = np.asarray(rho, dtype=float)
    return (1 - rho * rho) / (rho * rho)


@dataclass(frozen=True, eq=False)
class LinearFlow:
    rule: object               # "standard" | "optimal" | callable F(b, h, Q, R)
    eta: object                # number > 0 | schedule alpha -> eta | "greedy"
    constraint: str            # "none" | "sphere"
    M: float                   # teacher norm |B|^2
    averages: str              # "closed" | "quadrature"

    def eta_at(self, alpha, R, Q):
        if self.eta == "greedy":
            e = greedy_eta(R, Q, self.M)
        else:
            e = self.eta(alpha) if callable(self.eta) else self.eta
        if not (_real(e) and e > 0):
            raise Refusal("precondition:eta", f"learning rate {e!r} at alpha = {alpha:.4g}: need a "
                          "finite eta > 0")
        return float(e)

    def _coef(self, R, Q):
        if self.rule == "optimal":
            if R <= 0:
                raise Refusal("no-solution:rho-zero", "the optimal modulation carries 1/rho: R <= 0")
            return Q / R, 1.0
        return 1.0, 1.0

    def moments(self, R, Q, alpha=0.0):
        """(<b F>, <h F>, <F^2>) at (R, Q), without eta."""
        M = self.M
        if self.averages == "closed":
            c, d = self._coef(R, Q)
            return (c * M - d * R, c * R - d * Q,
                    c * c * M - _PLANT["cross"] * 2 * c * d * R + d * d * Q)
        x, w = _GH
        z1, z2 = np.meshgrid(x, x, indexing="ij")
        w2 = np.outer(w, w) / (2 * math.pi)
        b = math.sqrt(M) * z1
        h = (R / math.sqrt(M)) * z1 + math.sqrt(max(Q - R * R / M, 0.0)) * z2
        if callable(self.rule):
            F = np.asarray(self.rule(b, h, Q, R), dtype=float) * np.ones_like(b)
        else:
            c, d = self._coef(R, Q)
            F = c * b - d * h
        return (float(np.sum(w2 * b * F)), float(np.sum(w2 * h * F)), float(np.sum(w2 * F * F)))

    def rates(self, alpha, R, Q):
        """(dR/dalpha, dQ/dalpha): the documented ODE."""
        bf, hf, f2 = self.moments(R, Q, alpha)
        eta = self.eta_at(alpha, R, Q)
        gain = 2 * eta * hf + _PLANT["eta2"] * eta * eta * f2
        omega = gain / (2 * Q) if self.constraint == "sphere" else 0.0
        return eta * bf - omega * R, gain - 2 * omega * Q

    def rho_rate(self, alpha, R, Q):
        dR, dQ = self.rates(alpha, R, Q)
        return (dR - R * dQ / (2 * Q)) / math.sqrt(Q * self.M)


def linear_flow(rule="standard", eta=1.0, constraint="none", M=1.0, averages="closed"):
    """The online linear-perceptron flow (module docstring)."""
    if not callable(rule) and rule not in RULES:
        raise Refusal("unsupported:rule", f"rule {rule!r}: known rules {', '.join(RULES)}, or a "
                      "callable F(b, h, Q, R)")
    if not (callable(eta) or eta == "greedy" or (_real(eta) and eta > 0)):
        raise Refusal("precondition:eta", f"eta={eta!r}: need a finite learning rate > 0, a schedule "
                      "eta(alpha), or 'greedy'")
    if rule == "optimal" and (callable(eta) or eta != 1):
        raise Refusal("precondition:optimal-eta", "the optimal modulation (KC Eq. (9)) maximises "
                      "drho/dalpha at eta = 1; with another eta it is not the optimum")
    if constraint not in CONSTRAINTS:
        raise Refusal("precondition:constraint", f"constraint={constraint!r}: none | sphere")
    if not (_real(M) and M > 0):
        raise Refusal("precondition:teacher-norm", f"M={M!r}: need a finite teacher norm |B|^2 > 0")
    if averages not in ("closed", "quadrature"):
        raise Refusal("precondition:averages", f"averages={averages!r}: closed | quadrature")
    return LinearFlow(rule, eta, constraint, float(M), "quadrature" if callable(rule) else averages)


def _init(flow, init):
    if init is None:
        raise Refusal("precondition:state-point", "integrate_linear needs the initial state (R0, Q0)")
    try:
        R0, Q0 = (float(v) for v in init)
    except (TypeError, ValueError):
        raise Refusal("precondition:state-point", f"init={init!r}: need (R0, Q0)") from None
    M = flow.M
    need_q = flow.constraint == "sphere" or flow.rule != "standard" or flow.eta == "greedy"
    ok = (math.isfinite(R0) and math.isfinite(Q0) and Q0 >= 0 and R0 * R0 <= Q0 * M
          and M + Q0 - 2 * R0 > 0 and (Q0 > 0 or not need_q))
    if not ok:
        raise Refusal("precondition:initial-overlap", f"(R0, Q0) = ({R0}, {Q0}): need Q0 >= 0 (> 0 on "
                      "the sphere, for the optimal or greedy rule and for a callable F), R0^2 <= Q0 M, "
                      "and J0 != B")
    if (flow.rule == "optimal" or flow.eta == "greedy") and R0 <= 0:
        raise Refusal("precondition:initial-overlap", "the optimal rule and the greedy eta carry "
                      "1/rho: start at R0 > 0 (e.g. R0 = 1e-3)")
    return R0, Q0


def integrate_linear(flow, alpha_max, init=None, alphas=None, rtol=1e-10, atol=1e-18):
    """LinTrajectory(alpha, R, Q, rho, eg) of `flow` from init = (R0, Q0) at `alphas` (default 50
    equally spaced points up to alpha_max)."""
    if not (_real(alpha_max) and alpha_max > 0):
        raise Refusal("precondition:alpha", f"alpha_max={alpha_max!r}: need a finite alpha > 0")
    R0, Q0 = _init(flow, init)
    outs = sorted(float(a) for a in (alphas if alphas is not None else
                                     np.linspace(alpha_max / 50, alpha_max, 50)))
    if not outs or outs[0] <= 0 or outs[-1] > alpha_max * (1 + 1e-12):
        raise Refusal("precondition:alpha", f"alphas must lie in (0, alpha_max = {alpha_max}]")
    M = flow.M

    def f(a, y):
        D, Q = y
        dR, dQ = flow.rates(a, (M + Q - D) / 2, Q)
        return np.array([dQ - 2 * dR, dQ])
    ys = _dopri(f, 0.0, (M + Q0 - 2 * R0, Q0), outs, rtol, atol, hi=math.inf)
    D = np.array([y[0] for y in ys])
    Q = np.array([y[1] for y in ys])
    R = (M + Q - D) / 2
    return LinTrajectory(np.array(outs), R, Q, R / np.sqrt(Q * M), D / 2)


def one_minus_rho(D, Q, M):
    """1 - rho from D = M + Q - 2R without cancellation at Q = M."""
    return (D - (math.sqrt(M) - math.sqrt(Q)) ** 2) / (2 * math.sqrt(Q * M))


def asymptote(flow, init, alpha_big=60.0, alpha_tail=20.0):
    """Asymptote(alpha, rho, eg, tail) on the sphere: rho and e_g at alpha_big (refused
    undetermined:asymptote unless they agree with 3 alpha_big / 4 to 1e-9), and for the optimal rule
    at Q = M the tail constant e^alpha (1 - rho) at alpha_tail (KC Eq. (31))."""
    if flow.constraint != "sphere" or callable(flow.rule) or callable(flow.eta) or flow.eta != 1:
        raise Refusal("precondition:asymptote", "asymptotes are measured for the named rules at "
                      "eta = 1 on the sphere (KC Eqs. (24), (25), (31))")
    tr = integrate_linear(flow, alpha_big, init, alphas=(0.75 * alpha_big, alpha_big))
    if abs(tr.rho[1] - tr.rho[0]) > 1e-9 or abs(tr.eg[1] - tr.eg[0]) > 1e-9:
        raise Refusal("undetermined:asymptote", f"rho moved {abs(tr.rho[1] - tr.rho[0]):.3g} between "
                      f"alpha = {0.75 * alpha_big:g} and {alpha_big:g}: not converged")
    tail = None
    Q0 = float(init[1])
    if flow.rule == "optimal" and abs(Q0 - flow.M) <= 1e-12 * flow.M:
        t = integrate_linear(flow, alpha_tail, init, alphas=(alpha_tail,))
        tail = math.exp(alpha_tail) * one_minus_rho(2 * t.eg[0], t.Q[0], flow.M)
    return Asymptote(alpha_big, float(tr.rho[1]), float(tr.eg[1]), tail)


def standard_closed_form(alpha, eta, init, M=1.0):
    """(R, Q, e_g) of the standard rule at constant eta, unconstrained (module docstring)."""
    R0, Q0 = (float(v) for v in init)
    D = (M + Q0 - 2 * R0) * math.exp((eta * eta - 2 * eta) * alpha)
    R = M - (M - R0) * math.exp(-eta * alpha)
    return R, D + 2 * R - M, D / 2


def sphere_standard_closed_form(alpha, init, M=1.0):
    """rho of the standard rule at eta = 1 on the sphere Q = Q0 (KC Eqs. (23)-(24) solved)."""
    R0, Q0 = (float(v) for v in init)
    m = M / Q0
    top = 2 * math.sqrt(m) / (1 + m)
    return top + (R0 / math.sqrt(Q0 * M) - top) * math.exp(-(1 + m) * alpha / 2)


def optimal_closed_form(alpha, init, M=1.0, constraint="none"):
    """(R, Q, e_g) of the optimal rule (KC Eqs. (17)-(18) solved)."""
    R0, Q0 = (float(v) for v in init)
    r0 = R0 / math.sqrt(Q0 * M)
    rho = math.sqrt(1 - (1 - r0 * r0) * math.exp(-alpha))
    Q = Q0 * rho * rho / (r0 * r0) if constraint == "none" else Q0
    R = rho * math.sqrt(Q * M)
    return R, Q, (M + Q - 2 * R) / 2


__all__ = ["Asymptote", "CONSTRAINTS", "LinTrajectory", "LinearFlow", "OPTIMALITY_CLASS", "RULES",
           "asymptote", "greedy_eta", "hypothesis_temperature", "integrate_linear", "linear_flow",
           "one_minus_rho", "optimal_closed_form", "sphere_standard_closed_form",
           "standard_closed_form"]
