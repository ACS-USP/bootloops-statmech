"""Online learning of a drifting teacher: the O1 flow composed with a drift term (O3).

Engine convention (odes.py, Biehl & Riegler 1994): |B| = 1, inputs xi ~ N(0, I_N), label b = sign(B . xi)
flipped with probability `noise` (= VKC's chi / 2, see drift_rules.py), J <- J + (eta / N) F b xi,
rho = R / sqrt(Q), eps = arccos(rho) / pi against the CURRENT teacher, alpha = examples / N.  After each
example the teacher moves:

  random         B <- (B + (sqrt(2D) / N) u) / |B + (sqrt(2D) / N) u|, u ~ N(0, I_N), so that
                 B(alpha) . B(alpha') = e^{-D |alpha - alpha'|}.  This is VKC's model (Vicente,
                 Kinouchi & Caticha, Machine Learning 32, 179 (1998), Eqs. (4), (26): drift vector eta/N
                 with <eta_i eta_j> = 2D delta_ij, norm kept by Lambda; their Fig. 3 text: 'the
                 autocorrelation of the concept vector decays exponentially ... e^{-D (alpha - alpha')}')
                 and KC93's (Kinouchi & Caticha, J. Phys. A 26, 6161 (1993), Eqs. (3)-(6): B . B' =
                 1 - T/N per example, so T = D).  Both papers attribute it to Biehl & Schwarze, EPL 20,
                 733 (1992) (NOT fetched).
  deterministic  B turned in the plane of (B, J) away from J by the angle arccos(1 - D/N^2) per example
                 (VKC Eqs. (28)-(30), the 'worst case', after Biehl & Schwarze, J. Phys. A 26, 2651
                 (1993), NOT fetched).
  jumps          (integrate_drift) piecewise constant teachers: at each alpha_k the teacher is
                 replaced by an uncorrelated one; R is reset to jump_overlap sqrt(Q) (VKC Sec. V,
                 KC93 Fig. 7, the 'Wisconsin test').
  D may be a schedule D(alpha) >= 0 (VKC Fig. 7 uses D0 sin^2(2 pi nu alpha); KC93 Figs. 4, 6 switch
  the drift on and off).

With uncorrelated drift and inputs (VKC's C_{S eta} = 0) the order-parameter flow of O1 gains a term
that does not involve the modulation (VKC Eqs. (10), (27), (31); KC93 Eq. (14)):

  random:         drho/dalpha += -D rho          (dtheta/dalpha += D cot theta)
  deterministic:  drho/dalpha += -sqrt(2D (1 - rho^2))   (dtheta/dalpha += sqrt(2D))

and d ln Q / dalpha is unchanged.  DriftFlow composes this onto odes.Flow for the O1 rules (hebb,
perceptron, adatron, optimal, or a callable F(h, b, Q)) and onto the VKC rules (drift_rules.py:
annealed_hebb, step, symmetric, or a Modulation W(Delta, theta)), which are scale-free and read the
true rho.  N -> oo self-averaging is ASSUMED (VKC p. 7 sketch a Fokker-Planck argument, their ref.
[22], not fetched): physics-level, not final.

KC93 lower bound (Sec. 3): the optimal (expected-stability) rule maximises drho/dalpha at every rho
among modulated Hebbian single-example rules (O1 OPTIMALITY_CLASS), and the drift term does not
involve W, so from the same rho0 its eps(alpha) and its eps_inf bound every such rule's from below.

stationary(flow): eps_inf from dtheta/dalpha = 0 (Q eliminated for the perceptron through
d ln Q / dalpha = 0).  small_drift(): eps_inf ~ c D^(1/delta) measured at small D; delta from VKC
Eq. (34): 1/beta + 2 (random), 2/beta + 2 (deterministic), beta the driftless exponent, for the
scale-free rules; the perceptron at a fixed rate is derived separately (drift_exponent).

_PLANT: private planted-error hook (tests only): "drift" multiplies D in the drift term; "coupling"
adds a W-dependent drift term -coupling D rho <W^2> / 2 to the VKC rules' flow.
"""
import math
from collections import namedtuple
from dataclasses import dataclass

import numpy as np

from ..core import Refusal
from . import odes
from .drift_rules import VKC_RULES, Modulation, vkc_averages, vkc_modulation

KINDS = ("random", "deterministic")
BETA = {"annealed_hebb": 0.5, "step": 1.0, "symmetric": 1.0, **odes.EXPONENT}   # VKC Tables I-II; O1
_PLANT = {"drift": 1.0, "coupling": 0.0}
Stationary = namedtuple("Stationary", "eps theta Q unstable")
SmallDrift = namedtuple("SmallDrift", "exponent exponent_derived coefficient Ds eps")


def _real(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


@dataclass(frozen=True, eq=False)
class DriftFlow:
    rule: object               # O1 name / callable F(h, b, Q), VKC name, or a Modulation
    base: object               # odes.Flow for the O1 rules, else None
    D: object                  # a number >= 0 or a schedule alpha -> D(alpha) >= 0
    kind: str
    noise: float
    averages: str
    modulation: object         # W(Delta, theta) for the VKC rules and Modulations, else None

    @property
    def scale_free(self):
        return self.base is None or (not callable(self.base.rule) and odes.RULES[self.base.rule].scale == 1)

    @property
    def singular(self):        # the rule carries 1/rho: rho must stay > 0
        return self.base is None or self.base.rule == "optimal"

    def D_at(self, alpha):
        d = self.D(alpha) if callable(self.D) else self.D
        if not (_real(d) and d >= 0):
            raise Refusal("precondition:drift", f"D = {d!r} at alpha = {alpha:.4g}: need a finite D >= 0")
        return float(d)

    def drift_theta(self, alpha, theta):
        """The drift's dtheta/dalpha (module docstring)."""
        d = self.D_at(alpha) * _PLANT["drift"]
        if self.kind == "random":
            return d * math.cos(theta) / math.sin(theta)
        return math.sqrt(2 * d)

    def _theta_lnq(self, alpha, theta, lnq):
        if self.base is not None:
            dth, dl = self.base._theta_lnq(alpha, theta, lnq)
        else:
            a = vkc_averages(self.rule, theta, self.noise, route=self.averages)
            rho = math.cos(theta)
            drho = a.drive - rho * a.a2 / 2 - _PLANT["coupling"] * self.D_at(alpha) * rho * a.a2 / 2
            dth, dl = -drho / math.sin(theta), 2 * a.ax + a.a2
        return dth + self.drift_theta(alpha, theta), dl

    def theta_rate(self, alpha, theta, Q=1.0):
        return self._theta_lnq(alpha, theta, math.log(Q))[0]

    def rho_rate(self, alpha, theta, Q=1.0):
        """drho/dalpha at (theta, Q)."""
        return -math.sin(theta) * self.theta_rate(alpha, theta, Q)


def drift_flow(rule, D=0.0, kind="random", eta=1.0, noise=0.0, noise_estimate=None, averages="closed"):
    """The flow of `rule` learning a teacher drifting at rate D (module docstring).  The VKC rules
    and Modulations run at unit learning rate (they are the optima of their class there)."""
    if not callable(D) and not (_real(D) and D >= 0):
        raise Refusal("precondition:drift", f"D={D!r}: need a drift rate D >= 0 (or a schedule D(alpha))")
    if kind not in KINDS:
        raise Refusal("precondition:drift-kind", f"drift={kind!r}: {' | '.join(KINDS)} (abrupt changes: "
                      "integrate_drift(jumps=...))")
    if isinstance(rule, Modulation) or rule in VKC_RULES:
        if not (_real(noise) and 0 <= noise < 0.5):
            raise Refusal("precondition:noise", f"noise={noise!r}: a flip probability in [0, 1/2)")
        if noise_estimate is not None:
            raise Refusal("precondition:known-parameter", "noise_estimate is read by the optimal rule only "
                          "(the VKC rules use the true noise level)")
        if callable(eta) or eta != 1:
            raise Refusal("precondition:optimal-eta", f"{getattr(rule, 'name', rule)} is the optimum of its "
                          "class at unit learning rate (fold any rescaling into a Modulation)")
        if averages not in ("closed", "quadrature"):
            raise Refusal("precondition:averages", f"averages={averages!r}: closed | quadrature")
        w = rule if isinstance(rule, Modulation) else Modulation(
            lambda d, th, r=rule, n=float(noise): vkc_modulation(r, d, th, n), rule)
        return DriftFlow(rule, None, D, kind, float(noise), averages, w)
    if not callable(rule) and rule not in odes.RULES:
        raise Refusal("unsupported:rule", f"rule {rule!r}: {', '.join(odes.RULES)}, {', '.join(VKC_RULES)}, "
                      "a callable F(h, b, Q) or a drift_rules.Modulation")
    base = odes.perceptron_flow(rule, eta=eta, noise=noise, noise_estimate=noise_estimate, averages=averages)
    return DriftFlow(rule, base, D, kind, float(noise), base.averages, None)


def _start(flow, init):
    th0, l0 = odes._init(flow.base if flow.base is not None else flow, init)
    if flow.singular and float(init[0]) <= 0:
        raise Refusal("precondition:initial-overlap", f"{getattr(flow.rule, 'name', flow.rule)} carries "
                      "1/rho: start at R0 > 0")
    return th0, l0


def integrate_drift(flow, alpha_max, init=None, alphas=None, jumps=(), jump_overlap=0.0, rtol=1e-9, atol=1e-11):
    """odes.Trajectory of a DriftFlow from init = (R0, Q0); `jumps`: alphas at which the teacher is
    replaced by an uncorrelated one (R <- jump_overlap sqrt(Q)); outputs at a jump are taken before it."""
    if not (_real(alpha_max) and alpha_max > 0):
        raise Refusal("precondition:alpha", f"alpha_max={alpha_max!r}: need a finite alpha > 0")
    th, lq = _start(flow, init)
    outs = sorted(float(a) for a in (alphas if alphas is not None else np.linspace(alpha_max / 50, alpha_max, 50)))
    if not outs or outs[0] <= 0 or outs[-1] > alpha_max * (1 + 1e-12):
        raise Refusal("precondition:alpha", f"alphas must lie in (0, alpha_max = {alpha_max}]")
    jumps = sorted(float(j) for j in jumps)
    if any(not (0 < j < alpha_max) for j in jumps):
        raise Refusal("precondition:alpha", f"jumps {jumps} must lie in (0, alpha_max)")
    if jumps and not (_real(jump_overlap) and abs(jump_overlap) < 1 and (jump_overlap > 0 or not flow.singular)):
        raise Refusal("precondition:initial-overlap", f"jump_overlap={jump_overlap!r}: |rho| < 1 after a jump, "
                      "and > 0 for a rule that carries 1/rho")
    hi = math.pi / 2 if flow.singular else math.pi
    def f(a, y):                    # a trial stage beyond rho = 0 is rejected (step shrinks), not evaluated
        if flow.singular and not y[0] < math.pi / 2:
            return np.array([math.nan, math.nan])
        return np.array(flow._theta_lnq(a, y[0], y[1]))
    t, y, res = 0.0, np.array([th, lq]), []
    for end in jumps + [float(alpha_max)]:
        seg = [a for a in outs if t < a <= end]
        ys = odes._dopri(f, t, y, seg + ([] if seg and seg[-1] == end else [end]), rtol, atol, hi)
        res += ys[:len(seg)]
        t, y = end, ys[-1].copy()
        y[0] = math.acos(jump_overlap)                        # the next teacher (unused after the last)
    thv = np.array([v[0] for v in res])
    with np.errstate(over="ignore"):
        Q = np.exp([v[1] for v in res])
    return odes.Trajectory(np.array(outs), np.sqrt(Q) * np.cos(thv), Q, thv / math.pi)


def hebb_drift_closed_form(alpha, D, eta=1.0, noise=0.0, init=(0.0, 1.0)):
    """(R, Q, eps) of plain Hebb learning under random drift D > 0 (derived: dR/dalpha = eta a - D R,
    dQ/dalpha = 2 eta a R + eta^2, a = (1 - 2 noise) sqrt(2/pi)): R -> eta a / D while Q grows
    linearly, so rho -> 0 (no tracking)."""
    R0, Q0 = (float(v) for v in init)
    a = (1 - 2 * noise) * math.sqrt(2 / math.pi)
    r_inf = eta * a / D
    e = math.exp(-D * alpha)
    R = r_inf + (R0 - r_inf) * e
    Q = Q0 + eta * eta * alpha + 2 * eta * a * (r_inf * alpha + (R0 - r_inf) * (1 - e) / D)
    return R, Q, math.atan2(math.sqrt(max(Q - R * R, 0.0)), R) / math.pi


def _theta_fn(flow):
    """dtheta/dalpha as a function of theta alone at the stationary Q (module docstring)."""
    if flow.scale_free:
        return lambda th: flow._theta_lnq(0.0, th, 0.0)[0], lambda th: None
    if callable(flow.base.rule):
        raise Refusal("unsupported:stationary", "stationary states are located for the named rules only")
    eta = flow.base.eta_at(0.0)

    def sqrt_q(th):
        a, _ = flow.base._avg(th, 1.0)
        if a.ax >= 0:
            raise Refusal("no-solution:no-tracking", f"{flow.base.rule}: <F b x> = {a.ax:.3g} >= 0 at theta = "
                          f"{th:.3g}, so |J| grows without bound and rho -> 0 under drift (plain Hebb: R -> "
                          "eta a / D while Q ~ eta^2 alpha; no forgetting, VKC p. 17)")
        return -eta * a.a2 / (2 * a.ax)
    return lambda th: flow._theta_lnq(0.0, th, 2 * math.log(sqrt_q(th)))[0], lambda th: sqrt_q(th) ** 2


def stationary(flow):
    """Stationary(eps, theta, Q, unstable) at a constant D > 0: the single stable zero of dtheta/dalpha
    on a scan of theta (geometric near 0), refined by bisection; Q is None for the scale-free rules."""
    if callable(flow.D) or flow.D <= 0:
        raise Refusal("precondition:drift", "stationary states need a constant drift D > 0 (without drift "
                      "eps -> 0: use odes.asymptote)")
    if flow.base is not None and flow.base.rule == "hebb":
        raise Refusal("no-solution:no-tracking", "plain Hebb: R -> eta a / D while Q ~ eta^2 alpha, so rho -> "
                      "0 and eps -> 1/2 (hebb_drift_closed_form; VKC p. 17, 'no internal forgetting mechanism')")
    fn, q_of = _theta_fn(flow)
    hi = math.pi / 2 - 1e-9 if flow.singular else math.pi - 1e-9
    grid = np.concatenate([np.geomspace(1e-7, 0.5, 30), np.linspace(0.5, hi, 25)[1:]])
    vals = [fn(float(t)) for t in grid]
    stable, unstable = [], []
    for (t0, v0), (t1, v1) in zip(zip(grid, vals), zip(grid[1:], vals[1:])):
        if (v0 > 0) != (v1 > 0):
            lo, hi_ = float(t0), float(t1)
            for _ in range(80):
                mid = (lo + hi_) / 2
                if (fn(mid) > 0) == (v0 > 0):
                    lo = mid
                else:
                    hi_ = mid
            (stable if v0 > 0 else unstable).append((lo + hi_) / 2)
    if not stable:
        raise Refusal("no-solution:no-tracking", f"no stable stationary overlap for D = {flow.D}: dtheta/dalpha "
                      "keeps one sign on the scan")
    if len(stable) > 1:
        raise Refusal("undetermined:stationary", f"several stable stationary states theta = {stable}: the one "
                      "reached depends on the initial state")
    return Stationary(stable[0] / math.pi, stable[0], q_of(stable[0]), tuple(unstable))


def drift_exponent(rule, kind):
    """1/delta: VKC Eq. (34) with the driftless exponent beta (BETA) for the scale-free rules; for the
    perceptron at a fixed rate (Q free) derived here: at the stationary |J|, sqrt(Q*) = -eta <F^2> /
    (2 <F b x>) ~ eta sqrt(2 pi) / (pi theta), the flow becomes dtheta/dalpha = -theta^2 / 4 + drift
    (noiseless, theta -> 0), so theta^3 = 4D (random) and theta^2 = 4 sqrt(2D) (deterministic):
    delta = 3 and 4, NOT VKC's 5 and 8 (p. 15, obtained from beta = 1/3 through Eq. (34), which
    assumes the Q-independent form (35)); the coefficients are then 4^(1/3)/pi and 2^(5/4)/pi, those
    of VKC's step rule (Tables I-II), independent of eta."""
    if rule == "perceptron":
        return 1 / 3 if kind == "random" else 1 / 4
    b = BETA[rule]
    return 1 / (1 / b + 2) if kind == "random" else 1 / (2 / b + 2)


def small_drift(rule, kind="random", noise=0.0, eta=1.0, Ds=(1e-12, 1e-11), slope_tol=0.01):
    """SmallDrift: the log-slope of eps_inf between Ds[0] and Ds[1], the derived exponent 1/delta, and
    the coefficient eps_inf(Ds[0]) / Ds[0]^(1/delta); a slope further than slope_tol from 1/delta is
    refused undetermined:small-drift."""
    if rule == "hebb":
        raise Refusal("no-solution:no-tracking", "plain Hebb does not track a drifting rule (eps -> 1/2)")
    if rule not in BETA:
        raise Refusal("unsupported:rule", f"rule {rule!r}: small-drift laws for {', '.join(BETA)}")
    eps = [stationary(drift_flow(rule, D=d, kind=kind, eta=eta, noise=noise)).eps for d in Ds]
    slope = math.log(eps[1] / eps[0]) / math.log(Ds[1] / Ds[0])
    z = drift_exponent(rule, kind)
    if abs(slope - z) > slope_tol:
        raise Refusal("undetermined:small-drift", f"{rule}/{kind}: log-slope {slope:.4f} is not the derived "
                      f"1/delta = {z:.4f} (+- {slope_tol}) at D = {Ds}")
    return SmallDrift(slope, z, eps[0] / Ds[0] ** z, tuple(Ds), tuple(eps))


__all__ = ["BETA", "DriftFlow", "KINDS", "SmallDrift", "Stationary", "drift_exponent", "drift_flow",
           "hebb_drift_closed_form", "integrate_drift", "small_drift", "stationary"]
