"""The VKC 1998 modulation functions, their Gaussian averages, and closed-form stationary states.

Modulations (Vicente, Kinouchi & Caticha, Machine Learning 32, 179 (1998), arXiv:cond-mat/9801297,
Sec. II.C), per unit |J| as functions of the stability Delta = b x and of the TRUE overlap
rho = cos theta, with lambda = sqrt(1 - rho^2) / rho and the flip probability `noise` = chi / 2
(VKC's chi is the flip probability in their Eq. (1) but chi / 2 in Eqs. (20)-(25); this code follows
Eqs. (20)-(25), as online/optimal.py does, so chi = 2 noise and 1 - chi = 1 - 2 noise):

  annealed_hebb  W = sqrt(2/pi) lambda^2 rho (1 - chi)                              (Eq. (22))
  step           W = lambda^2 rho (1 - chi) / (sqrt(2 pi) [chi/2 + (1 - chi) arccos(-rho sgn Delta)/pi])
                                                                                    (Eq. (23))
  symmetric      W = sqrt(2/pi) lambda (1 - chi) e^{-Delta^2 / (2 lambda^2)}          (Eq. (24))
  optimal        W = (1 - chi) lambda phi(Delta/lambda) / (chi/2 + (1 - chi) H(-Delta/lambda))
                                                                                    (Eq. (25) = O1's w_star)

Each is the conditional mean W = <b (y - rho x)>_{H|V} / rho of VKC Eq. (17) over its information
set V (VKC_CLASS): it maximises drho/dalpha among modulations of the same V that may use the true rho
and the noise level, at unit learning rate, single examples, N -> oo.  Not claimed optimal outside
that class (several examples, batch learning, or an estimated rho).

Averages (fields.Averages: <W b y>, <W b x>, <W^2>, <W b (y - rho x)>):
  closed      annealed_hebb and step are combinations of O1's closed Hebb and perceptron moments
              (fields.closed); symmetric in closed form (_symmetric_closed, derived); any other
              Modulation uses the conditional route below
  conditional_averages(g, theta, noise): for g = g(Delta), the teacher field integrated
              analytically given x (y | x ~ N(rho x, s^2), b flipped with probability noise),
              leaving a 1D Gauss-Legendre integral over x, panels on the scales 1 and s / |rho|
  quadrature  fields.quadrature (2D polar Gauss-Legendre), the independent route

Closed-form stationary states (checked against the ODE in tests/test_online_drift.py):
  annealed_hebb_stationary: with W_AH, drho/dalpha = (1 - chi)^2 s^4 / (pi rho) - drift (derived);
    random drift (- D rho): s^2 = c rho, c = sqrt(pi D) / (1 - chi), rho = (sqrt(c^2 + 4) - c) / 2;
    deterministic (- sqrt(2D) s): s^3 / rho = pi sqrt(2D) / (1 - chi)^2, solved by bisection.
  kc93_I, kc93_stationary: Kinouchi & Caticha, J. Phys. A 26, 6161 (1993) Eqs. (17)-(23) (noiseless
    optimal rule): drho/dalpha = rho (I(lambda) - T), I = <W_ES^2> / 2.  Derived here:
    I(lambda) = (lambda^3 / 2 pi) int Du e^{-(1 + lambda^2) u^2 / 2} / H(u); their Eq. (22) PRINTS
    (1 - lambda^2) (diverging for lambda >= 1): a typo, which leaves their small-T constant (Eq. (25))
    unchanged; kc93_I(printed=True) evaluates the printed form for the record (both by Gauss-Legendre
    in u, a route independent of fields.py's panel integral in Delta).  Random drift:
    I(lambda) = T = D; deterministic (VKC Eq. (31)): I(lambda) = sqrt(2D) lambda.

_PLANT: private planted-error hook (tests only): "norm" is the sqrt(2 pi) of the conditional route and
of the symmetric closed form.
"""
import math

import numpy as np
from scipy.special import log_ndtr, ndtr

from ..core import Refusal
from . import fields
from .optimal import w_star as optimal_w

VKC_RULES = ("annealed_hebb", "step", "symmetric")
VKC_CLASS = {
    "annealed_hebb": "V = {b}: W may depend on the observed label only (VKC Eqs. (18)-(22))",
    "step": "V = {b, sign x}: W may depend on the label and the student's output (VKC Eq. (23))",
    "symmetric": "V = {b, |x|}: W may depend on the label and |x| (VKC Eq. (24))",
    "optimal": "V = {b, x}: W may depend on the label and the student field (VKC Eq. (25); O1 OPTIMALITY_CLASS)",
}
_PLANT = {"norm": math.sqrt(2 * math.pi)}
_SQ2PI = math.sqrt(2 * math.pi)


class Modulation:
    """A modulation W(Delta, theta) per unit |J| (numpy-vectorised in Delta) that may read the true
    overlap rho = cos theta: the scale-free rule J <- J + (1/N) |J| W b xi."""

    def __init__(self, fn, name="modulation"):
        if not callable(fn):
            raise Refusal("unsupported:rule", "Modulation needs a callable W(Delta, theta)")
        self.fn, self.name = fn, name

    def __call__(self, delta, theta):
        return np.asarray(self.fn(np.asarray(delta, dtype=float), theta), dtype=float) * \
            np.ones_like(np.asarray(delta, dtype=float))


def _rho_s(theta):
    rho, s = math.cos(theta), math.sin(theta)
    if rho <= 0:
        raise Refusal("precondition:overlap", f"rho = {rho:.3g}: the VKC modulations carry 1/rho "
                      "(lambda = sqrt(1 - rho^2) / rho) and are defined for rho > 0")
    return rho, s


def vkc_modulation(rule, delta, theta, noise=0.0):
    """W(Delta) per unit |J| of a VKC rule (see the module docstring), noise = chi / 2."""
    rho, s = _rho_s(theta)
    d = np.asarray(delta, dtype=float)
    k = 1 - 2 * noise
    if s == 0:
        return np.zeros_like(d)
    lam = s / rho
    if rule == "annealed_hebb":
        return math.sqrt(2 / math.pi) * lam * lam * rho * k * np.ones_like(d)
    if rule == "step":
        right = noise + k * math.acos(-rho) / math.pi          # sgn Delta = +1
        wrong = noise + k * math.acos(rho) / math.pi           # sgn Delta = -1
        return lam * lam * rho * k / _SQ2PI / np.where(d > 0, right, wrong)
    if rule == "symmetric":
        return math.sqrt(2 / math.pi) * lam * k * np.exp(-d * d / (2 * lam * lam))
    raise Refusal("unsupported:rule", f"rule {rule!r}: VKC rules {', '.join(VKC_RULES)} (optimal: optimal_w)")


def _panels(scale):
    pos = {12.0, 1.0, 2.0, 4.0, 8.0} | {min(12.0, scale * 2.0 ** j) for j in range(-4, 7)}
    edges = sorted({-e for e in pos} | pos | {0.0})
    x, w = np.polynomial.legendre.leggauss(20)
    a, b = np.array(edges[:-1]), np.array(edges[1:])
    mid, half = (a + b) / 2, (b - a) / 2
    return (mid[:, None] + half[:, None] * x[None, :]).ravel(), (half[:, None] * w[None, :]).ravel()


def conditional_averages(g, theta, noise=0.0):
    """fields.Averages of a modulation g(Delta) (vectorised) by the 1D conditional route."""
    rho, s = math.cos(theta), math.sin(theta)
    if s == 0:
        return fields.Averages(0.0, 0.0, 0.0, 0.0)
    x, w = _panels(s / abs(rho) if rho else 12.0)
    norm = _PLANT["norm"]
    psi = w * np.exp(-x * x / 2) / norm
    k = 1 - 2 * noise
    out = np.zeros(4)
    for b in (1.0, -1.0):
        m = rho * x
        u = b * m / s
        phi_u = np.exp(-u * u / 2) / norm
        cdf = ndtr(u)
        gx = np.asarray(g(b * x), dtype=float) * np.ones_like(x)
        p_b = noise + k * cdf
        py_b = noise * b * m + k * (b * m * cdf + s * phi_u)
        out += [np.sum(psi * gx * py_b), np.sum(psi * gx * b * x * p_b), np.sum(psi * gx * gx * p_b),
                k * s * np.sum(psi * gx * phi_u)]
    return fields.Averages(*(float(v) for v in out))


def vkc_averages(rule, theta, noise=0.0, route="closed"):
    """fields.Averages of a VKC rule (a name in VKC_RULES, or a Modulation) at theta."""
    if route not in ("closed", "quadrature"):
        raise Refusal("precondition:averages", f"route={route!r}: closed | quadrature")
    w = rule if isinstance(rule, Modulation) else (lambda d, th: vkc_modulation(rule, d, th, noise))
    if route == "quadrature":
        return fields.quadrature(lambda x, b: w(b * x, theta), theta, noise, n_r=96, n_phi=64)
    if rule == "annealed_hebb" or rule == "step":
        rho, s = _rho_s(theta)
        if s == 0:
            return fields.Averages(0.0, 0.0, 0.0, 0.0)
        H, P = fields.closed("hebb", theta, noise), fields.closed("perceptron", theta, noise)
        if rule == "annealed_hebb":
            c = float(w(np.array([1.0]), theta)[0])
            return fields.Averages(c * H.ay, c * H.ax, c * c, c * H.drive)
        cp, cm = (float(v) for v in w(np.array([1.0, -1.0]), theta))
        dc = cm - cp
        return fields.Averages(cp * H.ay + dc * P.ay, cp * H.ax + dc * P.ax,
                               cp * cp * (1 - P.a2) + cm * cm * P.a2, cp * H.drive + dc * P.drive)
    if rule == "symmetric":
        return _symmetric_closed(theta, noise)
    return conditional_averages(lambda d: w(d, theta), theta, noise)


def _symmetric_closed(theta, noise):
    """W_SW = A e^{-c x^2 / 2}, A = sqrt(2/pi) lambda (1 - 2 noise), c = 1/lambda^2 = (rho/s)^2 (derived):
    <W^2> = A^2 / sqrt(1 + 2c); <W b (y - rho x)> = 2 (1 - 2 noise) s A / (sqrt(2 pi) sqrt(1 + 2c));
    <W b x> = (1 - 2 noise) A 2 sqrt(c) / (sqrt(2 pi) (1 + c) sqrt(1 + 2c)) (Stein's lemma on
    E[x (2 Phi(sqrt(c) x) - 1)]); <W b y> = drive + rho <W b x>."""
    rho, s = _rho_s(theta)
    if s == 0:
        return fields.Averages(0.0, 0.0, 0.0, 0.0)
    k = 1 - 2 * noise
    c = (rho / s) ** 2
    A = math.sqrt(2 / math.pi) * (s / rho) * k
    r = math.sqrt(1 + 2 * c)
    drive = 2 * k * s * A / (_PLANT["norm"] * r)
    ax = k * A * 2 * math.sqrt(c) / (_PLANT["norm"] * (1 + c) * r)
    return fields.Averages(drive + rho * ax, ax, A * A / r, drive)


# ---- closed-form stationary states ------------------------------------------------------------
def _bisect(fn, lo, hi, it=200):
    flo = fn(lo)
    for _ in range(it):
        mid = (lo + hi) / 2
        if (fn(mid) > 0) == (flo > 0):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def annealed_hebb_stationary(D, kind="random", noise=0.0):
    """eps_inf of the annealed Hebb rule (closed form; module docstring)."""
    k = 1 - 2 * noise
    if kind == "random":
        c = math.sqrt(math.pi * D) / k
        return math.acos((math.sqrt(c * c + 4) - c) / 2) / math.pi
    target = math.pi * math.sqrt(2 * D) / (k * k)
    return _bisect(lambda t: math.sin(t) ** 3 / math.cos(t) - target, 1e-12, math.pi / 2 - 1e-12) / math.pi


def kc93_I(lam, printed=False):
    """KC93's I(lambda) = <W_ES^2> / 2 of the noiseless optimal rule (derived form; printed=True:
    their Eq. (22) as printed, which diverges for lambda >= 1, returned as inf).  Gauss-Legendre in u
    (4 panels x 48 nodes on [-L, L], L where the integrand is below e^-40), 1/H(u) from log_ndtr."""
    if printed and lam >= 1:
        return math.inf
    a = 1 - lam * lam if printed else 1 + lam * lam
    L = math.sqrt(80 / min(a, 1.0))
    x, w = np.polynomial.legendre.leggauss(48)
    edges = (-L, -3.0, 0.0, 3.0, L)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        u = (lo + hi) / 2 + (hi - lo) / 2 * x
        tot += (hi - lo) / 2 * float(np.sum(w * np.exp(-u * u / 2 - a * u * u / 2 - log_ndtr(-u))))
    return lam ** 3 / (2 * math.pi) * tot / math.sqrt(2 * math.pi)


def kc93_stationary(D, kind="random"):
    """eps_inf of the noiseless optimal rule from KC93's I(lambda) (random: I = D; deterministic:
    I = sqrt(2D) lambda): Illinois regula falsi on ln lambda, bracketed around the small-lambda guess
    I ~ c lambda^3 (c = 0.36, a starting point only; the root is located on the full I)."""
    rhs = (lambda lam: D) if kind == "random" else (lambda lam: math.sqrt(2 * D) * lam)
    f = lambda t: math.log(kc93_I(math.exp(t)) / rhs(math.exp(t)))      # increasing in t
    t0 = math.log(D / 0.36) / 3 if kind == "random" else math.log(math.sqrt(2 * D) / 0.36) / 2
    a, b = t0 - 1.0, t0 + 1.0
    fa, fb = f(a), f(b)
    while fa > 0:
        a, fa = a - 2.0, f(a - 2.0)
    while fb < 0:
        b, fb = b + 2.0, f(b + 2.0)
    side = 0
    for _ in range(60):
        t = b - fb * (b - a) / (fb - fa)
        ft = f(t)
        if abs(ft) < 1e-14 or b - a < 1e-13:
            break
        if (ft > 0) == (fb > 0):
            b, fb = t, ft
            if side == -1:
                fa /= 2
            side = -1
        else:
            a, fa = t, ft
            if side == 1:
                fb /= 2
            side = 1
    return math.atan(math.exp(t)) / math.pi


__all__ = ["Modulation", "VKC_CLASS", "VKC_RULES", "annealed_hebb_stationary", "conditional_averages",
           "kc93_I", "kc93_stationary", "optimal_w", "vkc_averages", "vkc_modulation"]
