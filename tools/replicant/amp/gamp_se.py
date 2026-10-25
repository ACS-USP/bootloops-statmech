"""State evolution (SE) of Bayes-optimal GAMP for the teacher-student perceptron
(amp/gamp.py), its fixed points, the RS potential and the Ising-prior thresholds.

Convention (Barbier, Krzakala, Macris, Miolane & Zdeborova, PNAS 116, 5451 (2019),
arXiv:1708.03395, "BKMMZ", eqs. (4)-(6), (10)): rho = E w*^2 = 1, sign channel,
    q^{t+1} = 2 psi_P0'(r^t),   r^t = 2 alpha Psi_out'(q^t),   q^0 = 0 (GAMP's start).
Channel.  Psi_out(q) = 2 E_v H(-s v) ln H(-s v), s = sqrt(q/(1-q)), H(x) = int_x^oo Dz;
    r = 2 alpha Psi_out'(q) = alpha E[g_out^2]
      = (2 alpha / (1-q)) E_xi[ phi(s xi)^2 / H(-s xi) ]                       (xi form)
      = alpha / (pi sqrt(1-q)) E_t[ exp(-q t^2/2) / H(-sqrt(q) t) ]          (t form)
(xi = sqrt(1-q) t; the t form has no sharp scale as q -> 1).  The t form is iterated
(120-node Gauss-Hermite, float64, H through scipy.special.log_ndtr; the prior maps by a
uniform trapezoid rule, h = 0.01 on |Z| <= 11); the fixed point is audited with the xi
form and the prior map by adaptive QUADPACK quadrature (scipy.integrate.quad: an
independent rule), |F(q) - q| <= 1e-9, else the result says not converged.
Priors.  2 psi_P0'(r) = E[x_hat X0], x_hat the posterior mean of X0 given X0 + Z/sqrt(r):
    gaussian N(0, 1): psi = r/2 - ln(1 + r)/2,          q = r/(1 + r);
    ising +-1:        psi = E ln cosh(r + sqrt(r) Z) - r/2,   q = E tanh(r + sqrt(r) Z).
q = 1 is an exact (absorbing) fixed point for ising (r = oo, f_RS = -ln 2: one
consistent weight vector out of 2^N); the gaussian prior has none at q = 1.
RS potential f_RS(q, r) = psi_P0(r) + alpha Psi_out(q) - r q/2 (BKMMZ eq. (4)); its
critical points are the SE fixed points, and the Bayes-optimal overlap q* is the global
maximiser (BKMMZ Thm 1-2, for almost every alpha).
Errors.  mse = 1 - q; eps_Bayes = arccos(sqrt q)/pi (plug-in sign(x_hat . xi) predictor,
equal to the Bayes-optimal one for the sign channel, BKMMZ Sec. 2.3); eps_Gibbs =
arccos(q)/pi (a posterior sample has overlap q with the teacher: Nishimori).
Gaussian prior vs the spherical teacher of the replica route (catalog teacher_student):
the conjugate relation q = r/(1 + r) is the spherical stationarity r = q/(1 - q), so the
SE fixed-point
equation reads q/sqrt(1-q) = (alpha/pi) E_t[exp(-q t^2/2)/H(-sqrt(q) t)], Gyorgyi's
Gibbs equation with R = q: the agreement of the two codes is an implementation check of
an identity, not new physics.

Status.  Deterministic N -> oo predictions.  GAMP's SE for GLMs is stated by BKMMZ as
Claim 1, "because some steps of the proof are missing" in Rangan (ISIT 2011,
arXiv:1010.5141, whose abstract "argues" it); Javanmard & Montanari (Inf. Inference 2,
115 (2013), arXiv:1211.5164, abstract read) state SE for general AMP on Gaussian
matrices covering GAMP, which BKMMZ do not count as a proof.  So these values are NOT
theorem-backed here;
the RS formula for the Bayes-optimal errors (BKMMZ Thm 1-2) is.  Bayes-optimal only on
the Nishimori line (teacher drawn from the same prior, matched noiseless channel).
"""
import math
from dataclasses import dataclass, field

import mpmath as mp
import numpy as np
from scipy.integrate import quad
from scipy.special import log_ndtr, ndtr

from ..core import Refusal

PRIORS = ("gaussian", "ising")
_Z, _W = np.polynomial.hermite_e.hermegauss(120)
_W = _W / _W.sum()
# prior maps: tanh / ln cosh of r + sqrt(r) Z have poles at distance pi/(2 sqrt r) from the
# real Z axis, which slows Gauss-Hermite at large r; a uniform trapezoid rule converges
# like exp(-2 pi (pi/(2 sqrt r))/h) instead (h = 0.01: < 1e-13 for r <= 1e4)
_ZT = np.linspace(-11, 11, 2201)
_WT = np.exp(-_ZT**2 / 2)
_WT = _WT / _WT.sum()
_ONE = 1 - 1e-13          # above this an ising overlap is the absorbing q = 1
_MAX_ITERS = 20000
_LN2 = math.log(2)


@dataclass
class SEResult:
    overlap: float                  # q: the SE limit from q^0 = 0 (GAMP's start)
    mse: float                      # 1 - q
    r: float
    trajectory: list                # q^0 = 0, q^1, ...
    converged: bool
    eps_bayes: float
    eps_gibbs: float
    free_entropy: float             # f_RS at (q, r)
    fixed_points: list = field(default_factory=list)   # [(q, f_RS)] of every fixed point
    optimal_overlap: float = None   # the fixed point maximising f_RS (Bayes-optimal q*)
    gamp_is_optimal: bool = None
    alpha: float = 0.0
    prior: str = "gaussian"
    label: str = "deterministic: N -> oo state-evolution prediction (quadrature)"
    note: str = ""


@dataclass(frozen=True)
class Thresholds:
    alpha_it: float                 # f_RS(GAMP fixed point) = -ln 2
    alpha_amp: float                # the low fixed point disappears (spinodal)
    note: str = ""


def _num(x, name):
    try:
        v = float(mp.mpf(x))
    except (TypeError, ValueError):
        raise Refusal(f"precondition:numeric-{name}", f"{name}={x!r} is not a number")
    if not math.isfinite(v):
        raise Refusal(f"precondition:finite-{name}", f"{name}={x!r} is not finite")
    return v


def _alpha(alpha):
    a = _num(alpha, "alpha")
    if a < 0:
        raise Refusal("precondition:nonnegative-alpha", f"alpha={alpha} < 0 (alpha = M/N)")
    return a


def _prior(prior):
    if prior not in PRIORS:
        raise Refusal("precondition:known-prior", f"prior={prior!r}; known: {', '.join(PRIORS)}")
    return prior


def _rhat(q, alpha):
    """r = 2 alpha Psi_out'(q), the t form above (a hook the tests plant)."""
    if q >= 1:
        return math.inf
    g = np.exp(-q * _Z**2 / 2 - log_ndtr(math.sqrt(q) * _Z))
    return alpha / (math.pi * math.sqrt(1 - q)) * float(_W @ g)


def _prior_map(prior, r):
    """q = 2 psi_P0'(r)."""
    if r == math.inf:
        return 1.0
    if prior == "gaussian":
        return r / (1 + r)
    return float(_WT @ np.tanh(r + math.sqrt(r) * _ZT))


def _F(prior, q, alpha):
    return _prior_map(prior, _rhat(q, alpha))


def _rhat_quadpack(q, alpha):
    """The xi form of r by adaptive QUADPACK quadrature (the audit rule)."""
    s = math.sqrt(q / (1 - q))
    f = lambda x: math.exp(-(s * x)**2 - x * x / 2 - float(log_ndtr(s * x))) / (
        2 * math.pi * math.sqrt(2 * math.pi))
    return 2 * alpha / (1 - q) * sum(quad(f, lo, hi, epsabs=1e-15, epsrel=1e-13, limit=200)[0]
                                     for lo, hi in ((-np.inf, 0), (0, np.inf)))


def _prior_map_quadpack(prior, r):
    if prior == "gaussian":
        return r / (1 + r)
    f = lambda z: math.tanh(r + math.sqrt(r) * z) * math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    c = -math.sqrt(r)
    return sum(quad(f, lo, hi, epsabs=1e-15, epsrel=1e-13, limit=200)[0]
               for lo, hi in ((-np.inf, c), (c, np.inf)))


def _Psi_out(q):
    """2 E_v H(-s v) ln H(-s v), trapezoid h = 0.01: in v on |v| <= 11 for s < 1, else in
    t = s v on |t| <= 12 (where H(-t) ln H(-t) is negligible beyond), weight N(0, s^2)."""
    if q >= 1:
        return 0.0
    s = math.sqrt(q / (1 - q))
    if s < 1:
        u, w = s * _ZT, _WT
    else:
        u = np.linspace(-12, 12, 2401)
        w = np.exp(-u**2 / (2 * s * s))
        w = w / (s * math.sqrt(2 * math.pi)) * (u[1] - u[0])
    return 2 * float(w @ (ndtr(u) * log_ndtr(u)))


def _psi_prior(prior, r):
    if prior == "gaussian":
        return r / 2 - math.log1p(r) / 2
    x = r + math.sqrt(r) * _ZT
    logcosh = np.abs(x) + np.log1p(np.exp(-2 * np.abs(x))) - _LN2
    return float(_WT @ logcosh) - r / 2


def free_entropy(q, alpha, prior="gaussian"):
    """f_RS at the SE pair (q, r(q)); -ln 2 at the ising q = 1 point."""
    _prior(prior)
    a = _alpha(alpha)
    if q >= _ONE:
        if prior == "ising":
            return -_LN2
        raise Refusal("precondition:overlap-range", "q = 1 is not a gaussian-prior fixed point")
    r = _rhat(q, a)
    return _psi_prior(prior, r) + a * _Psi_out(q) - r * q / 2


def _fixed_points(prior, a, q_gamp):
    """Grid scan of F(q) - q on [0, 1) (240 points), refined by bisection, plus the
    trajectory limit and (ising) q = 1."""
    grid = np.linspace(0, 1 - 1e-6, 241)
    g = [_F(prior, q, a) - q for q in grid]
    roots = [q_gamp]
    for q1, q2, g1, g2 in zip(grid, grid[1:], g, g[1:]):
        if g1 * g2 < 0:
            lo, hi, glo = float(q1), float(q2), g1
            for _ in range(80):
                mid = (lo + hi) / 2
                gm = _F(prior, mid, a) - mid
                if (gm > 0) == (glo > 0):
                    lo, glo = mid, gm
                else:
                    hi = mid
            roots.append((lo + hi) / 2)
    if prior == "ising":
        roots.append(1.0)
    out = []
    for r in sorted(roots):
        if not out or abs(r - out[-1]) > 1e-7:
            out.append(r)
        elif r == q_gamp:
            out[-1] = r
    return [(q, free_entropy(q, a, prior)) for q in out]


def _errors(q):
    if q >= _ONE:
        return 0.0, 0.0
    return math.acos(math.sqrt(q)) / math.pi, math.acos(q) / math.pi


def _iterate(prior, a, iters, tol):
    tr = [0.0]
    for _ in range(iters or _MAX_ITERS):
        q = _F(prior, tr[-1], a)
        tr.append(1.0 if q >= _ONE else q)
        if iters is None and (abs(tr[-1] - tr[-2]) < tol or tr[-1] == 1.0):
            break
    return tr


def perceptron_state_evolution(alpha, prior="gaussian", teacher=True, *, iters=None,
                               tol=1e-14) -> SEResult:
    """SE from q^0 = 0.  iters=None runs to |q^{t+1} - q^t| < tol (at most 20000 steps);
    an integer runs exactly that many steps.  Every fixed point is listed with f_RS;
    `optimal_overlap` is the maximiser (Bayes-optimal), `gamp_is_optimal` whether GAMP's
    fixed point is it."""
    prior = _prior(prior)
    if teacher is not True:
        raise Refusal("not-implemented:mismatched-teacher-student",
                      "only the Bayes-optimal teacher-student setting (teacher drawn from the "
                      "prior, matched sign channel: the Nishimori line) is implemented")
    a = _alpha(alpha)
    if iters is not None and (not isinstance(iters, int) or iters < 1):
        raise Refusal("precondition:positive-iters", f"iters={iters!r} is not a positive int")
    tr = _iterate(prior, a, iters, tol)
    q = tr[-1]
    converged = q == 1.0 or abs(tr[-1] - tr[-2]) < 1e-12
    note = []
    if converged and q < 1:
        audit = abs(_prior_map_quadpack(prior, _rhat_quadpack(q, a)) - q)
        if audit > 1e-9:
            converged = False
            note.append(f"xi-form audit residual {audit:.2g} > 1e-9: reported not converged")
        else:
            note.append(f"fixed point audited by the xi form (adaptive QUADPACK): residual "
                        f"{audit:.2g}")
    elif not converged:
        note.append(f"trajectory not converged after {len(tr) - 1} steps")
    fps = _fixed_points(prior, a, q) if converged and iters is None else []
    best = max(fps, key=lambda p: p[1])[0] if fps else None
    eb, eg = _errors(q)
    if best is not None and best != q:
        note.append(f"GAMP's fixed point q = {q:.6g} is NOT Bayes-optimal: f_RS is larger at "
                    f"q* = {best:.6g} (hard phase)")
    note.append("Bayes-optimal only on the Nishimori line (teacher from the same prior, "
                "matched noiseless sign channel); GAMP SE for GLMs is BKMMZ Claim 1 (from "
                "Rangan 2011), not a theorem applied here")
    r = _rhat(min(q, 1.0), a) if q < 1 else math.inf
    return SEResult(q, 1 - q, r, tr, converged, eb, eg,
                    free_entropy(q, a, prior) if converged else math.nan, fps, best,
                    None if best is None else best == q, a, prior, note="; ".join(note))


def _min_gap(a, lo=0.5, hi=0.99):
    """min over q in [lo, hi] of F(q) - q (ising): grid, then golden section."""
    qs = np.linspace(lo, hi, 50)
    g = [_F("ising", q, a) - q for q in qs]
    k = min(max(int(np.argmin(g)), 1), len(qs) - 2)
    x0, x1 = float(qs[k - 1]), float(qs[k + 1])
    phi = (math.sqrt(5) - 1) / 2
    for _ in range(40):
        c, d = x1 - phi * (x1 - x0), x0 + phi * (x1 - x0)
        if _F("ising", c, a) - c < _F("ising", d, a) - d:
            x1 = d
        else:
            x0 = c
    x = (x0 + x1) / 2
    return _F("ising", x, a) - x


def ising_thresholds(*, bracket_it=(1.15, 1.35), bracket_amp=(1.35, 1.7)) -> Thresholds:
    """alpha_IT: root of f_RS(GAMP fixed point) + ln 2 by bisection on `bracket_it`;
    alpha_AMP: bisection on the sign of the minimum of F(q) - q over q in [0.5, 0.99]
    (the low fixed point exists iff it is <= 0).  A bracket without a sign change is
    refused (no-solution)."""
    def gap(a):
        tr = _iterate("ising", a, None, 1e-14)
        if tr[-1] == 1.0 or abs(tr[-1] - tr[-2]) > 1e-12:
            raise Refusal("no-solution", f"alpha_IT bracket: no low fixed point at {a}")
        return free_entropy(tr[-1], a, "ising") + _LN2
    lo, hi = bracket_it
    glo, ghi = gap(lo), gap(hi)
    if glo * ghi > 0:
        raise Refusal("no-solution", f"f_RS + ln 2 does not change sign on {bracket_it}")
    while hi - lo > 1e-7:
        mid = (lo + hi) / 2
        gm = gap(mid)
        if (gm > 0) == (glo > 0):
            lo, glo = mid, gm
        else:
            hi = mid
    a_it = (lo + hi) / 2
    lo, hi = bracket_amp
    if not (_min_gap(lo) < 0 < _min_gap(hi)):
        raise Refusal("no-solution", f"the spinodal is not bracketed by {bracket_amp}")
    while hi - lo > 1e-7:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _min_gap(mid) < 0 else (lo, mid)
    return Thresholds(a_it, (lo + hi) / 2,
                      "ising prior, sign channel, rho = 1; alpha_IT: f_RS of the GAMP fixed "
                      "point = -ln 2 (perfect recovery); alpha_AMP: spinodal of the low "
                      "fixed point; bisection to 1e-7 in alpha")
