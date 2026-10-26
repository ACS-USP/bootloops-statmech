"""The variationally optimal online modulation of the perceptron, and its noise robustness.

Variational step (Kinouchi & Caticha, J. Phys. A 25, 6243 (1992), NOT fetched: read through
Vicente, Kinouchi & Caticha, Machine Learning 32, 179 (1998), arXiv:cond-mat/9801297, Eqs. (10),
(13)-(17), and Biehl & Riegler, Europhys. Lett. 28, 525 (1994), Eq. (10)): in normalised fields
the overlap obeys drho/dalpha = rho <W (b y / rho - Delta) - W^2 / 2> (VKC Eq. (10) without drift,
Delta = b x); the functional derivative in W(V), V = {b, x}, vanishes at

  W*(Delta) = < b y / rho - Delta >_{y | Delta, b} = (s/rho) (1 - 2 lam) phi(u) / (lam + (1 - 2 lam) Phi(u)),

u = rho Delta / s, s = sqrt(1 - rho^2), lam the flip probability of the label (the y average
done here; it is VKC Eq. (25) with their chi = 2 lam and their lambda = s/rho, and at lam = 0
BR Eq. (10), whose F = |J| W*).  With W* the drift of rho is rho <W*^2> / 2 and <W* Delta> = 0.

OPTIMALITY_CLASS states the class it is optimal in.  As rho -> 1 (Delta = s t / rho):
eps -> C / alpha with C = 2 / I(lam), I = (1 - 2 lam)^2 int Dt e^{-t^2/2} / (lam + (1 - 2 lam) H(t))
(derived here; C(0) = 0.8834, the 0.88 of VKC Tables I-II and BR p. 4).  Copelli et al.,
Europhys. Lett. 37, 427 (1997), Eq. (8) PRINT I with Dt e^{-t^2}, which gives C(0) = 1.325: a
discrepancy with their own group's 0.88, read as a typo (dt / sqrt(2 pi) written as Dt);
asymptotic_constant(printed=True) evaluates the printed form, for the record only.

Noise robustness (Copelli et al. 1997, output noise, their K = 1): the rule built on an estimate
Lam of the true lam.  noise_boundaries(lam) locates, by bisection on Lam and the sign of drho/dalpha
of the ODE flow (odes.py), the Lam below which rho = 0 attracts (no learning; near rho = 0) and
below which rho = 1 repels (imperfect learning; near rho = 1).
"""
import math
from collections import namedtuple

import mpmath as mp
import numpy as np

from ..core import Refusal
from .fields import log_ghat

OPTIMALITY_CLASS = (
    "maximises drho/dalpha (so the decrease of eps per example) at every alpha, among modulated "
    "Hebbian online updates J <- J + (1/N) F b xi whose F depends on the current example only "
    "through the observed label b and the student field h (and |J|), and that may use the TRUE "
    "overlap rho and the noise level (Kinouchi-Caticha 1992; VKC 1998 Eqs. (13)-(17)); not claimed "
    "optimal against rules that use several examples or other information, nor against batch "
    "(offline) learning; with a learning rate eta != 1 it is no longer the maximiser")

NoiseBoundaries = namedtuple("NoiseBoundaries", "no_learning robust robust_error")


def _overlap(rho):
    if not (isinstance(rho, (int, float)) and 0 < rho <= 1):
        raise Refusal("precondition:overlap", f"rho={rho!r}: the optimal modulation needs 0 < rho <= 1")
    return float(rho)


def w_star(delta, theta, noise=0.0):
    """W*(Delta) per unit |J| at overlap cos(theta) for an assumed flip probability `noise`."""
    rho, s = math.cos(theta), math.sin(theta)
    _overlap(rho)
    d = np.asarray(delta, dtype=float)
    if s == 0:
        return np.zeros_like(d)
    u = rho * d / s
    return (s / rho) * (1 - 2 * noise) * np.exp(-u * u / 2 - 0.5 * math.log(2 * math.pi)
                                                - log_ghat(u, noise))


class OptimalModulation:
    """F*(h, b, Q) = sqrt(Q) W*(b h / sqrt Q) at a given rho and assumed noise (BR's f_opt)."""
    optimality_class = OPTIMALITY_CLASS

    def __init__(self, rho, noise=0.0):
        self.rho = _overlap(rho)
        if not (isinstance(noise, (int, float)) and 0 <= noise < 0.5):
            raise Refusal("precondition:noise", f"noise={noise!r}: a flip probability in [0, 1/2)")
        self.noise = noise
        self._theta = math.acos(self.rho)

    def __call__(self, h, b, Q):
        j = math.sqrt(Q)
        return j * w_star(np.asarray(b) * np.asarray(h) / j, self._theta, self.noise)


def optimal_modulation(rho, noise=0.0):
    """The optimal modulation as a callable F(h, b, Q); see OPTIMALITY_CLASS."""
    return OptimalModulation(rho, noise)


def asymptotic_constant(noise=0.0, printed=False):
    """C = lim alpha eps(alpha) of the optimal rule with the noise known: 2 / I(noise) (mpmath).
    printed=True: Copelli et al. Eq. (8) as printed (Dt e^{-t^2}), for the record."""
    if not (isinstance(noise, (int, float)) and 0 <= noise < 0.5):
        raise Refusal("precondition:noise", f"noise={noise!r}: a flip probability in [0, 1/2)")
    with mp.workdps(20):
        lam = mp.mpf(noise)
        a = 1 if printed else mp.mpf(1) / 2
        f = lambda t: (mp.exp(-t * t / 2 - a * t * t) / mp.sqrt(2 * mp.pi)
                       / (lam + (1 - 2 * lam) * mp.erfc(t / mp.sqrt(2)) / 2))
        return float(2 / ((1 - 2 * lam) ** 2 * mp.quad(f, [-mp.inf, -3, 0, 3, mp.inf])))


def _bisect(fn, lo, hi, it=50):
    flo = fn(lo)
    if (flo > 0) == (fn(hi) > 0):
        return None
    for _ in range(it):
        mid = (lo + hi) / 2
        fm = fn(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return (lo + hi) / 2


def noise_boundaries(noise):
    """NoiseBoundaries(no_learning, robust, robust_error) for the true flip probability `noise`:
    the estimates Lam at which drho/dalpha changes sign near rho = 0 (at rho = 1e-6) and near
    rho = 1 (at theta = 1e-4; robust_error = the shift when theta = 1e-3 is used instead).  None
    where drho/dalpha keeps one sign on 0 <= Lam <= noise (no boundary in the window)."""
    from .odes import perceptron_flow
    if not (isinstance(noise, (int, float)) and 0 < noise < 0.5):
        raise Refusal("precondition:noise", f"noise={noise!r}: need 0 < noise < 1/2")

    def drho(theta):
        return lambda est: -perceptron_flow("optimal", noise=noise,
                                            noise_estimate=est).theta_rate(0.0, theta, 1.0)
    nl = _bisect(drho(math.acos(1e-6)), 0.0, float(noise))
    rb = _bisect(drho(1e-4), 0.0, float(noise))
    rb3 = _bisect(drho(1e-3), 0.0, float(noise))
    err = None if rb is None or rb3 is None else abs(rb - rb3)
    return NoiseBoundaries(nl, rb, err)


__all__ = ["NoiseBoundaries", "OPTIMALITY_CLASS", "OptimalModulation", "asymptotic_constant",
           "noise_boundaries", "optimal_modulation", "w_star"]
