"""Cavity model for the spherical perceptron (Gardner volume), used by dense.py.

Conventions (as in families/perceptron.py): P = alpha N patterns with i.i.d.
N(0, 1) components, labels absorbed; J on the sphere J.J = N stores the set
if lambda_mu = J.xi^mu/sqrt(N) >= kappa for every mu; S = (1/N) ln(fraction
of the sphere).  H(x) = int_x^inf Dz, E(x) = phi(x)/H(x).

Cavity argument (Mezard, "The space of interactions in neural networks:
Gardner's computation with the cavity method", J. Phys. A 22, 2181 (1989)):
  remove one pattern xi^0 (one constraint): in the version space of the other
    patterns, xi^0 is independent of J, so its stability field lambda_0 is
    Gaussian: mean sqrt(q) t over the disorder (t ~ N(0, 1), q the typical
    overlap of two couplings in the version space) and spread 1 - q.
  add the constraint Theta(lambda_0 - kappa): it keeps the fraction H(x) of the
    volume, x = (kappa - sqrt(q) t)/sqrt(1 - q).  The truncated Gaussian has
    mean sqrt(q) t + sqrt(1 - q) E(x) (the reaction: the constraint pushes the
    field) and variance (1 - q)(1 - E(E - x)).
  self-consistency: each pattern pushes the coupling centre by the cavity force
    g = (shift of <lambda_0>)/(1 - q) along xi^0/sqrt(N); the P pushes are
    independent, so the squared length of the centre's displacement closes as
    q/(1 - q)^2 = alpha E_t g^2.
  entropy (thermodynamic integration; the cavity gives dS/dalpha, not S):
    adding one pattern multiplies the volume by H(x), so dS/dalpha = E_t ln H(x)
    with S(0) = 0.  S(alpha*) = alpha* L(q*) - int_0^{q*} alpha(q) L'(q) dq (by
    parts, alpha(q) from the closure), L(q) = E_t ln H(x), and L'(q) =
    -E_t[E(x) dx/dq], dx/dq = (kappa - t/sqrt(q))/(2 (1 - q)^{3/2}) (chain rule
    under the integral, no Gaussian integration by parts); composite
    Gauss-Legendre in s = -ln(1 - q).
  stability: the variance reduction 1 - Var_post/Var_prior = E(E - x) is the
    pattern's response; lambda = 1 - alpha E_t[(E(E - x))^2] > 0 is the cavity
    condition for a single RS state (the same quantity as the replicon).
  capacity (q -> 1): (1 - q) g -> (kappa - t)_+, so the closure becomes
    1 = alpha_c E_t[(kappa - t)_+^2]: alpha_c = 1/int_{-inf}^kappa Dt (kappa - t)^2,
    integrated numerically (= 2 at kappa = 0); lambda -> 1 - alpha_c P(t < kappa)
    (the response is 1 where the constraint binds, 0 elsewhere).
  Above capacity the version space is empty: S = -inf (same as the replica side).

Ising couplings are not covered by this cavity module (refused).
Numerics: entropy float64 (cavity/_quad.py), capacity mpmath at 15 digits.
"""
import math
from dataclasses import dataclass
from functools import lru_cache

import mpmath as mp
import numpy as np

from ..core import Refusal
from ..families.common import CAVITY_CAVEAT, Capacity
from . import _quad as Q

_MEZARD = "Mezard, J. Phys. A 22, 2181 (1989)"
_ERFC = np.frompyfunc(math.erfc, 1, 1)
_XA = 25.0
_SQ2PI = math.sqrt(2 * math.pi)
_GL_S, _GW_S = np.polynomial.legendre.leggauss(32)


@dataclass(frozen=True)
class GardnerPoint:
    alpha: object
    q: object
    S: object
    stability: object
    kappa: object
    caveat: str = CAVITY_CAVEAT


def _series(x):                           # x H(x)/phi(x), asymptotic (x >= 25)
    y = 1 / (x * x)
    return 1 - y + 3 * y**2 - 15 * y**3 + 105 * y**4


def _lnH(x):
    small = np.minimum(x, _XA)
    big = np.maximum(x, _XA)
    direct = np.log(_ERFC(small / math.sqrt(2)).astype(float) / 2)
    return np.where(x < _XA, direct, -big * big / 2 - np.log(big * _SQ2PI) + np.log(_series(big)))


def _E(x):
    small = np.minimum(x, _XA)
    big = np.maximum(x, _XA)
    direct = np.exp(-small * small / 2) / _SQ2PI / (_ERFC(small / math.sqrt(2)).astype(float) / 2)
    return np.where(x < _XA, direct, big / _series(big))


class _Field:
    desc = "lambda_0 ~ N(sqrt(q) t, 1 - q), t ~ N(0, 1)"

    def __init__(self, kappa):
        self.k = kappa

    def grid(self, q):
        """t-nodes and x(t) at overlap q."""
        s, r = math.sqrt(q), math.sqrt(1 - q)
        t, w = Q.nodes(Q.kink_cuts(self.k / s, r / s) + (0.0,), L=12.0)
        return t, w, (self.k - s * t) / r


class _Site:
    desc = ("constraint Theta(lambda_0 - kappa): kept fraction H(x); mean shift "
            "sqrt(1 - q) E(x); variance ratio 1 - E(E - x)")

    @staticmethod
    def shift(q, x):
        return math.sqrt(1 - q) * _E(x)

    @staticmethod
    def var_ratio(x):
        e = _E(x)
        return 1 - e * (e - x)


def _alpha_of(q, law, site):
    t, w, x = law.grid(q)
    g = site.shift(q, x) / (1 - q)
    return q / ((1 - q)**2 * float(w @ (g * g)))


def _entropy(a, k, law, site):
    f = lambda u: math.log(_alpha_of(1 / (1 + math.exp(-u)), law, site) / a)
    u = Q.illinois(f, -30.0, 28.0, xtol=1e-15)
    q = 1 / (1 + math.exp(-u))
    res = abs(_alpha_of(q, law, site) / a - 1)
    if res > 1e-12:
        raise Refusal("no-solution:audit", f"cavity closure residual {res} at q = {q}")

    def L(q):
        t, w, x = law.grid(q)
        return float(w @ _lnH(x))

    def dL(q):
        t, w, x = law.grid(q)
        return -float(w @ (_E(x) * (k - t / math.sqrt(q)))) / (2 * (1 - q)**1.5)

    smax = -math.log1p(-q)
    npan = max(1, math.ceil(smax / 1.5))
    integral = 0.0
    for i in range(npan):
        lo, hi = smax * i / npan, smax * (i + 1) / npan
        for xs, ws in zip(_GL_S, _GW_S):
            s = (hi - lo) / 2 * xs + (hi + lo) / 2
            qq = -math.expm1(-s)
            integral += (hi - lo) / 2 * ws * _alpha_of(qq, law, site) * dL(qq) * (1 - qq)
    S = a * L(q) - integral
    t, w, x = law.grid(q)
    lam = 1 - a * float(w @ (1 - site.var_ratio(x))**2)
    return q, S, lam


class GardnerCavity:
    name = "perceptron"
    cites = {"remove_site": _MEZARD + " (remove one pattern)",
             "cavity_field_law": _MEZARD + " (Gaussian cavity stability field)",
             "add_site": _MEZARD + " (add the constraint; volume fraction H(x))",
             "self_consistency": _MEZARD + " (closure for q; Gardner, J. Phys. A 21, 257 (1988))"}

    def check(self, p):
        v = p.get("variables", "spherical")
        if v == "ising":
            raise Refusal("not-implemented:cavity-ising",
                          "the cavity module covers spherical couplings only (J.J = N); "
                          "use the replica side for Ising couplings")
        if v != "spherical":
            raise Refusal("precondition:known-variables",
                          f"variables = {v!r}: use 'spherical' (J.J = N) or 'ising' (J = +-1)")
        k, a = mp.mpf(p.get("kappa", 0)), mp.mpf(p.get("alpha", 0))
        if not mp.isfinite(k):
            raise Refusal("precondition:finite-kappa", f"kappa = {k}")
        if not a >= 0:
            raise Refusal("precondition:nonnegative-alpha", f"alpha = {a}")
        return {"alpha": a, "kappa": k, "variables": v}

    def remove(self, p):
        return {"desc": "removed pattern xi^0 (one constraint); couplings on J.J = N"}

    def field_law(self, p, sys_):
        return _Field(float(p["kappa"]))

    def add(self, p, sys_, law):
        return _Site()

    def close(self, p, sys_, law, site):
        a, k = p["alpha"], p["kappa"]
        if a == 0:
            return GardnerPoint(a, mp.mpf(0), mp.mpf(0), mp.mpf(1), k)
        cap = capacity(mp.nstr(k, 25))
        if a >= cap.alpha_c:
            return GardnerPoint(a, mp.mpf(1), mp.ninf, None, k)   # empty version space
        q, S, lam = _entropy(float(a), float(k), law, site)
        if lam < 0:
            raise Refusal("unstable:rs", f"cavity stability 1 - alpha E[(E(E - x))^2] = "
                          f"{lam:.6g} < 0 at alpha = {mp.nstr(a, 8)}: RS is AT-unstable, "
                          "its entropy is not reported")
        return GardnerPoint(a, mp.mpf(q), mp.mpf(S), mp.mpf(lam), k)

    def summary(self, sol):
        return f"q = {mp.nstr(sol.q, 10)}, S = {mp.nstr(sol.S, 10)}"


_MARGINAL = mp.mpf("1e-13")      # |lambda| below the quadrature error: reported as 0


@lru_cache(maxsize=None)
def capacity(kappa_key) -> Capacity:
    with mp.workdps(15):
        k = mp.mpf(kappa_key)
        c = 1 / mp.sqrt(2 * mp.pi)
        dens = lambda t: c * mp.exp(-t * t / 2)
        D = mp.quad(lambda t: (k - t)**2 * dens(t), [-mp.inf, k - 3, k])
        lam = mp.quad(lambda t: ((k - t)**2 - 1) * dens(t), [-mp.inf, k - 3, k]) / D
        lam = mp.mpf(0) if abs(lam) < _MARGINAL else lam
        ac = 1 / D
        qn = 1 - 1e-8                               # audit: the closure tends to 1/D
        rel = abs(_alpha_of(qn, _Field(float(k)), _Site()) * float(D) - 1)
        if rel > 1e-3:
            raise Refusal("no-solution:audit", f"closure alpha(q) at 1 - q = 1e-8 is {rel:.3g} "
                          "away from the q -> 1 limit 1/D")
        if k > 0:
            note = "kappa > 0: cavity stability > 0 at capacity (convex version space)"
        elif k == 0:
            note = "kappa = 0: cavity stability -> 0 at alpha_c (marginal)"
        else:
            note = "kappa < 0: cavity stability < 0 at capacity; RS alpha_c is not the capacity (RSB needed)"
        return Capacity(ac, bool(lam >= 0), "rs", "spherical", k, lam,
                        "cavity route (q -> 1 of the closure): " + note,
                        caveat=CAVITY_CAVEAT)
