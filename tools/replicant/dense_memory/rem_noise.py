"""The REM noise function phi_{alpha,rho}(lam) of the dense associative memory with P = e^{alpha N} patterns
(Lucibello & Mezard, Phys. Rev. Lett. 132, 077301 (2024), arXiv:2304.14964v4, Sec. II and App. A).

The energy of the memory, E(x) = -(1/lam) log sum_mu exp(lam x.xi^mu) + |x|^2/2 (their Eq. (1)), splits
for an x independent of patterns mu >= 2 into signal exp(lam x.xi^1) and noise exp(lam N Phi(x)), where
Phi is minus the free-energy density of a Random Energy Model with P - 1 i.i.d. levels
eps_mu = x.xi^mu/N and inverse temperature lam (their Eq. (4)).  Its almost sure limit phi = lim Phi(x)
(their (A15), Eq. (6): E(x) ~ -max(x.xi^1, N phi) + |x|^2/2) is computed here by FOUR routes that share
only zeta / the rate function (ensembles.py):

  closed    the printed closed forms: Gaussian (5) = (D5) with lam* (D6); spherical (C5) and the
            unnumbered line above it; binary (DERIVED HERE): (A17) with zeta = log cosh
  legendre  (A17) from zeta alone: lam* from lam zeta'(lam) - zeta(lam) = alpha, eps* = zeta'(lam*),
            phi = (alpha + zeta(lam))/lam below lam*, eps* above
  1rsb      (A24)-(A27): phi = (1/lam) min_{0 < m <= 1} [alpha + zeta(lam m)]/m, the one-step-RSB
            functional over the pattern LABELS, with Parisi parameter m (stationarity (A25), m = 1 at
            lam*, (A26)); this is the replica derivation the paper gives in App. A 4
  laplace   (A7): max over the levels eps in [eps_m, eps*] of alpha - s(eps) + lam eps, from the rate
            function s directly (Laplace's method on the density of levels)

The uncondensed phase (lam < lam*) is the REM paramagnet (annealed = quenched, m = 1); the condensed
(frozen) phase (lam >= lam*) is dominated by the largest level eps* (m < 1).  These are phases of the
AUXILIARY REM (the paper stresses that its energies and temperature "have nothing to do with" the memory's):
the memory itself is analysed at zero temperature through E(x).
"""
from dataclasses import dataclass

import mpmath as mp

from ..core import Refusal
from . import ensembles as EN
from .ensembles import _prec, binary_rate, check_ensemble, dzeta, num, rate, zeta

# planted-error hook for the tests: {"lam_star": factor} scales the closed condensation threshold
_PLANT = {}
_ITER = 160


def _bisect(f, lo, hi, *, what):
    """Root of an increasing f on [lo, hi] (f(lo) <= 0 <= f(hi)) by bisection at working precision."""
    flo, fhi = f(lo), f(hi)
    if flo > 0 or fhi < 0:
        raise Refusal("no-solution:bracket", f"{what}: no sign change on [{mp.nstr(lo, 5)}, "
                      f"{mp.nstr(hi, 5)}] (f = {mp.nstr(flo, 5)}, {mp.nstr(fhi, 5)})")
    for _ in range(_ITER):
        mid = (lo + hi) / 2
        if f(mid) <= 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _alpha(alpha):
    a = num(alpha, "alpha")
    if a < 0:
        raise Refusal("precondition:known-parameter", f"alpha = {alpha!r}: need alpha >= 0")
    return a


def _lam(lam):
    return num(lam, "lambda", positive=True)


def eps_star(ens, alpha, rho=1):
    """eps*(alpha, rho): the largest of the e^{alpha N} levels, the root of s_rho(eps) = alpha above the
    typical value 0 (App. A 3 step 3); the support edge (rho, or 1 for binary) when the rate function is
    bounded and alpha >= s(edge) (binary at alpha >= log 2)."""
    with _prec():
        r, a = check_ensemble(ens, rho), _alpha(alpha)
        if ens == "gaussian":
            return r * mp.sqrt(2 * a)                      # s = eps^2/(2 rho^2) = alpha
        if ens == "spherical":
            return r * mp.sqrt(1 - mp.exp(-2 * a))         # -log(1 - eps^2/rho^2)/2 = alpha
        if a >= mp.log(2):
            return mp.mpf(1)
        return _bisect(lambda e: binary_rate(e) - a, mp.mpf(0), mp.mpf(1), what="binary eps*")


def condensation_lambda(ens, alpha, rho=1, method="closed"):
    """lam*(alpha, rho) = s'(eps*), where the REM condenses.  closed: Gaussian (D6) sqrt(2 alpha)/rho,
    spherical (C5) e^{2 alpha} sqrt(1 - e^{-2 alpha})/rho, binary atanh(eps*); legendre: the root of
    lam zeta'(lam) - zeta(lam) = alpha.  +inf for the binary ensemble at alpha >= log 2 (never condensed)."""
    with _prec():
        r, a = check_ensemble(ens, rho), _alpha(alpha)
        if ens == "binary" and a >= mp.log(2):
            return mp.inf
        if method == "closed":
            k = mp.mpf(_PLANT.get("lam_star", 1))
            if ens == "gaussian":
                return k * mp.sqrt(2 * a) / r
            if ens == "spherical":
                return k * mp.exp(2 * a) * mp.sqrt(1 - mp.exp(-2 * a)) / r
            return k * mp.atanh(eps_star(ens, a, r))
        if method != "legendre":
            raise Refusal("precondition:known-parameter", f"method = {method!r}: closed or legendre")
        if a == 0:
            return mp.mpf(0)
        g = lambda l: l * dzeta(ens, l, r) - zeta(ens, l, r) - a
        hi = mp.mpf(1)
        while g(hi) < 0:
            hi *= 2
            if hi > mp.mpf(10) ** 12:
                raise Refusal("no-solution:bracket", f"lam* beyond 1e12 at alpha = {alpha!r}")
        return _bisect(g, mp.mpf(0), hi, what="condensation lambda")


@dataclass(frozen=True)
class NoisePoint:
    """One evaluation of the REM noise function: phi, the phase, lam*, eps* and the Parisi parameter m
    (1 in the uncondensed phase, lam*/lam in the condensed one, (A26)-(A27))."""
    phi: mp.mpf
    phase: str
    lam_star: mp.mpf
    eps_star: mp.mpf
    m: mp.mpf


def noise(ens, alpha, lam, rho=1, method="closed"):
    """NoisePoint at (alpha, lam, rho) by the `closed` or `legendre` route (module docstring)."""
    with _prec():
        r, a, l = check_ensemble(ens, rho), _alpha(alpha), _lam(lam)
        es = eps_star(ens, a, r)
        ls = condensation_lambda(ens, a, r, method)
        if l < ls:
            if ens == "gaussian" and method == "closed":
                phi = a / l + l * r**2 / 2                           # Eq. (5) / (D5)
            elif ens == "spherical" and method == "closed":
                q = mp.sqrt(1 + 4 * l**2 * r**2)                     # the line above (C5)
                phi = (a + (q - 1 - mp.log((1 + q) / 2)) / 2) / l
            else:
                phi = (a + zeta(ens, l, r)) / l                      # (A17), uncondensed
            return NoisePoint(phi, "uncondensed", ls, es, mp.mpf(1))
        return NoisePoint(es, "condensed", ls, es, ls / l)


def phi(ens, alpha, lam, rho=1, method="closed"):
    """phi_{alpha,rho}(lam) by route `closed`, `legendre`, `1rsb` or `laplace` (module docstring)."""
    if method in ("closed", "legendre"):
        return noise(ens, alpha, lam, rho, method).phi
    if method == "1rsb":
        return phi_1rsb(ens, alpha, lam, rho)[0]
    if method == "laplace":
        return phi_laplace(ens, alpha, lam, rho)
    raise Refusal("precondition:known-parameter", f"method = {method!r}: closed, legendre, 1rsb or laplace")


def phi_1rsb(ens, alpha, lam, rho=1):
    """(phi, m*) from the 1RSB functional over pattern labels, (A24)-(A27):
    phi = (1/lam) min_{0 < m <= 1} [alpha + zeta(lam m)]/m.  The functional is convex in m, so its
    minimum on (0, 1] is the stationary point of (A25), lam m zeta'(lam m) - zeta(lam m) = alpha, when that
    lies below 1 (condensed) and m = 1 otherwise (uncondensed).  Found by bisection on (A25), never by
    inserting lam*."""
    with _prec():
        r, a, l = check_ensemble(ens, rho), _alpha(alpha), _lam(lam)
        G = lambda m: l * m * dzeta(ens, l * m, r) - zeta(ens, l * m, r) - a     # (A25) * m^2 / ... sign
        if a == 0:
            m = mp.mpf(1) if G(mp.mpf(1)) <= 0 else mp.mpf(0)
        elif G(mp.mpf(1)) <= 0:
            m = mp.mpf(1)
        else:
            m = _bisect(G, mp.mpf(0), mp.mpf(1), what="Parisi parameter")
        if m == 0:                                                 # alpha = 0, condensed: phi = 0
            return mp.mpf(0), m
        return (a + zeta(ens, l * m, r)) / (l * m), m


def phi_laplace(ens, alpha, lam, rho=1):
    """(A7): (1/lam) max over eps in [-eps*, eps*] of alpha - s(eps) + lam eps, by golden-section search
    on the concave objective, using the rate function s only (no zeta)."""
    with _prec():
        r, a, l = check_ensemble(ens, rho), _alpha(alpha), _lam(lam)
        es = eps_star(ens, a, r)
        f = lambda e: a - rate(ens, e, r) + l * e
        lo, hi = -es, es
        if es == 0:
            return mp.mpf(0)
        g = (mp.sqrt(5) - 1) / 2
        c, d = hi - g * (hi - lo), lo + g * (hi - lo)
        fc, fd = f(c), f(d)
        for _ in range(_ITER):
            if fc > fd:
                hi, d, fd = d, c, fc
                c = hi - g * (hi - lo)
                fc = f(c)
            else:
                lo, c, fc = c, d, fd
                d = lo + g * (hi - lo)
                fd = f(d)
        best = max(f((lo + hi) / 2), f(es), f(-es))
        return best / l


__all__ = ["NoisePoint", "condensation_lambda", "eps_star", "noise", "phi", "phi_1rsb", "phi_laplace"]
