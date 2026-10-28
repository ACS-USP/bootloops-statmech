"""Retrieval thresholds of the dense associative memory (Lucibello & Mezard, Phys. Rev. Lett. 132, 077301
(2024), arXiv:2304.14964v4, Secs. II-VI and Apps. B-E).

  alpha_1(ens, lam)                 the single-pattern retrieval threshold, Eq. (7): the root of
                                    phi_{alpha,1}(lam) = 1 (rem_noise.phi); `level` replaces the 1 (an
                                    EXTENSION, not in the paper: see below)
  alpha_1_closed(ens, lam)          the same threshold by Legendre duality, no root finding:
                                    s(level) if lam >= s'(level), else lam level - zeta(lam)
                                    (Gaussian: the paper's (D7), lam (1 - lam/2) below 1, 1/2 above)
  lambda_1(ens, alpha)              its inverse: the smallest lam that retrieves at load alpha
  alpha_c_lb_gaussian(lam)          the lower bound on the all-patterns capacity of Gaussian patterns,
                                    Eqs. (9)-(10), (D8)-(D13); log(2)/4 for lam >= LAMBDA_SAT = 0.70091...
  capacity(ens, lam)                (value, kind): spherical and binary alpha_c = alpha_1 (union bound,
                                    (C6)-(C7); the binary case is DERIVED HERE, see ensembles.py), Gaussian
                                    the lower bound above
  cos_theta_c(ens, alpha, lam)      the basin of attraction, Sec. IV: rho cos(theta) = phi (one-step
                                    retrieval; "we conjecture [theta_c] to be tight"), and its large-lam
                                    limit the Gilbert-Varshamov cosine c_max(alpha) = eps*(alpha, 1) (App. B)
  cos_theta_few_patterns(P, N)      (B8)-(B9): sqrt(2 log P / N) for P << e^N
  ramsauer_lower_bound(lam)         the Ramsauer et al. bound as restated in App. E, Eq. (E7)
  scaled_threshold(lam_tilde)       Sec. VI: with lam = lam~/N^a and P = exp(alpha~ N^{1-a}),
                                    alpha~_1 = alpha~_c = lam~

EXTENSION `level` (not in the paper).  Eq. (7) compares the signal x.xi^1 = N (x = xi^1, unit norm) with the
noise N phi.  A query x whose signal overlap with the retrieved state is `level` < 1 instead (for example
the normalised sum of two patterns, level = 1/sqrt(2)) has the same noise phi_{alpha,1}(lam) when x is
independent of the other patterns (the paper's own condition for Eq. (6)), so the same argument gives
phi_{alpha,1}(lam) < level.  The package computes it, labels it not-computed (never final) and does NOT
bank it: the paper does not treat it.
"""
import mpmath as mp

from ..core import Refusal
from . import ensembles as EN
from . import rem_noise as RN
from .ensembles import _prec, check_ensemble, drate, num, rate, zeta
from .rem_noise import _bisect, eps_star

# planted-error hook for the tests: {"alpha_1": factor} scales the closed threshold
_PLANT = {}
_ALPHA_CAP = mp.mpf(60)


def _level(level):
    return num(level, "level", positive=True)


def _check_level(ens, lv, r):
    """A signal overlap cannot exceed the largest overlap of the ensemble (rho for spherical, 1 for binary)."""
    edge = {"gaussian": mp.inf, "spherical": r, "binary": mp.mpf(1)}[ens]
    if lv > edge:
        raise Refusal("no-solution:level-beyond-support",
                      f"level {mp.nstr(lv, 5)} exceeds the largest possible overlap {mp.nstr(edge, 5)}")


def alpha_1(ens, lam, level=1, rho=1, method="closed"):
    """sup{alpha : phi_{alpha,rho}(lam) < level}: Eq. (7) at rho = 1, level = 1.  Root of the increasing
    function phi - level in alpha, by bisection; method picks the phi route (rem_noise.phi)."""
    with _prec():
        r, l, lv = check_ensemble(ens, rho), num(lam, "lambda", positive=True), _level(level)
        _check_level(ens, lv, r)
        f = lambda a: RN.phi(ens, a, l, r, method) - lv
        hi = mp.mpf("0.0625")
        while f(hi) < 0:
            hi *= 2
            if hi > _ALPHA_CAP:
                raise Refusal("no-solution:no-threshold",
                              f"phi_(alpha,{mp.nstr(r, 3)})({mp.nstr(l, 5)}) stays below level "
                              f"{mp.nstr(lv, 5)} for every alpha <= {mp.nstr(_ALPHA_CAP, 3)}")
        return _bisect(f, mp.mpf(0), hi, what="alpha_1")


def alpha_1_closed(ens, lam, level=1, rho=1):
    """alpha_1 by Legendre duality of Eq. (A17): the threshold phi = level sits in the uncondensed
    phase (alpha = lam level - zeta(lam)) when lam < s'(level), and in the condensed phase
    (alpha = s(level), eps* = level) otherwise.  Gaussian, rho = level = 1: (D7); spherical: always
    uncondensed (s' = inf at the edge), alpha_1 = lam - zeta(lam), the phase boundary the paper plots
    in Fig. 1; binary: lam - log cosh(lam) (DERIVED HERE)."""
    with _prec():
        r, l, lv = check_ensemble(ens, rho), num(lam, "lambda", positive=True), _level(level)
        k = mp.mpf(_PLANT.get("alpha_1", 1))
        _check_level(ens, lv, r)
        if l >= drate(ens, lv, r):
            return k * rate(ens, lv, r)
        return k * (l * lv - zeta(ens, l, r))


def lambda_1(ens, alpha, level=1, rho=1):
    """The smallest lam with phi_{alpha,rho}(lam) < level (phi decreases in lam, Sec. II): the inverse of
    alpha_1.  Refused (no-solution:no-retrieval) when even lam -> inf (phi -> eps*(alpha)) does not
    get below the level, i.e. alpha is above the hard-max (Gilbert-Varshamov) limit."""
    with _prec():
        r, a, lv = check_ensemble(ens, rho), num(alpha, "alpha"), _level(level)
        es = eps_star(ens, a, r)
        if es >= lv:
            raise Refusal("no-solution:no-retrieval",
                          f"alpha = {mp.nstr(a, 6)}: even lam -> inf leaves phi -> eps* = "
                          f"{mp.nstr(es, 8)} >= level {mp.nstr(lv, 5)} (no pattern is retrieved at any lam)")
        f = lambda l: lv - RN.phi(ens, a, l, r)
        lo, hi = mp.mpf("1e-12"), mp.mpf(1)
        while f(hi) < 0:
            hi *= 2
            if hi > mp.mpf(10) ** 12:
                raise Refusal("no-solution:bracket", f"lambda_1 beyond 1e12 at alpha = {alpha!r}")
        return _bisect(f, lo, hi, what="lambda_1")


# ---- all-patterns retrieval: the Gaussian lower bound ----------------------------------------------
def _rate_norm(r):
    """I~(r) = (r^2 - 1)/2 - log r, the rate function of the scaled pattern norm (D8)."""
    return (r**2 - 1) / 2 - mp.log(r)


def _gauss_A(alpha, lam):
    """A(alpha, lam) of Eq. (9) for Gaussian patterns, (D12)-(D13): the cost of the cheapest way for a
    typical pattern to fail (an atypical norm r, or an atypically large REM free energy)."""
    r0 = mp.sqrt(alpha / (lam * (1 - lam / 2))) if lam < 1 else mp.sqrt(2 * alpha)       # (D12)
    first = _rate_norm(r0) if r0 < 1 else mp.mpf(0)           # min over r <= r0 of I~ (I~ decreases below 1)
    r2 = max(r0, 1 / mp.sqrt(2))                              # min over r >= r0 of I~(r) + r^2/2 - alpha
    second = _rate_norm(r2) + r2**2 / 2 - alpha               # stationary at r^2 = 1/2
    return min(first, second)


def alpha_c_lb_gaussian(lam):
    """The lower bound alpha_c^lb(lam) on the capacity for Gaussian patterns: the root of
    A(alpha, lam) = alpha, Eq. (10) with (D13).  log(2)/4 for lam >= LAMBDA_SAT; ~ lam at small lam."""
    with _prec():
        l = num(lam, "lambda", positive=True)
        f = lambda a: _gauss_A(a, l) - a
        # A - alpha is positive at small alpha (every failure costs more than the log P union bound)
        return _bisect(lambda a: -f(a), mp.mpf("1e-14"), mp.mpf("0.49"), what="alpha_c^lb")


def lambda_saturation():
    """The smallest lam at which alpha_c^lb(lam) reaches log(2)/4 ("lam >~ 0.70091", App. D 2): the
    second branch of A is then the active one with r^2 = 1/2, i.e. I~(r0) = log(2)/4 with
    r0^2 = alpha/(lam (1 - lam/2))."""
    with _prec():
        a = mp.log(2) / 4
        r0 = mp.findroot(lambda r: _rate_norm(r) - a, mp.mpf("0.6"))
        return 1 - mp.sqrt(1 - 2 * a / r0**2)         # lam (1 - lam/2) = alpha / r0^2


def capacity(ens, lam):
    """(alpha_c-or-bound, kind).  spherical: alpha_c = alpha_1 exactly (C6)-(C7); binary: the same
    union-bound argument gives alpha_1 (DERIVED HERE, with the retrieval criterion phi < 1 of Eq. (7));
    gaussian: the lower bound alpha_c^lb < alpha_1 (a gap at any lam; Fig. 8's simulations sit near it)."""
    check_ensemble(ens)
    if ens == "gaussian":
        return alpha_c_lb_gaussian(lam), "lower-bound"
    return alpha_1(ens, lam), "equals-alpha_1"


# ---- basins of attraction and geometry -----------------------------------------------------------
def cos_theta_c(ens, alpha, lam, rho=1, r_xi=1):
    """cos(theta_c) = phi_{alpha,rho}(lam)/(rho r_xi) (Sec. IV): an initial x at norm rho sqrt(N) and
    angle theta from a typical pattern (r_xi = 1) is retrieved in one step when rho r_xi cos(theta) >
    phi.  A value >= 1 means no initial condition is inside the one-step basin."""
    with _prec():
        r = check_ensemble(ens, rho)
        return RN.phi(ens, alpha, lam, r) / (r * num(r_xi, "r_xi", positive=True))


def cos_theta_nearest(ens, alpha, rho=1):
    """c_max(alpha) rho: the cosine to the nearest of the e^{alpha N} other patterns (the continuous
    Gilbert-Varshamov distance, App. B), lim_{lam->inf} cos theta_c = eps*(alpha, rho)/rho * rho."""
    with _prec():
        return eps_star(ens, alpha, rho)


def cos_theta_few_patterns(P, N):
    """(B8)-(B9): for P << e^N the REM is condensed at any finite lam and cos(theta_c) = sqrt(2 log P/N),
    so the basins are very large (cos -> 0 like sqrt(log N/N) for P = c N^k)."""
    with _prec():
        P, N = num(P, "P", positive=True), num(N, "N", positive=True)
        return mp.sqrt(2 * mp.log(P) / N)


def theta_c(cos_value):
    """The angle (radians) of a cos(theta_c) in [0, 1); refused (no-solution:no-basin) at >= 1."""
    c = num(cos_value, "cos")
    if c >= 1:
        raise Refusal("no-solution:no-basin", f"cos(theta_c) = {mp.nstr(c, 8)} >= 1: no one-step basin")
    return mp.acos(max(c, mp.mpf(-1)))


# ---- the Ramsauer et al. bound and the scaled regime ----------------------------------------------
def ramsauer_lower_bound(lam):
    """alpha_{c,rams}^lb(lam) = (1/4) log( (2 lam/5) / W0(2 lam/5) ), Eq. (E7): the thermodynamic limit of
    Theorem 3 of Ramsauer et al. (their reference [8], which this package has NOT read), as restated by
    the paper.  ~ lam/10 at small lam, ~ (1/4) log lam at large lam; "a large gap" below alpha_1."""
    with _prec():
        x = 2 * num(lam, "lambda", positive=True) / 5
        return mp.log(x / mp.lambertw(x).real) / 4


def scaled_threshold(lam_tilde):
    """Sec. VI: energy -(N^a/lam~) log sum exp(lam~ x.xi/N^a) with P = exp(alpha~ N^{1-a}), 0 < a < 1:
    the noise is N^{1-a} alpha~, the signal lam~ N^{1-a} for ANY pattern ensemble, so retrieval of a
    typical pattern and of all patterns coincide at alpha~_1 = alpha~_c = lam~."""
    return num(lam_tilde, "lambda_tilde", positive=True)


__all__ = ["alpha_1", "alpha_1_closed", "alpha_c_lb_gaussian", "capacity", "cos_theta_c",
           "cos_theta_few_patterns", "cos_theta_nearest", "lambda_1", "lambda_saturation",
           "ramsauer_lower_bound", "scaled_threshold", "theta_c"]
