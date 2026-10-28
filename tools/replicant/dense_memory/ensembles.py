"""Pattern ensembles of the dense associative memory: the cumulant generating function zeta and the
rate function s of the overlap eps = x.xi/N (Lucibello & Mezard, Phys. Rev. Lett. 132, 077301 (2024),
arXiv:2304.14964v4 (PDF read; journal version not checked), Appendix A 2-3, Eqs. (A10)-(A16)).

For patterns xi^mu in R^N drawn i.i.d. from a rotationally invariant law with E|xi|^2 = N and a
configuration x with |x| = rho sqrt(N) independent of them, the overlap eps_mu = x.xi^mu/N has the
large-deviation rate s_rho(eps), the Legendre transform of

    zeta_rho(lam) = lim (1/N) log E exp(lam x.xi)                           (A16)

    ensemble    zeta_rho(lam)                                           s_rho(eps)              source
    gaussian    lam^2 rho^2 / 2                                         eps^2 / (2 rho^2)       (D4), (D3)
    spherical   (q - 1 - log((1 + q)/2)) / 2,  q = sqrt(1+4 lam^2 rho^2)  -log(1-eps^2/rho^2)/2    (C3), (C4)
    binary      log cosh(lam)   (rho = 1, x in {-1,+1}^N, xi in {-1,+1}^N)  I(eps)                  DERIVED HERE

`binary` is NOT in the paper (its Conclusions only say that binary patterns "should" be treatable): it is
the paper's recipe (A16) applied to xi_i = +-1, E exp(lam x_i xi_i) = cosh(lam), and its rate function is
I(eps) = ((1+eps) log(1+eps) + (1-eps) log(1-eps))/2, the function printed in Demircigil et al.
(arXiv:1702.01929, Thm 1.3).  It needs rho = 1 (Ising x has |x|^2 = N) and its levels are bounded,
|eps| <= 1, so s is finite only there (the REM is then bounded: alpha_max = log 2 = I(1)).

Functions take mpmath numbers (decimal strings are parsed at 30 digits) and return mpf.
"""
import mpmath as mp

from ..core import Refusal

ENSEMBLES = ("gaussian", "spherical", "binary")
_DPS = 30
# planted-error hook for the tests: scale a closed form by a factor, e.g. {"zeta": 2}
_PLANT = {}


def _prec():
    return mp.workdps(max(_DPS, mp.mp.dps))


def num(x, name, *, positive=False):
    try:
        with _prec():
            v = mp.mpf(x)
    except (ValueError, TypeError):
        raise Refusal("precondition:known-parameter", f"{name} = {x!r} is not a number")
    if not mp.isfinite(v) or (positive and not v > 0):
        raise Refusal("precondition:known-parameter",
                      f"{name} = {x!r}: need a finite number" + (" > 0" if positive else ""))
    return v


def check_ensemble(ens, rho=1):
    """The validated rho (mpf) for ensemble `ens`; refuses an unknown ensemble and a binary rho != 1."""
    if ens not in ENSEMBLES:
        raise Refusal("precondition:known-parameter",
                      f"ensemble = {ens!r}: need one of {', '.join(ENSEMBLES)}")
    r = num(rho, "rho", positive=True)
    if ens == "binary" and r != 1:
        raise Refusal("unsupported:ensemble-rho",
                      f"binary patterns with Ising x have |x|^2 = N, so rho = 1 only (rho = {rho!r}); "
                      "a continuous x with +-1 patterns is not rotationally invariant and needs the "
                      "full vector law, which this entry does not treat")
    return r


def zeta(ens, lam, rho=1):
    """zeta_rho(lam), Eq. (A16) (closed forms of the table above)."""
    with _prec():
        r, l = check_ensemble(ens, rho), num(lam, "lambda")
        k = mp.mpf(_PLANT.get("zeta", 1))
        if ens == "gaussian":
            return k * l**2 * r**2 / 2
        if ens == "spherical":
            q = mp.sqrt(1 + 4 * l**2 * r**2)
            return k * (q - 1 - mp.log((1 + q) / 2)) / 2
        return k * mp.log(mp.cosh(l))


def dzeta(ens, lam, rho=1):
    """d zeta/d lam = eps~(lam), the overlap level that dominates the uncondensed sum."""
    with _prec():
        r, l = check_ensemble(ens, rho), num(lam, "lambda")
        if ens == "gaussian":
            return l * r**2
        if ens == "spherical":
            return 2 * l * r**2 / (1 + mp.sqrt(1 + 4 * l**2 * r**2))
        return mp.tanh(l)


def rate(ens, eps, rho=1):
    """s_rho(eps), the rate function; +inf outside the support of the ensemble."""
    with _prec():
        r, e = check_ensemble(ens, rho), num(eps, "eps")
        if ens == "gaussian":
            return e**2 / (2 * r**2)
        if ens == "spherical":
            return mp.inf if abs(e) >= r else -mp.log(1 - e**2 / r**2) / 2
        return binary_rate(e)


def drate(ens, eps, rho=1):
    """s'_rho(eps) = (d zeta/d lam)^{-1}(eps): the inverse temperature at which level eps dominates;
    +inf at and beyond the edge of a bounded support."""
    with _prec():
        r, e = check_ensemble(ens, rho), num(eps, "eps")
        if ens == "gaussian":
            return e / r**2
        if ens == "spherical":
            return mp.inf if abs(e) >= r else (e / r**2) / (1 - e**2 / r**2)
        return mp.inf if abs(e) >= 1 else mp.atanh(e)


def binary_rate(x):
    """I(x) = ((1+x) log(1+x) + (1-x) log(1-x))/2 on [-1, 1] (+inf outside; I(+-1) = log 2); the rate
    function of a +-1 overlap and the function Demircigil et al. call I (Thm 1.3)."""
    with _prec():
        x = mp.mpf(x)
        if abs(x) > 1:
            return mp.inf
        t = lambda u: u * mp.log(u) if u > 0 else mp.mpf(0)
        return (t(1 + x) + t(1 - x)) / 2


def alpha_max(ens):
    """Largest alpha with a finite maximal overlap: log 2 for binary (P > 2^N repeats patterns),
    +inf for the unbounded rate functions."""
    return mp.log(2) if ens == "binary" else mp.inf


__all__ = ["ENSEMBLES", "alpha_max", "binary_rate", "check_ensemble", "drate", "dzeta", "num", "rate",
           "zeta"]
