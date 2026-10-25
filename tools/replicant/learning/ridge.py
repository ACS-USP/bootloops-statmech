"""Ridge and ridgeless (min-norm) least squares on isotropic Gaussian design: the asymptotic risk.

Model (Hastie, Montanari, Rosset & Tibshirani, "Surprises in high-dimensional ridgeless least
squares interpolation", Ann. Statist. 50, 949 (2022), arXiv:1903.08560 -- HMRT -- Eqs. (1), (2),
(4), (5); Sec. 3 isotropic features):
    X in R^{n x p}, i.i.d. N(0, 1) entries (rows x_i ~ N(0, I_p): Sigma = I),
    y = X beta + sigma xi,  xi ~ N(0, I_n),  |beta|^2 = r^2 ("signal"),  gamma = p / n,
    ridge  beta_lam = argmin_b (1/n)|y - X b|^2 + lam |b|^2 = (X^T X + n lam I)^{-1} X^T y,
    ridgeless beta_0 = lim_{lam -> 0+} beta_lam = (X^T X)^+ X^T y (the min-norm least squares fit).
Reported quantity: HMRT's out-of-sample risk R_X = E[(x0^T beta_hat - x0^T beta)^2 | X] =
|beta_hat - beta|^2 (Sigma = I), i.e. the EXCESS risk, bias + variance (Sec. 2.1).  The test MSE
with a fresh noisy label, E (y0 - x0^T beta_hat)^2 = R + sigma^2, is Dobriban & Wager's predictive
risk (Ann. Statist. 46, 247 (2018), arXiv:1507.03003, Thm 2.1, noise variance 1).

Closed forms (n, p -> oo, p/n -> gamma; almost sure limits):
  ridgeless (HMRT Eq. (8): Prop. 2 for gamma < 1, Thm 1 for gamma > 1)
      gamma < 1:  R = sigma^2 gamma / (1 - gamma)               (no bias)
      gamma > 1:  R = r^2 (1 - 1/gamma) + sigma^2 / (gamma - 1)  (bias + variance)
    divergent at gamma = 1 when sigma^2 > 0 (refused there: neither statement covers gamma = 1);
  ridge (HMRT Cor. 5, Eq. (43)), m the Stieltjes transform of the Marchenko-Pastur law of
  X^T X / n, m(z) = [1 - gamma - z - sqrt((1 - gamma - z)^2 - 4 gamma z)] / (2 gamma z):
      R = r^2 lam^2 m'(-lam) + sigma^2 gamma (m(-lam) - lam m'(-lam));
  optimal penalty (HMRT Cor. 5; DW Thm 2.1 with unit noise; Advani & Saxe Sec. 2.2):
      lam* = sigma^2 gamma / r^2,  R* = sigma^2 gamma m(-lam*).
Derivation of Eq. (43) (checked in latex_learning.py): with S = X^T X / n, bias
lam^2 beta^T (S + lam)^-2 beta -> r^2 lam^2 (1/p) tr (S + lam)^-2 = r^2 lam^2 m'(-lam) (rotation
invariance) and variance (sigma^2/n) tr S (S + lam)^-2 = sigma^2 gamma (m - lam m').
Evaluation here (no cancellation for any gamma, lam > 0): with b = 1 - gamma + lam and
D = b^2 + 4 gamma lam, the MP quadratic gamma z m^2 - (1 - gamma - z) m + 1 = 0 at z = -lam gives
    m = 2 / (b + sqrt D)  (b >= 0),   m = (sqrt D - b) / (2 gamma lam)  (b < 0),
    m' = m (1 + gamma m) / sqrt D,    m - lam m' = (1 - lam m) / sqrt D,
so bias = r^2 lam^2 m (1 + gamma m) / sqrt D and variance = sigma^2 gamma (1 - lam m) / sqrt D
(mpmath at 30 digits).  The planted-error hook _SCALE multiplies the bias / variance terms of the
ridge and ridgeless closed forms (tests plant a missing bias or a scaled variance; the replica saddle
point in ridge_rs.py, the simulation in ridge_sim.py and HMRT's separate R* = sigma^2 gamma m(-lam*)
in optimal_test_error do not read it).
"""
import mpmath as mp

from ..core import Refusal

_DPS = 30
_SCALE = {"bias": 1, "variance": 1}     # private: planted-error hook (tests only)


def _num(x, name, *, positive=False):
    """A finite real number >= 0 (> 0 with positive=True) as an mpf, else Refusal precondition:<name>."""
    if isinstance(x, bool) or x is None:
        raise Refusal(f"precondition:{name}", f"{name}={x!r}: need a finite number")
    try:
        with mp.workdps(_DPS):      # a decimal string keeps 30 digits (1 - gamma near the peak)
            v = mp.mpf(repr(x)) if isinstance(x, float) else mp.mpf(x)
    except (TypeError, ValueError):
        raise Refusal(f"precondition:{name}", f"{name}={x!r}: need a finite number") from None
    if not mp.isfinite(v) or v < 0 or (positive and v == 0):
        raise Refusal(f"precondition:{name}",
                      f"{name}={x!r}: need a finite number {'> 0' if positive else '>= 0'}")
    return v


def _args(gamma, lam, sigma2, signal):
    return (_num(gamma, "gamma", positive=True), _num(lam, "lambda"), _num(sigma2, "sigma2"),
            _num(signal, "signal"))


def stieltjes_mp(gamma, lam):
    """(m(-lam), m'(-lam), sqrt D) of the Marchenko-Pastur law with ratio gamma, lam > 0
    (stable forms of the module docstring; call inside mp.workdps)."""
    b = 1 - gamma + lam
    sD = mp.sqrt(b * b + 4 * gamma * lam)
    m = 2 / (b + sD) if b >= 0 else (sD - b) / (2 * gamma * lam)
    return m, m * (1 + gamma * m) / sD, sD


def ridgeless_bias_variance(gamma, sigma2, signal=1):
    """(bias, variance) of the min-norm least squares limit (HMRT Eq. (8)); gamma = 1 refused."""
    g, _, s2, r2 = _args(gamma, 0, sigma2, signal)
    if g == 1:
        raise Refusal("precondition:gamma-not-one",
                      "gamma = 1: the ridgeless risk diverges at the interpolation threshold for "
                      "sigma^2 > 0 (HMRT Eq. (8) states the limits for gamma < 1 and gamma > 1 only)")
    with mp.workdps(_DPS):
        if g < 1:
            bias, var = mp.mpf(0), s2 * g / (1 - g)
        else:
            bias, var = r2 * (1 - 1 / g), s2 / (g - 1)
        return +(_SCALE["bias"] * bias), +(_SCALE["variance"] * var)


def ridgeless_test_error(gamma, sigma2, signal=1):
    """Limiting excess risk of min-norm least squares (HMRT Eq. (8)): mpf."""
    b, v = ridgeless_bias_variance(gamma, sigma2, signal)
    with mp.workdps(_DPS):
        return b + v


def ridge_bias_variance(gamma, lam, sigma2=0, signal=1):
    """(bias, variance) of the limiting ridge risk (HMRT Cor. 5, Eq. (43)); lam = 0 is the
    ridgeless limit."""
    g, lm, s2, r2 = _args(gamma, lam, sigma2, signal)
    if lm == 0:
        return ridgeless_bias_variance(g, s2, r2)
    with mp.workdps(_DPS):
        m, _, sD = stieltjes_mp(g, lm)
        bias = r2 * lm * lm * m * (1 + g * m) / sD
        var = s2 * g * (1 - lm * m) / sD
        return +(_SCALE["bias"] * bias), +(_SCALE["variance"] * var)


def ridge_test_error(gamma, lam, sigma2=0, signal=1):
    """Limiting excess risk R = bias + variance of ridge with penalty lam (HMRT Eq. (43)): mpf.
    The test MSE with a fresh noisy label is R + sigma2."""
    b, v = ridge_bias_variance(gamma, lam, sigma2, signal)
    with mp.workdps(_DPS):
        return b + v


def optimal_lambda(gamma, sigma2, signal=1):
    """lam* = sigma^2 gamma / r^2 (HMRT Cor. 5; DW Thm 2.1 lam* = gamma / alpha^2 at unit noise).
    sigma2 = 0 gives 0: the noiseless risk r^2 int lam^2/(s + lam)^2 dF(s) increases with lam, so
    its infimum is the ridgeless limit lam -> 0+ (no positive minimiser)."""
    g, _, s2, r2 = _args(gamma, 0, sigma2, signal)
    if r2 == 0:
        raise Refusal("precondition:signal", "signal = 0: lam* = sigma^2 gamma / r^2 is infinite "
                      "(the null estimator beta_hat = 0 is optimal)")
    with mp.workdps(_DPS):
        return s2 * g / r2


def optimal_test_error(gamma, sigma2, signal=1):
    """Limiting excess risk at lam*: R* = sigma^2 gamma m(-lam*) (HMRT Cor. 5); sigma2 = 0 -> the
    ridgeless limit (refused at gamma = 1)."""
    lam = optimal_lambda(gamma, sigma2, signal)
    if lam == 0:
        return ridgeless_test_error(gamma, sigma2, signal)
    g, _, s2, _ = _args(gamma, 0, sigma2, signal)
    with mp.workdps(_DPS):
        return s2 * g * stieltjes_mp(g, lam)[0]      # HMRT's own form: _SCALE does not enter


def argmin_lambda(gamma, sigma2, signal=1, lo="1e-6", hi="1e3", tol="1e-14"):
    """Numerical minimiser of ridge_test_error over lam in [lo, hi] by golden-section search in
    log lam (no formula for lam* used): the check of HMRT's lam* statement.  Refuses a minimiser
    at an end of the bracket (no interior minimum found)."""
    g, _, s2, r2 = _args(gamma, 0, sigma2, signal)
    with mp.workdps(_DPS):
        a, b = mp.log(mp.mpf(lo)), mp.log(mp.mpf(hi))
        f = lambda t: ridge_test_error(g, mp.exp(t), s2, r2)
        k = (mp.sqrt(5) - 1) / 2
        c, d = b - k * (b - a), a + k * (b - a)
        fc, fd = f(c), f(d)
        while b - a > mp.mpf(tol):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - k * (b - a)
                fc = f(c)
            else:
                a, c, fc = c, d, fd
                d = a + k * (b - a)
                fd = f(d)
        t = (a + b) / 2
        if t - mp.log(mp.mpf(lo)) < 1e-6 or mp.log(mp.mpf(hi)) - t < 1e-6:
            raise Refusal("no-solution:bracket", f"the risk minimum is at an end of [{lo}, {hi}]")
        return mp.exp(t)


__all__ = ["argmin_lambda", "optimal_lambda", "optimal_test_error", "ridge_bias_variance",
           "ridge_test_error", "ridgeless_bias_variance", "ridgeless_test_error", "stieltjes_mp"]
