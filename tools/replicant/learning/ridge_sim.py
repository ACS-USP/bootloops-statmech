"""Finite-n ridge / min-norm least squares by direct simulation (independent of the closed forms).

Per named seed (numpy default_rng(seed)): X in R^{n x p} with i.i.d. N(0, 1) entries,
beta = sqrt(r^2) g / |g| with g ~ N(0, I_p) (uniform direction, |beta|^2 = r^2 exactly: HMRT
Thm 1 takes |beta|^2 = r^2), y = X beta + sigma xi.  Fit by the thin SVD X = U diag(s) V^T:
    ridge  beta_hat = V diag(s / (s^2 + n lam)) U^T y   (= (X^T X + n lam I)^{-1} X^T y),
    lam = 0: beta_hat = V diag(1/s) U^T y over s > s_max max(n, p) eps (the min-norm fit).
Two finite-n risks, both from the SVD of the sampled X (no asymptotic formula):
  risk_x   HMRT's R_X(beta_hat; beta) = E[(x0 . (beta_hat - beta))^2 | X] (Sec. 2.1, the quantity
           their theorems state), averaged exactly over the training noise and x0 ~ N(0, I_p) by
           Lemma 1's identities: bias |beta - V f V^T beta|^2 with f = s^2 / (s^2 + n lam) (f = 1
           on the kept singular values at lam = 0) plus variance sigma^2 sum s^2 / (s^2 + n lam)^2
           (sigma^2 sum 1/s^2 at lam = 0);  ridge_simulate returns it;
  risk     |beta_hat - beta|^2 for the one sampled noise vector (the fit itself; an unbiased but
           much noisier estimate of R_X: its spread is dominated by the noise quadratic form and
           is right-skewed for gamma < 1, so the bands use risk_x).
Ensembles over named seeds give a 99% Student-t band (amp._stats.student_band): statistical,
never certified.  R_X concentrates at rate 1/n for gamma < 1 (a linear spectral statistic) while its
mean differs from the limit by O(1/n) (for Gaussian X and gamma < 1 exactly
E R_X = sigma^2 p / (n - p - 1), not sigma^2 gamma / (1 - gamma)), so a fixed-n band need not
contain the limit; extrapolate_in_n fits mean_n = R_inf + c / n over several n (weighted least
squares on the bands' standard errors) and returns the band of R_inf.
"""
from dataclasses import dataclass

import numpy as np

from ..amp._stats import student_band
from ..core import Refusal
from ..csp import _checks
from .ridge import _num


def _size(k, name):
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 10:
        raise Refusal("precondition:size", f"{name}={k!r}: need an int >= 10")
    return int(k)


@dataclass(frozen=True)
class RidgeFit:
    beta: np.ndarray
    beta_hat: np.ndarray
    risk: float                  # |beta_hat - beta|^2 (this noise draw)
    train_residual: float        # |y - X beta_hat|^2 / n
    bias_x: float                # HMRT's B_X: |E(beta_hat | X) - beta|^2
    variance_x: float            # HMRT's V_X: tr Cov(beta_hat | X)
    risk_x: float                # R_X = B_X + V_X


def ridge_fit(n, p, lam, sigma2, seed, signal=1):
    """One sampled problem and its ridge (lam > 0) or min-norm (lam = 0) fit by SVD."""
    n, p = _size(n, "n"), _size(p, "p")
    lam, s2, r2 = (float(_num(lam, "lambda")), float(_num(sigma2, "sigma2")),
                   float(_num(signal, "signal")))
    rng = np.random.default_rng(_checks.seed(seed))
    X = rng.standard_normal((n, p))
    g = rng.standard_normal(p)
    beta = np.sqrt(r2) * g / np.linalg.norm(g)
    y = X @ beta + np.sqrt(s2) * rng.standard_normal(n)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    if lam > 0:
        w = s / (s * s + n * lam)
        f = s * w
    else:
        keep = s > s[0] * max(n, p) * np.finfo(float).eps
        w = np.where(keep, 1 / np.where(keep, s, 1), 0.0)
        f = keep.astype(float)
    beta_hat = Vt.T @ (w * (U.T @ y))
    d = beta_hat - beta
    r = y - X @ beta_hat
    e = beta - Vt.T @ (f * (Vt @ beta))
    bias, var = float(e @ e), s2 * float(w @ w)
    return RidgeFit(beta, beta_hat, float(d @ d), float(r @ r) / n, bias, var, bias + var)


def ridge_simulate(n, p, lam, sigma2, seed, signal=1) -> float:
    """HMRT's finite-n risk R_X for one named seed (module docstring)."""
    return ridge_fit(n, p, lam, sigma2, seed, signal).risk_x


def ridge_ensemble(n, p, lam, sigma2, seeds, signal=1):
    """amp._stats.Band (99% Student-t) of ridge_simulate over the named seeds (>= 2)."""
    seeds = _checks.seeds(seeds)
    return student_band([ridge_simulate(n, p, lam, sigma2, s, signal) for s in seeds])


def extrapolate_in_n(bands, conf=0.99):
    """{n: Band} over >= 2 sizes -> Band of R_inf in mean_n = R_inf + c / n (weighted least
    squares, weights 1 / stderr^2; stderr of R_inf from the fit covariance).  The Student-t
    quantile uses df = min over sizes of (seeds - 1): conservative, the sizes are independent."""
    from ..amp._stats import Band, t_quantile
    if len(bands) < 2:
        raise Refusal("precondition:two-sizes", "extrapolation in 1/n needs at least two sizes n")
    ns = sorted(bands)
    A = np.array([[1.0, 1.0 / n] for n in ns])
    w = np.array([1 / bands[n].stderr ** 2 for n in ns])
    y = np.array([bands[n].mean for n in ns])
    cov = np.linalg.inv(A.T @ (w[:, None] * A))
    coef = cov @ (A.T @ (w * y))
    df = min(bands[n].df for n in ns)
    t = t_quantile((1 + conf) / 2, df)
    se = float(np.sqrt(cov[0, 0]))
    return Band(float(coef[0]), se, t * se, df, conf, t)


__all__ = ["RidgeFit", "extrapolate_in_n", "ridge_ensemble", "ridge_fit", "ridge_simulate"]
