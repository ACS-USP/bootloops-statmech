"""GAMP for the teacher-student perceptron (Bayes-optimal, noiseless sign channel), finite N.

Model and conventions (Barbier, Krzakala, Macris, Miolane & Zdeborova, PNAS 116, 5451
(2019), arXiv:1708.03395, "BKMMZ", Sec. 2): teacher w* in R^N with i.i.d. components
drawn from the prior P0 (gaussian N(0, 1) or ising +-1; rho = E w*^2 = 1); patterns
Phi (M x N, M = alpha N) with i.i.d. N(0, 1) entries; labels y = sign(Phi w*/sqrt N).
The posterior uses the same P0 and channel (Bayes-optimal: the Nishimori line).
<u> is the average over components.  GAMP (Rangan, ISIT 2011, arXiv:1010.5141; the scalar-
variance form displayed in BKMMZ Sec. 2.4), x_hat^0 = E P0 = 0, v^0 = rho = 1, g^0 = 0:
    V^t     = <v^{t-1}>
    omega^t = Phi x_hat^{t-1}/sqrt N - V^t g^{t-1}                 (Onsager term)
    g^t     = g_out(y, omega^t, V^t),   g_out = d/d omega ln H(-y omega/sqrt V)
            = y phi(u)/(sqrt V H(-u)),  u = y omega/sqrt V
    lambda^t = alpha <(g^t)^2>
    R^t     = x_hat^{t-1} + Phi^T g^t/(sqrt N lambda^t)
    x_hat^t = mean, v^t = variance of P0(x) exp(-lambda^t (R^t - x)^2/2):
            gaussian: lambda R/(1 + lambda), 1/(1 + lambda);  ising: tanh(lambda R), 1 - x_hat^2.
g_out is the omega-derivative of ln Z_out (Rangan's form), so V g is dimensionally the
correction to omega.  BKMMZ's displayed g_Pout is instead the mean of x under
P_out(y | omega + sqrt(V) x) exp(-x^2/2), i.e. sqrt(V) d/d omega ln Z_out, so their
displayed omega and lambda lines differ from these by powers of sqrt(V); the code follows
the dimensionally consistent Rangan form.  lambda uses <g^2> (as BKMMZ); its
Nishimori-equivalent form -alpha <d g/d omega> is not used.  Overlap q_t = x_hat^t . w*/N
(needs the teacher), q_0 = 0; the SE of amp/gamp_se.py predicts q_t at each t.

Status (stated in every result note): finite-N iterates are STATISTICAL, never
certified; one run has no error bar (gamp_ensemble: 99% Student-t band over named seeds).
GAMP SE for GLMs is stated as Claim 1 in BKMMZ (from Rangan 2011, steps of the proof
missing; Javanmard & Montanari 2013 state SE for general AMP covering GAMP, not counted as
a proof by BKMMZ).  Bayes-optimal only on the Nishimori line: with a teacher from
another law (e.g. a +-1 teacher under the gaussian prior, or vice versa) the iteration
still runs but is not Bayes-optimal and the Bayes-optimal SE does not describe it.
The ising prior has a hard phase (alpha_IT, alpha_AMP) where GAMP's fixed point is not
the Bayes-optimal one (gamp_se.ising_thresholds).
If <v> falls below 1e-6, or <g^2> underflows to 0 (ising: the marginals collapsed on
+-1 values, e.g. perfect recovery), the iterate is kept fixed and the note says so: in
the noiseless channel at finite N no label is then marginal on the scale sqrt(V), so
lambda = alpha <g^2> underflows and the next step would revert x_hat to the prior mean
(a float64 breakdown, measured at N = 800: V = 1e-8, lambda = 4e-85).
The init is deterministic (x_hat^0 = 0): `seed` names the instance in the result and is
the seed of perceptron_instance in gamp_ensemble; it does not enter the iteration.
"""
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import log_ndtr

from ..core import Refusal
from ._stats import Band, student_band
from .gamp_se import PRIORS, _prior

_VALIDITY = (
    "statistical (finite N, one instance; not certified).  Bayes-optimal only on the "
    "Nishimori line (teacher drawn from the same prior, matched noiseless sign channel); "
    "GAMP state evolution for GLMs is BKMMZ Claim 1 (from Rangan 2011), not a theorem "
    "applied here")
_onsager_scale = 1.0    # private: tests plant 0.8 / 1.2 to measure what the bands resolve
_SQ2PI = math.sqrt(2 * math.pi)
_V_FLOOR = 1e-6


@dataclass
class GAMPResult:
    x_hat: np.ndarray
    v: np.ndarray
    overlap_trace: list      # q_t = x_hat^t . w*/N, t = 0..iters (None without a teacher)
    norm_trace: list         # x_hat^t . x_hat^t / N
    converged: bool          # <(x_hat^T - x_hat^{T-1})^2> < tol
    increments: list = field(default_factory=list)
    alpha: float = 0.0
    N: int = 0
    prior: str = "gaussian"
    seed: int = 0
    iters: int = 0
    label: str = "statistical"
    note: str = _VALIDITY


@dataclass
class GAMPEnsemble:
    values: list             # q_T per seed
    eps: list                # plug-in generalization error arccos(cos(x_hat, w*))/pi per seed
    converged: list
    band: Band               # 99% Student-t band of the mean overlap
    eps_band: Band
    seeds: tuple
    N: int
    alpha: float
    prior: str
    iters: int
    label: str = "statistical"
    note: str = _VALIDITY


def perceptron_instance(N, alpha, prior, seed):
    """(Phi, y, w*) from numpy.random.default_rng([seed, 3]): M = round(alpha N)."""
    _prior(prior)
    if not isinstance(N, int) or N < 2:
        raise Refusal("precondition:size", f"N={N!r} must be an int >= 2")
    a = float(alpha)
    M = int(round(a * N))
    if not (a > 0 and M >= 1):
        raise Refusal("precondition:positive-alpha", f"alpha={alpha} gives M = {M} patterns")
    rng = np.random.default_rng([seed, 3])
    w = rng.normal(size=N) if prior == "gaussian" else rng.choice(np.array([-1.0, 1.0]), N)
    X = rng.normal(size=(M, N))
    y = np.where(X @ w >= 0, 1.0, -1.0)
    return X, y, w


def _check(X, y, teacher):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or min(X.shape) < 1 or not np.all(np.isfinite(X)):
        raise Refusal("precondition:pattern-matrix",
                      f"Phi has shape {X.shape}; need a finite M x N matrix")
    y = np.asarray(y, dtype=float)
    if y.shape != (X.shape[0],):
        raise Refusal("precondition:label-count", f"{y.shape} labels for {X.shape[0]} patterns")
    if not np.all(np.abs(y) == 1):
        raise Refusal("precondition:sign-labels", "labels must be +-1 (noiseless sign channel)")
    if teacher is not None:
        teacher = np.asarray(teacher, dtype=float)
        if teacher.shape != (X.shape[1],):
            raise Refusal("precondition:teacher-length",
                          f"teacher has shape {teacher.shape}; need ({X.shape[1]},)")
    return X, y, teacher


def _gout(y, omega, V):
    """(g_out, u): d/d omega ln H(-y omega/sqrt V), with phi/H via log_ndtr (no underflow)."""
    u = y * omega / math.sqrt(V)
    return y * np.exp(-u * u / 2 - log_ndtr(u)) / (_SQ2PI * math.sqrt(V)), u


def gamp_perceptron(X, y, prior, *, iters, seed, teacher=None, tol=1e-10) -> GAMPResult:
    """Run `iters` GAMP steps (module docstring) on patterns X (M x N), labels y."""
    prior = _prior(prior)
    X, y, teacher = _check(X, y, teacher)
    if not isinstance(iters, int) or iters < 1:
        raise Refusal("precondition:positive-iters", f"iters={iters!r} is not a positive int")
    M, N = X.shape
    a, sq = M / N, math.sqrt(N)
    xh, v, g = np.zeros(N), np.ones(N), np.zeros(M)
    ov = [0.0] if teacher is not None else None
    norms, incs, frozen = [0.0], [], False
    for _ in range(iters):
        V = float(v.mean())
        if V >= _V_FLOOR:
            omega = X @ xh / sq - _onsager_scale * V * g
            g_new, _ = _gout(y, omega, V)
            lam = a * float(g_new @ g_new) / M
        if V < _V_FLOOR or not (math.isfinite(lam) and lam > 1e-300):
            frozen = True             # marginals collapsed: a fixed point to float64
            incs.append(0.0)
            norms.append(norms[-1])
            if teacher is not None:
                ov.append(ov[-1])
            continue
        g = g_new
        R = xh + X.T @ g / (sq * lam)
        if prior == "gaussian":
            new, v = lam * R / (1 + lam), np.full(N, 1 / (1 + lam))
        else:
            new = np.tanh(lam * R)
            v = 1 - new**2
        incs.append(float((new - xh) @ (new - xh) / N))
        xh = new
        norms.append(float(xh @ xh / N))
        if teacher is not None:
            ov.append(float(xh @ teacher / N))
    note = _VALIDITY + ("; the variances collapsed (<v> < 1e-6 or <g^2> underflow): the "
                        "iterate was frozen as a float64 fixed point" if frozen else "")
    return GAMPResult(xh, v, ov, norms, bool(incs[-1] < tol), incs, a, N, prior, seed, iters,
                      note=note)


def gamp_ensemble(N, alpha, prior="gaussian", *, seeds, iters) -> GAMPEnsemble:
    """One fresh instance per named seed (perceptron_instance); 99% Student-t bands of the
    final overlap and of the plug-in generalization error."""
    seeds = tuple(seeds)
    vals, eps, conv = [], [], []
    for s in seeds:
        X, y, w = perceptron_instance(N, alpha, prior, s)
        r = gamp_perceptron(X, y, prior, iters=iters, seed=s, teacher=w)
        vals.append(r.overlap_trace[-1])
        nx = math.sqrt(r.norm_trace[-1] * N) * float(np.linalg.norm(w))
        c = 0.0 if nx == 0 else max(-1.0, min(1.0, r.overlap_trace[-1] * N / nx))
        eps.append(math.acos(c) / math.pi)
        conv.append(r.converged)
    note = _VALIDITY + f"; seeds {list(seeds)}, prior {prior}, N = {N}, alpha = {alpha}, " \
        f"t = {iters}"
    return GAMPEnsemble(vals, eps, conv, student_band(vals), student_band(eps), seeds, N,
                        float(alpha), prior, iters, note=note)


__all__ = ["GAMPEnsemble", "GAMPResult", "PRIORS", "gamp_ensemble", "gamp_perceptron",
           "perceptron_instance"]
