"""Rank-one spiked Wigner model: Bayes-optimal state evolution, the RS potential, the threshold,
and finite-N Bayes AMP with spectral initialisation.

Model (Montanari & Venkataramanan, Ann. Statist. 49, 321 (2021), arXiv:1711.01682, Eq. (2.1)):
    Y = (lambda / N) x x^T + W,   W ~ GOE(N): W_ij ~ N(0, 1/N) for i < j, W_ii ~ N(0, 2/N),
x_i i.i.d. from a zero-mean, unit-variance prior (PRIORS: gaussian N(0, 1), rademacher +-1).
The top eigenvalue of Y separates from the semicircle edge 2 iff lambda > 1 (to lambda + 1/lambda;
Feral-Peche 2007, Benaych-Georges-Nadakuditi 2011, as cited by Lelarge-Miolane, not fetched).
Lelarge & Miolane (Probab. Theory Relat. Fields 173, 859 (2019), arXiv:1611.03888) and Deshpande,
Abbe & Montanari (arXiv:1507.08685) write Y = sqrt(lambda_LM / n) X X^T + Z, Z of unit variance:
lambda_LM = lambda^2.  Barbier et al. (arXiv:1606.04142) write w = s s^T / sqrt(n) + sqrt(Delta) z:
Delta = 1 / lambda^2.

Scalar channel y = gamma X + sqrt(gamma) G (MV Eq. (2.26)), denoiser F(y; gamma) = E[X | y]
(the PRIORS hook: gaussian y / (1 + gamma), rademacher tanh(y)); mmse(gamma) = E (X - F)^2 by
Gauss-Hermite quadrature over G (and over X for the gaussian prior).  State evolution (MV (2.24)):
    gamma_{t+1} = lambda^2 (1 - mmse(gamma_t)),
whose fixed points are found here by bracketing the roots of lambda^2 (1 - mmse(gamma)) - gamma on
(0, lambda^2] (gamma = 0 is always one).  At the largest fixed point gamma* (the limit from the
spectral start gamma_0 = lambda^2 - 1 and from the informative start lambda^2):
    q = gamma*/lambda^2   (LM's q*: the overlap x.X of a posterior sample, Nishimori),
    overlap = sqrt(q)     (cosine |<xhat, x>| / (|xhat| |x|), MV Eq. (2.28)),
    mmse = 1 - q          (vector MSE up to the global sign, MV Eq. (2.29)),
    matrix_mmse = 1 - q^2 (LM Theorem 1; for the gaussian prior LM Eq. (10)).
`bayes` is True when gamma* is stationary for, and the lowest stationary value of, the RS potential
    Psi(gamma) = lambda^2/4 + gamma^2/(4 lambda^2) - gamma/2 + I(gamma)
(DAM Eq. (9) with lambda_DAM = lambda^2, minimised: DAM Lemma 3.3; MV Prop. 2.2 says "maximizes",
which contradicts DAM's min and the sign of Psi'' at 0: we follow DAM), with I(gamma) the scalar
mutual information from its cited closed form (gaussian 1/2 log(1 + gamma), LM Sec. 2.3; rademacher
gamma - E log cosh(gamma + sqrt(gamma) Z), DAM Eq. (6)) by QUADPACK, independent of the hook.

AMP (MV Eqs. (2.20)-(2.21), Bayes AMP): x^0 = sqrt(N lambda^2 (lambda^2 - 1)) phi_1,
    x^{t+1} = Y f_t(x^t) - b_t f_{t-1}(x^{t-1}),  f_t = lambda F(.; gamma_t),  b_t = <f_t'(x^t)>,
f_{-1} = 0, estimate xhat^t = F(x^t; gamma_t).  MV Theorem 2 takes gamma_t from SE; here gamma_t is
estimated from the iterate (|x^t|^2 / N -> gamma_t^2 + gamma_t), so AMP does not call this
module's SE: a consistent plug-in, not literally the theorem's algorithm.  `reaction=False` drops
the Onsager term (non-linear power iteration, the positive control).  Finite-N values are
statistical (99% Student-t bands over named seeds), never certified.
"""
import math
from dataclasses import dataclass

import mpmath as mp
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy import integrate, optimize

from ..amp._stats import Band, student_band
from ..core import Refusal
from ..csp import _checks

# G-quadrature: trapezoid on [-10, 10], h = 0.01 (exponentially accurate for these analytic
# integrands: the poles of tanh(gamma + sqrt(gamma) z) sit at |Im z| = pi / (2 sqrt(gamma)); a
# Gauss-Hermite rule with 101 nodes was off by 3e-7 at gamma = 3.7, by 2.7e-6 at gamma = 9)
_Z = np.linspace(-10.0, 10.0, 2001)
_WZ = np.exp(-_Z**2 / 2) / math.sqrt(2 * math.pi) * (_Z[1] - _Z[0])
_onsager_scale = 1.0    # private: tests plant 0.5 to show the band resolves the Onsager term
_SPECTRAL = ("AMP needs lambda > 1: the spectral start of MV Theorem 2 exists only above the "
             "spectral threshold (below it the Bayes estimate is trivial, q = 0)")


@dataclass(frozen=True)
class Prior:
    """Zero-mean, unit-variance prior: X-quadrature (nodes, weights), the posterior-mean denoiser
    F(y, gamma) for y = gamma X + sqrt(gamma) G and its y-derivative dF, sample(rng, n)."""
    name: str
    nodes: np.ndarray
    weights: np.ndarray
    F: object
    dF: object
    sample: object


def _gauss_prior():
    x, w = hermegauss(21)
    return Prior("gaussian", x, w / math.sqrt(2 * math.pi), lambda y, g: y / (1 + g),
                 lambda y, g: np.full_like(y, 1 / (1 + g)), lambda rng, n: rng.normal(size=n))


def _rademacher_prior():
    return Prior("rademacher", np.array([-1.0, 1.0]), np.array([0.5, 0.5]), lambda y, g: np.tanh(y),
                 lambda y, g: 1 - np.tanh(y)**2,
                 lambda rng, n: rng.choice(np.array([-1.0, 1.0]), size=n))


PRIORS = {"gaussian": _gauss_prior(), "rademacher": _rademacher_prior()}


def _prior(name):
    if not isinstance(name, str) or name not in PRIORS:
        raise Refusal("precondition:known-prior",
                      f"prior={name!r}; known: {', '.join(PRIORS)} (zero mean, unit variance)")
    return PRIORS[name]


def _snr(x):
    if isinstance(x, bool) or x is None:
        raise Refusal("precondition:snr", f"snr={x!r}: lambda must be a number >= 0")
    try:
        v = mp.mpf(x)
    except (TypeError, ValueError):
        raise Refusal("precondition:snr", f"snr={x!r}: lambda must be a number >= 0") from None
    if not mp.isfinite(v) or v < 0:
        raise Refusal("precondition:snr", f"snr={x!r}: lambda must be a finite number >= 0")
    return v


def scalar_mmse(prior, gamma):
    """E (X - F(gamma X + sqrt(gamma) G; gamma))^2 by quadrature over X and G (binary64)."""
    P = _prior(prior)
    g = float(gamma)
    y = g * P.nodes[:, None] + math.sqrt(g) * _Z[None, :]
    err = (P.nodes[:, None] - P.F(y, g))**2
    return float(P.weights @ err @ _WZ)


def _mutual_info(prior, g):
    if prior == "gaussian":
        return 0.5 * math.log1p(g)
    if g == 0:
        return 0.0
    s = math.sqrt(g)

    def f(z):
        a = abs(g + s * z)
        return math.exp(-z * z / 2) * (a + math.log1p(math.exp(-2 * a)) - math.log(2))
    v, _ = integrate.quad(f, -math.inf, math.inf, epsabs=1e-15, epsrel=1e-13, limit=200)
    return g - v / math.sqrt(2 * math.pi)


def spiked_potential(snr, gamma, prior="gaussian"):
    """RS potential Psi(gamma; lambda) (DAM Eq. (9), lambda_DAM = lambda^2), minimised over gamma."""
    lam = float(_snr(snr))
    _prior(prior)
    g = float(gamma)
    if lam == 0:
        raise Refusal("precondition:snr", "the potential needs lambda > 0")
    return mp.mpf(lam**2 / 4 + g**2 / (4 * lam**2) - g / 2 + _mutual_info(prior, g))


@dataclass(frozen=True)
class SEResult:
    snr: object
    prior: str
    gamma: float
    q: float
    overlap: float
    mmse: float
    matrix_mmse: float
    fixed_points: tuple
    bayes: bool
    potential: float
    note: str = ""


def _roots(lam2, prior):
    g = lambda x: lam2 * (1 - scalar_mmse(prior, x)) - x
    grid = np.concatenate([lam2 * np.geomspace(1e-9, 1e-2, 15), lam2 * np.linspace(1e-2, 1, 40)[1:]])
    vals = [g(x) for x in grid]
    out = [0.0]
    for (x0, v0), (x1, v1) in zip(zip(grid, vals), zip(grid[1:], vals[1:])):
        if v0 == 0:
            out.append(float(x0))
        elif v0 * v1 < 0:
            out.append(optimize.brentq(g, x0, x1, xtol=1e-15, rtol=1e-15, maxiter=200))
    return tuple(out)


def spiked_state_evolution(snr, prior="gaussian") -> SEResult:
    """Fixed points of MV Eq. (2.24); reports the largest (see the module docstring)."""
    lam = _snr(snr)
    _prior(prior)
    L2 = float(lam)**2
    if L2 == 0:
        return SEResult(lam, prior, 0.0, 0.0, 0.0, 1.0, 1.0, (0,), True, 0.0, "lambda = 0: no signal")
    roots = _roots(L2, prior)
    if len(roots) > 2:
        raise Refusal("ambiguous:several-fixed-points",
                      f"lambda = {mp.nstr(lam, 8)}, prior {prior}: nontrivial SE fixed points "
                      f"{[f'{r:.10g}' for r in roots[1:]]}; this module reports only the unique one")
    gs = roots[-1]
    psi = {r: float(spiked_potential(lam, r, prior)) for r in roots}
    h = 1e-5 * max(gs, 1e-3)
    stationary = gs == 0 or abs(float(spiked_potential(lam, gs + h, prior)) -
                                float(spiked_potential(lam, gs - h, prior))) / (2 * h) < 1e-6
    bayes = bool(stationary and psi[gs] <= min(psi.values()) + 1e-13)
    q = gs / L2
    note = (f"largest SE fixed point gamma* = {gs:.12g} of {len(roots)} (gamma = 0 included); "
            f"{'' if bayes else 'NOT '}the global minimum of the RS potential (DAM Eq. (9))")
    roots = tuple(0 if r == 0 else r for r in roots)
    return SEResult(lam, prior, gs, q, math.sqrt(q), 1 - q, 1 - q * q, roots, bayes, psi[gs], note)


def spiked_threshold(prior="gaussian"):
    """lambda_c = (-mmse'(0))^(-1/2): where gamma = 0 stops being a stable SE fixed point
    (slope lambda^2 (-mmse'(0)) = 1), by Richardson extrapolation of (mmse(0) - mmse(h))/h."""
    _prior(prior)
    m0 = scalar_mmse(prior, 0.0)
    D = lambda h: (m0 - scalar_mmse(prior, h)) / h
    hs = (1e-4, 5e-5, 2.5e-5)
    r1 = [2 * D(hs[i + 1]) - D(hs[i]) for i in range(2)]
    slope = (4 * r1[1] - r1[0]) / 3
    return mp.mpf(1 / math.sqrt(slope))


# ---- finite N --------------------------------------------------------------------------------
def spiked_matrix(N, snr, prior, seed):
    """(Y, x): W ~ GOE(N) from default_rng([seed, 1]), x ~ prior from default_rng([seed, 2])."""
    if isinstance(N, bool) or not isinstance(N, int) or N < 10:
        raise Refusal("precondition:size", f"N={N!r} must be an int >= 10")
    seed = _checks.seed(seed)
    lam = float(_snr(snr))
    P = _prior(prior)
    A = np.random.default_rng([seed, 1]).normal(size=(N, N)) / math.sqrt(N)
    W = (A + A.T) / math.sqrt(2)
    x = P.sample(np.random.default_rng([seed, 2]), N)
    return lam / N * np.outer(x, x) + W, x


@dataclass
class AMPResult:
    xhat: np.ndarray
    x: np.ndarray
    gamma_trace: list            # empirical gamma_t, t = 0..iters
    increments: list             # <xhat^t - xhat^{t-1}, same>, t = 1..iters
    converged: bool
    snr: float = 0.0
    prior: str = ""
    seed: int = 0
    reaction: bool = True
    label: str = "statistical"


def _check_Y(Y):
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2 or Y.shape[0] != Y.shape[1] or Y.shape[0] < 10:
        raise Refusal("precondition:square-matrix", f"Y has shape {Y.shape}; need N x N, N >= 10")
    if not np.all(np.isfinite(Y)):
        raise Refusal("precondition:finite-matrix", "Y has non-finite entries")
    if float(np.max(np.abs(Y - Y.T))) > 1e-12 * (float(np.max(np.abs(Y))) or 1.0):
        raise Refusal("precondition:symmetric-matrix", "the spiked Wigner Y must be symmetric")
    return Y


def _gamma_hat(x):
    return (-1 + math.sqrt(1 + 4 * float(x @ x) / x.size)) / 2


def spiked_amp(Y, prior, *, snr, iters, seed, init="spectral", reaction=True, tol=1e-12) -> AMPResult:
    """Bayes AMP on Y (module docstring); `seed` seeds the eigensolver's start vector."""
    from scipy.sparse.linalg import eigsh
    Y = _check_Y(Y)
    P = _prior(prior)
    lam = float(_snr(snr))
    seed = _checks.seed(seed)
    if isinstance(iters, bool) or not isinstance(iters, int) or iters < 1:
        raise Refusal("precondition:positive-iters", f"iters={iters!r} is not a positive int")
    if init != "spectral":
        raise Refusal("precondition:known-init", f"init={init!r}; supported: 'spectral' "
                      "(MV Eq. (2.20)); a start independent of Y has zero overlap for these "
                      "zero-mean priors, so SE would stay at gamma = 0")
    if lam <= 1:
        raise Refusal("precondition:snr-above-one", f"lambda = {lam}: {_SPECTRAL}")
    N = Y.shape[0]
    v0 = np.random.default_rng([seed, 3]).normal(size=N)
    _, vec = eigsh(Y, k=1, which="LA", v0=v0, tol=1e-12)
    x = math.sqrt(N * lam**2 * (lam**2 - 1)) * vec[:, 0]
    f_old = np.zeros(N)
    gam = _gamma_hat(x)
    gammas, incs, xhat = [gam], [], P.F(x, gam)
    for _ in range(iters):
        f = lam * P.F(x, gam)
        b = lam * float(np.mean(P.dF(x, gam))) if reaction else 0.0
        x = Y @ f - _onsager_scale * b * f_old
        f_old = f
        gam = _gamma_hat(x)
        new = P.F(x, gam)
        incs.append(float((new - xhat) @ (new - xhat) / N))
        xhat = new
        gammas.append(gam)
    return AMPResult(xhat, x, gammas, incs, bool(incs[-1] < tol), lam, prior, seed, reaction)


def amp_metrics(xhat, x0):
    """Sign-free accuracy of xhat against the truth: q = |<xhat, x0>|/N, cosine, the vector MSE
    min_s |s xhat - x0|^2 / N and the matrix MSE |x0 x0^T - xhat xhat^T|_F^2 / N^2."""
    N = x0.size
    c, a, b = abs(float(xhat @ x0)) / N, float(xhat @ xhat) / N, float(x0 @ x0) / N
    return {"q": c, "overlap": c / math.sqrt(a * b) if a > 0 else 0.0, "mse": a + b - 2 * c,
            "matrix_mse": b * b - 2 * c * c + a * a}


@dataclass
class AMPEnsemble:
    q: list
    overlap: list
    mse: list
    matrix_mse: list
    converged: list
    bands: dict                  # key -> amp._stats.Band (99% Student-t)
    seeds: tuple
    N: int
    snr: float
    prior: str
    iters: int
    reaction: bool
    label: str = "statistical"
    note: str = ""


def spiked_amp_ensemble(N, snr, prior, *, seeds, iters, reaction=True) -> AMPEnsemble:
    """One fresh (W, x) per named seed; 99% Student-t bands of the sign-free metrics."""
    seeds = _checks.seeds(seeds)
    lam = float(_snr(snr))
    _prior(prior)
    if lam <= 1:
        raise Refusal("precondition:snr-above-one", f"lambda = {lam}: {_SPECTRAL}")
    runs = []
    for s in seeds:
        Y, x0 = spiked_matrix(N, snr, prior, s)
        r = spiked_amp(Y, prior, snr=snr, iters=iters, seed=s, reaction=reaction)
        runs.append((amp_metrics(r.xhat, x0), r.converged))
    cols = {k: [m[k] for m, _ in runs] for k in ("q", "overlap", "mse", "matrix_mse")}
    bands = {k: student_band(v) for k, v in cols.items()}
    note = (f"statistical: Bayes AMP (spectral start, empirical gamma_t) on {len(seeds)} matrices, "
            f"N = {N}, t = {iters}, seeds {list(seeds)}; 99% Student-t bands; MV Theorem 2 is the "
            "N -> oo statement at fixed t (with SE gamma_t)" +
            ("" if reaction else "; reaction=False: NO Onsager term (power iteration), the control"))
    return AMPEnsemble(cols["q"], cols["overlap"], cols["mse"], cols["matrix_mse"],
                       [c for _, c in runs], bands, seeds, N, lam, prior, iters, reaction, note=note)


__all__ = ["AMPEnsemble", "AMPResult", "Band", "PRIORS", "Prior", "SEResult", "amp_metrics",
           "scalar_mmse", "spiked_amp", "spiked_amp_ensemble", "spiked_matrix", "spiked_potential",
           "spiked_state_evolution", "spiked_threshold"]
