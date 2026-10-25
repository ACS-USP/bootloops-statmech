"""AMP for the SK model (Bolthausen's TAP iteration with the Onsager term), finite N.

Model and conventions: H = -sum_{i<j} J_ij s_i s_j - h sum_i s_i, beta = 1/T; J is N x N,
symmetric, zero diagonal, off-diagonal variance 1/N (Gaussian N(0, 1/N), or +-1/sqrt(N)).
<a, b> = (1/N) sum_i a_i b_i.  The iteration (Bayati & Montanari 2011, eq. (4.1), with
f(x) = tanh(beta (h + x)) and lambda_t = <f'(x^t)> = beta (1 - <m^t, m^t>)):
    x^{t+1} = J m^t - beta (1 - Q_t) m^{t-1},   m^{t+1} = tanh(beta (h + x^{t+1})),
    Q_t = <m^t, m^t>,   m^{-1} = 0.
Multiplying by beta gives the TAP form m = tanh(beta h + beta J m - beta^2 (1 - q) m)
at a fixed point.  Bolthausen (Commun. Math. Phys. 325, 333 (2014), arXiv:1201.2891, Sec. 2)
uses the same scheme with the SE fixed point q in place of Q_t, h_B = beta h, and
m^(0) = 0, m^(1) = sqrt(q) 1 (init="bolthausen" here: that start, but the empirical Q_t
in the Onsager term as everywhere else).  `reaction=False` drops the Onsager
term (naive mean field, the positive control).

init="random" (default): m^0 = tanh(beta (h + g)), g ~ N(0, 1) iid from
numpy.random.default_rng([seed, 2]); sk_matrix draws J from default_rng([seed, 1]), so
the start is independent of J (required by the SE theorem).
damping d in [0, 1): m^{t+1} <- (1 - d) m^{t+1} + d m^t.  It keeps the fixed points but
the SE no longer describes the iterates (the Onsager coefficient is for d = 0).

Status and validity (stated in every result note): finite-N iterates are STATISTICAL,
never certified; a single run carries no error bar (sk_amp_ensemble gives a 99% Student-t
band over named seeds).  SE is a theorem for each fixed t as N -> oo for Gaussian J
(Bayati-Montanari Thm 4; Bolthausen for tanh started at the SE fixed point).  Bayati-
Montanari Thm 4 is stated for a GOE-type G with diagonal variance 2/N; the code uses a zero
diagonal (Bolthausen's g has zero diagonal): the diagonal is an O(N^-1/2) operator-norm
perturbation not literally covered by Thm 4 as stated.  Convergence of the iterates as
t -> oo is proved only on the RS-stable side of the AT line (beta^2 E cosh^-4 <= 1) with
h > 0 (Bolthausen Thm 2.1).  In the spin-glass phase (h = 0, beta > 1) the iterates need
not converge, and any fixed point need not describe the equilibrium measure.  +-J: not
covered by Bayati, Lelarge & Montanari (Ann. Appl. Probab. 25, 753 (2015), arXiv:1207.7321,
abstract read: polynomial nonlinearities); Chen & Lam (arXiv:2003.10431, abstract only;
entry assumptions not checked) claim universality for Lipschitz iterations with a
symmetric random matrix.  The +-J test is numerical evidence, not a theorem application.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from ..core import Refusal
from ._stats import Band, student_band
from .state_evolution import _check_beta_h, sk_state_evolution

LAWS = ("gaussian", "pm_one")
INITS = ("random", "bolthausen")
_VALIDITY = (
    "statistical (finite N, one matrix, one seed; not certified).  SE holds for each fixed t "
    "as N -> oo for Gaussian J (Bayati-Montanari 2011 Thm 4; Bolthausen 2014); convergence "
    "of the iterates is a theorem only on the RS-stable side of the AT line (beta^2 E "
    "cosh^-4 <= 1) with h > 0 (Bolthausen Thm 2.1); for h = 0, beta > 1 the iterates need "
    "not converge and a fixed point need not be the equilibrium state; +-J: not covered by "
    "Bayati-Lelarge-Montanari 2015 (polynomial nonlinearities); Chen-Lam arXiv:2003.10431 "
    "claim universality for Lipschitz iterations (abstract only, assumptions not checked); "
    "the +-J agreement is numerical evidence")
_onsager_scale = 1.0    # private: tests plant 0.9 / 1.1 to measure what the bands resolve


@dataclass
class AMPResult:
    m: np.ndarray
    q_trace: list            # Q_t = <m^t, m^t>, t = 0..iters
    converged: bool          # increment <m^T - m^{T-1}, same> < tol
    increments: list = field(default_factory=list)   # t = 1..iters
    beta: float = 0.0
    h: float = 0.0
    N: int = 0
    seed: int = 0
    iters: int = 0
    reaction: bool = True
    damping: object = None
    init: str = "random"
    label: str = "statistical"
    note: str = _VALIDITY


@dataclass
class AMPEnsemble:
    values: list             # Q_T per seed
    increments: list         # final increment per seed
    converged: list
    band: Band               # 99% Student-t band of the mean of `values`
    seeds: tuple
    law: str
    N: int
    iters: int
    reaction: bool
    label: str = "statistical"
    note: str = _VALIDITY


def sk_matrix(N, seed, law="gaussian"):
    """Symmetric N x N, zero diagonal, off-diagonal variance 1/N, from default_rng([seed, 1])."""
    if law not in LAWS:
        raise Refusal("precondition:known-law", f"law={law!r}; known: {', '.join(LAWS)}")
    if not isinstance(N, int) or N < 2:
        raise Refusal("precondition:size", f"N={N!r} must be an int >= 2")
    rng = np.random.default_rng([seed, 1])
    if law == "gaussian":
        A = rng.normal(size=(N, N)) / math.sqrt(N)
    else:
        A = rng.choice(np.array([-1.0, 1.0]), size=(N, N)) / math.sqrt(N)
    J = np.triu(A, 1)
    return J + J.T


def check_matrix(J):
    """Refuse anything that is not an SK coupling matrix in this normalisation."""
    J = np.asarray(J, dtype=float)
    if J.ndim != 2 or J.shape[0] != J.shape[1] or J.shape[0] < 2:
        raise Refusal("precondition:square-matrix", f"J has shape {J.shape}; need N x N, N >= 2")
    if not np.all(np.isfinite(J)):
        raise Refusal("precondition:finite-matrix", "J has non-finite entries")
    N = J.shape[0]
    scale = float(np.max(np.abs(J))) or 1.0
    asym = float(np.max(np.abs(J - J.T)))
    if asym > 1e-12 * scale:
        raise Refusal("precondition:symmetric-matrix",
                      f"max |J - J^T| = {asym:.3g}: AMP/SE here need a symmetric J")
    diag = float(np.max(np.abs(np.diag(J))))
    if diag > 1e-14:
        raise Refusal("precondition:zero-diagonal",
                      f"max |J_ii| = {diag:.3g}: the SK Hamiltonian has no self-coupling")
    M = N * (N - 1) / 2
    v = float(N * np.sum(np.triu(J, 1)**2) / M)
    tol = max(0.05, 6 * math.sqrt(2 / M))
    if abs(v - 1) > tol:
        raise Refusal("precondition:coupling-variance",
                      f"N * mean(J_ij^2) = {v:.4g}, not 1 within {tol:.3g}: couplings must "
                      f"have variance 1/N (Gaussian N(0, 1/N) or +-1/sqrt(N))")
    return J


def sk_amp(J, beta, *, iters, seed, damping=None, h=0.0, reaction=True, init="random",
           tol=1e-10) -> AMPResult:
    """Run `iters` AMP steps on J (see the module docstring for the iteration)."""
    J = check_matrix(J)
    b, hh = (float(v) for v in _check_beta_h(beta, h))
    if not isinstance(iters, int) or iters < 1:
        raise Refusal("precondition:positive-iters", f"iters={iters!r} is not a positive int")
    if damping is not None and not 0 <= float(damping) < 1:
        raise Refusal("precondition:damping-range", f"damping={damping} outside [0, 1)")
    if init not in INITS:
        raise Refusal("precondition:known-init",
                      f"init={init!r}; supported: 'random' (m^0 = tanh(beta(h + g)), g iid "
                      f"N(0, 1) from the seed, independent of J) and 'bolthausen' "
                      f"(m^0 = sqrt(q_SE) 1, h != 0); a J-dependent start breaks SE")
    N = J.shape[0]
    if init == "random":
        g = np.random.default_rng([seed, 2]).normal(size=N)
        m = np.tanh(b * (hh + g))
    else:
        if hh == 0:
            raise Refusal("precondition:bolthausen-init-field",
                          "Bolthausen's start sqrt(q) 1 is defined for h > 0 (at h = 0 the "
                          "SE fixed point is not unique); use init='random'")
        se = sk_state_evolution(b, hh)
        m = math.sqrt(float(se.q)) * np.sign(hh) * np.ones(N)
    m_old = np.zeros(N)
    qs, incs = [float(m @ m / N)], []
    for _ in range(iters):
        lam = _onsager_scale * b * (1 - qs[-1]) if reaction else 0.0
        x = J @ m - lam * m_old
        new = np.tanh(b * (hh + x))
        if damping:
            new = (1 - damping) * new + damping * m
        m_old, m = m, new
        qs.append(float(m @ m / N))
        incs.append(float((m - m_old) @ (m - m_old) / N))
    note = _VALIDITY + ("" if reaction else "; reaction=False: NAIVE mean field (no Onsager "
                        "term), the positive control, not AMP") + (
        f"; damping={damping}: SE does not apply to damped iterates" if damping else "")
    return AMPResult(m, qs, bool(incs[-1] < tol), incs, b, hh, N, seed, iters, reaction,
                     damping, init, note=note)


def sk_amp_ensemble(N, beta, h=0.0, *, seeds, iters, law="gaussian", reaction=True,
                    tol=1e-10) -> AMPEnsemble:
    """One fresh matrix and start per named seed; 99% Student-t band of Q_iters."""
    seeds = tuple(seeds)
    runs = [sk_amp(sk_matrix(N, s, law), beta, h=h, iters=iters, seed=s, reaction=reaction,
                   tol=tol) for s in seeds]
    vals = [r.q_trace[-1] for r in runs]
    note = runs[0].note + f"; seeds {list(seeds)}, law {law}, N = {N}, t = {iters}"
    return AMPEnsemble(vals, [r.increments[-1] for r in runs], [r.converged for r in runs],
                       student_band(vals), seeds, law, N, iters, reaction, note=note)
