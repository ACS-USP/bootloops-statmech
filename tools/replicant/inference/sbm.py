"""Symmetric two-group stochastic block model: the Kesten-Stigum line, a sampler, and belief
propagation with the external field (Decelle, Krzakala, Moore & Zdeborova, Phys. Rev. E 84, 066106
(2011), arXiv:1109.3041; DKMZ).

Convention (Mossel, Neeman & Sly 2015, arXiv:1202.1499, Def. 1.1 with p = a/n, q = b/n; Massoulie
2014, arXiv:1311.3085, Sec. 1.2): labels sigma_i in {+1, -1} i.i.d. uniform; each pair {i, j} is
an edge independently with probability a/N if sigma_i = sigma_j, b/N otherwise.  In DKMZ's
notation q = 2, n_a = 1/2, c_in = a, c_out = b, mean degree c = (a + b)/2.  Any a, b >= 0 with
a + b > 0 is accepted (a < b is the disassortative case; the threshold condition is symmetric).
Kesten-Stigum: tau = (a - b)^2 / (2 (a + b)) > 1 (DKMZ Eq. (44), |c_in - c_out| > q sqrt(c)).
What is proved: for a + b > 2 and tau <= 1 no estimator is correlated with the partition (MNS 2015,
Theorem 2.1, which includes tau = 1); for tau > 1 an efficient estimator is (Massoulie 2014,
Theorem 1.1; Mossel, Neeman & Sly, arXiv:1311.4115, Theorem 2.1).  Neither proof analyses BP from
random messages (MNS arXiv:1311.4115, Sec. 2.1: "a difficult task").

BP (DKMZ Eqs. (26)-(28)) for q = 2 in log-ratio form, message p = psi^{i->j}_+:
    L_{i->j} = -dh + sum_{k in di \\ j} u(p_{k->i}),   u(p) = log[(a p + b (1-p)) / (b p + a (1-p))],
    dh = h_+ - h_- = (a - b)/N sum_k (2 psi^k_+ - 1)        (the field (27) from non-edges),
marginals L_i = -dh + sum_{k in di} u(p_{k->i}).  Updates are synchronous (DKMZ update in random
sequential order) with dh solved each sweep to be consistent with the marginals, messages start i.i.d. uniform on [0, 1] (DKMZ "random initial condition"),
optional damping, messages clipped to [1e-12, 1 - 1e-12] inside u.  Estimate sigmahat_i = sign(L_i)
(ties broken by the seed); overlap Q = |(1/N) sum_i sigma_i sigmahat_i| = DKMZ Eq. (5) with
n_a = 1/2 maximised over the global flip.  `factorized` when every |psi^i_+ - 1/2| < 1e-6 (DKMZ's
factorized fixed point, Q = 0 up to the random tie-break).  Finite-N values are statistical.
"""
import math
from dataclasses import dataclass

import mpmath as mp
import numpy as np

from ..amp._stats import Band, student_band
from ..core import Refusal
from ..csp import _checks

_field_scale = 1.0      # private: tests plant 0 (field dropped) to show dkmz_residual detects it
_CLIP = 1e-12


def _deg(a, b):
    out = []
    for x in (a, b):
        if isinstance(x, bool) or x is None:
            raise Refusal("precondition:degree-parameters",
                          f"a, b = {a!r}, {b!r}: need numbers a, b >= 0 with a + b > 0")
        try:
            v = mp.mpf(x)
        except (TypeError, ValueError):
            raise Refusal("precondition:degree-parameters",
                          f"a, b = {a!r}, {b!r}: need numbers a, b >= 0 with a + b > 0") from None
        if not mp.isfinite(v) or v < 0:
            raise Refusal("precondition:degree-parameters",
                          f"a, b = {a!r}, {b!r}: need finite a, b >= 0 with a + b > 0")
        out.append(v)
    if out[0] + out[1] == 0:
        raise Refusal("precondition:degree-parameters", "a + b = 0: the graph has no edges")
    return out


@dataclass(frozen=True)
class KSResult:
    a: object
    b: object
    mean_degree: object
    margin: object            # tau = (a - b)^2 / (2 (a + b)); detectable iff tau > 1
    detectable: bool
    impossibility_proved: bool   # tau <= 1 is covered by MNS 2015 Thm 2.1 (needs a + b > 2)


def sbm_kesten_stigum(a, b, q=2) -> KSResult:
    if q != 2 or isinstance(q, bool):
        raise Refusal("unsupported:q", f"q = {q!r}: only q = 2 symmetric groups are implemented")
    a, b = _deg(a, b)
    with mp.workdps(30):
        tau = (a - b)**2 / (2 * (a + b))
        c = (a + b) / 2
    return KSResult(a, b, c, tau, bool(tau > 1), bool(tau <= 1 and a + b > 2))


@dataclass(frozen=True)
class KSLine:
    c: object
    a: object
    b: object
    ratio: object             # a / b on the line = 1 / eps_c in DKMZ's notation


def sbm_ks_line(c):
    """The assortative point (a, b) on the KS line with mean degree c: bisection of tau - 1 along
    a + b = 2c (tau increases with a on [c, 2c], tau(2c, 0) = c), 110 steps at 30 digits."""
    c = _deg(c, 0)[0]
    if c <= 1:
        raise Refusal("no-transition:ks-line", f"c = {c}: tau <= c <= 1 on the whole line a + b = 2c")
    with mp.workdps(30):
        lo, hi = c, 2 * c
        for _ in range(110):
            mid = (lo + hi) / 2
            if sbm_kesten_stigum(mid, 2 * c - mid).margin > 1:
                hi = mid
            else:
                lo = mid
        a = (lo + hi) / 2
        return KSLine(c, a, 2 * c - a, a / (2 * c - a))


# ---- sampler ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SBMGraph:
    N: int
    a: float
    b: float
    seed: int
    labels: np.ndarray        # +-1, int8
    edges: np.ndarray         # (M, 2) int64, i < j, no repeats


def _tri_pairs(t, n):
    """Index t in [0, n(n-1)/2) of the pairs i < j of range(n) (row-major) -> (i, j)."""
    t = np.asarray(t, dtype=np.int64)
    m = 2 * n - 1
    i = np.floor((m - np.sqrt(np.maximum(m * m - 8.0 * t, 0))) / 2).astype(np.int64)
    off = lambda r: r * (2 * n - r - 1) // 2
    i = np.where(off(i + 1) <= t, i + 1, i)
    i = np.where(off(i) > t, i - 1, i)
    return i, t - off(i) + i + 1


def _sample_count(rng, P, p):
    k = int(rng.binomial(P, p)) if P > 0 else 0
    return rng.choice(P, size=k, replace=False) if k else np.zeros(0, dtype=np.int64)


def sbm_sample(N, a, b, seed) -> SBMGraph:
    """Labels from default_rng([seed, 1]); per block an exact Binomial edge count and a uniform
    set of that many distinct pairs (each pair an independent Bernoulli)."""
    if isinstance(N, bool) or not isinstance(N, int) or N < 10:
        raise Refusal("precondition:size", f"N={N!r} must be an int >= 10")
    seed = _checks.seed(seed)
    a, b = (float(v) for v in _deg(a, b))
    if a > N or b > N:
        raise Refusal("precondition:edge-probability", f"a/N, b/N = {a / N:.3g}, {b / N:.3g} must be <= 1")
    rng = np.random.default_rng([seed, 1])
    labels = rng.choice(np.array([1, -1], dtype=np.int8), size=N)
    plus, minus = np.flatnonzero(labels == 1), np.flatnonzero(labels == -1)
    parts = []
    for grp in (plus, minus):
        n = grp.size
        i, j = _tri_pairs(_sample_count(rng, n * (n - 1) // 2, a / N), n)
        parts.append(np.stack([grp[i], grp[j]], axis=1))
    t = _sample_count(rng, plus.size * minus.size, b / N)
    if minus.size:
        parts.append(np.stack([plus[t // minus.size], minus[t % minus.size]], axis=1))
    E = np.concatenate(parts).astype(np.int64) if parts else np.zeros((0, 2), dtype=np.int64)
    E = np.sort(E, axis=1)
    E = E[np.lexsort((E[:, 1], E[:, 0]))]
    return SBMGraph(N, a, b, seed, labels, E)


# ---- belief propagation ---------------------------------------------------------------------------
@dataclass
class BPResult:
    overlap: float
    converged: bool
    iterations: int
    max_change: float
    factorized: bool
    marginals: np.ndarray     # psi^i_+
    messages: np.ndarray      # p_e, e < M: edges[e, 0] -> edges[e, 1]; e >= M: the reverse
    field: float              # dh = h_+ - h_-
    label: str = "statistical"


def _u(a, b, p):
    p = np.clip(p, _CLIP, 1 - _CLIP)
    return np.log((a * p + b * (1 - p)) / (b * p + a * (1 - p)))


def _sigmoid(L):
    return 0.5 * (1 + np.tanh(L / 2))


def _field(S, a, b, N):
    """dh solving dh = s (a - b)/N sum_i (2 sigmoid(S_i - dh) - 1) (s = _field_scale): the field
    (27) consistent with the marginals (28) of the current messages.  The right side decreases
    in dh, so the root is unique; bisection (a field lagging one synchronous sweep behind the
    marginals oscillates with period 2 when (a - b)/2 > 1)."""
    k = _field_scale * (a - b) / N
    lo, hi = -abs(k) * N - 1.0, abs(k) * N + 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if mid - k * float(np.sum(np.tanh((S - mid) / 2))) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def sbm_bp(graph, a, b, *, iters, seed, tol=1e-7, damping=0.0) -> BPResult:
    if not isinstance(graph, SBMGraph):
        raise Refusal("precondition:sbm-graph", "sbm_bp needs an SBMGraph from sbm_sample")
    a, b = (float(v) for v in _deg(a, b))
    seed = _checks.seed(seed)
    if isinstance(iters, bool) or not isinstance(iters, int) or iters < 1:
        raise Refusal("precondition:positive-iters", f"iters={iters!r} is not a positive int")
    if not 0 <= float(damping) < 1:
        raise Refusal("precondition:damping", f"damping={damping} outside [0, 1)")
    N, E = graph.N, graph.edges
    M = len(E)
    src, dst = np.concatenate([E[:, 0], E[:, 1]]), np.concatenate([E[:, 1], E[:, 0]])
    rev = np.concatenate([np.arange(M, 2 * M), np.arange(M)])
    p = np.random.default_rng([seed, 2]).random(2 * M)
    dh, change, it, conv = 0.0, math.inf, 0, False
    for it in range(1, iters + 1):
        u = _u(a, b, p)
        S = np.bincount(dst, weights=u, minlength=N)
        dh = _field(S, a, b, N)
        new = _sigmoid(S[src] - u[rev] - dh)
        if damping:
            new = (1 - damping) * new + damping * p
        change = float(np.max(np.abs(new - p))) if M else 0.0
        p = new
        if change < tol:
            conv = True
            break
    L = np.bincount(dst, weights=_u(a, b, p), minlength=N) - dh
    psi = _sigmoid(L)
    est = np.sign(L)
    ties = est == 0
    est[ties] = np.random.default_rng([seed, 3]).choice(np.array([-1.0, 1.0]), size=int(ties.sum()))
    Q = abs(float(np.mean(graph.labels * est)))
    return BPResult(Q, conv, it, change, bool(np.max(np.abs(psi - 0.5)) < 1e-6), psi, p, dh)


def dkmz_residual(graph, a, b, messages):
    """max over directed edges of |DKMZ Eq. (26) evaluated at `messages` - messages|, in plain
    Python product form (an independent implementation of the update): the field (27) is solved
    self-consistently with the marginals (28) by bisection on dh."""
    a, b = float(a), float(b)
    N, E = graph.N, graph.edges
    M = len(E)
    inc = [[] for _ in range(N)]                   # node -> [(directed edge into it, sender)]
    for e in range(2 * M):
        s, d = (E[e, 0], E[e, 1]) if e < M else (E[e - M, 1], E[e - M, 0])
        inc[d].append((e, s))

    def weights(k, skip=None):
        wp = wm = 1.0
        for e, s in inc[k]:
            if s == skip:
                continue
            q = min(max(float(messages[e]), _CLIP), 1 - _CLIP)
            wp *= a * q + b * (1 - q)
            wm *= b * q + a * (1 - q)
        return wp, wm

    W = [weights(k) for k in range(N)]

    def field_gap(dh):
        tot = sum((wp * math.exp(-dh) - wm) / (wp * math.exp(-dh) + wm) for wp, wm in W)
        return dh - (a - b) / N * tot

    lo, hi = -abs(a - b) - 1.0, abs(a - b) + 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if field_gap(mid) < 0 else (lo, mid)
    dh = (lo + hi) / 2
    worst = 0.0
    for e in range(2 * M):
        i, j = (E[e, 0], E[e, 1]) if e < M else (E[e - M, 1], E[e - M, 0])
        wp, wm = weights(i, skip=j)
        worst = max(worst, abs(wp * math.exp(-dh) / (wp * math.exp(-dh) + wm) - float(messages[e])))
    return worst


@dataclass
class BPEnsemble:
    overlaps: list
    converged: list
    factorized: list
    iterations: list
    band: Band                # 99% Student-t band of the mean overlap
    seeds: tuple
    N: int
    a: float
    b: float
    label: str = "statistical"
    note: str = ""


def sbm_bp_ensemble(N, a, b, *, seeds, iters, damping=0.0) -> BPEnsemble:
    seeds = _checks.seeds(seeds)
    runs = [sbm_bp(sbm_sample(N, a, b, s), a, b, iters=iters, seed=s, damping=damping)
            for s in seeds]
    ov = [r.overlap for r in runs]
    note = (f"statistical: BP (DKMZ Eqs. (26)-(28), synchronous) on {len(seeds)} graphs, N = {N}, "
            f"seeds {list(seeds)}, converged on {sum(r.converged for r in runs)}, factorized on "
            f"{sum(r.factorized for r in runs)}; 99% Student-t band of the mean overlap")
    return BPEnsemble(ov, [r.converged for r in runs], [r.factorized for r in runs],
                      [r.iterations for r in runs], student_band(ov), seeds, N, float(a), float(b),
                      note=note)


__all__ = ["BPEnsemble", "BPResult", "KSLine", "KSResult", "SBMGraph", "dkmz_residual", "sbm_bp",
           "sbm_bp_ensemble", "sbm_kesten_stigum", "sbm_ks_line", "sbm_sample"]
