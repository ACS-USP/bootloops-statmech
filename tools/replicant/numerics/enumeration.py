"""Exact enumeration (E11): +-J SK spectra and Ising perceptron solution counts.

  gray_states(m)                    the 2^m states of m spins in binary-reflected Gray order
  random_pm_one_J(N, seed)          a +-J SK matrix J_ij = sigma_ij / sqrt(N) (named seed)
  enumerate_pm_one_sk(J)            PmOneSpectrum: the exact integer density of states
  exact_ground_energy(J)            min_s H(s) / N for any real symmetric J (N <= 26)
  ising_perceptron_counts(N, P, seed)   #{J in {-1,+1}^N storing patterns 1..p}, p = 0..P
  ising_perceptron_log_count(alpha, N, seed)   ln of that count at P = alpha N (-inf if 0)
  ising_perceptron_capacity(counts) the largest p with a nonzero count (M_N)

SK conventions: H = -sum_{i<j} J_ij s_i s_j, h = 0.  For +-J, J_ij = sigma_ij / sqrt(N) with
sigma_ij = +-1, so H = -K(s) / sqrt(N) with the integer K(s) = sum_{i<j} sigma_ij s_i s_j in
[-M, M], M = N(N-1)/2, K = M (mod 2).  The spectrum is the histogram g_K over all 2^N states
(exact integers), and Z(beta) = sum_K g_K exp(beta K / sqrt N).

Method (both models): the spins are split into two blocks A and B; each block's states are
listed in binary-reflected Gray order, built by reflection (G_{m+1} = [G_m, +1 ; reverse(G_m),
-1], consecutive rows differ by one spin) together with the block's own energies by the same
doubling, O(2^m m) vectorised.  The cross term between the blocks is a matrix product
S_A W_AB S_B^T, taken in row chunks.  For +-J the products are integers below 2^53 in float64,
so the histogram is exact.  With h = 0, H is even under s -> -s: the last spin is fixed to +1
and the counts are doubled.  The perceptron has no such symmetry; its count over the pairs
(A, B) is filtered pattern by pattern (the survivors of the first p patterns, so one pass
gives every p).  Ties J.xi = 0 have probability zero for Gaussian patterns and are counted as
not stored.  Exact up to float64 rounding of the Gaussian fields (perceptron) and exact in
integers (SK).  Cost grows as 2^N: N = 16 takes milliseconds, N = 26 about a second.
"""
import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ..core import Refusal

MAX_N_SK = 26
MAX_N_PERCEPTRON = 30
_CHUNK = 1 << 22          # entries per cross-term chunk


def gray_states(m):
    """(2^m, m) float array of +-1 in binary-reflected Gray order (row 0 all +1)."""
    S = np.zeros((1, 0))
    for _ in range(m):
        S = np.vstack([np.hstack([S, np.ones((len(S), 1))]),
                       np.hstack([S[::-1], -np.ones((len(S), 1))])])
    return S


def _block(W, idx):
    """Gray-ordered states of the spins idx and their internal energies sum_{i<j} W_ij s_i s_j,
    both built by the reflection doubling (the new spin's field from the earlier ones)."""
    S, K = np.zeros((1, 0)), np.zeros(1)
    for t, c in enumerate(idx):
        col = S @ W[idx[:t], c]
        S = np.vstack([np.hstack([S, np.ones((len(S), 1))]),
                       np.hstack([S[::-1], -np.ones((len(S), 1))])])
        K = np.concatenate([K + col, K[::-1] - col[::-1]])
    return S, K


def _check_matrix(J):
    J = np.asarray(J, dtype=float)
    if J.ndim != 2 or J.shape[0] != J.shape[1] or J.shape[0] < 2 or not np.isfinite(J).all() \
            or not np.array_equal(J, J.T) or np.any(np.diag(J) != 0):
        raise Refusal("precondition:coupling-matrix",
                      f"need a finite symmetric N x N matrix (N >= 2) with zero diagonal, got "
                      f"shape {J.shape}")
    if len(J) > MAX_N_SK:
        raise Refusal("precondition:enumeration-size",
                      f"N = {len(J)}: exact enumeration here is for N <= {MAX_N_SK}")
    return J


def _sk_chunks(W):
    """K(s) = sum_{i<j} W_ij s_i s_j for every state with s_{N-1} = +1, in chunks (2D)."""
    N = len(W)
    free = list(range(N - 1))
    a = len(free) // 2
    A, B = free[:a], free[a:]
    SA, KA = _block(W, A)
    SB, KB = _block(W, B)
    SB = np.hstack([SB, np.ones((len(SB), 1))])              # the fixed spin s_{N-1} = +1
    KB = KB + SB[:, :-1] @ W[B, N - 1]
    C = SA @ W[np.ix_(A, B + [N - 1])]
    rows = max(1, _CHUNK // len(SB))
    for r in range(0, len(SA), rows):
        yield KA[r:r + rows, None] + KB[None, :] + C[r:r + rows] @ SB.T


@dataclass(frozen=True)
class PmOneSpectrum:
    """counts[K + M] = number of states with K(s) = K (all 2^N states), E = -K / sqrt N."""
    N: int
    M: int
    counts: np.ndarray

    def _logw(self, beta):
        K = np.arange(-self.M, self.M + 1)
        nz = self.counts > 0
        return K[nz], np.log(self.counts[nz].astype(float)) + beta * K[nz] / math.sqrt(self.N)

    def log_Z(self, beta):
        _, lw = self._logw(float(beta))
        top = lw.max()
        return float(top + math.log(np.exp(lw - top).sum()))

    def mean_energy_per_spin(self, beta):
        K, lw = self._logw(float(beta))
        w = np.exp(lw - lw.max())
        return float(-(w @ K) / w.sum() / math.sqrt(self.N) / self.N)

    @property
    def ground_energy(self):
        """Ground-state energy (total, not per spin): -K_max / sqrt N."""
        return -int(np.nonzero(self.counts)[0].max() - self.M) / math.sqrt(self.N)


def random_pm_one_J(N, seed):
    """J_ij = sigma_ij / sqrt(N), sigma_ij = +-1 iid (i < j) from default_rng(seed)."""
    rng = np.random.default_rng(seed)
    sig = np.triu(rng.choice((-1.0, 1.0), size=(N, N)), 1)
    return (sig + sig.T) / math.sqrt(N)


def enumerate_pm_one_sk(J):
    J = _check_matrix(J)
    N = len(J)
    sig = J * math.sqrt(N)
    if np.abs(np.abs(sig[np.triu_indices(N, 1)]) - 1).max() > 1e-12:
        raise Refusal("precondition:pm-one-couplings",
                      "enumerate_pm_one_sk needs J_ij = +-1/sqrt(N) off the diagonal")
    sig = np.rint(sig)
    M = N * (N - 1) // 2
    counts = np.zeros(2 * M + 1, dtype=np.int64)
    for K in _sk_chunks(sig):
        counts += np.bincount(np.rint(K).astype(np.int64).ravel() + M, minlength=2 * M + 1)
    return PmOneSpectrum(N, M, 2 * counts)


def exact_ground_energy(J):
    """min_s H(s) / N, H = -sum_{i<j} J_ij s_i s_j, by the same blocks (float64)."""
    J = _check_matrix(J)
    return -max(float(K.max()) for K in _sk_chunks(J)) / len(J)


# ---- Ising perceptron ---------------------------------------------------------------
def _check_seed(seed):
    ok = isinstance(seed, (int, np.integer)) and not isinstance(seed, bool) and seed >= 0
    ok = ok or (isinstance(seed, (list, tuple)) and seed and
                all(isinstance(x, (int, np.integer)) and x >= 0 for x in seed))
    if not ok:
        raise Refusal("precondition:seed", f"seed = {seed!r}: need an int >= 0 or a list of them")


def ising_perceptron_counts(N, P, seed):
    """counts[p] for p = 0..P: the number of J in {-1,+1}^N with J.xi^mu > 0 for mu < p, the
    patterns being the rows of default_rng(seed).standard_normal((P, N)) (nested in P)."""
    _check_seed(seed)
    if not (isinstance(N, int) and 2 <= N <= MAX_N_PERCEPTRON):
        raise Refusal("precondition:enumeration-size",
                      f"N = {N!r}: exact counting here is for 2 <= N <= {MAX_N_PERCEPTRON}")
    if not (isinstance(P, int) and P >= 0):
        raise Refusal("precondition:integer-patterns", f"P = {P!r}: need an int >= 0")
    xi = np.random.default_rng(seed).standard_normal((P, N))
    counts = np.zeros(P + 1, dtype=np.int64)
    counts[0] = 2**N
    if P == 0:
        return counts
    a = N // 2
    SA, SB = gray_states(a), gray_states(N - a)
    FA = np.ascontiguousarray((SA @ xi[:, :a].T).T)            # P x 2^a
    FB = np.ascontiguousarray((SB @ xi[:, a:].T).T)            # P x 2^b
    rows = max(1, _CHUNK // FB.shape[1])
    for r in range(0, FA.shape[1], rows):
        ia, ib = np.nonzero(FA[0, r:r + rows, None] + FB[0, None, :] > 0)
        ia = ia + r
        counts[1] += len(ia)
        for p in range(1, P):
            if not len(ia):
                break
            keep = FA[p][ia] + FB[p][ib] > 0
            ia, ib = ia[keep], ib[keep]
            counts[p + 1] += len(ia)
    return counts


def _patterns(alpha, N):
    try:
        a = Fraction(str(alpha)) if isinstance(alpha, float) else Fraction(alpha)
    except (TypeError, ValueError):
        a = None
    if a is None or a < 0 or (a * N).denominator != 1:
        raise Refusal("precondition:integer-patterns",
                      f"alpha = {alpha!r}, N = {N!r}: P = alpha N must be an integer >= 0")
    return int(a * N)


def ising_perceptron_log_count(alpha, N, seed):
    """ln #{J : J.xi^mu > 0, mu = 1..alpha N} for one pattern set; -inf when the count is 0
    (the caller states how it handles empty solution sets)."""
    if not (isinstance(N, int) and 2 <= N <= MAX_N_PERCEPTRON):
        raise Refusal("precondition:enumeration-size",
                      f"N = {N!r}: exact counting here is for 2 <= N <= {MAX_N_PERCEPTRON}")
    P = _patterns(alpha, N)
    c = int(ising_perceptron_counts(N, P, seed)[P])
    return math.log(c) if c else -math.inf


def ising_perceptron_capacity(counts):
    """M_N: the largest p with counts[p] > 0 (Ding & Sun's definition, kappa = 0); refused
    when every drawn pattern is stored (draw more patterns)."""
    counts = np.asarray(counts)
    if counts[-1] > 0:
        raise Refusal("no-solution:capacity",
                      f"all {len(counts) - 1} drawn patterns are stored; draw more")
    return int(np.nonzero(counts)[0].max())
