"""Exact Boltzmann distributions of the q-state Potts / Ising model on small L x L tori, by listing every
one of the q^N states (N = L^2): the targets of the detailed-balance chi^2 tests and the exact <|m|>,
<m^2> of ruling A-R6 (no Kaufman form exists for them).

Conventions (the whole montecarlo package).  Torus bonds: every site (x, y) (id y L + x) bonds to
(x + 1, y) and (x, y + 1) mod L, 2N bonds; at L = 2 the torus is a multigraph (two bonds between each
neighbouring pair), as in lattice/kaufman.py and lattice/transfer.py.  Colours 0 .. q - 1.
  q = 2 (Ising):  spins s = 1 - 2 colour, H = -J sum s_i s_j with J = 1, beta the Ising beta;
                  E = 2 n_unsat - 2N; m = |sum s| / N; m^2 = (sum s / N)^2.
  q >= 3 (Potts): H = -sum delta(c_i, c_j) (Beffara & Duminil-Copin's Theorem 3 convention), E = -n_sat;
                  m = (q max_a n_a / N - 1) / (q - 1); m^2 = (q sum_a n_a^2 / N^2 - 1) / (q - 1)
                  (both reduce to the Ising |m|, m^2 at q = 2).
  e = E / N; C = beta^2 Var(E) / N (k_B = 1), the c_L of lattice/torus_thermo.py at q = 2.

  torus_bonds(L) -> (u, v) int arrays (2N bonds)
  enumerate_torus(L, q) -> Enumerated (states (q^N, N) int8, n_sat, E, abs_m, m2, cls)
      cls: the class id of each state: (n_sat, the sorted colour-count vector), the chi^2 cells
  exact_distribution(L, q, beta) -> Exact (probs over states, class probs, e, abs_m, m2, C)
      beta: a rational (int, Fraction, decimal string) or "beta_c" (the q model's critical point,
      fk.py); weights in mpmath at 40 digits, then float
  kaufman_histogram_identity(L) -> (holds, detail): at q = 2 the enumerated histogram of n_unsat is
      exactly kaufman_polynomial(L, L)'s integer coefficients (two exact routes to the same g(E))
Sizes: q^N <= MAX_STATES (2^16: Ising L <= 4, Potts q = 3 L <= 3), else precondition:exact-size.
"""
from dataclasses import dataclass
from functools import lru_cache

import mpmath as mp
import numpy as np

from ..core import Refusal

MAX_STATES = 1 << 16


def torus_bonds(L):
    i = np.arange(L * L)
    x, y = i % L, i // L
    right = y * L + (x + 1) % L
    down = ((y + 1) % L) * L + x
    return np.concatenate([i, i]), np.concatenate([right, down])


@dataclass(frozen=True)
class Enumerated:
    L: int
    q: int
    states: np.ndarray
    n_sat: np.ndarray
    E: np.ndarray
    abs_m: np.ndarray
    m2: np.ndarray
    cls: np.ndarray
    n_cls: int


def _check(L, q):
    if isinstance(L, bool) or not isinstance(L, int) or L < 2:
        raise Refusal("precondition:size", f"L = {L!r}: need an int >= 2")
    if isinstance(q, bool) or not isinstance(q, int) or q < 2:
        raise Refusal("precondition:potts-q", f"q = {q!r}: need an int >= 2")
    if q ** (L * L) > MAX_STATES:
        raise Refusal("precondition:exact-size", f"q^N = {q}^{L * L} > {MAX_STATES} states")


def order_parameters(counts, N, q):
    """(|m|, m^2) from the colour counts n_a (last axis) under the conventions above."""
    c = np.asarray(counts, float)
    return (q * c.max(axis=-1) / N - 1) / (q - 1), (q * (c * c).sum(axis=-1) / N ** 2 - 1) / (q - 1)


def energy(n_sat, N, q):
    return 2 * (2 * N - n_sat) - 2 * N if q == 2 else -n_sat


@lru_cache(maxsize=8)
def enumerate_torus(L, q):
    _check(L, q)
    N = L * L
    idx = np.arange(q ** N, dtype=np.int64)
    states = np.stack([(idx // q ** k) % q for k in range(N)], 1).astype(np.int8)
    u, v = torus_bonds(L)
    n_sat = (states[:, u] == states[:, v]).sum(axis=1)
    counts = np.stack([(states == a).sum(axis=1) for a in range(q)], 1)
    abs_m, m2 = order_parameters(counts, N, q)
    key = np.sort(counts, axis=1)
    key = np.concatenate([n_sat[:, None], key], 1)
    _, cls = np.unique(key, axis=0, return_inverse=True)
    cls = cls.ravel()
    return Enumerated(L, q, states, n_sat, energy(n_sat, N, q), abs_m, m2, cls, int(cls.max()) + 1)


@dataclass(frozen=True)
class Exact:
    probs: np.ndarray
    class_probs: np.ndarray
    e: float
    abs_m: float
    m2: float
    C: float


def potts_coupling(q, beta):
    """The Potts coupling K of exp(K sum delta): 2 beta for the Ising convention (q = 2), beta else."""
    return 2 * beta if q == 2 else beta


def exact_distribution(L, q, beta):
    from .fk import beta_value
    en = enumerate_torus(L, q)
    N = L * L
    with mp.workdps(40):
        b = beta_value(q, beta)
        levels = np.unique(en.n_sat)
        lw = {int(k): potts_coupling(q, b) * int(k) for k in levels}
        top = max(lw.values())
        w = {k: mp.exp(x - top) for k, x in lw.items()}
        mult = {int(k): int((en.n_sat == k).sum()) for k in levels}
        Z = mp.fsum(mult[k] * w[k] for k in w)
        p_level = {k: w[k] / Z for k in w}
        probs = np.array([float(p_level[int(k)]) for k in en.n_sat])
        probs /= probs.sum()
        E = {k: energy(k, N, q) for k in w}
        mE = mp.fsum(mult[k] * p_level[k] * E[k] for k in w)
        vE = mp.fsum(mult[k] * p_level[k] * (E[k] - mE) ** 2 for k in w)
        e, C = float(mE / N), float(b * b * vE / N)
    cp = np.bincount(en.cls, probs, en.n_cls)
    return Exact(probs, cp, e, float(probs @ en.abs_m), float(probs @ en.m2), C)


def kaufman_histogram_identity(L):
    from ..lattice.kaufman import kaufman_polynomial
    en = enumerate_torus(L, 2)
    N = L * L
    hist = np.bincount(2 * N - en.n_sat, minlength=2 * N + 1)
    want = kaufman_polynomial(L, L).coefficients
    got = tuple(int(x) for x in hist)
    holds = got == tuple(want)
    return holds, (f"{L} x {L} torus: the enumerated n_unsat histogram equals Kaufman's integer polynomial "
                   f"({len(want)} coefficients, sum {sum(want)} = 2^{N})" if holds else
                   f"{L} x {L} torus: enumerated {got} != Kaufman {tuple(want)}")


__all__ = ["Enumerated", "Exact", "MAX_STATES", "energy", "enumerate_torus", "exact_distribution",
           "kaufman_histogram_identity", "order_parameters", "potts_coupling", "torus_bonds"]
