"""Replica-free routes to the perceptron capacities (independent checks).

  cover_count(p, N)                     Cover (1965): separable dichotomies, exact ints
  cover_separable_fraction(p, N)        cover_count / 2^p (Fraction)
  ising_capacity_enumeration(N, samples, seed)
                                        mean, s.e. of M_N/N by enumerating {-1,+1}^N

Cover's count is for p points in general position in R^N and homogeneous
hyperplanes (labels arbitrary), by the recurrence quoted in Pastore et al.,
arXiv:2005.10002, Eqs. (15)-(16): C(n, p+1) = C(n, p) + C(n-1, p),
C(n >= 1, 1) = 2, C(0, p) = 0; their Eq. (14) closed form
2 sum_{k<N} binom(p-1, k) is checked against it in the tests, not used here.
The fraction is exactly 1/2 at p = 2N (binomial symmetry), whence alpha_c = 2.

Enumeration: Gaussian patterns (labels absorbed), M_N = largest M such that
the first M patterns have a J in {-1,+1}^N with J.xi > 0 for all of them
(the definition of Ding & Sun, arXiv:1809.07742, Eq. (1), kappa = 0).
Statistical (float64, seeded) and finite-N: it can tell 0.83 from 4/pi,
not certify digits.
"""
import itertools
from fractions import Fraction

import numpy as np

from ..core import Refusal


def cover_count(p, N):
    if not (isinstance(p, int) and isinstance(N, int) and p >= 1 and N >= 1):
        raise Refusal("precondition:positive-integers",
                      f"p = {p!r}, N = {N!r}: Cover's count needs integers p, N >= 1")
    row = [0] + [2] * N                 # C(n, 1) for n = 0..N
    for _ in range(p - 1):
        row = [0] + [row[n] + row[n - 1] for n in range(1, N + 1)]
    return row[N]


def cover_separable_fraction(p, N):
    return Fraction(cover_count(p, N), 2**p)


def ising_capacity_enumeration(N, samples, seed=0):
    if not (isinstance(N, int) and 2 <= N <= 16):
        raise Refusal("precondition:small-N",
                      f"N = {N!r}: exact enumeration of 2^N couplings needs 2 <= N <= 16")
    if not (isinstance(samples, int) and samples >= 2):
        raise Refusal("precondition:samples", f"samples = {samples!r}: need an int >= 2")
    rng = np.random.default_rng(seed)
    J = np.array(list(itertools.product((-1.0, 1.0), repeat=N)))
    m_max = 3 * N
    out = np.empty(samples)
    for i in range(samples):
        alive = np.cumprod(J @ rng.standard_normal((N, m_max)) > 0, axis=1).any(axis=0)
        if alive[-1]:
            raise Refusal("no-solution", f"enumeration: a sample stored all {m_max} patterns")
        out[i] = alive.sum()
    return float(out.mean() / N), float(out.std(ddof=1) / N / np.sqrt(samples))
