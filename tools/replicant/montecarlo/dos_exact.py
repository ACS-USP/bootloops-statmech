"""The exact density of states g(E) of the 2D Ising model on the L x L torus: the target of the Wang-Landau and
multicanonical checks (M2, ruling A-R3), and the exact canonical averages it implies.

Source.  x^N Z = P(x) = sum_j c_j x^j with x = e^{-2K}, N = L^2: c_j is the number of the 2^N configurations
with j unsatisfied bonds (lattice/kaufman.py: Kaufman's torus partition function interpolated in ball
arithmetic, every coefficient ball of radius < 1/4 holding exactly one integer).  So g(E) = c_j at
E = 2j - 2N (H = -sum s_i s_j over the 2N torus bonds, J = 1, as montecarlo/exact_small.py).  On the torus j is
even (every row and every column cycle carries an even number of unsatisfied bonds), so the level index used
here is k = j / 2 in 0 .. N and E_k = 4k - 2N.
  L <= 12: kaufman.kaufman_polynomial(L, L) (its MAX_SIDE).
  13 <= L <= 16: the same interpolation (2N + 1 rational nodes, arb Vandermonde solve, integers certified by
      balls of radius < 1/4), run here on kaufman._z because kaufman_polynomial stops at 12 (measured cost at
      L = 16: 35 s, 2116 bits; slow tier only).  Checked in the slow tier: sum c_j = 2^N, c_j = c_{2N - j}
      at even L, and the moments u_L, c_L at beta = 2/5 against torus_thermo_kaufman (Kaufman's formula
      differentiated in arb_series: an independent route).

  exact_dos(L) -> ExactDOS(L, k (levels with g > 0), E, g (python ints), ln_g (float))   (cached)
  canonical(E, ln_g, beta, N) -> Canonical(u, C, ln_z)      float log-sum-exp averages of any ln g estimate:
      u = <E>/N, C = beta^2 Var(E)/N, ln_z = ln Z / N (Z = sum_E g(E) e^{-beta E})
  exact_canonical(L, beta) -> Canonical at 40 digits from the integers (mpmath), then float
Refusals: precondition:size (L not an int in 2 .. 16).
"""
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

import mpmath as mp
import numpy as np

from ..core import Refusal

MAX_L = 16
KAUFMAN_SIDE = 12


@dataclass(frozen=True)
class ExactDOS:
    L: int
    k: np.ndarray          # level indices k = n_unsat / 2 with g > 0
    E: np.ndarray          # energies 4k - 2N
    g: tuple               # python ints, one per level
    ln_g: np.ndarray       # float ln g


@dataclass(frozen=True)
class Canonical:
    u: float
    C: float
    ln_z: float


def check_L(L, top=MAX_L):
    if isinstance(L, bool) or not isinstance(L, int) or not 2 <= L <= top:
        raise Refusal("precondition:size", f"L = {L!r}: need an int in 2 .. {top}")
    return L


def _interpolated(L):
    """kaufman_polynomial's interpolation for 12 < L <= 16 (see the module docstring)."""
    from flint import arb_mat
    from ..lattice.ball import arb_q, baller
    from ..lattice.kaufman import _z
    N = L * L
    n = 2 * N + 1
    nodes = [Fraction(i, n + 1) for i in range(1, n + 1)]
    prec = 64 + 4 * n
    while prec <= 1 << 15:
        with baller().hygiene.ctx_guard(prec=prec):
            xs = [arb_q(x) for x in nodes]
            ys = [_z(L, L, -x.log() / 2) * x ** N for x in xs]
            c = arb_mat([[x ** j for j in range(n)] for x in xs]).solve(arb_mat([[y] for y in ys]))
            balls = [c[j, 0] for j in range(n)]
            if all(b.is_finite() and b.rad() < 0.25 for b in balls):
                ints = [b.unique_fmpz() for b in balls]
                if any(i is None for i in ints):
                    raise Refusal("no-solution:not-an-integer-polynomial", f"{L} x {L} torus at {prec} bits")
                return tuple(int(i) for i in ints)
        prec *= 2
    raise Refusal("no-solution:certification", f"{L} x {L} torus: coefficient radii not < 1/4")


@lru_cache(maxsize=None)
def exact_dos(L):
    check_L(L)
    if L <= KAUFMAN_SIDE:
        from ..lattice.kaufman import kaufman_polynomial
        cs = kaufman_polynomial(L, L).coefficients
    else:
        cs = _interpolated(L)
    N = L * L
    if any(cs[j] for j in range(1, len(cs), 2)):
        raise Refusal("no-solution:sampler-invariant", f"{L} x {L} torus: an odd number of unsatisfied bonds")
    k = np.array([j // 2 for j in range(0, 2 * N + 1, 2) if cs[j]])
    g = tuple(cs[2 * i] for i in k)
    with mp.workdps(30):
        lng = np.array([float(mp.log(x)) for x in g])
    return ExactDOS(L, k, 4 * k - 2 * N, g, lng)


def canonical(E, ln_g, beta, N):
    E = np.asarray(E, float)
    a = np.asarray(ln_g, float) - float(beta) * E
    top = a.max()
    w = np.exp(a - top)
    Z = w.sum()
    p = w / Z
    u = float(p @ E)
    var = float(p @ (E - u) ** 2)
    return Canonical(u / N, float(beta) ** 2 * var / N, float((np.log(Z) + top) / N))


def exact_canonical(L, beta):
    d = exact_dos(L)
    N = L * L
    with mp.workdps(40):
        b = mp.mpf(beta) if not isinstance(beta, Fraction) else mp.mpf(beta.numerator) / beta.denominator
        a = [mp.log(gi) - b * int(e) for gi, e in zip(d.g, d.E)]
        top = max(a)
        w = [mp.exp(x - top) for x in a]
        Z = mp.fsum(w)
        u = mp.fsum(wi * int(e) for wi, e in zip(w, d.E)) / Z
        var = mp.fsum(wi * (int(e) - u) ** 2 for wi, e in zip(w, d.E)) / Z
        return Canonical(float(u / N), float(b * b * var / N), float((mp.log(Z) + top) / N))


__all__ = ["Canonical", "ExactDOS", "KAUFMAN_SIDE", "MAX_L", "canonical", "check_L", "exact_canonical", "exact_dos"]
