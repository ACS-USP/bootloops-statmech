"""Transfer matrices: the exact Lx x Ly torus polynomial (integers) and the 1D Ising ring.

Square-lattice torus (no Onsager or Kaufman input).  With x = e^{-2K}, N = Lx Ly sites and 2N
bonds (sizes 2: the torus multigraph, two bonds between the same pair),
  Z = sum_config e^{K sum s s'} = e^{2NK} P(x),  P(x) = sum_{j=0}^{2N} c_j x^j,
c_j = the number of configurations with j unsatisfied bonds.  The row-to-row transfer matrix
T[s, s'] = x^{u(s) + |s xor s'|} (u(s): unsatisfied bonds inside the periodic row s of Ly spins)
gives P = Tr T^Lx.  It is computed exactly in Python integers by Kronecker substitution, x = 2^B
with B = N + 1 > log2 max c_j (sum c_j = 2^N), so the base-2^B digits of Tr T^Lx are the c_j.
Ly <= 7 (128 states) and Lx <= 16.

1D Ising ring (J > 0, field h; Wikipedia 'Ising model', Sec. One dimension): with a = e^K, b = e^{beta h},
  T = [[a b, 1/a], [1/a, a/b]],  lambda_+- = e^K cosh(beta h) +- sqrt(e^{2K} sinh^2(beta h) + e^{-2K}),
  Z_N = Tr T^N = lambda_+^N + lambda_-^N,  -beta f = ln lambda_+.
ising1d_ring_identity checks Z_N three ways in exact rationals: brute force over the 2^N
configurations, the rational matrix trace, and the closed form in Q(sqrt D) (D = a^2 ((b - 1/b)/2)^2
+ 1/a^2: lambda_+^N + lambda_-^N = 2 sum_{j even} C(N, j) p^{N-j} D^{j/2}, p = a (b + 1/b)/2).

  torus_partition_polynomial(Lx, Ly) -> tuple of ints c_0 .. c_2N
  ising_transfer_matrix_free_energy(L, beta, Lx=None, J=1) -> Ball   -beta f = ln Z / N of the
      Lx x L torus (Lx defaults to L), Z from the exact polynomial
  ising1d_free_energy(beta, h, J=1) -> Ball                          ln lambda_+
  ising1d_ring_identity(N, a, b) -> RingCheck(brute, trace, closed)  exact Fractions
  ising1d_ring_balls(N, beta, h, J=1) -> (Ball, Ball)                Tr T^N and the closed form
_HOOKS["d_sign"] (+1): planted-error hook of the closed form's discriminant (tests).
"""
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import comb

from ..core import Refusal
from .ball import Ball, arb_q, certify, rational
from .coupling import coupling

MAX_LX, MAX_LY = 16, 7
_HOOKS = {"d_sign": 1}


def _side(name, L, top):
    if isinstance(L, bool) or not isinstance(L, int) or not 2 <= L <= top:
        raise Refusal("precondition:size", f"{name} = {L!r}: need an int in 2 .. {top}")
    return L


def _matmul(A, B):
    n = len(A)
    Bt = list(zip(*B))
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]


def torus_partition_polynomial(Lx, Ly):
    Lx, Ly = _side("Lx", Lx, MAX_LX), _side("Ly", Ly, MAX_LY)
    N = Lx * Ly
    B = N + 1
    n, mask = 1 << Ly, (1 << Ly) - 1
    rot = lambda s: ((s << 1) | (s >> (Ly - 1))) & mask
    u = [bin(s ^ rot(s)).count("1") for s in range(n)]
    T = [[1 << (B * (u[s] + bin(s ^ t).count("1"))) for t in range(n)] for s in range(n)]
    R, P, e = None, T, Lx                      # T^Lx by squaring
    while e:
        if e & 1:
            R = P if R is None else _matmul(R, P)
        e >>= 1
        if e:
            P = _matmul(P, P)
    z = sum(R[s][s] for s in range(n))
    coeffs = []
    for _ in range(2 * N + 1):
        coeffs.append(z & ((1 << B) - 1))
        z >>= B
    if z:
        raise Refusal("no-solution:audit", f"{Lx} x {Ly}: Kronecker digits overflow")
    return tuple(coeffs)


def ising_transfer_matrix_free_energy(L, beta, Lx=None, J=1) -> Ball:
    Lx = L if Lx is None else Lx
    cs = torus_partition_polynomial(Lx, L)
    N = Lx * L
    c = coupling(beta, J)

    def thunk():
        K = c.arb()
        x = (-2 * K).exp()
        p = 0
        for coef in reversed(cs):
            p = p * x + coef
        return 2 * K + p.log() / N
    return certify(thunk)


def ising1d_free_energy(beta, h, J=1) -> Ball:
    K = rational(beta, "beta") * rational(J, "J")
    H = rational(beta, "beta") * rational(h, "h", positive=False)

    return certify(lambda: _lambdas(arb_q(K), arb_q(H))[0].log())


def _lambdas(K, H):
    sh = H.sinh()
    root = ((2 * K).exp() * sh * sh + (-2 * K).exp()).sqrt()
    return K.exp() * H.cosh() + root, K.exp() * H.cosh() - root


def ising1d_ring_balls(N, beta, h, J=1):
    """(Tr T^N, lambda_+^N + lambda_-^N) as Balls at the given rational beta, h: the ring
    identity in ball arithmetic at the run's own parameters (overlap expected)."""
    if isinstance(N, bool) or not isinstance(N, int) or not 2 <= N <= 1000:
        raise Refusal("precondition:size", f"N = {N!r}: need an int in 2 .. 1000")
    K = rational(beta, "beta") * rational(J, "J")
    H = rational(beta, "beta") * rational(h, "h", positive=False)

    def trace():
        from flint import arb_mat
        k, x = arb_q(K), arb_q(H)
        T = arb_mat([[(k + x).exp(), (-k).exp()], [(-k).exp(), (k - x).exp()]])
        P = T ** N
        return P[0, 0] + P[1, 1]

    def closed():
        lp, lm = _lambdas(arb_q(K), arb_q(H))
        return lp ** N + lm ** N
    return certify(trace), certify(closed)


@dataclass(frozen=True)
class RingCheck:
    brute: Fraction
    trace: Fraction
    closed: Fraction


def ising1d_ring_identity(N, a, b) -> RingCheck:
    if isinstance(N, bool) or not isinstance(N, int) or not 2 <= N <= 16:
        raise Refusal("precondition:size", f"N = {N!r}: need an int in 2 .. 16")
    a, b = rational(a, "a"), rational(b, "b")
    T = {(1, 1): a * b, (-1, -1): a / b, (1, -1): 1 / a, (-1, 1): 1 / a}
    brute = Fraction(0)
    for s in product((1, -1), repeat=N):
        w = Fraction(1)
        for i in range(N):
            w *= T[s[i], s[(i + 1) % N]]
        brute += w
    M = [[T[1, 1], T[1, -1]], [T[-1, 1], T[-1, -1]]]
    R = [[Fraction(1), Fraction(0)], [Fraction(0), Fraction(1)]]
    for _ in range(N):
        R = _matmul(R, M)
    p = a * (b + 1 / b) / 2
    D = a * a * ((b - 1 / b) / 2) ** 2 + _HOOKS["d_sign"] / (a * a)
    closed = 2 * sum(comb(N, j) * p ** (N - j) * D ** (j // 2) for j in range(0, N + 1, 2))
    return RingCheck(brute, R[0][0] + R[1][1], closed)


__all__ = ["MAX_LX", "MAX_LY", "RingCheck", "ising1d_free_energy", "ising1d_ring_balls",
           "ising1d_ring_identity",
           "ising_transfer_matrix_free_energy", "torus_partition_polynomial"]
