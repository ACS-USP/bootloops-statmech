"""Spin-spin correlations of the square-lattice Ising model at T_c (zero field, J = 1), certified.

Sources read:
  Deift, Its & Krasovsky, arXiv:1207.4990 (2012), Eqs. (103)-(107): at T = T_c the diagonal
    correlation <s_{1,1} s_{1+n,1+n}> is the n x n Toeplitz determinant of the symbol
    phi(theta) = e^{i (pi - theta)/2} (0 < theta < 2 pi), whose Fourier coefficients are
    (phi)_k = (2/pi) / (2k + 1)  (Eq. (105));  as n -> oo
    <s s_nn> = e^{1/4} 2^{1/12} A^{-3} n^{-1/4} (1 - 1/(64 n^2) + ...)  (Eq. (106)), A Glaisher's
    constant (Eq. (107)).  (Witte & Forrester, arXiv:1105.4389, Eqs. (1.5), (1.6), give the same
    determinant for general k = sinh^2 2K; Lyberg & McCoy, arXiv:math-ph/0612051, Eqs. (8)-(12).)
  Perk & Au-Yang, arXiv:0901.1931 (2009), Eqs. (31), (32): for k = 1,
    ln C(n, n) = ln C(n-1, n-1) + r_n,  r_1 = ln(2/pi),  r_{n+1} = r_n - ln(1 - 1/(4 n^2)),
    i.e. C(n, n) = (2/pi)^n prod_{k=1}^{n-1} (1 - 1/(4k^2))^{-(n-k)}.
  Onsager, Phys. Rev. 65, 117 (1944), Eq. (121): u_c = -sqrt 2 J; with two bonds per site and
    isotropy u = -2 J <s s'>_nn, so the nearest-neighbour correlation at T_c is sqrt 2 / 2.

  diagonal_correlation(n) -> Ball           Perk & Au-Yang's product (n <= 4096)
  diagonal_toeplitz(n) -> Ball              det[(phi)_{i-j}] (DIK Eq. (105)), the integer-ratio
                                            matrix [1/(2(i-j)+1)] eliminated in exact Fractions,
                                            times (2/pi)^n (n <= 60)
  diagonal_amplitude() -> Ball              e^{1/4} 2^{1/12} A^{-3}  (DIK Eq. (106))
  diagonal_asymptote(n) -> Ball             the amplitude n^{-1/4} (1 - 1/(64 n^2))
  diagonal_log_slope(n1, n2) -> Ball        ln[C(n2)/C(n1)] / ln(n2/n1) -> -eta (eta = 1/4)
  critical_nn_correlation() -> Ball         -u_c / 2 (onsager_thermo.onsager_energy(BETA_C))
_HOOKS["product_denominator"] (4: the 1/(4k^2) of Eq. (32); planted 2).
"""
from fractions import Fraction

from ..core import Refusal
from .ball import Ball, certify

_HOOKS = {"product_denominator": 4}
MAX_N, MAX_TOEPLITZ = 4096, 60


def _n(n, top=MAX_N):
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= top:
        raise Refusal("precondition:distance", f"n = {n!r}: need an int in 1 .. {top}")
    return n


def _product(n):
    from flint import arb
    q = _HOOKS["product_denominator"]
    lg = n * (2 / arb.pi()).log()
    for k in range(1, n):
        lg -= (n - k) * (1 - arb(1) / (q * k * k)).log()
    return lg.exp()


def diagonal_correlation(n) -> Ball:
    n = _n(n)
    return certify(lambda: _product(n))


def _cauchy_det(n):
    """det[1/(2(i - j) + 1)]_{i,j < n} by exact Fraction elimination."""
    M = [[Fraction(1, 2 * (i - j) + 1) for j in range(n)] for i in range(n)]
    det = Fraction(1)
    for c in range(n):
        p = next((r for r in range(c, n) if M[r][c]), None)
        if p is None:
            return Fraction(0)
        if p != c:
            M[c], M[p] = M[p], M[c]
            det = -det
        det *= M[c][c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            if f:
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return det


def diagonal_toeplitz(n) -> Ball:
    n = _n(n, MAX_TOEPLITZ)
    d = _cauchy_det(n)

    def thunk():
        from flint import arb
        from .ball import arb_q
        return arb_q(d) * (2 / arb.pi()) ** n
    return certify(thunk)


def _amplitude():
    from flint import arb
    return (arb(1) / 4).exp() * arb(2) ** (arb(1) / 12) / arb.const_glaisher() ** 3


def diagonal_amplitude() -> Ball:
    return certify(_amplitude)


def diagonal_asymptote(n) -> Ball:
    n = _n(n)

    def thunk():
        from flint import arb
        return _amplitude() * arb(n) ** (-arb(1) / 4) * (1 - arb(1) / (64 * n * n))
    return certify(thunk)


def diagonal_log_slope(n1, n2, digits=30) -> Ball:
    n1, n2 = _n(n1), _n(n2)
    if n1 == n2:
        raise Refusal("precondition:distance", "n1 = n2: need two distances")

    def thunk():
        from flint import arb
        return (_product(n2) / _product(n1)).log() / (arb(n2) / n1).log()
    return certify(thunk, digits)


def critical_nn_correlation() -> Ball:
    from .coupling import BETA_C
    from .onsager_thermo import onsager_energy
    return certify(lambda: -onsager_energy(BETA_C).arb / 2)


__all__ = ["MAX_N", "MAX_TOEPLITZ", "critical_nn_correlation", "diagonal_amplitude",
           "diagonal_asymptote", "diagonal_correlation", "diagonal_log_slope", "diagonal_toeplitz"]
