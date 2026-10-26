"""Exact high-temperature series of the square-lattice Ising free energy (E8), two routes.

Convention: isotropic nearest-neighbour coupling, zero field, K = beta J, v = tanh K, u = v^2.
    g(v) = ln Z/N - ln 2 - 2 ln cosh K = sum_n a_n v^n        (a_n = 0 for odd n; a_4 = 1)
is the high-temperature free energy (Z = 2^N cosh^{2N} K sum over even subgraphs of v^|E|).  Only
v (default) and u = v^2 are offered; any other variable is refused (unsupported:variable).

Route 1, the Onsager double integral (Onsager 1944, as printed by Duminil-Copin 2022, arXiv:2208.00864,
Eq. (3.6)): -beta f = ln 2 + (1/2) <ln[cosh^2 2K - sinh 2K (cos t1 + cos t2)]>, <.> the average over
the two angles.  With cosh 2K = (1 + u)/(1 - u), sinh 2K = 2v/(1 - u) and 2 ln cosh K = -ln(1 - u):
    g = ln(1 + u) + (1/2) <ln(1 - y C)>,   y = 2v (1 - u)/(1 + u)^2,   C = cos t1 + cos t2
      = ln(1 + u) - sum_{k >= 1} [4^k M_2k / (4k)] W^k,   W = y^2/4 = u (1 - u)^2 / (1 + u)^4,
M_m = <C^m> integrated exactly: <cos^j t> = binom(j, j/2)/2^j (j even, else 0) from the binomial
expansion of ((e^{it} + e^{-it})/2)^j, and M_m = sum_j binom(m, j) <cos^j><cos^(m-j)> (the hook
_HOOKS["moment"], so tests can plant an error).  The u-series is assembled by Horner in W with
integer arithmetic over one common denominator.

Route 2, independent of the integral: the even-subgraph polynomial P_L(v) of the L x L torus by a
parity transfer matrix (state: the occupied horizontal bonds crossing a column cut; inside a column
the vertical bonds are fixed by the site parities up to complement, weight v^|e| + v^(L - |e|)).
Window (derived here, verified in tests/test_series.py for L = 3 .. 8).  ln P_L is a sum over
connected clusters of polygon components; a cluster of total length k that wraps nothing has linear
extent <= k/2 and embeds as in the plane when k/2 < L.  So (1/L^2) ln P_L = g for every power v^k with
k < L; at k = L the 2L straight winding loops add exactly 2/L.  For ODD L more is true for EVEN k: a
closed even subgraph in homology class (a, b) mod 2 has length = (a + b) L = a + b mod 2, and
contractible cycles have even length (they lift to the plane), so every odd component winds (length
>= L) and an even-length cluster needs two of them or one of class (1, 1) (length >= 2L): the even
coefficients agree through v^(2L - 2), and differ first at v^(2L) (L = 7: a4 .. a12).  States 2^L:
L <= 8.  No lattice of this size constrains the recognised operator, which needs >= 62 u-terms
(v^122) at its (degree, order) = (8, 3): the operator rests on the exact identity of series/elliptic.py
and the held-out terms, not on this route.
"""
from fractions import Fraction
from math import comb, lcm

import mpmath as mp
import numpy as np

from ..core import Refusal

VARIABLES = ("v", "u")
MAX_ORDER = 600                     # in v (300 terms in u)
TORUS_SIZES = range(3, 9)


def trig_moment(j):
    """<cos^j t> over a period, exactly: binom(j, j/2) / 2^j for even j, 0 for odd j."""
    return Fraction(comb(j, j // 2), 2**j) if j % 2 == 0 else Fraction(0)


def onsager_moment(m):
    """<(cos t1 + cos t2)^m> over both angles, exactly (binomial expansion, independent angles)."""
    return sum((comb(m, j) * trig_moment(j) * trig_moment(m - j) for j in range(m + 1)), Fraction(0))


_HOOKS = {"moment": onsager_moment}
_CACHE = {}
_TRANSFER = {}
_TORUS = {}


def _int(x, name, lo, hi):
    if isinstance(x, bool) or not isinstance(x, int) or not lo <= x <= hi:
        raise Refusal(f"precondition:{name}", f"{name}={x!r}: need an int in [{lo}, {hi}]")
    return x


def _u_series(n):
    """b_0 .. b_{n-1} of g in u (route 1); cached per moment hook (prefix-stable)."""
    hook = _HOOKS["moment"]
    got = _CACHE.get(hook)
    if got is not None and len(got) >= n:
        return list(got[:n])
    N = n - 1
    # W = u (1 - u)^2 (1 + u)^-4, integer coefficients up to u^N
    inv = [(-1)**j * comb(j + 3, 3) for j in range(N + 1)]
    num = [1, -2, 1] + [0] * N
    w = [0] + [sum(num[i] * inv[k - i] for i in range(min(k, 2) + 1)) for k in range(N)]
    c = [Fraction(0)] + [Fraction(4**k) * hook(2 * k) / (4 * k) for k in range(1, N + 1)]
    D = 1
    for x in c:
        D = lcm(D, x.denominator)
    ci = [int(x * D) for x in c]
    acc = [0] * (N + 1)                               # Horner: acc = c_k + W acc, then W acc
    for k in range(N, 0, -1):
        new = [0] * (N + 1)
        for i in range(1, N + 1):
            if w[i]:
                for j in range(N + 1 - i):
                    if acc[j]:
                        new[i + j] += w[i] * acc[j]
        new[0] += ci[k]
        acc = new
    G = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(N + 1 - i):
            G[i + j] += w[i] * acc[j]
    out = [Fraction(0)] + [Fraction((-1)**(m + 1), m) - Fraction(G[m], D) for m in range(1, N + 1)]
    _CACHE[hook] = tuple(out)
    return out


def ising2d_free_energy_series(order, variable="v"):
    """[a_0, ..., a_order] (Fractions) of g(v) = ln Z/N - ln 2 - 2 ln cosh K in v = tanh K, from
    the Onsager integral expanded exactly; variable="u": [b_0, ..., b_order] in u = v^2."""
    if variable not in VARIABLES:
        raise Refusal("unsupported:variable", f"variable {variable!r}: the series is offered in "
                      "v = tanh K (default) or u = v^2 only")
    lim = MAX_ORDER if variable == "v" else MAX_ORDER // 2
    order = _int(order, "order", 0, lim)
    if variable == "u":
        return _u_series(order + 1)
    b = _u_series(order // 2 + 1)
    return [b[n // 2] if n % 2 == 0 else Fraction(0) for n in range(order + 1)]


# ---- route 2: the exact L x L torus -----------------------------------------------------
def _transfer(L, D):
    """[D + 1, 2^L, 2^L] float array: coefficient of v^d in T(b, b') (cached per (L, D))."""
    if (L, D) in _TRANSFER:
        return _TRANSFER[(L, D)]
    S = 1 << L
    b = np.arange(S)
    x = b[:, None] ^ b[None, :]                           # parity b_i + b'_i of each site
    bits = (x[..., None] >> np.arange(L)) & 1
    run = np.cumsum(bits[..., 1:], axis=-1) % 2           # vertical bond i joins sites i, i+1
    e = run.sum(axis=-1)                                  # |e| with e_0 = 0; the complement: L - |e|
    w = ((b[None, :, None] >> np.arange(L)) & 1).sum(axis=-1)    # |b'|
    ok = bits.sum(axis=-1) % 2 == 0
    T = np.zeros((D + 1, S, S))
    for deg in (w + e, w + L - e):
        for d in range(D + 1):
            T[d] += ok & (deg == d)
    _TRANSFER[(L, D)] = T
    return T


def _pmul(A, B, D):
    out = np.zeros_like(A)
    for i in range(D + 1):
        for j in range(D + 1 - i):
            out[i + j] += A[i] @ B[j]
    if out.max() >= 2.0**52:
        raise Refusal("precondition:torus-size", "even-subgraph counts beyond exact float range")
    return out


def torus_polynomial(L, order):
    """[p_0, ..., p_order]: even subgraphs of the L x L torus with k edges (exact integers)."""
    _int(L, "torus-size", TORUS_SIZES.start, TORUS_SIZES.stop - 1)
    D = _int(order, "order", 0, 2 * L)
    T, R, e = _transfer(L, D), None, L
    while e:                                              # T^L by binary powers
        if e & 1:
            R = T if R is None else _pmul(R, T, D)
        e >>= 1
        if e:
            T = _pmul(T, T, D)
    return [int(round(np.trace(R[d]))) for d in range(D + 1)]


def torus_free_energy_series(L, order):
    """[c_0, ..., c_order]: (1/L^2) ln P_L(v) (P_L = torus_polynomial), exact Fractions."""
    if (L, order) in _TORUS:
        return list(_TORUS[(L, order)])
    p = torus_polynomial(L, order)
    q = [Fraction(0)] * (order + 1)                      # q = ln p, p_0 = 1: k q_k = k p_k - ...
    for k in range(1, order + 1):
        q[k] = p[k] - sum((j * q[j] * p[k - j] for j in range(1, k)), Fraction(0)) / k
    _TORUS[(L, order)] = tuple(x / (L * L) for x in q)
    return list(_TORUS[(L, order)])


def torus_window(L):
    """(all, even): the L x L torus equals the bulk series for every v^k with k <= all = L - 1, and for
    even k <= even (2L - 2 for odd L, L - 2 for even L, where v^L itself differs)."""
    L = _int(L, "torus-size", TORUS_SIZES.start, TORUS_SIZES.stop - 1)
    return L - 1, (2 * L - 2 if L % 2 else L - 2)




def onsager_free_energy_numeric(v, dps=20):
    """g(v) from the Onsager integral by quadrature (mpf; |v| < sqrt 2 - 1): the inner angle done
    exactly, <ln(a - b cos t)> = ln((a + sqrt(a^2 - b^2))/2), the outer one by mp.quad.  A numerical
    route used by the LaTeX consistency checks; it shares no code with the exact expansion."""
    with mp.workdps(dps + 10):
        v = mp.mpf(v)
        c2, s2 = (1 + v * v) / (1 - v * v), 2 * v / (1 - v * v)

        def inner(t):
            a, b = c2 * c2 - s2 * mp.cos(t), s2
            return mp.log((a + mp.sqrt(a * a - b * b)) / 2)
        lnz = mp.log(2) + mp.quad(inner, [0, mp.pi]) / (2 * mp.pi)
        return +(lnz - mp.log(2) - 2 * mp.log(mp.cosh(mp.atanh(v))))


__all__ = ["MAX_ORDER", "VARIABLES", "ising2d_free_energy_series", "onsager_free_energy_numeric",
           "onsager_moment", "torus_free_energy_series", "torus_polynomial", "torus_window", "trig_moment"]
