"""The elliptic structure of the Onsager free energy, as an exact operator identity (E8).

From the expansion in series/ising2d.py, g(u) = ln(1 + u) - G(W), G(W) = sum_k binom(2k, k)^2 W^k/(4k),
W = u (1 - u)^2/(1 + u)^4.  With theta_W G = (F - 1)/4, F(z) = sum_k binom(2k, k)^2 (z/16)^k =
2F1(1/2, 1/2; 1; z) (= (2/pi) K(k) at z = k^2, DLMF 19.5.1), z = 16 W, and u W'/W = (u^2 - 6u + 1)/(1 - u^2):
    theta g = R(u) + s(u) F(z(u)),   R = (1 - 3u)/(4(1 - u)),   s = -(u^2 - 6u + 1)/(4(1 - u^2)).
F solves the hypergeometric equation z(1 - z) F'' + (1 - 2z) F' - F/4 = 0 (DLMF 15.10.1 at a = b = 1/2,
c = 1).  z = 16 W is Onsager's modulus squared: k = 2 sinh 2K / cosh^2 2K = 4v(1 - u)/(1 + u)^2.

elliptic_factor_check(op) tests, exactly (rational functions as Fraction coefficient lists):
  (i)   op = M theta (no theta^0 term), M of order 2;
  (ii)  M(s F) = 0, with F represented as A F_z + B F_z' (A, B rational in u) and F_z'' eliminated by
        the hypergeometric equation: A = B = 0 is a sufficient condition;
  (iii) M(R) = op.rhs.
Together: op g = M(theta g) = M(R) + M(s F) = P exactly, i.e. the guessed operator is proved for the
Onsager series (given the hypergeometric equation), and its order-2 factor is the pulled-back
Legendre / complete-elliptic-integral operator.
"""
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class EllipticCheck:
    right_factor_theta: bool
    factor_order: int
    kills_sF: bool
    maps_R_to_rhs: bool

    @property
    def ok(self):
        return self.right_factor_theta and self.factor_order == 2 and self.kills_sF and self.maps_R_to_rhs


_MEMO = {}


def elliptic_factor_check(op):
    """EllipticCheck of (i)-(iii) for an Operator in u (memoised per operator)."""
    key = (frozenset(op.terms.items()), tuple(op.rhs))
    if key not in _MEMO:
        _MEMO[key] = _check(op)
    return _MEMO[key]


# ---- rational functions in u as (numerator, denominator) coefficient lists (exact Fractions) ----
def _trim(p):
    while p and p[-1] == 0:
        p = p[:-1]
    return p


def _pm(p, q):
    if not p or not q:
        return []
    out = [Fraction(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a:
            for j, b in enumerate(q):
                out[i + j] += a * b
    return _trim(out)


def _pa(p, q):
    n = max(len(p), len(q))
    return _trim([(p[i] if i < len(p) else 0) + (q[i] if i < len(q) else 0) for i in range(n)])


def _pd(p):
    return _trim([i * p[i] for i in range(1, len(p))])


def _P(*c):
    return _trim([Fraction(x) for x in c])


class _R:
    __slots__ = ("n", "d")

    def __init__(self, n, d=None):
        self.n, self.d = _trim(list(n)), _trim(list(d)) if d is not None else [Fraction(1)]

    def __add__(self, o):
        if not self.n:
            return o
        if not o.n:
            return self
        if self.d == o.d:
            return _R(_pa(self.n, o.n), self.d)
        return _R(_pa(_pm(self.n, o.d), _pm(o.n, self.d)), _pm(self.d, o.d))

    def __mul__(self, o):
        return _R(_pm(self.n, o.n), _pm(self.d, o.d))

    def deriv(self):
        if not self.n:
            return self
        return _R(_pa(_pm(_pd(self.n), self.d), [-c for c in _pm(self.n, _pd(self.d))]), _pm(self.d, self.d))

    def is_zero(self):
        return not self.n


def _check(op):
    lead = op.coefficient(0)
    if lead or op.order != 3:
        return EllipticCheck(not lead, op.order - 1, False, False)
    P = {k: _R(list(op.coefficient(k))) for k in (1, 2, 3)}
    X = _R(_P(0, 1))
    one_p_x4 = _pm(_pm(_P(1, 1), _P(1, 1)), _pm(_P(1, 1), _P(1, 1)))
    z = _R(_pm(_P(0, 16), _pm(_P(1, -1), _P(1, -1))), one_p_x4)          # 16 u (1 - u)^2 / (1 + u)^4
    zp = z.deriv()
    zz = z * _R(_pa(z.d, [-c for c in z.n]), z.d)                        # z (1 - z)
    c1 = _R(_pm(_P(Fraction(1, 4)), zz.d), zz.n)                         # F_zz = c1 F + c2 F_z
    c2 = _R(_pm(_pa(_pm(_P(-1), z.d), _pm(_P(2), z.n)), zz.d), _pm(zz.n, z.d))

    def theta(pair):                                   # theta (A F + B F_z)
        a, b = pair
        return X * (a.deriv() + b * zp * c1), X * (a * zp + b.deriv() + b * zp * c2)

    def M(pair):
        t1 = theta(pair)
        t2 = theta(t1)
        return tuple(P[3] * t2[i] + P[2] * t1[i] + P[1] * pair[i] for i in range(2))

    s = _R(_P(Fraction(-1, 4), Fraction(6, 4), Fraction(-1, 4)), _P(1, 0, -1))   # -(u^2-6u+1)/(4(1-u^2))
    R = _R(_P(1, -3), _P(4, -4))                                                  # (1 - 3u)/(4(1 - u))
    kills = all(c.is_zero() for c in M((s, _R([]))))
    t1 = X * R.deriv()
    mr = P[3] * (X * t1.deriv()) + P[2] * t1 + P[1] * R
    diff = mr + _R([-c for c in op.rhs]) if op.rhs else mr
    return EllipticCheck(True, 2, kills, diff.is_zero())


__all__ = ["EllipticCheck", "elliptic_factor_check"]
