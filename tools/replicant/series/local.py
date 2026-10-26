"""Local analysis of a theta-form operator at a finite point: exact exponents and logarithms.

  QuadNumber(a, b, d)          a + b sqrt(d) (Fractions, d a squarefree integer; d = 1 or b = 0 is
                               rational), exact field arithmetic in Q(sqrt d)
  local_exponents(op, x0)      LocalExponents(point, regular, indicial, integer_exponents,
                               other_exponents, log_exponents)
  critical_exponent(op, x0)    the smallest local exponent with a non-analytic solution (mpf)

Method.  L = sum_k p_k(x) theta^k is rewritten in d/dx form with Stirling numbers of the second kind
(theta^k = sum_j S(k, j) x^j d^j), shifted to t = x - x0 by exact Taylor expansion in Q(sqrt d), and
written as L = t^m sum_l t^l Q_l(vartheta), vartheta = t d/dt, using t^j d^j = vartheta (vartheta - 1)
... (vartheta - j + 1).  The point is regular singular iff deg Q_0 = order (Fuchs); Q_0 is the
indicial polynomial.  Its integer roots are found exactly (Cauchy bound, exact evaluation, deflation);
the rest numerically (mpmath polyroots, reported, not used for logarithms).  Logarithms among the
integer exponents: the Frobenius recursion Q_0(rho0 + n) c_n = -sum_l Q_l(rho0 + n - l) c_{n-l}
is run from the smallest integer root with every free coefficient kept as a parameter; at a root
rho0 + n where the right-hand side is not identically zero, or at a repeated root, a solution carries
ln t times a solution of leading exponent rho0 + n (Frobenius).  That exponent is recorded.
"""
from dataclasses import dataclass
from fractions import Fraction
from math import isqrt

import mpmath as mp

from ..core import Refusal


def _squarefree(n):
    """(squarefree part with sign, s) with n = part * s^2."""
    s, f, m = 1, 2, abs(n)
    while f * f <= m:
        while m % (f * f) == 0:
            m //= f * f
            s *= f
        f += 1
    return m * (1 if n > 0 else -1), s


class QuadNumber:
    __slots__ = ("a", "b", "d")

    def __init__(self, a, b=0, d=1):
        a, b = Fraction(a), Fraction(b)
        if d == 0:
            raise ValueError("d = 0")
        dd, s = _squarefree(d)
        b *= s
        if dd == 1 or b == 0:
            a, b, dd = a + (b if dd == 1 else 0), Fraction(0), 1
        self.a, self.b, self.d = a, b, dd

    def _co(self, o):
        o = o if isinstance(o, QuadNumber) else QuadNumber(o)
        if o.b and self.b and o.d != self.d:
            raise ValueError("different quadratic fields")
        return o, (self.d if self.b else o.d)

    def __add__(self, o):
        o, d = self._co(o)
        return QuadNumber(self.a + o.a, self.b + o.b, d)
    __radd__ = __add__

    def __neg__(self):
        return QuadNumber(-self.a, -self.b, self.d)

    def __sub__(self, o):
        return self + (-(o if isinstance(o, QuadNumber) else QuadNumber(o)))

    def __rsub__(self, o):
        return QuadNumber(o) - self

    def __mul__(self, o):
        o, d = self._co(o)
        return QuadNumber(self.a * o.a + self.b * o.b * d, self.a * o.b + self.b * o.a, d)
    __rmul__ = __mul__

    def __truediv__(self, o):
        o, d = self._co(o)
        n = o.a * o.a - o.b * o.b * d
        if n == 0:
            raise ZeroDivisionError("QuadNumber division by zero")
        return self * QuadNumber(o.a / n, -o.b / n, d)

    def __eq__(self, o):
        if not isinstance(o, QuadNumber):
            try:
                o = QuadNumber(o)
            except (TypeError, ValueError):
                return NotImplemented
        return (self.a, self.b, self.d if self.b else 1) == (o.a, o.b, o.d if o.b else 1)

    def __hash__(self):
        return hash(self.a) if self.b == 0 else hash((self.a, self.b, self.d))

    def __bool__(self):
        return bool(self.a or self.b)

    def is_integer(self):
        return self.b == 0 and self.a.denominator == 1

    def mpf(self):
        """Numeric value (mpf, or mpc for d < 0) at the working precision."""
        r = mp.mpf(self.a.numerator) / self.a.denominator
        if not self.b:
            return r
        return r + mp.mpf(self.b.numerator) / self.b.denominator * mp.sqrt(self.d)

    def __repr__(self):
        return str(self.a) if not self.b else f"{self.a} + {self.b}*sqrt({self.d})"


def roots_of(coeffs):
    """Exact roots of a rational polynomial of degree 1 or 2 (coeffs ascending): [QuadNumber]."""
    c = [Fraction(x) for x in coeffs]
    if len(c) == 2:
        return [QuadNumber(-c[0] / c[1])]
    if len(c) != 3:
        raise ValueError("degree 1 or 2 only")
    disc = c[1] * c[1] - 4 * c[2] * c[0]
    num, den = disc.numerator * disc.denominator, disc.denominator * disc.denominator
    r = isqrt(abs(num)) if num >= 0 else -1
    if num >= 0 and r * r == num:
        s = Fraction(r, disc.denominator)
        return [QuadNumber((-c[1] + s) / (2 * c[2])), QuadNumber((-c[1] - s) / (2 * c[2]))]
    return [QuadNumber(-c[1] / (2 * c[2]), sg / (2 * c[2] * den), num) for sg in (1, -1)]


# ---- polynomials with QuadNumber coefficients (ascending lists) --------------------------
def _ev(p, x):
    out = QuadNumber(0)
    for c in reversed(p):
        out = out * x + c
    return out


def _stirling2(k, j):
    S = [[1]]
    for n in range(1, k + 1):
        S.append([0] * (n + 1))
        for i in range(1, n + 1):
            S[n][i] = (S[n - 1][i] if i < n else 0) * i + S[n - 1][i - 1]
    return S[k][j] if j <= k else 0


def _falling(rho, j):
    out = QuadNumber(1)
    for i in range(j):
        out = out * (rho - i)
    return out


def _local_form(op, x0):
    """(m, {l: Q_l as dict j -> coefficient}) with L = t^m sum_l t^l Q_l(vartheta)."""
    s = op.order
    q = {}                                               # d/dx form: q_j(x) ascending
    for (i, k), c in op.terms.items():
        for j in range(1, k + 1) if k else (0,):
            st = _stirling2(k, j) if k else 1
            q.setdefault(j, {})
            q[j][i + j] = q[j].get(i + j, 0) + c * st
    e = {}                                               # e[(i, j)]: t^i d^j
    for j, poly in q.items():
        deg = max(poly)
        coeffs = [Fraction(poly.get(n, 0)) for n in range(deg + 1)]
        e.update(_taylor(coeffs, x0, j))
    e = {k: v for k, v in e.items() if v}
    m = min(i - j for i, j in e)
    Q = {}
    for (i, j), c in e.items():
        Q.setdefault(i - j - m, {})[j] = c
    if s not in Q.get(0, {}):
        return m, Q, False
    return m, Q, True


def _taylor(coeffs, x0, j):
    """{(i, j): coefficient of t^i} of the polynomial coeffs(x0 + t) (repeated synthetic division)."""
    out, p = {}, [QuadNumber(c) for c in coeffs]
    i = 0
    while p:
        r, q = QuadNumber(0), []
        for c in reversed(p):
            r = r * x0 + c
            q.append(r)
        out[(i, j)] = q.pop()                             # remainder = p(x0)
        p = list(reversed(q))
        i += 1
    return out


def _Qeval(Ql, rho):
    return sum((c * _falling(rho, j) for j, c in Ql.items()), QuadNumber(0))


@dataclass(frozen=True)
class LocalExponents:
    point: QuadNumber
    regular: bool
    indicial: tuple                 # Q_0 coefficients ascending in rho (QuadNumber)
    integer_exponents: tuple        # with multiplicity, ascending
    other_exponents: tuple          # numeric (mpf / mpc), not analysed for logarithms
    log_exponents: tuple            # leading exponents of the solutions multiplied by ln t


def _indicial_poly(Q0, s):
    """Q_0(rho) = sum_j c_j rho(rho-1)..(rho-j+1) as ascending coefficients."""
    poly = [QuadNumber(0)] * (s + 1)
    for j, c in Q0.items():
        f = [QuadNumber(1)]
        for i in range(j):                                # multiply by (rho - i)
            f = [QuadNumber(0)] + f
            for n in range(len(f) - 1):
                f[n] = f[n] - f[n + 1] * i
        for n, x in enumerate(f):
            poly[n] = poly[n] + c * x
    return poly


def _deflate(p, r):
    q, acc = [], QuadNumber(0)
    for c in reversed(p):
        acc = acc * r + c
        q.append(acc)
    q.pop()
    return list(reversed(q))


def local_exponents(op, x0):
    x0 = x0 if isinstance(x0, QuadNumber) else QuadNumber(x0)
    m, Q, regular = _local_form(op, x0)
    if not regular:
        return LocalExponents(x0, False, (), (), (), ())
    s = op.order
    p = _indicial_poly(Q[0], s)
    with mp.workdps(30):
        lead = abs(p[-1].mpf())
        bound = 1 + int(max(abs(c.mpf()) for c in p[:-1]) / lead) if s else 0
    ints, rest = [], list(p)
    for n in range(-bound, bound + 1):
        while len(rest) > 1 and not _ev(rest, QuadNumber(n)):
            ints.append(n)
            rest = _deflate(rest, QuadNumber(n))
    other = ()
    if len(rest) > 1:
        with mp.workdps(30):
            other = tuple(mp.polyroots([c.mpf() for c in reversed(rest)], maxsteps=200, extraprec=60))
    return LocalExponents(x0, True, tuple(p), tuple(sorted(ints)), other, _logs(Q, sorted(ints)))


def _logs(Q, ints):
    if not ints:
        return ()
    rho0, top = ints[0], ints[-1]
    mult = {r: ints.count(r) for r in ints}
    params = 0
    c = []                                                # c[n]: list of coefficients per parameter
    logs = []
    for n in range(top - rho0 + 1):
        rhs = [QuadNumber(0)] * params
        for l in range(1, n + 1):
            Ql = Q.get(l)
            if Ql is None:
                continue
            f = _Qeval(Ql, QuadNumber(rho0 + n - l))
            if f:
                for k, x in enumerate(c[n - l]):
                    rhs[k] = rhs[k] - f * x
        q0 = _Qeval(Q[0], QuadNumber(rho0 + n))
        if q0:
            c.append([x / q0 for x in rhs])
            continue
        if any(rhs) or mult[rho0 + n] > 1:
            logs.append(rho0 + n)
        piv = next((k for k, x in enumerate(rhs) if x), None)
        if piv is not None:                               # impose rhs . params = 0
            sub = [-x / rhs[piv] for x in rhs]
            sub[piv] = QuadNumber(0)
            c = [[(x + row[piv] * sub[k]) if k != piv else QuadNumber(0)
                  for k, x in enumerate(row)] for row in c]
        c = [row + [QuadNumber(0)] for row in c]
        params += 1
        c.append([QuadNumber(0)] * (params - 1) + [QuadNumber(1)])
    return tuple(logs)


def critical_exponent(op, x0):
    """Smallest local exponent at x0 carrying a non-analytic solution: a negative integer
    exponent, an integer exponent multiplied by ln t, or a real non-integer exponent (mpf).
    Refused when x0 is not regular singular or every local solution is analytic."""
    loc = local_exponents(op, x0)
    if not loc.regular:
        raise Refusal("unsupported:irregular-singularity", f"x0 = {loc.point}: deg Q_0 < order")
    cands = [mp.mpf(r) for r in loc.integer_exponents if r < 0] + \
            [mp.mpf(r) for r in loc.log_exponents] + \
            [mp.re(r) for r in loc.other_exponents if abs(mp.im(r)) < mp.mpf("1e-20")]
    if not cands:
        raise Refusal("no-solution:analytic-point", f"x0 = {loc.point}: exponents "
                      f"{loc.integer_exponents}, no logarithm: every local solution is analytic")
    return min(cands)


__all__ = ["LocalExponents", "QuadNumber", "critical_exponent", "local_exponents", "roots_of"]
