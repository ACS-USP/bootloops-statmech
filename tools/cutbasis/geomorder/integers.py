# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of BootLoops 1.0; see LICENSE and NOTICE beside this file.
"""The ordering integers of a form omega_mu[Q] on CP^n (arXiv:2511.15381 sect. 3.5-3.6):
    |mu| = sum_i mu_i ;  r = number of consecutive non-zero residues along EVEN divisors ;  o = pole order of
    omega^0 = Q / prod_i P_i^{c_i} eta  with c_i = mu_i (even P_i) or mu_i + 1/2 (odd P_i), after blow-ups at
    non-normal crossings (floor rule, additive at normal crossings, max over points) ;  w = n + r.

n = 0: exact.  n = 1: exact at symbolic kinematics (`_n1`); at numeric kinematics, the path the selector runs, one
(kinematics, prime) sample (`_n1_modp`).  For n = 2 the local analysis (points, multiplicities, blow-ups, residues on curves) is done at random
integer kinematics modulo word-size primes p chosen so that all relevant points are F_p-rational; every verdict is the
consensus of NSAMPLES independent (kinematics, prime) samples (two independent primes).  n >= 3: |mu| exact,
r and o computed with the normal-crossing assumption and FLAGGED (not certified).
"""
from __future__ import annotations
import random
from math import comb
from dataclasses import dataclass, field
from math import floor
import sympy as sp
try:
    import flint
except ImportError:   # pragma: no cover
    flint = None
from .baikov import BaikovRep
from .iota import FormTerm

NSAMPLES = 2
def _prime_pool(n=150, start=2147483647):
    out = []; q = start
    while len(out) < n:
        if sp.isprime(q): out.append(q)
        q -= 2
    return out


PRIMES_SMALLSET = [2147483647, 2147483629, 2147483587, 2147483579, 2147483563, 2147483549, 2147483543, 2147483497,
          2147483489, 2147483477, 2147483423, 2147483399, 2147483353, 2147483323, 2147483269, 2147483249,
          2147483237, 2147483179, 2147483171, 2147483137, 2147483123, 2147483077, 2147483069, 2147483059]
PRIMES = _prime_pool()


@dataclass
class Integers:
    absmu: int
    r: int
    o: int
    w: int
    n: int
    flags: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def tup(self):
        return (self.w, self.o, self.absmu)


# ----------------------------------------------------------------------------------------------------------------
# symbolic helpers
# ----------------------------------------------------------------------------------------------------------------

def form_data(rep: BaikovRep, term: FormTerm):
    """return list of (P_hom, c_i, parity, mu_i) for i=0..ND and Q_hom"""
    out = [(rep.z0, sp.Rational(term.mu[0]) + (sp.Rational(1, 2) if rep.a0 == -1 else 0), 'odd' if rep.a0 == -1 else 'even', term.mu[0])]
    for d, m in zip(rep.divisors, term.mu[1:]):
        c = sp.Rational(m) + (sp.Rational(1, 2) if d.parity == 'odd' else 0)
        out.append((d.P_hom, c, d.parity, m))
    return out, term.Q_hom


# ----------------------------------------------------------------------------------------------------------------
# n = 1 : exact
# ----------------------------------------------------------------------------------------------------------------

def _n1(rep, term):
    divs, Q = form_data(rep, term)
    z0, z1 = rep.z0, rep.zsyms[0]
    absmu = sum(term.mu)
    o = max([int(floor(c)) for (_, c, _, m) in divs] + [0])
    r = 0
    detail = {}
    for idx, (P, c, par, m) in enumerate(divs):
        if par != 'even' or m < 1:
            continue
        if m == 1:
            r = 1; detail[f'P{idx}'] = 'simple pole: residue != 0'; break
        # higher pole along an even point: residue test. chart: if idx==0 use t=z0 with z1=1 else t=z1 with z0=1
        if idx == 0:
            var, sub = z0, {z1: 1}
        else:
            var, sub = z1, {z0: 1}
        p = sp.Poly(sp.expand(P.subs(sub)), var)
        alpha = sp.Symbol('alpha')
        # h = (p(var) - p(alpha))/(var - alpha)  exact polynomial
        h = sp.quo(sp.expand(p.as_expr() - p.as_expr().subs(var, alpha)), var - alpha, var)
        g = sp.expand(Q.subs(sub))
        for jdx, (Pj, cj, parj, mj) in enumerate(divs):
            if jdx == idx or cj == 0:
                continue
            g = g * sp.expand(Pj.subs(sub)) ** (-cj)
        expr = g * h ** (-m)
        der = sp.diff(expr, var, int(m) - 1)
        val = sp.together(der.subs(var, alpha))
        num, den = sp.fraction(val)
        # num may contain fractional powers of Pj(alpha): factor them out — collect as polynomial in alpha times algebraic unit
        num = sp.expand(sp.powsimp(num, force=True))
        # replace Pj(alpha)^(half-integers) by symbols to isolate the polynomial part
        reps = {}
        for jdx, (Pj, cj, parj, mj) in enumerate(divs):
            if jdx == idx:
                continue
            base = sp.expand(Pj.subs(sub).subs(var, alpha))
            reps[base] = sp.Symbol(f'B{jdx}')
        numr = num
        for base, sym in reps.items():
            numr = numr.replace(lambda e: e.is_Pow and sp.expand(e.base) == base, lambda e: sym ** e.exp)
        # numr is now polynomial in alpha and B-symbols with possibly fractional exponents; zero test modulo p(alpha):
        # evaluate B symbols can't vanish at alpha (distinct divisors) -> treat as invertible: test each coefficient? simpler:
        # substitute B_j -> Pj(alpha) back only for INTEGER powers; fractional powers factor out globally if a single monomial in B's
        numr = sp.expand(numr)
        polyB = sp.Poly(numr, *reps.values()) if reps else None
        zero = True
        if polyB is None:
            rem = sp.rem(sp.expand(numr), p.as_expr().subs(var, alpha), alpha)
            zero = sp.expand(rem) == 0
        else:
            # group monomials by fractional parts of exponents; within a group substitute back and reduce mod p(alpha)
            groups = {}
            for monom, coeff in zip(polyB.monoms(), polyB.coeffs()):
                keyf = tuple(e - floor(e) for e in monom)
                term_expr = coeff
                for (base, sym), e in zip(reps.items(), monom):
                    term_expr = term_expr * base ** int(floor(e))
                groups[keyf] = groups.get(keyf, 0) + term_expr
            for keyf, ge in groups.items():
                gnum, gden = sp.fraction(sp.together(ge))
                rem = sp.rem(sp.expand(gnum), sp.expand(p.as_expr().subs(var, alpha)), alpha)
                if sp.expand(rem) != 0:
                    zero = False; break
        if not zero:
            r = 1; detail[f'P{idx}'] = f'pole order {m}: residue != 0'; break
        else:
            detail[f'P{idx}'] = f'pole order {m}: residue == 0'
    return Integers(absmu, r, o, 1 + r, 1, [], detail)


def _n1_modp(rep, term, seed=1):
    """n=1 at NUMERIC kinematics: r via Laurent expansion mod p at the roots of each even divisor with a pole."""
    divs, Q = form_data(rep, term)
    z0, z1 = rep.z0, rep.zsyms[0]
    absmu = sum(term.mu)
    o = max([int(floor(c)) for (_, c, _, m) in divs] + [0])
    rng = random.Random(seed)
    detail = {}
    r = 0
    for idx, (P, c, par, m) in enumerate(divs):
        if par != 'even' or m < 1:
            continue
        if m == 1:
            r = 1; detail[f'P{idx}'] = 'simple pole'; break
        # chart
        if idx == 0:
            var, sub, pts_fixed = z0, {z1: 1}, [0]
        else:
            var, sub, pts_fixed = z1, {z0: 1}, None
        polys = []
        for jdx, (Pj, cj, parj, mj) in enumerate(divs):
            e = sp.Poly(sp.expand(Pj.subs(sub)), var)
            polys.append(([int(sp.Rational(cc).p) * 1 for cc in []], e, cj))
        Qc = sp.Poly(sp.expand(Q.subs(sub)), var) if sp.expand(Q.subs(sub)) != 0 else None
        if Qc is None:
            continue
        found = None
        for ptry in rng.sample(PRIMES, len(PRIMES)):
            def tomod(poly):
                cs = [sp.Rational(cc) for cc in reversed(poly.all_coeffs())]
                try:
                    return [(int(cc.p) * pow(int(cc.q), -1, ptry)) % ptry for cc in cs]
                except ValueError:
                    return None
            Pi_mod = tomod(polys[idx][1])
            if Pi_mod is None: continue
            if pts_fixed is not None:
                roots = pts_fixed
            else:
                if not _splits(Pi_mod, ptry):
                    continue
                roots = [rt for rt, mult in (_roots_mod(Pi_mod, ptry) or [])]
            ok = True
            nonzero = False
            K = int(2 * m + 4)
            for rt in roots:
                total = sp.Rational(0); pieces = []
                for jdx, (_, pj, cj) in enumerate(polys):
                    if cj == 0: continue
                    cm = tomod(pj)
                    if cm is None: ok = False; break
                    pl = P2(ptry, {(i, 0): v for i, v in enumerate(cm)}).shift(rt, 0)
                    oj = _ord_univ(pl, 0)
                    total += sp.Rational(cj) * oj
                    pieces.append((pl, -sp.Rational(cj)))
                if not ok: break
                qm = tomod(Qc)
                if qm is None: ok = False; break
                ql = P2(ptry, {(i, 0): v for i, v in enumerate(qm)}).shift(rt, 0)
                total -= _ord_univ(ql, 0)
                if total.q != 1 or total < 1:
                    continue
                ser = _unit_series(ql, 0, ptry, K)
                for pl, ej in pieces:
                    ser = _mul_series(ser, _pow_series(_unit_series(pl, 0, ptry, K), ej, ptry, K), ptry, K)
                coeff = ser[int(total) - 1] if int(total) - 1 <= K else 0
                if coeff % ptry:
                    nonzero = True; break
            if not ok:
                continue
            found = nonzero
            break
        if found is None:
            raise RuntimeError('n=1 mod-p residue test: no usable prime')
        if found:
            r = 1; detail[f'P{idx}'] = f'pole {m}: residue != 0 (mod p)'; break
        detail[f'P{idx}'] = f'pole {m}: residue == 0 (mod p)'
    return Integers(absmu, r, o, 1 + r, 1, [], detail)


# ----------------------------------------------------------------------------------------------------------------
# n = 2 : local analysis mod p
# ----------------------------------------------------------------------------------------------------------------

class P2:
    """sparse bivariate polynomial over Z/p: dict (i,j)->c"""
    __slots__ = ('p', 'd')

    def __init__(self, p, d=None):
        self.p = p; self.d = {k: v % p for k, v in (d or {}).items() if v % p}

    @staticmethod
    def from_sympy(expr, x, y, p, sub=None):
        e = sp.expand(expr.subs(sub or {}))
        P = sp.Poly(e, x, y)
        d = {}
        for (i, j), c in zip(P.monoms(), P.coeffs()):
            c = sp.Rational(c)
            d[(i, j)] = (int(c.p) * pow(int(c.q), -1, p)) % p
        return P2(p, d)

    def copy(self): return P2(self.p, dict(self.d))
    def is_zero(self): return not self.d

    def __add__(self, o):
        d = dict(self.d)
        for k, v in o.d.items(): d[k] = (d.get(k, 0) + v) % self.p
        return P2(self.p, d)

    def __sub__(self, o):
        d = dict(self.d)
        for k, v in o.d.items(): d[k] = (d.get(k, 0) - v) % self.p
        return P2(self.p, d)

    def __mul__(self, o):
        if not isinstance(o, P2):
            o = int(o)
            return P2(self.p, {k: v * o for k, v in self.d.items()})
        d = {}
        for (i, j), a in self.d.items():
            for (k, l), b in o.d.items():
                d[(i + k, j + l)] = (d.get((i + k, j + l), 0) + a * b) % self.p
        return P2(self.p, d)

    def pow(self, n):
        r = P2(self.p, {(0, 0): 1}); b = self
        while n:
            if n & 1: r = r * b
            b = b * b; n >>= 1
        return r

    def trunc(self, K):
        return P2(self.p, {k: v for k, v in self.d.items() if k[0] + k[1] <= K})

    def ord(self):
        return min((i + j for (i, j) in self.d), default=10 ** 9)

    def lowest(self):
        o = self.ord(); return P2(self.p, {k: v for k, v in self.d.items() if k[0] + k[1] == o})

    def eval(self, x, y):
        p = self.p; return sum(c * pow(x, i, p) * pow(y, j, p) for (i, j), c in self.d.items()) % p

    def shift(self, a, b):
        """poly(x+a, y+b)"""
        p = self.p
        res = {}
        for (i, j), c in self.d.items():
            for s in range(i + 1):
                cx = comb(i, s) * pow(a, i - s, p) % p
                if not cx: continue
                for t in range(j + 1):
                    cy = comb(j, t) * pow(b, j - t, p) % p
                    if not cy: continue
                    res[(s, t)] = (res.get((s, t), 0) + c * cx * cy) % p
        return P2(p, res)

    def subs_linear(self, a11, a12, a21, a22):
        """poly(a11 x + a12 y, a21 x + a22 y)"""
        X = P2(self.p, {(1, 0): a11, (0, 1): a12}); Y = P2(self.p, {(1, 0): a21, (0, 1): a22})
        res = P2(self.p)
        # Horner-free: accumulate monomials
        cacheX = {0: P2(self.p, {(0, 0): 1})}; cacheY = {0: P2(self.p, {(0, 0): 1})}
        for (i, j), c in self.d.items():
            if i not in cacheX: cacheX[i] = X.pow(i)
            if j not in cacheY: cacheY[j] = Y.pow(j)
            res = res + (cacheX[i] * cacheY[j]) * c
        return res

    def blowup_x(self):
        """chart x = u, y = u v : returns poly in (u,v)"""
        d = {}
        for (i, j), c in self.d.items(): d[(i + j, j)] = (d.get((i + j, j), 0) + c) % self.p
        return P2(self.p, d)

    def blowup_y(self):
        """chart x = u v, y = v : poly in (u,v)"""
        d = {}
        for (i, j), c in self.d.items(): d[(i, i + j)] = (d.get((i, i + j), 0) + c) % self.p
        return P2(self.p, d)

    def div_xpow(self):
        """divide by the maximal power of x; return (k, quotient)"""
        if not self.d: return 0, self
        k = min(i for (i, j) in self.d)
        return k, P2(self.p, {(i - k, j): c for (i, j), c in self.d.items()})

    def div_ypow(self):
        if not self.d: return 0, self
        k = min(j for (i, j) in self.d)
        return k, P2(self.p, {(i, j - k): c for (i, j), c in self.d.items()})

    def univ_y0(self):
        """coefficients of poly(x, 0) as list"""
        deg = max((i for (i, j) in self.d if j == 0), default=-1)
        return [self.d.get((i, 0), 0) for i in range(deg + 1)]

    def deriv(self, var):
        d = {}
        for (i, j), c in self.d.items():
            if var == 0 and i > 0: d[(i - 1, j)] = (d.get((i - 1, j), 0) + c * i) % self.p
            if var == 1 and j > 0: d[(i, j - 1)] = (d.get((i, j - 1), 0) + c * j) % self.p
        return P2(self.p, d)

    def to_sympy(self, x, y):
        return sum(sp.Integer(c) * x ** i * y ** j for (i, j), c in self.d.items())


def _roots_mod(coeffs, p):
    """roots in F_p of univariate poly with coeff list (low->high). returns list of (root, multiplicity); None if poly == 0"""
    while coeffs and coeffs[-1] % p == 0:
        coeffs = coeffs[:-1]
    if not coeffs:
        return None
    if len(coeffs) == 1:
        return []
    f = flint.nmod_poly(coeffs, p)
    out = []
    for fac, m in f.factor()[1]:
        if fac.degree() == 1:
            c0, c1 = int(fac[0]), int(fac[1])
            out.append(((-c0 * pow(c1, -1, p)) % p, m))
    return out


def _splits(coeffs, p):
    """True if the univariate poly splits into linear factors mod p"""
    while coeffs and coeffs[-1] % p == 0:
        coeffs = coeffs[:-1]
    if not coeffs or len(coeffs) == 1:
        return True
    f = flint.nmod_poly(coeffs, p)
    return all(fac.degree() == 1 for fac, m in f.factor()[1])


# --- local pole order with blow-ups --------------------------------------------------------------------------

def _local_order(curves, Qloc, p, depth=0, maxdepth=12, log=None):
    """curves: list of (P2 poly vanishing at origin? not necessarily, c exponent) in local affine coords centered at the
    point under study (origin).  Qloc: P2 numerator.  The form is Qloc/prod P^c dx dy.  Returns the pole order at the
    origin (max over infinitely-near points after blow-ups).  Divisors not through the origin are units -> dropped."""
    through = [(P, c) for (P, c) in curves if P.eval(0, 0) == 0 and c != 0]
    if not through:
        return 0
    mults = [P.ord() for P, c in through]
    qord = Qloc.ord() if not Qloc.is_zero() else 0
    smooth = all(m == 1 for m in mults)
    if len(through) == 1 and smooth:
        P, c = through[0]
        # order along a smooth curve = floor(c) unless Q vanishes identically along it (cannot: reduced) -> floor(c)
        # but Q may vanish at the point to some order along the curve: pole order at generic nearby curve point is floor(c) anyway
        return max(int(floor(c)), 0)
    if len(through) == 2 and smooth:
        (P1, c1), (P2_, c2) = through
        l1 = P1.lowest(); l2 = P2_.lowest()
        a1, b1 = l1.d.get((1, 0), 0), l1.d.get((0, 1), 0)
        a2, b2 = l2.d.get((1, 0), 0), l2.d.get((0, 1), 0)
        det = (a1 * b2 - a2 * b1) % p
        if det != 0:
            # normal crossing. Newton-diagram refinement with local coords (s,t)=(P1,P2):
            K = int(floor(c1)) + int(floor(c2)) + 1
            supp = _support_in_curvilinear(P1, P2_, Qloc, p, K)
            best = 0
            for (a, b) in supp:
                val = max(int(floor(c1 - a)), 0) + max(int(floor(c2 - b)), 0)
                best = max(best, val)
            if log is not None: log.append(('normal', depth, float(c1), float(c2), sorted(supp)[:6], best))
            return best
    if depth >= maxdepth:
        if log is not None: log.append(('MAXDEPTH', depth))
        return max(sum(max(int(floor(c)), 0) * 1 for P, c in through), 0)
    # blow up the origin: exceptional divisor exponent cE = sum m_i c_i - ord(Q) - 1
    cE = sum(m * c for m, (P, c) in zip(mults, through)) - qord - 1
    best = 0
    # chart A: x=u, y=u v  (E: u=0). strict transforms: P(u,uv)/u^m
    for chart in ('A', 'B'):
        new = []
        for (P, c), m in zip(through, mults):
            Pb = P.blowup_x() if chart == 'A' else P.blowup_y()
            k, Pt = Pb.div_xpow() if chart == 'A' else Pb.div_ypow()
            assert k == m, (k, m)
            new.append((Pt, c))
        Qb = Qloc.blowup_x() if chart == 'A' else Qloc.blowup_y()
        kq, Qt = (Qb.div_xpow() if chart == 'A' else Qb.div_ypow())
        E = P2(p, {(1, 0): 1}) if chart == 'A' else P2(p, {(0, 1): 1})
        allc = new + ([(E, cE)] if cE != 0 else [])
        # points on E to examine: intersections of strict transforms with E (v-roots of Pt(0,v)), chart B only needs v... chart B: E is v=0, points u with Pt(u,0)=0; but chart B origin (u=0) corresponds to direction x=0 which is NOT in chart A (v=inf) -> in chart B examine only u=0.
        pts = set()
        if chart == 'A':
            for Pt, c in new:
                coeffs = [Pt.d.get((0, j), 0) for j in range(0, 1 + max((j for (i, j) in Pt.d if i == 0), default=0))]
                rts = _roots_mod(coeffs, p)
                if rts is None:
                    continue  # E-component? shouldn't happen for strict transform
                for rt, m in rts: pts.add(rt)
            # also where Qt vanishes on E? not needed for pole order (only decreases)
            cand = [(0, v) for v in pts]
        else:
            cand = [(0, 0)] if any(Pt.eval(0, 0) == 0 for Pt, c in new) else []
            # generic point of E in chart B also covered by chart A generic
        # generic point on E (no strict transform): order floor(cE)
        best = max(best, max(int(floor(cE)), 0))
        for (a, b) in cand:
            shifted = [(P.shift(a, b), c) for P, c in allc]
            Qs = Qt.shift(a, b)
            best = max(best, _local_order(shifted, Qs, p, depth + 1, maxdepth, log))
    if log is not None: log.append(('blowup', depth, [float(c) for P, c in through], mults, float(cE), best))
    return best


def _support_in_curvilinear(P1, P2_, Q, p, K):
    """Support (set of (a,b)) of Q expanded in powers of (s,t)=(P1,P2) up to total degree K, at the origin where P1,P2
    are smooth and transversal.  Uses truncated power-series inversion."""
    # linear parts
    a1, b1 = P1.d.get((1, 0), 0), P1.d.get((0, 1), 0)
    a2, b2 = P2_.d.get((1, 0), 0), P2_.d.get((0, 1), 0)
    det = (a1 * b2 - a2 * b1) % p
    inv = pow(det, -1, p)
    # x,y in terms of (s,t) to first order: [x,y] = M^{-1}[s,t]
    m11, m12, m21, m22 = (b2 * inv) % p, (-b1 * inv) % p, (-a2 * inv) % p, (a1 * inv) % p
    # iterate: X_{k+1} = X_k - M^{-1} (F(X_k) - (s,t))  as truncated polys in (s,t)
    S = P2(p, {(1, 0): 1}); T = P2(p, {(0, 1): 1})
    X = S * m11 + T * m12
    Y = S * m21 + T * m22
    for _ in range(K + 1):
        F1 = _compose(P1, X, Y, K) - S
        F2 = _compose(P2_, X, Y, K) - T
        X = (X - (F1 * m11 + F2 * m12)).trunc(K)
        Y = (Y - (F1 * m21 + F2 * m22)).trunc(K)
    Qst = _compose(Q, X, Y, K)
    supp = set(Qst.d.keys())
    # minimal elements suffice but return all (small)
    if not supp:
        # Q vanishes to order > K at the point: pole order then <= 0 contributions; return a far point
        return {(K + 1, 0), (0, K + 1)}
    return supp


def _compose(P, X, Y, K):
    """P(X(s,t), Y(s,t)) truncated at total degree K"""
    res = P2(P.p)
    px = {0: P2(P.p, {(0, 0): 1})}; py = {0: P2(P.p, {(0, 0): 1})}
    for (i, j), c in P.d.items():
        if i not in px:
            for k in range(max(px) + 1, i + 1): px[k] = (px[k - 1] * X).trunc(K)
        if j not in py:
            for k in range(max(py) + 1, j + 1): py[k] = (py[k - 1] * Y).trunc(K)
        res = res + (px[i] * py[j]).trunc(K) * c
    return res.trunc(K)


# --- points of CP^2 ------------------------------------------------------------------------------------------

def _projective_points(polys_h, zsyms, p, kinsub):
    """All points (over F_p) where >=2 of the homogeneous polys vanish or one is singular.  polys_h: list of sympy
    homogeneous polys in (z0,z1,z2) with integer kinematics substituted.  Returns list of (chart, (a,b)) with chart in
    {0,1,2} = index of coordinate set to 1, (a,b) the other two coords in order; None if the configuration does not
    split over F_p (caller retries another prime)."""
    z0, z1, z2 = zsyms
    charts = [(0, (z1, z2), {z0: 1}), (1, (z0, z2), {z1: 1}), (2, (z0, z1), {z2: 1})]
    found = set()
    specialized = [sp.expand(P.subs(kinsub)) for P in polys_h]

    def add_point(ch, a, b):
        # canonical projective rep: coordinates (c0,c1,c2)
        co = [None, None, None]
        co[ch] = 1
        others = [i for i in range(3) if i != ch]
        co[others[0]] = a % p; co[others[1]] = b % p
        # normalize: first nonzero = 1
        for i in range(3):
            if co[i] % p:
                inv = pow(co[i], -1, p); co = tuple((c * inv) % p for c in co); break
        found.add(co)
    systems = []
    npol = len(specialized)
    for i in range(npol):
        for j in range(i + 1, npol):
            systems.append((specialized[i], specialized[j]))
        Pi = specialized[i]
        if sp.Poly(Pi, z0, z1, z2).total_degree() >= 2:
            systems.append((Pi, sp.expand(sp.diff(Pi, z1)), sp.expand(sp.diff(Pi, z2)), sp.expand(sp.diff(Pi, z0))))
    for sysp in systems:
        # solve in chart z0=1 first (points with z0!=0), then on the line z0=0 in chart z1=1 (z1!=0), then the point [0:0:1]
        for ch, (x, y), sub in charts:
            eqs = [sp.expand(P.subs(sub)) for P in sysp]
            if ch == 1:
                eqs = [sp.expand(e.subs(z0, 0)) for e in eqs]  # only points with z0 = 0 (x = z0 = 0 is substituted here)
            if ch == 2:
                vals = [sp.sympify(e).subs({z0: 0, z1: 0}) for e in eqs]
                if all(int(v) % p == 0 for v in vals):
                    add_point(2, 0, 0)
                continue
            res = _solve2(eqs, x, y, p, fix_x0=(ch == 1))
            if res is None:
                return None
            for (a, b) in res:
                add_point(ch, a, b)
    return found


def _projective_points_system(sysp, Z, p):
    """projective F_p points of one polynomial system (list of homogeneous integer polys); None if it does not split"""
    z0, z1, z2 = Z
    found = set()
    def norm(co):
        for i in range(3):
            if co[i] % p:
                inv = pow(co[i], -1, p); return tuple((c * inv) % p for c in co)
        return None
    res = _solve2([sp.expand(P.subs(z0, 1)) for P in sysp], z1, z2, p)
    if res is None: return None
    for (a, b) in res: found.add(norm((1, a, b)))
    res = _solve2([sp.expand(P.subs(z1, 1)) for P in sysp], z0, z2, p, fix_x0=True)
    if res is None: return None
    for (a, b) in res: found.add(norm((0, 1, b)))
    if all(int(sp.expand(P.subs({z0: 0, z1: 0, z2: 1}))) % p == 0 for P in sysp):
        found.add((0, 0, 1))
    return found


def _solve2(eqs, x, y, p, fix_x0=False):
    """common F_p-roots of >=2 bivariate integer polys. If fix_x0: impose x=0. None if an eliminant doesn't split."""
    eqs = [sp.Poly(sp.expand(e), x, y, modulus=p) for e in eqs if sp.expand(e) != 0]
    eqs = [e for e in eqs if not e.is_zero]
    pts = set()
    if fix_x0:
        us = []
        for e in eqs:
            u = sp.Poly(e.as_expr().subs(x, 0), y, modulus=p)
            us.append(u)
        us = [u for u in us if not u.is_zero]
        if len(us) < 1:
            return set()   # whole line? ignore
        g = us[0]
        for u in us[1:]:
            g = sp.gcd(g, u)
        coeffs = [int(c) % p for c in reversed(g.all_coeffs())]
        if not _splits(coeffs, p):
            return None
        rts = _roots_mod(coeffs, p) or []
        return {(0, r) for r, m in rts}
    if len(eqs) < 2:
        return set()
    # eliminate y via resultant of first two, then filter with all
    f, g = eqs[0], eqs[1]
    try:
        R = sp.Poly(sp.resultant(f.as_expr(), g.as_expr(), y), x, modulus=p)
    except Exception:
        return set()
    if R.is_zero:
        # common component: f,g share a factor; use gcd-free parts
        h = sp.gcd(f, g)
        f2 = sp.Poly(sp.quo(f.as_expr(), h.as_expr(), x, y, modulus=p), x, y, modulus=p)
        R = sp.Poly(sp.resultant(f2.as_expr(), g.as_expr(), y), x, modulus=p)
        if R.is_zero:
            return set()
    coeffs = [int(c) % p for c in reversed(R.all_coeffs())]
    if not _splits(coeffs, p):
        return None
    for xr, m in (_roots_mod(coeffs, p) or []):
        us = [sp.Poly(e.as_expr().subs(x, xr), y, modulus=p) for e in eqs]
        us = [u for u in us if not u.is_zero]
        if not us:
            continue
        gg = us[0]
        for u in us[1:]:
            gg = sp.gcd(gg, u)
        cf = [int(c) % p for c in reversed(gg.all_coeffs())]
        if not _splits(cf, p):
            return None
        for yr, mm in (_roots_mod(cf, p) or []):
            pts.add((xr % p, yr % p))
    return pts


# --- residues for n=2 ------------------------------------------------------------------------------------------

def _rho_geom(PH, idx, chart_idx, Z, cache=None):
    """Q-independent part of the residue 1-form data along divisor idx in chart Z[chart_idx]=1:
    (Pc, F0 = [restrictions of the other divisors] + [dPc/da], a, b).  Cached per (idx, chart) when a cache dict is given."""
    key = ('rho_geom', idx, chart_idx)
    if cache is not None and key in cache:
        return cache[key]
    others = [i for i in range(3) if i != chart_idx]
    a, b = Z[others[0]], Z[others[1]]
    sub = {Z[chart_idx]: 1}
    Pc = sp.expand(PH[idx].subs(sub))
    if sp.expand(sp.diff(Pc, a)) == 0:
        a, b = b, a
    dPa = sp.expand(sp.diff(Pc, a))
    F0 = [sp.expand(Pj.subs(sub)) for j, Pj in enumerate(PH) if j != idx] + [dPa]
    out = (Pc, F0, a, b)
    if cache is not None:
        cache[key] = out
    return out


def _rho_data(PH, CH, QH, idx, m, chart_idx, Z, cache=None):
    """Build the residue 1-form data along divisor idx in chart Z[chart_idx]=1.
    PH: list of homogeneous integer polys, CH: exponents c_i, QH numerator. Returns (Pc, F, E, N, a, b):
    rho = N * prod F^E  db  on C={Pc=0}, after (m-1) transversal derivatives."""
    Pc, F0, a, b = _rho_geom(PH, idx, chart_idx, Z, cache)
    sub = {Z[chart_idx]: 1}
    F, E = [], []
    k = 0
    for j, cj in enumerate(CH):
        if j == idx:
            continue
        if cj != 0:
            F.append(F0[k]); E.append(-sp.Rational(cj))
        k += 1
    F.append(F0[-1]); E.append(sp.Rational(-1))
    N = sp.expand(QH.subs(sub))
    for _ in range(int(m) - 1):
        prodF = sp.Integer(1)
        for Fj in F: prodF = sp.expand(prodF * Fj)
        newN = sp.expand(sp.diff(N, a) * prodF)
        for j, (Fj, Ej) in enumerate(zip(F, E)):
            if Ej == 0:
                continue
            rest = sp.Integer(1)
            for i, Fi in enumerate(F):
                if i != j: rest = rest * Fi
            newN = sp.expand(newN + N * Ej * sp.diff(Fj, a) * rest)
        N = newN
        E = [Ej - 1 for Ej in E]
        E[-1] = E[-1] - 1
    return Pc, F, E, N, a, b


def _cached_points_pair(PH, i, j, Z, primes, cache):
    """(prime, points) of {P_i = P_j = 0} with the first prime in `primes` for which the points are rational; cached."""
    key = ('pair', i, j)
    if key in cache:
        return cache[key]
    out = (None, None)
    for p in primes:
        pts = _projective_points_pair(PH[i], PH[j], Z, p)
        if pts is not None:
            out = (p, pts); break
    cache[key] = out
    return out


def _residue_chain_n2(rep, term, kinsub, primes, log, rng=None):
    """r in {0,1,2} for a form on CP^2, mod p at integer kinematics; every (curve, divisor) point system uses its own
    prime from `primes` (first one for which the points are F_p-rational).  Returns (r, detail) or None.
    All Q-independent geometry (restricted divisors, Groebner bases, point sets, local charts) is cached on
    the rep (rep._geom) so that only the numerator work is repeated per form."""
    divs, Q = form_data(rep, term)
    Z = (rep.z0, rep.zsyms[0], rep.zsyms[1])
    cache = _geom_cache(rep, kinsub)
    if 'PH' not in cache:
        cache['PH'] = [_intpoly(P.subs(kinsub)) for (P, c, par, mm) in divs]
    PH = cache['PH']
    CH = [c for (P, c, par, mm) in divs]
    QH = _intpoly(Q.subs(kinsub))
    best = 0
    detail = {}
    plist = list(primes)
    for idx, (P, c, par, m) in enumerate(divs):
        if par != 'even' or m < 1:
            continue
        ch0 = 0 if idx != 0 else 1
        Pc, F, E, N, a, b = _rho_data(PH, CH, QH, idx, m, ch0, Z, cache)
        gkey = ('groebner', idx, ch0)
        if gkey not in cache:
            cache[gkey] = sp.groebner([Pc], a, b, order='grevlex', domain=sp.QQ)
        G = cache[gkey]
        rem = G.reduce(sp.expand(N))[1]
        if sp.expand(rem) == 0:
            detail[f'P{idx}'] = f'residue along P{idx} (pole {m}) vanishes identically'
            continue
        rr = 1
        found2 = False
        unresolved = False
        for j in range(len(PH)):
            if j == idx or CH[j] == 0:
                continue
            puse, pts = _cached_points_pair(PH, idx, j, Z, plist, cache)
            if pts is None:
                unresolved = True
                detail.setdefault('flags', []).append(f'points of P{idx} cap P{j} split over no prime in pool')
                continue
            p = puse
            for co in pts:
                ch = next(i for i in range(3) if co[i] == 1)
                Pc, F, E, N, a, b = _rho_data(PH, CH, QH, idx, m, ch, Z, cache)
                qa = co[Z.index(a)]; qb = co[Z.index(b)]
                lkey = ('local', idx, ch, p, co)
                if lkey not in cache:
                    PcL_ = P2.from_sympy(Pc, a, b, p).shift(qa, qb)
                    F0 = _rho_geom(PH, idx, ch, Z, cache)[1]
                    FL0 = [P2.from_sympy(Fj, a, b, p).shift(qa, qb) for Fj in F0]
                    cache[lkey] = (PcL_, FL0)
                PcL, FL0 = cache[lkey]
                if PcL.d.get((1, 0), 0) % p == 0 and PcL.d.get((0, 1), 0) % p == 0:
                    if log is not None: log.append(('singular point of C skipped (FLAG)', idx, co))
                    detail.setdefault('flags', []).append(f'singular point on P{idx} at {co}')
                    continue
                # FL in the order of F (restrictions of divisors with c != 0, then dPa)
                FL = []
                k = 0
                for jj, cj in enumerate(CH):
                    if jj == idx:
                        continue
                    if cj != 0:
                        FL.append(FL0[k])
                    k += 1
                FL.append(FL0[-1])
                NL = P2.from_sympy(N, a, b, p).shift(qa, qb)
                res = _analyze_generic(PcL, FL, E, NL, p, 10)
                if log is not None: log.append(('pt on C', idx, co, None if res is None else (str(res[2]), res[0], res[1])))
                if res and res[0] and res[1]:
                    found2 = True
                    break
            if found2:
                break
        if found2:
            rr = 2
        detail[f'P{idx}'] = f'residue along P{idx} (pole {m}) nonzero; second residue nonzero: {found2}' + (' (some point systems unresolved)' if unresolved else '')
        best = max(best, rr)
        if best == 2:
            break
    return best, detail


def _geom_cache(rep, kinsub):
    """per-rep, per-kinematic-point cache of Q-independent geometry"""
    key = tuple(sorted((str(k), int(v)) for k, v in kinsub.items()))
    store = getattr(rep, '_geom', None)
    if store is None:
        store = {}
        try:
            rep._geom = store
        except Exception:
            pass
    return store.setdefault(key, {})


def _projective_points_pair(P1h, P2h, Z, p):
    """projective F_p points of {P1h=P2h=0} (homogeneous in Z=(z0,z1,z2)); None if not split. Points as normalized
    coordinate triples (first nonzero coordinate = 1)."""
    z0, z1, z2 = Z
    found = set()
    def norm(co):
        for i in range(3):
            if co[i] % p:
                inv = pow(co[i], -1, p); return tuple((c * inv) % p for c in co)
        return None
    # chart z0=1
    res = _solve2([sp.expand(P1h.subs(z0, 1)), sp.expand(P2h.subs(z0, 1))], z1, z2, p)
    if res is None: return None
    for (a, b) in res: found.add(norm((1, a, b)))
    # line z0=0, chart z1=1: solve P(0,1,z2)=0
    res = _solve2([sp.expand(P1h.subs(z1, 1)), sp.expand(P2h.subs(z1, 1))], z0, z2, p, fix_x0=True)
    if res is None: return None
    for (a, b) in res: found.add(norm((0, 1, b)))
    # point [0:0:1]
    if all(int(sp.expand(P.subs({z0: 0, z1: 0, z2: 1}))) % p == 0 for P in (P1h, P2h)):
        found.add((0, 0, 1))
    return found


def _intpoly(expr):
    e = sp.together(sp.expand(expr))
    n, d = sp.fraction(e)
    assert not d.free_symbols, d
    return sp.expand(n)


def _analyze_generic(PcL, FL, EL, NL, p, K):
    """Point at origin on the smooth curve branch PcL=0; rho = NL prod FL^EL db (b = slot-1 variable).  Returns
    (integer_pole>=1, residue_nonzero, order) or None if the curve is singular here."""
    da = PcL.d.get((1, 0), 0); db = PcL.d.get((0, 1), 0)
    extra = []
    if da % p:
        phi = _newton_param(PcL, p, K, solve_for=0)
        subsfun = lambda poly: _compose(poly, phi, P2(p, {(0, 1): 1}), K); slot = 1
    elif db % p:
        phi = _newton_param(PcL, p, K, solve_for=1)
        subsfun = lambda poly: _compose(poly, P2(p, {(1, 0): 1}), phi, K); slot = 0
        extra.append(phi.deriv(0))      # db = phi'(a) da
    else:
        return None
    total = sp.Rational(0); pieces = []
    for FLj, Ej in zip(FL, EL):
        if Ej == 0: continue
        rj = subsfun(FLj); oj = _ord_univ(rj, slot); total -= sp.Rational(Ej) * oj; pieces.append((rj, sp.Rational(Ej)))
    rN = subsfun(NL); total -= _ord_univ(rN, slot)
    for jd in extra: total -= _ord_univ(jd, 0)
    if total.q != 1 or total < 1:
        return (False, False, total)
    if total == 1:
        return (True, True, total)
    ser = _unit_series(rN, slot, p, K)
    for rj, Ej in pieces:
        ser = _mul_series(ser, _pow_series(_unit_series(rj, slot, p, K), Ej, p, K), p, K)
    for jd in extra:
        ser = _mul_series(ser, _unit_series(jd, 0, p, K), p, K)
    coeff = ser[int(total) - 1] if int(total) - 1 <= K else 0
    return (True, coeff % p != 0, total)


def _ord_univ(poly, slot):
    if poly.is_zero(): return 10 ** 6
    return min(k[slot] for k in poly.d)


def _unit_series(poly, slot, p, K):
    """poly (univariate in slot) = tau^o * unit ; return unit coefficients list length K+1 normalized? (not normalized)"""
    o = _ord_univ(poly, slot)
    coeffs = [0] * (K + 1)
    for k, c in poly.d.items():
        e = k[slot] - o
        if 0 <= e <= K: coeffs[e] = (coeffs[e] + c) % p
    return coeffs


def _mul_series(A, B, p, K):
    C = [0] * (K + 1)
    for i, x in enumerate(A):
        if not x: continue
        for j, y in enumerate(B):
            if i + j > K: break
            C[i + j] = (C[i + j] + x * y) % p
    return C


def _pow_series(U, e, p, K):
    """U(t)^e for unit series U (U[0]!=0), rational e; overall constant U0^e dropped (set to 1) — fine for zero tests."""
    u0 = U[0] % p
    inv = pow(u0, -1, p)
    V = [(c * inv) % p for c in U]   # 1 + ...
    # (1+x)^e = sum binom(e,k) x^k ; compute via log/exp-free recurrence: W' = e W V'/V
    W = [0] * (K + 1); W[0] = 1
    # recurrence: k W_k = sum_{j=1..k} (e*j - (k-j)) V_j W_{k-j}
    en, ed = int(sp.Rational(e).p), int(sp.Rational(e).q)
    einv = pow(ed, -1, p); emod = (en * einv) % p
    for k in range(1, K + 1):
        acc = 0
        for j in range(1, k + 1):
            acc = (acc + ((emod * j - (k - j)) % p) * V[j] % p * W[k - j]) % p
        W[k] = acc * pow(k, -1, p) % p
    return W


def _newton_param(PcL, p, K, solve_for=0):
    """PcL(x,y) with PcL(0,0)=0 and d/d(x if solve_for==0 else y) != 0 at 0: return phi as P2 poly in the OTHER variable
    (slot 1 if solve_for==0 else slot 0) with x = phi(y) (resp y = phi(x)), truncated at degree K."""
    if solve_for == 1:
        # swap roles
        sw = P2(p, {(j, i): c for (i, j), c in PcL.d.items()})
        ph = _newton_param(sw, p, K, 0)
        return P2(p, {(j, i): c for (i, j), c in ph.d.items()})
    d = PcL.d.get((1, 0), 0)
    inv = pow(d, -1, p)
    T = P2(p, {(0, 1): 1})
    phi = P2(p)
    for _ in range(K + 1):
        F = _compose(PcL, phi, T, K)
        phi = (phi - F * inv).trunc(K)
    return phi


# ----------------------------------------------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------------------------------------------

def integers(rep: BaikovRep, term: FormTerm, nsamples=NSAMPLES, seed=1, log=None) -> Integers:
    n = rep.n
    absmu = int(sum(term.mu))
    if n == 0:
        return Integers(absmu, 0, 0, 0, 0, [], {})
    if n == 1:
        return _n1(rep, term) if rep.family.kin else _n1_modp(rep, term, seed=seed)
    if n == 2:
        return _n2(rep, term, nsamples, seed, log)
    # n >= 3: generic-position estimate (NOT certified): r = min(n, number of even divisors with a pole), assuming all
    # residues non-zero and proper intersections; o = sum of the n largest floor(c_i) over active divisors (deepest
    # normal crossing of n divisors in general position).
    divs, Q = form_data(rep, term)
    floors = sorted((int(floor(c)) for (_, c, _, m) in divs if c > 0), reverse=True)
    o = sum(floors[:n]) if floors else 0
    r = min(n, sum(1 for (_, c, par, m) in divs if par == 'even' and m >= 1))
    return Integers(absmu, r, o, n + r, n, [f'n={n}>=3: generic-position estimate'], {})


def _n2(rep, term, nsamples, seed, log):
    divs, Q = form_data(rep, term)
    z0, z1, z2 = rep.z0, rep.zsyms[0], rep.zsyms[1]
    Z = (z0, z1, z2)
    kin = sorted(rep.family.kin.values(), key=str)
    rng = random.Random(seed)
    results = []
    flags = []
    attempts = 0
    fam_has_symbols = bool(kin)
    if not fam_has_symbols:
        nsamples = 1
    while len(results) < nsamples and attempts < 40:
        attempts += 1
        kinsub = {k: rng.randint(3, 200) for k in kin}
        # avoid degenerate kinematics: all divisors must stay nonzero & distinct
        try:
            spec = [sp.expand(sp.together(P.subs(kinsub))) for (P, c, par, m) in divs]
            spec = [sp.fraction(sp.together(s_))[0] for s_ in spec]   # clear denominators (kinematic)
        except Exception:
            continue
        if any(s_ == 0 for s_ in spec):
            continue
        # pole order: local analysis at the points of every divisor pair (and singular points), each pair with its own
        # prime chosen so that the pair's point set splits over F_p
        o = max([int(floor(c)) for (_, c, _, m) in divs] + [0])
        Qs = sp.fraction(sp.together(sp.expand(Q.subs(kinsub))))[0]
        cache = _geom_cache(rep, kinsub)
        if 'spec' not in cache:
            cache['spec'] = spec
        systems = []
        for i in range(len(spec)):
            for j in range(i + 1, len(spec)):
                if divs[i][1] == 0 and divs[j][1] == 0:
                    continue
                systems.append((('pair', i, j), (spec[i], spec[j])))
            if sp.Poly(spec[i], *Z).total_degree() >= 2 and divs[i][1] != 0:
                systems.append((('sing', i), (spec[i], sp.expand(sp.diff(spec[i], Z[1])), sp.expand(sp.diff(spec[i], Z[2])), sp.expand(sp.diff(spec[i], Z[0])))))
        for skey, sysp in systems:
            pkey = ('pts', skey)
            if pkey not in cache:
                found_p = None
                for p in rng.sample(PRIMES, len(PRIMES)):
                    pts = _projective_points_system(sysp, Z, p)
                    if pts is not None:
                        found_p = (p, pts); break
                cache[pkey] = found_p
            found_p = cache[pkey]
            if found_p is None:
                flags.append('a point system did not split over any prime in the pool')
                continue
            p, pts = found_p
            for co in pts:
                ch = next(i for i in range(3) if co[i] == 1)
                othersZ = [Z[i] for i in range(3) if i != ch]
                a0, b0 = [co[i] for i in range(3) if i != ch]
                sub = {Z[ch]: 1}
                ckey = ('curves', p, co)
                if ckey not in cache:
                    cache[ckey] = [P2.from_sympy(sp.expand(Psp.subs(sub)), othersZ[0], othersZ[1], p).shift(a0, b0) for Psp in spec]
                curves = [(Pl, c) for Pl, (P, c, par, m) in zip(cache[ckey], divs) if c != 0]
                Ql = P2.from_sympy(sp.expand(Qs.subs(sub)), othersZ[0], othersZ[1], p).shift(a0, b0)
                lg = [] if log is not None else None
                op = _local_order(curves, Ql, p, log=lg)
                if log is not None: log.append(('point', co, op, lg))
                o = max(o, op)
        rr = _residue_chain_n2(rep, term, kinsub, rng.sample(PRIMES, len(PRIMES)), log, rng)
        r, det = rr
        if det.get('flags'):
            flags.extend(det['flags'])
        results.append((r, o, None, dict(kinsub), det))
        if not fam_has_symbols:
            break
    if len(results) < nsamples:
        flags.append(f'only {len(results)} samples succeeded')
    if not results:
        return Integers(int(sum(term.mu)), -1, -1, -1, 2, flags + ['FAILED'], {})
    rs = {x[0] for x in results}; os_ = {x[1] for x in results}
    if len(rs) > 1 or len(os_) > 1:
        flags.append(f'INCONSISTENT samples: r={rs} o={os_}')
    r = max(rs); o = max(os_)
    return Integers(int(sum(term.mu)), r, o, 2 + r, 2, flags, {'samples': [(x[2], x[3], x[0], x[1]) for x in results], 'residues': results[0][4]})


def order_key(rep, terms, **kw):
    """(w,o,|mu|) of an iota image = lexicographic max over its terms (the convention stated in iota.py). Returns (key, per-term Integers)."""
    ints = [integers(rep, t, **kw) for t in terms]
    key = max(i.tup() for i in ints)
    return key, ints
