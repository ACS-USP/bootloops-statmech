# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of BootLoops 1.0; see LICENSE and NOTICE beside this file.
"""Multivariate rational-function reconstruction from samples modulo word-size primes.

A coefficient function c(x_1..x_k) (x = (eps, kinematic invariants)) is known only through black-box evaluations
mod p.  Reconstruction:
  1. degree probe on a random affine line x = x0 + lam*v (univariate rational reconstruction in lam gives the
     total degrees (dn, dd) of numerator and denominator for a generic line);
  2. dense ansatz num(x) - y*den(x) = 0 with all monomials of total degree <= dn resp. <= dd, nullspace over F_p
     at >= #unknowns + 4 random points: nullity must be exactly 1 (minimal degrees), then verified on spare points;
  3. normalization: the lexicographically smallest monomial of den has coefficient 1 (canonical, prime-independent);
  4. two primes: coefficients CRT-combined, rational reconstruction (Wang), verified against fresh samples at a
     third prime.  Nothing is accepted without the third-prime check.
"""
from __future__ import annotations
import itertools
from fractions import Fraction
import flint
import sympy as sp


def monomials_upto(nvars, D):
    out = []
    for tot in range(D + 1):
        for c in itertools.combinations_with_replacement(range(nvars), tot):
            e = [0] * nvars
            for i in c:
                e[i] += 1
            out.append(tuple(e))
    return out


def eval_poly(poly, x, p):
    """poly: dict exponent-tuple -> int; x: list of ints"""
    acc = 0
    for mono, c in poly.items():
        t = c
        for xi, e in zip(x, mono):
            if e:
                t = t * pow(xi, e, p) % p
        acc = (acc + t) % p
    return acc


def eval_ratfunc(num, den, x, p):
    d = eval_poly(den, x, p)
    if d == 0:
        return None
    return eval_poly(num, x, p) * pow(d, -1, p) % p


def ratrec_univariate(xs, ys, p):
    """num/den (coefficient lists low->high, den lowest nonzero coefficient = 1) of minimal total degree fitting the
    points with >= 1 spare validation point; None if nothing fits."""
    n = len(xs)
    for tot in range(0, n - 1):
        for dd in range(0, tot + 1):
            dn = tot - dd
            m = dn + dd + 2
            if m - 1 > n - 1:
                continue
            M = flint.nmod_mat(n, m, p)
            for r in range(n):
                xp = 1
                for c in range(dn + 1):
                    M[r, c] = xp; xp = xp * xs[r] % p
                xp = 1
                for c in range(dd + 1):
                    M[r, dn + 1 + c] = (-ys[r] * xp) % p; xp = xp * xs[r] % p
            vecs, nullity = M.nullspace()
            if nullity != 1:
                continue
            v = [int(vecs[k, 0]) for k in range(m)]
            num = v[:dn + 1]; den = v[dn + 1:]
            if all(c == 0 for c in den):
                continue
            ok = True
            for r in range(n):
                nv = sum(c * pow(xs[r], k, p) for k, c in enumerate(num)) % p
                dv = sum(c * pow(xs[r], k, p) for k, c in enumerate(den)) % p
                if dv == 0 or (nv * pow(dv, -1, p) - ys[r]) % p:
                    ok = False; break
            if ok:
                fn = flint.nmod_poly(num, p); fd = flint.nmod_poly(den, p)
                g = fn.gcd(fd)
                if g.degree() > 0:
                    fn = fn // g; fd = fd // g
                num = [int(fn[k]) for k in range(fn.degree() + 1)] if fn.degree() >= 0 else [0]
                den = [int(fd[k]) for k in range(fd.degree() + 1)]
                lead = next(c for c in den if c)
                inv = pow(lead, -1, p)
                return [c * inv % p for c in num], [c * inv % p for c in den]
    return None


def fit_dense(points, values, p, dn, dd, nvars):
    """Dense fit with total-degree bounds. Returns (num dict, den dict) normalized (lex-smallest den monomial = 1)
    or None if the nullity is not 1 or validation fails."""
    mn = monomials_upto(nvars, dn); md = monomials_upto(nvars, dd)
    m = len(mn) + len(md)
    n = len(points)
    if n < m + 2:
        return None
    nfit = m + 2
    M = flint.nmod_mat(nfit, m, p)
    for r in range(nfit):
        x = points[r]; y = values[r]
        for c, mono in enumerate(mn):
            M[r, c] = eval_poly({mono: 1}, x, p)
        for c, mono in enumerate(md):
            M[r, len(mn) + c] = (-y * eval_poly({mono: 1}, x, p)) % p
    vecs, nullity = M.nullspace()
    if nullity != 1:
        return None
    v = [int(vecs[k, 0]) for k in range(m)]
    num = {mono: v[c] for c, mono in enumerate(mn) if v[c]}
    den = {mono: v[len(mn) + c] for c, mono in enumerate(md) if v[len(mn) + c]}
    if not den:
        return None
    lead = den[min(den)]
    inv = pow(lead, -1, p)
    num = {k: c * inv % p for k, c in num.items()}
    den = {k: c * inv % p for k, c in den.items()}
    for r in range(n):
        val = eval_ratfunc(num, den, points[r], p)
        if val is None or val != values[r] % p:
            return None
    return num, den


def wang(a, m):
    """rational reconstruction of a mod m: r/s with |r|, s <= sqrt(m/2); None if none."""
    a %= m
    bound = int((m // 2) ** 0.5)
    r0, r1 = m, a
    s0, s1 = 0, 1
    while r1 > bound:
        q = r0 // r1
        r0, r1 = r1, r0 - q * r1
        s0, s1 = s1, s0 - q * s1
    if s1 == 0 or abs(s1) > bound:
        return None
    if s1 < 0:
        r1, s1 = -r1, -s1
    from math import gcd
    if gcd(r1, s1) != 1:
        return None
    return Fraction(r1, s1)


def crt2(a1, p1, a2, p2):
    m = p1 * p2
    return (a1 * p2 * pow(p2, -1, p1) + a2 * p1 * pow(p1, -1, p2)) % m


def combine_primes(fits):
    """fits: list of ((num, den), p). Same monomial supports required. Returns (num_Q, den_Q) dicts of Fractions."""
    (num0, den0), p0 = fits[0]
    for (num, den), p in fits[1:]:
        if set(num) != set(num0) or set(den) != set(den0):
            raise ValueError('monomial support differs between primes')
    out = []
    for which in (0, 1):
        res = {}
        for mono in (num0 if which == 0 else den0):
            a, m = fits[0][0][which][mono], fits[0][1]
            for (nd), p in fits[1:]:
                a = crt2(a, m, nd[which][mono], p); m = m * p
            q = wang(a, m)
            if q is None:
                raise ValueError(f'rational reconstruction failed for monomial {mono}')
            res[mono] = q
        out.append(res)
    return out[0], out[1]


def to_sympy(num, den, syms):
    def poly(d):
        e = sp.Integer(0)
        for mono, c in d.items():
            t = sp.Rational(c.numerator, c.denominator)
            for s, k in zip(syms, mono):
                if k:
                    t *= s ** k
            e += t
        return e
    return sp.cancel(poly(num) / poly(den))


def ratfunc_mod(expr, syms, x, p):
    """evaluate a sympy rational function with rational coefficients at integer point x mod p"""
    num, den = sp.fraction(sp.together(expr))
    def ev(e):
        P = sp.Poly(sp.expand(e), *syms)
        acc = 0
        for mono, c in zip(P.monoms(), P.coeffs()):
            c = sp.Rational(c)
            t = int(c.p) * pow(int(c.q), -1, p) % p
            for xi, k in zip(x, mono):
                if k:
                    t = t * pow(xi, k, p) % p
            acc = (acc + t) % p
        return acc
    d = ev(den)
    if d == 0:
        return None
    return ev(num) * pow(d, -1, p) % p
