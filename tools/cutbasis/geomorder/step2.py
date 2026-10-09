# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of BootLoops 1.0; see LICENSE and NOTICE beside this file.
"""STEP 2 of arXiv:2511.15381 (sect. 4.3, Theorem 'eps-factorization'): rotate an F-compatible Laurent-eps maximal-cut
DE  dJ = A J,  A = sum_{k=-n}^{1} eps^k A^(k)(x), into eps-factorized form dK = eps Atilde^(1) K, K = R_2^{-1} J.

Scope: ONE kinematic variable x, TWO filtration blocks (the paper's worked example, lines 3480-3528 of the arXiv v2 TeX source; this covers
every n=1 sector: block 1 = |mu| minimal, block 2 = the rest), exact arithmetic with truncated power series in
u = x - x0 around a regular point x0 (the new transcendental functions = the series solutions of the first-order
linear systems; for an elliptic sector these are the period integrals).  Equations implemented verbatim:
   dR11 = A0_11 R11 + A1_12 R21m          (R21m := R^(-1)_21)
   dR21m = Am1_21 R11 + A0_22 R21m
   dR22 = A0_22 R22 - R21m R11^{-1} A1_12 R22
   dR21z = R22^{-1} ( A0_21 R11 + A1_22 R21m - R21m R11^{-1} A1_11 R11 )     (R21z := R^(0)_21)
   R_2 = [[R11, 0],[R21m/eps, R22]] . [[1,0],[R21z,1]]
Output: the series of R blocks, the transformed matrix split by eps power, and the exact check that the eps^0 and
eps^-1 parts of the transformed DE vanish identically to the truncation order.
"""
from __future__ import annotations
from fractions import Fraction
import sympy as sp

# ------------------------------------------------------------------------------------------------ series arithmetic
# a series = list of Fraction of length N (coefficients of u^0..u^{N-1}); a matrix of series = list of lists.

def s_zero(N): return [Fraction(0)] * N
def s_const(c, N): v = s_zero(N); v[0] = Fraction(c); return v
def s_add(a, b): return [x + y for x, y in zip(a, b)]
def s_sub(a, b): return [x - y for x, y in zip(a, b)]
def s_scale(a, c): return [x * c for x in a]
def s_mul(a, b):
    N = len(a); out = s_zero(N)
    for i, x in enumerate(a):
        if x == 0: continue
        for j in range(N - i):
            out[i + j] += x * b[j]
    return out
def s_inv(a):
    N = len(a); assert a[0] != 0
    out = s_zero(N); out[0] = 1 / a[0]
    for m in range(1, N):
        acc = Fraction(0)
        for k in range(1, m + 1): acc += a[k] * out[m - k]
        out[m] = -acc / a[0]
    return out
def s_deriv(a):
    N = len(a); return [a[i + 1] * (i + 1) for i in range(N - 1)] + [Fraction(0)]
def s_from_ratfunc(expr, x, x0, N):
    """Taylor coefficients of a rational function of x at x0 (exact)."""
    num, den = sp.fraction(sp.together(sp.sympify(expr)))
    u = sp.Symbol('_u')
    pn = sp.Poly(sp.expand(sp.sympify(num).subs(x, x0 + u)), u); pd = sp.Poly(sp.expand(sp.sympify(den).subs(x, x0 + u)), u)
    def coeffs(P):
        c = s_zero(N)
        for (e,), v in zip(P.monoms(), P.coeffs()):
            if e < N: c[e] = Fraction(int(sp.Rational(v).p), int(sp.Rational(v).q))
        return c
    cn, cd = coeffs(pn), coeffs(pd)
    if cd[0] == 0: raise ValueError('x0 is a pole')
    return s_mul(cn, s_inv(cd))

# matrices of series
def m_zero(r, c, N): return [[s_zero(N) for _ in range(c)] for _ in range(r)]
def m_eye(r, N): M = m_zero(r, r, N); [M[i][i].__setitem__(0, Fraction(1)) for i in range(r)]; return M
def m_add(A, B): return [[s_add(a, b) for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]
def m_sub(A, B): return [[s_sub(a, b) for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]
def m_mul(A, B):
    r, k, c = len(A), len(B), len(B[0]); N = len(A[0][0])
    out = m_zero(r, c, N)
    for i in range(r):
        for j in range(c):
            acc = s_zero(N)
            for m in range(k): acc = s_add(acc, s_mul(A[i][m], B[m][j]))
            out[i][j] = acc
    return out
def m_deriv(A): return [[s_deriv(a) for a in row] for row in A]
def m_inv(A):
    """inverse of a square series matrix via adjugate for size<=3 or Gauss (generic): use Gauss-Jordan over series field"""
    n = len(A); N = len(A[0][0])
    M = [[list(a) for a in row] + [list(e) for e in m_eye(n, N)[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = next(r for r in range(col, n) if M[r][col][0] != 0)
        M[col], M[piv] = M[piv], M[col]
        inv = s_inv(M[col][col])
        M[col] = [s_mul(e, inv) for e in M[col]]
        for r in range(n):
            if r == col: continue
            f = M[r][col]
            if all(x == 0 for x in f): continue
            M[r] = [s_sub(e, s_mul(f, ec)) for e, ec in zip(M[r], M[col])]
    return [row[n:] for row in M]

def solve_linear_ode(Mfun, Ffun, Y0, N):
    """Y' = M Y + F with M (dxd series matrix), F (d x c series matrix), Y0 (d x c constants). Power-series recursion."""
    d = len(Mfun); c = len(Y0[0])
    Y = [[s_zero(N) for _ in range(c)] for _ in range(d)]
    for i in range(d):
        for j in range(c): Y[i][j][0] = Fraction(Y0[i][j])
    for m in range(N - 1):
        # coefficient m of (M Y + F) determines (m+1) Y_{m+1}
        for i in range(d):
            for j in range(c):
                acc = Ffun[i][j][m]
                for k in range(d):
                    Mi = Mfun[i][k]; Yk = Y[k][j]
                    for t in range(m + 1):
                        if Mi[t]: acc += Mi[t] * Yk[m - t]
                Y[i][j][m + 1] = acc / (m + 1)
    return Y

# ------------------------------------------------------------------------------------------------ step 2 (two blocks)

def step2_two_blocks(Ak, d1, x, x0, N=24):
    """Ak: dict k -> full matrix (list of lists of sympy expr in x) for k in {-1,0,1}; block sizes d1, d2.
    Returns dict with series of R blocks and the transformed DE split by eps power (as series matrices)."""
    nF = len(Ak[1])
    d2 = nF - d1
    def blk(k, I, J):
        M = Ak.get(k)
        rows = range(0, d1) if I == 1 else range(d1, nF)
        cols = range(0, d1) if J == 1 else range(d1, nF)
        return [[s_from_ratfunc(M[i][j] if M is not None else 0, x, x0, N) for j in cols] for i in rows]
    A0_11, A1_11, A1_12 = blk(0, 1, 1), blk(1, 1, 1), blk(1, 1, 2)
    Am1_21, A0_21 = blk(-1, 2, 1), blk(0, 2, 1)
    A0_22, A1_22 = blk(0, 2, 2), blk(1, 2, 2)
    # F-compatibility checks of the input (needed structure): A^(-1), A^(0) must vanish in blocks (1,2); A^(-1)_11, A^(-1)_22 zero
    def is_zero_blk(k, I, J):
        M = Ak.get(k)
        if M is None: return True
        rows = range(0, d1) if I == 1 else range(d1, nF); cols = range(0, d1) if J == 1 else range(d1, nF)
        return all(sp.simplify(sp.sympify(M[i][j])) == 0 for i in rows for j in cols)
    structure_ok = is_zero_blk(-1, 1, 1) and is_zero_blk(-1, 2, 2) and is_zero_blk(-1, 1, 2) and is_zero_blk(0, 1, 2)
    # (1) coupled system for (R11; R21m): Y' = [[A0_11, A1_12],[Am1_21, A0_22]] Y, Y(0) = generic column(s): take d1 columns
    Mbig = [A0_11[i] + A1_12[i] for i in range(d1)] + [Am1_21[i] + A0_22[i] for i in range(d2)]
    # initial condition: R11(0)=1 (d1 x d1 identity), R21m(0)=0  -> det R11 != 0 near x0
    Y0 = [[1 if i == j else 0 for j in range(d1)] for i in range(d1)] + [[0] * d1 for _ in range(d2)]
    Y = solve_linear_ode(Mbig, m_zero(nF, d1, N), Y0, N)
    R11 = Y[:d1]; R21m = Y[d1:]
    # (2) R22' = (A0_22 - R21m R11^{-1} A1_12) R22, R22(0) = 1
    R11inv = m_inv(R11)
    M22 = m_sub(A0_22, m_mul(m_mul(R21m, R11inv), A1_12))
    R22 = solve_linear_ode(M22, m_zero(d2, d2, N), [[1 if i == j else 0 for j in range(d2)] for i in range(d2)], N)
    # (3) R21z' = R22^{-1} (A0_21 R11 + A1_22 R21m - R21m R11^{-1} A1_11 R11), R21z(0) = 0  (pure quadrature)
    R22inv = m_inv(R22)
    rhs = m_mul(R22inv, m_sub(m_add(m_mul(A0_21, R11), m_mul(A1_22, R21m)), m_mul(m_mul(m_mul(R21m, R11inv), A1_11), R11)))
    R21z = solve_linear_ode(m_zero(d2, d2, N), rhs, [[0] * d1 for _ in range(d2)], N)
    # assemble R2 = R2m * R2z with eps symbolic: R2m = [[R11,0],[R21m/eps, R22]], R2z = [[1,0],[R21z,1]]
    # R2 = [[R11, 0], [R21m/eps + R22 R21z, R22]]
    # transformed A: Atil = R2^{-1} A R2 - R2^{-1} dR2 ; do it with explicit eps bookkeeping: represent matrices as dict eps-power -> series matrix
    def full(blocks):  # blocks dict (I,J)->series matrix -> full nF x nF
        M = m_zero(nF, nF, N)
        for (I, J), B in blocks.items():
            r0 = 0 if I == 1 else d1; c0 = 0 if J == 1 else d1
            for i, row in enumerate(B):
                for j, s in enumerate(row): M[r0 + i][c0 + j] = s
        return M
    R2 = {0: full({(1, 1): R11, (2, 2): R22, (2, 1): m_mul(R22, R21z)}), -1: full({(2, 1): R21m})}
    # inverse: R2 = R2m R2z ; R2^{-1} = R2z^{-1} R2m^{-1}; R2m^{-1} = [[R11^-1, 0], [-R22^-1 R21m R11^-1 / eps, R22^-1]], R2z^{-1} = [[1,0],[-R21z,1]]
    R2m_inv = {0: full({(1, 1): R11inv, (2, 2): R22inv}), -1: full({(2, 1): [[s_scale(s, Fraction(-1)) for s in row] for row in m_mul(m_mul(R22inv, R21m), R11inv)]})}
    R2z_inv = {0: full({(1, 1): m_eye(d1, N), (2, 2): m_eye(d2, N), (2, 1): [[s_scale(s, Fraction(-1)) for s in row] for row in R21z]})}
    def pmul(P, Q):
        out = {}
        for a, Ma in P.items():
            for b, Mb in Q.items():
                k = a + b
                prod = m_mul(Ma, Mb)
                out[k] = m_add(out[k], prod) if k in out else prod
        return out
    def pderiv(P): return {k: m_deriv(M) for k, M in P.items()}
    def psub(P, Q):
        out = dict(P)
        for k, M in Q.items(): out[k] = m_sub(out[k], M) if k in out else [[s_scale(s, Fraction(-1)) for s in row] for row in M]
        return out
    R2inv = pmul(R2z_inv, R2m_inv)
    Aser = {k: [[s_from_ratfunc(Ak[k][i][j], x, x0, N) for j in range(nF)] for i in range(nF)] for k in Ak}
    Atil = psub(pmul(pmul(R2inv, Aser), R2), pmul(R2inv, pderiv(R2)))
    # truncation: derivative loses one order and products accumulate; trust orders < N-2
    trust = N - 3
    report = {}
    for k, M in sorted(Atil.items()):
        mx = max((abs(float(s[t])) for row in M for s in row for t in range(trust)), default=0.0)
        exact_zero = all(s[t] == 0 for row in M for s in row for t in range(trust))
        report[k] = {'identically_zero_to_order': exact_zero, 'max_abs_coeff': mx}
    return {'structure_ok_input': structure_ok, 'x0': str(x0), 'N': N, 'trust_order': trust, 'eps_powers_after': report,
            'R11': R11, 'R21m': R21m, 'R22': R22, 'R21z': R21z, 'Atil': Atil}


def series_str(s, k=6):
    return ' + '.join(f'({c})u^{i}' for i, c in enumerate(s[:k]) if c != 0) or '0'
