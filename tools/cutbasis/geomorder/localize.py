# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of BootLoops 1.0; see LICENSE and NOTICE beside this file.
"""Algorithm 1 of arXiv:2511.15381 (construction of the masters on localizations) — point localizations.

n = 1: the 0-skeleton is the set of roots of the even divisors (incl. P_0 = z_0, the point [0:1]).  For every Galois
orbit of N points the N simplest forms (by (w,o,|mu|,tie-break)) with a non-degenerate N x N residue matrix are the
"preferred candidates" and receive a = -2n = -2 (lines 2812-2826 and 2853 of the arXiv v2 TeX source; CP^1 example lines 2461-2561).  Running the global
Laporta (cohomology.CutSystem.solve) with the key (a,w,o,|mu|,...) then yields the basis of H^1_omega; masters that were
not candidates get a = -w' from "Find additional masters" (Algorithm 3) with w' = n (the full space, r'=0) i.e. a = -1.

n = 2 point localizations (the full n = 2 construction, points -> curves -> surface, is localize2.py): points = pairs of even divisors meeting (any multiplicity) and even-multiplicity
contacts of an even with an odd divisor, after blow-up where needed; 2-fold residues via the residue 1-form machinery of
integers.py.  Candidates get a = -4.  Curve localizations (a = -3 masters) are in localize2.py.
All numerics mod p at numeric kinematics (the CutSystem's prime, chosen so that the point sets split).
"""
from __future__ import annotations
import sympy as sp
import flint
from .integers import P2, _roots_mod, _splits, _unit_series, _mul_series, _pow_series, _ord_univ, form_data, PRIMES
from .iota import make_term


def _modp_coeffs(poly, var, p):
    cs = [sp.Rational(cc) for cc in reversed(sp.Poly(poly, var).all_coeffs())]
    return [(int(cc.p) * pow(int(cc.q), -1, p)) % p for cc in cs]


def good_prime_n1(rep, rng):
    """prime for which every even affine divisor p_i(z1) splits into linear factors"""
    z1 = rep.zsyms[0]
    for p in rng.sample(PRIMES, len(PRIMES)):
        ok = True
        for d in rep.divisors:
            try:
                cm = _modp_coeffs(sp.expand(d.p_affine), z1, p)
            except ValueError:
                ok = False; break
            if d.parity == 'even' and not _splits(cm, p):
                ok = False; break
        if ok:
            return p
    raise RuntimeError('no splitting prime found')


def points_n1(rep, p):
    """0-skeleton of CP^1: list of orbits; orbit = {'div': index (0 = P_0), 'points': [(chart, alpha)], 'factor': expr}
    chart 'inf' means z0=0 (variable z0 at 0 with z1=1); chart 'aff' means z0=1, variable z1 = alpha."""
    z0, z1 = rep.z0, rep.zsyms[0]
    orbits = []
    if True:   # P_0 even or odd? a0 == -1 means odd -> no residue there
        if rep.a0 == 0:
            orbits.append({'div': 0, 'points': [('inf', 0)], 'factor': z0})
    for i, d in enumerate(rep.divisors, start=1):
        if d.parity != 'even':
            continue
        pa = sp.expand(d.p_affine)
        _, fl = sp.factor_list(sp.Poly(pa, z1))
        for f, mult in fl:
            fe = f.as_expr()
            cm = _modp_coeffs(fe, z1, p)
            rts = [rt for rt, mm in (_roots_mod(cm, p) or [])]
            if len(rts) != sp.Poly(fe, z1).degree():
                raise RuntimeError('divisor does not split mod p')
            orbits.append({'div': i, 'points': [('aff', rt) for rt in rts], 'factor': fe})
    return orbits


def residue_n1(rep, mu, Q_hom, point, p, K=None):
    """Residue (mod p, up to a point-dependent constant common to all forms) of omega_mu[Q] at the point.
    Returns 0 if the form is regular there or the Laurent coefficient vanishes."""
    divs, _ = form_data(rep, make_term(rep, mu, Q_hom))
    z0, z1 = rep.z0, rep.zsyms[0]
    chart, alpha = point
    if chart == 'inf':
        var, sub = z0, {z1: 1}
    else:
        var, sub = z1, {z0: 1}
    Qc = sp.expand(sp.sympify(Q_hom).subs(sub))
    if Qc == 0:
        return 0
    total = sp.Rational(0); pieces = []; const = 1
    maxc = 0
    for jdx, (Pj, cj, parj, mj) in enumerate(divs):
        if cj == 0:
            continue
        cm = _modp_coeffs(sp.expand(Pj.subs(sub)), var, p)
        pl = P2(p, {(i, 0): v for i, v in enumerate(cm)}).shift(alpha, 0)
        oj = _ord_univ(pl, 0)
        total += sp.Rational(cj) * oj
        pieces.append((pl, -sp.Rational(cj)))
        # f-dependent constant u_j0^{-mu_j}
        u = _unit_series(pl, 0, p, 1)
        const = (const * pow(u[0], -int(mj), p)) % p if mj else const
        maxc = max(maxc, int(cj) + 1)
    qm = _modp_coeffs(Qc, var, p) if sp.Poly(Qc, var).degree() >= 0 else [int(Qc) % p]
    ql = P2(p, {(i, 0): v for i, v in enumerate(qm)}).shift(alpha, 0)
    oq = _ord_univ(ql, 0)
    total -= oq
    if total.q != 1 or total < 1:
        return 0
    K = K or int(total) + 2
    ser = _unit_series(ql, 0, p, K)
    # the unit series of Q keeps its own leading coefficient (not normalized) -> fine
    for pl, ej in pieces:
        ser = _mul_series(ser, _pow_series(_unit_series(pl, 0, p, K), ej, p, K), p, K)
    coeff = ser[int(total) - 1] if int(total) - 1 <= K else 0
    return (coeff * const) % p


def point_candidates_n1(cs, verbose=False, integrals_only=False):
    """Assign a = -2 to the preferred candidates from the 0-skeleton (n=1). Returns dict column -> a."""
    rep = cs.rep
    assert rep.n == 1
    p = cs.p
    orbits = points_n1(rep, p)
    # columns sorted by bare key
    cols = [c for c in cs.collist if (c[0] == 'I' or not integrals_only)]
    keys = {c: cs.form_key(c[0], c[1]) for c in cols}
    cols.sort(key=lambda c: keys[c])
    Z = [rep.z0, rep.zsyms[0]]
    a_of = {}
    report = []
    for orb in orbits:
        N = len(orb['points'])
        chosen = []
        mat_rows = []
        for c in cols:
            mu, mono = c
            if mu == 'I':
                terms, tb = cs.icol_terms[c]
                vec = [0] * N
                for t in terms:
                    if t.mu[orb['div']] < 1:
                        continue
                    cf = sp.Rational(sp.sympify(t.coeff).subs(sp.Symbol('eps'), cs.e0_rational)) if sp.sympify(t.coeff).free_symbols else sp.Rational(t.coeff)
                    cfm = (int(cf.p) * pow(int(cf.q), -1, p)) % p
                    vv = [residue_n1(rep, t.mu, t.Q_hom, pt, p) for pt in orb['points']]
                    vec = [(a + cfm * b) % p for a, b in zip(vec, vv)]
            else:
                if mu[orb['div']] < 1:
                    continue   # no pole on this divisor -> zero residue
                Q = sp.Integer(1)
                for z, e in zip(Z, mono): Q *= z ** e
                vec = [residue_n1(rep, mu, Q, pt, p) for pt in orb['points']]
            if all(v == 0 for v in vec):
                continue
            trial = mat_rows + [vec]
            Mx = flint.nmod_mat(len(trial), N, p)
            for i, row in enumerate(trial):
                for j, v in enumerate(row): Mx[i, j] = v
            if Mx.rank() > len(mat_rows):
                mat_rows.append(vec); chosen.append(c)
                if len(chosen) == N:
                    break
        for c in chosen:
            a_of[c] = -2
        report.append((orb['div'], str(orb['factor']), N, chosen))
        if verbose:
            print(f"  orbit on P{orb['div']} factor {orb['factor']} ({N} pt): candidates {chosen}")
    return a_of, report


def algorithm1_n1(cs, verbose=False, integrals_only=False):
    """Full Algorithm 1 for n=1: point candidates (a=-2) then global Laporta with key (a,w,o,|mu|,...); additional
    masters get a=-1 (= -w' with w'=n=1).  Returns (masters list of (col, key, a), a_map for all columns)."""
    a_cand, report = point_candidates_n1(cs, verbose=verbose, integrals_only=integrals_only)
    masters, keys = cs.solve(a_of=lambda c: a_cand.get(c, 0), verbose=verbose, prefer_integrals=integrals_only)
    a_final = {}
    out = []
    for m in masters:
        if m[0] != 'I' and sum(m[0]) > cs._M:
            continue   # truncation boundary artifact
        a = a_cand.get(m, 0)
        if a == 0:
            a = -1   # Algorithm 3: additional master at weight w' = n + r' = 1
        a_final[m] = a
        k = keys[m]
        if integrals_only:
            k = (k[0],) + k[2:]
        out.append((m, k, a))
    # non-masters: a = 0 (Algorithm 2 step 4 / Algorithm 3 step 4) — candidates that were eliminated lose their a
    cs.a_solve = dict(a_cand)          # the exact a-map used in the solve (needed to re-solve at other kinematic samples)
    return out, a_final, report
