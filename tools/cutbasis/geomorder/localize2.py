# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of BootLoops 1.0; see LICENSE and NOTICE beside this file.
"""Algorithm 1 for n = 2 sectors (points -> curves -> surface), arXiv:2511.15381 sect. 4.1, in the numeric mod-p framework.

Realization used here (documented deviation from the letter of the paper, same mathematics):
 * 0-skeleton: points q on an even curve C_i = {P_i = 0} where the restricted twist has INTEGER local exponent (even
   points: intersections with other even divisors, even-multiplicity contacts with odd divisors).  Candidate at q = the
   simplest column (form or Feynman-integral image) with non-zero two-fold residue Res_q Res_{P_i}; a = -2n = -4.
 * Localization on C_i: instead of generating the localized IBPs with the modified twist U^loc, we use that the residue
   map sends global relations to relations among residues: the localized system = all global rows (IBP, cancellation,
   image rows) whose columns all have mu_i <= 1, projected onto the columns with mu_i = 1 (mu_i = 0 columns are regular
   along C_i and have zero residue).  Masters of this projected system (minimal basis under (a, w, o, |mu|, dots), with
   a = -4 for the point candidates lying on C_i) that are not point candidates get a = -(n + 1) = -3 (Algorithm 3 on the
   localization).  (Rows containing mu_i >= 2 columns are not used: possible loss of some localized relations -> the
   candidate set can only be too LARGE, never too small; the global run then eliminates.)
 * Merge (Algorithm 2) is replaced by the union of candidates; the global Laporta with (a, w, o, |mu|, dots) decides.
 * Global run: masters not yet flagged get a = -n = -2; non-masters a = 0.
Validated against the paper's Moller sector 93 (a=-4: Psi_1100[1], Psi_1010[.], Psi_0110[z0]; a=-3: Psi_0010[1], Psi_0011[z0^2];
a=-2: Psi_0001[1]) in geomorder/tests.
"""
from __future__ import annotations
import random
import sympy as sp
import flint
from .integers import (P2, form_data, _rho_data, _projective_points_pair, _newton_param, _compose, _intpoly, _ord_univ,
                       _unit_series, _mul_series, _pow_series, PRIMES)
from .iota import make_term


# ---------------------------------------------------------------------------------------------------------------- residues

def _laurent_value(PcL, FL, EL, NL, p, K):
    """At the origin of the smooth curve branch PcL=0: rho = NL prod FL^EL db.  Returns (order, value) where rho ~ value *
    tau^(-order) dtau * (point-dependent constant common to all forms: the u0^(half-integer) parts) ; value includes the
    Laurent coefficient at tau^-1 when order >= 1 is an integer (the residue), else None."""
    da = PcL.d.get((1, 0), 0); db = PcL.d.get((0, 1), 0)
    extra = []
    if da % p:
        phi = _newton_param(PcL, p, K, solve_for=0)
        subsfun = lambda poly: _compose(poly, phi, P2(p, {(0, 1): 1}), K); slot = 1
    elif db % p:
        phi = _newton_param(PcL, p, K, solve_for=1)
        subsfun = lambda poly: _compose(poly, P2(p, {(1, 0): 1}), phi, K); slot = 0
        extra.append(phi.deriv(0))
    else:
        return None
    total = sp.Rational(0); pieces = []; const = 1
    for FLj, Ej in zip(FL, EL):
        if Ej == 0: continue
        rj = subsfun(FLj); oj = _ord_univ(rj, slot); total -= sp.Rational(Ej) * oj
        us = _unit_series(rj, slot, p, 1)
        Ej = sp.Rational(Ej)
        eint = Ej if Ej.q == 1 else Ej + sp.Rational(1, 2)     # drop the common u0^(-1/2) of odd divisors
        const = const * pow(us[0], int(eint) % (p - 1), p) % p if eint >= 0 else const * pow(pow(us[0], -1, p), int(-eint), p) % p
        pieces.append((rj, Ej))
    rN = subsfun(NL); total -= _ord_univ(rN, slot)
    for jd in extra: total -= _ord_univ(jd, 0)
    if total.q != 1 or total < 1:
        return (total, None)
    ser = _unit_series(rN, slot, p, K)
    for rj, Ej in pieces:
        ser = _mul_series(ser, _pow_series(_unit_series(rj, slot, p, K), Ej, p, K), p, K)
    for jd in extra:
        ser = _mul_series(ser, _unit_series(jd, 0, p, K), p, K)
    coeff = ser[int(total) - 1] if int(total) - 1 <= K else 0
    return (total, (coeff * const) % p)


class PointData:
    """Even points of the 0-skeleton with everything needed to evaluate two-fold residues of forms there."""
    def __init__(self, rep, rng):
        self.rep = rep
        self.Z = (rep.z0, rep.zsyms[0], rep.zsyms[1])
        # geometry at eps-independent level: divisors with parity
        self.PH = [rep.z0] + [sp.expand(d.P_hom) for d in rep.divisors]
        self.PH = [_intpoly(P) for P in self.PH]
        self.par = ['even' if rep.a0 == 0 else 'odd'] + [d.parity for d in rep.divisors]
        self.points = []    # dicts: {'co': coords, 'p': prime, 'V': sorted divisors vanishing at the point, 'curves': even ones}
        nd = len(self.PH)
        for i in range(nd):
            for j in range(i + 1, nd):
                if self.par[i] != 'even' and self.par[j] != 'even':
                    continue
                pts = None; puse = None
                for pr in rng.sample(PRIMES, len(PRIMES)):
                    pts = _projective_points_pair(self.PH[i], self.PH[j], self.Z, pr)
                    if pts is not None:
                        puse = pr; break
                if pts is None:
                    continue
                z0, z1, z2 = self.Z
                for co in pts:
                    V = [k for k in range(nd) if int(sp.expand(self.PH[k].subs({z0: co[0], z1: co[1], z2: co[2]}))) % puse == 0]
                    if (V[0], V[1]) != (i, j):
                        continue      # this point belongs to the lexicographically first pair of its vanishing set
                    curves = [k for k in V if self.par[k] == 'even']
                    if curves:
                        self.points.append({'co': co, 'p': puse, 'V': V, 'curves': curves})

    def residue2(self, mu, Q_hom, pt, K=8, curve=None):
        """two-fold residue (value mod p up to a point constant) of omega_mu[Q] at pt (dict from self.points);
        0 if no integer pole >= 1 there or the Laurent coefficient vanishes."""
        rep = self.rep
        i = pt['curves'][0] if curve is None else curve; p = pt['p']; co = pt['co']
        divs, _ = form_data(rep, make_term(rep, mu, Q_hom))
        m = divs[i][3]
        if divs[i][2] != 'even' or m < 1:
            return 0
        CH = [c for (P, c, par, mm) in divs]
        QH = _intpoly(sp.sympify(Q_hom)) if sp.sympify(Q_hom) != 0 else sp.Integer(0)
        if QH == 0:
            return 0
        ch = next(k for k in range(3) if co[k] == 1)
        Pc, F, E, N, a, b = _rho_data(self.PH, CH, QH, i, m, ch, self.Z)
        qa = co[self.Z.index(a)]; qb = co[self.Z.index(b)]
        PcL = P2.from_sympy(Pc, a, b, p).shift(qa, qb)
        if PcL.d.get((1, 0), 0) % p == 0 and PcL.d.get((0, 1), 0) % p == 0:
            return 0   # singular point of the curve: not handled
        FL = [P2.from_sympy(Fj, a, b, p).shift(qa, qb) for Fj in F]
        NL = P2.from_sympy(N, a, b, p).shift(qa, qb)
        res = _laurent_value(PcL, FL, E, NL, p, K)
        if res is None or res[1] is None:
            return 0
        return res[1]


# ---------------------------------------------------------------------------------------------------------------- algorithm 1

def _column_form(cs, c):
    """(mu, Q_hom, coeff-list) describing a column: form columns -> single term; integral columns -> image terms"""
    if c[0] == 'I':
        terms, tb = cs.icol_terms[c]
        out = []
        for t in terms:
            cf = sp.sympify(t.coeff)
            cf = sp.Rational(cf.subs(sp.Symbol('eps'), cs.e0_rational)) if cf.free_symbols else sp.Rational(cf)
            out.append((t.mu, t.Q_hom, cf))
        return out
    mu, mono = c
    Q = sp.Integer(1)
    for z, e in zip(cs.Z, mono): Q *= z ** e
    return [(mu, Q, sp.Integer(1))]


def point_candidates_n2(cs, pd: PointData, cols_sorted, verbose=False):
    """For every even point of the 0-skeleton: the simplest column with a non-zero two-fold residue there (via any even
    divisor through the point) -> a = -4.  Returns (a_map, entries) with entries = (point index, candidate, curves)."""
    a_of = {}
    entries = []
    nvar = cs.rep.n
    # a point candidate is a SIMPLE form (|mu| <= n+1); a higher-|mu| 'simplest column with a residue' only
    # appears when the residue evaluation failed for the true candidate (seen on Moller 93 in one chart) -> skip and flag
    cols_sorted = [c for c in cols_sorted if c[0] == 'I' or sum(c[0]) <= nvar + 1]
    for n, pt in enumerate(pd.points):
        p = pt['p']
        chosen = None
        for c in cols_sorted:
            hit = False
            for cur in pt['curves']:
                val = 0
                for mu, Q, cf in _column_form(cs, c):
                    if mu[cur] < 1:
                        continue
                    r = pd.residue2(mu, Q, pt, curve=cur)
                    if r:
                        cfm = (int(cf.p) * pow(int(cf.q), -1, p)) % p
                        val = (val + cfm * r) % p
                if val % p:
                    hit = True; break
            if hit:
                chosen = c; break
        if chosen is not None:
            a_of[chosen] = -4
        entries.append((n, chosen, list(pt['curves'])))
        if verbose:
            print(f"  point {pt['co']} V={pt['V']} even curves={pt['curves']}: candidate {chosen}")
    return a_of, entries


class LocSystem:
    """IBP system of the localization on the even divisor P_i (sect. 3.7.2-3.7.3): objects = forms with mu_i = 1 and Q mod
    P_i (represented by global column labels (mu, monomial)); twist U^loc = P_s^(alpha_s + d_i b_i eps/(2 d_s)) prod_{k!=i,s}
    P_k^alpha_k (the equation after line 2203 of the arXiv v2 TeX source); relations: tangential IBPs with the logarithmic vector fields
    L_jk = (d_k P_i) d_j - (d_j P_i) d_k  (TeX label ibp_localised_Pn of the arXiv v2 TeX source, chart-free form), cancellation identities for m != i with the
    U^loc exponents (TeX label eq_cancellation_localised_Pn), and the reduction Q -> Q mod P_i (omega_mu[P_i q] has no residue)."""
    def __init__(self, cs, i):
        self.cs = cs; self.i = i
        self.p = cs.p
        # scale polynomial: first k != i with generic exponent in U^loc
        self.bloc = list(cs.b)
        di, bi = cs.deg[i], cs.b[i]
        self.s = None
        for k in range(cs.ND + 1):
            if k == i: continue
            bk = sp.Rational(cs.b[k]) + sp.Rational(di * bi, cs.deg[k])
            if cs.amin[k] % 2 != 0 or bk != 0:
                self.s = k; self.bloc[k] = bk; break
        if self.s is None:
            raise RuntimeError('no generic scale polynomial')
        self.rows = []
        self.cols = {}; self.collist = []

    def col(self, mu, mono):
        key = (tuple(mu), tuple(mono))
        # map to the global column index space lazily: we keep our own list but with global labels
        if key not in self.cols:
            self.cols[key] = len(self.collist); self.collist.append(key)
        return self.cols[key]

    def add_poly(self, row, mu, polyd, factor):
        cs = self.cs
        if any(m < 0 for m in mu) or mu[self.i] != 1:
            return
        for mono, c in polyd.items():
            v = (factor * cs.modp(c)) % self.p
            if v:
                k = self.col(mu, mono)
                row[k] = (row.get(k, 0) + v) % self.p

    def generate(self, M):
        cs = self.cs; i = self.i; nv = cs.n + 1
        from .cohomology import monomials, _compositions
        Z = cs.Z
        Pi = cs.P[i]
        dPi_e = [sp.expand(sp.diff(Pi, z)) for z in Z]
        for tot in range(1, M + 1):
            for mu in _compositions(tot, cs.ND + 1):
                if mu[i] != 1:
                    continue
                # localized seed degree: sum_{k != i} mu_k d_k - dU - n + 1
                dq_plus = int(sum(mu[k] * cs.deg[k] for k in range(cs.ND + 1) if k != i) - cs.dU - cs.n + 1)
                dq = cs.dQ(mu)      # degree of Q for columns omega_mu[Q] (global formula, mu_i = 1 included)
                # IBPs
                for Mplus in monomials(nv, dq_plus):
                    Qp = sp.Integer(1)
                    for z, e in zip(Z, Mplus): Qp *= z ** e
                    for j in range(nv):
                        for k in range(j + 1, nv):
                            # L_jk Q+ and L_jk P_m
                            LQ = sp.expand(dPi_e[k] * sp.diff(Qp, Z[j]) - dPi_e[j] * sp.diff(Qp, Z[k]))
                            row = {}
                            if LQ != 0:
                                self.add_poly(row, mu, cs._pdict(LQ), cs.inv_e0)
                            for m in range(cs.ND + 1):
                                if m == i: continue
                                LP = sp.expand(dPi_e[k] * sp.diff(cs.P[m], Z[j]) - dPi_e[j] * sp.diff(cs.P[m], Z[k]))
                                if LP == 0: continue
                                mu2 = list(mu); mu2[m] += 1
                                self.add_poly(row, tuple(mu2), cs._pdict(sp.expand(Qp * LP)), 1)
                            row = {k_: v for k_, v in row.items() if v}
                            if row: self.rows.append(row)
                # cancellation identities (m != i) with U^loc exponents, and P_i-reduction (m == i)
                if dq >= 0:
                    for mono in monomials(nv, dq):
                        for m in range(cs.ND + 1):
                            mu2 = list(mu); mu2[m] += 1
                            if m == i:
                                # omega_{mu+e_i}[P_i mono] has mu_i = 2: not an object; instead: omega_mu[P_i q] = 0 for q of degree dq - d_i
                                continue
                            row = {}
                            self.add_poly(row, tuple(mu2), cs.mulmono(cs.Pd[m], mono), 1)
                            fac = sp.Rational(cs.amin[m], 2) - mu[m]
                            val = (cs.modp(fac) * cs.inv_e0 + cs.modp(sp.Rational(self.bloc[m]) / 2)) % self.p
                            self.add_poly(row, tuple(mu), {tuple(mono): 1}, (-val) % self.p)
                            row = {k_: v for k_, v in row.items() if v}
                            if row: self.rows.append(row)
                    # Q mod P_i: omega_mu[P_i * q] == 0 (zero residue) for all q of degree dq - d_i
                    for qmono in monomials(nv, dq - cs.deg[i]):
                        row = {}
                        self.add_poly(row, mu, cs.mulmono(cs.Pd[i], qmono), 1)
                        row = {k_: v for k_, v in row.items() if v}
                        if row: self.rows.append(row)

    def add_integral_column(self, nu, terms):
        """residue of an integral image: only terms with mu_i = 1 contribute (mu_i = 0: regular; mu_i >= 2 not allowed -> skip column)"""
        cs = self.cs
        if any(t.mu[self.i] >= 2 for t in terms):
            return None
        key = ('I', tuple(int(x) for x in nu))
        if not any(t.mu[self.i] == 1 for t in terms):
            return None
        self.cols[key] = len(self.collist); self.collist.append(key)
        row = {self.cols[key]: 1}
        for t in terms:
            if t.mu[self.i] != 1: continue
            c = sp.sympify(t.coeff)
            c = sp.Rational(c.subs(sp.Symbol('eps'), cs.e0_rational)) if c.free_symbols else sp.Rational(c)
            self.add_poly(row, t.mu, cs._pdict(sp.expand(t.Q_hom)), (self.p - cs.modp(c)) % self.p)
        self.rows.append({k: v for k, v in row.items() if v})
        return key

    def solve(self, keyfun, want=()):
        """RREF; returns (masters, reductions) with reductions[c] = {master: coeff} for requested pivot columns c."""
        ncol = len(self.collist)
        if ncol == 0:
            return [], {}
        order = sorted(range(ncol), key=lambda n: keyfun(self.collist[n]), reverse=True)
        pos = {ci: k for k, ci in enumerate(order)}
        Mx = flint.nmod_mat(len(self.rows), ncol, self.p)
        for r, row in enumerate(self.rows):
            for c, v in row.items():
                Mx[r, pos[c]] = v
        R, rank = Mx.rref()
        piv = {}
        for r in range(rank):
            for cc in range(ncol):
                if int(R[r, cc]) != 0:
                    piv[cc] = r; break
        masters = [self.collist[order[cc]] for cc in range(ncol) if cc not in piv]
        red = {}
        for c in want:
            if c not in self.cols: continue
            cc = pos[self.cols[c]]
            if cc not in piv: continue
            r = piv[cc]; inv = pow(int(R[r, cc]), -1, self.p)
            red[c] = {self.collist[order[k]]: (-int(R[r, k]) * inv) % self.p for k in range(cc + 1, ncol) if int(R[r, k]) != 0}
        return masters, red


def localized_masters(cs, i, keys, a_point, M, verbose=False, integral_cols=()):
    """Masters of the localization on even divisor i (LocSystem), ordered by (a_point, w, o, |mu|, dots); also the
    reductions of the point candidates that are objects of this localization but not masters of it (for the merge)."""
    ls = LocSystem(cs, i)
    ls.generate(M)
    for nu, terms in integral_cols:
        ls.add_integral_column(nu, terms)
    def keyfun(c):
        if c not in keys:
            keys[c] = cs.form_key(c[0], c[1], None, False)
        return (a_point.get(c, 0),) + tuple(keys[c][1:])
    ms, red = ls.solve(keyfun, want=[c for c in a_point if c in ls.cols])
    boundary = {m for m in ms if m[0] != 'I' and sum(m[0]) > M}
    ms = [m for m in ms if m not in boundary]
    ms.sort(key=keyfun)
    if verbose:
        print(f'  localization P{i} (scale P{ls.s}): {len(ls.rows)} rows x {len(ls.collist)} cols, masters {ms}')
    return ms, red, ls


def merge_candidates(entries, locinfo, keys, p, verbose=False):
    """Algorithm 2 (merge on the 1-skeleton), 'lazy' form as in the paper's worked examples: for every point
    entry (curve i, point q, candidate b, partner divisor j) the glued objects are Res_i b and Res_j b; a reduction
    equation is added when b is an object of that localization but not one of its masters.  Laporta on the glued
    system; surviving candidates keep a=-4, the others get a=0."""
    cands = []
    for (n_, b, curves) in entries:
        if b is not None and b not in cands: cands.append(b)
    cols = {}; collist = []
    def col(c):
        if c not in cols: cols[c] = len(collist); collist.append(c)
        return cols[c]
    rows = []
    seen = set()
    for (n_, b, curves) in entries:
        if b is None: continue
        for cur in curves:
            if cur not in locinfo or (b, cur) in seen: continue
            seen.add((b, cur))
            ms, red, ls = locinfo[cur]
            if b in red:      # object of localization `cur`, reducible there
                row = {col(b): 1}
                for mcol, v in red[b].items():
                    row[col(mcol)] = (row.get(col(mcol), 0) - v) % p
                rows.append(row)
    for c in cands: col(c)
    if not rows:
        return dict((c, -4) for c in cands)
    order = sorted(range(len(collist)), key=lambda n: ((-4 if collist[n] in cands else 0),) + tuple(keys.get(collist[n], (0, 9, 9, 9))[1:]), reverse=True)
    pos = {ci: k for k, ci in enumerate(order)}
    Mx = flint.nmod_mat(len(rows), len(collist), p)
    for r, row in enumerate(rows):
        for c, v in row.items(): Mx[r, pos[c]] = v
    R, rank = Mx.rref()
    piv = set()
    for r in range(rank):
        for cc in range(len(collist)):
            if int(R[r, cc]) != 0: piv.add(cc); break
    out = {c: (0 if pos[cols[c]] in piv else -4) for c in cands}
    if verbose:
        print('  merge: candidates kept a=-4:', [c for c in cands if out[c] == -4], ' demoted:', [c for c in cands if out[c] == 0])
    return out


def algorithm1_n2(cs, seed=1, integrals_only=False, verbose=False):
    """Returns (masters [(col, key, a)], a_map, report).  cs: generated CutSystem (with integral columns if wanted)."""
    rep = cs.rep
    assert rep.n == 2
    rng = random.Random(seed)
    pd = PointData(rep, rng)
    cols = [c for c in cs.collist if (c[0] == 'I' or not integrals_only)]
    keys = {c: cs.form_key(c[0], c[1], None, integrals_only) for c in cs.collist}
    cols_sorted = sorted(cols, key=lambda c: keys[c])
    # drop truncation-boundary forms from candidacy
    cols_sorted = [c for c in cols_sorted if c[0] == 'I' or sum(c[0]) <= cs._M]
    a_pt, preport = point_candidates_n2(cs, pd, cols_sorted, verbose=verbose)
    lreport = {}
    even = [i for i, par in enumerate(pd.par) if par == 'even']
    locinfo = {}
    icolinfo = [(c[1], cs.icol_terms[c][0]) for c in cs.collist if c[0] == 'I']
    for i in even:
        ms, red, ls = localized_masters(cs, i, keys, dict(a_pt), cs._M, verbose=verbose, integral_cols=icolinfo)
        locinfo[i] = (ms, red, ls)
        lreport[i] = ms
    a_merged = merge_candidates(preport, locinfo, keys, cs.p, verbose=verbose)
    a_all = {c: a for c, a in a_merged.items() if a < 0}
    for i in even:
        for mcol in locinfo[i][0]:
            if mcol not in a_pt and mcol not in a_all:
                a_all[mcol] = -3
    masters, keys2 = cs.solve(a_of=lambda c: a_all.get(c, 0), verbose=verbose, prefer_integrals=integrals_only)
    cs.a_solve = dict(a_all)           # the exact a-map used in the solve
    out = []; a_final = {}
    for m in masters:
        if m[0] != 'I' and sum(m[0]) > cs._M:
            continue
        a = a_all.get(m, 0)
        if a == 0:
            a = -2
        a_final[m] = a
        k = keys2[m]
        if integrals_only:
            k = (k[0],) + k[2:]
        out.append((m, k, a))
    out.sort(key=lambda t: (t[2], t[1][1:]))
    return out, a_final, {'points': [(str(pd.points[n_]['co']), pd.points[n_]['V'], str(b)) for n_, b, cv in preport], 'localizations': {i: [str(c) for c in v] for i, v in lreport.items()}, 'n_points': len(pd.points), 'a_candidates_after_merge': {str(k): v for k, v in a_merged.items()}}
