# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of BootLoops 1.0; see LICENSE and NOTICE beside this file.
"""Maximal-cut twisted-cohomology Laporta (arXiv:2511.15381 sect. 3.4, 4.1) at numeric kinematics over F_p.
"TeX label" names below are label keys of the arXiv v2 TeX source of the paper.

Columns = forms omega_mu[m] (mu in N^{N_D+1}, m a monomial in z_0..z_n of degree d_Q(mu)); rows = IBP identities
(TeX label eq_ibp) for seeds (mu, j, Q_+ monomial) and cancellation identities (TeX label eq_cancellation); distribution identities
are implicit in the monomial basis.  Gaussian elimination with columns ordered by the geometric key, most complicated
first, gives the master integrands (= standard representatives, the minimal elements of their classes) as the
non-pivot columns.  'a' (Algorithm 1-3 preference from localizations) enters as the leading slot of the key.

Scope of this file: the GLOBAL system on CP^n (any n) + point-localization candidates (Algorithm 1 step 1 and the a=-2n
assignment) for n=1 and n=2.  Curve localizations (n=2, Algorithm 1 recursion + merge) live in localize2.py.
"""
from __future__ import annotations
import itertools, random, time
from dataclasses import dataclass, field
import sympy as sp
import flint
from .baikov import BaikovRep
from .iota import make_term
from .integers import integers, PRIMES


def monomials(nvars, deg):
    if deg < 0:
        return []
    out = []
    for c in itertools.combinations_with_replacement(range(nvars), deg):
        e = [0] * nvars
        for i in c: e[i] += 1
        out.append(tuple(e))
    return out


@dataclass
class CutSystem:
    rep: BaikovRep
    p: int
    e0: int                      # eps value mod p
    cols: dict = field(default_factory=dict)      # (mu, mono) -> index
    collist: list = field(default_factory=list)
    rows: list = field(default_factory=list)      # list of dict col->val mod p
    keycache: dict = field(default_factory=dict)
    icol_terms: dict = field(default_factory=dict)

    # ---- basic data
    def __post_init__(self):
        rep = self.rep
        self.n = rep.n
        self.Z = [rep.z0] + list(rep.zsyms)
        self.P = [rep.z0] + [d.P_hom for d in rep.divisors]
        self.deg = [1] + [d.deg for d in rep.divisors]
        self.amin = [rep.a0] + [d.a_min for d in rep.divisors]
        self.b = [rep.b0] + [d.b for d in rep.divisors]
        self.ND = len(self.P) - 1
        self.dU = rep.dU
        # polynomial dicts for P_i and their partial derivatives: {mono: coeff (Rational)}
        self.Pd = [self._pdict(P) for P in self.P]
        self.dPd = [[self._pdict(sp.diff(P, z)) for z in self.Z] for P in self.P]
        self.inv_e0 = pow(self.e0, -1, self.p)

    def _pdict(self, expr):
        expr = sp.expand(expr)
        if expr == 0:
            return {}
        Pp = sp.Poly(expr, *self.Z)
        return {tuple(m): sp.Rational(c) for m, c in zip(Pp.monoms(), Pp.coeffs())}

    def modp(self, c):
        c = sp.Rational(c)
        return (int(c.p) * pow(int(c.q), -1, self.p)) % self.p

    def dQ(self, mu):
        return int(sum(mu[i] * self.deg[i] for i in range(self.ND + 1)) - self.dU - self.n - 1)

    def col(self, mu, mono):
        key = (tuple(mu), tuple(mono))
        if key not in self.cols:
            self.cols[key] = len(self.collist); self.collist.append(key)
        return self.cols[key]

    def add_integral_column(self, nu, terms, tiebreak):
        """Add an unknown X_nu for a Feynman integral whose iota image (at the system's eps value) is `terms`
        (list of FormTerm, eps-free), with the row  X_nu - sum(image columns) = 0.  tiebreak = tuple used to order
        integral columns among themselves (smaller = simpler, e.g. (dots, sps, ...))."""
        key = ('I', tuple(int(x) for x in nu))
        if key in self.cols:
            return self.cols[key]
        self.cols[key] = len(self.collist); self.collist.append(key)
        self.icol_terms[key] = (terms, tiebreak)
        row = {self.cols[key]: 1}
        for t in terms:
            c = sp.sympify(t.coeff)
            polyd = self._pdict(sp.expand(c * t.Q_hom))
            self.add_poly(row, t.mu, polyd, self.p - 1)
        row = {k: v for k, v in row.items() if v}
        self.rows.append(row)
        return self.cols[key]

    # ---- relations
    def add_poly(self, row, mu, polyd, factor):
        """row += factor * omega_mu[poly] expanded in monomials (factor in F_p)"""
        if any(m < 0 for m in mu):
            return
        for mono, c in polyd.items():
            v = (factor * self.modp(c)) % self.p
            if v:
                k = self.col(mu, mono)
                row[k] = (row.get(k, 0) + v) % self.p

    @staticmethod
    def mulmono(d, mono):
        return {tuple(a + b for a, b in zip(m, mono)): c for m, c in d.items()}

    def ibp(self, mu, j, Mplus):
        """(1/eps) omega_mu[d_j M] + sum_i omega_{mu+e_i}[M d_j P_i] = 0"""
        row = {}
        if Mplus[j] > 0:
            dM = {tuple(m - (1 if k == j else 0) for k, m in enumerate(Mplus)): Mplus[j]}
            self.add_poly(row, mu, dM, self.inv_e0)
        for i in range(self.ND + 1):
            dP = self.dPd[i][j]
            if not dP:
                continue
            mu2 = list(mu); mu2[i] += 1
            self.add_poly(row, tuple(mu2), self.mulmono(dP, Mplus), 1)
        row = {k: v for k, v in row.items() if v}
        if row:
            self.rows.append(row)

    def cancellation(self, mu, j, mono):
        """omega_{mu+e_j}[P_j m] = (1/eps)(a_j/2 - mu_j + b_j eps/2) omega_mu[m]"""
        row = {}
        mu2 = list(mu); mu2[j] += 1
        self.add_poly(row, tuple(mu2), self.mulmono(self.Pd[j], mono), 1)
        fac = (sp.Rational(self.amin[j], 2) - mu[j]) * sp.Integer(1)   # eps^-1 part
        val = (self.modp(fac) * self.inv_e0 + self.modp(sp.Rational(self.b[j], 2))) % self.p
        self.add_poly(row, tuple(mu), {tuple(mono): 1}, (-val) % self.p)
        row = {k: v for k, v in row.items() if v}
        if row:
            self.rows.append(row)

    def generate(self, M, mu_filter=None):
        """all IBP seeds and cancellation identities with |mu| <= M (mu_i >= 0). mu_filter(mu)->bool restricts seeds."""
        t0 = time.time()
        nv = self.n + 1
        for tot in range(0, M + 1):
            for mu in _compositions(tot, self.ND + 1):
                if mu_filter and not mu_filter(mu):
                    continue
                dq = self.dQ(mu)
                # IBP seeds: Q+ of degree dq+1
                for Mplus in monomials(nv, dq + 1):
                    for j in range(nv):
                        self.ibp(mu, j, Mplus)
                # cancellation: m of degree dq (columns omega_mu[m]) and each j
                if dq >= 0:
                    for mono in monomials(nv, dq):
                        for j in range(self.ND + 1):
                            self.cancellation(mu, j, mono)
        return time.time() - t0

    # ---- keys and elimination
    def form_key(self, mu, mono, a_of=None, prefer_integrals=False):
        if mu == 'I':
            key = ('I', mono)
            if key in self.keycache:
                k = self.keycache[key]
            else:
                terms, tb = self.icol_terms[key]
                ks = []
                for t in terms:
                    I = integers(self.rep, t, nsamples=1)
                    ks.append((I.w, I.o, int(sum(t.mu)), I.r))
                k = max(ks) if ks else (0, 0, 0, 0)
                self.keycache[key] = k
            a = a_of(key) if a_of else 0
            w, o, am, r = k
            tb = self.icol_terms[key][1]
            # class (a,w,o,|mu|), then: integrals (flag 0) before pure forms (flag 1), then the integral tiebreak
            return (a, 0, w, o, am, 0, tuple(tb), ()) if prefer_integrals else (a, w, o, am, 0, tuple(tb), ())
        key = (tuple(mu), tuple(mono))
        if key in self.keycache:
            k = self.keycache[key]
        else:
            Q = sp.Integer(1)
            for z, e in zip(self.Z, mono):
                Q *= z ** e
            # reduce monomial divisors: if P_i is a monomial (single z) with mu_i>0 and divides Q -> equivalent lower form
            mu_r = list(mu); Qr = Q
            changed = True
            while changed:
                changed = False
                for i in range(self.ND + 1):
                    if mu_r[i] > 0 and len(self.Pd[i]) == 1:
                        (pm, pc), = self.Pd[i].items()
                        Pi = sp.Integer(1)
                        for z, e in zip(self.Z, pm): Pi *= z ** e
                        q, r = sp.div(Qr, Pi, *self.Z)
                        if r == 0:
                            Qr = sp.expand(q); mu_r[i] -= 1; changed = True
            term = make_term(self.rep, tuple(mu_r), Qr)
            I = integers(self.rep, term, nsamples=1)
            k = (I.w, I.o, int(sum(mu_r)), I.r)
            self.keycache[key] = k
        a = a_of(key) if a_of else 0
        w, o, am, r = k
        # tie-breaks ("the dots"): d_Q ascending is implied by |mu| at fixed structure; then mu tuple, then monomial
        return (a, 1, w, o, am, 1, tuple(mu), tuple(mono)) if prefer_integrals else (a, w, o, am, 1, tuple(mu), tuple(mono))

    def solve(self, a_of=None, verbose=False, prefer_integrals=False):
        """RREF with columns sorted most-complicated-first; returns list of master columns (mu, mono) sorted by key."""
        t0 = time.time()
        ncol = len(self.collist)
        keys = {c: self.form_key(c[0], c[1], a_of, prefer_integrals) for c in self.collist}
        order = sorted(range(ncol), key=lambda i: keys[self.collist[i]], reverse=True)   # most complex first
        pos = {ci: k for k, ci in enumerate(order)}
        nrow = len(self.rows)
        Mx = flint.nmod_mat(nrow, ncol, self.p)
        for r, row in enumerate(self.rows):
            for c, v in row.items():
                Mx[r, pos[c]] = v
        R, rank = Mx.rref()
        pivots = set()
        r = 0
        for r in range(rank):
            # first nonzero in row r
            for c in range(ncol):
                if int(R[r, c]) != 0:
                    pivots.add(c); break
        masters = [self.collist[order[c]] for c in range(ncol) if c not in pivots]
        masters.sort(key=lambda c: keys[c])
        if verbose:
            print(f'  solve: {nrow} rows x {ncol} cols, rank {rank}, {len(masters)} non-pivot, {time.time()-t0:.1f}s')
        return masters, keys


def _compositions(total, k):
    if k == 1:
        yield (total,); return
    for i in range(total + 1):
        for rest in _compositions(total - i, k - 1):
            yield (i,) + rest


def masters_on_cut(rep: BaikovRep, M=5, p=None, e0=None, a_of=None, verbose=False, seed=0):
    """Standard representatives (basis of H^n_omega) by truncated Laporta at |mu|<=M. Returns (masters, info).
    masters = list of ((mu, mono), key).  Stability must be checked by the caller (compare M and M+1)."""
    rng = random.Random(seed)
    p = p or rng.choice(PRIMES)
    if rep.n == 1 and not rep.family.kin:
        from .localize import good_prime_n1
        p = good_prime_n1(rep, rng)
    e0r = sp.Rational(7, 13) if e0 is None else sp.Rational(e0)
    e0 = (int(e0r.p) * pow(int(e0r.q), -1, p)) % p
    cs = CutSystem(rep, p, e0)
    cs._M = M
    cs.e0_rational = e0r
    tg = cs.generate(M)
    masters, keys = cs.solve(a_of=a_of, verbose=verbose)
    # drop truncation-boundary artifacts: columns with |mu| == M+1 cannot be reduced within the truncation
    inner = [m for m in masters if m[0] == 'I' or sum(m[0]) <= M]
    boundary = [m for m in masters if m[0] != 'I' and sum(m[0]) > M]
    info = {'p': p, 'e0': e0, 'M': M, 'rows': len(cs.rows), 'cols': len(cs.collist), 'gen_s': tg,
            'n_nonpivot': len(masters), 'n_boundary_nonpivot': len(boundary)}
    return [(m, keys[m]) for m in inner], info, cs


def fmt(rep, col):
    mu, mono = col
    if mu == 'I':
        return f"{rep.family.name.split('@')[0]}{list(mono)}"
    Z = [rep.z0] + list(rep.zsyms)
    Q = '*'.join(f'{z}^{e}' if e > 1 else f'{z}' for z, e in zip(Z, mono) if e) or '1'
    return f"omega_{list(mu)}[{Q}]"
