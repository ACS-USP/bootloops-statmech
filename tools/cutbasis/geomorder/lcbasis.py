# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of BootLoops 1.0; see LICENSE and NOTICE beside this file.
"""Linear-combination (BASISLC) realization of the geometric master basis.

For one sector: the Algorithm 1-3 basis of H (forms with their 'a') is fixed once at a base numeric kinematic point.
V = span of the images of the candidate Feynman integrals (dim V <= dim H; the mismatch comes from symmetries).
The geometric realization of V is its ECHELON basis with respect to the geometric order on the H-basis: for every
pivot form j in J (the simplest forms reachable as leading terms) one element of V whose image has coefficient 1 on
omega_j and 0 on every other pivot form.  Each such element is written as the sparsest linear combination of the
simplest candidate integrals (support fixed at the base point, anchor integral coefficient 1); the remaining
coefficients are rational functions of (eps, kinematics) obtained by solving the same small linear system at many
numeric samples mod p (each sample = one specialized cut system + one rref) and reconstructing with ratrec.py
(two primes + third-prime verification).  Output: Kira preferred_masters text (bare integrals + BASISLC blocks).
"""
from __future__ import annotations
import copy, itertools, random, time
from multiprocessing import Pool
import sympy as sp
import flint
from .baikov import Family, numeric_spec, representation, loop_by_loop, eps
from .fastiota import FastRep
from .cohomology import CutSystem, fmt
from .localize import algorithm1_n1, good_prime_n1
from .localize2 import algorithm1_n2
from .integers import PRIMES
from .sweep import enumerate_integrals
from .masters import kira_tiebreak
from . import ratrec

PRIMES61 = [2305843009213693951, 2305843009213693921, 2305843009213693907, 2305843009213693877]  # 2^61-1 and the next primes below


# ------------------------------------------------------------------------------------------------------ specialization
def _subs_expr(e, sub):
    if isinstance(e, sp.Basic):
        return sp.expand(sp.sympify(e).subs(sub))
    return e


def specialize_rep(srep, values):
    """Numeric copy of a symbolic BaikovRep at integer kinematics `values` (dict symbol-name -> int): same divisor
    order, same names, same monic normalization (rejected if a leading coefficient vanishes at the point)."""
    fam = srep.family
    sub = {fam.kin[k]: sp.Integer(v) for k, v in values.items()}
    nspec = numeric_spec(fam.spec, values=values)
    nspec['scale'] = str(sp.sympify(fam.spec.get('scale', 1)).subs(sub))
    NF = Family(nspec)
    NF._dotsyms = dict(fam._dotsyms)
    rep = copy.copy(srep)
    rep.family = NF
    divs = []
    zs = [srep.z0] + list(srep.zsyms)
    for d in srep.divisors:
        d2 = copy.copy(d)
        d2.p_affine = sp.expand(sp.together(d.p_affine.subs(sub)))
        d2.P_hom = sp.expand(sp.together(d.P_hom.subs(sub)))
        n_, den_ = sp.fraction(sp.together(d2.P_hom))
        if den_.free_symbols or d2.P_hom == 0:
            raise ValueError('bad point: divisor degenerates')
        za = list(srep.zsyms)
        Ps = sp.Poly(d.p_affine, *za); Pn = sp.Poly(d2.p_affine, *za)
        if Pn.total_degree() != Ps.total_degree() or Pn.LC() != 1 or sp.Poly(d2.P_hom, *zs).total_degree() != d.deg:
            raise ValueError('bad point: leading coefficient of a divisor vanishes')
        divs.append(d2)
    rep.divisors = divs
    rep.grams = [copy.copy(g) for g in srep.grams]
    for g in rep.grams:
        g.poly_sigma = sp.expand(g.poly_sigma.subs(sub))
    rep.const_factors = [(sp.sympify(c).subs(sub), A, B) for (c, A, B) in srep.const_factors]
    rep.gram_cut_factorization = {k: (sp.sympify(c).subs(sub), dict(f)) for k, (c, f) in srep.gram_cut_factorization.items()}
    ld = {}
    for k, v in srep.loopdata.items():
        v2 = dict(v)
        v2['sol'] = {s: sp.expand(sp.sympify(e).subs(sub)) for s, e in v['sol'].items()}
        ld[k] = v2
    rep.loopdata = ld
    return rep


def symbolic_rep(spec, sector):
    """best good loop-by-loop representation (symbolic kinematics) or democratic; (rep, note)"""
    F = Family(spec)
    return representation(F, sorted(sector))


# ------------------------------------------------------------------------------------------------------ cut solves
def _rref_reductions(cs, order_key, want):
    """RREF of cs with columns sorted by order_key descending (most complex first).  Returns (nonpivot columns,
    reductions {col: {master col: value}} for the columns in `want` that are pivots)."""
    ncol = len(cs.collist)
    order = sorted(range(ncol), key=lambda i: order_key(cs.collist[i]), reverse=True)
    pos = {ci: k for k, ci in enumerate(order)}
    Mx = flint.nmod_mat(len(cs.rows), ncol, cs.p)
    for r, row in enumerate(cs.rows):
        for c, v in row.items():
            Mx[r, pos[c]] = v
    R, rank = Mx.rref()
    pivrow = {}
    for r in range(rank):
        for c in range(ncol):
            if int(R[r, c]) != 0:
                pivrow[c] = r; break
    nonpiv = [cs.collist[order[c]] for c in range(ncol) if c not in pivrow]
    red = {}
    for col in want:
        if col not in cs.cols:
            continue
        pc = pos[cs.cols[col]]
        if pc not in pivrow:
            red[col] = None; continue
        r = pivrow[pc]; inv = pow(int(R[r, pc]), -1, cs.p)
        red[col] = {cs.collist[order[k]]: (-int(R[r, k]) * inv) % cs.p for k in range(pc + 1, ncol) if int(R[r, k]) != 0}
    return nonpiv, red


def _order_key_factory(cs, a_of):
    def key(c):
        if c[0] == 'I':
            return (1, tuple(cs.icol_terms[c][1]))
        return (0,) + tuple(cs.form_key(c[0], c[1], a_of, prefer_integrals=False))
    return key


class SectorLC:
    """Base analysis + LC structure for one sector."""

    def __init__(self, spec, sector, seed=5, M=5, cand_dots=2, cand_sps=2, eps_value=sp.Rational(7, 13), verbose=False, pool_size=40, max_support=3):
        self.spec = spec; self.sector = sorted(sector); self.seed = seed; self.M = M
        self.cand_dots, self.cand_sps = cand_dots, cand_sps
        self.eps_value = sp.Rational(eps_value); self.verbose = verbose
        self.pool_size = pool_size; self.max_support = max_support
        self.log = []
        self.srep, self.note = symbolic_rep(spec, self.sector)
        self.rep_alternatives = []     # other good loop orders with the same n (used on suspicion, see base())
        if self.srep.kind == 'loop_by_loop':
            F_ = Family(spec)
            import itertools as _it
            for order in _it.permutations(F_.loops):
                if list(order) == list(self.srep.loop_order):
                    continue
                try:
                    r_ = loop_by_loop(F_, self.sector, loop_order=list(order))
                except Exception:
                    continue
                if r_.good and r_.n == self.srep.n:
                    self.rep_alternatives.append(r_)
        self.N = len(spec['props'])
        self.fam = spec['name']
        self.kinsyms = list(spec.get('symbols', []))

    # ---- base point
    def base(self):
        """Base analysis; if the H-basis looks suspicious (a master at the truncation boundary |mu| >= M-1, or a
        localization flag), retry with the alternative loop orders and keep the first clean one (all attempts logged)."""
        res = self._base_once()
        def suspicious(r):
            if not r.get('dimH') or r.get('n') == 0 or not hasattr(self, 'H'):
                return False       # n = 0: one master (the corner), no cut system, nothing to retry
            if any(sum(c[0]) >= self.M - 1 for c in self.H):
                return True
            # a point of the 0-skeleton lying on >= 2 even divisors must carry a two-fold-residue candidate; 'None' there
            # means the residue evaluation failed in this chart (seen on Moller 93, k1-first chart) -> try another chart
            even = set([0] if self.srep.a0 == 0 else []) | {i + 1 for i, d in enumerate(self.srep.divisors) if d.parity == 'even'}
            for pt in r.get('point_report', []) or []:
                co, V, cand = pt
                if V == 'n1':
                    continue
                if cand in (None, 'None') and sum(1 for v in V if v in even) >= 2:
                    return True
            return False
        attempts = [(list(self.srep.loop_order), [h['form'] for h in res.get('H', [])], suspicious(res))]
        alts = list(self.rep_alternatives)
        while suspicious(res) and alts:
            alt = alts.pop(0)
            self.log.append(f'H-basis suspicious with loop order {self.srep.loop_order} (master at |mu|>=M-1): retrying with {alt.loop_order}')
            self.srep = alt; self.note = f'loop_by_loop n={alt.n} order={alt.loop_order} (retry)'
            res = self._base_once()
            attempts.append((list(alt.loop_order), [h['form'] for h in res.get('H', [])], suspicious(res)))
        res['loop_order_attempts'] = attempts
        if suspicious(res):
            res.setdefault('flags', []).append('H-basis suspicious (truncation-boundary master or a two-even-divisor point without a residue candidate) in every loop order: NOT trusted')
        return res

    def _base_once(self):
        t0 = time.time()
        rng = random.Random(self.seed)
        srep = self.srep
        if srep.n == 0:
            nu = tuple(1 if i in self.sector else 0 for i in range(1, self.N + 1))
            self.result = {'n': 0, 'masters': [{'nu': list(nu), 'a': 0, 'key': [0, 0, 1], 'form': 'omega_[1][1]', 'lc': None}], 'dimH': 1, 'dimV': 1}
            self.plan = None
            self.masters = [{'form': 'omega_[1][1]', 'a': 0, 'key': [0, 0, 1], 'lc': [(list(nu), sp.Integer(1))]}]
            return self.result
        if srep.n >= 3 or srep.kind != 'loop_by_loop':
            self.result = {'n': srep.n, 'kind': srep.kind, 'masters': [], 'flags': [f'n={srep.n} ({srep.kind}): not analyzed (n<=2 loop-by-loop only)']}
            return self.result
        # numeric base point (small primes chosen by the seed)
        nspec = numeric_spec(self.spec, seed=self.seed)
        vals = {k: int(v) for k, v in nspec['_numeric_values'].items()}
        self.base_values = vals
        rep = specialize_rep(srep, vals)
        self.rep = rep
        p = good_prime_n1(rep, rng) if rep.n == 1 else rng.choice(PRIMES)
        e0r = self.eps_value
        e0 = (int(e0r.p) * pow(int(e0r.q), -1, p)) % p
        cs = CutSystem(rep, p, e0); cs._M = self.M; cs.e0_rational = e0r
        cs.generate(self.M)
        self.keycache = cs.keycache
        # H-basis on the pure form system
        if rep.n == 1:
            res, a_final, report = algorithm1_n1(cs, verbose=self.verbose, integrals_only=False)
            point_report = [('n1', 'n1', str(ch)) for d, f, n_, ch in report]
        else:
            res, a_final, report = algorithm1_n2(cs, seed=self.seed, verbose=self.verbose, integrals_only=False)
            point_report = [(str(co), [int(v) for v in V] if not isinstance(V, str) else V, str(b)) for co, V, b in report['points']]
        self.a_solve = dict(cs.a_solve)
        H = [c for c, k, a in res]
        self.H = H
        self.H_info = [{'form': fmt(rep, c), 'a': a, 'key': [k[1], k[2], k[3]]} for c, k, a in res]
        self.Hkey = {c: (a, k[1], k[2], k[3]) for c, k, a in res}
        # integral columns (all candidates) at the base point
        fr = FastRep(rep)
        cands = enumerate_integrals(self.N, self.sector, len(self.sector) + self.cand_dots, self.cand_sps)
        cands.sort(key=lambda nu: kira_tiebreak(nu, self.sector))
        self.cands = []
        self.images = {}
        self.img_absmu = {}
        for nu in cands:
            try:
                T = fr.iota(nu, eps_value=e0r)
            except Exception as e:
                self.log.append(f'iota failed {nu}: {e!r}'); continue
            if not T or max(sum(t.mu) for t in T) > self.M:
                continue
            cs.add_integral_column(nu, T, kira_tiebreak(nu, self.sector))
            self.cands.append(tuple(nu)); self.images[tuple(nu)] = T
            self.img_absmu[tuple(nu)] = max(sum(t.mu) for t in T)
        a_of = lambda c: self.a_solve.get(c, 0)
        okey = _order_key_factory(cs, a_of)
        want = [('I', nu) for nu in self.cands]
        nonpiv, red = _rref_reductions(cs, okey, want)
        self.base_nonpiv = nonpiv
        if not set(H) <= set(nonpiv):
            raise RuntimeError('H-basis forms are not all non-pivots of the base system with integral columns')
        boundary = [c for c in nonpiv if c not in H]
        # reductions onto H (drop integrals touching boundary artifacts)
        self.R = {}
        for nu in self.cands:
            r = red.get(('I', nu))
            if r is None:
                continue
            if any(c not in self.Hkey for c in r):
                self.log.append(f'{nu}: reduction touches truncation-boundary columns, dropped'); continue
            self.R[nu] = r
        self.p_base = p
        self.result = {'n': rep.n, 'note': self.note, 'dimH': len(H), 'H': self.H_info, 'p_base': p, 'base_values': vals, 'point_report': point_report,
                       'n_candidates': len(self.cands), 'n_reduced': len(self.R), 'n_boundary_nonpivot': len(boundary), 'wall_base_s': time.time() - t0}
        return self.result

    # ---- structure
    def structure(self):
        """Leading forms L of V (echelon of the image space from the complex end) and, per leading form l, the sparsest
        support S of Kira-simplest integrals whose combination has image = omega_l + (forms simpler than l), with the
        coefficients uniquely determined (rank |S|).  Stores self.plan = list of dicts, simplest master first."""
        p = self.p_base
        Hc = sorted(self.H, key=lambda c: self.Hkey[c], reverse=True)     # most complex first
        self.Hc = Hc
        cand = [nu for nu in self.cands if nu in self.R]
        N = len(Hc)
        Mx = flint.nmod_mat(len(cand), N, p)
        for i, nu in enumerate(cand):
            for j, c in enumerate(Hc):
                Mx[i, j] = self.R[nu].get(c, 0)
        Rr, rank = Mx.rref()
        L = []
        for r in range(rank):
            for j in range(N):
                if int(Rr[r, j]) != 0:
                    L.append(Hc[j]); break
        self.L = L; self.dimV = rank
        pool = cand[:self.pool_size]

        def system(S, cons, l):
            M = flint.nmod_mat(len(cons), len(S), p); Mb = flint.nmod_mat(len(cons), len(S) + 1, p)
            for col, nu in enumerate(S):
                for row, c in enumerate(cons):
                    M[row, col] = self.R[nu].get(c, 0); Mb[row, col] = M[row, col]
            Mb[cons.index(l), len(S)] = 1
            return M, Mb

        def solvable(S, cons, l):
            M, Mb = system(S, cons, l)
            k = len(S)
            if M.rank() == k and Mb.rank() == k:
                b = flint.nmod_mat(len(cons), 1, p); b[cons.index(l), 0] = 1
                sol = _solve_overdet(M, b, p, k)
                if sol is not None and all(v % p for v in sol):
                    return sol
            return None

        plan = []
        for l in L:
            above = Hc[:Hc.index(l)]
            cons = above + [l]
            found = None
            for k in range(1, self.max_support + 1):
                for S in itertools.combinations(pool, k):
                    sol = solvable(list(S), cons, l)
                    if sol is not None:
                        found = (list(S), sol); break
                if found:
                    break
            if found is None:
                # incremental: grow the support in Kira order until the constraints are solvable, then prune
                B = []
                for nu in pool:
                    B.append(nu)
                    M, Mb = system(B, cons, l)
                    if Mb.rank() == M.rank() and M.rank() >= 1:
                        # consistent; need a solution with coefficient on l = 1: check rank condition of the l-row
                        b = flint.nmod_mat(len(cons), 1, p); b[cons.index(l), 0] = 1
                        # drop dependent columns to make the solution unique
                        keep = []
                        for nu_ in B:
                            trial = keep + [nu_]
                            Mt, _ = system(trial, cons, l)
                            if Mt.rank() == len(trial):
                                keep = trial
                        sol = solvable(keep, cons, l)
                        if sol is not None:
                            # prune zeros and re-check uniqueness
                            S2 = [nu_ for nu_, v in zip(keep, sol) if v % p]
                            sol2 = solvable(S2, cons, l)
                            if sol2 is not None:
                                found = (S2, sol2); break
                if found is None:
                    raise RuntimeError(f'no support in the pool of {len(pool)} integrals for leading form {fmt(self.rep, l)}')
                self.log.append(f'leading form {fmt(self.rep, l)}: no support <= {self.max_support}; incremental support of size {len(found[0])}')
            S, sol = found
            anchor = 0
            inv = pow(sol[anchor], -1, p)
            coeffs = [v * inv % p for v in sol]
            plan.append({'j': l, 'above': above, 'form': fmt(self.rep, l), 'a': self.Hkey[l][0], 'key': list(self.Hkey[l][1:]), 'support': S, 'anchor': S[anchor], 'base_coeffs': coeffs})
        plan.sort(key=lambda e: self.Hkey[e['j']])      # simplest first (Kira file order)
        self.plan = plan
        self.S_all = sorted({nu for e in plan for nu in e['support']}, key=lambda nu: kira_tiebreak(nu, self.sector))
        self.result.update({'dimV': rank, 'L': [fmt(self.rep, c) for c in L],
                            'plan': [{'form': e['form'], 'a': e['a'], 'key': e['key'], 'support': [list(nu) for nu in e['support']], 'anchor': list(e['anchor']), 'base_coeffs': e['base_coeffs']} for e in plan]})
        return plan

    # ---- one sample (picklable worker data)
    def sample_task(self, S_all=None, M=None):
        return {'spec': self.spec, 'sector': self.sector, 'M': M or self.M, 'S_all': list(S_all if S_all is not None else self.S_all),
                'H': list(self.H), 'a_solve': self.a_solve, 'keycache': self.keycache, 'loop_order': self.srep.loop_order}


def _solve_overdet(A, b, p, k):
    """solve A c = b (A: m x k, full column rank, consistent) via k independent rows"""
    m = A.nrows()
    rows = []
    for r in range(m):
        trial = flint.nmod_mat(len(rows) + 1, k, p)
        for i, rr in enumerate(rows + [r]):
            for c in range(k):
                trial[i, c] = A[rr, c]
        if trial.rank() > len(rows):
            rows.append(r)
        if len(rows) == k:
            break
    if len(rows) < k:
        return None
    Asq = flint.nmod_mat(k, k, p); bsq = flint.nmod_mat(k, 1, p)
    for i, rr in enumerate(rows):
        for c in range(k):
            Asq[i, c] = A[rr, c]
        bsq[i, 0] = b[rr, 0]
    x = Asq.solve(bsq)
    return [int(x[i, 0]) for i in range(k)]


def solve_support(R, cons, l, S, anchor, p):
    """R: {nu: {form: val}}; cons = forms more complex than l followed by l; returns coefficient list (anchor = 1)
    or None if the support is not uniquely solvable / inconsistent / anchor vanishes."""
    k = len(S)
    A = flint.nmod_mat(len(cons), k, p); b = flint.nmod_mat(len(cons), 1, p); b[cons.index(l), 0] = 1
    for col, nu in enumerate(S):
        if nu not in R:
            return None
        for row, c in enumerate(cons):
            A[row, col] = R[nu].get(c, 0)
    if A.rank() != k:
        return None
    sol = _solve_overdet(A, b, p, k)
    if sol is None:
        return None
    for row in range(len(cons)):
        if sum(int(A[row, col]) * sol[col] for col in range(k)) % p != int(b[row, 0]):
            return None
    ia = S.index(anchor)
    if sol[ia] % p == 0:
        return None
    inv = pow(sol[ia], -1, p)
    return [v * inv % p for v in sol]


_W = {}


def _worker_init(task):
    _W['task'] = task
    spec = task['spec']
    F = Family(spec)
    _W['srep'] = loop_by_loop(F, task['sector'], loop_order=task['loop_order'])


def sample_point(args):
    """reductions of the task's integrals onto the H-basis at one (eps, kinematics) point mod p.
    args = (point dict, e0r Rational, p).  Returns {'point','e0r','p','R': {nu: {form: val}}, 'wall'} or {'error'}."""
    point, e0r, p = args
    task = _W['task']; srep = _W['srep']
    t0 = time.time()
    try:
        rep = specialize_rep(srep, point)
    except ValueError as e:
        return {'point': point, 'p': p, 'error': str(e)}
    e0r = sp.Rational(e0r)
    e0 = (int(e0r.p) * pow(int(e0r.q), -1, p)) % p
    cs = CutSystem(rep, p, e0); cs._M = task['M']; cs.e0_rational = e0r
    cs.keycache = dict(task['keycache'])
    cs.generate(task['M'])
    fr = FastRep(rep)
    added = []
    for nu in task['S_all']:
        try:
            T = fr.iota(nu, eps_value=e0r)
        except Exception:
            continue
        if not T or max(sum(t.mu) for t in T) > task['M']:
            continue
        cs.add_integral_column(nu, T, kira_tiebreak(nu, task['sector']))
        added.append(nu)
    a_solve = task['a_solve']
    okey = _order_key_factory(cs, lambda c: a_solve.get(c, 0))
    nonpiv, red = _rref_reductions(cs, okey, [('I', nu) for nu in added])
    if not set(task['H']) <= set(nonpiv):
        return {'point': point, 'p': p, 'error': 'H-basis differs at this point'}
    Hset = set(task['H'])
    R = {}
    for nu in added:
        r = red.get(('I', nu))
        if r is None or any(c not in Hset for c in r):
            continue
        R[nu] = r
    return {'point': point, 'e0r': str(e0r), 'p': p, 'R': R, 'wall': time.time() - t0}


# ------------------------------------------------------------------------------------------------------ driver
def random_points(kinsyms, p, n, rng, line=None):
    """n random points (kin dict, eps Rational); if line=(x0, v): x0 + lam*v, lam = 1..n (eps = (x0_0 + lam v_0)/1e7)."""
    pts = []
    for i in range(n):
        if line is None:
            eps_r = sp.Rational(rng.randrange(2, 10 ** 6), rng.randrange(10 ** 6, 10 ** 7))
            kin = {k: rng.randrange(2, p - 1) for k in kinsyms}
        else:
            x0, v = line
            lam = i + 1
            eps_r = sp.Rational(x0[0] + lam * v[0], 10 ** 7)
            kin = {k: (x0[1 + j] + lam * v[1 + j]) % p for j, k in enumerate(kinsyms)}
        pts.append((kin, eps_r))
    return pts


def _eps_mod(e0r, p):
    e0r = sp.Rational(e0r)
    return int(e0r.p) * pow(int(e0r.q), -1, p) % p


def reconstruct_sector(sl: SectorLC, workers=16, maxdeg=12, seed=11, verbose=True, logf=None, max_support=3, max_candidates=300):
    """Reconstruction for one analyzed sector after sl.base() and sl.structure():
    (1) probe along a random line with ALL pool integrals -> for every leading form choose the support whose
        coefficients have the lowest total degree (among the sparsest solvable supports);
    (2) dense samples at two 61-bit primes for the union of chosen supports; multivariate fit; CRT; rationals;
    (3) verification at a third prime on fresh points.  Returns the masters list with sympy coefficients."""
    rng = random.Random(seed)
    kin = sl.kinsyms
    nv = 1 + len(kin)
    syms = [eps] + [sp.Symbol(k) for k in kin]
    p1, p2, p3 = PRIMES61[0], PRIMES61[1], PRIMES61[2]
    pool = [nu for nu in sl.cands if nu in sl.R][:sl.pool_size]
    def LOG(msg):
        if logf:
            with open(logf, 'a') as fh:
                fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")

    Mstate = {'bad': set()}

    def run(points_eps, p, S_all):
        absmu = getattr(sl, 'img_absmu', {})
        Ms = max([absmu.get(nu, sl.M) for nu in S_all] + [min(3, sl.M)])
        while True:
            Ms = max(Ms, min(m for m in range(Ms, sl.M + 1) if m not in Mstate['bad']))
            task = sl.sample_task(S_all, M=Ms)
            args = [(kinv, e0r, p) for (kinv, e0r) in points_eps]
            with Pool(min(workers, len(args)), initializer=_worker_init, initargs=(task,)) as pool_:
                out = list(pool_.imap_unordered(sample_point, args))
            good = [o for o in out if 'R' in o]
            bad = [o for o in out if 'error' in o]
            LOG(f"sector {sl.sector} p={p} M={Ms}: {len(good)} good, {len(bad)} bad samples of {len(args)}; wall/sample {sorted(round(o['wall'],1) for o in good)[:2]}..{sorted(round(o['wall'],1) for o in good)[-2:]}; errors {[o['error'] for o in bad][:3]}")
            if len(good) >= len(args) - 2 or Ms >= sl.M:
                return good
            Mstate['bad'].add(Ms); Ms += 1

    def pointvec(o, p):
        return [_eps_mod(o['e0r'], p)] + [o['point'][k] % p for k in kin]

    # ---- (1) probe on a random line with the whole pool
    x0 = [rng.randrange(2, 10 ** 6)] + [rng.randrange(2, p1 - 1) for _ in kin]
    v = [rng.randrange(2, 10 ** 6)] + [rng.randrange(2, p1 - 1) for _ in kin]
    nprobe = 2 * maxdeg + 6
    probe = run(random_points(kin, p1, nprobe, rng, line=(x0, v)), p1, pool)
    if len(probe) < nprobe - 2:
        raise RuntimeError(f'probe: only {len(probe)} good samples')
    lam_of = lambda o: int((sp.Rational(o['e0r']) * 10 ** 7 - x0[0]) / v[0]) % p1
    chosen = []
    for e in sl.plan:
        l = e['j']; cons = e['above'] + [l]
        cands = []
        # supports solvable at the base point, sparsest first, Kira order
        for k in range(1, max_support + 1):
            for S in itertools.combinations(pool, k):
                if solve_support(sl.R, cons, l, list(S), S[0], sl.p_base) is not None:
                    cands.append(list(S))
                    if len(cands) >= max_candidates:
                        break
            if len(cands) >= max_candidates or (cands and k >= 1 and len(cands) >= 40):
                break
        if not cands:
            cands = [e['support']]
        best = None
        for S in cands:
            anchor = S[0]
            xs, ys = [], []
            ok = True
            for o in probe:
                c = solve_support(o['R'], cons, l, S, anchor, p1)
                if c is None:
                    ok = False; break
                xs.append(lam_of(o)); ys.append(c)
            if not ok:
                continue
            degs = []
            for i in range(len(S)):
                if i == 0:
                    degs.append((0, 0)); continue
                rr = ratrec.ratrec_univariate(xs, [y[i] for y in ys], p1)
                if rr is None:
                    ok = False; break
                degs.append((len(rr[0]) - 1, len(rr[1]) - 1))
            if not ok:
                continue
            score = (max(dn + dd for dn, dd in degs), len(S), sum(pool.index(nu) for nu in S))
            if best is None or score < best[0]:
                best = (score, S, degs)
        if best is None:
            raise RuntimeError(f'no support with reconstructible coefficients (maxdeg {maxdeg}) for {e["form"]}')
        score, S, degs = best
        chosen.append({'j': l, 'above': e['above'], 'form': e['form'], 'a': e['a'], 'key': e['key'], 'support': S, 'anchor': S[0], 'degs': degs})
        LOG(f"  {e['form']}: support {S} degrees {degs} (from {len(cands)} candidate supports)")
    sl.plan = chosen
    sl.S_all = sorted({nu for e in chosen for nu in e['support']}, key=lambda nu: kira_tiebreak(nu, sl.sector))
    Dn = max([d[0] for e in chosen for d in e['degs']] + [0]); Dd = max([d[1] for e in chosen for d in e['degs']] + [0])
    need_dense = any(len(e['support']) > 1 for e in chosen)
    exprs = {}
    nsamp = 0; nver = 0
    if need_dense:
        nunk = len(ratrec.monomials_upto(nv, Dn)) + len(ratrec.monomials_upto(nv, Dd))
        nsamp = nunk + 6
        LOG(f"dense ansatz: total degrees <= ({Dn},{Dd}) in {nv} variables -> {nunk} unknowns, {nsamp} samples per prime")
        fits = {}
        for p in (p1, p2):
            good = run(random_points(kin, p, nsamp, rng), p, sl.S_all)
            for e in chosen:
                cons = e['above'] + [e['j']]
                pts_, vals_ = [], []
                for o in good:
                    c = solve_support(o['R'], cons, e['j'], e['support'], e['anchor'], p)
                    if c is None:
                        continue
                    pts_.append(pointvec(o, p)); vals_.append(c)
                if len(pts_) < nunk + 2:
                    raise RuntimeError(f'too few good samples at p={p} for {e["form"]}: {len(pts_)}')
                for i in range(1, len(e['support'])):
                    dn, dd = e['degs'][i]
                    fit = ratrec.fit_dense(pts_, [c[i] for c in vals_], p, dn, dd, nv)
                    if fit is None:
                        fit = ratrec.fit_dense(pts_, [c[i] for c in vals_], p, dn + 1, dd + 1, nv)
                    if fit is None:
                        raise RuntimeError(f'dense fit failed for {e["form"]} coefficient {i} at p={p}')
                    fits.setdefault((str(e['j']), i), []).append((fit, p))
        for (key, i), fl in fits.items():
            numQ, denQ = ratrec.combine_primes(fl)
            exprs[(key, i)] = ratrec.to_sympy(numQ, denQ, syms)
        # ---- (3) third-prime verification
        good = run(random_points(kin, p3, 4, rng), p3, sl.S_all)
        for o in good:
            x = pointvec(o, p3)
            for e in chosen:
                cons = e['above'] + [e['j']]
                c = solve_support(o['R'], cons, e['j'], e['support'], e['anchor'], p3)
                if c is None:
                    continue
                for i in range(1, len(e['support'])):
                    val = ratrec.ratfunc_mod(exprs[(str(e['j']), i)], syms, x, p3)
                    if val is None or val != c[i]:
                        raise RuntimeError(f'third-prime verification FAILED for {e["form"]} coefficient {i}')
                    nver += 1
    masters = []
    for e in chosen:
        lc = [(list(e['support'][0]), sp.Integer(1))] + [(list(nu), exprs[(str(e['j']), i)]) for i, nu in enumerate(e['support']) if i > 0]
        masters.append({'form': e['form'], 'a': e['a'], 'key': e['key'], 'lc': lc})
    sl.result['reconstruction'] = {'plan': [{'form': e['form'], 'support': [list(nu) for nu in e['support']], 'degrees': e['degs']} for e in chosen],
                                   'probe_points': nprobe, 'samples_per_prime': nsamp, 'primes': [p1, p2], 'verify_prime': p3, 'n_verified_values': nver,
                                   'masters': [{'form': m['form'], 'a': m['a'], 'key': m['key'], 'lc': [(nu, str(c)) for nu, c in m['lc']]} for m in masters]}
    sl.masters = masters
    return masters


def kira_coeff(expr, d0, symbols_map=None):
    """sympy coefficient in (eps, kin) -> Kira string in (d, kin): eps = (d0 - d)/2"""
    d = sp.Symbol('d')
    e = sp.cancel(sp.sympify(expr).subs(eps, (d0 - d) / 2))
    s = str(e).replace('**', '^').replace(' ', '')     # with Kira 3.1 a coefficient string with spaces stalled the run (GUIDE.md FOOTGUNS)
    return s


def preferred_text(fam, sectors_results, d0=4):
    """Kira preferred_masters text from a list of per-sector results (each with 'masters' entries having 'lc' or 'nu')."""
    bare, blocks = [], []
    for res in sectors_results:
        for m in res.get('masters', []):
            lc = m.get('lc')
            if lc is None:
                if m.get('nu') is not None:
                    bare.append(f"{fam}[{','.join(str(x) for x in m['nu'])}]")
                continue
            if len(lc) == 1 and sp.simplify(sp.sympify(lc[0][1]) - 1) == 0:
                bare.append(f"{fam}[{','.join(str(x) for x in lc[0][0])}]")
            else:
                blocks.append('\n'.join(f"{fam}[{','.join(str(x) for x in nu)}]*({kira_coeff(c, d0)})" for nu, c in lc))
    txt = '\n'.join(bare) + '\n'
    for b in blocks:
        txt += '\n' + b + '\n'
    return txt
