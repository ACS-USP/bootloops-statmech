# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of BootLoops 1.0; see LICENSE and NOTICE beside this file.
"""Loop-by-loop maximal-cut Baikov representation (arXiv:2511.15381 sect. 2.5) for one sector of a family.

Family spec (python dict, JSON-able):
  loops      : ['k1','k2',...]
  ext        : ['p1','p2',...]            independent external momenta
  sp_rules   : {'p1*p1': '0', 'p1*p2': 's/2', ...}   external scalar products (both orders accepted)
  props      : [{'mom': {'k1':1,'p1':-1}, 'm2': 'm2'}, ...]   sigma_i = sigma_sign*(q_i^2 - m2_i)
  sigma_sign : -1 (paper: sigma = -q^2+m^2) or +1 (Kira/FIRE style D = q^2-m^2); default +1
  symbols    : ['s','t','m2']            kinematic symbols
  d0         : 4                          D = d0 - 2*eps  (even integer d0)
  scale      : 't'                        mu^2 used for dimensionless z_j = sigma_j/scale
A sector is a list of 1-based propagator indices (the cut lines).  The remaining props are the ISP pool.

Construction (Frellesvig-Papadopoulos loop-by-loop): loops are integrated innermost-first in `loop_order`.  For loop k the
"external" momenta Q_k are a basis of the span of all momentum combinations v for which k.v occurs (in k's own
propagators and in Gram polynomials inherited from inner loops); E_k=|Q_k|; k needs E_k+1 Baikov variables = its
assigned cut propagators plus ISPs from the pool.  Measure factor per loop:  Gram(Q_k)^{(E_k-D+1)/2} * Gram(k,Q_k)^{(D-E_k-2)/2}.
Composite momenta (p1+p2, ...) are kept unexpanded as dot symbols so that E_k is the true span dimension.

Output: BaikovRep: Gram factors with exponents BEFORE the cut (polynomials in the sigma variables), and the on-cut
factorization into irreducible divisors p_i(z) with raw exponents alpha_i=(a_i+b_i eps)/2, minimal-twist data
(a_i^min in {-1,0}, shift s_i=(a_raw-a_min)/2 so that the corner integrand = U_min * prod p_i^shift_i), homogenized P_i,
P_0=z0 with (a0,b0), d_U, goodness flag.
"""
from __future__ import annotations
import itertools, json
from dataclasses import dataclass, field
from fractions import Fraction
import sympy as sp

eps = sp.Symbol('eps')


def _parse(expr, symtab):
    if isinstance(expr, (int, float)):
        return sp.nsimplify(expr)
    return sp.sympify(str(expr), locals=symtab)


def _rank_basis(vectors):
    """Greedy basis (sublist) of a list of rational tuples."""
    basis, rows = [], []
    for v in vectors:
        M = sp.Matrix(rows + [list(v)])
        if M.rank() > len(rows):
            rows.append(list(v)); basis.append(v)
    return basis


class Family:
    def __init__(self, spec: dict):
        self.spec = spec
        self.name = spec.get('name', 'fam')
        self.loops = list(spec['loops'])
        self.ext = list(spec['ext'])
        self.kin = {s: sp.Symbol(s) for s in spec.get('symbols', [])}
        self.symtab = dict(self.kin)
        self.sigma_sign = int(spec.get('sigma_sign', 1))
        self.d0 = int(spec.get('d0', 4))
        self.scale = _parse(spec.get('scale', 1), self.symtab)
        self.sp_ext = {}
        for k, v in spec['sp_rules'].items():
            a, b = [t.strip() for t in k.split('*')]
            val = _parse(v, self.symtab)
            self.sp_ext[(a, b)] = val
            self.sp_ext[(b, a)] = val
        for a in self.ext:
            for b in self.ext:
                if (a, b) not in self.sp_ext:
                    raise ValueError(f'missing external scalar product rule {a}*{b}')
        self.basis = self.loops + self.ext
        self.nb = len(self.basis)
        self.props = []
        for p in spec['props']:
            mom = {k: sp.nsimplify(v) for k, v in p['mom'].items()}
            for k in mom:
                if k not in self.basis:
                    raise ValueError(f'unknown momentum {k} in propagator {p}')
            m2 = _parse(p.get('m2', 0), self.symtab)
            self.props.append((mom, m2))
        self.nprops = len(self.props)
        self.sig = [sp.Symbol(f'sig{i+1}') for i in range(self.nprops)]
        self._dotsyms = {}

    # ---- vectors -------------------------------------------------------
    def vec(self, mom):
        return tuple(sp.nsimplify(mom.get(b, 0)) for b in self.basis)

    def loop_part(self, v, k):
        i = self.basis.index(k)
        return v[i]

    def strip(self, v, names):
        """zero the components in names"""
        return tuple(0 if b in names else c for b, c in zip(self.basis, v))

    def is_zero(self, v):
        return all(c == 0 for c in v)

    def dotsym(self, k, v):
        """symbol for k . v where v is a vector with no component along already-integrated loops nor k itself"""
        # normalize v to primitive integer direction with positive leading coefficient; return (symbol, scale)
        nz = [c for c in v if c != 0]
        if not nz:
            return sp.Integer(0)
        fr = [Fraction(int(sp.numer(c)), int(sp.denom(c))) for c in v]
        from math import gcd
        den = 1
        for f in fr:
            den = den * f.denominator // gcd(den, f.denominator)
        ints = [int(f * den) for f in fr]
        g = 0
        for x in ints:
            g = gcd(g, abs(x))
        ints = [x // g for x in ints]
        lead = next(x for x in ints if x != 0)
        sgn = 1 if lead > 0 else -1
        ints = [sgn * x for x in ints]
        key = (k, tuple(ints))
        if key not in self._dotsyms:
            label = '_'.join(f'{c}{b}' for b, c in zip(self.basis, ints) if c != 0).replace('-', 'm')
            self._dotsyms[key] = (sp.Symbol(f'DOT__{k}__{label}'), tuple(ints))
        sym, _ = self._dotsyms[key]
        factor = sp.Rational(sgn * g, den)
        return factor * sym

    def dot_vec(self, v, w, later_loops):
        """v.w for vectors over basis whose loop components are only in later_loops (not yet integrated)."""
        res = sp.Integer(0)
        vE = self.strip(v, self.loops)
        wE = self.strip(w, self.loops)
        # ext.ext
        for a, ca in zip(self.basis, vE):
            if ca == 0: continue
            for b, cb in zip(self.basis, wE):
                if cb == 0: continue
                res += ca * cb * self.sp_ext[(a, b)]
        # loop terms
        for i, l in enumerate(later_loops):
            vl = self.loop_part(v, l); wl = self.loop_part(w, l)
            if vl != 0 and wl != 0:
                res += vl * wl * self.dotsym(l, self.unit(l))
            # l . (other part of w excluding loops up to and including l in later order -> to avoid double counting use loops after l plus ext)
            after = later_loops[i + 1:]
            w_rest = self.strip(w, [x for x in self.loops if x not in after])  # keep only loops after l and ext
            v_rest = self.strip(v, [x for x in self.loops if x not in after])
            if vl != 0 and not self.is_zero(w_rest):
                res += vl * self.dotsym(l, w_rest)
            if wl != 0 and not self.is_zero(v_rest):
                res += wl * self.dotsym(l, v_rest)
        return sp.expand(res)

    def unit(self, name):
        return tuple(1 if b == name else 0 for b in self.basis)

    def gram(self, vecs, later_loops):
        n = len(vecs)
        if n == 0:
            return sp.Integer(1)
        M = sp.zeros(n, n)
        for i in range(n):
            for j in range(i, n):
                M[i, j] = M[j, i] = self.dot_vec(vecs[i], vecs[j], later_loops)
        return sp.expand(M.det(method='berkowitz'))

    def sigma_expr(self, i, later_loops):
        mom, m2 = self.props[i - 1]
        v = self.vec(mom)
        return sp.expand(self.sigma_sign * (self.dot_vec(v, v, later_loops) - m2))


@dataclass
class GramFactor:
    name: str
    poly_sigma: sp.Expr      # polynomial in sigma variables and kinematics, before cut
    A: int                   # exponent = (A + B*eps)/2
    B: int


@dataclass
class Divisor:
    name: str
    p_affine: sp.Expr
    P_hom: sp.Expr
    deg: int
    a_raw: int
    b: int
    a_min: int
    parity: str
    shift: int


@dataclass
class BaikovRep:
    family: Family
    sector: list
    loop_order: list
    isps: list
    n: int
    grams: list
    const_factors: list
    divisors: list
    a0: int
    b0: int
    dU: sp.Rational
    zsyms: list
    z0: sp.Symbol
    gram_cut_factorization: dict
    good: bool
    externals: dict = field(default_factory=dict)
    shifts: dict = field(default_factory=dict)
    loopdata: dict = field(default_factory=dict)
    kind: str = 'loop_by_loop'
    notes: list = field(default_factory=list)

    def kdot(self, k, w):
        """k'.w (k' = shifted loop momentum of loop k) expressed through loop-k Baikov variables (sigma symbols) and
        dot symbols of later loops; raises ValueError if w is not in span(Q_k)."""
        fam = self.family
        ld = self.loopdata[k]
        Q, u, sol = ld['Q'], ld['u'], ld['sol']
        w = tuple(sp.nsimplify(c) for c in w)
        if fam.is_zero(w):
            return sp.Integer(0)
        if w == tuple(sp.Integer(c) for c in fam.unit(k)):
            return sp.expand(u[0].subs(sol))
        E = len(Q)
        if E == 0:
            raise ValueError('not in span')
        Qmat = sp.Matrix([list(q) for q in Q])
        c = sp.symbols(f'c0:{E}')
        res = sp.linsolve((Qmat.T, sp.Matrix([list(w)]).T), *c)
        if not res:
            raise ValueError('not in span')
        cvals = list(res)[0]
        if any(ci.free_symbols for ci in cvals):
            raise ValueError('underdetermined')
        return sp.expand(sum(ci * u[j + 1] for j, ci in enumerate(cvals)).subs(sol))

    def resolve(self, expr, maxiter=10):
        """substitute all U__ and DOT__ symbols through the loop chain until only sigma/kinematic symbols remain"""
        fam = self.family
        expr = sp.expand(expr)
        for _ in range(maxiter):
            syms = [x for x in expr.free_symbols if str(x).startswith('DOT__') or str(x).startswith('U__')]
            if not syms:
                return expr
            rep = {}
            for sym in syms:
                nm = str(sym)
                if nm.startswith('U__'):
                    k = nm.split('__')[1]
                    rep[sym] = self.loopdata[k]['sol'][sym]
                else:
                    for (kk, ints), (s2, _) in fam._dotsyms.items():
                        if s2 == sym:
                            w = tuple(sp.Integer(c) for c in ints)
                            # account for the shift of loop kk: DOT(kk, w) symbols created AFTER construction mean k'.w already
                            rep[sym] = self.kdot(kk, w)
                            break
            expr = sp.expand(expr.xreplace(rep))
        raise RuntimeError('resolve did not converge: ' + str(expr.free_symbols))

    def div(self, name):
        for d in self.divisors:
            if d.name == name:
                return d
        raise KeyError(name)

    def summary(self):
        lines = [f'family={self.family.name} sector={self.sector} loop_order(inner first)={self.loop_order} z=sigma{self.isps}/({self.family.scale}) n={self.n} good={self.good}',
                 f'  P0 = z0  (a0={self.a0}, b0={self.b0})  dU={self.dU}']
        for k in self.loop_order:
            lines.append(f'  loop {k}: shift(ref prop, v0)={self.shifts.get(k)} E={len(self.externals.get(k, []))} Q={self.externals.get(k)}')
        for d in self.divisors:
            lines.append(f'  {d.name} = {d.P_hom}   deg={d.deg} alpha_raw=({d.a_raw}{d.b:+d}eps)/2  a_min={d.a_min} {d.parity} shift={d.shift}')
        for g in self.grams:
            c, fac = self.gram_cut_factorization[g.name]
            lines.append(f'  {g.name}: exponent ({g.A}{g.B:+d}eps)/2 ; on cut = [{c}] * ' + ' * '.join(f'{k}^{m}' for k, m in fac.items()))
        lines.append('  U_min = z0^(%s) * ' % sp.nsimplify(f'{self.a0}/2+{self.b0}/2*eps') + ' * '.join(f'{d.name}^({sp.Rational(d.a_min,2)}{"+" if d.b>=0 else "-"}{abs(sp.Rational(d.b,2))}eps)' for d in self.divisors))
        for nt in self.notes:
            lines.append('  note: ' + nt)
        return '\n'.join(lines)


def loop_by_loop(fam: Family, sector, loop_order=None, isp_choice=None, verbose=False) -> BaikovRep:
    sector = sorted(int(i) for i in sector)
    cut = set(sector)
    pool = [i for i in range(1, fam.nprops + 1) if i not in cut]
    loop_order = list(loop_order) if loop_order else list(fam.loops)
    integrated = []
    used_isps = []
    pending = []    # [name, expr, A, B]
    externals = {}
    shifts = {}
    loopdata = {}
    for li, k in enumerate(loop_order):
        later = loop_order[li + 1:]

        def assignable(i):
            mom = fam.props[i - 1][0]
            return mom.get(k, 0) != 0 and not any(mom.get(l, 0) != 0 for l in integrated)
        mine = [i for i in sector if assignable(i)]
        cand_isps = [i for i in pool if i not in used_isps and assignable(i)]
        if isp_choice and k in isp_choice:
            trial_sets = [tuple(isp_choice[k])]
        else:
            trial_sets = []
            for r in range(0, len(cand_isps) + 1):
                trial_sets += list(itertools.combinations(cand_isps, r))
        uk_t = tuple(sp.Integer(c) for c in fam.unit(k))

        def k_syms_in(polys):
            out = []
            for (kk, ints), (sym, _) in list(fam._dotsyms.items()):
                if kk == k and any(pe.has(sym) for pe in polys):
                    out.append((tuple(sp.Integer(c) for c in ints), sym))
            return out
        chosen = None
        for ts in trial_sets:
            vars_k = mine + list(ts)
            others = {}
            for i in vars_k:
                mom = fam.props[i - 1][0]
                v = fam.vec(mom)
                ck = fam.loop_part(v, k)
                others[i] = (ck, tuple(c / ck for c in fam.strip(v, integrated + [k])))   # q_i = ck*(k + other)
            shift_cands = [None] + [i for i in vars_k if not fam.is_zero(others[i][1])]
            for i0 in shift_cands:
                v0 = others[i0][1] if i0 is not None else tuple(sp.Integer(0) for _ in fam.basis)
                # shifted pending polys: k = k' - v0
                pend2 = [list(item) for item in pending]
                if i0 is not None:
                    repl = {}
                    for w, sym in k_syms_in([it[1] for it in pend2]):
                        if w == uk_t:
                            repl[sym] = sym - 2 * fam.dotsym(k, v0) + fam.dot_vec(v0, v0, later)
                        else:
                            repl[sym] = sym - fam.dot_vec(v0, w, later)
                    for it in pend2:
                        it[1] = sp.expand(it[1].xreplace(repl))
                sh_others = {i: tuple(sp.nsimplify(a - b) for a, b in zip(others[i][1], v0)) for i in vars_k}
                pend_vecs = [w for w, sym in k_syms_in([it[1] for it in pend2]) if w != uk_t]
                V = [v for v in list(sh_others.values()) + pend_vecs if not fam.is_zero(v)]
                Q = _rank_basis(V)
                E = len(Q)
                if len(vars_k) != E + 1:
                    continue
                u = [sp.Symbol(f'U__{k}__{j}') for j in range(E + 1)]
                Qmat = sp.Matrix([list(q) for q in Q]) if E else None

                def k_dot(vother, u=u, Qmat=Qmat, E=E):
                    if fam.is_zero(vother):
                        return sp.Integer(0)
                    if tuple(vother) == uk_t:
                        return u[0]
                    solv = sp.Matrix([list(vother)]).T
                    c = sp.symbols(f'c0:{E}') if E else ()
                    if not E:
                        raise ValueError('not in span')
                    res = sp.linsolve((Qmat.T, solv), *c)
                    if not res:
                        raise ValueError('not in span')
                    cvals = list(res)[0]
                    if any(ci.free_symbols for ci in cvals):
                        raise ValueError('underdetermined')
                    return sp.expand(sum(ci * u[j + 1] for j, ci in enumerate(cvals)))
                eqs = []
                try:
                    for i in vars_k:
                        mom, m2 = fam.props[i - 1]
                        ck = others[i][0]
                        oth = sh_others[i]
                        expr = ck ** 2 * (u[0] + 2 * k_dot(oth) + fam.dot_vec(oth, oth, later)) - m2
                        eqs.append(sp.Eq(sp.expand(fam.sigma_sign * expr), fam.sig[i - 1]))
                    # pending dot symbols must be expressible
                    for w in pend_vecs:
                        k_dot(w)
                    sol = sp.solve(eqs, u, dict=True)
                except ValueError:
                    continue
                if len(sol) != 1 or any(x not in sol[0] for x in u):
                    continue
                chosen = (list(ts), Q, sol[0], u, k_dot, pend2, i0, v0)
                break
            if chosen is not None:
                break
        if chosen is None:
            raise RuntimeError(f'loop {k}: no valid Baikov variable completion (mine={mine}, cand_isps={cand_isps})')
        ts, Q, sol, u, k_dot, pend2, i0, v0 = chosen
        pending = pend2
        shifts[k] = (i0, dict((b, c) for b, c in zip(fam.basis, v0) if c != 0))
        used_isps += ts
        externals[k] = [dict((b, c) for b, c in zip(fam.basis, q) if c != 0) for q in Q]
        loopdata[k] = {'Q': list(Q), 'E': len(Q), 'u': list(u), 'sol': dict(sol), 'v0': v0, 'later': list(later), 'integrated_before': list(integrated)}
        E = len(Q)
        # Gram matrices in terms of u's and outer dots
        vecsQ = list(Q)
        # G_full(k,Q): entries k.k=u0, k.q_j=u_{j}, q_i.q_j = dot_vec
        M = sp.zeros(E + 1, E + 1)
        M[0, 0] = u[0]
        for j in range(E):
            M[0, j + 1] = M[j + 1, 0] = u[j + 1]
        for i in range(E):
            for j in range(i, E):
                M[i + 1, j + 1] = M[j + 1, i + 1] = fam.dot_vec(vecsQ[i], vecsQ[j], later)
        G_full = sp.expand(M.det(method='berkowitz'))
        G_ext = sp.expand(M[1:, 1:].det(method='berkowitz')) if E else sp.Integer(1)
        A_ext, B_ext = (E - fam.d0 + 1), 2
        A_full, B_full = (fam.d0 - E - 2), -2
        G_full = sp.expand(G_full.subs(sol))
        G_ext = sp.expand(G_ext.subs(sol))
        # substitute k's dot symbols in pending polys: DOT__k__v -> k_dot(v) with u->sol
        dsub = {}
        for (kk, ints), (sym, _) in list(fam._dotsyms.items()):
            if kk == k:
                vv = tuple(sp.Integer(c) for c in ints)
                try:
                    dsub[sym] = sp.expand(k_dot(vv).subs(sol))
                except ValueError:
                    dsub[sym] = None
        for item in pending:
            for sym, val in dsub.items():
                if item[1].has(sym):
                    if val is None:
                        raise RuntimeError(f'{item[0]}: dot {sym} not expressible for loop {k}')
                    item[1] = sp.expand(item[1].subs(sym, val))
        pending.append([f'Gfull_{k}', G_full, A_full, B_full])
        if E > 0:
            pending.append([f'Gext_{k}', G_ext, A_ext, B_ext])
        integrated.append(k)
        if verbose:
            print(f'loop {k}: vars={mine}+isps{ts} shift={shifts[k]} E={E} Q={externals[k]}')
    return _assemble(fam, sector, loop_order, used_isps, pending, externals, shifts, loopdata, 'loop_by_loop')


def _assemble(fam, sector, loop_order, isps, pending, externals, shifts, loopdata, kind):
    grams = []
    for name, expr, A, B in pending:
        left = [s for s in expr.free_symbols if str(s).startswith('DOT__') or str(s).startswith('U__')]
        if left:
            raise RuntimeError(f'{name}: unresolved symbols {left}')
        grams.append(GramFactor(name, expr, A, B))
    n = len(isps)
    zs = [sp.Symbol(f'z{j+1}') for j in range(n)]
    z0 = sp.Symbol('z0')
    cutsub = {fam.sig[i - 1]: 0 for i in sector}
    zsub = {fam.sig[isps[j] - 1]: zs[j] * fam.scale for j in range(n)}
    kin = sorted(fam.kin.values(), key=str)
    dom = sp.QQ.frac_field(*kin) if kin else sp.QQ
    divisors = {}
    order = []
    gram_fac = {}
    const_factors = []
    for g in grams:
        expr = sp.expand(g.poly_sigma.subs(cutsub).subs(zsub))
        if expr == 0:
            raise RuntimeError(f'{g.name} vanishes on the cut — degenerate representation')
        if n == 0 or not (set(sp.sympify(expr).free_symbols) & set(zs)):
            gram_fac[g.name] = (sp.factor(expr), {})
            const_factors.append((sp.factor(expr), g.A, g.B))
            continue
        P = sp.Poly(expr, *zs, domain=dom)
        const, flist = P.factor_list()
        const = const.as_expr() if hasattr(const, 'as_expr') else sp.sympify(const)
        fac = {}
        for f, m in flist:
            fm = f.monic()
            lc = f.LC()
            const = const * (lc.as_expr() if hasattr(lc, 'as_expr') else sp.sympify(lc)) ** m
            key = sp.expand(fm.as_expr())
            if key not in divisors:
                nm = f'P{len(divisors) + 1}'
                divisors[key] = {'name': nm, 'A': 0, 'B': 0, 'deg': fm.total_degree()}
                order.append(key)
            divisors[key]['A'] += m * g.A
            divisors[key]['B'] += m * g.B
            fac[divisors[key]['name']] = fac.get(divisors[key]['name'], 0) + m
        gram_fac[g.name] = (sp.factor(const), fac)
    divs = []
    for key in order:
        d = divisors[key]
        a_raw, b = d['A'], d['B']
        a_min = 0 if a_raw % 2 == 0 else -1
        shift = (a_raw - a_min) // 2
        deg = d['deg']
        P_h = sp.expand(sp.cancel(sp.together(z0 ** deg * key.subs({z: z / z0 for z in zs}, simultaneous=True))))
        divs.append(Divisor(d['name'], key, P_h, deg, a_raw, b, a_min, 'odd' if a_min == -1 else 'even', shift))
    sum_odd_deg = sum(d.deg for d in divs if d.parity == 'odd')
    a0 = 0 if sum_odd_deg % 2 == 0 else -1
    b0 = -sum(d.b * d.deg for d in divs)
    dU = sp.Rational(-1, 2) * (sum_odd_deg + (1 if a0 == -1 else 0))
    # 'good' (sect. 2.5.3): every exponent alpha_i = (a_i + b_i eps)/2 must be non-integer, incl. alpha_0 for n>0
    good = all(not (d.a_raw % 2 == 0 and d.b == 0) for d in divs) and (n == 0 or not (a0 == 0 and b0 == 0))
    notes = []
    for d in divs:
        if d.a_raw % 2 == 0 and d.b == 0:
            notes.append(f'{d.name} has integer eps-independent exponent {d.a_raw}/2 -> not regulated by the twist (bad representation)')
    if n > 0 and a0 == 0 and b0 == 0:
        notes.append('alpha_0 = 0: z0 not regulated (bad representation)')
    rep = BaikovRep(fam, sector, loop_order, isps, n, grams, const_factors, divs, a0, b0, dU, zs, z0, gram_fac, good, externals, notes)
    rep.shifts = shifts
    rep.loopdata = loopdata
    rep.kind = kind
    return rep


def best_representation(fam: Family, sector, verbose=False):
    """Try all loop orders (automatic minimal ISP completion); return (best good rep with minimal n, all tried)."""
    tried = []
    for order in itertools.permutations(fam.loops):
        try:
            rep = loop_by_loop(fam, sector, loop_order=list(order))
            tried.append(rep)
        except Exception as e:   # noqa
            if verbose:
                print('order', order, 'failed:', repr(e))
    goodreps = [r for r in tried if r.good]
    pick = min(goodreps, key=lambda r: r.n) if goodreps else (min(tried, key=lambda r: r.n) if tried else None)
    return pick, tried


def load_family(path_or_dict):
    if isinstance(path_or_dict, dict):
        return Family(path_or_dict)
    with open(path_or_dict) as f:
        return Family(json.load(f))


def numeric_spec(spec: dict, seed=1, lo=3, hi=97, values=None):
    """Copy of a family spec with every kinematic symbol replaced by a random integer (distinct primes-ish values).
    Used to run iota/integers fast at generic numeric kinematics; the caller must check that the divisor structure
    (number, degrees, parities, exponents) matches the symbolic representation (accidental factorization guard)."""
    import random as _r
    rng = _r.Random(seed)
    syms = list(spec.get('symbols', []))
    vals = {}
    pool = [v for v in range(lo, hi) if all(v % q for q in range(2, int(v ** 0.5) + 1))]
    rng.shuffle(pool)
    for s_, v in zip(syms, pool):
        vals[s_] = v
    if values:
        for k_, v in values.items():
            vals[k_] = sp.nsimplify(v)
    symtab = {s_: sp.Symbol(s_) for s_ in syms}
    def sub(e):
        if isinstance(e, (int, float)):
            return str(e)
        return str(sp.sympify(str(e), locals=symtab).subs({symtab[k]: v for k, v in vals.items()}))
    new = dict(spec)
    new['symbols'] = []
    new['sp_rules'] = {k: sub(v) for k, v in spec['sp_rules'].items()}
    new['props'] = [{'mom': dict(p['mom']), 'm2': sub(p.get('m2', 0))} for p in spec['props']]
    new['scale'] = sub(spec.get('scale', 1))
    new['name'] = spec.get('name', 'fam') + f'@num{seed}'
    new['_numeric_values'] = vals
    return new


def structure(rep):
    """divisor structure signature used to guard numeric specializations"""
    return (rep.n, tuple(sorted((d.deg, d.a_raw, d.b) for d in rep.divisors)), rep.a0, rep.b0, tuple(rep.isps), tuple(rep.loop_order))


def democratic(fam: Family, sector) -> BaikovRep:
    """Democratic (standard) Baikov representation on the maximal cut: one polynomial B = Gram(l_1..l_L, p_1..p_E) in ALL
    propagator variables, exponent (D-L-E-1)/2; every non-sector propagator is a Baikov variable (no missing ISPs).
    Always 'good' (sect. 2.5.3) but n = N_props - |sector| is larger than loop-by-loop."""
    sector = sorted(int(i) for i in sector)
    L = len(fam.loops); E = len(fam.ext)
    # unknown scalar products l_i.l_j, l_i.p_a
    unk = {}
    for i, a in enumerate(fam.loops):
        for b in fam.loops[i:]:
            unk[(a, b)] = unk[(b, a)] = sp.Symbol(f'SPd_{a}_{b}')
        for b in fam.ext:
            unk[(a, b)] = unk[(b, a)] = sp.Symbol(f'SPd_{a}_{b}')
    def dot(a, b):
        if a in fam.ext and b in fam.ext:
            return fam.sp_ext[(a, b)]
        return unk[(a, b)]
    def sq(mom):
        return sp.expand(sum(ca * cb * dot(a, b) for a, ca in mom.items() for b, cb in mom.items()))
    eqs = []
    for i, (mom, m2) in enumerate(fam.props):
        eqs.append(sp.Eq(sp.expand(fam.sigma_sign * (sq(mom) - m2)), fam.sig[i]))
    syms = sorted(set(unk.values()), key=str)
    if len(syms) != fam.nprops:
        raise RuntimeError(f'democratic: {len(syms)} scalar products vs {fam.nprops} propagators (family not complete)')
    sol = sp.solve(eqs, syms, dict=True)
    if len(sol) != 1:
        raise RuntimeError('democratic: propagators do not determine the scalar products')
    sol = sol[0]
    allm = fam.loops + fam.ext
    n_ = len(allm)
    M = sp.zeros(n_, n_)
    for i in range(n_):
        for j in range(n_):
            M[i, j] = dot(allm[i], allm[j])
    B = sp.expand(M.det(method='berkowitz').subs(sol))
    A = fam.d0 - L - E - 1
    pending = [['B', B, A, -2]]
    isps = [i for i in range(1, fam.nprops + 1) if i not in sector]
    return _assemble(fam, sector, list(fam.loops), isps, pending, {}, {}, {}, 'democratic')


def representation(fam: Family, sector, verbose=False):
    """Policy: best good loop-by-loop representation with minimal n; if none (degenerate cut Gram or an unregulated
    divisor), the democratic representation. Returns (rep, note)."""
    pick, tried = best_representation(fam, sector, verbose=verbose)
    if pick is not None and pick.good:
        return pick, f'loop_by_loop n={pick.n} order={pick.loop_order}'
    why = 'no valid loop-by-loop completion' if pick is None else 'loop-by-loop not good: ' + '; '.join(pick.notes)
    rep = democratic(fam, sector)
    return rep, f'democratic n={rep.n} ({why})'
