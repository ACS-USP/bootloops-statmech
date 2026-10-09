# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of BootLoops 1.0; see LICENSE and NOTICE beside this file.
"""Fast iota-hat at NUMERIC kinematics using python-flint multivariate polynomials (fmpq_mpoly).

Same mathematics as geomorder.iota (residues of higher poles by sigma-derivatives of the uncut loop-by-loop
integrand; missing ISPs by loop-by-loop transverse averaging; eps-split; lowest terms; homogenization) but all
polynomial algebra is done in the ring Q[sig_1..sig_N, eps] with flint.  Requires a BaikovRep built from a
numeric_spec() family (no kinematic symbols).  Returns geomorder.iota.FormTerm objects, so geomorder.integers works
unchanged.  Cross-checked against the symbolic path in tests.
"""
from __future__ import annotations
from math import comb
import itertools
import sympy as sp
import flint
from .baikov import BaikovRep, eps
from .iota import FormTerm, MissingISP


def _fmpq(x):
    x = sp.Rational(x)
    return flint.fmpq(int(x.p), int(x.q))


class TD:
    """term dict: {shift tuple: mpoly}; value = sum_m C_m * prod_G G^{e_G - m_G}"""
    __slots__ = ('d', 'nG', 'ctx')

    def __init__(self, ctx, nG, d=None):
        self.ctx = ctx; self.nG = nG; self.d = {} if d is None else {k: v for k, v in d.items() if not v.is_zero()}

    @staticmethod
    def const(ctx, nG, poly):
        return TD(ctx, nG, {tuple([0] * nG): poly})

    def copy(self):
        return TD(self.ctx, self.nG, dict(self.d))

    def __add__(self, o):
        d = dict(self.d)
        for k, v in o.d.items():
            d[k] = d[k] + v if k in d else v
        return TD(self.ctx, self.nG, d)

    def __sub__(self, o):
        d = dict(self.d)
        for k, v in o.d.items():
            d[k] = d[k] - v if k in d else -v
        return TD(self.ctx, self.nG, d)

    def __mul__(self, o):
        if isinstance(o, TD):
            d = {}
            for k1, v1 in self.d.items():
                for k2, v2 in o.d.items():
                    k = tuple(a + b for a, b in zip(k1, k2))
                    d[k] = d[k] + v1 * v2 if k in d else v1 * v2
            return TD(self.ctx, self.nG, d)
        # scalar or mpoly
        return TD(self.ctx, self.nG, {k: v * o for k, v in self.d.items()})

    def shift(self, gidx, by):
        d = {}
        for k, v in self.d.items():
            k2 = list(k); k2[gidx] += by
            d[tuple(k2)] = v
        return TD(self.ctx, self.nG, d)

    def pow(self, n):
        r = TD.const(self.ctx, self.nG, self.ctx.from_dict({tuple([0] * self.ctx.nvars()): 1}))
        for _ in range(int(n)):
            r = r * self
        return r

    def is_zero(self):
        return not self.d


class FastRep:
    def __init__(self, rep: BaikovRep):
        fam = rep.family
        if fam.kin:
            raise ValueError('FastRep needs a numeric family (use baikov.numeric_spec)')
        self.rep = rep
        self.fam = fam
        N = fam.nprops
        names = tuple(f'sig{i+1}' for i in range(N)) + ('eps',)
        self.ctx = flint.fmpq_mpoly_ctx.get(names, 'lex')
        self.gens = self.ctx.gens()
        self.sig = {fam.sig[i]: self.gens[i] for i in range(N)}
        self.epsg = self.gens[N]
        self.one = self.ctx.from_dict({tuple([0] * (N + 1)): 1})
        self.symmap = dict(self.sig); self.symmap[eps] = self.epsg
        self.grams = [(g.name, self.to_mpoly(g.poly_sigma), g.A, g.B) for g in rep.grams]
        self.nG = len(self.grams)
        self.gidx = {g[0]: i for i, g in enumerate(self.grams)}
        self.eG = [self.const(sp.Rational(A, 2)) + self.const(sp.Rational(B, 2)) * self.epsg for (_, _, A, B) in self.grams]
        # divisors as polynomials in the ISP sigma variables: p_i(z = sigma/scale)
        self.scale = sp.Rational(fam.scale)
        zsub = {rep.zsyms[j]: fam.sig[rep.isps[j] - 1] / self.scale for j in range(rep.n)}
        self.pt = [self.to_mpoly(sp.expand(d.p_affine.subs(zsub, simultaneous=True))) for d in rep.divisors]
        self.gram_cut = []
        for (name, G, A, B) in self.grams:
            c, fac = rep.gram_cut_factorization[name]
            self.gram_cut.append((_fmpq(c), [fac.get(d.name, 0) for d in rep.divisors]))
        # verify the cut factorization numerically in this ring
        cutsub = {f'sig{i}': 0 for i in rep.sector}
        for (name, G, A, B), (c, Mv) in zip(self.grams, self.gram_cut):
            lhs = G.subs(cutsub)
            rhs = self.one * c
            for pi, m in zip(self.pt, Mv):
                for _ in range(m):
                    rhs = rhs * pi
            # p_affine are monic in z; in sigma variables p(sigma/scale) differs from the sigma-monic version by scale powers -> compare up to constant
            q, r = divmod(lhs, rhs)
            if not r.is_zero() or not q.is_constant():
                raise RuntimeError(f'cut factorization check failed for {name}')
        self.cutsub_all = cutsub
        self._loopcache = {}

    # -- conversions ----------------------------------------------------------------------------------------
    def const(self, x):
        return self.one * _fmpq(x)

    def to_mpoly(self, expr):
        expr = sp.expand(expr)
        P = sp.Poly(expr, *([s for s in self.fam.sig] + [eps]))
        d = {}
        for monom, c in zip(P.monoms(), P.coeffs()):
            d[tuple(monom)] = _fmpq(c)
        if not d:
            return self.ctx.from_dict({tuple([0] * self.ctx.nvars()): 0})
        return self.ctx.from_dict(d)

    def to_sympy_z(self, poly):
        """mpoly in ISP sigmas (+eps) -> sympy expr in z (sigma_j = scale z_j)"""
        rep = self.rep
        out = sp.Integer(0)
        N = self.fam.nprops
        zmap = {rep.isps[j] - 1: rep.zsyms[j] * self.scale for j in range(rep.n)}
        for monom, c in zip(poly.monoms(), poly.coeffs()):
            term = sp.Rational(int(c.p), int(c.q))
            for i, e in enumerate(monom[:N]):
                if e:
                    if i not in zmap:
                        raise RuntimeError('non-ISP sigma survived the cut')
                    term *= zmap[i] ** e
            if monom[N]:
                term *= eps ** monom[N]
            out += term
        return sp.expand(out)

    # -- missing ISP machinery -------------------------------------------------------------------------------
    def loopinfo(self, k):
        if k in self._loopcache:
            return self._loopcache[k]
        rep, fam = self.rep, self.fam
        ld = rep.loopdata[k]
        Q, E, later = ld['Q'], ld['E'], ld['later']
        Gq = [[None] * E for _ in range(E)]
        for a in range(E):
            for b in range(a, E):
                Gq[a][b] = Gq[b][a] = rep.resolve(fam.dot_vec(Q[a], Q[b], later))
        Gm = sp.Matrix(E, E, lambda a, b: Gq[a][b]) if E else sp.Matrix(0, 0, [])
        adj = Gm.adjugate() if E else None
        det = sp.expand(Gm.det()) if E else sp.Integer(1)
        u = [rep.resolve(x) for x in ld['u']]
        info = {'Q': Q, 'E': E, 'later': later, 'v0': ld['v0'], 'before': ld['integrated_before'],
                'adj': [[self.to_mpoly(adj[a, b]) for b in range(E)] for a in range(E)] if E else [],
                'det': self.to_mpoly(det), 'u': [self.to_mpoly(x) for x in u]}
        # det must be proportional to Gext_k
        if E:
            ge = self.grams[self.gidx[f'Gext_{k}']][1]
            q, r = divmod(info['det'], ge)
            if not r.is_zero() or not q.is_constant():
                raise RuntimeError(f'det Gram(Q_{k}) is not proportional to Gext_{k}')
            info['det_over_gext'] = q   # constant mpoly
        else:
            info['det_over_gext'] = self.one
        self._loopcache[k] = info
        return info

    def numerator_td(self, nu):
        """TD for prod_{ISP j} sigma_j^{-nu_j} with missing ISPs averaged (times an irrelevant overall eps-polynomial)."""
        rep, fam = self.rep, self.fam
        td = TD.const(self.ctx, self.nG, self.one)
        missing = {}
        for j in range(1, fam.nprops + 1):
            if j in rep.sector or nu[j - 1] == 0:
                continue
            if nu[j - 1] > 0:
                raise ValueError('positive index outside the sector')
            if j in rep.isps:
                td = td * (self.sig[fam.sig[j - 1]] ** (-nu[j - 1]))
            else:
                mom = fam.props[j - 1][0]
                k = next((l for l in rep.loop_order if mom.get(l, 0) != 0), None)
                if k is None:
                    raise MissingISP(f'sigma{j} has no loop momentum')
                missing.setdefault(k, []).append(j)
        for k, js in missing.items():
            info = self.loopinfo(k)
            Q, E, later, v0, before = info['Q'], info['E'], info['later'], info['v0'], info['before']
            gi_ext = self.gidx.get(f'Gext_{k}'); gi_full = self.gidx[f'Gfull_{k}']
            adj, u = info['adj'], info['u']
            dq = info['det_over_gext']          # det = dq * Gext
            A, Bc, bvecs, ws = [], [], [], []
            for j in js:
                mom, m2 = fam.props[j - 1]
                v = fam.vec(mom); ck = fam.loop_part(v, k)
                if any(mom.get(l, 0) != 0 for l in before):
                    raise MissingISP(f'sigma{j} involves an already-integrated loop')
                other = tuple(c / ck for c in fam.strip(v, before + [k]))
                w = tuple(sp.nsimplify(a - b) for a, b in zip(other, v0))
                bvec = [self.to_mpoly(rep.resolve(fam.dot_vec(Q[a], w, later))) for a in range(E)]
                w2 = self.to_mpoly(rep.resolve(fam.dot_vec(w, w, later)))
                sgn = fam.sigma_sign
                # A_j = sgn ck^2 (u0 + w2) - sgn m2   [shift 0]  +  2 sgn ck^2 * (sum_a (adj b)_a u_{a+1}) / det   [Gext shift +1]
                part0 = (u[0] + w2) * _fmpq(sgn * ck ** 2) - self.const(sgn * sp.Rational(m2))
                tdA = TD.const(self.ctx, self.nG, part0)
                if E:
                    lin = self.ctx.from_dict({tuple([0] * self.ctx.nvars()): 0})
                    for a in range(E):
                        ab = self.ctx.from_dict({tuple([0] * self.ctx.nvars()): 0})
                        for b in range(E):
                            ab = ab + adj[a][b] * bvec[b]
                        lin = lin + ab * u[a + 1]
                    lin = lin * _fmpq(2 * sgn * ck ** 2)
                    # divide by det = dq*Gext : multiply by 1/dq and shift Gext +1
                    lin = lin * flint.fmpq(1, 1)
                    inv_dq = flint.fmpq(1, 1) / flint.fmpq(dq.to_dict()[tuple([0] * self.ctx.nvars())])
                    t1 = TD(self.ctx, self.nG, {tuple([0] * self.nG): lin * inv_dq}).shift(gi_ext, +1)
                    tdA = tdA + t1
                A.append(tdA); Bc.append(_fmpq(2 * sgn * ck ** 2)); bvecs.append(bvec); ws.append(w)
            nm = len(js)
            # S_ab = w_a.w_b [shift0] - (b_a^T adj b_b)/det [Gext +1]
            S = {}
            inv_dq = flint.fmpq(1, 1) / flint.fmpq(dq.to_dict()[tuple([0] * self.ctx.nvars())])
            for a in range(nm):
                for b in range(a, nm):
                    s0 = self.to_mpoly(rep.resolve(fam.dot_vec(ws[a], ws[b], later)))
                    tdS = TD.const(self.ctx, self.nG, s0)
                    if E:
                        q = self.ctx.from_dict({tuple([0] * self.ctx.nvars()): 0})
                        for i1 in range(E):
                            for i2 in range(E):
                                q = q + bvecs[a][i1] * adj[i1][i2] * bvecs[b][i2]
                        tdS = tdS - TD(self.ctx, self.nG, {tuple([0] * self.nG): q * inv_dq}).shift(gi_ext, +1)
                    S[(a, b)] = S[(b, a)] = tdS
            # expand prod_j (A_j + B_j X_j)^{n_j}
            ns = [-nu[j - 1] for j in js]
            mmax = sum(ns) // 2
            dperp = self.const(fam.d0 - E) - self.epsg * 2
            # overall multiply by prod_{i<mmax} (dperp + 2i) so that every piece is polynomial in eps
            acc = TD(self.ctx, self.nG, {})
            ranges = [range(n_ + 1) for n_ in ns]
            # precompute powers
            Apow = [[A[a].pow(e) for e in range(ns[a] + 1)] for a in range(nm)]
            for avec in itertools.product(*ranges):
                deg = sum(avec)
                if deg % 2:
                    continue
                m = deg // 2
                coef = TD.const(self.ctx, self.nG, self.one)
                scal = flint.fmpq(1, 1)
                for a in range(nm):
                    coef = coef * Apow[a][ns[a] - avec[a]]
                    scal *= comb(ns[a], avec[a]) * Bc[a] ** avec[a]
                # angular average of prod X_a^{avec_a}: (prod a_j!)/(2m)! * (2m-1)!! * [t^a] S(t)^m * kperp2^m / prod_{i<m}(dperp+2i)
                if m == 0:
                    avg = TD.const(self.ctx, self.nG, self.one)
                else:
                    avg = self._t_coeff_S_pow(S, nm, avec, m)
                    num = 1
                    for a in range(nm):
                        num *= sp.factorial(avec[a])
                    num = sp.Rational(num * _dfact2(m), sp.factorial(deg))
                    avg = avg * _fmpq(num)
                    # kperp^2^m = (Gfull/Gext)^m
                    avg = avg.shift(gi_full, -m)
                    if gi_ext is not None:
                        avg = avg.shift(gi_ext, +m)
                epsfac = self.one
                for i in range(m, mmax):
                    epsfac = epsfac * (dperp + self.const(2 * i))
                acc = acc + (coef * avg) * (epsfac * scal)
            td = td * acc
        return td

    def _t_coeff_S_pow(self, S, nm, avec, m):
        """[t^avec] (sum_ab t_a t_b S_ab)^m with TD coefficients"""
        # polynomial in t represented as dict exponent-tuple -> TD
        base = {}
        for a in range(nm):
            for b in range(nm):
                e = [0] * nm; e[a] += 1; e[b] += 1; e = tuple(e)
                base[e] = base[e] + S[(a, b)] if e in base else S[(a, b)]
        cur = {tuple([0] * nm): TD.const(self.ctx, self.nG, self.one)}
        for _ in range(m):
            nxt = {}
            for e1, c1 in cur.items():
                for e2, c2 in base.items():
                    e = tuple(x + y for x, y in zip(e1, e2))
                    if any(x > y for x, y in zip(e, avec)):
                        continue
                    nxt[e] = nxt[e] + c1 * c2 if e in nxt else c1 * c2
            cur = nxt
        return cur.get(tuple(avec), TD(self.ctx, self.nG, {}))

    # -- the map ------------------------------------------------------------------------------------------------
    def iota(self, nu, eps_value=None):
        """eps_value=None: split the image by powers of eps (list of FormTerms); eps_value=q (rational): return the single
        eps-free form iota(I)|_{eps=q} (convention: integers of an integral = integers of its image at generic eps)."""
        rep = self.rep
        nu = [int(x) for x in nu]
        for i in rep.sector:
            if nu[i - 1] < 1:
                raise ValueError(f'index {i} must be >= 1 (sector {rep.sector})')
        try:
            td = self.numerator_td(nu)
        except (MissingISP, ValueError, RuntimeError):
            td = self.numerator_general(nu)
        # dots
        for i in rep.sector:
            k = nu[i - 1] - 1
            if k == 0:
                continue
            var = f'sig{i}'
            dG = [G.derivative(var) for (_, G, _, _) in self.grams]
            for _ in range(k):
                new = {}
                for m, c in td.d.items():
                    dc = c.derivative(var)
                    if not dc.is_zero():
                        new[m] = new[m] + dc if m in new else dc
                    for a in range(self.nG):
                        if dG[a].is_zero():
                            continue
                        m2 = list(m); m2[a] += 1; m2 = tuple(m2)
                        val = c * (self.eG[a] - self.const(m[a])) * dG[a]
                        new[m2] = new[m2] + val if m2 in new else val
                td = TD(self.ctx, self.nG, new)
            td = TD(self.ctx, self.nG, {m: c.subs({var: 0}) for m, c in td.d.items()})
        # cut everything
        td = TD(self.ctx, self.nG, {m: c.subs(self.cutsub_all) for m, c in td.d.items()})
        if td.is_zero():
            return []
        # exponents of p_i in the denominator per term: ex_i = sum_G M_Gi m_G - shift_i
        nd = len(rep.divisors)
        shifts = [d.shift for d in rep.divisors]
        terms = []
        for m, c in td.d.items():
            ex = [sum(self.gram_cut[a][1][i] * m[a] for a in range(self.nG)) - shifts[i] for i in range(nd)]
            const = flint.fmpq(1, 1)
            for a in range(self.nG):
                if m[a]:
                    const *= self.gram_cut[a][0] ** (-m[a]) if m[a] < 0 else flint.fmpq(1, 1) / (self.gram_cut[a][0] ** m[a])
            terms.append((ex, c * const))
        D = [max(t[0][i] for t in terms) for i in range(nd)]
        N = None
        ppow_cache = {}
        def ppow(i, e):
            key = (i, e)
            if key not in ppow_cache:
                ppow_cache[key] = self.pt[i] ** e if e else self.one
            return ppow_cache[key]
        for ex, c in terms:
            t = c
            for i in range(nd):
                e = D[i] - ex[i]
                if e:
                    t = t * ppow(i, e)
            N = t if N is None else N + t
        Dpos = list(D)
        for i in range(nd):
            if Dpos[i] < 0:
                N = N * ppow(i, -Dpos[i]); Dpos[i] = 0
        if N.is_zero():
            return []
        # split by eps powers (or evaluate at generic eps)
        Nn = self.fam.nprops
        if eps_value is not None:
            N = N.subs({'eps': _fmpq(eps_value)})
            if N.is_zero():
                return []
        pieces = {}
        for monom, cf in zip(N.monoms(), N.coeffs()):
            k = monom[Nn]
            mon2 = tuple(list(monom[:Nn]) + [0])
            pieces.setdefault(k, {})[mon2] = cf
        pieces = {k: self.ctx.from_dict(d) for k, d in pieces.items()}
        groups = []   # [Nref mpoly, coeff sympy in eps]
        for k in sorted(pieces):
            Nk = pieces[k]
            placed = False
            for grp in groups:
                q, r = divmod(Nk, grp[0])
                if r.is_zero() and q.is_constant():
                    qc = q.to_dict().get(tuple([0] * self.ctx.nvars()), flint.fmpq(0))
                    grp[1] = grp[1] + sp.Rational(int(qc.p), int(qc.q)) * eps ** k
                    placed = True
                    break
            if not placed:
                groups.append([Nk, eps ** k])
        out = []
        for Nref, ceps in groups:
            mu = list(Dpos)
            Qp = Nref
            for i in range(nd):
                while mu[i] > 0:
                    q, r = divmod(Qp, self.pt[i])
                    if r.is_zero():
                        Qp = q; mu[i] -= 1
                    else:
                        break
            # primitive: divide by content (rational) — take term_content
            Qz = self.to_sympy_z(Qp)
            if rep.zsyms:
                P = sp.Poly(Qz, *rep.zsyms)
                cont = sp.Rational(1)
                coeffs = P.coeffs()
                if coeffs:
                    from functools import reduce
                    from math import gcd
                    nums = [sp.Rational(c_) for c_ in coeffs]
                    g = reduce(gcd, [int(abs(x.p)) for x in nums]); l = reduce(lambda a, b: a * b // gcd(a, b), [int(x.q) for x in nums])
                    cont = sp.Rational(g, l) * (1 if nums[0] > 0 else -1)
                Qz = sp.expand(Qz / cont)
                ceps = ceps * cont
                degQ = sp.Poly(Qz, *rep.zsyms).total_degree()
            else:
                degQ = 0
            sum_mu_d = sum(mu[i] * rep.divisors[i].deg for i in range(nd))
            tval = degQ - sum_mu_d + rep.dU + rep.n + 1
            if tval >= 0:
                mu0 = int(tval); extra = 0
            else:
                mu0 = 0; extra = int(-tval)
            z0 = rep.z0
            Qh = sp.expand(z0 ** (degQ + extra) * Qz.subs({z: z / z0 for z in rep.zsyms}, simultaneous=True)) if rep.zsyms else z0 ** extra * Qz
            Qh = sp.expand(sp.cancel(Qh))
            out.append(FormTerm(sp.factor(ceps), tuple([mu0] + mu), Qz, Qh, degQ + extra))
        return out


def _dfact2(m):
    r = 1
    for i in range(1, 2 * m, 2):
        r *= i
    return r


def _numerator_general(self, nu):
    """General missing-ISP treatment (nested transverse averaging, loop by loop).  State = dict shift-tuple -> sympy
    polynomial numerator (in sigma, eps, not-yet-resolved DOT symbols); denominators are tracked ONLY as powers of the
    Gram polynomials (det Gram(Q_k) = const * Gext_k), so no rational-function algebra is ever needed."""
    rep, fam = self.rep, self.fam
    sgn = fam.sigma_sign
    expr = sp.Integer(1)
    for j in range(1, fam.nprops + 1):
        if j in rep.sector or nu[j - 1] == 0:
            continue
        n_ = -nu[j - 1]
        if j in rep.isps:
            expr *= fam.sig[j - 1] ** n_
            continue
        mom, m2 = fam.props[j - 1]
        k = next((l for l in rep.loop_order if mom.get(l, 0) != 0), None)
        if k is None:
            raise MissingISP(f'sigma{j} has no loop momentum')
        ld = rep.loopdata[k]
        before = ld['integrated_before']; v0 = ld['v0']; later = ld['later']
        if any(mom.get(l, 0) != 0 for l in before):
            raise MissingISP(f'sigma{j} involves an already-integrated loop')
        v = fam.vec(mom); ck = fam.loop_part(v, k)
        other = tuple(c / ck for c in fam.strip(v, before + [k]))
        w = tuple(sp.nsimplify(a_ - b_) for a_, b_ in zip(other, v0))
        kk2 = fam.dotsym(k, fam.unit(k))
        kw = fam.dotsym(k, w) if not fam.is_zero(w) else 0
        ww = fam.dot_vec(w, w, later) if not fam.is_zero(w) else 0
        expr *= (sgn * ck ** 2 * (kk2 + 2 * kw + ww) - sgn * m2) ** n_
    state = {tuple([0] * self.nG): sp.expand(expr)}
    for k in rep.loop_order:
        ld = rep.loopdata[k]
        Q, E, later = ld['Q'], ld['E'], ld['later']
        allsyms = [(ints, sym) for (kk, ints), (sym, _) in list(fam._dotsyms.items()) if kk == k]
        present = [(ints, sym) for ints, sym in allsyms if any(numr.has(sym) for numr in state.values())]
        if not present:
            continue
        gi_ext = self.gidx.get(f'Gext_{k}')
        Gq = sp.Matrix(E, E, lambda a_, b_: fam.dot_vec(Q[a_], Q[b_], later)) if E else None
        det = rep.resolve(sp.expand(Gq.det())) if E else sp.Integer(1)       # = cdet * Gext_k(sigma)
        if E:
            ge = next(g.poly_sigma for g in rep.grams if g.name == f'Gext_{k}')
            cdet = sp.cancel(det / ge)
            if cdet.free_symbols:
                raise RuntimeError('det Gram(Q) not proportional to Gext')
        else:
            cdet = sp.Integer(1)
        adj = Gq.adjugate() if E else None
        uvec = sp.Matrix([sp.expand(ld['u'][a_ + 1].subs(ld['sol'])) for a_ in range(E)]) if E else None
        uk = tuple(sp.Integer(c) for c in fam.unit(k))
        inspan = {}; trans = []   # trans: (sym, Lnum (so that sym = Lnum/det + X), X, wv, bvec)
        for ints, sym in present:
            wv = tuple(sp.Integer(c) for c in ints)
            if wv == uk:
                inspan[sym] = sp.expand(ld['u'][0].subs(ld['sol'])); continue
            try:
                inspan[sym] = rep.kdot(k, wv); continue
            except ValueError:
                pass
            bvec = sp.Matrix([fam.dot_vec(Q[a_], wv, later) for a_ in range(E)]) if E else None
            Lnum = sp.expand((bvec.T * adj * uvec)[0, 0]) if E else sp.Integer(0)    # k'.w_par = Lnum/det
            X = sp.Symbol(f'Xperp_{k}_{len(trans)}')
            trans.append((sym, Lnum, X, wv, bvec))
        # substitute in-span symbols (polynomial)
        state = {m_: sp.expand(numr.xreplace(inspan)) for m_, numr in state.items()}
        if not trans:
            continue
        tsyms = [t[0] for t in trans]
        Xs = [t[2] for t in trans]
        # expand each numerator as polynomial in the transverse DOT symbols; sym^e -> (Lnum + X det)^e / det^e
        new = {}
        for m_, numr in state.items():
            Pn = sp.Poly(numr, *tsyms)
            for monom, coeff in zip(Pn.monoms(), Pn.coeffs()):
                etot = sum(monom)
                term = coeff
                for (sym, Lnum, X, wv, bvec), e in zip(trans, monom):
                    if e:
                        term = term * (Lnum + X * det) ** e
                m2_ = list(m_)
                if etot:
                    m2_[gi_ext] += etot
                    term = term / cdet ** etot
                m2_ = tuple(m2_)
                new[m2_] = sp.expand(new.get(m2_, 0) + sp.expand(term))
        state = new
        # transverse averaging over X's
        nm = len(Xs)
        Snum = sp.zeros(nm, nm)   # S = Snum/det
        for a_ in range(nm):
            for b_ in range(a_, nm):
                val = fam.dot_vec(trans[a_][3], trans[b_][3], later) * det
                if E:
                    val = val - (trans[a_][4].T * adj * trans[b_][4])[0, 0]
                Snum[a_, b_] = Snum[b_, a_] = sp.expand(val)
        tt = sp.symbols(f'tq0:{nm}')
        Sform = sp.expand(sum(tt[a_] * tt[b_] * Snum[a_, b_] for a_ in range(nm) for b_ in range(nm)))
        gfull = next(g.poly_sigma for g in rep.grams if g.name == f'Gfull_{k}')
        dperp = fam.d0 - E - 2 * eps
        mmax = 0
        for numr in state.values():
            Pn = sp.Poly(numr, *Xs)
            mmax = max(mmax, max(sum(mon) for mon in Pn.monoms()) // 2)
        Spow = {0: sp.Integer(1)}
        new = {}
        for m_, numr in state.items():
            Pn = sp.Poly(numr, *Xs)
            for monom, coeff in zip(Pn.monoms(), Pn.coeffs()):
                deg = sum(monom)
                if deg % 2:
                    continue
                mm = deg // 2
                m2_ = list(m_)
                if mm == 0:
                    term = coeff
                else:
                    if mm not in Spow:
                        Spow[mm] = sp.expand(Sform ** mm)
                    tc = sp.Poly(Spow[mm], *tt).coeff_monomial(sp.Mul(*[tt[a_] ** monom[a_] for a_ in range(nm)]))
                    fac = sp.Integer(1)
                    for a_ in range(nm):
                        fac *= sp.factorial(monom[a_])
                    term = coeff * tc * fac / sp.factorial(deg) * sp.Integer(_dfact2(mm)) * gfull ** mm / cdet ** mm
                    # denominators: det^mm (from S) * Gext^mm (from kperp2) -> Gext shift += 2 mm ; Gfull in numerator
                    m2_[gi_ext] += 2 * mm
                for i in range(mm, mmax):
                    term = term * (dperp + 2 * i)
                m2_ = tuple(m2_)
                new[m2_] = sp.expand(new.get(m2_, 0) + sp.expand(term))
        state = {m_: v for m_, v in new.items() if v != 0}
    # any remaining DOT/U symbols must resolve polynomially
    out = {}
    for m_, numr in state.items():
        if any(str(x).startswith(('DOT__', 'U__')) for x in numr.free_symbols):
            numr = rep.resolve(numr)
        bad = [x for x in numr.free_symbols if x not in set(fam.sig) and x != eps]
        if bad:
            raise RuntimeError(f'general numerator: unresolved symbols {bad}')
        out[m_] = self.to_mpoly(numr)
    return TD(self.ctx, self.nG, out)


FastRep.numerator_general = _numerator_general
