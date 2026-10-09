# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of BootLoops 1.0; see LICENSE and NOTICE beside this file.
"""The map iota-hat : Feynman integral I_nu (one sector) -> element of the form space A_omega (arXiv:2511.15381, TeX label def_input_data of the arXiv v2 TeX source).

Given the loop-by-loop representation `rep` (geomorder.baikov.BaikovRep) of the sector, the maximal-cut integrand of
I_nu = int prod sigma_i^{-nu_i} is obtained by taking, for every cut line i, the residue of order nu_i at sigma_i=0:
    Res_{sigma_i=0} f/sigma_i^{nu}  =  (1/(nu-1)!) d^{nu-1} f / d sigma_i^{nu-1} |_{sigma_i=0},
applied to f = prod_G G(sigma)^{e_G} * prod_{ISP j} sigma_j^{-nu_j}.  Derivatives of G^{e} are tracked symbolically as
shift vectors.  The result is written relative to the MINIMAL twist U_min as a rational function
    Phi(z;eps) = N(z,eps) / prod_i p_i(z)^{M_i}
and split by powers of eps into eps-free forms omega_mu[Q] (distribution identity):  iota(I_nu) = sum_j c_j(eps) omega_{mu^(j)}[Q^(j)].
Each term is reduced to lowest terms (trivial cancellation identities) and homogenized (mu_0 from the degree constraint
d_Q = sum_i mu_i d_i - d_U - n - 1).  Overall kinematic constants are kept in c_j but are irrelevant for the ordering.

Convention: when the image is a genuine multi-term combination (eps-dependent
numerator that does not factorize), the ordering integers of I_nu are the lexicographic MAX of (w,o,|mu|) over the terms.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import factorial
import sympy as sp
from .baikov import BaikovRep, eps


@dataclass
class FormTerm:
    coeff: sp.Expr          # c_j(eps, kin)
    mu: tuple               # (mu_0, mu_1, ..., mu_ND) w.r.t. [P0] + rep.divisors order
    Q_affine: sp.Expr       # numerator in affine chart z0=1
    Q_hom: sp.Expr          # homogeneous numerator
    degQ: int

    def key(self):
        return (self.mu, sp.srepr(sp.expand(self.Q_hom)))

    def __str__(self):
        return f'({sp.factor(self.coeff)}) * omega_{list(self.mu)}[{sp.factor(self.Q_hom)}]'


class MissingISP(Exception):
    pass


def iota(rep: BaikovRep, nu, keep_constants=False, strict_missing=False):
    fam = rep.family
    nu = [int(x) for x in nu]
    assert len(nu) == fam.nprops
    for i in rep.sector:
        if nu[i - 1] < 1:
            raise ValueError(f'index {i} must be >=1 in sector {rep.sector}')
    for j in range(1, fam.nprops + 1):
        if j not in rep.sector and nu[j - 1] > 0:
            raise ValueError(f'index {j} is not a cut line of sector {rep.sector} but nu={nu[j-1]}>0 (super-sector integral)')
    grams = rep.grams
    nG = len(grams)
    eG = [sp.Rational(g.A, 2) + sp.Rational(g.B, 2) * eps for g in grams]
    sig = fam.sig
    # numerator from ISPs (present ones: monomial; missing ones: loop-by-loop transverse averaging)
    num = numerator_expr(rep, nu, strict=strict_missing)
    terms = {tuple([0] * nG): num}
    # derivatives for dots
    overall = sp.Integer(1)
    for i in rep.sector:
        k = nu[i - 1] - 1
        if k == 0:
            continue
        overall /= factorial(k)
        s = sig[i - 1]
        dG = [sp.diff(g.poly_sigma, s) for g in grams]
        for _ in range(k):
            new = {}
            for m, c in terms.items():
                dc = sp.diff(c, s)
                if dc != 0:
                    new[m] = sp.expand(new.get(m, 0) + dc)
                for a in range(nG):
                    if dG[a] == 0:
                        continue
                    m2 = list(m); m2[a] += 1; m2 = tuple(m2)
                    new[m2] = sp.expand(new.get(m2, 0) + c * (eG[a] - m[a]) * dG[a])
            terms = {m: c for m, c in new.items() if c != 0}
        # we may set sigma_i = 0 now (derivatives w.r.t. other sigmas commute with this substitution)
        terms = {m: sp.expand(c.subs(s, 0)) for m, c in terms.items()}
        terms = {m: c for m, c in terms.items() if c != 0}
        # dG is recomputed per cut line from the full polynomials; substituting s = 0 in the Gram polynomials is not allowed
    # set all cut sigmas to zero, ISPs to z
    cutsub = {sig[i - 1]: 0 for i in rep.sector}
    zsub = {sig[rep.isps[j] - 1]: rep.zsyms[j] * fam.scale for j in range(rep.n)}
    divs = rep.divisors
    psyms = {d.name: sp.Symbol('__' + d.name) for d in divs}
    # G|cut = const_G * prod p_i^{M_Gi}
    total = sp.Integer(0)
    for m, c in terms.items():
        cc = sp.expand(c.subs(cutsub).subs(zsub))
        if cc == 0:
            continue
        t = cc
        for a, g in enumerate(grams):
            if m[a] == 0:
                continue
            const, fac = rep.gram_cut_factorization[g.name]
            t = t * const ** (-m[a])
            for pn, mult in fac.items():
                t = t * psyms[pn] ** (-mult * m[a])
        total += t
    # multiply by prod p_i^{shift_i} (raw twist -> minimal twist)
    for d in divs:
        total = total * psyms[d.name] ** d.shift
    total = sp.together(sp.expand(total) * overall)
    N, D = sp.fraction(total)
    paff = {psyms[d.name]: d.p_affine for d in divs}
    N = sp.expand(N.subs(paff))
    D = sp.expand(sp.sympify(D).subs(paff))
    # identify D = const * prod p_i^{M_i} by factorization over Q(kin)
    kin_ = sorted(fam.kin.values(), key=str) + [eps]
    dom_ = sp.QQ.frac_field(*kin_)
    M = {d.name: 0 for d in divs}
    if rep.zsyms and (set(D.free_symbols) & set(rep.zsyms)):
        DP = sp.Poly(D, *rep.zsyms, domain=dom_)
        Dconst, flist = DP.factor_list()
        Dconst = Dconst.as_expr() if hasattr(Dconst, 'as_expr') else sp.sympify(Dconst)
        bykey = {sp.expand(d.p_affine): d.name for d in divs}
        for f, mult in flist:
            fm = f.monic(); lc = f.LC(); lc = lc.as_expr() if hasattr(lc, 'as_expr') else sp.sympify(lc)
            Dconst = Dconst * lc ** mult
            key = sp.expand(fm.as_expr())
            if key not in bykey:
                raise RuntimeError(f'denominator factor {key} is not a divisor of the representation')
            M[bykey[key]] += mult
    else:
        Dconst = D
    # split numerator by powers of eps
    NP = sp.Poly(N, eps)
    pieces = []   # (eps power, Nj(z))
    for (k,), co in NP.terms():
        pieces.append((k, sp.expand(co)))
    # merge proportional pieces
    kin = sorted(fam.kin.values(), key=str)
    dom = sp.QQ.frac_field(*kin) if kin else sp.QQ
    zs = rep.zsyms
    groups = []   # list of [Nref(z), coeff(eps)]
    for k, Nj in pieces:
        placed = False
        for grp in groups:
            ratio = sp.cancel(Nj / grp[0])
            if not (set(sp.sympify(ratio).free_symbols) & set(zs)):
                grp[1] = grp[1] + ratio * eps ** k
                placed = True
                break
        if not placed:
            groups.append([Nj, eps ** k])
    out = []
    for Nref, ceps in groups:
        # reduce Nref / prod p^M to lowest terms
        mu = dict(M)
        Q = Nref
        const = sp.Integer(1) / Dconst
        if zs:
            QP = sp.Poly(Q, *zs, domain=dom)
            for d in divs:
                while mu[d.name] > 0:
                    PP = sp.Poly(d.p_affine, *zs, domain=dom)
                    q, r = QP.div(PP)
                    if r.is_zero:
                        QP = q; mu[d.name] -= 1
                    else:
                        break
            # pull out constant content
            Q = QP.as_expr()
        cQ, fl = sp.factor_list(sp.Poly(Q, *zs, domain=dom)) if zs else (Q, [])
        # keep Q primitive-ish: move overall kinematic factor into coeff
        if zs:
            cQe = cQ.as_expr() if hasattr(cQ, 'as_expr') else sp.sympify(cQ)
            Qprim = sp.expand(sp.cancel(Q / cQe)) if cQe != 0 else Q
            const = const * cQe
            Q = Qprim
        degQ = sp.Poly(Q, *zs).total_degree() if zs else 0
        # homogenize: mu_0
        sum_mu_d = sum(mu[d.name] * d.deg for d in divs)
        tval = degQ - sum_mu_d + rep.dU + rep.n + 1
        # d_Q = sum_{i>=0} mu_i d_i - dU - n - 1  with d_0 = 1
        if tval >= 0:
            mu0 = int(tval); extra = 0
        else:
            mu0 = 0; extra = int(-tval)
        z0 = rep.z0
        Qh = sp.expand(z0 ** (degQ + extra) * Q.subs({z: z / z0 for z in zs}, simultaneous=True)) if zs else z0 ** extra * Q
        Qh = sp.expand(sp.cancel(Qh))
        mu_t = tuple([mu0] + [mu[d.name] for d in divs])
        out.append(FormTerm(sp.factor(ceps * const) if keep_constants else sp.factor(ceps * const), mu_t, Q, Qh, degQ + extra))
    return out


def corner(rep: BaikovRep):
    nu = [1 if i in rep.sector else 0 for i in range(1, rep.family.nprops + 1)]
    return iota(rep, nu)


def make_term(rep: BaikovRep, mu, Q_hom, coeff=1):
    """build a FormTerm by hand (for fixtures/tests): mu tuple (mu_0..mu_ND), Q_hom sympy expr in z0..zn"""
    Q_hom = sp.sympify(Q_hom)
    zs = rep.zsyms
    Qa = sp.expand(Q_hom.subs(rep.z0, 1))
    deg = sp.Poly(Q_hom, rep.z0, *zs).total_degree() if (zs or Q_hom.has(rep.z0)) else 0
    return FormTerm(sp.sympify(coeff), tuple(int(m) for m in mu), Qa, sp.expand(Q_hom), deg)


def _dfact2(m):
    r = 1
    for i in range(1, 2 * m, 2):
        r *= i
    return r


def numerator_expr(rep: BaikovRep, nu, strict=False):
    """Numerator of the integrand as a rational function of the sigma variables: product over ISPs of sigma_j^{-nu_j};
    ISPs that are not Baikov variables of the representation ("missing ISPs", sect. 2.5.3) are rewritten loop by loop:
    sigma_j = A_j + B_j (k'_perp . w_j,perp) and the transverse monomials are replaced by their angular averages
    <(k_perp.w_perp)^{2m}> = (2m-1)!!/prod_{i<m}(d_perp+2i) (k_perp^2)^m (w_perp^2)^m, k_perp^2 = G(k,Q)/G(Q),
    d_perp = D - E_k (standard loop-by-loop tensor reduction).  strict=True raises MissingISP instead."""
    fam = rep.family
    sig = fam.sig
    num = sp.Integer(1)
    missing = {}
    for j in range(1, fam.nprops + 1):
        if j in rep.sector or nu[j - 1] == 0:
            continue
        if j in rep.isps:
            num *= sig[j - 1] ** (-nu[j - 1])
        else:
            if strict:
                raise MissingISP(f'ISP sigma{j} (nu={nu[j-1]}) is not a Baikov variable of this representation (z = sigma{rep.isps})')
            mom = fam.props[j - 1][0]
            k = next((l for l in rep.loop_order if mom.get(l, 0) != 0), None)
            if k is None:
                raise MissingISP(f'ISP sigma{j} contains no loop momentum')
            missing.setdefault(k, []).append(j)
    for k, js in missing.items():
        ld = rep.loopdata[k]
        Q, E, later, v0 = ld['Q'], ld['E'], ld['later'], ld['v0']
        before = ld['integrated_before']
        # Gram of Q and its inverse (resolved to sigma variables)
        Gq = sp.zeros(E, E)
        for a_ in range(E):
            for b_ in range(a_, E):
                Gq[a_, b_] = Gq[b_, a_] = rep.resolve(fam.dot_vec(Q[a_], Q[b_], later))
        Ginv = Gq.inv() if E else None
        u0 = rep.resolve(ld['u'][0])
        ws, A, B = [], [], []
        for j in js:
            mom, m2 = fam.props[j - 1]
            v = fam.vec(mom)
            ck = fam.loop_part(v, k)
            if any(mom.get(l, 0) != 0 for l in before):
                raise MissingISP(f'ISP sigma{j} involves an already-integrated loop; cannot assign to loop {k}')
            other = tuple(c / ck for c in fam.strip(v, before + [k]))
            w = tuple(sp.nsimplify(a - b) for a, b in zip(other, v0))
            bvec = sp.Matrix([rep.resolve(fam.dot_vec(Q[a_], w, later)) for a_ in range(E)]) if E else None
            w2 = rep.resolve(fam.dot_vec(w, w, later))
            if E:
                cvec = Ginv * bvec
                L = sp.expand(sum(cvec[a_] * rep.resolve(ld['u'][a_ + 1]) for a_ in range(E)))
            else:
                L = sp.Integer(0)
            ws.append((w, bvec))
            A.append(sp.together(fam.sigma_sign * (ck ** 2 * (u0 + 2 * L + w2) - m2)))
            B.append(2 * fam.sigma_sign * ck ** 2)
        # transverse inner products
        nm = len(js)
        S = sp.zeros(nm, nm)
        for a_ in range(nm):
            for b_ in range(a_, nm):
                wa, ba = ws[a_]; wb, bb = ws[b_]
                val = rep.resolve(fam.dot_vec(wa, wb, later))
                if E:
                    val = val - (ba.T * Ginv * bb)[0, 0]
                S[a_, b_] = S[b_, a_] = sp.together(val)
        # k_perp^2 = Gfull_k / Gext_k (polynomials in sigma)
        gf = next(g.poly_sigma for g in rep.grams if g.name == f'Gfull_{k}')
        ge = next((g.poly_sigma for g in rep.grams if g.name == f'Gext_{k}'), sp.Integer(1))
        kperp2 = gf / ge
        dperp = fam.d0 - E - 2 * eps
        X = sp.symbols(f'X0:{nm}')
        tt = sp.symbols(f'tq0:{nm}')
        poly = sp.Integer(1)
        for a_, j in enumerate(js):
            poly *= (A[a_] + B[a_] * X[a_]) ** (-nu[j - 1])
        poly = sp.Poly(sp.expand(poly), *X)
        Sform = sp.expand(sum(tt[a_] * tt[b_] * S[a_, b_] for a_ in range(nm) for b_ in range(nm)))
        acc = sp.Integer(0)
        Spow = {0: sp.Integer(1)}
        for monom, coeff in zip(poly.monoms(), poly.coeffs()):
            deg = sum(monom)
            if deg == 0:
                acc += coeff; continue
            if deg % 2:
                continue
            m = deg // 2
            if m not in Spow:
                Spow[m] = sp.expand(Sform ** m)
            tc = sp.Poly(Spow[m], *tt).coeff_monomial(sp.Mul(*[tt[a_] ** monom[a_] for a_ in range(nm)]))
            fac = sp.Integer(1)
            for a_ in range(nm):
                fac *= sp.factorial(monom[a_])
            cm = sp.Integer(_dfact2(m))
            den = sp.Integer(1)
            for i in range(m):
                den *= (dperp + 2 * i)
            acc += coeff * tc * fac / sp.factorial(deg) * cm / den * kperp2 ** m
        num *= acc
    return sp.together(num)
