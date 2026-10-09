# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of BootLoops 1.0; see LICENSE and NOTICE beside this file.
"""Maximal-cut differential equation of a master basis and the paper's refined statement: is dJ = A J with A a LAURENT POLYNOMIAL
in eps, with powers inside the F-filtration window  -(|mu|_i - |mu|_j) <= k <= 1  (arXiv:2511.15381, TeX label refined_statement,
lines 3078-3091 of the arXiv v2 TeX source)?

Method: on the cut, in the omega-normalization of the paper (C_rel, C_clutch built into the IBP/cancellation identities),
    d/dx omega_mu[Q] = sum_i eps * omega_{mu+e_i}[Q dP_i/dx] + omega_mu[dQ/dx]   (+ dlog(C_B C_abs) * omega_mu[Q], a
    scalar diagonal shift linear in eps, dropped here and reported separately),
derived from U = prod P_i^{alpha_i} and C_rel(mu+e_i) = C_rel(mu) (alpha_i - mu_i), C_clutch = eps^{-|mu|}.
The derivative rows are reduced modulo the cut system (cohomology.CutSystem) onto the chosen basis at K numeric eps values
(mod p, numeric kinematics); each matrix entry is reconstructed as a rational function of eps and its denominator tested
for being a pure power of eps.  Bases: 'geometric' = Algorithm-1 H-basis (forms, with a), 'kira' = images of a given
list of Feynman integrals (single-form images are used in omega-normalization; multi-term images are flagged).
"""
from __future__ import annotations
import random, time
import sympy as sp
import flint
from .baikov import Family, numeric_spec, representation, loop_by_loop, democratic, eps
from .fastiota import FastRep
from .cohomology import CutSystem, fmt
from .localize import algorithm1_n1, good_prime_n1
from .integers import PRIMES
from .sweep import enumerate_integrals
from .masters import kira_tiebreak


def _symbolic_dP(spec, sector, nrep, nvals):
    """d(P_i)/dx_s evaluated at the numeric kinematic point, for the monic symbolic divisors matched to the numeric rep's
    divisors.  Returns ({symbol: [poly for i=0..ND]}, symbolic rep); index 0 is P_0 = z0 (derivative 0)."""
    F = Family(spec)
    srep = loop_by_loop(F, sector, loop_order=nrep.loop_order) if nrep.kind == 'loop_by_loop' else democratic(F, sector)
    kin = {sp.Symbol(k): v for k, v in nvals.items()}
    out = {str(s): [sp.Integer(0)] * (len(nrep.divisors) + 1) for s in kin}
    used = set()
    mapping = {}
    for idx, d in enumerate(nrep.divisors, start=1):
        hit = None
        for ds in srep.divisors:
            if ds.name in used:
                continue
            val = sp.expand(sp.cancel(sp.together(ds.P_hom.subs(kin))))
            q = sp.cancel(val / d.P_hom)
            if not q.free_symbols and q != 0:
                hit = (ds, q); break
        if hit is None:
            raise RuntimeError(f'cannot match numeric divisor {d.name}={d.P_hom} to a symbolic one')
        ds, q = hit
        used.add(ds.name)
        mapping[ds.name] = (idx, q)
        if (ds.a_raw, ds.b) != (d.a_raw, d.b):
            raise RuntimeError('exponent mismatch between symbolic and numeric divisor')
        for s in kin:
            der = sp.expand(sp.cancel(sp.together(sp.diff(ds.P_hom, s).subs(kin))) / q)
            out[str(s)][idx] = der
    return out, srep, mapping


class CutDE:
    def __init__(self, spec, sector, seed=5, M=5, K=12, cand_dots=2, cand_sps=2, verbose=False, values=None, prime=None):
        self.spec, self.sector, self.seed, self.M, self.K = spec, sorted(sector), seed, M, K
        self.nspec = numeric_spec(spec, seed=seed, values=values)
        self.nvals = self.nspec['_numeric_values']
        self.NF = Family(self.nspec)
        self.rep, self.note = representation(self.NF, self.sector)
        if self.rep.n not in (1, 2):
            raise NotImplementedError(f'cut DE implemented for n<=2 sectors (sector has n={self.rep.n}, {self.note})')
        self.fr = FastRep(self.rep)
        self.dP, self.srep, self.divmap = _symbolic_dP(spec, self.sector, self.rep, self.nvals)
        self.symimg = {}
        self.keycache = {}
        self.cand = enumerate_integrals(self.NF.nprops, self.sector, len(self.sector) + cand_dots, cand_sps)
        self.verbose = verbose
        self.rng = random.Random(seed)
        self.p = prime if prime else (good_prime_n1(self.rep, self.rng) if self.rep.n == 1 else PRIMES[0])

    def symbolic_image(self, nu):
        """iota-hat of a Feynman integral with SYMBOLIC kinematics (slow path), re-indexed to the numeric rep's divisor
        order: list of (coeff(eps, x) sympy, mu tuple (numeric order), Q_hom(x) sympy)."""
        nu = tuple(nu)
        if nu in self.symimg:
            return self.symimg[nu]
        from .iota import iota as slow_iota
        terms = slow_iota(self.srep, list(nu))
        out = []
        nd = len(self.rep.divisors)
        for t in terms:
            mu = [0] * (nd + 1); mu[0] = t.mu[0]
            qfix = sp.Integer(1)
            for j, ds in enumerate(self.srep.divisors, start=1):
                idx, q = self.divmap[ds.name]
                mu[idx] = t.mu[j]
                # symbolic P = q * numeric P  ->  1/P_sym^m = q^-m / P_num^m
                qfix = qfix * sp.Integer(1) / sp.sympify(q) ** t.mu[j]
            out.append((sp.sympify(t.coeff) * qfix, tuple(mu), sp.sympify(t.Q_hom)))
        self.symimg[nu] = out
        return out

    def build(self, e0r, extra_integrals=()):
        p = self.p
        e0 = (int(sp.Rational(e0r).p) * pow(int(sp.Rational(e0r).q), -1, p)) % p
        cs = CutSystem(self.rep, p, e0); cs._M = self.M; cs.e0_rational = sp.Rational(e0r)
        cs.keycache = self.keycache   # (w,o,|mu|) keys are eps-independent: share across samples
        cs.generate(self.M)
        images = {}
        for nu in list(self.cand) + [tuple(x) for x in extra_integrals if tuple(x) not in self.cand]:
            try:
                terms = self.fr.iota(nu, eps_value=sp.Rational(e0r))
            except Exception:
                continue
            if not terms or max(sum(t.mu) for t in terms) > self.M:
                continue
            cs.add_integral_column(nu, terms, kira_tiebreak(nu, self.sector))
            images[tuple(nu)] = terms
        return cs, images

    def deriv_row(self, cs, col, images, xname):
        """row dict for d/dx of a basis column (form or integral column), as {column index: value mod p} with the new
        unknown NOT included (caller adds it)."""
        row = {}
        dPs = self.dP[xname]
        def add_form(mu, Qpoly, factor):
            # d/dx omega_mu[Q] = sum_i eps*omega_{mu+e_i}[Q dP_i] + omega_mu[dQ/dx]; Q numeric poly (no x dep) -> second term 0
            for i, dPi in enumerate(dPs):
                if dPi == 0:
                    continue
                mu2 = list(mu); mu2[i] += 1
                polyd = cs._pdict(sp.expand(Qpoly * dPi))
                cs.add_poly(row, tuple(mu2), polyd, (factor * cs.e0) % cs.p)
        if col[0] == 'I':
            kin = {sp.Symbol(k): v for k, v in self.nvals.items()}
            xs = sp.Symbol(xname)
            for c, mu, Qx in self.symbolic_image(col[1]):
                c_e = c.subs(eps, cs.e0_rational)
                c_num = sp.Rational(sp.together(c_e.subs(kin)))
                dc_num = sp.Rational(sp.together(sp.diff(c_e, xs).subs(kin)))
                Q_num = sp.expand(Qx.subs(kin)); dQ_num = sp.expand(sp.diff(Qx, xs).subs(kin))
                # d/dx [c omega_mu[Q]] = dc*omega_mu[Q] + c*omega_mu[dQ] + c*sum_i eps omega_{mu+e_i}[Q dP_i]
                if dc_num != 0:
                    cs.add_poly(row, tuple(mu), cs._pdict(Q_num), cs.modp(dc_num))
                if dQ_num != 0:
                    cs.add_poly(row, tuple(mu), cs._pdict(dQ_num), cs.modp(c_num))
                add_form(mu, Q_num, cs.modp(c_num))
        else:
            mu, mono = col
            Q = sp.Integer(1)
            for z, e in zip(cs.Z, mono): Q *= z ** e
            add_form(mu, Q, 1)
        return row

    def matrix_at(self, e0r, basis_spec, xname):
        """A(eps=e0r) mod p for basis given as list of columns (('I',nu) or (mu,mono)). Returns (A rows list, basis, info)."""
        cs, images = self.build(e0r, extra_integrals=[c[1] for c in basis_spec if c[0] == 'I'])
        bset = set(basis_spec)
        missing = [c for c in basis_spec if c not in cs.cols]
        if missing:
            raise RuntimeError(f'basis columns not in system: {missing}')
        # derivative unknowns
        ycols = []
        for b in basis_spec:
            yk = ('Y', (xname,) + (tuple(b[1]) if b[0] == 'I' else (tuple(b[0]), tuple(b[1]))))
            cs.cols[yk] = len(cs.collist); cs.collist.append(yk)
            row = self.deriv_row(cs, b, images, xname)
            row[cs.cols[yk]] = (cs.p - 1)     # -Y + d/dx(b) = 0
            cs.rows.append({k: v for k, v in row.items() if v})
            ycols.append(yk)
        # order: Y most complex, basis least complex
        def keyf(c):
            if c[0] == 'Y':
                return (10 ** 6,)
            if c in bset:
                return (-10 ** 6, basis_spec.index(c))
            return None
        base_keys = {}
        for c in cs.collist:
            kf = keyf(c)
            base_keys[c] = kf
        # use cs.form_key for the rest
        def a_of(c):
            return 0
        keys = {}
        for c in cs.collist:
            if base_keys[c] is not None:
                keys[c] = base_keys[c]
            else:
                k = cs.form_key(c[0], c[1], a_of, prefer_integrals=False)
                keys[c] = (0,) + tuple(k)
        ncol = len(cs.collist)
        order = sorted(range(ncol), key=lambda i: keys[cs.collist[i]], reverse=True)
        pos = {ci: k for k, ci in enumerate(order)}
        Mx = flint.nmod_mat(len(cs.rows), ncol, cs.p)
        for r, row in enumerate(cs.rows):
            for c, v in row.items():
                Mx[r, pos[c]] = v
        R, rank = Mx.rref()
        # locate pivot rows for Y columns
        A = {}
        leaks = {}
        colat = {k: cs.collist[order[k]] for k in range(ncol)}
        pivrow = {}
        for r in range(rank):
            for c in range(ncol):
                if int(R[r, c]) != 0:
                    pivrow[c] = r; break
        for yk in ycols:
            pc = pos[cs.cols[yk]]
            if pc not in pivrow:
                raise RuntimeError('derivative unknown not pivoted')
            r = pivrow[pc]
            inv = pow(int(R[r, pc]), -1, cs.p)
            rowvals = {}
            leak = []
            for c in range(pc + 1, ncol):
                v = int(R[r, c])
                if v:
                    col = colat[c]
                    val = (-v * inv) % cs.p
                    if col in bset:
                        rowvals[basis_spec.index(col)] = val
                    else:
                        leak.append((fmt(self.rep, col) if col[0] != 'Y' else str(col), val))
            A[ycols.index(yk)] = rowvals
            if leak:
                leaks[ycols.index(yk)] = leak
        return A, leaks, cs.p

    def reconstruct(self, basis_spec, xname, eps_values=None):
        """A(eps) entries as rational functions mod p via multi-point interpolation + rational reconstruction."""
        K = self.K
        eps_values = eps_values or [sp.Rational(97 + 6 * i, 389 + 11 * i) for i in range(K)]
        samples = []
        leaks_all = {}
        for ev in eps_values:
            A, leaks, p = self.matrix_at(ev, basis_spec, xname)
            samples.append((ev, A))
            for k, v in leaks.items():
                leaks_all.setdefault(k, []).append((str(ev), v[:3]))
        nb = len(basis_spec)
        p = self.p
        xs = [(int(ev.p) * pow(int(ev.q), -1, p)) % p for ev, _ in samples]
        entries = {}
        for i in range(nb):
            for j in range(nb):
                ys = [samples[k][1].get(i, {}).get(j, 0) for k in range(len(samples))]
                entries[(i, j)] = _ratrec_univariate(xs, ys, p)
        return entries, leaks_all

    def report(self, basis_spec, labels, absmu, xnames):
        out = {'sector': self.sector, 'rep': self.note, 'p': self.p, 'seed': self.seed, 'nvals': self.nvals, 'M': self.M, 'K': self.K,
               'basis': labels, 'absmu': absmu, 'x': {}}
        for xname in xnames:
            entries, leaks = self.reconstruct(basis_spec, xname)
            table = {}
            laurent = True; window = True
            for (i, j), rr in entries.items():
                if rr is None:
                    table[f'{i},{j}'] = {'status': 'reconstruction FAILED (degree > K/2-1?)'}; laurent = False; continue
                num, den = rr
                if all(c == 0 for c in num):
                    table[f'{i},{j}'] = {'zero': True}; continue
                # den monomial? den = list of coeffs low->high
                nz = [k for k, c in enumerate(den) if c % self.p]
                is_mono = len(nz) == 1
                dpow = nz[0] if is_mono else None
                nzn = [k for k, c in enumerate(num) if c % self.p]
                minp = (nzn[0] - dpow) if is_mono else None
                maxp = (nzn[-1] - dpow) if is_mono else None
                lo = -(absmu[i] - absmu[j])
                ok_window = is_mono and minp >= lo and maxp <= 1
                table[f'{i},{j}'] = {'laurent': is_mono, 'eps_powers': [minp, maxp] if is_mono else None, 'window': [lo, 1], 'in_window': ok_window,
                                     'num_deg': nzn[-1], 'den_coeffs_nonzero_at': nz}
                laurent = laurent and is_mono
                window = window and ok_window
            out['x'][xname] = {'entries': table, 'all_laurent': laurent, 'all_in_window': window, 'leaks': {str(k): v for k, v in leaks.items()}}
        return out


def _ratrec_univariate(xs, ys, p):
    """Find num/den (coeff lists low->high) with deg num + deg den <= len(xs)-2, den normalized, fitting ys at xs mod p.
    Tries (dn, dd) splits with minimal total degree. Returns None if nothing fits with one spare point for validation."""
    n = len(xs)
    for tot in range(0, n - 1):
        for dd in range(0, tot + 1):
            dn = tot - dd
            # unknowns: num coeffs (dn+1) + den coeffs (dd+1) with den leading... solve homogeneous: num(x) - y den(x) = 0
            m = dn + dd + 2
            if m - 1 > n - 1:   # need at least m-1 equations + 1 validation
                continue
            M = flint.nmod_mat(n, m, p)
            for r in range(n):
                xp = 1
                for c in range(dn + 1):
                    M[r, c] = xp; xp = xp * xs[r] % p
                xp = 1
                for c in range(dd + 1):
                    M[r, dn + 1 + c] = (-ys[r] * xp) % p; xp = xp * xs[r] % p
            # nullspace
            ns = M.nullspace()
            vecs, nullity = ns[0], ns[1]
            if nullity < 1:
                continue
            v = [int(vecs[k, 0]) for k in range(m)]
            num = v[:dn + 1]; den = v[dn + 1:]
            if all(c == 0 for c in den):
                continue
            # validate on all points
            ok = True
            for r in range(n):
                nv = sum(c * pow(xs[r], k, p) for k, c in enumerate(num)) % p
                dv = sum(c * pow(xs[r], k, p) for k, c in enumerate(den)) % p
                if dv == 0 or (nv * pow(dv, -1, p) - ys[r]) % p:
                    ok = False; break
            if ok:
                # normalize: remove common power? gcd of polys mod p
                fn = flint.nmod_poly(num, p); fd = flint.nmod_poly(den, p)
                g = fn.gcd(fd)
                if g.degree() > 0:
                    fn = fn // g; fd = fd // g
                num = [int(fn[k]) for k in range(fn.degree() + 1)] if fn.degree() >= 0 else [0]
                den = [int(fd[k]) for k in range(fd.degree() + 1)]
                return num, den
    return None


def run_sector(spec, sector, kira_masters=None, seed=5, M=6, K=14, verbose=False):
    """Claim-(A) test for one n=1 sector.  Bases: 'H_forms' = Algorithm-1 basis of pure forms (the paper's objects, with
    the monomial tie-break); 'geometric_integrals' = Algorithm-1 integral masters (what Kira receives), bare
    normalization; 'kira' = the supplied integral list, bare normalization.  Integral-basis derivatives are exact
    (symbolic-kinematics images)."""
    t0 = time.time()
    cd = CutDE(spec, sector, seed=seed, M=M, K=K, verbose=verbose)
    xnames = list(spec['symbols'])
    res = {'sector': sector, 'note': cd.note, 'D': f"{spec.get('d0',4)}-2eps", 'xnames': xnames, 'bases': {}}
    # (1) pure-form H basis
    e0r = sp.Rational(97, 389)
    p = cd.p; e0 = (int(e0r.p) * pow(int(e0r.q), -1, p)) % p
    cs0 = CutSystem(cd.rep, p, e0); cs0._M = M; cs0.e0_rational = e0r; cs0.keycache = cd.keycache; cs0.generate(M)
    forms, a_final, report = algorithm1_n1(cs0, verbose=verbose, integrals_only=False)
    basis = [c for c, k, a in forms]
    rg = cd.report(basis, [fmt(cd.rep, c) for c in basis], [int(k[3]) for c, k, a in forms], xnames)
    rg['a'] = [a for c, k, a in forms]; rg['keys_w_o_absmu'] = [list(k[1:4]) for c, k, a in forms]
    rg['normalization'] = 'paper omega-normalization (C_rel, C_clutch built in); exact'
    res['bases']['H_forms'] = rg
    # (2) geometric integral masters
    cs, images = cd.build(e0r, extra_integrals=kira_masters or [])
    ints, _, _ = algorithm1_n1(cs, verbose=False, integrals_only=True)
    bi = [c for c, k, a in ints if c[0] == 'I']
    if bi and len(bi) == len(ints):
        ri = cd.report(bi, [fmt(cd.rep, c) for c in bi], [int(k[3]) for c, k, a in ints], xnames)
        ri['a'] = [a for c, k, a in ints]
        ri['normalization'] = 'bare Feynman integrals (as Kira would differentiate them); exact derivatives'
        ri['images'] = {fmt(cd.rep, c): [str(t) for t in images[c[1]]] for c in bi}
        res['bases']['geometric_integrals'] = ri
    else:
        res['bases']['geometric_integrals'] = {'note': f'integrals do not span H (dim H={len(forms)}, integral masters={len(bi)} of {len(ints)}); form masters remain', 'basis': [fmt(cd.rep, c) for c, k, a in ints]}
    # (3) Kira basis
    if kira_masters:
        cols = [('I', tuple(nu)) for nu in kira_masters if ('I', tuple(nu)) in cs.cols]
        notes = [f'{nu}: image not in system' for nu in kira_masters if ('I', tuple(nu)) not in cs.cols]
        if cols:
            try:
                rk = cd.report(cols, [str(c[1]) for c in cols], [int(max(sum(t.mu) for t in images[c[1]])) for c in cols], xnames)
                rk['normalization'] = 'bare Feynman integrals; exact derivatives'; rk['notes'] = notes
                rk['images'] = {str(c[1]): [str(t) for t in images[c[1]]] for c in cols}
                res['bases']['kira'] = rk
            except Exception as e:
                res['bases']['kira'] = {'error': repr(e), 'notes': notes}
    res['wall_s'] = time.time() - t0
    res['stamp'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    return res


def summarize(res):
    lines = [f"sector {res['sector']} ({res['note']}), x = {res['xnames']}"]
    for bname, r in res['bases'].items():
        if 'x' not in r:
            lines.append(f"  basis {bname}: {r}"); continue
        lines.append(f"  basis {bname}: {r['basis']}  |mu|={r['absmu']}" + (f" a={r.get('a')}" if r.get('a') else '') + f"  [{r.get('normalization','')}]")
        for x, xr in r['x'].items():
            lines.append(f"    d/d{x}: all Laurent in eps: {xr['all_laurent']}; all inside F-window: {xr['all_in_window']}; leaks: {len(xr['leaks'])}")
            for k, e in sorted(xr['entries'].items()):
                if e.get('zero'): continue
                lines.append(f"      A[{k}] laurent={e.get('laurent')} eps^{e.get('eps_powers')} window={e.get('window')} in={e.get('in_window')} {e.get('status','')}")
    return '\n'.join(lines)
