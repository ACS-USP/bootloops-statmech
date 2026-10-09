#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of BootLoops 1.0; see LICENSE and NOTICE beside this file.
"""tablecheck.py — compare two Kira reduction tables (kira2math .m rule files) of the SAME target list onto DIFFERENT
master bases, and extract per-table complexity metrics.

Consistency (exact mod p): parse both tables; let A reduce targets onto basis M_A and B onto M_B.
Every master of M_B must itself appear as a target (or master) of table A, giving the basis change T (|M_B| x |M_A|) over
F_p at random (kinematics, d); then for every common target x:  coeff_A(x) == coeff_B(x) . T  (mod p).  Two primes/points.
Metrics per table: bytes, number of rules, total number of monomial terms, max total degree of numerator and denominator
polynomials (in d and the invariants), number of DISTINCT irreducible denominator factors and the list of those that
are not in a user-supplied "alphabet" (spurious-denominator census), max integer coefficient bit length.

Usage: tablecheck.py --a A.m --b B.m [--symbols d,t] [--alphabet 't,1+t,d-4,...'] [--json out.json]
"""
import argparse, json, random, re, sys, time
import sympy as sp

RULE_RE = re.compile(r'^([A-Za-z]\w*\[[-0-9, ]+\])\s*->\s*$')
TERM_RE = re.compile(r'^\s*\+\s*([A-Za-z]\w*\[[-0-9, ]+\])\*\((.*)\)\s*$')


def parse_table(path):
    rules = {}
    cur = None
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip() or line.strip() in '{},':
                if line.strip() == ',':
                    cur = None
                continue
            m = RULE_RE.match(line.strip())
            if m:
                cur = m.group(1).replace(' ', ''); rules[cur] = []; continue
            m = TERM_RE.match(line)
            if m and cur is not None:
                rules[cur].append((m.group(1).replace(' ', ''), m.group(2)))
                continue
            if line.strip().endswith('-> 0') or line.strip().endswith('->0'):
                k = line.split('->')[0].strip().replace(' ', ''); rules[k] = []
                continue
    return rules


def masters_of(rules):
    ms = set()
    for k, terms in rules.items():
        for m, c in terms:
            ms.add(m)
    return sorted(ms)


def evalmod(expr_str, sub, p, symtab):
    e = sp.sympify(expr_str.replace('^', '**'), locals=symtab)
    v = e.subs(sub)
    v = sp.Rational(v)
    return (int(v.p) * pow(int(v.q), -1, p)) % p


def parse_lc_defs(path):
    """Kira preferred_masters file -> {'BASISLC[n]': [(integral name, coeff string), ...]} (blocks of FAM[...]*(c) lines
    separated by blank lines, numbered 1.. in file order; bare lines are ignored)."""
    blocks, cur = [], []
    for line in open(path):
        line = line.strip()
        if not line:
            if cur: blocks.append(cur); cur = []
            continue
        m = re.match(r'^([A-Za-z]\w*\[[-0-9, ]+\])\*\((.*)\)$', line)
        if m:
            cur.append((m.group(1).replace(' ', ''), m.group(2)))
        elif cur:
            blocks.append(cur); cur = []
    if cur: blocks.append(cur)
    return {f'BASISLC[{i+1}]': b for i, b in enumerate(blocks)}


def sector_of(name):
    if name.startswith('BASISLC'):
        return (-1, 0, 0, 0)
    idx = [int(x) for x in re.search(r'\[(.*)\]', name).group(1).split(',')]
    return sum(1 << i for i, v in enumerate(idx) if v > 0), sum(1 for v in idx if v > 0), sum(v - 1 for v in idx if v > 0), sum(-v for v in idx if v < 0)


def metrics(path, rules, symtab, alphabet, dsym='d'):
    """Global and per-block (target sector -> master sector) complexity census of a table."""
    import os
    syms = list(symtab.values())
    dS = symtab.get(dsym)
    glob = {'terms': 0, 'max_deg_num': 0, 'max_deg_den': 0, 'max_coeff_bits': 0, 'bytes': os.path.getsize(path), 'rules': len(rules)}
    factors = {}      # monic factor string -> {'class': pure_d|kinematic|mixed, 'count': n}
    blocks = {}       # "tsec->msec" -> dict
    fcache = {}
    def classify(fe):
        fs = fe.free_symbols
        if dS is not None and fs == {dS}: return 'pure_d'
        if dS is None or dS not in fs: return 'kinematic'
        return 'mixed'
    for k, terms in rules.items():
        tsec, tl, tdots, tsps = sector_of(k)
        for m, c in terms:
            msec, ml, mdots, msps = sector_of(m)
            bkey = f'{tsec}->{msec}'
            B = blocks.setdefault(bkey, {'target_sector': tsec, 'master_sector': msec, 't_lines': tl, 'm_lines': ml, 'terms': 0, 'max_deg_num': 0, 'max_deg_den': 0, 'bytes': 0, 'factors': {}})
            glob['terms'] += 1; B['terms'] += 1; B['bytes'] += len(c)
            e = sp.sympify(c.replace('^', '**'), locals=symtab)
            num, den = sp.fraction(sp.together(e))
            try:
                pn = sp.Poly(sp.expand(num), *syms); pd = sp.Poly(sp.expand(den), *syms)
            except sp.PolynomialError:
                continue
            dn, dd = pn.total_degree(), pd.total_degree()
            glob['max_deg_num'] = max(glob['max_deg_num'], dn); glob['max_deg_den'] = max(glob['max_deg_den'], dd)
            B['max_deg_num'] = max(B['max_deg_num'], dn); B['max_deg_den'] = max(B['max_deg_den'], dd)
            for cf in pn.coeffs() + pd.coeffs():
                glob['max_coeff_bits'] = max(glob['max_coeff_bits'], int(abs(sp.Rational(cf).p)).bit_length())
            dkey = str(pd.as_expr())
            if dkey not in fcache:
                fl = []
                for fac, mult in sp.factor_list(pd)[1]:
                    fe = fac.as_expr()
                    if not fe.free_symbols: continue
                    mon = sp.Poly(fe, *syms).monic().as_expr()
                    fl.append((str(mon), classify(mon), mult))
                fcache[dkey] = fl
            for mon, cls, mult in fcache[dkey]:
                F_ = factors.setdefault(mon, {'class': cls, 'count': 0, 'max_power': 0}); F_['count'] += 1; F_['max_power'] = max(F_['max_power'], mult)
                bf = B['factors'].setdefault(mon, {'class': cls, 'count': 0, 'max_power': 0}); bf['count'] += 1; bf['max_power'] = max(bf['max_power'], mult)
    alpha = set()
    for a in alphabet:
        ae = sp.sympify(a.replace('^', '**'), locals=symtab)
        alpha.add(str(sp.Poly(ae, *syms).monic().as_expr()) if ae.free_symbols else str(ae))
    def summar(fd):
        out = {'pure_d': 0, 'kinematic': 0, 'mixed': 0, 'kinematic_not_in_alphabet': 0}
        for mon, v in fd.items():
            out[v['class']] += 1
            if v['class'] == 'kinematic' and mon not in alpha: out['kinematic_not_in_alphabet'] += 1
        return out
    glob['factor_classes'] = summar(factors)
    glob['factors'] = factors
    for bk, B in blocks.items():
        B['factor_classes'] = summar(B['factors'])
    glob['blocks'] = blocks
    glob['file'] = path
    return glob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True); ap.add_argument('--b', required=True)
    ap.add_argument('--symbols', default='d,t'); ap.add_argument('--alphabet', default='')
    ap.add_argument('--json', default=None); ap.add_argument('--nchecks', type=int, default=2)
    ap.add_argument('--no-metrics', action='store_true')
    ap.add_argument('--lc-defs', default=None, help='preferred_masters file defining BASISLC[n] masters of table B (and/or A)')
    args = ap.parse_args()
    syms = [s.strip() for s in args.symbols.split(',') if s.strip()]
    symtab = {s: sp.Symbol(s) for s in syms}
    A = parse_table(args.a); B = parse_table(args.b)
    MA = masters_of(A); MB = masters_of(B)
    out = {'a': args.a, 'b': args.b, 'masters_a': MA, 'masters_b': MB, 'n_rules_a': len(A), 'n_rules_b': len(B)}
    # basis change: each master of B as a combination of MA using table A (masters of A map to themselves)
    lcdefs = parse_lc_defs(args.lc_defs) if args.lc_defs else {}
    def row_in_A(x, sub, p):
        if x in MA:
            return {x: 1}
        if x in A:
            return {m: evalmod(c, sub, p, symtab) for m, c in A[x]}
        if x in lcdefs:
            acc = {}
            for name, c in lcdefs[x]:
                r = row_in_A(name, sub, p)
                if r is None:
                    return None
                cv = evalmod(c, sub, p, symtab)
                for m, v in r.items():
                    acc[m] = (acc.get(m, 0) + cv * v) % p
            return acc
        return None
    rng = random.Random(1)
    primes = [2147483647, 2147483629, 2147483587]
    checks = []
    ok_all = True
    t0 = time.time()
    for i in range(args.nchecks):
        p = primes[i % len(primes)]
        sub = {symtab[s]: rng.randint(5, 10 ** 6) for s in syms}
        T = {}
        missing = []
        for mb in MB:
            r = row_in_A(mb, sub, p)
            if r is None: missing.append(mb)
            else: T[mb] = r
        nbad = 0; ncmp = 0; bad_examples = []
        for x, termsB in B.items():
            if x not in A and x not in MA:
                continue
            lhs = row_in_A(x, sub, p)
            rhs = {}
            okx = True
            for mb, c in termsB:
                if mb not in T: okx = False; break
                cv = evalmod(c, sub, p, symtab)
                for ma, tv in T[mb].items():
                    rhs[ma] = (rhs.get(ma, 0) + cv * tv) % p
            if not okx:
                continue
            ncmp += 1
            keys = set(lhs) | set(rhs)
            if any((lhs.get(k, 0) - rhs.get(k, 0)) % p for k in keys):
                nbad += 1
                if len(bad_examples) < 3: bad_examples.append(x)
        checks.append({'prime': p, 'point': {str(k): v for k, v in sub.items()}, 'compared': ncmp, 'mismatches': nbad, 'missing_basis_rows': missing, 'bad_examples': bad_examples})
        ok_all = ok_all and nbad == 0 and ncmp > 0 and not missing
    out['checks'] = checks; out['consistent'] = ok_all; out['check_wall_s'] = time.time() - t0
    if not args.no_metrics:
        alphabet = [a for a in args.alphabet.split(',') if a.strip()]
        out['metrics_a'] = metrics(args.a, A, symtab, alphabet)
        out['metrics_b'] = metrics(args.b, B, symtab, alphabet)
    out['stamp'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    js = json.dumps(out, indent=1, default=str)
    if args.json:
        with open(args.json, 'w') as f: f.write(js)
    print(json.dumps({k: v for k, v in out.items() if k in ('consistent', 'checks', 'n_rules_a', 'n_rules_b')}, default=str))
    if not args.no_metrics:
        for tag in ('metrics_a', 'metrics_b'):
            m = out[tag]; print(tag, {k: m[k] for k in ('bytes', 'rules', 'terms', 'max_deg_num', 'max_deg_den', 'max_coeff_bits')}, m['factor_classes'])
            print('   factors:', {k: (v['class'], v['count'], v['max_power']) for k, v in sorted(m['factors'].items())})
    sys.exit(0 if ok_all else 1)


if __name__ == '__main__':
    main()
