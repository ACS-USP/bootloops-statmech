#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of BootLoops 1.0; see LICENSE and NOTICE beside this file.
"""Builder of the committed planted-defect fixtures that selftest.py reads (maintainer use; the self-test never runs it).

  fixtures/tablecheck/A.txt, B.txt, pref   two reduction tables of the same targets on two bases related by the
                                       linear combination in pref (B must certify against A)
  fixtures/tablecheck/Bbad.txt           B with one coefficient changed by +1 (must be refused, exit 1)
  fixtures/ratrec/samples.json         mod-p samples of a known rational function at two primes, a verification
                                       point at a third prime, and a copy of the first sample set with one value
                                       corrupted (the dense fit must reject it)
  fixtures/doublebox_kira/echelon.preferred, alt.preferred
                                       two preferred-masters lists for the double-box top sector that make the
                                       reducer choose different masters (the reducer check needs both to certify
                                       and the tables to differ)

Run from the package directory: python3 fixtures/build_fixtures.py
"""
import json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import sympy as sp
from geomorder import ratrec
from geomorder.lcbasis import PRIMES61


def write_table(path, rules):
    with open(path, 'w') as f:
        f.write('{\n')
        for k, (tgt, terms) in enumerate(rules):
            f.write(f'{tgt} ->\n')
            for m, c in terms:
                f.write(f' + {m}*({c})\n')
            f.write(',\n' if k < len(rules) - 1 else '\n')
        f.write('}\n')


def tablecheck_fixtures():
    d, t = sp.symbols('d t')
    # masters of A: m1, m2 ; targets x1, x2 ; basis of B: n1 = m1, n2 = m1 + (t/(d-3)) m2  -> m2 = (d-3)/t (n2 - n1)
    m1, m2, n1, n2 = 'fam[1,1,1,0]', 'fam[1,1,2,0]', 'fam[1,1,1,0]', 'BASISLC[1]'
    A = {'fam[1,1,1,-1]': [(m1, '(d-4)/t'), (m2, '3*t/(d-2)')], 'fam[2,1,1,0]': [(m1, 't^2'), (m2, '(d-1)/(t+1)')]}
    c = t / (d - 3)
    def rot(terms):
        a = sp.sympify(dict(terms)[m1].replace('^', '**')); b = sp.sympify(dict(terms)[m2].replace('^', '**'))
        # a m1 + b m2 = (a - b (d-3)/t) n1 + b (d-3)/t n2
        return [(n1, str(sp.cancel(a - b / c)).replace('**', '^')), (n2, str(sp.cancel(b / c)).replace('**', '^'))]
    B = {k: rot(v) for k, v in A.items()}
    Bbad = {k: list(v) for k, v in B.items()}
    Bbad['fam[2,1,1,0]'][0] = (n1, '(' + Bbad['fam[2,1,1,0]'][0][1] + ')+1')      # the planted wrong coefficient
    D = os.path.join(HERE, 'tablecheck'); os.makedirs(D, exist_ok=True)
    write_table(os.path.join(D, 'A.txt'), list(A.items())); write_table(os.path.join(D, 'B.txt'), list(B.items())); write_table(os.path.join(D, 'Bbad.txt'), list(Bbad.items()))
    open(os.path.join(D, 'pref'), 'w').write(f'{m1}*(1)\n{m2}*({str(c).replace("**", "^").replace(" ", "")})\n')
    print(f'wrote {D}/A.txt B.txt Bbad.txt pref')


def ratrec_fixtures():
    e, x, y = sp.symbols('eps x y')
    syms = [e, x, y]
    f = (3 * e ** 2 * x - 2 * y + 7) / (e * x * y + 5 * y ** 2 - 1)
    rng = random.Random(3)
    p1, p2, p3 = PRIMES61[:3]
    out = {'function': str(f), 'symbols': [str(s) for s in syms], 'degrees': [3, 3], 'primes': [p1, p2], 'samples': {}}
    for p in (p1, p2):
        pts, vals = [], []
        while len(pts) < 50:
            xx = [rng.randrange(2, p - 1) for _ in range(3)]
            v = ratrec.ratfunc_mod(f, syms, xx, p)
            if v is None:
                continue
            pts.append(xx); vals.append(v)
        out['samples'][str(p)] = {'points': pts, 'values': vals}
    xx = [rng.randrange(2, p3 - 1) for _ in range(3)]
    out['verify'] = {'prime': p3, 'point': xx, 'value': ratrec.ratfunc_mod(f, syms, xx, p3)}
    pts, vals = [], []
    while len(pts) < 50:
        xx = [rng.randrange(2, p1 - 1) for _ in range(3)]
        v = ratrec.ratfunc_mod(f, syms, xx, p1)
        if v is None:
            continue
        pts.append(xx); vals.append(v)
    vals[7] = (vals[7] + 1) % p1                                                    # the planted corrupted sample
    out['corrupted'] = {'prime': p1, 'points': pts, 'values': vals, 'corrupted_index': 7}
    D = os.path.join(HERE, 'ratrec'); os.makedirs(D, exist_ok=True)
    with open(os.path.join(D, 'samples.json'), 'w') as fh:
        json.dump(out, fh, indent=1); fh.write('\n')
    print(f'wrote {D}/samples.json')


def doublebox_lists():
    D = os.path.join(HERE, 'doublebox_kira')
    # the selection of the doublebox_select check (corner + the ISP in slot 9)
    open(os.path.join(D, 'echelon.preferred'), 'w').write('db[1,1,1,1,1,1,1,0,0]\ndb[1,1,1,1,1,1,1,0,-1]\n')
    # a different legitimate basis (corner + one dot): certifies against stock, gives a different and larger table
    open(os.path.join(D, 'alt.preferred'), 'w').write('db[1,1,1,1,1,1,2,0,0]\ndb[1,1,1,1,1,1,1,0,0]\n')
    print(f'wrote {D}/echelon.preferred alt.preferred')


if __name__ == '__main__':
    tablecheck_fixtures(); ratrec_fixtures(); doublebox_lists()
