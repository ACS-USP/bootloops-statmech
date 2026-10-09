#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of BootLoops 1.0; see LICENSE and NOTICE beside this file.
"""Regenerate the committed known answers that selftest.py compares against (maintainer use only; the self-test never
runs it, and a change in these files is a change of the tool's documented output that must be explained in the commit).

  fixtures/expected_doublebox.json         the doublebox_select check (H-basis, leading forms, masters)
  examples/moeller_tower/expected/quick/   the moeller_quick check: the README's quick command (sector 42)

Run from the package directory: python3 fixtures/regenerate.py [--doublebox] [--quick]   (both when no flag is given)
"""
import argparse, json, os, shutil, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)


def doublebox():
    from geomorder.lcbasis import SectorLC
    spec = json.load(open(os.path.join(HERE, 'paper_doublebox.json')))
    sl = SectorLC(spec, [1, 2, 3, 4, 5, 6, 7], seed=5, M=4, cand_dots=1, cand_sps=1)
    r = sl.base(); sl.structure()
    got = {'dimH': r['dimH'], 'dimV': sl.dimV, 'H': sorted((h['form'], h['a'], h['key']) for h in r['H']),
           'masters': [(e['form'], [list(nu) for nu in e['support']]) for e in sl.plan]}
    with open(os.path.join(HERE, 'expected_doublebox.json'), 'w') as fh:
        json.dump(got, fh, indent=1); fh.write('\n')
    print('wrote fixtures/expected_doublebox.json')


def quick():
    ex = os.path.join(PKG, 'examples', 'moeller_tower'); dst = os.path.join(ex, 'expected', 'quick')
    with tempfile.TemporaryDirectory(prefix='cutbasis_regen_') as tmp:
        out = os.path.join(tmp, 'out')
        subprocess.run([sys.executable, '-B', os.path.join(PKG, 'cutbasis.py'), 'select', '--config', os.path.join(ex, 'input'), '--family', 'moeller', '--top', '107',
                        '--sectors', '42', '--out', out, '--workers', '4'], check=True)
        os.makedirs(dst, exist_ok=True)
        for f in ('moeller.echelon.preferred', 'moeller.lc.preferred', 'moeller.integrals.preferred'):
            shutil.copy(os.path.join(out, f), os.path.join(dst, f))
    print('wrote examples/moeller_tower/expected/quick/')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--doublebox', action='store_true'); ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    if a.doublebox or not (a.doublebox or a.quick): doublebox()
    if a.quick or not (a.doublebox or a.quick): quick()
