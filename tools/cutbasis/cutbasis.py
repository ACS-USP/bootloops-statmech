#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of BootLoops 1.0; see LICENSE and NOTICE beside this file.
"""cutbasis — geometric master-basis selector on the maximal cut (arXiv:2511.15381, Algorithms 1-3) for Kira.

Subcommands
  census    representation census of a family: per sector the Baikov representation (loop-by-loop or democratic), the
            number of cut variables n, and whether the sector is analyzable (n <= 2 loop-by-loop, or n = 0)
  select    per analyzable sector: the basis of the maximal-cut twisted cohomology H chosen by the paper's order relation
            (a, w, o, |mu|), realized as Feynman integrals in two ways and written as Kira preferred_masters files:
              <fam>.echelon.preferred    single-integral pre-images of the leading forms of V (one per class that has one)
              <fam>.lc.preferred         the same plus BASISLC linear combinations (symbolic coefficients) for the others
              <fam>.integrals.preferred  the Laporta-with-integral-columns realization (single integrals, dots allowed)
  evaluate  run the reducer on a workload with each candidate list, certify every table against the stock table and
            keep the smallest certified one (see evaluate.py)

Typical use:
  python3 cutbasis.py census   --config CONFIG --family FAM --top SECTOR
  python3 cutbasis.py select   --config CONFIG --family FAM --top SECTOR --out OUT [--workers 8]
  python3 cutbasis.py evaluate --config CONFIG --family FAM --jobs jobs_template.yaml --targets TARGETS --out OUT/eval \\
          --candidates echelon=OUT/FAM.echelon.preferred integrals=OUT/FAM.integrals.preferred --symbols d,t,m2
(the sector list defaults to every subsector of --top; --sectormappings DIR restricts it to Kira's nonTrivialSector list)
Exit codes: 0 done; 1 a sector analysis, a reducer run or a certification failed (named line); 2 an option the tool
cannot act on (unknown --family, a sector that is not a subsector of --top, a malformed --candidates entry or a missing
file: one line '[cutbasis] usage error: ...' on stderr, nothing written); 4 evaluate found no Kira or Fermat executable.
"""
from __future__ import annotations
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geomorder.kirayaml import UsageError, USAGE_RC, family_header, parse_sectors


def cmd_census(a):
    fam, N, _ = family_header(a.config, a.family)
    sectors = _sector_list(a, fam, N)
    from geomorder.kirayaml import read_kira_family, sector_props
    from geomorder.baikov import Family, representation
    spec = read_kira_family(a.config, a.family, d0=a.d0); F = Family(spec)
    rows = []
    for sid in sectors:
        S = sector_props(sid, N)
        try:
            rep, note = representation(F, S)
            rows.append({'sector': sid, 'props': S, 'kind': rep.kind, 'n': rep.n, 'analyzable': rep.n == 0 or (rep.kind == 'loop_by_loop' and rep.n <= 2),
                         'divisors': [(d.deg, d.parity, d.a_raw, d.b) for d in rep.divisors], 'note': note})
        except Exception as e:
            rows.append({'sector': sid, 'props': S, 'kind': 'FAIL', 'n': None, 'analyzable': False, 'note': repr(e)})
        r = rows[-1]
        print(f"sector {sid:5d} {str(r['props']):32s} {r['kind']:12s} n={r['n']} {'analyzable' if r['analyzable'] else '-'} divisors={r.get('divisors')}")
    na = sum(1 for r in rows if r['analyzable'])
    print(f'{len(rows)} sectors, {na} analyzable (n<=2 loop-by-loop or n=0)')
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        json.dump({'family': fam, 'sectors': rows, 'stamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}, open(os.path.join(a.out, f'{fam}.census.json'), 'w'), indent=1, default=str)


def _sector_list(a, fam, N):
    """The sectors to analyze; a UsageError (exit 2) names a --sectors entry that is not a sector of the family or not
    a subsector of --top, and a --top outside the family."""
    if a.sectors:
        return parse_sectors(a.sectors, a.top, N, fam)
    parse_sectors(str(a.top), a.top, N, fam)
    if a.sectormappings:
        path = os.path.join(a.sectormappings, fam, 'nonTrivialSector')
        if not os.path.isfile(path):
            raise UsageError(f'--sectormappings {a.sectormappings}: no {fam}/nonTrivialSector there')
        nt = [int(l.split()[0]) for l in open(path) if l.strip()]
        return sorted(s for s in nt if s | a.top == a.top)
    return [s for s in range(1, a.top + 1) if s | a.top == a.top]


def cmd_select(a):
    fam, N, _ = family_header(a.config, a.family)
    _sector_list(a, fam, N)
    import geo_master_select
    argv = ['--config', a.config, '--top', str(a.top), '--out', a.out, '--M', str(a.M), '--dots', str(a.dots), '--sps', str(a.sps), '--workers', str(a.workers), '--seed', str(a.seed), '--maxdeg', str(a.maxdeg), '--d0', str(a.d0)]
    if a.family: argv += ['--family', a.family]
    if a.sectors: argv += ['--sectors', a.sectors]
    if a.sectormappings: argv += ['--sectormappings', a.sectormappings]
    if a.no_lc: argv += ['--no-reconstruct']
    sys.argv = ['geo_master_select.py'] + argv
    nfail = geo_master_select.main()
    # split the echelon output into single-integral and LC lists
    txt = open(os.path.join(a.out, f'{fam}.preferred')).read()
    bare = [l for l in txt.split('\n') if l.strip() and '*(' not in l]
    open(os.path.join(a.out, f'{fam}.echelon.preferred'), 'w').write('\n'.join(bare) + '\n')
    os.replace(os.path.join(a.out, f'{fam}.preferred'), os.path.join(a.out, f'{fam}.lc.preferred'))
    if not a.no_integrals:
        import imode
        sectors = _sector_list(a, fam, N)
        sys.argv = ['imode.py', '--config', a.config, '--sectors', ','.join(str(s) for s in sectors), '--out', os.path.join(a.out, 'integrals'), '--M', str(a.M), '--workers', str(a.workers), '--seed', str(a.seed), '--d0', str(a.d0)] + (['--family', a.family] if a.family else [])
        nfail += imode.main()
        os.replace(os.path.join(a.out, 'integrals', f'{fam}.imode.preferred'), os.path.join(a.out, f'{fam}.integrals.preferred'))
    print(f'[cutbasis] wrote {fam}.echelon.preferred, {fam}.lc.preferred' + ('' if a.no_integrals else f', {fam}.integrals.preferred') + f' under {a.out}')
    if nfail:
        print(f'[cutbasis] FAILED: {nfail} sector analysis(es) did not complete (lines marked FAILED above); the lists are incomplete', flush=True)
        return 1
    return 0


def cmd_evaluate(a):
    import evaluate
    return evaluate.run(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('census', 'select'):
        p = sub.add_parser(name)
        p.add_argument('--config', required=True); p.add_argument('--family', default=None); p.add_argument('--top', type=int, required=True)
        p.add_argument('--sectors', default=None); p.add_argument('--sectormappings', default=None); p.add_argument('--d0', type=int, default=4)
        p.add_argument('--out', default=None if name == 'census' else 'cutbasis_out')
        if name == 'select':
            p.add_argument('--M', type=int, default=5); p.add_argument('--dots', type=int, default=2); p.add_argument('--sps', type=int, default=2)
            p.add_argument('--workers', type=int, default=8); p.add_argument('--seed', type=int, default=5); p.add_argument('--maxdeg', type=int, default=12)
            p.add_argument('--no-lc', action='store_true', help='skip the symbolic reconstruction of linear-combination masters')
            p.add_argument('--no-integrals', action='store_true', help='skip the Laporta-with-integral-columns realization')
    import evaluate
    evaluate.add_arguments(sub.add_parser('evaluate', help='run the reducer with each candidate list and certify (exit 4: no Kira or Fermat found)'))
    a = ap.parse_args()
    try:
        return {'census': cmd_census, 'select': cmd_select, 'evaluate': cmd_evaluate}[a.cmd](a)
    except UsageError as e:
        print(f'[cutbasis] usage error: {e}', file=sys.stderr, flush=True)
        return USAGE_RC


if __name__ == '__main__':
    sys.exit(main() or 0)
