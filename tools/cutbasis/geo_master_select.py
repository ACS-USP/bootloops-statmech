#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of BootLoops 1.0; see LICENSE and NOTICE beside this file.
"""geo_master_select.py — geometric master selector (arXiv:2511.15381 Algorithms 1-3 on the maximal cut) for Kira.

For every sector of a family (inside the Kira seed box) with a good loop-by-loop Baikov representation with at most
two cut variables (and every one-line/trivial n=0 sector): the basis of the twisted cohomology H on the maximal cut
chosen by the paper's order relation (a, w, o, |mu|), realized as Feynman integrals — single integrals where a basis
form has a single-integral pre-image, linear combinations with symbolic coefficients (Kira BASISLC blocks) otherwise.

usage: geo_master_select.py --config <kira config dir> --family NAME --top SECTOR --out DIR
         [--sectors 107,73,...] [--sectormappings DIR] [--M 5] [--dots 2] [--sps 2] [--workers 8] [--d0 4]
         [--no-reconstruct] [--maxdeg 8] [--seed 5]
outputs in DIR: <fam>.preferred (Kira preferred_masters text), summary.json, ckpt/sec<id>.pkl|json, select.log
Exit codes: 0 every analyzed sector completed; 1 a sector failed (its error is named in select.log and summary.json;
a sector whose cut system is empty at the truncation --M, dim H = 0 with columns left at the truncation boundary, is
such a failure: raise --M); 2 a usage error (unknown --family, a --sectors entry that is not a subsector of --top).
"""
from __future__ import annotations
import argparse, json, os, pickle, sys, time, traceback
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp
from geomorder.kirayaml import read_kira_family, sector_props, sector_id, UsageError, USAGE_RC, family_header, parse_sectors
from geomorder.baikov import Family, representation
from geomorder.lcbasis import SectorLC, reconstruct_sector, preferred_text


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _sector_work(args):
    spec, sector, seed, M, dots, sps, maxsup, pool = args
    t0 = time.time()
    try:
        sl = SectorLC(spec, sector, seed=seed, M=M, cand_dots=dots, cand_sps=sps, max_support=maxsup, pool_size=pool)
        r = sl.base()
        if r.get('n') and r.get('dimH') == 0 and r.get('n_boundary_nonpivot'):
            # every non-pivot column of the cut system sits at the truncation boundary: the truncation is too small for
            # this sector (a scaleless sector has dim H = 0 with no boundary columns and is not this case)
            return {'sector': sector_id(sector), 'props': sector,
                    'error': f"dim H = 0 at M={M} with {r['n_boundary_nonpivot']} column(s) at the truncation boundary and "
                             f"{r.get('n_candidates', 0)} candidate integrals inside |mu| <= M: the truncation is too small for this sector; raise --M",
                    'wall_s': time.time() - t0}
        if r.get('dimH') and hasattr(sl, 'H'):
            sl.structure()
        return sl
    except Exception as e:
        return {'sector': sector_id(sector), 'props': sector, 'error': repr(e) + traceback.format_exc()[-1200:], 'wall_s': time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True); ap.add_argument('--family', default=None); ap.add_argument('--d0', type=int, default=4)
    ap.add_argument('--top', type=int, required=True); ap.add_argument('--sectors', default=None); ap.add_argument('--sectormappings', default=None)
    ap.add_argument('--out', required=True); ap.add_argument('--M', type=int, default=5); ap.add_argument('--dots', type=int, default=2); ap.add_argument('--sps', type=int, default=2)
    ap.add_argument('--workers', type=int, default=8); ap.add_argument('--seed', type=int, default=5); ap.add_argument('--no-reconstruct', action='store_true')
    ap.add_argument('--maxdeg', type=int, default=8); ap.add_argument('--max-support', type=int, default=4); ap.add_argument('--pool', type=int, default=40)
    a = ap.parse_args()
    fam, N, _ = family_header(a.config, a.family)
    if a.sectors:
        sectors = parse_sectors(a.sectors, a.top, N, fam)
    else:
        parse_sectors(str(a.top), a.top, N, fam)
        if a.sectormappings:
            path = os.path.join(a.sectormappings, fam, 'nonTrivialSector')
            if not os.path.isfile(path):
                raise UsageError(f'--sectormappings {a.sectormappings}: no {fam}/nonTrivialSector there')
            nt = [int(l.split()[0]) for l in open(path) if l.strip()]
            sectors = sorted(s for s in nt if s | a.top == a.top)
        else:
            sectors = [s for s in range(1, a.top + 1) if s | a.top == a.top]
    os.makedirs(a.out, exist_ok=True); ck = os.path.join(a.out, 'ckpt'); os.makedirs(ck, exist_ok=True)
    logf = open(os.path.join(a.out, 'select.log'), 'a')
    def L(s):
        logf.write(f'{stamp()} {s}\n'); logf.flush(); print(s, flush=True)
    spec = read_kira_family(a.config, a.family, d0=a.d0)
    F = Family(spec)
    t0 = time.time()
    L(f'[select] family {fam} nprops {N} top {a.top}: {len(sectors)} sectors; M={a.M} dots<={a.dots} sps<={a.sps}')
    # phase A: representation census
    cls = {}
    for sid in sectors:
        S = sector_props(sid, N)
        try:
            rep, note = representation(F, S)
            analyzed = rep.n == 0 or (rep.kind == 'loop_by_loop' and rep.n <= 2)
            cls[sid] = {'props': S, 'kind': rep.kind, 'n': rep.n, 'note': note, 'analyzed': analyzed}
        except Exception as e:
            cls[sid] = {'props': S, 'kind': 'FAIL', 'n': None, 'note': repr(e), 'analyzed': False}
    nA = sum(1 for v in cls.values() if v['analyzed'])
    L(f'[select] census: {nA} analyzed (n<=2 loop-by-loop or n=0), {len(sectors) - nA} other; {time.time() - t0:.0f}s')
    # phase B: base + structure per analyzed sector (checkpointed)
    tasks = []
    for sid in sectors:
        if cls[sid]['analyzed'] and not os.path.exists(os.path.join(ck, f'sec{sid}.pkl')):
            tasks.append((spec, cls[sid]['props'], a.seed, a.M, a.dots, a.sps, a.max_support, a.pool))
    if tasks:
        with Pool(min(a.workers, len(tasks))) as pool:
            for res in pool.imap_unordered(_sector_work, tasks):
                if isinstance(res, dict):
                    sid = res['sector']
                    with open(os.path.join(ck, f'sec{sid}.json.tmp'), 'w') as fh: json.dump(res, fh, default=str)
                    os.replace(os.path.join(ck, f'sec{sid}.json.tmp'), os.path.join(ck, f'sec{sid}.json'))
                    L(f'[select] sector {sid} FAILED: {res["error"][:300]}')
                    continue
                sid = sector_id(res.sector)
                with open(os.path.join(ck, f'sec{sid}.pkl.tmp'), 'wb') as fh: pickle.dump(res, fh)
                os.replace(os.path.join(ck, f'sec{sid}.pkl.tmp'), os.path.join(ck, f'sec{sid}.pkl'))
                r = res.result
                L(f"[select] sector {sid} {res.sector} n={r.get('n')} dimH={r.get('dimH')} dimV={r.get('dimV')} plan={[ (e['form'], len(e['support'])) for e in r.get('plan', [])]} {r.get('wall_base_s', 0):.0f}s")
    # phase C: reconstruction where needed
    sls = {}
    for sid in sectors:
        f = os.path.join(ck, f'sec{sid}.pkl')
        if os.path.exists(f):
            sls[sid] = pickle.load(open(f, 'rb'))
    for sid, sl in sorted(sls.items()):
        if getattr(sl, 'masters', None) is not None:
            continue
        plan = getattr(sl, 'plan', None)
        if not plan:
            continue
        if all(len(e['support']) == 1 for e in plan):
            sl.masters = [{'form': e['form'], 'a': e['a'], 'key': e['key'], 'lc': [(list(e['support'][0]), sp.Integer(1))]} for e in plan]
            sl.result['reconstruction'] = 'not needed (all single-integral pre-images)'
        elif a.no_reconstruct:
            continue
        else:
            t1 = time.time()
            L(f'[select] sector {sid}: reconstructing {sum(1 for e in plan if len(e["support"]) > 1)} linear-combination masters')
            try:
                reconstruct_sector(sl, workers=a.workers, maxdeg=a.maxdeg, verbose=False, logf=os.path.join(a.out, 'select.log'))
            except Exception as e:
                L(f'[select] sector {sid}: reconstruction FAILED {e!r}'); sl.result['reconstruction_error'] = repr(e); continue
            L(f'[select] sector {sid}: reconstruction done {time.time() - t1:.0f}s: ' + '; '.join(m['form'] + ' = ' + ' + '.join(f'({c})*{fam}{nu}' for nu, c in m['lc']) for m in sl.masters))
        with open(f + '.tmp', 'wb') as fh: pickle.dump(sl, fh)
        os.replace(f + '.tmp', f)
    # phase D: write
    per = []
    summary = {'family': fam, 'top': a.top, 'M': a.M, 'dots': a.dots, 'sps': a.sps, 'stamp': stamp(), 'sectors': {}}
    for sid in sectors:
        c = cls[sid]
        entry = dict(c)
        sl = sls.get(sid)
        if sl is not None:
            entry.update({k: v for k, v in sl.result.items() if k not in ('H',)})
            entry['H'] = sl.result.get('H')
            ms = getattr(sl, 'masters', None)
            if ms is None and sl.result.get('n') == 0:
                ms = [{'form': m['form'], 'a': m['a'], 'key': m['key'], 'lc': [(m['nu'], sp.Integer(1))]} for m in sl.result['masters']]
            if ms is not None:
                entry['masters'] = [{'form': m['form'], 'a': m['a'], 'key': m['key'], 'lc': [(nu, str(c)) for nu, c in m['lc']]} for m in ms]
                per.append((len(c['props']), sid, {'masters': ms}))
            else:
                entry['masters'] = None
        elif os.path.exists(os.path.join(ck, f'sec{sid}.json')):
            entry['error'] = json.load(open(os.path.join(ck, f'sec{sid}.json')))['error']
        summary['sectors'][sid] = entry
    per.sort(key=lambda t: (t[0], t[1]))
    txt = preferred_text(fam, [t[2] for t in per], d0=a.d0)
    with open(os.path.join(a.out, f'{fam}.preferred.tmp'), 'w') as fh: fh.write(txt)
    os.replace(os.path.join(a.out, f'{fam}.preferred.tmp'), os.path.join(a.out, f'{fam}.preferred'))
    summary['n_preferred_lines'] = sum(1 for l in txt.splitlines() if l.strip())
    summary['wall_total_s'] = time.time() - t0
    with open(os.path.join(a.out, 'summary.json.tmp'), 'w') as fh: json.dump(summary, fh, indent=1, default=str)
    os.replace(os.path.join(a.out, 'summary.json.tmp'), os.path.join(a.out, 'summary.json'))
    L(f'[select] wrote {fam}.preferred ({summary["n_preferred_lines"]} lines) + summary.json; {time.time() - t0:.0f}s total')
    for sid in sectors:
        d = summary['sectors'][sid]
        L(f"  sector {sid:4d} {str(d['props']):26s} {d['kind']:12s} n={d['n']} {'ANALYZED' if d.get('analyzed') else 'other'} dimH={d.get('dimH')} dimV={d.get('dimV')} masters={[(m['form'], [nu for nu, c in m['lc']]) for m in (d.get('masters') or [])]} err={'error' in d}")
    failed = sorted(sid for sid, d in summary['sectors'].items() if 'error' in d or 'reconstruction_error' in d)
    summary['failed_sectors'] = failed
    with open(os.path.join(a.out, 'summary.json.tmp'), 'w') as fh: json.dump(summary, fh, indent=1, default=str)
    os.replace(os.path.join(a.out, 'summary.json.tmp'), os.path.join(a.out, 'summary.json'))
    if failed:
        L(f"[select] FAILED: {len(failed)} sector(s) did not complete: {', '.join(str(s) for s in failed)} (error text in summary.json and ckpt/sec<id>.json)")
    return len(failed)


if __name__ == '__main__':
    try:
        sys.exit(1 if main() else 0)
    except UsageError as e:
        print(f'[select] usage error: {e}', file=sys.stderr, flush=True)
        sys.exit(USAGE_RC)
