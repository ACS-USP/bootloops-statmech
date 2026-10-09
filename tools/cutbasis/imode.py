#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of BootLoops 1.0; see LICENSE and NOTICE beside this file.
"""Integrals-mode geometric masters (Laporta with the Feynman integrals as columns): Algorithm 1-3 Laporta on the maximal cut WITH the candidate
Feynman integrals as columns (integrals preferred within a class, Kira tie-breaks) -> single-integral masters per sector.
usage: imode.py --config DIR --family NAME --sectors a,b,c --out DIR [--M 5] [--workers 8] [--seed 5] [--d0 4]"""
import sys, os, json, time, argparse, traceback
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geomorder.kirayaml import read_kira_family, sector_props, sector_id
from geomorder.baikov import Family, representation
from geomorder.masters import sector_masters

def work(args):
    spec, sector, seed, M = args
    try:
        return sector_masters(spec, list(sector), seed=seed, M=M)
    except Exception as e:
        return {'sector': sector_id(list(sector)), 'props': list(sector), 'error': repr(e) + traceback.format_exc()[-600:], 'masters': [], 'flags': ['failed']}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--family'); ap.add_argument('--sectors', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--M', type=int, default=5); ap.add_argument('--workers', type=int, default=8); ap.add_argument('--seed', type=int, default=5); ap.add_argument('--d0', type=int, default=4)
    a = ap.parse_args(); os.makedirs(os.path.join(a.out, 'ckpt'), exist_ok=True)
    spec = read_kira_family(a.config, a.family, d0=a.d0); fam = spec['name']; N = len(spec['props']); F = Family(spec)
    sectors = [int(x) for x in a.sectors.split(',')]
    log = open(os.path.join(a.out, 'imode.log'), 'a')
    def L(s): log.write(time.strftime('%Y-%m-%dT%H:%M:%SZ ', time.gmtime()) + s + '\n'); log.flush(); print(s, flush=True)
    cls = {}
    for sid in sectors:
        S = sector_props(sid, N)
        try:
            rep, note = representation(F, S); cls[sid] = {'props': S, 'n': rep.n, 'kind': rep.kind, 'analyzed': rep.n == 0 or (rep.kind == 'loop_by_loop' and rep.n <= 2)}
        except Exception:
            cls[sid] = {'props': S, 'n': None, 'kind': 'FAIL', 'analyzed': False}
    tasks = [(spec, tuple(cls[s]['props']), a.seed, a.M) for s in sectors if cls[s]['analyzed'] and not os.path.exists(os.path.join(a.out, 'ckpt', f'masters{s}.json'))]
    L(f'[imode] {fam}: {len(sectors)} sectors, {sum(1 for s in sectors if cls[s]["analyzed"])} analyzed, {len(tasks)} to run')
    if tasks:
        with Pool(min(a.workers, len(tasks))) as pool:
            for res in pool.imap_unordered(work, tasks):
                sid = res['sector']; f = os.path.join(a.out, 'ckpt', f'masters{sid}.json')
                json.dump(res, open(f + '.tmp', 'w'), default=str); os.replace(f + '.tmp', f)
                L(f"[imode] sector {sid}: {[m.get('nu') or m.get('form') for m in res.get('masters', [])]} a={[m['a'] for m in res.get('masters', [])]} flags={len(res.get('flags', []))} {res.get('wall_s', 0):.0f}s")
    pref = []
    summ = {'family': fam, 'sectors': {}}
    for sid in sectors:
        f = os.path.join(a.out, 'ckpt', f'masters{sid}.json')
        if not os.path.exists(f):
            summ['sectors'][sid] = dict(cls[sid]); continue
        res = json.load(open(f)); summ['sectors'][sid] = dict(cls[sid], masters=res.get('masters'), flags=res.get('flags'), error=res.get('error'))
        for m in res.get('masters', []):
            if m.get('nu') is not None:
                pref.append((len(cls[sid]['props']), sid, int(m['a']), tuple(m.get('key') or ()), f"{fam}[{','.join(str(x) for x in m['nu'])}]"))
    lines = [x[-1] for x in sorted(set(pref))]
    open(os.path.join(a.out, f'{fam}.imode.preferred'), 'w').write('\n'.join(lines) + '\n')
    json.dump(summ, open(os.path.join(a.out, 'imode_summary.json'), 'w'), indent=1, default=str)
    L(f'[imode] wrote {fam}.imode.preferred ({len(lines)} lines)')
    failed = sorted(sid for sid, d in summ['sectors'].items() if d.get('error') or 'failed' in (d.get('flags') or []))
    if failed:
        L(f"[imode] FAILED: {len(failed)} sector(s) did not complete: {', '.join(str(s) for s in failed)} (error text in imode_summary.json)")
    return len(failed)

if __name__ == '__main__':
    sys.exit(1 if main() else 0)
