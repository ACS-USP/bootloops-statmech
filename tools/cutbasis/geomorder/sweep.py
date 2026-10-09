# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of BootLoops 1.0; see LICENSE and NOTICE beside this file.
"""Sector sweep -> Kira rank files.

For every non-trivial sector of a family (inside the Kira seed box r<=R, s<=S):
  * representation (loop-by-loop if good, else democratic);
  * ANALYZED sectors (loop-by-loop with n<=1; n=0 trivially): Algorithm 1 masters with their 'a'
    (geomorder.masters), and the key (a, w, o, |mu|) for EVERY integral in the box (w,o,|mu| from the iota-hat image at
    generic eps, the convention of fastiota.FastRep.iota; a = master's a if the integral is a master, else 0);
  * OTHER sectors (n>=2 or democratic): keys (w,o,|mu|) are generic-position estimates and 'a' is not available; two
    policies are written: rank file G ("faithful-where-available": no rank lines for these sectors -> Kira's stock
    order inside them) and rank file H ("heuristic everywhere": a_heur = -w if r == n else 0, for integrals with
    dots <= --dcap-other; deeper integrals default to the maximal rank).
Two numeric kinematic seeds are cross-checked for analyzed sectors (disagreement -> flag, max key).
Outputs in --out: <fam>.ranks.G, <fam>.ranks.H, <fam>.preferred (masters of analyzed sectors, Kira syntax),
<fam>.sweep.json (everything, incl. per-sector master analysis), ckpt/ (resumable).
"""
from __future__ import annotations
import argparse, itertools, json, os, time, traceback, signal
from multiprocessing import Pool
import sympy as sp

from .kirayaml import read_kira_family, sector_props, sector_id
from .baikov import Family, numeric_spec, representation, structure
from .fastiota import FastRep
from .integers import integers

EPS_VALUES = (sp.Rational(7, 13), sp.Rational(11, 17))
_CACHE = {}
_G = {}


def enumerate_integrals(nprops, sector, rmax, smax, dmax=None):
    t = len(sector)
    dots_max = rmax - t if dmax is None else min(rmax - t, dmax)
    if dots_max < 0:
        return []
    isps = [i for i in range(1, nprops + 1) if i not in sector]
    def comps(total_max, k):
        if k == 0:
            yield (); return
        for tot in range(total_max + 1):
            for c in itertools.combinations_with_replacement(range(k), tot):
                v = [0] * k
                for i in c: v[i] += 1
                yield tuple(v)
    dot_list = list(dict.fromkeys(comps(dots_max, t)))
    sp_list = list(dict.fromkeys(comps(smax, len(isps))))
    out = []
    for dv in dot_list:
        for sv in sp_list:
            nu = [0] * nprops
            for i, d in zip(sector, dv): nu[i - 1] = 1 + d
            for j, e in zip(isps, sv): nu[j - 1] = -e
            out.append(tuple(nu))
    return out


def _init(spec):
    _G['spec'] = spec


def _get(sector_t, seed):
    key = (sector_t, seed)
    if key not in _CACHE:
        nspec = numeric_spec(_G['spec'], seed=seed)
        F = Family(nspec)
        rep, note = representation(F, list(sector_t))
        fr = FastRep(rep)
        _CACHE[key] = (rep, fr, note)
    return _CACHE[key]


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


PER_INTEGRAL_TIMEOUT = int(os.environ.get('GEOMORDER_INT_TIMEOUT', '20'))


def _work(args):
    sector_t, seed, nus = args
    t0 = time.time()
    signal.signal(signal.SIGALRM, _alarm)
    try:
        rep, fr, note = _get(sector_t, seed)
    except Exception as e:
        return {'sector': sector_t, 'seed': seed, 'error': 'rep: ' + repr(e) + traceback.format_exc()[-800:], 'keys': {}, 'flags': {}, 'nerr': 0, 'wall': time.time() - t0}
    keys, flags, nerr = {}, {}, 0
    ntimeout = 0
    for nu in nus:
        try:
            ks = []
            signal.alarm(PER_INTEGRAL_TIMEOUT)
            for ev in EPS_VALUES:
                T = fr.iota(nu, eps_value=ev)
                if not T:
                    ks.append((0, 0, 0, 0)); continue
                I = [integers(rep, t, nsamples=1) for t in T]
                best = max(I, key=lambda i: (i.w, i.o, i.absmu))
                k = (best.w, best.o, best.absmu, best.r)
                fl = [f for i in I for f in i.flags]
                if fl:
                    flags[str(list(nu))] = sorted(set(fl))
                ks.append(k)
            if len(set(ks)) > 1:
                flags.setdefault(str(list(nu)), []).append(f'eps-dependence: {ks}')
            keys[str(list(nu))] = max(ks)
            signal.alarm(0)
        except _Timeout:
            nerr += 1; ntimeout += 1
            flags[str(list(nu))] = [f'TIMEOUT>{PER_INTEGRAL_TIMEOUT}s (default rank)']
        except Exception as e:
            signal.alarm(0)
            nerr += 1
            flags[str(list(nu))] = ['ERROR ' + repr(e)[:200]]
    signal.alarm(0)
    return {'sector': sector_t, 'seed': seed, 'note': note, 'kind': rep.kind, 'n': rep.n, 'struct': str(structure(rep)),
            'keys': keys, 'flags': flags, 'nerr': nerr, 'ntimeout': ntimeout, 'wall': time.time() - t0, 'count': len(nus)}


def _masters_work(args):
    spec, sector, seed, M = args
    from .masters import sector_masters
    try:
        return sector_masters(spec, list(sector), seed=seed, M=M)
    except Exception as e:
        return {'sector': sector_id(list(sector)), 'props': list(sector), 'error': repr(e) + traceback.format_exc()[-800:], 'masters': [], 'flags': ['masters analysis failed']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--family', default=None)
    ap.add_argument('--top', type=int, required=True)
    ap.add_argument('--r', type=int, required=True)
    ap.add_argument('--s', type=int, required=True)
    ap.add_argument('--sectormappings', default=None)
    ap.add_argument('--sectors', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seeds', default='5,7')
    ap.add_argument('--chunk', type=int, default=300)
    ap.add_argument('--d0', type=int, default=4)
    ap.add_argument('--dcap-other', type=int, default=1, help='dots cap for non-analyzed sectors (heuristic file)')
    ap.add_argument('--dcap-analyzed', type=int, default=None, help='dots cap for analyzed sectors (default: full seed box)')
    ap.add_argument('--M', type=int, default=5, help='|mu| truncation of the maxcut system')
    ap.add_argument('--nmax-analyzed', type=int, default=1)
    ap.add_argument('--external', default=None, help='JSON {sector_id: {"amap": {"fam[...]": a}, "preferred_block": "verbatim preferred-masters text"}} for sectors whose masters/a come from elsewhere (e.g. the paper); keys are computed exactly if n<=2')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ck = os.path.join(args.out, 'ckpt'); os.makedirs(ck, exist_ok=True)
    spec = read_kira_family(args.config, args.family, d0=args.d0)
    fam = spec['name']
    F = Family(spec)
    N = F.nprops
    seeds = [int(x) for x in args.seeds.split(',')]
    if args.sectors:
        sectors = [int(x) for x in args.sectors.split(',')]
    elif args.sectormappings:
        path = os.path.join(args.sectormappings, fam, 'nonTrivialSector')
        nt = [int(l.split()[0]) for l in open(path) if l.strip()]
        sectors = sorted(s for s in nt if s | args.top == args.top)
    else:
        sectors = [s for s in range(1, args.top + 1) if s | args.top == args.top]
    t0 = time.time()
    external = json.load(open(args.external)) if args.external else {}
    external = {int(k): v for k, v in external.items()}
    # --- phase A: representation census (symbolic) to classify sectors
    cls = {}
    for sid in sectors:
        S = sector_props(sid, N)
        try:
            rep, note = representation(F, S)
            analyzed = (rep.kind == 'loop_by_loop' and rep.n <= args.nmax_analyzed) or rep.n == 0 or (sid in external and rep.n <= 2)
            cls[sid] = {'props': S, 'kind': rep.kind, 'n': rep.n, 'note': note, 'analyzed': analyzed, 'external': sid in external}
        except Exception as e:
            cls[sid] = {'props': S, 'kind': 'FAIL', 'n': None, 'note': repr(e), 'analyzed': False}
    nA = sum(1 for v in cls.values() if v['analyzed'])
    print(f'[sweep] family={fam} nprops={N} top={args.top} r<={args.r} s<={args.s}: {len(sectors)} sectors, {nA} analyzed (n<={args.nmax_analyzed} loop-by-loop or n=0), {len(sectors)-nA} other; {time.time()-t0:.0f}s', flush=True)
    # --- phase B: masters of analyzed sectors (one seed) in parallel
    for sid in sectors:
        if cls[sid].get('external'):
            ext = external[sid]
            res = {'sector': sid, 'props': cls[sid]['props'], 'external': True, 'masters': [{'nu': [int(x) for x in k[k.index('[')+1:k.index(']')].split(',')], 'a': int(a), 'key': None, 'form': k} for k, a in ext.get('amap', {}).items()], 'preferred_block': ext.get('preferred_block', ''), 'flags': ['masters/a supplied externally']}
            f = os.path.join(ck, f'masters{sid}.json')
            with open(f + '.tmp', 'w') as fh: json.dump(res, fh)
            os.replace(f + '.tmp', f)
    mtasks = [(spec, tuple(cls[sid]['props']), seeds[0], args.M) for sid in sectors if cls[sid]['analyzed'] and not cls[sid].get('external') and not os.path.exists(os.path.join(ck, f'masters{sid}.json'))]
    if mtasks:
        with Pool(min(args.workers, len(mtasks))) as pool:
            for res in pool.imap_unordered(_masters_work, mtasks):
                sid = res['sector']
                f = os.path.join(ck, f'masters{sid}.json')
                with open(f + '.tmp', 'w') as fh: json.dump(res, fh, default=str)
                os.replace(f + '.tmp', f)
                print(f"[sweep] masters sector {sid}: {[m.get('nu') or m.get('form') for m in res.get('masters', [])]} a={[m['a'] for m in res.get('masters', [])]} flags={len(res.get('flags', []))}", flush=True)
    masters = {sid: json.load(open(os.path.join(ck, f'masters{sid}.json'))) for sid in sectors if cls[sid]['analyzed']}
    # --- phase C: keys
    tasks = []
    todo = {}
    for sid in sectors:
        S = tuple(cls[sid]['props'])
        if cls[sid]['kind'] == 'FAIL':
            continue
        dmax = args.dcap_analyzed if cls[sid]['analyzed'] else args.dcap_other
        nus = enumerate_integrals(N, list(S), args.r, args.s, dmax)
        sds = seeds if cls[sid]['analyzed'] else seeds[:1]
        todo[sid] = (len(nus), sds)
        for seed in sds:
            if os.path.exists(os.path.join(ck, f'sec{sid}_seed{seed}.json')):
                continue
            for i in range(0, len(nus), args.chunk):
                tasks.append((S, seed, nus[i:i + args.chunk]))
    print(f'[sweep] key tasks: {len(tasks)} ({sum(v[0]*len(v[1]) for v in todo.values())} integral evaluations)', flush=True)
    partial = {}
    if tasks:
        with Pool(args.workers, initializer=_init, initargs=(spec,)) as pool:
            done = 0
            for res in pool.imap_unordered(_work, tasks):
                done += 1
                sid = sector_id(list(res['sector'])); seed = res['seed']
                P = partial.setdefault((sid, seed), {'sector': sid, 'props': list(res['sector']), 'seed': seed, 'keys': {}, 'flags': {}, 'nerr': 0, 'wall': 0.0, 'note': res.get('note'), 'kind': res.get('kind'), 'n': res.get('n'), 'struct': res.get('struct'), 'error': res.get('error')})
                P['keys'].update(res['keys']); P['flags'].update(res.get('flags', {})); P['nerr'] += res.get('nerr', 0); P['wall'] += res.get('wall', 0.0)
                if res.get('error'): P['error'] = res['error']
                if len(P['keys']) + P['nerr'] >= todo[sid][0] or P.get('error'):
                    f = os.path.join(ck, f'sec{sid}_seed{seed}.json')
                    with open(f + '.tmp', 'w') as fh: json.dump(P, fh)
                    os.replace(f + '.tmp', f)
                if done % 25 == 0 or done == len(tasks):
                    print(f'[sweep] {done}/{len(tasks)} key tasks, {time.time()-t0:.0f}s', flush=True)
    # --- phase D: merge and write
    side = {'family': fam, 'top': args.top, 'r': args.r, 's': args.s, 'seeds': seeds, 'eps_values': [str(e) for e in EPS_VALUES], 'M': args.M, 'dcap_other': args.dcap_other, 'sectors': {}}
    linesG, linesH, pref, pref_blocks = [], [], [], []
    maxG = maxH = 0
    for sid in sectors:
        c = cls[sid]
        if c['kind'] == 'FAIL':
            side['sectors'][sid] = dict(c, error='representation failed'); continue
        per = [json.load(open(os.path.join(ck, f'sec{sid}_seed{seed}.json'))) for seed in todo[sid][1] if os.path.exists(os.path.join(ck, f'sec{sid}_seed{seed}.json'))]
        if not per:
            side['sectors'][sid] = dict(c, error='no key checkpoints'); continue
        merged, disagreements = {}, 0
        for nu_s in per[0]['keys']:
            ks = [tuple(p_['keys'][nu_s]) for p_ in per if nu_s in p_['keys']]
            if len(set(ks)) > 1: disagreements += 1
            merged[nu_s] = max(ks)
        flags_union = {}
        for p_ in per:
            for k, v in p_['flags'].items(): flags_union.setdefault(k, []).extend(v)
        n_ = c['n']
        amap = {}
        minfo = masters.get(sid)
        if c['analyzed'] and minfo:
            for m in minfo.get('masters', []):
                if m.get('nu') is not None:
                    amap[str(list(m['nu']))] = int(m['a'])
                    if not minfo.get('external'):
                        pref.append((len(c['props']), sid, int(m['a']), tuple(m.get('key') or ()), f"{fam}[{','.join(str(x) for x in m['nu'])}]"))
            if minfo.get('external') and minfo.get('preferred_block'):
                pref_blocks.append((len(c['props']), sid, minfo['preferred_block']))
        # keys: G (analyzed only) and H
        keyG, keyH = {}, {}
        for nu_s, (w, o, am, r) in merged.items():
            a = amap.get(nu_s, 0)
            if c['analyzed']:
                keyG[nu_s] = (a, w, o, am)
                keyH[nu_s] = (a, w, o, am)
            else:
                ah = -w if (n_ is not None and r == n_) else 0
                keyH[nu_s] = (ah, w, o, am)
        for keyset, lines, which in ((keyG, linesG, 'G'), (keyH, linesH, 'H')):
            uniq = sorted(set(keyset.values()))
            rank_of = {k: i for i, k in enumerate(uniq)}
            if uniq:
                if which == 'G': maxG = max(maxG, len(uniq) - 1)
                else: maxH = max(maxH, len(uniq) - 1)
            for nu_s, k in keyset.items():
                nu = json.loads(nu_s)
                lines.append(f"{fam}[{','.join(str(x) for x in nu)}] {rank_of[k]}")
        side['sectors'][sid] = dict(c, n_integrals=len(merged), seed_disagreements=disagreements, n_flagged=len(flags_union),
                                    distinct_keys_G=len(set(keyG.values())), distinct_keys_H=len(set(keyH.values())),
                                    masters=minfo, keys={k: list(v) for k, v in merged.items()}, amap=amap,
                                    errors=[p_.get('error') for p_ in per if p_.get('error')], nerr=sum(p_.get('nerr',0) for p_ in per), wall_s=sum(p_['wall'] for p_ in per))
    def wr(path, header, lines):
        with open(path + '.tmp', 'w') as fh:
            fh.write(header + '\n' + '\n'.join(lines) + ('\n' if lines else ''))
        os.replace(path + '.tmp', path)
    stamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    wr(os.path.join(args.out, f'{fam}.ranks.G'), f'# geomorder rank file G (faithful where available: analyzed sectors only) family {fam} top {args.top} r<={args.r} s<={args.s} {stamp}', linesG)
    wr(os.path.join(args.out, f'{fam}.ranks.H'), f'# geomorder rank file H (G + heuristic a=-w for r=n elsewhere, dots<={args.dcap_other}) family {fam} {stamp}', linesH)
    pref_sorted = [x[-1] for x in sorted(set(pref))]
    ptxt = '\n'.join(pref_sorted) + '\n'
    for lines_, sid, block in sorted(pref_blocks):
        ptxt += '\n' + block.strip('\n') + '\n'
    with open(os.path.join(args.out, f'{fam}.preferred') + '.tmp', 'w') as fh: fh.write(ptxt)
    os.replace(os.path.join(args.out, f'{fam}.preferred') + '.tmp', os.path.join(args.out, f'{fam}.preferred'))
    side['stamp'] = stamp; side['n_lines_G'] = len(linesG); side['n_lines_H'] = len(linesH); side['max_rank_G'] = maxG; side['max_rank_H'] = maxH
    side['wall_total_s'] = time.time() - t0
    sf = os.path.join(args.out, f'{fam}.sweep.json')
    with open(sf + '.tmp', 'w') as fh: json.dump(side, fh, default=str)
    os.replace(sf + '.tmp', sf)
    print(f'[sweep] wrote {len(linesG)} G lines (max rank {maxG}), {len(linesH)} H lines (max rank {maxH}), {len(set(pref))}+{len(pref_blocks)} preferred entries; {time.time()-t0:.0f}s total')
    for sid in sectors:
        d = side['sectors'][sid]
        print(f"  sector {sid:4d} {str(d['props']):28s} {d['kind']:12s} n={d['n']} {'ANALYZED' if d.get('analyzed') else 'other   '} ints={d.get('n_integrals',0):6d} keysG={d.get('distinct_keys_G',0):3d} keysH={d.get('distinct_keys_H',0):3d} masters={[m.get('nu') or m.get('form') for m in (d.get('masters') or {}).get('masters', [])] if d.get('masters') else '-'} disagree={d.get('seed_disagreements',0)} nerr={d.get('nerr',0)} err={len(d.get('errors',[]))}")


if __name__ == '__main__':
    main()
