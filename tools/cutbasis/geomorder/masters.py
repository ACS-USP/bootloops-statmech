# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of BootLoops 1.0; see LICENSE and NOTICE beside this file.
"""Per-sector analysis: representation, maxcut cohomology system with Feynman-integral columns, Algorithm 1 (localize.py
for n = 1, localize2.py for n = 2), masters with their 'a' values.  The output feeds the sweep (rank key = (a, w, o, |mu|)) and the Kira
preferred_masters file."""
from __future__ import annotations
import time, signal
import sympy as sp
from .baikov import Family, numeric_spec, representation
from .fastiota import FastRep
from .cohomology import masters_on_cut, fmt
from .localize import algorithm1_n1
from .localize2 import algorithm1_n2
from .sweep import enumerate_integrals
from .kirayaml import sector_id


def kira_tiebreak(nu, sector):
    dots = sum(nu[i - 1] - 1 for i in sector)
    sps = sum(-x for j, x in enumerate(nu, start=1) if j not in sector)
    return (dots, sps, tuple(-x if (j + 1) not in sector else 0 for j, x in enumerate(nu)), tuple(nu))


def sector_masters(spec, sector, seed=5, M=5, cand_dots=2, cand_sps=2, eps_value=sp.Rational(7, 13), verbose=False, integrals_only=True):
    """Returns dict with: rep summary, masters = [(nu tuple or form string, a, (w,o,|mu|))], flags."""
    t0 = time.time()
    sector = sorted(sector)
    nspec = numeric_spec(spec, seed=seed)
    F = Family(nspec)
    rep, note = representation(F, sector)
    out = {'sector': sector_id(sector), 'props': sector, 'rep': note, 'kind': rep.kind, 'n': rep.n, 'flags': [], 'masters': []}
    N = F.nprops
    if rep.n == 0:
        nu = tuple(1 if i in sector else 0 for i in range(1, N + 1))
        out['masters'] = [{'nu': list(nu), 'a': 0, 'key': [0, 0, 1], 'form': 'omega_[1][1]'}]
        out['dimH'] = 1
        out['wall_s'] = time.time() - t0
        return out
    if rep.n >= 3 or rep.kind != 'loop_by_loop':
        out['flags'].append(f'n={rep.n} ({rep.kind}): localization algorithm implemented for loop-by-loop n<=2 only -> no geometric masters for this sector')
        out['wall_s'] = time.time() - t0
        return out
    fr = FastRep(rep)
    ms, info, cs = masters_on_cut(rep, M=M, e0=eps_value, verbose=verbose, seed=seed)
    out['pure_form_masters'] = [fmt(rep, m) for m, k in ms]
    out['system'] = info
    # integral columns
    cands = enumerate_integrals(N, sector, len(sector) + cand_dots, cand_sps)
    nadd = 0
    def _alarm(signum, frame):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, _alarm)
    for nu in cands:
        try:
            signal.alarm(20)
            terms = fr.iota(nu, eps_value=eps_value)
            signal.alarm(0)
        except Exception as e:
            signal.alarm(0)
            out['flags'].append(f'iota failed for {nu}: {e!r}'); continue
        if not terms:
            continue
        if max(sum(t.mu) for t in terms) > M:
            continue   # image outside the truncated system
        cs.add_integral_column(nu, terms, kira_tiebreak(nu, sector))
        nadd += 1
    out['n_integral_columns'] = nadd
    # faithful form-level result (paper's H basis) for the record, then the integrals-preferred variant for Kira
    if rep.n <= 1:
        res_forms, _, report_forms = algorithm1_n1(cs, verbose=False, integrals_only=False)
        out['H_basis_forms'] = [(fmt(rep, c), a, [k[1], k[2], k[3]]) for c, k, a in res_forms]
        res, a_final, report = algorithm1_n1(cs, verbose=verbose, integrals_only=integrals_only)
        out['point_report'] = [(d, f, n_, [fmt(rep, c) for c in ch]) for d, f, n_, ch in report]
    else:
        res_forms, _, report_forms = algorithm1_n2(cs, seed=seed, verbose=False, integrals_only=False)
        out['H_basis_forms'] = [(fmt(rep, c), a, [k[1], k[2], k[3]]) for c, k, a in res_forms]
        res, a_final, report = algorithm1_n2(cs, seed=seed, verbose=verbose, integrals_only=integrals_only)
        out['point_report'] = report
    masters = []
    for col, key, a in res:
        entry = {'a': a, 'key': [key[1], key[2], key[3]], 'form': fmt(rep, col)}
        if col[0] == 'I':
            entry['nu'] = list(col[1])
        else:
            entry['nu'] = None
            out['flags'].append(f'form master {fmt(rep, col)} remains (integrals do not span its class: dim V < dim H, or candidates dots<={cand_dots}, sps<={cand_sps} too small)')
        masters.append(entry)
    out['masters'] = masters
    out['dimH'] = len(masters)
    out['wall_s'] = time.time() - t0
    return out


if __name__ == '__main__':
    import argparse, json
    from .kirayaml import read_kira_family, sector_props
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True); ap.add_argument('--family', default=None)
    ap.add_argument('--sector', type=int, required=True); ap.add_argument('--M', type=int, default=5)
    ap.add_argument('--seed', type=int, default=5); ap.add_argument('-v', action='store_true')
    a = ap.parse_args()
    spec = read_kira_family(a.config, a.family)
    N = len(spec['props'])
    r = sector_masters(spec, sector_props(a.sector, N), seed=a.seed, M=a.M, verbose=a.v)
    print(json.dumps(r, indent=1, default=str))
