#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of BootLoops 1.0; see LICENSE and NOTICE beside this file.
"""evaluate.py — run Kira on a workload with each candidate preferred-masters list, certify every table against the
stock table (mod p, two points) and report the sizes, degrees and denominator-factor census; the smallest certified
table wins, and the stock table is in the running (a candidate larger than stock is never called best).

usage: evaluate.py --config DIR --family NAME --jobs jobs_template.yaml --targets FILE --out DIR
         --candidates name1=pref1 [name2=pref2 ...] [--symbols d,t,m2] [--alphabet 't,m2,...'] [--kira PATH]
         [--fermat PATH] [--parallel 4]
The jobs template must contain the line '    preferred_masters: PREFERRED' (removed for the stock run) and reduce the
targets file named 'targets' (the union of the workload and every candidate's integrals is written to it).

Exit codes: 0 every run finished and every candidate table certified; 1 a Kira run failed or a candidate table did
not certify (named in the output and in evaluate.json); 2 a usage error (a --candidates entry without '=', a candidate name with characters
other than letters, digits, _ . -, a candidate named 'stock', a missing candidate, jobs, targets or config file, a --family not in the config: one line
'[evaluate] usage error: ...' on stderr, nothing written); 4 no Kira or no Fermat executable found (nothing is written).
"""
from __future__ import annotations
import argparse, json, os, re, shutil, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geomorder.kirayaml import UsageError, USAGE_RC, family_header

HERE = os.path.dirname(os.path.abspath(__file__))
MISSING_PROGRAM_RC = 4


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def integrals_of(pref_path, fam):
    out = set()
    for line in open(pref_path):
        m = re.match(r'^\s*(%s\[[-0-9,\s]+\])' % re.escape(fam), line)
        if m:
            out.add(m.group(1).replace(' ', ''))
    return out


def run_kira(workdir, kira, parallel, log, env_extra=None):
    t0 = time.time()
    env = dict(os.environ); env.update(env_extra or {})
    with open(os.path.join(workdir, 'kira_stdout.log'), 'w') as fh:
        r = subprocess.run([kira, f'--parallel={parallel}', 'jobs.yaml'], cwd=workdir, stdout=fh, stderr=subprocess.STDOUT, env=env)
    wall = time.time() - t0
    log(f'  kira rc={r.returncode} wall={wall:.1f}s in {workdir}')
    return r.returncode, wall


def find_programs(kira, fermat):
    """Resolve the Kira and Fermat executables the way Kira itself looks for them; return (kira, fermat, reason)."""
    k = kira or os.environ.get('KIRA') or shutil.which('kira')
    if k and os.path.isfile(k) and os.access(k, os.X_OK):
        kira_path = os.path.abspath(k)
    elif k and shutil.which(k):
        kira_path = shutil.which(k)
    else:
        return None, None, "no Kira executable found (PATH has no 'kira'; pass --kira PATH or set KIRA=/path/to/kira)"
    f = fermat or os.environ.get('FERMATPATH') or shutil.which('fer64')
    if not (f and os.path.isfile(f) and os.access(f, os.X_OK)):
        return kira_path, None, "no Fermat executable found (Kira needs one; pass --fermat PATH or set FERMATPATH=/path/to/fer64)"
    return kira_path, os.path.abspath(f), None


def add_arguments(ap):
    ap.add_argument('--config', required=True, help='Kira configuration directory (integralfamilies.yaml, kinematics.yaml)')
    ap.add_argument('--family', required=True); ap.add_argument('--jobs', required=True, help='jobs template with the line "preferred_masters: PREFERRED"')
    ap.add_argument('--targets', required=True, help='workload target list'); ap.add_argument('--out', required=True)
    ap.add_argument('--candidates', nargs='+', required=True, help='name=preferred_masters_file, one per candidate')
    ap.add_argument('--symbols', default='d', help='comma-separated symbols of the tables (d and the invariants)')
    ap.add_argument('--alphabet', default='', help='comma-separated expected denominator factors; others are counted as kinematic_not_in_alphabet')
    ap.add_argument('--kira', default=None, help='Kira executable (default: KIRA, then kira on PATH)')
    ap.add_argument('--parallel', type=int, default=4); ap.add_argument('--sectormappings', default=None)
    ap.add_argument('--fermat', default=None, help='Fermat executable (default: FERMATPATH, then fer64 on PATH)')
    ap.add_argument('--preload', default=None, help='optional LD_PRELOAD (e.g. jemalloc) for the Kira runs')
    return ap


METRIC_KEYS = ('bytes', 'rules', 'terms', 'max_deg_num', 'max_deg_den', 'max_coeff_bits')


def check_arguments(a):
    """The candidate dictionary {name: absolute path}; a UsageError (exit 2) names the first option that cannot be acted
    on. Runs before the reducer lookup and before anything is written."""
    family_header(a.config, a.family)
    for opt, path in (('--jobs', a.jobs), ('--targets', a.targets)):
        if not os.path.isfile(path):
            raise UsageError(f'{opt} {path}: no such file')
    cands = {}
    for c in a.candidates:
        if '=' not in c:
            raise UsageError(f'--candidates {c}: expected name=preferred_masters_file')
        name, path = c.split('=', 1)
        if not name or not path:
            raise UsageError(f'--candidates {c}: expected name=preferred_masters_file')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
            raise UsageError(f'--candidates {c}: the name may contain only letters, digits, _ . - (it names the run directory run_{name})')
        if name == 'stock':
            raise UsageError(f"--candidates {c}: 'stock' is the name of the reference run; name the candidate differently")
        if name in cands:
            raise UsageError(f'--candidates {c}: the candidate name {name} is used twice')
        if not os.path.isfile(path):
            raise UsageError(f'--candidates {c}: no such file {path}')
        cands[name] = os.path.abspath(path)
    return cands


def run(a):
    cands = check_arguments(a)
    kira, fermat, reason = find_programs(a.kira, a.fermat)
    if reason:
        print(f'[evaluate] REFUSED: {reason}', flush=True)
        return MISSING_PROGRAM_RC
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, 'evaluate.log'), 'a')
    def log(s):
        logf.write(f'{stamp()} {s}\n'); logf.flush(); print(s, flush=True)
    fam = a.family
    base = [l.strip().replace(' ', '') for l in open(a.targets) if l.strip()]
    union = set(base)
    for path in cands.values():
        union |= integrals_of(path, fam)
    targets = sorted(union)
    log(f'[evaluate] family {fam}: {len(base)} workload targets + candidates -> {len(targets)} targets; runs: stock, {", ".join(cands)}')
    log(f'[evaluate] kira {kira}; fermat {fermat}')
    template = open(a.jobs).read()
    if 'preferred_masters: PREFERRED' not in template:
        log('[evaluate] FAILED: the jobs template needs the line "preferred_masters: PREFERRED"'); return 1
    runs = {'stock': None}; runs.update(cands)
    results = {}
    for name, pref in runs.items():
        D = os.path.join(a.out, f'run_{name}')
        table = os.path.join(D, 'results', fam, 'kira_targets.m')
        if os.path.exists(table):
            log(f'[evaluate] run {name}: table exists, skipping the run')
            results[name] = {'rc': 0, 'wall_s': None, 'reused': True}
        else:
            shutil.rmtree(D, ignore_errors=True); os.makedirs(D)
            shutil.copytree(a.config, os.path.join(D, 'config'))
            if a.sectormappings:
                shutil.copytree(a.sectormappings, os.path.join(D, 'sectormappings'))
            open(os.path.join(D, 'targets'), 'w').write('\n'.join(targets) + '\n')
            jobs = template
            if pref is None:
                jobs = '\n'.join(l for l in jobs.split('\n') if 'preferred_masters: PREFERRED' not in l)
            else:
                shutil.copy(pref, os.path.join(D, 'preferred'))
                jobs = jobs.replace('preferred_masters: PREFERRED', 'preferred_masters: preferred')
            open(os.path.join(D, 'jobs.yaml'), 'w').write(jobs)
            env_extra = {'FERMATPATH': fermat}
            if a.preload: env_extra['LD_PRELOAD'] = a.preload
            rc, wall = run_kira(D, kira, a.parallel, log, env_extra)
            results[name] = {'rc': rc, 'wall_s': wall}
        results[name]['table'] = table
        if results[name]['rc'] != 0 or not os.path.exists(table):
            results[name]['status'] = 'failed'; results[name]['bytes'] = None; results[name]['certified'] = False
            log(f"[evaluate] run {name} FAILED: kira rc={results[name]['rc']}, table {'missing' if not os.path.exists(table) else 'present'} (see {os.path.join(D, 'kira_stdout.log')})")
        else:
            results[name]['status'] = 'ran'; results[name]['bytes'] = os.path.getsize(table)
    if results['stock']['status'] != 'ran':
        log('[evaluate] FAILED: the stock run did not produce a table; nothing can be certified')
        _write(a, fam, results, None, log); return 1
    stock_table = results['stock']['table']
    for name in cands:
        if results[name]['status'] != 'ran':
            continue
        rec = os.path.join(a.out, f'tablecheck_stock_vs_{name}.json')
        cmd = [sys.executable, os.path.join(HERE, 'tablecheck.py'), '--a', stock_table, '--b', results[name]['table'], '--symbols', a.symbols, '--alphabet', a.alphabet, '--json', rec, '--lc-defs', cands[name]]
        r = subprocess.run(cmd, capture_output=True, text=True)
        try:
            j = json.load(open(rec))
            results[name]['certified'] = bool(j['consistent']); results[name]['metrics'] = {k: j['metrics_b'][k] for k in METRIC_KEYS}; results[name]['factor_classes'] = j['metrics_b']['factor_classes']
            results['stock']['metrics'] = {k: j['metrics_a'][k] for k in METRIC_KEYS}; results['stock']['factor_classes'] = j['metrics_a']['factor_classes']
        except Exception as e:
            results[name]['certified'] = False; results[name]['error'] = repr(e) + r.stdout[-500:] + r.stderr[-500:]
        log(f"[evaluate] run {name}: certified={results[name].get('certified')} bytes={results[name]['bytes']} metrics={results[name].get('metrics')} factors={results[name].get('factor_classes')}")
    # the smallest certified table; stock is in the running, and a tie goes to stock
    ok = ['stock'] + [n for n in cands if results[n].get('certified') and results[n].get('bytes')]
    best = min(ok, key=lambda n: results[n]['bytes'])
    _write(a, fam, results, best, log)
    bad = [n for n in runs if results[n]['status'] != 'ran' or (n != 'stock' and not results[n].get('certified'))]
    if bad:
        log(f"[evaluate] FAILED: {', '.join(bad)} did not run or did not certify; best certified candidate: {best if best != 'stock' else 'none (stock is the smallest certified table)'}")
        return 1
    if best == 'stock':
        log(f"[evaluate] best certified candidate: none; the stock table ({results['stock']['bytes']} B) is the smallest certified table")
    else:
        log(f"[evaluate] best certified candidate: {best} (stock {results['stock']['bytes']} B -> {results[best]['bytes']} B)")
    return 0


def _write(a, fam, results, best, log):
    verdict = {'stamp': stamp(), 'family': fam, 'runs': results, 'best_certified_candidate': best, 'stock_bytes': results['stock'].get('bytes'),
               'gain': (results['stock']['bytes'] / results[best]['bytes']) if best and results[best].get('bytes') else None}
    json.dump(verdict, open(os.path.join(a.out, 'evaluate.json'), 'w'), indent=1, default=str)
    # the comparable part (no stamps, walls or paths): what an example's expected/ directory carries
    summary = {'family': fam, 'runs': {}, 'best_certified_candidate': best}
    for n, r in results.items():
        summary['runs'][n] = {'status': r.get('status'), 'bytes': r.get('bytes'), 'certified': (None if n == 'stock' else r.get('certified')),
                              'metrics': r.get('metrics'), 'factor_classes': r.get('factor_classes')}
    with open(os.path.join(a.out, 'evaluate_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1, sort_keys=True); fh.write('\n')
    log(f"[evaluate] wrote evaluate.json and evaluate_summary.json under {a.out}")


def main(argv=None):
    a = add_arguments(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)).parse_args(argv)
    try:
        return run(a)
    except UsageError as e:
        print(f'[evaluate] usage error: {e}', file=sys.stderr, flush=True)
        return USAGE_RC


if __name__ == '__main__':
    sys.exit(main())
