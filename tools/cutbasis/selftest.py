#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of BootLoops 1.0; see LICENSE and NOTICE beside this file.
"""cutbasis self-test: known-answer checks, committed planted-defect fixtures, a small part of the worked example, and
(when Kira and Fermat are reachable) the reducer comparison.  Run from the package directory; needs only the declared
Python dependencies (sympy, python-flint, pyyaml) except for the reducer check.

Checks
  paper_controls     23 statements of arXiv:2511.15381 reproduced by script (twists, images, integers, blow-ups)
  doublebox_select   the paper's double-box top sector: H-basis, leading forms and single-integral masters equal the
                     committed expected output (fixtures/expected_doublebox.json)
  tablecheck_defect  two committed reduction tables on rotated bases certify as equal (fixtures/tablecheck/A.txt, B.txt);
                     the committed table with one wrong coefficient (Bbad.txt) is refused with exit 1
  ratrec_defect      a known rational function of three variables is reconstructed from the committed mod-p samples at
                     two primes and verified at a third (fixtures/ratrec/samples.json); the committed sample set with
                     one corrupted value is rejected
  coefficient_format Kira coefficient strings are written without spaces (with Kira 3.1 a run stalled on them)
  evaluate_refusal   evaluate with no Kira executable prints 'REFUSED: ...' and exits 4 before writing anything
  cli_errors         five options the tool cannot act on (a --sectors entry that is not a sector of the family, an
                     unknown --family, a --candidates entry without '=', a candidate name with a path separator,
                     a missing candidate file) each exit 2 with
                     one named 'usage error' line on stderr and write nothing
  moeller_quick      sector 42 of the worked example (examples/moeller_tower, the README's quick command): the three
                     preferred-masters lists equal the committed expected/quick/ files byte for byte
  reducer            (needs Kira and Fermat) evaluate on the double box with two candidate lists: both certify against
                     the stock table, their tables differ, and the larger one is never called best

Exit codes: 0 every check passed (the reducer check may be skipped with a named SKIP line); 1 a check failed;
4 only with CUTBASIS_REQUIRE_REDUCER=1 set and no Kira or Fermat found: the last line is 'REFUSED: <reason>'.
"""
import hashlib, json, os, subprocess, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sympy as sp

results = []
EX = os.path.join(HERE, 'examples', 'moeller_tower')
QUICK_LISTS = ('moeller.echelon.preferred', 'moeller.lc.preferred', 'moeller.integrals.preferred')


class SkipCheck(Exception):
    pass


def check(name, fn):
    t0 = time.time()
    try:
        info = fn()
        results.append((name, 'PASS', info, time.time() - t0)); print(f'PASS {name}  {info}  ({time.time()-t0:.1f}s)', flush=True)
    except SkipCheck as e:
        results.append((name, 'SKIP', str(e), time.time() - t0)); print(f'SKIP {name}  {e}', flush=True)
    except Exception as e:
        results.append((name, 'FAIL', repr(e), time.time() - t0)); print(f'FAIL {name}  {e!r}', flush=True)


def paper_controls():
    r = subprocess.run([sys.executable, '-B', '-m', 'geomorder.tests.test_paper'], cwd=HERE, capture_output=True, text=True)
    last = [l for l in r.stdout.splitlines() if 'PASS,' in l]
    if r.returncode != 0 or not last or ' 0 FAIL' not in last[-1]:
        raise RuntimeError('paper controls: ' + (last[-1] if last else r.stdout[-300:] + r.stderr[-300:]))
    return last[-1]


def doublebox_select():
    from geomorder.lcbasis import SectorLC
    spec = json.load(open(os.path.join(HERE, 'fixtures', 'paper_doublebox.json')))
    sl = SectorLC(spec, [1, 2, 3, 4, 5, 6, 7], seed=5, M=4, cand_dots=1, cand_sps=1)
    r = sl.base(); sl.structure()
    got = {'dimH': r['dimH'], 'dimV': sl.dimV, 'H': sorted((h['form'], h['a'], h['key']) for h in r['H']),
           'masters': [(e['form'], [list(nu) for nu in e['support']]) for e in sl.plan]}
    exp = json.load(open(os.path.join(HERE, 'fixtures', 'expected_doublebox.json')))
    if json.dumps(got, sort_keys=True) != json.dumps(exp, sort_keys=True):
        raise RuntimeError(f'double box: got {got} expected {exp}')
    return f"dimH={got['dimH']} masters={got['masters']}"


def tablecheck_defect():
    fx = os.path.join(HERE, 'fixtures', 'tablecheck')
    def tc(b):
        return subprocess.run([sys.executable, '-B', os.path.join(HERE, 'tablecheck.py'), '--a', os.path.join(fx, 'A.txt'), '--b', os.path.join(fx, b),
                               '--symbols', 'd,t', '--lc-defs', os.path.join(fx, 'pref'), '--no-metrics'], capture_output=True, text=True)
    ok, bad = tc('B.txt'), tc('Bbad.txt')
    if ok.returncode != 0:
        raise RuntimeError('consistent tables not certified: ' + ok.stdout[-300:] + ok.stderr[-300:])
    if bad.returncode != 1:
        raise RuntimeError(f'planted wrong coefficient NOT refused with exit 1 (rc {bad.returncode})')
    j = json.loads(bad.stdout.strip().splitlines()[-1])
    if j.get('consistent') is not False or 'fam[2,1,1,0]' not in j['checks'][0]['bad_examples']:
        raise RuntimeError('planted wrong coefficient refused for the wrong reason: ' + bad.stdout[-300:])
    return 'rotated basis certified; planted wrong coefficient refused (exit 1, rule fam[2,1,1,0] named)'


def ratrec_defect():
    from geomorder import ratrec
    fx = json.load(open(os.path.join(HERE, 'fixtures', 'ratrec', 'samples.json')))
    syms = sp.symbols(fx['symbols'])
    f = sp.sympify(fx['function'])
    dn, dd = fx['degrees']
    fits = []
    for p in fx['primes']:
        s = fx['samples'][str(p)]
        fit = ratrec.fit_dense(s['points'], s['values'], p, dn, dd, len(syms))
        if fit is None:
            raise RuntimeError(f'fit failed at p={p}')
        fits.append((fit, p))
    numQ, denQ = ratrec.combine_primes(fits)
    g = ratrec.to_sympy(numQ, denQ, syms)
    if sp.simplify(g - f) != 0:
        raise RuntimeError(f'reconstructed {g} != {f}')
    v = fx['verify']
    if ratrec.ratfunc_mod(g, syms, v['point'], v['prime']) != v['value']:
        raise RuntimeError('third-prime check failed on a correct function')
    c = fx['corrupted']
    if ratrec.fit_dense(c['points'], c['values'], c['prime'], dn, dd, len(syms)) is not None:
        raise RuntimeError('corrupted sample NOT rejected')
    return f'{f} reconstructed at two primes, verified at a third; corrupted sample rejected'


def coefficient_format():
    from geomorder.lcbasis import kira_coeff, preferred_text
    e = sp.Symbol('eps'); t = sp.Symbol('t')
    s = kira_coeff((3 * e ** 2 * t + 1) / (e - 2), 4)
    txt = preferred_text('fam', [{'masters': [{'form': 'w', 'a': -2, 'key': [1, 1, 1], 'lc': [([1, 1, 1, 0], sp.Integer(1)), ([1, 1, 1, -1], (3 * e ** 2 * t + 1) / (e - 2))]}]}], d0=4)
    if ' ' in s or any(' ' in l for l in txt.splitlines()):
        raise RuntimeError('coefficient string contains spaces: ' + s)
    if '**' in s:
        raise RuntimeError('coefficient string uses ** : ' + s)
    return s


def evaluate_refusal():
    fx = os.path.join(HERE, 'fixtures', 'doublebox_kira')
    env = {k: v for k, v in os.environ.items() if k not in ('KIRA', 'FERMATPATH')}
    env['PATH'] = os.pathsep.join(d for d in env.get('PATH', '').split(os.pathsep) if not os.path.exists(os.path.join(d, 'kira')))
    with tempfile.TemporaryDirectory(prefix='cutbasis_refusal_') as tmp:
        out = os.path.join(tmp, 'eval')
        r = subprocess.run([sys.executable, '-B', os.path.join(HERE, 'cutbasis.py'), 'evaluate', '--config', fx, '--family', 'db', '--jobs', os.path.join(fx, 'jobs_template.yaml'),
                            '--targets', os.path.join(fx, 'targets'), '--out', out, '--candidates', 'echelon=' + os.path.join(fx, 'echelon.preferred'), '--symbols', 'd,t'],
                           capture_output=True, text=True, env=env)
        lines = [l for l in r.stdout.splitlines() if l.strip()]
        if r.returncode != 4 or not lines or not lines[-1].startswith('[evaluate] REFUSED:'):
            raise RuntimeError(f'no-Kira evaluate: rc {r.returncode}, last line {lines[-1] if lines else None!r}; stderr {r.stderr[-200:]!r}')
        if os.path.exists(out):
            raise RuntimeError('no-Kira evaluate created its output directory before refusing')
    return 'exit 4, ' + lines[-1]


def cli_errors():
    """Five committed-fixture invocations with one bad option each: exit 2, one named line on stderr, no output dir."""
    fx = os.path.join(HERE, 'fixtures', 'doublebox_kira')
    inp = os.path.join(EX, 'input')
    with tempfile.TemporaryDirectory(prefix='cutbasis_cli_') as tmp:
        out = os.path.join(tmp, 'out')
        ev = ['evaluate', '--config', fx, '--family', 'db', '--jobs', os.path.join(fx, 'jobs_template.yaml'), '--targets', os.path.join(fx, 'targets'), '--out', out, '--symbols', 'd,t']
        cases = [
            ('sectors', ['select', '--config', inp, '--family', 'moeller', '--top', '107', '--sectors', '1000', '--out', out],
             '[cutbasis] usage error: --sectors 1000: sector 1000 is not one of the 9-propagator family moeller (1 to 511)'),
            ('family', ['census', '--config', inp, '--family', 'nosuch', '--top', '107'],
             f'[cutbasis] usage error: --family nosuch: not in {os.path.join(inp, "integralfamilies.yaml")} (families: moeller)'),
            ('candidates', ev + ['--candidates', 'X'],
             '[cutbasis] usage error: --candidates X: expected name=preferred_masters_file'),
            ('name', ev + ['--candidates', 'a/../../x=' + os.path.join(fx, 'echelon.preferred')],
             '[cutbasis] usage error: --candidates a/../../x=' + os.path.join(fx, 'echelon.preferred') + ': the name may contain only letters, digits, _ . - (it names the run directory run_a/../../x)'),
            ('missing', ev + ['--candidates', 'echelon=' + os.path.join(tmp, 'missing.preferred')],
             f'[cutbasis] usage error: --candidates echelon={os.path.join(tmp, "missing.preferred")}: no such file {os.path.join(tmp, "missing.preferred")}'),
        ]
        for name, argv, want in cases:
            r = subprocess.run([sys.executable, '-B', os.path.join(HERE, 'cutbasis.py')] + argv, capture_output=True, text=True, timeout=120)
            err = r.stderr.strip().splitlines()
            if r.returncode != 2 or len(err) != 1 or err[0] != want or r.stdout.strip():
                raise RuntimeError(f'{name}: rc {r.returncode}, stderr {err!r}, stdout {r.stdout[-200:]!r}; expected rc 2 and the line {want!r}')
            if os.path.exists(out):
                raise RuntimeError(f'{name}: the output directory was created before the refusal')
    return 'bad --sectors, --family, --candidates, a candidate name with a path separator and a missing candidate file: exit 2, one named line each, nothing written'


def _sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def moeller_quick():
    exp = os.path.join(EX, 'expected', 'quick')
    with tempfile.TemporaryDirectory(prefix='cutbasis_quick_') as tmp:
        out = os.path.join(tmp, 'out')
        r = subprocess.run([sys.executable, '-B', os.path.join(HERE, 'cutbasis.py'), 'select', '--config', os.path.join(EX, 'input'), '--family', 'moeller', '--top', '107',
                            '--sectors', '42', '--out', out, '--workers', '4'], capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            raise RuntimeError(f'quick select rc {r.returncode}: ' + r.stdout[-400:] + r.stderr[-400:])
        diffs = [f for f in QUICK_LISTS if _sha(os.path.join(out, f)) != _sha(os.path.join(exp, f))]
    if diffs:
        raise RuntimeError('quick select output differs from expected/quick: ' + ', '.join(diffs))
    return 'sector 42: echelon, lc and integrals lists equal expected/quick byte for byte'


def reducer():
    from evaluate import find_programs
    kira, fermat, reason = find_programs(None, None)
    if reason:
        raise SkipCheck(reason)
    fx = os.path.join(HERE, 'fixtures', 'doublebox_kira')
    with tempfile.TemporaryDirectory(prefix='cutbasis_reducer_') as tmp:
        out = os.path.join(tmp, 'eval')
        cmd = [sys.executable, '-B', os.path.join(HERE, 'cutbasis.py'), 'evaluate', '--config', fx, '--family', 'db', '--jobs', os.path.join(fx, 'jobs_template.yaml'),
               '--targets', os.path.join(fx, 'targets'), '--out', out, '--candidates', 'echelon=' + os.path.join(fx, 'echelon.preferred'), 'alt=' + os.path.join(fx, 'alt.preferred'),
               '--symbols', 'd,t', '--alphabet', 't,t+1', '--kira', kira, '--fermat', fermat, '--parallel', '2']
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        try:
            v = json.load(open(os.path.join(out, 'evaluate.json')))
        except Exception:
            raise RuntimeError(f'evaluate did not finish (rc {r.returncode}): ' + r.stdout[-400:] + r.stderr[-400:])
        if r.returncode != 0:
            raise RuntimeError(f'evaluate rc {r.returncode}: ' + r.stdout[-400:])
        runs = v['runs']
        for n in ('echelon', 'alt'):
            if not runs[n].get('certified'):
                raise RuntimeError(f'double-box {n} table not certified: ' + json.dumps(runs[n])[:300])
        shas = {n: _sha(runs[n]['table']) for n in ('stock', 'echelon', 'alt')}
        if len(set(shas.values())) != 3:
            raise RuntimeError(f'the three tables are not all different: {shas}')
        if runs['alt']['bytes'] <= runs['stock']['bytes']:
            raise RuntimeError(f"alt list ({runs['alt']['bytes']} B) was expected to give a larger table than stock ({runs['stock']['bytes']} B)")
        if v['best_certified_candidate'] == 'alt':
            raise RuntimeError('a candidate larger than stock was called best')
    return f"stock {runs['stock']['bytes']} B, echelon {runs['echelon']['bytes']} B, alt {runs['alt']['bytes']} B; all three tables differ, both candidates certified, best={v['best_certified_candidate']}"


if __name__ == '__main__':
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    for name, fn in [('paper_controls', paper_controls), ('doublebox_select', doublebox_select), ('tablecheck_defect', tablecheck_defect),
                     ('ratrec_defect', ratrec_defect), ('coefficient_format', coefficient_format), ('evaluate_refusal', evaluate_refusal),
                     ('cli_errors', cli_errors), ('moeller_quick', moeller_quick), ('reducer', reducer)]:
        check(name, fn)
    nfail = sum(1 for r in results if r[1] == 'FAIL')
    skipped = [r for r in results if r[1] == 'SKIP']
    npass = sum(1 for r in results if r[1] == 'PASS')
    if nfail:
        print(f'OVERALL FAIL: {npass}/{len(results)} checks pass, {nfail} failed' + (f", {len(skipped)} skipped" if skipped else ''))
        sys.exit(1)
    if skipped and os.environ.get('CUTBASIS_REQUIRE_REDUCER'):
        # the reducer check is required: refuse, reason last (exit 4 = the toolkit's refusal code)
        print(f'{npass}/{len(results)} checks pass; the reducer check was required (CUTBASIS_REQUIRE_REDUCER=1) and could not run')
        print(f'REFUSED: {skipped[0][2]}')
        sys.exit(4)
    print(f"OVERALL PASS: {npass}/{len(results)} checks pass" + (f"; skipped: {', '.join(r[0] + ' (' + r[2] + ')' for r in skipped)}" if skipped else ''))
    sys.exit(0)
