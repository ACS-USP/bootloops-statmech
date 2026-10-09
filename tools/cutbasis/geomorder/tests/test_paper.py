# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of BootLoops 1.0; see LICENSE and NOTICE beside this file.
"""Positive controls against arXiv:2511.15381 (run: python3 -m geomorder.tests.test_paper from the package directory).
Each check cites the paper statement it reproduces; "L<number>" is a line number of the arXiv v2 TeX source
(eps_factor_rev2.tex), as are the "line" ranges and the "TeX label" names in the geomorder docstrings."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sympy as sp
from geomorder.baikov import load_family, loop_by_loop
from geomorder.iota import iota, make_term, MissingISP
from geomorder.integers import integers, P2, _local_order

HOME = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FAM = os.path.join(HOME, 'fixtures')
results = []

def check(name, cond, info=''):
    results.append((name, bool(cond), info))
    print(('PASS ' if cond else 'FAIL ') + name + ('  | ' + str(info) if info else ''))

def exps(rep):
    """set of (parity, b, deg) of divisors and (a0,b0)"""
    return sorted((d.parity, d.b, d.deg) for d in rep.divisors), (rep.a0, rep.b0)

t0 = time.time()
# ---------------- double box (sect 5.2, lines 3817-4020 of the arXiv v2 TeX source): U = z0^{2eps} z1^{-2eps} (z1-z0)^{-eps} (z1-(x+1) z0)^{eps}
fam = load_family(os.path.join(FAM, 'paper_doublebox.json'))
rep = loop_by_loop(fam, [1, 2, 3, 4, 5, 6, 7], loop_order=['k2', 'k1'])
E, (a0, b0) = exps(rep)
check('doublebox: n=1, good', rep.n == 1 and rep.good, rep.n)
check('doublebox: twist exponents {z1:-2eps,(z1-z0):-eps,(z1-(1+x)z0):+eps}, z0^{2eps} (L3900-3912)',
      E == sorted([('even', -4, 1), ('even', -2, 1), ('even', 2, 1)]) and (a0, b0) == (0, 4), (E, a0, b0))
z0, z1 = rep.z0, rep.zsyms[0]
names = {sp.expand(d.P_hom): d.name for d in rep.divisors}
s, t = sp.symbols('s t')
Pz1 = names[z1]; Pz1mz0 = names[sp.expand(z1 - z0)]
idx = {d.name: i + 1 for i, d in enumerate(rep.divisors)}
T = iota(rep, [1, 1, 1, 1, 1, 1, 1, 0, 0])
mu = T[0].mu
check('doublebox: iota(I_111111100) = omega_0110[1] i.e. 1/(z1 (z1-z0)) (L3979)', len(T) == 1 and mu[idx[Pz1]] == 1 and mu[idx[Pz1mz0]] == 1 and sum(mu) == 2 and T[0].Q_hom == 1, str(T[0]))
T = iota(rep, [1, 1, 1, 1, 1, 1, 1, -1, 0])
mu = T[0].mu
check('doublebox: iota(I_1111111(-1)0) = omega_1010[1] i.e. 1/(z0 (z1-z0)) (L3981)', len(T) == 1 and mu[0] == 1 and mu[idx[Pz1mz0]] == 1 and sum(mu) == 2, str(T[0]))
try:
    iota(rep, [1, 1, 1, 1, 1, 1, 1, 0, -1], strict_missing=True); check('doublebox: sigma9 is a missing ISP', False)
except MissingISP:
    check('doublebox: sigma9 is a missing ISP (loop-by-loop, L795-806)', True)
for mu_, Q in [((0, 1, 0, 1), 1), ((1, 1, 0, 0), 1)]:
    I = integers(rep, make_term(rep, mu_, Q))
    check(f'doublebox: masters have (r,o,|mu|)=(1,1,2), w=2 [Hodge diagram h^(1,1)... L3990-4015] for mu={mu_}', (I.r, I.o, I.absmu, I.w) == (1, 1, 2, 2), (I.r, I.o, I.absmu))

# ---------------- Moller sector 93 (sect 5.4, L4163-4680)
fam = load_family(os.path.join(FAM, 'paper_moeller.json'))
rep = loop_by_loop(fam, [1, 3, 4, 5, 7], loop_order=['k2', 'k1'])
E, (a0, b0) = exps(rep)
check('moeller93: n=2, good', rep.n == 2 and rep.good)
check('moeller93: U = P0^{2eps} P1^{2eps} P2^{-eps} P3^{-1/2-eps} (L4225-4245)',
      E == sorted([('even', 4, 1), ('even', -2, 2), ('odd', -2, 2)]) and (a0, b0) == (0, 4), (E, a0, b0))
z0, (z1, z2) = rep.z0, rep.zsyms
m2 = sp.Symbol('m2')
# identify divisors: Plin (paper P1), Pconic (paper P2, even deg 2), Podd (paper P3)
Plin = next(d.name for d in rep.divisors if d.deg == 1)
Pcon = next(d.name for d in rep.divisors if d.deg == 2 and d.parity == 'even')
Podd = next(d.name for d in rep.divisors if d.parity == 'odd')
idx = {d.name: i + 1 for i, d in enumerate(rep.divisors)}
def MU(mu0, lin, con, odd):
    m = [0] * (len(rep.divisors) + 1); m[0] = mu0; m[idx[Plin]] = lin; m[idx[Pcon]] = con; m[idx[Podd]] = odd; return tuple(m)
T = iota(rep, [1, 0, 1, 1, 1, 0, 1, 0, 0]); check('moeller93: iota(I_101110100) = omega_1100[1] (L4650)', len(T) == 1 and T[0].mu == MU(1, 1, 0, 0) and T[0].Q_hom == 1, str(T[0]))
T = iota(rep, [1, 0, 1, 2, 1, 0, 1, 0, 0]); check('moeller93: iota(I_101210100) = omega_0010[1] (L4664)', len(T) == 1 and T[0].mu == MU(0, 0, 1, 0) and T[0].Q_hom == 1, str(T[0]))
Ta = iota(rep, [1, -1, 1, 2, 1, 0, 1, 0, 0]); Tb = iota(rep, [1, 0, 1, 2, 1, 0, 1, -1, 0])
check('moeller93: I_101210100[z2] pieces map to omega_1010[z_i] (L4652,L4656-4659)', len(Ta) == 1 and len(Tb) == 1 and Ta[0].mu == MU(1, 0, 1, 0) and Tb[0].mu == MU(1, 0, 1, 0), (str(Ta[0]), str(Tb[0])))
T = iota(rep, [1, 0, 1, 1, 2, 0, 1, 0, 0]); check('moeller93: iota(I_101120100) single term in omega_1110[.] (combines with I_101210100[z2] to omega_0110[z2], L4654)', len(T) == 1 and T[0].mu == MU(1, 1, 1, 0), str(T[0]))
z2p = z1 + z2 + (m2 / s - t / s) * z0   # paper's z2 in our coordinates
hodge = {  # name: (mu, Q, expected (r,o))  — basis of H^2_omega L4600-4612 and Hodge diagram L4560-4598
    'omega_0001[1]': (MU(0, 0, 0, 1), 1, (0, 1)),
    'omega_0010[1]': (MU(0, 0, 1, 0), 1, (1, 1)),
    'omega_0011[z0^2]': (MU(0, 0, 1, 1), z0 ** 2, (1, 2)),
    'omega_1100[1]': (MU(1, 1, 0, 0), 1, (2, 2)),
    'omega_1010[z2]': (MU(1, 0, 1, 0), z2p, (2, 2)),
    'omega_0110[z0]': (MU(0, 1, 1, 0), z0, (2, 2)),
}
for name, (mu_, Q, exp) in hodge.items():
    I = integers(rep, make_term(rep, mu_, Q))
    check(f'moeller93: {name} has (r,o)={exp} (Hodge diagram L4560-4598; a=-4,r=2,o=2 statement L4318)', (I.r, I.o) == exp and not I.flags, (I.r, I.o, I.flags))

# ---------------- electron self-energy 3-loop (sect 5.5, L4683+; twist of example_blow_up L1724-1735)
fam = load_family(os.path.join(FAM, 'paper_eselfenergy.json'))
rep = loop_by_loop(fam, [1, 2, 3, 4], loop_order=['k3', 'k2', 'k1'])
E, (a0, b0) = exps(rep)
check('eself: n=2 good; U = P0^{4eps} z2^{eps} (z2+z0)^{-2eps} z1^{-1/2} (z1+4x z0)^{-1/2-eps} P5^{-1/2-eps} (L1724-1735)',
      rep.n == 2 and rep.good and E == sorted([('even', 2, 1), ('even', -4, 1), ('odd', 0, 1), ('odd', -2, 1), ('odd', -2, 2)]) and (a0, b0) == (0, 8), (E, a0, b0))
# pole-order examples with blow-ups (L1770-1790): eta/(z1 z2 sqrt(P5)) has o=2 at [1:0:0],[1:0:-x],[1:-x:0]; eta/(z1 P5) has o=3 at [1:0:-x]
p = 2147483647; x = 17
Z1, Z2 = sp.symbols('z1 z2'); P5 = (Z2 - Z1) ** 2 + 2 * x * (Z1 + Z2) + x ** 2
def loc(pt, curves, Q=1):
    a, b = pt
    cs = [(P2.from_sympy(sp.sympify(P), Z1, Z2, p).shift(a % p, b % p), sp.Rational(c)) for P, c in curves]
    return _local_order(cs, P2.from_sympy(sp.sympify(Q), Z1, Z2, p).shift(a % p, b % p), p)
check('blow-up example: eta/(z1 z2 sqrt P5) o=2 at the three points (L1770-1779)', [loc(pt, [(Z1, 1), (Z2, 1), (P5, sp.Rational(1, 2))]) for pt in [(0, 0), (0, -x), (-x, 0)]] == [2, 2, 2])
check('blow-up example: eta/(z1 P5) o=3 at [1:0:-x] (L1780-1788)', loc((0, -x), [(Z1, 1), (P5, 1)]) == 3)
check('normal crossing additivity dz1/z1 ^ dz2/z2^2 -> 3 (L1766)', loc((0, 0), [(Z1, 1), (Z2, 2)]) == 3)

npass = sum(1 for r in results if r[1]); nfail = len(results) - npass
print(f'\n{npass}/{len(results)} PASS, {nfail} FAIL, wall {time.time()-t0:.1f}s')
# the per-check record goes to CUTBASIS_PAPER_RECORD when set (a scratch path), never into the package tree
rec = os.environ.get('CUTBASIS_PAPER_RECORD')
if rec:
    with open(rec, 'w') as f:
        json.dump({'stamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'results': results, 'npass': npass, 'nfail': nfail}, f, indent=1, default=str)
sys.exit(1 if nfail else 0)
