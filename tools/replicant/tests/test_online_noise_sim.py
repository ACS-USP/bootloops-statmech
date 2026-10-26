"""O1: finite-N simulation of the online perceptron (an independent route to the ODEs) and the
noise robustness of the optimal rule (Copelli, Eichhorn, Kinouchi, Biehl, Simonetti, Riegler &
Caticha, Europhys. Lett. 37, 427 (1997): output noise, K = 1 of their tree machines).

Conventions as tests/test_online_perceptron.py.  The simulator (online/simulate.py) imports none of
online/fields.py, odes.py, optimal.py: it re-implements the modulations and draws the examples.
Bands: 99% Student-t over named seeds (amp/_stats.student_band).
"""
import ast
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.catalog import catalog, check_ref
from replicant.core import Refusal
from replicant.online import odes as O
from replicant.online import optimal as OPT
from replicant.online import simulate as SIM
from replicant.testing import slow

ONLINE = Path(__file__).resolve().parent.parent / "online"


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


def _ode_eps(rule, alpha, init, **kw):
    return O.integrate(O.perceptron_flow(rule, **kw), alpha, init).eps[-1]


def _planted_eps(rule, alpha, init, factor=0.0, **kw):
    """The ODE with the eta^2 <F^2> term of dQ/dalpha scaled by `factor` (odes._PLANT hook;
    0 drops it)."""
    saved = dict(O._PLANT)
    try:
        O._PLANT["eta2"] = factor
        return _ode_eps(rule, alpha, init, **kw)
    finally:
        O._PLANT.clear()
        O._PLANT.update(saved)
        assert O._PLANT == saved


def test_simulator_shares_no_code_with_the_ode():
    tree = ast.parse((ONLINE / "simulate.py").read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.add((node.module or "") + ":" + ",".join(a.name for a in node.names))
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
    for bad in ("fields", "odes", "optimal"):
        assert not any(bad in n for n in names), names


def test_ode_vs_simulation_band():
    """Default tier: N = 200, alpha = 4, 24 named seeds, perceptron rule from (R, Q) = (0, 1).
    The ODE value lies in the 99% band; the planted ODE (no eta^2 term) lies outside it."""
    sim = SIM.simulate_perceptron("perceptron", 200, 4.0, range(24))
    ode = _ode_eps("perceptron", 4.0, (0.0, 1.0))
    bad = _planted_eps("perceptron", 4.0, (0.0, 1.0))
    b = sim.band
    assert abs(ode - b.mean) <= b.half_width, (ode, b)
    assert abs(bad - b.mean) > 3 * b.half_width, (bad, b)
    assert len(sim.eps_seeds) == 24 and sim.N == 200


@slow
def test_ode_vs_simulation_band_large_n():
    """N = 1000, alpha = 10, 20 seeds per rule (fresh seeds per rule), noiseless and with output
    noise 0.1; each ODE value inside its band, each planted ODE outside.  The plant here HALVES the
    eta^2 term (a subtler error than dropping it; dropped, the noisy perceptron's theta reaches 0
    at finite alpha and the planted ODE has no solution at alpha = 10)."""
    cases = (("hebb", {}, (0.0, 1.0)), ("perceptron", {}, (0.0, 1.0)),
             ("adatron", {}, (0.0, 1.0)), ("optimal", {}, (0.1, 1.0)),
             ("perceptron", {"noise": 0.1}, (0.0, 1.0)),
             ("optimal", {"noise": 0.1}, (0.1, 1.0)))
    for i, (rule, kw, init) in enumerate(cases):
        seeds = range(100 * (i + 1), 100 * (i + 1) + 20)
        sim = SIM.simulate_perceptron(rule, 1000, 10.0, seeds, init=init, **kw)
        ode, bad = _ode_eps(rule, 10.0, init, **kw), _planted_eps(rule, 10.0, init, 0.5, **kw)
        b = sim.band
        print(f"{rule} {kw}: sim {b.mean:.5f} +- {b.half_width:.5f}, ode {ode:.5f}, "
              f"planted {bad:.5f} ({abs(bad - b.mean) / b.half_width:.1f} half-widths)")
        assert abs(ode - b.mean) <= b.half_width, (rule, kw, ode, b)
        assert abs(bad - b.mean) > b.half_width, (rule, kw, bad, b)


def test_simulation_refusals():
    _refuses(lambda: SIM.simulate_perceptron("perceptron", 5, 1.0, range(4)), "precondition:size")
    _refuses(lambda: SIM.simulate_perceptron("perceptron", 50, 1.0, [-1, 2]),
             "precondition:named-seeds")
    _refuses(lambda: SIM.simulate_perceptron("oja", 50, 1.0, range(4)), "unsupported:rule")
    _refuses(lambda: SIM.simulate_perceptron("hebb", 50, 1.0, range(4), eta=0), "precondition:eta")
    _refuses(lambda: SIM.simulate_perceptron("hebb", 50, 1.0, [3]), "precondition:two-samples")
    # the optimal rule from rho0 = 0.01 at N = 200: some seeds cross rho = 0 (measured: 4 of 20)
    _refuses(lambda: SIM.simulate_perceptron("optimal", 200, 5.0, range(20), noise=0.2,
                                             init=(0.01, 1.0)), "undetermined:rho-sign")


# ---- noise robustness (paper 6, output noise) -----------------------------------------------
def test_noise_robustness():
    """Copelli et al. 1997: with output-noise level lam and the optimal rule built on an estimate
    Lam, (i) learning fails (rho = 0 attractive) below Lam = 2 lam - 1/2 (their text p. 5, the
    dashed line of Fig. 1a); (ii) perfect learning is lost below the solid line of Fig. 1a (read
    off the figure); both computed here from the sign of drho/dalpha of the ODE near rho = 0 and
    rho = 1 (bisection on Lam), compared through the catalog with the bank."""
    e = catalog()["online_perceptron"]
    refs = [r for r in e.refs if "boundary" in r.key]
    assert len(refs) >= 4
    for ref in refs:
        out = check_ref(e, ref)
        assert out.ok, (ref.key, out)
    # trajectories on either side (lam = 0.3: no-learning line 0.1, robust line ~0.150)
    lam = 0.3
    nl, rb, _ = OPT.noise_boundaries(lam)
    assert nl < rb < lam
    def run(est):                       # eps at alpha = 500 and 1000 from rho0 = 0.1
        return O.integrate(O.perceptron_flow("optimal", noise=lam, noise_estimate=est), 1000.0,
                           (0.1, 1.0), alphas=(500.0, 1000.0), rtol=1e-7).eps
    for est in (lam, rb + 0.03):                            # perfect learning: eps ~ C / alpha
        e5, e10 = run(est)
        assert e10 < 0.02 and 0.45 < e10 / e5 < 0.6, (est, e5, e10)
    e5, e10 = run(0.5 * (nl + rb))                          # imperfect: a plateau eps > 0
    assert 0.05 < e10 < 0.45 and abs(e10 - e5) < 1e-4, (e5, e10)
    # below the dashed line rho = 0 attracts and is reached at finite alpha (W* ~ 1/rho)
    _refuses(lambda: run(nl - 0.03), "no-solution:rho-zero")
    # positive control: with the eta^2 term dropped drho/dalpha > 0 for every estimate: no line
    saved = dict(O._PLANT)
    try:
        O._PLANT["eta2"] = 0.0
        assert OPT.noise_boundaries(0.4)[:2] == (None, None)
    finally:
        O._PLANT.clear()
        O._PLANT.update(saved)
    assert O._PLANT == saved
    # the matched optimum is C = 2 / I(lam) (their Eq. (8), with the corrected Gaussian weight)
    assert OPT.asymptotic_constant(0.1) > OPT.asymptotic_constant(0.0)
    assert math.isfinite(OPT.asymptotic_constant(0.45))
    _refuses(lambda: OPT.noise_boundaries(0.5), "precondition:noise")
