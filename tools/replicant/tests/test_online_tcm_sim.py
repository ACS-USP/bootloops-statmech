"""O4: finite-N online learning in the tree committee machine against the TCM ODE, at K = 3 (and 5, slow).

online/simulate_tcm.py builds the TCM teacher and student itself (K branches of N/K Gaussian inputs, the
majority label, the update J_k <- J_k + (1/N) F_k xi_k) and shares no code with the ODE side (tcm.py,
tcm_flow.py, odes.py, fields.py, optimal.py).  The nonlocal rule is CC95 Eqs. (19)-(20) as printed; the
local rule is the branch posterior with the tie probability typed again (math.comb).  Bands: 99%
Student-t over named seeds; a planted ODE error (the 1/K of the eta^2 term x 1.5 or dropped, tcm_flow._PLANT)
must fall outside them.
"""
import ast
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.core import Refusal
from replicant.online import simulate_tcm as S
from replicant.online import tcm_flow as TF
from replicant.testing import slow


@contextmanager
def planted(module, attr="_PLANT", **changes):
    hook = getattr(module, attr)
    saved = dict(hook)
    try:
        hook.update(changes)
        yield
    finally:
        hook.clear()
        hook.update(saved)
        assert getattr(module, attr) is hook and hook == saved


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


def test_simulator_shares_no_code_with_the_ode():
    parts = set()
    for n in ast.walk(ast.parse(Path(S.__file__).read_text())):
        if isinstance(n, ast.Import):
            parts |= {c for a in n.names for c in a.name.split(".")}
        elif isinstance(n, ast.ImportFrom):
            parts |= set((n.module or "").split(".")) | {a.name for a in n.names}
    banned = {"odes", "fields", "optimal", "tcm", "tcm_flow", "drift", "drift_rules", "simulate", "online"}
    assert not parts & banned, parts & banned
    assert {"amp", "_stats"} <= parts                          # the walk sees dotted paths


def _band(rule, K, N, alpha, seeds, init, plant, eta=1.0):
    sim = S.simulate_tcm(rule, K, N, alpha, seeds, eta=eta, init=init)
    f = TF.tcm_flow(rule, K, eta=eta)
    ode = TF.integrate_tcm(f, alpha, init, alphas=(alpha,)).eps[0]
    with planted(TF, inv_k=plant):
        bad = TF.integrate_tcm(f, alpha, init, alphas=(alpha,)).eps[0]
    return sim.band, ode, bad


def test_ode_vs_simulation_band_k3():
    """K = 3, N = 3 x 100, alpha = 3, 24 named seeds, the local optimal rule from rho0 = 0.3 and Hebb
    (eta = 1.5): the ODE inside the 99% band, the ODE with the 1/K of the eta^2 term multiplied by 1.5
    (local rule; dropping it, x K, makes rho reach 0) or dropped (Hebb) outside it."""
    seeds = tuple(range(700, 724))
    for rule, init, eta, plant in (("local_optimal", (0.3, 1.0), 1.0, 1.5), ("hebb", (0.0, 1.0), 1.5, 3.0)):
        band, ode, bad = _band(rule, 3, 300, 3.0, seeds, init, plant, eta)
        assert abs(ode - band.mean) <= band.half_width, (rule, ode, band)
        assert abs(bad - band.mean) > 3 * band.half_width, (rule, bad, band)


@slow
def test_ode_vs_simulation_band_large_n():
    """N = 3 x 300, 20 fresh seeds, alpha = 4: CC95's nonlocal rule (Eqs. (19)-(20) as printed) and the
    local rule at K = 3, Hebb at K = 3, the local rule at K = 5 (N = 5 x 180); plants (the eta^2 term's 1/K
    x 1.5 for the optimal rules, dropped for Hebb) excluded."""
    seeds = tuple(range(3000, 3020))
    for rule, K, N, init, eta in (("optimal", 3, 900, (0.3, 1.0), 1.0), ("local_optimal", 3, 900, (0.3, 1.0), 1.0),
                                  ("hebb", 3, 900, (0.0, 1.0), 1.0), ("local_optimal", 5, 900, (0.3, 1.0), 1.0)):
        band, ode, bad = _band(rule, K, N, 4.0, seeds, init, float(K) if rule == "hebb" else 1.5, eta)
        print(f"{rule}/K={K}: sim {band.mean:.5f} +- {band.half_width:.5f} ode {ode:.5f} planted {bad:.5f} "
              f"({abs(bad - band.mean) / band.half_width:.1f} hw)")
        assert abs(ode - band.mean) <= band.half_width, (rule, K, ode, band)
        assert abs(bad - band.mean) > 2 * band.half_width, (rule, K, bad, band)


def test_simulation_refusals():
    seeds = (0, 1)
    _refuses(lambda: S.simulate_tcm("oja", 3, 30, 1.0, seeds), "unsupported:rule")
    _refuses(lambda: S.simulate_tcm("optimal", 5, 50, 1.0, seeds, init=(0.3, 1.0)), "unsupported:K")
    _refuses(lambda: S.simulate_tcm("hebb", 4, 40, 1.0, seeds), "precondition:K")
    _refuses(lambda: S.simulate_tcm("hebb", 3, 31, 1.0, seeds), "precondition:size")
    _refuses(lambda: S.simulate_tcm("hebb", 3, 6, 1.0, seeds), "precondition:size")
    _refuses(lambda: S.simulate_tcm("local_optimal", 3, 30, 1.0, seeds), "precondition:initial-overlap")
    _refuses(lambda: S.simulate_tcm("local_optimal", 3, 30, 1.0, seeds, init=(0.3, 1.0), eta=2.0),
             "precondition:optimal-eta")
    _refuses(lambda: S.simulate_tcm("hebb", 3, 30, -1.0, seeds), "precondition:alpha")
    _refuses(lambda: S.simulate_tcm("hebb", 3, 30, 1.0, "01"), "precondition:named-seeds")
    _refuses(lambda: S.simulate_tcm("hebb", 3, 30, 1.0, seeds, init=("a", 1.0)), "precondition:state-point")
    _refuses(lambda: S.simulate_tcm("local_optimal", 3, 30, 2.0, (5,), init=(0.01, 1.0)), "undetermined:rho-sign")
