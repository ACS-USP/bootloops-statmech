"""O5: finite-N online gradient descent in the soft committee machine against the Saad-Solla ODE.

online/simulate_scm.py builds the teacher (orthonormal B_n, so T = delta exactly) and the student (J_i placed
to realise the requested initial (Q, R) exactly, as BRW do by Gram-Schmidt) and runs the update
J_i <- J_i + (eta/N) delta_i xi itself, with its own erf units and its own eps formula; it shares no code with
scm.py, scm_flow.py or plateau.py (the AST test).  The ODE starts from the same (Q, R).  Bands: 99% Student-t
over named seeds at every checkpoint alpha; the ODE without its eta^2 (I4) term must fall outside them.
"""
import ast
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.core import Refusal
from replicant.online import scm as S
from replicant.online import scm_flow as F
from replicant.online import simulate_scm as SIM
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
    for n in ast.walk(ast.parse(Path(SIM.__file__).read_text())):
        if isinstance(n, ast.Import):
            parts |= {c for a in n.names for c in a.name.split(".")}
        elif isinstance(n, ast.ImportFrom):
            parts |= set((n.module or "").split(".")) | {a.name for a in n.names}
    banned = {"scm", "scm_flow", "plateau", "odes", "fields", "optimal", "tcm", "tcm_flow", "simulate", "online"}
    assert not parts & banned, parts & banned
    assert {"amp", "_stats"} <= parts


def _init(xr):
    """Nearly identical students with an explicit R asymmetry xr (BRW Eq. (16) with X = 0.02)."""
    return np.array([[0.5, 0.48], [0.48, 0.5]]), np.array([[0.0, 0.0], [0.2 * xr, xr]])


def _compare(N, eta, alphas, seeds, init, plant=0.0):
    sim = SIM.simulate_scm(2, 2, N, eta, alphas, seeds, init=init)
    f = S.scm_flow(2, 2, eta)
    ode = F.integrate_scm(f, alphas[-1], init, alphas=alphas).eps
    with planted(S, eta2=plant):
        bad = F.integrate_scm(f, alphas[-1], init, alphas=alphas).eps
    return sim, ode, bad


def test_initial_overlaps_are_realised_exactly():
    Q, R = _init(0.05)
    sim = SIM.simulate_scm(2, 2, 60, 1.0, (0.5,), (1, 2), init=(Q, R), return_initial=True)
    for q, r in sim.initial:
        assert np.allclose(q, Q, atol=1e-12) and np.allclose(r, R, atol=1e-12)


def test_ode_vs_simulation_band_small_n():
    """Default tier: N = 80, eta = 1.5, 12 named seeds, alpha = 4, 8, 16 (the approach to the plateau only;
    the passage through it is the slow test): the ODE is inside every 99% band; the ODE without the eta^2 term
    is outside at least one band by more than 3 half-widths."""
    alphas = (4.0, 8.0, 16.0)
    sim, ode, bad = _compare(80, 1.5, alphas, tuple(range(300, 312)), _init(0.1))
    for b, o in zip(sim.bands, ode):
        assert abs(o - b.mean) <= b.half_width, (o, b)
    assert max(abs(x - b.mean) / b.half_width for b, x in zip(sim.bands, bad)) > 3, (bad, sim.bands)


@slow
def test_ode_vs_simulation_through_the_plateau():
    """N = 1000, eta = 1.5, 16 named seeds, from nearly identical students (Q_12 = 0.48) with an R asymmetry
    0.01: the ODE trajectory goes to the permutation-symmetric plateau (BRW's less symmetric point, eps =
    0.0346) and escapes inside the window (online/plateau.py: status escaped); at alpha = 10, ..., 80 the ODE
    is inside every 99% band; the ODE without the eta^2 term is outside (plateau 0.011 instead of 0.035)."""
    from replicant.online import plateau as P
    alphas = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)
    init = _init(0.01)
    rep = P.analyse(S.scm_flow(2, 2, 1.5), init, 100.0)
    assert rep.status == "escaped" and rep.branch == "permutation" and rep.alpha_exit < 60, rep
    sim, ode, bad = _compare(1000, 1.5, alphas, tuple(range(4000, 4016)), init)
    for a, b, o, x in zip(alphas, sim.bands, ode, bad):
        print(f"alpha {a:5.1f}: sim {b.mean:.5f} +- {b.half_width:.5f}  ode {o:.5f}  planted {x:.5f} "
              f"({abs(x - b.mean) / b.half_width:.1f} hw)")
        assert abs(o - b.mean) <= b.half_width, (a, o, b)
    assert ode[1] > 0.034 and ode[-1] < 0.002                      # on the plateau, then out
    assert max(abs(x - b.mean) / b.half_width for b, x in zip(sim.bands, bad)) > 5


def test_simulation_refusals():
    seeds = (0, 1)
    init = _init(0.1)
    _refuses(lambda: SIM.simulate_scm(2, 2, 10, 1.0, (1.0,), seeds, init=init), "precondition:size")
    _refuses(lambda: SIM.simulate_scm(2, 2, 50, 0.0, (1.0,), seeds, init=init), "precondition:eta")
    _refuses(lambda: SIM.simulate_scm(2, 2, 50, 1.0, (), seeds, init=init), "precondition:alpha")
    _refuses(lambda: SIM.simulate_scm(2, 2, 50, 1.0, (2.0, 1.0), seeds, init=init), "precondition:alpha")
    _refuses(lambda: SIM.simulate_scm(2, 2, 50, 1.0, (1.0,), "01", init=init), "precondition:named-seeds")
    _refuses(lambda: SIM.simulate_scm(2, 2, 50, 1.0, (1.0,), seeds, init=(np.eye(2), 2 * np.eye(2))),
             "precondition:initial-overlap")
    _refuses(lambda: SIM.simulate_scm(2, 2, 50, 1.0, (1.0,), seeds, init=None), "precondition:state-point")
    _refuses(lambda: SIM.simulate_scm(2, 2, 50, 1.0, (1.0,), seeds, init=init, teacher="noisy"), "unsupported:teacher")
