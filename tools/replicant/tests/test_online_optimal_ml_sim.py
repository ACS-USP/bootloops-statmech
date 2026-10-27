"""O6: finite-N simulation of VC97's optimal algorithm against the ODE (online/simulate_optimal_ml.py).

The simulator shares no code with the ODE side (the imports are inspected): it types VC97's Eq. (3) again, takes
H and G by central differences of it, and computes yhat by its own level-curve integral (M = 2).  Bands are 99%
Student-t over named seeds.  A finding recorded here: where det H crosses 0 (alpha ~ 1 for VC97's Fig. 1 start)
F* diverges, and a finite-N run that steps close to the crossing is kicked far (norms ~ 30 at N = 1000), so the
bands after the crossing are wide and heavy-tailed; the pre-crossing band is the sharp check.
"""
import ast
import os
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.core import Refusal
from replicant.online import optimal_ml_flow as OF
from replicant.online import optimal_multilayer as O
from replicant.online import simulate_optimal_ml as SIM
from replicant.testing import slow


@contextmanager
def planted(**changes):
    hook = O._PLANT
    saved = dict(hook)
    try:
        hook.update(changes)
        yield
    finally:
        hook.clear()
        hook.update(saved)
        assert O._PLANT is hook and hook == saved


def _z(ode, band):
    return abs(ode - band.mean) / band.half_width


def test_simulator_imports_no_ode_code_and_refuses():
    allowed = {"math", "dataclasses", "numpy", "scipy.special", "amp._stats", "core", "csp"}
    for n in ast.walk(ast.parse(Path(SIM.__file__).read_text())):
        if isinstance(n, ast.Import):
            assert all(a.name in allowed for a in n.names), [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            assert n.module in allowed, n.module
    init = OF.vc97_initial(2, 1)[:2]
    for args, reason in (((2, 1, 10, (0.5,), (0,), init), "precondition:size"),
                         ((2, 3, 100, (0.5,), (0,), init), "unsupported:M"),
                         ((2, 1, 100, (0.5, 0.2), (0,), init), "precondition:alpha")):
        try:
            SIM.simulate_optimal(*args)
            raise AssertionError(reason)
        except Refusal as r:
            assert r.reason == reason


def test_tiny_band_before_the_crossing():
    """K = 2, M = 1, N = 60, 8 seeds, alpha = 0.5 (before det H = 0): the ODE is inside the band and the ODE with H
    read as de/dQ (hess = 0.5) is outside it."""
    Q0, R0, _ = OF.vc97_initial(2, 1)
    b = SIM.simulate_optimal(2, 1, 60, (0.5,), tuple(range(8)), (Q0, R0)).bands[0]
    ode = OF.integrate_optimal(O.optimal_flow(2, 1), 0.5, (Q0, R0), alphas=(0.5,), rtol=1e-6).eps[0]
    with planted(hess=0.5):
        bad = OF.integrate_optimal(O.optimal_flow(2, 1), 0.5, (Q0, R0), alphas=(0.5,), rtol=1e-6).eps[0]
    assert _z(ode, b) < 1 and _z(bad, b) > 1, (b, ode, bad)


@slow
def test_k2m1_bands_through_the_crossing():
    """K = 2, M = 1, N = 1000, 16 seeds: the ODE is inside the bands at alpha = 0.5, 3, 5 and the planted W (x 1.1)
    ODE is outside the alpha = 0.5 band."""
    Q0, R0, _ = OF.vc97_initial(2, 1)
    al = (0.5, 3.0, 5.0)
    res = SIM.simulate_optimal(2, 1, 1000, al, tuple(range(16)), (Q0, R0))
    tr = OF.integrate_optimal(O.optimal_flow(2, 1), 5.0, (Q0, R0), alphas=al, rtol=1e-8)
    assert all(_z(e, b) < 1 for e, b in zip(tr.eps, res.bands)), (tr.eps, res.bands)
    with planted(w=1.1):
        bad = OF.integrate_optimal(O.optimal_flow(2, 1), 0.5, (Q0, R0), alphas=(0.5,), rtol=1e-8).eps[0]
    assert _z(bad, res.bands[0]) > 1


@slow
def test_k2m2_band_before_the_spike():
    """K = M = 2, N = 400, 8 seeds, alpha = 0.3, 0.45 (before the det H = 0 crossing, where the M = 2 quadrature is
    trusted): the ODE (W by info_matrix) is inside both bands, the planted W (x 1.1) outside one."""
    Q0, R0, _ = OF.vc97_initial(2, 2)
    al = (0.3, 0.45)
    res = SIM.simulate_optimal(2, 2, 400, al, tuple(range(8)), (Q0, R0))
    kw = dict(rtol=1e-4, atol=1e-6)
    tr = OF.integrate_optimal(O.optimal_flow(2, 2), 0.45, (Q0, R0), alphas=al, **kw)
    assert all(_z(e, b) < 1 for e, b in zip(tr.eps, res.bands)), (tr.eps, res.bands)
    with planted(w=1.1):
        bad = OF.integrate_optimal(O.optimal_flow(2, 2), 0.45, (Q0, R0), alphas=al, **kw)
    assert max(_z(e, b) for e, b in zip(bad.eps, res.bands)) > 1
