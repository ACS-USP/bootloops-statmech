"""O3: finite-N simulation of online learning with a drifting teacher against the drift ODE.

online/simulate_drift.py moves the teacher itself (random: B <- (B + sqrt(2D)/N u)/|.|; deterministic:
B turned away from J by arccos(1 - D/N^2) per example; jumps: B replaced) and shares no code with the
ODE side (online/drift.py, drift_rules.py, odes.py, fields.py, optimal.py).  Bands: 99% Student-t
over named seeds; planted ODE errors (drift._PLANT["drift"]) must fall outside them.
"""
import ast
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.core import Refusal
from replicant.online import drift as DR
from replicant.online import simulate_drift as S
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
    src = Path(S.__file__).read_text()
    parts = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            parts |= {c for a in n.names for c in a.name.split(".")}
        elif isinstance(n, ast.ImportFrom):
            parts |= set((n.module or "").split(".")) | {a.name for a in n.names}
    banned = {"odes", "fields", "optimal", "drift", "drift_rules", "linear", "linear_offline", "online"}
    assert not parts & banned, parts & banned
    assert {"amp", "_stats", "simulate"} <= parts                  # the walk sees dotted paths


def _band(rule, D, kind, init, alpha, seeds, N=200, plant=0.0, **kw):
    sim = S.simulate_drift(rule, N, alpha, seeds, D=D, drift=kind, init=init, **kw)
    f = DR.drift_flow(rule, D=D, kind=kind, eta=kw.get("eta", 1.0), noise=kw.get("noise", 0.0))
    jk = {k: kw[k] for k in ("jumps", "jump_overlap") if k in kw}
    ode = DR.integrate_drift(f, alpha, init, alphas=(alpha,), **jk).eps[0]
    with planted(DR, drift=plant):
        bad = DR.integrate_drift(f, alpha, init, alphas=(alpha,), **jk).eps[0]
    return sim.band, ode, bad


def test_teacher_drift_alone():
    """With no learning (F = 0) the student is fixed: random drift gives rho = rho0 e^{-D alpha}
    (VKC p. 11, the e^{-D (alpha - alpha')} autocorrelation), deterministic drift theta = theta0 +
    sqrt(2D) alpha (derived from VKC Eq. (28)); the simulator and the ODE both reproduce them."""
    still = lambda h, b, Q: 0.0 * h
    seeds = tuple(range(12))
    for kind, D in (("random", 0.3), ("deterministic", 0.02)):
        sim = S.simulate_drift(still, 200, 2.0, seeds, D=D, drift=kind, init=(0.8, 1.0))
        want = (math.acos(0.8 * math.exp(-2 * D)) if kind == "random" else
                math.acos(0.8) + math.sqrt(2 * D) * 2) / math.pi
        assert abs(sim.band.mean - want) <= sim.band.half_width + 1e-6, (kind, sim.band, want)   # O(1/N^2) steps
        ode = DR.integrate_drift(DR.drift_flow(still, D=D, kind=kind), 2.0, (0.8, 1.0), alphas=(2.0,)).eps[0]
        assert abs(ode - want) < 1e-8, (kind, ode, want)


def test_ode_vs_simulation_band():
    """N = 200, alpha = 3, 24 named seeds: the optimal rule under random drift D = 0.1 (VKC Fig. 3)
    and the annealed Hebb rule under deterministic drift D = 0.01 (VKC Fig. 5): the ODE inside the
    99% band, the ODE without the drift term (planted) outside it."""
    seeds = tuple(range(500, 524))
    for rule, D, kind in (("optimal", 0.1, "random"), ("annealed_hebb", 0.01, "deterministic")):
        band, ode, bad = _band(rule, D, kind, (0.3, 1.0), 3.0, seeds)
        assert abs(ode - band.mean) <= band.half_width, (rule, ode, band)
        assert abs(bad - band.mean) > 3 * band.half_width, (rule, bad, band)


@slow
def test_ode_vs_simulation_band_large_n():
    """N = 1000, 20 fresh seeds, alpha = 4: every rule kind, both drifts, output noise, a jump and a
    drift schedule; plants (the drift term halved) excluded."""
    seeds = tuple(range(2000, 2020))
    cases = (("perceptron", 0.1, "random", (0.0, 1.0), {}), ("perceptron", 0.02, "deterministic", (0.0, 1.0), {"eta": 2.0}),
             ("hebb", 0.3, "random", (0.0, 1.0), {}), ("adatron", 0.005, "deterministic", (0.0, 1.0), {}),
             ("step", 0.01, "deterministic", (0.3, 1.0), {}), ("symmetric", 0.05, "random", (0.3, 1.0), {}),
             ("optimal", 0.1, "random", (0.3, 1.0), {"noise": 0.1}),
             ("annealed_hebb", 0.1, "random", (0.3, 1.0), {"noise": 0.1}),
             ("step", 0.0, "random", (0.3, 1.0), {"jumps": (2.0,), "jump_overlap": 0.2}),
             ("optimal", lambda a: 0.5 * math.sin(math.pi * a / 4) ** 2, "random", (0.3, 1.0), {}))
    for rule, D, kind, init, kw in cases:
        band, ode, bad = _band(rule, D, kind, init, 4.0, seeds, N=1000, plant=0.5, **kw)
        print(f"{rule}/{kind}/{kw}: sim {band.mean:.5f} +- {band.half_width:.5f} ode {ode:.5f} planted {bad:.5f} "
              f"({abs(bad - band.mean) / band.half_width:.1f} hw)")
        assert abs(ode - band.mean) <= band.half_width, (rule, kind, kw, ode, band)
        if D != 0.0:
            assert abs(bad - band.mean) > 2 * band.half_width, (rule, kind, kw, bad, band)


def test_simulation_refusals():
    seeds = (0, 1)
    _refuses(lambda: S.simulate_drift("oja", 50, 1.0, seeds), "unsupported:rule")
    _refuses(lambda: S.simulate_drift("step", 50, 1.0, seeds, drift="sudden"), "precondition:drift-kind")
    _refuses(lambda: S.simulate_drift("step", 50, 1.0, seeds, D=-1.0), "precondition:drift")
    _refuses(lambda: S.simulate_drift("step", 50, 1.0, seeds, eta=2.0), "precondition:optimal-eta")
    _refuses(lambda: S.simulate_drift("symmetric", 50, 1.0, seeds), "precondition:initial-overlap")
    _refuses(lambda: S.simulate_drift("hebb", 50, 1.0, seeds, jumps=(2.0,)), "precondition:alpha")
    _refuses(lambda: S.simulate_drift("step", 50, 1.0, seeds, init=(0.3, 1.0), jumps=(0.5,)),
             "precondition:initial-overlap")
    _refuses(lambda: S.simulate_drift("hebb", 5, 1.0, seeds), "precondition:size")
