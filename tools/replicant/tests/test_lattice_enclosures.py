"""Printed certified values are enclosures away from beta_c, and the radius-stall rule (T15).

E9 deferred minors: test_lattice.test_printed_certified_values_are_enclosures checks the printed
value +- error of minus_beta_f at beta_c only; here every printed physical value of three more
certified runs (2D Ising at beta = 0.6, 1D Ising in a field, the Bethe ferromagnet at h = 0.02)
must contain the ball recomputed by the library call that defines it.  The radius-stall rule of
lattice.ball.certify must not treat an exact ball (radius 0 twice) as a stall: exercised with a
stub in place of tools/baller, whose solve asks for two attempts (unreachable through baller
itself, which stops at the first exact ball).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from replicant.core import Refusal


def _encloses(v, err, ball):
    from flint import arb
    from replicant.lattice.ball import baller
    with baller().hygiene.ctx_guard(prec=300):
        lo, hi = arb(v) - arb(err), arb(v) + arb(err)
        return lo.upper() <= ball.lower() and ball.upper() <= hi.lower()


def test_printed_values_enclose_their_balls_off_critical():
    from replicant.catalog import catalog
    from replicant.cli import result_json
    from replicant.compose import compose
    from replicant import lattice as L
    targets = {
        ("ising2d", "exact", (("beta", "0.6"),)): {
            "minus_beta_f": lambda: L.onsager_free_energy("0.6"),
            "magnetization": lambda: L.yang_magnetization("0.6")},
        ("ising1d", "exact", (("beta", "1"), ("h", "1/2"))): {
            "minus_beta_f": lambda: L.ising1d_free_energy(1, "1/2")},
        ("bethe_ising", "cavity", (("h", "0.02"),)): {
            "phi": lambda: L.bethe_ising_free_energy(3, 1, "0.02").phi,
            "magnetization": lambda: L.bethe_ising_free_energy(3, 1, "0.02").selected.magnetization},
    }
    for (name, route, params), keys in targets.items():
        e = catalog()[name]
        res = compose(e.spec, route).run(**dict(params))
        js = result_json(name, e.spec, res)
        for k, ball in keys.items():
            v, err = js["values"][k], js["errors"][k]
            assert isinstance(v, str) and len(v) >= 40, (name, k, v)
            assert _encloses(v, err, ball().arb), (name, k, v, err)
            assert not _encloses(v[:22], err, ball().arb), (name, k)   # a 20-digit print is not


class _StubBaller:
    """run returns an exact ball at every precision; solve calls attempt at two precisions."""
    class SolveRefused(Exception):
        pass

    class _R:
        def __init__(self, value):
            self.value = value

    def run(self, thunk, dps):
        return self._R(thunk())

    def solve(self, attempt, digits, start_dps=None):
        attempt(30)
        r = attempt(60)
        return type("Res", (), {"ball": r.value, "certified_digits": digits})()


def test_exact_ball_is_not_a_radius_stall():
    from flint import arb
    from replicant.lattice import ball as B
    saved = B.baller
    B.baller = lambda: _StubBaller()
    try:
        out = B.certify(lambda: arb(3))
    except Refusal as r:
        raise AssertionError(f"an exact ball was refused: {r}")
    finally:
        B.baller = saved
    assert B.baller is saved and out.arb == arb(3)
