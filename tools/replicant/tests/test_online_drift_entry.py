"""O3: the `online_drift` catalog entry, its LaTeX blocks and its GUIDE rows.

The physics checks are in test_online_drift.py and test_online_drift_sim.py; conventions as there.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.catalog import catalog, check_ref, run_cached
from replicant.compose import compose
from replicant.core import Refusal
from replicant.online import drift_rules as VR
from replicant.refdata import check, load
from replicant.testing import slow

BANK = load("online_drift")


def _ok(key, got):
    return check(BANK[key], got, label=key).ok


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


def test_catalog_entry_runs_and_refuses():
    e = catalog()["online_drift"]
    v = run_cached(e, "online", {})[0]                           # every default (generic tables)
    assert v["rule"] == "optimal" and v["drift"] == "random" and "eps_inf" not in v
    assert _ok("random_optimal_coeff", v["eps_coefficient"])
    v = e.run(route="online", rule="annealed_hebb", D="0.1", alpha="5")
    assert abs(v["eps_inf"] - VR.annealed_hebb_stationary(0.1, "random", 0.0)) < 1e-10
    assert v["eps_inf_bound"] <= v["eps_inf"] and v["eps"] >= v["eps_bound"] - 1e-7
    assert not e.finality["eps_inf"].final and not e.finality["eps"].final
    _refuses(lambda: e.run(route="online", rule="oja"), "unsupported:rule")
    _refuses(lambda: e.run(route="online", drift="sudden", D="0.1"), "precondition:drift-kind")
    _refuses(lambda: e.run(route="online", rule="hebb", D="0.1"), "no-solution:no-tracking")
    _refuses(lambda: compose(e.spec, "online").run(Dee="0.1"), "precondition:known-parameter")
    for ref in e.refs[:2]:                     # test_catalog checks all of them in the default tier
        out = check_ref(e, ref)
        assert out.ok, out


@slow
def test_latex_blocks_checked_and_a_wrong_one_caught():
    """The online_drift sympy blocks agree with the code at their first point (test_latex checks all);
    the annealed Hebb stationary block with its expression off by 1% is caught."""
    from dataclasses import replace

    import sympy as sp
    from replicant.tex import verify_block
    blocks = [b for b in catalog()["online_drift"].derivation("online") if b.check is not None]
    assert len(blocks) == 4 and all(not verify_block(b, max_points=1) for b in blocks)
    b = next(b for b in blocks if "random drift: drho" in b.title)
    wrong = replace(b, expr=sp.Eq(b.expr.lhs, b.expr.rhs * sp.Rational(101, 100)))
    assert verify_block(wrong, max_points=1)


def test_guide_is_true():
    """The GUIDE's online_drift rows: every refusal reason listed is raised by the code, and the
    worked example's numbers are what the entry returns (one through the CLI)."""
    import io
    import json
    from pathlib import Path

    from replicant.cli import main
    root = Path(__file__).resolve().parent.parent
    guide = (root / "GUIDE.md").read_text()
    assert "| `online_drift`" in guide and "`online_drift` (online): `rule`" in guide
    code = "".join(p.read_text() for p in [*(root / "online").glob("drift*.py"), root / "online" / "simulate_drift.py",
                                           root / "online" / "odes.py", root / "catalog_online_drift.py"])
    for reason in ("unsupported:rule", "unsupported:stationary", "precondition:drift", "precondition:drift-kind",
                   "precondition:optimal-eta", "precondition:known-parameter", "precondition:noise",
                   "precondition:initial-overlap", "precondition:alpha", "precondition:averages",
                   "precondition:overlap", "undetermined:stationary", "undetermined:small-drift",
                   "undetermined:rho-sign", "no-solution:no-tracking"):
        assert f'"{reason}"' in code, reason
    out = io.StringIO()
    assert main(["run", "online_drift", "--route", "online"], out, io.StringIO()) == 0
    v = json.loads(out.getvalue())["values"]
    e = catalog()["online_drift"]
    v.update(e.run(route="online", rule="annealed_hebb", D="0.1", alpha="5"))
    for key, fmt in (("eps_exponent", ".6f"), ("eps_coefficient", ".6f"), ("eps_inf", ".6f"),
                     ("eps_inf_closed", ".6f"), ("eps_inf_bound", ".6f"), ("eps", ".6f"), ("eps_bound", ".6f")):
        assert f"`{key}` {v[key]:{fmt}}" in guide, (key, v[key])
    v = e.run(route="online", D="0.01", drift="deterministic")
    assert f"`eps_inf` {v['eps_inf']:.6f}" in guide and f"`eps_inf_kc93` {v['eps_inf_kc93']:.6f}" in guide
    assert "`drift.stationary(drift.drift_flow(\"optimal\", D=0.1)).eps` -> 0.195856" in guide
