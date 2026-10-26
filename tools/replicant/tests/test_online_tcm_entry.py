"""O4: the `online_tcm` catalog entry, its LaTeX blocks and its GUIDE rows.

The physics checks are in test_online_tcm.py and test_online_tcm_sim.py; conventions as there.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.catalog import UNCLAIMED, catalog, check_ref, run_cached
from replicant.compose import compose
from replicant.core import Refusal
from replicant.online import tcm as T
from replicant.refdata import check, load
from replicant.testing import slow

BANK = load("online_tcm")


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
    e = catalog()["online_tcm"]
    v = run_cached(e, "online", {})[0]                           # every default: CC95's rule at K = 3
    assert v["rule"] == "optimal" and v["K"] == 3 and _ok("cc95_kappa", v["eps_constant"])
    assert "noise_equivalent" not in v                           # the nonlocal rule has no perceptron image
    v = run_cached(e, "online", {"rule": "local_optimal", "K": "3", "alpha": "5"})[0]      # shared with the GUIDE test
    assert v["noise_equivalent"] == T.effective_noise(3) and v["equivalence_gap"] < 1e-8
    assert abs(v["eps_constant"] / v["eps_constant_perceptron"] - 1) < 2e-3
    assert not e.finality["eps"].final and not e.finality["eps_constant"].final
    _refuses(lambda: e.run(route="online", rule="oja"), "unsupported:rule")
    _refuses(lambda: e.run(route="online", K="5"), "unsupported:K")
    _refuses(lambda: e.run(route="online", rule="hebb", K="4"), "precondition:K")
    _refuses(lambda: e.run(route="online", rule="hebb", K="three"), "precondition:K")
    _refuses(lambda: e.run(route="online", rule="local_optimal", eta="2"), "precondition:optimal-eta")
    _refuses(lambda: e.run(route="online", rule="hebb", N=30), "precondition:state-point")
    _refuses(lambda: compose(e.spec, "online").run(Kay="3"), "precondition:known-parameter")
    for key in ("cc95_fig1_es_a3p3", "cc95_fig1_es_a6p7", "cc95_fig1_es_a10", "cc95_fig1_es_a16p5"):
        assert ("online_tcm", key) in UNCLAIMED                 # checked by the slow test (nonlocal ODE to 16.5)
    for ref in e.refs[:2]:                     # test_catalog checks all of them in the default tier
        out = check_ref(e, ref)
        assert out.ok, out


@slow
def test_latex_blocks_checked_and_a_wrong_one_caught():
    """The online_tcm sympy blocks agree with the code at their first point (test_latex checks all);
    CC95 Eq. (5) with one term off by 1% is caught."""
    from dataclasses import replace

    import sympy as sp
    from replicant.tex import verify_block
    blocks = [b for b in catalog()["online_tcm"].derivation("online") if b.check is not None]
    assert len(blocks) >= 3 and all(not verify_block(b, max_points=1) for b in blocks)
    b = next(b for b in blocks if "Eq. (5)" in b.title)
    wrong = replace(b, expr=sp.Eq(b.expr.lhs, b.expr.rhs * sp.Rational(101, 100)))
    assert verify_block(wrong, max_points=1)


def test_guide_is_true():
    """The GUIDE's online_tcm rows: every refusal reason listed is raised by the code, and the worked
    example's numbers are what the entry returns (one through the CLI)."""
    import io
    import json
    from pathlib import Path

    from replicant.cli import main
    root = Path(__file__).resolve().parent.parent
    guide = (root / "GUIDE.md").read_text()
    assert "| `online_tcm`" in guide and "`online_tcm` (online): `rule`" in guide
    code = "".join(p.read_text() for p in [*(root / "online").glob("tcm*.py"), root / "online" / "simulate_tcm.py",
                                           root / "catalog_online_tcm.py"])
    for reason in ("unsupported:rule", "unsupported:K", "precondition:K", "precondition:eta",
                   "precondition:optimal-eta", "precondition:initial-overlap", "precondition:state-point",
                   "precondition:alpha", "precondition:overlap", "precondition:asymptote", "precondition:size",
                   "precondition:known-parameter", "undetermined:asymptote", "undetermined:rho-sign"):
        assert f'"{reason}"' in code, reason
    out = io.StringIO()
    assert main(["run", "online_tcm", "--route", "online"], out, io.StringIO()) == 0
    v = json.loads(out.getvalue())["values"]
    assert f"`eps_constant` {v['eps_constant']:.6f}" in guide, v["eps_constant"]
    v = run_cached(catalog()["online_tcm"], "online", {"rule": "local_optimal", "K": "3", "alpha": "5"})[0]
    for key in ("noise_equivalent", "eps", "rho_perceptron_mapped", "eps_constant", "eps_constant_perceptron"):
        assert f"`{key}` {v[key]:.6f}" in guide, (key, v[key])
