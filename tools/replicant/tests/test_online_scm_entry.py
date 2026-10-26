"""O5: the `online_scm` catalog entry, its LaTeX blocks and its GUIDE rows.

The physics checks are in test_online_scm.py, test_online_scm_plateau.py and test_online_scm_sim.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.catalog import UNCLAIMED, catalog, run_cached
from replicant.compose import compose
from replicant.core import Refusal
from replicant.refdata import check, load
from replicant.testing import slow

BANK = load("online_scm")


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
    e = catalog()["online_scm"]
    v = run_cached(e, "online", {})[0]                    # K = M = 2, eta = 1.5 (BRW's scenario); shared with a RefCheck
    assert v["K"] == 2 and v["M"] == 2 and _ok("brw96_fig2_height", v["eps_sym_identical"])
    assert _ok("brw96_fig1_eps", v["eps_sym_permutation"]) and "plateau_status" not in v
    assert not e.finality["eps_sym_identical"].final and not e.finality["plateau_height"].final
    assert e.finality["eps_sim"].ansatz == "finite-N"
    _refuses(lambda: e.run(route="online", K="0"), "precondition:K")
    _refuses(lambda: e.run(route="online", eta="-1"), "precondition:eta")
    _refuses(lambda: e.run(route="online", teacher="noisy"), "unsupported:teacher")
    _refuses(lambda: e.run(route="online", init="random"), "unsupported:init")
    _refuses(lambda: e.run(route="online", sym="odd"), "unsupported:sym")
    _refuses(lambda: e.run(route="online", N=100), "precondition:state-point")
    _refuses(lambda: e.run(route="online", seeds=[1, 2]), "precondition:known-parameter")
    _refuses(lambda: compose(e.spec, "online").run(Kay="3"), "precondition:known-parameter")
    st, _ = e.execute("online", K="3", sym="identical")             # eta = 1.5: the identical root is odd
    assert any("odd identical root" in n for n in st["notes"]) and "eps_sym_permutation" not in st["values"]
    for key in ("ss_eta_max_coefficient", "brw96_eq17_log_slope", "ss_trapping_eta_exponent"):
        assert ("online_scm", key) in UNCLAIMED                # integrations: checked by the slow tests


@slow
def test_plateau_through_the_entry_and_the_cli():
    """alpha given: the learning curve and the plateau report (BRW's initial conditions, X = 1e-6), and the GUIDE's
    plateau numbers; the default run through the CLI's build_result equals the cached one."""
    from pathlib import Path

    from replicant.cli import build_result
    guide = (Path(__file__).resolve().parent.parent / "GUIDE.md").read_text()
    e = catalog()["online_scm"]
    assert build_result("online_scm", "online", {})[1].values == run_cached(e, "online", {})[0]
    v = run_cached(e, "online", {"alpha": "120"})[0]
    for key in ("plateau_height", "plateau_entry", "plateau_exit"):
        assert f"`{key}` {v[key]:.6f}" in guide, (key, v[key])
    assert v["plateau_status"] == "escaped" and v["plateau_branch"] == "identical"
    assert _ok("brw96_fig2_height", v["plateau_height"]) and v["plateau_exit"] > v["plateau_entry"]
    assert v["plateau_rtol_spread"] < 1e-4 and v["eps"] < 1e-3
    v = run_cached(e, "online", {"alpha": "40"})[0]
    assert v["plateau_status"] == "undetermined" and "plateau_exit" not in v and "plateau_length" not in v


@slow
def test_latex_blocks_checked_and_a_wrong_one_caught():
    from dataclasses import replace

    import sympy as sp
    from replicant.tex import verify_block
    blocks = [b for b in catalog()["online_scm"].derivation("online") if b.check is not None]
    assert len(blocks) >= 5 and all(not verify_block(b, max_points=1) for b in blocks)
    b = next(b for b in blocks if "I3" in b.title)
    wrong = replace(b, expr=sp.Eq(b.expr.lhs, b.expr.rhs * sp.Rational(101, 100)))
    assert verify_block(wrong, max_points=1)


def test_guide_is_true():
    """The GUIDE's online_scm rows: every refusal reason listed is raised by the code, and the worked example's
    numbers are what the entry returns (the default run shared with the RefChecks; the CLI path and the plateau
    numbers: the slow test above)."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    guide = (root / "GUIDE.md").read_text()
    assert "| `online_scm`" in guide and "`online_scm` (online): `K`" in guide
    code = "".join(p.read_text() for p in [*(root / "online").glob("scm*.py"), root / "online" / "plateau.py",
                                           root / "online" / "simulate_scm.py", root / "catalog_online_scm.py",
                                           root / "online" / "odes.py"])
    for reason in ("precondition:K", "precondition:M", "precondition:eta", "unsupported:teacher", "unsupported:init", "unsupported:sym",
                   "unsupported:branch", "precondition:symmetric", "precondition:state-point",
                   "precondition:initial-overlap", "precondition:alpha", "precondition:threshold", "precondition:grid",
                   "precondition:size", "precondition:known-parameter", "no-solution:fixed-point",
                   "no-solution:critical-rate", "no-solution:step-size"):
        assert f'"{reason}"' in code, reason
    v = run_cached(catalog()["online_scm"], "online", {})[0]
    for key in ("eps_sym_identical", "eps_sym_permutation", "tau_esc_identical"):
        assert f"`{key}` {v[key]:.6f}" in guide, (key, v[key])
