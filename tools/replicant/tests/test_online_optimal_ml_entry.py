"""O6: `rule=optimal` of the `online_scm` catalog entry, its LaTeX blocks and its GUIDE rows.

The physics checks are in test_online_optimal_ml.py and test_online_optimal_ml_sim.py.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.catalog import UNCLAIMED, catalog, run_cached
from replicant.core import Refusal
from replicant.refdata import check, load

BANK = load("online_scm")
OPT22 = {"rule": "optimal", "K": "2", "M": "2"}                   # shared with the RefChecks (run_cached)


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


def test_entry_rule_optimal_runs_and_refuses():
    e = catalog()["online_scm"]
    v = run_cached(e, "online", OPT22)[0]
    assert v["rule"] == "optimal" and check(BANK["vc97_fig2_eps_plateau"], v["eps_plateau_opt"], label="p").ok
    assert "eps_sym_identical" not in v and not e.finality["eps_plateau_opt"].final
    assert "optimal" in e.finality["eps_plateau_opt"].note
    d = e.run(route="online", K="2", M="1", sym="none")             # rule = gd: O5's steps, none of O6's
    assert d["K"] == 2 and "eps_plateau_gd" not in d and "Q11_init" not in d
    for kw, reason in (({"rule": "fast"}, "unsupported:rule"), ({"rule": "optimal", "M": "3"}, "unsupported:M"),
                       ({"rule": "optimal", "sym": "none"}, "precondition:known-parameter"),
                       ({"rule": "optimal", "teacher": "graded"}, "unsupported:teacher"),
                       ({"rule": "optimal", "init": "ss"}, "unsupported:init"),
                       ({"rule": "optimal", "K": "3", "M": "1"}, "unsupported:init"),
                       ({"rule": "optimal", "N": 100}, "precondition:state-point")):
        _refuses(lambda kw=kw: e.run(route="online", **kw), reason)
    for k in ("vc97_fig1_eps_a2p5", "vc97_fig1_hess_hi_a4"):
        assert ("online_scm", k) in UNCLAIMED


def test_latex_blocks_agree():
    from replicant.tex import verify_block
    blocks = [b for b in catalog()["online_scm"].derivation("online") if b.check is not None and "VC97" in b.title
              or (b.check is not None and "plateau of the optimal flow" in b.title)]
    assert len(blocks) >= 3 and all(not verify_block(b, max_points=1) for b in blocks)


def test_guide_is_true():
    root = Path(__file__).resolve().parent.parent
    guide = (root / "GUIDE.md").read_text()
    assert "`rule` (gd, the" in guide and "`rule=optimal`" in guide
    code = "".join(p.read_text() for p in [root / "online" / "optimal_multilayer.py", root / "online" / "optimal_ml_flow.py",
                                           root / "online" / "simulate_optimal_ml.py", root / "catalog_online_scm_opt.py"])
    for reason in ("unsupported:rule", "unsupported:M", "no-solution:singular-hessian", "precondition:known-parameter",
                   "unsupported:init", "precondition:initial-overlap", "no-solution:step-size"):
        assert f'"{reason}"' in code, reason
    v = run_cached(catalog()["online_scm"], "online", OPT22)[0]
    for key in ("eps_plateau_opt", "hess_hi_plateau", "eps_plateau_gd", "Q11_init"):
        assert f"`{key}` {v[key]:.6f}" in guide, (key, v[key])
