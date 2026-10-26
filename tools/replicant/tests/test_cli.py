"""CLI: `python3 -m replicant list | run <entry|spec.json> [--route R] [--param k=v ...]
[--out result.json]` (run with tools/ on the path).  The default tier calls
replicant.cli.main in-process (stdout / stderr captured); one slow test runs the real
module entry point in a subprocess."""
import io
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.catalog import catalog
from replicant.cli import main, parse_param
from replicant.testing import slow

TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _main(argv):
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_cli_list_names_all_entries():
    code, out, _ = _main(["list"])
    assert code == 0
    for name, e in catalog().items():
        line = next(l for l in out.splitlines() if l.split()[0] == name)
        assert ",".join(sorted(e.routes)) in line and e.cite.split(";")[0] in line, line


def test_cli_refusal_prints_REFUSED_and_exits_2():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "spec.json")
        with open(path, "w") as f:
            json.dump({"family": "xyz", "variables": "ising", "disorder": "gaussian"}, f)
        code, out, err = _main(["run", path, "--out", os.path.join(d, "r.json")])
        assert code == 2 and not os.path.exists(os.path.join(d, "r.json"))
    assert err.startswith("REFUSED: missing-step:family:xyz: "), err
    code, _, err = _main(["run", "pspin", "--route", "cavity"])
    assert code == 2 and err.startswith("REFUSED: missing-step:route:cavity: "), err
    code, _, err = _main(["run", "no_such_entry"])
    assert code == 2 and err.startswith("REFUSED: "), err


def test_cli_writes_json_and_rendered_trace():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "result.json")
        code, out, err = _main(["run", "sk", "--param", "beta=2", "--param", "h=0",
                                "--out", path])
        assert code == 0, err
        with open(path) as f:
            res = json.load(f)
    assert res["rigor"] == "physics-level" and res["route"] == "replica"
    assert res["stable"] is False and res["final"] is False
    assert float(res["values"]["q"]) > 0.4 and res["values"]["ansatz"] == "rs"
    assert res["report"]["kind"] == "catalog" and res["report"]["adapted"] == []
    assert res["trace"]["steps"] == [s.name for s in catalog()["sk"].steps]
    assert any("NOT final" in n for n in res["notes"])
    assert res["finality"]["q"] == {**res["finality"]["q"], "state": "unstable", "final": False}
    assert res["report"]["relative_to"].startswith("catalog entry 'sk' itself")
    assert res["params"] == {"beta": 2, "h": 0}
    # the rendered trace goes to stderr, with citations
    assert "sk-rs-saddle [Sherrington" in err and "1. " in err, err


def test_cli_json_spec_composes_and_prints_to_stdout():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "spec.json")
        with open(path, "w") as f:
            json.dump({"family": "sk", "variables": "ising", "disorder": "gaussian",
                       "params": {"beta": None, "h": None}}, f)
        code, out, err = _main(["run", path, "--param", "beta=2", "--param", "h=0"])
    assert code == 0, err
    res = json.loads(out)
    assert res["report"]["kind"] == "catalog" and res["stable"] is False
    assert {r["key"] for r in res["references"]} == {"tc", "rs_ground_energy",
                                                     "rsb1_ground_energy"}
    assert all(r["ok"] for r in res["references"])


def test_cli_param_parsing():
    assert parse_param("alpha=0.1") == ("alpha", "0.1")       # decimal string, no float
    assert parse_param("k=2") == ("k", 2)
    assert parse_param("learning=bayes") == ("learning", "bayes")
    assert parse_param("ys=0.3,0.5,0.7") == ("ys", ("0.3", "0.5", "0.7"))
    assert parse_param("ys=0.3,") == ("ys", ("0.3",))           # trailing comma: a 1-list
    assert parse_param("require_stable=true") == ("require_stable", True)


def test_cli_unexpected_error_prints_ERROR_and_exits_1():
    code, out, err = _main(["run", "sk", "--param", "beta=abc", "--param", "h=0"])
    assert code == 1 and err.startswith("ERROR: ") and "Traceback" not in err, err


def test_cli_unknown_parameter_is_refused_with_exit_2():
    """Final review, Important 1: a typo'd or undeclared parameter is REFUSED, not
    ignored (the reproduction ran RS at kappa = 0 and reported final: true)."""
    # (ansatz is no longer an unknown name since E2: --param ansatz=1rsb selects the level-(a)
    # 1RSB route, test_cli_ansatz_param_selects_the_1rsb_route; "variant" takes its place)
    code, out, err = _main(["run", "perceptron", "--param", "alpha=1.5", "--param",
                            "variant=1rsb", "--param", "kapa=0.3"])
    assert code == 2 and out == "", (code, out)
    assert err.startswith("REFUSED: precondition:known-parameter: "), err
    assert "kapa, variant" in err and "accepted: alpha, kappa" in err, err


@slow
def test_cli_1rsb_route_refuses_unknown_parameters():
    """Fix round 1, Important 3: the 1RSB route refuses an unread name (composed and catalog
    specs alike) instead of noting it."""
    import tempfile
    spec = {"family": "perceptron", "variables": "ising", "disorder": "gaussian",
            "constraint": "margin kappa", "params": {"alpha": None}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(spec, f)
    try:
        code, out, err = _main(["run", f.name, "--param", "ansatz=1rsb", "--param",
                                "frozen=true", "--param", "solve_for=alpha", "--param",
                                "kappa=0", "--param", "kapa=0.3"])
    finally:
        os.unlink(f.name)
    assert code == 2 and err.startswith("REFUSED: precondition:known-parameter"), err
    assert "kapa" in err and "accepted: kappa" in err, err
    code, out, err = _main(["run", "sk", "--param", "ansatz=1rsb", "--param", "beta=2",
                            "--param", "h=0", "--param", "hh=5"])
    assert code == 2 and err.startswith("REFUSED: precondition:known-parameter"), err
    assert "hh" in err and "accepted: beta, h" in err, err


def test_cli_ansatz_param_selects_the_1rsb_route():
    code, out, err = _main(["run", "sk", "--param", "ansatz=2rsb"])
    assert code == 2 and err.startswith("REFUSED: missing-step:ansatz:2rsb"), err
    code, out, err = _main(["run", "sk", "--param", "ansatz=1rsb", "--param", "beta=2"])
    assert code == 2 and err.startswith("REFUSED: precondition:complete-point"), err


def test_selftest_summary_line_states_the_full_tier_cost():
    from replicant.selftest import summary_line
    line = summary_line(235, 0, 68)
    assert line.startswith("replicant selftest: 235 passed, 0 failed, 68 skipped "), line
    assert "--full" in line and "18 minutes" in line, line


@slow
def test_cli_module_entry_point_in_subprocess():
    p = subprocess.run([sys.executable, "-m", "replicant", "list"], cwd=TOOLS,
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0 and "sk" in p.stdout, p.stderr
    p = subprocess.run([sys.executable, "-m", "replicant", "run", "pspin", "--route", "cavity"],
                       cwd=TOOLS, capture_output=True, text=True, timeout=60)
    assert p.returncode == 2 and p.stderr.startswith("REFUSED: missing-step:route:cavity"), p
