"""Catalog route `amp` (E3b) for the `sk` and `teacher_student` entries, and its CLI.

The route runs amp/ (state evolution, and finite-N AMP / GAMP when N is given): SE
values carry `exact-by-literature` only where the cited paper proves SE for the
setting (SK: Bayati-Montanari 2011 Thm 4, AMP with a Lipschitz nonlinearity on a
Gaussian GOE-type matrix); GAMP's SE for the GLM is stated as a claim (Barbier et al.
2019, Claim 1, from Rangan 2011), so its values are `not-computed`; finite-N values are
statistical (mean +- stderr over named seeds, 99% Student-t half-width in the meta
values), never final.  The comparison targets (families/) are called only afterwards.
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.catalog import catalog
from replicant.cli import main
from replicant.compose import compose
from replicant.core import Refusal
from replicant.spec import ModelSpec
from replicant.testing import slow


def _refusal(fn):
    try:
        out = fn()
    except Refusal as r:
        return r
    raise AssertionError(f"expected Refusal, got {out!r}")


def _cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_amp_route_in_catalog_and_cli():
    cat = catalog()
    assert cat["sk"].routes == {"replica", "cavity", "amp"}
    assert cat["teacher_student"].routes == {"replica", "amp"}
    # list shows the route
    code, out, _ = _cli("list")
    assert code == 0
    for name in ("sk", "teacher_student"):
        line = next(l for l in out.splitlines() if l.startswith(name + " "))
        assert "amp" in line.split("routes=")[1].split()[0], line
    # sk: SE value exact-by-literature (BM11 Thm 4), no finite-N value without N
    code, out, err = _cli("run", "sk", "--route", "amp", "--param", "beta=1.5")
    assert code == 0, err
    doc = json.loads(out)
    from replicant.families.sk import sk_rs                     # target, after the run
    assert abs(float(doc["values"]["q_se"]) - float(sk_rs("1.5").q)) < 1e-12
    fin = doc["finality"]["q_se"]
    assert fin["state"] == "exact-by-literature" and fin["final"], fin
    assert "Bayati" in fin["cite"] and "Thm 4" in fin["cite"] and "Gibbs" in fin["note"], fin
    assert doc["route"] == "amp" and doc["statistical"] == [] and "q_amp" not in doc["values"]
    # teacher-student: GAMP SE is a claim, not a theorem: not-computed
    code, out, err = _cli("run", "teacher_student", "--route", "amp", "--param", "alpha=2",
                          "--param", "learning=bayes")
    assert code == 0, err
    doc = json.loads(out)
    from replicant.families.teacher_student import teacher_student_error
    assert abs(float(doc["values"]["eps_se"]) - float(teacher_student_error(2, "bayes"))) < 1e-6
    for k in ("q_se", "mse_se", "eps_se"):
        st = doc["finality"][k]
        assert st["state"] == "not-computed" and not st["final"], (k, st)
        assert "Claim 1" in st["note"] and "Rangan" in st["note"], st
    # refusals: unknown parameters (route-specific sets), missing state point, gibbs SE ok
    code, _, err = _cli("run", "sk", "--route", "amp", "--param", "beta=1.5", "--param",
                        "fullrsb=true")
    assert code == 2 and "precondition:known-parameter" in err and "fullrsb" in err, err
    code, _, err = _cli("run", "sk", "--route", "amp", "--param", "beta=1.5", "--param",
                        "ansatz=1rsb")
    # `ansatz` is the composer's switch (cli.build_result passes it to compose), so it is
    # refused by the composer, naming the route, before the entry's parameter check
    assert code == 2 and "missing-step:route:amp" in err and "no ansatz switch" in err, err
    code, _, err = _cli("run", "teacher_student", "--route", "amp", "--param", "alpha=2",
                        "--param", "prior=ising")
    assert code == 2 and "precondition:known-parameter" in err, err
    r = _refusal(lambda: compose(cat["sk"].spec, "amp").run())
    assert r.reason == "precondition:state-point" and "beta" in r.detail, r
    r = _refusal(lambda: compose(cat["teacher_student"].spec, "amp").run(alpha=1,
                                                                        learning="map"))
    assert r.reason == "precondition:known-learning", r
    g = compose(cat["teacher_student"].spec, "amp").run(alpha=2, learning="gibbs")
    assert abs(g.values["eps_se"] - teacher_student_error(2, "gibbs")) < 1e-6
    # entries without the route, and composed specs, refuse it
    r = _refusal(lambda: compose(cat["perceptron"].spec, "amp"))
    assert r.reason == "missing-step:route:amp", r
    r = _refusal(lambda: compose(ModelSpec("sk", "ising", "pm_one"), "amp"))
    assert r.reason == "missing-step:route:amp" and "catalog" in r.detail, r


@slow
def test_amp_route_finite_N_runs():
    """With N the route also runs finite-N AMP / GAMP over named seeds: mean +- stderr
    (errors), the 99% Student-t half-width and the seeds in the values, statistical and
    never final; the SE at the same t lies inside the band at these sizes."""
    cat = catalog()
    # sk finite N: mean +- stderr, a Student-t half-width, the named seeds; never final
    code, out, err = _cli("run", "sk", "--route", "amp", "--param", "beta=1.5", "--param",
                          "h=0.8", "--param", "N=300", "--param", "seeds=0,1,2,3",
                          "--param", "iters=60")
    assert code == 0, err
    doc = json.loads(out)
    v = doc["values"]
    assert doc["statistical"] == ["q_amp"] and float(doc["errors"]["q_amp"]) > 0
    assert v["seeds"] == [0, 1, 2, 3] and v["N"] == 300 and v["law"] == "gaussian"
    assert abs(float(v["q_amp"]) - float(v["q_se_t"])) <= float(v["q_amp_halfwidth99"])
    assert doc["finality"]["q_amp"]["state"] == "not-computed" and doc["final"] is False
    assert "statistical" in doc["finality"]["q_amp"]["note"]
    assert doc["finality"]["q_se_t"]["state"] == "exact-by-literature"
    # +-J: SE is not proved for it (the q_se_t label says so)
    res = compose(cat["sk"].spec, "amp").run(beta="1.5", h="0.8", N=200, seeds=(0, 1),
                                             iters=20, law="pm_one")
    assert res.finality["q_se_t"].state == "not-computed", res.finality["q_se_t"]
    res = compose(cat["teacher_student"].spec, "amp").run(alpha="1", N=200, seeds=(0, 1, 2),
                                                          iters=20)
    assert set(res.statistical) == {"q_gamp", "eps_gamp"} and res.final is False
    assert abs(res.values["q_gamp"] - res.values["q_se_t"]) <= res.values["q_gamp_halfwidth99"]


def test_amp_route_finite_N_parameters_need_N_and_labels_follow_the_law():
    """Fix round 1.  (a) A finite-N knob (sk: seeds, iters, law; teacher_student: seeds,
    iters) without N is refused precondition:known-parameter naming N, never echoed and
    ignored.  (b) law is validated even so (precondition:known-law).  (c) With a +-J law
    every SE label is not-computed (SE is proved for Gaussian J only), so nothing on that
    run is final; with the Gaussian law q_se stays exact-by-literature."""
    cat = catalog()
    sk, ts = compose(cat["sk"].spec, "amp"), compose(cat["teacher_student"].spec, "amp")
    for calc, p, knob in ((sk, {"beta": "1.5"}, {"law": "pm_one"}),
                          (sk, {"beta": "1.5"}, {"seeds": (1, 2)}),
                          (sk, {"beta": "1.5"}, {"iters": 10}),
                          (ts, {"alpha": 2}, {"seeds": (1, 2)}),
                          (ts, {"alpha": 2}, {"iters": 10})):
        r = _refusal(lambda: calc.run(**p, **knob))
        assert r.reason == "precondition:known-parameter", (knob, r)
        assert "need N" in r.detail and next(iter(knob)) in r.detail, r.detail
    code, _, err = _cli("run", "sk", "--route", "amp", "--param", "beta=1.5", "--param",
                        "law=bogus")
    assert code == 2 and "precondition:known-parameter" in err and "need N" in err, err
    r = _refusal(lambda: sk.run(beta="1.5", N=20, law="bogus"))
    assert r.reason == "precondition:known-law", r
    res = sk.run(beta="1.5", h="0.8", N=60, seeds=(0, 1), iters=5, law="pm_one")
    for k in ("q_se", "q_se_t", "q_amp"):
        assert res.finality[k].state == "not-computed", (k, res.finality[k])
    assert "+-J" in res.finality["q_se"].note and res.final is False
    g = sk.run(beta="1.5", h="0.8", N=60, seeds=(0, 1), iters=5)
    assert g.finality["q_se"].state == "exact-by-literature"


def test_touched_modules_leave_line_headroom():
    """Fix round 1 (file size): the three files the amp route touched stay <= 380 lines."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("catalog_entries.py", "compose.py", "latex.py"):
        with open(os.path.join(here, name)) as f:
            n = sum(1 for _ in f)
        assert n <= 380, (name, n)


def test_amp_route_malformed_seeds_are_refused():
    """T15 (E3b deferred minor): a single integer (`--param seeds=1`), an empty list or a
    negative seed is refused precondition:named-seeds on both amp entries (it used to end
    in ERROR TypeError, exit 1, and an empty list silently fell back to the default)."""
    for argv in (("sk", "--param", "beta=1.5"), ("teacher_student", "--param", "alpha=1")):
        code, _, err = _cli("run", *argv, "--route", "amp", "--param", "N=60", "--param",
                            "seeds=1")
        assert code == 2 and "REFUSED: precondition:named-seeds" in err, (argv, code, err)
    cat = catalog()
    sk, ts = compose(cat["sk"].spec, "amp"), compose(cat["teacher_student"].spec, "amp")
    for calc, p in ((sk, {"beta": "1.5"}), (ts, {"alpha": 1})):
        for bad in ((), (-1,), (0, "x")):
            r = _refusal(lambda: calc.run(**p, N=60, seeds=bad, iters=2))
            assert r.reason == "precondition:named-seeds", (bad, r)
