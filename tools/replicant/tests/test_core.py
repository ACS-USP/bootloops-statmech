import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.core import Refusal, State, Step, run_pipeline


def _step(name, key, cite=None, needs=()):
    return Step(name,
                lambda s: [n for n in needs if n not in s],
                lambda s: s.with_(**{key: True}),
                cite)


def test_pipeline_records_trace_in_order():
    _, trace = run_pipeline([_step("a", "x"), _step("b", "y")], State())
    assert trace.to_json()["steps"] == ["a", "b"]


def test_failed_precondition_refuses_with_named_reason():
    steps = [_step("a", "x", needs=("gaussian_disorder",)), _step("b", "y")]
    try:
        run_pipeline(steps, State())
    except Refusal as r:
        assert r.reason == "precondition:gaussian_disorder"
    else:
        raise AssertionError("expected Refusal")
    ran = []
    spy = Step("spy", lambda s: [], lambda s: ran.append(1) or s)
    try:
        run_pipeline([_step("a", "x", needs=("g",)), spy], State())
    except Refusal:
        pass
    assert not ran


def test_render_contains_citations():
    _, trace = run_pipeline([_step("a", "x", cite="Gardner 1988")], State())
    assert "Gardner 1988" in trace.render()
