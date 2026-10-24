import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.core import Refusal
from replicant.refdata import check, load


def _bank(entry: dict) -> Path:
    root = Path(tempfile.mkdtemp())
    (root / "t.json").write_text(json.dumps({"k": entry}))
    return root


_GOOD = {"value": "0.7632", "tol": "1e-4", "cite": "Some Paper 2000, Eq. 1",
         "independent_route": "series extrapolation"}


def _refusal(entry):
    try:
        load("t", root=_bank(entry))
    except Refusal as r:
        return r.reason
    raise AssertionError("expected Refusal")


def test_load_refuses_entry_without_cite():
    assert _refusal({k: v for k, v in _GOOD.items() if k != "cite"}) == "unbanked:no-citation"


def test_load_refuses_entry_without_independent_route():
    e = {k: v for k, v in _GOOD.items() if k != "independent_route"}
    assert _refusal(e) == "unbanked:no-independent-route"


def test_check_uses_tolerance():
    ref = load("t", root=_bank(_GOOD))["k"]
    assert check(ref, "0.76325", label="x").ok
    assert not check(ref, "0.7640", label="x").ok


def test_disagreeing_sources_widen_tolerance():
    e = {**_GOOD, "sources_disagree": [{"value": "0.7633", "cite": "Other 2001"}]}
    ref = load("t", root=_bank(e))["k"]
    assert [s["value"] for s in ref.sources_disagree] == ["0.7633"]
    assert check(ref, "0.7632", label="x").ok
    assert check(ref, "0.7633", label="x").ok
    assert not check(ref, "0.7700", label="x").ok
