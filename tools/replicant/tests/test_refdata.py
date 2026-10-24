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
    assert check(ref, "0.76335", label="x").ok  # 5e-5 from alternate, 1.5e-4 from main
    assert not check(ref, "0.7640", label="x").ok  # near neither
    bare = load("t", root=_bank(_GOOD))["k"]
    assert not check(bare, "0.76335", label="x").ok


def test_tolerance_edge_is_inclusive():
    ref = load("t", root=_bank(_GOOD))["k"]
    assert check(ref, "0.7633", label="x").ok
    assert not check(ref, "0.76331", label="x").ok


def test_check_result_carries_label():
    ref = load("t", root=_bank(_GOOD))["k"]
    assert check(ref, "0.7632", label="lbl").label == "lbl"


def test_whitespace_cite_is_missing():
    assert _refusal({**_GOOD, "cite": "  "}) == "unbanked:no-citation"
    assert _refusal({**_GOOD, "independent_route": " "}) == "unbanked:no-independent-route"


def _malformed(entry):
    assert _refusal(entry) == "unbanked:malformed"


def test_malformed_entries_refuse():
    _malformed({k: v for k, v in _GOOD.items() if k != "value"})
    _malformed({k: v for k, v in _GOOD.items() if k != "tol"})
    _malformed("not a dict")
    _malformed({**_GOOD, "value": "abc"})
    _malformed({**_GOOD, "tol": "x"})
    _malformed({**_GOOD, "tol": "-1e-4"})
    _malformed({**_GOOD, "sources_disagree": [{"value": "zz", "cite": "c"}]})
    _malformed({**_GOOD, "sources_disagree": [{"cite": "c"}]})
    _malformed({**_GOOD, "sources_disagree": "nope"})


def test_missing_file_and_bad_json_refuse():
    root = Path(tempfile.mkdtemp())
    for setup, reason in ((None, "unbanked:missing-entry"), ("{", "unbanked:malformed")):
        if setup:
            (root / "t.json").write_text(setup)
        try:
            load("t", root=root)
        except Refusal as r:
            assert r.reason == reason
        else:
            raise AssertionError("expected Refusal")


def test_modelspec_equality():
    from replicant.spec import ModelSpec
    a = ModelSpec("sk", "ising", "gaussian", params={"beta": "1"})
    assert a == ModelSpec("sk", "ising", "gaussian", params={"beta": "1"})
    assert a != ModelSpec("sk", "spherical", "gaussian")


def test_non_finite_numbers_refuse():
    # tol="inf" would make check() accept anything; nan never compares.
    for bad in ("inf", "+inf", "-inf", "nan"):
        _malformed({**_GOOD, "tol": bad})
        _malformed({**_GOOD, "value": bad})
        _malformed({**_GOOD, "sources_disagree": [{"value": bad, "cite": "c"}]})


def assert_tolerance_covers_spread(entry, *, root=None):
    """Data rule: every sources_disagree value lies within tol of the main value."""
    import mpmath as mp
    for key, ref in load(entry, root=root).items():
        with mp.workdps(30):
            main, tol = mp.mpf(ref.value), mp.mpf(ref.tol)
            for s in ref.sources_disagree:
                gap = abs(mp.mpf(s["value"]) - main)
                assert gap <= tol, f"{entry}/{key}: {s['value']} is {gap} from {ref.value} > tol {ref.tol}"


def test_banked_tolerances_cover_spread():
    for f in sorted((Path(__file__).parent.parent / "reference").glob("*.json")):
        assert_tolerance_covers_spread(f.stem)


def test_spread_helper_catches_narrow_tol():
    e = {**_GOOD, "sources_disagree": [{"value": "0.7634", "cite": "Other 2001"}]}
    try:
        assert_tolerance_covers_spread("t", root=_bank(e))
    except AssertionError:
        return
    raise AssertionError("helper accepted a tol narrower than the spread")
