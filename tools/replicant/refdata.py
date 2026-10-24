"""Banked reference values: cited numbers a replica result is checked against.

`load("entry")` reads `reference/<entry>.json` (see reference/_schema.md).
An entry with no citation or no independent route is refused, never used.
Values and tolerances are finite decimal strings parsed with mpmath, not floats.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from mpmath import mp, mpf

from replicant.core import Refusal

_ROOT = Path(__file__).parent / "reference"
_DPS = 30


@dataclass(frozen=True)
class RefValue:
    key: str
    value: str
    tol: str
    cite: str
    independent_route: str
    sources_disagree: tuple = ()  # of {"value": str, "cite": str}


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    got: str
    want: str
    tol: str
    label: str = ""


def _bad(entry, key, what):
    return Refusal("unbanked:malformed", f"{entry}/{key}: {what}")


def _num(entry, key, field, text):
    """Parse a decimal string with mpmath or refuse as malformed."""
    if not isinstance(text, str):
        raise _bad(entry, key, f"{field} must be a decimal string")
    try:
        with mp.workdps(_DPS):
            x = mpf(text)
    except (ValueError, TypeError):
        raise _bad(entry, key, f"{field} is not a number: {text!r}")
    if not mp.isfinite(x):   # tol="inf" would let check() accept anything
        raise _bad(entry, key, f"{field} must be finite: {text!r}")
    return x


def load(entry: str, *, root: Path | None = None) -> dict[str, RefValue]:
    path = (root or _ROOT) / f"{entry}.json"
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        raise Refusal("unbanked:missing-entry", str(path))
    except ValueError as exc:
        raise Refusal("unbanked:malformed", f"{entry}: bad JSON ({exc})")
    if not isinstance(raw, dict):
        raise Refusal("unbanked:malformed", f"{entry}: top level must be an object")
    out = {}
    for key, e in raw.items():
        if not isinstance(e, dict):
            raise _bad(entry, key, "entry must be an object")
        for f in ("cite", "independent_route"):
            v = e.get(f)
            if not (isinstance(v, str) and v.strip()):
                raise Refusal("unbanked:no-citation" if f == "cite"
                              else "unbanked:no-independent-route", f"{entry}/{key}")
        for f in ("value", "tol"):
            if f not in e:
                raise _bad(entry, key, f"missing {f}")
        _num(entry, key, "value", e["value"])
        if _num(entry, key, "tol", e["tol"]) < 0:
            raise _bad(entry, key, "tol must be >= 0")
        dis = e.get("sources_disagree", [])
        if not isinstance(dis, list):
            raise _bad(entry, key, "sources_disagree must be a list")
        for s in dis:
            if not isinstance(s, dict) or "value" not in s:
                raise _bad(entry, key, "sources_disagree item needs a value")
            _num(entry, key, "sources_disagree value", s["value"])
        out[key] = RefValue(key, e["value"], e["tol"], e["cite"],
                            e["independent_route"], tuple(dis))
    return out


def check(ref: RefValue, got, *, label: str) -> CheckResult:
    """ok if `got` is within tol of the main value or of any disagreeing
    source's value (the tolerance covers the spread).  `got` may be a decimal
    string or a number; a float carries its own rounding error."""
    with mp.workdps(_DPS):
        g, tol = mpf(str(got)), mpf(ref.tol)
        wants = [ref.value] + [s["value"] for s in ref.sources_disagree]
        ok = any(abs(g - mpf(w)) <= tol for w in wants)
    return CheckResult(ok, str(got), ref.value, ref.tol, label)
