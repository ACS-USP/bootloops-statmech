"""Banked reference values: cited numbers a replica result is checked against.

`load("entry")` reads `reference/<entry>.json` (see reference/_schema.md).
An entry with no citation or no independent route is refused, never used.
Values and tolerances are decimal strings parsed with mpmath, not floats.
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


def load(entry: str, *, root: Path | None = None) -> dict[str, RefValue]:
    path = (root or _ROOT) / f"{entry}.json"
    raw = json.loads(path.read_text())
    out = {}
    for key, e in raw.items():
        if not e.get("cite"):
            raise Refusal("unbanked:no-citation", f"{entry}/{key}")
        if not e.get("independent_route"):
            raise Refusal("unbanked:no-independent-route", f"{entry}/{key}")
        out[key] = RefValue(key, e["value"], e["tol"], e["cite"],
                            e["independent_route"],
                            tuple(e.get("sources_disagree", ())))
    return out


def check(ref: RefValue, got, *, label: str) -> CheckResult:
    """ok if `got` is within tol of the main value or of any disagreeing
    source's value (the tolerance covers the spread).  `got` may be a decimal
    string or a number; a float carries its own rounding error."""
    old = mp.dps
    mp.dps = _DPS
    try:
        g, tol = mpf(str(got)), mpf(ref.tol)
        wants = [ref.value] + [s["value"] for s in ref.sources_disagree]
        ok = any(abs(g - mpf(w)) <= tol for w in wants)
        return CheckResult(ok, str(got), ref.value, ref.tol)
    finally:
        mp.dps = old
