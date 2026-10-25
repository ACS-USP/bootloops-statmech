"""compose.Result (a run's values, finality, labels and trace) and the per-value finality
rule _finalise; split out of compose.py, which re-exports both (see its docstring for the
finality rules).
"""
from dataclasses import dataclass, field

from .catalog import CatalogEntry, compare, run_cached
from .core import Refusal, Trace
from .spec import ModelSpec


@dataclass
class Result:
    values: dict
    trace: Trace
    rigor: str = "physics-level"
    errors: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    route: str = "replica"
    stable: bool | None = None      # the saddle's computed flag (values["stable"]), if any
    final: bool | None = False      # all values final (None: no physical values)
    finality: dict = field(default_factory=dict)    # value key -> catalog.Stability
    statistical: tuple = ()         # value keys estimated by sampling (stderr in errors)
    report: object = None           # compose.CompositionReport
    params: dict = field(default_factory=dict)
    entry: CatalogEntry = None
    spec: ModelSpec = None          # the spec that was run (for writers, e.g. latex.py)

    def reference_checks(self, *, slow=False):
        """The entry's banked comparisons for this route (RefOutcome list); slow and
        statistical ones only with slow=True.  A check whose parameters are this run's
        reads this result's values; the others run the entry at their own parameters.
        Composed calculations have no banked keys: []."""
        if self.entry is None:
            return []
        out = []
        for ref in self.entry.refs:
            if ref.route != self.route or ((ref.slow or ref.statistical) and not slow):
                continue
            if dict(ref.params) == dict(self.params):
                out.append(compare(self.entry, ref, self.values, self.errors))
            else:
                out.append(compare(self.entry, ref, *run_cached(self.entry, ref.route,
                                                                ref.params)))
        return out


def _finalise(res: Result, finality, require_final):
    res.finality = finality
    res.stable = res.values.get("stable")
    res.final = all(st.final for st in finality.values()) if finality else None
    bad = {k: st for k, st in finality.items() if not st.final}
    for k, st in bad.items():
        res.notes.append(f"{k}: {st.ansatz.upper()} value is NOT final ({st.state}"
                         + (f"; {st.note}" if st.note else "") + ")")
    if require_final and bad:
        states = [(st.state, st.ansatz) for st in bad.values()]
        unstable = [a for st, a in states if st == "unstable"]
        if unstable:
            reason = f"unstable:{unstable[0]}"
        elif any(a == "rs" for _, a in states):
            reason = "unknown-stability:rs"
        else:
            reason = f"not-final:{states[0][1]}"
        raise Refusal(reason, "; ".join(f"{k}: {st.ansatz} {st.state}" + (f" ({st.note})"
                                                                           if st.note else "")
                                        for k, st in bad.items()),
                      trace=res.trace, step="require-final")
    return res


__all__ = ["Result"]
