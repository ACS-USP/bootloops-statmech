"""Core: Refusal, Step, State, Trace, run_pipeline.

A calculation is a sequence of Steps over an immutable State.  Each step
names its preconditions (`requires` returns the missing ones); a missing
precondition is a named Refusal, never a silent fallback.  Every applied
step is recorded in the Trace with its citation.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable, Sequence


class Refusal(Exception):
    """A named, fail-closed refusal; `reason` is kebab-case, e.g.
    `precondition:gaussian_disorder`."""

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


class State(Mapping):
    """Immutable mapping; `with_` returns an updated copy."""

    def __init__(self, data: Mapping | None = None, **kw):
        self._d = {**(data or {}), **kw}

    def __getitem__(self, k):
        return self._d[k]

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def with_(self, **kw) -> "State":
        return State(self._d, **kw)

    def __repr__(self):
        return f"State({self._d!r})"


@dataclass(frozen=True)
class Step:
    name: str
    requires: Callable[[State], list[str]]
    apply: Callable[[State], State]
    cite: str | None = None


@dataclass(frozen=True)
class TraceEntry:
    step: str
    cite: str | None
    summary: str


class Trace:
    def __init__(self):
        self.entries: list[TraceEntry] = []

    def add(self, entry: TraceEntry) -> None:
        self.entries.append(entry)

    def render(self) -> str:
        lines = []
        for i, e in enumerate(self.entries, 1):
            cite = f" [{e.cite}]" if e.cite else ""
            lines.append(f"{i}. {e.step}{cite}: {e.summary}")
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {"steps": [e.step for e in self.entries],
                "entries": [{"step": e.step, "cite": e.cite,
                             "summary": e.summary} for e in self.entries]}


def run_pipeline(steps: Sequence[Step], state: State) -> tuple[State, Trace]:
    trace = Trace()
    for step in steps:
        missing = step.requires(state)
        if missing:
            raise Refusal(f"precondition:{missing[0]}",
                          f"step {step.name!r} needs {', '.join(missing)}")
        new = step.apply(state)
        added = sorted(set(new) - set(state))
        changed = sorted(k for k in state if k in new and new[k] is not state[k])
        trace.add(TraceEntry(step.name, step.cite,
                             "set " + ", ".join(added + changed) if added or changed
                             else "no change"))
        state = new
    return state, trace
