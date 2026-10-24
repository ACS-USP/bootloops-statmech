"""Core: Refusal, Step, State, Trace, run_pipeline.

A calculation is a sequence of Steps over an immutable State.  Each step
names its preconditions (`requires` returns the missing ones); a missing
precondition is a named Refusal, never a silent fallback.  Every applied
step is recorded in the Trace with its citation.  A Refusal raised by run_pipeline
(a missing precondition, or one raised inside a step) carries the partial Trace of the
steps applied before it (`refusal.trace`) and the refusing step's name
(`refusal.step`; the outermost pipeline's, a nested one's is `refusal.inner_trace`),
so a refused calculation can still be written out.  A Step may carry
`tex(old_state, new_state) -> list[tex.TexBlock]`: the equations it displays; the
TraceEntry keeps it and renders the blocks only when asked (`entry.tex`).
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Callable, Sequence


class Refusal(Exception):
    """A named, fail-closed refusal; `reason` is kebab-case, e.g.
    `precondition:gaussian_disorder`."""

    def __init__(self, reason: str, detail: str = "", *, trace=None, step=None):
        self.reason = reason
        self.detail = detail
        self.trace = trace      # the partial Trace (steps applied before), if known
        self.step = step        # the name of the step that refused, if known
        self.inner_trace = None  # a nested pipeline's partial trace, if any
        self.route = None       # the route of the calculation, set by compose (writers)
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
    # Optional: a sentence for the trace, computed from the new state
    # (e.g. which order an expansion dropped).
    note: Callable[[State], str] | None = None
    # Optional: the equations this step displays, (old, new) -> list[tex.TexBlock]
    tex: Callable[[State, State], list] | None = None


@dataclass(frozen=True)
class TraceEntry:
    step: str
    cite: str | None
    summary: str
    render_tex: Callable[[], list] | None = field(default=None, compare=False, repr=False)

    @property
    def tex(self) -> list:
        """The step's TexBlocks (rendered now; [] for a step without a tex hook)."""
        return list(self.render_tex()) if self.render_tex else []


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
        try:
            missing = step.requires(state)
            if missing:
                raise Refusal(f"precondition:{missing[0]}",
                              f"step {step.name!r} needs {', '.join(missing)}")
            new = step.apply(state)
        except Refusal as r:
            # the outermost pipeline's partial trace is the calculation's; a nested
            # pipeline's (e.g. a solver's) is kept as inner_trace
            if r.trace is not None and getattr(r, "inner_trace", None) is None:
                r.inner_trace = r.trace
            r.trace, r.step = trace, step.name
            raise
        added = sorted(set(new) - set(state))
        changed = sorted(k for k in state if k in new and new[k] is not state[k])
        summary = ("set " + ", ".join(added + changed) if added or changed
                   else "no change")
        if step.note:
            summary += "; " + step.note(new)
        tex = (lambda f=step.tex, old=state, cur=new: f(old, cur)) if step.tex else None
        trace.add(TraceEntry(step.name, step.cite, summary, tex))
        state = new
    return state, trace
