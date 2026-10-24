"""Composer: match a ModelSpec against the catalog, or assemble its calculation from
level-(a) steps, and say what was reused, adapted, or missing.

  nearest_entry(spec) -> CatalogEntry | None
  compose(spec, route="replica") -> ComposedCalculation(steps, report)
  ComposedCalculation.run(*, require_final=False, **params) -> Result

Matching.  A spec IS a catalog entry when its structural key (family, variables,
disorder, constraint, prior) equals the entry's; spec.params are run parameters (numeric
ones become defaults of run), not part of the match.  nearest_entry: among entries of
the same family, the one agreeing on most of (constraint, prior), ties broken by most of
(variables, disorder), then by catalog order; no entry of that family -> None.

Catalog spec: the entry's pipeline for the route (level (b), or its cavity route) is
reused whole: report.reused = its step names, adapted = missing = [].

Any other spec: the calculation is assembled from the level-(a) steps
  replicate -> average-disorder -> introduce-order-parameters -> rs-ansatz ->
  replica-limit -> rs-stationarity -> solve-derived
through the registered hooks (derive.FAMILIES, derive.SITES, derive.LAWS); nothing is
looked up in the catalog.  Each step declares the spec fields it consumes (STEP_INPUTS);
a step is ADAPTED when one of those fields differs from the nearest entry's spec (the
Adaptation names the hook now used, e.g. disorder:pm_one, site:ising) and REUSED
otherwise.  A hook that does not exist is a Refusal, never a partial result:
  missing-step:family:<f>        no derive.FAMILIES hook (detail: what exists, nearest)
  missing-step:site:<v>          the family has no hook for these variables
  missing-step:disorder:<d>      the family's average does not handle this law
  missing-step:constraint:<c>, missing-step:prior:<p>   not implemented by the hooks
  missing-step:route:<r>         unknown route, a catalog entry without that route, or
                                 cavity for a composed spec (there is no generic cavity
                                 composer)
The detail wraps the underlying step's own refusal reason where there is one.

Results.  Result.rigor is always "physics-level".  An answer from a saddle point is
FINAL only when its stability was established (stable is True); stable False or None
(unknown, e.g. replicon not derived) is stated in Result.notes as "NOT final", and
run(require_final=True) refuses with unstable:<ansatz> / unknown-stability:<ansatz>.
"""
from dataclasses import dataclass, field

import sympy as sp

from . import derive as D
from .catalog import CatalogEntry, catalog, compare, run_cached
from .core import Refusal, State, Step, Trace, run_pipeline
from .solve_derived import lift_numeric_beta, solve_derived_rs
from .spec import ModelSpec

ROUTES = ("replica", "cavity")
FIELDS = ("family", "variables", "disorder", "constraint", "prior")
LEVEL_A_STEPS = ("replicate", "average-disorder", "introduce-order-parameters", "rs-ansatz",
                 "replica-limit", "rs-stationarity", "solve-derived")
# spec fields each level-(a) step consumes (the hook it calls depends on them)
STEP_INPUTS = {"replicate": ("family",),
               "average-disorder": ("disorder", "constraint", "prior"),
               "introduce-order-parameters": ("variables",),
               "rs-ansatz": ("variables",),
               "replica-limit": (),
               "rs-stationarity": (),
               "solve-derived": ("variables",)}     # discrete-variable entropy check
_HOOK = {"disorder": "disorder", "variables": "site", "constraint": "constraint",
         "prior": "prior", "family": "family"}
_SOLVER_KNOBS = ("branch", "quad", "dps", "seed", "seeds", "gh_nodes", "audit_tol")


@dataclass(frozen=True)
class Adaptation:
    step: str
    field: str
    was: object
    now: object
    hook: str

    def __str__(self):
        return f"{self.step} ({self.field}: {self.was} -> {self.now}; hook {self.hook})"


@dataclass(frozen=True)
class CompositionReport:
    """reused / adapted: step names / Adaptations (module docstring).  missing is always
    [] on a returned calculation: a step the composer cannot supply is a Refusal
    (missing-step:...), never a partial calculation; the field is kept for the interface."""
    reused: list
    adapted: list
    missing: list
    nearest: str | None = None
    kind: str = "catalog"           # "catalog" | "composed"

    def to_json(self):
        return {"kind": self.kind, "nearest": self.nearest, "reused": list(self.reused),
                "adapted": [vars(a) for a in self.adapted], "missing": list(self.missing)}


@dataclass
class Result:
    values: dict
    trace: Trace
    rigor: str = "physics-level"
    errors: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    route: str = "replica"
    stable: bool | None = None
    final: bool | None = False      # None: headline quantities only, no saddle to judge
    statistical: tuple = ()         # value keys estimated by sampling (stderr in errors)
    report: CompositionReport = None
    params: dict = field(default_factory=dict)
    entry: CatalogEntry = None

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


def _key(spec):
    return tuple(getattr(spec, f) for f in FIELDS)


def nearest_entry(spec: ModelSpec):
    best, score = None, None
    for e in catalog().values():
        if e.spec.family != spec.family:
            continue
        s = (sum(getattr(e.spec, f) == getattr(spec, f) for f in ("constraint", "prior")),
             sum(getattr(e.spec, f) == getattr(spec, f) for f in ("variables", "disorder")))
        if score is None or s > score:
            best, score = e, s
    return best


def _not_final(stable, ansatz, detail=""):
    why = "unstable" if stable is False else "unknown"
    text = (f"the {ansatz.upper()} saddle is unstable here (stable=False)" if stable is False
            else "its stability is unknown (stable=None: not established by this pipeline)")
    return why, f"{ansatz.upper()} answer is NOT final: {text}" + (f"; {detail}" if detail else "")


def _finalise(res: Result, require_final, detail=""):
    ansatz = res.values.get("ansatz")
    if ansatz is None:              # headline quantities only: no saddle to judge
        res.final = res.stable = None
        return res
    res.stable = res.values.get("stable")
    res.final = res.stable is True
    if not res.final:
        why, text = _not_final(res.stable, ansatz, detail)
        res.notes.append(text)
        if require_final:
            reason = "unstable" if why == "unstable" else "unknown-stability"
            raise Refusal(f"{reason}:{ansatz}", text)
    return res


@dataclass
class ComposedCalculation:
    spec: ModelSpec
    route: str
    steps: list
    report: CompositionReport
    entry: CatalogEntry = None

    def run(self, *, require_final=False, **params) -> Result:
        if self.report.kind == "catalog":
            return self._run_catalog(require_final, params)
        return self._run_composed(require_final, params)

    def _run_catalog(self, require_final, params):
        p = {**{k: v for k, v in self.spec.params.items()
                if not (isinstance(v, sp.Basic) and v.free_symbols)}, **params}
        state, trace = self.entry.execute(self.route, **p)
        values = state["values"]
        detail = (f"replicon {values['replicon']}" if values.get("replicon") is not None
                  else "")
        res = Result(dict(values), trace, errors=dict(state["errors"]),
                     notes=[self.entry.caveats[self.route], *state["notes"]], route=self.route,
                     statistical=tuple(state["statistical"]), report=self.report, params=p,
                     entry=self.entry)
        return _finalise(res, require_final, detail)

    def _run_composed(self, require_final, params):
        # self.steps were built for self.spec; a numeric beta is lifted to the symbol
        # first (so the entropy can be derived), and the same builder makes the steps run
        knobs = {k: params.pop(k) for k in _SOLVER_KNOBS if k in params}
        spec, point, lift = lift_numeric_beta(self.spec, dict(params))
        state, trace = run_pipeline(level_a_steps(spec, point, knobs), State())
        sol = state["solution"]
        # phi = lim (1/(N n)) ln E Z^n at the RS saddle: -beta f for a thermal model, the
        # entropy (log volume / count per variable) for the perceptron (solve_derived.py)
        values = {**sol.values, "phi": sol.free_energy, "entropy": sol.entropy,
                  "replicon": sol.replicon, "branch": sol.branch, "ansatz": "rs",
                  "stable": sol.stable}
        res = Result(values, trace, errors={"residual": sol.residual,
                                            "quad_error": sol.quad_error},
                     notes=[sol.caveat, *lift, sol.note], route="replica", report=self.report,
                     params={**point, **knobs})
        return _finalise(res, require_final)


def level_a_steps(spec, point=None, knobs=None):
    """The seven composed Steps for `spec`; the last solves the derived equations at
    `point` (solve_derived_rs, solver `knobs`)."""
    n = sp.Symbol("n", positive=True)

    def solve(s):
        sol = solve_derived_rs(spec, derived=(s["rs_equations"], s, Trace()),
                               **(knobs or {}), **(point or {}))
        return s.with_(solution=sol)

    return [D.replicate(spec, n), D.average_disorder(), D.introduce_order_parameters(),
            D.rs_ansatz(), D.replica_limit(), D.rs_stationarity(),
            Step("solve-derived", lambda s: [] if "rs_equations" in s else ["rs_equations"],
                 solve, None, lambda s: s["solution"].trace.entries[-1].summary)]


# ---- compose ------------------------------------------------------------------------
def _families():
    return ", ".join(sorted(D.FAMILIES))


def _check_hooks(spec, near):
    """Refuse with missing-step:<hook> for any level-(a) hook the spec needs and lacks."""
    where = f"; nearest catalog entry: {near.name!r}" if near else "; no catalog entry of " \
        "that family"
    fam = D.FAMILIES.get(spec.family)
    if fam is None:
        raise Refusal(f"missing-step:family:{spec.family}",
                      f"not a catalog spec, and level (a) has no Family hook for "
                      f"{spec.family!r} (derive.FAMILIES: {_families()}){where}")
    try:
        D.replicate(spec, sp.Symbol("n", positive=True)).requires(State())
    except Refusal as r:
        raise Refusal(f"missing-step:site:{spec.variables}", f"{r.reason}: {r.detail}{where}")
    if not fam.supports_disorder(spec.disorder):
        try:
            D._law(spec)
            under = f"{spec.family}'s Family.average does not handle {spec.disorder!r}"
        except Refusal as r:
            under = r.detail
        raise Refusal(f"missing-step:disorder:{spec.disorder}",
                      f"precondition:supported-disorder-law: {under}{where}")
    for f, allowed in (("constraint", fam.constraints), ("prior", fam.priors)):
        v = getattr(spec, f)
        if v not in allowed:
            raise Refusal(f"missing-step:{f}:{v}",
                          f"the {spec.family} level-(a) hooks implement {f} in {allowed}; "
                          f"{v!r} would be silently ignored by them{where}")


def compose(spec: ModelSpec, route: str = "replica") -> ComposedCalculation:
    if route not in ROUTES:
        raise Refusal(f"missing-step:route:{route}", f"known routes: {', '.join(ROUTES)}")
    for e in catalog().values():
        if _key(e.spec) == _key(spec):
            if route not in e.routes:
                raise Refusal(f"missing-step:route:{route}",
                              f"catalog entry {e.name!r} ({spec.family}) has routes "
                              f"{', '.join(sorted(e.routes))}")
            steps = list(e.route_steps[route])
            return ComposedCalculation(spec, route, steps, CompositionReport(
                [s.name for s in steps], [], [], e.name, "catalog"), e)
    near = nearest_entry(spec)
    _check_hooks(spec, near)
    if route != "replica":
        raise Refusal(f"missing-step:route:{route}",
                      "a spec outside the catalog is composed from level-(a) replica steps "
                      "only; the cavity pipelines exist for catalog entries "
                      "(cavity/dense.py, cavity/sparse.py), no generic cavity composer")
    adapted, reused = [], []
    for step in LEVEL_A_STEPS:
        diffs = [f for f in STEP_INPUTS[step]
                 if near is not None and getattr(near.spec, f) != getattr(spec, f)]
        for f in diffs:
            adapted.append(Adaptation(step, f, getattr(near.spec, f), getattr(spec, f),
                                      f"{_HOOK[f]}:{getattr(spec, f)}"))
        if not diffs:
            reused.append(step)
    if near is None:            # a level-(a) family with no catalog entry: nothing reused
        reused = []
    report = CompositionReport(reused, adapted, [], near.name if near else None, "composed")
    return ComposedCalculation(spec, route, level_a_steps(spec), report)


__all__ = ["Adaptation", "CompositionReport", "ComposedCalculation", "LEVEL_A_STEPS",
           "Result", "STEP_INPUTS", "compose", "level_a_steps", "nearest_entry"]
