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
looked up in the catalog.  The report is relative to the level-(a) derivation of the
nearest entry's spec (report.relative_to): both step lists are built (level_a_steps) and,
step by step, the registry objects each step's code resolves (hooks_used: the Family
hook it calls, the SITES entry, the LAWS entry, the constraint / prior the family hook
implements, the solver's discrete-variable rule) are compared.  A step is ADAPTED when
any of them differs (the Adaptation names the hook now used, e.g. disorder:pm_one,
site:ising) and REUSED otherwise.  A hook that does not exist is a Refusal, never a partial result:
  missing-step:family:<f>        no derive.FAMILIES hook (detail: what exists, nearest)
  missing-step:site:<v>          the family has no hook for these variables
  missing-step:disorder:<d>      the family's average does not handle this law
  missing-step:constraint:<c>, missing-step:prior:<p>   not implemented by the hooks
  missing-step:route:<r>         unknown route, a catalog entry without that route, or
                                 cavity for a composed spec (there is no generic cavity
                                 composer)
The detail wraps the underlying step's own refusal reason where there is one.

Results.  Result.rigor is always "physics-level".  Finality is per value
(Result.finality: key -> catalog.Stability): catalog values carry their entry's
declaration; composed values the solver's computed flag (True -> stable, False ->
unstable, None -> not-computed).  Only "stable" and "exact-by-literature" are final;
every other value gets a "NOT final (<state>)" note, Result.final = all values final,
and run(require_final=True) refuses if ANY value is not final: unstable:<ansatz>, else
unknown-stability:rs (RS, stability not computed), else not-final:<ansatz>.
"""
from dataclasses import dataclass, field

import sympy as sp

from . import derive as D
from . import solve_derived as SD
from .catalog import CatalogEntry, catalog, compare, from_flag, run_cached
from .core import Refusal, State, Step, Trace, run_pipeline
from .solve_derived import lift_numeric_beta, solve_derived_rs
from .spec import ModelSpec

ROUTES = ("replica", "cavity")
FIELDS = ("family", "variables", "disorder", "constraint", "prior")
LEVEL_A_STEPS = ("replicate", "average-disorder", "introduce-order-parameters", "rs-ansatz",
                 "replica-limit", "rs-stationarity", "solve-derived")
_HOOK = {"disorder": "disorder", "variables": "site", "constraint": "constraint",
         "prior": "prior", "family": "family"}
LEVEL_A_CITE = "replicant level (a) derivation (this package)"


def hooks_used(step, spec):
    """{spec field: the registry object step `step` resolves for `spec`}: what the
    step's code looks up (derive.replicate: Family.formal and the SITES entry it checks;
    average_disorder: Family.average, which reads LAWS and implements the declared
    constraint / prior; introduce_order_parameters: Family.order_parameters and SITES;
    rs_ansatz: the site's reduction rule; solve-derived: the discrete-variable rule)."""
    fam, site = D.FAMILIES[spec.family], D.SITES.get(spec.variables)
    return {"replicate": {"family": fam.formal, "variables": site},
            "average-disorder": {"family": fam.average,
                                 "disorder": D.LAWS.get(spec.disorder, spec.disorder),
                                 "constraint": spec.constraint, "prior": spec.prior},
            "introduce-order-parameters": {"family": fam.order_parameters, "variables": site},
            "rs-ansatz": {"variables": site},
            "replica-limit": {},
            "rs-stationarity": {},
            "solve-derived": {"variables": spec.variables in SD._DISCRETE}}[step]
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
    relative_to: str = ""           # what reused / adapted are measured against

    def to_json(self):
        return {"kind": self.kind, "nearest": self.nearest, "relative_to": self.relative_to,
                "reused": list(self.reused),
                "adapted": [vars(a) for a in self.adapted], "missing": list(self.missing)}


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
    report: CompositionReport = None
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
        res = Result(dict(values), trace, errors=dict(state["errors"]),
                     notes=[self.entry.caveats[self.route], *state["notes"]], route=self.route,
                     statistical=tuple(state["statistical"]), report=self.report, params=p,
                     entry=self.entry, spec=self.spec)
        return _finalise(res, self.entry.finality_of(values), require_final)

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
                     params={**point, **knobs}, spec=self.spec)
        rep = "" if sol.replicon is None else f"derived replicon {sol.replicon}"
        fin = {k: from_flag(sol.stable, "rs", LEVEL_A_CITE, rep or "replicon not derived")
               for k in values if k not in ("branch", "ansatz", "stable")}
        return _finalise(res, fin, require_final)


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
                 solve, LEVEL_A_CITE, lambda s: s["solution"].trace.entries[-1].summary,
                 D._tex("solve-derived"))]


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
                [s.name for s in steps], [], [], e.name, "catalog",
                f"catalog entry {e.name!r} itself ({route} pipeline)"), e)
    near = nearest_entry(spec)
    _check_hooks(spec, near)
    if route != "replica":
        raise Refusal(f"missing-step:route:{route}",
                      "a spec outside the catalog is composed from level-(a) replica steps "
                      "only; the cavity pipelines exist for catalog entries "
                      "(cavity/dense.py, cavity/sparse.py), no generic cavity composer")
    adapted, reused = [], []
    steps = level_a_steps(spec)
    if near is None:            # a level-(a) family with no catalog entry: nothing to compare
        relative = f"nothing (no catalog entry of family {spec.family!r})"
    else:
        relative = (f"level-(a) derivation of catalog entry {near.name!r} (spec "
                    f"{dict(zip(FIELDS, _key(near.spec)))})")
        theirs = level_a_steps(near.spec)
        assert [s.name for s in theirs] == [s.name for s in steps] == list(LEVEL_A_STEPS)
        for st in steps:
            mine, other = hooks_used(st.name, spec), hooks_used(st.name, near.spec)
            diffs = [f for f in mine if mine[f] != other[f]]
            for f in diffs:
                adapted.append(Adaptation(st.name, f, getattr(near.spec, f), getattr(spec, f),
                                          f"{_HOOK[f]}:{getattr(spec, f)}"))
            if not diffs:
                reused.append(st.name)
    report = CompositionReport(reused, adapted, [], near.name if near else None, "composed",
                               relative)
    return ComposedCalculation(spec, route, steps, report)


__all__ = ["Adaptation", "CompositionReport", "ComposedCalculation", "LEVEL_A_STEPS",
           "Result", "compose", "hooks_used", "level_a_steps", "nearest_entry"]
