"""Composer: match a ModelSpec against the catalog, or assemble its calculation from
level-(a) steps, and say what was reused, adapted, or missing.

  nearest_entry(spec) -> CatalogEntry | None
  compose(spec, route="replica", ansatz="rs") -> ComposedCalculation(steps, report)
  ComposedCalculation.run(*, require_final=False, **params) -> Result

Matching.  A spec IS a catalog entry when its structural key (family, variables,
disorder, constraint, prior) equals the entry's; spec.params are run parameters (numeric
ones become defaults of run), not part of the match.  nearest_entry: among entries of
the same family, the one agreeing on most of (constraint, prior), ties broken by most of
(variables, disorder), then by catalog order; no entry of that family -> None.

Catalog spec: the entry's pipeline for the route (level (b), or its cavity route) is
reused whole: report.reused = its step names, adapted = missing = [].  Its run
parameters (spec.params numbers + run(**params)) must be names the route reads
(CatalogEntry.parameters[route]); any other name is refused precondition:known-parameter
(detail: the offending and the accepted names), never silently ignored.  (A composed
calculation instead notes "unused parameters [...]", solve_derived.py.)

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
                                 cavity / amp for a composed spec (no generic composer)
The detail wraps the underlying step's own refusal reason where there is one.

ansatz="1rsb": the level-(a) 1RSB route for any spec with level-(a) hooks (compose_rsb.py:
report relative to the spec's own RS derivation; values not-computed); any other ansatz is
refused missing-step:ansatz:<a>.

Results.  Result.rigor is "physics-level", except for a catalog route its entry declares
"certified" (CatalogEntry.rigor): there every physical value must carry its ball radius in errors,
else the run is refused precondition:certified-radius; or "statistical" (the `sample` route of the
percolation and Monte Carlo entries): means over named seeds with stderr in errors, never certified, never
final.  Finality is per value (Result.finality: key -> catalog.Stability): catalog values carry their entry's
declaration; composed values the solver's computed flag (True -> stable, False ->
unstable, None -> not-computed), with a note naming the derived replicon and its rules
(derive_replicon: SK, the spherical and Ising perceptron and Hopfield have one; Hopfield
also the derived m-mode, solve_derived_stability).  Only "stable" and
"exact-by-literature" are final; every other value gets a "NOT final (<state>)" note,
Result.final = all values final, and run(require_final=True) refuses if ANY value is not
final: unstable:<ansatz>, else unknown-stability:rs (RS, stability not computed), else
not-final:<ansatz>.
"""
from dataclasses import dataclass

import mpmath as mp
import sympy as sp

from . import derive as D
from . import solve_derived as SD
from .catalog import CatalogEntry, catalog, from_flag
from .compose_result import Result, _finalise
from .core import Refusal, State, Step, Trace, run_pipeline
from .solve_derived import lift_numeric_beta, solve_derived_rs
from .spec import ModelSpec

ROUTES = ("replica", "cavity", "amp", "series", "exact", "bound", "online", "sample")
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


@dataclass
class ComposedCalculation:
    spec: ModelSpec
    route: str
    steps: list
    report: CompositionReport
    entry: CatalogEntry = None
    ansatz: str = "rs"

    def run(self, *, require_final=False, **params) -> Result:
        try:
            if self.ansatz == "1rsb":
                from .compose_rsb import run_1rsb
                return run_1rsb(self, require_final, dict(params))
            if self.report.kind == "catalog":
                return self._run_catalog(require_final, params)
            return self._run_composed(require_final, params)
        except Refusal as r:
            r.route = r.route or self.route      # a refused document names its route
            raise

    def _run_catalog(self, require_final, params):
        p = {**{k: v for k, v in self.spec.params.items()
                if not (isinstance(v, sp.Basic) and v.free_symbols)}, **params}
        accepted = self.entry.parameters.get(self.route, frozenset())
        unknown = sorted(set(p) - accepted)
        if unknown:
            raise Refusal("precondition:known-parameter",
                          f"catalog entry {self.entry.name!r} ({self.route} route) does not "
                          f"read {', '.join(unknown)}; accepted: {', '.join(sorted(accepted))}")
        state, trace = self.entry.execute(self.route, **p)
        values = state["values"]
        rigor = self.entry.rigor.get(self.route, "physics-level")
        if rigor == "certified":
            bare = sorted(k for k in values if k not in self.entry.meta and k not in state["errors"])
            if bare:
                raise Refusal("precondition:certified-radius", f"{self.entry.name}: certified route "
                              f"{self.route!r} returned {', '.join(bare)} without a radius in errors")
        res = Result(dict(values), trace, rigor=rigor, errors=dict(state["errors"]),
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
        fin = {k: from_flag(sol.stable, "rs", LEVEL_A_CITE, _stability_note(sol))
               for k in values if k not in ("branch", "ansatz", "stable")}
        return _finalise(res, fin, require_final)


def _stability_note(sol):
    """What the composed finality rests on: the derived replicon (and its rules) and, for
    discrete variables, the sign of the RS entropy."""
    if sol.replicon is None:
        return "replicon not derived for this family (stability UNKNOWN)"
    out = f"derived replicon {mp.nstr(sol.replicon, 8)} ({sol.replicon_route})"
    if sol.mode is not None:
        out += f"; derived mode of the extra order parameters {mp.nstr(sol.mode, 8)}"
    if sol.stable is False and sol.replicon > 0:
        out += ("; unstable because the extras' mode is negative" if sol.mode is not None
                and sol.mode < 0 else
                "; unstable because the RS entropy is negative for discrete variables")
    elif sol.stable is None:
        out += "; stability UNKNOWN (a mode was not computed)"
    return out


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


ANSATZE = ("rs", "1rsb")


def compose(spec: ModelSpec, route: str = "replica", ansatz: str = "rs") -> ComposedCalculation:
    try:
        if ansatz not in ANSATZE:
            raise Refusal(f"missing-step:ansatz:{ansatz}", f"known ansatze: {', '.join(ANSATZE)}")
        if ansatz == "1rsb":
            return _compose_rsb(spec, route)
        return _compose(spec, route)
    except Refusal as r:
        r.route = r.route or route
        raise


def _compose_rsb(spec, route):
    from .compose_rsb import level_a_rsb_steps, rsb_report
    if route != "replica":
        raise Refusal(f"missing-step:route:{route}",
                      f"ansatz 1rsb is a level-(a) replica route; route {route!r} has no "
                      "ansatz switch (the CLI sends --param ansatz here only for specs that are "
                      "not catalog entries, and --param level_a_ansatz for any spec)")
    _check_hooks(spec, nearest_entry(spec))
    return ComposedCalculation(spec, route, level_a_rsb_steps(spec), rsb_report(spec),
                               ansatz="1rsb")


def _compose(spec, route):
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
                      "only; the cavity and amp pipelines exist for catalog entries "
                      "(cavity/dense.py, cavity/sparse.py, catalog_amp.py), no generic "
                      "cavity or AMP composer")
    adapted, reused = [], []
    steps = level_a_steps(spec)
    if near is None:            # a level-(a) family with no catalog entry: nothing to compare
        relative = f"nothing (no catalog entry of family {spec.family!r})"
    else:
        relative = (f"level-(a) derivation of catalog entry {near.name!r} (spec "
                    f"{dict(zip(FIELDS, _key(near.spec)))})")
        theirs = level_a_steps(near.spec)
        if not [s.name for s in theirs] == [s.name for s in steps] == list(LEVEL_A_STEPS):
            raise ValueError(f"level-(a) step lists differ: {[s.name for s in steps]} (spec) "
                             f"vs {[s.name for s in theirs]} ({near.name!r}); expected "
                             f"{list(LEVEL_A_STEPS)}")
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


__all__ = ["ANSATZE", "Adaptation", "CompositionReport", "ComposedCalculation", "LEVEL_A_STEPS",
           "Result", "compose", "hooks_used", "level_a_steps", "nearest_entry"]
