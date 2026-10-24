"""Composer route ansatz="1rsb" (compose.compose(spec, ansatz="1rsb")): the level-(a)
1RSB derivation of any spec with level-(a) hooks, solved by solve_derived_rsb.

  level_a_rsb_steps(spec, point=None, knobs=None) -> the Steps
      replicate -> average-disorder -> introduce-order-parameters -> rsb-ansatz ->
      rsb-replica-limit [-> zero-temperature-limit] -> rsb-stationarity ->
      solve-derived-1rsb
  rsb_report(spec, knobs=None) -> CompositionReport relative to the RS level-(a) derivation
      of the SAME spec, computed by comparing compose.level_a_steps(spec) with
      level_a_rsb_steps(spec, knobs): steps with the same name and builder are reused (here
      replicate, average-disorder, introduce-order-parameters); the RS ansatz, n -> 0,
      stationarity and solve steps are ADAPTED to their 1RSB versions (field "ansatz",
      rs -> 1rsb); with zero_temperature=true the zero-temperature-limit step is listed as a
      new step.  run_1rsb recomputes it with the run's knobs.
  run_1rsb(calc, require_final, params) -> Result

Run knobs (popped from run(**params)): frozen, zero_temperature, branch, seeds, fixed,
solve_for, audit, audit_tol (solve_derived_rsb.py); everything else is a model parameter,
and a name the derived functional does not read is refused precondition:known-parameter.
Finality: every value is "not-computed" (ansatz 1rsb): the 1RSB replicon (stability toward
further RSB) is Task 5's (E4), and no composed 1RSB value has a banked target.  So
run(require_final=True) refuses not-final:1rsb.
"""
import mpmath as mp
import sympy as sp

from . import derive as D
from . import derive_rsb as R
from .catalog import from_flag
from .core import State, Step, Trace, run_pipeline
from .solve_derived import lift_numeric_beta
from .solve_derived_rsb import solve_derived_1rsb

KNOBS = ("frozen", "zero_temperature", "branch", "seeds", "fixed", "solve_for", "audit",
         "audit_tol")
FINALITY_NOTE = ("1RSB value: stability toward further RSB (the 1RSB replicon) is not "
                 "computed here (Task 5, E4); no banked 1RSB target for a composed value "
                 "(no independent target)")


def level_a_rsb_steps(spec, point=None, knobs=None):
    knobs = dict(knobs or {})
    frozen, t0 = bool(knobs.get("frozen", False)), bool(knobs.get("zero_temperature", False))
    n = sp.Symbol("n", positive=True)

    def solve(s):
        eqs = s["rsb_equations"]
        sol = solve_derived_1rsb(spec, derived=(eqs, s, Trace()), **knobs, **(point or {}))
        return s.with_(rsb_solution=sol)

    steps = [D.replicate(spec, n), D.average_disorder(), D.introduce_order_parameters(),
             R.rsb_ansatz(frozen), R.rsb_replica_limit()]
    steps += [R.zero_temperature_limit(), R.rsb_stationarity("rsb_t0")] if t0 else \
        [R.rsb_stationarity()]
    return steps + [Step("solve-derived-1rsb",
                         lambda s: [] if "rsb_equations" in s else ["rsb_equations"], solve,
                         "replicant level (a) derivation (this package)",
                         lambda s: s["rsb_solution"].trace.entries[-1].summary,
                         D._tex("solve-derived-1rsb"))]


def _builder(step):
    """The function that built a Step (its apply closure's qualified name and module)."""
    return (step.apply.__module__, step.apply.__qualname__)


def rsb_report(spec, knobs=None):
    """Computed from the two step lists: a step of the 1RSB list with the same name and the
    same builder as the RS one at that position is REUSED (the same Family / SITES / LAWS
    hooks, since the spec is the same); the remaining RS steps are ADAPTED to the remaining
    1RSB steps in order; a 1RSB step left over (zero-temperature-limit) is a new step."""
    from .compose import Adaptation, CompositionReport, FIELDS, level_a_steps
    rs, rsb = level_a_steps(spec), level_a_rsb_steps(spec, knobs=knobs)
    k = 0
    while k < min(len(rs), len(rsb)) and rs[k].name == rsb[k].name and \
            _builder(rs[k]) == _builder(rsb[k]):
        k += 1
    new = [s.name for s in rsb[k:]]
    old = [s.name for s in rs[k:]]
    adapted, j = [], 0
    for name in new:
        partner = next((o for o in old[j:] if o.split("-")[-1] == name.split("-")[-1]
                        or o.replace("rs-", "rsb-") == name or o + "-1rsb" == name
                        or o.replace("replica-", "rsb-replica-") == name), None)
        if partner is None:
            adapted.append(Adaptation("(new step)", "ansatz", "rs", "1rsb", name))
        else:
            j = old.index(partner) + 1
            adapted.append(Adaptation(partner, "ansatz", "rs", "1rsb", name))
    rel = (f"the RS level-(a) derivation of the same spec "
           f"({dict(zip(FIELDS, (getattr(spec, f) for f in FIELDS)))})")
    return CompositionReport([s.name for s in rsb[:k]], adapted, [], None, "composed", rel)


def run_1rsb(calc, require_final, params):
    from .compose import LEVEL_A_CITE, Result, _finalise
    knobs = {k: params.pop(k) for k in KNOBS if k in params}
    if knobs.get("zero_temperature"):
        spec, point, lift = calc.spec, dict(params), []
    else:
        spec, point, lift = lift_numeric_beta(calc.spec, dict(params))
    state, trace = run_pipeline(level_a_rsb_steps(spec, point, knobs), State())
    sol = state["rsb_solution"]
    values = {**sol.values, "phi": sol.free_energy, "branch": sol.branch, "ansatz": "1rsb",
              "stable": sol.stable}
    if sol.energy is not None:
        values["energy"] = sol.energy
    mode = ("frozen 1RSB (q1 = 1)" if knobs.get("frozen") else
            "T = 0 1RSB" if knobs.get("zero_temperature") else "full 1RSB")
    res = Result(values, trace, errors={"residual": sol.residual, "quad_error": sol.quad_error},
                 notes=[sol.caveat, *lift, f"ansatz 1rsb: {mode}", sol.note], route="replica",
                 report=rsb_report(calc.spec, knobs),
                 params={**point, **{k: v for k, v in knobs.items()
                                                         if k != "seeds"}}, spec=calc.spec)
    fin = {k: from_flag(None, "1rsb", LEVEL_A_CITE, FINALITY_NOTE)
           for k in values if k not in ("branch", "ansatz", "stable")}
    return _finalise(res, fin, require_final)


def describe(sol):
    """One line for the TeX hook of solve-derived-1rsb."""
    fmt = lambda v: mp.nstr(v, 12) if isinstance(v, (mp.mpf, mp.mpc)) else str(v)
    vals = ", ".join(f"{k} = {fmt(v)}" for k, v in sol.values.items())
    return (f"{vals}; psi = {fmt(sol.free_energy)}; branch {sol.branch}; residual "
            f"{fmt(sol.residual)} (independent quadrature), quadrature error "
            f"{fmt(sol.quad_error)}; Hessian eigenvalues {[round(e, 8) for e in sol.hessian]}; "
            f"stable = None (not computed).  {sol.note}")
