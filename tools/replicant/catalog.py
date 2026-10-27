"""Catalog: published calculations as recorded pipelines of typed Steps.

  catalog() -> dict[str, CatalogEntry]      the v1 entries (built once)
  check_ref(entry, ref) -> RefOutcome       run the entry where `ref` says, compare
  UNCLAIMED                                 bank keys no entry reproduces, with reasons

An entry is a ModelSpec, its citation, the routes it has ("replica" = the level-(b)
family pipeline; "cavity" = the cavity pipeline), one list of Steps per route, and the
banked keys it reproduces (RefCheck).  Each family call is wrapped as a core.Step with
its citation, so a run returns a core.Trace a reader can follow to the paper.  Steps
read State["params"] and add to State["values"] (plus "errors": numerical error bars,
"statistical": the value keys that come from sampling, "notes").  A step that does not
apply to the given parameters (e.g. a T = 0 headline when beta is given) is recorded
in the trace as "skipped: ...", never silently dropped.

Finality is per value: each entry declares, for every physical value key, a Stability
(stable / unstable / not-computed / exact-by-literature, with the ansatz and a cite), or a
callable reading the run's computed values["stable"]; compose.Result reports it per key
and refuses require_final when any value is not final.  Honest labels: every route is physics-level (replica n -> 0 and
RS/1RSB ansaetze, or the cavity clustering assumptions) unless the entry declares it "certified"
(CatalogEntry.rigor: the lattice and bound entries, whose values are ball enclosures with their radii in
errors); population-dynamics numbers are statistical (value +- stderr over independent runs),
never certified.

Each entry also exposes derivation(route): the equations a LaTeX document displays
(latex.py), kept next to the family code in latex_entries.py / latex_cavity.py.

Matching to model specifications (which entry a spec is, which is nearest) lives in
compose.py; the entries themselves in catalog_entries.py, catalog_sparse.py and
catalog_csp.py (constraint satisfaction) and catalog_inference.py (codes, inference; the
spiked Wigner and block-model entries in catalog_spiked_sbm.py) and catalog_learning.py
(high-dimensional learning: ridge) and catalog_series.py (exact series: series_ising2d, route
"series") and catalog_lattice.py (solvable lattice models, certified: ising2d, ising1d, route
"exact"; bethe_ising, route "cavity") and catalog_bounds.py (Guerra's certified upper bounds on the SK
pressure: sk_bounds, route "bound") and catalog_online.py (online learning, route "online":
online_perceptron).
"""
import math
from dataclasses import dataclass, field
from typing import Callable

import mpmath as mp

from .core import Refusal, State, Step, run_pipeline
from .refdata import check, load
from .spec import ModelSpec


@dataclass(frozen=True, eq=False)
class RefCheck:
    """Banked key `file`/`key` is reproduced by running the entry on `route` with
    `params` and reading `extract(values)`.  Statistical keys are compared in a 99%
    Student-t band on hypot(stderr, tol), stderr = error(errors), df = runs - 1."""
    file: str
    key: str
    extract: Callable[[dict], object]
    statistical: bool = False
    params: dict = field(default_factory=dict)
    route: str = "replica"
    slow: bool = False
    error: Callable[[dict], object] | None = None
    runs: int | None = None


@dataclass(frozen=True)
class RefOutcome:
    file: str
    key: str
    ok: bool
    got: str
    want: str
    tol: str
    statistical: bool
    detail: str = ""


STATES = ("stable", "unstable", "not-computed", "exact-by-literature")
FINAL_STATES = ("stable", "exact-by-literature")


@dataclass(frozen=True)
class Stability:
    """Finality of one reported value.  state: "stable" (stability computed by this
    pipeline and positive), "unstable" (computed and negative, or the computed entropy
    rules the ansatz out), "not-computed" (no stability computed here), or
    "exact-by-literature" (the cited source establishes the value; needs `cite`).
    Only "stable" and "exact-by-literature" are final."""
    state: str
    ansatz: str
    cite: str = ""
    note: str = ""

    def __post_init__(self):
        if self.state not in STATES:
            raise ValueError(f"stability state {self.state!r} not in {STATES}")
        if self.state == "exact-by-literature" and not self.cite.strip():
            raise ValueError("exact-by-literature needs a cite")

    @property
    def final(self):
        return self.state in FINAL_STATES


def from_flag(stable, ansatz, cite="", note=""):
    """Stability from a computed flag: True -> stable, False -> unstable, None ->
    not-computed."""
    state = {True: "stable", False: "unstable"}.get(stable, "not-computed")
    return Stability(state, ansatz, cite, note)


def saddle(cite="", note=""):
    """Declaration for values of a saddle step that records values['ansatz'] and the
    computed values['stable']."""
    return lambda v: from_flag(v.get("stable"), v.get("ansatz", "rs"), cite, note)


@dataclass(frozen=True, eq=False)
class CatalogEntry:
    name: str
    spec: ModelSpec
    steps: list            # the default route's Steps (route_steps[default_route])
    cite: str
    routes: set
    refs: list
    run: Callable[..., dict]          # run(route=None, **params) -> values
    route_steps: dict = field(default_factory=dict)
    caveats: dict = field(default_factory=dict)       # route -> honest label
    description: str = ""
    default_route: str = "replica"
    # value key -> Stability, or a callable(values) -> Stability (state-point values);
    # `meta`: value keys that are not physical results (echoed parameters, labels,
    # diagnostics).  Every other value key must have a finality declaration.
    finality: dict = field(default_factory=dict)
    meta: frozenset = frozenset()
    # route -> frozenset of the run-parameter names that route's steps read (the composer
    # refuses any other name: precondition:known-parameter, never silently ignored)
    parameters: dict = field(default_factory=dict)
    # route -> rigor label of its Result ("physics-level" unless declared): "certified" for a
    # route whose every physical value is a ball enclosure with its radius in errors (lattice/)
    rigor: dict = field(default_factory=dict)

    def finality_of(self, values):
        """{key: Stability} for the physical values of one run; a key with no
        declaration is a Refusal (precondition:finality-declared), never assumed final."""
        out = {}
        for k in values:
            if k in self.meta:
                continue
            d = self.finality.get(k)
            if d is None:
                raise Refusal("precondition:finality-declared",
                              f"catalog entry {self.name!r}: value {k!r} has no finality "
                              "declaration")
            out[k] = d(values) if callable(d) else d
        return out

    def derivation(self, route=None):
        """The displayed equations of this entry's route ([tex.TexBlock]; latex_entries.py,
        latex_cavity.py): sympy forms checked against the family code, or cited strings."""
        from .latex_entries import derivation
        return list(derivation(self.name, route or self.default_route))

    def execute(self, route=None, **params):
        """(final State, Trace) of the route's pipeline at `params`."""
        route = route or self.default_route
        if route not in self.routes:
            raise Refusal(f"missing-step:route:{route}",
                          f"catalog entry {self.name!r} has routes "
                          f"{', '.join(sorted(self.routes))}")
        init = State(params=dict(params), values={}, errors={}, statistical=(), notes=(),
                     step_note="")
        return run_pipeline(self.route_steps[route], init)


@dataclass(frozen=True)
class Out:
    values: dict
    note: str = ""
    errors: dict = field(default_factory=dict)
    statistical: tuple = ()
    notes: tuple = ()


def fstep(name, cite, compute, *, when=None, skip=""):
    """A family call as a Step: compute(params) -> Out.  `when(params)` False -> the
    step is recorded as skipped (with `skip` as the reason) and changes no value."""
    def apply(s):
        p = s["params"]
        if when is not None and not when(p):
            return s.with_(step_note=f"skipped: {skip}")
        out = compute(p)
        return s.with_(values={**s["values"], **out.values},
                       errors={**s["errors"], **out.errors},
                       statistical=tuple(dict.fromkeys(s["statistical"] + tuple(out.statistical))),
                       notes=s["notes"] + tuple(out.notes),
                       step_note=out.note or "computed " + ", ".join(out.values))

    return Step(name, lambda s: [] if "params" in s else ["params"], apply, cite,
                lambda s: s["step_note"])


def entry(name, spec, cite, route_steps, refs, caveats, description, default_route="replica",
          finality=None, meta=(), parameters=None, rigor=None):
    holder = {}
    if set(parameters or {}) != set(route_steps):
        raise ValueError(f"catalog entry {name!r}: declare the run parameters of every route "
                         f"({', '.join(sorted(route_steps))})")

    def run(route=None, **params):
        return holder["e"].execute(route, **params)[0]["values"]

    e = CatalogEntry(name, spec, route_steps[default_route], cite, set(route_steps), refs, run,
                     route_steps, caveats, description, default_route, dict(finality or {}),
                     frozenset(meta) | {"ansatz", "stable"},
                     {r: frozenset(v) for r, v in (parameters or {}).items()}, dict(rigor or {}))
    holder["e"] = e
    return e


def has(*keys):
    return lambda p: all(p.get(k) is not None for k in keys)


def lacks(*keys):
    return lambda p: all(p.get(k) is None for k in keys)


def mpf30(x):
    """Decimal string of an mpf at 30 digits (an mpf keeps its own precision; str()
    would print only the global 15)."""
    return mp.nstr(x, 30) if isinstance(x, mp.mpf) else x


# ---- bank keys no entry claims ---------------------------------------------------------
_HELD_OUT = ("Ising perceptron: held out of the catalog on purpose (Task 16 composes it "
             "from the spherical perceptron entry, and the composer must not look its "
             "answer up); level (b) gardner_capacity('ising') reproduces it in "
             "tests/test_perceptron.py")
_HELD_OUT_SK = ("+-J SK: held out of the catalog (Task 16); the composed level-(a) "
                "calculation is checked against it in tests/test_heldout_sk.py, never a "
                "catalog pipeline")
_GAMP = ("Ising-prior (binary teacher) GAMP thresholds: located by bisection on the state "
         "evolution and RS potential of amp/gamp_se.py (ising_thresholds) in tests/test_gamp.py; "
         "the teacher_student catalog entry is the spherical teacher, so no pipeline outputs it")
_MC = ("finite-N literature value (Boettcher 2005, Table 1) reproduced by exact enumeration "
       "(N = 15) or parallel tempering (N = 31) in tests/test_pt.py (slow): a check of the "
       "numerics/ machinery, not the output of any pipeline step")
UNCLAIMED = {
    ("mc", "pm_one_ground_energy_n15"): _MC,
    ("mc", "pm_one_ground_energy_n31"): _MC,
    ("heldout", "sk_pm_one_tc"): _HELD_OUT_SK,
    ("heldout", "sk_pm_one_rs_ground_energy"): _HELD_OUT_SK,
    ("sk", "at_small_field_coefficient"):
        "an h -> 0 extrapolation of h^2/(1 - T_AT)^3 over three AT-line points, done in "
        "tests/test_sk.py (slow): not the output of any pipeline step",
    ("perceptron", "ising_rs_capacity"): _HELD_OUT,
    ("perceptron", "ising_1rsb_capacity"): _HELD_OUT,
    ("perceptron", "ising_1rsb_q"): _HELD_OUT,
    ("perceptron", "ising_1rsb_qhat"): _HELD_OUT,
    ("gamp", "ising_alpha_it"): _GAMP,
    ("gamp", "ising_alpha_amp"): _GAMP,
    ("hopfield", "low_alpha_error_prefactor"):
        "an alpha -> 0 extrapolation of (1 - m) e^{1/(2 alpha)}/sqrt(alpha) from two T = 0 "
        "points, done in tests/test_hopfield.py: not a pipeline output",
    ("hopfield", "tg_sqrt_alpha_coefficient"):
        "located by bisection on the existence of the spin-glass branch in "
        "tests/test_hopfield.py (slow): not a pipeline output",
    ("hopfield", "alpha_c_rs_max_over_T"):
        "the maximum over T of alpha_c(T), located by a three-point scan in "
        "tests/test_hopfield.py (slow): not a pipeline output",
}

_KSAT = ("random K-SAT thresholds banked from the literature (MMZ 2006 / KMRSZ 2007 tables): "
         "replicant locates the SP-population root for K = 3 only (catalog_csp.ksat, "
         "crossing=true), and implements no reconstruction / condensation recursion")
UNCLAIMED.update({("ksat", k): _KSAT for k in (
    "alpha_s_k4", "alpha_s_k5", "alpha_s_k6", "alpha_s_k7", "alpha_c_k3", "alpha_d_k4",
    "alpha_c_k4", "alpha_d_k5", "alpha_c_k5", "alpha_d_k6", "alpha_c_k6")})
UNCLAIMED[("ksat", "alpha_d_k3")] = (
    "the reference point of tests/test_ksat.py::test_bp_converges_below_alpha_d_and_not_above "
    "(BP converges at 3.5, not at 4.0): bracketed, not computed (no reconstruction recursion)")
UNCLAIMED[("ksat", "alpha_sp_onset_k3")] = (
    "bracketed in tests/test_ksat.py (SP population trivial at 3.6, non-trivial at 4.2), not "
    "located: near the onset the population relaxes slowly (labelled undetermined)")
UNCLAIMED[("sbm", "inverse_eps_c_q4_c16")] = (
    "DKMZ's q = 4, c = 16 ratio: inference/sbm.py implements q = 2 only (q = 4 is refused "
    "unsupported:q), so no pipeline outputs it")
UNCLAIMED[("ksat", "alpha_1rsb_stability_k3")] = (
    "MMZ's 1RSB stability window edge, cited in the finality note of alpha_s_sp; replicant "
    "computes no SP stability")


# ---- reference checks --------------------------------------------------------------------
_RUNS = {}

UNCLAIMED[("online_linear", "online_eq20_rho_a2")] = (
    "KC95 Eq. (20) is a time-dependent rule (a callable F(b, h, Q, R, alpha)), not a named catalog rule: "
    "tests/test_online_linear.py test_kc_eq20_time_dependent_rule integrates it and checks this key")
UNCLAIMED.update({("online_drift", k): (
    "VKC p. 15's delta = 5, 8 for the perceptron at a fixed rate apply Eq. (34) outside its derivation: the "
    "decoupled form (35) needs W independent of J (KC93 p. 6), and the free-norm perceptron's |J| self-tunes; "
    "replicant's ODE gives 1/delta = 1/3, 1/4 with eps_inf = (4D)^(1/3)/pi, 2^(5/4)/pi D^(1/4) (online/drift.py "
    "drift_exponent), and delta = 3 violates VKC's inequality (42); checked against this key's failure in "
    "tests/test_online_drift.py") for k in ("random_perceptron_exponent", "det_perceptron_exponent")})
UNCLAIMED.update({("online_tcm", f"cc95_fig1_es_a{a}"): (
    "CC95 Fig. 1 read off the figure: the nonlocal K = 3 ODE (a 3D field average per step) to alpha = 16.5 costs "
    "about 3.5 s at rtol 1e-9 (measured by the O4 reviewer), so it is checked in the slow test "
    "tests/test_online_tcm.py test_cc95_figure1_and_the_integrated_asymptote") for a in ("3p3", "6p7", "10", "16p5")})
UNCLAIMED.update({("online_scm", k): "VC97 Fig. 1 points along the optimal learning curve (one ODE run checks them all): "
                  "tests/test_online_optimal_ml.py test_vc97_fig1_learning_curve_and_hessian" for k in (
    *(f"vc97_fig1_eps_{a}" for a in ("a2p5", "a3", "a3p5", "a4", "a5", "a6")), "vc97_fig1_hess_hi_a4", "vc97_fig1_hess_lo_a4")})
UNCLAIMED.update({("online_scm", f"vc97_fig2_eps_a{a}"): "the K = M = 2 optimal curve takes about a minute: "
                  "tests/test_online_optimal_ml.py test_k2m2_curve_reaches_the_plateau (slow)" for a in ("10", "12", "16")})
UNCLAIMED.update({("online_scm", k): f"integrated over a plateau or to K = 12 (seconds): checked in the slow test {t}" for k, t in (
    ("ss_eta_max_coefficient", "tests/test_online_scm.py test_ss_eta_max_first_order_in_one_over_k"),
    ("brw96_eq17_log_slope", "tests/test_online_scm_plateau.py test_brw96_eq17_half_log_slope_and_fig2_shift"),
    ("ss_trapping_eta_exponent", "tests/test_online_scm_plateau.py test_ss_fig1_plateau_and_trapping_time"))})

def _freeze(p):
    return tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in p.items()))


def run_cached(e: CatalogEntry, route, params):
    """(values, errors) of one run, memoised per (entry, route, params) in this process
    (several banked keys come from one population-dynamics run)."""
    key = (e.name, route, _freeze(params))
    if key not in _RUNS:
        state, _ = e.execute(route, **params)
        _RUNS[key] = (state["values"], state["errors"])
    return _RUNS[key]


def compare(e: CatalogEntry, ref: RefCheck, values, errors) -> RefOutcome:
    want = load(ref.file)[ref.key]
    try:
        got = ref.extract(values)
    except (KeyError, TypeError) as exc:
        raise Refusal("precondition:reference-quantity",
                      f"{e.name}: {ref.file}/{ref.key} not in this run's values ({exc!r})")
    if not ref.statistical:
        c = check(want, mpf30(got), label=f"{e.name}:{ref.key}")
        return RefOutcome(ref.file, ref.key, c.ok, c.got, c.want, c.tol, False)
    from .cavity._stats import t_quantile
    err = float(ref.error(errors))
    t = t_quantile(0.995, ref.runs - 1)
    band = t * math.hypot(err, float(want.tol))
    ok = abs(float(got) - float(want.value)) <= band
    return RefOutcome(ref.file, ref.key, ok, repr(float(got)), want.value, want.tol, True,
                      f"stderr {err:.3g}, 99% Student-t band {band:.3g} (t = {t:.4g}, "
                      f"df = {ref.runs - 1})")


def check_ref(e: CatalogEntry, ref: RefCheck) -> RefOutcome:
    values, errors = run_cached(e, ref.route, ref.params)
    return compare(e, ref, values, errors)


_CATALOG = {}


def catalog() -> dict:
    if not _CATALOG:
        from .catalog_bounds import bounds_entries
        from .catalog_csp import csp_entries
        from .catalog_entries import dense_entries
        from .catalog_inference import inference_entries
        from .catalog_lattice import lattice_entries
        from .catalog_learning import learning_entries
        from .catalog_online import online_entries
        from .catalog_series import series_entries
        from .catalog_sparse import sparse_entries
        for e in (dense_entries() + sparse_entries() + csp_entries() + inference_entries()
                  + learning_entries() + series_entries() + lattice_entries() + bounds_entries()
                  + online_entries()):
            _CATALOG[e.name] = e
    return dict(_CATALOG)


__all__ = ["CatalogEntry", "RefCheck", "RefOutcome", "Stability", "UNCLAIMED", "catalog",
           "check_ref", "from_flag", "saddle"]
