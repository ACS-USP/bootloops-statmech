"""Composer: matching a ModelSpec against the catalog, reuse / adaptation reports,
composition from level-(a) steps for a model NOT in the catalog, and refusals.

Conventions: a spec MATCHES a catalog entry when (family, variables, disorder,
constraint, prior) are equal (params are run parameters).  nearest_entry ranks the
entries of the same family by (constraint, prior) first, then (variables, disorder).
A composed calculation runs replicate -> average-disorder -> introduce-order-parameters
-> rs-ansatz -> replica-limit -> rs-stationarity -> solve-derived; it is RS and
physics-level, and it is never presented as final unless stable is True.
SK: J_ij ~ N(0, 1/N) (gaussian) or +-1/sqrt(N) (pm_one), field h, beta = 1/T.
"""
import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant import derive as D
from replicant.agree import default_spec
from replicant.catalog import catalog, run_cached
from replicant.compose import LEVEL_A_STEPS, Result, compose, nearest_entry
from replicant.core import Refusal
from replicant.families.common import CAVEAT, CAVITY_CAVEAT
from replicant.spec import ModelSpec
from replicant.testing import slow


def _refusal(fn):
    try:
        out = fn()
    except Refusal as r:
        return r
    raise AssertionError(f"expected Refusal, got {out!r}")


def _sk(disorder="gaussian", variables="ising"):
    return ModelSpec("sk", variables, disorder, params={"beta": D.BETA, "h": D.H_FIELD})


def _ising_perceptron():
    s = default_spec("perceptron")
    return ModelSpec("perceptron", "ising", s.disorder, s.constraint, s.prior, dict(s.params))


# ---- matching ----------------------------------------------------------------------
def test_nearest_entry_sk_pm_one_is_sk():
    assert nearest_entry(_sk("pm_one")).name == "sk"


def test_nearest_entry_ising_perceptron_is_the_perceptron_entry():
    e = nearest_entry(_ising_perceptron())
    assert e.name == "perceptron" and e.spec.variables == "spherical"


def test_nearest_entry_unknown_family_is_none():
    assert nearest_entry(ModelSpec("xyz", "ising", "gaussian")) is None


def test_nearest_entry_prefers_constraint_and_prior_over_variables():
    """A real tie-break: register a second perceptron entry that matches the query's
    variables and disorder but not its constraint; the catalogued entry (same constraint)
    must win.  The registry is restored afterwards."""
    import dataclasses
    from replicant import catalog as C
    base = catalog()["perceptron"]
    fake = dataclasses.replace(base, name="perceptron_parity",
                               spec=ModelSpec("perceptron", "ising", "pm_one", "parity"))
    query = ModelSpec("perceptron", "ising", "pm_one", "margin kappa")
    C._CATALOG["perceptron_parity"] = fake
    try:
        assert nearest_entry(query).name == "perceptron"      # (constraint, prior) first
        q2 = ModelSpec("perceptron", "ising", "pm_one", "parity", prior="x")
        assert nearest_entry(q2).name == "perceptron_parity"  # constraint matches the fake
    finally:
        del C._CATALOG["perceptron_parity"]
    assert "perceptron_parity" not in catalog()


# ---- catalog specs: reuse ----------------------------------------------------------
def test_compose_known_spec_reports_all_reused():
    for name, e in catalog().items():
        for route in sorted(e.routes):
            c = compose(e.spec, route)
            r = c.report
            assert r.kind == "catalog" and r.nearest == name, (name, route)
            assert r.reused == [s.name for s in e.route_steps[route]], (name, route, r.reused)
            assert r.adapted == [] and r.missing == [], (name, route)
            assert r.relative_to.startswith(f"catalog entry {name!r} itself"), r.relative_to
            assert [s.name for s in c.steps] == r.reused
    # and the run is the entry's family pipeline, with its trace and citations
    from replicant.families.sk import sk_rs
    res = compose(catalog()["sk"].spec).run(beta="2.0", h=0)
    assert isinstance(res, Result) and res.route == "replica"
    assert res.trace.to_json()["steps"] == [s.name for s in catalog()["sk"].steps]
    assert all(e["cite"] for e in res.trace.to_json()["entries"])
    assert res.values["q"] == sk_rs("2.0").q and res.values["ansatz"] == "rs"


def test_compose_catalog_route_not_supported_refuses():
    r = _refusal(lambda: compose(catalog()["pspin"].spec, "cavity"))
    assert r.reason == "missing-step:route:cavity" and "pspin" in r.detail, r
    r = _refusal(lambda: compose(catalog()["sk"].spec, "tap"))
    assert r.reason == "missing-step:route:tap", r


# ---- composition reports ------------------------------------------------------------
def test_compose_pm_one_sk_reports_adapted_disorder_step():
    c = compose(_sk("pm_one"))
    r = c.report
    assert r.kind == "composed" and r.nearest == "sk"
    assert [(a.step, a.field, a.was, a.now, a.hook) for a in r.adapted] == \
        [("average-disorder", "disorder", "gaussian", "pm_one", "disorder:pm_one")]
    assert r.reused == [n for n in LEVEL_A_STEPS if n != "average-disorder"]
    assert r.missing == [] and "level-(a) derivation of catalog entry 'sk'" in r.relative_to
    assert [s.name for s in c.steps] == list(LEVEL_A_STEPS)


def test_compose_ising_perceptron_reports_adapted_site():
    r = compose(_ising_perceptron()).report
    assert r.kind == "composed" and r.nearest == "perceptron"
    # replicate checks the variables against SITES / Family.variables: adapted too
    assert [a.step for a in r.adapted] == ["replicate", "introduce-order-parameters",
                                           "rs-ansatz", "solve-derived"], r.adapted
    assert all((a.field, a.was, a.now, a.hook) == ("variables", "spherical", "ising",
                                                   "site:ising") for a in r.adapted)
    assert r.reused == ["average-disorder", "replica-limit", "rs-stationarity"]
    assert "'perceptron'" in r.relative_to


# ---- refusals -----------------------------------------------------------------------
def test_compose_unknown_family_refuses_naming_missing_step():
    r = _refusal(lambda: compose(ModelSpec("xyz", "ising", "gaussian")))
    assert r.reason.startswith("missing-step:") and r.reason == "missing-step:family:xyz", r
    assert "FAMILIES" in r.detail and "catalog" in r.detail, r.detail


def test_compose_refusals_name_the_hook_and_wrap_the_reason():
    r = _refusal(lambda: compose(_sk(variables="spherical")))
    assert r.reason == "missing-step:site:spherical", r
    assert "precondition:supported-variables" in r.detail, r.detail
    r = _refusal(lambda: compose(_sk("lognormal")))
    assert r.reason == "missing-step:disorder:lognormal", r
    assert "precondition:supported-disorder-law" in r.detail, r.detail
    s = default_spec("perceptron")
    r = _refusal(lambda: compose(ModelSpec("perceptron", "spherical", "pm_one", s.constraint)))
    assert r.reason == "missing-step:disorder:pm_one", r
    r = _refusal(lambda: compose(ModelSpec("perceptron", "spherical", "gaussian", "parity")))
    assert r.reason == "missing-step:constraint:parity", r
    r = _refusal(lambda: compose(ModelSpec("perceptron", "ising", "gaussian", s.constraint,
                                           prior="sparse")))
    assert r.reason == "missing-step:prior:sparse", r
    r = _refusal(lambda: compose(ModelSpec("rem", "ising", "pm_one")))
    assert r.reason == "missing-step:family:rem" and "rem" in r.detail, r
    r = _refusal(lambda: compose(_sk("pm_one"), "cavity"))      # no generic cavity composer
    assert r.reason == "missing-step:route:cavity", r


# ---- results: labels, stability, finality -------------------------------------------
def test_result_declares_physics_level():
    rep = compose(catalog()["sk"].spec).run(beta="2.0", h=0)
    assert rep.rigor == "physics-level" and CAVEAT in rep.notes
    assert rep.stable is False and rep.final is False
    cav = compose(catalog()["sk"].spec, "cavity").run(beta=0.5, h=0)
    assert cav.rigor == "physics-level" and cav.route == "cavity" and CAVITY_CAVEAT in cav.notes
    assert cav.stable is True and cav.final is True
    assert abs(cav.values["q"]) < 1e-12 and cav.trace.to_json()["steps"][1] == "remove_site"
    head = compose(catalog()["rem"].spec).run(require_final=True)    # exact by literature
    assert head.rigor == "physics-level" and head.final is True
    assert head.finality["T_c"].state == "exact-by-literature" and "Derrida" in \
        head.finality["T_c"].cite


def test_compose_pm_one_sk_says_rs_not_final():
    """The one default-tier composed solve (low dps, float Gauss-Hermite, then the
    solver's independent mpmath audit; the polished path is in the slow tier): +-J SK at beta = 1.5, h = 0 is
    below T_c and AT-unstable; the composer must say the RS answer is not final.
    Independent route: universality (pm_one = gaussian at leading order in 1/N), so q
    must equal the level-(b) Gaussian SK solution, which nothing here fed."""
    from replicant.families.sk import sk_rs
    before = mp.mp.dps
    res = compose(_sk("pm_one")).run(beta="1.5", h=0, dps=12, quad="gh")   # float GH + audit
    assert mp.mp.dps == before
    assert res.rigor == "physics-level" and res.route == "replica"
    assert res.stable is False and res.final is False
    assert any("NOT final" in n and "unstable" in n for n in res.notes), res.notes
    assert {st.state for st in res.finality.values()} == {"unstable"}
    assert res.trace.entries[-1].cite == "replicant level (a) derivation (this package)"
    assert abs(res.values["q"] - sk_rs("1.5").q) < 1e-9
    assert res.trace.to_json()["steps"] == list(LEVEL_A_STEPS)
    assert "dropped fourth cumulant" in res.trace.render()
    assert res.errors["residual"] < 1e-9 and res.report.adapted[0].hook == "disorder:pm_one"


def test_require_final_refuses():
    sk = compose(catalog()["sk"].spec)
    r = _refusal(lambda: sk.run(beta="2.0", h=0, require_final=True))
    assert r.reason == "unstable:rs" and "replicon" in r.detail, r
    vb = compose(catalog()["viana_bray"].spec, "cavity")
    small = dict(k=2, beta=1.3, pop=300, sweeps=10, burn_in=30, runs=4, seed=0)
    r = _refusal(lambda: vb.run(require_final=True, **small))
    assert r.reason == "unknown-stability:rs" and "not-computed" in r.detail, r
    r = _refusal(lambda: sk.run(beta="1.5", h=0, ansatz="1rsb", require_final=True))
    assert r.reason == "not-final:1rsb", r
    ts = compose(catalog()["teacher_student"].spec).run(alpha=2, require_final=True)
    assert ts.final and ts.finality["eps"].state == "exact-by-literature"
    cav = compose(catalog()["sk"].spec, "cavity")
    assert cav.run(beta=0.5, h=0, require_final=True).final      # stable passes through


def test_headline_ground_energies_are_not_final():
    """`python3 -m replicant run sk` (no state point): the RS ground energy comes from an
    AT-unstable saddle and the 1RSB one is not the exact (full-RSB) answer."""
    res = compose(catalog()["sk"].spec).run()
    fin = res.finality
    assert fin["rs_ground_energy"].state == "unstable" and "replicon" in fin["rs_ground_energy"].note
    assert fin["rsb1_ground_energy"].state == "not-computed"
    assert fin["tc"].final and res.final is False
    assert any(n.startswith("rs_ground_energy: RS value is NOT final (unstable") for n in res.notes)
    assert any(n.startswith("rsb1_ground_energy: 1RSB value is NOT final (not-computed")
               for n in res.notes)
    r = _refusal(lambda: compose(catalog()["sk"].spec).run(require_final=True))
    assert r.reason == "unstable:rs" and "rs_ground_energy" in r.detail \
        and "rsb1_ground_energy" in r.detail, r


def test_every_catalog_value_has_a_finality_declaration():
    """Guard: no un-annotated physical value (finality_of refuses on one)."""
    from replicant.catalog import Stability
    runs = [(n, r.route, r.params) for n, e in catalog().items() for r in e.refs
            if not (r.slow or r.statistical)]
    runs += [("sk", "replica", {"beta": "2.0", "h": 0}), ("sk", "cavity", {"beta": 2}),
             ("rem", "replica", {"beta": 1}), ("perceptron", "cavity", {"alpha": "0.5"}),
             ("teacher_student", "replica", {"alpha": 2}),
             ("hopfield", "cavity", {"alpha": "0.05", "beta": 20}),
             ("viana_bray", "cavity", dict(k=2, beta=1.3, pop=300, sweeps=10, burn_in=30,
                                           runs=4, seed=0))]
    for name, route, params in runs:
        e = catalog()[name]
        values, _ = run_cached(e, route, params)        # the pipeline's own run (memoised)
        fin = e.finality_of(values)
        assert fin and all(isinstance(st, Stability) for st in fin.values()), (name, params)
        assert set(fin) | (set(values) & e.meta) == set(values)
    bad = catalog()["sk"]
    import dataclasses
    stripped = dataclasses.replace(bad, finality={})
    assert _refusal(lambda: stripped.finality_of({"tc": 1})).reason == \
        "precondition:finality-declared"


def test_population_dynamics_values_are_labelled_statistical():
    vb = compose(catalog()["viana_bray"].spec, "cavity")
    res = vb.run(k=2, beta=1.3, pop=300, sweeps=10, burn_in=30, runs=4, seed=0)
    assert set(res.statistical) >= {"q", "e", "f"}
    assert all(res.errors[k] > 0 for k in ("q", "e", "f")), res.errors
    assert any("statistical" in n for n in res.notes) and res.rigor == "physics-level"


def test_reference_checks_available_through_result():
    res = compose(catalog()["perceptron"].spec).run()
    checks = res.reference_checks()
    assert checks and all(c.ok for c in checks), checks
    assert {c.key for c in checks} == {"spherical_capacity_kappa0"}


# ---- run parameters of catalog entries (final review, Important 1) ----------------------
# The parameter names the GUIDE's INPUTS line and worked examples use, per entry and route.
_DOCUMENTED = {
    ("rem", "replica"): {"beta", "N_scale"},
    ("sk", "replica"): {"beta", "h", "fullrsb", "ansatz"},
    ("sk", "cavity"): {"beta", "h"},
    ("pspin", "replica"): {"p", "beta"},
    ("ising_pspin", "replica"): {"p", "beta"},
    ("perceptron", "replica"): {"alpha", "kappa"},
    ("perceptron", "cavity"): {"alpha", "kappa"},
    ("teacher_student", "replica"): {"alpha", "learning"},
    ("sk", "amp"): {"beta", "h", "N", "seeds", "iters", "law"},
    ("teacher_student", "amp"): {"alpha", "learning", "N", "seeds", "iters"},
    ("hopfield", "replica"): {"alpha", "beta", "branch"},
    ("hopfield", "cavity"): {"alpha", "beta", "branch"},
    ("xorsat", "cavity"): {"K", "alpha", "N", "seeds"},
    ("ksat", "cavity"): {"K", "alpha", "N", "seeds", "pop", "sweeps", "crossing"},
    ("ldpc_bec", "cavity"): {"l", "r", "eps", "N", "seeds"},
    ("spiked_wigner", "amp"): {"snr", "prior", "N", "seeds", "iters"},
    ("sbm", "cavity"): {"a", "b", "c", "N", "seeds", "iters"},
    ("ridge", "replica"): {"gamma", "lam", "sigma2", "signal", "n", "seeds"},
    ("series_ising2d", "series"): {"order", "fit_terms", "max_order", "max_degree", "L"},
    ("ising2d", "exact"): {"beta", "L", "Ly"},
    ("ising1d", "exact"): {"beta", "h", "N"},
    ("bethe_ising", "cavity"): {"k", "beta", "h", "branch"},
    ("sk_bounds", "bound"): {"beta", "h", "K", "digits"},
    ("online_perceptron", "online"): {"rule", "alpha", "eta", "noise", "noise_estimate", "R0", "Q0",
                                      "N", "seeds"},
    ("online_linear", "online"): {"rule", "constraint", "eta", "M", "R0", "Q0", "alpha", "N", "seeds"},
    ("online_drift", "online"): {"rule", "drift", "D", "eta", "noise", "noise_estimate", "R0", "Q0", "alpha", "N",
                                 "seeds"},
    ("online_tcm", "online"): {"rule", "K", "eta", "R0", "Q0", "alpha", "N", "seeds"},
    ("online_scm", "online"): {"K", "M", "eta", "teacher", "critical", "init", "X", "Nx", "init_seed", "alpha", "B", "N",
                               "seeds"},
    ("viana_bray", "cavity"): {"graph", "k", "c", "J", "beta", "ansatz", "m", "ys", "pop",
                               "inner", "sweeps", "burn_in", "runs", "seed",
                               "require_equilibrated"},
}


def test_catalog_run_refuses_an_unknown_parameter_naming_the_accepted_ones():
    """A typo'd run parameter (kapa for kappa) is refused before anything runs, never
    silently ignored (it would yield the kappa = 0 answer labelled final)."""
    e = catalog()["perceptron"]
    r = _refusal(lambda: compose(e.spec).run(alpha="1.5", kapa="0.3"))
    assert r.reason == "precondition:known-parameter", r
    assert "kapa" in r.detail and "alpha, kappa" in r.detail, r.detail
    # the perceptron entry is RS only: an ansatz parameter is not one of its parameters
    r = _refusal(lambda: compose(e.spec).run(alpha="1.5", ansatz="1rsb"))
    assert r.reason == "precondition:known-parameter" and "ansatz" in r.detail, r
    r = _refusal(lambda: compose(catalog()["sk"].spec, "cavity").run(beta=2, fullrsb=True))
    assert r.reason == "precondition:known-parameter" and "fullrsb" in r.detail, r


def test_declared_parameters_cover_the_documented_names_and_the_banked_runs():
    cat = catalog()
    assert {(n, r) for n, e in cat.items() for r in e.routes} == set(_DOCUMENTED)
    for (name, route), names in _DOCUMENTED.items():
        assert names <= cat[name].parameters[route], (name, route,
                                                      names - cat[name].parameters[route])
    for name, e in cat.items():
        for ref in e.refs:
            assert set(ref.params) <= e.parameters[ref.route], (name, ref.key)
        numeric = {k for k, v in e.spec.params.items() if not getattr(v, "free_symbols", None)}
        assert numeric <= e.parameters[e.default_route], name


def test_composed_path_still_notes_unused_parameters():
    res = compose(_sk("pm_one")).run(beta="1.5", h=0, dps=12, quad="gh", foo=1)
    assert any("unused parameters ['foo']" in n for n in res.notes), res.notes


# ---- perceptron entropy finality (final review, Important 2) ----------------------------
def test_perceptron_entropy_finality_is_its_computed_replicon():
    e = catalog()["perceptron"]
    res = compose(e.spec).run(alpha="0.5")
    lam = res.values["replicon"]
    assert lam > 0 and res.finality["S"].state == "stable", (lam, res.finality["S"])
    assert res.values["stable"] is True
    res = compose(e.spec).run(alpha="1.5", kappa=-1)
    lam = res.values["replicon"]
    assert res.finality["S"].state == ("stable" if lam > 0 else "unstable"), (lam, res.finality)
    assert res.finality["alpha_c"].state == "unstable"


def test_perceptron_entropy_label_follows_a_negative_replicon():
    """Positive control: if gardner_entropy's refusal were relaxed and the replicon came
    back negative, the entropy is labelled unstable (not the old hard-coded stable)."""
    import dataclasses
    from replicant import catalog_entries as CE
    orig = CE.gardner_entropy_point

    def bypassed(alpha, variables, kappa=0):
        S, pt = orig(alpha, variables, kappa)
        return S, dataclasses.replace(pt, replicon=mp.mpf("-0.25"))

    CE.gardner_entropy_point = bypassed
    try:
        res = compose(catalog()["perceptron"].spec).run(alpha="0.5")
    finally:
        CE.gardner_entropy_point = orig
    assert res.finality["S"].state == "unstable" and res.values["stable"] is False
    assert res.values["replicon"] == mp.mpf("-0.25") and not res.final


def test_teacher_student_gibbs_finality_states_the_nishimori_argument():
    """Final review, Minor 6: the exact-by-literature label of eps (Gibbs) rests on
    Gibbs learning being posterior sampling (families/teacher_student.py: J uniform on
    the version space, a uniform spherical teacher prior); the note says so."""
    res = compose(catalog()["teacher_student"].spec).run(alpha=2)
    st = res.finality["eps"]
    assert st.state == "exact-by-literature" and res.values["learning"] == "gibbs"
    assert ("Gibbs learning on the version space with a uniform prior is posterior sampling, "
            "so the Nishimori identity applies (overlap concentration of Barbier et al.)"
            ) in st.note, st.note


# ---- composer invariants (final review, Minors 4 and 8) ---------------------------------
def test_compose_level_a_step_mismatch_is_an_error_not_an_assert():
    """The step-list invariant holds under python -O too (no bare assert)."""
    import replicant.compose as C
    orig = C.level_a_steps
    near = nearest_entry(_sk("pm_one")).spec
    C.level_a_steps = lambda spec, *a, **k: orig(spec, *a, **k)[:-1] if spec == near \
        else orig(spec, *a, **k)
    try:
        compose(_sk("pm_one"))
    except ValueError as exc:
        assert "level-(a) step" in str(exc), exc
    else:
        raise AssertionError("a step-list mismatch must raise")
    finally:
        C.level_a_steps = orig


def test_hooks_used_names_every_registry_a_step_consults():
    """Pin compose.hooks_used against what the level-(a) steps actually look up, for the
    cheap composed SK pm_one solve: derive.FAMILIES (which Family hook is called),
    derive.SITES, derive.LAWS and solve_derived._DISCRETE are replaced by recording
    wrappers (restored in finally) and every step runs once.  Pinned: the fields a step
    touches are a SUBSET of the fields hooks_used names for it, and each object it names
    for a touched registry is the one the step fetched.  Limit (why not equality):
    rs-ansatz reads the site only through the functional built by
    introduce-order-parameters, and the constraint / prior entries of average-disorder are
    declarative (Family.constraints / priors are checked by the composer, not read by any
    step), so those names cannot be observed as lookups."""
    import dataclasses
    from replicant import solve_derived as SD
    from replicant.compose import hooks_used, level_a_steps
    from replicant.core import State, run_pipeline
    spec = _sk("pm_one")
    touched, cur = {}, [None]

    def rec(field, obj):
        touched.setdefault(cur[0], {}).setdefault(field, []).append(obj)
        return obj

    class RecDict(dict):
        def __init__(self, field, d):
            super().__init__(d)
            self.field = field

        def __getitem__(self, k):
            return rec(self.field, dict.__getitem__(self, k))

        def get(self, k, default=None):
            return rec(self.field, dict.get(self, k, default))

        def __contains__(self, k):
            rec(self.field, dict.get(self, k))
            return dict.__contains__(self, k)

    class RecTuple(tuple):
        def __contains__(self, k):
            rec("variables", tuple.__contains__(self, k))
            return tuple.__contains__(self, k)

    def wrap_family(fam):
        hooks = {}
        for f in dataclasses.fields(fam):
            v = getattr(fam, f.name)
            if callable(v):
                hooks[f.name] = (lambda v, n: lambda *a, **k: (rec("family", getattr(fam, n)),
                                                              v(*a, **k))[1])(v, f.name)
        return dataclasses.replace(fam, **hooks)

    saved = [(D, "FAMILIES", D.FAMILIES), (D, "SITES", D.SITES), (D, "LAWS", D.LAWS),
             (SD, "_DISCRETE", SD._DISCRETE)]
    steps = level_a_steps(spec, {"beta": "1.5", "h": 0}, {"dps": 12, "quad": "gh"})

    def tagged(st):
        def on(fn):
            def g(s):
                cur[0] = st.name
                return fn(s)
            return g
        return dataclasses.replace(st, requires=on(st.requires), apply=on(st.apply))

    D.FAMILIES = RecDict("family", {k: wrap_family(v) for k, v in D.FAMILIES.items()})
    D.SITES, D.LAWS = RecDict("variables", D.SITES), RecDict("disorder", D.LAWS)
    SD._DISCRETE = RecTuple(SD._DISCRETE)
    try:
        run_pipeline([tagged(s) for s in steps], State())
    finally:
        for mod, n, v in saved:
            setattr(mod, n, v)
    assert set(touched) <= set(LEVEL_A_STEPS), touched
    assert touched.get("replicate") and touched.get("average-disorder"), touched
    for step in LEVEL_A_STEPS:
        named, seen = hooks_used(step, spec), touched.get(step, {})
        assert set(seen) <= set(named), (step, set(seen) - set(named))
        for field, objs in seen.items():
            if field == "family":     # the Family hook hooks_used names is one called here
                assert any(o is named["family"] for o in objs), (step, objs)
            elif field == "variables" and step == "solve-derived":
                assert named["variables"] in objs, (step, objs)
            else:
                assert any(o is named[field] for o in objs), (step, field, objs)


# ---- slow: composed solves -------------------------------------------------------------
@slow
def test_composed_ising_perceptron_is_not_presented_as_final():
    # beyond the frozen-1RSB capacity (alpha = 1.0): replicon > 0 but S < 0, not final
    res = compose(_ising_perceptron()).run(alpha="1.0", kappa=0, dps=15)
    assert res.stable is False and not res.final and res.values["replicon"] > 0
    assert any("NOT final" in n for n in res.notes), res.notes
    assert all("entropy is negative" in f.note for f in res.finality.values())
    r = _refusal(lambda: compose(_ising_perceptron()).run(alpha="1.0", kappa=0, dps=15,
                                                          require_final=True))
    assert r.reason == "unstable:rs", r


@slow
def test_composed_pm_one_sk_above_at_line_is_final():
    res = compose(_sk("pm_one")).run(beta="1.5", h=1, branch="nontrivial", dps=15,
                                     require_final=True)
    assert res.stable is True and res.final


@slow
def test_composed_numeric_beta_in_spec():
    s = ModelSpec("sk", "ising", "pm_one", params={"beta": "1.5", "h": D.H_FIELD})
    a = compose(s).run(h=0, dps=12)
    b = compose(_sk("pm_one")).run(beta="1.5", h=0, dps=12)
    assert abs(a.values["q"] - b.values["q"]) < 1e-10
