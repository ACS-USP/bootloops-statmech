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


# ---- slow: composed solves -------------------------------------------------------------
@slow
def test_composed_ising_perceptron_is_not_presented_as_final():
    res = compose(_ising_perceptron()).run(alpha="0.5", kappa=0, dps=15)
    assert res.stable is not True and not res.final
    assert any("NOT final" in n for n in res.notes), res.notes
    r = _refusal(lambda: compose(_ising_perceptron()).run(alpha="0.5", kappa=0, dps=15,
                                                          require_final=True))
    assert r.reason in ("unknown-stability:rs", "unstable:rs"), r


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
