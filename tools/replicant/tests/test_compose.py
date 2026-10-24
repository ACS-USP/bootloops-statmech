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
from replicant.catalog import catalog
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
    # same family, two candidate entries would differ; here: a perceptron with the
    # catalogued constraint but other variables and disorder is still the perceptron
    s = ModelSpec("perceptron", "ising", "pm_one", "margin kappa")
    assert nearest_entry(s).name == "perceptron"


# ---- catalog specs: reuse ----------------------------------------------------------
def test_compose_known_spec_reports_all_reused():
    for name, e in catalog().items():
        for route in sorted(e.routes):
            c = compose(e.spec, route)
            r = c.report
            assert r.kind == "catalog" and r.nearest == name, (name, route)
            assert r.reused == [s.name for s in e.route_steps[route]], (name, route, r.reused)
            assert r.adapted == [] and r.missing == [], (name, route)
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
    assert r.missing == []
    assert [s.name for s in c.steps] == list(LEVEL_A_STEPS)


def test_compose_ising_perceptron_reports_adapted_site():
    r = compose(_ising_perceptron()).report
    assert r.kind == "composed" and r.nearest == "perceptron"
    sites = [a for a in r.adapted if a.hook == "site:ising"]
    assert sites and {a.step for a in sites} >= {"introduce-order-parameters", "rs-ansatz"}
    assert all((a.field, a.was, a.now) == ("variables", "spherical", "ising") for a in r.adapted)
    assert "average-disorder" in r.reused and "replicate" in r.reused
    assert not set(r.reused) & {a.step for a in r.adapted}


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
    head = compose(catalog()["rem"].spec).run()          # headline numbers: no saddle judged
    assert head.rigor == "physics-level" and head.final is None and "T_c" in head.values


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
    assert abs(res.values["q"] - sk_rs("1.5").q) < 1e-9
    assert res.trace.to_json()["steps"] == list(LEVEL_A_STEPS)
    assert "dropped fourth cumulant" in res.trace.render()
    assert res.errors["residual"] < 1e-9 and res.report.adapted[0].hook == "disorder:pm_one"


def test_require_final_refuses():
    sk = compose(catalog()["sk"].spec)
    r = _refusal(lambda: sk.run(beta="2.0", h=0, require_final=True))
    assert r.reason == "unstable:rs" and "replicon" in r.detail, r
    ts = compose(catalog()["teacher_student"].spec)
    r = _refusal(lambda: ts.run(alpha=2, require_final=True))
    assert r.reason == "unknown-stability:rs", r
    res = ts.run(alpha=2)
    assert res.stable is None and any("NOT final" in n and "unknown" in n for n in res.notes)
    cav = compose(catalog()["sk"].spec, "cavity")
    assert cav.run(beta=0.5, h=0, require_final=True).final      # stable passes through


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
