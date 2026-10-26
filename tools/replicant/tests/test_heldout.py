"""Held-out models (Task 16): the project's acceptance test of reuse.

T15 split this file by model: test_heldout_sk.py (+-J SK routes (a)-(d)),
test_heldout_perceptron.py (Ising perceptron routes (a)-(d), finality, not looked up),
test_heldout_mc.py (E11 routes SK (e)-(g), Perceptron (e)); shared helpers in
heldout_common.py.  This file keeps the positive controls, the catalog guards and the
documents.

Two models that are NOT catalog entries are assembled by compose() from the level-(a)
steps of their nearest entries, solved, and checked against routes that did not feed
them; a positive control shows each check can fail.

Conventions.
  +-J SK: J_ij = +-1/sqrt(N) with equal probability (i < j), H = -sum_{i<j} J_ij s_i s_j
    - h sum_i s_i, s_i = +-1, beta = 1/T.  phi = lim (1/(N n)) ln E Z^n = -beta f (RS);
    entropy S = phi - beta dphi/dbeta, energy per spin u = (S - phi)/beta.
  Ising perceptron: J in {-1,+1}^N, P = alpha N Gaussian patterns, kappa = 0; a pattern
    is stored if J.xi > 0.  Entropy per weight S(alpha) = lim (1/N) E ln #solutions (RS:
    phi of the composed calculation, no temperature).

Independent routes (none is computed by the code under test, none feeds it):
  SK (a)  ln E Z at n = 1 by brute force over coupling signs (N <= 6) and by the per-pair
          sign average (N = 7..12), against derive.annealed_log_Z and against the composed
          functional at n = 1 (whose deviation is the stated dropped fourth cumulant).
  SK (b)  T_c = 1, RS ground energy -sqrt(2/pi) (bank targets, reference/heldout.json and
          sk.json) and q at beta = 1.5 equal to the Gaussian catalog family's value
          (families.sk.sk_rs, called only as the comparison target, AFTER the composed
          solve): universality at leading order in 1/N.
  SK (c)  the de Almeida-Thouless eigenvalue 1 - beta^2 E sech^4(beta sqrt(q) z),
          recomputed here by mpmath quadrature at the composed q.
  Perceptron (a) S -> ln 2 as alpha -> 0 (counting: 2^N vectors); (b) the zero of the
          composed S(alpha) against the banked Krauth-Mezard / Ding-Sun threshold (the
          root of the RS entropy) and its q, qhat; (c) the Jensen / annealed bound
          S <= ln E Z / N = ln 2 (1 - alpha) (each J satisfies one pattern with
          probability exactly 1/2, independently over patterns: E Z = 2^N 2^-P);
          (d) exact enumeration at N <= 14 (statistical, named seeds, Student-t bands
          plus an explicit finite-size slack c/N, c = 1, an assumption stated here).
  SK (d)  AMP run on actually sampled +-J matrices (amp/sk.py, which imports none of the
          replica, cavity or family code): the composed q at beta = 1.5, h = 0.8 against
          the 99% Student-t band of the finite-N AMP q over named seeds (E3b).
  E11 (tests/heldout_mc.py; Monte Carlo and exact enumeration, numerics/; none uses the
  composer, every comparison target is computed after the measurement):
  SK (e)  Gaussian SK ground energy by parallel tempering (N = 16..128, N^-2/3 extrapolation)
          against the banked Parisi value; the RS value must be excluded (control).
  SK (f)  +-J SK energy at beta = 2 by PT (N = 64, 128, 256) against the composed RS and 1RSB
          energies; what the bands resolve is printed, only resolved contrasts are asserted.
  SK (g)  exact E ln Z (Gray-code enumeration, N = 12..22, beta = 0.7, 0.9) against ln E Z: the
          quenched-minus-annealed gap and the moment closure of numerics/moments.py; this
          tells quenched from annealed, which the v1 check (a) (n = 1 only) cannot.
  Perceptron (e)  exact counts at N = 12..24 extrapolated in 1/N against the composed S(alpha)
          and its zero; the spherical site must be excluded (control).  (d) stays: it is the
          v1 N <= 14 check; (e) is the larger-N route, not a replacement.
"""
import os
import sys

import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant import derive as D
from replicant.catalog import catalog
from replicant.compose import compose, nearest_entry
from replicant.core import Refusal
from replicant.testing import slow

from heldout_common import (_annealed_failures, _enumeration_failures, _entropy,
                            _ising_perceptron, _no_catalog, _pm_one_sk, _refusal, _run,
                            _sk_sign_flipped, _sk_tc_failures, _small_alpha_failures,
                            _universality_failures, _with, _zero_failures)


# ---- 3. positive controls -----------------------------------------------------------
@slow
def test_wrong_ansatz_is_rejected():
    import dataclasses
    # (a) RS forced where the replicon is negative: +-J SK at beta = 2
    calc = compose(_pm_one_sk())
    r = _refusal(lambda: calc.run(beta="2", h=0, dps=12, require_final=True))
    assert r.reason == "unstable:rs" and "replicon" in r.detail, r
    # (b) a sign-flipped saddle equation (SK order-parameter hook) fails the T_c and
    # universality routes: the flipped functional has no stationary point in the domain
    flipped = dataclasses.replace(D.FAMILIES["sk"], order_parameters=_sk_sign_flipped)

    def flipped_checks():
        bad = compose(_pm_one_sk())
        res = _run(bad, beta="1.5", h=0, dps=12)
        return _sk_tc_failures(bad), _universality_failures(res, "1.5"), res
    tc_f, uni_f, res = _with(D.FAMILIES, "sk", flipped, flipped_checks)
    assert tc_f and uni_f, (tc_f, uni_f)
    assert isinstance(res, Refusal) and res.reason == "no-solution" and \
        res.step == "rs-stationarity", res
    assert D.FAMILIES["sk"].order_parameters is not _sk_sign_flipped      # restored
    assert not _universality_failures(calc.run(beta="1.5", h=0, dps=12), "1.5")
    # (b') a solvable wrong model, not a refusal: the +-J law with its variance doubled
    # (J = +-sqrt(2/N)) has T_c = sqrt(2); the T_c route must see the wrong number
    law = D.LAWS["pm_one"]
    doubled = dataclasses.replace(
        law, cgf=lambda t, N: sp.log(sp.cosh(sp.sqrt(2) * t / sp.sqrt(N))))
    tc_wrong = _with(D.LAWS, "pm_one", doubled, lambda: _sk_tc_failures(compose(_pm_one_sk())))
    assert any("beta=0.95" in f for f in tc_wrong) and \
        any("paramagnetic replicon" in f for f in tc_wrong), tc_wrong
    assert D.LAWS["pm_one"] is law
    # (c) the Ising entropic term replaced by the spherical one: the composed Ising spec
    # with the spherical site.  S -> ln 2, the zero of S and the enumeration detect it; the
    # annealed bound does NOT (a spherical log-volume is <= 0 < ln2 (1 - alpha), alpha < 1)
    def spherical_checks():
        bad = compose(_ising_perceptron())
        zero, a0 = _zero_failures(bad)
        S_of = lambda a: _entropy(bad, a, dps=12)
        return (_small_alpha_failures(bad), zero, _enumeration_failures(S_of, a0),
                _annealed_failures(bad, ("0.5",)))
    small, zero, enum, ann = _with(D.SITES, "ising", D.SITES["spherical"], spherical_checks)
    assert small and zero and enum, (small, zero, enum)
    assert not ann, ann                 # the limitation, stated: this check is blind here
    assert D.SITES["ising"] is not D.SITES["spherical"]


# ---- 4. not in the catalog (the not-looked-up perceptron check is in test_heldout_perceptron.py)
def test_heldout_models_not_in_catalog():
    """Neither spec is a catalog entry (structural key); the nearest entries are the
    Gaussian SK and the spherical perceptron; and the composed +-J SK solves with every
    catalog family function made to raise (the one cheap default-tier composed solve:
    float Gauss-Hermite, dps 12, then the solver's independent audit).  The comparison
    target is called only afterwards."""
    from replicant.compose import FIELDS
    keys = [tuple(getattr(e.spec, f) for f in FIELDS) for e in catalog().values()]
    for spec, near in ((_pm_one_sk(), "sk"), (_ising_perceptron(), "perceptron")):
        assert tuple(getattr(spec, f) for f in FIELDS) not in keys, spec
        assert all(spec != e.spec for e in catalog().values())
        assert nearest_entry(spec).name == near
        assert compose(spec).report.kind == "composed"
    assert nearest_entry(_ising_perceptron()).spec.variables == "spherical"
    res, calls = _no_catalog(lambda: compose(_pm_one_sk()).run(beta="1.5", h=0, dps=12,
                                                               quad="gh"))
    assert calls == [] and res.report.kind == "composed" and res.entry is None
    from replicant.families.sk import sk_rs
    assert abs(res.values["q"] - sk_rs("1.5").q) < 1e-9
    # the guard itself works: a catalog run under it raises
    from replicant import catalog as C
    try:
        _no_catalog(lambda: compose(catalog()["sk"].spec).run(beta="1.5", h=0))
    except AssertionError as e:
        assert "composer called the catalog" in str(e)
    else:
        raise AssertionError("the no-catalog guard did not fire on a catalog run")
    assert C.CatalogEntry.execute.__name__ == "execute"           # restored


# ---- 5. documents -------------------------------------------------------------------
@slow
def test_heldout_documents_render():
    import shutil
    import tempfile
    import warnings
    from replicant.latex import OverfullDisplay, escape, overfull, render_latex, write_latex
    docs = {"sk_pm_one": compose(_pm_one_sk()).run(beta="1.5", h=0, dps=15),
            "ising_perceptron": compose(_ising_perceptron()).run(alpha="0.5", kappa=0, dps=15)}
    for name, res in docs.items():
        doc = render_latex(res)
        assert "Composed model" in doc, name
        assert escape(res.report.relative_to) in doc, name
        for a in res.report.adapted:
            assert escape(a.hook) in doc and escape(a.step) in doc, (name, a)
        for s in res.report.reused:
            assert escape(s) in doc, (name, s)
        # E1: the Ising perceptron at alpha = 0.5 is replicon-stable with S > 0 (final);
        # +-J SK at beta = 1.5 is AT-unstable (NOT final); both show the derived replicon
        assert ("NOT final" in doc) == (name == "sk_pm_one"), name
        assert "Replicon of the RS saddle (level (a))" in doc, name
    if not (shutil.which("tectonic") or shutil.which("pdflatex")):
        print("SKIP test_heldout_documents_render (compile): no TeX engine on PATH")
        return
    with tempfile.TemporaryDirectory() as d:
        for name, res in docs.items():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", OverfullDisplay)
                try:
                    write_latex(res, os.path.join(d, f"{name}.tex"), pdf=True)
                except Refusal as r:
                    if r.reason == "tex-engine:unavailable":
                        print(f"SKIP test_heldout_documents_render (compile): {r.detail}")
                        return
                    raise
            log = os.path.join(d, f"{name}.log")
            assert os.path.exists(os.path.join(d, f"{name}.pdf")), name
            bad = [o for o in overfull(log, 1.0) if o[2] == "display"]
            assert not bad and not caught, (name, bad, [str(w.message) for w in caught])
