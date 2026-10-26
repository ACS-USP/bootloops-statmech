"""Held-out model 2, the Ising perceptron (split out of tests/test_heldout.py, T15; the
conventions and the list of independent routes are in test_heldout.py's docstring): routes
Perceptron (a) S -> ln 2, (b) the zero of S against the banked threshold, (c) the annealed
bound, (d) exact enumeration at N <= 14, the computed finality (E1), and the solve with the
catalog made to raise."""
import math
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.compose import compose
from replicant.testing import slow

from heldout_common import (ENUM_N, ENUM_SAMPLES, MARKOV_SEEDS, _annealed_failures, _counts,
                            _enumeration_failures, _entropy, _ising_perceptron, _no_catalog,
                            _refusal, _small_alpha_failures, _zero_failures)


# ---- 2. Ising perceptron ------------------------------------------------------------
@slow
def test_ising_perceptron_composed_matches_independent_routes():
    calc = compose(_ising_perceptron())
    r = calc.report
    assert r.kind == "composed" and r.nearest == "perceptron"
    assert [a.step for a in r.adapted] == ["replicate", "introduce-order-parameters",
                                           "rs-ansatz", "solve-derived"], r.adapted
    assert {a.hook for a in r.adapted} == {"site:ising"}
    assert r.reused == ["average-disorder", "replica-limit", "rs-stationarity"]
    assert not _small_alpha_failures(calc), _small_alpha_failures(calc)          # (a)
    fails, alpha0 = _zero_failures(calc)                                       # (b)
    assert not fails, fails
    grid = ("0.1", "0.25", "0.5", "0.75", "1.0")
    assert not _annealed_failures(calc, grid), _annealed_failures(calc, grid)  # (c)
    cache = {}
    S_of = lambda a: cache.setdefault(a, _entropy(calc, a))
    assert not _enumeration_failures(S_of, alpha0), _enumeration_failures(S_of, alpha0)
    # (d) enumeration sanity, rigorous: median ln Z <= ln(2 E Z) (Markov), i.e. the
    # median of (1/N) ln #solutions <= ln2 (1 - alpha) + ln2/N at N = 12; statistical
    # (sample median, named seeds)
    for P, seed in MARKOV_SEEDS.items():
        c = _counts(ENUM_N, P, ENUM_SAMPLES, seed)
        with np.errstate(divide="ignore"):
            med = float(np.median(np.log(c) / ENUM_N))
        assert med <= math.log(2) * (1 - P / ENUM_N + 1 / ENUM_N), (P, med)
    # (e) finality computed (E1): replicon-stable with S > 0 below the zero (final),
    # False (negative entropy) beyond it
    below = calc.run(alpha="0.5", kappa=0, dps=15)
    assert below.stable is True and below.final and below.values["replicon"] > 0
    above = calc.run(alpha="1.0", kappa=0, dps=15)
    assert above.values["entropy"] < 0 and above.stable is False and not above.final
    assert _refusal(lambda: calc.run(alpha="1.0", kappa=0, dps=15,
                                     require_final=True)).reason == "unstable:rs"
    assert calc.run(alpha="0.5", kappa=0, dps=15, require_final=True).final

@slow
def test_heldout_ising_perceptron_not_looked_up():
    """The composed Ising perceptron solves with the catalog made to raise.  Afterwards it
    is compared with Task 8's level-(b) gardner_entropy('ising') as a consistency check
    only: that is the code path R13 keeps out of the catalog, so it is NOT counted as an
    independent route (those are in test_ising_perceptron_composed_matches_...)."""
    res, calls = _no_catalog(lambda: compose(_ising_perceptron()).run(alpha="0.5", kappa=0,
                                                                      dps=15))
    assert calls == [] and res.entry is None
    from replicant.families.perceptron import gardner_entropy
    with mp.workdps(20):
        want = gardner_entropy("0.5", "ising")
    assert abs(res.values["entropy"] - mp.mpf(getattr(want, "value", want))) < 1e-10
