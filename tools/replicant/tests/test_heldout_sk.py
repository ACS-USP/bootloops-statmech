"""Held-out model 1, SK with +-J couplings (split out of tests/test_heldout.py, T15; the
conventions and the list of independent routes are in test_heldout.py's docstring):
routes SK (a) n = 1 enumeration, (b) T_c, universality of q and the RS ground energy, (c) the
AT eigenvalue recomputed by quadrature, (d) AMP on sampled +-J matrices (E3b)."""
import itertools
import math
import os
import sys

import mpmath as mp
import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant import derive as D
from replicant.compose import LEVEL_A_STEPS, compose
from replicant.core import State, run_pipeline
from replicant.refdata import check, load
from replicant.spec import ModelSpec
from replicant.testing import slow

from heldout_common import (_pm_one_sk, _refusal, _sk_tc_failures, _universality_failures,
                            _with)


def _at_eigenvalue(beta, q):
    """1 - beta^2 E_z sech^4(beta sqrt(q) z), z ~ N(0, 1): the AT replicon at h = 0,
    computed here (mpmath quadrature), not by the package."""
    with mp.workdps(30):
        b, q = mp.mpf(beta), mp.mpf(q)
        f = lambda z: mp.exp(-z * z / 2) / mp.sqrt(2 * mp.pi) * mp.sech(b * mp.sqrt(q) * z)**4
        return 1 - b**2 * mp.quad(f, [-mp.inf, 0, mp.inf])


SK_BETAS = ("4", "5", "6", "8", "10")


def _sk_ground_energy(calc):
    """(e0 from 5 points, e0 from the 4 lowest-T points): polynomial in T without a linear
    term (de/dT = T dS/dT -> 0, the RS entropy being finite at T = 0)."""
    us = {}
    for b in SK_BETAS:
        r = calc.run(beta=b, h=0, dps=15, branch="nontrivial")
        us[b] = (r.values["entropy"] - r.values["phi"]) / mp.mpf(b)
    with mp.workdps(30):
        def fit(bs):
            T = [1 / mp.mpf(b) for b in bs]
            A = mp.matrix([[1] + [t**p for p in range(2, len(bs) + 1)] for t in T])
            return mp.lu_solve(A, mp.matrix([us[b] for b in bs]))[0]
        return fit(SK_BETAS), fit(SK_BETAS[1:])


# ---- 1. +-J SK ----------------------------------------------------------------------
def test_sk_pm_one_annealed_matches_enumeration():
    """(a), default tier.  ln E Z at n = 1, beta = 0.7, h = 0.  Brute force: N <= 6 over
    every coupling sign pattern and every spin state (no identity used); N = 7..12 the
    per-pair average over the two signs, state by state (exact since the pairs are
    independent).  Both against derive.annealed_log_Z (the pm_one Family hook) to 1e-12,
    and against the COMPOSED functional at n = 1 (the composer's own replicate /
    average-disorder / introduce-order-parameters steps), whose gap to the exact value is
    the dropped fourth cumulant: exact - (N e + c) = -beta^4/24 (1 - 1/N) + O(1/N^2)."""
    b = 0.7
    spec = ModelSpec("sk", "ising", "pm_one", params={"beta": sp.Rational(7, 10), "h": 0})
    exact = D.annealed_log_Z(spec, D.N_SYM)
    st, _ = run_pipeline([D.replicate(spec, 1), D.average_disorder(),
                          D.introduce_order_parameters()], State())
    F = st["functional"]
    composed_n1 = D.N_SYM * F.expr + F.correction
    assert F.missing == "O(1)", F.missing        # the stated order of what was dropped
    for N in range(2, 13):
        pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        S = np.array(list(itertools.product((1.0, -1.0), repeat=N)))
        X = np.stack([S[:, i] * S[:, j] for i, j in pairs], axis=1)      # states x pairs
        if N <= 6:
            G = np.array(list(itertools.product((1.0, -1.0), repeat=len(pairs))))
            avg = np.exp(b / math.sqrt(N) * (G @ X.T)).sum() / len(G)
        else:
            avg = np.prod(np.cosh(b / math.sqrt(N) * X), axis=1).sum()
        enum = math.log(avg)
        want = float(exact.subs(D.N_SYM, N))
        assert abs(enum - want) < 1e-12 * max(1, abs(want)), (N, enum, want)
        gap = enum - float(composed_n1.subs(D.N_SYM, N))
        assert abs(gap + b**4 / 24) <= b**4 / (12 * N), (N, gap)


@slow
def test_sk_pm_one_composed_matches_independent_routes():
    calc = compose(_pm_one_sk())
    r = calc.report
    assert r.kind == "composed" and r.nearest == "sk"
    assert [(a.step, a.hook) for a in r.adapted] == [("average-disorder", "disorder:pm_one")]
    assert r.reused == [s for s in LEVEL_A_STEPS if s != "average-disorder"]
    # (b) T_c = 1 by three readings of the composed calculation
    assert not _sk_tc_failures(calc), _sk_tc_failures(calc)
    # (b) universality of q at beta = 1.5 (target computed after the composed solve)
    res = calc.run(beta="1.5", h=0, dps=15)
    assert not _universality_failures(res, "1.5")
    assert "dropped fourth cumulant" in res.trace.render() and "O(1/N) per spin" in \
        res.trace.render()
    # (b) RS ground energy: T -> 0 extrapolation of the composed energies
    e5, e4 = _sk_ground_energy(calc)
    closed = -mp.sqrt(2 / mp.pi)
    assert check(load("heldout")["sk_pm_one_rs_ground_energy"], mp.nstr(e5, 15),
                 label="pm_one e0").ok, e5
    assert check(load("sk")["rs_ground_energy"], mp.nstr(e5, 15), label="e0").ok, e5
    assert abs(e5 - closed) <= 3 * abs(e5 - e4), (e5, e4, closed)
    # (c) AT-unstable, NOT final; the replicon equals the independently computed AT one
    assert res.stable is False and res.final is False
    assert res.values["replicon"] < 0
    assert abs(res.values["replicon"] - _at_eigenvalue("1.5", res.values["q"])) < 1e-10
    assert {st.state for st in res.finality.values()} == {"unstable"}
    assert any("NOT final" in n for n in res.notes)
    rf = _refusal(lambda: calc.run(beta="1.5", h=0, dps=15, require_final=True))
    assert rf.reason == "unstable:rs" and "replicon" in rf.detail, rf


AMP_N, AMP_SEEDS, AMP_ITERS = 2000, tuple(range(8)), 100


def _amp_failures(q, band):
    """The composed q inside the 99% Student-t band of the finite-N AMP q."""
    return [] if abs(float(q) - band.mean) <= band.half_width else \
        [f"composed q {float(q):.5f} outside AMP {band.mean:.5f} +- {band.half_width:.5f}"]


@slow
def test_heldout_pm_one_sk_q_vs_amp_on_pm_one_matrix():
    """(d) The composed +-J SK q at beta = 1.5, h = 0.8 (the RS-stable side of the AT line,
    where the AMP iterates converge) against AMP on sampled +-J matrices J_ij =
    +-1/sqrt(N): N = 2000, seeds 0-7, t = 100, 99% Student-t band (df = 7).  AMP does not
    use the composer or any family (amp/ imports none of them) and is run after the
    composed solve.  What it rests on: AMP's state evolution is a theorem for Gaussian J
    (Bayati-Montanari 2011); for +-J it is numerical evidence (amp/sk.py), so this route
    checks the composed universality claim against an independent sampled computation,
    not against a theorem.  Positive controls: the composed calculation with the planted
    doubled +-J variance (J = +-sqrt(2/N), the (b') law) falls outside the band, and so
    does naive mean-field AMP (no Onsager term) against the composed q."""
    import dataclasses
    from replicant import amp as A
    calc = compose(_pm_one_sk())
    res = calc.run(beta="1.5", h="0.8", dps=15)
    assert res.report.kind == "composed" and res.entry is None
    q = res.values["q"]
    ens = A.sk_amp_ensemble(AMP_N, 1.5, 0.8, seeds=AMP_SEEDS, iters=AMP_ITERS, law="pm_one")
    b = ens.band
    print(f"+-J AMP N={AMP_N} t={AMP_ITERS}: q = {b.mean:.5f} +- {b.half_width:.5f} "
          f"(stderr {b.stderr:.5f}); composed {float(q):.5f}; converged {sum(ens.converged)}"
          f"/{len(ens.converged)}")
    assert not _amp_failures(q, b), _amp_failures(q, b)
    assert all(ens.converged), ens.converged
    law = D.LAWS["pm_one"]
    doubled = dataclasses.replace(
        law, cgf=lambda t, N: sp.log(sp.cosh(sp.sqrt(2) * t / sp.sqrt(N))))
    wrong = _with(D.LAWS, "pm_one", doubled,
                  lambda: compose(_pm_one_sk()).run(beta="1.5", h="0.8", dps=15))
    assert D.LAWS["pm_one"] is law
    print(f"planted doubled variance: composed q = {float(wrong.values['q']):.5f}")
    assert _amp_failures(wrong.values["q"], b), wrong.values["q"]
    naive = A.sk_amp_ensemble(AMP_N, 1.5, 0.8, seeds=AMP_SEEDS, iters=AMP_ITERS,
                              law="pm_one", reaction=False).band
    print(f"naive mean field: q = {naive.mean:.5f} +- {naive.half_width:.5f}")
    assert _amp_failures(q, naive), naive
