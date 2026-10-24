"""Full RSB for SK (families/parisi.py): K-step Parisi staircases, extrapolated in K.

Conventions (families/parisi.py docstring): SK with J_ij ~ N(0, 1/N), h = 0, energies per
spin.  The replica functional is MAXIMISED over the staircase parameters (the n -> 0 rule
of families/sk.py, = Guerra's bound), so the T = 0 energies rise with the number of steps:
  E_RS < E_1RSB < E_2RSB < ... < E_FRSB   (-0.7979 < -0.76526 < -0.76361 < -0.76317).
Published numbers are compared through refdata.check against reference/sk.json only; the
banked full-RSB value is a TARGET of our own K-extrapolation, never an input.  All solves
are slow (one default-tier test: the refusals, which compute nothing).
"""
import math
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.core import Refusal
from replicant.refdata import check, load
from replicant.testing import slow

REF = load("sk")
GRID = 8     # the headline extrapolation: levels K = 6, 7, 8


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


def test_refusal_when_grid_too_small():
    """Default tier (computes nothing): the extrapolation needs three levels."""
    from replicant.families.parisi import krsb_ground_energy, parisi_ground_energy, sk_frsb
    r = _refuses(lambda: parisi_ground_energy(grid=2), "precondition:grid-too-small")
    assert "three levels" in r.detail
    _refuses(lambda: parisi_ground_energy(grid=2.5), "precondition:grid-too-small")
    _refuses(lambda: krsb_ground_energy(0), "precondition:grid-range")
    _refuses(lambda: krsb_ground_energy(99), "precondition:grid-range")
    _refuses(lambda: sk_frsb(2.0, grid=2), "precondition:grid-too-small")
    _refuses(lambda: sk_frsb(0.5), "precondition:temperature-range")   # paramagnet: no RSB


@slow
def test_parisi_energy_within_banked_range():
    """value +- error_estimate contains the banked OSS 2007 value; the raw level E_8 is
    below it (variational bound).  Sources disagree slightly (Crisanti-Rizzo -0.76321 +-
    3e-5, Parisi 1980 -0.7633 +- 1e-4): check() accepts any within the banked tol 2e-4."""
    from replicant.families.parisi import parisi_ground_energy
    before = mp.mp.dps
    r = parisi_ground_energy(grid=GRID)
    assert mp.mp.dps == before
    assert r.label == "numerical" and "numerical" in r.caveat
    ref = REF["fullrsb_ground_energy"]
    assert check(ref, r.value, label="parisi").ok
    banked = mp.mpf(ref.value)
    assert abs(r.value - banked) <= r.error_estimate, (r.value, r.error_estimate)
    assert r.levels[-1].E < banked < r.value + r.error_estimate
    assert r.error_estimate < mp.mpf("1e-5")           # far inside the bank's 2e-4 spread
    assert len(ref.sources_disagree) == 2


@slow
def test_frsb_below_1rsb():
    """Ordering of the T = 0 energies (maximised functional: each level is higher)."""
    from replicant.families.parisi import krsb_ground_energy, parisi_ground_energy
    from replicant.families.sk import sk_ground_energy
    Es = [krsb_ground_energy(K).E for K in range(1, GRID + 1)]
    assert all(a < b for a, b in zip(Es, Es[1:])), Es
    frsb = parisi_ground_energy(grid=GRID).value
    rs = sk_ground_energy("rs")
    assert rs < Es[0] < Es[1] < frsb
    assert frsb - Es[0] > mp.mpf("2e-3")        # 1RSB -> FRSB: +0.0021, not a rounding effect
    for key, got in (("rs_ground_energy", rs), ("rsb1_ground_energy", Es[0]),
                     ("rsb2_ground_energy", Es[1]), ("fullrsb_ground_energy", frsb)):
        assert check(REF[key], repr(float(got)), label=key).ok, (key, got)


@slow
def test_error_estimate_shrinks_with_grid():
    from replicant.families.parisi import parisi_ground_energy
    errs = [parisi_ground_energy(grid=g).error_estimate for g in (5, 6, 7, GRID)]
    assert all(a > b for a, b in zip(errs, errs[1:])), errs
    vals = [parisi_ground_energy(grid=g).value for g in (5, 6, 7, GRID)]
    for v, e in zip(vals, errs):                 # each band contains the finer estimate
        assert abs(v - vals[-1]) <= e, (v, e)


@slow
def test_k1_reproduces_1rsb_and_k2_reproduces_2rsb():
    """Independent of the banked full-RSB number: K = 1 is sk.py's own T = 0 1RSB (to
    1e-10) and the bank's 1RSB value; K = 2, 3 are Stojnic's Table 3 '3 (full)', '4 (full)'
    rows (2RSB, 3RSB)."""
    from replicant.families.parisi import krsb_ground_energy
    from replicant.families.sk import sk_ground_energy
    k1, k2, k3 = (krsb_ground_energy(K) for K in (1, 2, 3))
    assert abs(k1.E - float(sk_ground_energy("1rsb"))) < 1e-10
    assert check(REF["rsb1_ground_energy"], repr(k1.E), label="K=1").ok
    assert check(REF["rsb2_ground_energy"], repr(k2.E), label="K=2").ok
    assert check(REF["rsb3_ground_energy"], repr(k3.E), label="K=3").ok


@slow
def test_levels_are_audited_local_maxima():
    """Every level: gradient ~ 0, Hessian negative definite, grid change negligible; and
    the adjoint gradient agrees with a finite difference of the discrete functional."""
    from replicant.families import parisi as P
    for K in range(1, GRID + 1):
        lv = P.krsb_ground_energy(K)
        assert lv.grad_max <= 1e-7 and lv.hess_max_eig < 0, (K, lv.grad_max, lv.hess_max_eig)
        assert lv.grid_change < 1e-9, (K, lv.grid_change)
        assert 0 < lv.q[0] and all(a < b for a, b in zip(lv.q, lv.q[1:])) and lv.q[-1] < 1
        assert 0 < lv.mu[0] and all(a < b for a, b in zip(lv.mu, lv.mu[1:]))
    q, mu = np.array([0.3, 0.6, 0.85]), np.array([0.8, 1.7, 3.5])     # off the saddle
    E, gq, gmu = P._levels(q, mu, dh=0.04)
    d = 1e-6
    for i in range(3):
        e = np.eye(3)[i] * d
        fd_q = (P._levels(q + e, mu, dh=0.04, grad=False)
                - P._levels(q - e, mu, dh=0.04, grad=False)) / (2 * d)
        fd_m = (P._levels(q, mu + e, dh=0.04, grad=False)
                - P._levels(q, mu - e, dh=0.04, grad=False)) / (2 * d)
        assert abs(fd_q - gq[i]) < 1e-8 and abs(fd_m - gmu[i]) < 1e-8, (i, fd_q, gq, fd_m, gmu)


@slow
def test_sk_ground_energy_fullrsb_uses_the_parisi_solver():
    from replicant.families.parisi import parisi_ground_energy
    from replicant.families.sk import sk_ground_energy
    e = sk_ground_energy("fullrsb")
    assert e == parisi_ground_energy(grid=GRID).value
    assert check(REF["fullrsb_ground_energy"], e, label="sk_ground_energy").ok


@slow
def test_sk_frsb_at_T_half():
    """T = 0.5 (beta = 2), h = 0.  Level 1 is sk_1rsb's point (f to 1e-9, e to 1e-8);
    f rises with K (maximised) and the extrapolated full-RSB f lies above the 1RSB and RS
    ones by a few 1e-5 (accuracy: error_estimate, the whole extrapolated tail, < 1e-5)."""
    from replicant.families.parisi import sk_frsb
    from replicant.families.sk import sk_1rsb, sk_rs
    s, one, rs = sk_frsb(2.0), sk_1rsb(2.0), sk_rs("2.0")
    assert "numerical" in s.caveat
    assert abs(s.f_levels[0] - float(one.f)) < 1e-9
    assert all(a < b for a, b in zip(s.f_levels, s.f_levels[1:])), s.f_levels
    assert rs.f < one.f < s.f
    assert mp.mpf("1e-5") < s.f - one.f < mp.mpf("1e-4")
    assert s.error_estimate < mp.mpf("1e-5")
    assert s.grad_max <= 1e-7 and s.hess_max_eig < 0
    assert len(s.qs) == len(s.ms) + 1 and all(0 < m < 1 for m in s.ms)
    assert abs(s.e - one.e) < mp.mpf("1e-3")      # e of the finest level, near 1RSB's
    assert not math.isnan(float(s.e))


@slow
def test_catalog_reports_fullrsb_with_its_band():
    """The SK entry computes the full-RSB energy only behind its flag, with the numerical
    error in the run's errors and a finality that cites the Parisi-formula proofs."""
    from replicant.catalog import catalog
    e = catalog()["sk"]
    state, _ = e.execute(fullrsb=True)
    v, errs = state["values"], state["errors"]
    assert "fullrsb_ground_energy" not in e.run()          # default run: not computed
    fin = e.finality_of(v)["fullrsb_ground_energy"]
    assert fin.state == "exact-by-literature" and "Talagrand" in fin.cite
    assert errs["fullrsb_ground_energy"] > 0 and "error" in fin.note
    assert check(REF["fullrsb_ground_energy"], repr(float(v["fullrsb_ground_energy"])),
                 label="catalog").ok
