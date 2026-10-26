"""Parallel tempering (E11): numerics/pt.py.

Conventions: H = -sum_{i<j} J_ij s_i s_j (h = 0), J symmetric with zero diagonal; energies per
spin.  Independent route for detailed balance: the exact +-J spectrum of
numerics/enumeration.py (Gray-code blocks; no Monte Carlo), whose thermal mean energy at the
same beta is compared with the PT time average.  Statistical: one PT run per disorder sample,
independent seeds; the band is a 99% Student-t band over samples (the per-sample run averages
are independent; within one run the sweeps are correlated, which the sample average sidesteps).
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from replicant.amp._stats import student_band
from replicant.core import Refusal
from replicant.numerics import enumeration as E
from replicant.numerics import pt as PT
from replicant.testing import slow


def _refusal(fn):
    try:
        fn()
    except Refusal as r:
        return r
    raise AssertionError("expected a Refusal")


def _planted(module, key, value, fn):
    reg = module._HOOKS
    old = reg[key]
    reg[key] = value
    try:
        return fn()
    finally:
        reg[key] = old
        assert module._HOOKS is reg and reg[key] is old


DB_N, DB_SAMPLES, DB_SWEEPS, DB_BETAS = 8, 48, 600, (0.5, 1.0, 1.5, 2.0)


def _detailed_balance_residuals(seed):
    """Per sample: PT time average of e (second half of the run) minus the exact thermal mean
    energy per spin at the same beta, for each beta.  Returns (samples x betas) array."""
    Js = np.stack([E.random_pm_one_J(DB_N, [seed, k]) for k in range(DB_SAMPLES)])
    res = PT.pt_sk(Js, DB_BETAS, DB_SWEEPS, seed)
    tr = res.energy_per_spin_trace[DB_SWEEPS // 2:]          # sweeps x samples x betas
    exact = np.array([[E.enumerate_pm_one_sk(J).mean_energy_per_spin(b) for b in DB_BETAS]
                      for J in Js])
    return tr.mean(axis=0) - exact, res


def test_pt_detailed_balance_against_exact_enumeration():
    """PT time averages at N = 8 against the exact spectrum at every beta: the 99% band of the
    mean residual over 48 samples contains 0.  Positive control: the next test."""
    r, res = _detailed_balance_residuals(20261026)
    for k, b in enumerate(DB_BETAS):
        band = student_band(r[:, k])
        assert abs(band.mean) <= band.half_width, (b, band)
    assert res.energy_per_spin_trace.shape == (DB_SWEEPS, DB_SAMPLES, len(DB_BETAS))
    assert (res.swap_acceptance > 0.2).all(), res.swap_acceptance
    # bookkeeping: the tracked energy equals the energy recomputed from the final spins
    assert np.allclose(res.final_energy_per_spin, res.energy_per_spin_trace[-1], atol=1e-12)


def test_pt_detailed_balance_control_detects_planted_acceptance():
    """The planted Metropolis rule exp(-0.8 beta dE) (detailed balance at beta' = 0.8 beta, so
    the wrong equilibrium) is detected by the same gate at every beta."""
    r, _ = _planted(PT, "acceptance_scale", 0.8, lambda: _detailed_balance_residuals(20261026))
    for k, b in enumerate(DB_BETAS):
        band = student_band(r[:, k])
        print(f"planted beta={b}: residual {band.mean:+.4f} +- {band.half_width:.4f}")
        assert abs(band.mean) > band.half_width, (b, band)


def test_pt_finds_exact_ground_states_small_n():
    """Gaussian J, N = 12: the PT minimum over the run equals the exact enumeration minimum
    in every sample (an independent route for the ground-state finder)."""
    rng = np.random.default_rng(5)
    Js = []
    for _ in range(16):
        G = np.triu(rng.standard_normal((12, 12)), 1)
        Js.append((G + G.T) / math.sqrt(12))
    Js = np.stack(Js)
    res = PT.pt_sk(Js, PT.geometric_betas(0.3, 4.0, 8), 200, 1)
    exact = np.array([E.exact_ground_energy(J) for J in Js])
    assert np.allclose(res.min_energy_per_spin, exact, atol=1e-12), (res.min_energy_per_spin, exact)
    assert (res.min_sweep < 100).all()


def test_pt_equilibration_diagnostic_flags_a_cold_start_transient():
    """Window comparison (log binning: [T/4, T/2) against [T/2, T)): a short run read from
    sweep 0 (the random, infinite-temperature start) is flagged at beta = 2; a 400-sweep run
    read over its default windows passes."""
    Js = np.stack([E.random_pm_one_J(8, [3, k]) for k in range(24)])
    short = PT.equilibration(PT.pt_sk(Js, DB_BETAS, 4, 9), beta_index=3, start=0)
    assert not short.equilibrated and short.drift < 0, short.text
    ok = PT.equilibration(PT.pt_sk(Js, DB_BETAS, 400, 9), beta_index=3)
    assert ok.equilibrated, ok.text
    assert set(vars(ok)) >= {"equilibrated", "drift", "half_width", "text"}


def test_pt_refusals():
    J = E.random_pm_one_J(6, 0)
    assert _refusal(lambda: PT.pt_sk(J, (1.0, 0.5), 10, 0)).reason == "precondition:betas"
    assert _refusal(lambda: PT.pt_sk(J, (0.5, 1.0), 0, 0)).reason == "precondition:sweeps"
    assert _refusal(lambda: PT.pt_sk(J, (0.5, 1.0), 10, "x")).reason == "precondition:seed"
    asym = J.copy()
    asym[0, 1] += 1
    assert _refusal(lambda: PT.pt_sk(asym, (0.5, 1.0), 10, 0)).reason == \
        "precondition:coupling-matrix"
    assert _refusal(lambda: PT.sk_ground_energy_finite_size((16,), 10, 0)).reason == \
        "precondition:sizes"
    # sweeps=0 is refused, not silently replaced by the default
    assert _refusal(lambda: PT.sk_ground_energy_finite_size((8, 10, 12), 2, 0, sweeps=0)
                    ).reason == "precondition:sweeps"
    from replicant.numerics import finite_size as F
    assert _refusal(lambda: F.sk_energy_finite_size(2.0, (8, 10, 12), 2, 0, nbetas=0)
                    ).reason == "precondition:nbetas"


@slow
def test_pm_one_ground_energies_match_boettcher_table():
    """+-J SK ground energies per spin at N = 15 (exact enumeration, 20000 samples) and N = 31
    (PT minima, two runs per sample must agree, 800 samples) against Boettcher 2005 Table 1
    (reference/mc.json): our 99% t-band widened by 2.58 x the source's quoted s.e."""
    from replicant.numerics import finite_size as F
    from replicant.refdata import load
    ref = load("mc")
    J15 = F.pm_one_sk_couplings(15, 20000, [2026102605, 15])
    e15 = np.array([E.exact_ground_energy(J) for J in J15])
    J31 = F.pm_one_sk_couplings(31, 800, [2026102605, 31])
    bs = PT.geometric_betas(0.4, 4.0, 11)
    a = PT.pt_sk(J31, bs, 600, [2026102605, 31, 1]).min_energy_per_spin
    b = PT.pt_sk(J31, bs, 600, [2026102605, 31, 2]).min_energy_per_spin
    assert np.allclose(a, b, rtol=1e-9, atol=0), int((~np.isclose(a, b, rtol=1e-9)).sum())
    for key, e in (("pm_one_ground_energy_n15", e15), ("pm_one_ground_energy_n31", a)):
        band = student_band(e)
        want, tol = float(ref[key].value), float(ref[key].tol)
        print(f"{key}: ours {band.mean:.5f} +- {band.half_width:.5f} (99% t), Boettcher "
              f"{want} +- {tol}")
        assert abs(band.mean - want) <= band.half_width + 2.58 * tol, (key, band, want)


def test_pt_detailed_balance_control_detects_reversed_exchange():
    """The exchange criterion with its sign reversed (exp(-(beta_k - beta_{k+1})(E_k -
    E_{k+1})): it favours the hotter configuration at the larger beta) breaks detailed balance
    of the product measure; the same gate detects it at every beta (measured residuals
    -0.130, +0.034, +0.051, +0.060 against 99% half-widths 0.015, 0.007, 0.010, 0.010)."""
    r, _ = _planted(PT, "exchange_sign", -1.0, lambda: _detailed_balance_residuals(20261026))
    for k, b in enumerate(DB_BETAS):
        band = student_band(r[:, k])
        print(f"reversed exchange beta={b}: residual {band.mean:+.4f} +- {band.half_width:.4f}")
        assert abs(band.mean) > band.half_width, (b, band)
