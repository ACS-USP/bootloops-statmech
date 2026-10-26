"""Held-out routes by Monte Carlo and exact enumeration (E11; split out of
tests/test_heldout.py, T15): SK (e) Gaussian ground energy by parallel tempering, SK (f) +-J
energy at beta = 2 against the composed RS and 1RSB energies, SK (g) exact E ln Z against
ln E Z and the moment closure, Perceptron (e) exact counts at N = 12..24.  The logic is in
tests/heldout_mc.py; the conventions and the full list of routes are in test_heldout.py's
docstring.  None uses the composer to measure; every comparison target is computed after the
measurement."""
import os
import sys

import mpmath as mp
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant import derive as D
from replicant.compose import compose
from replicant.refdata import load
from replicant.testing import slow

import heldout_mc as MC

from heldout_common import (_entropy, _ising_perceptron, _pm_one_sk, _with, _zero_failures)



@slow
def test_pt_sk_energy_extrapolates_to_parisi():
    """(e) Gaussian SK ground energy: PT minima (two independent runs per sample must agree;
    exact enumeration at N = 16) at N = 16, 32, 64, 128, extrapolated as e_inf + a N^-2/3
    (band: 99% t plus the omega = 3/4 and drop-N = 16 shifts).  The banked Parisi values lie
    inside; the RS value -sqrt(2/pi) lies outside (positive control).  1RSB (-0.76526) is 2e-3
    from Parisi, below the attainable band: printed, not asserted."""
    fails, lines = MC.ground_state_check()
    print("\n".join(lines))
    assert not fails, fails


@slow
def test_pm_one_sk_finite_T_energy_vs_composed_rs_and_1rsb():
    """(f) +-J SK at beta = 2 (T = 0.5).  PT energies at N = 64, 128, 256 (equilibration
    diagnostic required at each size) against the composed RS energy (S - phi)/beta and the
    composed 1RSB energy -d phi/d beta (central difference), both computed by the composer.
    Asserted: at N = 256 both composed values lie below the 99% band (the brief's "RS-vs-PT
    difference exceeds the band"; a finite-size statement only, it does NOT discriminate RS
    from 1RSB: 1RSB lies below too); the extrapolated band (omega = 2/3, stressed by 3/4 and 1)
    contains the composed 1RSB energy; the planted doubled +-J variance (composed RS energy) is
    excluded.  Margins (printed): the plant lies ~20 band widths from the centre, RS only 0.012
    (0.65 band widths: NOT excluded).  "1RSB closer than RS" is printed with the measured
    resolution and not asserted (not resolved at this budget)."""
    import dataclasses
    calc = compose(_pm_one_sk())
    law = D.LAWS["pm_one"]
    doubled = dataclasses.replace(
        law, cgf=lambda t, N: sp.log(sp.cosh(sp.sqrt(2) * t / sp.sqrt(N))))
    planted = _with(D.LAWS, "pm_one", doubled, lambda: MC.composed_energy_rs(compose(_pm_one_sk())))
    assert D.LAWS["pm_one"] is law
    fails, lines = MC.finite_t_check(MC.composed_energy_rs(calc),
                                     MC.composed_energy_1rsb(compose(_pm_one_sk(), ansatz="1rsb")),
                                     planted)
    print("\n".join(lines))
    assert not fails, fails


@slow
def test_enumeration_logZ_vs_annealed_and_quenched_expansion():
    """(g) +-J SK (the held-out law), beta = 0.7 and 0.9 (< 1), N = 12, 16, 20, 22: exact
    ln Z per sample by Gray-code enumeration; E ln Z - ln E Z over many samples (99% t band).
    Asserted at every (N, beta): the annealed value 0 is excluded (quenched != annealed, per
    spin a gap of O(1/N)).  The +-J second-order closure -L2/2 (exact finite-N moments) is
    compared within the band plus |order 2 - order 3| only where that tolerance excludes 0
    (otherwise the comparison cannot fail where annealed would pass): asserted at beta = 0.7,
    where it is required to be resolved at every N; at beta = 0.9 the truncation estimate
    (0.03-0.065) is the size of the signal, so it is printed "NOT resolved", not asserted.
    Control: at beta = 0.7 the same closure with Gaussian-law moments is excluded (the law's
    correction matters); at 0.9 printed only."""
    fails, lines = MC.log_z_check()
    print("\n".join(lines))
    assert not fails, fails


@slow
def test_ising_perceptron_log_count_vs_composed_entropy():
    """(e) Exact counts by meet-in-the-middle Gray-code blocks at N = 12, 16, 20, 24 (the
    largest size the 2-minute budget allows: N = 24 is ~0.1 s per pattern set), nested
    patterns P = 3N.  ln 0 hazard: the estimator is E ln(1 + count) / N (defined at every
    sample), and the band includes its shift to the estimator conditioned on count > 0.
    1/N extrapolation, band = 99% t + the largest shift over (drop N = 12, quadratic in 1/N,
    conditioned estimator).  Asserted: the composed S(alpha) at 0.25, 0.5, 0.75 inside; the
    annealed ln2 (1 - alpha) excluded at every alpha (measured: resolved at all three); the
    capacity extrapolation contains the composed zero and the banked threshold.  Controls:
    the composed spherical-site entropy (alpha = 0.25, 0.5), the spherical capacity 2 and the
    annealed capacity 1 are excluded."""
    calc = compose(_ising_perceptron())
    S = {a: _entropy(calc, a) for a in MC.PC_ALPHAS}
    zfails, zero = _zero_failures(calc)
    assert not zfails, zfails

    def spherical():
        bad = compose(_ising_perceptron())
        return {a: _entropy(bad, a, dps=12) for a in ("0.25", "0.5")}
    sph = _with(D.SITES, "ising", D.SITES["spherical"], spherical)
    assert D.SITES["ising"] is not D.SITES["spherical"]
    bank = mp.mpf(load("perceptron")["ising_1rsb_capacity"].value)
    fails, lines = MC.perceptron_check(S, zero, sph, bank)
    print("\n".join(lines))
    assert not fails, fails

