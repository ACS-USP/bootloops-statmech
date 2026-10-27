"""M1, default tier: the shared Monte Carlo statistics module (montecarlo/stats.py) on series with known answers,
each check with a positive control that must fail."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from replicant.core import Refusal
from replicant.montecarlo import stats as ST


def _ar1(phi, n, seed):
    """A stationary AR(1) series x_t = phi x_{t-1} + e_t as the truncated moving average sum_k phi^k e_{t-k}
    (k < K with phi^K < 1e-15: exact to that level)."""
    K = int(math.ceil(math.log(1e-15) / math.log(phi)))
    e = np.random.default_rng(seed).standard_normal(n + K)
    return np.convolve(e, phi ** np.arange(K), mode="valid")[:n]


def test_tau_int_recovers_the_ar1_value_and_rejects_a_wrong_one():
    # AR(1): rho(t) = phi^t, tau_int = (1 + phi) / (2 (1 - phi)) exactly; 8 named seeds, 99% Student-t band
    phi = 0.8
    exact = (1 + phi) / (2 * (1 - phi))                      # 4.5
    ts = [ST.tau_int(_ar1(phi, 20000, s)) for s in range(8)]
    b = ST.band([t.tau for t in ts])
    assert abs(b.mean - exact) <= b.half_width + 0.05 * exact, (b, exact)   # + the window's bias, < 5% at c = 6
    assert all(t.reliable and t.window >= 6 * t.tau for t in ts)
    assert abs(b.mean - 1.5 * exact) > b.half_width              # control: a 50% wrong tau is excluded
    se = np.mean([t.stderr for t in ts])
    assert 0.3 < se / (b.stderr * math.sqrt(8)) < 3               # Eq. (4.13) error vs the seed spread
    short = ST.tau_int(_ar1(0.99, 400, 1))
    assert not short.reliable and "UNRELIABLE" in short.text      # n < 100 tau: flagged


def test_binning_and_jackknife_on_a_correlated_series():
    # the binned stderr over the naive one tends to sqrt(2 tau_int) = 3 for phi = 0.8 (8 named seeds, t-band)
    xs = [_ar1(0.8, 32768, s) for s in range(8)]
    ratio = ST.band([ST.binned(x, 32).stderr / (x.std(ddof=1) / math.sqrt(x.size)) for x in xs])
    assert abs(ratio.mean - 3) <= ratio.half_width and abs(ratio.mean - 1) > ratio.half_width
    x = xs[3]
    curve = ST.binning_curve(x)
    assert curve[0][1] < curve[-3][1] and curve[0][0] == 1
    var = ST.jackknife(lambda a: a[0].var(), [x], 32)
    assert abs(var.estimate - 1 / (1 - 0.64)) < 5 * var.stderr    # Var = 1 / (1 - phi^2)
    assert abs(var.estimate - 1.0) > 5 * var.stderr               # control: the innovation variance is excluded


def test_equilibration_flags_a_drift_and_passes_a_stationary_series():
    rng = np.random.default_rng(11)
    flat = rng.standard_normal((8, 400))
    assert ST.equilibration(flat).equilibrated
    drift = flat + np.linspace(0, 1, 400)[None, :]
    eq = ST.equilibration(drift)
    assert not eq.equilibrated and "undetermined" in eq.text
    try:
        ST.equilibration(flat[:1])
    except Refusal as r:
        assert r.reason == "precondition:windows"
    else:
        raise AssertionError("one run accepted")


def test_chi2_accepts_the_true_law_and_rejects_a_wrong_one():
    p = np.array([0.5, 0.3, 0.15, 0.04, 0.01])
    rng = np.random.default_rng(5)
    obs = np.bincount(rng.choice(5, 4000, p=p), minlength=5)
    ok = ST.chi2_gof(obs, p, 0.01)
    assert ok.ok and ok.df == 4 and ok.alpha == 0.01
    wrong = ST.chi2_gof(obs, np.array([0.45, 0.35, 0.15, 0.04, 0.01]), 0.01)
    assert not wrong.ok
    zero = ST.chi2_gof(np.array([10, 0, 1]), np.array([0.6, 0.4, 0.0]), 0.01)
    assert not zero.ok                                           # a count in an impossible cell


def test_power_fit_separates_a_power_from_zero_and_from_a_log():
    Ls = np.array([8, 16, 32, 64])
    rng = np.random.default_rng(2)
    power = 2.0 * Ls[None, :] ** 0.5 * np.exp(0.01 * rng.standard_normal((8, 4)))
    f = ST.power_fit(Ls, power)
    assert abs(f.z.mean - 0.5) < f.z.half_width + 0.01 and f.separable_from_zero and f.separable_from_log
    log = (1 + 0.8 * np.log(Ls))[None, :] * np.exp(0.01 * rng.standard_normal((8, 4)))
    g = ST.power_fit(Ls, log)
    assert not g.separable_from_log and "NOT separable from a logarithm" in g.text   # control
    flat = 3.0 * np.exp(0.05 * rng.standard_normal((8, 4)))
    assert not ST.power_fit(Ls, flat).separable_from_zero
