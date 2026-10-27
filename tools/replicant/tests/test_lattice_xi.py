"""2D Ising correlation length, critical exponents and critical correlations (I2):
lattice/correlation2d.py, lattice/strip_tm.py, lattice/critical2d.py, catalog_lattice_xi.py.

Routes.  Onsager's strip spectrum (certified balls) against the strip transfer matrix built and
diagonalised directly (numpy, symmetry sectors; no shared code); the strip -> bulk sequence against
the exact bulk forms 2(K* - K) (T > T_c) and 4(K - K*) (T < T_c), a finite-size consistency check
with a stated band; nu and eta as consistency checks on exact data; at T_c Perk & Au-Yang's product
against the Toeplitz determinant of DIK's symbol, DIK's asymptote, and the cylinder's correlations.
Planted errors go through each module's _HOOKS (restored in try/finally, identity asserted).
"""
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import mpmath as mp

from replicant.core import Refusal
from replicant.refdata import load
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


def _bank(key):
    return load("lattice")[key]


def _near(x, key, extra=0):
    rec = _bank(key)
    with mp.workdps(45):
        x = x.mid() if hasattr(x, "mid") else x
        return abs(x - mp.mpf(rec.value)) <= mp.mpf(rec.tol) + extra


def _run(**kw):
    from replicant.catalog import catalog
    from replicant.compose import compose
    return compose(catalog()["ising2d"].spec, "exact").run(**kw)


# ---- the strip: two routes --------------------------------------------------------------------
def test_strip_onsager_spectrum_equals_direct_transfer_matrix():
    from replicant.lattice import correlation2d as C, strip_tm as S
    for beta in ("0.3", "beta_c", "0.6"):
        for n in (2, 3, 4, 5, 6):
            g, (do, de) = C.strip_gaps(n, beta), S.strip_spectrum(n, beta)
            assert abs(g.odd.mid() - do) < 1e-9 and abs(g.even.mid() - de) < 1e-9, (beta, n)
            assert g.odd.rad() < mp.mpf("1e-39") and g.even.rad() < mp.mpf("1e-39")
    assert _refusal(lambda: S.strip_spectrum(11, "0.3")).reason == "precondition:size"
    assert _refusal(lambda: C.strip_gaps(1, "0.3")).reason == "precondition:size"


def test_planted_strip_boundary_conditions_are_refused():
    """Two wrong boundary conditions: the direct matrix without the seam bond (open strip), and
    Onsager's odd sector with |gamma_0| (sign rule of Eq. (99) dropped).  Each breaks the n = 4, 6
    agreement; the catalog step refuses no-solution:strip-disagreement."""
    from replicant.lattice import correlation2d as C, strip_tm as S

    def disagree(beta):
        g, d = C.strip_gaps(4, beta), S.strip_spectrum(4, beta)
        assert max(abs(g.odd.mid() - d[0]), abs(g.even.mid() - d[1])) > 1e-3, beta
        r = _refusal(lambda: _run(beta=beta))
        assert r.reason == "no-solution:strip-disagreement", r
    _planted(S, "boundary", "open", lambda: [disagree(b) for b in ("0.3", "0.6")])
    _planted(C, "gamma0_signed", False, lambda: disagree("0.3"))
    assert _run(beta="0.3").values["correlation_length"]          # restored


# ---- the correlation length ---------------------------------------------------------------------
def test_correlation_length_bank_and_strip_limit():
    from replicant.lattice import correlation2d as C
    assert _near(C.correlation_length("0.3"), "ising2d_correlation_length_beta0_3")
    assert _near(C.correlation_length("0.6"), "ising2d_correlation_length_beta0_6")
    for beta in ("0.3", "0.6"):                 # more points: the slow grid
        lim, xi_inv = C.strip_limit(beta), C.inverse_correlation_length(beta).mid()
        assert lim.resolved and abs(lim.estimate - xi_inv) <= lim.band, (beta, lim)
    near = C.strip_limit("0.45")                 # band 9e-3 > 1/xi / 10: printed, not asserted
    assert not near.resolved and "lambda_2" in near.pair
    assert "lambda_1" in C.strip_limit("0.3").pair
    assert _refusal(lambda: C.correlation_length("beta_c")).reason == "precondition:off-critical"
    v = _run(beta="0.6").values
    assert v["strip_limit"]["resolved"] and "4 (K - K*)" in v["xi_branch"]
    v = _run(beta="0.45").values                  # not resolved: no refusal, reported
    assert v["strip_limit"]["resolved"] is False


def test_planted_wrong_xi_branch_is_refused():
    """The T > T_c form used below T_c (1/xi = 2(K - K*)) is half the strip's two-particle gap: the
    strip estimate at beta = 0.6 (1.15667, band 5.8e-6) excludes 0.578; the step refuses."""
    from replicant.lattice import correlation2d as C

    def check():
        lim = C.strip_limit("0.6")
        assert abs(lim.estimate - C.inverse_correlation_length("0.6").mid()) > 100 * lim.band
        assert _refusal(lambda: _run(beta="0.6")).reason == "no-solution:strip-limit"
    _planted(C, "below_factor", 2, check)
    assert _run(beta="0.6").values["strip_limit"]["resolved"]


# ---- exponents ---------------------------------------------------------------------------------
def test_exponent_consistency_nu_and_eta():
    from replicant.lattice import correlation2d as C, critical2d as D
    for d1, d2 in (("-1e-6", "-1e-8"), ("1e-6", "1e-8")):
        s = C.xi_log_slope(d1, d2).mid()
        assert abs(s + 1) < mp.mpf("1e-5") and abs(s + 1) > mp.mpf("1e-9"), (d1, s)  # O(d) seen
    assert _near(C.xi_amplitude("-1e-12"), "ising2d_correlation_length_amplitude_plus")
    ratio = C.xi_amplitude("-1e-12").mid() / C.xi_amplitude("1e-12").mid()
    assert _near(ratio, "ising2d_correlation_length_amplitude_ratio")
    # Cardy: n ln(lambda_0/lambda_1) at beta_c -> pi eta, corrections ~ 0.32/n^2 (observed)
    a = {n: C.critical_gap_scaled(n).mid() for n in (64, 128, 256)}
    pi4 = mp.mpf(_bank("ising2d_strip_critical_gap_amplitude").value)
    assert 0 < a[256] - pi4 < a[128] - pi4 < mp.mpf("3e-5")
    assert abs((4 * a[256] - a[128]) / 3 - pi4) < mp.mpf("1e-8")
    assert abs((4 * a[256] - a[128]) / 3 - pi4 / 2) > mp.mpf("0.3")   # the brief's pi eta / 2 is excluded
    s = D.diagonal_log_slope(64, 128).mid()
    assert abs(s + mp.mpf(1) / 4) < mp.mpf("1e-5")
    v = _run().values
    assert "nu_above" in v["exponent_checks"] and v["strip_critical_gap_n256"]


def test_planted_wrong_exponents_are_refused():
    from replicant.lattice import correlation2d as C
    for key, val, what in (("nu", Fraction(1, 2), "nu_above"), ("eta", Fraction(1, 2), "eta_cardy"),
                           ("eta", Fraction(1, 8), "eta_diagonal")):
        def check():
            r = _refusal(lambda: _run())
            assert r.reason == "no-solution:exponent-mismatch" and what in r.detail, (key, r)
        _planted(C, key, val, check)
    assert _run().values["nu_loglog_below"]


# ---- critical correlations -------------------------------------------------------------------------
def test_critical_diagonal_correlation_routes():
    from replicant.lattice import critical2d as D
    for n in range(1, 13):
        assert D.diagonal_correlation(n).overlaps(D.diagonal_toeplitz(n)), n
    assert _near(D.diagonal_correlation(1), "ising2d_critical_diagonal_correlation_n1")
    with mp.workdps(45):                 # n = 2: (2/pi)^2 (1 - 1/4)^-1 = 16 / (3 pi^2)
        assert abs(D.diagonal_correlation(2).mid() - 16 / (3 * mp.pi ** 2)) < mp.mpf("1e-39")
    with mp.workdps(30):                 # DIK Eq. (105) from the symbol of Eq. (104), by quadrature
        for p in (-2, -1, 0, 1, 3):
            f = lambda t: mp.exp(1j * (mp.pi - t) / 2) * mp.exp(-1j * p * t)
            a = mp.quad(f, [0, mp.pi, 2 * mp.pi]) / (2 * mp.pi)
            assert abs(a - 2 / (mp.pi * (2 * p + 1))) < mp.mpf("1e-25"), p
    assert _near(D.diagonal_amplitude(), "ising2d_critical_diagonal_amplitude")
    for n, band in ((64, "1e-9"), (128, "1e-10")):    # next term O(n^-4)
        r = D.diagonal_correlation(n).mid() / D.diagonal_asymptote(n).mid() - 1
        assert abs(r) < mp.mpf(band), (n, r)
    assert _refusal(lambda: D.diagonal_toeplitz(61)).reason == "precondition:distance"


def test_critical_nn_and_strip_correlations():
    from replicant.lattice import critical2d as D, strip_tm as S
    nn = D.critical_nn_correlation()
    assert _near(nn, "ising2d_critical_nn_correlation")
    diffs = []
    for L in (3, 4, 6, 8):
        row, col, diag = S.strip_correlations(L, "beta_c")
        assert abs((row + col) / 2 - nn.mid()) < 1e-13, L      # observed identity at every width
        diffs.append(abs(diag - 2 / mp.pi))
    assert all(b < a for a, b in zip(diffs, diffs[1:])) and diffs[-1] < 3e-4, diffs
    row, col, _ = S.strip_correlations(6, "0.3")
    assert abs((row + col) / 2 - nn.mid()) > 0.3                # off beta_c it is not sqrt 2 / 2


def test_planted_critical_correlation_is_refused():
    from replicant.lattice import critical2d as D

    def check():
        assert not D.diagonal_correlation(3).overlaps(D.diagonal_toeplitz(3))
        # the exponent step (diagonal log-slope) refuses first; the correlation step on its own too
        assert _refusal(lambda: _run(n=3)).reason == "no-solution:exponent-mismatch"
        r = _refusal(lambda: XI._crit_corr({"n": 3}))
        assert r.reason == "no-solution:critical-correlation" and "Toeplitz" in r.detail, r
    from replicant import catalog_lattice_xi as XI
    _planted(D, "product_denominator", 2, check)
    assert _refusal(lambda: _run(n=0)).reason == "precondition:distance"


def test_catalog_xi_certified_printed_enclosures_and_precision():
    from flint import arb, ctx
    from replicant.catalog import catalog
    from replicant.cli import result_json
    from replicant.lattice import correlation2d as C, critical2d as D
    from replicant.lattice.ball import baller
    before, fprec = mp.mp.dps, ctx.prec
    e = catalog()["ising2d"]
    for params, keys in (({"beta": "0.6"}, {"correlation_length": lambda: C.correlation_length("0.6")}),
                         ({"n": 5}, {"critical_diagonal_correlation": lambda: D.diagonal_correlation(5),
                                     "critical_nn_correlation": D.critical_nn_correlation})):
        res = _run(**params)
        js = result_json("ising2d", e.spec, res)
        assert res.rigor == "certified"
        for k, ball in keys.items():
            v, err = js["values"][k], js["errors"][k]
            with baller().hygiene.ctx_guard(prec=300):
                b = ball().arb
                assert (arb(v) - arb(err)).upper() <= b.lower() and b.upper() <= (arb(v) + arb(err)).lower()
            assert res.finality[k].state == "exact-by-literature" and "Eq" in res.finality[k].cite
    res = _run()
    assert "correlation_length" not in res.values
    assert res.finality["nu_loglog_above"].state == "not-computed"
    assert mp.mp.dps == before and ctx.prec == fprec


# ---- slow tier -------------------------------------------------------------------------------
@slow
def test_slow_direct_transfer_matrix_to_width_10():
    from replicant.lattice import correlation2d as C, strip_tm as S
    for beta in ("0.3", "beta_c", "0.6"):
        for n in (8, 10):
            g, (do, de) = C.strip_gaps(n, beta), S.strip_spectrum(n, beta)
            assert abs(g.odd.mid() - do) < 1e-9 and abs(g.even.mid() - de) < 1e-9, (beta, n)
    row, col, diag = S.strip_correlations(10, "beta_c")
    assert abs((row + col) / 2 - mp.sqrt(2) / 2) < 1e-13 and abs(diag - 2 / mp.pi) < 1.2e-4


@slow
def test_slow_toeplitz_to_40_and_strip_limit_grid():
    from replicant.lattice import correlation2d as C, critical2d as D
    for n in (20, 30, 40):
        assert D.diagonal_correlation(n).overlaps(D.diagonal_toeplitz(n)), n
    for beta in ("0.1", "0.2", "0.35", "0.4", "0.42", "0.43", "0.47", "0.5", "0.55", "0.7", "1", "2"):
        lim, xi_inv = C.strip_limit(beta), C.inverse_correlation_length(beta).mid()
        assert abs(lim.estimate - xi_inv) <= lim.band, (beta, lim)
        assert lim.resolved or beta in ("0.42", "0.47"), (beta, lim)
