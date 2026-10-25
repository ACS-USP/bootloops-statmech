"""E7: ridge and ridgeless least squares on isotropic Gaussian design (double descent).

Convention (learning/ridge.py; Hastie, Montanari, Rosset & Tibshirani, Ann. Statist. 50, 949
(2022), arXiv:1903.08560, Eqs. (1)-(2), (4)-(5)): X in R^{n x p} with i.i.d. N(0, 1) entries
(rows N(0, I_p)), y = X beta + sigma xi, |beta|^2 = r^2 ("signal"), gamma = p/n, ridge
beta_lam = (X^T X + n lam I)^{-1} X^T y.  Reported risk: the excess risk
R = E[(x0 . (beta_hat - beta))^2 | X] = |beta_hat - beta|^2 (HMRT's R_X, bias + variance); the
test MSE with a fresh noisy label is R + sigma^2 (Dobriban & Wager's predictive risk).
Bank: reference/ridge.json (formula evaluations at chosen points, sympy exact arithmetic).
"""
import io
import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.catalog import catalog, check_ref
from replicant.compose import compose
from replicant.core import Refusal
from replicant.learning import ridge as R
from replicant.learning import ridge_rs as RS
from replicant.learning import ridge_sim as SIM
from replicant.refdata import check, load
from replicant.testing import slow

BANK = load("ridge")


def _ok(key, got):
    return check(BANK[key], got, label=key).ok


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


class _Planted:
    """Scale one term of the closed form (the private hook R._SCALE) inside a block; restore it."""
    def __init__(self, term, value):
        self.term, self.value = term, value

    def __enter__(self):
        self.saved = dict(R._SCALE)
        R._SCALE[self.term] = self.value
        return self

    def __exit__(self, *exc):
        R._SCALE.clear()
        R._SCALE.update(self.saved)
        assert R._SCALE == {"bias": 1, "variance": 1}
        return False


_RIDGELESS = (("ridgeless_g0p5_s1_r1", "0.5", "1", "1"), ("ridgeless_g0p9_s1_r1", "0.9", "1", "1"),
              ("ridgeless_g1p1_s1_r1", "1.1", "1", "1"), ("ridgeless_g2_s1_r1", "2", "1", "1"),
              ("ridgeless_g4_s0p25_r1", "4", "0.25", "1"))
_RIDGE = (("ridge_g1_lam0p5_s1_r1", "1", "0.5", "1", "1"),
          ("ridge_g2_lam0p1_s0p5_r2", "2", "0.1", "0.5", "2"),
          ("ridge_g0p5_lam1_s1_r1", "0.5", "1", "1", "1"))


# ---- closed forms vs the bank, the peak, the lambda -> 0 limit ------------------------------
def test_ridgeless_peak_at_gamma_one_and_matches_bank():
    """TARGET: HMRT Eq. (8) (Thm 1 for gamma > 1, Prop. 2 for gamma < 1) evaluated at chosen points.
    Routes: the code's piecewise limit; the lambda -> 0+ limit of its Stieltjes-transform ridge
    risk (HMRT Cor. 5, Eq. (43)); the RS saddle point at small lambda (learning/ridge_rs.py)."""
    dps = mp.mp.dps
    for key, g, s2, r2 in _RIDGELESS:
        assert _ok(key, R.ridgeless_test_error(g, s2, r2)), key
        lam0 = R.ridge_test_error(g, "1e-15", s2, r2)
        assert abs(lam0 - R.ridgeless_test_error(g, s2, r2)) < 1e-9 * (1 + lam0), (key, lam0)
        rs = RS.ridge_replica_rs(g, "1e-12", s2, r2).risk
        assert abs(rs - R.ridgeless_test_error(g, s2, r2)) < 1e-8 * (1 + rs), (key, rs)
    with _Planted("bias", 0):                    # positive control: the bank sees a missing bias
        assert not _ok("ridgeless_g2_s1_r1", R.ridgeless_test_error("2", "1", "1"))
    # lam = 0 is the ridgeless limit itself
    assert R.ridge_test_error("2", 0, "1", "1") == R.ridgeless_test_error("2", "1", "1")
    # divergence from both sides: R ~ sigma^2 / |1 - gamma| (below: gamma sigma^2/(1 - gamma))
    for k in (2, 4, 6):
        below = R.ridgeless_test_error("0." + "9" * k, "1", "1")
        above = R.ridgeless_test_error("1." + "0" * (k - 1) + "1", "1", "1")
        assert abs(below - (10**k - 1)) < 1e-12 * 10**k, (k, below)          # gamma/(1 - gamma)
        assert abs(above - (10**k + mp.mpf(1) / (10**k + 1))) < 1e-12 * 10**k, (k, above)
    ups = [R.ridgeless_test_error(g, "1", "1") for g in ("0.5", "0.9", "0.99")]
    downs = [R.ridgeless_test_error(g, "1", "1") for g in ("1.01", "1.1", "2")]
    assert ups == sorted(ups) and downs == sorted(downs, reverse=True)
    # at gamma = 1 the ridge risk blows up as lam -> 0+ (variance ~ lam^(-1/2))
    r6, r8 = R.ridge_test_error("1", "1e-6", "1", "1"), R.ridge_test_error("1", "1e-8", "1", "1")
    assert r6 > 400 and 9 < r8 / r6 < 11, (r6, r8)
    r = _refuses(lambda: R.ridgeless_test_error("1", "1", "1"), "precondition:gamma-not-one")
    assert "diverges" in r.detail
    # noiseless: no peak, the bias alone (0 below gamma = 1)
    assert R.ridgeless_test_error("0.5", 0, "1") == 0
    assert mp.mp.dps == dps


def test_ridge_closed_form_matches_bank_and_the_replica_saddle():
    """TARGET: HMRT Cor. 5 Eq. (43) at chosen points (bank, from the printed m(z) by sympy).
    Route 2: the RS saddle-point equations of the linear student (Loureiro et al. 2022, Thm 1 /
    Eq. (A.38), isotropic), solved without the Stieltjes transform; planted bias / variance errors
    in the closed form are seen by the saddle point (it does not read the hook)."""
    dps = mp.mp.dps
    for key, g, lam, s2, r2 in _RIDGE:
        assert _ok(key, R.ridge_test_error(g, lam, s2, r2)), key
    grid = [(g, lam, s2, r2) for g in ("0.3", "1", "2.5") for lam in ("0.05", "0.7")
            for s2, r2 in (("0", "1"), ("0.5", "2"))]
    for g, lam, s2, r2 in grid:
        cf, rs = R.ridge_test_error(g, lam, s2, r2), RS.ridge_replica_rs(g, lam, s2, r2)
        assert abs(cf - rs.risk) < 1e-12 * (1 + cf), (g, lam, s2, r2, cf, rs.risk)
        b, v = R.ridge_bias_variance(g, lam, s2, r2)
        assert abs(b + v - cf) < 1e-14 * (1 + cf) and b >= 0 and v >= 0
    for term, scale in (("bias", 0), ("variance", "1.1")):
        with _Planted(term, mp.mpf(scale)):
            cf = R.ridge_test_error("2.5", "0.05", "0.5", "2")
            rs = RS.ridge_replica_rs("2.5", "0.05", "0.5", "2").risk
            assert abs(cf - rs) > 1e-3, (term, cf, rs)
    assert mp.mp.dps == dps


# ---- simulation: independent of the closed form ----------------------------------------------
_SEEDS = tuple(range(20))


def _band(n, gamma, lam, s2, r2, seeds=_SEEDS):
    p = int(round(float(mp.mpf(gamma)) * n))
    assert p == float(mp.mpf(gamma)) * n
    return SIM.ridge_ensemble(n, p, lam, s2, seeds, r2)


def _limit_band(ns, case, seeds=_SEEDS):
    """Raw bands per n and the band of the 1/n extrapolation R_inf (ridge_sim.extrapolate_in_n)."""
    raw = {n: _band(n, *case, seeds=seeds) for n in ns}
    return raw, SIM.extrapolate_in_n(raw)


def _inside(band, x):
    return abs(float(x) - band.mean) <= band.half_width


_CASES = [("0.5", 0, "1", "1"), ("2", 0, "1", "1"), ("1", "0.5", "1", "1")]
# planted variance factors the bands resolve (variance terms 1, 1, 1/3; 0.174 for the lam* case)
_VAR_DEFAULT = {"0.5": "1.1", "2": "1.1", "1": "1.5"}


def test_simulation_matches_closed_form():
    """Default tier: HMRT's finite-n risk R_X from an SVD fit (ridge_sim; never calls the closed
    form) at n = 80, 160 over 20 named seeds each, extrapolated in 1/n; the 99% Student-t band of
    R_inf (half-widths 0.037, 0.061, 0.051 at gamma = 0.5, 2 and the ridge point) contains the
    closed form.  Controls planted in R._SCALE: the ridgeless risk at gamma = 2 and the ridge risk
    at gamma = 1, lam = 0.5 without their bias (shifts 0.5 and 1/3) fall outside, as do a +10%
    variance at gamma = 0.5 and 2 (shift 0.1) and a +50% variance at the ridge point (shift 1/6;
    +10% there, 0.033, is below this band's resolution).
    Why extrapolate: R_X's mean is off the limit by O(1/n), as large as the O(1/n) fixed-n band
    (gamma = 0.5: E R_X = p/(n - p - 1) exactly for Gaussian X, the Wishart mean; the n = 160 band
    contains 80/79; at n = 200 over seeds 0-19 the fixed-n band 1.015 +- 0.011 misses the limit 1).
    n = 400..1000: the slow test below."""
    for c in _CASES:
        raw, inf = _limit_band((80, 160), c)
        cf = R.ridge_test_error(*c)
        assert _inside(inf, cf), (c, inf, cf)
        assert inf.half_width < 0.07, inf
        if c[0] != "0.5":
            with _Planted("bias", 0):
                assert not _inside(inf, R.ridge_test_error(*c)), (c, inf)
        with _Planted("variance", mp.mpf(_VAR_DEFAULT[c[0]])):
            assert not _inside(inf, R.ridge_test_error(*c)), (c, inf)
        if c[0] == "0.5":
            assert _inside(raw[160], mp.mpf(80) / 79), raw[160]


@slow
def test_simulation_matches_closed_form_large_n():
    """n = 400 and 1000, 24 named seeds per point, extrapolated in 1/n: ridgeless gamma = 0.5, 2;
    ridge at gamma = 1 (lam = 0.5) and at the optimal lam* = sigma^2 gamma / r^2 for gamma = 2,
    r^2 = 2.  R_inf half-widths 0.006, 0.020, 0.015, 0.035.  Controls: missing bias (shifts 0.5,
    1/3, 1.11); variance +5% at gamma = 0.5, 2 (0.05), +25% at the two ridge points (0.083, 0.044);
    raw band half-widths shrink with n."""
    seeds = tuple(range(24))
    factor = {"0.5": "1.05", "2": "1.05", "1": "1.25", "lam*": "1.25"}
    for c in _CASES + [("2", "0.5", "0.5", "2")]:
        raw, inf = _limit_band((400, 1000), c, seeds)
        assert _inside(inf, R.ridge_test_error(*c)), (c, inf)
        if c[0] != "0.5":
            with _Planted("bias", 0):
                assert not _inside(inf, R.ridge_test_error(*c)), (c, inf)
        with _Planted("variance", mp.mpf(factor["lam*" if c[1] == "0.5" and c[0] == "2" else c[0]])):
            assert not _inside(inf, R.ridge_test_error(*c)), (c, inf)
        assert raw[1000].half_width < raw[400].half_width, (c, raw)


def test_ridge_simulate_is_reproducible_and_uses_the_seed():
    a, b = SIM.ridge_simulate(60, 30, "0.1", "0.5", 3), SIM.ridge_simulate(60, 30, "0.1", "0.5", 3)
    c = SIM.ridge_simulate(60, 30, "0.1", "0.5", 4)
    assert a == b and a != c and isinstance(a, float)
    # lam = 0 above the interpolation threshold is the min-norm interpolator: zero training error
    fit = SIM.ridge_fit(40, 80, 0, "1", 0)
    assert fit.train_residual < 1e-10, fit.train_residual


# ---- the optimal penalty ------------------------------------------------------------------------
def test_optimal_lambda_removes_the_peak():
    """HMRT Cor. 5 (and Dobriban-Wager Thm 2.1, Advani-Saxe Sec. 2.2): lam* = sigma^2 gamma / r^2
    minimises the limiting risk, which is then sigma^2 gamma m(-lam*).  Checks: lam* against the
    bank; a numerical argmin of ridge_test_error (golden section, no formula for lam*) agrees, and
    a wrong rule sigma^2 / r^2 does not; R* + sigma^2 against Dobriban-Wager Eq. (8) (bank); R* is
    finite at gamma = 1, increasing in gamma, below the ridgeless risk and the null risk r^2."""
    dps = mp.mp.dps
    assert _ok("lambda_opt_g2_s1_r1", R.optimal_lambda("2", "1", "1"))
    assert _ok("lambda_opt_g2_s1_r5", R.optimal_lambda("2", "1", "5"))
    for key, g, a2 in (("optimal_test_mse_g0p5_a1", "0.5", "1"), ("optimal_test_mse_g1_a1", "1", "1"),
                       ("optimal_test_mse_g2_a1", "2", "1"), ("optimal_test_mse_g2_a5", "2", "5")):
        assert _ok(key, R.optimal_test_error(g, "1", a2) + 1), key
    # positive control: the wrong rule lam = sigma^2 / r^2 misses the DW value at gamma = 2
    assert not _ok("optimal_test_mse_g2_a1", R.ridge_test_error("2", "1", "1", "1") + 1)
    for g, s2, r2 in (("0.5", "1", "1"), ("1", "1", "5"), ("3", "0.5", "2")):
        lam_num = R.argmin_lambda(g, s2, r2)
        lam_star = R.optimal_lambda(g, s2, r2)
        assert abs(lam_num - lam_star) < 1e-8 * (1 + lam_star), (g, lam_num, lam_star)
        assert abs(R.optimal_test_error(g, s2, r2) - R.ridge_test_error(g, lam_star, s2, r2)) < 1e-20
        wrong = mp.mpf(s2) / mp.mpf(r2)          # planted rule without gamma
        if mp.mpf(g) != 1:
            assert abs(lam_num - wrong) > 1e-2 and \
                R.ridge_test_error(g, wrong, s2, r2) > R.ridge_test_error(g, lam_star, s2, r2) + 1e-6
    for r2 in ("1", "5"):
        gs = [mp.mpf(x) / 10 for x in range(1, 101)]
        opt = [R.optimal_test_error(g, "1", r2) for g in gs]
        assert all(b > a for a, b in zip(opt, opt[1:])), r2          # monotone increasing in gamma
        assert all(x < mp.mpf(r2) for x in opt)                          # below the null risk r^2
        for g, x in zip(gs, opt):
            if g != 1:
                assert x < R.ridgeless_test_error(g, "1", r2), (g, x)
        assert mp.isfinite(R.optimal_test_error("1", "1", r2))
    # sigma^2 = 0: no positive minimiser; the infimum is the ridgeless limit (lam -> 0+)
    assert R.optimal_lambda("2", 0, "1") == 0
    assert R.optimal_test_error("2", 0, "1") == R.ridgeless_test_error("2", 0, "1")
    assert mp.mp.dps == dps


# ---- refusals -----------------------------------------------------------------------------------
def test_ridge_refusals_are_named():
    for bad in (0, -1, "-0.5", "x", float("nan"), float("inf"), None, True):
        _refuses(lambda: R.ridge_test_error(bad, "0.1"), "precondition:gamma")
        _refuses(lambda: R.ridgeless_test_error(bad, "1"), "precondition:gamma")
        _refuses(lambda: RS.ridge_replica_rs(bad, "0.1"), "precondition:gamma")
    for bad in (-1, "-1e-3", "y", float("nan"), None):
        _refuses(lambda: R.ridge_test_error("2", bad), "precondition:lambda")
        _refuses(lambda: SIM.ridge_simulate(50, 50, bad, "1", 0), "precondition:lambda")
    _refuses(lambda: RS.ridge_replica_rs("2", 0), "precondition:lambda")
    for bad in (-1, "-0.1", "z", None):
        _refuses(lambda: R.ridge_test_error("2", "0.1", bad), "precondition:sigma2")
        _refuses(lambda: R.ridgeless_test_error("2", bad), "precondition:sigma2")
        _refuses(lambda: SIM.ridge_simulate(50, 50, "0.1", bad, 0), "precondition:sigma2")
    for bad in (-1, "w", None):
        _refuses(lambda: R.ridge_test_error("2", "0.1", "1", bad), "precondition:signal")
    _refuses(lambda: R.optimal_lambda("2", "1", 0), "precondition:signal")
    for n, p in ((5, 50), (50, 5), (50.0, 50), ("50", 50), (True, 50)):
        _refuses(lambda: SIM.ridge_simulate(n, p, "0.1", "1", 0), "precondition:size")
    for bad in (-1, "0", 1.5, True, None):
        _refuses(lambda: SIM.ridge_simulate(50, 50, "0.1", "1", bad), "precondition:named-seeds")
    for bad in ([], "01", (0, -2)):
        _refuses(lambda: SIM.ridge_ensemble(50, 50, "0.1", "1", bad), "precondition:named-seeds")
    _refuses(lambda: SIM.ridge_ensemble(50, 50, "0.1", "1", (0,)), "precondition:two-samples")
    b = SIM.ridge_ensemble(50, 50, "0.1", "1", (0, 1))
    _refuses(lambda: SIM.extrapolate_in_n({50: b}), "precondition:two-sizes")


# ---- catalog entry --------------------------------------------------------------------------------
def test_catalog_entry_ridge():
    dps = mp.mp.dps
    e = catalog()["ridge"]
    assert e.routes == {"replica"} and e.default_route == "replica"
    assert all(check_ref(e, r).ok for r in e.refs if not r.slow)
    res = compose(e.spec).run(gamma="2", lam="0.1", sigma2="0.5", signal="2")
    v, fin = res.values, res.finality
    assert abs(v["risk"] - mp.mpf(BANK["ridge_g2_lam0p1_s0p5_r2"].value)) < 1e-12
    assert abs(v["risk_rs"] - v["risk"]) < 1e-12 and abs(v["test_mse"] - v["risk"] - mp.mpf("0.5")) < 1e-14
    for k in ("risk", "bias", "variance", "test_mse", "risk_ridgeless", "lambda_opt", "risk_opt"):
        assert fin[k].state == "exact-by-literature" and "1903.08560" in fin[k].cite, k
    assert fin["risk_rs"].state == "exact-by-literature" and "2102.08127" in fin["risk_rs"].cite
    assert "not derive" in e.caveats["replica"] and "random-matrix" in e.caveats["replica"]
    # gamma = 1: the ridgeless step is skipped (divergent), the rest runs
    res1 = compose(e.spec).run(gamma="1", lam="0.5")
    assert "risk_ridgeless" not in res1.values and "risk" in res1.values
    # finite n: statistical, never final
    res2 = compose(e.spec).run(gamma="2", lam="0.1", n=60, seeds=[0, 1, 2, 3])
    assert "risk_sim" in res2.statistical and res2.errors["risk_sim"] > 0
    assert res2.finality["risk_sim"].state == "not-computed"
    for params, reason in (({}, "precondition:state-point"),
                           ({"gamma": "2", "seeds": [0, 1]}, "precondition:known-parameter"),
                           ({"gamma": "2", "lamda": "0.1"}, "precondition:known-parameter")):
        _refuses(lambda: compose(e.spec).run(**params), reason)
    _refuses(lambda: compose(e.spec, "cavity"), "missing-step:route:cavity")
    assert e.derivation("replica")
    from replicant.cli import main
    err = io.StringIO()
    code = main(["run", "ridge", "--param", "gamma=2", "--param", "n=50", "--param", "seeds=-1,"],
                io.StringIO(), err)
    assert code == 2 and "REFUSED: precondition:named-seeds" in err.getvalue(), err.getvalue()
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "GUIDE.md")) as f:
        guide = f.read()
    assert "| `ridge`" in guide and "`precondition:gamma`" in guide and "`:gamma-not-one`" in guide
    assert mp.mp.dps == dps
