"""E6b (a): the rank-one spiked Wigner model and its Bayes-optimal AMP.

Convention (inference/spiked.py; Montanari & Venkataramanan 2021, Eq. (2.1)):
    Y = (lambda / N) x x^T + W,   W ~ GOE(N): W_ij ~ N(0, 1/N) (i < j), W_ii ~ N(0, 2/N),
x_i i.i.d. from a zero-mean, unit-variance prior (gaussian | rademacher).  The spectral
(BBP-type) edge is at lambda = 1 in this convention.  Lelarge & Miolane (2019) write
Y = sqrt(lambda_LM / n) X X^T + Z with unit-variance Z, so lambda_LM = lambda^2.

State evolution (MV Eq. (2.24)): gamma_{t+1} = lambda^2 (1 - mmse(gamma_t)), mmse the scalar
channel y = sqrt(gamma) X + G.  At the reported fixed point gamma*:
    q        = gamma* / lambda^2       (= LM's q*(lambda_LM): the overlap x.X of a posterior sample)
    overlap  = sqrt(q)                  (normalised cosine |<xhat, x>| / (|xhat| |x|), MV (2.28))
    mmse     = 1 - q                    (vector MSE up to the global sign, MV (2.29))
    matrix_mmse = 1 - q^2               (LM Theorem 1, the theorem quantity)
Gaussian prior (LM Sec. 2.3, Eq. (10)): q* = max(0, 1 - 1/lambda_LM), i.e. q = 1 - 1/lambda^2 here,
and the matrix MMSE is (1/lambda^2)(2 - 1/lambda^2) for lambda > 1, 1 below.  Bank: reference/spiked.json.
"""
import math
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.catalog import catalog, check_ref
from replicant.compose import compose
from replicant.core import Refusal
from replicant.inference import spiked as S
from replicant.refdata import check, load
from replicant.testing import slow

BANK = load("spiked")


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
    """Swap one PRIORS entry for the duration of a block; restore it with an identity check."""
    def __init__(self, name, prior):
        self.name, self.prior = name, prior

    def __enter__(self):
        self.saved = S.PRIORS[self.name]
        S.PRIORS[self.name] = self.prior
        return self

    def __exit__(self, *exc):
        S.PRIORS[self.name] = self.saved
        assert S.PRIORS[self.name] is self.saved
        return False


def _wrong_gaussian():
    g = S.PRIORS["gaussian"]
    return S.Prior("gaussian", g.nodes, g.weights, lambda y, gam: y / (1 + 2 * gam),
                   lambda y, gam: np.full_like(y, 1 / (1 + 2 * gam)), g.sample)


_GAUSS_KEYS = (("1p5", "1.5"), ("2", "2"), ("3", "3"))


# ---- the Gaussian closed form (bank) ---------------------------------------------------------
def test_spiked_se_mmse_matches_bank_formula():
    """TARGET: LM Eq. (10) / Sec. 2.3 (q* = 1 - 1/lambda_LM, lambda_LM = lambda^2).  Route: the
    code's SE fixed point, from the posterior-mean denoiser of the PRIORS hook, a trapezoid rule over
    G and Gauss-Hermite over X (no closed form typed in).  Finite-N AMP: the slow test below."""
    dps = mp.mp.dps
    for tag, lam in _GAUSS_KEYS:
        se = S.spiked_state_evolution(lam, "gaussian")
        assert _ok(f"q_gauss_snr{tag}", se.q), (lam, se)
        assert _ok(f"matrix_mmse_gauss_snr{tag}", se.matrix_mmse), (lam, se)
        L = float(lam)
        assert abs(se.overlap**2 - (1 - 1 / L**2)) < 1e-10 and abs(se.mmse - 1 / L**2) < 1e-10
        assert se.bayes and len([g for g in se.fixed_points if g > 0]) == 1, se
    low = S.spiked_state_evolution("0.8", "gaussian")
    assert low.q == 0 and low.fixed_points == (0,) and _ok("matrix_mmse_gauss_snr0p8", low.matrix_mmse)
    # positive control: a planted denoiser in the hook (posterior mean for the wrong channel)
    with _Planted("gaussian", _wrong_gaussian()):
        bad = S.spiked_state_evolution("2", "gaussian")
        assert not _ok("q_gauss_snr2", bad.q), bad
        assert not bad.bayes, "the planted denoiser is not stationary for the RS potential"
    assert mp.mp.dps == dps


def test_spiked_se_fixed_point_is_the_global_minimum_of_the_rs_potential():
    """Independent route: Psi(gamma) = lam^2/4 + gamma^2/(4 lam^2) - gamma/2 + I(gamma) (DAM Eq. (9)
    with lambda_DAM = lam^2), I from the cited closed forms by mpmath quadrature (not the SE
    quadrature): its minimiser over [0, lam^2] is the SE root."""
    for prior in ("gaussian", "rademacher"):
        for lam in ("1.3", "2"):
            se = S.spiked_state_evolution(lam, prior)
            L2 = float(lam)**2
            grid = [L2 * k / 400 for k in range(401)]
            vals = [float(S.spiked_potential(lam, g, prior)) for g in grid]
            gmin = grid[int(np.argmin(vals))]
            assert abs(gmin - se.gamma) <= L2 / 400, (prior, lam, gmin, se.gamma)
            assert vals[0] > float(S.spiked_potential(lam, se.gamma, prior))
            h = 1e-5
            d = float(S.spiked_potential(lam, se.gamma + h, prior)
                      - S.spiked_potential(lam, se.gamma - h, prior)) / (2 * h)
            assert abs(d) < 1e-7, (prior, lam, d)


def test_spiked_rademacher_se_is_not_closed_form_but_below_gaussian():
    """MMSE_rademacher <= MMSE_gaussian at the same lambda.  Why: for ANY unit-variance prior the
    linear estimator sqrt(gamma) y / (1 + gamma) has error 1/(1 + gamma) = mmse_gaussian(gamma), so
    mmse_P(gamma) <= mmse_gaussian(gamma) pointwise (the Gaussian is the worst prior of a given
    variance), and the SE map gamma -> lam^2 (1 - mmse(gamma)) is monotone, so its largest fixed
    point is ordered the same way."""
    for g in (0.05, 0.3, 1.0, 3.0, 10.0):
        mr, mg = S.scalar_mmse("rademacher", g), S.scalar_mmse("gaussian", g)
        assert mr < mg and abs(mg - 1 / (1 + g)) < 1e-12, (g, mr, mg)
    for lam in ("0.5", "1.2", "1.5", "2", "3"):
        r, gs = S.spiked_state_evolution(lam, "rademacher"), S.spiked_state_evolution(lam, "gaussian")
        assert r.mmse <= gs.mmse + 1e-14 and r.matrix_mmse <= gs.matrix_mmse + 1e-14, (lam, r, gs)
        if float(lam) > 1:
            assert r.q > gs.q + 1e-3, (lam, r.q, gs.q)          # strictly better: not the closed form
            assert r.bayes
        else:
            assert r.q == gs.q == 0
    # the Rademacher fixed point solves q = E tanh(lam^2 q + lam sqrt(q) Z) (LM Sec. 2.3), by mpmath
    r = S.spiked_state_evolution("1.5", "rademacher")
    with mp.workdps(20):
        L2 = mp.mpf("2.25")
        rhs = mp.quad(lambda z: mp.tanh(L2 * r.q + mp.sqrt(L2 * r.q) * z) * mp.npdf(z), [-mp.inf, mp.inf])
    assert abs(rhs - r.q) < 1e-10, (rhs, r.q)
    # positive control: a planted (non-Bayes) denoiser tanh(y/2) breaks the ordering
    rad = S.PRIORS["rademacher"]
    bad = S.Prior("rademacher", rad.nodes, rad.weights, lambda y, gam: np.tanh(y / 2),
                  lambda y, gam: (1 - np.tanh(y / 2)**2) / 2, rad.sample)
    with _Planted("rademacher", bad):
        assert S.scalar_mmse("rademacher", 1.0) > S.scalar_mmse("gaussian", 1.0)


def test_spiked_threshold_is_one_for_both_priors():
    """lambda_c from the linearisation of the SE map at gamma = 0 (slope lambda^2 (-mmse'(0)),
    computed by Richardson extrapolation on the hook's quadrature): 1 for unit-variance priors."""
    for prior in ("gaussian", "rademacher"):
        lc = S.spiked_threshold(prior)
        assert isinstance(lc, mp.mpf) and _ok(f"lambda_c_{prior}", lc), (prior, lc)
        assert S.spiked_state_evolution("0.99", prior).q == 0
        assert S.spiked_state_evolution("1.01", prior).q > 0
    g = S.PRIORS["gaussian"]
    var2 = S.Prior("gaussian", g.nodes * math.sqrt(2), g.weights, lambda y, gam: 2 * y / (1 + 2 * gam),
                   lambda y, gam: np.full_like(y, 2 / (1 + 2 * gam)), g.sample)
    with _Planted("gaussian", var2):         # planted: variance-2 prior -> lambda_c = 1/2
        assert not _ok("lambda_c_gaussian", S.spiked_threshold("gaussian"))


def test_spiked_se_near_the_threshold():
    """Fix round 1 (Important 1): at lambda = 1 -+ 1e-5 and at 1 the SE neither crashes nor reports
    a root made of quadrature noise.  1 - mmse is computed as 2 E[X F] - E[F^2] (no cancellation
    against E X^2 = 1), so g(gamma) = lambda^2 (1 - mmse) - gamma has relative accuracy ~1e-15
    down to the bracketing floor 1e-9 lambda^2 (roots below it are not searched)."""
    for prior in ("gaussian", "rademacher"):
        for lam in ("0.99999", "1"):
            se = S.spiked_state_evolution(lam, prior)
            assert se.q == 0 and se.fixed_points == (0,) and se.bayes, (prior, lam, se)
        se = S.spiked_state_evolution("1.00001", prior)
        assert len(se.fixed_points) == 2 and se.bayes, se
        if prior == "gaussian":
            want = 1 - 1 / mp.mpf("1.00001")**2
            # the root's condition number is 1/(lambda^2 - 1) = 5e4: 1e-9 relative (measured 1.2e-10)
            assert abs(se.q - want) < 1e-9 * float(want), (se.q, want)
        else:                                   # rademacher q = lambda^2 - 1 + O((lambda^2 - 1)^2)
            assert abs(se.q - 2.00001e-5) < 1e-9, se.q
        assert abs(S.scalar_mmse(prior, 1e-9) - (1 - S.scalar_overlap(prior, 1e-9))) < 1e-14
    _refuses(lambda: S.spiked_potential("2", -1e-9, "rademacher"), "precondition:gamma")
    from replicant.cli import main
    import io
    for lam in ("0.99999", "1", "1.00001"):
        out, err = io.StringIO(), io.StringIO()
        assert main(["run", "spiked_wigner", "--route", "amp", "--param", f"snr={lam}",
                     "--param", "prior=rademacher"], out, err) == 0, err.getvalue()


# ---- finite N ----------------------------------------------------------------------------------
def test_spiked_matrix_normalisation():
    Y, x = S.spiked_matrix(400, "2", "rademacher", 0)
    W = Y - 2 / 400 * np.outer(x, x)
    off = W[np.triu_indices(400, 1)]
    assert abs(400 * off.var() - 1) < 0.02 and abs(400 * np.diag(W).var() - 2) < 0.35
    assert np.allclose(Y, Y.T) and set(np.unique(x)) == {-1.0, 1.0}
    top = np.linalg.eigvalsh(Y)[-1]
    assert abs(top - 2.5) < 0.1, top          # lambda + 1/lambda (outlier above the edge 2)


def test_spiked_amp_small_n_runs_and_reports_sign_free_metrics():
    ens = S.spiked_amp_ensemble(300, "2", "gaussian", seeds=(0, 1, 2), iters=15)
    assert len(ens.q) == 3 and all(0 < q < 1 for q in ens.q)
    assert abs(ens.bands["q"].mean - 0.75) < 0.15     # sanity only at N = 300; the test is slow
    Y, _ = S.spiked_matrix(300, "2", "gaussian", 0)
    r = S.spiked_amp(Y, "gaussian", snr="2", iters=5, seed=0)
    assert r.xhat.shape == (300,) and len(r.gamma_trace) == 6


@slow
def test_spiked_amp_n2000_band_contains_bank_formula():
    """N = 2000, seeds 0-31: the 99% Student-t band of q_hat = |<xhat, x>| / N contains the banked
    Gaussian closed form at lambda = 1.5 and 2 (and the code's SE value), and the band of the
    matrix MSE contains the banked matrix MMSE.  Planted controls inside the engine: the Onsager
    term removed (power iteration: same direction, wrong normalisation; N -> oo q =
    sqrt((lam^2 - 1)/(lam^2 + 1)) = 0.620 at 1.5, 0.775 at 2) and scaled by 0.5: their bands
    exclude the banked value.  Per-matrix spread of q_hat at N = 2000 is about 0.03 (build), so
    the bands (about +-0.013) resolve the 0.065 (lambda = 1.5) and 0.025 (lambda = 2) shifts.  The
    lambda = 1.5 controls are the robust ones; the lambda = 2 no-Onsager control is weaker (its band
    0.7763 +- 0.0117 misses the bank by 0.0146, its gap to the Bayes-AMP mean 0.0249 is about 2.1
    half-widths; fixed seeds)."""
    seeds = tuple(range(32))
    for tag, lam in (("1p5", "1.5"), ("2", "2")):
        want = float(BANK[f"q_gauss_snr{tag}"].value)
        ens = S.spiked_amp_ensemble(2000, lam, "gaussian", seeds=seeds, iters=30)
        b = ens.bands["q"]
        assert abs(b.mean - want) <= b.half_width, (lam, b)
        assert abs(b.mean - float(S.spiked_state_evolution(lam, "gaussian").q)) <= b.half_width
        mb = ens.bands["matrix_mse"]
        assert abs(mb.mean - float(BANK[f"matrix_mmse_gauss_snr{tag}"].value)) <= mb.half_width, mb
        naive = S.spiked_amp_ensemble(2000, lam, "gaussian", seeds=seeds, iters=30, reaction=False)
        assert abs(naive.bands["q"].mean - want) > naive.bands["q"].half_width, (lam, naive.bands)
    saved = S._onsager_scale
    try:
        S._onsager_scale = 0.5
        half = S.spiked_amp_ensemble(2000, "1.5", "gaussian", seeds=seeds, iters=30)
    finally:
        S._onsager_scale = saved
    assert S._onsager_scale == saved
    assert abs(half.bands["q"].mean - float(BANK["q_gauss_snr1p5"].value)) > half.bands["q"].half_width


@slow
def test_spiked_amp_n2000_rademacher_band_contains_se():
    seeds = tuple(range(32))
    for lam in ("1.5", "2"):
        se = S.spiked_state_evolution(lam, "rademacher")
        ens = S.spiked_amp_ensemble(2000, lam, "rademacher", seeds=seeds, iters=30)
        b = ens.bands["q"]
        assert abs(b.mean - se.q) <= b.half_width, (lam, b, se.q)
        g = S.spiked_state_evolution(lam, "gaussian")
        assert abs(b.mean - g.q) > b.half_width, (lam, b, g.q)     # the Gaussian value is excluded


# ---- refusals ----------------------------------------------------------------------------------
def test_spiked_refusals_are_named():
    for bad in (-1, "-0.5", "x", float("nan"), None, True):
        _refuses(lambda: S.spiked_state_evolution(bad, "gaussian"), "precondition:snr")
    for bad in ("laplace", "", None):
        _refuses(lambda: S.spiked_state_evolution("2", bad), "precondition:known-prior")
        _refuses(lambda: S.spiked_threshold(bad), "precondition:known-prior")
    r = _refuses(lambda: S.spiked_amp_ensemble(300, "0.9", "gaussian", seeds=(0, 1), iters=5),
                 "precondition:snr-above-one")
    assert "spectral" in r.detail
    for bad in (5, 300.0, "300"):
        _refuses(lambda: S.spiked_matrix(bad, "2", "gaussian", 0), "precondition:size")
    for bad in (-1, "a", 1.5, True):
        _refuses(lambda: S.spiked_matrix(300, "2", "gaussian", bad), "precondition:named-seeds")
    for bad in ([], "x", (0, -2)):
        _refuses(lambda: S.spiked_amp_ensemble(300, "2", "gaussian", seeds=bad, iters=5),
                 "precondition:named-seeds")
    Y, _ = S.spiked_matrix(50, "2", "gaussian", 0)
    _refuses(lambda: S.spiked_amp(Y[:, :40], "gaussian", snr="2", iters=3, seed=0),
             "precondition:square-matrix")
    A = Y.copy()
    A[0, 1] += 1
    _refuses(lambda: S.spiked_amp(A, "gaussian", snr="2", iters=3, seed=0), "precondition:symmetric-matrix")
    _refuses(lambda: S.spiked_amp(Y, "gaussian", snr="2", iters=0, seed=0), "precondition:positive-iters")
    _refuses(lambda: S.spiked_amp(Y, "gaussian", snr="2", iters=3, seed=0, init="warm"),
             "precondition:known-init")


# ---- catalog entry -----------------------------------------------------------------------------
def test_catalog_entry_spiked_wigner():
    dps = mp.mp.dps
    e = catalog()["spiked_wigner"]
    assert e.routes == {"amp"} and e.default_route == "amp"
    res = compose(e.spec, "amp").run()
    assert _ok("lambda_c_gaussian", res.values["lambda_c"])
    assert res.finality["lambda_c"].state == "exact-by-literature" and res.final
    assert "Lelarge" in res.finality["lambda_c"].cite
    pt = compose(e.spec, "amp").run(snr="2")
    assert _ok("q_gauss_snr2", pt.values["q_se"]) and _ok("matrix_mmse_gauss_snr2", pt.values["matrix_mmse_se"])
    assert pt.finality["matrix_mmse_se"].state == "exact-by-literature" and pt.final
    assert "Montanari" in pt.finality["overlap_se"].cite
    rad = compose(e.spec, "amp").run(snr="1.5", prior="rademacher")
    g15 = compose(e.spec, "amp").run(snr="1.5")
    assert rad.values["q_se"] > g15.values["q_se"] and rad.final
    fin = compose(e.spec, "amp").run(snr="2", N=300, seeds=(0, 1, 2), iters=10)
    assert "q_amp" in fin.statistical and not fin.final
    assert fin.finality["q_amp"].state == "not-computed" and fin.errors["q_amp"] > 0
    r = _refuses(lambda: compose(e.spec, "amp").run(lam="2"), "precondition:known-parameter")
    assert "snr" in r.detail
    _refuses(lambda: compose(e.spec, "amp").run(snr="2", seeds=(0, 1)), "precondition:known-parameter")
    _refuses(lambda: compose(e.spec, "amp").run(N=300), "precondition:state-point")
    _refuses(lambda: compose(e.spec, "replica"), "missing-step:route:replica")
    assert e.derivation("amp")
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "GUIDE.md")) as f:
        guide = f.read()
    assert "| `spiked_wigner`" in guide and "precondition:snr-above-one" in guide
    assert all(check_ref(e, r).ok for r in e.refs if not r.slow)
    assert mp.mp.dps == dps
