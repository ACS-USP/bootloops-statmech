"""O4: online learning in the tree committee machine (papers 4, 5 of the online spec).

Copelli & Caticha, J. Phys. A 28, 1615 (1995) (CC95; read from the preprint IFUSP/P-1126, October 1994,
publica-sbi.if.usp.br/PDFs/pd1126.pdf, linked from repositorio.usp.br item 000878722).  Copelli, Kinouchi &
Caticha, Phys. Rev. E 53, 6341 (1996) (CKC96): ONLY THE ABSTRACT was read (OpenAlex record of the APS
abstract); the full text, its epsilon(K) and its figures were not fetched (APS 403, no preprint on arXiv, the
neuroprose file is gone, not among the IFUSP preprints P-1127-P-1300).

Convention (online/tcm.py): K odd non-overlapping branches of N/K inputs, |B_k| = 1, xi ~ N(0, I), teacher
label sigma_B = sign(sum_k sign(B_k . xi_k)) (CC95 Eq. (1)), J_k <- J_k + (1/N) F_k xi_k (CC95 Eq. (6) with
Omega = 0, eta folded into F), alpha = examples / N (= CKC96's alpha_CM, examples per adjustable weight),
eps = P(TCM student != TCM teacher), branch-symmetric state rho_k = rho.
Bank: reference/online_tcm.json.
"""
import ast
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.core import Refusal
from replicant.online import odes
from replicant.online import optimal as OPT
from replicant.online import tcm as T
from replicant.online import tcm_flow as TF
from replicant.refdata import check, load
from replicant.testing import slow

BANK = load("online_tcm")


def _ok(key, got):
    return check(BANK[key], got, label=key).ok


@contextmanager
def planted(module, attr="_PLANT", **changes):
    """Set module.<attr> entries for the block; restore the SAME dict object (identity asserted)."""
    hook = getattr(module, attr)
    saved = dict(hook)
    try:
        hook.update(changes)
        yield
    finally:
        hook.clear()
        hook.update(saved)
        assert getattr(module, attr) is hook and hook == saved


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


def test_tie_probability_and_mapping():
    """The internal-representation average: P(the other K-1 teacher units tie) is the central binomial
    C(K-1, (K-1)/2) / 2^(K-1) (typed here), and the mapped noise is (1 - P0)/2: 0, 1/4, 5/16 ..."""
    for K in (1, 3, 5, 7, 11, 21):
        p0 = math.comb(K - 1, (K - 1) // 2) / 2 ** (K - 1)
        assert abs(T.tie_probability(K) - p0) < 1e-15
        assert abs(T.effective_noise(K) - (1 - p0) / 2) < 1e-15
        assert sum(T.tie_distribution(K).values()) == 1.0 or abs(sum(T.tie_distribution(K).values()) - 1) < 1e-14
    assert T.effective_noise(3) == 0.25 and T.perceptron_alpha(2.0, 3) == 6.0


def test_committee_error_is_cc95_eq5():
    """eps_committee (a DP over the branch states) against CC95 Eq. (5) typed here, for K = 3 and
    unequal branch errors; K = 1 is the branch error; the small-e slope is K P0."""
    eq5 = lambda e1, e2, e3: e1 * e2 * e3 + 0.5 * (e1 + e2 + e3 - e1 * e2 - e1 * e3 - e2 * e3)
    for es in ((0.1, 0.2, 0.3), (0.01, 0.4, 0.05), (0.5, 0.5, 0.5), (0.0, 0.0, 0.25)):
        assert abs(T.eps_committee(es, 3) - eq5(*es)) < 1e-15, es
    assert abs(T.eps_committee(0.37, 1) - 0.37) < 1e-15
    for K in (3, 5, 9):
        e = 1e-7
        assert abs(T.eps_committee(e, K) / e - K * T.tie_probability(K)) < 1e-5
    assert abs(T.eps_committee(0.5, 7) - 0.5) < 1e-14       # an uncorrelated student


def test_tcm_route_shares_no_code_with_the_perceptron_side():
    """O-R2: tcm.py and tcm_flow.py import nothing of O1's averages (fields), optimal modulation
    (optimal) or simulators; tcm_flow takes only the integrator _dopri from odes; the averages and the
    rates never read the mapping (effective_noise, perceptron_alpha)."""
    for mod, allowed in ((T, set()), (TF, {"_dopri"})):
        tree = ast.parse(Path(mod.__file__).read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.level >= 1 and n.module in ("odes", "fields", "optimal",
                                                                               "simulate", "drift"):
                assert {a.name for a in n.names} <= allowed, (mod.__name__, n.module)
            if isinstance(n, ast.ImportFrom) and n.level >= 1:
                assert n.module not in ("fields", "optimal", "simulate", "simulate_tcm"), n.module
    for fn in (T.averages, T._local_grid, T._nonlocal_grid, TF.TcmFlow._theta_lnq, TF.TcmFlow.rates):
        src = ast.unparse(ast.parse(__import__("inspect").getsource(fn).strip()))
        assert "effective_noise" not in src and "perceptron_alpha" not in src and "mapping" not in src


_TCM = {}


def _gap(K, rule="local_optimal", alphas=(0.5, 2.0, 5.0), eta=1.0):
    """(largest |rho_TCM(alpha) - rho_perceptron(K alpha; noise = effective_noise(K))|, the TCM
    trajectory).  The TCM trajectory is integrated once per (K, rule, eta) and reused: it never reads
    the mapping (asserted under the plant below)."""
    init = (0.05, 1.0)
    key = (K, rule, eta, alphas)
    if key not in _TCM:
        _TCM[key] = TF.integrate_tcm(TF.tcm_flow(rule, K, eta=eta), max(alphas), init, alphas=alphas)
    tr = _TCM[key]
    if rule == "local_optimal":
        pf = odes.perceptron_flow("optimal", noise=T.effective_noise(K))
    else:
        pf = odes.perceptron_flow("hebb", eta=eta / K, noise=T.effective_noise(K))
    pa = [T.perceptron_alpha(a, K) for a in alphas]
    pt = odes.integrate(pf, max(pa), init, alphas=pa)
    return float(np.max(np.abs(tr.rho - pt.R / np.sqrt(pt.Q)))), tr


def test_ckc96_equivalence_local_learning_is_a_noisy_perceptron():
    """CKC96 (abstract): local learning in a TCM with K hidden units is learning in a simple perceptron
    with a corresponding noise level.  Two routes: the TCM ODE (tcm_flow: its own field averages over the
    other branches' internal representations) and O1's perceptron ODE with output noise at the mapped
    noise effective_noise(K) and time K alpha.  A planted wrong mapping (noise or time) breaks the
    identity; the TCM trajectory itself does not move under the mapping plant (K = 5, 7: slow test)."""
    g, tr0 = _gap(3)
    assert g < 1e-8, g
    g, _ = _gap(3, rule="hebb", eta=1.5)                    # any local rule: F_TCM = K F_perceptron
    assert g < 1e-8, g
    with planted(T, mapping=0.8):
        bad, _ = _gap(3)
        fresh = TF.integrate_tcm(TF.tcm_flow("local_optimal", 3), 5.0, (0.05, 1.0), alphas=(0.5, 2.0, 5.0))
    assert bad > 1e-3, bad
    assert np.array_equal(tr0.rho, fresh.rho)                # the TCM route never reads the mapping
    with planted(T, time=1 / 3):
        assert _gap(3)[0] > 1e-3


def test_cc95_eq21_is_the_nonlocal_rate():
    """CC95 Eq. (21) typed here (with 1 - r written as H(-y2)H(-y3) + H(-y1)D, which is the same), on a
    tensor Gauss-Hermite grid in y_j = h_j / lambda_j, against the TCM route's nonlocal rate (a posterior
    over the internal representations in the variables u = x cot theta).  The local rule's rate (which
    ignores the other fields) is far from Eq. (21)."""
    from scipy.special import ndtr
    H = lambda x: ndtr(-x)
    gx, gw = np.polynomial.hermite_e.hermegauss(120)
    gw = gw / math.sqrt(2 * math.pi)
    for theta in (0.5, 0.9, 1.3):
        rho, lam = math.cos(theta), math.tan(theta)
        y1, y2, y3 = (gx[:, None, None] / lam, gx[None, :, None] / lam, gx[None, None, :] / lam)
        D = H(-y2) * H(y3) + H(y2) * H(-y3)
        r = H(y2) * H(y3) + H(y1) * D
        r1 = H(-y2) * H(-y3) + H(-y1) * D
        w = gw[:, None, None] * gw[None, :, None] * gw[None, None, :]
        den = r * r1                                         # 0 only where D^2 e^{-y1^2} underflows too
        integ = np.where(den > 0, np.exp(-y1 ** 2) * D ** 2 / np.where(den > 0, den, 1.0), 0.0)
        eq21 = 3 / (4 * math.pi) * rho * lam ** 2 * np.sum(w * integ)
        a = T.averages("optimal", theta, 3)
        drho = a.drive - rho * a.a2 / (2 * 3)                # dR/sqrt(Q) - rho dQ/(2Q) at eta = 1
        assert abs(eq21 / drho - 1) < 1e-6, (theta, eq21, drho)   # measured 1.6e-7, 1e-13, 4e-16
        loc = T.averages("local_optimal", theta, 3)                # positive control: the local rule's rate
        assert abs(eq21 / (loc.drive - rho * loc.a2 / 6) - 1) > 1e-2


def test_cc95_norm_follows_the_overlap():
    """CC95 Eqs. (23)-(25): with the optimal weight the branch norm obeys the overlap's equation, so
    J_k / rho_k stays constant: d ln |J| / dalpha = d ln rho / dalpha (here d ln Q = 2 d ln rho), for the
    nonlocal and the local rule; Hebb (positive control) does not."""
    for rule in ("optimal", "local_optimal"):
        f = TF.tcm_flow(rule, 3)
        for theta in (0.4, 1.1):
            R, Q = math.cos(theta) * 2.0, 4.0
            dR, dQ = f.rates(0.0, R, Q)
            dlnrho = dR / R - dQ / (2 * Q)
            assert abs(dQ / Q - 2 * dlnrho) < 1e-10 * abs(dlnrho) + 1e-13, (rule, theta)
    dR, dQ = TF.tcm_flow("hebb", 3).rates(0.0, 0.8, 1.0)
    assert abs(dQ - 2 * (dR / 0.8 - dQ / 2)) > 0.1


def test_cc95_pseudo_hebb_closed_form():
    """CC95 Eq. (30): rho(alpha) = (1 + 2 pi / (3 alpha))^(-1/2) from rho(0) = 0 (K = 3; plain Hebb from
    J = 0 is their greedy pseudo-Hebb: rho does not depend on the constant step), from the TCM ODE started
    at Q0 = 1e-12; its asymptote sqrt(3 / 2 pi) alpha^(-1/2) from the TCM Hebb closed form (checked
    against the ODE) at alpha = 1e10."""
    f = TF.tcm_flow("hebb", 3, eta=0.7)
    tr = TF.integrate_tcm(f, 20.0, (0.0, 1e-12), alphas=(1.0, 5.0, 20.0))
    for a, rho in zip((1, 5, 20), tr.rho):
        assert _ok(f"cc95_pseudo_hebb_rho_a{a}", rho), (a, rho)
    cf = [TF.hebb_closed_form(a, 3, 0.7, (0.0, 1e-12)) for a in (1.0, 5.0, 20.0)]
    assert max(abs(R / math.sqrt(Q) - r) for (R, Q, _), r in zip(cf, tr.rho)) < 1e-8
    assert not _ok("cc95_pseudo_hebb_rho_a5", float(tr.rho[0]))        # a wrong alpha fails
    big = 1e10
    assert _ok("cc95_pseudo_hebb_coeff", math.sqrt(big) * TF.hebb_closed_form(big, 3, 1.0, (0.0, 1.0))[2])
    assert not _ok("cc95_pseudo_hebb_coeff", math.sqrt(big) * TF.hebb_closed_form(big, 5, 1.0, (0.0, 1.0))[2])


def test_cc95_asymptotic_constant_of_the_nonlocal_rule():
    """CC95 Eq. (32): eps ~ kappa / alpha, kappa = 2 [int Dx e^{-x^2/2} / H(x)]^{-1} ~ 0.88 (the
    printed integral evaluated by mpmath here: 0.883400).  TCM route: the rate limit -dtheta/dalpha /
    theta^2 at theta = 1e-3, 5e-4 (Richardson) with eps/theta -> K P0 / pi.  Positive control: the local
    rule's constant at K = 3 (0.5 C(1/4) = 2.665) is far from it."""
    with mp.workdps(20):
        H = lambda x: mp.erfc(x / mp.sqrt(2)) / 2
        integ = mp.quad(lambda x: mp.npdf(x) * mp.e ** (-x * x / 2) / H(x), [-mp.inf, 0, 5, mp.inf])
        formula = float(2 / integ)
    assert _ok("cc95_kappa_formula", formula)
    kappa = TF.rate_limit_constant(TF.tcm_flow("optimal", 3))
    assert _ok("cc95_kappa", kappa) and _ok("cc95_kappa_formula", kappa), kappa
    assert abs(kappa - OPT.asymptotic_constant(0.0)) < 1e-4          # = the perceptron's (CC95 p. 13)
    loc = TF.rate_limit_constant(TF.tcm_flow("local_optimal", 3))
    assert not _ok("cc95_kappa", loc) and abs(loc - 2.6652) < 1e-3, loc


def _local_constant(K):
    """(TCM route: log-slope and alpha eps at alpha = 1e4, perceptron route: P0(K) C(lambda_K))."""
    slope, const = TF.asymptote_tcm(TF.tcm_flow("local_optimal", K), (0.05, 1.0))
    return slope, const, T.tie_probability(K) * OPT.asymptotic_constant(T.effective_noise(K))


def test_asymptotic_constant_against_K():
    """Local learning: eps ~ kappa_loc(K) / alpha.  TCM route: its ODE at alpha = 1e4 (eps alpha, and
    the log-slope, CKC96 abstract: exponent 1 at finite K).  Perceptron route: P0(K) C(lambda_K) with
    O1's C = 2 / I(lambda) at the mapped noise.  At K = 3 kappa_loc = 2.665, three times the nonlocal
    0.88 (K = 1, 5 and the growth with K: the slow test).  Hebb: exponent 1/2 (CKC96 abstract),
    constant sqrt(K / 2 pi) (derived; K = 3 is CC95's)."""
    slope, const, perc = _local_constant(3)
    assert abs(const / perc - 1) < 2e-3 and _ok("ckc96_optimal_exponent", slope), (slope, const, perc)
    assert abs(TF.rate_limit_constant(TF.tcm_flow("local_optimal", 3)) / perc - 1) < 1e-4
    slope, const = TF.asymptote_tcm(TF.tcm_flow("hebb", 3), (0.0, 1.0))
    assert _ok("ckc96_hebb_exponent", slope) and not _ok("ckc96_optimal_exponent", slope)
    assert abs(const - math.sqrt(3 / (2 * math.pi))) < 2e-2


@slow
def test_asymptotic_constant_grows_with_K():
    """K = 1 (the perceptron's 0.8834), 3, 5, 7: the TCM route equals the perceptron route at each K
    (and the identity holds along the whole curve at K = 5, 7), and kappa_loc grows with K."""
    prev = 0.0
    for K in (1, 3, 5, 7):
        slope, const, perc = _local_constant(K)
        assert abs(const / perc - 1) < 2e-3 and _ok("ckc96_optimal_exponent", slope), (K, slope, const, perc)
        assert const > prev
        prev = const
        if K >= 5:
            assert _gap(K)[0] < 1e-8
    assert abs(_local_constant(1)[2] - OPT.asymptotic_constant(0.0)) < 1e-15


def test_hebb_ode_vs_closed_form_and_planted_term():
    """TCM Hebb: R = R0 + eta a alpha, a = P0 sqrt(2/pi), Q - R^2 = Q0 - R0^2 + eta^2 alpha / K (derived);
    the ODE agrees; the planted 1/K of the eta^2 term (inv_k = K, i.e. dropped) does not."""
    f = TF.tcm_flow("hebb", 5, eta=1.3)
    tr = TF.integrate_tcm(f, 7.0, (0.2, 1.5), alphas=(7.0,))
    R, Q, eps = TF.hebb_closed_form(7.0, 5, 1.3, (0.2, 1.5))
    assert abs(tr.R[0] - R) < 1e-8 and abs(tr.Q[0] - Q) < 1e-7 and abs(tr.eps[0] - eps) < 1e-9
    with planted(TF, inv_k=5.0):
        tr2 = TF.integrate_tcm(f, 7.0, (0.2, 1.5), alphas=(7.0,))
    assert abs(tr2.eps[0] - eps) > 1e-2


def test_refusals():
    _refuses(lambda: TF.tcm_flow("oja", 3), "unsupported:rule")
    _refuses(lambda: TF.tcm_flow("optimal", 5), "unsupported:K")
    _refuses(lambda: TF.tcm_flow("hebb", 4), "precondition:K")
    _refuses(lambda: TF.tcm_flow("hebb", 0), "precondition:K")
    _refuses(lambda: TF.tcm_flow("hebb", True), "precondition:K")
    _refuses(lambda: TF.tcm_flow("local_optimal", 3, eta=2.0), "precondition:optimal-eta")
    _refuses(lambda: TF.tcm_flow("hebb", 3, eta=-1.0), "precondition:eta")
    _refuses(lambda: TF.integrate_tcm(TF.tcm_flow("optimal", 3), 1.0, (0.0, 1.0)), "precondition:initial-overlap")
    _refuses(lambda: TF.integrate_tcm(TF.tcm_flow("hebb", 3), 1.0, None), "precondition:state-point")
    _refuses(lambda: TF.integrate_tcm(TF.tcm_flow("hebb", 3), -1.0, (0.0, 1.0)), "precondition:alpha")
    _refuses(lambda: TF.rate_limit_constant(TF.tcm_flow("hebb", 3)), "precondition:asymptote")
    _refuses(lambda: T.averages("hebb", 4.0, 3), "precondition:overlap")
    _refuses(lambda: T.eps_committee((0.1, 0.2), 3), "precondition:K")


@slow
def test_cc95_figure1_and_the_integrated_asymptote():
    """CC95 Fig. 1 (expected stability, K = 3, read off the figure) against the nonlocal TCM ODE from
    rho0 = 1e-3; the pseudo-Hebb curve read off the same figure agrees with Eq. (30), which calibrates the
    reading.  The nonlocal constant from the ODE integrated to alpha = 1e4 agrees with the rate limit."""
    tr = TF.integrate_tcm(TF.tcm_flow("optimal", 3), 16.5, (1e-3, 1.0), alphas=(3.3, 6.7, 10.0, 16.5))
    for a, e in zip(("3p3", "6p7", "10", "16p5"), tr.eps):
        assert _ok(f"cc95_fig1_es_a{a}", float(e)), (a, e)
    slope, const = TF.asymptote_tcm(TF.tcm_flow("optimal", 3), (0.05, 1.0))
    assert abs(slope - 1) < 1e-2 and abs(const - TF.rate_limit_constant(TF.tcm_flow("optimal", 3))) < 2e-3
