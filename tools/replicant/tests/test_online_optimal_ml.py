"""O6: the variationally optimal online algorithm of the soft committee machine (Vicente & Caticha 1997).

Convention (online/optimal_multilayer.py): VC97's, which is Saad & Solla's soft committee machine (K student, M
teacher erf units, M_nm = delta_nm, J_k <- J_k + (F_k / N) S, alpha = examples / N, e_g = <(Sigma_B - Sigma_J)^2>/2)
with the learning rate folded into the modulation F.  The optimal F = H^-1 G <y>_{H|V} - x (VC97 Eq. (4)) uses
every student field, the teacher's output and the exact order parameters; it is greedy (the largest decay of e_g
per example).  Banked numbers are targets only: Fig. 1-3 readings of VC97's N = 5000 simulations (read from the
vector data of the arXiv PostScript figures), compared with replicant's ODE and with O5's gradient-descent ODE.
"""
import math
import os
import sys
from contextlib import contextmanager
from functools import lru_cache

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.core import Refusal
from replicant.online import optimal_ml_flow as OF
from replicant.online import optimal_multilayer as O
from replicant.online import scm as S
from replicant.online import scm_flow as SF
from replicant.refdata import check, load
from replicant.testing import slow

BANK = load("online_scm")


@contextmanager
def planted(module, **changes):
    hook = module._PLANT
    saved = dict(hook)
    try:
        hook.update(changes)
        yield
    finally:
        hook.clear()
        hook.update(saved)
        assert module._PLANT is hook and hook == saved


def _ok(key, got):
    return check(BANK[key], got, label=key).ok


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


def _state(rng, K, M):
    """A random admissible (Q, R): rows of J and B drawn in R^(K + M + 2)."""
    B = np.linalg.qr(rng.normal(size=(K + M + 2, M)))[0].T
    J = rng.normal(size=(K, K + M + 2)) * 0.5 + 0.4 * rng.normal(size=(K, M)) @ B
    return J @ J.T, J @ B.T


def _gauss(C, n):
    """Tensor Gauss-Hermite nodes and weights of N(0, C) (symmetric square root)."""
    d = C.shape[0]
    w, V = np.linalg.eigh(C)
    A = V * np.sqrt(np.clip(w, 0, None))
    x, wt = np.polynomial.hermite_e.hermegauss(n)
    U = np.stack(np.meshgrid(*([x] * d), indexing="ij"), -1).reshape(-1, d)
    W = np.prod(np.stack(np.meshgrid(*([wt / math.sqrt(2 * math.pi)] * d), indexing="ij"), -1).reshape(-1, d), 1)
    return U @ A.T, W


def _rates_by_quadrature(Q, R, F, n=14):
    """M = 1: dQ, dR of VC97 Eq. (2) by Gauss-Hermite over (x, y) of the pointwise modulation F(x, y) (K x nodes);
    no algebra of the optimal module is used (for M = 1 the teacher output fixes y: y = sqrt2 erfinv(Sigma_B))."""
    K = Q.shape[0]
    C = np.block([[Q, R], [R.T, np.eye(1)]])
    Z, W = _gauss(C, n)
    x, y = Z[:, :K].T, Z[:, K]
    f = F(x, y)
    dR = (f * y * W).sum(1)[:, None]
    dQ = np.einsum("in,jn,n->ij", x, f, W) + np.einsum("in,jn,n->ij", f, x, W) + np.einsum("in,jn,n->ij", f, f, W)
    return dQ, dR


def _decay(Q, R, F):
    """de_g/dalpha of a modulation F: e_g from O5's scm.eps (SS's I2 route) by a central difference along the
    quadrature rates: independent of optimal_multilayer's Eq. (3), its gradients and its W."""
    dQ, dR = _rates_by_quadrature(Q, R, F)
    flow, h = S.scm_flow(Q.shape[0], 1, 1.0), 1e-6
    return (flow.eps(Q + h * dQ, R + h * dR) - flow.eps(Q - h * dQ, R - h * dR)) / (2 * h)


def test_vc97_eq3_and_its_gradients():
    """VC97 Eq. (3) (typed in optimal_multilayer) equals O5's (1/2) s I2 s; D = de/dQ_ij (ordered pairs) and
    G = -de/dR by central differences of O5's eps; the R-dependence of D_ii planted out is caught."""
    rng = np.random.default_rng(6)
    for K, M in ((2, 1), (2, 2), (3, 2)):
        flow = S.scm_flow(K, M, 1.0)
        for _ in range(3):
            Q, R = _state(rng, K, M)
            assert abs(O.eps_vc(Q, R) - flow.eps(Q, R)) < 1e-13
            D, G = O.gradients(Q, R)
            h = 1e-6
            for i in range(K):
                for j in range(K):
                    E = np.zeros((K, K))
                    E[i, j] = h
                    num = (flow.eps(Q + E, R) - flow.eps(Q - E, R)) / (2 * h) if i == j else \
                        (flow.eps(Q + E + E.T, R) - flow.eps(Q - E - E.T, R)) / (4 * h)
                    assert abs(D[i, j] - num) < 1e-7, (K, M, i, j)
            for k in range(K):
                for n in range(M):
                    E = np.zeros((K, M))
                    E[k, n] = h
                    assert abs(G[k, n] + (flow.eps(Q, R + E) - flow.eps(Q, R - E)) / (2 * h)) < 1e-7
    Q, R = _state(rng, 2, 1)
    with planted(O, dqr=0.0):
        D = O.gradients(Q, R)[0]
    E = np.diag([1e-6, 0.0])
    assert abs(D[0, 0] - (S.scm_flow(2, 1, 1.0).eps(Q + E, R) - S.scm_flow(2, 1, 1.0).eps(Q - E, R)) / 2e-6) > 1e-3


def test_m1_rates_against_quadrature_of_the_modulation():
    """M = 1: the closed ODE dR = A W - R, dQ = A W A^T - Q (W = 1, derived from VC97 Eqs. (2), (4)) against
    Gauss-Hermite of the pointwise F* = A y - x; a planted W (x 1.05) is caught."""
    rng = np.random.default_rng(7)
    flow = O.optimal_flow(2, 1)
    for _ in range(3):
        Q, R = _state(rng, 2, 1)
        A = O.modulation_matrix(Q, R)
        dQ, dR = flow.rates(Q, R)
        qQ, qR = _rates_by_quadrature(Q, R, lambda x, y: A @ y[None] - x)
        assert np.abs(dQ - qQ).max() < 1e-10 and np.abs(dR - qR).max() < 1e-10
        with planted(O, w=1.05):
            pQ, pR = flow.rates(Q, R)
        assert np.abs(pQ - qQ).max() > 1e-3


_FIG1_AL = (0.3,) + tuple(np.round(np.arange(0.8, 1.4001, 0.02), 10)) + (2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0)


@lru_cache(maxsize=None)
def _fig1():
    Q0, R0, q11 = OF.vc97_initial(2, 1)
    return q11, OF.integrate_optimal(O.optimal_flow(2, 1), 8.0, (Q0, R0), alphas=_FIG1_AL, rtol=1e-6)


def test_optimality_positive_control():
    """VC97's objective (the decay of e_g per example) at fixed (Q, R), by the independent quadrature route.
    Where H (= delta^2 de/dalpha / delta F delta F) is positive definite, every perturbation of F* decays more slowly:
    a scale 1 +- 0.05, an R misestimated by 10%, a nonlinear tanh perturbation; the excess is quadratic.  Where H
    has a negative eigenvalue (early learning; VC97's Fig. 1 inset), F* is a saddle and a perturbation along the
    negative mode decays faster: the check can fail and does where the paper says it must."""
    tr = _fig1()[1]
    i, j = (int(np.argmin(np.abs(tr.alpha - a))) for a in (0.3, 4.0))
    (Qs, Rs), (Qp, Rp) = (tr.Q[i], tr.R[i]), (tr.Q[j], tr.R[j])
    assert tr.eig_lo[i] < 0 < tr.eig_lo[j]
    A = O.modulation_matrix(Qp, Rp)
    best = _decay(Qp, Rp, lambda x, y: A @ y[None] - x)
    assert best < 0
    wrong = O.modulation_matrix(Qp, 0.9 * Rp)
    phi = lambda x, y: np.tanh(x + y[None])
    for F in (lambda x, y: 1.05 * (A @ y[None] - x), lambda x, y: 0.95 * (A @ y[None] - x),
              lambda x, y: wrong @ y[None] - x, lambda x, y: A @ y[None] - x + 0.05 * phi(x, y)):
        assert _decay(Qp, Rp, F) > best + 1e-7
    with planted(O, hess=0.5):                                     # a wrong F* (H misnormalised) where H > 0
        bad = O.modulation_matrix(Qp, Rp)
    assert _decay(Qp, Rp, lambda x, y: bad @ y[None] - x) > best + 1e-6
    d1 = _decay(Qp, Rp, lambda x, y: A @ y[None] - x + 0.02 * phi(x, y)) - best
    d2 = _decay(Qp, Rp, lambda x, y: A @ y[None] - x + 0.04 * phi(x, y)) - best
    assert 3.5 < d2 / d1 < 4.5                                     # second order: the first variation vanishes
    w, V = np.linalg.eigh(O.hessian(Qs, Rs))
    A = O.modulation_matrix(Qs, Rs)
    star = _decay(Qs, Rs, lambda x, y: A @ y[None] - x)
    v = V[:, 0][:, None]
    assert _decay(Qs, Rs, lambda x, y: A @ y[None] - x + 0.3 * v * np.tanh(y)[None]) < star - 1e-6


def test_vc97_fig1_learning_curve_and_hessian():
    """K = 2, M = 1 (VC97 Fig. 1): Q11 is fitted to the inset's top Hessian eigenvalue at alpha = 0 (an input: the
    paper gives only Q11 in [0, 0.5]); everything else is a prediction of the ODE against the figure's single
    N = 5000 simulation, with tolerances that are 99% prediction bands of one N = 5000 run (bank cites): the
    inset's lower eigenvalue at 0, its dip near alpha = 1 (the det H = 0 crossing; weak), its values at 4 and 8, and
    the learning curve at alpha = 2.5 ... 6.  Positive controls: gradient descent's curve misses the curve keys, and
    H read as de/dQ (half the second variation) misses the inset's alpha = 0 key."""
    q11, tr = _fig1()
    assert 0 < q11 < 0.5
    at = lambda a: int(np.argmin(np.abs(tr.alpha - a)))
    assert _ok("vc97_fig1_hess_lo_a0", OF.vc97_initial_eigs(2, 1)[0])
    i = 1 + int(np.argmin(tr.eig_hi[1:32]))
    assert _ok("vc97_fig1_hess_dip", tr.eig_hi[i]) and _ok("vc97_fig1_hess_dip_alpha", tr.alpha[i])
    for a, t in ((4, "a4"), (8, "a8")):
        assert _ok(f"vc97_fig1_hess_hi_{t}", tr.eig_hi[at(a)]) and _ok(f"vc97_fig1_hess_lo_{t}", tr.eig_lo[at(a)])
    keys = (("a2p5", 2.5), ("a3", 3.0), ("a3p5", 3.5), ("a4", 4.0), ("a5", 5.0), ("a6", 6.0))
    for t, a in keys:
        assert _ok(f"vc97_fig1_eps_{t}", tr.eps[at(a)]), (a, tr.eps[at(a)])
    Q0, R0, _ = OF.vc97_initial(2, 1)
    gd = SF.integrate_scm(S.scm_flow(2, 1, 1.5), 6.0, (Q0, R0), alphas=tuple(a for _, a in keys), rtol=1e-7)
    assert not any(_ok(f"vc97_fig1_eps_{t}", e) for (t, _), e in zip(keys, gd.eps))
    with planted(O, hess=0.5):
        lo, hi = np.linalg.eigvalsh(O.hessian(Q0, R0))
    assert not _ok("vc97_fig1_hess_lo_a0", lo)


def test_beats_gradient_descent_in_the_papers_regime():
    """VC97 Fig. 1 (K = 2, M = 1) against O5's gradient descent at the paper's eta = 1.5 from the same overlaps: the
    optimal error is below gradient descent's at every alpha in [2, 8] and has no plateau (below 1e-3 by alpha = 8),
    while gradient descent sits on its permutation-symmetric plateau, whose O5 value is the figure's BP plateau."""
    q11, tr = _fig1()
    Q0, R0, _ = OF.vc97_initial(2, 1)
    al = (2.0, 3.0, 4.0, 6.0, 8.0)
    gd = SF.integrate_scm(S.scm_flow(2, 1, 1.5), 8.0, (Q0, R0), alphas=al, rtol=1e-7)
    opt = [tr.eps[int(np.argmin(np.abs(tr.alpha - a)))] for a in al]
    assert all(o < g for o, g in zip(opt, gd.eps)) and opt[-1] < 1e-3 < 0.015 < gd.eps[-1]
    fp = SF.symmetric_fixed_point(S.scm_flow(2, 1, 1.5), "permutation")
    assert _ok("vc97_fig1_bp_plateau", fp.eps) and not _ok("vc97_fig1_bp_plateau", opt[-1])


def test_k2m2_plateau_fixed_point():
    """K = M = 2: on the identical-student manifold the optimal flow forces Sigma singular (x fixes y1 + y2, so
    W = (1/2) 1 1^T) and its fixed point is a root of a(r) = r at q = 2 r^2 (computed, not typed).  Its error,
    Hessian eigenvalues and Q12 are VC97's plateau (Fig. 2, its inset, Fig. 3's inset); the smallest eigenvalue
    vanishes there (VC97's 'functional robustness').  Gradient descent's plateau (O5, eta = 1.5) is Fig. 2's BP
    plateau and fails the optimal one."""
    p = O.plateau_k2m2()
    assert _ok("vc97_fig2_eps_plateau", p.eps) and _ok("vc97_fig3_q12_plateau", p.q)
    assert _ok("vc97_fig2_hess_hi_plateau", p.eigs[1]) and _ok("vc97_fig2_hess_lo_plateau", p.eigs[0])
    assert abs(p.eigs[0]) < 1e-7 and abs(p.r - 1 / math.sqrt(6)) < 1e-7 and abs(p.q - 1 / 3) < 1e-7
    gd = SF.symmetric_fixed_point(S.scm_flow(2, 2, 1.5), "permutation")
    assert _ok("vc97_fig2_bp_plateau", gd.eps) and not _ok("vc97_fig2_eps_plateau", gd.eps)
    assert gd.eps > 2.5 * p.eps                                   # 'vastly reduced' plateau
    Q, R = 2 * p.r ** 2 * np.ones((2, 2)), p.r * np.ones((2, 2))
    dQ, dR = O.optimal_flow(2, 2).rates(Q, R)
    assert max(np.abs(dQ).max(), np.abs(dR).max()) < 1e-7


def test_singular_posterior_two_roots_in_one_cell():
    """Regression: Sigma = 0.6 u u^T put both roots of the line search in one grid cell (NaN before); the extrema
    are now bracketed first.  W is finite, exchange-symmetric and W - (1 - Sigma) is PSD."""
    u = np.array([1.0, -1.0]) / math.sqrt(2)
    Sig = 0.6 * np.outer(u, u)
    W = O.info_matrix(Sig, 2)
    assert np.all(np.isfinite(W)) and abs(W[0, 0] - W[1, 1]) < 1e-12
    assert np.linalg.eigvalsh(W - (np.eye(2) - Sig)).min() > -1e-14


def test_refusals_and_the_class():
    _refuses(lambda: O.optimal_flow(2, 3), "unsupported:M")
    _refuses(lambda: O.optimal_flow(0, 1), "precondition:K")
    _refuses(lambda: O.modulation_matrix(np.zeros((2, 2)), np.zeros((2, 1))), "no-solution:singular-hessian")
    _refuses(lambda: OF.integrate_optimal(O.optimal_flow(2, 1), -1.0, OF.vc97_initial(2, 1)[:2]), "precondition:alpha")
    _refuses(lambda: OF.vc97_initial(3, 1), "unsupported:init")
    for word in ("every student field", "teacher's output", "order parameters", "greedy", "saddle", "nonlocal",
                 "replicant's term", "no maximiser"):
        assert word in O.OPTIMALITY_CLASS, word


@slow
def test_m2_information_matrix_against_an_independent_route():
    """W = <yhat yhat^T> (M = 2) by optimal_multilayer's route against a dense midpoint rule in v = g(y1) inside a
    4-dimensional Gauss-Hermite outer integral (no code shared): they agree to 3e-3 away from the identical-student
    manifold; symmetry W11 = W22 where Sigma is exchange-symmetric; a planted W is caught; the identical-manifold
    bound w11 + w12 > 2 beta for beta < 1/2 (the teacher output adds information), so the plateau needs Sigma
    singular.  Near that manifold (Sigma = lb u u^T + ls v v^T, u = (1, -1)/sqrt2, v = (1, 1)/sqrt2) what matters is
    the excess information v^T W v - (1 - ls) > 0, which vanishes as ls -> 0: the PSD assembly keeps it positive and
    within 30% of an independent outer grid (sinh-clustered along v), while the direct form sum w yhat yhat^T
    (planted) keeps the quadrature's cross-term residue and turns it negative."""
    from scipy.special import erf, erfinv

    def brute(Sig, nmu=10, nz=16, nv=1500):
        Sig = np.asarray(Sig, float)
        Si = np.linalg.inv(Sig)
        mus, wm = _gauss(np.eye(2) - Sig, nmu)
        zs, wz = _gauss(Sig, nz)
        out = np.zeros((2, 2))
        for mu, a in zip(mus, wm):
            y = mu + zs
            s = np.clip(erf(y[:, 0] / math.sqrt(2)) + erf(y[:, 1] / math.sqrt(2)), -2 + 1e-15, 2 - 1e-15)
            lo, hi = np.maximum(-1, s - 1), np.minimum(1, s + 1)
            v = lo[:, None] + (hi - lo)[:, None] * (np.arange(nv) + 0.5) / nv
            y1, y2 = math.sqrt(2) * erfinv(v), math.sqrt(2) * erfinv(np.clip(s[:, None] - v, -1 + 1e-16, 1 - 1e-16))
            d1, d2 = y1 - mu[0], y2 - mu[1]
            lf = -0.5 * (Si[0, 0] * d1 * d1 + 2 * Si[0, 1] * d1 * d2 + Si[1, 1] * d2 * d2) + 0.5 * (y1 * y1 + y2 * y2)
            f = np.exp(lf - lf.max(1, keepdims=True))
            yh = np.stack([(f * y1).sum(1), (f * y2).sum(1)], -1) / f.sum(1)[:, None]
            out += a * np.einsum("m,ma,mb->ab", wz, yh, yh)
        return out
    for Sig in ([[0.6, -0.3], [-0.3, 0.6]], [[0.3, 0.1], [0.1, 0.2]], [[0.05, 0.01], [0.01, 0.02]]):
        w = O.info_matrix(np.array(Sig, float), 2)
        assert np.abs(w - brute(Sig)).max() < 5e-3, (Sig, w)
        with planted(O, w=1.03):
            assert np.abs(O.info_matrix(np.array(Sig, float), 2) - brute(Sig)).max() > 1e-2
    for beta in (0.1, 0.3, 0.45, 0.49):
        w = O.info_matrix(np.eye(2) - beta * np.ones((2, 2)), 2)
        assert abs(w[0, 0] - w[1, 1]) < 1e-6 and w[0, 0] + w[0, 1] > 2 * beta + 1e-4, (beta, w)
    u, v = np.array([1.0, -1.0]) / math.sqrt(2), np.array([1.0, 1.0]) / math.sqrt(2)
    for ls in (0.02, 0.005, 0.002):
        Sig = 0.95 * np.outer(u, u) + ls * np.outer(v, v)
        a, b = (v @ w @ v - (1 - ls) for w in (O.info_matrix(Sig, 2), _w_clustered(Sig)))
        assert a > 0 and b > 0 and abs(a - b) < 0.3 * b, (ls, a, b)
    with planted(O, direct=1.0):
        assert v @ O.info_matrix(0.95 * np.outer(u, u) + 0.002 * np.outer(v, v), 2) @ v - 0.998 < 0


def _w_clustered(Sig, na=12, nb=41, nz=12):
    """An independent outer grid for W: Gauss-Hermite along Sigma's large eigenvector, a sinh-clustered trapezoid
    (scale sqrt(lam_small)) along the small one, Gauss-Hermite over z; the PSD assembly."""
    lam, U = np.linalg.eigh(Sig)
    es, eb = U[:, 0], U[:, 1]
    xa, wa = np.polynomial.hermite_e.hermegauss(na)
    a = xa * math.sqrt(max(1 - lam[1], 0.0))
    c, sdb = math.sqrt(lam[0]), math.sqrt(1 - lam[0])
    W = math.asinh(9 * sdb / c)
    w = np.linspace(-W, W, nb)
    b = c * np.sinh(w)
    jb = c * np.cosh(w) * (w[1] - w[0]) * np.exp(-b * b / (2 * sdb ** 2)) / (math.sqrt(2 * math.pi) * sdb)
    jb[0] *= 0.5
    jb[-1] *= 0.5
    mu = (a[:, None, None] * eb + b[None, :, None] * es).reshape(-1, 2)
    wm = (wa[:, None] / math.sqrt(2 * math.pi) * jb[None, :]).ravel()
    z, wz = _gauss(Sig, nz)
    y = mu[:, None, :] + z[None]
    M = np.broadcast_to(mu[:, None, :], y.shape)
    d = O.yhat2(M, O.g(y[..., 0]) + O.g(y[..., 1]), Sig) - M
    d = np.where(np.all(np.isfinite(d), -1)[..., None], d, 0.0)
    return np.eye(2) - Sig + np.einsum("m,z,mza,mzb->ab", wm, wz, d, d)


@slow
def test_k2m2_ode_through_the_spike():
    """VC97 Figs. 2, 3 (K = M = 2): from the paper's initial overlaps the ODE passes the det H = 0 crossing near
    alpha = 0.5 (the lower Hessian eigenvalue changes sign, the norms jump and Q12 turns negative: Fig. 3's
    'anti-correlated weights'; the jump is ~4 in the ODE, ~13 in the paper's N = 5000 run) and is below gradient
    descent (O5, eta = 1.5) at alpha = 3; the catalog reports a curve whose min eig Sigma drops below
    SIGMA_RELIABLE as undetermined (the threshold raised here so that a short run reaches it)."""
    flow = O.optimal_flow(2, 2)
    Q0, R0, _ = OF.vc97_initial(2, 2)
    tr = OF.integrate_optimal(flow, 3.0, (Q0, R0), alphas=(0.3, 0.6, 1.0, 3.0), rtol=1e-4, atol=1e-6)
    assert tr.eig_lo[0] < 0 < tr.eig_lo[1] and tr.Q[1][0, 1] < -2.0 and tr.Q[1][0, 0] > 2.0
    gd = SF.integrate_scm(S.scm_flow(2, 2, 1.5), 3.0, (Q0, R0), alphas=(3.0,), rtol=1e-7)
    assert tr.eps[-1] < gd.eps[-1]
    from replicant.catalog import catalog
    saved = OF.SIGMA_RELIABLE
    try:
        OF.SIGMA_RELIABLE = 0.5
        v = catalog()["online_scm"].run(route="online", rule="optimal", K="2", M="2", alpha="1.5")
    finally:
        OF.SIGMA_RELIABLE = saved
    assert v["curve_status"] == "undetermined" and "eps" not in v


@slow
def test_k2m2_curve_reaches_the_plateau():
    """The integrated K = M = 2 curve approaches plateau_k2m2 (e* = 0.011611): within 6e-4 at alpha = 16, and inside
    the Fig. 2 keys at alpha = 10, 12, 16 (99% one-run bands from replicant's N = 5000 runs).  Planted control: the
    direct assembly of W (whose cross-term residue fakes a fixed point) stalls above e* + 1e-3."""
    flow = O.optimal_flow(2, 2)
    Q0, R0, _ = OF.vc97_initial(2, 2)
    e_star = O.plateau_k2m2().eps
    al = (10.0, 12.0, 16.0)
    tr = OF.integrate_optimal(flow, 16.0, (Q0, R0), alphas=al, rtol=1e-4, atol=1e-6)
    assert 0 < tr.eps[-1] - e_star < 6e-4 and tr.eps[0] > tr.eps[1] > tr.eps[2]
    for t, e in zip(("a10", "a12", "a16"), tr.eps):
        assert _ok(f"vc97_fig2_eps_{t}", e), (t, e)
    assert tr.sig_min[-1] > OF.SIGMA_RELIABLE
    with planted(O, direct=1.0):
        bad = OF.integrate_optimal(flow, 16.0, (Q0, R0), alphas=(16.0,), rtol=1e-4, atol=1e-6)
    assert bad.eps[0] - e_star > 1e-3, bad.eps
