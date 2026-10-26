"""O3: online learning of drifting rules (papers 2, 8, 14 of the online spec).

Kinouchi & Caticha, J. Phys. A 26, 6161 (1993) (KC93; read from the preprint IFUSP/P-1044, March
1993); Vicente, Kinouchi & Caticha, Machine Learning 32, 179 (1998), arXiv:cond-mat/9801297 (VKC).
Biehl & Schwarze (EPL 20, 733 (1992); J. Phys. A 26, 2651 (1993)) were NOT fetched (IOP bot check,
no preprint found): nothing is banked from them.

Convention (online/drift.py): the O1 engine (|B| = 1, xi ~ N(0, I), J <- J + (eta/N) F b xi) with a
teacher that moves after every example: random drift B <- (B + sqrt(2D)/N u) / |.|, u ~ N(0, I)
(VKC Eqs. (4), (26); KC93 Eqs. (3)-(6) with T = D), so B(alpha) . B(alpha') = e^{-D |alpha - alpha'|};
deterministic drift: B turned away from J by the angle arccos(1 - D/N^2) per example (VKC Eqs.
(28)-(30)).  noise = the flip probability = VKC's chi / 2 (O1, online/optimal.py).
Bank: reference/online_drift.json.
"""
import math
import os
import sys
from contextlib import contextmanager

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.core import Refusal
from replicant.online import drift as DR
from replicant.online import drift_rules as VR
from replicant.online import fields, odes
from replicant.refdata import check, load
from replicant.testing import slow

BANK = load("online_drift")
VKC_NAMES = ("annealed_hebb", "step", "symmetric")


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


def _eps(flow, alpha, init):
    return float(DR.integrate_drift(flow, alpha, init, alphas=(alpha,)).eps[0])


# ---- the VKC modulations and their averages -----------------------------------------------------
def test_vkc_averages_closed_vs_quadrature():
    """Annealed Hebb / step (combinations of O1's closed Hebb and perceptron moments) and symmetric
    (the 1D conditional route) against the 2D polar quadrature of fields.py; the conditional route
    also reproduces O1's closed forms for its named rules.  Planted: drift_rules._PLANT["norm"]."""
    for theta in (0.3, 0.8, 1.2):
        for lam in (0.0, 0.15):
            for rule in VKC_NAMES:
                c = VR.vkc_averages(rule, theta, lam)
                q = VR.vkc_averages(rule, theta, lam, route="quadrature")
                assert max(abs(a - b) for a, b in zip(c, q)) < 1e-10, (rule, theta, lam, c, q)
            for rule in ("hebb", "perceptron", "adatron", "optimal"):
                g = odes.RULES[rule].g
                c = VR.conditional_averages(lambda d: g(d, theta, lam), theta, lam)
                want = fields.closed(rule, theta, lam)
                assert max(abs(a - b) for a, b in zip(c, want)) < 1e-10, (rule, theta, lam, c, want)
    g = lambda d: VR.vkc_modulation("symmetric", d, 0.8, 0.1)
    for fn in (lambda: VR.vkc_averages("symmetric", 0.8, 0.1), lambda: VR.conditional_averages(g, 0.8, 0.1)):
        ref = fn()
        with planted(VR, norm=math.sqrt(math.pi)):
            bad = fn()
        assert abs(bad.drive - ref.drive) > 1e-3 and fn() == ref


def _directional(flow, theta, basis, h=1e-3):
    """d(drho/dalpha)/d epsilon along each basis function at the flow's own modulation (central
    differences of the flow's rate; drho/dalpha is quadratic in W, so they are exact up to rounding)."""
    base = flow.modulation
    out = []
    for psi in basis:
        up = DR.drift_flow(VR.Modulation(lambda d, th, p=psi: base(d, th) + h * p(d)), D=flow.D,
                           kind=flow.kind, noise=flow.noise, averages="quadrature")
        dn = DR.drift_flow(VR.Modulation(lambda d, th, p=psi: base(d, th) - h * p(d)), D=flow.D,
                           kind=flow.kind, noise=flow.noise, averages="quadrature")
        out.append((up.rho_rate(0.0, theta) - dn.rho_rate(0.0, theta)) / (2 * h))
    return np.array(out)


_CLASS_BASIS = {      # smooth functions inside each modulation's information class (VKC Sec. II.C)
    "annealed_hebb": [lambda d: np.ones_like(d)],
    "step": [lambda d: (d > 0).astype(float), lambda d: (d < 0).astype(float)],
    "symmetric": [lambda d, k=k: np.exp(-d * d / (2 * k)) for k in (0.3, 1.0, 3.0)],
    "optimal": [lambda d, k=k: np.exp(-d * d / (2 * k)) * (1 + np.tanh(d)) for k in (0.3, 3.0)]
               + [lambda d: np.tanh(2 * d)],
}


def _vkc_flow(rule, D=0.0, kind="random", noise=0.0, typed_noise=None):
    t = noise if typed_noise is None else typed_noise
    w = (lambda d, th: VR.vkc_modulation(rule, d, th, t)) if rule != "optimal" else \
        (lambda d, th: VR.optimal_w(d, th, t))
    return DR.drift_flow(VR.Modulation(w), D=D, kind=kind, noise=noise, averages="quadrature")


@slow
def test_vkc_modulations_maximise_drho_in_their_class():
    """Each typed VKC modulation (Eqs. (22)-(25), chi = 2 noise) is a stationary point of drho/dalpha
    (the flow's rate, quadrature averages) along every direction of its information class; typing
    chi = noise instead (the unmapped convention) is not."""
    for rule, basis in _CLASS_BASIS.items():
        for theta, lam in ((0.5, 0.0), (0.9, 0.1)):
            g = _directional(_vkc_flow(rule, noise=lam), theta, basis)
            assert np.max(np.abs(g)) < 1e-9, (rule, theta, lam, g)
            if lam:
                bad = _directional(_vkc_flow(rule, noise=lam, typed_noise=lam / 2), theta, basis)
                assert np.max(np.abs(bad)) > 1e-4, (rule, bad)


def test_vkc_claim_modulations_do_not_depend_on_drift():
    """VKC Sec. III, p. 9: 'under the assumption of uncorrelated drift and input vectors, the
    modulation functions do not depend on the drift parameters (in contrast with the explicit
    dependence on the examples noise level chi)'.  Tested as stated: the maximiser of drho/dalpha
    within each class, found from the flow's rate (gradient and Hessian by central differences in a
    class basis), is the same at D = 0, D = 0.3 (random) and D = 0.01 (deterministic), and
    it changes with the noise level.  Planted: a W-dependent drift term (drift._PLANT["coupling"]).
    This check is close to structural: drift.py composes the drift term independently of W, so the
    test confirms the composition and the class maximisation rather than the physics; the
    independent evidence that the drift enters as composed is the finite-N band
    (test_online_drift_sim.py), where the teacher drifts in the simulation.  The typed W* is checked
    to be stationary in its class at every D here (default tier); all four VKC rules and the chi
    mapping in test_vkc_modulations_maximise_drho_in_their_class (slow)."""
    def argmax(flow, basis, theta):
        n = len(basis)
        g0 = _directional(DR.drift_flow(VR.Modulation(lambda d, th: 0 * d), D=flow.D, kind=flow.kind,
                                        noise=flow.noise, averages="quadrature"), theta, basis)
        hess = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                f = DR.drift_flow(VR.Modulation(lambda d, th, i=i: basis[i](d)), D=flow.D,
                                  kind=flow.kind, noise=flow.noise, averages="quadrature")
                hess[i, j] = _directional(f, theta, [basis[j]])[0] - g0[j]
        return np.linalg.solve(hess, -g0)
    theta = 0.7
    for rule in ("step", "optimal"):
        basis = _CLASS_BASIS[rule]
        w0 = argmax(_vkc_flow(rule), basis, theta)
        for D, kind in ((0.3, "random"), (0.01, "deterministic")):
            w = argmax(_vkc_flow(rule, D=D, kind=kind), basis, theta)
            assert np.max(np.abs(w - w0)) < 1e-8, (rule, D, kind, w, w0)
            assert np.max(np.abs(_directional(_vkc_flow(rule, D=D, kind=kind), theta, basis))) < 1e-9
        wn = argmax(_vkc_flow(rule, noise=0.1), basis, theta)
        assert np.max(np.abs(wn - w0)) > 1e-2                              # the noise contrast
        with planted(DR, coupling=1.0):
            wp = argmax(_vkc_flow(rule, D=0.3), basis, theta)
        assert np.max(np.abs(wp - w0)) > 1e-2                              # planted control


# ---- the drift composed onto the O1 flow -------------------------------------------------------
def test_drift_is_composed_onto_the_o1_flow():
    """D = 0 reproduces odes.Flow exactly; random drift adds -D rho to drho/dalpha (VKC Eq. (27),
    KC93 Eq. (14)), deterministic drift -sqrt(2D (1 - rho^2)) (VKC Eq. (31)); d ln Q/dalpha unchanged."""
    for rule, eta in (("hebb", 0.7), ("perceptron", 1.3), ("adatron", 1.0), ("optimal", 1.0)):
        base = odes.perceptron_flow(rule, eta=eta, noise=0.1)
        for theta, lnq in ((0.4, 0.2), (1.1, -0.5)):
            b = base._theta_lnq(0.3, theta, lnq)
            assert DR.drift_flow(rule, eta=eta, noise=0.1)._theta_lnq(0.3, theta, lnq) == b
            r = DR.drift_flow(rule, D=0.2, eta=eta, noise=0.1)._theta_lnq(0.3, theta, lnq)
            d = DR.drift_flow(rule, D=0.02, kind="deterministic", eta=eta, noise=0.1)._theta_lnq(0.3, theta, lnq)
            s, rho = math.sin(theta), math.cos(theta)
            assert abs((r[0] - b[0]) * s - 0.2 * rho) < 1e-13 and r[1] == b[1]
            assert abs((d[0] - b[0]) * s - math.sqrt(0.04) * s) < 1e-13 and d[1] == b[1]


def test_hebb_with_random_drift_closed_form_and_no_tracking():
    """Plain Hebb (W J = 1): dR/dalpha = eta a - D R, dQ/dalpha = 2 eta a R + eta^2 (derived), so R
    saturates at eta a / D while Q grows: rho -> 0, eps -> 1/2 (VKC p. 17: Hebb 'has no internal
    forgetting mechanism').  The ODE matches the closed form; with the drift planted away it does not;
    stationary() refuses no-solution:no-tracking."""
    f = DR.drift_flow("hebb", D=0.3, eta=0.8, noise=0.1)
    tr = DR.integrate_drift(f, 6.0, (0.2, 1.5), alphas=(2.0, 6.0))
    for k, a in enumerate((2.0, 6.0)):
        R, Q, eps = DR.hebb_drift_closed_form(a, 0.3, eta=0.8, noise=0.1, init=(0.2, 1.5))
        assert abs(tr.R[k] - R) < 1e-8 and abs(tr.Q[k] - Q) < 1e-7 and abs(tr.eps[k] - eps) < 1e-9
    with planted(DR, drift=0.0):
        bad = DR.integrate_drift(f, 6.0, (0.2, 1.5), alphas=(6.0,)).eps[0]
    assert abs(bad - DR.hebb_drift_closed_form(6.0, 0.3, 0.8, 0.1, (0.2, 1.5))[2]) > 1e-2
    far = DR.hebb_drift_closed_form(1e6, 0.3, 0.8, 0.1, (0.2, 1.5))[2]
    assert 0.5 - far < 2e-3
    _refuses(lambda: DR.stationary(f), "no-solution:no-tracking")


# ---- stationary tracking error ----------------------------------------------------------------
def test_drift_stationary_error():
    """eps_inf(D) for the annealed Hebb and the optimal rule: the ODE's stationary state (bisection on
    dtheta/dalpha) against independent evaluations: annealed Hebb in closed form (derived from
    VKC Eqs. (22), (27), (31)); optimal noiseless from KC93's I(lambda) = T (Eqs. (17)-(23), Gauss-Legendre in u),
    with the deterministic analogue I(lambda) = sqrt(2D) lambda.  The integrated ODE reaches it from
    both sides (equilibration diagnostic)."""
    for kind, D in (("random", 0.1), ("random", 0.02), ("deterministic", 0.01)):
        for lam in (0.0, 0.1):
            s = DR.stationary(DR.drift_flow("annealed_hebb", D=D, kind=kind, noise=lam))
            assert abs(s.eps - VR.annealed_hebb_stationary(D, kind, lam)) < 1e-10, (kind, D, lam, s)
        o = DR.stationary(DR.drift_flow("optimal", D=D, kind=kind))
        assert abs(o.eps - VR.kc93_stationary(D, kind)) < 1e-9, (kind, D, o)
    f = DR.drift_flow("optimal", D=0.1)
    e_inf = DR.stationary(f).eps
    for init in ((0.05, 1.0), (0.995, 1.0)):
        assert abs(_eps(f, 30.0, init) - e_inf) < 1e-7, init
    p = DR.drift_flow("perceptron", D=0.1)                 # Q free: (theta, Q) reach (theta*, Q*)
    sp_ = DR.stationary(p)
    tr = DR.integrate_drift(p, 150.0, (0.0, 1.0), alphas=(150.0,))
    assert abs(tr.eps[0] - sp_.eps) < 1e-6 and abs(tr.Q[0] - sp_.Q) < 1e-4, (tr, sp_)
    with planted(DR, drift=0.5):
        assert abs(DR.stationary(f).eps - e_inf) > 1e-2


def test_kc93_printed_eq22_is_a_typo():
    """KC93 Eq. (22) as printed has e^{-(1 - lambda^2) u^2 / 2}; the average of W_ES^2 / 2 over the
    fields gives (1 + lambda^2) (derived in drift_rules.py): the corrected form equals the engine's
    <W*^2>/2 (fields.closed) to 1e-10; the printed one does not (it diverges for lambda >= 1).
    Both agree at lambda -> 0 (so KC93's small-T constant, Eq. (25), is unaffected)."""
    for theta in (0.2, 0.9):
        lam = math.tan(theta)
        engine = fields.closed("optimal", theta).a2 / 2
        assert abs(VR.kc93_I(lam) - engine) < 1e-10 * max(1.0, engine), (theta, engine)
        assert abs(VR.kc93_I(lam, printed=True) - engine) > 1e-3 * engine
    assert abs(VR.kc93_I(0.01, printed=True) / VR.kc93_I(0.01) - 1) < 1e-3
    with mp.workdps(20):
        c = (mp.pi ** 2 / 2 * mp.quad(lambda u: mp.npdf(u) * mp.exp(-u * u / 2) / (mp.erfc(u / mp.sqrt(2)) / 2),
                                      [-mp.inf, 0, mp.inf])) ** (-mp.mpf(1) / 3)
    assert _ok("random_optimal_coeff", float(c))                          # KC93 Eq. (25) evaluated


def test_kc93_lower_bound_holds_for_every_rule():
    """KC93 Sec. 3: the expected-stability (optimal) rule maximises drho/dalpha pointwise in rho and
    the drift term does not involve W, so its eps(alpha) lower-bounds every modulated Hebbian rule
    from the same rho0 (ODE comparison), and its eps_inf every rule's eps_inf.  Checked with
    tolerance 1e-7 (integrator rtol 1e-9) for each rule the engine runs; planted violation: the
    drift halved for the rule under test (drift._PLANT["drift"]) beats the bound."""
    init, alphas = (0.2, 1.0), (1.0, 4.0)
    rules = (("hebb", 1.0), ("hebb", 0.4), ("perceptron", 1.0), ("perceptron", 3.0), ("adatron", 1.0),
             ("adatron", 0.5), ("annealed_hebb", 1.0), ("step", 1.0), ("symmetric", 1.0))
    for kind, D in (("random", 0.05), ("deterministic", 0.005)):
        opt = DR.integrate_drift(DR.drift_flow("optimal", D=D, kind=kind), 4.0, init, alphas=alphas).eps
        e_opt = DR.stationary(DR.drift_flow("optimal", D=D, kind=kind)).eps
        mis = DR.integrate_drift(DR.drift_flow("optimal", D=D, kind=kind, noise=0.0, noise_estimate=0.1),
                                 4.0, init, alphas=alphas).eps
        assert np.all(mis >= opt - 1e-7)
        for rule, eta in rules:
            f = DR.drift_flow(rule, D=D, kind=kind, eta=eta)
            e = DR.integrate_drift(f, 4.0, init, alphas=alphas).eps
            assert np.all(e >= opt - 1e-7), (rule, eta, kind, e, opt)
            if rule != "hebb":
                assert DR.stationary(f).eps >= e_opt - 1e-10, (rule, eta, kind)
        with planted(DR, drift=0.5):
            e = DR.integrate_drift(DR.drift_flow("step", D=D, kind=kind), 4.0, init, alphas=alphas).eps
        assert e[-1] < opt[-1] - 1e-3                                       # the plant violates it


@slow
def test_kc93_lower_bound_with_noise_and_more_rules():
    init, alphas = (0.3, 1.0), (0.5, 2.0, 8.0)
    for lam in (0.05, 0.2):
        for kind, D in (("random", 0.1), ("deterministic", 0.02)):
            opt = DR.integrate_drift(DR.drift_flow("optimal", D=D, kind=kind, noise=lam), 8.0, init, alphas=alphas).eps
            for rule, eta, est in (("hebb", 1.0, None), ("perceptron", 0.5, None), ("perceptron", 2.0, None),
                                   ("adatron", 1.0, None), ("adatron", 2.0, None), ("optimal", 1.0, 0.0),
                                   ("optimal", 1.0, lam / 2), ("annealed_hebb", 1.0, None), ("step", 1.0, None),
                                   ("symmetric", 1.0, None)):
                f = DR.drift_flow(rule, D=D, kind=kind, eta=eta, noise=lam, noise_estimate=est)
                e = DR.integrate_drift(f, 8.0, init, alphas=alphas).eps
                assert np.all(e >= opt - 1e-7), (lam, kind, rule, eta, est, e, opt)


# ---- banked small-drift laws (VKC Tables I-II, KC93 Eqs. (24)-(25), VKC p. 15) -------------------
_TABLE = (("annealed_hebb", "random"), ("symmetric", "random"), ("step", "random"), ("optimal", "random"),
          ("annealed_hebb", "deterministic"), ("symmetric", "deterministic"), ("step", "deterministic"),
          ("optimal", "deterministic"))


def test_small_drift_laws_vs_banked():
    """eps_inf ~ c D^(1/delta) for small D: the measured log-slope (D = 1e-8 -> 1e-7) and the
    coefficient against VKC Tables I-II (and KC93 Eq. (25) for the optimal rule, random drift); the
    exponents obey VKC Eq. (34), delta = 1/beta + 2 (random), 2/beta + 2 (deterministic), for these
    decoupled rules (W independent of |J|).  The perceptron at a fixed rate is the exception: its |J|
    self-tunes under drift and delta = 3, 4 (drift.drift_exponent), not VKC p. 15's 5, 8 (banked,
    UNCLAIMED), with the step rule's coefficients (4D)^(1/3)/pi, 2^(5/4)/pi D^(1/4).  Controls: each
    coefficient against the next rule's key; the perceptron against VKC's exponents."""
    got = {}
    for rule, kind in _TABLE:
        sd = DR.small_drift(rule, kind)
        tag = "random" if kind == "random" else "det"
        assert _ok(f"{tag}_{rule}_exponent", sd.exponent), (rule, kind, sd)
        assert _ok(f"{tag}_{rule}_coeff", sd.coefficient), (rule, kind, sd)
        got[(rule, kind)] = sd
    for (r1, k1), (r2, k2) in zip(_TABLE, _TABLE[1:] + _TABLE[:1]):
        tag = "random" if k2 == "random" else "det"
        assert not _ok(f"{tag}_{r2}_coeff", got[(r1, k1)].coefficient), (r1, r2)
    for kind, tag in (("random", "random"), ("deterministic", "det")):
        # VKC p. 15 claim delta = 5, 8 for the perceptron at a fixed rate (banked, UNCLAIMED): the
        # ODE, with |J| at its stationary value, gives the derived 1/3, 1/4 and the step rule's
        # coefficients (drift.drift_exponent)
        sd = DR.small_drift("perceptron", kind)
        assert not _ok(f"{tag}_perceptron_exponent", sd.exponent), (kind, sd)
        assert abs(sd.exponent - (1 / 3 if kind == "random" else 0.25)) < 1e-3
        assert _ok(f"{tag}_step_coeff", sd.coefficient), (kind, sd)
        assert _ok(f"{tag}_step_coeff", DR.small_drift("perceptron", kind, eta=3.0).coefficient)


def test_vkc_figure_curves_vs_banked():
    """e_G^oo(D) of VKC Figs. 4 (random) and 6 (deterministic), read off the figures, against the
    stationary states of the four VKC rules; control: the optimal reading against annealed Hebb."""
    for kind, tag, Ds in (("random", "fig4", ("0.1", "0.45")), ("deterministic", "fig6", ("0.035", "0.045"))):
        for D in Ds:
            for rule in ("annealed_hebb", "symmetric", "step", "optimal"):
                e = DR.stationary(DR.drift_flow(rule, D=float(D), kind=kind)).eps
                assert _ok(f"{tag}_{rule}_d{D.replace('.', 'p')}", e), (kind, D, rule, e)
            e = DR.stationary(DR.drift_flow("annealed_hebb", D=float(D), kind=kind)).eps
            assert not _ok(f"{tag}_optimal_d{D.replace('.', 'p')}", e)


# ---- abrupt changes and schedules --------------------------------------------------------------
def test_abrupt_change_and_drift_schedule():
    """A jump to an uncorrelated teacher resets R to jump_overlap sqrt(Q) (VKC Sec. V, KC93 Fig. 7);
    after it the annealed rules (W ~ 1/rho) relearn as from that overlap, while plain Hebb (|J| grown
    by the first rule) relearns more slowly than from tabula rasa (VKC p. 18, 'perseveres').  A drift
    schedule D(alpha) (VKC Fig. 7) equals the constant drift where it is constant."""
    f = DR.drift_flow("step")
    tr = DR.integrate_drift(f, 15.0, (0.1, 1.0), alphas=(15.0,), jumps=(10.0,), jump_overlap=0.1)
    fresh = DR.integrate_drift(f, 5.0, (0.1, 1.0), alphas=(5.0,))
    assert abs(tr.eps[0] - fresh.eps[0]) < 1e-8
    h = DR.drift_flow("hebb")
    after = DR.integrate_drift(h, 15.0, (0.0, 1.0), alphas=(15.0,), jumps=(10.0,), jump_overlap=0.0).eps[0]
    assert after > DR.integrate_drift(h, 5.0, (0.0, 1.0), alphas=(5.0,)).eps[0] + 0.05
    sched = DR.drift_flow("optimal", D=lambda a: 0.1 if a < 3 else 0.0)
    a = _eps(sched, 3.0, (0.2, 1.0))
    assert abs(a - _eps(DR.drift_flow("optimal", D=0.1), 3.0, (0.2, 1.0))) < 1e-8
    _refuses(lambda: DR.integrate_drift(f, 15.0, (0.1, 1.0), jumps=(10.0,)), "precondition:initial-overlap")
    _refuses(lambda: DR.stationary(sched), "precondition:drift")


# ---- refusals and the catalog entry ------------------------------------------------------------
def test_refusals():
    _refuses(lambda: DR.drift_flow("oja"), "unsupported:rule")
    _refuses(lambda: DR.drift_flow("hebb", D=-0.1), "precondition:drift")
    _refuses(lambda: DR.drift_flow("hebb", D=0.1, kind="sudden"), "precondition:drift-kind")
    _refuses(lambda: DR.drift_flow("step", eta=0.5), "precondition:optimal-eta")
    _refuses(lambda: DR.drift_flow("step", noise_estimate=0.1), "precondition:known-parameter")
    _refuses(lambda: DR.drift_flow("hebb", noise=0.6), "precondition:noise")
    _refuses(lambda: DR.integrate_drift(DR.drift_flow("symmetric"), 1.0, (0.0, 1.0)),
             "precondition:initial-overlap")
    _refuses(lambda: DR.stationary(DR.drift_flow("optimal")), "precondition:drift")
    _refuses(lambda: DR.stationary(DR.drift_flow(lambda h, b, Q: b * 0 + 1.0, D=0.1)),
             "unsupported:stationary")
    _refuses(lambda: DR.small_drift("hebb", "random"), "no-solution:no-tracking")
    _refuses(lambda: VR.vkc_averages("step", 0.5, 0.0, route="mc"), "precondition:averages")
