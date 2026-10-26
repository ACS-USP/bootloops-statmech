"""O1: the online-learning engine and the online perceptron (papers 1, 11, 6 of the online spec).

Convention (online/odes.py; Biehl & Riegler, Europhys. Lett. 28, 525 (1994), Eqs. (1)-(6)):
teacher B with |B| = 1, inputs xi with i.i.d. N(0, 1) components, observed label b = sign(B . xi)
flipped with probability `noise` (Copelli et al., Europhys. Lett. 37, 427 (1997), Eq. (7)),
update J <- J + (eta / N) F(h, b, Q) b xi with h = J . xi, order parameters R = J . B and
Q = J . J (BR's Q is |J|; here Q = |J|^2), alpha = examples / N, eps = arccos(R / sqrt Q) / pi
(the probability of disagreeing with the TRUE teacher).
Bank: reference/online_perceptron.json.
"""
import math
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.catalog import RefCheck, catalog, check_ref
from replicant.compose import compose
from replicant.core import Refusal
from replicant.online import fields as F
from replicant.online import odes as O
from replicant.online import optimal as OPT
from replicant.refdata import check, load
from replicant.testing import slow

BANK = load("online_perceptron")
RULES = ("hebb", "perceptron", "adatron", "optimal")


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


def _g(rule, theta, noise_est=0.0):
    """The modulation per unit scale g(b x) of a named rule as a plain callable (x, b) -> f, for
    the quadrature route (odes.RULES holds the same functions; the closed forms never call them)."""
    return lambda x, b: O.RULES[rule].g(b * x, theta, noise_est)


# ---- Gaussian averages: closed forms vs an independent quadrature ---------------------------
def test_fields_closed_vs_quadrature():
    """fields.closed (quadrant moments, a 1D panel integral for the optimal rule) against
    fields.quadrature (polar Gauss-Legendre over (x, w), the label summed explicitly) at three
    overlaps and two noise levels; the optimal rule also with a mis-estimated noise.  The
    quadrature runs at 96 x 64 nodes per sector: at its default 48 x 32 the optimal rule at
    rho = 0.95 with a mis-estimated noise is off by 2e-6 relative (measured; 96 x 64 and
    160 x 128 agree with closed() to 1e-13).  Measured largest gap: 4e-14 absolute."""
    for rule in RULES:
        for rho in (0.3, 0.7, 0.95):
            th = math.acos(rho)
            for noise in (0.0, 0.15):
                ests = (noise, 0.05) if rule == "optimal" and noise else (noise,)
                for est in ests:
                    c = F.closed(rule, th, noise, est)
                    q = F.quadrature(_g(rule, th, est), th, noise, n_r=96, n_phi=64)
                    for a, b in zip(c, q):
                        assert abs(a - b) <= 1e-12, (rule, rho, noise, est, c, q)


def test_fields_quadrature_against_mpmath_and_planted_normalisation():
    """A third rule (mpmath 1D quadrature of the quadrant moments, x | y ~ N(rho y, 1 - rho^2))
    at one point, and the positive
    control: a closed form typed with sqrt(pi) for sqrt(2 pi) (fields._PLANT['norm']) is caught."""
    th, noise = math.acos(0.6), 0.1
    rho, s = math.cos(th), math.sin(th)
    with mp.workdps(20):
        def quadrant(sx, sy):          # E[y; sign x = sx, sign y = sy]: x | y ~ N(rho y, s^2)
            px = lambda y: mp.ncdf(sx * rho * y / s)
            f = lambda y: y * mp.npdf(y) * px(y)
            return mp.quad(f, [0, mp.inf] if sy > 0 else [-mp.inf, 0])
        # perceptron <Theta(-b x) b y>: b = sign y w.p. 1 - noise
        want = ((1 - noise) * (quadrant(-1, 1) - quadrant(1, -1))
                - noise * (quadrant(1, 1) - quadrant(-1, -1)))
    assert abs(F.closed("perceptron", th, noise).ay - float(want)) < 1e-12
    saved = dict(F._PLANT)
    try:
        F._PLANT["norm"] = math.sqrt(math.pi)
        bad = F.closed("perceptron", th, noise)
        assert abs(bad.ay - F.quadrature(_g("perceptron", th), th, noise).ay) > 1e-2, bad
    finally:
        F._PLANT.clear()
        F._PLANT.update(saved)
    assert F._PLANT == saved


# ---- Hebb: the ODE against its closed form --------------------------------------------------
def test_hebb_closed_form():
    """The integrated ODE equals the closed form R = R0 + eta a alpha, Q = Q0 + eta^2 alpha +
    2 eta a R0 alpha + eta^2 a^2 alpha^2, a = (1 - 2 noise) sqrt(2/pi), to 1e-8; its large-alpha
    limit sqrt(alpha) eps -> 1/((1 - 2 noise) sqrt(2 pi)) is BR Eq. (7) at noise 0 (banked)."""
    for eta, noise, init in ((1.0, 0.0, (0.0, 1.0)), (0.5, 0.2, (0.3, 2.0))):
        tr = O.integrate(O.perceptron_flow("hebb", eta=eta, noise=noise), 50.0, init,
                         alphas=(1.0, 7.0, 50.0))
        for a, R, Q, e in zip(tr.alpha, tr.R, tr.Q, tr.eps):
            cR, cQ, ce = O.hebb_closed_form(a, eta=eta, noise=noise, init=init)
            assert abs(R - cR) < 1e-8 * max(1, cR) and abs(Q - cQ) < 1e-8 * cQ, (a, R, cR, Q, cQ)
            assert abs(e - ce) < 1e-8, (a, e, ce)
    # the documented rates (R, Q form) are the derivative of the closed form: dR = eta a,
    # dQ = 2 eta a R + eta^2 (Hebb: <b h> = a R)
    f = O.perceptron_flow("hebb", eta=0.5, noise=0.2)
    a = 0.6 * math.sqrt(2 / math.pi)
    dR, dQ = f.rates(3.0, 1.2, 4.0)
    assert abs(dR - 0.5 * a) < 1e-14 and abs(dQ - (2 * 0.5 * a * 1.2 + 0.25)) < 1e-14, (dR, dQ)
    big = 1e12
    _, _, e = O.hebb_closed_form(big)
    assert check(BANK["hebb_eps_constant"], e * math.sqrt(big), label="hebb").ok
    # positive control: a closed form with the eta^2 alpha (diffusion) term dropped is far off
    tr = O.integrate(O.perceptron_flow("hebb"), 7.0, (0.0, 1.0))
    R = math.sqrt(2 / math.pi) * 7.0
    assert abs(math.acos(R / math.sqrt(1 + 2 / math.pi * 49)) / math.pi - tr.eps[-1]) > 0.05


# ---- learning curves vs the banked asymptotes -----------------------------------------------
def test_rules_learning_curves_vs_banked():
    """Exponents and constants of eps ~ C alpha^-p for Hebb, perceptron, AdaTron (Biehl-Riegler
    Eqs. (7), (8), (15)) and the optimal rule (0.88 / alpha: Vicente-Kinouchi-Caticha 1998
    Tables I-II, BR p. 4 quoting Kinouchi-Caticha 1992), through the catalog entry."""
    e = catalog()["online_perceptron"]
    keys = {r.key for r in e.refs}
    for rule in RULES:
        assert {f"{rule}_eps_exponent", f"{rule}_eps_constant"} <= keys, rule
    for ref in e.refs:
        if ref.key.endswith(("_exponent", "_constant")):
            out = check_ref(e, ref)
            assert out.ok, (ref.key, out)
    # positive control: the perceptron's constant is not AdaTron's, nor its exponent Hebb's
    assert not check_ref(e, RefCheck("online_perceptron", "adatron_eps_constant",
                                     lambda v: v["eps_constant"], params={"rule": "perceptron"},
                                     route="online")).ok
    assert not check_ref(e, RefCheck("online_perceptron", "hebb_eps_exponent",
                                     lambda v: v["eps_exponent"], params={"rule": "perceptron"},
                                     route="online")).ok


def test_optimal_versus_batch_bayes_teacher_student():
    """Comparison only (never an input): the online optimal rule is slower than the batch
    Bayes-optimal student of the teacher_student entry (Opper-Haussler), its asymptotic constant
    about twice Bayes', and below the batch curve nowhere on a grid."""
    from replicant.families.teacher_student import asymptotic_constant, teacher_student_error
    c_on = OPT.asymptotic_constant(0)
    c_bayes = float(asymptotic_constant("bayes"))
    assert 1.9 < c_on / c_bayes < 2.1, (c_on, c_bayes)
    tr = O.integrate(O.perceptron_flow("optimal"), 20.0, (0.01, 1.0), alphas=(20.0,))
    for a, eps in zip(tr.alpha, tr.eps):
        assert eps > float(teacher_student_error(a, "bayes")), (a, eps)
    assert "modulated" in OPT.OPTIMALITY_CLASS and "rho" in OPT.OPTIMALITY_CLASS
    f = OPT.optimal_modulation(0.8, noise=0.1)
    assert f.optimality_class == OPT.OPTIMALITY_CLASS and f.rho == 0.8 and f.noise == 0.1


def test_optimal_asymptote_matches_its_formula_with_noise():
    """ODE route (integration to alpha = 1e4) vs the mpmath integral 2 / I(noise),
    I = (1 - 2 noise)^2 int Dt e^{-t^2/2} / (noise + (1 - 2 noise) H(t)): two different codes."""
    for noise in (0.0, 0.2):
        tr = O.integrate(O.perceptron_flow("optimal", noise=noise), 1e4, (0.01, 1.0), alphas=(1e4,),
                         rtol=1e-7)                     # 2e-6 relative on the constant (measured)
        c = OPT.asymptotic_constant(noise)
        assert abs(tr.eps[-1] * 1e4 / c - 1) < 2e-3, (noise, tr.eps[-1] * 1e4, c)
    # the formula as printed in Copelli et al. Eq. (8) (Dt e^{-t^2}) gives 1.325 at noise 0, not
    # the 0.88 the same group reports: recorded as a discrepancy (online/optimal.py docstring)
    assert abs(OPT.asymptotic_constant(0, printed=True) - 1.3248) < 1e-3


# ---- refusals -------------------------------------------------------------------------------
def test_refusals():
    _refuses(lambda: O.perceptron_flow("hebbian"), "unsupported:rule")
    for eta in (0, -1, float("nan")):
        _refuses(lambda: O.perceptron_flow("hebb", eta=eta), "precondition:eta")
    _refuses(lambda: O.perceptron_flow("perceptron", noise=0.5), "precondition:noise")
    _refuses(lambda: O.perceptron_flow("hebb", noise_estimate=0.1), "precondition:known-parameter")
    _refuses(lambda: O.perceptron_flow("optimal", eta=0.5), "precondition:optimal-eta")
    flow = O.perceptron_flow("perceptron")
    _refuses(lambda: O.integrate(flow, 5.0, None), "precondition:state-point")
    _refuses(lambda: O.integrate(flow, 0.0, (0.0, 1.0)), "precondition:alpha")
    _refuses(lambda: O.integrate(flow, 5.0, (2.0, 1.0)), "precondition:initial-overlap")
    _refuses(lambda: O.integrate(O.perceptron_flow("optimal"), 5.0, (0.0, 1.0)),
             "precondition:initial-overlap")
    sched = O.perceptron_flow("hebb", eta=lambda a: 1 - a)          # turns negative at alpha = 1
    _refuses(lambda: O.integrate(sched, 3.0, (0.0, 1.0)), "precondition:eta")
    e = catalog()["online_perceptron"]
    _refuses(lambda: compose(e.spec, "online").run(), "precondition:state-point")
    _refuses(lambda: compose(e.spec, "online").run(rule="hebb", kappa=1), "precondition:known-parameter")
    _refuses(lambda: compose(e.spec, "online").run(rule="widrow", alpha=2), "unsupported:rule")
    _refuses(lambda: compose(e.spec, "online").run(rule="hebb", alpha=2, eta=0), "precondition:eta")


def test_callable_rule_and_schedule():
    """A callable F(h, b, Q) goes through the quadrature route; written as Hebb it reproduces the
    named Hebb flow, and a 1/(1 + alpha) schedule is integrated (and differs from eta = 1)."""
    hebb = O.integrate(O.perceptron_flow("hebb"), 4.0, (0.0, 1.0))
    mine = O.integrate(O.perceptron_flow(lambda h, b, Q: np.ones_like(h)), 4.0, (0.0, 1.0))
    assert abs(hebb.eps[-1] - mine.eps[-1]) < 1e-9, (hebb.eps[-1], mine.eps[-1])
    ann = O.integrate(O.perceptron_flow("hebb", eta=lambda a: 1 / (1 + a)), 4.0, (0.0, 1.0))
    assert abs(ann.eps[-1] - hebb.eps[-1]) > 1e-3


# ---- catalog entry --------------------------------------------------------------------------
def test_catalog_entry_values_and_labels():
    e = catalog()["online_perceptron"]
    assert e.routes == {"online"} and e.default_route == "online"
    res = compose(e.spec, "online").run(rule="perceptron", alpha="3")
    tr = O.integrate(O.perceptron_flow("perceptron"), 3.0, (0.0, 1.0), alphas=(3.0,))
    assert abs(res.values["eps"] - tr.eps[-1]) < 1e-12              # the entry wires the engine
    assert all(not s.final for s in res.finality.values()), res.finality
    assert "physics-level" in res.finality["eps"].note or "assumed" in res.finality["eps"].note
    hebb = compose(e.spec, "online").run(rule="hebb", alpha="3", noise="0.1")
    assert hebb.values["hebb_closed_form_gap"] < 1e-9
    assert "eps_constant" in hebb.values                       # Hebb has an asymptote with noise
    noisy = compose(e.spec, "online").run(rule="perceptron", alpha="3", noise="0.1")
    assert "eps_constant" not in noisy.values                  # skipped: no power law claimed


def test_guide_is_true():
    """The GUIDE's online rows: every refusal reason it lists is raised by the code, and the worked
    example's perceptron numbers are what the CLI prints."""
    import io
    import json
    from pathlib import Path

    from replicant.cli import main
    root = Path(__file__).resolve().parent.parent
    guide = (root / "GUIDE.md").read_text()
    assert "| `online_perceptron`" in guide and "`online_perceptron` (online): `rule`" in guide
    code = "".join(p.read_text() for p in [*(root / "online").glob("*.py"), root / "catalog_online.py"])
    for reason in ("unsupported:rule", "precondition:eta", "precondition:noise", "precondition:optimal-eta",
                   "precondition:initial-overlap", "precondition:alpha", "precondition:overlap",
                   "precondition:averages", "precondition:asymptote", "undetermined:asymptote",
                   "undetermined:rho-sign", "no-solution:rho-zero", "no-solution:step-size",
                   "no-solution:overflow"):
        assert f'"{reason}"' in code, reason
    out = io.StringIO()
    assert main(["run", "online_perceptron", "--route", "online", "--param", "rule=perceptron"], out,
                io.StringIO()) == 0
    v = json.loads(out.getvalue())["values"]
    assert f"`eps_exponent` {v['eps_exponent']:.5f}" in guide, v
    assert f"`eps_constant` {v['eps_constant']:.6f}" in guide, v
