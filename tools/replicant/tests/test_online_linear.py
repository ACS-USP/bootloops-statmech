"""O2: the linear perceptron, online versus offline (paper 3 of the online spec).

Kinouchi & Caticha, Phys. Rev. E 52, 2878 (1995) (KC95; read from the preprint IFUSP/P-1157, May
1995).  Convention (online/linear.py): teacher B with |B|^2 = M, inputs xi with i.i.d. N(0, 1)
components, linear outputs b = B . xi and h = J . xi, update J <- (1 - Omega/N) J + (eta/N) F xi,
R = J . B, Q = J . J, rho = R / sqrt(Q M), e_g = <(b - h)^2> / 2 = (M + Q - 2R) / 2 (KC Eq. (4)),
alpha = examples / N.  constraint "sphere" holds Q fixed through Omega (KC Eq. (8));
"none" is Omega = 0.  Bank: reference/online_linear.json.
"""
import ast
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.catalog import catalog, check_ref, run_cached
from replicant.compose import compose
from replicant.core import Refusal
from replicant.learning import ridge
from replicant.online import linear as L
from replicant.online import linear_offline as OFF
from replicant.refdata import check, load
from replicant.testing import slow

BANK = load("online_linear")


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


def _rho(flow, alpha, init):
    return float(L.integrate_linear(flow, alpha, init, alphas=(alpha,)).rho[0])


# ---- Gaussian moments ---------------------------------------------------------------------------
def test_moments_closed_vs_quadrature():
    """<b F>, <h F>, <F^2> of the named rules: closed bilinear forms vs a 2D Gauss-Hermite rule
    over (b, h); the planted cross term (linear._PLANT["cross"]) moves <F^2> outside 1e-12."""
    pts = ((0.3, 1.0, 1.0), (0.7, 2.0, 0.5), (-0.2, 0.4, 3.0))         # (R, Q, M)
    for rule, eta in (("standard", 1.0), ("standard", 0.4), ("optimal", 1.0)):
        for R, Q, M in pts:
            if rule == "optimal" and R <= 0:
                continue
            fc = L.linear_flow(rule, eta=eta, M=M)
            fq = L.linear_flow(rule, eta=eta, M=M, averages="quadrature")
            mc, mq = fc.moments(R, Q, 0.5), fq.moments(R, Q, 0.5)
            assert max(abs(a - b) for a, b in zip(mc, mq)) < 1e-12, (rule, R, Q, M, mc, mq)
            with planted(L, cross=0.0):
                mp_ = fc.moments(R, Q, 0.5)
            assert abs(mp_[2] - mq[2]) > 1e-2
            assert fc.moments(R, Q, 0.5) == mc


# ---- the ODE against its closed forms (derived in online/linear.py) -----------------------------
def test_ode_vs_closed_forms():
    """Integrated ODE vs the closed forms: standard rule at constant eta (unconstrained), KC Eq. (23)
    on the sphere, and the optimal rule (KC Eqs. (17)-(18)); the eta^2 term dropped (planted) misses
    them."""
    for eta in (0.5, 1.0, 1.5):
        tr = L.integrate_linear(L.linear_flow("standard", eta=eta), 3.0, (0.1, 0.5), alphas=(3.0,))
        R, Q, eg = L.standard_closed_form(3.0, eta, (0.1, 0.5))
        assert abs(tr.R[0] - R) < 1e-8 and abs(tr.Q[0] - Q) < 1e-8 and abs(tr.eg[0] - eg) < 1e-9
    for M, Q0 in ((1.0, 1.0), (1.0, 4.0), (1.0, 0.25), (2.0, 1.0)):
        f = L.linear_flow("standard", constraint="sphere", M=M)
        tr = L.integrate_linear(f, 2.5, (0.0, Q0), alphas=(2.5,))
        assert abs(tr.Q[0] - Q0) < 1e-9
        assert abs(tr.rho[0] - L.sphere_standard_closed_form(2.5, (0.0, Q0), M)) < 1e-8
    for cons in ("none", "sphere"):
        f = L.linear_flow("optimal", constraint=cons)
        tr = L.integrate_linear(f, 4.0, (0.05, 0.8), alphas=(4.0,))
        R, Q, eg = L.optimal_closed_form(4.0, (0.05, 0.8), constraint=cons)
        assert abs(tr.R[0] - R) < 1e-8 and abs(tr.Q[0] - Q) < 1e-8, (cons, tr, R, Q)
    with planted(L, eta2=0.0):
        tr = L.integrate_linear(L.linear_flow("standard"), 3.0, (0.1, 0.5), alphas=(3.0,))
    assert abs(tr.eg[0] - L.standard_closed_form(3.0, 1.0, (0.1, 0.5))[2]) > 1e-2


def test_tabula_rasa_optimal_is_the_standard_rule():
    """KC p. 9: unconstrained, from sqrt Q = rho sqrt M the optimal F reduces to F = b - h; the
    optimal ODE from (R0, Q0) = (eps, eps) and the standard closed form from the same point agree,
    and e_g = (M - Q0) e^-alpha / 2 (derived) along it."""
    for M in (1.0, 2.0):
        e = 1e-6 * M
        tr = L.integrate_linear(L.linear_flow("optimal", M=M), 3.0, (e, e), alphas=(1.0, 3.0))
        for k, a in enumerate((1.0, 3.0)):
            R, Q, eg = L.standard_closed_form(a, 1.0, (e, e), M=M)
            assert abs(tr.R[k] - R) < 1e-8 and abs(tr.Q[k] - Q) < 1e-8
            assert abs(tr.eg[k] - (M - e) * math.exp(-a) / 2) < 1e-9


# ---- banked online results (KC95 Eqs. (19), (24)-(26), (31)) -----------------------------------
def test_online_learning_curves_vs_banked():
    f = L.linear_flow("optimal", constraint="sphere")
    for a, key in ((1.0, "online_optimal_rho_a1"), (3.0, "online_optimal_rho_a3")):
        assert _ok(key, _rho(f, a, (1e-3, 1.0)))
    std = L.linear_flow("standard", constraint="sphere")
    assert _ok("online_standard_rho_a1", _rho(std, 1.0, (0.0, 1.0)))
    assert not _ok("online_optimal_rho_a1", _rho(std, 1.0, (0.0, 1.0)))      # control


def test_online_asymptotes_vs_banked():
    """rho_max (KC Eq. (24)) and e_g at alpha -> oo (Eq. (25)) for the unrealizable sphere Q != M,
    and the tail 1 - rho_opt ~ e^-alpha / 2 (Eq. (31)), measured on the ODE."""
    a = L.asymptote(L.linear_flow("standard", constraint="sphere"), (0.0, 0.25))
    assert _ok("standard_rho_max_m4", a.rho)
    a = L.asymptote(L.linear_flow("standard", constraint="sphere"), (0.0, 4.0))
    assert _ok("standard_eg_inf_q4", a.eg)
    o = L.asymptote(L.linear_flow("optimal", constraint="sphere"), (0.01, 4.0))
    assert _ok("optimal_eg_inf_q4", o.eg)
    assert not _ok("optimal_eg_inf_q4", a.eg)                               # control
    t = L.asymptote(L.linear_flow("optimal", constraint="sphere"), (1e-3, 1.0))
    assert _ok("optimal_tail_constant", t.tail)


# ---- optimality and the learning-rate schedules ------------------------------------------------
def test_optimal_rule_bounds_its_class():
    """KC p. 9: rho_opt(alpha) is an upper bound for any online F in the class (OPTIMALITY_CLASS);
    the standard rule at eta = 1, at the greedy eta*(alpha), and two callable modulations stay
    below it.  Control: the optimal ODE with the eta^2 term dropped (not a member of any class)
    exceeds the bound."""
    assert "rho" in L.OPTIMALITY_CLASS and "greedy" in L.OPTIMALITY_CLASS
    init = (0.05, 1.0)
    for cons in ("sphere", "none"):
        top = [_rho(L.linear_flow("optimal", constraint=cons), a, init) for a in (2.0,)]
        for flow in (L.linear_flow("standard", constraint=cons),
                     L.linear_flow("standard", eta="greedy", constraint=cons),
                     L.linear_flow(lambda b, h, Q, R: 0.7 * b - 1.2 * h, constraint=cons),
                     L.linear_flow(lambda b, h, Q, R: np.tanh(b) - h, constraint=cons)):
            for t, a in zip(top, (2.0,)):
                assert _rho(flow, a, init) < t - 1e-4
        with planted(L, eta2=0.0):
            assert _rho(L.linear_flow("optimal", constraint=cons), 2.0, init) > top[0] + 1e-2


def test_greedy_eta_and_hypothesis_temperature():
    """greedy_eta = the eta maximising drho/dalpha for F = eta (b - h) (derived): 1 on the tabula
    rasa manifold R = Q (KC p. 9), (1 + rho)/(2 rho) on the sphere Q = M = 1; perturbing it lowers
    drho/dalpha.  Along the optimal trajectory, KC's learning-rate parameter Gamma = (1 - rho^2) /
    rho^2 (Eq. (13)) follows 1 / (e^alpha - 1) from rho0 -> 0 (derived from Eq. (19))."""
    assert abs(L.greedy_eta(0.3, 0.3, 1.0) - 1) < 1e-14
    assert abs(L.greedy_eta(0.6, 1.0, 1.0) - 1.6 / 1.2) < 1e-14
    for cons in ("sphere", "none"):
        for R, Q in ((0.2, 1.0), (0.7, 2.0)):
            e = L.greedy_eta(R, Q, 1.0)
            best = L.linear_flow("standard", eta=e, constraint=cons).rho_rate(0.0, R, Q)
            for d in (-0.1, 0.1):
                assert L.linear_flow("standard", eta=e + d, constraint=cons).rho_rate(0.0, R, Q) < best
            assert L.linear_flow("optimal", constraint=cons).rho_rate(0.0, R, Q) > best
    tr = L.integrate_linear(L.linear_flow("optimal", constraint="sphere"), 3.0, (1e-4, 1.0),
                            alphas=(0.5, 1.0, 3.0))
    for a, g in zip(tr.alpha, L.hypothesis_temperature(tr.rho)):
        assert abs(g - 1 / math.expm1(a)) < 1e-6 * (1 + g), (a, g)


def test_kc_eq20_time_dependent_rule():
    """KC Eq. (20): on the sphere Q = M = 1 the optimal rule with the theoretical rho(alpha) in place
    of the true rho, F = b / sqrt(1 - e^-alpha) - h, is a time-dependent callable rule; integrated
    from alpha0 = 0.01 on the curve (it is singular at alpha = 0), it stays on Eq. (19)
    self-consistently.  A wrong schedule b / (1 - e^-alpha) - h leaves it."""
    a0 = 0.01
    r0 = math.sqrt(-math.expm1(-a0))
    good = L.linear_flow(lambda b, h, Q, R, a: b / math.sqrt(-math.expm1(-a)) - h, constraint="sphere")
    bad = L.linear_flow(lambda b, h, Q, R, a: b / -math.expm1(-a) - h, constraint="sphere")
    tr = L.integrate_linear(good, 2.0, (r0, 1.0), alphas=(0.5, 2.0), alpha0=a0)
    for a, r in zip(tr.alpha, tr.rho):
        assert abs(r - math.sqrt(-math.expm1(-a))) < 1e-8, (a, r)
    assert _ok("online_eq20_rho_a2", tr.rho[1])
    wrong = L.integrate_linear(bad, 2.0, (r0, 1.0), alphas=(2.0,), alpha0=a0).rho[0]
    assert not _ok("online_eq20_rho_a2", wrong) and abs(wrong - tr.rho[1]) > 1e-3, wrong


# ---- offline (KC95 Sec. 4) and the second route through ridge (E7) -----------------------------
def test_offline_vs_banked():
    for a, key, alg in ((0.5, "offline_standard_rho_a0p5", "standard"),
                        (0.5, "offline_optimal_rho_a0p5", "optimal")):
        assert _ok(key, OFF.offline(a, alg).rho)
    assert _ok("offline_optimal_eg_a0p25", OFF.offline(0.25, "optimal").eg)
    assert _ok("offline_standard_eg_a0p25", OFF.offline(0.25, "standard").eg)
    assert not _ok("offline_optimal_rho_a0p5", OFF.offline(0.5, "standard").rho)
    assert OFF.offline(1.5, "standard").rho == 1 and OFF.offline(1.5, "optimal").rho == 1


def test_offline_pseudoinverse_vs_ridge():
    """KC Sec. 4.4: the unconstrained offline solution is the pseudoinverse (rho = sqrt alpha, and
    its direction is the constrained optimum).  Ridge (learning/ridge.py, HMRT Eq. (8)) at sigma2
    = 0, gamma = p/n = 1/alpha, signal r^2 = M gives the excess risk |beta_hat - beta|^2 = 2 e_g;
    with R = Q (a projection of B) rho = sqrt(1 - risk / M).  A planted ridge bias (ridge._SCALE)
    breaks the agreement."""
    for a in (0.25, 0.5, 0.8, 2.0):
        for M in (1.0, 3.0):
            r = OFF.ridge_route(a, M)
            o = OFF.offline(a, "standard", constraint="none", M=M)
            assert abs(r.eg - o.eg) < 1e-12 and abs(r.rho - o.rho) < 1e-12, (a, M, r, o)
            assert abs(r.rho - OFF.offline(a, "optimal", M=M).rho) < 1e-12
    assert _ok("offline_optimal_rho_a0p5", OFF.ridge_route(0.5).rho)
    with planted(ridge, attr="_SCALE", bias=0.5):
        assert abs(OFF.ridge_route(0.5).rho - math.sqrt(0.5)) > 1e-2


def test_online_offline_comparison():
    """KC p. 11: optimal online beats offline standard learning (sphere, Q = M) up to alpha ~ 0.71;
    the crossing is located on the ODE trajectory against the offline zero-error geometry.  Offline
    optimal beats optimal online for every alpha (KC p. 3); unconstrained, the pseudoinverse e_g
    (1 - alpha)/2 is below the online e^-alpha / 2 (derived)."""
    x = OFF.crossing_alpha()
    assert _ok("online_offline_crossing", x)
    import mpmath as mp
    with mp.workdps(30):                       # Eq. (19) = Eq. (27), an independent root
        root = mp.findroot(lambda a: mp.sqrt(1 - mp.exp(-a)) - a, mp.mpf("0.7"))
    assert abs(x - float(root)) < 1e-6, (x, root)
    with planted(L, eta2=0.0):                                  # control: a wrong online flow
        try:
            bad = OFF.crossing_alpha()
        except Refusal as r:
            assert r.reason == "undetermined:crossing"
            bad = None
    assert bad is None or not _ok("online_offline_crossing", bad)
    f = L.linear_flow("optimal", constraint="sphere")
    for a in (0.3, 0.6, 0.9, 1.2):
        on = _rho(f, a, (1e-4, 1.0))
        assert (on > OFF.offline(a, "standard").rho) == (a < x)
        assert on < OFF.offline(a, "optimal").rho
        u = L.integrate_linear(L.linear_flow("standard"), a, (0.0, 0.0), alphas=(a,)).eg[0]
        assert u > OFF.offline(a, "standard", constraint="none").eg


# ---- finite-N simulation -----------------------------------------------------------------------
def test_simulator_shares_no_code_with_the_ode():
    """Every imported module path (dotted components of `import a.b`, `from .a.b import c`, and the
    imported names, which may be modules) avoids the ODE side's modules."""
    src = Path(L.__file__).with_name("simulate_linear.py").read_text()
    parts = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            parts |= {c for a in n.names for c in a.name.split(".")}
        elif isinstance(n, ast.ImportFrom):
            parts |= set((n.module or "").split(".")) | {a.name for a in n.names}
    banned = {"linear", "linear_offline", "odes", "fields", "optimal", "online"}
    assert not parts & banned, parts & banned
    assert "amp" in parts and "_stats" in parts                     # the walk sees dotted paths


def test_ode_vs_simulation_band():
    """N = 200, alpha = 2, 24 named seeds: the ODE e_g lies in the 99% Student-t band of the
    simulation for the optimal rule on the sphere and the standard rule at eta = 0.5; the ODE with
    the eta^2 term dropped (linear._PLANT) falls outside both."""
    from replicant.online import simulate_linear as S
    seeds = tuple(range(300, 324))
    for rule, cons, eta, init in (("optimal", "sphere", 1.0, (0.1, 1.0)),
                                  ("standard", "none", 0.5, (0.0, 1.0))):
        sim = S.simulate_linear(rule, 200, 2.0, seeds, eta=eta, constraint=cons, init=init)
        f = L.linear_flow(rule, eta=eta, constraint=cons)
        ode = L.integrate_linear(f, 2.0, init, alphas=(2.0,)).eg[0]
        assert abs(ode - sim.band.mean) <= sim.band.half_width, (rule, ode, sim.band)
        with planted(L, eta2=0.0):
            bad = L.integrate_linear(f, 2.0, init, alphas=(2.0,)).eg[0]
        assert abs(bad - sim.band.mean) > 3 * sim.band.half_width, (rule, bad, sim.band)


def test_offline_simulation_band():
    """Offline zero-temperature solutions at N = 200, 10 seeds, rho in the band of the geometry.
    method="construct" builds them (pseudoinverse, its normalisation, and a uniform point of the
    zero-error set: the derivation's own assumption); method="descent" assumes nothing (gradient
    descent on the standard energy from a random start on the sphere, from 0 unconstrained).  The
    standard and optimal answers are told apart."""
    from replicant.online import simulate_linear as S
    seeds = tuple(range(10))
    for alg, cons, how in (("standard", "sphere", "construct"), ("optimal", "sphere", "construct"),
                           ("standard", "sphere", "descent"),
                           ("standard", "none", "descent")):
        b = S.simulate_offline(alg, 200, 0.5, seeds, constraint=cons, method=how).rho_band
        want = OFF.offline(0.5, alg, constraint=cons).rho
        assert abs(want - b.mean) <= b.half_width + 1e-12, (alg, cons, how, want, b)
        if alg == "standard" and cons == "sphere":
            assert abs(OFF.offline(0.5, "optimal").rho - b.mean) > 0.1
    _refuses(lambda: S.simulate_offline("optimal", 50, 0.5, seeds, method="descent"),
             "precondition:method")


@slow
def test_ode_vs_simulation_band_large_n():
    """N = 1000, alpha = 3, 20 fresh seeds: optimal and standard on both constraints, the greedy
    schedule, M = 2; plants (eta^2 term halved) excluded."""
    from replicant.online import simulate_linear as S
    seeds = tuple(range(1000, 1020))
    cases = (("optimal", "sphere", 1.0, (0.1, 1.0), 1.0), ("optimal", "none", 1.0, (0.1, 0.5), 1.0),
             ("standard", "sphere", 1.0, (0.0, 1.0), 1.0), ("standard", "none", 0.5, (0.0, 1.0), 2.0),
             ("standard", "none", 1.5, (0.2, 1.0), 1.0), ("standard", "sphere", "greedy", (0.1, 1.0), 1.0))
    for rule, cons, eta, init, M in cases:
        sim = S.simulate_linear(rule, 1000, 3.0, seeds, eta=eta, constraint=cons, init=init, M=M)
        f = L.linear_flow(rule, eta=eta, constraint=cons, M=M)
        ode = L.integrate_linear(f, 3.0, init, alphas=(3.0,)).eg[0]
        assert abs(ode - sim.band.mean) <= sim.band.half_width, (rule, cons, eta, ode, sim.band)
        with planted(L, eta2=0.5):
            bad = L.integrate_linear(f, 3.0, init, alphas=(3.0,)).eg[0]
        print(f"{rule}/{cons}/eta={eta}: sim {sim.band.mean:.5f} +- {sim.band.half_width:.5f} ode "
              f"{ode:.5f} planted {bad:.5f} ({abs(bad - sim.band.mean) / sim.band.half_width:.1f} hw)")
        assert abs(bad - sim.band.mean) > 2 * sim.band.half_width


# ---- refusals ----------------------------------------------------------------------------------
def test_refusals():
    _refuses(lambda: L.linear_flow("hebb"), "unsupported:rule")
    _refuses(lambda: L.linear_flow("standard", eta=0), "precondition:eta")
    _refuses(lambda: L.linear_flow("standard", eta="fast"), "precondition:eta")
    _refuses(lambda: L.linear_flow("optimal", eta=0.5), "precondition:optimal-eta")
    _refuses(lambda: L.linear_flow("optimal", eta="greedy"), "precondition:optimal-eta")
    _refuses(lambda: L.linear_flow("standard", constraint="ball"), "precondition:constraint")
    _refuses(lambda: L.linear_flow("standard", M=0), "precondition:teacher-norm")
    _refuses(lambda: L.linear_flow("standard", averages="mc"), "precondition:averages")
    f = L.linear_flow("standard", constraint="sphere")
    _refuses(lambda: L.integrate_linear(f, 1.0), "precondition:state-point")
    _refuses(lambda: L.integrate_linear(f, 1.0, (0.0, 0.0)), "precondition:initial-overlap")
    _refuses(lambda: L.integrate_linear(f, 1.0, (2.0, 1.0)), "precondition:initial-overlap")
    _refuses(lambda: L.integrate_linear(f, -1.0, (0.0, 1.0)), "precondition:alpha")
    _refuses(lambda: L.integrate_linear(L.linear_flow("optimal"), 1.0, (0.0, 1.0)),
             "precondition:initial-overlap")
    _refuses(lambda: L.integrate_linear(L.linear_flow("standard", eta="greedy"), 1.0, (0.0, 0.0)),
             "precondition:initial-overlap")
    _refuses(lambda: OFF.offline(0.5, "adaline"), "unsupported:algorithm")
    _refuses(lambda: OFF.offline(0, "standard"), "precondition:alpha")
    _refuses(lambda: OFF.offline(0.5, "standard", M=1, Q=0.25), "precondition:offline-window")
    _refuses(lambda: OFF.offline(0.5, "standard", constraint="ball"), "precondition:constraint")


def test_catalog_entry_runs_and_refuses():
    e = catalog()["online_linear"]
    v = run_cached(e, "online", {})[0]                           # every default (generic tables)
    assert v["rule"] == "optimal" and v["constraint"] == "sphere"
    v = e.run(route="online", alpha="0.5")
    assert abs(v["rho"] - math.sqrt(1 - math.exp(-0.5))) < 1e-5
    assert abs(v["rho_offline_standard"] - 0.5) < 1e-12 and abs(v["rho_offline_optimal_ridge"]
                                                                - math.sqrt(0.5)) < 1e-12
    assert not e.finality["rho"].final and not e.finality["rho_offline_optimal"].final
    _refuses(lambda: compose(e.spec, "online").run(rul="optimal"), "precondition:known-parameter")
    _refuses(lambda: e.run(route="online", rule="standard", eta="greedy", constraint="none",
                           R0=0, Q0=0, alpha=1), "precondition:initial-overlap")
    _refuses(lambda: e.run(route="online", rule="hebb"), "unsupported:rule")
    _refuses(lambda: e.run(route="online", constraint="ball"), "precondition:constraint")
    _refuses(lambda: e.run(route="online", rule=lambda b, h, Q, R: b - h), "unsupported:rule")
    for ref in e.refs[:2]:                     # test_catalog checks all 13 in the default tier
        out = check_ref(e, ref)
        assert out.ok, out


def test_guide_is_true():
    """The GUIDE's online_linear rows: every refusal reason listed is raised by the code, and the
    worked example's numbers are what the CLI prints."""
    import io
    import json

    from replicant.cli import main
    root = Path(__file__).resolve().parent.parent
    guide = (root / "GUIDE.md").read_text()
    assert "| `online_linear`" in guide and "`online_linear` (online): `rule`" in guide
    code = "".join(p.read_text() for p in [*(root / "online").glob("linear*.py"),
                                           root / "online" / "simulate_linear.py",
                                           root / "catalog_online_linear.py"])
    for reason in ("unsupported:rule", "unsupported:algorithm", "precondition:eta",
                   "precondition:optimal-eta", "precondition:constraint", "precondition:teacher-norm",
                   "precondition:averages", "precondition:state-point", "precondition:initial-overlap",
                   "precondition:alpha", "precondition:asymptote", "precondition:offline-window",
                   "undetermined:asymptote", "undetermined:crossing", "no-solution:rho-zero",
                   "precondition:method", "undetermined:descent"):
        assert f'"{reason}"' in code, reason
    out = io.StringIO()
    assert main(["run", "online_linear", "--route", "online", "--param", "alpha=0.5"], out,
                io.StringIO()) == 0
    v = json.loads(out.getvalue())["values"]
    assert "alpha_cross" not in v and "rho_inf" not in v
    v.update(run_cached(catalog()["online_linear"], "online", {})[0])     # the run without alpha
    for key, fmt in (("rho", ".6f"), ("gamma_temperature", ".6f"), ("alpha_cross", ".6f"),
                     ("rho_offline_optimal_ridge", ".6f"), ("tail_constant", ".7f")):
        assert f"`{key}` {v[key]:{fmt}}" in guide, (key, v[key])
