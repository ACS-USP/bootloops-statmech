"""Solvable lattice models, certified (E9): lattice/ (onsager.py, kaufman.py, transfer.py, bethe.py).

Conventions: J = 1 unless stated, K = beta J; values are -beta f = ln Z / N.  Certified values are
lattice.Ball (python-flint arb through tools/baller: baller.solve / baller.run), radius < 1e-30 at
the default 40 digits.  Independent checks: mpmath quadrature (a different rule and formula) for
Onsager, the exact integer transfer matrix for Kaufman, brute-force enumeration for the 1D ring,
the sparse cavity code (cavity/sparse.py: BP on sampled graphs, population dynamics) for Bethe
(deterministic implementation agreement there, not a statistical estimate).
Planted errors go through each module's _HOOKS registry, restored in try/finally.
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
    """fn() with module._HOOKS[key] = value; the registry is restored (identity asserted)."""
    reg = module._HOOKS
    old = reg[key]
    reg[key] = value
    try:
        return fn()
    finally:
        reg[key] = old
        assert module._HOOKS is reg and reg[key] is old


def _flint_prec():
    from flint import ctx
    return ctx.prec


# ---- Onsager ---------------------------------------------------------------------------------
def test_onsager_certified_value_matches_bank():
    from replicant.lattice import BETA_C, onsager, onsager_free_energy
    before, fprec = mp.mp.dps, _flint_prec()
    b = onsager_free_energy(BETA_C)
    assert b.rad() < mp.mpf("1e-30"), b
    want = load("lattice")["onsager_critical_minus_beta_f"]
    with mp.workdps(40):
        assert abs(b.mid() - mp.mpf(want.value)) <= mp.mpf(want.tol)
        # independent: mpmath tanh-sinh at 50 digits of the Onsager integral (inner angle closed)
        ind = onsager.onsager_free_energy_mp(BETA_C, dps=50)
        assert abs(ind - b.mid()) < b.rad() + mp.mpf("1e-45"), (ind, b)
    # planted: a wrong Catalan coefficient (3G/pi) is caught by the bank and by the quadrature
    bad = _planted(onsager, "catalan_coefficient", 3, lambda: onsager_free_energy(BETA_C))
    with mp.workdps(40):
        assert abs(bad.mid() - mp.mpf(want.value)) > 1e-3 and not bad.contains(ind)
    assert mp.mp.dps == before and _flint_prec() == fprec


def test_onsager_integral_certified_away_from_critical():
    from replicant.lattice import onsager, onsager_free_energy
    for beta in ("0.3", Fraction(1, 2), "0.8"):
        b = onsager_free_energy(beta)
        assert b.rad() < mp.mpf("1e-30"), (beta, b)
        with mp.workdps(40):
            ind = onsager.onsager_free_energy_mp(beta, dps=45)
            assert abs(ind - b.mid()) < b.rad() + mp.mpf("1e-40"), (beta, ind, b)
    # J scales K: (beta, J) = (0.15, 2) is K = 0.3
    assert onsager_free_energy("0.15", J=2).overlaps(onsager_free_energy("0.3"))
    # high-temperature limit: -beta f -> ln 2 + 2 ln cosh K + O(K^4) (series a4 = 1)
    small = onsager_free_energy("0.001")
    with mp.workdps(40):
        lead = mp.log(2) + 2 * mp.log(mp.cosh(mp.mpf("0.001")))
        assert abs(small.mid() - lead - mp.mpf("1e-12")) < mp.mpf("1e-17")   # a4 v^4, v ~ K


def test_onsager_double_integral_trapezoid_and_large_torus():
    """The double integral itself (periodic trapezoid, exponentially convergent off beta_c) and
    the Kaufman torus approaching the bulk value: two routes sharing no code with the ball."""
    from replicant.lattice import kaufman_partition_function, onsager, onsager_free_energy
    b = onsager_free_energy("0.3")
    with mp.workdps(40):
        two = onsager.onsager_free_energy_mp("0.3", dps=30, method="trapezoid", nodes=160)
        assert abs(two - b.mid()) < mp.mpf("1e-25"), (two, b)
        gaps = []
        for L in (8, 16, 24):
            z = kaufman_partition_function(L, L, "0.3")
            gaps.append(abs(mp.log(z.mid()) / L**2 - b.mid()))
        # measured 2e-4, 4e-7, 1.4e-9
        assert gaps[0] > gaps[1] > gaps[2] and gaps[2] < mp.mpf("1e-8"), gaps


def test_onsager_certifies_near_critical():
    """Within 1e-4 .. 1e-22 of beta_c the cancellation-free integrand certifies 40 digits quickly;
    the -beta f values bracket the critical constant (energy continuous at beta_c)."""
    from replicant.lattice import BETA_C, onsager_free_energy
    crit = onsager_free_energy(BETA_C)
    for beta in ("0.4407", "0.44068", "0.44069", "0.4406867935097715126163"):
        b = onsager_free_energy(beta)
        assert b.rad() < mp.mpf("1e-40") and b.digits >= 40, (beta, b)
        with mp.workdps(40):
            gap = abs(b.mid() - crit.mid())
            assert gap < 2 * abs(mp.mpf(beta) - mp.mpf("0.44068679350977151262")), (beta, gap)


@slow
def test_onsager_naive_integrand_stalls_and_is_refused_fast():
    import time
    from replicant.lattice import onsager, onsager_free_energy
    t = time.time()
    r = _refusal(lambda: _planted(onsager, "naive_integrand", True,
                                  lambda: onsager_free_energy("0.4407")))
    assert r.reason == "no-solution:radius-stall" and time.time() - t < 10, (r, time.time() - t)


def test_printed_certified_values_are_enclosures():
    """The value +- error printed in the JSON (cli.result_json) and in the LaTeX solution table
    contains the ball, for every physical value of a certified run."""
    import re
    from replicant.catalog import catalog
    from replicant.cli import result_json
    from replicant.compose import compose
    from replicant.latex import render_latex
    from replicant.lattice import BETA_C, onsager_free_energy
    from replicant.tex import escape
    from flint import arb
    e = catalog()["ising2d"]
    res = compose(e.spec, "exact").run()
    doc, js = render_latex(res), result_json("ising2d", e.spec, res)
    ball = onsager_free_energy(BETA_C).arb
    rows = 0
    for k in res.values:
        if k in e.meta:
            continue
        row = next(l for l in doc.splitlines() if l.startswith(escape(k) + " &"))
        v, err = re.search(r"& ([-0-9.e]+) \$\\pm\$ ([-0-9.e]+) \(certified ball\)", row).groups()
        assert v == js["values"][k] and err == js["errors"][k], (k, row, js["values"][k])
        rows += 1
    v, err = js["values"]["minus_beta_f"], js["errors"]["minus_beta_f"]
    with __import__("replicant.lattice.ball", fromlist=["baller"]).baller().hygiene.ctx_guard(prec=300):
        lo, hi = arb(v) - arb(err), arb(v) + arb(err)
        assert lo.upper() < ball.lower() and ball.upper() < hi.lower(), (v, err)
        bad = arb(v[:22]) + arb("1e-42")          # a 20-digit print with the ball's radius
        assert not (bad - arb("1e-42") <= ball.lower() and ball.upper() <= bad + arb("1e-42"))
    assert rows >= 4


# ---- Kaufman versus the transfer matrix ------------------------------------------------------
def test_kaufman_equals_transfer_matrix_exact():
    from replicant.lattice import kaufman, kaufman_polynomial, torus_partition_polynomial
    from replicant.lattice import kaufman_partition_function
    for Lx, Ly in ((2, 2), (2, 3), (3, 4), (4, 3), (4, 4)):
        tm = torus_partition_polynomial(Lx, Ly)
        N = Lx * Ly
        assert len(tm) == 2 * N + 1 and sum(tm) == 2**N and tm[0] == 2, (Lx, Ly, tm)
        rec = kaufman_polynomial(Lx, Ly)
        assert rec.coefficients == tm and rec.max_radius < mp.mpf("0.5"), (Lx, Ly, rec)
        # held-out rational points (not interpolation nodes), one near x_c = sqrt 2 - 1
        for x in (Fraction(1, 7), Fraction(29, 70), Fraction(9, 10)):
            z = kaufman_partition_function(Lx, Ly, kaufman.beta_of_x(x))
            exact = sum(c * x**j for j, c in enumerate(tm)) / x**N      # Z = x^-N P(x)
            assert z.contains(exact), (Lx, Ly, x, z)
    # planted: gamma_0 unsigned (|2K + ln tanh K|) is caught by the exact comparison
    tm = torus_partition_polynomial(3, 4)
    bad = _planted(kaufman, "gamma0", lambda K: abs(2 * K + K.tanh().log()),
                   lambda: kaufman_polynomial(3, 4, require_integers=False))
    assert bad.coefficients != tm


@slow
def test_kaufman_equals_transfer_matrix_larger_tori():
    from replicant.lattice import kaufman_polynomial, torus_partition_polynomial
    for Lx, Ly in ((5, 5), (6, 5), (6, 6)):
        assert kaufman_polynomial(Lx, Ly).coefficients == torus_partition_polynomial(Lx, Ly)


def test_transfer_matrix_free_energy_ball():
    from replicant.lattice import BETA_C, ising_transfer_matrix_free_energy, torus_partition_polynomial
    b = ising_transfer_matrix_free_energy(3, "0.4", Lx=4)
    tm = torus_partition_polynomial(4, 3)
    with mp.workdps(40):
        K = mp.mpf("0.4")
        x = mp.exp(-2 * K)
        want = (2 * 12 * K + mp.log(sum(c * x**j for j, c in enumerate(tm)))) / 12
        assert abs(b.mid() - want) < mp.mpf("1e-35") and b.rad() < mp.mpf("1e-30")
    assert ising_transfer_matrix_free_energy(4, BETA_C).rad() < mp.mpf("1e-30")


# ---- Yang ------------------------------------------------------------------------------------
def test_yang_magnetization_exponent_one_eighth():
    from replicant.lattice import yang_magnetization
    from replicant.lattice.onsager import yang_loglog_slope
    m = yang_magnetization(1)
    with mp.workdps(40):
        assert abs(m.mid() - (1 - mp.sinh(2)**-4) ** (mp.mpf(1) / 8)) < mp.mpf("1e-35")
    s = yang_loglog_slope("1e-6", "1e-8")
    want = load("lattice")["yang_exponent"]
    assert s.rad() < mp.mpf("1e-25")
    with mp.workdps(30):
        assert abs(s.mid() - mp.mpf(want.value)) < mp.mpf(want.tol), s
        # the derived correction: M^8 = 8 sqrt2 d (1 - (9/sqrt2) d + O(d^2)), so the slope is
        # 1/8 - (9/(8 sqrt2)) (d2 - d1)/ln(d2/d1) + O(d^2) = 1/8 - 1.71013e-7
        d1, d2 = mp.mpf("1e-6"), mp.mpf("1e-8")
        corr = 9 / (8 * mp.sqrt(2)) * (d2 - d1) / mp.log(d2 / d1)
        assert abs(s.mid() - (mp.mpf(1) / 8 - corr)) < mp.mpf("1e-11"), (s, corr)
        assert abs(s.mid() - mp.mpf(1) / 8) < mp.mpf("2e-7")
        # positive control: the slope of m^2 (exponent 1/4) is outside the band
        assert abs(2 * s.mid() - mp.mpf(want.value)) > mp.mpf(want.tol)
    # refused at and below beta_c (the paramagnet, m* = 0): a certified comparison decides
    for beta in ("0.44", Fraction(2, 5), "0.4406867935"):
        r = _refusal(lambda: yang_magnetization(beta))
        assert r.reason == "precondition:ordered-phase", r
    assert yang_magnetization("0.4406867936").mid() > 0


# ---- 1D Ising --------------------------------------------------------------------------------
def test_ising1d_ring_identity_exact_and_ball():
    from replicant.lattice import ising1d_free_energy, transfer
    for N in range(2, 11):
        for a, b in ((Fraction(3, 2), Fraction(1)), (Fraction(5, 4), Fraction(7, 3)),
                     (Fraction(2, 3), Fraction(1, 5))):
            chk = transfer.ising1d_ring_identity(N, a, b)
            assert chk.brute == chk.trace == chk.closed, (N, a, b, chk)
    bad = _planted(transfer, "d_sign", -1, lambda: transfer.ising1d_ring_identity(6, Fraction(3, 2),
                                                                                  Fraction(7, 3)))
    assert bad.closed != bad.brute
    want = load("lattice")["ising1d_minus_beta_f_beta1_h0"]
    b = ising1d_free_energy(1, 0)
    with mp.workdps(40):
        assert abs(b.mid() - mp.mpf(want.value)) <= mp.mpf(want.tol) and b.rad() < mp.mpf("1e-30")
        K, H = mp.mpf("0.7"), mp.mpf("0.3") * mp.mpf("0.7")
        lam = mp.exp(K) * mp.cosh(H) + mp.sqrt(mp.exp(2 * K) * mp.sinh(H)**2 + mp.exp(-2 * K))
        b = ising1d_free_energy("0.7", "0.3")
        assert abs(b.mid() - mp.log(lam)) < b.rad() + mp.mpf("1e-38"), b


# ---- Bethe lattice ---------------------------------------------------------------------------
def test_bethe_fixed_points_certified_and_branches():
    from replicant.lattice import bethe_critical_beta, bethe_ising_fixed_points, bethe_ising_free_energy
    want = load("lattice")["bethe_beta_c_k3"]
    with mp.workdps(40):
        assert abs(bethe_critical_beta(3).mid() - mp.mpf(want.value)) <= mp.mpf(want.tol)
    assert len(bethe_ising_fixed_points(3, "0.54", 0)) == 1
    fps = bethe_ising_fixed_points(3, "0.56", 0)
    assert [f.branch for f in fps] == ["lower", "middle", "upper"]
    fps = bethe_ising_fixed_points(3, 1, 0)
    lo, mid, up = fps
    assert not mid.stable and lo.stable and up.stable and mid.cavity_field.mid() == 0
    assert lo.phi.overlaps(up.phi) and up.phi.mid() > mid.phi.mid()
    assert lo.magnetization.overlaps(-up.magnetization)
    for f in fps:                       # the two free-energy formulas agree at every fixed point
        assert f.phi.overlaps(f.phi_dm), f
    from replicant.lattice import bethe
    r = _refusal(lambda: _planted(bethe, "link_weight", 3, lambda: bethe_ising_fixed_points(3, 1, 0)))
    assert r.reason == "no-solution:audit", r                   # planted: the two forms disagree
    res = bethe_ising_free_energy(3, 1, 0)
    assert res.selected is None and res.phi.overlaps(up.phi)
    r = _refusal(lambda: res.magnetization)
    assert r.reason == "ambiguous:branch" and "lower" in r.detail and "upper" in r.detail, r
    assert bethe_ising_free_energy(3, 1, 0, branch="upper").magnetization.mid() > 0.9
    # h > 0: Dembo-Montanari select the unique non-negative fixed point (three exist here)
    pos = bethe_ising_free_energy(3, 1, "0.02")
    assert len(pos.fixed_points) == 3 and pos.selected.branch == "upper"
    assert bethe_ising_free_energy(3, 1, "-0.02").selected.branch == "lower"
    # paramagnet: -beta f = ln 2 + (k/2) ln cosh beta
    para = bethe_ising_free_energy(4, "0.2", 0)
    with mp.workdps(40):
        want = mp.log(2) + 2 * mp.log(mp.cosh(mp.mpf("0.2")))
        assert abs(para.phi.mid() - want) < para.phi.rad() + mp.mpf("1e-38"), para.phi


def test_bethe_ising_matches_viana_bray_regular():
    """Comparison targets from the earlier sparse code.  Both are deterministic implementation
    agreement, not statistical estimates: the paramagnetic population collapses exactly to h = 0
    (stderr 0), and BP reaches the homogeneous fixed point on every regular instance (spread at
    float rounding).  Tolerances: 1e-12 and 1e-9 (float64 arithmetic of cavity/sparse.py)."""
    import numpy as np
    from replicant.cavity import graphs as G
    from replicant.cavity import sparse as S
    from replicant.catalog import catalog
    from replicant.lattice import bethe_ising_free_energy
    # (a) viana_bray (cavity, regular +-J, k = 2 excess degree = coordination 3): above both
    # transitions the +-J and the ferromagnetic Bethe free energies are the same paramagnet
    vb = catalog()["viana_bray"].execute("cavity", k=2, beta="0.4", pop=300, sweeps=10, burn_in=60,
                                         runs=4, seed=3)[0]
    f_vb, err = vb["values"]["f"], vb["errors"]["f"]
    assert vb["values"]["phase"] == "paramagnet", vb["values"]      # every run collapsed to h = 0
    phi = bethe_ising_free_energy(3, "0.4", 0).phi
    assert err == 0 and abs(float(f_vb) + float(phi.mid()) / 0.4) <= 1e-12, (f_vb, err, phi)
    # (b) BP (cavity/sparse.bp_fixed_point) on sampled ferromagnetic 3-regular graphs, named seeds
    beta, h = 1.0, 0.1
    res = bethe_ising_free_energy(3, 1, "0.1")
    got = []
    for seed in (101, 102, 103, 104):
        rng = np.random.default_rng(seed)
        edges = G.random_regular(200, 3, rng)
        g = G.Graph(200, edges, np.ones(len(edges)))
        bp = S.bp_fixed_point(g, beta, None, seed=seed, field=h, require_converged=True)
        got.append((bp.log_z / 200, float(np.mean(bp.marginals))))
    a = np.array(got)
    assert np.all(np.abs(a[:, 0] - float(res.phi.mid())) <= 1e-9), (a, res.phi)       # every seed
    assert np.all(np.abs(a[:, 1] - float(res.magnetization.mid())) <= 1e-9), a


# ---- refusals --------------------------------------------------------------------------------
def test_lattice_refusals():
    from replicant.lattice import (bethe_ising_free_energy, ising1d_free_energy, kaufman_polynomial,
                                   onsager_free_energy, torus_partition_polynomial)
    cases = [(lambda: onsager_free_energy(0), "precondition:beta"),
             (lambda: onsager_free_energy(True), "precondition:beta"),
             (lambda: onsager_free_energy("nan"), "precondition:beta"),
             (lambda: onsager_free_energy("0.3", J=-1), "precondition:J"),
             (lambda: torus_partition_polynomial(1, 4), "precondition:size"),
             (lambda: torus_partition_polynomial(4, 9), "precondition:size"),
             (lambda: kaufman_polynomial(2, 1), "precondition:size"),
             (lambda: ising1d_free_energy(-1, 0), "precondition:beta"),
             (lambda: ising1d_free_energy(1, "x"), "precondition:h"),
             (lambda: bethe_ising_free_energy(1, 1, 0), "precondition:coordination"),
             (lambda: bethe_ising_free_energy(3.0, 1, 0), "precondition:coordination"),
             (lambda: bethe_ising_free_energy(3, 1, 0, branch="top"), "precondition:branch"),
             (lambda: bethe_ising_free_energy(3, "0.3", 0, branch="upper"), "precondition:branch")]
    for fn, reason in cases:
        r = _refusal(fn)
        assert r.reason == reason, (reason, r)


# ---- catalog: certified label and radii ------------------------------------------------------
def test_lattice_entries_are_certified_with_radii_and_others_unchanged():
    from replicant.catalog import catalog
    from replicant.compose import compose
    cat = catalog()
    for name, params in (("ising2d", {}), ("ising2d", {"beta": "0.6"}), ("ising1d", {}),
                         ("bethe_ising", {"beta": 1, "h": "0.02"}), ("ising2d_anisotropic", {}),
                         ("ising_triangular", {}), ("ising_honeycomb", {})):
        e = cat[name]
        res = compose(e.spec, e.default_route).run(**params)
        assert res.rigor == "certified", (name, res.rigor)
        phys = [k for k in res.values if k not in e.meta]
        assert phys and all(k in res.errors for k in phys), (name, set(phys) - set(res.errors))
        assert all(mp.mpf(res.errors[k]) < mp.mpf("1e-25") for k in phys), (name, res.errors)
    # v1 behaviour unchanged: no other entry declares a rigor (E10's sk_bounds is certified too:
    # tests/test_guerra.py), and a run stays physics-level
    assert all(not e.rigor for n, e in cat.items()
               if n not in ("ising2d", "ising1d", "bethe_ising", "sk_bounds", "ising2d_anisotropic",
                            "ising_triangular", "ising_honeycomb", "percolation_bethe",
                            "percolation_graph", "percolation_lattice", "percolation_kcore",
                            "percolation_bootstrap", "cluster_mc", "wang_landau_mc",
                            "histogram_reweighting_mc"))
    assert compose(cat["rem"].spec).run().rigor == "physics-level"
    # a certified route that returns a value without its radius is refused
    from replicant.catalog import Out, entry, fstep
    from replicant.core import Refusal as R
    bad = entry("bad", cat["ising1d"].spec, "x", {"exact": [fstep("s", "x", lambda p: Out({"v": 1}))]},
                [], {"exact": ""}, "", default_route="exact", finality={"v": cat["ising1d"].finality[
                    "minus_beta_f"]}, parameters={"exact": ()}, rigor={"exact": "certified"})
    from replicant.compose import ComposedCalculation, CompositionReport
    calc = ComposedCalculation(bad.spec, "exact", bad.steps, CompositionReport([], [], []), bad)
    r = _refusal(lambda: calc.run())
    assert r.reason == "precondition:certified-radius" and "v" in r.detail, r
    m = compose(cat["ising2d"].spec, "exact").run(beta="0.6")
    mag = m.finality["magnetization"]
    assert mag.state == "exact-by-literature" and mag.final
    assert not m.final and not m.finality["yang_exponent_loglog"].final     # the slope: a check only
    bp = compose(cat["bethe_ising"].spec, "cavity").run(beta=1, h=0)
    assert "magnetization" not in bp.values and "phi" in bp.values      # h = 0: m not chosen
    assert bp.finality["beta_c"].state == "not-computed"            # a closed-form threshold
    assert "Frame 2" in m.finality["torus_minus_beta_f_tm"].cite   # exact by construction (TM)
    assert bp.finality["phi_upper"].state == "stable" and bp.finality["phi_middle"].state == "unstable"
