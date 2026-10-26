"""Solvable lattice models, certified (E9): lattice/ (onsager.py, kaufman.py, transfer.py, bethe.py).

Conventions: J = 1 unless stated, K = beta J; values are -beta f = ln Z / N.  Certified values are
lattice.Ball (python-flint arb through tools/baller: baller.solve / baller.run), radius < 1e-30 at
the default 40 digits.  Independent checks: mpmath quadrature (a different rule and formula) for
Onsager, the exact integer transfer matrix for Kaufman, brute-force enumeration for the 1D ring,
the sparse cavity code (cavity/sparse.py: BP on sampled graphs, population dynamics) for Bethe.
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
        assert gaps[0] > gaps[1] > gaps[2] and gaps[2] < mp.mpf("1e-8"), gaps     # measured 2e-4, 4e-7, 1.4e-9


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
    """Comparison targets from the earlier sparse code (statistical, Student-t bands)."""
    import numpy as np
    from replicant.cavity import graphs as G
    from replicant.cavity import sparse as S
    from replicant.cavity._stats import t_quantile
    from replicant.catalog import catalog
    from replicant.lattice import bethe_ising_free_energy
    # (a) viana_bray (cavity, regular +-J, k = 2 excess degree = coordination 3): above both
    # transitions the +-J and the ferromagnetic Bethe free energies are the same paramagnet
    vb = catalog()["viana_bray"].execute("cavity", k=2, beta="0.4", pop=300, sweeps=10, burn_in=60,
                                         runs=4, seed=3)[0]
    f_vb, err = vb["values"]["f"], vb["errors"]["f"]
    assert vb["values"]["phase"] == "paramagnet", vb["values"]      # every run collapsed to h = 0
    phi = bethe_ising_free_energy(3, "0.4", 0).phi
    band = t_quantile(0.995, 3) * float(np.hypot(err, 1e-12))     # float64 rounding of the PD
    assert abs(float(f_vb) + float(phi.mid()) / 0.4) <= band, (f_vb, phi, band)
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
    se = a.std(axis=0, ddof=1) / np.sqrt(len(a))
    t = t_quantile(0.995, len(a) - 1)
    assert abs(a[:, 0].mean() - float(res.phi.mid())) <= t * np.hypot(se[0], 1e-9), (a, res.phi)
    assert abs(a[:, 1].mean() - float(res.magnetization.mid())) <= t * np.hypot(se[1], 1e-9)


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
                         ("bethe_ising", {"beta": 1, "h": "0.02"})):
        e = cat[name]
        res = compose(e.spec, e.default_route).run(**params)
        assert res.rigor == "certified", (name, res.rigor)
        phys = [k for k in res.values if k not in e.meta]
        assert phys and all(k in res.errors for k in phys), (name, set(phys) - set(res.errors))
        assert all(mp.mpf(res.errors[k]) < mp.mpf("1e-25") for k in phys), (name, res.errors)
    # v1 behaviour unchanged: no other entry declares a rigor, and a run stays physics-level
    assert all(not e.rigor for n, e in cat.items() if n not in ("ising2d", "ising1d", "bethe_ising"))
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
    assert m.finality["magnetization"].state == "exact-by-literature" and m.finality["magnetization"].final
    assert not m.final and not m.finality["yang_exponent_loglog"].final     # the slope: a check only
    bp = compose(cat["bethe_ising"].spec, "cavity").run(beta=1, h=0)
    assert "magnetization" not in bp.values and "phi" in bp.values      # h = 0: m not chosen
    assert bp.finality["phi_upper"].state == "stable" and bp.finality["phi_middle"].state == "unstable"
