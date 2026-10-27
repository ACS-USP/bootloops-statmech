"""Exact observables of the 1D and 2D Ising entries (I1): lattice/observables1d.py,
lattice/onsager_thermo.py, lattice/torus_thermo.py, catalog_lattice_obs.py.

Routes.  1D: the N-ring (Tr V^N with d/dH by exact truncated series; exact rationals at rational
a = e^K, b = e^H, equal to brute-force enumeration) against Baxter's closed forms, with the N -> oo
gap made explicit (exact finite-N correction and an upper bound, both from ln lambda_{1,2} by
arb_series, never from the closed forms).  2D: Onsager's Eqs. (116), (117) against the derivatives
of his single integral under the integral sign (acb.integral), the K-derivatives of Kaufman's torus Z
(arb_series) and of the exact integer polynomial, and E8's high-temperature series.  Planted errors
go through each module's _HOOKS (restored in try/finally, identity asserted).  J = 1 throughout.
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


def _near(ball, rec, extra=0):
    with mp.workdps(45):
        return abs(ball.mid() - mp.mpf(rec.value)) <= mp.mpf(rec.tol) + extra


# ---- 1D: the ring ------------------------------------------------------------------------------
def test_ising1d_ring_exact_equals_brute_force():
    from replicant.lattice import observables1d as O
    for N, a, b, r in ((5, 3, 2, 1), (7, 2, 1, 2), (6, Fraction(5, 2), Fraction(1, 3), 3)):
        assert O.ring_observables(N, a, b, r) == O.ring_brute(N, a, b, r), (N, a, b)
    x = O.ring_observables(9, 3, 2, 1)
    assert all(isinstance(getattr(x, f), Fraction) for f in ("Z", "m", "chi_over_beta", "corr"))


def test_ising1d_exact_rational_ring_converges_to_closed_forms():
    """Exact rationals at a = 3, b = 2 (K = ln 3, H = ln 2): |ring - closed| <= the explicit
    bound q |l'| (m), q (N l'^2 + |l''|) (chi / beta), rho^{N-r} + 2 q |l'| (G), and the bound
    goes to 0 as rho^N."""
    from flint import arb
    from replicant.lattice import observables1d as O
    from replicant.lattice.ball import arb_q, baller
    with baller().hygiene.ctx_guard(prec=400):
        k, x = arb(3).log(), arb(2).log()
        lp, lm = O._log_lambda_series(k, x)
        ell = lm - lp
        l1, l2 = O._coef(ell, 1).abs_upper(), (2 * O._coef(ell, 2)).abs_upper()
        rho = O._coef(ell, 0).exp()
        closed = (O._m(k, x), O._chi(k, x, arb(1)), O._corr(k, x, 1))
        for N, lim in ((12, "1e-5"), (40, "1e-22"), (80, "1e-46")):
            ring = O.ring_observables(N, 3, 2, 1)
            q = rho ** N
            bounds = (q * l1, q * (N * l1 * l1 + l2), rho ** (N - 1) + 2 * q * l1)
            for got, c, bd in zip((ring.m, ring.chi_over_beta, ring.corr), closed, bounds):
                assert (arb_q(got) - c).abs_upper() <= bd, (N, got, c, bd)
            assert max(bounds) < arb(lim), (N, bounds)


def test_ising1d_ring_check_identity_and_limit_and_bank():
    from replicant.lattice import observables1d as O
    small, big = O.ring_check(8, 1, "1/2", 1), O.ring_check(200, 1, "1/2", 1)
    assert small.ok and big.ok
    # N = 8: the ring differs from the bulk (the identity is what checks it); N = 200: the
    # ring alone reproduces the bank (closed forms not used), within the explicit bound
    assert small.observables["magnetization"].bound.mid() > mp.mpf("1e-4")
    for name, key in (("magnetization", "ising1d_magnetization_beta1_h1_2"),
                      ("susceptibility", "ising1d_susceptibility_beta1_h1_2"),
                      ("correlation", "ising1d_correlation_beta1_h1_2"),
                      ("correlation_length", "ising1d_correlation_length_beta1_h1_2")):
        o = big.observables[name]
        assert o.bound.mid() < mp.mpf("1e-80"), (name, o.bound)
        assert _near(o.ring, _bank(key), o.bound.mid()), (name, o.ring)
        assert _near(o.closed, _bank(key)), name


def test_ising1d_closed_forms_two_susceptibilities_and_zero_field():
    from replicant.lattice import observables1d as O
    for beta, h in ((1, "1/2"), ("0.7", "-0.3"), (2, 0)):
        a, b = O.ising1d_susceptibility(beta, h), O.ising1d_susceptibility_fluctuation(beta, h)
        assert a.overlaps(b) and a.rad() < mp.mpf("1e-40"), (beta, h, a, b)
    with mp.workdps(45):     # h = 0: chi = beta e^{2K} (arXiv:2511.15399 Eq. (15)), m = 0
        assert abs(O.ising1d_susceptibility(2, 0).mid() - 2 * mp.exp(4)) < mp.mpf("1e-38")
        assert abs(O.ising1d_correlation_length(1, 0).mid() + 1 / mp.log(mp.tanh(1))) < mp.mpf("1e-38")
    assert O.ising1d_magnetization(1, 0).contains(0)
    assert _refusal(lambda: O.ising1d_correlation(1, 0, -1)).reason == "precondition:distance"
    assert _refusal(lambda: O.ring_check(8, 1, 0, 7)).reason == "precondition:distance"


def test_ising1d_planted_closed_forms_fail_the_ring():
    """One planted error per 1D observable route: each fails the ring check (the N = 8 identity
    and the N = 200 bound) and the catalog step refuses."""
    from replicant.catalog import catalog
    from replicant.compose import compose
    from replicant.lattice import observables1d as O
    wrong = {"magnetization": lambda K, H: H.sinh() / (H.sinh() ** 2 + (-2 * K).exp()).sqrt(),
             "susceptibility": lambda K, H, b: b * (-K).exp() * H.cosh() / O._d(K, H).sqrt(),
             "correlation": lambda K, H, r: O._rho(K, H) ** r,          # amplitude 1, not 1 - m^2
             "correlation_length": lambda K, H: 2 * O._xi(K, H)}
    e = catalog()["ising1d"]
    for key, fn in wrong.items():
        def check():
            for N in (8, 200):
                o = O.ring_check(N, 1, "1/2", 1).observables[key]
                assert not (o.identity and o.within_bound), (key, N)
            r = _refusal(lambda: compose(e.spec, "exact").run(h="1/2"))
            assert r.reason == "no-solution:ring-limit" and key in r.detail, (key, r)
        _planted(O, key, fn, check)
    assert O.ring_check(8, 1, "1/2", 1).ok                   # restored


# ---- 2D: Onsager's energy and specific heat ------------------------------------------------------
def test_onsager_energy_and_specific_heat_two_certified_routes():
    from flint import ctx
    from replicant.lattice import observables1d as O, onsager_thermo as T, torus_thermo as TT
    before, fprec = mp.mp.dps, ctx.prec
    for beta in ("0.3", "0.6", "0.4407"):
        u, c = T.onsager_energy(beta), T.onsager_specific_heat(beta)
        ui, ci = T.onsager_energy_integral(beta), T.onsager_specific_heat_integral(beta)
        assert u.overlaps(ui) and c.overlaps(ci), (beta, u, ui, c, ci)
        assert max(u.rad(), c.rad(), ui.rad(), ci.rad()) < mp.mpf("1e-39"), beta
    O.ring_check(20, 1, "1/2", 2)
    TT.torus_thermo_kaufman(8, 8, "0.3")
    T.series_energy_specific_heat("0.2", 40)
    assert mp.mp.dps == before and ctx.prec == fprec          # precision only by workdps / ctx_guard


def test_onsager_critical_energy_amplitude_and_divergence():
    from replicant.lattice import BETA_C
    from replicant.lattice import onsager_thermo as T
    assert _near(T.onsager_energy(BETA_C), _bank("onsager_critical_energy"))
    with mp.workdps(40):                    # Eq. (116) at a rational 1e-30 from beta_c: -> -sqrt 2
        assert abs(T.onsager_energy("0.44068679350977151261630466249").mid() + mp.sqrt(2)) < 1e-27
    A = _bank("onsager_specific_heat_amplitude")
    assert _near(T.specific_heat_amplitude(), A)
    band = mp.mpf("1e-5")          # |slope - A| = 4.3e-7 at d = 1e-6, 1e-8 (O(d ln d)); excludes 2A
    for d1, d2 in (("1e-6", "1e-8"), ("-1e-6", "-1e-8")):
        s = T.specific_heat_log_slope(d1, d2)
        assert _near(s, A, band) and not _near(s, A, mp.mpf("1e-7")), (d1, s)
    assert _refusal(lambda: T.onsager_specific_heat(BETA_C)).reason == "precondition:off-critical"
    assert _refusal(lambda: T.specific_heat_log_slope("1e-6", "-1e-8")).reason == "precondition:d"
    # planted: a wrong amplitude (4/pi) fails the bank and the certified log-slope
    def wrong():
        a = T.specific_heat_amplitude()
        s = T.specific_heat_log_slope("1e-6", "1e-8")
        assert not _near(a, A) and abs(a.mid() - s.mid()) > 100 * band, (a, s)
    _planted(T, "amplitude_coefficient", 4, wrong)


def test_torus_derivatives_polynomial_kaufman_and_bulk_sequence():
    from replicant.lattice import onsager_thermo as O
    from replicant.lattice import torus_thermo as T
    for L, Ly in ((3, 3), (4, 3), (4, 4)):
        for beta in ("0.3", "beta_c"):
            pu, pc = T.torus_thermo_polynomial(L, Ly, beta)
            ku, kc = T.torus_thermo_kaufman(L, Ly, beta)
            assert pu.overlaps(ku) and pc.overlaps(kc), (L, Ly, beta)

    def gaps(beta):
        u, c = O.onsager_energy(beta), O.onsager_specific_heat(beta)
        out = []
        for L in (16, 32, 64, 128):
            tu, tc = T.torus_thermo_kaufman(L, L, beta)
            out.append(max(abs(tu.mid() - u.mid()), abs(tc.mid() - c.mid())))
        return out
    for beta in ("0.3", "0.6"):
        g = gaps(beta)                   # finite-size sequence: each gap below the previous^1.5
        assert all(b < a ** 1.5 for a, b in zip(g, g[1:])) and g[-1] < mp.mpf("1e-30"), (beta, g)
    # planted: the wrong derivative order (c_L from the first K-derivative) breaks the sequence
    _planted(T, "derivative_order", 1, lambda: _assert_far(gaps("0.3")))


def _assert_far(g):
    assert g[-1] > mp.mpf("1e-2"), g


def test_critical_torus_sequence_gives_the_amplitude():
    """At beta_c the L x L torus (Kaufman's Z, arb_series): (c_2L - c_L)/ln 2 - A = 1.9e-3, 9.6e-4
    at L = 64 -> 128 -> 256 (observed ~ 0.246/L, not proved); Richardson in 1/L leaves 2.7e-6 (band
    1e-5); u_L -> -sqrt 2 likewise (L (u_L + sqrt 2) = -0.6224).  A finite-size consistency check."""
    from replicant.lattice import torus_thermo as T
    A = mp.mpf(_bank("onsager_specific_heat_amplitude").value)
    th = {L: T.torus_thermo_kaufman(L, L, "beta_c") for L in (64, 128, 256)}
    c = {L: th[L][1].mid() for L in th}
    s1, s2 = ((c[128] - c[64]) / mp.log(2), (c[256] - c[128]) / mp.log(2))
    assert 0 < s2 - A < s1 - A < mp.mpf("2e-3"), (s1 - A, s2 - A)
    assert abs(2 * s2 - s1 - A) < mp.mpf("1e-5"), 2 * s2 - s1 - A
    lu = [L * (th[L][0].mid() + mp.sqrt(2)) for L in (64, 128, 256)]
    assert all(abs(x + mp.mpf("0.6224")) < mp.mpf("1e-4") for x in lu), lu


def test_series_route_agrees_where_it_converges():
    from replicant.lattice import onsager_thermo as O
    for beta, tol in (("0.15", "1e-45"), ("0.2", "1e-34")):     # v^120 tails: 1e-48, 1e-37
        u, c, last = O.series_energy_specific_heat(beta, 120)
        assert abs(u - O.onsager_energy(beta).mid()) < mp.mpf(tol), beta
        assert abs(c - O.onsager_specific_heat(beta).mid()) < mp.mpf(tol), beta
        assert last < mp.mpf(tol)
    assert _refusal(lambda: O.series_energy_specific_heat("0.5")).reason == \
        "precondition:series-convergence"


# ---- catalog -----------------------------------------------------------------------------------
def test_catalog_observables_certified_and_printed_enclosures():
    from flint import arb
    from replicant.catalog import catalog
    from replicant.cli import result_json
    from replicant.compose import compose
    from replicant.lattice import BETA_C, observables1d as O1, onsager_thermo as OT
    from replicant.lattice.ball import baller
    cat = catalog()
    runs = {("ising2d", (("beta", "0.6"),)): {"energy": lambda: OT.onsager_energy("0.6"),
                                            "specific_heat": lambda: OT.onsager_specific_heat("0.6")},
            ("ising2d", ()): {"energy": lambda: OT.onsager_energy(BETA_C),
                              "specific_heat_amplitude": OT.specific_heat_amplitude},
            ("ising1d", (("h", "1/2"),)): {
                "magnetization": lambda: O1.ising1d_magnetization(1, "1/2"),
                "correlation_length": lambda: O1.ising1d_correlation_length(1, "1/2")}}
    for (name, params), keys in runs.items():
        e = cat[name]
        res = compose(e.spec, "exact").run(**dict(params))
        js = result_json(name, e.spec, res)
        assert res.rigor == "certified"
        for k, ball in keys.items():
            v, err = js["values"][k], js["errors"][k]
            with baller().hygiene.ctx_guard(prec=300):
                b = ball().arb
                assert (arb(v) - arb(err)).upper() <= b.lower() and b.upper() <= (arb(v) + arb(err)).lower()
            assert res.finality[k].state == "exact-by-literature" and "Eq" in res.finality[k].cite
        if not params and name == "ising2d":
            assert "specific_heat" not in res.values and "logarithmic" in res.values[
                "specific_heat_divergence"]
            assert res.finality["specific_heat_amplitude_loglog"].state == "not-computed"


# ---- slow tier ---------------------------------------------------------------------------------
@slow
def test_slow_ising1d_rings_up_to_1000():
    from replicant.lattice import observables1d as O
    for N, beta, h, r in ((1000, 3, 0, 5), (1000, "1/2", "2", 3), (500, 2, "-1/4", 10), (64, 1, "1/2", 30)):
        c = O.ring_check(N, beta, h, r)
        assert c.ok, (N, beta, h, {k: (o.identity, o.within_bound) for k, o in c.observables.items()})


@slow
def test_slow_onsager_routes_on_a_grid():
    from replicant.lattice import onsager_thermo as T
    grid = ("0.1", "0.2", "0.35", "0.43", "0.4406867935", "0.4406867936", "0.45", "0.8", "1.5")
    for beta in grid:
        u, c = T.onsager_energy(beta), T.onsager_specific_heat(beta)
        assert u.overlaps(T.onsager_energy_integral(beta)), beta
        assert c.overlaps(T.onsager_specific_heat_integral(beta)), beta
    for beta in ("0.1", "0.25", "0.3", "0.35"):
        su, sc, last = T.series_energy_specific_heat(beta, 600)
        assert abs(su - T.onsager_energy(beta).mid()) < 10 * last + mp.mpf("1e-45"), beta
        assert abs(sc - T.onsager_specific_heat(beta).mid()) < 10 * last + mp.mpf("1e-45"), beta


@slow
def test_slow_polynomial_tori_equal_kaufman_derivatives():
    from replicant.lattice import torus_thermo as T
    for L, Ly in ((5, 5), (6, 6), (16, 3), (7, 7)):
        for beta in ("0.3", "beta_c", "0.6"):
            pu, pc = T.torus_thermo_polynomial(L, Ly, beta)
            ku, kc = T.torus_thermo_kaufman(L, Ly, beta)
            assert pu.overlaps(ku) and pc.overlaps(kc), (L, Ly, beta)
