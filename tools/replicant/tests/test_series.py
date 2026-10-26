"""E8: exact high-temperature series of the 2D Ising (Onsager) free energy and its holonomic operator.

Convention (series/ising2d.py): square lattice, isotropic coupling, zero field, K = beta J,
v = tanh K, u = v^2.  g(v) = ln Z/N - ln 2 - 2 ln cosh K = sum_n a_n v^n (only even n; a_4 = 1),
b_n = a_{2n} the same series in u.  Route 1: the Onsager double integral (Duminil-Copin 2022,
arXiv:2208.00864, Eq. (3.6)) expanded exactly (trigonometric moments in exact arithmetic).  Route 2
(independent, no Onsager integral): the exact even-subgraph polynomial of the L x L torus by a
parity transfer matrix.  Recognition: tools/annihilator (pf_from_series, find_recurrence_fast,
nullspace_mod_fast, rec_to_theta) on the u-series, 2-prime confirmed, held-out terms verified here.
Bank: reference/series.json.
"""
import os
import sys
from fractions import Fraction
from math import comb

import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.catalog import catalog, check_ref
from replicant.core import Refusal
from replicant.refdata import check, load
from replicant.series import ising2d as I
from replicant.series import (NoOperatorWithinBounds, Operator, QuadNumber, critical_exponent,
                              elliptic_factor_check, ising2d_free_energy_series, local_exponents,
                              recognize_holonomic, singularities, torus_free_energy_series)
from replicant.testing import slow

BANK = load("series")
U_C = QuadNumber(3, -2, 2)                       # 3 - 2 sqrt 2 = (sqrt 2 - 1)^2


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


def _u(n):
    return ising2d_free_energy_series(n - 1, variable="u")


class _PlantedMoment:
    """Replace the trigonometric moment hook I._HOOKS['moment'] inside a block; restore it."""
    def __init__(self, fn):
        self.fn = fn

    def __enter__(self):
        self.saved = I._HOOKS["moment"]
        I._HOOKS["moment"] = self.fn
        return self

    def __exit__(self, *exc):
        I._HOOKS["moment"] = self.saved
        assert I._HOOKS["moment"] is I.onsager_moment
        return False


# ---- route 1: the Onsager integral, expanded exactly ------------------------------------
def test_series_from_the_onsager_integral_is_exact_and_matches_the_bank():
    before = mp.mp.dps
    a = ising2d_free_energy_series(20)
    assert all(isinstance(x, Fraction) for x in a) and len(a) == 21
    assert a[:4] == [0, 0, 0, 0] and all(a[n] == 0 for n in range(1, 21, 2))
    for n in (4, 6, 8, 10):                        # Finch's table (Harris-Meir) at d = 2
        assert check(BANK[f"a{n}"], str(mp.mpf(a[n].numerator) / a[n].denominator),
                     label=f"a{n}").ok, (n, a[n])
    assert ising2d_free_energy_series(10, variable="u") == a[::2]
    # the moments are integrated, not typed: <(cos t1 + cos t2)^2k> 4^k = C(2k, k)^2
    for k in range(12):
        assert I.onsager_moment(2 * k) * 4**k == comb(2 * k, k)**2
        assert I.onsager_moment(2 * k + 1) == 0
    assert mp.mp.dps == before


def test_series_refuses_other_variables_and_bad_orders():
    _refuses(lambda: ising2d_free_energy_series(10, variable="K"), "unsupported:variable")
    _refuses(lambda: ising2d_free_energy_series(-1), "precondition:order")
    _refuses(lambda: ising2d_free_energy_series(2.5), "precondition:order")
    _refuses(lambda: ising2d_free_energy_series(I.MAX_ORDER + 1), "precondition:order")


# ---- route 2: exact finite tori ---------------------------------------------------------
def test_series_matches_exact_finite_lattice():
    """The L x L torus agrees with the bulk exactly for k < L; at k = L the 2L straight
    non-contractible loops add 2L v^L / L^2 = (2/L) v^L (positive control: the first wrong
    coefficient differs, by exactly that)."""
    bulk = ising2d_free_energy_series(8)
    for L in (4, 5, 6, 8):
        tor = torus_free_energy_series(L, L)
        assert tor[:L] == bulk[:L], (L, tor, bulk)
        assert tor[L] != bulk[L] and tor[L] - bulk[L] == Fraction(2, L), (L, tor[L], bulk[L])
    _refuses(lambda: torus_free_energy_series(9, 9), "precondition:torus-size")
    _refuses(lambda: torus_free_energy_series(2, 2), "precondition:torus-size")


def test_torus_route_detects_a_planted_moment_error():
    def wrong(m):                                  # an extra 1/16 in the fourth moment
        return I.onsager_moment(m) + (Fraction(1, 16) if m == 4 else 0)
    tor = torus_free_energy_series(8, 8)
    with _PlantedMoment(wrong):
        bad = ising2d_free_energy_series(8)
    assert bad[:8] != tor[:8]
    assert ising2d_free_energy_series(8)[:8] == tor[:8]       # hook restored, cache not stale


# ---- recognition ------------------------------------------------------------------------
def test_onsager_free_energy_is_d_finite_order_three_with_an_order_two_elliptic_factor():
    """Fit on 70 coefficients of the u-series, verify on 31 held out (u^70 ... u^100)."""
    b = _u(101)
    rec = recognize_holonomic(b, max_order=4, max_degree=10, fit_terms=70)
    op = rec.operator
    assert (op.order, op.degree) == (3, 8), (op.order, op.degree)
    assert rec.verified_terms == 31 and rec.fit_terms == 70
    assert rec.certificate.found and len(rec.certificate.primes) == 2
    assert op.coefficient(0) == ()                 # L = M theta: constants are solutions
    assert op.rhs and any(op.rhs)                  # L g = P(u): the boundary terms
    assert all(x == 0 for x in op.apply(b)[len(op.rhs):])
    # minimal within bounds: no theta-order <= 2 operator of degree <= 10 on these terms
    r = _refuses(lambda: recognize_holonomic(b, max_order=2, max_degree=10, fit_terms=70),
                 "no-operator-within-bounds")
    assert r.certificate.max_order == 2 and r.certificate.equations > r.certificate.unknowns


def test_elliptic_factor_is_the_pulled_back_hypergeometric_operator():
    """theta g = R(u) + s(u) F(16 W(u)), F = 2F1(1/2, 1/2; 1; z) = (2/pi) K(k), k^2 = 16 W; the
    guessed order-2 factor M kills s F exactly (F'' eliminated by the hypergeometric ODE) and
    maps R to the boundary polynomial: a proof that L g = P, not a fit."""
    rec = recognize_holonomic(_u(101), max_order=4, max_degree=10, fit_terms=70)
    chk = elliptic_factor_check(rec.operator)
    assert chk.right_factor_theta and chk.kills_sF and chk.maps_R_to_rhs, chk
    # negative control: scale the theta^1 coefficient of M; the identity fails
    t = dict(rec.operator.terms)
    t[(1, 1)] = t[(1, 1)] * 2
    bad = elliptic_factor_check(Operator(t, rec.operator.rhs))
    assert not (bad.kills_sF and bad.maps_R_to_rhs)


def test_singularity_at_critical_point_and_log_exponent():
    op = recognize_holonomic(_u(101), max_order=4, max_degree=10, fit_terms=70).operator
    sing = singularities(op)
    pts = {s.point: s for s in sing}
    assert U_C in pts and QuadNumber(3, 2, 2) in pts and 1 in pts and -1 in pts, list(pts)
    assert not [s for s in sing if s.kind == "apparent"], sing
    loc = local_exponents(op, U_C)
    assert loc.regular and loc.integer_exponents == (0, 2, 2) and loc.log_exponents == (2,), loc
    rho = critical_exponent(op, U_C)
    assert rho == 2 and 2 - rho == 0               # (u - u_c)^2 ln|u - u_c|: alpha = 0
    assert check(BANK["alpha"], str(2 - rho), label="alpha").ok
    assert local_exponents(op, Fraction(1, 2)).integer_exponents == (0, 1, 2)     # ordinary point
    _refuses(lambda: critical_exponent(op, Fraction(1, 2)), "no-solution:analytic-point")
    with mp.workdps(30):
        assert check(BANK["v_c"], mp.nstr(mp.sqrt(U_C.mpf()), 30), label="v_c").ok
    # positive control: perturb the theta^2 coefficient; u_c stays singular, the exponents move
    t = {k: (c * Fraction(11, 10) if k[1] == 2 else c) for k, c in op.terms.items()}
    loc2 = local_exponents(Operator(t, op.rhs), U_C)
    assert loc2.integer_exponents != (0, 2, 2) and critical_exponent(Operator(t, op.rhs), U_C) != 2


def test_coefficient_asymptotics_carry_the_log_exponent():
    """Independent of the operator: n (1 - u_c b_n / b_{n-1}) -> 3 for (u - u_c)^2 ln (b_n ~
    u_c^-n n^-3); for theta g = sum n b_n u^n -> 2.  Richardson in 1/n, band 0.02."""
    b = _u(101)
    with mp.workdps(30):
        uc = U_C.mpf()

        def est(seq):
            x = [n * (1 - uc * mp.mpf(seq[n].numerator) / seq[n].denominator
                      / (mp.mpf(seq[n - 1].numerator) / seq[n - 1].denominator))
                 for n in (98, 99, 100)]
            return (100**2 * x[2] - 2 * 99**2 * x[1] + 98**2 * x[0]) / 2   # kills d/n + e/n^2
        assert abs(est(b) - 3) < mp.mpf("0.02"), est(b)
        h = [n * c for n, c in enumerate(b)]
        assert abs(est(h) - 2) < mp.mpf("0.02"), est(h)


def test_recognition_refuses_when_series_is_not_holonomic_at_bounds():
    b = _u(101)
    bad = list(b)
    bad[30] += 1                                   # one corrupted coefficient in the fit window
    r = _refuses(lambda: recognize_holonomic(bad, max_order=4, max_degree=10, fit_terms=70),
                 "no-operator-within-bounds")
    c = r.certificate
    assert (c.max_order, c.max_degree, c.n_terms) == (4, 10, 70) and not c.found
    assert c.unknowns == 55 and c.equations == 60 and len(c.primes) == 2, c
    late = list(b)
    late[90] += 1                                  # corrupted beyond the fit window
    _refuses(lambda: recognize_holonomic(late, max_order=4, max_degree=10, fit_terms=70),
             "no-solution:held-out")
    # integer partitions (natural boundary at |x| = 1: not D-finite, Flajolet-Gerhold-Salvy)
    p = [1] + [0] * 69
    for k in range(1, 70):
        for n in range(k, 70):
            p[n] += p[n - k]
    _refuses(lambda: recognize_holonomic(p, max_order=4, max_degree=10),
             "no-operator-within-bounds")
    _refuses(lambda: recognize_holonomic(b[:40], max_order=4, max_degree=10),
             "precondition:series-length")
    _refuses(lambda: recognize_holonomic(b, max_order=0, max_degree=10), "precondition:bounds")


# ---- the catalog entry ------------------------------------------------------------------
def test_series_entry_runs_and_labels_finality():
    e = catalog()["series_ising2d"]
    assert e.routes == {"series"}
    v = e.run()
    assert v["operator_order"] == 3 and v["operator_degree"] == 8 and v["elliptic_factor_order"] == 2
    assert v["heldout_verified"] >= 30 and v["elliptic_identity"] is True
    assert v["torus_agrees_through"] == 7
    fin = e.finality_of(v)
    assert all(st.final for st in fin.values()), {k: s.state for k, s in fin.items() if not s.final}
    assert fin["alpha"].state == "exact-by-literature" and fin["operator_order"].state == "stable"
    for ref in e.refs:
        assert check_ref(e, ref).ok, ref.key
    _refuses(lambda: e.run(max_order=2), "no-operator-within-bounds")
    _refuses(lambda: e.run(order=100), "precondition:series-length")


@slow
def test_recognition_on_a_longer_series_holds_out_eighty_terms():
    b = _u(161)
    rec = recognize_holonomic(b, max_order=4, max_degree=10, fit_terms=80)
    assert (rec.operator.order, rec.operator.degree) == (3, 8) and rec.verified_terms == 81


@slow
def test_susceptibility_has_no_operator_within_bounds():
    """Stretch: the 2D Ising high-temperature susceptibility (OEIS A002906, from the Guttmann-
    Jensen series; natural boundary |v| = 1 per Nickel 1999 / Orrick et al. 2001).  Claim: no
    operator within the bounds, with the certificate; never a proof of non-D-finiteness."""
    from replicant.series import susceptibility_series
    chi = susceptibility_series(400)
    assert chi[:6] == [1, 4, 12, 36, 100, 276]
    r = _refuses(lambda: recognize_holonomic(chi, max_order=6, max_degree=40),
                 "no-operator-within-bounds")
    assert r.certificate.unknowns == 41 * 7 and r.certificate.equations == 400 - 40
