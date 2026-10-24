import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.core import Refusal
from replicant.families.sk import (sk_1rsb, sk_at_temperature, sk_ground_energy,
                                   sk_replicon, sk_rs)
from replicant.refdata import check, load
from replicant.testing import slow

REF = load("sk")


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, r.reason
        return r
    raise AssertionError(f"expected Refusal {reason}")


def test_tc_is_one():
    hot, cold = sk_rs("0.9"), sk_rs("1.1")
    assert hot.q == 0 and hot.stable            # paramagnet above T_c
    assert cold.q > mp.mpf("0.01")              # q > 0 just below T_c
    tc = mp.mpf(REF["tc"].value)
    assert 1 / mp.mpf("1.1") < tc < 1 / mp.mpf("0.9")


def test_rs_zero_temperature_energy():
    e0 = sk_ground_energy("rs")
    with mp.workdps(30):
        assert abs(e0 + mp.sqrt(2 / mp.pi)) < mp.mpf("1e-20")
    assert check(REF["rs_ground_energy"], e0, label="RS e0").ok


def test_rs_unstable_below_tc_at_zero_field():
    assert sk_replicon("2.0") < 0                     # q > 0 RS branch, T = 0.5
    assert abs(sk_replicon("0.5") - mp.mpf("0.75")) < mp.mpf("1e-12")   # 1 - beta^2
    assert abs(sk_replicon("2.0", q=0) + 3) < mp.mpf("1e-12")  # paramagnet continued below T_c
    assert sk_rs("0.5").stable and not sk_rs("2.0").stable


def test_require_stable_refuses():
    r = _refuses(lambda: sk_rs("2.0", require_stable=True), "unstable:rs")
    assert "replicon" in r.detail
    assert sk_rs("0.5", require_stable=True).stable     # stable case passes through


def test_field_uses_at_line():
    h = mp.mpf("0.3")
    assert sk_rs("0.5", h).q > 0                      # no T_c at h != 0: q > 0 always
    assert sk_replicon(1 / mp.mpf("0.70"), h) > 0     # above the AT line: RS stable
    assert sk_replicon(1 / mp.mpf("0.60"), h) < 0     # below it: unstable
    # Leading-order AT line from the banked coefficient, h^2 = c (1 - T)^3, lands
    # in the same window up to its O(1 - T) corrections (1 - T ~ 0.4 here).
    c = mp.mpf(REF["at_small_field_coefficient"].value)
    t_asym = 1 - (h**2 / c) ** (mp.mpf(1) / 3)
    assert mp.mpf("0.55") < t_asym < mp.mpf("0.70")


@slow
def test_at_line_small_field_scaling():
    # h^2/(1-T_AT)^3 -> c as h -> 0; quadratic extrapolation in tau = 1 - T_AT.
    taus, ratios = [], []
    for h in ("0.01", "0.02", "0.04"):
        tau = 1 - sk_at_temperature(h)
        taus.append(tau)
        ratios.append(mp.mpf(h)**2 / tau**3)
    est =sum(r * mp.fprod((0 - t2) / (t1 - t2) for t2 in taus if t2 is not t1)
              for t1, r in zip(taus, ratios))
    assert check(REF["at_small_field_coefficient"], est, label="AT coeff").ok, est
    assert abs(sk_at_temperature("0.3") - mp.mpf("0.652")) < mp.mpf("0.01")


def test_1rsb_energy_matches_reference():
    e1 = sk_ground_energy("1rsb")
    assert check(REF["rsb1_ground_energy"], e1, label="1RSB e0").ok, e1


def test_1rsb_above_rs():
    # n -> 0: the replica free energy is MAXIMISED over the order parameters,
    # so 1RSB lies above RS, and its ground energy is less negative.
    s, r = sk_1rsb("2.0"), sk_rs("2.0")
    assert 0 <= s.q0 < s.q1 < 1 and 0 < s.m < 1
    assert s.f > r.f + mp.mpf("1e-4")
    assert sk_ground_energy("1rsb") > sk_ground_energy("rs") + mp.mpf("0.03")


def test_1rsb_energy_is_beta_derivative():
    # e = d(beta f)/d beta along the solution (total derivative), central difference.
    b, d = mp.mpf("2.0"), mp.mpf("1e-4")
    up, dn = sk_1rsb(b + d), sk_1rsb(b - d)
    fd = ((b + d) * up.f - (b - d) * dn.f) / (2 * d)
    assert abs(fd - sk_1rsb(b).e) < mp.mpf("1e-6")


def test_1rsb_in_field_and_rs_energy():
    s = sk_1rsb("2.0", "0.3")
    assert 0 < s.q0 < s.q1 < 1
    assert s.f > sk_rs("2.0", "0.3").f       # RS unstable there (below AT): 1RSB above
    r = sk_rs("1.5")                          # RS energy e = -beta (1 - q^2)/2 at h = 0
    assert abs(r.e + mp.mpf("1.5") * (1 - r.q**2) / 2) < mp.mpf("1e-12")


def test_quadratic_term_from_ansatz():
    import sympy as sp
    from replicant.families.sk import quadratic_term
    q, q0, q1, m = sp.symbols("q q0 q1 m")
    assert sp.expand(quadratic_term("rs") - (1 - q)**2) == 0
    assert sp.expand(quadratic_term("rsb") - (1 - 2*q1 + (1 - m)*q1**2 + m*q0**2)) == 0
    assert sp.expand(quadratic_term("rsb").subs({q0: q, q1: q}) - (1 - q)**2) == 0  # q0 = q1: RS


def test_fullrsb_is_banked_not_computed():
    assert "fullrsb_ground_energy" in REF
    _refuses(lambda: sk_ground_energy("fullrsb"), "not-implemented:fullrsb")
    _refuses(lambda: sk_ground_energy("2rsb"), "precondition:known-ansatz")


def test_outputs_say_physics_level():
    for sol in (sk_rs("2.0"), sk_1rsb("2.0")):
        assert "physics-level" in sol.caveat
