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
    hot, cold = sk_rs("0.9"), sk_rs("1.0001")
    assert hot.q == 0 and hot.stable            # paramagnet above T_c
    # just below T_c, q > 0 and follows the leading-order law q = (beta^2-1)/(2 beta^4)
    b = mp.mpf("1.0001")
    assert abs(cold.q / ((b**2 - 1) / (2 * b**4)) - 1) < mp.mpf("0.01")
    tc = mp.mpf(REF["tc"].value)
    assert 1 / mp.mpf("1.0001") < tc < 1 / mp.mpf("0.9")


def test_rs_q_does_not_pass_paramagnet_off_below_tc():
    import replicant.families.sk as sk
    real = sk.fixed_point

    def failing(*a, **k):
        raise Refusal("no-solution", "forced")
    sk.fixed_point = failing
    try:
        assert sk._rs_q.__wrapped__("0.9", "0") == 0       # above T_c: q = 0 is the answer
        _refuses(lambda: sk._rs_q.__wrapped__("2.0", "0"), "no-solution")   # below: re-raise
    finally:
        sk.fixed_point = real


def test_rs_zero_temperature_energy():
    e0 = sk_ground_energy("rs")
    with mp.workdps(30):
        assert abs(e0 + mp.sqrt(2 / mp.pi)) < mp.mpf("1e-20")
    assert check(REF["rs_ground_energy"], e0, label="RS e0").ok


def test_rs_unstable_below_tc_at_zero_field():
    assert sk_replicon("2.0") < 0                     # q > 0 RS branch, T = 0.5
    assert abs(sk_replicon("0.9") - (1 - mp.mpf("0.9")**2)) < mp.mpf("1e-12")   # 1 - beta^2 at q = 0
    assert abs(sk_replicon("2.0", q=0) + 3) < mp.mpf("1e-12")  # paramagnet continued below T_c
    assert sk_rs("0.9").stable and not sk_rs("2.0").stable


def test_require_stable_refuses():
    r = _refuses(lambda: sk_rs("2.0", require_stable=True), "unstable:rs")
    assert "replicon" in r.detail
    assert sk_rs("0.9", require_stable=True).stable     # stable case passes through


def test_field_uses_at_line():
    h = mp.mpf("0.3")
    assert sk_rs("0.5", h).q > 0                      # no T_c at h != 0: q > 0 always
    assert sk_replicon(1 / mp.mpf("0.70"), h) > 0     # above the AT line: RS stable
    assert sk_replicon(1 / mp.mpf("0.60"), h) < 0     # below it: unstable
    # Small field: the sign change sits at 1 - T = (h^2/c_eff)^(1/3) with
    # c_eff = c (1 + O(1-T)); at h = 0.04 (1 - T ~ 0.1) the bracket below
    # admits c_eff in about (1.27, 1.75): it excludes c = 1 and c = 2, and
    # contains the banked c = 4/3 up to its ~13% first correction.
    c = mp.mpf(REF["at_small_field_coefficient"].value)
    hs = mp.mpf("0.04")
    lo, hi = mp.mpf("0.097"), mp.mpf("0.108")       # 1 - T window
    assert sk_replicon(1 / (1 - lo), hs) > 0 > sk_replicon(1 / (1 - hi), hs)
    assert lo < (hs**2 / c) ** (mp.mpf(1) / 3) < hi


@slow
def test_at_line_small_field_scaling():
    # h^2/(1-T_AT)^3 -> c as h -> 0; quadratic extrapolation in tau = 1 - T_AT.
    taus, ratios = [], []
    for h in ("0.01", "0.02", "0.04"):
        tau = 1 - sk_at_temperature(h)
        taus.append(tau)
        ratios.append(mp.mpf(h)**2 / tau**3)
    est = sum(r * mp.fprod((0 - t2) / (t1 - t2) for t2 in taus if t2 is not t1)
              for t1, r in zip(taus, ratios))
    assert check(REF["at_small_field_coefficient"], est, label="AT coeff").ok, est
    # regression pin of replicant's own output (T_AT(0.3) = 0.651854), not a reference
    assert abs(sk_at_temperature("0.3") - mp.mpf("0.651854")) < mp.mpf("1e-5")


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


def test_1rsb_collapse_says_why():
    for args in (("0.9",), ("1.2", "0.3")):           # paramagnet; h != 0 above the AT line
        r = _refuses(lambda: sk_1rsb(*args), "no-solution")
        assert "collapses to RS: RS is stable here" in r.detail, r.detail


def test_1rsb_point_is_local_maximum():
    from replicant.families.sk import _rsb_hessian_eigs
    for args in (("2.0",), ("2.0", "0.3")):
        s = sk_1rsb(*args)
        assert max(_rsb_hessian_eigs(float(args[0]), float((args + ("0",))[1]),
                                     float(s.q0), float(s.q1), float(s.m))) < 0


def test_ground_1rsb_picks_larger_stationary_point():
    from replicant.families.sk import _ground_1rsb_points
    pts = _ground_1rsb_points()            # [(q0, mu, E, d2E/dq0^2)], q0 = 0 point included
    assert any(p[0] == 0 for p in pts) and any(p[0] > 0.4 for p in pts)
    best = max(pts, key=lambda p: p[2])
    assert best[0] > 0.4 and best[3] < 0 and best[2] == sk_ground_energy("1rsb")


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


def test_cache_key_ignores_global_precision():
    from replicant.families.sk import _key
    plain = _key("0.9")
    with mp.workdps(40):             # an earlier test may leave mp.dps raised
        assert _key("0.9") == _key(0.9) == plain
