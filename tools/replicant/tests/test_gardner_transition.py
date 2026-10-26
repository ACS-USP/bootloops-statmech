"""E4: the Gardner transition (stability of the 1RSB solution toward further RSB).

Conventions: couplings of variance p!/(2 N^(p-1)) (J = 1), zero field, q0 = 0.  The 1RSB
replicon of the innermost block, normalised to be dimensionless,
    Lambda_R = 1 - Sigma(q1) chi_R,   Sigma(q) = beta^2 p (p-1) q^(p-2) / 2,
with chi_R = (1 - q1)^2 for spherical spins (Crisanti & Leuzzi 2006, Eq. (31), times
(1 - q1)^2) and chi_R = E_m[sech^4] for Ising spins (Montanari & Ricci-Tersenghi 2003,
Eq. (2.6)).  The formula is checked against an independent rule: the second derivative
of the explicit integer-n replica functional along a replicon vector (finite differences,
exact spin enumeration for Ising), at integer block size m.

Bank: reference/ising_pspin.json (Rizzo 2013, Yeo & Moore 2020).  The pure spherical
p-spin has no Gardner transition; its replicon is computed positive on the whole
static line (scan below).
"""
import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.catalog import catalog, check_ref
from replicant.compose import compose
from replicant.core import Refusal
from replicant.families import gardner as G
from replicant.families import ising_pspin as I
from replicant.families.pspin import pspin_critical_temperature, pspin_spherical_1rsb
from replicant.refdata import check, load
from replicant.testing import slow

BANK = load("ising_pspin")


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


# ---- banked transition temperatures of the Ising p-spin ------------------------------------
def test_gardner_temperature_matches_bank():
    dps = mp.mp.dps
    tg = G.pspin_gardner_temperature(3, variables="ising")
    assert check(BANK["tg_p3"], mp.nstr(tg, 12), label="tg").ok, tg
    ts = I.ising_pspin_critical_temperature(3)
    td = I.ising_pspin_critical_temperature(3, "dynamic")
    assert check(BANK["ts_p3"], mp.nstr(ts, 12), label="ts").ok, ts
    assert check(BANK["td_p3"], mp.nstr(td, 12), label="td").ok, td
    # positive control: the comparison is not vacuous
    assert not check(BANK["tg_p3"], mp.nstr(tg * mp.mpf("1.001"), 12), label="tg").ok
    assert mp.mp.dps == dps


def test_spherical_pspin_has_no_gardner_temperature():
    r = _refuses(lambda: G.pspin_gardner_temperature(3), "no-transition:gardner")
    assert "replicon" in r.detail and "T_s" in r.detail and "scan" in r.detail, r.detail
    _refuses(lambda: G.pspin_gardner_temperature(3, variables="potts"),
             "precondition:known-variables")


# ---- the replicon sign on a temperature grid (no root-finding in T) -----------------------
def test_replicon_sign_scan():
    tg = mp.mpf(BANK["tg_p3"].value)            # a target, never an input of the replicon
    for T in ("0.16", "0.23", "0.25", "0.5"):
        lam = G.onersb_replicon("ising_pspin", p=3, beta=1 / mp.mpf(T))
        assert (lam > 0) == (mp.mpf(T) > tg), (T, lam)
        assert abs(lam) > mp.mpf("1e-3"), (T, lam)
    # the spherical pure p-spin: positive on the whole static line, 0 < T < T_s
    ts = pspin_critical_temperature(3)
    for T in ("0.05", "0.3", "0.58"):
        assert mp.mpf(T) < ts
        lam = G.onersb_replicon("pspin", p=3, beta=1 / mp.mpf(T))
        assert lam > mp.mpf("0.2"), (T, lam)


def test_spherical_replicon_is_the_closed_form_on_the_static_line():
    """CONSISTENCY check, not independent evidence (it uses _y from the same pspin.py): on the
    static line beta^2 q^(p-2)(1-q)^2 = 2y/p, so Lambda_R = 1 - (p-1) y for all T < T_s."""
    from replicant.families.pspin import _y
    for p in (3, 4):
        want = 1 - (p - 1) * _y(p)
        for T in ("0.1", "0.3"):
            got = G.onersb_replicon("pspin", p=p, beta=1 / mp.mpf(T))
            assert abs(got - want) < mp.mpf("1e-14"), (p, T, got, want)


# ---- the formula against the explicit integer-n functional --------------------------------
def test_replicon_formula_matches_explicit_n_functional():
    """Points where Sigma chi_R is of order one (an Ising point with q near 1 has
    chi_R ~ 1e-10 and would agree trivially)."""
    pts = {"spherical": (("2", "0.7", 8, 4), ("3", "0.85", 10, 5), ("1.5", "0.5", 8, 4)),
           "ising": (("1", "0.3", 8, 4), ("1.2", "0.5", 5, 5), ("0.8", "0.6", 4, 4))}
    for variables, rows in pts.items():
        for beta, q, n, m in rows:
            got = G.explicit_n_replicon(variables, 3, beta, q, n, m)
            want = G.replicon_formula(variables, 3, beta, q, m)
            assert abs(1 - want) > mp.mpf("0.04"), (variables, beta, q, want)
            assert abs(got - want) < mp.mpf("1e-12"), (variables, beta, q, n, m, got, want)


def test_planted_error_in_replicon_formula_is_caught():
    """Plant errors in the registered susceptibility rules (G.CHI_R): sech^2 for sech^4
    (Ising) and (1 - q) for (1 - q)^2 (spherical) are caught by the explicit-n comparison;
    with sech^2 the static-line replicon never changes sign (T_G refused); a 5% error in the
    Ising rule moves T_G outside the bank."""
    orig = dict(G.CHI_R)
    sech4 = orig["ising"]
    try:
        G.CHI_R["ising"] = lambda p, beta, q, m: G.ising_block_average(
            beta, p, q, m, lambda t: 1 - t * t)
        G.CHI_R["spherical"] = lambda p, beta, q, m: 1 - mp.mpf(q)
        for v, b, q in (("spherical", "2", "0.7"), ("ising", "1", "0.3")):
            got = G.explicit_n_replicon(v, 3, b, q, 8, 4)
            assert abs(got - G.replicon_formula(v, 3, b, q, 4)) > mp.mpf("1e-3"), v
        _refuses(lambda: G.pspin_gardner_temperature(3, variables="ising"),
                 "no-transition:gardner")
        G.CHI_R["ising"] = lambda p, beta, q, m: mp.mpf("1.05") * sech4(p, beta, q, m)
        tg = G.pspin_gardner_temperature(3, variables="ising")
        assert not check(BANK["tg_p3"], mp.nstr(tg, 12), label="tg").ok, tg
        assert abs(G.explicit_n_replicon("ising", 3, "1", "0.3", 8, 4)
                   - G.replicon_formula("ising", 3, "1", "0.3", 4)) > mp.mpf("1e-3")
    finally:
        G.CHI_R.clear()
        G.CHI_R.update(orig)
    assert all(G.CHI_R[k] is orig[k] for k in orig) and set(G.CHI_R) == set(orig)
    tg = G.pspin_gardner_temperature(3, variables="ising")
    assert check(BANK["tg_p3"], mp.nstr(tg, 12), label="tg").ok


def test_spherical_gardner_answer_follows_its_own_scan():
    """Fix round 1 (Important 1): the spherical no-transition reason is given only when every
    scanned replicon is positive.  Planted rules (registry restored, identity checked):
    5 (1 - q)^2 makes the replicon negative everywhere (refused unstable:1rsb, not the
    no-transition reason); (1 - q)^2 beta/2 makes it 1 - (p-1) y beta/2, which crosses zero at
    T = (p-1) y / 2 inside (0, T_s): the root must be found and returned."""
    from replicant.families.pspin import _y
    orig = dict(G.CHI_R)
    sph = orig["spherical"]
    try:
        G.CHI_R["spherical"] = lambda p, beta, q, m: 5 * sph(p, beta, q, m)
        r = _refuses(lambda: G.pspin_gardner_temperature(3), "unstable:1rsb")
        assert "positive for all" not in r.detail and "negative" in r.detail, r.detail
        G.CHI_R["spherical"] = lambda p, beta, q, m: sph(p, beta, q, m) * mp.mpf(beta) / 2
        tg = G.pspin_gardner_temperature(3)
        assert abs(tg - 2 * _y(3) / 2) < mp.mpf("1e-9"), tg
    finally:
        G.CHI_R.clear()
        G.CHI_R.update(orig)
    assert all(G.CHI_R[k] is orig[k] for k in orig) and set(G.CHI_R) == set(orig)
    _refuses(lambda: G.pspin_gardner_temperature(3), "no-transition:gardner")


def test_ising_tg_cache_holds_the_rule_object():
    """Fix round 1 (minor 2): the T_G cache is keyed on the rule object itself (kept alive by
    the key), never on id(rule), which CPython may reuse after a planted lambda is freed."""
    G.pspin_gardner_temperature(3, variables="ising")
    assert any(k[1] is G.CHI_R["ising"] for k in G._TG_CACHE), list(G._TG_CACHE)
    assert all(callable(k[1]) for k in G._TG_CACHE)


# ---- the Ising p-spin 1RSB solver --------------------------------------------------------
def test_ising_pspin_1rsb_solution_and_its_checks():
    s = I.ising_pspin_1rsb(3, 1 / mp.mpf("0.4"))
    assert 0 < s.m < 1 and s.q0 == 0 and 0.5 < s.q1 < 1, s
    assert len(s.q_roots) == 2 and s.q1 == max(s.q_roots), s.q_roots   # larger-q branch
    assert s.slopes[0] > 1 > s.slopes[1] > 0, s.slopes       # the smaller root is unstable
    # e = d(beta f)/d beta, by a finite difference of the reported f (independent of e);
    # m held at the static value (beta f is stationary in m: an O(h^2) effect)
    b, h = 1 / mp.mpf("0.4"), mp.mpf("1e-4")
    bf = lambda x: x * I.ising_pspin_1rsb(3, x, m=s.m).f
    assert abs((bf(b + h) - bf(b - h)) / (2 * h) - s.e) < 1e-6, s.e
    # the residuals at the solution, re-evaluated by adaptive mpmath quadrature
    assert max(abs(x) for x in I.audit_residuals(3, b, s.q1, s.m)) < 1e-10
    # above T_s: the paramagnet, refused with a reason
    r = _refuses(lambda: I.ising_pspin_1rsb(3, 1 / mp.mpf("0.66")), "no-solution")
    assert "paramagnet" in r.detail
    _refuses(lambda: I.ising_pspin_1rsb(2, 3), "precondition:p-at-least-3")
    _refuses(lambda: I.ising_pspin_1rsb(3, -1), "precondition:positive-beta")


def test_m1_replicon_vanishes_at_the_dynamic_fold():
    """Rizzo 2013, Sec. V: the marginal (replicon zero) and spinodal lines cross at m = 1,
    T = T_d.  T_d is located here as the fold of the m = 1 q equation (no replicon used):
    the m = 1 replicon must go to 0 there (like sqrt(T_d - T)) and be O(1) away from it."""
    td = I.ising_pspin_critical_temperature(3, "dynamic")
    lam = [G.onersb_replicon("ising_pspin", p=3, beta=1 / (td - d), m=1)
           for d in (mp.mpf("1e-2"), mp.mpf("1e-4"), mp.mpf("1e-6"))]
    assert lam[0] > lam[1] > lam[2] > 0 and lam[2] < 3e-3 and lam[0] > 0.1, lam
    assert abs(lam[1] / lam[2] - 10) < 0.5, lam                  # square-root approach


def test_q_equation_roots_agree_with_a_grid_scan():
    """The root finder assumes one hump of E_m[tanh^2] - q; a dense sign scan (no
    assumption) finds the same nonzero roots at a few (beta, m)."""
    import numpy as np
    for b, m in ((2.5, 0.55), (1 / 0.6, 0.9), (4.0, 0.3), (1.5, 1.0)):
        grid = np.concatenate([np.linspace(1e-4, 0.99, 400), 1 - np.geomspace(1e-2, 1e-13, 60)])
        v = [I._qres(q, 3, b, m) for q in grid]
        scan = [grid[i] for i in range(len(grid) - 1) if v[i] * v[i + 1] < 0]
        roots = I._q_roots(3, b, m)
        assert len(scan) == len(roots) and all(abs(a - c) < 3e-3 for a, c in zip(scan, roots)), \
            (b, m, scan, roots)


def test_ising_pspin_m1_is_the_paramagnet_free_energy():
    """At m = 1 the 1RSB functional equals the paramagnet -beta f = beta^2/4 + ln 2 for any
    q (an identity of the functional, not of the solver)."""
    for b, q in (("1.6", "0.3"), ("2.5", "0.9")):
        got = I.minus_beta_f(3, b, q, 1)
        assert abs(got - (mp.mpf(b)**2 / 4 + mp.log(2))) < 1e-12, got


# ---- catalog finality --------------------------------------------------------------------
def test_pspin_finality_is_conditional():
    cat = catalog()
    ps = compose(cat["pspin"].spec).run(beta=2)
    for k in ("q0", "q1", "m", "f", "e", "replicon_1rsb"):
        st = ps.finality[k]
        assert st.state == "stable" and "replicon" in st.note, (k, st)
    assert ps.final and ps.values["replicon_1rsb"] > 0
    head = compose(cat["pspin"].spec).run()
    for k in ("T_s", "q_s"):
        assert head.finality[k].state == "stable" and "replicon" in head.finality[k].note
    assert head.finality["T_d"].state == "exact-by-literature"
    ising = cat["ising_pspin"]
    hi = compose(ising.spec).run(beta=1 / mp.mpf("0.3"))
    assert hi.final and hi.finality["q1"].state == "stable", hi.finality["q1"]
    lo = compose(ising.spec).run(beta=5)                           # T = 0.2 < T_G
    assert lo.finality["q1"].state == "unstable" and not lo.final, lo.finality["q1"]
    r = _refuses(lambda: compose(ising.spec).run(require_final=True, beta=5), "unstable:1rsb")
    hd = compose(ising.spec).run()
    assert hd.finality["T_G"].state == "stable" and hd.finality["T_d"].state == "not-computed"
    for ref in ising.refs:
        assert check_ref(ising, ref).ok, ref.key


@slow
def test_pspin_finality_documents_are_updated():
    """GUIDE and LaTeX carry the computed finality (string checks; slow tier for budget)."""
    cat = catalog()
    guide = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "GUIDE.md")).read()
    assert "no Gardner transition" in guide and "`ising_pspin`" in guide
    assert "| exact-by-literature (1RSB) |\n| `perceptron`" not in guide
    titles = [b.title for b in cat["pspin"].derivation()]
    assert any("replicon" in t.lower() for t in titles), titles
    assert any("replicon" in b.title.lower() for b in cat["ising_pspin"].derivation())


# ---- the Ising perceptron ---------------------------------------------------------------
def test_ising_perceptron_1rsb_stability():
    r = _refuses(lambda: G.ising_perceptron_1rsb_replicon("0.8"),
                 "not-implemented:frozen-1rsb-replicon")
    assert "q1 = 1" in r.detail and "not computed" in r.detail, r.detail
    _refuses(lambda: G.ising_perceptron_1rsb_replicon("0.9"), "precondition:alpha-range")


def test_onersb_replicon_refuses_unknown_inputs():
    _refuses(lambda: G.onersb_replicon("sk", beta=2), "missing-step:family:sk")
    _refuses(lambda: G.onersb_replicon("pspin", p=3, beta=2, kapa=1),
             "precondition:known-parameter")
    _refuses(lambda: G.onersb_replicon("pspin", p=3), "precondition:state-point")


@slow
def test_gardner_temperature_p4_and_scan_density():
    """p = 4: a Gardner temperature exists (MRT 2003: 'qualitatively valid for any finite
    p > 2'); not banked (no fetched source tabulates it).  A dense scan at p = 3 changes
    sign once, at T_G."""
    tg4 = G.pspin_gardner_temperature(4, variables="ising")
    assert 0 < tg4 < I.ising_pspin_critical_temperature(4)
    tg = G.pspin_gardner_temperature(3, variables="ising")
    Ts = [mp.mpf(k) / 100 for k in range(12, 65)]
    signs = [G.onersb_replicon("ising_pspin", p=3, beta=1 / T) > 0 for T in Ts]
    flips = [Ts[i] for i in range(len(Ts) - 1) if signs[i] != signs[i + 1]]
    assert len(flips) == 1 and flips[0] < tg < flips[0] + mp.mpf("0.01"), flips


def test_spherical_no_transition_detail_is_tied_to_the_scan():
    """T15 (E4 deferred minors): the no-transition detail opens with the scanned points and
    their resolution limit, and says "equal ... T-independent" only when the scanned values
    are equal; a planted T-dependent positive rule (chi_R times T: replicon 1 - (p-1) y T > 0)
    still gives no-transition but the detail reports the spread instead."""
    r = _refuses(lambda: G.pspin_gardner_temperature(3), "no-transition:gardner")
    assert r.detail.startswith("1RSB replicon positive") and "5 scanned" in r.detail, r.detail
    assert "would be missed" in r.detail and "T-independent" in r.detail, r.detail
    orig = dict(G.CHI_R)
    sph = orig["spherical"]
    try:
        G.CHI_R["spherical"] = lambda p, beta, q, m: sph(p, beta, q, m) / mp.mpf(beta)
        r = _refuses(lambda: G.pspin_gardner_temperature(3), "no-transition:gardner")
        assert "differ by up to" in r.detail and "T-independent" not in r.detail, r.detail
    finally:
        G.CHI_R.clear()
        G.CHI_R.update(orig)
    assert all(G.CHI_R[k] is orig[k] for k in orig) and set(G.CHI_R) == set(orig)


def test_ising_tg_cache_does_not_reuse_a_stale_value():
    """T15 (E4 deferred minor): with the registered rule's T_G cached first, a planted rule
    gets its own search (a different T_G), and the restored rule gets the cached value back."""
    base = G.pspin_gardner_temperature(3, variables="ising")
    orig = dict(G.CHI_R)
    sech4 = orig["ising"]
    try:
        G.CHI_R["ising"] = lambda p, beta, q, m: mp.mpf("1.05") * sech4(p, beta, q, m)
        planted = G.pspin_gardner_temperature(3, variables="ising")
    finally:
        G.CHI_R.clear()
        G.CHI_R.update(orig)
    assert all(G.CHI_R[k] is orig[k] for k in orig) and set(G.CHI_R) == set(orig)
    assert abs(planted - base) > mp.mpf("1e-4"), (planted, base)
    assert G.pspin_gardner_temperature(3, variables="ising") == base
