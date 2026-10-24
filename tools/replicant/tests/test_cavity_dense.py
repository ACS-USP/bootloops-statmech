"""Cavity route, RS, fully connected models: SK, Hopfield, spherical perceptron.

Conventions (stated once, the same as the replica families they are checked against):
  SK:        J_ij ~ N(0, 1/N), field h, beta = 1/T; q, f, e per spin.
  Hopfield:  AGS, H = -(1/2N) sum_ij sum_mu xi_i^mu xi_j^mu s_i s_j, p = alpha N, one
             condensed pattern; (m, q, r), r = (1/alpha) sum_{mu>1} N <m_mu>^2.
  Perceptron (spherical, Gardner): J.J = N, store J.xi^mu/sqrt(N) >= kappa,
             P = alpha N Gaussian patterns; S = (1/N) ln(fraction of the sphere).
The cavity code (replicant/cavity) never imports the replica families; the replica
functions are used HERE only, as the comparison target.
"""
import os
import sys
import time

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant import cavity as CV
from replicant.cavity import dense as D
from replicant.cavity import (cavity_gardner_capacity, cavity_gardner_entropy,
                              cavity_hopfield_rs, cavity_sk_rs, run_cavity)
from replicant.core import Refusal, State, run_pipeline
from replicant.families.common import CAVITY_CAVEAT
from replicant.families.hopfield import hopfield_rs
from replicant.families.perceptron import gardner_capacity, gardner_entropy
from replicant.families.sk import sk_rs
from replicant.refdata import check, load
from replicant.testing import slow

STEP_NAMES = ["remove_site", "cavity_field_law", "add_site", "self_consistency"]


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


def _sk_agree(c, r, tol="1e-12"):
    tol = mp.mpf(tol)
    return (abs(c.q - r.q) < tol and abs(c.f - r.f) < tol and abs(c.e - r.e) < tol
            and c.stable == r.stable and abs(c.replicon - r.replicon) < mp.mpf("1e-10"))


def _hop_agree(c, r, tol="1e-12"):
    tol = mp.mpf(tol)
    return (c.branch == r.branch and abs(c.m - r.m) < tol and abs(c.q - r.q) < tol
            and abs(c.r - r.r) < tol * (1 + abs(r.r)) and c.stable == r.stable
            and c.mode_stable == r.mode_stable
            and abs(c.mode_eig - r.mode_eig) < mp.mpf("1e-7")
            and abs(c.replicon - r.replicon) < mp.mpf("1e-8") * (1 + abs(r.replicon)))


# ---- SK -------------------------------------------------------------------------
def test_cavity_sk_equals_replica_sk():
    dps = mp.mp.dps
    for beta, h in (("1.5", "0"), ("0.9", "0.3")):
        c, r = cavity_sk_rs(beta, h), sk_rs(beta, h)
        assert _sk_agree(c, r), (beta, h, c, r)
        assert c.q > 0 and c.caveat == CAVITY_CAVEAT
    # paramagnet: h = 0 above T_c the cavity field has zero variance, q = 0 exactly
    c, r = cavity_sk_rs("0.9"), sk_rs("0.9")
    assert c.q == 0 == r.q and _sk_agree(c, r), (c, r)
    # same branch semantics below T_c: the q > 0 branch is returned (not q = 0),
    # labelled AT-unstable at h = 0, and require_stable refuses for the same reason
    c = cavity_sk_rs("1.5")
    assert c.q > mp.mpf("0.3") and not c.stable and c.replicon < 0
    rr = _refuses(lambda: cavity_sk_rs("1.5", require_stable=True), "unstable:rs")
    assert not sk_rs("1.5").stable      # sk_rs(require_stable=True) refuses exactly then
    assert "replicon" in rr.detail and "cavity" in rr.detail, rr.detail
    assert cavity_sk_rs("0.9", "0.3", require_stable=True).stable
    _refuses(lambda: cavity_sk_rs("-1"), "precondition:beta-positive")
    assert mp.mp.dps == dps


# ---- Hopfield -----------------------------------------------------------------------
def test_cavity_hopfield_equals_replica_hopfield():
    dps = mp.mp.dps
    for a, b in (("0.1", 20), ("0.05", 3)):          # points test_hopfield also solves
        for br in ("retrieval", "spinglass"):
            c, r = cavity_hopfield_rs(a, b, branch=br), hopfield_rs(a, b, branch=br)
            assert _hop_agree(c, r), (a, b, br, c, r)
            assert c.caveat == CAVITY_CAVEAT and "cavity" in c.note
    c, r = cavity_hopfield_rs("0.05", "0.8", branch="paramagnet"), \
        hopfield_rs("0.05", "0.8", branch="paramagnet")
    assert c.m == c.q == c.r == 0 and _hop_agree(c, r), (c, r)
    assert cavity_hopfield_rs("0.1", 20) == cavity_hopfield_rs("0.1", 20, branch="retrieval")
    assert mp.mp.dps == dps


def test_cavity_hopfield_refusals_say_why():
    # retrieval above alpha_c(T): alpha_c(T) named (the replica side's refusal at the
    # same point costs ~4 s and is compared in the slow test below)
    r = _refuses(lambda: cavity_hopfield_rs("0.15", 20), "no-solution:retrieval")
    assert "alpha_c(T)" in r.detail and "fold" in r.detail, r.detail
    # retrieval at T >= 1: no Mattis root
    r = _refuses(lambda: cavity_hopfield_rs("0.05", "0.9"), "no-solution:retrieval")
    _refuses(lambda: hopfield_rs("0.05", "0.9"), "no-solution:retrieval")
    assert "tanh" in r.detail
    # spin glass above T_g = 1 + sqrt(alpha)
    r = _refuses(lambda: cavity_hopfield_rs("0.05", "0.7", branch="spinglass"),
                 "no-solution:spinglass")
    _refuses(lambda: hopfield_rs("0.05", "0.7", branch="spinglass"), "no-solution:spinglass")
    assert "T_g" in r.detail
    # paramagnet below T = 1: the pattern-overlap reweighting is not normalisable
    r = _refuses(lambda: cavity_hopfield_rs("0.1", 20, branch="paramagnet"),
                 "no-solution:paramagnet")
    _refuses(lambda: hopfield_rs("0.1", 20, branch="paramagnet"), "no-solution:paramagnet")
    assert "normalis" in r.detail, r.detail
    _refuses(lambda: cavity_hopfield_rs("0.1", 5, branch="mixture"), "precondition:known-branch")
    _refuses(lambda: cavity_hopfield_rs("-0.1", 5), "precondition:alpha-positive")
    _refuses(lambda: cavity_hopfield_rs("0.1", 0), "precondition:beta-positive")


def test_cavity_hopfield_energy_is_beta_derivative_of_replica_f():
    """The cavity energy (separate formula, from the reweighted pattern overlaps)
    equals d(beta f_AGS)/d beta at the stationary point (replica side, numerically)."""
    from replicant.families.hopfield import hopfield_free_energy
    a, b = mp.mpf("0.05"), mp.mpf(3)
    c = cavity_hopfield_rs(a, b)
    e = D.cavity_hopfield_energy(a, b)
    with mp.workdps(20):
        d = mp.diff(lambda bb: bb * hopfield_free_energy(c.m, c.q, c.r, a, bb), b)
    assert abs(e - d) < mp.mpf("1e-9"), (e, d)


@slow
def test_cavity_hopfield_retrieval_boundary_equals_replica():
    """alpha_c(T) from the cavity curve alpha(m) vs the replica fold, and the same
    refusal just above it."""
    from replicant.cavity import dense_hopfield as DH
    from replicant.families.hopfield import hopfield_alpha_c
    b = 20
    ac_cav = DH._fold(float(b), DH._Law(1.0, float(b), True))[0]
    ac_rep = hopfield_alpha_c(b)
    assert abs(ac_cav - ac_rep) < 1e-9, (ac_cav, ac_rep)
    for a in (ac_rep * mp.mpf("1.001"), mp.mpf("0.15")):
        _refuses(lambda: cavity_hopfield_rs(a, b), "no-solution:retrieval")
        _refuses(lambda: hopfield_rs(a, b), "no-solution:retrieval")
    a = ac_rep * mp.mpf("0.999")             # just below: both find the retrieval point
    assert _hop_agree(cavity_hopfield_rs(a, b), hopfield_rs(a, b), tol="1e-10")


# ---- perceptron ---------------------------------------------------------------------
def test_cavity_perceptron_equals_replica_gardner_spherical():
    dps = mp.mp.dps
    s_c, s_r = cavity_gardner_entropy(1, "spherical"), gardner_entropy(1, "spherical")
    assert abs(s_c - s_r) < mp.mpf("1e-9"), (s_c, s_r)
    c0 = cavity_gardner_capacity("spherical")
    assert check(load("perceptron")["spherical_capacity_kappa0"], c0.alpha_c,
                 label="cavity alpha_c(0)").ok, c0
    assert abs(c0.alpha_c - 2) < mp.mpf("1e-12") and c0.ansatz == "rs"
    assert c0.caveat == CAVITY_CAVEAT
    k = mp.mpf("0.5")
    c, r = cavity_gardner_capacity("spherical", k), gardner_capacity("spherical", k)
    with mp.workdps(30):        # the spherical integral formula, closed form
        H = mp.erfc(-k / mp.sqrt(2)) / 2
        phi = mp.exp(-k * k / 2) / mp.sqrt(2 * mp.pi)
        formula = 1 / ((1 + k * k) * H + k * phi)
    assert abs(c.alpha_c - formula) < mp.mpf("1e-12") and abs(c.alpha_c - r.alpha_c) < 1e-12
    assert c.stable == r.stable and abs(c.replicon - r.replicon) < mp.mpf("1e-12"), (c, r)
    assert c0.stable == gardner_capacity("spherical").stable
    # above capacity: same semantics as the replica side (empty version space, -inf)
    assert cavity_gardner_entropy("2.5", "spherical") == mp.ninf \
        == gardner_entropy("2.5", "spherical")
    assert mp.mp.dps == dps


def test_cavity_perceptron_refusals_say_why():
    r = _refuses(lambda: cavity_gardner_entropy("0.5", "ising"), "not-implemented:cavity-ising")
    assert "spherical" in r.detail
    _refuses(lambda: cavity_gardner_capacity("ising"), "not-implemented:cavity-ising")
    _refuses(lambda: cavity_gardner_entropy(-1, "spherical"), "precondition:nonnegative-alpha")
    _refuses(lambda: cavity_gardner_capacity("potts"), "precondition:known-variables")
    # kappa < 0: the cavity stability (variance-reduction) quantity is negative near
    # capacity, the entropy is refused as RS-unstable, as on the replica side
    km = mp.mpf("-0.5")
    a = cavity_gardner_capacity("spherical", km).alpha_c * mp.mpf("0.98")
    r = _refuses(lambda: cavity_gardner_entropy(a, "spherical", km), "unstable:rs")
    assert "stability" in r.detail and not cavity_gardner_capacity("spherical", km).stable


@slow
def test_cavity_perceptron_refusal_matches_replica_below_kappa_zero():
    km = mp.mpf("-0.5")
    a = gardner_capacity("spherical", km).alpha_c * mp.mpf("0.98")
    _refuses(lambda: cavity_gardner_entropy(a, "spherical", km), "unstable:rs")
    _refuses(lambda: gardner_entropy(a, "spherical", km), "unstable:rs")
    c, r = cavity_gardner_capacity("spherical", km), gardner_capacity("spherical", km)
    assert c.stable == r.stable and abs(c.replicon - r.replicon) < 1e-12
    for k in ("-0.5", "0", "0.5", "1"):          # entropy below capacity, several kappa
        a = gardner_capacity("spherical", mp.mpf(k)).alpha_c * mp.mpf("0.6")
        assert abs(cavity_gardner_entropy(a, "spherical", k) - gardner_entropy(a, "spherical", k)) \
            < 1e-9, k


# ---- steps and trace -----------------------------------------------------------------
def test_cavity_trace_names_cavity_steps():
    for model, params in (("sk", dict(beta="1.5", h="0")),
                          ("hopfield", dict(alpha="0.05", beta=3, branch="retrieval")),
                          ("perceptron", dict(alpha=1, kappa=0))):
        state, trace = run_cavity(model, **params)
        js = trace.to_json()
        assert js["steps"] == STEP_NAMES, js["steps"]
        assert all("route=cavity" in e["summary"] and e["cite"] for e in js["entries"]), js
        assert state["route"] == "cavity" and "solution" in state
    assert [s.name for s in CV.cavity_steps()] == STEP_NAMES


def test_cavity_steps_refuse_out_of_order():
    steps = CV.cavity_steps()
    st = State(model=D.MODELS["sk"], params={"beta": mp.mpf(1), "h": mp.mpf(0)})
    _refuses(lambda: run_pipeline(steps[1:], st), "precondition:cavity_system")
    _refuses(lambda: run_pipeline(steps[:1] + steps[2:], st), "precondition:field_law")
    _refuses(lambda: run_pipeline(steps, State(params={})), "precondition:model")


def test_planted_error_makes_cross_check_fail():
    """A deliberately wrong cavity step must break the cavity-vs-replica agreement."""
    good = run_cavity("sk", beta="1.5", h="0")[0]["solution"]
    bad = run_cavity(D.SKCavity(reaction=False), beta="1.5", h="0")[0]["solution"]
    ref = sk_rs("1.5", "0")
    assert _sk_agree(good, ref) and not _sk_agree(bad, ref)
    assert abs(bad.e - ref.e) > mp.mpf("0.1"), bad      # missing Onsager term beta(1 - q)
    good = run_cavity("hopfield", alpha="0.05", beta=3, branch="retrieval")[0]["solution"]
    bad = run_cavity(D.HopfieldCavity(reaction=False), alpha="0.05", beta=3,
                     branch="retrieval")[0]["solution"]
    ref = hopfield_rs("0.05", 3)
    assert _hop_agree(good, ref) and not _hop_agree(bad, ref)
    assert abs(bad.r - ref.r) > mp.mpf("0.1"), bad      # noise alpha q instead of alpha r


def test_cavity_modules_do_not_import_replica_families():
    import inspect
    import re
    from replicant.cavity import _quad, dense_hopfield, dense_perceptron
    for mod in (D, dense_hopfield, dense_perceptron, _quad, CV):
        src = re.sub(r"cavity_\w+", "CAV", inspect.getsource(mod))   # our own names
        for bad in ("families.sk", "families.hopfield", "families.perceptron",
                    "families import sk", "families import hopfield", "families import perceptron",
                    "_perceptron_float", "sk_rs", "hopfield_rs", "gardner_", "solve import"):
            assert bad not in src, (mod.__name__, bad)
        for line in src.splitlines():          # imports: numpy, mpmath, core, common only
            if re.match(r"\s*(from|import)\s", line):
                assert re.match(r"\s*(import (math|mpmath as mp|numpy as np)|from (dataclasses|"
                                r"functools|\.\.core|\.\.families\.common|\.|\.dense|\._quad|"
                                r"\.dense_\w+) import)", line), (mod.__name__, line)


# ---- genuinely independent numerical check (finite N) ---------------------------------
@slow
def test_sk_amp_finite_n_overlap_matches_cavity_q():
    """AMP (TAP with the Onsager reaction term) on ONE sampled SK matrix, N = 2000,
    seed 0, beta = 1.5, h = 0, run for a fixed number of iterations from a random
    start independent of J.  Statistical/finite-size check only, not certified:
    the empirical q_t = (1/N) sum m_i^2 tracks the cavity q within an honest band
    (state-evolution fluctuations are O(N^-1/2) ~ 0.02; band 0.05)."""
    t0 = time.time()
    N, beta, iters = 2000, 1.5, 25
    rng = np.random.default_rng(0)
    A = rng.normal(size=(N, N)) / np.sqrt(N)
    J = (A + A.T) / np.sqrt(2)
    np.fill_diagonal(J, 0)
    q_cav = float(cavity_sk_rs("1.5").q)
    m_old = np.zeros(N)
    m = np.tanh(beta * np.sqrt(0.5) * rng.normal(size=N))     # q_0 > 0, independent of J
    for _ in range(iters):
        b = np.mean(1 - m * m)
        x = beta * (J @ m) - beta**2 * b * m_old               # Onsager reaction term
        m_old, m = m, np.tanh(x)
    q_emp = float(np.mean(m * m))
    assert abs(q_emp - q_cav) < 0.05, (q_emp, q_cav)
    # the same iteration WITHOUT the reaction term overshoots (naive mean field)
    m_old, m = np.zeros(N), np.tanh(beta * np.sqrt(0.5) * rng.normal(size=N))
    for _ in range(iters):
        m = np.tanh(beta * (J @ m))
    assert abs(float(np.mean(m * m)) - q_cav) > 0.1
    assert time.time() - t0 < 30
