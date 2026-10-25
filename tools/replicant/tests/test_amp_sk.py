"""amp/: AMP, state evolution and TAP for SK (E3a), an independent finite-N route.

Conventions (amp/sk.py): H = -sum_{i<j} J_ij s_i s_j - h sum_i s_i, J symmetric, zero
diagonal, off-diagonal variance 1/N (Gaussian or +-1/sqrt(N)); phi = -beta f per spin.
families/sk.py (`sk_rs`, `sk_replicon`) and solve_derived.py appear HERE only, as
comparison targets: nothing from them feeds amp/.  Finite-N checks are statistical: a
99% Student-t band over the named seeds 0-5 (df = 5); the half-widths are printed.
"""
import inspect
import os
import re
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant import amp as A
from replicant.amp import sk as AS
from replicant.amp import state_evolution as SE
from replicant.amp import tap as AT
from replicant.core import Refusal
from replicant.families.sk import sk_replicon, sk_rs
from replicant.testing import slow

SEEDS = range(6)
N = 1500


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        assert r.detail, r.reason
        return r
    raise AssertionError(f"expected Refusal {reason}")


def _inside(band, x):
    return abs(x - band.mean) <= band.half_width


# ---- state evolution (default tier) ---------------------------------------------------
def test_state_evolution_fixed_point_equals_rs_q():
    from replicant.agree import default_spec
    from replicant.solve_derived import solve_derived_rs
    dps = mp.mp.dps
    se = A.sk_state_evolution("1.5")
    assert se.converged and se.q > mp.mpf("0.3"), se
    ref = sk_rs("1.5")                                    # comparison target only
    assert abs(se.q - ref.q) < mp.mpf("1e-10"), (se.q, ref.q)
    der = solve_derived_rs(default_spec("sk"), beta="1.5", h=0, branch="nontrivial", dps=12)
    assert abs(se.q - der.values["q"]) < mp.mpf("1e-10"), (se.q, der.values)
    # h = 0, beta > 1: q = 0 is also a fixed point (SE-unstable, slope beta^2); both listed
    assert [float(p.q) for p in se.fixed_points][0] == 0.0 and len(se.fixed_points) == 2
    assert abs(se.fixed_points[0].slope - mp.mpf("2.25")) < mp.mpf("1e-6")
    assert se.fixed_points[1].slope < 1
    # the trajectory is the recursion itself, monotone from q0 = 1 down to q*
    tr = se.trajectory
    assert tr[0] == 1 and all(a >= b for a, b in zip(tr, tr[1:])), tr[:5]
    assert abs(tr[-1] - float(se.q)) < 1e-9
    # with a field: one fixed point, equal to the RS q
    se1 = A.sk_state_evolution("1.5", "0.8")
    assert len(se1.fixed_points) == 1 and abs(se1.q - sk_rs("1.5", "0.8").q) < 1e-10
    assert mp.mp.dps == dps


def test_two_time_slope_is_the_at_eigenvalue():
    """Bolthausen's psi'(q) (the correlation map between successive iterates), computed
    by differencing the bivariate SE map, equals 1 - lambda_AT (sk_replicon: target)."""
    for b, h in (("1.5", "0"), ("1.5", "0.8")):
        se = A.sk_state_evolution(b, h)
        lam = sk_replicon(b, h)
        assert abs((1 - se.corr_slope) - lam) < 1e-5, (b, h, se.corr_slope, lam)
    assert A.sk_state_evolution("1.5", "0.8").amp_converges is True
    assert A.sk_state_evolution("1.5").amp_converges is False


def test_planted_error_in_state_evolution_is_caught():
    good = SE._field
    SE._field = lambda b, h, q, z: b * (h + (2 * q)**0.5 * z)    # doubled field variance
    try:
        bad = A.sk_state_evolution("1.5")
    finally:
        SE._field = good
    assert SE._field is good
    assert abs(bad.q - sk_rs("1.5").q) > mp.mpf("0.05"), bad.q


def test_state_evolution_paramagnet():
    se = A.sk_state_evolution("0.7")
    assert se.q == 0 and len(se.fixed_points) == 1 and se.converged
    assert abs(se.fixed_points[0].slope - mp.mpf("0.49")) < 1e-6


def test_two_time_state_evolution_predicts_amp_increments():
    tt = A.sk_two_time_state_evolution("1.5", "0.8", iters=12)
    inc = tt.increment
    assert inc[1] > 0.3 and inc[-1] < 1e-2 and all(a > b for a, b in zip(inc[1:], inc[2:]))
    tt0 = A.sk_two_time_state_evolution("1.5", "0", iters=12)
    # h = 0, zero-mean start: successive iterates decorrelate (psi(0) = 0), increment -> 2 q*
    assert abs(tt0.increment[-1] - 2 * float(sk_rs("1.5").q)) < 1e-3, tt0.increment


# ---- refusals (default tier) ----------------------------------------------------------
def test_refusals_say_why():
    J = AS.sk_matrix(8, 0, "pm_one")
    _refuses(lambda: A.sk_amp(J + np.triu(np.ones((8, 8)), 1) / 8, 1, iters=3, seed=0),
             "precondition:symmetric-matrix")
    D = J.copy()
    D[0, 0] = 0.1
    _refuses(lambda: A.sk_amp(D, 1, iters=3, seed=0), "precondition:zero-diagonal")
    r = _refuses(lambda: A.sk_amp(3 * J, 1, iters=3, seed=0), "precondition:coupling-variance")
    assert "1/N" in r.detail
    _refuses(lambda: A.sk_amp(J[:, :7], 1, iters=3, seed=0), "precondition:square-matrix")
    _refuses(lambda: A.sk_amp(J, -1, iters=3, seed=0), "precondition:beta-nonnegative")
    r = _refuses(lambda: A.sk_amp(J, 1, iters=3, seed=0, init="spectral"),
                 "precondition:known-init")
    assert "random" in r.detail and "independent of J" in r.detail
    _refuses(lambda: A.sk_amp(J, 1, iters=3, seed=0, init="bolthausen"),
             "precondition:bolthausen-init-field")
    _refuses(lambda: A.sk_amp(J, 1, iters=0, seed=0), "precondition:positive-iters")
    _refuses(lambda: A.sk_amp(J, 1, iters=3, seed=0, damping=1), "precondition:damping-range")
    _refuses(lambda: A.sk_matrix(8, 0, "cauchy"), "precondition:known-law")
    _refuses(lambda: A.sk_state_evolution(-1), "precondition:beta-nonnegative")
    _refuses(lambda: A.sk_state_evolution(1, q0=2), "precondition:q0-range")
    m = np.zeros(8)
    _refuses(lambda: A.sk_tap_free_energy(m + 2, J, 1), "precondition:magnetization-range")
    _refuses(lambda: A.sk_tap_free_energy(m[:7], J, 1), "precondition:shape")
    _refuses(lambda: A.sk_tap_free_energy(m, J, 1, onsager="third"), "precondition:known-onsager")
    _refuses(lambda: A.sk_tap_free_energy(m, J, -1), "precondition:beta-nonnegative")
    _refuses(lambda: A.sk_tap_gradient(m + 1, J, 1), "precondition:open-interval")
    _refuses(lambda: A.student_band([1.0]), "precondition:two-samples")


def test_tap_free_energy_small_exact_cases():
    """phi_TAP at m = 0 is ln 2 + beta^2/4 (the paramagnet's exact -beta f per spin at
    N -> oo); the naive and Plefka variants differ from it exactly as written."""
    J = AS.sk_matrix(8, 1, "pm_one")
    m0 = np.zeros(8)
    assert abs(A.sk_tap_free_energy(m0, J, "0.5") - (mp.log(2) + mp.mpf("0.0625"))) < 1e-14
    assert abs(A.sk_tap_free_energy(m0, J, "0.5", onsager="none") - mp.log(2)) < 1e-14
    # Plefka with J_ij^2 = 1/N on i != j: (beta^2/4)(N-1)/N at m = 0
    assert abs(A.sk_tap_free_energy(m0, J, "0.5", onsager="plefka")
               - (mp.log(2) + mp.mpf("0.0625") * 7 / 8)) < 1e-14
    # analytic gradient = finite differences of the scalar functional (each variant)
    rng = np.random.default_rng(5)
    m = rng.uniform(-0.6, 0.6, 8)
    for ons in ("averaged", "plefka", "none"):
        g = A.sk_tap_gradient(m, J, 0.9, 0.2, onsager=ons)
        for k in range(8):
            e = np.zeros(8)
            e[k] = 1e-6
            fd = 8 * float(A.sk_tap_free_energy(m + e, J, 0.9, 0.2, onsager=ons)
                           - A.sk_tap_free_energy(m - e, J, 0.9, 0.2, onsager=ons)) / 2e-6
            assert abs(fd - g[k]) < 1e-6, (ons, k, fd, g[k])


def test_amp_modules_do_not_import_replica_code():
    from replicant.amp import _stats
    for mod in (AS, SE, AT, _stats, A):
        src = inspect.getsource(mod)
        for line in src.splitlines():
            if re.match(r"\s*(from|import)\s", line):
                assert re.match(r"\s*(import (math|mpmath as mp|numpy as np)|from (dataclasses|"
                                r"\.\.core|\.|\.sk|\.tap|\._stats|\.state_evolution) import)",
                                line), (mod.__name__, line)
        for bad in ("families", "derive_", "derive import", "solve_derived", "sk_rs", "sk_replicon",
                    "gauss_expect", "..cavity", "..solve"):
            assert bad not in src, (mod.__name__, bad)


# ---- finite N (slow tier) -------------------------------------------------------------
def _amp_band(law, beta, h, t, *, reaction=True):
    ens = A.sk_amp_ensemble(N, beta, h, seeds=SEEDS, iters=t, law=law, reaction=reaction)
    return ens


@slow
def test_amp_finite_N_matches_state_evolution():
    """Gaussian J, N = 1500, seeds 0-5.  (a) beta = 1.5, h = 0.8 (above the AT line,
    lambda_AT = 0.40): q_t at t = 100 against the SE fixed point; every run converged, as
    the two-time SE predicts.  (b) beta = 1.5, h = 0 (AT-unstable): q_t at t = 10 against
    the SE trajectory from the same initial law; the runs do NOT converge, and the
    increment matches the two-time SE value 2 q*.  Positive control: naive mean field (no
    Onsager term) falls outside both bands."""
    for beta, h, t in ((1.5, 0.8, 100), (1.5, 0.0, 10)):
        se = A.sk_state_evolution(beta, h, q0=A.random_init_q(beta, h), iters=t)
        target = se.trajectory[t]
        ens = _amp_band("gaussian", beta, h, t)
        print(f"AMP gaussian beta={beta} h={h} t={t}: q = {ens.band.mean:.4f} +- "
              f"{ens.band.half_width:.4f} (99% t, df=5); SE {target:.4f}")
        assert _inside(ens.band, target), (ens.band, target)
        naive = _amp_band("gaussian", beta, h, t, reaction=False)
        print(f"  naive mean field q = {naive.band.mean:.4f} (excluded)")
        assert not _inside(ens.band, naive.band.mean), (ens.band, naive.band)
        tt = A.sk_two_time_state_evolution(beta, h, iters=t)
        inc = A.student_band(ens.increments)
        print(f"  increment {inc.mean:.3g} +- {inc.half_width:.2g}; two-time SE "
              f"{tt.increment[t]:.3g}")
        if h > 0:
            assert all(ens.converged) and se.amp_converges
        else:
            assert not any(ens.converged) and not se.amp_converges
            assert _inside(inc, tt.increment[t]), (inc, tt.increment[t])
        assert ens.label == "statistical" and "Bayati" in ens.note


@slow
def test_pm_one_matrix_universality():
    """+-1/sqrt(N) couplings: the same q within the band (numerical universality; the
    theorem of Bayati-Lelarge-Montanari covers polynomial nonlinearities only)."""
    for beta, h, t in ((1.5, 0.8, 100), (1.5, 0.0, 10)):
        se = A.sk_state_evolution(beta, h, q0=A.random_init_q(beta, h), iters=t)
        ens = _amp_band("pm_one", beta, h, t)
        print(f"AMP pm_one beta={beta} h={h} t={t}: q = {ens.band.mean:.4f} +- "
              f"{ens.band.half_width:.4f}; SE {se.trajectory[t]:.4f}")
        assert _inside(ens.band, se.trajectory[t]), (ens.band, se.trajectory[t])
        assert not _inside(ens.band, _amp_band("pm_one", beta, h, t, reaction=False).band.mean)
        assert "polynomial" in ens.note


@slow
def test_tap_free_energy_stationary_at_amp_fixed_point():
    """beta = 1.5, h = 0.8, N = 1500, seeds 0-5.  At the converged AMP iterate the TAP
    functional is stationary (directional finite differences ~ 0; the naive functional is
    not); phi_TAP per spin lies in the Student-t band around the RS -beta f (sk_rs:
    target); the naive functional at the same points does not; tap_fixed_point polishes
    the AMP iterate by Newton and gives the same phi."""
    beta, h = 1.5, 0.8
    phis, naive, plefka = [], [], []
    for s in SEEDS:
        J = A.sk_matrix(N, s, "gaussian")
        r = A.sk_amp(J, beta, h=h, iters=150, seed=s)
        assert r.converged
        rng = np.random.default_rng([s, 7])
        dirs = [r.m] + [rng.normal(size=N) for _ in range(2)]
        # eps = 1e-8: spins near saturation (1 - m^2 ~ 1e-5) make the cubic term of a
        # 1e-6 central difference ~1e-2; rounding at 1e-8 is ~1e-4
        dd = lambda ons: [N * float(A.sk_tap_free_energy(r.m + 1e-8 * u, J, beta, h, onsager=ons)
                                    - A.sk_tap_free_energy(r.m - 1e-8 * u, J, beta, h,
                                                           onsager=ons)) / 2e-8 for u in dirs]
        assert max(abs(x) for x in dd("averaged")) < 1e-2, dd("averaged")
        assert max(abs(x) for x in dd("none")) > 1, dd("none")       # positive control
        phis.append(float(A.sk_tap_free_energy(r.m, J, beta, h)))
        naive.append(float(A.sk_tap_free_energy(r.m, J, beta, h, onsager="none")))
        plefka.append(float(A.sk_tap_free_energy(r.m, J, beta, h, onsager="plefka")))
        if s == 0:
            tp = A.tap_fixed_point(J, beta, h=h, seed=s)
            assert tp.converged and tp.residual < 1e-10 and abs(tp.phi - phis[0]) < 1e-6, tp
            assert "exponentially many" in tp.note and tp.label == "statistical"
    band = A.student_band(phis)
    target = -mp.mpf(beta) * sk_rs(beta, h).f                 # comparison target only
    print(f"phi_TAP = {band.mean:.4f} +- {band.half_width:.4f}; RS -beta f {float(target):.4f}; "
          f"naive {np.mean(naive):.4f}")
    assert _inside(band, float(target)), (band, target)
    assert not _inside(band, float(np.mean(naive)))
    assert abs(np.mean(plefka) - band.mean) < band.half_width / 4, (np.mean(plefka), band)


@slow
def test_paramagnet_trivial_below_tc():
    """beta = 0.7, h = 0: SE q -> 0, AMP m -> 0 (the Onsager term makes the linearised
    iteration contract at rate beta); naive mean field keeps q > 0 (its linearisation
    grows at rate beta * lambda_max(J) ~ 2 beta = 1.4)."""
    se = A.sk_state_evolution(0.7)
    assert se.q == 0 and se.amp_converges is True
    ens = _amp_band("gaussian", 0.7, 0.0, 60)
    assert ens.band.mean + ens.band.half_width < 1e-6 and all(ens.converged), ens.band
    nv = _amp_band("gaussian", 0.7, 0.0, 60, reaction=False)
    assert nv.band.mean - nv.band.half_width > 0.1, nv.band
    print(f"paramagnet: AMP q {ens.band.mean:.2e}; naive {nv.band.mean:.3f}")
