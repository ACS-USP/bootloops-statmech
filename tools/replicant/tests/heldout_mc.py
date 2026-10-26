"""Held-out routes by Monte Carlo and exact enumeration (E11), used by tests/test_heldout.py.

Each check returns (failures, lines): failures empty = the check passes; lines are printed
by the test (the measured bands, including contrasts that are NOT resolved and therefore not
asserted, lessons 2 and 11).  All results are statistical (named seeds, 99% Student-t bands)
and finite-size; none is certified.  Seeds: 2026102601..04 (one per check).

  ground_state_check()    Gaussian SK e_0 by PT at N = 16..128, N^-2/3 extrapolation, against
                          the banked Parisi value; RS -sqrt(2/pi) must be excluded (control)
  finite_t_check(calc_rs, calc_1rsb, planted_rs)
                          +-J SK energy at beta = 2 by PT (N = 64, 128, 256) against the
                          composed RS and 1RSB energies
  log_z_check()           exact E ln Z - ln E Z at N = 12..22 against the moment closure
  perceptron_check(calc, spherical)
                          exact Ising perceptron counts at N = 12..24 against the composed
                          entropy and zero
"""
import math

import mpmath as mp
import numpy as np

from replicant.amp._stats import student_band, t_quantile
from replicant.numerics import enumeration as E
from replicant.numerics import finite_size as F
from replicant.numerics import moments as Mo
from replicant.refdata import load

# ---- (1) T = 0, Gaussian SK ---------------------------------------------------------
GS_SEED = 2026102601
GS_SAMPLES = {16: 1024, 32: 400, 64: 160, 128: 64}


def ground_state_check():
    fs = F.sk_ground_energy_finite_size(tuple(GS_SAMPLES), GS_SAMPLES, GS_SEED)
    fails, lines = [], list(fs.notes)
    lines += [f"  N = {N}: {fs.means[N]:.5f} +- {fs.stderrs[N]:.5f} (s.e.), {fs.diagnostics[N]}"
              for N in fs.sizes]
    # the PT ground states at N = 16 against exact enumeration of the same matrices
    J16 = F.gaussian_sk_couplings(16, GS_SAMPLES[16], [GS_SEED, 16, 0])
    exact = np.array([E.exact_ground_energy(J) for J in J16])
    if not np.allclose(fs.per_sample[16], exact, rtol=0, atol=1e-12):
        fails.append(f"N = 16: PT minima differ from exact enumeration in "
                     f"{int((~np.isclose(fs.per_sample[16], exact, atol=1e-12)).sum())} samples")
    sk = load("sk")
    parisi = [mp.mpf(sk["fullrsb_ground_energy"].value)] + \
        [mp.mpf(d["value"]) for d in sk["fullrsb_ground_energy"].sources_disagree]
    rs, rsb1 = mp.mpf(sk["rs_ground_energy"].value), mp.mpf(sk["rsb1_ground_energy"].value)
    lines.append(f"e_inf = {fs.e_inf:.5f} +- {fs.band:.5f}; Parisi {float(parisi[0]):.5f}, "
                 f"1RSB {float(rsb1):.5f} ({'excluded' if fs.excludes(rsb1) else 'NOT resolved'}"
                 f": gap to Parisi {float(rsb1 - parisi[0]):+.4f} vs band {fs.band:.4f}), "
                 f"RS {float(rs):.5f} ({'excluded' if fs.excludes(rs) else 'NOT excluded'})")
    fails += [f"Parisi value {float(p)} outside e_inf {fs.e_inf:.5f} +- {fs.band:.5f}"
              for p in parisi if fs.excludes(p)]
    if not fs.excludes(rs):
        fails.append(f"positive control: RS {float(rs)} inside {fs.e_inf:.5f} +- {fs.band:.5f}")
    return fails, lines


# ---- (2) beta = 2, +-J SK -----------------------------------------------------------
FT_SEED, FT_BETA = 2026102602, 2.0
FT_SAMPLES = {64: 256, 128: 64, 256: 32}


def composed_energy_rs(calc, beta="2"):
    r = calc.run(beta=beta, h=0, dps=15, branch="nontrivial")
    return (r.values["entropy"] - r.values["phi"]) / mp.mpf(beta)


def composed_energy_1rsb(calc, beta=2, d="0.005"):
    """u = -d phi / d beta by a central difference of the composed 1RSB phi = -beta f (the
    root of largest f, branch max-f); the O(d^2) error is ~1e-6 here."""
    with mp.workdps(20):
        b, d = mp.mpf(beta), mp.mpf(d)
        phi = lambda x: calc.run(beta=mp.nstr(x, 15), h=0, branch="max-f").values["phi"]
        return -(phi(b + d) - phi(b - d)) / (2 * d)


def finite_t_check(u_rs, u_1rsb, u_planted):
    fs = F.sk_energy_finite_size(FT_BETA, tuple(FT_SAMPLES), FT_SAMPLES, FT_SEED)
    fails, lines = [], list(fs.notes)
    lines += [f"  N = {N}: {fs.means[N]:.5f} +- {fs.stderrs[N]:.5f} (s.e.); {fs.diagnostics[N]}"
              for N in fs.sizes]
    big = fs.size_band(max(fs.sizes))
    lines.append(f"N = {max(fs.sizes)}: {big.mean:.5f} +- {big.half_width:.5f} (99% t); composed "
                 f"RS {float(u_rs):.5f}, 1RSB {float(u_1rsb):.5f}; extrapolated {fs.e_inf:.5f} +- "
                 f"{fs.band:.5f}")
    for name, u in (("RS", u_rs), ("1RSB", u_1rsb)):
        if not big.mean - float(u) > big.half_width:      # finite N lies above, outside the band
            fails.append(f"N = {max(fs.sizes)}: composed {name} {float(u):.5f} not below the band")
    if fs.excludes(u_1rsb):
        fails.append(f"composed 1RSB {float(u_1rsb):.5f} outside {fs.e_inf:.5f} +- {fs.band:.5f}")
    if not fs.excludes(u_planted):
        fails.append(f"positive control: planted {float(u_planted):.5f} inside the band")
    mid = (float(u_rs) + float(u_1rsb)) / 2
    lines.append(f"1RSB closer than RS: {'resolved' if fs.excludes(mid) else 'NOT resolved'} "
                 f"(midpoint {mid:.5f} at {abs(mid - fs.e_inf):.4f} from the centre, band "
                 f"{fs.band:.4f}); RS {'excluded' if fs.excludes(u_rs) else 'NOT excluded'} "
                 f"by the extrapolated band")
    return fails, lines


# ---- (3) beta < 1: quenched against annealed, +-J SK --------------------------------
LZ_SEED, LZ_BETAS = 2026102603, (0.7, 0.9)
LZ_SAMPLES = {12: 20000, 16: 20000, 20: 6000, 22: 3000}


def log_z_check():
    fails, lines = [], []
    for N, S in LZ_SAMPLES.items():
        X = {b: np.empty(S) for b in LZ_BETAS}
        L1 = {b: float(Mo.log_moments(N, b)[0]) for b in LZ_BETAS}
        for k in range(S):
            sp = E.enumerate_pm_one_sk(E.random_pm_one_J(N, [LZ_SEED, N, k]))
            for b in LZ_BETAS:
                X[b][k] = sp.log_Z(b) - L1[b]
        for b in LZ_BETAS:
            band = student_band(X[b])
            p2, p3 = Mo.quenched_gap(N, b, 2), Mo.quenched_gap(N, b, 3)
            g2, g3 = Mo.quenched_gap(N, b, 2, "gaussian"), Mo.quenched_gap(N, b, 3, "gaussian")
            ok_pred = abs(band.mean - p2) <= band.half_width + abs(p2 - p3)
            gauss_out = abs(band.mean - g2) > band.half_width + abs(g2 - g3)
            lines.append(f"N = {N}, beta = {b}: E lnZ - ln EZ = {band.mean:+.5f} +- "
                         f"{band.half_width:.5f} ({S} samples); +-J closure {p2:+.5f} (order 3 "
                         f"{p3:+.5f}); Gaussian-law closure {g2:+.5f} ({g3:+.5f}): "
                         f"{'excluded' if gauss_out else 'NOT resolved'}")
            if abs(band.mean) <= band.half_width:
                fails.append(f"N = {N}, beta = {b}: annealed (0) inside the band")
            if not ok_pred:
                fails.append(f"N = {N}, beta = {b}: +-J closure {p2:+.5f} outside the band")
            if b == 0.7 and not gauss_out:
                fails.append(f"N = {N}, beta = 0.7: Gaussian-law closure not excluded")
    return fails, lines


# ---- (4) Ising perceptron -----------------------------------------------------------
PC_SEED, PC_ALPHAS = 2026102604, ("0.25", "0.5", "0.75")
PC_SAMPLES = {12: 8000, 16: 8000, 20: 3000, 24: 600}


def _fit_1n(Ns, y, se, deg=1):
    x = 1 / np.array(Ns, float)
    w = 1 / np.array(se)**2
    A = np.vstack([x**k for k in range(deg + 1)]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    return float((cov @ (A.T @ (w * np.array(y))))[0]), float(math.sqrt(cov[0, 0]))


def extrapolate_1n(per_size, alt=None):
    """per_size: N -> per-sample values.  Base: weighted fit a + b/N.  Band = 99% t-band of a
    (df = smallest sample count - 1) + the largest shift over: without the smallest N, the
    quadratic a + b/N + c/N^2, and (when given) the same base fit of the alternative
    estimator `alt` (the ln 0 hazard).  Returns (centre, band, stat, syst)."""
    Ns = sorted(per_size)
    y = [per_size[N].mean() for N in Ns]
    se = [per_size[N].std(ddof=1) / math.sqrt(len(per_size[N])) for N in Ns]
    a, sa = _fit_1n(Ns, y, se)
    shifts = [_fit_1n(Ns[1:], y[1:], se[1:])[0], _fit_1n(Ns, y, se, 2)[0]]
    if alt is not None:
        shifts.append(extrapolate_1n(alt)[0])
    stat = t_quantile(0.995, min(len(v) for v in per_size.values()) - 1) * sa
    syst = max(abs(s - a) for s in shifts)
    return a, stat + syst, stat, syst


def perceptron_check(S_composed, zero_composed, S_spherical, alpha_c_bank):
    counts = {N: np.array([E.ising_perceptron_counts(N, 3 * N, [PC_SEED, N, k])
                           for k in range(S)]) for N, S in PC_SAMPLES.items()}
    fails, lines = [], []
    for a in PC_ALPHAS:
        P = {N: int(mp.mpf(a) * N) for N in counts}
        main = {N: np.log1p(c[:, P[N]]) / N for N, c in counts.items()}
        cond = {N: np.log(c[c[:, P[N]] > 0, P[N]]) / N for N, c in counts.items()}
        zeros = {N: float((c[:, P[N]] == 0).mean()) for N, c in counts.items()}
        s, band, stat, syst = extrapolate_1n(main, cond)
        ann = math.log(2) * (1 - float(a))
        verdict = "excluded" if abs(ann - s) > band else "NOT resolved"
        lines.append(f"alpha = {a}: S_inf = {s:.4f} +- {band:.4f} (stat {stat:.4f}, syst "
                     f"{syst:.4f}); empty fraction by N {zeros}; composed "
                     f"{float(S_composed[a]):.4f}; annealed {ann:.4f} ({verdict}); spherical "
                     f"site {float(S_spherical.get(a, mp.nan)):.4f}")
        if abs(float(S_composed[a]) - s) > band:
            fails.append(f"alpha = {a}: composed S {float(S_composed[a]):.4f} outside the band")
        if abs(ann - s) <= band:
            fails.append(f"alpha = {a}: annealed ln2 (1 - alpha) inside the band")
        if a in S_spherical and abs(float(S_spherical[a]) - s) <= band:
            fails.append(f"alpha = {a}: spherical-site entropy inside the band (control)")
    cap = {N: np.array([E.ising_perceptron_capacity(r) for r in c]) / N for N, c in counts.items()}
    m, band, stat, syst = extrapolate_1n(cap)
    lines.append(f"capacity M_N/N: {m:.4f} +- {band:.4f} (stat {stat:.4f}, syst {syst:.4f}); by N "
                 + ", ".join(f"{N}: {v.mean():.4f}" for N, v in cap.items())
                 + f"; composed zero {float(zero_composed):.5f}, bank {float(alpha_c_bank):.5f}")
    for name, v in (("composed zero", zero_composed), ("banked alpha_c", alpha_c_bank)):
        if abs(float(v) - m) > band:
            fails.append(f"{name} {float(v):.5f} outside capacity {m:.4f} +- {band:.4f}")
    if abs(2 - m) <= band:
        fails.append("spherical capacity 2 inside the capacity band (control)")
    return fails, lines
