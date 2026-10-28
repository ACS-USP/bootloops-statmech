"""Wang-Landau estimates of ln g(E) for the 2D Ising torus (M2), with the original schedule and the 1/t
refinement, and their error profile against the exact g(E) (dos_exact.py).

Schedules (F = ln f, the increment of S = ln g at every visit; flat_walk.py for the move):
  "wang_landau"  Wang & Landau, arXiv:cond-mat/0011174v1 (2001) (PRL 86, 2050), PDF read, p. 2: f0 = e (F0 = 1);
      the histogram H is "flat" when H(E) >= 80% of <H(E)> for all E; then f -> sqrt f (F -> F / 2) and H = 0;
      stop when f < f_final.
  "one_over_t"   Belardinelli & Pereyra, PRE 75, 046701 (2007), arXiv:cond-mat/0701672, PDF read, Sec. III
      steps (i)-(vi): F0 = 1; after a fixed period check H; if H(E) != 0 for all E, F -> F / 2 and H = 0;
      once F <= 1/t, F = F(t) = 1/t at every MC time from then on (H no longer used); stop when F < F_final.
      t = j / N_E: j trial moves of the walker, N_E its number of energy levels (their Sec. II: "t is
      normalized to the entire range of energy").
Which levels count.  "all E" is read as the levels the walker has visited so far (no exact knowledge enters
the walk); the run then checks every walker visited the same set (tests: = the exact set of g > 0).  WL and BP
walk a known energy range; here the range is discovered.  A new level (found by the moves of the last MC time) gets
S(new) = the largest S of the walker's visited levels within 2 of it (the levels it can have been entered from; a
new level at S = 0 would trap the walk there, every move out refused); a walker past its first stage (F < F0 or in
the 1/t phase) also restarts its schedule: F = F0, H = 0, the 1/t phase left.  To make late discoveries rare, a
walker leaves its first stage (F = F0) only when it has found no new level for the last half of its MC time, so a
restart at MC time t_r holds F = F0 until 2 t_r (a discovery after t_max / 2 leaves the walker undetermined; its
initial error then decays as 1/t, the order of the intrinsic 1/t bias).  These
rules are this package's, stated; restarts are counted per walker (WLRun.restarts).
Check period: every `check_every` MC time (BP give "some fixed sweeps (i.e., 1000 MC time)"; the default 10
here is this package's choice, stated, so the 1/t phase starts within ~100 MC time at L <= 12).
Additive constant: ln g(ground) = ln 2 (WL p. 2: "the number of ground states for the Ising model is 2"; BP:
"normalized with respect to the exact DOS at the ground state").  Error profile: eps(E) = |S_est - S_ex| / S_ex
(WL's eps(X) = |(X_sim - X_exact)/X_exact| for X = log g; BP Eq. (2)), <eps> its mean over the levels other
than the ground state (BP: (N - 1)^{-1} sum_E).

  wang_landau(L, seeds, walkers=8, schedule="one_over_t", t_max=10**4, F_final=1e-4, check_every=10,
              snapshots=()) -> WLRun
      per walker S normalised on the levels visited by every walker; per seed the mean over its walkers;
      mean and stderr over seeds; status "converged" (every walker reached F < F_final within t_max and all
      visited the same levels, the ground state among them) or "undetermined"; snapshots: MC times at which
      the normalised S of every walker is stored (the error-against-time measurement)
  error_profile(run, exact, conf=0.99) -> ErrorProfile
      signed deviation per level in a Bonferroni-corrected Student-t band over seeds (conf per level =
      1 - (1 - conf) / levels), eps per level, <eps> per seed and its band, and whether every level's band
      holds 0; first_walker=True uses each seed's first walker alone (the default-tier resolution)
  error_slope(run, exact, t1, t2) -> ErrorSlope    the saturation diagnostic: per-seed slope of ln <eps> against ln t
      between two snapshots, in a Student-t band; `decreasing` iff its upper edge < -1/4 (the midpoint of BP's
      t^{-1/2} and a saturated error's 0)
  refinement(t) -> 1/t (the 1/t phase)
_HOOKS["flatness"] ("min"; planted "max": the flatness test asks max H >= 0.8 <H> (WL) / H != 0 for some E
(1/t), so f is reduced at every check: a wrong flatness criterion), _HOOKS["time_exponent"] (1.0; planted 2:
F = t^-2, faster than 1/t, which BP say saturates).
"""
from dataclasses import dataclass

import numpy as np

from ..core import Refusal
from . import stats as ST
from .flat_walk import Walkers

SCHEDULES = ("one_over_t", "wang_landau")
FLATNESS = 0.8
F0 = 1.0
LN2 = float(np.log(2.0))
SLOPE_RULE = -0.25
_HOOKS = {"flatness": "min", "time_exponent": 1.0}


@dataclass
class WLRun:
    L: int
    schedule: str
    seeds: tuple
    walkers: int
    k: np.ndarray            # levels visited by every walker
    E: np.ndarray
    S: np.ndarray            # (walkers total, levels), ln g(ground) = ln 2
    per_seed: np.ndarray     # (seeds, levels)
    mean: np.ndarray
    stderr: np.ndarray
    t_end: np.ndarray        # MC time at which each walker stopped
    F_end: np.ndarray
    moves: int
    status: str
    text: str
    snapshots: dict
    restarts: np.ndarray     # schedule restarts per walker (a new level found)


@dataclass(frozen=True)
class ErrorProfile:
    E: np.ndarray
    deviation: np.ndarray    # mean over seeds of S_est - S_ex
    half_width: np.ndarray   # Bonferroni Student-t half-widths
    eps: np.ndarray          # mean over seeds of eps(E) per level
    eps_mean: ST.Band        # <eps> per seed, band over seeds
    within: bool
    worst: float             # max |deviation| / half_width
    text: str


def _int(name, x, lo, hi):
    if isinstance(x, bool) or not isinstance(x, (int, np.integer)) or not lo <= x <= hi:
        raise Refusal(f"precondition:{name}", f"{name} = {x!r}: need an int in {lo} .. {hi}")
    return int(x)


def check_params(L, seeds, walkers, schedule, t_max, F_final, check_every):
    from ..percolation.lattice2d import check_seeds
    from .dos_exact import check_L
    if not isinstance(schedule, str) or schedule not in SCHEDULES:
        raise Refusal("precondition:schedule", f"schedule = {schedule!r}: one of {', '.join(SCHEDULES)}")
    from ..lattice.ball import rational
    Ff = float(rational(F_final, "f-final"))
    if not 0 < Ff < F0:
        raise Refusal("precondition:f-final", f"F_final = {F_final!r}: need a number in (0, {F0:g})")
    return (check_L(L, 32), check_seeds(seeds), _int("walkers", walkers, 1, 256), schedule,
            _int("t-max", t_max, 10, 10 ** 7), Ff, _int("check-every", check_every, 1, 10 ** 4))


def refinement(t):
    """F(t) = 1/t of the 1/t phase (BP step (v)); planted exponent through _HOOKS["time_exponent"]."""
    return np.asarray(t, float) ** -float(_HOOKS["time_exponent"])


def _flat(H, vis, schedule):
    Hv = np.where(vis, H, np.nan)
    if schedule == "one_over_t":
        nz = (H > 0) | ~vis
        return nz.all(axis=1) if _HOOKS["flatness"] == "min" else ((H > 0) & vis).any(axis=1)
    mean = np.nanmean(Hv, axis=1)
    side = np.nanmin(Hv, axis=1) if _HOOKS["flatness"] == "min" else np.nanmax(Hv, axis=1)
    return side >= FLATNESS * mean


def _normalised(S, common):
    g0 = S[:, common[0]][:, None]
    return S[:, common] - g0 + LN2


def wang_landau(L, seeds, walkers=8, schedule="one_over_t", t_max=10 ** 4, F_final=1e-4, check_every=10,
                snapshots=()):
    L, seeds, W, schedule, t_max, F_final, check_every = check_params(L, seeds, walkers, schedule, t_max, F_final,
                                                                      check_every)
    wk = Walkers(L, seeds, W)
    R, K = wk.R, wk.K
    S, H = np.zeros(R * K), np.zeros(R * K)
    vis = np.zeros((R, K), bool)
    vis[np.arange(R), wk.k] = True
    F = np.full(R, F0)
    late = np.zeros(R, bool)               # in the 1/t phase
    done = np.zeros(R, bool)
    t_end = np.full(R, np.nan)
    moves, blocks = 0, 0
    restarts = np.zeros(R, int)
    t_new = np.zeros(R)                 # MC time of each walker's latest new level
    snaps, pending = {}, sorted(float(x) for x in snapshots)
    while True:
        ne = vis.sum(axis=1)
        block = int(ne.max())               # one MC time of the walker with the most levels
        wk.run(block, S, np.where(done, 0.0, F), H)
        moves += block
        blocks += 1
        Hm, S2 = H.reshape(R, K), S.reshape(R, K)
        new = (Hm > 0) & ~vis & ~done[:, None]
        if new.any():                       # the range grew (module docstring)
            pad = np.pad(np.where(vis, S2, -np.inf), ((0, 0), (2, 2)), constant_values=-np.inf)
            near = np.max([pad[:, 2 + d:2 + d + K] for d in (-2, -1, 1, 2)], axis=0)   # visited levels within 2
            S2[new] = np.maximum(S2[new], near[new])
            grew = new.any(axis=1) & ((F < F0) | late)   # a walker past its first stage restarts its schedule
            F, late = np.where(grew, F0, F), late & ~grew
            Hm[grew] = 0
            restarts += grew
        vis |= (Hm > 0) | new
        t = moves / vis.sum(axis=1)
        t_new = np.where(new.any(axis=1), t, t_new)
        if schedule == "one_over_t":
            F = np.where(late & ~done, refinement(t), F)
        if blocks % check_every == 0:
            settled = (F < F0) | (t >= 2 * t_new)    # leave the first stage only after no new level for t / 2
            flat = _flat(Hm, vis, schedule) & ~late & ~done & settled
            F = np.where(flat, F / 2, F)
            Hm[flat] = 0
            if schedule == "one_over_t":
                sw = ~late & ~done & (F <= 1 / t)
                late |= sw
                F = np.where(sw, refinement(t), F)
        newly = ~done & (F < F_final)
        t_end[newly] = t[newly]
        done |= newly
        while pending and t.min() >= pending[0]:
            snaps[pending.pop(0)] = S.reshape(R, K).copy()
        if done.all() or t.min() >= t_max:
            break
    wk.check()
    t_end = np.where(done, t_end, t)
    common = np.nonzero(vis.all(axis=0))[0]
    same = bool((vis == vis[0]).all())
    S2 = S.reshape(R, K)
    Sn = _normalised(S2, common)
    per_seed = Sn.reshape(len(seeds), W, -1).mean(axis=1)
    mean = per_seed.mean(axis=0)
    se = per_seed.std(axis=0, ddof=1) / np.sqrt(len(seeds))
    ok = bool(done.all() and same and common.size and common[0] == 0)
    why = [] if ok else [w for w, c in (("a walker did not reach F_final within t_max", not done.all()),
                                        ("walkers visited different levels", not same),
                                        ("the ground state was not visited", not (common.size and common[0] == 0)))
                         if c]
    text = (f"{schedule}: {len(seeds)} seeds x {W} walkers, {common.size} levels, F_final {F_final:g}, MC time "
            f"{np.nanmin(t_end):.4g} .. {np.nanmax(t_end):.4g} (t_max {t_max}): "
            + ("converged" if ok else "UNDETERMINED: " + "; ".join(why)))
    snaps = {tt: _normalised(x, common) for tt, x in snaps.items()}
    return WLRun(L, schedule, seeds, W, common, 4 * common - 2 * L * L, Sn, per_seed, mean, se, t_end, F,
                 moves, "converged" if ok else "undetermined", text, snaps, restarts)


def _eps(S, ex_lng):
    return np.abs(S[..., 1:] - ex_lng[1:]) / ex_lng[1:]


def _levels_match(run, exact):
    if not np.array_equal(run.k, exact.k):
        raise Refusal("no-solution:level-set", f"visited levels {run.k.tolist()} != the exact levels with g > 0 "
                      f"{exact.k.tolist()}")


def error_profile(run, exact, conf=0.99, first_walker=False):
    """first_walker: each seed's estimate is its first walker alone (the resolution of the default-tier test)
    instead of the mean of its walkers."""
    _levels_match(run, exact)
    n = len(run.seeds)
    est = run.S.reshape(n, run.walkers, -1)[:, 0, :] if first_walker else run.per_seed
    d = est - exact.ln_g
    m = d.shape[1] - 1                       # the ground level is fixed by the normalisation
    c = 1 - (1 - conf) / m
    t = ST.t_quantile((1 + c) / 2, n - 1)
    hw = t * d.std(axis=0, ddof=1) / np.sqrt(n)
    dm = d.mean(axis=0)
    eps_w = _eps(run.S, exact.ln_g)          # per walker, per level
    eps_seed = eps_w.reshape(n, run.walkers, -1).mean(axis=1)
    band = ST.band(eps_seed.mean(axis=1), conf)
    ratio = np.abs(dm[1:]) / np.maximum(hw[1:], 1e-300)
    within = bool((ratio <= 1).all())
    text = (f"ln g against the exact g(E) on {m + 1} levels ({'first walker' if first_walker else 'walker mean'} per "
            f"seed): max |deviation| {np.abs(dm).max():.3g}, worst |deviation| / band {ratio.max():.3g} (Bonferroni "
            f"{c:.5f} Student-t per level over {n} seeds, largest half-width {hw.max():.3g}); <eps> = {band.mean:.3g} "
            f"+- {band.half_width:.2g}: " + ("every level inside its band" if within else "OUTSIDE the band"))
    return ErrorProfile(run.E, dm, hw, eps_seed.mean(axis=0), band, within, float(ratio.max()), text)


@dataclass(frozen=True)
class ErrorSlope:
    t1: float
    t2: float
    slope: ST.Band
    decreasing: bool
    text: str


def error_slope(run, exact, t1, t2, conf=0.99):
    """The saturation diagnostic: per seed s = ln(<eps>(t2) / <eps>(t1)) / ln(t2 / t1), <eps> over the seed's walkers
    from the snapshots (a walker that stopped keeps its final S), in a Student-t band over seeds.  BP: <eps> ~ t^{-1/2}
    with 1/t (their Fig. 3: slope -0.488 +- 0.008 at L = 8); a saturated error has slope 0.  Rule: `decreasing` iff
    the band's upper edge is below -1/4, the midpoint of the two hypotheses (0 and -1/2)."""
    _levels_match(run, exact)
    n = len(run.seeds)
    e = [_eps(run.snapshots.get(float(t), run.S), exact.ln_g).mean(axis=1).reshape(n, run.walkers).mean(axis=1)
         for t in (t1, t2)]
    b = ST.band(np.log(e[1] / e[0]) / np.log(t2 / t1), conf)
    dec = b.mean + b.half_width < SLOPE_RULE
    text = (f"<eps> slope between MC time {t1:g} and {t2:g}: {b.mean:+.3f} +- {b.half_width:.3f} ({conf:.0%} Student-t "
            f"over {n} seeds): " + ("decreasing (upper edge < -1/4)" if dec else "SATURATED (upper edge >= -1/4)"))
    return ErrorSlope(t1, t2, b, bool(dec), text)


__all__ = ["ErrorProfile", "ErrorSlope", "SCHEDULES", "WLRun", "check_params", "error_profile", "error_slope",
           "refinement", "wang_landau"]
