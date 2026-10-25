"""Survey propagation (SP) for random K-SAT: on an instance, and as population dynamics of
the 1RSB energetic cavity at y -> oo (zero energy: the clusters of SAT assignments).

SP update (Braunstein, Mezard & Zecchina 2005, Eqs. (26)-(27); Mertens, Mezard & Zecchina
2006, Eqs. (1)-(2)).  For clause a, an input variable j != i of a, ps = prod (1 - eta_{b->j})
over the other clauses b where j has the SAME sign as in a (no "supporting" warning) and
pu = the product over the clauses where j has the OPPOSITE sign (no "impeding" warning):
    Pi^u = (1 - pu) ps,  Pi^s = (1 - ps) pu,  Pi^0 = ps pu,
    eta_{a->i} = prod_{j != i} Pi^u / (Pi^u + Pi^s + Pi^0).
The factor is a HOOK: SP_UPDATES[name](ps, pu) (default "bmz"), read at call time by every
routine here, so a planted rule reaches the instance and the population alike.  Its
independent route is `warning_factor_by_enumeration` (MMZ Sec. 2.1: each clause warns with
probability eta, independently; contradictory configurations are discarded).

Complexity (ln of the number of clusters per variable).  Instance: BMZ Eqs. (39)-(41),
    N Sigma = sum_a ln[prod_j (Pi^u + Pi^s + Pi^0)_{j->a} - prod_j Pi^u_{j->a}]
              - sum_i (n_i - 1) ln[P+_i + P-_i - P+_i P-_i],
P+-_i = prod over the clauses where i appears unnegated (negated) of (1 - eta).  Population:
MMZ Eq. (18), Sigma = Sigma_0 + alpha [K I_(K-1) - (K-1) I_K] ("complexity_mmz18"), and the
same expression rewritten with the fixed-point identity (17) (E x = (K alpha / 2) E phi and
E phi = -E ln(1 - prod_(K-1) rho)), which has a smaller variance ("complexity"):
    Sigma = E ln(pi + pi' - pi pi') - alpha (K - 1) E ln(1 - prod_(i=1..K) rho_i),
    rho = pi' (1 - pi) / (pi + pi' - pi pi'),  pi = e^-x, x = sum of Poisson(K alpha / 2) phi's,
    phi = -ln(1 - eta).
The population (MMZ Sec. 5.2, without the regularisation of Sec. 5.1: zero surveys are kept
in the population) starts from phi ~ Exponential(mean 2^(1-K)) (MMZ step (2)); each sweep
replaces all `pop` members in parallel; phi is capped at PHI_MAX (eta = 1 - e^-50) so that a
numerically certain warning never produces 0/0.

  ksat_sp(formula, *, iters, seed, tol, damping, rule) -> SPResult
  ksat_population_dynamics_sp(K, alpha, *, pop, sweeps, seed, burn_in, runs, rule) -> PDResult
      phase "trivial" (every run collapsed: max phi < 1e-12; Sigma = 0 to that accuracy),
      "nontrivial" (no drift of the mean survey, cavity/_stats.drift), or "undetermined"
      (drifting: not equilibrated, typically near the SP onset)
  ksat_sp_crossing(K, alphas, *, pop, sweeps, seed, burn_in, runs, rule) -> SPCrossing
      per run a straight line fitted to Sigma over the grid, its root; mean +- stderr over
      runs and the 99% Student-t half-width
Statistical: population values carry stderr over independent runs (SeedSequence(seed).spawn),
never certified.  The SP threshold is the cavity CONJECTURE alpha_s = root of Sigma (MMZ Sec.
2.3); for K = 3 it is not proven (Ding, Sly & Sun prove it for K >= k0, an unspecified
constant).
Refusals: unsupported:sp-rule, precondition:population-size, :runs, :sweeps, :burn-in,
:crossing-grid, :named-seeds, no-convergence:equilibration, no-solution:bracket (+ those of ksat.py).
"""
import itertools
import math
from dataclasses import dataclass

import numpy as np

from ..cavity._stats import drift, t_quantile
from ..cavity.sparse import PDResult
from ..core import Refusal
from . import _checks
from .ksat import _bp_iter_args, _k, group_products, others_product, var_products
from .xorsat import _alpha

PHI_MAX = 50.0
# Settings of tests/test_ksat.py::test_sp_complexity_vanishes_at_banked_threshold, measured
# before the assert (seed 0, ~37 s): alpha_s = 4.2623 +- 0.0108 (99%, 6 runs; per-run roots
# 4.2545 - 4.2705); planted 1.005 power: 4.3402 +- 0.0207.  Earlier, pop 50000, sweeps 40,
# independent (not warm-started) densities: 4.2641 +- 0.0078; planted 1.003: 4.3158 +- 0.0185.
CROSSING = {"alphas": ("4.18", "4.22", "4.26", "4.30", "4.34"), "pop": 30000, "sweeps": 30,
            "burn_in": 100, "runs": 6, "seed": 0, "max_halfwidth": 0.02}
TRIVIAL = 1e-12
_MMZ = ("Mertens, Mezard & Zecchina, Random Struct. Algorithms 28, 340 (2006), "
        "arXiv:cs/0309020")


def bmz_rule(ps, pu):
    """Pi^u / (Pi^u + Pi^s + Pi^0) = ps (1 - pu) / (ps + pu - ps pu)."""
    return ps * (1 - pu) / (ps + pu - ps * pu)


SP_UPDATES = {"bmz": bmz_rule}


def planted_power_rule(p):
    """A deliberately wrong factor (test positive control): the BMZ factor to the power p."""
    return lambda ps, pu: bmz_rule(ps, pu)**p


def _rule(name):
    if name not in SP_UPDATES:
        raise Refusal("unsupported:sp-rule",
                      f"SP rule {name!r}; implemented: {', '.join(sorted(SP_UPDATES))}")
    return SP_UPDATES[name]


def warning_factor_by_enumeration(same, opp):
    """P(j receives >= 1 impeding warning and no supporting one | no contradiction), by
    enumerating every warning configuration of the clauses `same` (supporting) and `opp`
    (impeding), each warning independently with its probability eta."""
    num = den = 0.0
    etas = list(same) + list(opp)
    for w in itertools.product((0, 1), repeat=len(etas)):
        p = math.prod(e if b else 1 - e for e, b in zip(etas, w))
        sup, imp = any(w[:len(same)]), any(w[len(same):])
        if sup and imp:
            continue
        den += p
        num += p if imp else 0.0
    return num / den


def audit_sp_rule(rule="bmz", cases=300, seed=0):
    """max |hook factor - enumeration| over random survey sets of 0..4 clauses per sign."""
    R, rng, worst = _rule(rule), np.random.default_rng(seed), 0.0
    draw = lambda n: [float(rng.choice([0.0, 1.0, rng.random()], p=[0.1, 0.05, 0.85]))
                      for _ in range(n)]
    for _ in range(cases):
        same, opp = draw(int(rng.integers(0, 5))), draw(int(rng.integers(0, 5)))
        ps, pu = math.prod(1 - e for e in same), math.prod(1 - e for e in opp)
        if ps + pu - ps * pu == 0:
            continue
        worst = max(worst, abs(float(R(np.float64(ps), np.float64(pu)))
                               - warning_factor_by_enumeration(same, opp)))
    return worst


# ---- SP on an instance -----------------------------------------------------------------
@dataclass(frozen=True)
class SPResult:
    converged: bool
    iterations: int
    residual: float
    trivial: bool            # every survey below 1e-8 (the paramagnetic fixed point eta = 0)
    complexity: float        # Sigma per variable, BMZ Eqs. (39)-(41) at the final surveys
    surveys: np.ndarray      # eta_{a->i}, shape (M, K)
    note: str = ""


def instance_complexity(f, eta):
    ps, pu = group_products(f, eta)
    tot, piu = ps + pu - ps * pu, (1 - pu) * ps
    pp, pm = var_products(f, eta)
    deg = np.bincount(f.var.ravel(), minlength=f.N)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (np.sum(np.log(np.prod(tot, axis=1) - np.prod(piu, axis=1)))
             - np.sum((deg - 1) * np.log(pp + pm - pp * pm)))
    return float(s) / f.N


def ksat_sp(f, *, iters=1000, seed=0, tol=1e-6, damping=0.0, rule="bmz"):
    """Synchronous SP sweeps from uniform random surveys (seed) with optional damping; BMZ
    use sequential random-order sweeps and precision 1e-3, so non-convergence here is a
    statement about this schedule."""
    R = _rule(rule)
    iters, damping = _bp_iter_args(iters, damping)
    eta = np.random.default_rng(_checks.seed(seed)).random(f.var.shape)
    res, it, ok, note = math.inf, 0, False, ""
    with np.errstate(invalid="ignore", divide="ignore"):
        for it in range(1, iters + 1):
            ps, pu = group_products(f, eta)
            new = others_product(R(ps, pu))
            if not np.all(np.isfinite(new)):
                note = "contradictory surveys (ps = pu = 0 on an edge)"
                break
            new = damping * eta + (1 - damping) * new
            res = float(np.max(np.abs(new - eta))) if new.size else 0.0
            eta = new
            if res < tol:
                ok = True
                break
    if not ok and not note:
        note = f"not converged: residual {res:.3g} >= tol {tol:g} after {it} sweeps"
    trivial = bool(eta.size == 0 or np.max(eta) < 1e-8)
    return SPResult(ok, it, res, trivial, instance_complexity(f, eta), eta, note)


# ---- population dynamics ---------------------------------------------------------------
def _psums(rng, phi, lam, n):
    """n sums of phi over Poisson(lam) members drawn uniformly from the population."""
    k = rng.poisson(lam, n)
    vals = phi[rng.integers(len(phi), size=int(k.sum()))]
    cs = np.concatenate(([0.0], np.cumsum(vals)))
    e = np.cumsum(k)
    return cs[e] - cs[e - k]


def _sweep(rng, phi, K, a, R):
    n, lam = len(phi), K * a / 2
    eta = np.ones(n)
    for _ in range(K - 1):
        eta *= R(np.exp(-_psums(rng, phi, lam, n)), np.exp(-_psums(rng, phi, lam, n)))
    with np.errstate(divide="ignore"):
        return np.minimum(-np.log1p(-eta), PHI_MAX)


def _measure(rng, phi, K, a, n):
    """(Sigma by identity-(17) form, Sigma by MMZ Eq. (18)) from n fresh samples."""
    lam = K * a / 2
    lr, L, s0 = np.zeros(n), np.zeros(n), np.zeros(n)
    lrs = []
    with np.errstate(divide="ignore"):
        for _ in range(K):
            x, z = _psums(rng, phi, lam, n), _psums(rng, phi, lam, n)
            p, q = np.exp(-x), np.exp(-z)
            den = p + q - p * q
            lrho = np.log(q * -np.expm1(-x)) - np.log(den)
            lrs.append(lrho)
            L += np.log(den) / K
            s0 += (x + z + np.log(den)) / K
        jk = np.log1p(-np.exp(np.sum(lrs, axis=0)))
        jk1 = np.mean([np.log1p(-np.exp(np.sum([l for j, l in enumerate(lrs) if j != d], axis=0)))
                       for d in range(K)], axis=0)
    a_ = float(a)
    return float(np.mean(L - a_ * (K - 1) * jk)), float(np.mean(s0 + a_ * (K * jk1 - (K - 1) * jk)))


def _pd_args(K, alpha, pop, sweeps, seed, burn_in, runs):
    K, a = _k(K), float(_alpha(alpha))
    seed = _checks.seed(seed)
    if isinstance(pop, bool) or not isinstance(pop, int) or pop < 100:
        raise Refusal("precondition:population-size", f"pop must be an int >= 100, got {pop!r}")
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 2:
        raise Refusal("precondition:runs", f"runs must be an int >= 2 for a stderr, got {runs!r}")
    if isinstance(sweeps, bool) or not isinstance(sweeps, int) or sweeps < 4:
        raise Refusal("precondition:sweeps", f"sweeps must be an int >= 4 (the drift test "
                      f"compares halves of the measured sweeps), got {sweeps!r}")
    if isinstance(burn_in, bool) or not isinstance(burn_in, int) or burn_in < max(2, sweeps // 2):
        raise Refusal("precondition:burn-in", f"burn_in must be an int >= max(2, sweeps // 2), "
                      f"got {burn_in!r}")
    return K, a, seed


def _run(K, a, rng, pop, burn_in, sweeps, R, phi=None):
    phi = rng.exponential(2.0**(1 - K), pop) if phi is None else phi
    traj, meas, s2, s1, zeros = [], [], [], [], []
    for t in range(burn_in + sweeps):
        phi = _sweep(rng, phi, K, a, R)
        if not np.all(np.isfinite(phi)):
            raise Refusal("no-solution:audit", f"SP population not finite at sweep {t}")
        m = float(np.mean(-np.expm1(-phi)))
        traj.append(m)
        if t >= burn_in:
            meas.append(m)
            zeros.append(float(np.mean(phi == 0)))
            x2, x1 = _measure(rng, phi, K, a, pop)
            s2.append(x2)
            s1.append(x1)
    return {"traj": traj, "meas": meas, "s2": float(np.mean(s2)), "s1": float(np.mean(s1)),
            "eta": float(np.mean(meas)), "zero": float(np.mean(zeros)), "trivial": bool(np.max(phi) < TRIVIAL), "phi": phi}


def _phase(runs_out, sweeps):
    if all(r["trivial"] for r in runs_out):
        return "trivial", f"every run collapsed to eta = 0 (max phi < {TRIVIAL:g})"
    drifting, diag = drift([r["meas"] for r in runs_out], [r["traj"] for r in runs_out],
                           max(2, sweeps // 2))
    if drifting:
        return "undetermined", (f"not equilibrated: {diag}; the values carry burn-in bias not "
                                "in stderr (near the SP onset relaxation is slow); increase "
                                "burn_in and runs")
    return "nontrivial", f"no drift of the mean survey detected: {diag}"


def ksat_population_dynamics_sp(K, alpha, *, pop, sweeps, seed, burn_in=100, runs=4,
                                rule="bmz"):
    K, a, seed = _pd_args(K, alpha, pop, sweeps, seed, burn_in, runs)
    R = _rule(rule)
    out = [_run(K, a, np.random.default_rng(ss), pop, burn_in, sweeps, R)
           for ss in np.random.SeedSequence(seed).spawn(runs)]
    phase, note = _phase(out, sweeps)
    keys = (("complexity", "s2"), ("complexity_mmz18", "s1"), ("mean_survey", "eta"),
            ("zero_survey_fraction", "zero"))
    val = {k: float(np.mean([r[s] for r in out])) for k, s in keys}
    err = {k: float(np.std([r[s] for r in out], ddof=1) / math.sqrt(runs)) for k, s in keys}
    params = dict(K=K, alpha=a, pop=pop, sweeps=sweeps, seed=seed, burn_in=burn_in, runs=runs,
                  rule=rule)
    return PDResult(val, err, "statistical", phase, note, params, None,
                    f"cavity route (1RSB, y -> oo, SP population, {_MMZ}, Sec. 5): statistical")


@dataclass(frozen=True)
class SPCrossing:
    K: int
    alphas: tuple
    alpha_s: float           # mean over runs of the per-run root of the fitted line
    stderr: float
    halfwidth99: float       # t_{0.995, runs-1} stderr
    runs: int
    roots: tuple
    sigma: tuple             # mean Sigma per alpha over runs
    sigma_stderr: tuple
    note: str = ""


def ksat_sp_crossing(K, alphas, *, pop, sweeps, seed, burn_in=100, runs=4, rule="bmz",
                     warm_burn_in=None):
    try:
        grid = tuple(sorted({float(_alpha(x)) for x in alphas}))
    except TypeError:
        grid = ()
    if len(grid) < 3:
        raise Refusal("precondition:crossing-grid",
                      f"alphas = {alphas!r}: need at least three distinct densities")
    K, _, seed = _pd_args(K, grid[0], pop, sweeps, seed, burn_in, runs)
    R = _rule(rule)
    warm = max(sweeps // 2, burn_in // 3) if warm_burn_in is None else warm_burn_in
    if isinstance(warm, bool) or not isinstance(warm, int) or warm < max(2, sweeps // 2):
        raise Refusal("precondition:burn-in", f"warm_burn_in must be an int >= max(2, sweeps "
                      f"// 2), got {warm!r}")
    table = []
    for ss in np.random.SeedSequence(seed).spawn(runs):      # one run: the grid in ascending
        rng, row, phi = np.random.default_rng(ss), [], None  # order, warm-started
        for a in grid:
            row.append(_run(K, a, rng, pop, burn_in if phi is None else warm, sweeps, R, phi))
            phi = row[-1]["phi"]
        table.append(row)
    for j, a in enumerate(grid):
        phase, note = _phase([row[j] for row in table], sweeps)
        if phase == "trivial":
            raise Refusal("no-solution:bracket", f"alpha = {a}: SP population trivial (below "
                          "the SP onset): the grid must lie on the non-trivial branch")
        if phase == "undetermined":
            raise Refusal("no-convergence:equilibration", f"alpha = {a}: {note}")
    sig = np.array([[c["s2"] for c in row] for row in table])
    mean, se = sig.mean(0), sig.std(0, ddof=1) / math.sqrt(runs)
    if not (mean[0] > 0 > mean[-1]):
        raise Refusal("no-solution:bracket", f"mean Sigma over runs does not change sign on "
                      f"the grid {grid}: {np.round(mean, 5).tolist()}")
    roots = []
    for row in sig:
        slope, icept = np.polyfit(grid, row, 1)
        if slope >= 0:
            raise Refusal("no-solution:bracket", f"a run's fitted Sigma(alpha) is not decreasing "
                          f"(slope {slope:.3g})")
        roots.append(-icept / slope)
    r = np.array(roots)
    m, s = float(r.mean()), float(r.std(ddof=1) / math.sqrt(runs))
    hw = t_quantile(0.995, runs - 1) * s
    return SPCrossing(K, grid, m, s, hw, runs, tuple(roots), tuple(mean), tuple(se),
                      f"root of the SP-population complexity: per run a line fitted to Sigma on "
                      f"{len(grid)} densities (ascending, each run warm-started from the previous "
                      f"density: burn_in {burn_in} then {warm}), {runs} runs (pop {pop}, sweeps "
                      f"{sweeps}); 99% Student-t half-width {hw:.3g}")


__all__ = ["CROSSING", "PHI_MAX", "SPCrossing", "SPResult", "SP_UPDATES", "audit_sp_rule", "bmz_rule",
           "instance_complexity", "ksat_population_dynamics_sp", "ksat_sp", "ksat_sp_crossing",
           "planted_power_rule", "warning_factor_by_enumeration"]
