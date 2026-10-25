"""Catalog entries `xorsat` (random K-XORSAT, E5a) and `ksat` (random K-SAT, E5b), route `cavity`.

Why `cavity` and not `replica`: every closed form here is a fixed point of zero-temperature
message passing on the factor graph (leaf removal = warning propagation, the y -> oo limit of
the 1RSB cavity equations, Mezard, Ricci-Tersenghi & Zecchina 2003, Sec. II), not a replica
functional; the package's `replica` route means a level-(b) replica pipeline.  The finite-N
part of the route (sampled instances, exact GF(2) elimination, leaf removal on the samples)
is an independent check run inside the same route, labelled statistical.

Conventions (csp/xorsat.py): N variables, M = alpha N checks on K distinct uniform variables,
uniform right-hand sides; default K = 3.  Run parameters:
  (none)          alpha_d, alpha_s and the core at the thresholds (always computed)
  alpha           the 2-core (q, V, C, lambda) and the complexity Sigma = ln 2 (V - C) at alpha
  alpha, N        also the satisfiable fraction over named seeds (exact GF(2) elimination;
                  99% Clopper-Pearson band) and the sampled core (leaf removal), seeds
                  default 0-19
Finality: the thresholds and the core are `exact-by-literature` (theorems: Molloy 2005 for the
2-core, Dubois-Mandler 2002 (K = 3) and Pittel-Sorkin 2016 (K >= 3) for alpha_s); Sigma is
`exact-by-literature` below alpha_s (Mezard-Montanari 2009, Thm 18.2) and `not-computed` above;
the finite-N values are statistical and `not-computed`.

`ksat` (csp/ksat.py, csp/sp.py): N variables, M = round(alpha N) OR-clauses of K literals on
distinct uniform variables, uniform signs; default K = 3.  Run parameters:
  (none)          alpha_1, the first-moment bound's root (an upper bound on the threshold), and
                  alpha_t (K >= 3), the tangency of the zero-survey equation (MMZ Eq. (39))
  alpha           also the annealed entropy ln 2 + alpha ln(1 - 2^-K)
  alpha, N        N <= 22: exhaustive counts on named seeds (default 0-19): sat_fraction,
                  entropy_exact_mean, ln_mean_z; N > 22: BP convergence fraction and the SP
                  complexity of the instances (default seeds 0-9)
  alpha, pop      the SP population (sweeps default 30; seed 0, 4 runs): complexity,
                  complexity_mmz18, mean_survey, sp_phase
  crossing=true   the root of the SP-population complexity for K = 3 (sp.CROSSING; ~40 s)
Finality: alpha_1 and the annealed entropy `exact-by-literature` (Ding-Sly-Sun Eq. (2.3):
E Z is exact); everything else statistical and `not-computed` (the SP threshold is a cavity
conjecture for K = 3).
"""
import math

import mpmath as mp

from .catalog import Out, RefCheck, Stability, entry, fstep, has
from .core import Refusal
from .csp import _checks
from .csp import ksat as S
from .csp import sp as SP
from .csp import xorsat as X
from .csp import xorsat_finite as XF
from .spec import ModelSpec

MRZ = ("Mezard, Ricci-Tersenghi & Zecchina, J. Stat. Phys. 111, 505 (2003), "
       "arXiv:cond-mat/0207140")
MOLLOY = ("Molloy, Random Struct. Algorithms 27, 124 (2005), Thm 1 (not fetched; as stated by "
          "Pittel & Sorkin, Combin. Probab. Comput. 25, 236 (2016), Sec. 7); Mezard & Montanari, "
          "Information, Physics, and Computation (2009), Thm 18.1")
DM = "Dubois & Mandler, C. R. Acad. Sci. Paris Ser. I 335, 963 (2002), Thm 3"
PS = "Pittel & Sorkin, Combin. Probab. Comput. 25, 236 (2016), arXiv:1212.3822, Thm 16"
MM_SIGMA = ("Mezard & Montanari, Information, Physics, and Computation (2009), Thm 18.2 (iii); "
            f"{MRZ}, Sec. III.E")
CAVEAT = ("cavity route (zero-temperature warning propagation / leaf removal on the factor "
          "graph): the thresholds and the 2-core are theorems of the cited papers; the "
          "finite-N fractions are statistical (binomial, named seeds), never certified")
_DEFAULT_SEEDS = tuple(range(20))


def _K(p):
    return p.get("K", 3)


def _thresholds(p):
    th = X.xorsat_thresholds(_K(p))
    return Out({"K": th.K, "alpha_d": th.alpha_d, "alpha_s": th.alpha_s, "q_d": th.q_d,
                "lambda_d": th.lambda_d, "core_d": th.core_d, "lambda_s": th.lambda_s},
               f"tangency of the leaf-removal map (alpha_d, the core jumps to {mp.nstr(th.core_d, 6)}"
               ") and the zero of Sigma = ln 2 (V - C) on the largest root (alpha_s)")


def _core(p):
    c = X.xorsat_core_size(_K(p), p["alpha"])
    return Out({"q": c.q, "q_hat": c.q_hat, "lam": c.lam, "core_variables": c.variables,
                "core_checks": c.checks, "complexity": c.complexity, "core_trivial": c.trivial},
               "empty core (alpha < alpha_d)" if c.trivial else
               f"largest leaf-removal root q = {mp.nstr(c.q, 8)}; Sigma = {mp.nstr(c.complexity, 6)}")


def _finite(p):
    K, N, a = _K(p), p["N"], p["alpha"]
    seeds = XF._seeds(p.get("seeds", _DEFAULT_SEEDS))
    r = XF.xorsat_sat_probability(K, N, a, seeds)
    cores = [XF.xorsat_sampled_core(K, N, a, s) for s in seeds]
    vals = {}
    errs = {"sat_fraction": r.halfwidth99}
    for key, xs in (("core_variables_sampled", [v for v, _ in cores]),
                    ("core_checks_sampled", [c for _, c in cores])):
        m = math.fsum(xs) / len(xs)
        sd = math.sqrt(math.fsum((x - m)**2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else math.nan
        vals[key], errs[key] = m, sd / math.sqrt(len(xs))
    stat = ("sat_fraction", "core_variables_sampled", "core_checks_sampled")
    return Out({"sat_fraction": r.fraction, "sat_ci99_lo": r.lo, "sat_ci99_hi": r.hi,
                "n_seeds": r.n, **vals},
               f"exact GF(2) elimination on {r.n} sampled instances (seeds {seeds[0]}..{seeds[-1]}):"
               f" {r.n_sat} satisfiable, 99% Clopper-Pearson [{r.lo:.3f}, {r.hi:.3f}]; leaf removal"
               " on the same instances", errors=errs, statistical=stat,
               notes=("statistical: sat_fraction (binomial over named seeds) and the sampled core "
                      "(mean +- stderr) are finite-N estimates, not certified; the error given for "
                      "sat_fraction is the larger half-width of its 99% Clopper-Pearson interval "
                      "(the binomial stderr is 0 when every seed agrees)",))


def _check_params(p):
    if p.get("N") is None and "seeds" in p:
        raise Refusal("precondition:known-parameter",
                      "seeds is a finite-N parameter: need N (and alpha) for the GF(2) check")
    if p.get("N") is not None and p.get("alpha") is None:
        raise Refusal("precondition:state-point", "the finite-N check needs alpha (and N)")


def _guard(compute):
    def run(p):
        _check_params(p)
        return compute(p)
    return run


def _alpha_s_label(v):
    k = v.get("K", 3)
    cite = f"{DM}; {PS}" if k == 3 else PS
    return Stability("exact-by-literature", "1rsb", cite,
                     "satisfiability threshold (theorem: N_c = M_c of the 2-core)"
                     + ("; DM's full proof is the FOCS 2002 version (not fetched)" if k == 3 else ""))


def _sigma_label(v):
    if v["complexity"] > 0 or v.get("core_trivial"):
        return Stability("exact-by-literature", "1rsb", MM_SIGMA,
                         "ln of the number of core solutions per variable, for alpha < alpha_s; "
                         "reading it as the number of solution clusters is the cavity picture, "
                         "made rigorous by Ibrahimi, Kanoria, Kraning & Montanari, Ann. Appl. "
                         "Probab. 25, 2743 (2015), arXiv:1107.5377, and Achlioptas & Molloy, "
                         "RSA 46, 197 (2015) (both abstract only); Mezard & Montanari's ch. 18 "
                         "Notes say that clusters being well connected remained to be proved")
    return Stability("not-computed", "1rsb", MM_SIGMA,
                     "Sigma <= 0: at or above alpha_s there are no solutions w.h.p. "
                     f"({PS}); the MAX-XORSAT ground energy is not computed: NOT final")


def xorsat():
    steps = [fstep("xorsat-thresholds", f"{MRZ}, Secs. II.A, III.B-D; {MOLLOY}; {PS}",
                   _guard(_thresholds)),
             fstep("xorsat-core", f"{MRZ}, Eqs. (30)-(33); {MOLLOY}", _core, when=has("alpha"),
                   skip="no alpha given"),
             fstep("xorsat-finite-n-gf2", "exact Gaussian elimination over GF(2) and leaf removal "
                   "on sampled instances (csp/xorsat_finite.py); Mezard & Montanari (2009), "
                   "Sec. 18.3.1 (peeling)", _finite, when=has("alpha", "N"),
                   skip="needs alpha and N")]
    refs = [RefCheck("xorsat", "alpha_d_k3", lambda v: v["alpha_d"], route="cavity"),
            RefCheck("xorsat", "alpha_s_k3", lambda v: v["alpha_s"], route="cavity"),
            RefCheck("xorsat", "lambda_d_k3", lambda v: v["lambda_d"], route="cavity"),
            RefCheck("xorsat", "lambda_s_k3", lambda v: v["lambda_s"], route="cavity")]
    refs += [RefCheck("xorsat", f"alpha_d_k{k}", lambda v: v["alpha_d"], params={"K": k},
                      route="cavity") for k in (4, 5)]
    refs += [RefCheck("xorsat", f"alpha_s_k{k}", lambda v: v["alpha_s"], params={"K": k},
                      route="cavity") for k in (4, 5, 6)]
    core = Stability("exact-by-literature", "1rsb", MOLLOY,
                     "2-core order and size of the random K-uniform hypergraph (largest "
                     "leaf-removal root; empty below alpha_d)")
    finite = Stability("not-computed", "1rsb", "csp/xorsat_finite.py",
                       "finite-N statistical estimate over named seeds: NOT final")
    fin = {"alpha_d": core, "q_d": core, "lambda_d": core, "core_d": core,
           "alpha_s": _alpha_s_label, "lambda_s": _alpha_s_label,
           **{k: core for k in ("q", "q_hat", "lam", "core_variables", "core_checks")},
           "complexity": _sigma_label,
           **{k: finite for k in ("sat_fraction", "core_variables_sampled",
                                  "core_checks_sampled")}}
    spec = ModelSpec("xorsat", "boolean", "none",
                     constraint="K-ary parity checks on a sparse random hypergraph",
                     params={"K": 3})
    return entry("xorsat", spec, MRZ, {"cavity": steps}, refs, {"cavity": CAVEAT},
                 "random K-XORSAT: leaf removal (alpha_d, 2-core), satisfiability threshold "
                 "(Sigma = 0), exact finite-N GF(2) checks", default_route="cavity",
                 finality=fin, meta=("K", "core_trivial", "sat_ci99_lo", "sat_ci99_hi",
                                     "n_seeds"),
                 parameters={"cavity": ("K", "alpha", "N", "seeds")})


# ---- random K-SAT (E5b) ------------------------------------------------------------------
_KS_MMZ = ("Mertens, Mezard & Zecchina, Random Struct. Algorithms 28, 340 (2006), "
           "arXiv:cs/0309020")
_KS_BMZ = ("Braunstein, Mezard & Zecchina, Random Struct. Algorithms 27, 201 (2005), "
           "arXiv:cs/0212002")
_KS_DSS = "Ding, Sly & Sun, Ann. of Math. 196, 1 (2022), arXiv:1411.0650"
KSAT_CAVEAT = ("cavity route (1RSB at y -> oo: survey propagation on the factor graph); the SP "
               "complexity and its root are the cavity conjecture (MMZ Sec. 2.3), statistical "
               "(population dynamics over named seeds), never certified; for K = 3 the threshold "
               f"is not a theorem ({_KS_DSS}: K >= k0 only)")


def _ks_check(p):
    if p.get("N") is None and "seeds" in p:
        raise Refusal("precondition:known-parameter", "seeds is a finite-N parameter: need N")
    if p.get("pop") is None and "sweeps" in p:
        raise Refusal("precondition:known-parameter", "sweeps is a population parameter: need pop")
    for k in ("N", "pop"):
        if p.get(k) is not None and p.get("alpha") is None:
            raise Refusal("precondition:state-point", f"{k} needs alpha (the clause density)")


def _ks_first_moment(p):
    _ks_check(p)
    K = _K(p)
    vals = {"K": K, "alpha_1": S.ksat_first_moment_threshold(K)}
    if K >= 3:
        vals["alpha_t"] = S.ksat_sp_support_threshold(K)
    if p.get("alpha") is not None:
        vals["annealed_entropy"] = S.ksat_annealed_bound(K, p["alpha"])
    return Out(vals, "first moment: (1/N) ln E Z = ln 2 + alpha ln(1 - 2^-K); alpha_1 its root"
               + ("; alpha_t: tangency of the zero-survey equation (MMZ Eq. (39))" if K >= 3 else
                  "; K = 2: no zero-survey tangency (alpha_t not computed)"))


def _mean_se(xs):
    m = math.fsum(xs) / len(xs)
    sd = math.sqrt(math.fsum((x - m)**2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else math.nan
    return m, sd / math.sqrt(len(xs))


def _ks_finite(p):
    K, N, a = _K(p), p["N"], p["alpha"]
    if isinstance(N, int) and not isinstance(N, bool) and N <= S.EXHAUSTIVE_MAX:
        seeds = _checks.seeds(p.get("seeds", tuple(range(20))))
        fs = [S.random_ksat(K, N, a, s) for s in seeds]
        z = [S.ksat_exact_count(f) for f in fs]
        sat = [math.log(x) / N for x in z if x > 0]
        k = len(sat)
        lo, hi = XF.clopper_pearson(k, len(z))
        vals = {"sat_fraction": k / len(z), "sat_ci99_lo": lo, "sat_ci99_hi": hi,
                "n_seeds": len(z), "ln_mean_z": (math.log(math.fsum(z) / len(z)) / N
                                                if any(z) else -math.inf)}
        errs = {"sat_fraction": max(k / len(z) - lo, hi - k / len(z))}
        stat = ["sat_fraction", "ln_mean_z"]
        if k:
            vals["entropy_exact_mean"], errs["entropy_exact_mean"] = _mean_se(sat)
            stat.append("entropy_exact_mean")
        return Out(vals, f"exhaustive counts of {len(z)} instances (seeds {seeds[0]}..{seeds[-1]}): "
                   f"{k} satisfiable", errors=errs, statistical=tuple(stat),
                   notes=("statistical: exact counts per instance, averaged over named seeds; "
                          "entropy_exact_mean = mean of ln Z / N over the satisfiable instances "
                          "(ln 0 excluded); ln_mean_z = (1/N) ln of the sample mean of Z",))
    seeds = _checks.seeds(p.get("seeds", tuple(range(10))))
    bp = S.ksat_bp_converged_fraction(K, N, a, seeds, iters=S.BP_WINDOW["iters"],
                                      damping=S.BP_WINDOW["damping"])
    sps = [SP.ksat_sp(S.random_ksat(K, N, a, s), iters=1000, seed=s) for s in seeds]
    conv = [r.complexity for r in sps if r.converged]
    vals = {"bp_converged_fraction": bp.fraction, "bp_ci99_lo": bp.lo, "bp_ci99_hi": bp.hi,
            "sp_converged": len(conv), "n_seeds": len(seeds),
            "sp_trivial_fraction": sum(r.trivial for r in sps if r.converged) / max(1, len(conv))}
    errs = {"bp_converged_fraction": max(bp.fraction - bp.lo, bp.hi - bp.fraction)}
    stat = ["bp_converged_fraction", "sp_trivial_fraction"]
    if conv:
        vals["sp_complexity_instances"], errs["sp_complexity_instances"] = _mean_se(conv)
        stat.append("sp_complexity_instances")
    return Out(vals, f"BP (damping {bp.damping}, {bp.iters} sweeps) converged on {bp.n_conv}/{bp.n}"
               f" instances; SP converged on {len(conv)}/{len(seeds)}", errors=errs,
               statistical=tuple(stat),
               notes=("statistical: BP convergence (99% Clopper-Pearson) and the SP complexity of "
                      "sampled instances (BMZ Eqs. (39)-(41), mean +- stderr over the converged "
                      "ones), named seeds",))


def _ks_population(p):
    r = SP.ksat_population_dynamics_sp(_K(p), p["alpha"], pop=p["pop"], sweeps=p.get("sweeps", 30),
                                       seed=0, burn_in=100, runs=4)
    return Out({**r.value, "sp_phase": r.phase}, f"SP population ({r.phase}): {r.note}",
               errors=dict(r.stderr), statistical=tuple(r.value),
               notes=("statistical: SP population dynamics (4 runs, seed 0, burn_in 100), mean "
                      "+- stderr over runs; phase " + r.phase,))


def _ks_crossing(p):
    if _K(p) != 3:
        raise Refusal("precondition:crossing-grid", "a density grid around the root of Sigma is "
                      "set (csp/sp.CROSSING) only for K = 3")
    c = SP.CROSSING
    r = SP.ksat_sp_crossing(3, c["alphas"], pop=c["pop"], sweeps=c["sweeps"], seed=c["seed"],
                            burn_in=c["burn_in"], runs=c["runs"])
    return Out({"alpha_s_sp": r.alpha_s, "alpha_s_sp_halfwidth99": r.halfwidth99,
                "crossing_runs": r.runs}, r.note, errors={"alpha_s_sp": r.stderr},
               statistical=("alpha_s_sp",),
               notes=("statistical: root of the SP-population complexity, mean +- stderr over "
                      f"{r.runs} runs; 99% Student-t half-width {r.halfwidth99:.3g}",))


def _truthy(key):
    return lambda p: p.get(key) in (True, 1, "1", "true")


def ksat():
    steps = [fstep("ksat-first-moment", f"{_KS_DSS}, Eq. (2.3) and Markov's inequality",
                   _ks_first_moment),
             fstep("ksat-finite-n", f"exhaustive counts (N <= {S.EXHAUSTIVE_MAX}) or BP / SP on "
                   f"sampled instances (csp/ksat.py, csp/sp.py); {_KS_BMZ}, Secs. III-IV",
                   _ks_finite, when=has("alpha", "N"), skip="needs alpha and N"),
             fstep("ksat-sp-population", f"{_KS_MMZ}, Secs. 3 and 5.2, Eq. (18)", _ks_population,
                   when=has("alpha", "pop"), skip="needs alpha and pop"),
             fstep("ksat-sp-crossing", f"{_KS_MMZ}, Sec. 5.3 (root of Sigma by regression)",
                   _ks_crossing, when=_truthy("crossing"), skip="crossing not requested")]
    refs = [RefCheck("ksat", "alpha_t_k3", lambda v: v["alpha_t"], route="cavity"),
            RefCheck("ksat", "alpha_s_k3", lambda v: v["alpha_s_sp"], statistical=True,
                     params={"crossing": True}, route="cavity", slow=True,
                     error=lambda e: e["alpha_s_sp"], runs=SP.CROSSING["runs"])]
    first = Stability("exact-by-literature", "none", f"{_KS_DSS}, Eq. (2.3)",
                      "first-moment (annealed) bound: E Z = 2^N (1 - 2^-K)^M exactly; alpha_1 "
                      "upper-bounds the SAT threshold (Markov); it is NOT the threshold")
    fin_n = Stability("not-computed", "rs", "csp/ksat.py, csp/sp.py",
                      "finite-N statistical estimate over named seeds: NOT final")
    cav = Stability("not-computed", "1rsb", _KS_MMZ,
                    "SP population dynamics (1RSB cavity, y -> oo): statistical, a cavity "
                    "conjecture: NOT final")
    cross = Stability("not-computed", "1rsb", f"{_KS_MMZ}; {_KS_DSS}",
                      "root of the SP complexity (statistical): the cavity conjecture for the "
                      "SAT threshold; for K = 3 not proven (Ding-Sly-Sun: K >= k0, an unspecified "
                      "constant); MMZ find the 1RSB solution stable at the root (Sec. 5.4): "
                      "NOT final")
    support = Stability("not-computed", "1rsb", f"{_KS_MMZ}, Eq. (39)",
                        "deterministic tangency of the zero-survey equation: where exactly-zero "
                        "surveys stop being all of them; not a phase transition (the non-zero "
                        "surveys collapse below the SP onset): NOT final")
    fin = {"alpha_1": first, "annealed_entropy": first, "alpha_t": support,
           **{k: fin_n for k in ("sat_fraction", "entropy_exact_mean", "ln_mean_z",
                                 "bp_converged_fraction", "sp_trivial_fraction",
                                 "sp_complexity_instances")},
           **{k: cav for k in ("complexity", "complexity_mmz18", "mean_survey",
                               "zero_survey_fraction")},
           "alpha_s_sp": cross}
    spec = ModelSpec("ksat", "boolean", "none",
                     constraint="K-literal OR clauses on a sparse random hypergraph",
                     params={"K": 3})
    return entry("ksat", spec, _KS_MMZ, {"cavity": steps}, refs, {"cavity": KSAT_CAVEAT},
                 "random K-SAT: first-moment bound, exact small-N counts, BP and survey "
                 "propagation on instances, SP-population complexity and its root",
                 default_route="cavity", finality=fin,
                 meta=("K", "sat_ci99_lo", "sat_ci99_hi", "n_seeds", "bp_ci99_lo", "bp_ci99_hi",
                       "sp_converged", "sp_phase", "alpha_s_sp_halfwidth99", "crossing_runs"),
                 parameters={"cavity": ("K", "alpha", "N", "seeds", "pop", "sweeps", "crossing")})


def csp_entries():
    return [xorsat(), ksat()]


__all__ = ["csp_entries", "ksat", "xorsat"]
