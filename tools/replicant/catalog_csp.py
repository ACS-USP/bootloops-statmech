"""Catalog entry `xorsat` (random K-XORSAT), route `cavity`.

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
"""
import math

import mpmath as mp

from .catalog import Out, RefCheck, Stability, entry, fstep, has
from .core import Refusal
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


def csp_entries():
    return [xorsat()]


__all__ = ["csp_entries", "xorsat"]
