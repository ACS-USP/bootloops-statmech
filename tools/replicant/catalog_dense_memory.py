"""Catalog entries `dense_memory` and `dense_memory_ising` (dense associative memories with exponentially many
patterns, route `replica`).

`dense_memory`  Lucibello & Mezard, "The Exponential Capacity of Dense Associative Memories", Phys. Rev. Lett.
132, 077301 (2024), arXiv:2304.14964v4 (PDF read; journal version not checked).  Continuous neurons x in R^N,
energy E(x) = -(1/lam) log sum_{mu=1}^P exp(lam x.xi^mu) + |x|^2/2, P = e^{alpha N} patterns, Gaussian or
spherical (|xi|^2 = N), so that the softmax denominator of an InfoNCE loss is the same exponential-interaction
energy.  This is a ZERO-TEMPERATURE retrieval analysis: the sum over the other patterns is the free energy of an
auxiliary Random Energy Model (the paper stresses its "energies" and "temperature" are not those of the memory),
and the 1RSB solution over pattern LABELS is what the replica method contributes.  There is no finite-
temperature replica-symmetric free energy of the memory, no spin-glass / paramagnetic phase diagram of the
memory and no replicon in the paper; the paper names the finite-temperature landscape an open problem.  What the
entry reports: the noise phi, its two REM phases (uncondensed / frozen) and the Parisi parameter m*, the
condensation line lam*(alpha), the single-pattern threshold alpha_1(lam) (Eq. (7)) and its inverse lambda_1(alpha),
the all-patterns capacity (equal to alpha_1 for spherical patterns, a lower bound alpha_c^lb for Gaussian ones),
the one-step basin cos(theta_c), the nearest-pattern cosine, the small-load and scaled-dot-product regimes and
the Ramsauer et al. bound as restated in App. E.

`dense_memory_ising`  Ising neurons x in {-1,+1}^N, +-1 patterns, F(x) = exp(lam x): the model of Demircigil,
Heusel, Lowe, Upgang & Vermet (J. Stat. Phys. 168, 288 (2017), arXiv:1702.01929v2, PDF read), proved there for
lam = 1.  The noise phi is the paper's recipe (A17) with zeta = log cosh: DERIVED HERE, not published (the paper's
Conclusions only say binary patterns "should" be treatable); the thresholds alpha_1_flip / alpha_1_energy are also
derived here; only `alpha_fixed_points` = I(1 - 2 radius)/2 (Thm 1.3, Remark 1.4) is a published theorem.  Every
derived value is labelled not-computed (never final).

Both: polynomial interactions F(x) = x^p are refused (`unsupported:interaction`).  Run parameters of `dense_memory`:
  ensemble   gaussian | spherical (default spherical)
  alpha      load, P = e^{alpha N} (>= 0)
  lam        interaction strength (> 0); the InfoNCE inverse temperature is lam N (see the mapping note)
  rho        norm of x, |x| = rho sqrt(N) (default 1)
  level      signal overlap that replaces 1 in Eq. (7): an EXTENSION, reported as alpha_1_level, not-computed
  P, N       P patterns of N neurons: (B8) cos(theta_c) = sqrt(2 log P/N); with alpha, lam, N and seeds the
             finite-N sampler (sim.rem_phi_mc) is run beside the formula
  lam_tilde  Sec. VI: lam = lam~/N^a, P = exp(alpha~ N^{1-a}): alpha~_1 = lam~
  seeds      named seeds of the sampler (default 0-19)
  interaction  exp (default); anything else is refused
`dense_memory_ising`: lam, alpha, radius (the Hamming fraction of Thm 1.3, default 0), P, N, seeds, interaction.
"""
import math

import mpmath as mp
import numpy as np

from .catalog import Out, RefCheck, Stability, entry, fstep
from .core import Refusal
from .csp import _checks
from .derive_dense_memory import label_replica_steps
from .dense_memory import ensembles as EN
from .dense_memory import ising as IS
from .dense_memory import rem_noise as RN
from .dense_memory import sim as SIM
from .dense_memory import thresholds as TH
from .spec import ModelSpec

LM = "Lucibello & Mezard, Phys. Rev. Lett. 132, 077301 (2024), arXiv:2304.14964v4"
DHLUV = ("Demircigil, Heusel, Lowe, Upgang & Vermet, J. Stat. Phys. 168, 288 (2017), arXiv:1702.01929v2")
_SEEDS = tuple(range(20))
_NON_RIGOROUS = ("derived in the paper by a Random Energy Model analysis of the noise term, which the authors call "
                 "partially non-rigorous (a large-deviation theorem of Fedrigo, Flandoli & Morandin, their ref. [14], "
                 "underlies the all-patterns bound); a zero-temperature retrieval threshold, not a phase of the "
                 "memory at finite temperature")

CAVEAT = ("replica route: the REM noise of the dense memory by the 1RSB replica calculation over pattern labels "
          "(Lucibello & Mezard App. A 4), checked here by four routes (closed forms, Legendre, the minimised 1RSB "
          "functional, Laplace's method on the rate function), by the exact finite-N law of the overlaps and by the "
          "actual gradient descent; Gaussian and spherical patterns, continuous neurons, exponential interaction.  "
          "Physics level; zero temperature; the thresholds are for patterns drawn i.i.d. at random, not for "
          "optimised embeddings")
CAVEAT_ISING = ("replica route: the REM noise with binary levels (zeta = log cosh), DERIVED HERE and not published, "
                "plus Demircigil et al.'s rigorous fixed-point exponent (lam = 1); the thresholds are checked by the "
                "exact finite-N law, the single-flip dynamics and Gray-code enumeration; physics level except the "
                "banked theorem")

_PARAMS = ("ensemble", "alpha", "lam", "rho", "level", "interaction", "p", "P", "N", "lam_tilde", "seeds")
_PARAMS_ISING = ("alpha", "lam", "radius", "interaction", "p", "P", "N", "seeds")
_META = ("ensemble", "rho", "alpha", "lam", "level", "phase", "phi_route_spread", "lambda_1_status", "P", "N",
         "lam_tilde", "seeds", "radius", "phi_mc_halfwidth99", "alpha_c_kind")


def _opt(p, key):
    return None if p.get(key) is None else p[key]


def _interaction(p):
    kind = p.get("interaction", "exp")
    if kind != "exp":
        raise Refusal("unsupported:interaction",
                      f"interaction = {kind!r}: only the exponential F(x) = exp(lam x) has an exponential number of "
                      "retrievable patterns here.  A polynomial F(x) = x^p stores O(N^(p-1)) patterns "
                      "(Krotov-Hopfield; quoted by Demircigil et al., Thm 1.1-1.2), so P = e^(alpha N) lies outside "
                      "its regime, and its thermodynamics is the p-spin glass: see the catalog entries `pspin` and "
                      "`ising_pspin`")


def _state(ens_fixed=None):
    def compute(p):
        _interaction(p)
        ens = ens_fixed or p.get("ensemble", "spherical")
        if ens == "binary" and ens_fixed is None:
            raise Refusal("unsupported:ensemble",
                          "binary (+-1) patterns need Ising neurons: use the catalog entry `dense_memory_ising` "
                          "(the continuous-neuron entry covers the gaussian and spherical ensembles of the paper)")
        rho = EN.check_ensemble(ens, p.get("rho", 1))
        known = _PARAMS_ISING if ens_fixed else _PARAMS
        extra = [k for k in p if k not in known]
        if extra:
            raise Refusal("precondition:known-parameter", f"unknown parameter(s) {extra}; accepted: {list(known)}")
        if not any(p.get(k) is not None for k in ("alpha", "lam", "lam_tilde")) and not (
                p.get("P") is not None and p.get("N") is not None):
            raise Refusal("precondition:state-point",
                          "give lam and/or alpha (and optionally rho, level, N, P, seeds), lam_tilde, or P with N")
        if p.get("alpha") is not None:
            RN._alpha(p["alpha"])
        if p.get("lam") is not None:
            RN._lam(p["lam"])
        out = {"ensemble": ens, "rho": rho}
        for k in ("alpha", "lam"):
            if p.get(k) is not None:
                out[k] = EN.num(p[k], k)
        lvl = p.get("level", 1)
        if ens_fixed is None:
            out["level"] = EN.num(lvl, "level", positive=True)
        return Out(out, f"{ens} patterns, rho = {mp.nstr(rho, 6)}, exponential interaction F(x) = exp(lam x)")
    return compute


def _have(*keys):
    return lambda p: all(p.get(k) is not None for k in keys)


def _condensation(p):
    ens, r, a = p.get("ensemble", "spherical"), p.get("rho", 1), p["alpha"]
    ls, es = RN.condensation_lambda(ens, a, r), RN.eps_star(ens, a, r)
    return Out({"lam_star": ls, "eps_star": es},
               f"lam* = {mp.nstr(ls, 10)} (the REM condenses above it); eps* = {mp.nstr(es, 10)}, the largest of "
               f"the e^(alpha N) overlaps (at rho = 1 the nearest-pattern cosine c_max, App. B)")


def _noise(ens_fixed=None):
    def compute(p):
        ens, r = ens_fixed or p.get("ensemble", "spherical"), p.get("rho", 1)
        pt = RN.noise(ens, p["alpha"], p["lam"], r)
        spread = max(abs(pt.phi - RN.phi(ens, p["alpha"], p["lam"], r, m)) for m in ("legendre", "1rsb", "laplace"))
        return Out({"phi": pt.phi, "phase": pt.phase, "m_star": pt.m, "phi_route_spread": spread},
                   f"phi = {mp.nstr(pt.phi, 12)} ({pt.phase}); Parisi m* = {mp.nstr(pt.m, 10)}; the Legendre, "
                   f"1RSB and Laplace routes agree to {mp.nstr(spread, 3)}")
    return compute


def _threshold(p):
    ens, lam = p.get("ensemble", "spherical"), p["lam"]
    a1 = TH.alpha_1(ens, lam)
    closed = TH.alpha_1_closed(ens, lam)
    lb, kind = TH.capacity(ens, lam)
    vals = {"alpha_1": a1, "alpha_c_lb": lb, "alpha_c_kind": kind}
    if ens == "gaussian":
        vals["lambda_saturation"] = TH.lambda_saturation()
    return Out(vals, f"alpha_1 = {mp.nstr(a1, 12)} (Eq. (7), rho = 1; the Legendre closed form differs by "
               f"{mp.nstr(abs(a1 - closed), 3)}); capacity {kind}: {mp.nstr(lb, 12)}")


def _lambda_one(p):
    ens, a = p.get("ensemble", "spherical"), p["alpha"]
    try:
        l1 = TH.lambda_1(ens, a)             # Eq. (7) is at rho = 1
    except Refusal as r:
        if r.reason != "no-solution:no-retrieval":
            raise
        return Out({"lambda_1_status": r.reason}, f"{r.reason}: {r.detail}", notes=(f"{r.reason}: {r.detail}",))
    return Out({"lambda_1": l1}, f"smallest retrieving lam at alpha = {mp.nstr(mp.mpf(a), 6)}: {mp.nstr(l1, 12)}")


def _basin(p):
    c = TH.cos_theta_c(p.get("ensemble", "spherical"), p["alpha"], p["lam"], p.get("rho", 1))
    return Out({"cos_theta_c": c}, f"cos(theta_c) = phi/rho = {mp.nstr(c, 10)}" + (" >= 1: no one-step basin" if c >= 1 else ""))


def _few(p):
    c = TH.cos_theta_few_patterns(p["P"], p["N"])
    return Out({"cos_theta_few": c}, f"(B8): sqrt(2 log P/N) = {mp.nstr(c, 10)}; valid for log P << N")


def _scaled(p):
    return Out({"alpha_tilde_1": TH.scaled_threshold(p["lam_tilde"])}, "alpha~_1 = alpha~_c = lam~ (Sec. VI)")


def _ramsauer(p):
    v = TH.ramsauer_lower_bound(p["lam"])
    return Out({"alpha_c_ramsauer_lb": v}, f"(E7): {mp.nstr(v, 10)}, below alpha_1 = a large gap")


def _level(p):
    v = TH.alpha_1(p.get("ensemble", "spherical"), p["lam"], p["level"], p.get("rho", 1))
    return Out({"alpha_1_level": v}, f"EXTENSION (not in the paper): phi < {mp.nstr(mp.mpf(p['level']), 6)} gives "
               f"alpha = {mp.nstr(v, 12)}")


def _finite_n(ens_fixed=None):
    def compute(p):
        ens = ens_fixed or p.get("ensemble", "spherical")
        N = p["N"]
        if isinstance(N, bool) or not isinstance(N, int) or N < 20:
            raise Refusal("precondition:size", f"N = {N!r}: need an int >= 20")
        seeds = _checks.seeds(p.get("seeds", _SEEDS))
        v = np.array(SIM.rem_phi_mc(ens, p["alpha"], p["lam"], N, seeds, p.get("rho", 1)))
        se = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")
        from .cavity._stats import t_quantile
        hw = float(t_quantile(0.995, len(v) - 1)) * se if len(v) > 1 else float("nan")
        vals = {"phi_mc": float(v.mean()), "phi_mc_halfwidth99": hw}
        notes = ("statistical: the quenched noise phi at finite N sampled from the exact overlap law (the number of "
                 "patterns per overlap bin is Poisson); it differs from the N -> oo limit by O(log N / N) in the "
                 "frozen phase, so the band need not contain it",)
        return Out(vals, f"N = {N}, seeds {list(seeds)}: {v.mean():.6g} +- {hw:.3g} (99% Student-t)",
                   errors={"phi_mc": se}, statistical=("phi_mc",), notes=notes)
    return compute


# ---- the Ising entry -----------------------------------------------------------------------
def _ising_thresholds(p):
    lam = p["lam"]
    flip, energy = IS.alpha_1_flip(lam), IS.alpha_1_energy(lam)
    return Out({"alpha_1_flip": flip, "alpha_1_energy": energy},
               f"typical-pattern single-flip stability (the sign-random noise of Demircigil et al.'s dynamics sits at "
               f"2 lam): {mp.nstr(flip, 12)}; the energy criterion phi < 1: {mp.nstr(energy, 12)}")


def _ising_fixed(p):
    v = IS.fixed_point_exponent(p.get("radius", 0))
    return Out({"alpha_fixed_points": v},
               f"I(1 - 2 radius)/2 = {mp.nstr(v, 14)} (Thm 1.3; radius 0: log(2)/2, Remark 1.4): the exponent below "
               "which all patterns are fixed points, proved for lam = 1; it is the birthday threshold of pattern pairs "
               "within Hamming distance 1")


def _ising_pairs(p):
    P, N = p["P"], p["N"]
    e1 = IS.expected_close_pairs(N, P, 1)
    return Out({"close_pairs_expected": e1, "p_no_close_pair": math.exp(-e1)},
               f"expected pairs at Hamming distance <= 1: {e1:.6g}; P(none) ~ exp(-that) = {math.exp(-e1):.6g}")


def _ising_flip_mc(p):
    N = p["N"]
    if isinstance(N, bool) or not isinstance(N, int) or N < 20:
        raise Refusal("precondition:size", f"N = {N!r}: need an int >= 20")
    seeds = _checks.seeds(p.get("seeds", _SEEDS))
    v = np.array(SIM.sign_noise_mc(float(p["alpha"]), float(p["lam"]), N, seeds))
    return Out({"phi_flip_mc": float(np.median(v))}, f"median over seeds {list(seeds)} of (1/(lam N)) log|sign-random "
               f"sum| = {np.median(v):.6g}; the formula phi(alpha, 2 lam) = {mp.nstr(RN.phi('binary', p['alpha'], 2 * mp.mpf(p['lam'])), 8)}",
               statistical=("phi_flip_mc",))


def _ising_phi_flip(p):
    v = RN.phi("binary", p["alpha"], 2 * EN.num(p["lam"], "lam"))
    return Out({"phi_flip": v}, f"sign-random noise exponent phi_binary(alpha, 2 lam) = {mp.nstr(v, 12)}")


# ---- finality -------------------------------------------------------------------------------
def _lm(eqs, note=""):
    """Stability of a value the paper establishes: `eqs` maps the ensemble to the equation citations."""
    def f(v):
        e = eqs.get(v.get("ensemble"), eqs.get("*", ""))
        return Stability("exact-by-literature", "rem-labels", f"{LM}, {e}", (note + "; " if note else "") + _NON_RIGOROUS)
    return f


def _derived(note):
    return Stability("not-computed", "rem-labels", "",
                     f"derived in this package, not published ({note}); cross-checked by sampling or enumeration; NOT final")


_PHI = {"gaussian": "Eq. (5), (D5)", "spherical": "Eq. (A17), the line above (C5)"}
_LAMS = {"gaussian": "Eq. (D6)", "spherical": "Eq. (C5)"}
_ALPHA1 = {"gaussian": "Eq. (7) with (D7)", "spherical": "Eq. (7) with (C3)-(C5), App. C 1"}


def _dm_finality():
    return {
        "phi": _lm(_PHI, "the noise phi_(alpha,rho)(lam); the energy is E ~ -max(x.xi^1, N phi) + |x|^2/2, Eq. (6)"),
        "m_star": _lm({"gaussian": "Eqs. (A25)-(A27)", "spherical": "Eqs. (A25)-(A27)"},
                      "Parisi parameter of the 1RSB solution over pattern labels, m* = min(1, lam*/lam); the second "
                      "derivative of the 1RSB functional is lam^2 zeta''/m > 0 (a minimum), derived in step dm-extremize"),
        "lam_star": _lm(_LAMS, "the REM condensation (freezing) line"),
        "eps_star": _lm({"*": "App. A 3 step 3, App. B (Gilbert-Varshamov); Eqs. (C5), (D6)"},
                        "eps*(alpha, rho): the largest overlap among e^(alpha N) patterns"),
        "alpha_1": _lm(_ALPHA1, "single-pattern retrieval threshold: the largest load at which a typical stored pattern "
                       "is retrieved"),
        "alpha_c_lb": _lm({"gaussian": "Eq. (10) with (D8)-(D13): a LOWER BOUND on the all-patterns capacity "
                           "(the paper's simulations sit near it, Fig. 8; whether it equals alpha_c is open)",
                           "spherical": "Eqs. (C6)-(C7): alpha_c = alpha_1 (union bound, no gap)"}),
        "lambda_saturation": _lm({"gaussian": "App. D 2 (printed 0.70091)"},
                                 "alpha_c^lb = log(2)/4 for lam >= this value"),
        "lambda_1": _lm({"*": "Eq. (7) inverted"}, "the smallest lam retrieving at load alpha"),
        "alpha_tilde_1": _lm({"*": "Sec. VI"}, "scaled dot product: alpha~_1 = alpha~_c = lam~"),
        "alpha_c_ramsauer_lb": _lm({"*": "App. E, Eq. (E7)"},
                                   "the thermodynamic limit of Theorem 3 of Ramsauer et al. (their ref. [8], NOT read "
                                   "here) for SPHERICAL patterns, as restated by the paper"),
        "cos_theta_c": _lm({"gaussian": "Sec. IV, App. D 3", "spherical": "Sec. IV, App. C 3"},
                           "the one-step retrieval boundary rho cos(theta) = phi, exact from Eq. (6) for a start "
                           "independent of the other patterns; the paper calls theta_c a LOWER BOUND on the basin of "
                           "attraction and conjectures it tight (a start in the noise-dominated region needs the "
                           "multi-step dynamics, which is not analysed)"),
        "cos_theta_few": Stability("exact-by-literature", "rem-labels", f"{LM}, Eqs. (B8)-(B9)",
                                   "the small-load limit cos(theta_c) = sqrt(2 log P/N), tested by the paper only for "
                                   "spherical patterns (Fig. 5); " + _NON_RIGOROUS),
        "alpha_1_level": Stability("not-computed", "rem-labels", "",
                                   "EXTENSION of Eq. (7) with the signal overlap `level` in place of 1: not in the "
                                   "paper, the noise argument applies to a query independent of the other patterns, "
                                   "and the real InfoNCE query is optimised against them; NOT final"),
        "phi_mc": Stability("not-computed", "finite-N", "dense_memory/sim.py",
                            "statistical: the quenched noise at finite N from the exact overlap law, mean +- stderr "
                            "over named seeds (99% Student-t half-width in phi_mc_halfwidth99); NOT final"),
    }


def _ising_finality():
    d = lambda note: _derived(note)
    return {
        "phi": d("REM noise with zeta = log cosh, the recipe of Lucibello & Mezard (A17) applied to binary levels"),
        "m_star": d("Parisi parameter of the label-1RSB solution, binary levels"),
        "lam_star": d("binary condensation line atanh(eps*)"),
        "eps_star": d("largest overlap, I(eps*) = alpha; infinite lam* and eps* = 1 for alpha >= log 2"),
        "phi_flip": d("sign-random noise at 2 lam"),
        "alpha_1_flip": d("single-flip stability of a typical pattern at a typical site, the REM noise at 2 lam"),
        "alpha_1_energy": d("Eq. (7) of Lucibello & Mezard transplanted to binary levels: lam - log cosh lam"),
        "alpha_fixed_points": Stability(
            "exact-by-literature", "rigorous", f"{DHLUV}, Thm 1.3 and Remark 1.4",
            "P(some pattern is not a fixed point of the single-flip dynamics) -> 0 for alpha < I(1 - 2 radius)/2, "
            "proved for F(x) = exp(x), i.e. lam = 1 and a start on the Hamming sphere of radius radius*N; I(1)/2 = "
            "log(2)/2 at radius 0 (all patterns); the theorem says nothing about alpha above it"),
        "close_pairs_expected": Stability("exact-by-literature", "exact", "exact binomial count",
                                          "C(P,2) (1 + N)/2^N: exact for i.i.d. uniform patterns; no model assumption"),
        "p_no_close_pair": Stability("not-computed", "poisson", "",
                                     "exp(-expected pairs): a Poisson approximation to the probability that no two "
                                     "patterns lie within Hamming distance 1; NOT final"),
        "phi_mc": Stability("not-computed", "finite-N", "dense_memory/sim.py",
                            "statistical: the quenched noise at finite N from the exact overlap law; NOT final"),
        "phi_flip_mc": Stability("not-computed", "finite-N", "dense_memory/sim.py",
                                 "statistical: median of (1/(lam N)) log |sign-random sum| over named seeds; NOT final"),
    }


# ---- the entries ------------------------------------------------------------------------------
def dense_memory():
    deriv = label_replica_steps()
    steps = [fstep("dm-interaction", f"{LM}, Eqs. (1)-(2); refuses a polynomial interaction", lambda p: _interaction(p) or Out({}, "exponential interaction F(x) = exp(lam x)")),
             fstep("dm-state-point", f"{LM}, Sec. II", _state()),
             *deriv,
             fstep("dm-condensation", f"{LM}, App. A 3, Eqs. (C5), (D6)", _condensation, when=_have("alpha"),
                   skip="no alpha given"),
             fstep("dm-noise", f"{LM}, Sec. II, Eqs. (5), (6), (A17), (A24)", _noise(), when=_have("alpha", "lam"),
                   skip="needs alpha and lam"),
             fstep("dm-single-pattern-threshold", f"{LM}, Eq. (7), App. C-D", _threshold, when=_have("lam"),
                   skip="no lam given"),
             fstep("dm-critical-lambda", f"{LM}, Sec. II, Eq. (7) inverted", _lambda_one, when=_have("alpha"),
                   skip="no alpha given"),
             fstep("dm-basin", f"{LM}, Sec. IV, App. C 3", _basin, when=_have("alpha", "lam"),
                   skip="needs alpha and lam"),
             fstep("dm-few-patterns", f"{LM}, App. B, Eq. (B8)", _few, when=_have("P", "N"), skip="needs P and N"),
             fstep("dm-scaled-dot-product", f"{LM}, Sec. VI", _scaled, when=_have("lam_tilde"), skip="no lam_tilde"),
             fstep("dm-ramsauer-bound", f"{LM}, App. E, Eq. (E7)", _ramsauer,
                   when=lambda p: p.get("lam") is not None and p.get("ensemble", "spherical") == "spherical",
                   skip="needs lam and spherical patterns (the bound is for spherical patterns)"),
             fstep("dm-level-extension", "EXTENSION of Eq. (7) (this package)", _level,
                   when=lambda p: p.get("lam") is not None and p.get("level") is not None and mp.mpf(p["level"]) != 1,
                   skip="level not given or 1 (the paper's Eq. (7))"),
             fstep("dm-finite-n", "exact-law REM sampler (dense_memory/sim.py)", _finite_n(),
                   when=_have("alpha", "lam", "N"), skip="needs alpha, lam and N")]
    return entry("dense_memory", ModelSpec("dense_memory", "real", "rotation_invariant", constraint="exp interaction"),
                 LM, {"replica": steps}, _dm_refs(), {"replica": CAVEAT},
                 "dense associative memory with P = e^(alpha N) Gaussian or spherical patterns: REM noise, retrieval "
                 "thresholds alpha_1(lam), lambda_1(alpha), all-patterns bound, basins (zero temperature)",
                 finality=_dm_finality(), meta=_META, parameters={"replica": _PARAMS})


def dense_memory_ising():
    steps = [fstep("dmi-interaction", f"{DHLUV}, Eq. (1); refuses a polynomial interaction", lambda p: _interaction(p) or Out({}, "exponential interaction F(x) = exp(lam x)")),
             fstep("dmi-state-point", f"{DHLUV}; derived here", _state("binary")),
             *label_replica_steps("binary"),
             fstep("dmi-condensation", "derived here from Lucibello & Mezard (A17), zeta = log cosh", _condensation_b,
                   when=_have("alpha"), skip="no alpha given"),
             fstep("dmi-noise", "derived here from Lucibello & Mezard (A17), (A24)", _noise("binary"),
                   when=_have("alpha", "lam"), skip="needs alpha and lam"),
             fstep("dmi-sign-random-noise", "derived here: the single-flip decision sum has fair random signs",
                   _ising_phi_flip, when=_have("alpha", "lam"), skip="needs alpha and lam"),
             fstep("dmi-thresholds", "derived here; Demircigil et al. Eq. (1) for the dynamics", _ising_thresholds,
                   when=_have("lam"), skip="no lam given"),
             fstep("dmi-fixed-points", f"{DHLUV}, Thm 1.3, Remark 1.4", _ising_fixed,
                   when=lambda p: p.get("lam") is None or mp.mpf(p["lam"]) == 1,
                   skip="Thm 1.3 is proved for F(x) = exp(x), i.e. lam = 1 only"),
             fstep("dmi-close-pairs", "exact binomial count (dense_memory/ising.py)", _ising_pairs,
                   when=_have("P", "N"), skip="needs P and N"),
             fstep("dmi-finite-n", "exact-law REM sampler (dense_memory/sim.py)", _finite_n("binary"),
                   when=_have("alpha", "lam", "N"), skip="needs alpha, lam and N"),
             fstep("dmi-flip-sampler", "sign-random REM sampler (dense_memory/sim.py)", _ising_flip_mc,
                   when=_have("alpha", "lam", "N"), skip="needs alpha, lam and N")]
    return entry("dense_memory_ising", ModelSpec("dense_memory", "ising", "pm_one", constraint="exp interaction"),
                 DHLUV, {"replica": steps}, _dmi_refs(), {"replica": CAVEAT_ISING},
                 "Ising dense associative memory with P = e^(alpha N) +-1 patterns (Demircigil et al.): rigorous "
                 "fixed-point exponent log(2)/2, REM noise and single-flip thresholds derived here",
                 finality=_ising_finality(), meta=_META, parameters={"replica": _PARAMS_ISING})


def _condensation_b(p):
    return _condensation({**p, "ensemble": "binary"})


def _dm_refs():
    ref = lambda key, ex, **pr: RefCheck("dense_memory", key, ex, params=pr)
    g, s = {"ensemble": "gaussian"}, {"ensemble": "spherical"}
    return [ref("gaussian_alpha1_plateau", lambda v: v["alpha_1"], lam="3", **g),
            ref("gaussian_alpha1_lam0p5", lambda v: v["alpha_1"], lam="0.5", **g),
            ref("gaussian_alpha_c_lb_plateau", lambda v: v["alpha_c_lb"], lam="2", **g),
            ref("gaussian_lambda_saturation", lambda v: v["lambda_saturation"], lam="2", **g),
            ref("gaussian_lam_star_a0p1", lambda v: v["lam_star"], alpha="0.1", **g),
            ref("gaussian_phi_uncondensed", lambda v: v["phi"], alpha="0.1", lam="0.2", **g),
            ref("gaussian_phi_condensed", lambda v: v["phi"], alpha="0.1", lam="2", **g),
            ref("gaussian_cos_theta_large_lambda", lambda v: v["cos_theta_c"], alpha="0.1", lam="20", **g),
            ref("spherical_lam_star_a0p3", lambda v: v["lam_star"], alpha="0.3", **s),
            ref("spherical_phi_uncondensed", lambda v: v["phi"], alpha="0.3", lam="1", **s),
            ref("spherical_phi_condensed", lambda v: v["phi"], alpha="0.3", lam="10", **s),
            ref("spherical_alpha1_lam1", lambda v: v["alpha_1"], lam="1", **s),
            ref("spherical_alpha1_lam3", lambda v: v["alpha_1"], lam="3", **s),
            ref("spherical_cos_theta_large_lambda", lambda v: v["cos_theta_c"], alpha="0.1", lam="20", **s),
            ref("scaled_threshold", lambda v: v["alpha_tilde_1"], lam_tilde="0.7"),
            ref("few_patterns_cos_theta", lambda v: v["cos_theta_few"], P=1000, N=100),
            ref("ramsauer_small_lambda_ratio", lambda v: v["alpha_c_ramsauer_lb"] / v["lam"], lam="1e-6", **s)]


def _dmi_refs():
    ref = lambda key, ex, **pr: RefCheck("dense_memory_ising", key, ex, params=pr)
    return [ref("fixed_points_radius0", lambda v: v["alpha_fixed_points"], lam="1", radius="0"),
            ref("fixed_points_radius0p1", lambda v: v["alpha_fixed_points"], lam="1", radius="0.1"),
            ref("fixed_points_radius0p25", lambda v: v["alpha_fixed_points"], lam="1", radius="0.25")]


def dense_memory_entries():
    return [dense_memory(), dense_memory_ising()]


__all__ = ["dense_memory", "dense_memory_entries", "dense_memory_ising"]
