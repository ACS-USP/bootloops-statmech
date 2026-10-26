"""Catalog entry `ldpc_bec` (regular LDPC codes on the binary erasure channel, E6a), route `cavity`.

Why `cavity`: belief propagation on the code's factor graph is the cavity method on a
locally tree-like graph; on the BEC its messages are erased / known, density evolution is the
replica-symmetric cavity recursion for their erasure probability (Mezard & Montanari 2009,
Sec. 15.3), and the trial entropy whose root gives the MAP threshold is the Bethe (RS cavity)
entropy of the code evaluated on a DE fixed point (Measson, Montanari & Urbanke 2008,
Definition 4, Theorem 10).  No replica functional is involved, so not `replica`.  The finite-N
part (configuration-model graphs, the peeling decoder over named seeds) is an independent
check inside the same route, labelled statistical.

Conventions (inference/ldpc.py): variable degree l, check degree r, edge-perspective
lambda(x) = x^(l-1), rho(x) = x^(r-1); defaults l = 3, r = 6.  Run parameters:
  (none)        design_rate, eps_bp, x_bp, eps_map, x_map (trial-entropy root) and
                eps_map_area (area-theorem route, quadrature); needs l >= 3
  eps           density evolution at eps: de_edge_erasure, de_check_erasure, de_bit_erasure
  eps, N        the peeling decoder on N-variable graphs over named seeds (default 0-19):
                peel_success_fraction (99% Clopper-Pearson), peel_residual_mean
Finality: eps_bp, x_bp `exact-by-literature` (Luby et al. 2001, Thm 2 and Mezard-Montanari
Eq. (15.34)); eps_map, x_map, eps_map_area `exact-by-literature` (MMU 2008 Example 6 for
(3, 6); Kudekar-Richardson-Urbanke 2010, Lemma 4, for regular ensembles); the DE values
`exact-by-literature` when DE converged, `not-computed` otherwise; finite-N values statistical
and `not-computed`.
"""
from .catalog import Out, RefCheck, Stability, entry, fstep, has
from .core import Refusal
from .csp import _checks
from .inference import ldpc as L
from .inference import ldpc_finite as LF
from .spec import ModelSpec

MMU = ("Measson, Montanari & Urbanke, IEEE Trans. Inform. Theory 54, 5277 (2008), "
       "arXiv:cs/0506083")
KRU = "Kudekar, Richardson & Urbanke, arXiv:1001.1826 (2010), Lemma 4"
LMSS = ("Luby, Mitzenmacher, Shokrollahi & Spielman, IEEE Trans. Inform. Theory 47, 569 (2001), "
        "Thm 2 and Prop. 2")
MM15 = "Mezard & Montanari, Information, Physics, and Computation (2009), Sec. 15.3, Eqs. (15.33)-(15.34)"
CAVEAT = ("cavity route (belief propagation = peeling on the code's factor graph; density "
          "evolution is its N -> oo limit): the thresholds are theorems of the cited papers; "
          "the finite-N fractions are statistical (binomial, named seeds), never certified")
_DEFAULT_SEEDS = tuple(range(20))


def _lr(p):
    return p.get("l", 3), p.get("r", 6)


def _check_params(p):
    if p.get("N") is None and "seeds" in p:
        raise Refusal("precondition:known-parameter",
                      "seeds is a finite-N parameter: need N (and eps) for the peeling check")
    if p.get("N") is not None and p.get("eps") is None:
        raise Refusal("precondition:state-point", "the finite-N check needs eps (and N)")


def _thresholds(p):
    _check_params(p)
    l, r = _lr(p)
    th = L.bec_thresholds(l, r)
    area = L.bec_map_threshold_area(l, r)
    return Out({"l": th.l, "r": th.r, "design_rate": th.design_rate, "eps_bp": th.eps_bp,
                "x_bp": th.x_bp, "eps_map": th.eps_map, "x_map": th.x_map, "eps_map_area": area},
               "tangency of eps(x) = x / lambda(1 - rho(1 - x)) (eps_BP) and the root of the "
               "trial entropy on (x_BP, 1) (eps_MAP); eps_map_area from the area theorem")


def _de(p):
    l, r = _lr(p)
    d = L.bec_density_evolution(l, r, p["eps"])
    return Out({"de_edge_erasure": d.edge_erasure, "de_check_erasure": d.check_erasure,
                "de_bit_erasure": d.erasure_prob, "de_converged": d.converged,
                "de_iterations": d.iterations},
               f"density evolution from x_0 = eps: {'converged' if d.converged else 'NOT converged'}"
               f" after {d.iterations} sweeps")


def _finite(p):
    l, r = _lr(p)
    seeds = _checks.seeds(p.get("seeds", _DEFAULT_SEEDS))
    s = LF.ldpc_peeling_success(l, r, p["N"], p["eps"], seeds)
    return Out({"peel_success_fraction": s.fraction, "peel_ci99_lo": s.lo, "peel_ci99_hi": s.hi,
                "peel_residual_mean": s.residual_mean, "n_seeds": s.n,
                "multi_edges_mean": s.multi_edges_mean},
               f"peeling decoder on {s.n} configuration-model graphs (seeds {seeds[0]}..{seeds[-1]}):"
               f" {s.n_success} decoded, 99% Clopper-Pearson [{s.lo:.3f}, {s.hi:.3f}]",
               errors={"peel_success_fraction": s.halfwidth99,
                       "peel_residual_mean": s.residual_stderr},
               statistical=("peel_success_fraction", "peel_residual_mean"),
               notes=("statistical: the success fraction (binomial over named seeds; its error is "
                      "the larger half-width of the 99% Clopper-Pearson interval) and the mean "
                      "residual erasure fraction (mean +- stderr) are finite-N estimates, not "
                      "certified",))


def _map_label(v):
    note = ("MAP threshold of the ensemble: root of the trial (Bethe) entropy, equal to the "
            "area-theorem value; an upper bound by MMU Lemmas 2-3, the MAP threshold by MMU "
            "Theorem 10 / Corollary 2")
    if (v.get("l"), v.get("r")) == (3, 6):
        note += ", whose conditions MMU verify for (3, 6) in Example 6"
    else:
        note += (f"; for ({v.get('l')}, {v.get('r')}) the equality is cited from KRU Lemma 4 "
                 "(stated there for every regular ensemble; proof referred to Richardson & "
                 "Urbanke, Modern Coding Theory (2008), not fetched)")
    ex6 = ", Ex. 6" if (v.get("l"), v.get("r")) == (3, 6) else ""
    return Stability("exact-by-literature", "rs", f"{MMU}, Lemmas 2-3, Thm 10{ex6}; {KRU}", note)


def _de_label(v):
    if v.get("de_converged"):
        return Stability("exact-by-literature", "rs", f"{MM15}; {LMSS}",
                         "N -> oo erasure probabilities of BP decoding (density evolution on the "
                         "tree-like ensemble; concentration of the finite-N decoder on DE as "
                         "summarised in the Notes of Mezard-Montanari ch. 15: the concentration "
                         "theorem itself, Richardson & Urbanke 2001, was not fetched)")
    return Stability("not-computed", "rs", MM15, "density evolution did not converge within "
                     f"{L.MAX_ITER} sweeps (critical slowing down at eps_BP): NOT final")


def ldpc_bec():
    steps = [fstep("ldpc-thresholds", f"{MM15}; {MMU}, Def. 4, Lemmas 2-4; {KRU}", _thresholds),
             fstep("ldpc-density-evolution", MM15, _de, when=has("eps"), skip="no eps given"),
             fstep("ldpc-finite-n-peeling", "peeling decoder on configuration-model graphs "
                   "(inference/ldpc_finite.py); Mezard & Montanari (2009), Sec. 15.3.1",
                   _finite, when=has("eps", "N"), skip="needs eps and N")]
    refs = [RefCheck("ldpc", "eps_bp_l3_r6", lambda v: v["eps_bp"], route="cavity"),
            RefCheck("ldpc", "eps_map_l3_r6", lambda v: v["eps_map"], route="cavity")]
    for l, r in ((4, 8), (5, 10), (6, 12), (7, 14)):
        refs += [RefCheck("ldpc", f"eps_bp_l{l}_r{r}", lambda v: v["eps_bp"],
                          params={"l": l, "r": r}, route="cavity"),
                 RefCheck("ldpc", f"eps_map_l{l}_r{r}", lambda v: v["eps_map"],
                          params={"l": l, "r": r}, route="cavity")]
    refs += [RefCheck("ldpc", f"eps_map_l{l}_r{r}", lambda v: v["eps_map_area"],
                      params={"l": l, "r": r}, route="cavity")
             for l, r in ((3, 4), (3, 5), (4, 6), (3, 7))]
    refs += [RefCheck("ldpc", "x_bp_l3_r6", lambda v: v["x_bp"], route="cavity")]
    refs += [RefCheck("ldpc", key, fn, params={"eps": "0.46"}, route="cavity")
             for key, fn in (("de_x_l3_r6_eps046", lambda v: v["de_edge_erasure"]),
                             ("de_y_l3_r6_eps046", lambda v: v["de_check_erasure"]),
                             ("de_determined_l3_r6_eps046", lambda v: 1 - v["de_bit_erasure"]))]
    bp = Stability("exact-by-literature", "rs", f"{LMSS}; {MM15}",
                   "BP (peeling) threshold: below it peeling recovers every bit w.h.p. (LMSS Thm 2, "
                   "for variable degrees >= 3); above it DE has a nonzero fixed point")
    rate = Stability("exact-by-literature", "rs", f"{MMU}, Lemma 3 (proof), Lemma 2",
                     "design rate r(lambda, rho) = 1 - Lambda'(1)/Gamma'(1) = 1 - l/r (the rate of "
                     "a code is at least this; MMU: r_as >= r(lambda, rho))")
    finite = Stability("not-computed", "rs", "inference/ldpc_finite.py",
                       "finite-N statistical estimate over named seeds: NOT final")
    fin = {"design_rate": rate, "eps_bp": bp, "x_bp": bp,
           **{k: _map_label for k in ("eps_map", "x_map", "eps_map_area")},
           **{k: _de_label for k in ("de_edge_erasure", "de_check_erasure", "de_bit_erasure")},
           **{k: finite for k in ("peel_success_fraction", "peel_residual_mean")}}
    spec = ModelSpec("ldpc_bec", "boolean", "none",
                     constraint="regular (l, r) LDPC code on the binary erasure channel",
                     params={"l": 3, "r": 6})
    return entry("ldpc_bec", spec, MMU, {"cavity": steps}, refs, {"cavity": CAVEAT},
                 "regular LDPC codes on the BEC: density evolution, BP threshold, MAP threshold "
                 "(Maxwell construction), finite-N peeling", default_route="cavity",
                 finality=fin, meta=("l", "r", "de_converged", "de_iterations", "peel_ci99_lo",
                                     "peel_ci99_hi", "n_seeds", "multi_edges_mean"),
                 parameters={"cavity": ("l", "r", "eps", "N", "seeds")})


def inference_entries():
    from .catalog_spiked_sbm import sbm, spiked_wigner
    return [ldpc_bec(), spiked_wigner(), sbm()]


__all__ = ["inference_entries", "ldpc_bec"]
