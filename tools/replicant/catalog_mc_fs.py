"""Catalog entry `histogram_reweighting_mc` (M2): Ferrenberg-Swendsen single and multiple histogram reweighting of
canonical 2D Ising runs on the L x L torus (montecarlo/reweight.py), the runs from M1's samplers (montecarlo/run.py).
Route sample (statistical):
  single-histogram     named-seed chains at beta0, reweighted to beta: u, C; the reliable window from histogram
                       overlap (O >= 1/2); a beta outside it is refused precondition:reweight-window; mc_status
                       "equilibrated" or "undetermined" (stats.equilibration, tau_int reliability)
  multiple-histogram   chains at each of `betas` combined (Kumar et al. Eqs. (19), (20)), u, C at beta; windows that
                       do not chain are refused precondition:histogram-overlap
  exact-targets        printed, never refusing: Kaufman's torus u, C at beta (L <= 64); exact g(E) values (L <= 12)
Run parameters: algorithm (swendsen_wang | wolff | metropolis), L (8), beta0 (beta_c), beta (9/20), betas
(2/5, beta_c, 12/25), seeds (0 .. 7), sweeps (1000), burn_in (200).  Run m of named seed s at betas[m] is the
chain of seed s + SEED_STRIDE (m + 1), so the multiple-histogram runs are independent of each other and of the
single-histogram run (seed s itself).
"""
from functools import lru_cache

import mpmath as mp
import numpy as np

from .catalog import Out, RefCheck, Stability, entry, fstep
from .catalog_mc_wl import BDC
from .montecarlo import fk as FK
from .spec import ModelSpec

KUMAR = ("Kumar, Bouzida, Swendsen, Kollman & Rosenberg, J. Comput. Chem. 13, 1011 (1992) (PDF read, pages "
         "rendered), Eqs. (17)-(20), (22) and the footnote to Eq. (19): the Ferrenberg-Swendsen single and multiple "
         "histogram equations (Ferrenberg & Swendsen, PRL 61, 2635 (1988); PRL 63, 1195 (1989): not fetched)")
BERG = "Berg, arXiv:cond-mat/9909236 (PDF read), Eq. (2.15) and the text below it (the reweighting range)"
CAVEAT = ("sample route, statistical: canonical chains (M1 samplers) over named seeds reweighted by the Ferrenberg-"
          "Swendsen equations as printed by Kumar et al. 1992; means over seeds +- stderr; never final; reweighting "
          "outside the overlap window (O >= 1/2, a rule of this package: |beta - beta0| <= 1.349 / sigma_E for a "
          "Gaussian histogram) is refused; mc_status 'undetermined' when a run drifts or a tau_int is unreliable")
SEED_STRIDE = 1_000_003
DEFAULTS = {"algorithm": "swendsen_wang", "L": 8, "beta0": "beta_c", "beta": "9/20",
            "betas": ("2/5", "beta_c", "12/25"), "seeds": tuple(range(8)), "sweeps": 1000, "burn_in": 200}
PARAMS = tuple(DEFAULTS)
STAT = ("energy_per_site_reweighted", "specific_heat_reweighted", "energy_per_site_multihistogram",
        "specific_heat_multihistogram")


def _p(p):
    from .core import Refusal
    from .montecarlo.run import check_params
    g = {k: p.get(k, d) for k, d in DEFAULTS.items()}
    g["algorithm"], _, g["L"], g["sweeps"], g["burn_in"], g["seeds"] = check_params(
        g["algorithm"], 2, g["L"], g["sweeps"], g["burn_in"], g["seeds"])
    if isinstance(g["betas"], (str, bytes)) or not hasattr(g["betas"], "__iter__") or len(tuple(g["betas"])) < 2:
        raise Refusal("precondition:betas", f"betas = {g['betas']!r}: need >= 2 inverse temperatures")
    g["betas"] = tuple(g["betas"])
    for k in ("beta0", "beta"):
        g[f"{k}_f"] = float(FK.beta_value(2, g[k]))
    g["betas_f"] = tuple(float(FK.beta_value(2, b)) for b in g["betas"])
    if len(set(g["betas_f"])) != len(g["betas_f"]):
        raise Refusal("precondition:betas", f"betas = {g['betas']!r}: need distinct values")
    return g


@lru_cache(maxsize=64)
def _chains(algorithm, L, beta, sweeps, burn_in, seeds):
    from .montecarlo import stats as ST
    from .montecarlo.run import chain_series, wolff_flips_per_sweep
    unit = (wolff_flips_per_sweep(2, L, FK.bond_probability(2, beta), burn_in, seeds[0]) if algorithm == "wolff"
            else None)                            # M1's rule: one Wolff sweep unit per run, from seeds[0]
    es = [chain_series(algorithm, 2, L, beta, sweeps, burn_in, s, unit)["E"].astype(float) for s in seeds]
    eq = ST.equilibration(np.array(es) / (L * L))
    rel = all(ST.tau_int(x).reliable for x in es)
    ok = eq.equilibrated and rel
    return es, ("equilibrated" if ok else "undetermined"), eq.text + ("" if rel else "; a tau_int is UNRELIABLE")


def _vals(rw, suffix):
    vals = {f"energy_per_site_{suffix}": rw.bands["u"].mean, f"specific_heat_{suffix}": rw.bands["C"].mean}
    errs = {f"energy_per_site_{suffix}": rw.bands["u"].stderr, f"specific_heat_{suffix}": rw.bands["C"].stderr}
    return vals, errs


def _single(p):
    from .montecarlo.reweight import single_histogram
    g = _p(p)
    es, status, eqt = _chains(g["algorithm"], g["L"], g["beta0"], g["sweeps"], g["burn_in"], g["seeds"])
    rw = single_histogram(es, g["beta0_f"], g["beta_f"], g["L"] ** 2)
    vals, errs = _vals(rw, "reweighted")
    bc, detail = FK.critical_beta(2)
    vals.update(beta_c=bc, algorithm=g["algorithm"], L=g["L"], beta0=g["beta0"], beta=g["beta"],
                single_window=list(rw.window), single_overlap=rw.overlap, single_status=status)
    errs["beta_c"] = mp.mpf(10) ** -30
    return Out(vals, rw.text, errors=errs, statistical=tuple(k for k in vals if k.endswith("_reweighted")),
               notes=(f"single histogram: {rw.text}; run at beta0: {eqt}", detail))


def _multi(p):
    from .montecarlo.reweight import multiple_histogram
    g = _p(p)
    runs, sts = [], []
    for m, (b, bf) in enumerate(zip(g["betas"], g["betas_f"])):
        seeds = tuple(s + SEED_STRIDE * (m + 1) for s in g["seeds"])
        es, status, _ = _chains(g["algorithm"], g["L"], b, g["sweeps"], g["burn_in"], seeds)
        runs.append((bf, es))
        sts.append(status)
    rw = multiple_histogram(runs, g["beta_f"], g["L"] ** 2)
    vals, errs = _vals(rw, "multihistogram")
    status = "equilibrated" if all(s == "equilibrated" for s in sts) else "undetermined"
    vals.update(betas=list(g["betas"]), multi_window=list(rw.window), multi_status=status)
    return Out(vals, rw.text, errors=errs, statistical=tuple(k for k in vals if k.endswith("_multihistogram")),
               notes=(f"multiple histogram: {rw.text}; runs {', '.join(sts)}",))


def _exact(p):
    g = _p(p)
    out, lines = {}, []
    if g["L"] <= 64:
        from .lattice.torus_thermo import torus_thermo_kaufman
        u, c = torus_thermo_kaufman(g["L"], g["L"], g["beta"])
        with mp.workdps(20):
            out.update(energy_kaufman=float(u.mid()), specific_heat_kaufman=float(c.mid()))
        lines.append(f"Kaufman torus at beta: u = {out['energy_kaufman']:.10g}, "
                     f"C = {out['specific_heat_kaufman']:.10g}")
    if g["L"] <= 12:
        from .montecarlo.dos_exact import exact_canonical
        ex = exact_canonical(g["L"], FK.beta_value(2, g["beta"]))
        out.update(energy_exact_dos=ex.u, specific_heat_exact_dos=ex.C)
        lines.append(f"exact g(E): u = {ex.u:.10g}, C = {ex.C:.10g}")
    return Out({"exact_targets": out}, "; ".join(lines) or "no exact target", notes=(
        "exact targets (printed, compared in tests): " + ("; ".join(lines) or "none at this size"),))


def _stat(key):
    def f(v):
        und = v.get(key) == "undetermined"
        return Stability("not-computed", "sample", f"{KUMAR}; {BERG}", "Monte Carlo over named seeds: NOT final"
                         + ("; UNDETERMINED: a canonical run drifted or a tau_int is unreliable" if und else ""))
    return f


FINALITY = {**{k: _stat("single_status" if k.endswith("_reweighted") else "multi_status") for k in STAT},
            "beta_c": Stability("exact-by-literature", "exact", BDC, "Theorem 3 at q = 2, halved for the Ising "
                                "convention; a second route -log(1 - p_sd) agrees to 1e-30")}
META = ("algorithm", "L", "beta0", "beta", "betas", "single_window", "single_overlap", "single_status", "multi_window",
        "multi_status", "exact_targets")
CHEAP = {"algorithm": "metropolis", "L": 4, "sweeps": 50, "burn_in": 5, "seeds": (0, 1), "beta": "11/25",
         "betas": ("2/5", "12/25")}


def histogram_reweighting_mc():
    steps = [fstep("single-histogram", f"{KUMAR}; {BERG}", _single), fstep("multiple-histogram", KUMAR, _multi),
             fstep("exact-targets", "lattice/torus_thermo.py (Kaufman); montecarlo/dos_exact.py", _exact)]
    refs = [RefCheck("lattice", "beta_c_square", lambda v: v["beta_c"], route="sample", params=CHEAP)]
    spec = ModelSpec("ising", "ising", "none", constraint="L x L torus, histogram reweighting")
    return entry("histogram_reweighting_mc", spec, f"{KUMAR}; {BERG}", {"sample": steps}, refs, {"sample": CAVEAT},
                 "Ferrenberg-Swendsen single and multiple histogram reweighting of canonical 2D Ising runs on L x L "
                 "tori: u, C at beta from runs at nearby beta, with the overlap window and its refusal",
                 default_route="sample", finality=FINALITY, meta=META, parameters={"sample": PARAMS},
                 rigor={"sample": "statistical"})


__all__ = ["CHEAP", "SEED_STRIDE", "histogram_reweighting_mc"]
