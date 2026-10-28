"""Catalog entry `population_annealing_mc` (M3): population annealing (Hukushima-Iba, as described by Machta 2010 and
Wang-Machta-Katzgraber 2015) of the 2D Ising model on the L x L torus or of one SK coupling matrix (Gaussian or
+-1/sqrt N) at small N: the free energy from the weight normalisations, the family diagnostics, and the exact ln Z.
Route sample (statistical):
  population-annealing  named seeds = independent runs (montecarlo/population.py): ln Z / N as the simple mean of the
                        runs' estimates (biased low by about var(beta F~) / 2, WMK Eq. (16); the estimate printed in
                        pa_diagnostics) and as Machta's weighted average (his Eq. (7), nominal-R normalisation,
                        an extinct run a zero weight, jackknifed over seeds), u at beta; pa_status "equilibrated" or "undetermined" (families: R~ / rho_t >= 100 per seed;
                        var(beta F~) <= 1/2; no extinct population); beta_c (ising2d, exact-by-literature)
  exact-targets         printed, never refusing: Kaufman's torus ln Z / N and u (lattice/kaufman.py,
                        lattice/torus_thermo.py, L <= 64); SK: ln Z and u by enumeration of the 2^N states (N <= 20;
                        +-J also through numerics/enumeration.enumerate_pm_one_sk)
Run parameters: model (ising2d | sk_gaussian | sk_pm_one), L (4; even, 4 .. 64), N (12; 2 .. 64), disorder (0: the
coupling matrix is sample 0 of numerics/finite_size's couplings(N, 1, disorder)), beta (beta_c for ising2d, 3/2 for
SK), population (R, 1000), temperatures (20, evenly spaced in beta from 0), sweeps (Metropolis sweeps per
temperature, 2; 1 .. 10^4: sweeps = 0 would be resampling only), seeds (0 .. 7).  L for an SK model, or N /
disorder for ising2d, is refused precondition:known-parameter (not read by that model).
"""
from functools import lru_cache

import mpmath as mp
import numpy as np

from .catalog import Out, RefCheck, Stability, entry, fstep
from .catalog_mc_wl import BDC
from .core import Refusal
from .montecarlo import fk as FK
from .spec import ModelSpec

MACHTA = ("Machta, PRE 82, 026704 (2010), arXiv:1006.0252v2 (PDF read), Sec. II Eqs. (1)-(4) (the algorithm, "
          "the free-energy estimator), Sec. III Eqs. (5)-(7) (weighted averages; var(beta F~) 'smaller than about "
          "0.5'); Hukushima & Iba, AIP Conf. Proc. 690, 200 (2003): not fetched, cited through Machta and WMK")
WMK = ("Wang, Machta & Katzgraber, PRE 92, 063307 (2015), arXiv:1508.05647v2 (PDF read), Sec. II A Eqs. (1), (2) "
       "and the nearest-integer resampling, Sec. III A Eq. (4), Sec. IV A Eqs. (12)-(16), (20), (23) (the bias of "
       "the free energy at finite R), Sec. IV B Eqs. (28), (30), (31) (rho_t, S_f, rho_s), Sec. V B (R >= 100 "
       "rho_s), Sec. VI B (rho_t preferred)")
CAVEAT = ("sample route, statistical: independent population-annealing runs over named seeds; means over seeds "
          "+- stderr; never certified, never final.  One run's ln Z~ is biased LOW by about var(beta F~)/2 "
          "(Wang-Machta-Katzgraber Eqs. (15), (16)); the weighted estimate (Machta Eq. (7)) removes the leading "
          "bias.  pa_status 'undetermined' when a seed has R~ / rho_t < 100 (too few families), var(beta F~) over "
          "seeds exceeds 1/2, or a population died out; Metropolis sweeps at each temperature (2D Ising: M1's "
          "checkerboard sweep; SK: sequential single-spin flips)")
MODELS = ("ising2d", "sk_gaussian", "sk_pm_one")
DEFAULTS = {"model": "ising2d", "L": 4, "N": 12, "disorder": 0, "beta": None, "population": 1000,
            "temperatures": 20, "sweeps": 2, "seeds": tuple(range(8))}
PARAMS = tuple(DEFAULTS)
STAT = ("ln_z_per_site_pa", "ln_z_per_site_weighted", "energy_per_site_pa")


def _int(name, v, lo, hi):
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not lo <= v <= hi:
        raise Refusal(f"precondition:{name}", f"{name} = {v!r}: need an int in {lo} .. {hi}")
    return int(v)


def _p(p):
    from .montecarlo.population import check_params
    g = {k: p.get(k, d) for k, d in DEFAULTS.items()}
    if g["model"] not in MODELS:
        raise Refusal("precondition:model", f"model = {g['model']!r}: one of {', '.join(MODELS)}")
    unused = sorted(k for k in (("N", "disorder") if g["model"] == "ising2d" else ("L",)) if k in p)
    if unused:                      # every accepted parameter is read or refused
        raise Refusal("precondition:known-parameter", f"{', '.join(unused)}: not a parameter of model = "
                      f"{g['model']} (ising2d reads L; the SK models read N and disorder)")
    if g["model"] == "ising2d":
        if isinstance(g["L"], bool) or not isinstance(g["L"], (int, np.integer)) or g["L"] % 2 or not 4 <= g["L"] <= 64:
            raise Refusal("precondition:checkerboard", f"L = {g['L']!r}: the checkerboard Metropolis sweep needs an "
                          "even L in 4 .. 64 (on the 2 x 2 torus it is not ergodic: tests/test_mc_cluster.py)")
        g["L"] = int(g["L"])
    else:
        g["N"], g["disorder"] = _int("size", g["N"], 2, 64), _int("disorder", g["disorder"], 0, 2 ** 32)
    if g["beta"] is None:
        g["beta"] = "beta_c" if g["model"] == "ising2d" else "3/2"
    if g["beta"] == "beta_c" and g["model"] != "ising2d":
        raise Refusal("precondition:beta", "beta = 'beta_c' is the 2D Ising critical point: give SK a rational beta")
    g["beta_f"] = float(FK.beta_value(2, g["beta"]))
    g["population"], g["temperatures"], g["sweeps"], g["seeds"] = check_params(
        g["population"], g["temperatures"], g["sweeps"], g["seeds"])
    return g


@lru_cache(maxsize=16)
def _couplings(model, N, disorder):
    from .numerics.finite_size import gaussian_sk_couplings, pm_one_sk_couplings
    law = gaussian_sk_couplings if model == "sk_gaussian" else pm_one_sk_couplings
    return law(N, 1, disorder)[0]


def _system(g):
    from .montecarlo.population import SK, Ising2D
    return Ising2D(g["L"]) if g["model"] == "ising2d" else SK(_couplings(g["model"], g["N"], g["disorder"]))


@lru_cache(maxsize=16)
def _run(model, L, N, disorder, beta_f, R, K, sweeps, seeds):
    from .montecarlo.population import anneal, summary
    g = {"model": model, "L": L, "N": N, "disorder": disorder}
    r = anneal(_system(g), beta_f, R, K, sweeps, seeds)
    return r, summary(r)


def _run_of(g):
    return _run(g["model"], g["L"], g["N"], g["disorder"], g["beta_f"], g["population"], g["temperatures"],
                g["sweeps"], g["seeds"])


def _pa(p):
    from .montecarlo.population import FAMILY_MIN, VAR_MAX
    g = _p(p)
    r, s = _run_of(g)
    vals = {"ln_z_per_site_pa": s.ln_z_mean.mean, "ln_z_per_site_weighted": s.ln_z_weighted,
            "energy_per_site_pa": s.energy.mean, "pa_status": s.status, "pa_summary": s.text, "model": g["model"],
            "beta": g["beta"],
            "pa_diagnostics": {"var_beta_f": s.var_beta_f, "ln_z_bias_per_site_estimate": s.bias_estimate,
                               "family_min": FAMILY_MIN, "var_max": VAR_MAX, "reasons": list(s.reasons),
                               "families": [[int(x), *(float(v) for v in f)] for x, f in zip(r.seeds, r.families)],
                               "families_columns": ["seed", "rho_t", "R~/rho_t", "rho_s", "e^S_f"],
                               "final_population": [int(v) for v in r.sizes[:, -1]]}}
    errs = {"ln_z_per_site_pa": s.ln_z_mean.stderr, "ln_z_per_site_weighted": s.ln_z_weighted_stderr,
            "energy_per_site_pa": s.energy.stderr}
    notes = [f"population annealing: {s.text}; 99% Student-t half-widths over seeds: ln_z_per_site_pa +- "
             f"{s.ln_z_mean.half_width:.3g}, energy_per_site_pa +- {s.energy.half_width:.3g}"]
    if g["model"] == "ising2d":
        bc, detail = FK.critical_beta(2)
        vals["beta_c"], errs["beta_c"] = bc, mp.mpf(10) ** -30
        vals["L"] = g["L"]
        notes.append(detail)
    else:
        vals.update(N=g["N"], disorder=g["disorder"])
    return Out(vals, s.text, errors=errs, statistical=STAT, notes=tuple(notes))


def _exact(p):
    g = _p(p)
    out, lines = {}, []
    if g["model"] == "ising2d":
        from .lattice.kaufman import kaufman_free_energy
        from .lattice.torus_thermo import torus_thermo_kaufman
        L, b = g["L"], g["beta"]
        u, _ = torus_thermo_kaufman(L, L, b)
        with mp.workdps(20):
            out.update(ln_z_per_site_exact=float(kaufman_free_energy(L, L, b).mid()), energy_exact=float(u.mid()))
        lines.append(f"Kaufman torus: ln Z / N = {out['ln_z_per_site_exact']:.12g}, u = {out['energy_exact']:.10g}")
    else:
        from .montecarlo.sk_metropolis import MAX_N_EXACT, exact_energy, exact_log_z
        J, N = _couplings(g["model"], g["N"], g["disorder"]), g["N"]
        if N <= MAX_N_EXACT:
            out.update(ln_z_per_site_exact=exact_log_z(J, g["beta_f"]) / N, energy_exact=exact_energy(J, g["beta_f"]))
            lines.append(f"enumeration of the 2^{N} states: ln Z / N = {out['ln_z_per_site_exact']:.12g}, "
                         f"u = {out['energy_exact']:.10g}")
            if g["model"] == "sk_pm_one":
                from .numerics.enumeration import enumerate_pm_one_sk
                out["ln_z_per_site_e11"] = enumerate_pm_one_sk(J).log_Z(g["beta_f"]) / N
                lines.append(f"E11 integer spectrum: ln Z / N = {out['ln_z_per_site_e11']:.12g}")
    return Out({"exact_targets": out}, "; ".join(lines) or "no exact target at this size", notes=(
        "exact targets (printed, compared in tests): " + ("; ".join(lines) or "none at this size"),))


def _stat(v):
    und = v.get("pa_status") == "undetermined"
    return Stability("not-computed", "sample", f"{MACHTA}; {WMK}", "Monte Carlo over named seeds: NOT final"
                     + ("; UNDETERMINED: " + "; ".join(v.get("pa_diagnostics", {}).get("reasons", [])) if und else ""))


FINALITY = {**{k: _stat for k in STAT},
            "beta_c": Stability("exact-by-literature", "exact", BDC, "Theorem 3 at q = 2, halved for the Ising "
                                "convention; a second route -log(1 - p_sd) agrees to 1e-30")}
META = ("model", "L", "N", "disorder", "beta", "pa_status", "pa_summary", "pa_diagnostics", "exact_targets")
CHEAP = {"population": 100, "temperatures": 5, "sweeps": 1, "seeds": (0, 1)}   # too few families: undetermined


def population_annealing_mc():
    steps = [fstep("population-annealing", f"{MACHTA}; {WMK}", _pa),
             fstep("exact-targets", "lattice/kaufman.py, lattice/torus_thermo.py (Kaufman); enumeration of the 2^N "
                   "SK states (montecarlo/sk_metropolis.py, numerics/enumeration.py)", _exact)]
    refs = [RefCheck("lattice", "beta_c_square", lambda v: v["beta_c"], route="sample", params=CHEAP)]
    spec = ModelSpec("ising", "ising", "none", constraint="population annealing: L x L torus, or one SK / +-J "
                     "coupling matrix at small N")
    return entry("population_annealing_mc", spec, f"{MACHTA}; {WMK}", {"sample": steps}, refs, {"sample": CAVEAT},
                 "Population annealing of the 2D Ising torus and of SK / +-J couplings at small N: the free energy "
                 "from the weight normalisations against exact ln Z, family-size diagnostics, over named seeds",
                 default_route="sample", finality=FINALITY, meta=META, parameters={"sample": PARAMS},
                 rigor={"sample": "statistical"})


__all__ = ["CHEAP", "population_annealing_mc"]
