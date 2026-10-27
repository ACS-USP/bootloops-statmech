"""Catalog entry `cluster_mc` (M1): Swendsen-Wang and Wolff cluster Monte Carlo (and the single-spin-flip
Metropolis baseline) for the q-state Potts model on the L x L torus, q = 2 the Ising model (montecarlo/).
Route sample (statistical):
  fk-coupling        the bond probability p = 1 - e^{-K} and the exact critical point beta_c(q) by two
                     routes (montecarlo/fk.py; Beffara & Duminil-Copin Theorems 1, 3)
  cluster-mc         named-seed chains: <e>, <|m|>, <m^2>, C (and SW's FK estimator of <m^2>), their
                     stderr over seeds, tau_int (montecarlo/stats.py), the equilibration diagnostic;
                     mc_status "undetermined" when the energy drifts or a tau_int is unreliable
  exact-targets      the exact values this run can be compared with, printed (never a refusal: a correct
                     sampler is not refused at runtime): q = 2, <e> and C of the same torus (Kaufman,
                     lattice/torus_thermo.py, L <= 64); <|m|>, <m^2> (and <e>, C) by enumeration when q^N <= 2^16
Run parameters: algorithm (swendsen_wang | wolff | metropolis, default swendsen_wang), q (2), L (8), beta
(the q model's beta_c; or a rational > 0: the Ising beta of H = -sum s s at q = 2, the Potts beta of
H = -sum delta at q >= 3), seeds (0 .. 7), sweeps (1000), burn_in (200).
"""
import mpmath as mp

from .catalog import Out, RefCheck, Stability, entry, fstep
from .montecarlo import fk as FK
from .montecarlo.run import ALGORITHMS, check_params, simulate
from .spec import ModelSpec

SS = ("Salas & Sokal, arXiv:hep-lat/9605018v2 (1997) (PDF read), Sec. 2.1, Eqs. (2.1)-(2.5) "
      "(Swendsen-Wang, the Edwards-Sokal joint weight, p = 1 - e^{-beta}), Eqs. (4.11), (4.13) and p. 13 (tau_int, "
      "the self-consistent window c ~ 6)")
LPS = ("Liu, Polkovnikov & Sandvik, arXiv:1310.6327v3 (2014) (PDF read), Secs. III B, C "
       "(SW: P = 1 - e^{-2J/T}, clusters flipped with probability 1/2; Wolff: the same bond rule from a random "
       "seed, flipped with probability one), Eq. (27), Table I")
BDC = ("Beffara & Duminil-Copin, arXiv:1006.5073v2 (PDF read), "
       "Eq. (1.1), Theorem 1 (p_c = sqrt q / (1 + sqrt q), q >= 1), Theorem 3 (beta_c = log(1 + sqrt q), q >= 2)")
KAUF = "lattice/torus_thermo.py (Kaufman's torus partition function as printed by Izmailian & Hu, Eqs. (10)-(16))"
CAVEAT = ("sample route, statistical: Swendsen-Wang / Wolff / Metropolis chains over named seeds; means over seeds "
          "+- stderr (99% Student-t half-widths in the notes), tau_int by the self-consistent window (c = 6); never "
          "certified, never final; mc_status 'undetermined' when the equilibration diagnostic drifts or a tau_int is "
          "unreliable.  Valid for any ferromagnetic couplings; on frustrated or disordered couplings SW / Wolff stay "
          "valid (bonds open only on satisfied edges) but lose their efficiency: the satisfied-edge clusters can "
          "percolate where the spins do not order (a statement of this package, no source fetched; not "
          "implemented here: uniform J only)")
REN = {"e": "energy_per_site_sampled", "abs_m": "abs_magnetisation_sampled", "m2": "m2_sampled",
       "C": "specific_heat_sampled", "m2_improved": "m2_fk_estimator_sampled"}
DEFAULTS = {"algorithm": "swendsen_wang", "q": 2, "L": 8, "beta": "beta_c", "sweeps": 1000, "burn_in": 200,
            "seeds": tuple(range(8))}


def _p(p):
    g = {k: p.get(k, d) for k, d in DEFAULTS.items()}
    g["algorithm"], g["q"], g["L"], g["sweeps"], g["burn_in"], g["seeds"] = check_params(
        g["algorithm"], g["q"], g["L"], g["sweeps"], g["burn_in"], g["seeds"])
    FK.beta_value(g["q"], g["beta"])
    return g


def _coupling(p):
    g = _p(p)
    bc, detail = FK.critical_beta(g["q"])
    with mp.workdps(40):
        b = FK.beta_value(g["q"], g["beta"])
        return Out({"beta_c": +bc, "algorithm": g["algorithm"], "q": g["q"], "L": g["L"],
                    "beta": mp.nstr(b, 20), "bond_probability": FK.bond_probability(g["q"], g["beta"])},
                   f"{detail}; at beta = {mp.nstr(b, 12)}: p = {FK.bond_probability(g['q'], g['beta']):.10g}",
                   errors={"beta_c": mp.mpf(10) ** -30})


def _mc(p):
    g = _p(p)
    r = simulate(g["algorithm"], g["q"], g["L"], g["beta"], g["sweeps"], g["burn_in"], g["seeds"])
    vals = {REN[k]: x for k, x in r.mean.items()}
    errs = {REN[k]: x for k, x in r.stderr.items()}
    vals.update(tau_int_energy=r.tau["e"], tau_int_m2=r.tau["m2"])
    errs.update(tau_int_energy=r.tau_stderr["e"], tau_int_m2=r.tau_stderr["m2"])
    hw = ", ".join(f"{REN[k]} +- {b.half_width:.3g}" for k, b in r.bands.items())
    meta = {"mc_status": r.status, "equilibration": r.equilibration.text, "n_seeds": len(r.seeds),
            "sweeps": r.sweeps, "burn_in": r.burn_in, "tau_reliable": r.tau_reliable,
            "wolff_flips_per_sweep": r.flips if g["algorithm"] == "wolff" else None}
    note = (f"{len(r.seeds)} x {r.sweeps} {g['algorithm']} sweeps, q = {g['q']}, {g['L']} x {g['L']} torus: <e> = "
            f"{r.mean['e']:.5g} +- {r.stderr['e']:.2g}, tau_int(e) = {r.tau['e']:.3g}; {r.status}")
    notes = (f"statistical: means over named seeds, stderr in errors; 99% Student-t half-widths: {hw}",
             f"equilibration: {r.equilibration.text}")
    return Out({**vals, **meta}, note, errors=errs, statistical=tuple(vals), notes=notes)


def _exact(p):
    g = _p(p)
    out, lines = {}, []
    if g["q"] == 2 and g["L"] <= 64:
        from .lattice.torus_thermo import torus_thermo_kaufman
        u, c = torus_thermo_kaufman(g["L"], g["L"], g["beta"])
        with mp.workdps(20):
            out.update(energy_exact=float(u.mid()), specific_heat_exact=float(c.mid()))
        lines.append(f"Kaufman torus: <e> = {out['energy_exact']:.10g}, C = {out['specific_heat_exact']:.10g}")
    if g["q"] ** (g["L"] ** 2) <= 1 << 16:
        from .montecarlo.exact_small import exact_distribution
        ex = exact_distribution(g["L"], g["q"], g["beta"])
        out.update(abs_magnetisation_exact=ex.abs_m, m2_exact=ex.m2, energy_enumerated=ex.e,
                   specific_heat_enumerated=ex.C)
        lines.append(f"enumeration: <|m|> = {ex.abs_m:.10g}, <m^2> = {ex.m2:.10g}")
    if not lines:
        lines.append("no exact target at this size (A-R6: <|m|>, <m^2> exact only by enumeration on small tori)")
    return Out({"exact_targets": out}, "; ".join(lines), notes=("exact targets (printed, compared in tests): "
                                                                 + "; ".join(lines),))


def _stat(v):
    und = v.get("mc_status") == "undetermined"
    return Stability("not-computed", "sample", f"{SS}; {LPS}", "Monte Carlo over named seeds: NOT final"
                     + ("; UNDETERMINED: the equilibration diagnostic drifted or a tau_int is unreliable" if und else ""))


FINALITY = {**{k: _stat for k in list(REN.values()) + ["tau_int_energy", "tau_int_m2"]},
            "beta_c": Stability("exact-by-literature", "exact", BDC, "Theorem 3 (q >= 2, H = -sum delta); at q = 2 "
                                "halved for H = -sum s s (beta_Potts = 2 beta_Ising, derived here); a second route "
                                "-log(1 - p_sd(q)) agrees to 1e-30")}
META = ("algorithm", "q", "L", "beta", "bond_probability", "mc_status", "equilibration", "n_seeds", "sweeps",
        "burn_in", "tau_reliable", "wolff_flips_per_sweep", "exact_targets")
PARAMS = ("algorithm", "q", "L", "beta", "seeds", "sweeps", "burn_in")
CHEAP = {"q": 3, "L": 2, "sweeps": 16, "burn_in": 4, "seeds": (0, 1)}


def cluster_mc():
    steps = [fstep("fk-coupling", f"{BDC}; {SS}; {LPS}", _coupling), fstep("cluster-mc", f"{SS}; {LPS}", _mc),
             fstep("exact-targets", f"{KAUF}; montecarlo/exact_small.py (enumeration)", _exact)]
    refs = [RefCheck("cluster_mc", "potts_beta_c_q3", lambda v: v["beta_c"], route="sample", params=CHEAP)]
    spec = ModelSpec("potts", "potts", "none", constraint="L x L torus, cluster Monte Carlo")
    return entry("cluster_mc", spec, f"{SS}; {LPS}; {BDC}", {"sample": steps}, refs, {"sample": CAVEAT},
                 "Swendsen-Wang and Wolff cluster Monte Carlo (Metropolis baseline) for the Ising and q-state "
                 "Potts models on L x L tori: energy, magnetisation, specific heat, tau_int over named seeds",
                 default_route="sample", finality=FINALITY, meta=META, parameters={"sample": PARAMS},
                 rigor={"sample": "statistical"})


__all__ = ["ALGORITHMS", "cluster_mc"]
