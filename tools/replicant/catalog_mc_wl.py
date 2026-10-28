"""Catalog entry `wang_landau_mc` (M2): Wang-Landau estimates of the density of states of the 2D Ising model on the
L x L torus (the 1/t refinement of Belardinelli & Pereyra, or the original schedule), a multicanonical run with
the WL weights, and canonical averages from both (montecarlo/).
Route sample (statistical):
  wang-landau       named-seed walkers (montecarlo/wang_landau.py): ln g(E) per level (meta ln_g_profile: E, mean,
                    stderr over seeds; ln g(ground) = ln 2), and from it ln Z / N, u, C at beta; wl_status
                    "converged" or "undetermined"; beta_c (exact-by-literature, the default beta)
  multicanonical    walkers with the fixed weights exp(-S_WL(E)) of each seed (montecarlo/multicanonical.py), u, C
                    and <|m|> at beta by Berg's Eq. (2.15); an explicit beta outside the window rule is refused
                    precondition:reweight-window, the default beta there labelled; muca_status "equilibrated" or
                    "undetermined" (also when the WL run has not reached the ground level: no multicanonical run)
  exact-targets     printed, never refusing (a correct sampler is not refused at runtime): Kaufman's torus u, C at
                    beta (L <= 64), the exact g(E) of montecarlo/dos_exact.py (L <= 12 here) with the ln g error
                    profile, <|m|> by enumeration (L <= 4)
Run parameters: L (4), seeds (0 .. 7), walkers (per seed, 8), schedule (one_over_t | wang_landau), t_max (MC time
cap, 4000), F_final (1/1000), beta (beta_c or a rational > 0), muca_sweeps (400), muca_burn_in (40).
"""
from functools import lru_cache

import mpmath as mp
import numpy as np

from .catalog import UNCLAIMED, Out, RefCheck, Stability, entry, fstep
from .montecarlo import fk as FK
from .spec import ModelSpec

WL = ("Wang & Landau, arXiv:cond-mat/0011174v1 (PRL 86, 2050 (2001)) (PDF read), Eq. (1) and p. 2 (f0 = e, the 80% "
      "flatness rule, f -> sqrt f, ln g(ground) = ln 2, eps(X))")
BP = ("Belardinelli & Pereyra, PRE 75, 046701 (2007), arXiv:cond-mat/0701672 (PDF read), Eq. (1), Eq. (2), Sec. III "
      "steps (i)-(vi) (the 1/t refinement)")
BERG = ("Berg, arXiv:cond-mat/9909236 (Fields Inst. Commun. 26, 2000) (PDF read), Eqs. (1.5)-(1.8), (2.11), (2.15), "
        "(2.16); Berg & Neuhaus, PRL 68, 9 (1992), arXiv:hep-lat/9202004 (PDF read), Eqs. (7), (10)")
BDC = ("Beffara & Duminil-Copin, arXiv:1006.5073v2 (PDF read), Theorem 3: beta_c = log(1 + sqrt q) (H = -sum delta), "
       "halved for H = -sum s s at q = 2")
CAVEAT = ("sample route, statistical: Wang-Landau walkers and multicanonical walkers over named seeds; means over "
          "seeds "
          "+- stderr; never certified, never final; wl_status / muca_status 'undetermined' when a walker missed "
          "F_final or a level, or the multicanonical energy drifts.  The 1/t estimate carries a finite-time "
          "systematic bias at the rarest (edge) levels, ln g underestimated by O(1/t) (measured in "
          "tests/test_mc_flat_slow.py; Wang & Landau's PRE 64, 056101 report a related edge effect of the original "
          "algorithm in restricted energy windows), larger than the seed stderr when many walkers are averaged; the "
          "multicanonical averages do not inherit it (tested: on 4 x 4 the WL canonical estimate is resolved as biased "
          "where the multicanonical one is not, tests/test_mc_flat_slow.py)")
STAT = ("energy_per_site_wl", "specific_heat_wl", "ln_z_per_site_wl", "energy_per_site_muca", "specific_heat_muca",
        "abs_magnetisation_muca")
DEFAULTS = {"L": 4, "seeds": tuple(range(8)), "walkers": 8, "schedule": "one_over_t", "t_max": 4000,
            "F_final": "1/1000", "beta": "beta_c", "muca_sweeps": 400, "muca_burn_in": 40}
PARAMS = tuple(DEFAULTS)
UNCLAIMED[("wang_landau_mc", "bp_one_over_t_eps_slope_l8")] = (
    "an error-against-time slope over MC time 8000 .. 32000 on the 8 x 8 torus (tens of seconds of walking): measured "
    "with its 99% Student-t band and printed in tests/test_mc_flat_slow.py test_error_saturation_and_its_slope; not "
    "a pipeline output")


def _p(p):
    from .montecarlo.wang_landau import check_params
    g = {k: p.get(k, d) for k, d in DEFAULTS.items()}
    g["L"], g["seeds"], g["walkers"], g["schedule"], g["t_max"], g["F_final"], _ = check_params(
        g["L"], g["seeds"], g["walkers"], g["schedule"], g["t_max"], g["F_final"], 10)
    for k in ("muca_sweeps", "muca_burn_in"):
        x = g[k]
        lo = 8 if k == "muca_sweeps" else 0
        if isinstance(x, bool) or not isinstance(x, (int, np.integer)) or not lo <= x <= 10 ** 6:
            from .core import Refusal
            raise Refusal(f"precondition:{k.replace('_', '-')}", f"{k} = {x!r}: need an int in {lo} .. 10^6")
    g["beta_f"] = float(FK.beta_value(2, g["beta"]))
    return g


@lru_cache(maxsize=8)
def _wl_run(L, seeds, walkers, schedule, t_max, F_final):
    from .montecarlo.wang_landau import wang_landau
    return wang_landau(L, seeds, walkers, schedule, t_max, F_final)


def _wl_of(g):
    return _wl_run(g["L"], g["seeds"], g["walkers"], g["schedule"], g["t_max"], g["F_final"])


@lru_cache(maxsize=8)
def _muca_run(key, sweeps, burn_in):
    from .montecarlo.multicanonical import muca_run
    r = _wl_run(*key)
    return muca_run(r.L, r.per_seed, r.k, r.seeds, r.walkers, sweeps, burn_in)


def _band_out(per, prefix):
    from .montecarlo import stats as ST
    vals, errs, hw = {}, {}, []
    for k, v in per.items():
        b = ST.band(v)
        vals[f"{k}{prefix}"], errs[f"{k}{prefix}"] = b.mean, b.stderr
        hw.append(f"{k}{prefix} +- {b.half_width:.3g}")
    return vals, errs, ", ".join(hw)


def _wl(p):
    from .montecarlo.dos_exact import canonical
    g = _p(p)
    r = _wl_of(g)
    N = g["L"] ** 2
    cs = [canonical(r.E, row, g["beta_f"], N) for row in r.per_seed]
    vals, errs, hw = _band_out({"energy_per_site": [c.u for c in cs], "specific_heat": [c.C for c in cs],
                                "ln_z_per_site": [c.ln_z for c in cs]}, "_wl")
    bc, detail = FK.critical_beta(2)
    vals.update(beta_c=bc, L=g["L"], schedule=g["schedule"], beta=g["beta"], wl_status=r.status, wl_summary=r.text,
                ln_g_profile=[[int(e), float(m), float(s)] for e, m, s in zip(r.E, r.mean, r.stderr)])
    errs["beta_c"] = mp.mpf(10) ** -30
    note = f"{r.text}; ln Z / N at beta {g['beta_f']:.6g}: {vals['ln_z_per_site_wl']:.6g}"
    return Out(vals, note, errors=errs, statistical=tuple(k for k in vals if k.endswith("_wl")),
               notes=(f"Wang-Landau: {r.text}; 99% Student-t half-widths over seeds: {hw}", detail))


def _muca(p):
    from .montecarlo.multicanonical import reweight
    g = _p(p)
    key = (g["L"], g["seeds"], g["walkers"], g["schedule"], g["t_max"], g["F_final"])
    if 0 not in set(int(x) for x in _wl_of(g).k):     # no ground level to start from: labelled, not refused
        why = "UNDETERMINED: the Wang-Landau run has not visited the ground level, no multicanonical run"
        return Out({"muca_status": "undetermined", "muca_summary": why}, why, notes=(why,))
    m = _muca_run(key, g["muca_sweeps"], g["muca_burn_in"])
    explicit = "beta" in p                    # an explicit out-of-window request is refused; the default is labelled
    rw = reweight(m, g["beta_f"], enforce=explicit)
    vals, errs, hw = _band_out({"energy_per_site": rw.per_seed["u"], "specific_heat": rw.per_seed["C"],
                                "abs_magnetisation": rw.per_seed["abs_m"]}, "_muca")
    status = m.status if _wl_of(g).status == "converged" and rw.in_window else "undetermined"
    vals.update(muca_status=status, muca_summary=m.text, reweight_summary=rw.text)
    return Out(vals, f"{m.text}; {rw.text}", errors=errs, statistical=tuple(k for k in vals if k.endswith("_muca")),
               notes=(f"multicanonical: {m.text}; 99% Student-t half-widths over seeds: {hw}",
                      f"reweighting (Berg Eq. (2.15)): {rw.text}"
                      + ("; UNDETERMINED: the Wang-Landau weights did not converge" if _wl_of(g).status != "converged"
                         else "") + ("" if rw.in_window else "; UNDETERMINED: the default beta is outside the "
                                     "multicanonical window (an explicit beta there is refused)")))


def _exact(p):
    g = _p(p)
    L, b = g["L"], g["beta"]
    out, lines = {}, []
    if L <= 64:
        from .lattice.torus_thermo import torus_thermo_kaufman
        u, c = torus_thermo_kaufman(L, L, b)
        with mp.workdps(20):
            out.update(energy_kaufman=float(u.mid()), specific_heat_kaufman=float(c.mid()))
        lines.append(f"Kaufman torus: u = {out['energy_kaufman']:.10g}, C = {out['specific_heat_kaufman']:.10g}")
    if L <= 12:
        from .core import Refusal
        from .montecarlo.dos_exact import exact_canonical, exact_dos
        from .montecarlo.wang_landau import error_profile
        ex = exact_canonical(L, FK.beta_value(2, b))
        out.update(energy_exact_dos=ex.u, specific_heat_exact_dos=ex.C, ln_z_per_site_exact=ex.ln_z)
        try:
            prof = error_profile(_wl_of(g), exact_dos(L))
            out.update(ln_g_max_deviation=float(np.abs(prof.deviation).max()), ln_g_mean_eps=prof.eps_mean.mean)
            lines.append(prof.text)
        except Refusal as r:
            lines.append(f"ln g profile not comparable: {r.detail}")
        lines.append(f"exact g(E): u = {ex.u:.10g}, C = {ex.C:.10g}, ln Z / N = {ex.ln_z:.10g}")
    if L <= 4:
        from .montecarlo.exact_small import exact_distribution
        out["abs_magnetisation_enumerated"] = exact_distribution(L, 2, b).abs_m
        lines.append(f"enumeration: <|m|> = {out['abs_magnetisation_enumerated']:.10g}")
    return Out({"exact_targets": out}, "; ".join(lines), notes=("exact targets (printed, compared in tests): "
                                                                 + "; ".join(lines),))


def _stat(key):
    def f(v):
        und = v.get(key) == "undetermined"
        return Stability("not-computed", "sample", f"{WL}; {BP}; {BERG}", "Monte Carlo over named seeds: NOT final"
                         + ("; UNDETERMINED" if und else ""))
    return f


FINALITY = {**{k: _stat("wl_status" if k.endswith("_wl") else "muca_status") for k in STAT},
            "beta_c": Stability("exact-by-literature", "exact", BDC, "Theorem 3 at q = 2, halved for the Ising "
                                "convention; a second route -log(1 - p_sd) agrees to 1e-30")}
META = ("L", "schedule", "beta", "wl_status", "wl_summary", "ln_g_profile", "muca_status", "muca_summary",
        "reweight_summary", "exact_targets")
CHEAP = {"L": 3, "walkers": 2, "t_max": 200, "F_final": "1/1000", "muca_sweeps": 60, "muca_burn_in": 4,
         "seeds": (0, 1)}      # stops at t_max before F_final: labelled undetermined (tests/test_mc_flat.py)


def wang_landau_mc():
    steps = [fstep("wang-landau", f"{WL}; {BP}", _wl), fstep("multicanonical", BERG, _muca),
             fstep("exact-targets", "montecarlo/dos_exact.py (Kaufman's integer polynomial); lattice/torus_thermo.py; "
                   "montecarlo/exact_small.py (enumeration)", _exact)]
    refs = [RefCheck("lattice", "beta_c_square", lambda v: v["beta_c"], route="sample", params=CHEAP)]
    spec = ModelSpec("ising", "ising", "none", constraint="L x L torus, flat-histogram Monte Carlo")
    return entry("wang_landau_mc", spec, f"{WL}; {BP}; {BERG}", {"sample": steps}, refs, {"sample": CAVEAT},
                 "Wang-Landau (1/t refinement) density of states and multicanonical sampling of the 2D Ising model on "
                 "L x L tori: ln g(E), canonical u, C, <|m|> at any beta, over named seeds",
                 default_route="sample", finality=FINALITY, meta=META, parameters={"sample": PARAMS},
                 rigor={"sample": "statistical"})


__all__ = ["CHEAP", "wang_landau_mc"]
