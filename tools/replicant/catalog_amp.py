"""Catalog route `amp` (E3b) for the `sk` and `teacher_student` entries: amp/ as a route.

  sk_amp_route() -> AmpRoute     state evolution of SK AMP; finite-N AMP when N is given
  ts_amp_route() -> AmpRoute     state evolution of Bayes-optimal GAMP (gaussian prior);
                                 finite-N GAMP when N is given
An AmpRoute carries the Steps, the honest caveat, the declared run parameters, the
finality declaration of every value key and the meta keys; catalog_entries.py adds it
to its entry.  The route calls amp/ only (no family, replica or cavity code).

Finality.  A state-evolution value is `exact-by-literature` only where the cited paper
proves SE for the setting: SK AMP with the Lipschitz tanh on a Gaussian (GOE-type) J,
Bayati & Montanari 2011 Thm 4 (and Bolthausen 2014 for this iteration).  GAMP's SE for
the perceptron (a GLM) is stated as Claim 1 by Barbier et al. 2019 (from Rangan 2011), so
its values are `not-computed`.  Finite-N values are statistical (mean +- stderr over the
named seeds; 99% Student-t half-width in the meta key <key>_halfwidth99), `not-computed`,
never final.  Defaults (echoed in the values): seeds 0-5, law gaussian; iters 100 (SK),
30 (GAMP).
"""
from dataclasses import dataclass

import mpmath as mp

from .catalog import Out, Stability, fstep, has
from .core import Refusal
from .csp import _checks

BM11 = ("Bayati & Montanari, IEEE Trans. Inf. Theory 57, 764 (2011), arXiv:1001.3448, Thm 4; "
        "Bolthausen, Commun. Math. Phys. 325, 333 (2014), arXiv:1201.2891")
BKMMZ = ("Barbier, Krzakala, Macris, Miolane & Zdeborova, PNAS 116, 5451 (2019), "
         "arXiv:1708.03395, Claim 1 and eq. (10); Rangan, IEEE ISIT 2011, arXiv:1010.5141")
SEEDS = (0, 1, 2, 3, 4, 5)
SK_CAVEAT = (
    "AMP route: state evolution (SE) is a deterministic N -> oo prediction for SK AMP "
    "(amp/sk.py), a theorem for Gaussian J at each fixed t (Bayati-Montanari 2011 Thm 4; "
    "Bolthausen 2014), not a statement about the Gibbs measure; finite-N AMP values are "
    "statistical (mean +- stderr over named seeds, 99% Student-t half-width), never "
    "certified; +-J matrices: numerical evidence only")
TS_CAVEAT = (
    "AMP route: Bayes-optimal GAMP (amp/gamp.py) with an i.i.d. N(0, 1) prior and teacher "
    "(the spherical model's overlap equation is the same); its state evolution is stated as "
    "a claim for GLMs (Barbier et al. 2019, Claim 1, from Rangan 2011), not a theorem here; "
    "Bayes-optimal only on the Nishimori line; finite-N values are statistical (mean +- "
    "stderr over named seeds, 99% Student-t half-width), never certified")


@dataclass(frozen=True)
class AmpRoute:
    steps: list
    caveat: str
    params: tuple
    finality: dict
    meta: tuple


def _need(p, key, model, finite):
    """The state point is required; the finite-N knobs (`finite`) are read only with N, so
    one given without N is refused rather than echoed and ignored."""
    if p.get(key) is None:
        raise Refusal("precondition:state-point", f"the {model} amp route needs {key}")
    stray = [k for k in finite if k in p] if p.get("N") is None else []
    if stray:
        raise Refusal("precondition:known-parameter",
                      f"{', '.join(stray)}: finite-N parameters need N (the {model} amp route "
                      "without N runs state evolution only, which reads "
                      f"{key}{', h' if model == 'sk' else ', learning'})")


def _stat(key, note):
    return Stability("not-computed", "amp-finite-N", BM11 if key == "q_amp" else BKMMZ,
                     "statistical: finite-N mean +- stderr over the named seeds (99% Student-t "
                     f"half-width in {key}_halfwidth99), never certified; " + note)


def _band_values(key, band):
    return {key: band.mean, f"{key}_halfwidth99": band.half_width}


# ---- SK ------------------------------------------------------------------------------
def _sk_se(p):
    from .amp import sk_state_evolution
    from .amp.sk import LAWS
    _need(p, "beta", "sk", ("seeds", "iters", "law"))
    law = p.get("law", "gaussian")
    if law not in LAWS:
        raise Refusal("precondition:known-law", f"law={law!r}; known: {', '.join(LAWS)}")
    se = sk_state_evolution(p["beta"], p.get("h", 0))
    return Out({"q_se": se.q, "law": law, "se_converged": se.converged,
                "se_fixed_points": [mp.nstr(f.q, 12) for f in se.fixed_points],
                "se_amp_converges": se.amp_converges}, "SE fixed point from Q_0 = 1: " + se.note)


def _sk_finite(p):
    from .amp import random_init_q, sk_amp_ensemble, sk_state_evolution
    seeds, iters, law = _checks.seeds(p.get("seeds", SEEDS)), p.get("iters", 100), p.get("law",
                                                                                      "gaussian")
    beta, h = p["beta"], p.get("h", 0)
    ens = sk_amp_ensemble(p["N"], float(mp.mpf(beta)), float(mp.mpf(h)), seeds=seeds,
                          iters=iters, law=law)
    se_t = sk_state_evolution(beta, h, q0=random_init_q(beta, h), iters=iters).trajectory[-1]
    vals = {**_band_values("q_amp", ens.band), "N": p["N"], "seeds": list(seeds),
            "iters": iters, "q_se_t": se_t, "q_amp_runs": ens.values,
            "amp_converged": ens.converged}
    return Out(vals, f"AMP over seeds {list(seeds)}: q = {ens.band.mean:.6g} +- "
               f"{ens.band.half_width:.3g} (99% Student-t, df = {ens.band.df}); SE at t = "
               f"{iters} from the same initial law: {se_t:.6g}",
               errors={"q_amp": ens.band.stderr}, statistical=("q_amp",), notes=(ens.note,))


_PM_ONE = ("+-J matrix: SE is not proved for it here (Bayati-Lelarge-Montanari 2015 cover "
           "polynomial nonlinearities); the value is the Gaussian-J SE, numerical evidence only "
           "for this law")


def _sk_q_se(v):
    if not v.get("se_converged"):
        return Stability("not-computed", "amp-se", BM11, "the SE trajectory did not converge")
    if v.get("law") != "gaussian":
        return Stability("not-computed", "amp-se", BM11, _PM_ONE)
    return Stability("exact-by-literature", "amp-se", BM11,
                     "what is proved: for Gaussian (GOE-type) J and the Lipschitz tanh, lim_N of "
                     "AMP's Q_t = <m^t, m^t> equals the SE Q_t at each fixed t (Thm 4 is stated "
                     "with diagonal variance 2/N; amp/sk.py uses a zero diagonal); q_se = lim_t "
                     "of SE from Q_0 = 1 (N -> oo first), the same limit as from AMP's random "
                     "start (Q_0 > 0) when one fixed point attracts; it is not a statement about "
                     "the Gibbs measure (at h = 0, beta > 1 the RS saddle it matches is "
                     "AT-unstable)")


def _sk_q_se_t(v):
    if v.get("law") != "gaussian":
        return Stability("not-computed", "amp-se", BM11, _PM_ONE)
    return Stability("exact-by-literature", "amp-se", BM11,
                     "SE Q_t at the finite t of the AMP runs, from the same initial law (proved "
                     "for each fixed t as N -> oo, Gaussian J)")


def sk_amp_route():
    steps = [fstep("amp-state-evolution", BM11, _sk_se),
             fstep("amp-finite-n", BM11 + " (finite-N iterates: statistical)", _sk_finite,
                   when=has("N"), skip="no N given (finite-N AMP runs only with N)")]
    fin = {"q_se": _sk_q_se, "q_se_t": _sk_q_se_t,
           "q_amp": _stat("q_amp", "AMP on sampled matrices (amp/sk.py)")}
    meta = ("se_converged", "se_fixed_points", "se_amp_converges", "N", "seeds", "iters",
            "law", "q_amp_halfwidth99", "q_amp_runs", "amp_converged")
    return AmpRoute(steps, SK_CAVEAT, ("beta", "h", "N", "seeds", "iters", "law"), fin, meta)


# ---- teacher-student -------------------------------------------------------------------
def _learning(p):
    learning = p.get("learning", "bayes")
    if learning not in ("bayes", "gibbs"):
        raise Refusal("precondition:known-learning",
                      f"learning = {learning!r}: use 'bayes' (posterior mean, GAMP's estimate) "
                      "or 'gibbs' (a posterior sample: overlap q by the Nishimori identity)")
    return learning


def _ts_se(p):
    from .amp import perceptron_state_evolution
    _need(p, "alpha", "teacher_student", ("seeds", "iters"))
    learning = _learning(p)
    se = perceptron_state_evolution(p["alpha"], "gaussian")
    eps = se.eps_bayes if learning == "bayes" else se.eps_gibbs
    return Out({"q_se": se.overlap, "mse_se": se.mse, "eps_se": eps, "learning": learning,
                "prior": "gaussian", "se_converged": se.converged},
               f"GAMP SE from q = 0 (gaussian prior): {se.note}")


def _ts_finite(p):
    from .amp import gamp_ensemble, perceptron_state_evolution
    seeds, iters = _checks.seeds(p.get("seeds", SEEDS)), p.get("iters", 30)
    ens = gamp_ensemble(p["N"], float(mp.mpf(p["alpha"])), "gaussian", seeds=seeds,
                        iters=iters)
    se_t = perceptron_state_evolution(p["alpha"], "gaussian", iters=iters).trajectory[-1]
    vals = {**_band_values("q_gamp", ens.band), **_band_values("eps_gamp", ens.eps_band),
            "N": p["N"], "seeds": list(seeds), "iters": iters, "q_se_t": se_t,
            "gamp_converged": ens.converged}
    return Out(vals, f"GAMP over seeds {list(seeds)}: q = {ens.band.mean:.6g} +- "
               f"{ens.band.half_width:.3g} (99% Student-t, df = {ens.band.df}); SE at t = "
               f"{iters}: {se_t:.6g}; eps_gamp is the plug-in (Bayes) error of the GAMP "
               "estimate whatever `learning`",
               errors={"q_gamp": ens.band.stderr, "eps_gamp": ens.eps_band.stderr},
               statistical=("q_gamp", "eps_gamp"), notes=(ens.note,))


def ts_amp_route():
    claim = Stability("not-computed", "gamp-se", BKMMZ,
                      "GAMP state evolution for GLMs is stated as Claim 1 by Barbier et al. "
                      "(from Rangan 2011, steps of the proof missing; Javanmard-Montanari 2013 "
                      "state SE for general AMP covering GAMP, not counted as a proof there): "
                      "no theorem is applied by this route.  The replica route's eps is "
                      "exact-by-literature (Barbier et al. Thm 2) and agrees with eps_se "
                      "(tests/test_gamp.py); the agreement checks an identity: with the "
                      "Gaussian prior the SE equation is Gyorgyi's spherical Gibbs equation")
    steps = [fstep("gamp-state-evolution", BKMMZ, _ts_se),
             fstep("gamp-finite-n", BKMMZ + " (finite-N iterates: statistical)", _ts_finite,
                   when=has("N"), skip="no N given (finite-N GAMP runs only with N)")]
    fin = {"q_se": claim, "mse_se": claim, "eps_se": claim, "q_se_t": claim,
           "q_gamp": _stat("q_gamp", "GAMP on sampled instances (amp/gamp.py)"),
           "eps_gamp": _stat("eps_gamp", "plug-in error of the GAMP estimate")}
    meta = ("learning", "prior", "se_converged", "N", "seeds", "iters", "q_gamp_halfwidth99",
            "eps_gamp_halfwidth99", "gamp_converged")
    return AmpRoute(steps, TS_CAVEAT, ("alpha", "learning", "N", "seeds", "iters"), fin, meta)


__all__ = ["AmpRoute", "sk_amp_route", "ts_amp_route"]
