"""Catalog entry `ridge` (ridge / ridgeless least squares, double descent, E7), route `replica`.

Model (learning/ridge.py; HMRT = Hastie, Montanari, Rosset & Tibshirani 2022): X in R^{n x p} with
i.i.d. N(0, 1) entries, y = X beta + sigma xi, |beta|^2 = r^2 (`signal`), gamma = p/n, ridge
(X^T X + n lam I)^{-1} X^T y.  Values are the EXCESS risk |beta_hat - beta|^2 (bias + variance);
`test_mse*` add the fresh label noise sigma^2.

Why `replica`: the learning curve of the linear student is the replica-symmetric result (the RS
saddle point of Loureiro et al. 2022, App. A / Eq. (A.38), evaluated by learning/ridge_rs.py), and
the entry reports it next to the random-matrix closed form (HMRT Cor. 5) it is checked against.
replicant's formal layer (derive*.py) does NOT derive this free energy: no Family with a teacher
replica and a quadratic loss is registered, so the RS equations are a cited closed form solved
here, not a level-(a) derivation.  The rigorous statements are random-matrix theory (HMRT, Dobriban
& Wager) and Gordon's comparison inequality (Loureiro et al. Thm 1).  Run parameters:
  gamma            required, > 0
  sigma2, signal   noise variance and |beta|^2 (default 1, 1)
  (always)         ridgeless limit risk_ridgeless (+ bias, variance, test_mse; skipped at gamma = 1,
                   where it diverges), optimal penalty lambda_opt, risk_opt, test_mse_opt (skipped
                   at sigma2 = 0: no positive minimiser)
  lam              ridge closed form risk, bias, variance, test_mse; with lam > 0 also the RS saddle
                   risk_rs, rs_V, rs_q, rs_m
  n (seeds)        finite-n R_X by SVD over named seeds (default 0-19; lam default 0): risk_sim
                   (mean +- stderr, 99% Student-t half-width), p = round(gamma n)
"""
import mpmath as mp

from .catalog import Out, RefCheck, Stability, entry, fstep, has
from .core import Refusal
from .csp import _checks
from .learning import ridge as R
from .learning import ridge_rs as RS
from .learning import ridge_sim as SIM
from .spec import ModelSpec

HMRT = ("Hastie, Montanari, Rosset & Tibshirani, Ann. Statist. 50, 949 (2022), arXiv:1903.08560")
DW = "Dobriban & Wager, Ann. Statist. 46, 247 (2018), arXiv:1507.03003, Thm 2.1, Eq. (8)"
AS = "Advani & Saxe, Neural Networks 132, 428 (2020), arXiv:1710.03667, Sec. 2.2, Eq. (21)"
LGC = ("Loureiro, Gerbelot, Cui, Goldt, Krzakala, Mezard & Zdeborova, J. Stat. Mech. (2022) 114001, arXiv:2102.08127, "
       "Thm 1 (Eq. (2.8)), App. A.5 Eqs. (A.37)-(A.38)")
CAVEAT = ("replica route: the RS saddle point of the linear student (Loureiro et al. 2022, a cited "
          "closed form specialised to isotropic design and solved here; replicant's formal layer does "
          "not derive it) and the random-matrix closed form of HMRT Cor. 5 it is checked against; the "
          "rigorous proofs are random-matrix theory (HMRT, Dobriban-Wager) and Gordon's inequality "
          "(Loureiro et al. Thm 1), for the proportional limit n, p -> oo, p/n -> gamma with isotropic "
          "features; finite-n values are statistical (named seeds, 99% Student-t), never certified")
_ASSUME = ("isotropic features (i.i.d. zero-mean unit-variance entries, finite 4 + eta moment), "
           "|beta|^2 = r^2, n, p -> oo with p/n -> gamma, almost sure limit")
_SEEDS = tuple(range(20))


def _g(p):
    return R._num(p["gamma"], "gamma", positive=True)


def _state(p):
    if p.get("gamma") is None:
        raise Refusal("precondition:state-point", "the ridge entry needs gamma = p/n (> 0)")
    if p.get("n") is None and "seeds" in p:
        raise Refusal("precondition:known-parameter", "seeds is a finite-n parameter: need n")
    s2, r2 = R._num(p.get("sigma2", 1), "sigma2"), R._num(p.get("signal", 1), "signal")
    lam = p.get("lam")
    return Out({"gamma": _g(p), "sigma2": s2, "signal": r2,
                "lam": None if lam is None else R._num(lam, "lambda")},
               "isotropic Gaussian design, linear teacher (HMRT Eqs. (1)-(2))")


def _ridgeless(p):
    g, s2, r2 = _g(p), p.get("sigma2", 1), p.get("signal", 1)
    b, v = R.ridgeless_bias_variance(g, s2, r2)
    with mp.workdps(30):
        return Out({"risk_ridgeless": b + v, "bias_ridgeless": b, "variance_ridgeless": v,
                    "test_mse_ridgeless": b + v + R._num(s2, "sigma2")},
                   f"min-norm least squares, HMRT Eq. (8), gamma {'<' if g < 1 else '>'} 1")


def _optimal(p):
    g, s2, r2 = _g(p), p.get("sigma2", 1), p.get("signal", 1)
    lam, risk = R.optimal_lambda(g, s2, r2), R.optimal_test_error(g, s2, r2)
    with mp.workdps(30):
        return Out({"lambda_opt": lam, "risk_opt": risk, "test_mse_opt": risk + R._num(s2, "sigma2")},
                   "lam* = sigma^2 gamma / r^2, R* = sigma^2 gamma m(-lam*) (HMRT Cor. 5)")


def _ridge(p):
    g, lam, s2, r2 = _g(p), p["lam"], p.get("sigma2", 1), p.get("signal", 1)
    b, v = R.ridge_bias_variance(g, lam, s2, r2)
    with mp.workdps(30):
        return Out({"risk": b + v, "bias": b, "variance": v, "test_mse": b + v + R._num(s2, "sigma2")},
                   "Marchenko-Pastur Stieltjes transform at z = -lam (HMRT Eq. (43))")


def _replica(p):
    g, lam, s2, r2 = _g(p), p["lam"], p.get("sigma2", 1), p.get("signal", 1)
    rs = RS.ridge_replica_rs(g, lam, s2, r2)
    gap = abs(rs.risk - R.ridge_test_error(g, lam, s2, r2))
    return Out({"risk_rs": rs.risk, "rs_V": rs.V, "rs_q": rs.q, "rs_m": rs.m,
                "rs_closed_form_gap": gap},
               f"RS saddle point (V root bracketed, q and m solved); |risk_rs - risk| = "
               f"{mp.nstr(gap, 3)}")


def _finite(p):
    n = p["n"]
    if isinstance(n, bool) or not isinstance(n, int):
        raise Refusal("precondition:size", f"n={n!r}: need an int >= 10")
    g, lam = _g(p), p.get("lam", 0)
    pdim = int(mp.nint(g * n))
    seeds = _checks.seeds(p.get("seeds", _SEEDS))
    band = SIM.ridge_ensemble(n, pdim, lam, p.get("sigma2", 1), seeds, p.get("signal", 1))
    return Out({"risk_sim": band.mean, "risk_sim_halfwidth99": band.half_width, "n": n, "p": pdim,
                "gamma_n": pdim / n, "seeds": list(seeds)},
               f"R_X over seeds {list(seeds)} at n = {n}, p = {pdim}: {band.mean:.6g} +- "
               f"{band.half_width:.3g} (99% Student-t, df = {band.df})",
               errors={"risk_sim": band.stderr}, statistical=("risk_sim",),
               notes=("statistical: finite-n R_X (HMRT Sec. 2.1) over named seeds; its mean differs "
                      "from the n -> oo limit by O(1/n) (for gamma < 1 exactly sigma^2 p/(n - p - 1) "
                      "for Gaussian X), so the band need not contain the limit",))


def _ridgeless_label(v):
    if v["gamma"] < 1:
        return Stability("exact-by-literature", "rs", f"{HMRT}, Prop. 2, Eq. (8)",
                         "gamma < 1: least squares risk -> sigma^2 gamma/(1 - gamma), no bias (HMRT "
                         "Prop. 2, a known random-matrix result they cite from Serdobolskii 2007 and "
                         f"Tulino-Verdu 2004, not fetched); {_ASSUME}")
    return Stability("exact-by-literature", "rs", f"{HMRT}, Thm 1, Eqs. (6)-(8)",
                     "gamma > 1: bias -> r^2 (1 - 1/gamma), variance -> sigma^2/(gamma - 1) "
                     f"(HMRT Thm 1); {_ASSUME}")


def _ridge_label(v):
    if v.get("lam") == 0:
        return _ridgeless_label(v)
    return Stability("exact-by-literature", "rs", f"{HMRT}, Cor. 5, Eq. (43); Thm 6, Eqs. (38)-(39)",
                     "lam > 0: R -> r^2 lam^2 m'(-lam) + sigma^2 gamma (m - lam m') (HMRT Cor. 5); the "
                     "split into bias (the r^2 term) and variance (the sigma^2 term) is Thm 6 "
                     f"(B proportional to |beta|^2, V to sigma^2); {_ASSUME}")


def _sim_label():
    return Stability("not-computed", "finite-n", "learning/ridge_sim.py",
                     "statistical: finite-n mean +- stderr over the named seeds (99% Student-t "
                     "half-width in risk_sim_halfwidth99), never certified: NOT final")


def ridge():
    steps = [fstep("ridge-state-point", f"{HMRT}, Eqs. (1)-(2), (5)", _state),
             fstep("ridge-ridgeless-limit", f"{HMRT}, Prop. 2, Thm 1, Eq. (8)", _ridgeless,
                   when=lambda p: _g(p) != 1, skip="gamma = 1: the ridgeless risk diverges "
                   "(the interpolation peak)"),
             fstep("ridge-optimal-penalty", f"{HMRT}, Cor. 5; {DW}; {AS}", _optimal,
                   when=lambda p: R._num(p.get("sigma2", 1), "sigma2") > 0,
                   skip="sigma2 = 0: no positive minimiser (the ridgeless limit is the infimum)"),
             fstep("ridge-closed-form", f"{HMRT}, Cor. 5, Eq. (43)", _ridge, when=has("lam"),
                   skip="no lam given"),
             fstep("ridge-replica-rs", LGC, _replica,
                   when=lambda p: p.get("lam") is not None and R._num(p["lam"], "lambda") > 0,
                   skip="needs lam > 0"),
             fstep("ridge-finite-n", "SVD fit over named seeds (learning/ridge_sim.py)", _finite,
                   when=has("n"), skip="no n given")]
    refs = []
    for key, g, s2, r2 in (("ridgeless_g0p5_s1_r1", "0.5", "1", "1"),
                           ("ridgeless_g0p9_s1_r1", "0.9", "1", "1"),
                           ("ridgeless_g1p1_s1_r1", "1.1", "1", "1"), ("ridgeless_g2_s1_r1", "2", "1", "1"),
                           ("ridgeless_g4_s0p25_r1", "4", "0.25", "1")):
        refs.append(RefCheck("ridge", key, lambda v: v["risk_ridgeless"],
                             params={"gamma": g, "sigma2": s2, "signal": r2}))
    for key, g, lam, s2, r2 in (("ridge_g1_lam0p5_s1_r1", "1", "0.5", "1", "1"),
                                ("ridge_g2_lam0p1_s0p5_r2", "2", "0.1", "0.5", "2"),
                                ("ridge_g0p5_lam1_s1_r1", "0.5", "1", "1", "1")):
        pr = {"gamma": g, "lam": lam, "sigma2": s2, "signal": r2}
        refs += [RefCheck("ridge", key, lambda v: v["risk"], params=pr),
                 RefCheck("ridge", key, lambda v: v["risk_rs"], params=pr)]
    for key, g, a2 in (("optimal_test_mse_g0p5_a1", "0.5", "1"), ("optimal_test_mse_g1_a1", "1", "1"),
                       ("optimal_test_mse_g2_a1", "2", "1"), ("optimal_test_mse_g2_a5", "2", "5")):
        refs.append(RefCheck("ridge", key, lambda v: v["test_mse_opt"],
                             params={"gamma": g, "sigma2": "1", "signal": a2}))
    for key, r2 in (("lambda_opt_g2_s1_r1", "1"), ("lambda_opt_g2_s1_r5", "5")):
        refs.append(RefCheck("ridge", key, lambda v: v["lambda_opt"],
                             params={"gamma": "2", "sigma2": "1", "signal": r2}))
    opt = Stability("exact-by-literature", "rs", f"{HMRT}, Cor. 5; {DW}",
                    "lam* = sigma^2 gamma / r^2 minimises the limiting ridge risk, R* = sigma^2 gamma "
                    "m(-lam*) (HMRT Cor. 5; DW Thm 2.1 part 2 at unit noise, their Eq. (8) for R* + 1; "
                    f"Advani-Saxe Sec. 2.2 state the same optimal strength); {_ASSUME}")
    rs = Stability("exact-by-literature", "rs", LGC,
                   "RS saddle point of ridge (replica derivation App. A, heuristic; proved for Gaussian "
                   "covariates by Gordon's inequality, Thm 1), specialised by replicant to Psi = Phi = "
                   "Omega = I, p = d, with the label noise entering q_hat as rho + sigma^2 (Thm 1 Eq. (2.8) "
                   "with f0(x) = x + sigma xi; Eq. (A.38) is printed for sigma = 0) and lam_L = lam/gamma; "
                   "the run reports |risk_rs - risk| (rs_closed_form_gap)")
    fin = {**{k: _ridgeless_label for k in ("risk_ridgeless", "bias_ridgeless", "variance_ridgeless",
                                            "test_mse_ridgeless")},
           **{k: _ridge_label for k in ("risk", "bias", "variance", "test_mse")},
           **{k: opt for k in ("lambda_opt", "risk_opt", "test_mse_opt")},
           **{k: rs for k in ("risk_rs", "rs_V", "rs_q", "rs_m")},
           "risk_sim": lambda v: _sim_label()}
    meta = ("gamma", "sigma2", "signal", "lam", "rs_closed_form_gap", "risk_sim_halfwidth99", "n", "p",
            "gamma_n", "seeds")
    spec = ModelSpec("ridge", "real", "gaussian", constraint="linear teacher, ridge / min-norm student")
    return entry("ridge", spec, HMRT, {"replica": steps}, refs, {"replica": CAVEAT},
                 "ridge / ridgeless least squares on isotropic Gaussian design: double descent at "
                 "gamma = 1, optimal ridge, RS saddle point", default_route="replica", finality=fin,
                 meta=meta, parameters={"replica": ("gamma", "lam", "sigma2", "signal", "n", "seeds")})


def learning_entries():
    return [ridge()]


__all__ = ["learning_entries", "ridge"]
