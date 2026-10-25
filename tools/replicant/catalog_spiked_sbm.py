"""Catalog entries `spiked_wigner` (route `amp`) and `sbm` (route `cavity`), E6b.

spiked_wigner: Y = (lambda/N) x x^T + W, W ~ GOE(N) (inference/spiked.py states the convention and
its map to Lelarge-Miolane's lambda_LM = lambda^2).  Route `amp`: Bayes-optimal state evolution
and finite-N Bayes AMP; no replica functional is evaluated (the RS potential enters only as the
check that the SE root is its global minimum).  Run parameters:
  (none)        lambda_c of the prior (default gaussian)
  snr           SE at lambda = snr: gamma_se, q_se, overlap_se, mmse_se, matrix_mmse_se
  snr, N        Bayes AMP on N x N matrices over named seeds (default 0-7, iters 30): q_amp,
                overlap_amp, mmse_amp, matrix_mmse_amp (mean +- stderr, 99% t half-width)
  prior         gaussian | rademacher
sbm: two groups, edge probability a/N inside, b/N across (inference/sbm.py).  Route `cavity`:
BP on the graph is the cavity method; the Kesten-Stigum line is the instability of its
factorized fixed point (DKMZ Sec. IV.A).  Run parameters:
  a, b          ks_margin tau = (a - b)^2 / (2 (a + b)) (detectable iff tau > 1)
  c             the KS point at mean degree c: a_ks, b_ks, inverse_eps_ks = a_ks / b_ks
  a, b, N       BP over named seeds (default 0-9, iters 2000): bp_overlap (statistical)
"""
from .catalog import Out, RefCheck, Stability, entry, fstep, has
from .core import Refusal
from .csp import _checks
from .inference import sbm as B
from .inference import spiked as S
from .spec import ModelSpec

LM = ("Lelarge & Miolane, Probab. Theory Relat. Fields 173, 859 (2019), arXiv:1611.03888, "
      "Thm 1, Prop. 4, Sec. 2.3")
DAM = ("Deshpande, Abbe & Montanari, arXiv:1507.08685 (2015; Information and Inference 2017), "
       "Thm 1.4, Thm 4.3, Lemma 3.3")
MV = ("Montanari & Venkataramanan, Ann. Statist. 49, 321 (2021), arXiv:1711.01682, Thm 2, "
      "Eqs. (2.24), (2.28)-(2.29), Prop. 2.2, Cor. 2.3")
DKMZ = ("Decelle, Krzakala, Moore & Zdeborova, Phys. Rev. E 84, 066106 (2011), arXiv:1109.3041, "
        "Eqs. (26)-(28), (44)")
MNS15 = ("Mossel, Neeman & Sly, arXiv:1202.1499 ('Reconstruction and estimation in the planted "
         "partition model', Probab. Theory Relat. Fields 2015), Thm 2.1")
MAS = "Massoulie, arXiv:1311.3085 (STOC 2014), Thm 1.1"
MNS18 = "Mossel, Neeman & Sly, arXiv:1311.4115 (Combinatorica 2018), Thm 2.1"
SW_CAVEAT = ("AMP route: state evolution is the N -> oo description of Bayes AMP (MV Thm 2, spectral "
             "start, lambda > 1); its fixed point gives the Bayes-optimal q of LM Thm 1 when it is "
             "the global minimum of the RS potential (checked by the run); finite-N AMP values are "
             "statistical (mean +- stderr over named seeds, 99% Student-t), never certified")
SBM_CAVEAT = ("cavity route: BP with the external field (DKMZ) on sampled graphs, statistical over "
              "named seeds; the Kesten-Stigum line is a theorem (MNS 2015 below, Massoulie 2014 / "
              "MNS 2018 above), BP from random messages is not proved to reach it")
_SEEDS_SW, _SEEDS_SBM = tuple(range(8)), tuple(range(10))


def _stray(p, finite, need):
    if p.get("N") is None:
        bad = [k for k in finite if k in p]
        if bad:
            raise Refusal("precondition:known-parameter", f"{', '.join(bad)}: finite-N parameters "
                          f"need N (and {need})")


def _band(key, band):
    return {key: band.mean, f"{key}_halfwidth99": band.half_width}


# ---- spiked Wigner -------------------------------------------------------------------------------
def _sw_threshold(p):
    _stray(p, ("seeds", "iters"), "snr")
    if p.get("N") is not None and p.get("snr") is None:
        raise Refusal("precondition:state-point", "finite-N AMP needs snr (and N)")
    prior = p.get("prior", "gaussian")
    return Out({"lambda_c": S.spiked_threshold(prior), "prior": prior},
               "lambda_c = (-mmse'(0))^(-1/2) from the linearised SE map at gamma = 0")


def _sw_se(p):
    se = S.spiked_state_evolution(p["snr"], p.get("prior", "gaussian"))
    return Out({"snr": se.snr, "gamma_se": se.gamma, "q_se": se.q, "overlap_se": se.overlap,
                "mmse_se": se.mmse, "matrix_mmse_se": se.matrix_mmse,
                "se_fixed_points": [float(g) for g in se.fixed_points], "se_bayes": se.bayes},
               se.note)


def _sw_finite(p):
    seeds, iters = _checks.seeds(p.get("seeds", _SEEDS_SW)), p.get("iters", 30)
    ens = S.spiked_amp_ensemble(p["N"], p["snr"], p.get("prior", "gaussian"), seeds=seeds,
                                iters=iters)
    vals = {"N": p["N"], "seeds": list(seeds), "iters": iters,
            "amp_converged_fraction": sum(ens.converged) / len(seeds)}
    keys = {"q": "q_amp", "overlap": "overlap_amp", "mse": "mmse_amp", "matrix_mse": "matrix_mmse_amp"}
    for k, name in keys.items():
        vals.update(_band(name, ens.bands[k]))
    b = ens.bands["q"]
    return Out(vals, f"Bayes AMP over seeds {list(seeds)}: q = {b.mean:.6g} +- {b.half_width:.3g} "
               f"(99% Student-t, df = {b.df})",
               errors={name: ens.bands[k].stderr for k, name in keys.items()},
               statistical=tuple(keys.values()), notes=(ens.note,))


def _sw_rs(v):
    if v.get("se_bayes") and v.get("snr") != 1:
        return Stability("exact-by-literature", "rs", LM,
                         "LM Thm 1: lim MMSE_n = 1 - q*^2 with q* the maximiser of F(lambda_LM, q), "
                         "lambda_LM = lambda^2 (for lambda in D, all but countably many); this run "
                         "computed that its SE root is the global minimum of the RS potential (DAM "
                         "Eq. (9)), i.e. q* = gamma*/lambda^2")
    return Stability("not-computed", "rs", LM, "the SE root was not shown to be the global minimum "
                     "of the RS potential (or lambda = 1, the transition point): NOT final")


def _sw_amp(v):
    if v.get("se_bayes") and v.get("snr", 0) > 1:
        return Stability("exact-by-literature", "amp-se", MV,
                         "MV Thm 2, Eqs. (2.28)-(2.29): the N -> oo then t -> oo cosine and "
                         "sign-resolved MSE of Bayes AMP with spectral start (lambda > 1; F "
                         "Lipschitz: bounded support or log-concave prior); Bayes optimal for the "
                         "cosine by Prop. 2.2 / Cor. 2.3 when the prior has bounded support "
                         "(rademacher; the gaussian prior is not covered by Prop. 2.2)")
    return Stability("not-computed", "amp-se", MV, "MV Thm 2 needs lambda > 1 (and the SE root to be "
                     "the global minimum of the RS potential): NOT final")


def _stat(src):
    return Stability("not-computed", "amp-finite-N", src,
                     "statistical: finite-N mean +- stderr over the named seeds (99% Student-t "
                     "half-width in <key>_halfwidth99), never certified: NOT final")


def spiked_wigner():
    steps = [fstep("spiked-threshold", f"{LM}; {DAM}", _sw_threshold),
             fstep("spiked-state-evolution", f"{MV}; {LM}; {DAM}", _sw_se, when=has("snr"),
                   skip="no snr given"),
             fstep("spiked-amp-finite-n", f"{MV} (finite-N iterates: statistical)", _sw_finite,
                   when=has("snr", "N"), skip="needs snr and N")]
    refs = [RefCheck("spiked", "lambda_c_gaussian", lambda v: v["lambda_c"], route="amp"),
            RefCheck("spiked", "lambda_c_rademacher", lambda v: v["lambda_c"],
                     params={"prior": "rademacher"}, route="amp")]
    for tag, lam in (("0p8", "0.8"), ("1p5", "1.5"), ("2", "2"), ("3", "3")):
        refs += [RefCheck("spiked", f"q_gauss_snr{tag}", lambda v: v["q_se"], params={"snr": lam},
                          route="amp"),
                 RefCheck("spiked", f"matrix_mmse_gauss_snr{tag}", lambda v: v["matrix_mmse_se"],
                          params={"snr": lam}, route="amp")]
    lc = Stability("exact-by-literature", "rs", f"{LM}; {DAM}",
                   "threshold for nontrivial estimation (LM Eq. (12)) at lambda = 1 for both priors "
                   "(LM Sec. 2.3; DAM Thm 1.4 for rademacher); for these priors it equals the "
                   "spectral threshold (LM Sec. 2.3, citing Feral-Peche 2007 and Benaych-Georges-"
                   "Nadakuditi 2011, not fetched) and the SE instability of gamma = 0 computed here")
    stat = _stat(MV)
    fin = {"lambda_c": lc, "gamma_se": _sw_rs, "q_se": _sw_rs, "matrix_mmse_se": _sw_rs,
           "overlap_se": _sw_amp, "mmse_se": _sw_amp,
           **{k: stat for k in ("q_amp", "overlap_amp", "mmse_amp", "matrix_mmse_amp")}}
    meta = ("prior", "snr", "se_fixed_points", "se_bayes", "N", "seeds", "iters",
            "amp_converged_fraction", "q_amp_halfwidth99", "overlap_amp_halfwidth99",
            "mmse_amp_halfwidth99", "matrix_mmse_amp_halfwidth99")
    spec = ModelSpec("spiked_wigner", "real", "gaussian", constraint="rank-one spike, Bayes-optimal",
                     prior="gaussian")
    return entry("spiked_wigner", spec, LM, {"amp": steps}, refs, {"amp": SW_CAVEAT},
                 "rank-one spiked Wigner: Bayes-optimal SE, threshold lambda_c = 1, finite-N Bayes AMP",
                 default_route="amp", finality=fin, meta=meta,
                 parameters={"amp": ("snr", "prior", "N", "seeds", "iters")})


# ---- stochastic block model ----------------------------------------------------------------------
def _sbm_check(p):
    has_ab = [k for k in ("a", "b") if p.get(k) is not None]
    if len(has_ab) == 1 or (not has_ab and p.get("c") is None):
        raise Refusal("precondition:state-point", "the sbm entry needs a and b (the KS margin, and "
                      "with N the BP overlap) or c (the KS point at mean degree c)")
    if p.get("N") is not None and not has_ab:
        raise Refusal("precondition:state-point", "finite-N BP needs a and b (and N)")
    _stray(p, ("seeds", "iters"), "a, b")


def _sbm_ks(p):
    _sbm_check(p)
    if p.get("a") is None:
        return Out({}, "no a, b given")
    ks = B.sbm_kesten_stigum(p["a"], p["b"])
    return Out({"ks_margin": ks.margin, "ks_detectable": ks.detectable,
                "mean_degree": ks.mean_degree, "ks_impossibility_proved": ks.impossibility_proved},
               f"tau = (a - b)^2 / (2 (a + b)) = {float(ks.margin):.6g}: "
               f"{'above' if ks.detectable else 'not above'} the Kesten-Stigum line")


def _sbm_line(p):
    line = B.sbm_ks_line(p["c"])
    return Out({"c": line.c, "a_ks": line.a, "b_ks": line.b, "inverse_eps_ks": line.ratio},
               "bisection of tau - 1 along a + b = 2c")


def _sbm_finite(p):
    seeds, iters = _checks.seeds(p.get("seeds", _SEEDS_SBM)), p.get("iters", 2000)
    ens = B.sbm_bp_ensemble(p["N"], p["a"], p["b"], seeds=seeds, iters=iters)
    n = len(seeds)
    vals = {**_band("bp_overlap", ens.band), "N": p["N"], "seeds": list(seeds), "iters": iters,
            "bp_converged_fraction": sum(ens.converged) / n,
            "bp_factorized_fraction": sum(ens.factorized) / n}
    return Out(vals, f"BP over seeds {list(seeds)}: overlap {ens.band.mean:.4g} +- "
               f"{ens.band.half_width:.3g} (99% Student-t, df = {ens.band.df})",
               errors={"bp_overlap": ens.band.stderr}, statistical=("bp_overlap",), notes=(ens.note,))


def _ks_label(v):
    if v.get("ks_detectable"):
        return Stability("exact-by-literature", "rs", f"{MAS}; {MNS18}",
                         "tau > 1: an efficient estimator correlated with the partition exists "
                         "(Massoulie Thm 1.1, MNS 2018 Thm 2.1)")
    if v.get("ks_impossibility_proved"):
        return Stability("exact-by-literature", "rs", MNS15,
                         "tau <= 1 and a + b > 2: no estimator is correlated with the partition "
                         "(MNS 2015 Thm 2.1, tau = 1 included)")
    return Stability("not-computed", "rs", MNS15, "tau <= 1 with a + b <= 2: outside the hypothesis "
                     "a + b > 2 of MNS 2015 Thm 2.1: NOT final")


def sbm():
    steps = [fstep("sbm-kesten-stigum", f"{DKMZ}; {MNS15}; {MAS}; {MNS18}", _sbm_ks),
             fstep("sbm-ks-line", f"{DKMZ}, Fig. 1 caption", _sbm_line, when=has("c"),
                   skip="no c given"),
             fstep("sbm-bp-finite-n", f"{DKMZ} (BP on sampled graphs: statistical)", _sbm_finite,
                   when=has("a", "b", "N"), skip="needs a, b and N")]
    refs = [RefCheck("sbm", "inverse_eps_c_q2_c3", lambda v: v["inverse_eps_ks"], params={"c": 3},
                     route="cavity")]
    line = Stability("exact-by-literature", "rs", f"{MNS15}; {MAS}; {MNS18}",
                     "the KS point on a + b = 2c (c > 1, so a + b > 2): detection impossible at and "
                     "below it (MNS 2015), possible above it (Massoulie 2014, MNS 2018)")
    stat = Stability("not-computed", "rs-bp-finite-N", DKMZ,
                     "statistical: mean +- stderr over named seeds (99% Student-t half-width in "
                     "bp_overlap_halfwidth99); BP from random messages is not proved to reach the "
                     "KS line (MNS 2018 Sec. 2.1): NOT final")
    fin = {"ks_margin": _ks_label, "a_ks": line, "b_ks": line, "inverse_eps_ks": line,
           "bp_overlap": stat}
    meta = ("ks_detectable", "mean_degree", "ks_impossibility_proved", "c", "N", "seeds", "iters",
            "bp_overlap_halfwidth99", "bp_converged_fraction", "bp_factorized_fraction")
    spec = ModelSpec("sbm", "ising", "none", constraint="two planted groups, sparse random graph")
    return entry("sbm", spec, DKMZ, {"cavity": steps}, refs, {"cavity": SBM_CAVEAT},
                 "two-group stochastic block model: Kesten-Stigum line, BP with the external field",
                 default_route="cavity", finality=fin, meta=meta,
                 parameters={"cavity": ("a", "b", "c", "N", "seeds", "iters")})


__all__ = ["sbm", "spiked_wigner"]
