"""Catalog entry `online_perceptron` (O1), route `online`: online learning of a perceptron teacher.

Model (online/odes.py; Biehl & Riegler 1994 Eqs. (1)-(6)): teacher |B| = 1, inputs xi_j i.i.d.
N(0, 1), label b = sign(B . xi) flipped with probability `noise` (Copelli et al. 1997 Eq. (7)),
update J <- J + (eta / N) F(h, b, Q) b xi, R = J . B, Q = J . J, alpha = examples / N,
eps = arccos(R / sqrt Q) / pi against the true teacher.  The N -> oo ODEs are physics-level (self-
averaging assumed, as in the papers; no concentration theorem fetched), so no value is final.
Run parameters:
  rule              required: hebb | perceptron | adatron | optimal
  eta, noise        learning rate (> 0, default 1; the optimal rule needs 1) and flip probability
                    (in [0, 1/2), default 0); noise_estimate: the noise the optimal rule assumes
  R0, Q0            initial state (default 0, 1, Biehl-Riegler Fig. 1; the optimal rule needs R0 > 0)
  (always)          asymptote eps ~ C alpha^-p from the ODE at alpha = 1e4 (eps_exponent measured,
                    eps_constant), for noise = 0 or the Hebb rule; for the optimal rule also
                    eps_constant_formula = 2 / I(noise) (derived integral)
  alpha             the learning curve at alpha: eps, R, Q, rho (Hebb: + its closed form)
  optimal, noise>0  noise-robustness boundaries in the estimate (Copelli et al. 1997 Fig. 1a)
  N (seeds)         finite-N simulation at alpha over named seeds (default 0-19): eps_sim
"""
import math

from .catalog import Out, RefCheck, Stability, entry, fstep, has
from .core import Refusal
from .csp import _checks
from .online import odes as O
from .online import optimal as OPT
from .online import simulate as SIM
from .spec import ModelSpec

BR = "Biehl & Riegler, Europhys. Lett. 28, 525 (1994) (preprint read, author's page)"
VKC = "Vicente, Kinouchi & Caticha, Machine Learning 32, 179 (1998), arXiv:cond-mat/9801297"
KC92 = "Kinouchi & Caticha, J. Phys. A 25, 6243 (1992) (not fetched)"
CEKBSRC = ("Copelli, Eichhorn, Kinouchi, Biehl, Simonetti, Riegler & Caticha, Europhys. Lett. 37, "
           "427 (1997) (preprint WUE-ITP-96-022 read, author's page)")
CAVEAT = ("online route: the N -> oo order-parameter ODEs of online learning (Biehl-Riegler 1994 "
          "Eqs. (4)-(5) with a learning rate and output noise), integrated here; self-averaging of R "
          "and Q is ASSUMED as in the papers (no concentration theorem fetched), so the values are "
          "physics-level, not final; the optimal rule is optimal only in its stated class "
          "(online/optimal.py OPTIMALITY_CLASS); finite-N values are statistical (named seeds, 99% "
          "Student-t), never certified")
_ODE_NOTE = ("N -> oo ODE limit: self-averaging assumed (BR 1994 p. 2); VKC 1998 sketch a "
             "Fokker-Planck argument (their ref. [22], not fetched); no theorem fetched: "
             "physics-level, NOT final")
ALPHA_BIG = 1e4
_SEEDS = tuple(range(20))


def _num(p, key, default=None):
    v = p.get(key, default)
    if v is None:
        return None
    if isinstance(v, bool):
        raise Refusal(f"precondition:{key}", f"{key}={v!r}: need a number")
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise Refusal(f"precondition:{key}", f"{key}={v!r}: need a number") from None
    if not math.isfinite(x):
        raise Refusal(f"precondition:{key}", f"{key}={v!r}: need a finite number")
    return x


def _flow(p):
    if p.get("rule") is None:
        raise Refusal("precondition:state-point", "the online_perceptron entry needs rule = hebb | "
                      "perceptron | adatron | optimal")
    if callable(p["rule"]):
        raise Refusal("unsupported:rule", "the catalog takes named rules; a callable rule goes "
                      "through online.odes.perceptron_flow")
    return O.perceptron_flow(p["rule"], eta=_num(p, "eta", 1), noise=_num(p, "noise", 0),
                             noise_estimate=_num(p, "noise_estimate"))


def _init(p):
    return _num(p, "R0", 0), _num(p, "Q0", 1)


def _state(p):
    f = _flow(p)
    O._init(f, _init(p))
    if p.get("N") is not None and p.get("alpha") is None:
        raise Refusal("precondition:state-point", "the finite-N simulation runs at alpha: give alpha")
    if p.get("N") is None and "seeds" in p:
        raise Refusal("precondition:known-parameter", "seeds is a finite-N parameter: need N")
    R0, Q0 = _init(p)
    notes = ()
    if f.rule == "optimal" and f.noise_est != f.noise:
        notes = (f"optimal rule built on noise_estimate {f.noise_est} != noise {f.noise}: a mis-estimated "
                 "modulation, OUTSIDE OPTIMALITY_CLASS (Copelli et al. 1997's robustness setting), not an "
                 "optimum",)
    return Out({"rule": f.rule, "eta": f.eta, "noise": f.noise, "noise_estimate": f.noise_est,
                "R0": R0, "Q0": Q0}, f"rule {f.rule}, eta {f.eta}, noise {f.noise} "
               f"(estimate {f.noise_est}), (R0, Q0) = ({R0}, {Q0}) (BR Eqs. (1)-(3))", notes=notes)


def _curve(p):
    f, a = _flow(p), _num(p, "alpha")
    tr = O.integrate(f, a, _init(p), alphas=(a,))
    R, Q, eps = float(tr.R[0]), float(tr.Q[0]), float(tr.eps[0])
    vals = {"eps": eps, "R": R, "Q": Q, "rho": R / math.sqrt(Q), "alpha": a}
    if f.rule == "hebb":
        _, _, ce = O.hebb_closed_form(a, f.eta, f.noise, _init(p))
        vals.update(eps_hebb_closed_form=ce, hebb_closed_form_gap=abs(ce - eps))
    return Out(vals, f"ODE (theta, ln Q) by Dormand-Prince 5(4), rtol 1e-9, to alpha = {a:g}: "
               f"eps = {eps:.6g}")


def _asym_when(p):
    return p.get("rule") == "hebb" or _num(p, "noise", 0) == 0


def _asymptote(p):
    f = _flow(p)
    slope, c = O.asymptote(f, _init(p), ALPHA_BIG)
    vals = {"eps_exponent": slope, "eps_constant": c, "alpha_big": ALPHA_BIG,
            "eps_exponent_derived": O.EXPONENT[f.rule]}
    notes = ()
    if f.rule == "optimal" and f.noise_est == f.noise:
        vals["eps_constant_formula"] = OPT.asymptotic_constant(f.noise)
    elif f.rule == "optimal":
        notes = (f"noise_estimate {f.noise_est} != noise {f.noise}: the mis-estimated rule is OUTSIDE "
                 "OPTIMALITY_CLASS (it is not the optimum of this flow), and 2 / I(noise) is the limit of "
                 "the matched rule only: eps_constant_formula not reported",)
    return Out(vals, f"log-slope {slope:.5f} (derived {O.EXPONENT[f.rule]:.4f}) and "
               f"alpha^p eps = {c:.6g} at alpha = {ALPHA_BIG:g}", notes=notes)


def _robust(p):
    nb = OPT.noise_boundaries(_num(p, "noise"))
    vals = {k: v for k, v in (("no_learning_boundary", nb.no_learning), ("robust_boundary", nb.robust),
                              ("robust_boundary_error", nb.robust_error)) if v is not None}
    notes = () if nb.no_learning is not None else (
        "no Lam in [0, noise] makes rho = 0 attractive at this noise level (no no-learning line)",)
    return Out(vals, "sign changes of drho/dalpha in the estimate Lam near rho = 0 and rho = 1",
               notes=notes)


def _finite(p):
    n = p["N"]
    if isinstance(n, bool) or not isinstance(n, int):
        raise Refusal("precondition:size", f"N={n!r}: need an int >= 10")
    f, a = _flow(p), _num(p, "alpha")
    seeds = _checks.seeds(p.get("seeds", _SEEDS))
    sim = SIM.simulate_perceptron(f.rule, n, a, seeds, eta=f.eta, noise=f.noise,
                                  noise_estimate=f.noise_est if f.rule == "optimal" else None,
                                  init=_init(p))
    b = sim.band
    return Out({"eps_sim": b.mean, "eps_sim_halfwidth99": b.half_width, "N": n, "seeds": list(seeds)},
               f"eps over seeds {list(seeds)} at N = {n}, alpha = {a:g}: {b.mean:.6g} +- "
               f"{b.half_width:.3g} (99% Student-t, df = {b.df})",
               errors={"eps_sim": b.stderr}, statistical=("eps_sim",),
               notes=("statistical: finite-N online learning over named seeds (online/simulate.py, "
                      "no code shared with the ODEs); it differs from the N -> oo ODE by O(1/N)",))


def online_perceptron():
    steps = [fstep("online-state-point", f"{BR}, Eqs. (1)-(3); {CEKBSRC}, Eq. (7)", _state),
             fstep("online-asymptote", f"{BR}, Eqs. (7), (8), (15); {VKC}, Tables I-II; {KC92}", _asymptote,
                   when=_asym_when, skip="noise > 0 with a rule other than Hebb: no power law is "
                   "claimed (the optimal rule's 2 / I(noise) is checked in the tests)"),
             fstep("online-learning-curve", f"{BR}, Eqs. (4)-(6)", _curve, when=has("alpha"),
                   skip="no alpha given"),
             fstep("online-noise-robustness", f"{CEKBSRC}, p. 5 and Fig. 1a", _robust,
                   when=lambda p: p.get("rule") == "optimal" and (_num(p, "noise", 0) or 0) > 0,
                   skip="needs rule = optimal and noise > 0"),
             fstep("online-finite-n", "online learning over named seeds (online/simulate.py)",
                   _finite, when=has("N"), skip="no N given")]
    refs = []
    for rule in ("hebb", "perceptron", "adatron", "optimal"):
        pr = {"rule": rule, **({"R0": "0.01"} if rule == "optimal" else {})}
        refs += [RefCheck("online_perceptron", f"{rule}_eps_exponent", lambda v: v["eps_exponent"],
                          params=pr, route="online"),
                 RefCheck("online_perceptron", f"{rule}_eps_constant", lambda v: v["eps_constant"],
                          params=pr, route="online")]
    for lam in ("0.3", "0.4"):
        refs.append(RefCheck("online_perceptron", f"no_learning_boundary_l{lam.replace('.', 'p')}",
                             lambda v: v["no_learning_boundary"], route="online",
                             params={"rule": "optimal", "noise": lam, "R0": "0.01"}))
    for lam in ("0.2", "0.3", "0.4"):
        refs.append(RefCheck("online_perceptron", f"robust_boundary_l{lam.replace('.', 'p')}",
                             lambda v: v["robust_boundary"], route="online",
                             params={"rule": "optimal", "noise": lam, "R0": "0.01"}))
    ode = Stability("not-computed", "online-ode", BR, _ODE_NOTE)
    asym = Stability("not-computed", "online-ode", f"{BR}; {VKC}, Tables I-II; {KC92}",
                     f"measured on the integrated ODE at alpha = {ALPHA_BIG:g} (log-slope over a factor "
                     f"2; constant alpha^p eps with the derived p); {_ODE_NOTE}")
    formula = Stability("not-computed", "online-ode", f"{VKC}, Eqs. (10), (13)-(17), (25)",
                        "2 / I(noise), the rho -> 1 limit of the optimal flow derived here (Copelli et "
                        "al. Eq. (8) up to the Gaussian weight; see online/optimal.py); " + _ODE_NOTE)
    rob = Stability("not-computed", "online-ode", CEKBSRC,
                    "located on the ODE flow (sign of drho/dalpha at rho = 1e-6 and at theta = 1e-4); "
                    + _ODE_NOTE)
    sim = Stability("not-computed", "finite-N", "online/simulate.py",
                    "statistical: finite-N mean +- stderr over the named seeds (99% Student-t half-width "
                    "in eps_sim_halfwidth99), never certified: NOT final")
    fin = {**{k: ode for k in ("eps", "R", "Q", "rho", "eps_hebb_closed_form")},
           **{k: asym for k in ("eps_exponent", "eps_constant")}, "eps_constant_formula": formula,
           "no_learning_boundary": rob, "robust_boundary": rob, "eps_sim": sim}
    meta = ("rule", "eta", "noise", "noise_estimate", "R0", "Q0", "alpha", "alpha_big",
            "eps_exponent_derived", "hebb_closed_form_gap", "robust_boundary_error",
            "eps_sim_halfwidth99", "N", "seeds")
    spec = ModelSpec("online_perceptron", "spherical", "gaussian", constraint="online", prior="teacher")
    return entry("online_perceptron", spec, f"{BR}; {VKC}; {KC92}; {CEKBSRC}", {"online": steps}, refs,
                 {"online": CAVEAT}, "online perceptron learning of a perceptron teacher: Hebb, "
                 "perceptron, AdaTron and the variationally optimal rule; output noise and its "
                 "robustness diagram", default_route="online", finality=fin, meta=meta,
                 parameters={"online": ("rule", "alpha", "eta", "noise", "noise_estimate", "R0", "Q0",
                                        "N", "seeds")})


def online_entries():
    from .catalog_online_linear import online_linear
    return [online_perceptron(), online_linear()]


__all__ = ["online_entries", "online_perceptron"]
