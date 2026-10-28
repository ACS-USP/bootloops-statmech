"""Catalog entry `online_drift` (O3), route `online`: online learning of a drifting perceptron teacher.

Model (online/drift.py): the O1 engine (|B| = 1, inputs N(0, 1), label flipped with probability `noise`
= VKC's chi / 2, J <- J + (eta/N) F b xi, eps = arccos(rho)/pi against the current teacher, alpha =
examples / N) with a teacher that drifts after each example: random (B <- (B + sqrt(2D)/N u)/|.|,
autocorrelation e^{-D alpha}; VKC 1998 Eqs. (4), (26), KC93 Eqs. (3)-(6) with T = D) or deterministic
(B turned away from J by arccos(1 - D/N^2); VKC Eqs. (28)-(31)).  The N -> oo ODEs are physics-level
(self-averaging assumed), so no value is final.
Run parameters:
  rule              optimal (default) | annealed_hebb | step | symmetric (VKC Eqs. (22)-(25), each optimal
                    in its information class, drift_rules.VKC_CLASS) | hebb | perceptron | adatron (O1)
  drift             random (default) | deterministic
  D                 the drift rate; without D: the small-drift law eps_inf ~ c D^(1/delta) (noise = 0)
  eta, noise        learning rate (O1 rules; the optimal and VKC rules need 1) and flip probability;
                    noise_estimate: the noise the optimal rule assumes
  R0, Q0            initial state (default R0 = 0.01 for the rules that read rho, else 0; Q0 = 1)
  (D > 0)           the stationary tracking error eps_inf and the KC93 bound eps_inf_bound (the optimal
                    rule at the same noise, drift and D); annealed Hebb: eps_inf_closed; optimal, noise
                    0: eps_inf_kc93 (KC93's I(lambda), Gauss-Legendre in u)
  alpha             the learning curve: eps, R, Q, rho, and the bound eps_bound (optimal rule)
  N (seeds)         finite-N online learning at alpha over named seeds (default 0-19): eps_sim
"""
import math

from .catalog import Out, RefCheck, Stability, entry, fstep, has
from .catalog_online import _num, nothing_computed, sim_band
from .core import Refusal
from .csp import _checks
from .online import drift as DR
from .online import drift_rules as VR
from .online import simulate_drift as SIM
from .spec import ModelSpec

VKC = "Vicente, Kinouchi & Caticha, Machine Learning 32, 179 (1998), arXiv:cond-mat/9801297"
KC93 = ("Kinouchi & Caticha, J. Phys. A 26, 6161 (1993) (preprint IFUSP/P-1044 read, "
        "repositorio.usp.br)")
BS = ("Biehl & Schwarze, Europhys. Lett. 20, 733 (1992) and J. Phys. A 26, 2651 (1993) (not fetched; "
      "the drift models as VKC and KC93 restate them)")
CAVEAT = ("online route: the N -> oo order-parameter ODEs of online learning (O1) with a drifting teacher "
          "(VKC Eqs. (10), (27), (31); KC93 Eq. (14)), integrated here; self-averaging ASSUMED as in the papers, "
          "so the values are physics-level, not final; each optimal rule is optimal only in its stated class "
          "(drift_rules.VKC_CLASS, odes/optimal OPTIMALITY_CLASS); finite-N values are statistical (named seeds, "
          "99% Student-t)")
_ODE_NOTE = ("N -> oo ODE limit: self-averaging assumed (VKC p. 7 sketch a Fokker-Planck argument, their ref. "
             "[22], not fetched): physics-level, NOT final")
_SEEDS = tuple(range(20))
_READS_RHO = ("optimal",) + VR.VKC_RULES


def _D(p):
    d = _num(p, "D")
    if d is not None and d < 0:
        raise Refusal("precondition:drift", f"D={d}: need a drift rate D >= 0")
    return d


def _flow(p, rule=None, D=None, est=True):
    rule = p.get("rule", "optimal") if rule is None else rule
    if callable(rule) and not isinstance(rule, str):
        raise Refusal("unsupported:rule", "the catalog takes named rules; a callable goes through online.drift")
    ne = _num(p, "noise_estimate") if est else None
    return DR.drift_flow(rule, D=(_D(p) or 0.0) if D is None else D, kind=p.get("drift", "random"),
                         eta=_num(p, "eta", 1), noise=_num(p, "noise", 0), noise_estimate=ne)


def _init(p, rule=None):
    rule = p.get("rule", "optimal") if rule is None else rule
    return _num(p, "R0", 0.01 if rule in _READS_RHO else 0.0), _num(p, "Q0", 1)


def _state(p):
    f = _flow(p)
    R0, Q0 = _init(p)
    DR._start(f, (R0, Q0))
    if p.get("N") is not None and p.get("alpha") is None:
        raise Refusal("precondition:state-point", "the finite-N simulation runs at alpha: give alpha")
    if p.get("N") is None and "seeds" in p:
        raise Refusal("precondition:known-parameter", "seeds is a finite-N parameter: need N")
    rule = p.get("rule", "optimal")
    notes = (f"optimal only in its class: {VR.VKC_CLASS[rule]}",) if rule in VR.VKC_CLASS else ()
    return Out({"rule": rule, "drift": f.kind, "D": _D(p), "eta": _num(p, "eta", 1), "noise": f.noise, "R0": R0,
                "Q0": Q0}, f"rule {rule}, {f.kind} drift D = {_D(p)}, noise {f.noise} (VKC Eqs. (4), (26)-(31))",
               notes=notes)


def _small_when(p):
    return p.get("D") is None and p.get("rule", "optimal") != "hebb" and _num(p, "noise", 0) == 0


def _small(p):
    sd = DR.small_drift(p.get("rule", "optimal"), p.get("drift", "random"), eta=_num(p, "eta", 1))
    return Out({"eps_exponent": sd.exponent, "eps_exponent_derived": sd.exponent_derived,
                "eps_coefficient": sd.coefficient},
               f"eps_inf at D = {sd.Ds}: {sd.eps}; log-slope {sd.exponent:.6f} (derived 1/delta = "
               f"{sd.exponent_derived:.6f}), coefficient {sd.coefficient:.6f}")


def _stat_when(p):
    d = _D(p)
    return d is not None and d > 0 and not (p.get("rule") == "hebb" and p.get("alpha") is not None)


def _stationary(p):
    f, rule = _flow(p), p.get("rule", "optimal")
    s = DR.stationary(f)
    b = DR.stationary(_flow(p, rule="optimal", est=False))
    vals = {"eps_inf": s.eps, "theta_inf": s.theta, "eps_inf_bound": b.eps, "bound_gap": s.eps - b.eps}
    if s.Q is not None:
        vals["Q_inf"] = s.Q
    if rule == "annealed_hebb":
        vals["eps_inf_closed"] = VR.annealed_hebb_stationary(f.D, f.kind, f.noise)
    if rule == "optimal" and f.noise == 0 and p.get("noise_estimate") is None:
        vals["eps_inf_kc93"] = VR.kc93_stationary(f.D, f.kind)
    return Out(vals, f"stationary dtheta/dalpha = 0 (bisection): eps_inf = {s.eps:.9g}; KC93 bound "
               f"(optimal rule) {b.eps:.9g}")


def _curve(p):
    f, a, rule = _flow(p), _num(p, "alpha"), p.get("rule", "optimal")
    init = _init(p)
    tr = DR.integrate_drift(f, a, init, alphas=(a,))
    R, Q, eps = (float(x[0]) for x in (tr.R, tr.Q, tr.eps))
    binit = (max(init[0], 1e-3 * math.sqrt(init[1])), init[1])        # rho0_opt >= rho0: still a bound
    eb = float(DR.integrate_drift(_flow(p, rule="optimal", est=False), a, binit, alphas=(a,)).eps[0])
    vals = {"eps": eps, "R": R, "Q": Q, "rho": R / math.sqrt(Q), "eps_bound": eb, "alpha": a}
    if rule == "hebb" and f.kind == "random" and f.D_at(0) > 0:
        vals["closed_form_gap"] = abs(DR.hebb_drift_closed_form(a, f.D, _num(p, "eta", 1), f.noise, init)[2] - eps)
    return Out(vals, f"ODE (theta, ln Q) by Dormand-Prince 5(4), rtol 1e-9, to alpha = {a:g}: eps = {eps:.9g}; "
               f"KC93 bound {eb:.9g}")


def _finite(p):
    n = p["N"]
    if isinstance(n, bool) or not isinstance(n, int):
        raise Refusal("precondition:size", f"N={n!r}: need an int >= 10")
    a, rule = _num(p, "alpha"), p.get("rule", "optimal")
    seeds = _checks.seeds(p.get("seeds", _SEEDS))
    b, und = sim_band(lambda: SIM.simulate_drift(rule, n, a, seeds, D=_D(p) or 0.0, drift=p.get("drift", "random"),
                                                 eta=_num(p, "eta", 1), noise=_num(p, "noise", 0),
                                                 noise_estimate=_num(p, "noise_estimate"), init=_init(p)), n, seeds)
    if und:                         # a seed reached rho <= 0: labelled, the ODE values are kept
        return und
    return Out({"eps_sim": b.mean, "eps_sim_halfwidth99": b.half_width, "N": n, "seeds": list(seeds)},
               f"eps over seeds {list(seeds)} at N = {n}, alpha = {a:g}: {b.mean:.6g} +- {b.half_width:.3g} "
               f"(99% Student-t, df = {b.df})", errors={"eps_sim": b.stderr}, statistical=("eps_sim",),
               notes=("statistical: finite-N online learning with the teacher drifting in the simulation "
                      "(online/simulate_drift.py, no code shared with the ODEs); it differs from the ODE by O(1/N)",))


_FIG = {"fig4": ("random", ("0.1", "0.45")), "fig6": ("deterministic", ("0.035", "0.045"))}
_TABLE = ("annealed_hebb", "symmetric", "step", "optimal")


def online_drift():
    steps = [fstep("online-drift-state-point", f"{VKC}, Eqs. (3)-(4), (22)-(31); {KC93}, Eqs. (3)-(14)", _state),
             fstep("online-drift-small-drift", f"{VKC}, Tables I-II, Eq. (34); {KC93}, Eqs. (24)-(25)", _small,
                   when=_small_when, skip="D given, rule hebb (no tracking) or noise > 0 (the tables are noiseless)"),
             fstep("online-drift-stationary", f"{VKC}, Eqs. (27), (31), Figs. 4, 6; {KC93}, Eqs. (17)-(23)",
                   _stationary, when=_stat_when, skip="no D > 0 given, or plain Hebb at alpha (no stationary state)"),
             fstep("online-drift-learning-curve", f"{VKC}, Eqs. (10)-(12), Figs. 3, 5; {KC93}, Sec. 3", _curve,
                   when=has("alpha"), skip="no alpha given"),
             fstep("online-drift-finite-n", "online learning over named seeds (online/simulate_drift.py)",
                   _finite, when=has("N"), skip="no N given")]
    R = lambda key, get, **pr: RefCheck("online_drift", key, get, params=pr, route="online")
    refs = [R("random_optimal_coeff", lambda v: v["eps_coefficient"])]
    for rule in _TABLE:
        for kind, tag in (("random", "random"), ("deterministic", "det")):
            refs += [R(f"{tag}_{rule}_exponent", lambda v: v["eps_exponent"], rule=rule, drift=kind)]
            if (rule, kind) != ("optimal", "random"):
                refs += [R(f"{tag}_{rule}_coeff", lambda v: v["eps_coefficient"], rule=rule, drift=kind)]
    for tag, (kind, Ds) in _FIG.items():
        refs += [R(f"{tag}_{rule}_d{D.replace('.', 'p')}", lambda v: v["eps_inf"], rule=rule, drift=kind, D=D)
                 for D in Ds for rule in _TABLE]
    ode = Stability("not-computed", "online-ode", f"{VKC}; {KC93}", _ODE_NOTE)
    stat = Stability("not-computed", "online-ode", f"{VKC}, Eqs. (27), (31)", "stationary zero of dtheta/dalpha "
                     "located by bisection (a single stable root on the scan, else refused); " + _ODE_NOTE)
    small = Stability("not-computed", "online-ode", f"{VKC}, Tables I-II", "log-slope and coefficient of the ODE's "
                      "eps_inf at D = 1e-12, 1e-11; " + _ODE_NOTE)
    closed = Stability("not-computed", "online-ode", VKC, "closed form derived here from VKC Eqs. (22), (27), (31) "
                       "(same N -> oo flow): " + _ODE_NOTE)
    kc = Stability("not-computed", "online-ode", f"{KC93}, Eqs. (17)-(23)", "KC93's I(lambda) by Gauss-Legendre in u (their Eq. "
                   "(22) with the derived (1 + lambda^2)); " + _ODE_NOTE)
    sim = Stability("not-computed", "finite-N", "online/simulate_drift.py", "statistical: finite-N mean +- stderr "
                    "over the named seeds (99% Student-t half-width in eps_sim_halfwidth99), never certified: NOT final")
    fin = {**{k: ode for k in ("eps", "R", "Q", "rho", "eps_bound")},
           **{k: stat for k in ("eps_inf", "theta_inf", "Q_inf", "eps_inf_bound")},
           **{k: small for k in ("eps_exponent", "eps_coefficient")}, "eps_inf_closed": closed,
           "eps_inf_kc93": kc, "eps_sim": sim}
    meta = ("rule", "drift", "D", "eta", "noise", "R0", "Q0", "alpha", "bound_gap", "closed_form_gap",
            "eps_exponent_derived", "eps_sim_halfwidth99", "N", "seeds", "sim_status")
    steps = [*steps, nothing_computed(fin, "give alpha (the learning curve) or a drift D > 0 (the stationary state and the small-drift law)")]
    spec = ModelSpec("online_drift", "spherical", "gaussian", constraint="online", prior="teacher")
    return entry("online_drift", spec, f"{VKC}; {KC93}; {BS}", {"online": steps}, refs, {"online": CAVEAT},
                 "online learning of a perceptron teacher that drifts (randomly or deterministically, VKC 1998, "
                 "KC93): the O1 rules and VKC's annealed Hebb, step, symmetric and optimal modulations, their "
                 "stationary tracking errors, small-drift laws and the KC93 lower bound",
                 default_route="online", finality=fin, meta=meta,
                 parameters={"online": ("rule", "drift", "D", "eta", "noise", "noise_estimate", "R0", "Q0",
                                        "alpha", "N", "seeds")})


__all__ = ["online_drift"]
