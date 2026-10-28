"""Catalog entry `online_linear` (O2), route `online`: the linear perceptron, online versus offline.

Model (online/linear.py; Kinouchi & Caticha, Phys. Rev. E 52, 2878 (1995), Eqs. (1)-(8) with
I = 1): teacher |B|^2 = M, inputs xi_j i.i.d. N(0, 1), linear outputs b = B . xi and h = J . xi,
update J <- (1 - Omega/N) J + (eta/N) F xi, R = J . B, Q = J . J, rho = R / sqrt(Q M),
e_g = (M + Q - 2R)/2 (KC Eq. (4)), alpha = examples / N, noiseless examples.  The N -> oo ODEs are
physics-level (self-averaging assumed, as in the paper), so no value is final.
Run parameters:
  rule              optimal (default; KC Eq. (9), class in online/linear.py OPTIMALITY_CLASS) |
                    standard (F = b - h, KC Eq. (22))
  constraint        sphere (default; Q held at Q0 by the decay Omega) | none
  eta               standard rule: a learning rate > 0 or "greedy" (the eta maximising drho/dalpha);
                    the optimal rule needs 1
  M, R0, Q0         teacher norm (default 1); initial state (default R0 = 1e-3 M for the optimal rule
                    or the greedy eta, else 0; Q0 = M)
  (no alpha; sphere, eta = 1) asymptote at alpha = 60: rho_inf, eg_inf (KC Eqs. (24), (25)); optimal
                    at Q0 = M: tail_constant = e^alpha (1 - rho) at alpha = 20 (Eq. (31))
  (no alpha; optimal, sphere, Q0 = M) alpha_cross: online optimal vs offline standard (KC p. 11)
  alpha             the online curve (R, Q, rho, eg, eps_sign) and the offline T = 0 solutions at the
                    same alpha (online/linear_offline.py), with ridge (E7) as a second route for the
                    pseudoinverse
  N (seeds)         finite-N online learning at alpha over named seeds (default 0-19): eg_sim
"""
import math

from .catalog import Out, RefCheck, Stability, entry, fstep, has
from .catalog_online import _num, nothing_computed
from .core import Refusal
from .csp import _checks
from .online import linear as L
from .online import linear_offline as OFF
from .online import simulate_linear as SIM
from .spec import ModelSpec

KC95 = ("Kinouchi & Caticha, Phys. Rev. E 52, 2878 (1995) (preprint IFUSP/P-1157 read, "
        "repositorio.usp.br)")
SST = "Seung, Sompolinsky & Tishby, Phys. Rev. A 45, 6056 (1992) (not fetched; quoted by KC95 Eq. (27))"
HMRT = "Hastie, Montanari, Rosset & Tibshirani, Ann. Statist. 50, 949 (2022), arXiv:1903.08560, Eq. (8)"
CAVEAT = ("online route: the N -> oo order-parameter ODEs of online learning of a linear teacher (KC95 "
          "Eqs. (5)-(8)), integrated here; self-averaging of R and Q is ASSUMED as in the paper, so the "
          "values are physics-level, not final; the optimal rule is optimal only in its stated class "
          "(online/linear.py OPTIMALITY_CLASS); the offline values are the zero-temperature solutions "
          "(derived geometry, KC95 Sec. 4); finite-N values are statistical (named seeds, 99% Student-t)")
_ODE_NOTE = ("N -> oo ODE limit: self-averaging assumed (KC95 Sec. 3.1, 'in the limit N -> oo'); no "
             "theorem fetched: physics-level, NOT final")
_SEEDS = tuple(range(20))


def _flow(p):
    rule, cons = p.get("rule", "optimal"), p.get("constraint", "sphere")
    if callable(rule):
        raise Refusal("unsupported:rule", "the catalog takes named rules; a callable rule goes "
                      "through online.linear.linear_flow")
    eta = p.get("eta", 1)
    eta = "greedy" if eta == "greedy" else _num(p, "eta", 1)
    return L.linear_flow(rule, eta=eta, constraint=cons, M=_num(p, "M", 1))


def _init(p, f):
    r0 = 1e-3 * f.M if (f.rule == "optimal" or f.eta == "greedy") else 0.0
    return _num(p, "R0", r0), _num(p, "Q0", f.M)


def _state(p):
    f = _flow(p)
    R0, Q0 = L._init(f, _init(p, f))
    if p.get("N") is not None and p.get("alpha") is None:
        raise Refusal("precondition:state-point", "the finite-N simulation runs at alpha: give alpha")
    if p.get("N") is None and "seeds" in p:
        raise Refusal("precondition:known-parameter", "seeds is a finite-N parameter: need N")
    notes = (f"optimal only in OPTIMALITY_CLASS: {L.OPTIMALITY_CLASS}",) if f.rule == "optimal" or \
        f.eta == "greedy" else ()
    return Out({"rule": f.rule, "constraint": f.constraint, "eta": f.eta, "M": f.M, "R0": R0, "Q0": Q0},
               f"rule {f.rule}, constraint {f.constraint}, eta {f.eta}, M = {f.M}, (R0, Q0) = ({R0}, "
               f"{Q0}) (KC95 Eqs. (1)-(5))", notes=notes)


def _asym_when(p):
    return (p.get("alpha") is None and p.get("constraint", "sphere") == "sphere" and p.get("eta", 1) != "greedy"
            and _num(p, "eta", 1) == 1)


def _asymptote(p):
    f = _flow(p)
    a = L.asymptote(f, _init(p, f))
    vals = {"rho_inf": a.rho, "eg_inf": a.eg, "alpha_big": a.alpha}
    if a.tail is not None:
        vals["tail_constant"] = a.tail
    return Out(vals, f"ODE at alpha = {a.alpha:g} (converged to 1e-9 against 3 alpha / 4): rho = "
               f"{a.rho:.9g}, e_g = {a.eg:.9g}")


def _curve(p):
    f, a = _flow(p), _num(p, "alpha")
    init = _init(p, f)
    tr = L.integrate_linear(f, a, init, alphas=(a,))
    R, Q, rho, eg = (float(x[0]) for x in (tr.R, tr.Q, tr.rho, tr.eg))
    vals = {"R": R, "Q": Q, "rho": rho, "eg": eg, "eps_sign": math.acos(max(-1.0, min(1.0, rho))) / math.pi,
            "alpha": a}
    if f.rule == "optimal":
        cR, cQ, _ = L.optimal_closed_form(a, init, f.M, f.constraint)
        vals.update(gamma_temperature=float(L.hypothesis_temperature(rho)),
                    closed_form_gap=max(abs(cR - R), abs(cQ - Q)))
    elif f.eta == "greedy":
        vals["eta_greedy"] = L.greedy_eta(R, Q, f.M)
    elif f.constraint == "none":
        cR, cQ, _ = L.standard_closed_form(a, f.eta, init, f.M)
        vals["closed_form_gap"] = max(abs(cR - R), abs(cQ - Q))
    elif f.eta == 1:
        vals["closed_form_gap"] = abs(L.sphere_standard_closed_form(a, init, f.M) - rho)
    return Out(vals, f"ODE (D = 2 e_g, Q) by Dormand-Prince 5(4), rtol 1e-10, to alpha = {a:g}: rho = "
               f"{rho:.9g}, e_g = {eg:.9g}")


def _offline(p):
    f, a = _flow(p), _num(p, "alpha")
    Q0 = _init(p, f)[1]
    vals, notes = {}, []
    for alg in OFF.ALGORITHMS:
        try:
            o = OFF.offline(a, alg, f.constraint, f.M, Q0 if f.constraint == "sphere" else None)
        except Refusal as r:
            notes.append(f"offline {alg}: {r.reason} ({r.detail})")
            continue
        vals.update({f"rho_offline_{alg}": o.rho, f"eg_offline_{alg}": o.eg})
    try:
        rr = OFF.ridge_route(a, f.M)
        vals.update(rho_offline_optimal_ridge=rr.rho, eg_pinv_ridge=rr.eg, ridge_gamma=rr.gamma)
    except Refusal as r:
        if r.reason != "precondition:gamma-not-one":
            raise
        notes.append("ridge route skipped at gamma = 1/alpha = 1: ridge refuses gamma = 1 because HMRT Eq. (8) "
                     "states the ridgeless limit for gamma != 1 only (it diverges there for sigma2 > 0); at the "
                     "sigma2 = 0 used here the bias r^2 (1 - 1/gamma) tends to 0, but no value is claimed")
    return Out(vals, f"zero-temperature offline solutions at alpha = {a:g} (zero-energy-set geometry; "
               f"ridge at gamma = 1/alpha, sigma2 = 0 for the pseudoinverse)", notes=tuple(notes))


def _cross_when(p):
    return (p.get("alpha") is None and p.get("rule", "optimal") == "optimal" and p.get("constraint", "sphere") == "sphere"
            and _num(p, "Q0", _num(p, "M", 1)) == _num(p, "M", 1))


def _crossing(p):
    x = OFF.crossing_alpha(_num(p, "M", 1))
    return Out({"alpha_cross": x}, f"rho_online_opt(alpha) = rho_offline_std(alpha) at alpha = {x:.6g}")


def _finite(p):
    n = p["N"]
    if isinstance(n, bool) or not isinstance(n, int):
        raise Refusal("precondition:size", f"N={n!r}: need an int >= 10")
    f, a = _flow(p), _num(p, "alpha")
    seeds = _checks.seeds(p.get("seeds", _SEEDS))
    sim = SIM.simulate_linear(f.rule, n, a, seeds, eta=f.eta, constraint=f.constraint, M=f.M,
                              init=_init(p, f))
    b = sim.band
    return Out({"eg_sim": b.mean, "eg_sim_halfwidth99": b.half_width, "rho_sim": sim.rho_band.mean,
                "N": n, "seeds": list(seeds)},
               f"e_g over seeds {list(seeds)} at N = {n}, alpha = {a:g}: {b.mean:.6g} +- {b.half_width:.3g} "
               f"(99% Student-t, df = {b.df})", errors={"eg_sim": b.stderr, "rho_sim": sim.rho_band.stderr},
               statistical=("eg_sim", "rho_sim"),
               notes=("statistical: finite-N online learning over named seeds (online/simulate_linear.py, no "
                      "code shared with the ODEs); it differs from the N -> oo ODE by O(1/N)",))


def online_linear():
    steps = [fstep("online-linear-state-point", f"{KC95}, Eqs. (1)-(5), (9), (22)", _state),
             fstep("online-linear-asymptote", f"{KC95}, Eqs. (24), (25), (31)", _asymptote, when=_asym_when,
                   skip="alpha given, or not the sphere at eta = 1 (asymptotes are reported without alpha)"),
             fstep("online-linear-crossing", f"{KC95}, p. 11 and Fig. 3", _crossing, when=_cross_when,
                   skip="alpha given, or not rule = optimal on the sphere with Q0 = M"),
             fstep("online-linear-learning-curve", f"{KC95}, Eqs. (6)-(8), (17)-(19), (23), (26)", _curve,
                   when=has("alpha"), skip="no alpha given"),
             fstep("online-linear-offline", f"{KC95}, Eqs. (27), (39), (47)-(49), (53)-(54), Sec. 4.4; {SST}; "
                   f"{HMRT}", _offline, when=has("alpha"), skip="no alpha given"),
             fstep("online-linear-finite-n", "online learning over named seeds (online/simulate_linear.py)",
                   _finite, when=has("N"), skip="no N given")]
    R = lambda key, get, **pr: RefCheck("online_linear", key, get, params=pr, route="online")
    refs = [R("online_optimal_rho_a1", lambda v: v["rho"], alpha="1"),
            R("online_optimal_rho_a3", lambda v: v["rho"], alpha="3"),
            R("online_standard_rho_a1", lambda v: v["rho"], rule="standard", alpha="1"),
            R("standard_rho_max_m4", lambda v: v["rho_inf"], rule="standard", Q0="0.25"),
            R("standard_eg_inf_q4", lambda v: v["eg_inf"], rule="standard", Q0="4"),
            R("optimal_eg_inf_q4", lambda v: v["eg_inf"], Q0="4", R0="0.01"),
            R("optimal_tail_constant", lambda v: v["tail_constant"]),
            R("online_offline_crossing", lambda v: v["alpha_cross"]),
            R("offline_standard_rho_a0p5", lambda v: v["rho_offline_standard"], alpha="0.5"),
            R("offline_optimal_rho_a0p5", lambda v: v["rho_offline_optimal"], alpha="0.5"),
            R("offline_optimal_rho_a0p5", lambda v: v["rho_offline_optimal_ridge"], alpha="0.5"),
            R("offline_optimal_eg_a0p25", lambda v: v["eg_offline_optimal"], alpha="0.25"),
            R("offline_standard_eg_a0p25", lambda v: v["eg_offline_standard"], alpha="0.25")]
    ode = Stability("not-computed", "online-ode", KC95, _ODE_NOTE)
    asym = Stability("not-computed", "online-ode", KC95, "measured on the integrated ODE (alpha = 60, "
                     "converged to 1e-9; the tail at alpha = 20); " + _ODE_NOTE)
    cross = Stability("not-computed", "online-ode", f"{KC95}, p. 11", "located on the integrated ODE of the "
                      "optimal rule (rho0 = 1e-4) against the offline zero-energy geometry (online/linear_offline.py "
                      "crossing_alpha, grid step 1e-3, linear interpolation); " + _ODE_NOTE)
    off = Stability("not-computed", "offline-t0", f"{KC95}, Sec. 4; {SST}",
                    "zero-temperature offline solution from the zero-energy-set geometry derived in "
                    "online/linear_offline.py (|P B|^2 = alpha M self-averaging assumed); KC's replica result "
                    "is RS (valid for noiseless examples, their ref. [7], not fetched): NOT final")
    rid = Stability("not-computed", "offline-t0", HMRT, "ridge's ridgeless excess risk mapped onto the "
                    "pseudoinverse (gamma = 1/alpha, sigma2 = 0, signal M; R = Q of a projection): a second "
                    "route through E7, NOT final here")
    sim = Stability("not-computed", "finite-N", "online/simulate_linear.py",
                    "statistical: finite-N mean +- stderr over the named seeds (99% Student-t half-width in "
                    "eg_sim_halfwidth99), never certified: NOT final")
    fin = {**{k: ode for k in ("R", "Q", "rho", "eg", "eps_sign", "gamma_temperature", "eta_greedy")},
           **{k: asym for k in ("rho_inf", "eg_inf", "tail_constant")}, "alpha_cross": cross,
           **{f"{q}_offline_{a}": off for q in ("rho", "eg") for a in OFF.ALGORITHMS},
           "rho_offline_optimal_ridge": rid, "eg_pinv_ridge": rid, "eg_sim": sim, "rho_sim": sim}
    meta = ("rule", "constraint", "eta", "M", "R0", "Q0", "alpha", "alpha_big", "closed_form_gap",
            "ridge_gamma", "eg_sim_halfwidth99", "N", "seeds")
    steps = [*steps, nothing_computed(fin, "give alpha (the learning curve), or rule = optimal without alpha (the asymptote and the crossing)")]
    spec = ModelSpec("online_linear", "spherical", "gaussian", constraint="online", prior="teacher")
    return entry("online_linear", spec, f"{KC95}; {SST}; {HMRT}", {"online": steps}, refs,
                 {"online": CAVEAT}, "online learning of a linear teacher by a linear perceptron with "
                 "quadratic error: the standard and the variationally optimal rule, constrained (sphere) "
                 "or not, compared with offline zero-temperature learning and with ridge",
                 default_route="online", finality=fin, meta=meta,
                 parameters={"online": ("rule", "constraint", "eta", "M", "R0", "Q0", "alpha", "N",
                                        "seeds")})


__all__ = ["online_linear"]
