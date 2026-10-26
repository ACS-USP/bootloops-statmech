"""Catalog entry `online_tcm` (O4), route `online`: online learning in the tree committee machine.

Model (online/tcm.py, tcm_flow.py; CC95 Sec. 2): K odd non-overlapping branches of N/K inputs (N(0, 1)),
|B_k| = 1, majority label sigma_B, J_k <- J_k + (1/N) F_k xi_k, alpha = examples / N (= CKC96's alpha_CM),
eps = P(student label != teacher label), branch-symmetric state.  The N -> oo ODEs are physics-level
(self-averaging assumed, as CC95 do), so no value is final.
Run parameters:
  rule              optimal (default; CC95's nonlocal expected-stability rule, K = 3) | local_optimal
                    (the branch sees only its own field; CKC96's equivalence) | hebb (F_k = eta sigma_B)
  K                 odd number of hidden units (default 3; optimal needs 3)
  eta               Hebb's learning rate (default 1; the optimal rules need 1)
  R0, Q0            branch state (default R0 = 0.01 for the optimal rules, else 0; Q0 = 1)
  (always)          the asymptote: optimal rules eps_constant = lim alpha eps from the rate at theta -> 0;
                    local_optimal and hebb also eps_exponent (ODE log-slope at alpha = 1e4); local_optimal
                    eps_constant_perceptron = P0 C(lambda_K) (O1's perceptron route); hebb eps_constant from
                    the closed form at alpha = 1e10
  (local rules)     the CKC96 mapping: noise_equivalent lambda_K, time_factor K; at alpha also the
                    perceptron ODE at the mapped noise (rho_perceptron_mapped, equivalence_gap)
  alpha             the learning curve: eps, eps_branch, R, Q, rho
  N (seeds)         finite-N simulation at alpha over named seeds (default 0-19): eps_sim
"""
import math

from .catalog import Out, RefCheck, Stability, entry, fstep, has
from .catalog_online import _num
from .core import Refusal
from .csp import _checks
from .online import odes as O
from .online import optimal as OPT
from .online import simulate_tcm as SIM
from .online import tcm as T
from .online import tcm_flow as TF
from .spec import ModelSpec

CC95 = ("Copelli & Caticha, J. Phys. A 28, 1615 (1995) (preprint IFUSP/P-1126 read, publica-sbi.if.usp.br, "
        "repositorio.usp.br)")
CKC96 = "Copelli, Kinouchi & Caticha, Phys. Rev. E 53, 6341 (1996) (abstract only; full text not fetched)"
CAVEAT = ("online route: the N -> oo order-parameter ODEs of online learning in the tree committee machine (CC95 "
          "Eqs. (7)-(8)), with the branch field averages over the other hidden units' internal representations "
          "computed here; self-averaging ASSUMED as in the paper, so the values are physics-level, not final; each "
          "optimal rule is optimal only in its stated class (online/tcm.py OPTIMALITY_CLASS); CKC96's equivalence is "
          "checked between the TCM route and O1's noisy perceptron route; finite-N values are statistical")
_ODE_NOTE = ("N -> oo ODE limit: self-averaging assumed (CC95 p. 6: the mean values 'as approximations (which will "
             "turn out to be very good)'); no theorem fetched: physics-level, NOT final")
ALPHA_BIG = 1e4
_SEEDS = tuple(range(20))
_OPT = ("local_optimal", "optimal")


def _K(p):
    k = p.get("K", 3)
    if isinstance(k, str):
        try:
            k = int(k)
        except ValueError:
            raise Refusal("precondition:K", f"K={k!r}: need an odd integer >= 1") from None
    T._check_k(k)
    return k


def _rule(p):
    r = p.get("rule", "optimal")
    if not isinstance(r, str):
        raise Refusal("unsupported:rule", "the catalog takes named rules (hebb, local_optimal, optimal)")
    return r


def _flow(p):
    return TF.tcm_flow(_rule(p), _K(p), eta=_num(p, "eta", 1))


def _init(p):
    return _num(p, "R0", 0.01 if _rule(p) in _OPT else 0.0), _num(p, "Q0", 1)


def _state(p):
    f = _flow(p)
    R0, Q0 = _init(p)
    TF._init(f, (R0, Q0))
    if p.get("N") is not None and p.get("alpha") is None:
        raise Refusal("precondition:state-point", "the finite-N simulation runs at alpha: give alpha")
    if p.get("N") is None and "seeds" in p:
        raise Refusal("precondition:known-parameter", "seeds is a finite-N parameter: need N")
    notes = (f"optimal only in its class: {T.OPTIMALITY_CLASS[f.rule]}",) if f.rule in _OPT else ()
    return Out({"rule": f.rule, "K": f.K, "eta": f.eta, "R0": R0, "Q0": Q0, "tie_probability": T.tie_probability(f.K)},
               f"rule {f.rule}, K = {f.K}, eta {f.eta}, branch state ({R0}, {Q0}) (CC95 Eqs. (1)-(6))", notes=notes)


def _asymptote(p):
    f = _flow(p)
    vals, notes = {"eps_exponent_derived": TF.EXPONENT[f.rule]}, ()
    if f.rule in _OPT:
        vals["eps_constant"] = TF.rate_limit_constant(f)
        how = "rate limit at theta = 1e-3, 5e-4"
    if f.rule != "optimal":
        slope, c = TF.asymptote_tcm(f, _init(p), ALPHA_BIG)
        vals.update(eps_exponent=slope, alpha_big=ALPHA_BIG)
        how = f"log-slope {slope:.5f} at alpha = {ALPHA_BIG:g}"
    else:
        notes = ("the nonlocal ODE (a 3D field average per step) is integrated to alpha = 1e4 in the slow tests only; "
                 "the exponent 1 is derived (theta ~ 1/(k alpha)), not measured here",)
    if f.rule == "local_optimal":
        vals["eps_constant_perceptron"] = T.tie_probability(f.K) * OPT.asymptotic_constant(T.effective_noise(f.K))
    if f.rule == "hebb":
        vals["eps_constant"] = math.sqrt(1e10) * TF.hebb_closed_form(1e10, f.K, f.eta, _init(p))[2]
    return Out(vals, f"{how}; eps_constant = {vals['eps_constant']:.6g}", notes=notes)


def _equivalence(p):
    f = _flow(p)
    lam = T.effective_noise(f.K)
    vals = {"noise_equivalent": lam, "time_factor": T.perceptron_alpha(1.0, f.K)}
    if p.get("alpha") is not None:
        a = _num(p, "alpha")
        tr = TF.integrate_tcm(f, a, _init(p), alphas=(a,))
        ap = T.perceptron_alpha(a, f.K)
        if f.rule == "local_optimal":
            pt = O.integrate(O.perceptron_flow("optimal", noise=lam), ap, _init(p), alphas=(ap,))
            rp = float(pt.R[0] / math.sqrt(pt.Q[0]))
        else:                       # O1's Hebb closed form (its ODE is stiff from a tiny Q0, CC95's J = 0)
            R, Q, _ = O.hebb_closed_form(ap, f.eta / f.K, lam, _init(p))
            rp = R / math.sqrt(Q)
        vals.update(rho_perceptron_mapped=rp, equivalence_gap=abs(rp - float(tr.rho[0])))
    return Out(vals, f"local learning in branch k = the perceptron with output noise {lam:.6g} on N/K inputs at "
               f"alpha_p = K alpha (CKC96 abstract; mapping derived here)")


def _curve(p):
    f, a = _flow(p), _num(p, "alpha")
    tr = TF.integrate_tcm(f, a, _init(p), alphas=(a,))
    vals = {"eps": float(tr.eps[0]), "eps_branch": float(tr.eps_branch[0]), "R": float(tr.R[0]),
            "Q": float(tr.Q[0]), "rho": float(tr.rho[0]), "alpha": a}
    return Out(vals, f"TCM ODE (theta, ln Q) by Dormand-Prince 5(4), rtol 1e-9, to alpha = {a:g}: eps = {vals['eps']:.6g}")


def _finite(p):
    n = p["N"]
    if isinstance(n, bool) or not isinstance(n, int):
        raise Refusal("precondition:size", f"N={n!r}: need an int multiple of K")
    f, a = _flow(p), _num(p, "alpha")
    seeds = _checks.seeds(p.get("seeds", _SEEDS))
    b = SIM.simulate_tcm(f.rule, f.K, n, a, seeds, eta=f.eta, init=_init(p)).band
    return Out({"eps_sim": b.mean, "eps_sim_halfwidth99": b.half_width, "N": n, "seeds": list(seeds)},
               f"eps over seeds {list(seeds)} at N = {n}, alpha = {a:g}: {b.mean:.6g} +- {b.half_width:.3g} "
               f"(99% Student-t, df = {b.df})", errors={"eps_sim": b.stderr}, statistical=("eps_sim",),
               notes=("statistical: finite-N online TCM learning (online/simulate_tcm.py, no code shared with the "
                      "ODEs); it differs from the ODE by O(1/N)",))


def online_tcm():
    local = lambda p: _rule(p) in ("hebb", "local_optimal")
    steps = [fstep("online-tcm-state-point", f"{CC95}, Eqs. (1)-(6)", _state),
             fstep("online-tcm-asymptote", f"{CC95}, Eqs. (30)-(32); {CKC96}", _asymptote),
             fstep("online-tcm-equivalence", f"{CKC96}; derived here (online/tcm.py)", _equivalence, when=local,
                   skip="the nonlocal rule reads the other branches: no perceptron image"),
             fstep("online-tcm-learning-curve", f"{CC95}, Eqs. (7)-(8), (21), Fig. 1", _curve, when=has("alpha"),
                   skip="no alpha given"),
             fstep("online-tcm-finite-n", "online TCM learning over named seeds (online/simulate_tcm.py)", _finite,
                   when=has("N"), skip="no N given")]
    R = lambda key, get, **pr: RefCheck("online_tcm", key, get, params=pr, route="online")
    refs = [R("cc95_kappa", lambda v: v["eps_constant"]), R("cc95_kappa_formula", lambda v: v["eps_constant"]),
            R("cc95_pseudo_hebb_coeff", lambda v: v["eps_constant"], rule="hebb"),
            R("ckc96_optimal_exponent", lambda v: v["eps_exponent"], rule="local_optimal"),
            R("ckc96_hebb_exponent", lambda v: v["eps_exponent"], rule="hebb")]
    refs += [R(f"cc95_pseudo_hebb_rho_a{a}", lambda v: v["rho"], rule="hebb", alpha=str(a), R0="0", Q0="1e-12")
             for a in (1, 5, 20)]
    ode = Stability("not-computed", "online-ode", CC95, _ODE_NOTE)
    asym = Stability("not-computed", "online-ode", f"{CC95}, Eq. (32); {CKC96}", "the rate limit (optimal rules) or the "
                     f"ODE log-slope at alpha = {ALPHA_BIG:g}; Hebb from the derived closed form; " + _ODE_NOTE)
    perc = Stability("not-computed", "online-ode", "online/optimal.py (O1)", "O1's derived C = 2/I(lambda) of the noisy "
                     "perceptron at the mapped noise, times P0 (derived here); " + _ODE_NOTE)
    mapping = Stability("not-computed", "online-ode", CKC96, "the tie probability of the other hidden units (exact "
                        "combinatorics) and the derived equivalence; " + _ODE_NOTE)
    sim = Stability("not-computed", "finite-N", "online/simulate_tcm.py", "statistical: finite-N mean +- stderr over the "
                    "named seeds (99% Student-t half-width in eps_sim_halfwidth99), never certified: NOT final")
    fin = {**{k: ode for k in ("eps", "eps_branch", "R", "Q", "rho", "rho_perceptron_mapped")},
           **{k: asym for k in ("eps_exponent", "eps_constant")}, "eps_constant_perceptron": perc,
           **{k: mapping for k in ("noise_equivalent", "tie_probability")}, "eps_sim": sim}
    meta = ("rule", "K", "eta", "R0", "Q0", "alpha", "alpha_big", "eps_exponent_derived", "time_factor",
            "equivalence_gap", "eps_sim_halfwidth99", "N", "seeds")
    spec = ModelSpec("online_tcm", "spherical", "gaussian", constraint="online", prior="teacher")
    return entry("online_tcm", spec, f"{CC95}; {CKC96}", {"online": steps}, refs, {"online": CAVEAT},
                 "online learning in the tree committee machine (K odd non-overlapping branches, majority vote): "
                 "CC95's optimal (expected stability) rule and pseudo-Hebb, the local optimal rule and CKC96's "
                 "equivalence with a noisy perceptron, asymptotic constants against K",
                 default_route="online", finality=fin, meta=meta,
                 parameters={"online": ("rule", "K", "eta", "R0", "Q0", "alpha", "N", "seeds")})


__all__ = ["online_tcm"]
