"""Catalog entry `online_scm` (O5), route `online`: online gradient descent in the soft committee machine.

Model (online/scm.py; Saad & Solla 1995): K student and M teacher erf hidden units, linear output, eps =
(sigma - tau)^2 / 2, J_i <- J_i + (eta/N) delta_i xi, alpha = examples / N, order parameters Q, R, T.  The
N -> oo ODEs are physics-level (self-averaging assumed, as SS do), so no value is final.
Run parameters:
  K, M        student and teacher hidden units (default 2, M = K)
  eta         the fixed learning rate (default 1.5, BRW's)
  teacher     isotropic (T = delta; default) | graded (T_nn = n)
  (always)    K >= 2, isotropic: the symmetric fixed points (identical students Q = C, BRW Eq. (20); the
              permutation-symmetric Q > C, BRW Eq. (21) / SS Fig. 1's plateau), their escape times, the
              eta^2-truncated (small-eta) point and its escape time in units of 1/eta; K = M, isotropic: the
              asymptotic eigenvalues lambda1, lambda2 at the optimum and the rate max(lambda1, 2 lambda2)
  sym         all (default) | none | a comma list of identical, permutation, small_eta: the symmetric points
              to compute
  critical    eta_max (only that root) or any other value: the critical learning rates: K = M: eta_max,
              eta_crossing (lambda1 = 2 lambda2), eta_fastest; K = 2, M = 1: BS's eta_c and eta_s
  init        brw16 (BRW Eq. (16), Fig. 5 choice; K = M = 2; default there) with X (1e-6), Nx (1000) |
              ss (SS's random overlaps; default otherwise) with init_seed (1)
  alpha       the learning curve to alpha and the plateau report (online/plateau.py), with B (0.05)
  N (seeds)   finite-N simulation at alpha over named seeds (default 0-11): eps_sim
  rule        gd (default: the above) | optimal (VC97's variationally optimal algorithm, O6:
              catalog_online_scm_opt.py, which lists its parameters)
"""
import math

import numpy as np

from .catalog import Out, RefCheck, Stability, entry, fstep, has
from .catalog_online import _num, nothing_computed
from . import catalog_online_scm_opt as OPT
from .core import Refusal
from .csp import _checks
from .online import plateau as P
from .online import scm as S
from .online import scm_flow as F
from .online import simulate_scm as SIM
from .spec import ModelSpec

SS = ("Saad & Solla, PRL 74, 4337 (1995); PRE 52, 4225 (1995) (not fetched; read through Saad & Solla, NIPS 8, 302 "
      "(1996) and MANNA 1995 proceedings (1997), Aston eprints 640, 657)")
BS = "Biehl & Schwarze, J. Phys. A 28, 643 (1995) (preprint LU TP 94-10 read)"
BRW = "Biehl, Riegler & Woehler, J. Phys. A 29, 4769 (1996) (preprint WUE-ITP-96-003 read)"
CAVEAT = ("online route: the N -> oo order-parameter ODEs of online gradient descent in the soft committee machine "
          "(SS's closed-form erf averages I3, I4, checked against Gauss-Hermite); self-averaging ASSUMED as in the "
          "papers, so the values are physics-level, not final; 'optimal' learning rates only in "
          "scm_flow.OPTIMALITY_CLASS; plateaus by BRW's definition, 'undetermined' when not left inside the window; "
          "finite-N values are statistical")
_ODE_NOTE = ("N -> oo ODE limit: self-averaging assumed (SS NIPS p. 304 call the equations 'exact' in the "
             "thermodynamic limit; no theorem fetched): physics-level, NOT final")
_SEEDS = tuple(range(12))


def _int(p, key, default):
    v = p.get(key, default)
    try:
        k = int(v) if not isinstance(v, bool) and float(v) == int(float(v)) else None
    except (TypeError, ValueError):
        k = None
    if k is None:
        raise Refusal(f"precondition:{key}", f"{key}={v!r}: need an integer")
    return k


def _flow(p, truncated=False):
    K = _int(p, "K", 2)
    return S.scm_flow(K, _int(p, "M", K), _num(p, "eta", 1.5), p.get("teacher", "isotropic"), truncated)


def _init(p, f):
    kind = p.get("init", "brw16" if f.K == f.M == 2 else "ss")
    if kind == "brw16" and f.K == f.M == 2:
        return F.brw_initial(_num(p, "X", 1e-6), _num(p, "Nx", 1000))
    if kind == "ss":
        return F.ss_initial(f.K, f.M, _int(p, "init_seed", 1))
    raise Refusal("unsupported:init", f"init {kind!r}: brw16 (K = M = 2) | ss")


def _state(p):
    f = _flow(p)
    F._state(f, _init(p, f))
    if p.get("N") is not None and p.get("alpha") is None:
        raise Refusal("precondition:state-point", "the finite-N simulation runs at alpha: give alpha")
    if p.get("N") is None and "seeds" in p:
        raise Refusal("precondition:known-parameter", "seeds is a finite-N parameter: need N")
    return Out({"K": f.K, "M": f.M, "eta": f.eta, "teacher": p.get("teacher", "isotropic"),
                "init": p.get("init", "brw16" if f.K == f.M == 2 else "ss")},
               f"SCM K = {f.K}, M = {f.M}, eta = {f.eta} (SS NIPS Eqs. (1)-(4))")


_SYM = ("identical", "permutation", "small_eta")


def _sym_parts(p):
    v = p.get("sym", "all")
    parts = _SYM if v == "all" else () if v == "none" else tuple(x.strip() for x in str(v).split(","))
    if any(x not in _SYM for x in parts):
        raise Refusal("unsupported:sym", f"sym {v!r}: all | none | a comma list of {', '.join(_SYM)}")
    return parts


def _odd(f, fp):
    """A root no better than the zero student, or with norms far beyond the teacher's, is flagged."""
    eps0 = f.eps(np.zeros((f.K, f.K)), np.zeros((f.K, f.M)))
    if fp.eps >= eps0 or fp.Q > 3 * float(f.T.max()):
        return (f"odd {fp.branch} root at eta = {f.eta:g}: eps = {fp.eps:.4g} (the zero student has {eps0:.4g}), "
                f"Q = {fp.Q:.4g}; it is a fixed point of the ODE but not a plateau a learning curve visits",)
    return ()


def _symmetric(p):
    f, vals, notes, parts = _flow(p), {}, [], _sym_parts(p)
    if "identical" in parts:
        a = F.symmetric_fixed_point(f, "identical")
        vals.update(eps_sym_identical=a.eps, R_sym_identical=a.R, Q_sym_identical=a.Q, tau_esc_identical=1 / a.lam_esc,
                    plateau_shift_per_100x=0.5 * math.log(100) / a.lam_esc)
        notes += _odd(f, a)
    if "permutation" in parts:
        try:
            b = F.symmetric_fixed_point(f, "permutation")
            vals.update(eps_sym_permutation=b.eps, R_sym_permutation=b.R, Q11_sym_permutation=b.Q,
                        Q12_sym_permutation=b.C, tau_esc_permutation=1 / b.lam_esc if b.lam_esc > 0 else math.inf)
            notes += _odd(f, b)
        except Refusal as r:
            notes.append(f"no permutation-symmetric point distinct from Q = C ({r.reason})")
    if "small_eta" in parts:
        t = F.symmetric_fixed_point(_flow(p, truncated=True), "identical")
        vals.update(eps_small_eta=t.eps, R_small_eta=t.R, Q_small_eta=t.Q, tau_small_eta=f.eta / t.lam_esc)
    return Out(vals, "symmetric fixed points: " + (", ".join(parts) or "none requested"), notes=tuple(notes))


def _asymptotics(p):
    m = F.optimum_modes(_flow(p))
    return Out({"lambda1": m.lam1, "lambda2": m.lam2, "asymptotic_rate": m.rate},
               f"at the optimum: lambda1 = {m.lam1:.6g}, lambda2 = {m.lam2:.6g}, rate max(lambda1, 2 lambda2) = {m.rate:.6g}")


def _critical(p):
    f = _flow(p)
    note = (f"optimal only in its class: {F.OPTIMALITY_CLASS}",)
    if f.K == f.M:
        r = F.critical_rates(f.K, need=("eta_max",) if p["critical"] == "eta_max" else
                             ("eta_max", "eta_crossing", "eta_fastest"))
        if "eta_crossing" not in r:
            return Out(r, f"eta_max {r['eta_max']:.6g} (critical=eta_max: the crossing and the fastest rate skipped)")
        return Out(r, f"eta_max {r['eta_max']:.6g}, lambda1 = 2 lambda2 at {r['eta_crossing']:.6g}, fastest "
                   f"asymptotic decay at {r['eta_fastest']:.6g}", notes=note)
    if (f.K, f.M) == (2, 1) and F._isotropic(f):
        r = F.bs95_k2_rates()
        return Out({"eta_c_specialised": r["eta_c"], "eta_s_symmetric": r["eta_s"]},
                   f"BS Sec. 4 (beta = 1): eta_c {r['eta_c']:.6g}, eta_s {r['eta_s']:.6g}")
    raise Refusal("precondition:symmetric", "critical rates are computed for K = M or (K, M) = (2, 1), isotropic")


def _curve(p):
    f, a = _flow(p), _num(p, "alpha")
    rep = P.analyse(f, _init(p, f), a, B=_num(p, "B", 0.05))
    eps = rep.eps_end
    vals = {"eps": eps, "alpha": a, "plateau_status": rep.status}
    if rep.status != "none":
        vals.update(plateau_branch=rep.branch, plateau_height=rep.eps_fp, plateau_eps_mid=rep.eps_mid,
                    plateau_entry=rep.alpha_entry, plateau_tau_esc=rep.tau_esc)
    if rep.runaway:
        vals["plateau_runaway"] = True
    if rep.status == "escaped":
        vals.update(plateau_exit=rep.alpha_exit, plateau_length=rep.length, plateau_rtol_spread=rep.exit_rtol_spread)
    return Out(vals, f"SCM ODE to alpha = {a:g}: eps = {eps:.6g}; plateau {rep.status}: {rep.reason}",
               notes=(f"plateau: {rep.status} ({rep.reason})",) if rep.status != "escaped" else ())


def _finite(p):
    n = p["N"]
    if isinstance(n, bool) or not isinstance(n, int):
        raise Refusal("precondition:size", f"N={n!r}: need an int >= 20")
    f, a = _flow(p), _num(p, "alpha")
    seeds = _checks.seeds(p.get("seeds", _SEEDS))
    teacher = p.get("teacher", "isotropic")
    b = SIM.simulate_scm(f.K, f.M, n, f.eta, (a,), seeds, init=_init(p, f), teacher=teacher).bands[0]
    return Out({"eps_sim": b.mean, "eps_sim_halfwidth99": b.half_width, "N": n, "seeds": list(seeds)},
               f"eps over seeds {list(seeds)} at N = {n}, alpha = {a:g}: {b.mean:.6g} +- {b.half_width:.3g} "
               f"(99% Student-t, df = {b.df})", errors={"eps_sim": b.stderr}, statistical=("eps_sim",),
               notes=("statistical: finite-N online gradient descent (online/simulate_scm.py, no code shared with "
                      "the ODEs), from the same initial overlaps; it differs from the ODE by O(1/sqrt N) on the "
                      "plateau's escape",))


def _sym_ok(p):
    f = _flow(p)
    return f.K >= 2 and F._isotropic(f)


def _asym_ok(p):
    f = _flow(p)
    return f.K == f.M and F._isotropic(f)


def _gd(when=None):
    return lambda p: not OPT.is_opt(p) and (when is None or when(p))


def _opt(when=None):
    return lambda p: OPT.is_opt(p) and (when is None or when(p))


def online_scm():
    steps = [fstep("online-scm-state-point", f"{SS}; NIPS Eqs. (1)-(4)", _state, when=_gd(), skip="rule=optimal"),
             fstep("online-scm-symmetric-points", f"{SS}, NIPS Eqs. (5)-(8); {BRW}, Eqs. (9)-(21)", _symmetric,
                   when=_gd(_sym_ok), skip="needs K >= 2 and an isotropic teacher"),
             fstep("online-scm-asymptotics", f"{SS}, NIPS p. 307, Fig. 2; {BS}, Eq. (12)", _asymptotics,
                   when=_gd(_asym_ok), skip="needs K = M and an isotropic teacher"),
             fstep("online-scm-critical-rates", f"{SS}, NIPS Eq. (9); {BS}, Secs. 3-4", _critical,
                   when=_gd(has("critical")), skip="no critical given"),
             fstep("online-scm-learning-curve", f"{BRW}, Eqs. (9)-(17); {SS}, NIPS Fig. 1", _curve, when=_gd(has("alpha")),
                   skip="no alpha given"),
             fstep("online-scm-finite-n", "online gradient descent over named seeds (online/simulate_scm.py)", _finite,
                   when=_gd(has("N")), skip="no N given")]
    vc = OPT.VC
    steps += [fstep("online-scm-optimal-state-point", f"{vc}, Eqs. (1)-(4), Figs. 1-2 captions", OPT.state,
                    when=_opt(), skip="rule=gd"),
              fstep("online-scm-optimal-plateaus", f"{vc}, Figs. 2-3; {SS} (gradient descent)", OPT.plateaus,
                    when=_opt(), skip="rule=gd"),
              fstep("online-scm-optimal-learning-curve", f"{vc}, Eq. (2), Figs. 1-3", OPT.curve,
                    when=_opt(has("alpha")), skip="rule=gd or no alpha given"),
              fstep("online-scm-optimal-finite-n", "VC97's modulation over named seeds (online/simulate_optimal_ml.py)",
                    OPT.finite, when=_opt(has("N")), skip="rule=gd or no N given")]
    R = lambda key, get, **pr: RefCheck("online_scm", key, get, params=pr, route="online")
    refs = [R("brw96_fig2_height", lambda v: v["eps_sym_identical"]),
            R("brw96_fig1_eps", lambda v: v["eps_sym_permutation"]), R("brw96_fig1_R", lambda v: v["R_sym_permutation"]),
            R("brw96_fig1_Q11", lambda v: v["Q11_sym_permutation"]),
            R("brw96_fig1_Q12", lambda v: v["Q12_sym_permutation"]),
            R("brw96_fig2_shift_per_100x", lambda v: v["plateau_shift_per_100x"]),
            R("ss_tau_k2", lambda v: v["tau_small_eta"])]
    refs += [R(f"brw96_{k}", lambda v, g=g: v[g], eta="2", sym="identical,permutation") for k, g in (
        ("eq20_R", "R_sym_identical"), ("eq20_Q", "Q_sym_identical"), ("eq20_eps", "eps_sym_identical"),
        ("eq21_R", "R_sym_permutation"), ("eq21_Q11", "Q11_sym_permutation"), ("eq21_Q12", "Q12_sym_permutation"),
        ("eq21_eps", "eps_sym_permutation"))]
    refs += [R(f"ss_fig1d_height_eta{t}", lambda v: v["eps_sym_permutation"], K="3", eta=e, sym="permutation")
             for t, e in (("0p1", "0.1"), ("0p3", "0.3"), ("0p5", "0.5"), ("0p7", "0.7"))]
    refs += [R(f"ss_fig2_lambda{i}_eta{t}", lambda v, i=i: v[f"lambda{i}"], K="3", eta=e, sym=sy)
             for t, e, sy in (("0p4", "0.4", "none"), ("1p0", "1.0", "small_eta"), ("1p5", "1.5", "none")) for i in (1, 2)]
    refs += [R(k, lambda v: v["eps_small_eta"], K="3", eta="1.0", sym="small_eta")
             for k in ("ss_trunc_eps_k3", "ss_trunc_eps_k3_formula")]
    refs += [R("ss_tau_k3", lambda v: v["tau_small_eta"], K="3", eta="1.0", sym="small_eta"),
             R("ss_fig2_eta_max_k3", lambda v: v["eta_max"], K="3", critical="eta_max", sym="none")]
    refs += [R(k, lambda v, g=g: v[g], K="1", critical="1") for k, g in (
        ("bs95_eta_c_single", "eta_max"), ("bs95_eta_c_single_formula", "eta_max"),
        ("bs95_eta_opt_single", "eta_crossing"), ("bs95_eta_opt_single_formula", "eta_crossing"))]
    refs += [R(k, lambda v, g=g: v[g], K="2", M="1", critical="1", sym="none") for k, g in (
        ("bs95_k2_eta_c", "eta_c_specialised"), ("bs95_k2_eta_s", "eta_s_symmetric"))]
    o22, o21 = dict(rule="optimal", K="2", M="2"), dict(rule="optimal", K="2", M="1")
    refs += [R(k, lambda v, g=g: v[g], **o22) for k, g in (
        ("vc97_fig2_eps_plateau", "eps_plateau_opt"), ("vc97_fig2_hess_hi_plateau", "hess_hi_plateau"),
        ("vc97_fig2_hess_lo_plateau", "hess_lo_plateau"), ("vc97_fig3_q12_plateau", "Q_plateau_opt"),
        ("vc97_fig2_bp_plateau", "eps_plateau_gd"))]
    refs += [R("vc97_fig1_bp_plateau", lambda v: v["eps_plateau_gd"], **o21),
             R("vc97_fig1_hess_lo_a0", lambda v: v["hess_lo0"], **o21)]
    refs += [R(k, lambda v, g=g: v[g], alpha="8", **o21) for k, g in (
        ("vc97_fig1_hess_dip", "hess_hi_dip"), ("vc97_fig1_hess_dip_alpha", "alpha_hess_hi_dip"),
        ("vc97_fig1_hess_hi_a8", "hess_hi"), ("vc97_fig1_hess_lo_a8", "hess_lo"))]
    ode = Stability("not-computed", "online-ode", SS, _ODE_NOTE)
    rate = Stability("not-computed", "online-ode", f"{SS}; {BS}", "roots of numerically differentiated eigenvalues "
                     f"of the ODE; 'optimal' only in scm_flow.OPTIMALITY_CLASS; " + _ODE_NOTE)
    sim = Stability("not-computed", "finite-N", "online/simulate_scm.py", "statistical: finite-N mean +- stderr over "
                    "the named seeds (99% Student-t half-width in eps_sim_halfwidth99), never certified: NOT final")
    phys = ("eps_sym_identical", "R_sym_identical", "Q_sym_identical", "tau_esc_identical", "plateau_shift_per_100x",
            "eps_sym_permutation", "R_sym_permutation", "Q11_sym_permutation", "Q12_sym_permutation",
            "tau_esc_permutation", "eps_small_eta", "R_small_eta", "Q_small_eta", "tau_small_eta", "lambda1", "lambda2",
            "asymptotic_rate", "eps", "plateau_height", "plateau_eps_mid", "plateau_entry", "plateau_exit",
            "plateau_length", "plateau_tau_esc")
    fin = {**{k: ode for k in phys}, **{k: rate for k in ("eta_max", "eta_crossing", "eta_fastest", "rate_crossing",
                                                            "rate_fastest", "eta_c_specialised", "eta_s_symmetric")},
           "eps_sim": sim}
    vcode = Stability("not-computed", "online-ode", OPT.VC, "VC97's optimal modulation in the N -> oo ODE limit "
                      "(self-averaging assumed; 'optimal' only in optimal_multilayer.OPTIMALITY_CLASS): physics-level, "
                      "NOT final")
    fin.update({k: vcode for k in OPT.VALUES})
    meta = ("K", "M", "eta", "teacher", "init", "alpha", "plateau_status", "plateau_branch", "plateau_rtol_spread",
            "plateau_runaway",
            "eps_sim_halfwidth99", "N", "seeds") + OPT.META
    hint = "give alpha (the learning curve), or the isotropic teacher (the symmetric points and asymptotics)"
    steps = [*steps, nothing_computed(fin, hint)]
    spec = ModelSpec("online_scm", "spherical", "gaussian", constraint="online", prior="teacher")
    return entry("online_scm", spec, f"{SS}; {BS}; {BRW}", {"online": steps}, refs, {"online": CAVEAT},
                 "online gradient descent in the soft committee machine (Saad-Solla ODEs, erf units): symmetric "
                 "plateaus (height, escape, BRW's plateau length), asymptotic eigenvalues and critical learning "
                 "rates, finite-N simulation", default_route="online", finality=fin, meta=meta,
                 parameters={"online": ("K", "M", "eta", "teacher", "sym", "critical", "init", "X", "Nx", "init_seed", "alpha",
                                        "B", "N", "seeds", "rule", "Q11")})


__all__ = ["online_scm"]
