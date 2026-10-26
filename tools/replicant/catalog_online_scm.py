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
  critical    (any value) the critical learning rates: K = M: eta_max, eta_crossing (lambda1 = 2 lambda2),
              eta_fastest; K = 2, M = 1: BS's eta_c and eta_s
  init        brw16 (BRW Eq. (16), Fig. 5 choice; K = M = 2; default there) with X (1e-6), Nx (1000) |
              ss (SS's random overlaps; default otherwise) with init_seed (1)
  alpha       the learning curve to alpha and the plateau report (online/plateau.py), with B (0.05)
  N (seeds)   finite-N simulation at alpha over named seeds (default 0-11): eps_sim
"""
import math

from .catalog import Out, RefCheck, Stability, entry, fstep, has
from .catalog_online import _num
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


def _symmetric(p):
    f, vals, notes = _flow(p), {}, []
    a = F.symmetric_fixed_point(f, "identical")
    vals.update(eps_sym_identical=a.eps, R_sym_identical=a.R, Q_sym_identical=a.Q, tau_esc_identical=1 / a.lam_esc,
                plateau_shift_per_100x=0.5 * math.log(100) / a.lam_esc)
    try:
        b = F.symmetric_fixed_point(f, "permutation")
        vals.update(eps_sym_permutation=b.eps, R_sym_permutation=b.R, Q11_sym_permutation=b.Q,
                    Q12_sym_permutation=b.C, tau_esc_permutation=1 / b.lam_esc if b.lam_esc > 0 else math.inf)
    except Refusal as r:
        notes.append(f"no permutation-symmetric point distinct from Q = C ({r.reason})")
    t = F.symmetric_fixed_point(_flow(p, truncated=True), "identical")
    vals.update(eps_small_eta=t.eps, R_small_eta=t.R, Q_small_eta=t.Q, tau_small_eta=f.eta / t.lam_esc)
    return Out(vals, f"identical-student point eps = {a.eps:.6g} (tau_esc {1 / a.lam_esc:.6g}); eta^2-truncated "
               f"eps = {t.eps:.6g}", notes=tuple(notes))


def _asymptotics(p):
    m = F.optimum_modes(_flow(p))
    return Out({"lambda1": m.lam1, "lambda2": m.lam2, "asymptotic_rate": m.rate},
               f"at the optimum: lambda1 = {m.lam1:.6g}, lambda2 = {m.lam2:.6g}, rate max(lambda1, 2 lambda2) = {m.rate:.6g}")


def _critical(p):
    f = _flow(p)
    note = (f"optimal only in its class: {F.OPTIMALITY_CLASS}",)
    if f.K == f.M:
        r = F.critical_rates(f.K)
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


def online_scm():
    steps = [fstep("online-scm-state-point", f"{SS}; NIPS Eqs. (1)-(4)", _state),
             fstep("online-scm-symmetric-points", f"{SS}, NIPS Eqs. (5)-(8); {BRW}, Eqs. (9)-(21)", _symmetric,
                   when=_sym_ok, skip="needs K >= 2 and an isotropic teacher"),
             fstep("online-scm-asymptotics", f"{SS}, NIPS p. 307, Fig. 2; {BS}, Eq. (12)", _asymptotics,
                   when=_asym_ok, skip="needs K = M and an isotropic teacher"),
             fstep("online-scm-critical-rates", f"{SS}, NIPS Eq. (9); {BS}, Secs. 3-4", _critical,
                   when=has("critical"), skip="no critical given"),
             fstep("online-scm-learning-curve", f"{BRW}, Eqs. (9)-(17); {SS}, NIPS Fig. 1", _curve, when=has("alpha"),
                   skip="no alpha given"),
             fstep("online-scm-finite-n", "online gradient descent over named seeds (online/simulate_scm.py)", _finite,
                   when=has("N"), skip="no N given")]
    R = lambda key, get, **pr: RefCheck("online_scm", key, get, params=pr, route="online")
    refs = [R("brw96_fig2_height", lambda v: v["eps_sym_identical"]),
            R("brw96_fig1_eps", lambda v: v["eps_sym_permutation"]), R("brw96_fig1_R", lambda v: v["R_sym_permutation"]),
            R("brw96_fig1_Q11", lambda v: v["Q11_sym_permutation"]),
            R("brw96_fig1_Q12", lambda v: v["Q12_sym_permutation"]),
            R("brw96_fig2_shift_per_100x", lambda v: v["plateau_shift_per_100x"]),
            R("ss_tau_k2", lambda v: v["tau_small_eta"])]
    refs += [R(f"brw96_{k}", lambda v, g=g: v[g], eta="2") for k, g in (
        ("eq20_R", "R_sym_identical"), ("eq20_Q", "Q_sym_identical"), ("eq20_eps", "eps_sym_identical"),
        ("eq21_R", "R_sym_permutation"), ("eq21_Q11", "Q11_sym_permutation"), ("eq21_Q12", "Q12_sym_permutation"),
        ("eq21_eps", "eps_sym_permutation"))]
    refs += [R(f"ss_fig1d_height_eta{t}", lambda v: v["eps_sym_permutation"], K="3", eta=e)
             for t, e in (("0p1", "0.1"), ("0p3", "0.3"), ("0p5", "0.5"), ("0p7", "0.7"))]
    refs += [R(f"ss_fig2_lambda{i}_eta{t}", lambda v, i=i: v[f"lambda{i}"], K="3", eta=e)
             for t, e in (("0p4", "0.4"), ("1p0", "1.0"), ("1p5", "1.5")) for i in (1, 2)]
    refs += [R(k, lambda v: v["eps_small_eta"], K="3", eta="1.0") for k in ("ss_trunc_eps_k3", "ss_trunc_eps_k3_formula")]
    refs += [R("ss_tau_k3", lambda v: v["tau_small_eta"], K="3", eta="1.0"),
             R("ss_fig2_eta_max_k3", lambda v: v["eta_max"], K="3", critical="1")]
    refs += [R(k, lambda v, g=g: v[g], K="1", critical="1") for k, g in (
        ("bs95_eta_c_single", "eta_max"), ("bs95_eta_c_single_formula", "eta_max"),
        ("bs95_eta_opt_single", "eta_crossing"), ("bs95_eta_opt_single_formula", "eta_crossing"))]
    refs += [R(k, lambda v, g=g: v[g], K="2", M="1", critical="1") for k, g in (
        ("bs95_k2_eta_c", "eta_c_specialised"), ("bs95_k2_eta_s", "eta_s_symmetric"))]
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
    meta = ("K", "M", "eta", "teacher", "init", "alpha", "plateau_status", "plateau_branch", "plateau_rtol_spread",
            "eps_sim_halfwidth99", "N", "seeds")
    spec = ModelSpec("online_scm", "spherical", "gaussian", constraint="online", prior="teacher")
    return entry("online_scm", spec, f"{SS}; {BS}; {BRW}", {"online": steps}, refs, {"online": CAVEAT},
                 "online gradient descent in the soft committee machine (Saad-Solla ODEs, erf units): symmetric "
                 "plateaus (height, escape, BRW's plateau length), asymptotic eigenvalues and critical learning "
                 "rates, finite-N simulation", default_route="online", finality=fin, meta=meta,
                 parameters={"online": ("K", "M", "eta", "teacher", "critical", "init", "X", "Nx", "init_seed", "alpha",
                                        "B", "N", "seeds")})


__all__ = ["online_scm"]
