"""`rule=optimal` of the catalog entry `online_scm` (O6): VC97's variationally optimal online algorithm.

Model and convention: online/optimal_multilayer.py (Vicente & Caticha 1997; Saad & Solla's soft committee
machine with the learning rate folded into the modulation F).  Run parameters with rule=optimal:
  K, M        student and teacher units: (2, 1) (VC97 Fig. 1) or (2, 2) (Figs. 2-3); default K = 2, M = K
  eta         the learning rate of the gradient-descent comparison (default 1.5, VC97's backpropagation)
  init        vc97 (the only one): Q22 = 5e-7, Q12 = 0, R = 0 and Q11 fitted to the Hessian inset at alpha = 0,
              or Q11 given
  (always)    the initial Hessian eigenvalues; K >= 2: gradient descent's permutation-symmetric plateau at eta
              (O5); K = M = 2: the optimal plateau (identical students, Sigma singular)
  alpha       the optimal learning curve to alpha (M = 2 takes about a minute to alpha = 16), its Hessian
              eigenvalues, the top eigenvalue's dip on alpha in [0.8, 1.4] (the det H = 0 crossing), and
              gradient descent's error from the same overlaps
  N (seeds)   finite-N simulation of the optimal algorithm at alpha over named seeds (default 0-11): eps_sim
O5's gradient-descent parameters (sym, critical, X, Nx, init_seed, B, a graded teacher) are refused here.
"""
import numpy as np

from .catalog import Out
from .catalog_online import _num
from .core import Refusal
from .csp import _checks
from .online import optimal_ml_flow as OF
from .online import optimal_multilayer as O
from .online import scm as S
from .online import scm_flow as F
from .online import simulate_optimal_ml as SIM

VC = ("Vicente & Caticha, J. Phys. A 30, L599 (1997) (arXiv cond-mat/9706015v1 read: the TeX source and the "
      "PostScript figures; the IOP version not fetched)")
_CLASS = f"'optimal' only in its class: {O.OPTIMALITY_CLASS}"
_GD_ONLY = ("sym", "critical", "X", "Nx", "init_seed", "B")
_DIP = tuple(np.round(np.arange(0.8, 1.4001, 0.02), 10))


def is_opt(p):
    v = p.get("rule", "gd")
    if v not in ("gd", "optimal"):
        raise Refusal("unsupported:rule", f"rule {v!r}: gd (online gradient descent, O5) | optimal (VC97, O6)")
    return v == "optimal"


def _int(p, key, default):
    v = p.get(key, default)
    try:
        k = int(v) if not isinstance(v, bool) and float(v) == int(float(v)) else None
    except (TypeError, ValueError):
        k = None
    if k is None:
        raise Refusal(f"precondition:{key}", f"{key}={v!r}: need an integer")
    return k


def _setup(p):
    K = _int(p, "K", 2)
    flow = O.optimal_flow(K, _int(p, "M", K))
    for k in _GD_ONLY:
        if p.get(k) is not None:
            raise Refusal("precondition:known-parameter", f"{k} is a gradient-descent parameter (rule=gd)")
    if p.get("teacher", "isotropic") != "isotropic":
        raise Refusal("unsupported:teacher", "VC97 take M_nm = delta_nm: rule=optimal needs the isotropic teacher")
    if p.get("init", "vc97") != "vc97":
        raise Refusal("unsupported:init", f"init {p.get('init')!r}: rule=optimal starts from vc97")
    Q0, R0, q11 = OF.vc97_initial(flow.K, flow.M, _num(p, "Q11"))
    return flow, Q0, R0, q11, _num(p, "eta", 1.5)


def state(p):
    flow, Q0, R0, q11, eta = _setup(p)
    if p.get("N") is not None and p.get("alpha") is None:
        raise Refusal("precondition:state-point", "the finite-N simulation runs at alpha: give alpha")
    if p.get("N") is None and "seeds" in p:
        raise Refusal("precondition:known-parameter", "seeds is a finite-N parameter: need N")
    lo, hi = np.linalg.eigvalsh(O.hessian(Q0, R0))[[0, -1]]
    vals = {"rule": "optimal", "K": flow.K, "M": flow.M, "eta": eta, "init": "vc97", "Q11_init": q11,
            "eps0": O.eps_vc(Q0, R0), "hess_lo0": float(lo), "hess_hi0": float(hi), "decay0": flow.decay(Q0, R0)}
    notes = (_CLASS,) + ((f"H has a negative eigenvalue ({lo:.4g}) at the start: F* is a saddle there and decay0 is "
                          "the stationary value, not a maximal decay",) if lo < 0 else ())
    return Out(vals, f"VC97 optimal SCM K = {flow.K}, M = {flow.M}, from Q11 = {q11:.6g} (Hessian eigenvalues "
               f"{lo:.4g}, {hi:.4g})", notes=notes)


def plateaus(p):
    flow, *_, eta = _setup(p)
    vals, notes = {}, []
    if flow.K >= 2:
        try:
            fp = F.symmetric_fixed_point(S.scm_flow(flow.K, flow.M, eta), "permutation")
            vals["eps_plateau_gd"] = fp.eps
        except Refusal as r:
            notes.append(f"no gradient-descent permutation plateau at eta = {eta:g} ({r.reason})")
    if (flow.K, flow.M) == (2, 2):
        pl = O.plateau_k2m2()
        vals.update(eps_plateau_opt=pl.eps, R_plateau_opt=pl.r, Q_plateau_opt=pl.q, hess_lo_plateau=float(pl.eigs[0]),
                    hess_hi_plateau=float(pl.eigs[1]))
    return Out(vals, "plateaus: " + ", ".join(f"{k} {v:.6g}" for k, v in vals.items()), notes=tuple(notes))


def curve(p):
    flow, Q0, R0, _, eta = _setup(p)
    a = _num(p, "alpha")
    if not a > 0:
        raise Refusal("precondition:alpha", f"alpha={a!r}: need alpha > 0")
    dip = tuple(x for x in _DIP if x < a) if a > 1.4 else ()
    track = tuple(float(round(x, 10)) for x in np.arange(0.5, a, 0.5) if round(x, 10) not in dip) if flow.M == 2 else ()
    al = tuple(sorted(dip + track + (a,)))
    tr = OF.integrate_optimal(flow, a, (Q0, R0), alphas=al,
                              **(dict(rtol=1e-4, atol=1e-6) if flow.M == 2 else dict(rtol=1e-6)))
    gd = F.integrate_scm(S.scm_flow(flow.K, flow.M, eta), a, (Q0, R0), alphas=(a,), rtol=1e-7)
    Q, R = tr.Q[-1], tr.R[-1]
    if flow.M == 2 and tr.sig_min.min() < OF.SIGMA_RELIABLE:
        why = (f"min eig Sigma = {tr.sig_min.min():.3g} < {OF.SIGMA_RELIABLE} on the way to alpha = {a:g} (checked every "
               "0.5 in alpha): the M = 2 quadrature of W is not verified there (optimal_ml_flow docstring); the plateau "
               "itself is eps_plateau_opt")
        return Out({"alpha": a, "curve_status": "undetermined"}, why, notes=(why,))
    vals = {"alpha": a, "curve_status": "ok", "eps": float(tr.eps[-1]), "hess_lo": float(tr.eig_lo[-1]),
            "hess_hi": float(tr.eig_hi[-1]), "Q11": float(Q[0, 0]), "R11": float(R[0, 0]), "eps_gd": float(gd.eps[-1])}
    if flow.K >= 2:
        vals.update(Q12=float(Q[0, 1]), Q22=float(Q[1, 1]))
    if dip:
        k = [i for i, x in enumerate(tr.alpha) if x in dip]
        i = k[int(np.argmin(tr.eig_hi[k]))]
        vals.update(hess_hi_dip=float(tr.eig_hi[i]), alpha_hess_hi_dip=float(tr.alpha[i]))
    return Out(vals, f"optimal ODE to alpha = {a:g}: eps = {vals['eps']:.6g} (gradient descent at eta = {eta:g}: "
               f"{vals['eps_gd']:.6g})", notes=(_CLASS,))


def finite(p):
    n = p["N"]
    if isinstance(n, bool) or not isinstance(n, int):
        raise Refusal("precondition:size", f"N={n!r}: need an int >= 20")
    flow, Q0, R0, *_ = _setup(p)
    a = _num(p, "alpha")
    seeds = _checks.seeds(p.get("seeds", tuple(range(12))))
    b = SIM.simulate_optimal(flow.K, flow.M, n, (a,), seeds, (Q0, R0)).bands[0]
    return Out({"eps_sim": b.mean, "eps_sim_halfwidth99": b.half_width, "N": n, "seeds": list(seeds)},
               f"optimal algorithm over seeds {list(seeds)} at N = {n}, alpha = {a:g}: {b.mean:.6g} +- "
               f"{b.half_width:.3g} (99% Student-t, df = {b.df})", errors={"eps_sim": b.stderr}, statistical=("eps_sim",),
               notes=("statistical: finite-N runs of VC97's modulation (online/simulate_optimal_ml.py, no code shared "
                      "with the ODE); near the det H = 0 crossing F* diverges and single runs can be kicked far",))


VALUES = ("Q11_init", "eps0", "hess_lo0", "hess_hi0", "decay0", "eps_plateau_gd", "eps_plateau_opt", "R_plateau_opt",
          "Q_plateau_opt", "hess_lo_plateau", "hess_hi_plateau", "hess_lo", "hess_hi", "Q11", "Q12", "Q22", "R11",
          "eps_gd", "hess_hi_dip", "alpha_hess_hi_dip")
META = ("rule", "curve_status")

__all__ = ["META", "VALUES", "VC", "curve", "finite", "is_opt", "plateaus", "state"]
