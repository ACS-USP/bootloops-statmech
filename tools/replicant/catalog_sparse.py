"""Catalog entry for the sparse-graph Ising spin glass (Viana-Bray / Bethe lattice),
cavity route only: BP / population dynamics (RS) and 1RSB.

Conventions (cavity/sparse.py): E = -sum_<ij> J_ij s_i s_j, J_ij = +-J (J = 1 default);
graph "regular" with excess degree k (connectivity k + 1, default k = 2) or "poisson"
with mean degree c.  beta = 1/T.  Run parameters:
  no beta              transition beta_c (B E tanh^2(beta_c J) = 1); for the regular +-1
                       graph also the exact T = 0 RS solution and the factorized T = 0
                       1RSB (deterministic, exact enumeration);
  no beta, ansatz 1rsb also the full T = 0 1RSB ground-state energy (surveys; statistical);
  beta, ansatz rs      RS population dynamics (statistical);
  beta, ansatz 1rsb, m 1RSB population of populations at Parisi parameter m (statistical).
Sampling settings (pop, sweeps, burn_in, runs, seed, inner, ys, ...) pass through to the
cavity functions; their defaults are those functions' (seed 0).  Statistical values are
labelled as such and carry stderr (over independent runs) in errors.
"""
from .catalog import Out, RefCheck, entry, fstep, lacks
from .cavity import sparse as S
from .cavity._rsb_common import FACTORIZED_CAVEAT, RSB_CAVEAT
from .cavity.rsb_sparse import pop_of_pops_1rsb
from .cavity.rsb_sparse_t0 import factorized_1rsb_t0, rsb_ground_state_energy
from .core import Refusal
from .families.common import CAVITY_CAVEAT
from .spec import ModelSpec

MP01 = "Mezard & Parisi, Eur. Phys. J. B 20, 217 (2001)"
MP03 = "Mezard & Parisi, J. Stat. Phys. 111, 1 (2003)"
VB = "Viana & Bray, J. Phys. C 18, 3037 (1985)"
_PD_KEYS = ("pop", "sweeps", "seed", "burn_in", "runs", "field", "init_field",
            "require_equilibrated")
_POP_KEYS = ("pop", "inner", "sweeps", "seed", "burn_in", "runs", "init_field", "init_spread",
             "require_equilibrated", "bary_samples")
_GGS_KEYS = ("ys", "pop", "sweeps", "seed", "burn_in", "runs", "init", "require_equilibrated")
_PD_DEFAULTS = {"pop": 2000, "sweeps": 10, "seed": 0}
_POP_DEFAULTS = {"pop": 1000, "inner": 100, "sweeps": 20, "seed": 0}
_GGS_DEFAULTS = {"ys": (0.3, 0.5, 0.7, 0.9, 1.1), "pop": 2000, "sweeps": 300, "seed": 0}


def _laws(p):
    graph = p.get("graph", "regular")
    if graph == "regular":
        deg = ("regular", int(p.get("k", 2)))
    elif graph == "poisson":
        deg = ("poisson", p.get("c", 3))
    else:
        raise Refusal("precondition:known-graph", f"graph = {graph!r}: use 'regular' or 'poisson'")
    return deg, ("pm1", float(p.get("J", 1.0)))


def _kw(p, keys, defaults):
    out = {**defaults, **{k: p[k] for k in keys if k in p}}
    if "ys" in out:
        out["ys"] = tuple(out["ys"])
    return out


def _regular_pm1(p):
    return p.get("graph", "regular") == "regular" and float(p.get("J", 1.0)) == 1.0


def _stat(r, keys, ansatz, caveat):
    values = {k: r.value[k] for k in keys}
    return Out({**values, "ansatz": ansatz, "stable": None, "phase": r.phase},
               f"{r.label}: {r.phase}; {r.note}",
               errors={k: r.stderr[k] for k in keys}, statistical=tuple(keys),
               notes=(f"statistical: {', '.join(keys)} are population-dynamics estimates "
                      "(value +- stderr over independent runs), not certified", caveat))


def _transition(p):
    deg, jl = _laws(p)
    bc = S.viana_bray_beta_c(deg, jl)
    return Out({"beta_c": bc, "T_c": 1 / bc}, f"B E tanh^2(beta_c J) = 1 on {deg}")


def _rs_t0(p):
    sol = S.rs_zero_temperature_pm1(int(p.get("k", 2)))
    return Out({"p0_rs_T0": sol["p0"], "e0_rs": sol["e"], "q_rs_T0": sol["q"]},
               "exact T = 0 RS recursion on integer cavity fields (sympy roots)")


def _fact_t0(p):
    sol = factorized_1rsb_t0(int(p.get("k", 2)))
    return Out({"e0_1rsb_factorized": sol["e0"], "mu_star_factorized": sol["y_star"]},
               "factorized 1RSB at T = 0, maximised over y = beta m (deterministic)",
               notes=(FACTORIZED_CAVEAT,))


def _pd(p):
    deg, jl = _laws(p)
    r = S.population_dynamics(deg, jl, p["beta"], **_kw(p, _PD_KEYS, _PD_DEFAULTS))
    return _stat(r, ("q", "e", "f"), "rs", CAVITY_CAVEAT)


def _pop(p):
    if p.get("m") is None:
        raise Refusal("precondition:parisi-m", "1RSB at finite T needs the Parisi parameter m")
    deg, jl = _laws(p)
    r = pop_of_pops_1rsb(deg, jl, p["beta"], p["m"], **_kw(p, _POP_KEYS, _POP_DEFAULTS))
    return _stat(r, ("e", "f", "q0", "q1"), "1rsb", RSB_CAVEAT)


def _ggs(p):
    deg, jl = _laws(p)
    r = rsb_ground_state_energy(deg, jl, **_kw(p, _GGS_KEYS, _GGS_DEFAULTS))
    return _stat(r, ("e0", "y_star"), "1rsb", RSB_CAVEAT)


def _ansatz(name, beta):
    return lambda p: p.get("ansatz", "rs") == name and (p.get("beta") is not None) == beta


def viana_bray():
    steps = [
        fstep("vb-transition", f"{VB}; {MP01}, Sec. IV", _transition),
        fstep("vb-rs-t0-exact", f"{MP03}, Sec. III.B", _rs_t0,
              when=lambda p: lacks("beta")(p) and _regular_pm1(p),
              skip="needs T = 0 (no beta) on the regular +-1 graph"),
        fstep("vb-1rsb-factorized-t0", f"{MP03}, Sec. V.F.1", _fact_t0,
              when=lambda p: lacks("beta")(p) and _regular_pm1(p),
              skip="needs T = 0 (no beta) on the regular +-1 graph"),
        fstep("vb-rs-population-dynamics", f"{MP01}, Secs. III-IV", _pd,
              when=_ansatz("rs", True), skip="needs beta and ansatz rs"),
        fstep("vb-1rsb-population-of-populations", f"{MP01}, Secs. V-VI", _pop,
              when=_ansatz("1rsb", True), skip="needs beta and ansatz 1rsb"),
        fstep("vb-1rsb-ground-state", f"{MP03}, Secs. IV-V", _ggs,
              when=_ansatz("1rsb", False), skip="needs T = 0 (no beta) and ansatz 1rsb"),
    ]
    k5 = {"k": 5, "beta": 1.25}
    rs5 = {**k5, "pop": 10000, "sweeps": 40, "burn_in": 150, "runs": 8, "seed": 11}
    pop5 = {**k5, "ansatz": "1rsb", "m": 0.24, "pop": 1000, "inner": 100, "sweeps": 20,
            "burn_in": 60, "runs": 8, "seed": 16, "init_spread": 1.0}
    ggs2 = {"ansatz": "1rsb", "ys": (0.3, 0.5, 0.7, 0.9, 1.1), "pop": 2000, "sweeps": 300,
            "burn_in": 150, "runs": 8, "seed": 12}

    def stat(key, obs, params):
        return RefCheck("viana_bray", key, lambda v: v[obs], statistical=True, params=params,
                        route="cavity", slow=True, error=lambda e: e[obs], runs=params["runs"])

    det = lambda key, obs, params=None: RefCheck("viana_bray", key, lambda v: v[obs],
                                                 params=params or {}, route="cavity")
    refs = [det("beta_c_regular_pm1_k2", "beta_c"),
            det("beta_c_poisson_pm1_c3", "beta_c", {"graph": "poisson", "c": 3}),
            det("p0_rs_T0_regular_pm1_k2", "p0_rs_T0"),
            det("e0_rs_regular_pm1_k2", "e0_rs"),
            det("e0_1rsb_factorized_regular_pm1_k2", "e0_1rsb_factorized"),
            det("mu_star_1rsb_factorized_regular_pm1_k2", "mu_star_factorized"),
            stat("q_rs_regular_pm1_k5_T0p8", "q", rs5),
            stat("e_rs_regular_pm1_k5_T0p8", "e", rs5),
            stat("f_rs_regular_pm1_k5_T0p8", "f", rs5),
            stat("e0_1rsb_regular_pm1_k2", "e0", ggs2),
            stat("e_1rsb_regular_pm1_k5_T0p8", "e", pop5),
            stat("q1_1rsb_regular_pm1_k5_T0p8", "q1", pop5),
            stat("f_1rsb_regular_pm1_k5_T0p8", "f", pop5)]
    spec = ModelSpec("viana_bray", "ising", "pm_one", constraint="sparse random graph")
    return entry("viana_bray", spec, f"{VB}; {MP01}; {MP03}", {"cavity": steps}, refs,
                 {"cavity": CAVITY_CAVEAT},
                 "sparse +-J spin glass (Viana-Bray / Bethe lattice): BP and population "
                 "dynamics, RS and 1RSB", default_route="cavity")


def sparse_entries():
    return [viana_bray()]
