"""TeX hooks of the level-(a) steps (derive.py, compose.level_a_steps): each renders the
sympy objects its step put in the State with sympy.latex (tex.sym), plus the step's own
notes as text.  Nothing here writes an equation by hand: what is displayed is what the
next step consumes.  Called lazily by core.TraceEntry.tex (only when a document is
written), so a calculation that is not written out pays nothing."""
import mpmath as mp
import sympy as sp

from . import derive as D
from .tex import sym, text

LNEZN = sp.Symbol("Phi_n")             # (1/N) ln E Z^n = extr Phi_n (+ correction / N)


def replicate(old, new):
    zn = new["Zn"]
    kind = "symbolic (continued to n -> 0 later)" if isinstance(zn.n, sp.Symbol) else \
        f"explicit, n = {zn.n}"
    return [text("Replicas", f"{zn.spec.family}: n replicas a = 1..n of the N "
                 f"{zn.spec.variables} variables, n {kind}; beta = {zn.beta}.  Tr is the "
                 "trace over every replicated variable; the disorder is still quenched."),
            sym("Replicated partition function", sp.Eq(sp.Symbol("Z") ** zn.n, zn.expr))]


def average_disorder(old, new):
    avg, zn = new["avg"], new["Zn"]
    out = [text("Disorder average", avg.note)]
    if isinstance(avg, D.Averaged):
        K = sp.Function("K")
        out.append(sym("Cumulant generating function of the coupling law, "
                       "K(t) = ln E exp(t J)", sp.Eq(K(avg.t), avg.law.cgf(avg.t, avg.N))))
        out.append(sym("Kept per pair (t = beta X_ij, X_ij = sum_a s_i^a s_j^a)",
                       sp.Eq(K(zn.beta * avg.X), avg.leading)))
        if avg.dropped != 0:
            out.append(sym("First dropped cumulant term (per pair)",
                           sp.Eq(sp.Symbol("Delta_K"), avg.dropped)))
    else:                                    # a pattern average (derive_families.py)
        lam = avg.weight.variables[0]
        out.append(sym("Weight per pattern and replica (hand-supplied)",
                       sp.Eq(sp.Function("g")(lam), avg.weight.expr)))
        if avg.dropped != 0:
            out.append(sym("Dropped cumulant term (per site and pattern)",
                           sp.Eq(sp.Symbol("Delta_K"), avg.dropped)))
    return out


def introduce_order_parameters(old, new):
    F = new["functional"]
    conj = ("imposed by conjugate fields qhat_ab (integrated along the imaginary axis)"
            if F.conjugate else "with no conjugate field (the volume term carries Q)")
    out = [text("Order parameters", "q_ab = (1/N) sum_i s_i^a s_i^b (a < b), " + conj +
                "; the variables then factorize over sites.  (1/N) ln E Z^n = extr Phi_n "
                "+ (correction)/N" + (f"; neglected: {F.missing}" if F.missing else "")),
           sym("Energy term", sp.Eq(sp.Symbol("E_n"), F.energy)),
           sym("Single-site (entropic) term", sp.Eq(sp.Symbol("S_n"), F.entropic)),
           sym("Order-parameter functional", sp.Eq(LNEZN, F.expr))]
    if F.correction != 0:
        out.append(sym("O(1) correction to ln E Z^n", sp.Eq(sp.Symbol("C_n"), F.correction)))
    return out


def saddle_equations(old, new):
    return [text("Saddle point", "N -> infinity: stationarity of Phi_n in every q_ab and "
                 "qhat_ab (explicit n)"), sym("Saddle-point equations", list(new["equations"]))]


def rs_ansatz(old, new):
    R = new["rs"]
    return [text("Replica-symmetric ansatz", "every replica-indexed order parameter set to "
                 "its RS value (" + ", ".join(map(str, R.params)) + "; the same for every "
                 "pair a != b, or every replica a); rules: " + "; ".join(R.rules)),
            sym("RS functional at symbolic n", sp.Eq(sp.Symbol("Phi_n^RS"), R.expr))]


def replica_limit(old, new):
    P = new["rs_free"]
    return [text("Continuation n -> 0", "phi = lim_{n->0} Phi_n^RS / n (polynomial part by "
                 "ansatz.n_to_zero, the rest by d/dn at n = 0: (1/n) ln E[f^n] -> E ln f); "
                 + P.caveat),
            sym("RS free-energy functional (per replica and variable)",
                sp.Eq(sp.Symbol("phi"), P.expr))]


def rs_stationarity(old, new):
    P = new["rs_free"]
    return [text("Stationarity", "d phi / d p = 0 for p in " +
                 ", ".join(str(p) for p in P.params) + " (unsimplified)"),
            sym("RS saddle-point equations", list(new["rs_equations"]))]


def solve_derived(old, new):
    sol = new["solution"]
    fmt = lambda v: mp.nstr(v, 12) if isinstance(v, (mp.mpf, mp.mpc)) else str(v)
    vals = ", ".join(f"{k} = {fmt(v)}" for k, v in sol.values.items())
    rep = "not derived" if sol.replicon is None else fmt(sol.replicon)
    return [text("Solution of the derived equations",
                 f"{vals}; phi = {fmt(sol.free_energy)}; branch {sol.branch or '-'}; "
                 f"residual {fmt(sol.residual)} (independent quadrature), quadrature error "
                 f"{fmt(sol.quad_error)}; derived replicon {rep}; stable = {sol.stable}.  "
                 + (sol.note or ""))]


def replicon_blocks(rep):
    """Blocks of a derive_replicon.Replicon: its rules (text), then each display piece the
    derivation holds (the slot weight, the slot ratios, E''_R, chi_R, lambda_R)."""
    out = [text("Replicon (level (a))", rep.note)]
    return out + [sym(title, obj) for title, obj in rep.parts]


def rsb_ansatz(old, new):
    R = new["rsb"]
    kind = ("frozen 1RSB (q1 = 1: the replicas of a block are one configuration, no qhat1)"
            if R.frozen else "1RSB (Parisi): blocks of x replicas, q1 / qhat1 in a block, "
            "q0 / qhat0 between blocks; x is Parisi's m")
    out = [text("One-step replica symmetry breaking", f"{kind}; unknowns "
                + ", ".join(map(str, R.params)) + "; rules: " + "; ".join(R.rules)),
           sym("1RSB functional at symbolic n", sp.Eq(sp.Symbol("Phi_n^1RSB"), R.expr))]
    if R.conditions:
        out.append(text("Assumed by the reduction (enforced by the solver)",
                        "; ".join(f"{c} > 0" for c in R.conditions)))
    return out


def rsb_replica_limit(old, new):
    P = new["rsb_free"]
    return [text("Continuation n -> 0", "phi = lim_{n->0} Phi_n^1RSB / n: polynomial part by "
                 "ansatz.n_to_zero, (1/n) ln E_t[F^(n/x)] -> (1/x) E_t ln F; " + P.caveat),
            sym("1RSB functional (per replica and variable)", sp.Eq(sp.Symbol("phi"), P.phi))]


def zero_temperature_limit(old, new):
    P = new["rsb_t0"]
    return [text("Zero temperature", P.rules[-1]),
            sym("lim phi/beta (= -e0)", sp.Eq(sp.Symbol("phi_0"), P.phi))]


def rsb_stationarity(old, new):
    P = new["rsb_t0"] if "rsb_t0" in new else new["rsb_free"]
    return [text("Stationarity", "d phi / d p = 0 for p in " + ", ".join(map(str, P.params))
                 + " (unsimplified; the solver eliminates the conjugates first)"),
            sym("1RSB saddle-point equations", list(new["rsb_equations"]))]


def solve_derived_1rsb(old, new):
    from .compose_rsb import describe
    return [text("Solution of the derived 1RSB equations", describe(new["rsb_solution"]))]


HOOKS = {"replicate": replicate, "average-disorder": average_disorder,
         "introduce-order-parameters": introduce_order_parameters,
         "saddle-equations": saddle_equations, "rs-ansatz": rs_ansatz,
         "replica-limit": replica_limit, "rs-stationarity": rs_stationarity,
         "solve-derived": solve_derived, "rsb-ansatz": rsb_ansatz,
         "rsb-replica-limit": rsb_replica_limit, "zero-temperature-limit": zero_temperature_limit,
         "rsb-stationarity": rsb_stationarity, "solve-derived-1rsb": solve_derived_1rsb}
