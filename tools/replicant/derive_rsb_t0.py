"""Conjugate elimination and the T = 0 limit of the derived 1RSB functional (derive_rsb.py).

  eliminate_conjugates(phi, params) -> (psi, remaining params, [(qhat, S)], notes)
      each conjugate qhat_j is solved from d phi/d q_j = 0 when that equation is linear in
      qhat_j outside every expectation and qhat_j occurs in none of them; psi(rest) =
      phi(rest, S(rest)) has the same stationary points (d psi/d q = phi_qhat S'(q),
      d psi/d x = phi_x), and it is the functional the replica rule extremises
  zero_temperature_limit() -> Step "zero-temperature-limit" (State key `rsb_t0`)
      psi with q1 = 1 and x = mu/beta, then lim_{beta->oo} psi/beta (derive_rsb docstring);
      the finite-beta positivity conditions are dropped (the T = 0 expectations are closed
      forms with no integrability condition left)
"""
from dataclasses import replace

import sympy as sp

from . import derive as D
from . import gauss
from . import gauss_rsb as GR
from .core import Refusal, Step
from .gauss import GaussE

_PAIRS = (("qhat0", "q0"), ("qhat1", "q1"))


def _masked_linear(e, x):
    atoms = sorted(e.atoms(GaussE), key=str)
    dum = [sp.Dummy("G") for _ in atoms]
    m = sp.expand(e.xreplace(dict(zip(atoms, dum))))
    try:
        p = sp.Poly(m, x)
    except sp.PolynomialError:
        return None
    if p.degree() != 1:
        return None
    back = dict(zip(dum, atoms))
    c1, c0 = p.coeff_monomial(x).xreplace(back), p.coeff_monomial(1).xreplace(back)
    return None if c1 == 0 or c1.has(x) else (c1, c0)


def _radicals(e):
    """sqrt(a b^2 ...) with positive b pulled out (sqrt(factor(arg)))."""
    return e.replace(lambda y: y.is_Pow and y.exp == sp.S.Half and not y.base.is_number,
                     lambda y: sp.sqrt(sp.factor(y.base)))


def eliminate_conjugates(phi, params):
    by = {p.name: p for p in params}
    rest, elim, notes, sol = list(params), [], [], {}
    for qh, q in _PAIRS:        # every equation is taken from the ORIGINAL phi
        if qh not in by or q not in by:
            continue
        x, v = by[qh], by[q]
        eq = sp.diff(phi, v)
        others = [by[h] for h, _ in _PAIRS if h in by and h != qh]
        if any(x in g.free_symbols or any(o in g.free_symbols for o in others)
               for g in eq.atoms(GaussE)):
            notes.append(f"{x} kept as an unknown: a conjugate occurs inside an expectation "
                         f"of d phi/d {v}")
            continue
        lin = _masked_linear(eq, x)
        if lin is None or any(lin[0].has(o) or lin[1].has(o) for o in others):
            notes.append(f"{x} kept as an unknown: d phi/d {v} is not linear in it alone")
            continue
        sol[x] = sp.cancel(-lin[1] / lin[0]) if not lin[1].atoms(GaussE) else -lin[1] / lin[0]
        notes.append(f"eliminated {x} = {sol[x]} (from d phi/d {v} = 0)")
    psi = _radicals(phi.subs(sol)) if sol else phi
    for x, S in sol.items():
        rest.remove(x)
        elim.append((x, S))
    return psi, tuple(rest), elim, notes


def _close_abs(expr):
    """E_u exp(c |a + s u| + r) -> closed form (abs_split, linearity, the exp-step rule)."""
    def one(g):
        if not g.f.has(sp.Abs):
            return g
        parts = gauss.linear(GaussE(sp.expand(GR.abs_split(g.f)), g.z))
        return parts.replace(lambda h: isinstance(h, GaussE) and h.z == g.z,
                             lambda h: gauss.expect_closed(h.f, h.z))
    return expr.replace(lambda y: isinstance(y, GaussE) and not y.f.atoms(GaussE), one)


def zero_temperature_limit() -> Step:
    from .derive_rsb import MU, Q1, X

    def requires(state):
        if "rsb_free" not in state:
            return ["rsb_free"]
        P = state["rsb_free"]
        if P.frozen:
            raise Refusal("precondition:not-frozen", "the T = 0 limit is for the full 1RSB")
        F = state["functional"]
        if F.energy.has(D.LnGaussProd) or F.energy.has(D.LogDet) or \
                F.entropic.has(D.LogDet):
            raise Refusal("precondition:polynomial-energy-for-t0",
                          "with a Gaussian-integral energy (or a spherical volume) the T = 0 "
                          "1RSB limit needs the scaling q1 = 1 - C/beta, not implemented")
        if D.BETA not in P.phi.free_symbols:
            raise Refusal("precondition:symbolic-beta", "phi has no symbolic beta")
        return []

    def apply(state):
        P = state["rsb_free"]
        beta = D.BETA
        psi, rest, elim, notes = eliminate_conjugates(P.phi, P.params)
        if any(p.name.startswith("qhat") for p in rest):
            raise Refusal("precondition:conjugates-eliminated", "; ".join(notes))
        psi = gauss.linear(sp.expand(psi.subs({Q1: 1, X: MU / beta}) / beta, deep=False))
        terms = sp.Add.make_args(sp.expand(psi, deep=False))
        rest_terms = [y for y in terms if y.atoms(GaussE)]
        out = sp.limit(sp.Add(*(y for y in terms if not y.atoms(GaussE))), beta, sp.oo)
        for term in rest_terms:
            r = GR.zero_temperature_power(term, beta)
            if r.has(beta):
                raise Refusal("precondition:tempered-power",
                              f"{term}: beta left after the zero-temperature rule")
            out += _close_abs(r)
        if not out.is_finite and out.has(sp.oo):
            raise Refusal("no-solution", f"lim phi/beta diverges: {out}")
        params = tuple(p for p in rest if p not in (Q1, X)) + (MU,)
        note = ("T = 0 1RSB ansatz q1 = 1, x = mu/beta; " + "; ".join(notes)
                + "; lim phi/beta: polynomial part by sympy limit, (2 cosh beta y)^(mu/beta) "
                "-> exp(mu |y|) (gauss_rsb.zero_temperature_power), E_u closed by abs_split "
                "and the exp-step rule")
        return state.with_(rsb_t0=replace(P, phi=out, params=params,
                                          rules=P.rules + (note,), fixed={Q1: sp.S(1)},
                                          conditions=()))

    return Step("zero-temperature-limit", requires, apply,
                note=lambda s: s["rsb_t0"].rules[-1], tex=D._tex("zero-temperature-limit"))
