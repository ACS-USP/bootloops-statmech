"""Catalog entry `sk_bounds` (E10), route `bound`: Guerra's upper bounds on the SK pressure, certified.

Model: SK (families/sk.py: J_ij ~ N(0, 1/N), field h, P = -beta f).  Why a route of its own (`bound`):
the values are rigorous upper bounds (Guerra, Comm. Math. Phys. 233, 1 (2003), Theorem 3), evaluated in
ball arithmetic (bounds/: python-flint arb through tools/baller), not a replica saddle point and not an
exact solution.  Steps:
  guerra-rs-bound     min over q of P_RS(q), every stationary point, the certified minimiser
                      (bounds/guerra.py)
  guerra-bound-chain  the certified P_K at the package's own 1RSB (and, K = 2, 2RSB) optimiser
                      outputs, and the chain check (bounds/chain.py: consistency of the optimisers with
                      the proved ordering of the infima; a verdict, not a theorem)
Run parameters: beta (rational > 0, default 2), h (default 0), K (1 or 2, default 1; K = 2 needs
h = 0, 1 < beta <= 3; the 2RSB ball takes 37 s at 12 digits, 116 s at 20), digits (certified digits
of the K-step balls, default 20; the RS values always 40).
"""
import mpmath as mp

from .bounds import check_bound_chain, guerra_rs_best_bound
from .catalog import Out, RefCheck, Stability, entry, fstep
from .core import Refusal
from .spec import ModelSpec

GUERRA = ("Guerra, Comm. Math. Phys. 233, 1 (2003), arXiv:cond-mat/0205123, Theorem 3 (bound), "
          "Eqs. (12), (14), (26) (the K-step functional)")
GUERRA_RS = f"{GUERRA}; RS case Eq. (9)"
GUERRA92 = "Guerra (1992), arXiv:1212.2905, Theorem 3, Eq. (3.6) (annealed for beta <= 1)"
TALAGRAND = "Talagrand, Ann. Math. 163, 221 (2006) (the infimum is the limit; not fetched)"
ARB = "python-flint arb through tools/baller (acb.integral, nested; interval Newton)"
CAVEAT = ("bound route, certified: each value is a ball enclosure (" + ARB + ") of Guerra's K-step "
          "functional, a rigorous upper bound on lim (1/N) E ln Z_N for any admissible parameters "
          "(" + GUERRA + "); radii in errors.  The chain verdict compares the package's own RS / "
          "1RSB / 2RSB optimiser outputs: consistency with the proved ordering of the infima, not a "
          "proof (a violation is a solver defect)")
_K = (1, 2)


def _put(**balls):
    vals, errs = {}, {}
    for k, b in balls.items():
        vals[k], errs[k] = b.value_and_error(40)
    return vals, errs


def _beta(p):
    return p.get("beta", 2)


def _K_of(p):
    K = p.get("K", 1)
    if isinstance(K, bool) or K not in _K:
        raise Refusal("precondition:K", f"K = {K!r}: the catalog route certifies K = 1 or 2 "
                      "(bounds.check_bound_chain takes K <= 5, higher levels numerical)")
    return K


def _rs(p):
    best = guerra_rs_best_bound(_beta(p), p.get("h", 0))
    v, e = _put(rs_bound=best, rs_q=best.q)
    sv, sve = best.second_variation.value_and_error(20)
    meta = {"rs_stationary": [f"{s.kind} at q = {s.q.render(12)}" for s in best.stationary],
            "rs_second_variation": f"{sv} +- {sve}", "rs_minimiser_stable": best.stable,
            "rs_certificate": best.certificate}
    return Out({**v, **meta}, f"{len(best.stationary)} stationary point(s) of P_RS(q) on [0, 1]; "
               f"minimum at q = {best.q.render(15)}", errors=e)


def _chain(p):
    K = _K_of(p)
    rep = check_bound_chain(_beta(p), p.get("h", 0), K, digits=p.get("digits", 20), certified_max=K)
    if len(rep.levels) == 1:
        return Out({"chain_verdict": "rs-only", "chain_links": [], "chain_violations": []},
                   rep.notes[0])
    balls = {f"rsb{lv.level}_bound": lv.bound for lv in rep.levels[1:]}
    v, e = _put(**balls)
    meta = {"chain_verdict": "consistent" if rep.ok else "violation" if rep.violations else "undecided",
            "chain_links": [f"{l.lower} <= {l.upper}: {l.verdict}" for l in rep.links],
            "chain_violations": list(rep.violations + rep.undecided)}
    for lv in rep.levels:
        meta[f"{lv.name}_reported"] = f"{mp.nstr(lv.reported, 17)} +- {mp.nstr(lv.tol, 3)}"
        meta[f"{lv.name}_functional"] = lv.functional
    return Out({**v, **meta}, f"chain {' >= '.join(lv.name for lv in rep.levels)}: "
               f"{meta['chain_verdict']}", errors=e)


def _finality():
    lit = lambda note: Stability("exact-by-literature", "guerra", GUERRA, note)
    rs_q = lambda v: Stability("stable" if v.get("rs_minimiser_stable") else "not-computed", "rs",
                               GUERRA_RS, "certified minimiser of P_RS(q); 'stable' iff the computed "
                               "second variation P_RS''(q*) = (beta^2/2) G'(q*) is certified > 0")
    return {"rs_bound": Stability("exact-by-literature", "rs", f"{GUERRA_RS}; {GUERRA92}",
                                  "certified minimum over q of the RS bound: an upper bound on the "
                                  "limit pressure (equal to it for beta <= 1, h = 0)"),
            "rs_q": rs_q,
            "rsb1_bound": lit("Guerra's bound at the package's 1RSB optimiser output, certified"),
            "rsb2_bound": lit("Guerra's bound at the package's 2-step staircase, certified")}


def sk_bounds():
    steps = [fstep("guerra-rs-bound", GUERRA_RS, _rs),
             fstep("guerra-bound-chain", f"{GUERRA}; {TALAGRAND}", _chain)]
    refs = [RefCheck("guerra", "annealed_pressure_beta_half", lambda v: v["rs_bound"],
                     params={"beta": "1/2"}, route="bound")]
    meta = ("rs_stationary", "rs_second_variation", "rs_minimiser_stable", "rs_certificate",
            "chain_verdict",
            "chain_links", "chain_violations") + tuple(
        f"{n}_{s}" for n in ("rs", "1rsb", "2rsb") for s in ("reported", "functional"))
    spec = ModelSpec("sk_bounds", "ising", "gaussian", constraint="Guerra's RSB upper bounds")
    return entry("sk_bounds", spec, f"{GUERRA}; {GUERRA92}", {"bound": steps}, refs,
                 {"bound": CAVEAT}, "SK pressure: Guerra's RS and K-step RSB upper bounds, certified, "
                 "and the bound chain on the package's own optimisers", default_route="bound",
                 finality=_finality(), meta=meta, parameters={"bound": ("beta", "h", "K", "digits")},
                 rigor={"bound": "certified"})


def bounds_entries():
    return [sk_bounds()]


__all__ = ["bounds_entries", "sk_bounds"]
