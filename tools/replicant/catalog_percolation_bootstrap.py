"""Catalog entry `percolation_bootstrap` (P3): bootstrap percolation -- the ell-neighbour infection
dynamics, unrelated to bootstrap resampling -- on the random d-regular graph, as the (d + 1 - ell)-core of
the initially uninfected set (Janson, arXiv:0804.1656, Sec. 5).  Split from catalog_percolation_kcore.py,
whose helpers, citations and caveats it shares; the behaviour is unchanged.  Routes:
  cavity  (certified)    the threshold q_c = 1 - t_c and the final infected fraction from the core
                         equation's certified fixed points; Balogh-Pittel's closed forms and DGM's own form
                         in mpmath must agree (else no-solution:route-disagreement)
  sample  (statistical)  direct spreading on sampled d-regular graphs over named seeds, checked against
                         peeling on every instance (percolation/kcore_sample.py)
Run parameters: d (default 4), ell (infection threshold, default 2), q (initial infection probability,
default 1/20); sample: the same, N, seeds
"""
from fractions import Fraction

from .catalog import RefCheck, Stability, entry, fstep, Out
from .catalog_lattice_common import _put
from .catalog_percolation_kcore import (BP, CAVEAT, DGM, J09, SAMPLE_CAVEAT, SEEDS, STAT, _KMETA, _cond,
                                        _f, _identity, _lit, _sample_out, _second)
from .core import Refusal
from .lattice.ball import arb_q, ball_of, baller
from .percolation.kcore import PREC, kcore_fixed_points
from .percolation.kcore_dgm import dgm_fixed, dgm_threshold
from .percolation.kcore_sample import sample_bootstrap
from .percolation.kcore_threshold import bootstrap_closed_form, occupation_threshold
from .spec import ModelSpec


def _boot_args(p):
    d, ell = p.get("d", 4), p.get("ell", 2)
    if isinstance(d, bool) or not isinstance(d, int) or not 3 <= d <= 1000:
        raise Refusal("precondition:degree", f"d = {d!r}: need an int in 3 .. 1000")
    if isinstance(ell, bool) or not isinstance(ell, int) or not 1 <= ell <= d - 1:
        raise Refusal("precondition:bootstrap-threshold", f"ell = {ell!r}: need an int in 1 .. d - 1 = "
                      f"{d - 1} (J09 Thm 5.1)")
    q = _f(p.get("q", "1/20"), "infection")
    if not 0 <= q <= 1:
        raise Refusal("precondition:infection", f"q = {p.get('q')!r}: need 0 <= q <= 1")
    return d, ell, q


def _boot_threshold(p):
    d, ell, q = _boot_args(p)
    T = occupation_threshold(("bethe", d), 2, d + 1 - ell, "site")
    with baller().hygiene.ctx_guard(prec=PREC):
        balls = {"q_c": ball_of(1 - T.value.arb), "y_star": T.s_star, "uninfected_at_q_c": T.core}
        cf = bootstrap_closed_form(d, ell)
        if cf is not None:
            balls["q_c_closed_form"] = ball_of(arb_q(cf[0]))
            if cf[1] is not None:
                balls["y_star_closed_form"] = ball_of(arb_q(cf[1]))
        for key in ("q_c", "y_star"):
            if f"{key}_closed_form" in balls and not balls[key].overlaps(balls[f"{key}_closed_form"]):
                raise Refusal("no-solution:route-disagreement", f"{key}: certified {balls[key]} vs the "
                              f"Balogh-Pittel closed form {balls[key + '_closed_form']}")
    notes = ()
    if not T.continuous:
        notes = (_second("DGM Eq. (8) in mpmath", lambda dps: (lambda g: {"q_c": 1 - g["threshold_t"]})(
            dgm_threshold(("finite", {d: Fraction(1)}), 2, d + 1 - ell, "site", None, dps)),
            [("q_c", balls["q_c"], "q_c")]),)
    v, e = _put(**balls)
    return Out({**v, "closed_form": cf is not None, "transition": "continuous" if T.continuous else
                "discontinuous"}, f"q_c = {balls['q_c'].render(12)} = 1 - t_c of the ({d + 1 - ell})-core "
               "of the uninfected set (J09 Sec. 5)" + ("; the Balogh-Pittel closed form agrees" if cf else ""),
               errors=e, notes=notes)


def _boot_final(p):
    d, ell, q = _boot_args(p)
    R = kcore_fixed_points(("bethe", d), 2, d + 1 - ell, ("site", 1 - q))
    with baller().hygiene.ctx_guard(prec=PREC):
        balls = {"p_hat": R.selected.x, "jacobian": R.selected.jacobian,
                 "final_infected": ball_of(1 - R.core.arb), "uninfected_core_edges": R.edges}
        _identity("uninfected_core_edges", R.edges, arb_q(d) * R.y.arb * R.selected.x.arb / 2, "d y x / 2")
    note = _second("DGM Eq. (1) in mpmath", lambda dps: (lambda g: {"f": 1 - g["core"]})(dgm_fixed(
        ("finite", {d: Fraction(1)}), 2, d + 1 - ell, "site", 1 - q, dps)),
        [("final_infected", balls["final_infected"], "f")], _cond(R.selected.jacobian))
    v, e = _put(**balls)
    return Out({**v, "phase": "partial infection" if R.selected.x.arb > 0 else "complete infection",
                "fixed_points": [r.x.render(12) for r in R.roots], "selected_stable": R.selected.stable},
               f"final infected fraction {balls['final_infected'].render(12)}; {note}", errors=e, notes=(note,))


def _boot_sample(p):
    d, ell, q = _boot_args(p)
    r = sample_bootstrap(d, ell, q, p.get("N", 2000), p.get("seeds", SEEDS))
    return _sample_out(r, {"final_infected": "final_infected_sampled",
                           "uninfected_core": "uninfected_core_sampled"}, "bootstrap spreading = core")


def percolation_bootstrap():
    refs = [RefCheck("percolation_kcore", "bootstrap_d4_l2_qc", lambda v: v["q_c"], route="cavity"),
            RefCheck("percolation_kcore", "bootstrap_d4_l3_qc", lambda v: v["q_c"], route="cavity",
                     params={"ell": 3})]
    lit = _lit("J09 Thm 5.1 (Balogh & Pittel): q_c = 1 - inf p/P(Bi(d-1, 1-p) <= ell-1), |A_f|/n -> 1 - "
               "(1 - q) P(Bi(d, 1-p^) <= ell-1) at the largest root p^; certified", f"{J09}; {BP}")
    fin = {k: lit for k in ("q_c", "y_star", "p_hat", "final_infected", "uninfected_core_edges")}
    fin.update({"q_c_closed_form": _lit("closed form for ell = 2 (d > 3) and ell = d - 1", BP),
                "y_star_closed_form": _lit("y* = (d-1)(d-3)/(d-2)^2 for ell = 2", BP),
                "uninfected_at_q_c": Stability("not-computed", "cavity", J09, "the q -> q_c- limit at the "
                                               "certified argmin (derived here)"),
                "jacobian": Stability("stable", "cavity", DGM, "F'(p^), certified < 1"),
                "final_infected_sampled": STAT, "uninfected_core_sampled": STAT})
    spec = ModelSpec("percolation", "occupation", "none", constraint="bootstrap percolation")
    return entry("percolation_bootstrap", spec, f"{J09}; {BP}",
                 {"cavity": [fstep("bootstrap-threshold", f"{J09}; {BP}", _boot_threshold),
                             fstep("bootstrap-final-state", J09, _boot_final)],
                  "sample": [fstep("bootstrap-sampled-graphs", "percolation/kcore_sample.py", _boot_sample)]},
                 refs, {"cavity": CAVEAT, "sample": SAMPLE_CAVEAT}, "bootstrap percolation (threshold ell "
                 "infection dynamics) on the random d-regular graph: critical q_c, final infected fraction",
                 default_route="cavity", finality=fin, meta=_KMETA,
                 parameters={"cavity": ("d", "ell", "q"), "sample": ("d", "ell", "q", "N", "seeds")},
                 rigor={"cavity": "certified", "sample": "statistical"})


__all__ = ["percolation_bootstrap"]
