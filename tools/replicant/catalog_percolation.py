"""Catalog entries for percolation on trees and random graphs (P1): `percolation_bethe` (the Bethe
lattice / random k-regular graph) and `percolation_graph` (Erdos-Renyi and configuration-model graphs
with a finite degree law), each with bond or site dilution.  Routes:
  cavity  (certified)    the generating-function equations, every fixed point in arb balls with its
                         computed stability (percolation/cavity.py), the cluster-size distribution
                         by two algorithms (percolation/clusters.py), and the closed forms
                         (percolation/closed_forms.py: no generating-function code) as second routes
                         that must overlap (else no-solution:route-disagreement)
  sample  (statistical)  finite graphs over named seeds, union-find (catalog_percolation_sample.py)
Run parameters:
  percolation_bethe   k (coordination, default 3), p (occupation, rational in [0, 1], default 3/4),
                      dilution ("bond" default | "site"), s_max (P_s for s <= s_max, default 6);
                      sample: N, seeds, s_max
  percolation_graph   law ("poisson" default | "finite" | "powerlaw": refused by name), c (Poisson
                      mean degree, default 2), pk (finite law "k:p,k:p" or {k: p}), tau (power law),
                      p (default 1), dilution, s_max; sample: the same law parameters, N, seeds, s_max
At p_c exactly the cavity route refuses ambiguous:critical-point (a marginal double root); a root it
cannot separate from u = 1 is undetermined:near-critical.
"""
from fractions import Fraction

from .catalog import Out, RefCheck, Stability, entry, fstep
from .catalog_lattice_common import _put
from .catalog_percolation_sample import (SAMPLE_CAVEAT, SAMPLE_META, SAMPLE_PARAMS, sample_finality,
                                         sample_step)
from .core import Refusal
from .lattice.ball import arb_q, ball_of, baller
from .percolation import closed_forms as CF
from .percolation.cavity import percolation_giant
from .percolation.clusters import S_MAX, cluster_distribution
from .percolation.laws import occupation, parse_law
from .spec import ModelSpec

NSW = ("Newman, Strogatz & Watts, Phys. Rev. E 64, 026118 (2001) (journal reference as in the task brief, not "
       "printed in the arXiv PDF), arXiv:cond-mat/0007235v2 (PDF read), "
       "Eqs. (9), (26)-(28), (31)-(38)")
CNSW = ("Callaway, Newman, Strogatz & Watts, arXiv:cond-mat/0007300v2 (PDF read; journal reference "
        "not checked), Eqs. (5)-(7), (11), (12), (14), (15)")
MR = ("Molloy & Reed, Random Struct. Alg. 6, 161 (1995) (PDF read), Theorem 1 and Q(D) = sum i(i - 2) "
      "lambda_i, p. 163")
MS = ("Malthe-Sorenssen, Percolation Theory Using Python, Lecture Notes in Physics 1029 (2024), Ch. 3, "
      "doi:10.1007/978-3-031-59900-2_3 (open-access PDF read), Eqs. (3.2)-(3.20), (3.27)")
CAVEAT = ("cavity route, certified: the generating-function (cavity) equations of NSW / CNSW, every fixed "
          "point in [0, 1] enclosed by interval Newton in arb (complete by convexity), its stability the "
          "certified f'(u) = t G1'(u); a monotone Galton-Watson bracket, the closed forms and a second "
          "cluster-size algorithm must overlap; the equations are exact for the locally tree-like N -> oo "
          "limit (physics-level for the random graphs: NSW, CNSW); finite graphs: the sample route")
DERIVED = "derived here (percolation/cavity.py docstring)"
AD = ("Abraham & Delmas, 'An introduction to Bienayme-Galton-Watson trees and their local limits', "
      "arXiv:1506.05571v3 (PDF read), Corollary 2.5 and Lemma 2.6 (extinction a.s. for mean m <= 1 unless "
      "p(1) = 1; P(extinction) = q, the smallest root of g(r) = r in [0, 1])")
GW = ("the cluster of an occupied root of the Bethe lattice, below the root, is a BGW tree with "
      "Bin(k - 1, p) offspring (m = (k - 1) p), and u = 1 - p + p q solves u = 1 - p + p u^(k-1) with q "
      "its extinction probability (" + DERIVED.split(" (")[0] + ")")
JL = ("Janson & Luczak, arXiv:0707.1786v1 (PDF read; Random Struct. Alg. 34, 197 (2009), journal "
      "reference as given by the P1 review, not checked), Theorem 2.3 and Remark 2.7")


def _p(p, default):
    dil = p.get("dilution", "bond")
    return occupation((dil, p.get("p", default)))


def _bethe_law(p):
    return ("bethe", p.get("k", 3)), _p(p, "3/4")


def _graph_law(p):
    law = p.get("law", "poisson")
    if law == "poisson":
        spec = ("poisson", p.get("c", 2))
    elif law == "finite":
        if p.get("pk") is None:
            raise Refusal("precondition:degree-law", "law = finite needs pk ('k:p,k:p' or {k: p})")
        spec = ("finite", p["pk"])
    elif law == "powerlaw":
        spec = ("powerlaw", p.get("tau", "5/2"))
    else:
        raise Refusal("unsupported:degree-law", f"law = {law!r}: poisson, finite (powerlaw refused)")
    return spec, _p(p, 1)


def _threshold(law_of, graph):
    def go(p):
        spec, dil = law_of(p)
        L = parse_law(spec)
        br, z = L.branching(), L.mean()
        balls = {"branching_G1p1": ball_of_q(br)}
        if graph:
            balls["molloy_reed_Q"] = ball_of_q(L.molloy_reed_Q())
        if br:
            tc = 1 / br
            balls.update(p_c=ball_of_q(tc), chi_amplitude=ball_of_q(z * tc * tc))
            if L.g1pp():
                B = 2 * z * br * br / L.g1pp()
                balls["p_inf_slope_at_p_c"] = ball_of_q(B if dil[0] == "bond" else tc * B)
        v, e = _put(**balls)
        return Out(v, f"G1'(1) = {br}; p_c = 1/G1'(1)" + (f" = {1 / br}" if br else " = oo (no giant)") +
                   "; (p_c - p) <s> -> <k> p_c^2 and dP_inf/dp at p_c+ (exact rationals)", errors=e)
    return go


def ball_of_q(q):
    from .lattice.ball import arb_q
    with baller().hygiene.ctx_guard(prec=256):
        return ball_of(arb_q(Fraction(q)))


def _closed(spec, dil, res):
    """Second routes that do not use the generating-function code: {key: Ball}."""
    kind, t = dil
    out = {}
    with CF.prec_guard(256):
        if spec[0] == "bethe":
            k = spec[1]
            if res.phase == "below threshold":
                out["mean_finite_cluster_size_closed_form"] = ball_of(arb_q(CF.bethe_mean_below(k, t)))
                out["P_inf_closed_form"] = ball_of(arb_q(0))
            else:
                u = CF.bethe_u(k, t)
                if u is not None:
                    s = 1 - u ** k
                    out["P_inf_closed_form"] = ball_of(s if kind == "bond" else arb_q(t) * s)
        elif spec[0] == "poisson":
            lam = Fraction(spec[1]) * t
            S = CF.er_giant(lam)
            out["P_inf_closed_form"] = ball_of(S if kind == "bond" else arb_q(t) * S)
            if res.mean_finite is not None:
                out["mean_finite_cluster_size_closed_form"] = ball_of(CF.er_mean(lam, S))
    return out


def _fixed(law_of):
    def go(p):
        spec, dil = law_of(p)
        res = percolation_giant(spec, dil)
        balls = {"u": res.selected.u, "jacobian": res.selected.jacobian, "P_inf": res.P_inf}
        trivial = next(f for f in res.fixed_points if f.branch == "trivial")
        balls["jacobian_trivial"] = trivial.jacobian
        if res.mean_finite is not None:
            balls["mean_finite_cluster_size"] = res.mean_finite
        second = _closed(spec, dil, res)
        for key, b in second.items():
            first = balls[key.replace("_closed_form", "")]
            if not first.overlaps(b):
                raise Refusal("no-solution:route-disagreement", f"{key}: {b} does not overlap the cavity "
                              f"value {first}")
        balls.update(second)
        v, e = _put(**balls)
        meta = {"phase": res.phase, "fixed_point_branches": [f.branch for f in res.fixed_points],
                "selected_branch": res.selected.branch, "selected_stable": res.selected.stable,
                "trivial_stable": trivial.stable, "dilution": dil[0], "occupation": str(dil[1])}
        if res.bracket is not None:
            meta["monotone_bracket"] = res.bracket
        return Out({**v, **meta}, f"{len(res.fixed_points)} fixed point(s) "
                   f"({', '.join(meta['fixed_point_branches'])}), {res.phase}; selected "
                   f"{res.selected.branch} (f'(u) < 1, certified)" +
                   (f"; second routes {', '.join(second)} overlap" if second else ""), errors=e)
    return go


def _clusters(law_of):
    def go(p):
        spec, dil = law_of(p)
        res = percolation_giant(spec, dil)
        s_max = p.get("s_max", 6)
        d = cluster_distribution(spec, dil, s_max, res.S_occupied)
        balls = {f"P_s{s}": b for s, b in enumerate(d.lagrange, 1)}
        third = "none"
        with CF.prec_guard(256):
            for s, b in enumerate(d.lagrange, 1):
                if spec[0] == "bethe":
                    c, third = arb_q(CF.bethe_ps(spec[1], dil[1], s)), "s g_s t^(s-1) (1-t)^((k-2)s+2)"
                elif spec[0] == "poisson":
                    c, third = CF.borel_ps(Fraction(spec[1]) * dil[1], s), "Borel (lambda s)^(s-1) e^(-lambda s)/s!"
                else:
                    break
                if not b.arb.overlaps(c):
                    raise Refusal("no-solution:route-disagreement", f"P_{s}: generating functions {b} vs "
                                  f"the closed form {c.str(12)}")
        v, e = _put(**balls)
        return Out({**v, "cluster_routes": f"series iteration = Lagrange inversion = closed form ({third})"},
                   f"P_s, s = 1 .. {s_max}, for an occupied vertex: series iteration and Lagrange inversion "
                   f"agree; closed form: {third}; partial sums <= 1 - S", errors=e)
    return go


def _stable(cite, graph):
    tail = ("; final as the tree-like N -> oo fixed point; its identification with the giant of a finite "
            "graph is NSW/CNSW physics-level, a theorem for undiluted laws with p_1 > 0 and a finite second "
            "moment (" + JL + ")") if graph else ""
    return lambda v: Stability("stable" if v.get("selected_stable") else "unstable", "cavity", cite,
                               ("the selected (smallest) fixed point: computed f'(u) = t G1'(u) < 1, "
                                "certified ball" + tail) if v.get("selected_stable") else "unstable selection")


def _bethe_theorem(cite):
    return lambda v: (Stability("exact-by-literature", "cavity", f"{AD}; {cite}", f"{GW}; P_inf = 1 - "
                                "G0(u) (bond; t times it for site), the root's k branches being independent; "
                                "certified ball, computed f'(u) < 1") if v.get("selected_stable") else
                      Stability("unstable", "cavity", cite, "unstable selection"))


def _finality(cite, graph):
    st = _stable(cite, graph)
    tc_note = ("1/G1'(1), an exact rational (radius 0); CNSW Eq. (12) (physics-level); t G1'(1) - 1 has "
               "the sign of the thinned law's Molloy-Reed Q = t (t <k(k-1)> - <k>) (" + DERIVED + ")")
    fin = {"branching_G1p1": Stability("exact-by-literature", "cavity", f"{NSW}: Eq. (9), G1 = G0'/z",
                                       "G1'(1) = <k(k-1)>/<k>, an exact rational"),
           "p_c": (Stability("not-computed", "cavity", CNSW, tc_note) if graph else
                   Stability("exact-by-literature", "cavity", f"{AD}; {MS}, Eq. (3.2)", "1/(k - 1), exact "
                             f"rational: {GW}, so the cluster is infinite with positive probability iff "
                             "m = (k - 1) p > 1")),
           "chi_amplitude": Stability("not-computed", "cavity", f"{MS}, Eq. (3.20)" if not graph else CNSW,
                                      "<k> p_c^2 = lim (p_c - p) <s>, exact rational, " + DERIVED),
           "p_inf_slope_at_p_c": Stability("not-computed", "cavity", CNSW if graph else f"{MS}, Eq. (3.15)",
                                           "2 <k> G1'(1)^2 / G1''(1) (times p_c for site), " + DERIVED +
                                           ("" if graph else "; MS (3.15): 6 for Z = 3 site")),
           "jacobian_trivial": lambda v: Stability("stable" if v.get("trivial_stable") else "unstable",
                                                   "cavity", cite, "f'(1) = t G1'(1), exact rational"),
           **{k: st for k in ("u", "jacobian", "P_inf", "mean_finite_cluster_size", "P_inf_closed_form",
                              "mean_finite_cluster_size_closed_form")},
           **{f"P_s{s}": Stability("not-computed", "cavity", f"{NSW}: Eqs. (26)-(28)", "certified "
                                   "coefficient of H0 (Lagrange inversion, " + DERIVED.split(" (")[0] +
                                   "; = NSW's series iteration); tree-like limit") for s in range(1, S_MAX + 1)}}
    if not graph:
        fin.update({k: _bethe_theorem(cite) for k in ("u", "P_inf", "P_inf_closed_form")})
    if graph:
        fin["molloy_reed_Q"] = Stability("exact-by-literature", "cavity", MR, "Q = sum k (k - 2) p_k, an "
                                         "exact rational; Theorem 1: Q > 0 a giant a.s., Q < 0 none (max "
                                         "degree < n^(1/4 - eps)); diluted laws: p_c above")
    return {**fin, **sample_finality()}


_META = ("phase", "fixed_point_branches", "selected_branch", "selected_stable", "trivial_stable",
         "dilution", "occupation", "monotone_bracket", "cluster_routes") + SAMPLE_META


def _entry(name, law_of, graph, spec, cite, refs, desc, params):
    steps = [fstep("percolation-threshold", f"{CNSW}; {MS}" if not graph else f"{CNSW}; {MR}",
                   _threshold(law_of, graph)),
             fstep("percolation-fixed-points", f"{CNSW}; {NSW}" + ("" if graph else f"; {MS}"), _fixed(law_of)),
             fstep("percolation-cluster-sizes", f"{NSW}; {CNSW}", _clusters(law_of))]
    sample = [sample_step("percolation-sampled-graphs", law_of, "finite random graphs, union-find "
                          "(percolation/sampling.py, percolation/unionfind.py)")]
    return entry(name, spec, cite, {"cavity": steps, "sample": sample}, refs,
                 {"cavity": CAVEAT, "sample": SAMPLE_CAVEAT}, desc, default_route="cavity",
                 finality=_finality(f"{CNSW}; {NSW}", graph), meta=_META,
                 parameters={"cavity": params + ("s_max",), "sample": params + SAMPLE_PARAMS},
                 rigor={"cavity": "certified", "sample": "statistical"})


def _rc(key, extract, params=None, statistical=False):
    if statistical:
        seeds = tuple(range(16))
        return RefCheck("percolation", key, extract, statistical=True, route="sample", slow=True,
                        params={**(params or {}), "N": 20000, "seeds": seeds},
                        error=lambda e, k=extract.key: e[k], runs=len(seeds))
    return RefCheck("percolation", key, extract, route="cavity", params=params or {})


class _V:
    """An extractor that remembers its value key (statistical RefChecks read the same key's error)."""
    def __init__(self, key):
        self.key = key

    def __call__(self, v):
        return v[self.key]


def percolation_bethe():
    site = {"dilution": "site", "p": "3/4"}
    refs = [_rc("bethe_pc_k3", _V("p_c")), _rc("bethe_site_p_inf_k3_p3_4", _V("P_inf"), site),
            _rc("bethe_site_p_inf_slope_k3", _V("p_inf_slope_at_p_c"), site),
            _rc("bethe_mean_cluster_k3_p1_4", _V("mean_finite_cluster_size"), {"p": "1/4"}),
            _rc("bethe_site_p_inf_k3_p3_4", _V("P_inf_sampled"), site, statistical=True)]
    spec = ModelSpec("percolation", "occupation", "none", constraint="Bethe lattice / random regular graph")
    return _entry("percolation_bethe", _bethe_law, False, spec, f"{MS}; {CNSW}", refs,
                  "bond / site percolation on the Bethe lattice (coordination k): p_c, P_inf, mean "
                  "cluster size, cluster sizes; certified fixed points and sampled graphs",
                  ("k", "p", "dilution"))


NSW24 = "0:86/1000,1:150/1000,2:363/1000,3:238/1000,4:109/1000,5:54/1000"


def percolation_graph():
    nsw24 = {"law": "finite", "pk": NSW24, "dilution": "site"}
    refs = [_rc("er_giant_c2", _V("P_inf")), _rc("er_mean_cluster_c2", _V("mean_finite_cluster_size")),
            _rc("nsw_eq24_site_threshold", _V("p_c"), nsw24), _rc("nsw_eq24_molloy_reed_Q", _V("molloy_reed_Q"), nsw24),
            _rc("er_giant_c2", _V("P_inf_sampled"), statistical=True)]
    spec = ModelSpec("percolation", "occupation", "none", constraint="configuration model / Erdos-Renyi")
    return _entry("percolation_graph", _graph_law, True, spec, f"{NSW}; {CNSW}; {MR}", refs,
                  "bond / site percolation on Erdos-Renyi and configuration-model graphs: Molloy-Reed "
                  "threshold, giant component, mean cluster size, cluster sizes; certified and sampled",
                  ("law", "c", "pk", "tau", "p", "dilution"))


def percolation_entries():
    return [percolation_bethe(), percolation_graph()]


__all__ = ["percolation_bethe", "percolation_entries", "percolation_graph"]
