"""Catalog entries for k-core and bootstrap percolation (P3): `percolation_kcore` (the k-core of
Erdos-Renyi, configuration-model and random regular graphs, and of K-uniform Poisson hypergraphs, with
site or bond dilution) and `percolation_bootstrap` (bootstrap percolation -- the k-neighbour infection
dynamics, unrelated to bootstrap resampling -- on the random d-regular graph).  Routes:
  cavity  (certified)    the core equation's fixed points in arb balls (percolation/kcore.py), the
                         birth point as a certified global minimum (percolation/kcore_threshold.py);
                         second routes that must agree (else no-solution:route-disagreement): DGM's
                         own form in mpmath (percolation/kcore_dgm.py), csp/xorsat.py's 2-core at
                         K >= 3, k = 2 (consumed, never edited), Balogh-Pittel's closed forms
  sample  (statistical)  finite graphs over named seeds, exact peeling / spreading
                         (percolation/kcore_sample.py)
Run parameters:
  percolation_kcore      law ("poisson" default | "finite" | "regular"), c (Poisson mean degree,
                         default 4), alpha (hyperedges per vertex: c = K alpha), pk (finite law),
                         d (regular degree, default 4), K (hyperedge size, default 2), k (core order,
                         default 3), dilution ("site" default | "bond"), p (occupation, default 1);
                         sample: the same (K >= 3: Poisson only), N, seeds
  percolation_bootstrap  d (default 4), ell (infection threshold, default 2), q (initial infection
                         probability, default 1/20); sample: the same, N, seeds
"""
from fractions import Fraction

import mpmath as mp

from .catalog import Out, RefCheck, Stability, entry, fstep
from .catalog_lattice_common import _put
from .core import Refusal
from .lattice.ball import arb_q, ball_of, baller, rational
from .percolation.kcore import PREC, kcore_fixed_points
from .percolation.kcore_dgm import TOL, dgm_fixed, dgm_threshold
from .percolation.kcore_sample import sample_bootstrap, sample_kcore
from .percolation.kcore_threshold import bootstrap_closed_form, degree_threshold, occupation_threshold
from .percolation.laws import occupation, parse_law
from .spec import ModelSpec

DGM = ("Dorogovtsev, Goltsev & Mendes, Phys. Rev. Lett. 96, 040601 (2006), arXiv:cond-mat/0509102v3 "
       "(PDF read), Eqs. (1), (2), (4)-(8), (11)")
JL07 = ("Janson & Luczak, 'A simple solution to the k-core problem', Random Struct. Alg. 30, 50 (2007) "
        "(journal reference as cited in arXiv:math/0612827), arXiv:math/0508453v1 (PDF read), Thm 1.1 "
        "(Pittel, Spencer & Wormald's theorem: lambda_k = min mu/psi_(k-1)(mu), v/n -> psi_k(mu_k), e/n -> "
        "mu_k^2/(2 lambda)) and Thm 2.3")
J09 = ("Janson, 'On percolation in random graphs with given vertex degrees', arXiv:0804.1656v1 (PDF read; "
       "journal reference not checked), Eq. (4.9), Thms 4.5 (site), 4.6 (bond), Remark 4.9 (k = 2), "
       "Thm 5.1 (bootstrap percolation on G(n, d), stating Balogh & Pittel), Thm A.1")
BP = ("Balogh & Pittel, Random Struct. Alg. 30, 257 (2007) (not fetched), as stated by Bartha & Pete, "
      "'Noise sensitivity in bootstrap percolation', arXiv:1509.08454v1 (PDF read, page 7 rendered), "
      "Eqs. (3.1), (3.2), Thm 3.1, and by Janson, arXiv:0804.1656, Thm 5.1")
MRZ = ("csp/xorsat.py (E5): Mezard, Ricci-Tersenghi & Zecchina, J. Stat. Phys. 111, 505 (2003), "
       "arXiv:cond-mat/0207140, Eqs. (11), (30), (32), (33)")
CAVEAT = ("cavity route, certified: the k-core equation x = a Psi(b x^(K-1)) of DGM / Janson, every "
          "fixed point in [0, 1] isolated and enclosed in arb, the largest (the peeling limit) selected "
          "with its certified F'(x) < 1; the birth point a certified global minimum; DGM's own form in "
          "mpmath must agree; exact for the locally tree-like N -> oo limit; finite graphs: the sample route")
SAMPLE_CAVEAT = ("sample route, statistical: finite random (hyper)graphs over named seeds, the k-core by "
                 "exact peeling, bootstrap percolation by direct spreading (with the exact identity "
                 "between them checked on every instance); means +- stderr, never final; meaningless "
                 "inside the transition window |t - t_c| <~ N^(-1/2)")
STAT = Stability("not-computed", "sample", "percolation/kcore_sample.py", "finite-N statistical estimate "
                 "over named seeds: NOT final")
SEEDS = tuple(range(8))


def _f(x, name):
    return rational(x, name, positive=False)


def _kcore_args(p):
    law, K = p.get("law", "poisson"), p.get("K", 2)
    if isinstance(K, bool) or not isinstance(K, int) or not 2 <= K <= 12:
        raise Refusal("precondition:hyperedge-size", f"K = {K!r}: need an int in 2 .. 12")
    if law == "poisson":
        if p.get("alpha") is not None and p.get("c") is not None:
            raise Refusal("precondition:mean-degree", "give c or alpha (c = K alpha), not both")
        c = _f(p["alpha"], "mean-degree") * K if p.get("alpha") is not None else _f(p.get("c", 4), "mean-degree")
        spec = ("poisson", c)
    elif law == "finite":
        if p.get("pk") is None:
            raise Refusal("precondition:degree-law", "law = finite needs pk ('d:p,d:p' or {d: p})")
        spec = ("finite", p["pk"])
    elif law == "regular":
        spec = ("bethe", p.get("d", 4))
    else:
        raise Refusal("unsupported:degree-law", f"law = {law!r}: poisson, finite, regular")
    return spec, K, p.get("k", 3), occupation((p.get("dilution", "site"), p.get("p", 1)))


def _dgm_law(spec):
    L = parse_law(spec)
    return ("poisson", Fraction(L.c)) if L.kind == "poisson" else ("finite", L.degree())


def _agree(key, ball, v, route):
    with baller().hygiene.ctx_guard(prec=PREC):
        with mp.workdps(40):
            d = abs(ball.mid(30) - v)
            if d > ball.rad() + TOL * max(1, abs(v)):
                raise Refusal("no-solution:route-disagreement", f"{key}: certified {ball.render(15)} vs "
                              f"{route} {mp.nstr(v, 15)}")


def _kcore_threshold(p):
    spec, K, k, (kind, t) = _kcore_args(p)
    T = occupation_threshold(spec, K, k, kind)
    balls = {"p_c": T.value, "y_at_p_c": T.s_star, "x_at_p_c": T.x, "core_at_p_c": T.core,
             "edges_at_p_c": T.edges}
    routes = []
    if spec[0] == "poisson" and t > 0:
        D = degree_threshold(K, k, kind, t)
        balls.update(c_k=D.value, mu_at_c_k=D.s_star, core_at_c_k=D.core, edges_at_c_k=D.edges)
        if K > 2:
            with baller().hygiene.ctx_guard(prec=PREC):
                balls["alpha_k"] = ball_of(D.value.arb / K)
    if not T.continuous:
        g = dgm_threshold(_dgm_law(spec), K, k, kind, t if spec[0] == "poisson" and t > 0 else None)
        for key, gk in (("p_c", "threshold_t"), ("y_at_p_c", "y_star"), ("core_at_p_c", "core_at_jump"),
                        ("c_k", "c_k"), ("mu_at_c_k", "mu_star")):
            if key in balls and gk in g:
                _agree(key, balls[key], g[gk], "DGM Eqs. (5)-(8)")
        routes.append("DGM (8) in mpmath")
        if K > 2 and k == 2 and spec[0] == "poisson" and t == 1:
            from .csp.xorsat import xorsat_thresholds
            X = xorsat_thresholds(K)
            _agree("alpha_k", balls["alpha_k"], X.alpha_d, "csp/xorsat.py alpha_d")
            _agree("mu_at_c_k", balls["mu_at_c_k"], X.lambda_d, "csp/xorsat.py lambda_d")
            routes.append("csp/xorsat.py alpha_d, lambda_d (E5)")
    v, e = _put(**balls)
    with baller().hygiene.ctx_guard(prec=PREC):
        possible = bool(T.value.arb <= 1)
    meta = {"transition": "continuous" if T.continuous else "discontinuous", "threshold_boxes": T.boxes,
            "threshold_at_endpoint": T.endpoint, "core_possible_at_p_le_1": possible,
            "hyperedge_size": K, "core_order": k, "threshold_routes": routes}
    return Out({**v, **meta}, f"birth point p_c = {T.value.render(12)} ({meta['transition']}; "
               f"{T.boxes} boxes)" + (f"; c_k = {balls['c_k'].render(12)}" if "c_k" in balls else "") +
               (f"; agrees with {', '.join(routes)}" if routes else ""), errors=e)


def _kcore_fixed(p):
    spec, K, k, (kind, t) = _kcore_args(p)
    R = kcore_fixed_points(spec, K, k, (kind, t))
    balls = {"x": R.selected.x, "y": R.y, "jacobian": R.selected.jacobian, "core_fraction": R.core,
             "core_edges": R.edges}
    g = dgm_fixed(_dgm_law(spec), K, k, kind, t)
    _agree("x", R.selected.x, g["x"], "DGM Eqs. (2), (4)")
    _agree("core_fraction", R.core, g["core"], "DGM Eqs. (1), (11)")
    routes = ["DGM (2), (4), (1) in mpmath"]
    if K > 2 and k == 2 and spec[0] == "poisson" and t == 1 and R.selected.x.arb > 0:
        from .csp.xorsat import xorsat_core_size
        with mp.workdps(40):                 # csp/xorsat.py parses alpha at the caller's precision
            X = xorsat_core_size(K, f"{spec[1].numerator}/{spec[1].denominator * K}")
        _agree("core_fraction", R.core, X.variables, "csp/xorsat.py V")
        _agree("core_edges", R.edges, X.checks, "csp/xorsat.py C")
        _agree("x", R.selected.x, X.q, "csp/xorsat.py q")
        routes.append("csp/xorsat.py 2-core V, C, q (E5)")
    v, e = _put(**balls)
    meta = {"phase": R.phase, "fixed_points": [r.x.render(12) for r in R.roots],
            "selected_stable": R.selected.stable, "hyperedge_size": K, "core_order": k,
            "fixed_point_routes": routes}
    if R.bracket is not None:
        meta["peeling_bracket"] = R.bracket
    return Out({**v, **meta}, f"{len(R.roots)} fixed point(s) in [0, 1]; the largest x = "
               f"{R.selected.x.render(12)} selected (F'(x) < 1, certified); core {R.core.render(12)}; "
               f"agrees with {', '.join(routes)}", errors=e)


def _kcore_sample(p):
    spec, K, k, dil = _kcore_args(p)
    r = sample_kcore(spec, K, k, dil, p.get("N", 2000), p.get("seeds", SEEDS))
    ren = {"core_fraction": "core_fraction_sampled", "core_edges": "core_edges_sampled"}
    return _sample_out(r, ren, "k-core by peeling")


def _sample_out(r, ren, what):
    vals = {ren[k]: r.mean[k] for k in ren}
    errs = {ren[k]: r.stderr[k] for k in ren}
    hw = ", ".join(f"{ren[k]} +- {r.halfwidth99[k]:.3g}" for k in ren)
    return Out({**vals, "N": r.N, "n_seeds": len(r.seeds)}, f"{len(r.seeds)} sampled graphs (seeds "
               f"{r.seeds[0]}..{r.seeds[-1]}), N = {r.N}: {what}", errors=errs, statistical=tuple(vals),
               notes=(f"statistical: means over named seeds, stderr in errors; 99% Student-t "
                      f"half-widths: {hw}",))


def _lit(note, cite):
    return Stability("exact-by-literature", "cavity", cite, note)


def _kcore_finality():
    def graph(note_k2, note_hyper):
        def f(v):
            if v.get("hyperedge_size", 2) == 2:
                return _lit(note_k2, f"{J09}; {JL07}; {DGM}")
            return Stability("stable" if v.get("selected_stable", True) else "unstable", "cavity",
                             f"{DGM}; {MRZ}", note_hyper)
        return f
    at_run = graph("J09 Thm 4.5 (site) / 4.6 (bond) (ii): v/n -> a P(D_p >= k), e/n -> lambda p^2/2 at "
                   "the largest root p^ (JL07 Thm 1.1 for G(n, c/n)), valid as the certified F'(x) < 1 "
                   "rules out a local maximum point; below the birth point (i): o_p(n), empty w.h.p. "
                   "for k >= 3", "K >= 3: the tree-like fixed point with computed F'(x) < 1 (derived "
                   "here; at k = 2 = csp/xorsat.py's leaf-removal 2-core); no theorem fetched")
    thr = lambda v: (_lit("pi_c = inf lambda p^2/h(p) (J09 Eq. (4.9)); k = 2: 1/G1'(1) (Remark 4.9); "
                          "certified global minimum", f"{J09}; {JL07}; {DGM}")
                     if v.get("hyperedge_size", 2) == 2 else
                     Stability("not-computed", "cavity", f"{DGM}; {MRZ}", "K >= 3: derived here (the "
                               "minimum of y/Psi(y)^(K-1)); at k = 2 = E5's alpha_d"))
    jump = Stability("not-computed", "cavity", f"{DGM}; {J09}", "the t -> t_c+ limit of the selected root "
                     "at the certified argmin (derived here); at the threshold itself the core's size "
                     "is not a law of large numbers (Janson & Luczak, arXiv:math/0612827, Thm 1.3)")
    fin = {k: at_run for k in ("x", "y", "core_fraction", "core_edges")}
    fin["jacobian"] = Stability("stable", "cavity", DGM, "F'(x) of the selected root, certified < 1")
    fin.update({k: thr for k in ("p_c", "c_k", "alpha_k")})
    fin.update({k: jump for k in ("y_at_p_c", "x_at_p_c", "core_at_p_c", "edges_at_p_c", "mu_at_c_k",
                                  "core_at_c_k", "edges_at_c_k")})
    fin.update({k: STAT for k in ("core_fraction_sampled", "core_edges_sampled")})
    return fin


def percolation_kcore():
    refs = [RefCheck("percolation_kcore", "er_3core_threshold", lambda v: v["c_k"], route="cavity"),
            RefCheck("percolation_kcore", "er_4core_threshold", lambda v: v["c_k"], route="cavity",
                     params={"k": 4, "c": 6}, slow=True),
            RefCheck("xorsat", "alpha_d_k3", lambda v: v["alpha_k"], route="cavity",
                     params={"K": 3, "k": 2, "alpha": 1}),
            RefCheck("xorsat", "lambda_d_k3", lambda v: v["mu_at_c_k"], route="cavity",
                     params={"K": 3, "k": 2, "alpha": 1})]
    spec = ModelSpec("percolation", "occupation", "none", constraint="k-core")
    params = ("law", "c", "alpha", "pk", "d", "K", "k", "dilution", "p")
    steps = [fstep("kcore-threshold", f"{DGM}; {J09}; {JL07}", _kcore_threshold),
             fstep("kcore-fixed-points", f"{DGM}; {J09}", _kcore_fixed)]
    return entry("percolation_kcore", spec, f"{DGM}; {JL07}; {J09}",
                 {"cavity": steps, "sample": [fstep("kcore-sampled-graphs", "percolation/kcore_sample.py",
                                                    _kcore_sample)]}, refs,
                 {"cavity": CAVEAT, "sample": SAMPLE_CAVEAT}, "the k-core of random graphs and K-uniform "
                 "hypergraphs (site / bond dilution): birth point, jump, core size; certified and sampled",
                 default_route="cavity", finality=_kcore_finality(), meta=_KMETA,
                 parameters={"cavity": params, "sample": params + ("N", "seeds")},
                 rigor={"cavity": "certified", "sample": "statistical"})


_KMETA = ("transition", "threshold_boxes", "threshold_at_endpoint", "core_possible_at_p_le_1",
          "hyperedge_size", "core_order", "threshold_routes", "phase", "fixed_points", "selected_stable",
          "fixed_point_routes", "peeling_bracket", "N", "n_seeds", "closed_form")


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
    if not T.continuous:
        g = dgm_threshold(("finite", {d: Fraction(1)}), 2, d + 1 - ell, "site", None)
        _agree("q_c", balls["q_c"], 1 - g["threshold_t"], "DGM Eq. (8)")
    v, e = _put(**balls)
    return Out({**v, "closed_form": cf is not None, "transition": "continuous" if T.continuous else
                "discontinuous"}, f"q_c = {balls['q_c'].render(12)} = 1 - t_c of the ({d + 1 - ell})-core "
               "of the uninfected set (J09 Sec. 5)" + ("; the Balogh-Pittel closed form agrees" if cf else ""),
               errors=e)


def _boot_final(p):
    d, ell, q = _boot_args(p)
    R = kcore_fixed_points(("bethe", d), 2, d + 1 - ell, ("site", 1 - q))
    with baller().hygiene.ctx_guard(prec=PREC):
        balls = {"p_hat": R.selected.x, "jacobian": R.selected.jacobian,
                 "final_infected": ball_of(1 - R.core.arb), "uninfected_core_edges": R.edges}
    g = dgm_fixed(("finite", {d: Fraction(1)}), 2, d + 1 - ell, "site", 1 - q)
    _agree("final_infected", balls["final_infected"], 1 - g["core"], "DGM Eq. (1)")
    v, e = _put(**balls)
    return Out({**v, "phase": "partial infection" if R.selected.x.arb > 0 else "complete infection",
                "fixed_points": [r.x.render(12) for r in R.roots], "selected_stable": R.selected.stable},
               f"final infected fraction {balls['final_infected'].render(12)}", errors=e)


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


def kcore_entries():
    return [percolation_kcore(), percolation_bootstrap()]


__all__ = ["kcore_entries", "percolation_bootstrap", "percolation_kcore"]
