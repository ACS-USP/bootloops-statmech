"""Catalog entries for k-core and bootstrap percolation (P3): `percolation_kcore` (the k-core of
Erdos-Renyi, configuration-model and random regular graphs, and of K-uniform Poisson hypergraphs, with
site or bond dilution) and `percolation_bootstrap` (bootstrap percolation -- the k-neighbour infection
dynamics, unrelated to bootstrap resampling -- on the random d-regular graph).  Routes:
  cavity  (certified)    the core equation's fixed points in arb balls (percolation/kcore.py), the
                         birth point as a certified global minimum (percolation/kcore_threshold.py);
                         second routes that must agree (else no-solution:route-disagreement): DGM's
                         own form in mpmath (percolation/kcore_dgm.py), csp/xorsat.py's 2-core at
                         K >= 3, k = 2 (consumed, never edited), Balogh-Pittel's closed forms, the edge
                         identity; an auxiliary route that does not compute is skipped with a note
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
from .percolation.kcore_dgm import DPS, TOL, dgm_fixed, dgm_threshold
from .percolation.kcore_sample import sample_bootstrap, sample_kcore
from .percolation.kcore_threshold import bootstrap_closed_form, degree_threshold, occupation_threshold
from .percolation.laws import occupation, parse_law
from .spec import ModelSpec

DGM = ("Dorogovtsev, Goltsev & Mendes, Phys. Rev. Lett. 96, 040601 (2006), arXiv:cond-mat/0509102v3 "
       "(PDF read), Eqs. (1), (2), (4)-(8), (11)")
PSW = ("Pittel, Spencer & Wormald, 'Sudden emergence of a giant k-core in a random graph', J. Combin. "
       "Theory B 67, 111 (1996), preprint cs.nyu.edu/~spencer/papers/k-core.pdf (PDF read, page 4 rendered): "
       "abstract (c_k = min lambda/pi_k(lambda), pi_k = P(Po >= k - 1)), Thms 1-3 (the newborn core ~ "
       "p_k(lambda_k) n, p_k = P(Po(lambda_k) >= k))")
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


_ROUTE_ERRORS = (ArithmeticError, ValueError, TypeError)
FIRST = ("p_c (and c_k) is the core's first appearance (J09 Eq. (4.9)); further first-order jumps of the "
         "core above it (J09 Remark 4.7: other local maxima of h(p)/p^2) are not enumerated; the run point "
         "always selects the largest root")


def _off(ball, v, eps, cond):
    """|ball - v| beyond the radius plus max(TOL, 10 eps / cond) (relative for |v| > 1): eps the route's
    precision, cond = |1 - F'(x)| for values read off a root (a near-double root amplifies eps)."""
    with baller().hygiene.ctx_guard(prec=PREC), mp.workdps(50):
        tol = max(TOL, 10 * mp.mpf(eps) / cond) * max(1, abs(v))
        return abs(ball.mid(45) - v) > ball.rad() + tol


def _second(name, compute, pairs, cond=1):
    """An auxiliary route against the certified balls: compute(dps) -> {key: mpf}; pairs [(label, ball,
    key)].  At DPS digits, then at 40 before refusing no-solution:route-disagreement; a route that fails
    to compute at both precisions is skipped with the reason (it never blocks a certified run)."""
    errs = []
    for dps in (DPS, 40):
        try:
            g = compute(dps)
        except _ROUTE_ERRORS as exc:
            errs.append(f"{dps} digits: {type(exc).__name__}: {exc}")
            continue
        bad = [(lab, b, g[k]) for lab, b, k in pairs if k in g and _off(b, g[k], mp.mpf(10) ** -dps, cond)]
        if not bad:
            return f"{name}: agrees" + (" at 40 digits (20 were not enough here)" if dps == 40 else "")
        if dps == 40:
            lab, b, v = bad[0]
            raise Refusal("no-solution:route-disagreement", f"{lab}: certified {b.render(15)} vs {name} "
                          f"{mp.nstr(v, 15)} (also at 40 digits)")
    return f"{name}: SKIPPED, it did not compute here ({'; '.join(errs)})"


def _csp(name, compute, pairs, cond=1):
    """csp/xorsat.py (30 digits, its own precision) against the balls; skipped if it fails."""
    try:
        g = compute()
    except _ROUTE_ERRORS + (Refusal,) as exc:     # e.g. its own residual audit near the jump
        return f"{name}: SKIPPED, it did not compute here ({type(exc).__name__}: {exc})"
    for lab, b, v in pairs:
        if _off(b, g(v), mp.mpf(10) ** -30, cond):
            raise Refusal("no-solution:route-disagreement", f"{lab}: certified {b.render(15)} vs {name} "
                          f"{mp.nstr(g(v), 15)}")
    return f"{name}: agrees"


def _identity(key, ball, other, what):
    """The edge identity e = <d> y x / K (JL07 Thm 1.1: mu^2/(2 lambda) at K = 2, t = 1; a hyperedge
    survives with probability b x^(K-1) x = y x): a second path through the selected root."""
    with baller().hygiene.ctx_guard(prec=PREC):
        if not ball.arb.overlaps(other):
            raise Refusal("no-solution:route-disagreement", f"{key} = {ball.render(15)} but {what} = "
                          f"{other.str(15)}: the edge identity e = <d> y x / K fails")


def _cond(jac):
    with mp.workdps(20):
        return max(mp.mpf(1) - mp.mpf(jac.arb.upper().str(20, radius=False)), mp.mpf(10) ** -300)


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
        with baller().hygiene.ctx_guard(prec=PREC):
            z = arb_q(parse_law(spec).mean())
            _identity("edges_at_p_c", T.edges, z * T.s_star.arb * T.x.arb / K, "<d> y x / K at the jump")
            if "c_k" in balls:
                _identity("edges_at_c_k", D.edges, D.s_star.arb * D.x.arb / K, "mu x / K at the jump")
        tck = t if spec[0] == "poisson" and t > 0 else None
        routes.append(_second("DGM Eqs. (5)-(8) in mpmath", lambda dps: dgm_threshold(
            _dgm_law(spec), K, k, kind, tck, dps), [(key, balls[key], gk) for key, gk in (
                ("p_c", "threshold_t"), ("y_at_p_c", "y_star"), ("core_at_p_c", "core_at_jump"),
                ("c_k", "c_k"), ("mu_at_c_k", "mu_star")) if key in balls]))
        if K > 2 and k == 2 and spec[0] == "poisson" and t == 1:
            from .csp.xorsat import xorsat_thresholds
            routes.append(_csp("csp/xorsat.py alpha_d, lambda_d (E5)", lambda: (
                lambda X: lambda key: getattr(X, key))(xorsat_thresholds(K)),
                [("alpha_k", balls["alpha_k"], "alpha_d"), ("mu_at_c_k", balls["mu_at_c_k"], "lambda_d")]))
    v, e = _put(**balls)
    with baller().hygiene.ctx_guard(prec=PREC):
        possible = bool(T.value.arb <= 1)
    meta = {"transition": "continuous" if T.continuous else "discontinuous", "threshold_boxes": T.boxes,
            "threshold_at_endpoint": T.endpoint, "core_possible_at_p_le_1": possible,
            "hyperedge_size": K, "core_order": k, "threshold_routes": routes}
    return Out({**v, **meta}, f"birth point p_c = {T.value.render(12)} ({meta['transition']}; "
               f"{T.boxes} boxes)" + (f"; c_k = {balls['c_k'].render(12)}" if "c_k" in balls else "") +
               (f"; {'; '.join(routes)}" if routes else ""), errors=e, notes=(FIRST,) + tuple(routes))


def _kcore_fixed(p):
    spec, K, k, (kind, t) = _kcore_args(p)
    R = kcore_fixed_points(spec, K, k, (kind, t))
    balls = {"x": R.selected.x, "y": R.y, "jacobian": R.selected.jacobian, "core_fraction": R.core,
             "core_edges": R.edges}
    with baller().hygiene.ctx_guard(prec=PREC):
        _identity("core_edges", R.edges, arb_q(parse_law(spec).mean()) * R.y.arb * R.selected.x.arb / K,
                  "<d> y x / K")
    cond = _cond(R.selected.jacobian)
    routes = [_second("DGM Eqs. (2), (4), (1) and Janson's edges in mpmath", lambda dps: dgm_fixed(
        _dgm_law(spec), K, k, kind, t, dps), [("x", R.selected.x, "x"), ("core_fraction", R.core, "core"),
                                              ("core_edges", R.edges, "edges")], cond)]
    if K > 2 and k == 2 and spec[0] == "poisson" and t == 1 and R.selected.x.arb > 0:
        from .csp.xorsat import xorsat_core_size

        def X():
            with mp.workdps(40):             # csp/xorsat.py parses alpha at the caller's precision
                c = xorsat_core_size(K, f"{spec[1].numerator}/{spec[1].denominator * K}")
            return lambda key: getattr(c, key)
        routes.append(_csp("csp/xorsat.py 2-core V, C, q (E5)", X, [
            ("core_fraction", R.core, "variables"), ("core_edges", R.edges, "checks"),
            ("x", R.selected.x, "q")], cond))
    v, e = _put(**balls)
    meta = {"phase": R.phase, "fixed_points": [r.x.render(12) for r in R.roots],
            "selected_stable": R.selected.stable, "hyperedge_size": K, "core_order": k,
            "fixed_point_routes": routes}
    if R.bracket is not None:
        meta["peeling_bracket"] = R.bracket
    return Out({**v, **meta}, f"{len(R.roots)} fixed point(s) in [0, 1]; the largest x = "
               f"{R.selected.x.render(12)} selected (F'(x) < 1, certified); core {R.core.render(12)}; "
               f"{'; '.join(routes)}", errors=e, notes=tuple(routes))


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
                          "certified global minimum; " + FIRST, f"{J09}; {JL07}; {PSW}; {DGM}")
                     if v.get("hyperedge_size", 2) == 2 else
                     Stability("not-computed", "cavity", f"{DGM}; {MRZ}", "K >= 3: derived here (the "
                               "minimum of y/Psi(y)^(K-1)); at k = 2 = E5's alpha_d; " + FIRST))
    jump = Stability("not-computed", "cavity", f"{DGM}; {J09}; {PSW}", "the t -> t_c+ limit of the selected "
                     "root at the certified argmin (derived here; for ER at t = 1, K = 2, PSW Thms 2, 3: the "
                     "newborn core ~ p_k(lambda_k) n); at the threshold itself the core's size is not a law "
                     "of large numbers (Janson & Luczak, arXiv:math/0612827, Thm 1.3)")
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
            RefCheck("percolation_kcore", "er_3core_newborn_size", lambda v: v["core_at_c_k"], route="cavity"),
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


def kcore_entries():
    return [percolation_kcore(), percolation_bootstrap()]


__all__ = ["kcore_entries", "percolation_bootstrap", "percolation_kcore"]
