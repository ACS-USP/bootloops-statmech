"""Catalog entries `ising_triangular` and `ising_honeycomb` (route `exact`, certified; I3): critical
points, free energies, the star-triangle and duality relations, the triangular antiferromagnet's
ground-state entropy.  New entries (not run parameters of ising2d), so ising2d stays byte-identical.

Every physical value is a python-flint arb ball (lattice/tri_honey.py, lattice/geom_tori.py); the
second and third routes of a value are values too (each its own ball), and must overlap the first
(else no-solution:route-disagreement).  Finite tori are exact (transfer matrix in balls, and the
enumeration of all configurations); the sequence of cylinders towards the bulk is a consistency
check with no proven bound (reported in meta; refused no-solution:torus-limit when it ends outside its band, 10 x
its last change).  Run parameters:
  ising_triangular (exact)  beta (rational > 0, default 1/5), J1, J2, J3 (nonzero rationals, default
                            1; any sign: the free energy; all > 0: the critical point), L (torus
                            side 3 .. 6, default 4)
  ising_honeycomb (exact)   beta (default 2/5), J1, J2, J3 (> 0, default 1), L (3 .. 4, default 3)
"""
import mpmath as mp

from .catalog import Out, RefCheck, Stability, entry, fstep
from .core import Refusal
from .lattice import anisotropic as A
from .lattice import geom_tori as G
from .lattice import tri_honey as T
from .lattice.ball import arb_q, baller, certify
from .spec import ModelSpec

BAX = "Baxter, Exactly Solved Models (1982) (ANU PDF, page images read)"
AU = ("Athanasopoulos & Ueltschi, Ann. Henri Poincare (2024), arXiv:2407.05303v2 (read), Theorem "
      "2.1(b) (Houtappel, Physica 16, 425 (1950), not fetched)")
PBM = "Pierre, Bernu & Messio, SciPost Phys. 19, 025 (2025), arXiv:2501.18569v2 (PDF read)"
ST = f"{BAX}, Eqs. (6.4.4)-(6.4.16)"
DUAL = f"{BAX}, Eqs. (6.3.6)-(6.3.8), (6.5.1)-(6.5.7)"
TORI = "exact finite tori (lattice/geom_tori.py; derived here)"
LIT = lambda cite, note: Stability("exact-by-literature", "exact", cite, note)
CONS = lambda cite, note: Stability("not-computed", "exact", cite, note)
AF_MAX_K = 3                # |K| beyond which the frustrated integrand is not certified (measured)


def _put(**balls):
    vals, errs = {}, {}
    for k, b in balls.items():
        vals[k], errs[k] = b.value_and_error(40)
    return vals, errs


def _s(x, d=6):
    return mp.nstr(x, d)


def _js(p, positive):
    Js = tuple(p.get(f"J{i}", 1) for i in (1, 2, 3))
    A.couplings(p.get("beta", "0.2"), Js, positive)       # validates (refusals precondition:J*)
    return Js


def _side(p, default, hi):
    L = p.get("L", default)
    if isinstance(L, bool) or not isinstance(L, int) or not 3 <= L <= hi:
        raise Refusal("precondition:L", f"L = {L!r}: need an int in 3 .. {hi}")
    return L


def _agree(name, first, **others):
    for k, b in others.items():
        if not first.overlaps(b):
            raise Refusal("no-solution:route-disagreement", f"{name}: {k} {b} does not overlap "
                          f"{first}")


def cylinder_sequence(geom, widths, Ks, bulk):
    """-beta f of infinite cylinders of circumference m (the Perron root of the row transfer
    matrix, enclosed by Collatz-Wielandt: lattice/geom_tori.cylinder_free_energy), m in `widths`:
    the sequence, its last change and the offset of its last term from the bulk.  The band is 10 x
    the last change (heuristic: it covers a geometric tail of ratio <= 0.9); a sequence ending
    outside it is refused no-solution:torus-limit.  `resolved`: band < 1e-4."""
    vals = [G.cylinder_free_energy(geom, m, Ks).mid() for m in widths]
    with mp.workdps(45):
        est, b = vals[-1], bulk.mid()
        band = max(10 * abs(vals[-1] - vals[-2]), mp.mpf("1e-30") * max(1, abs(est)))
        off = abs(est - b)
    out = {"widths": list(widths), "minus_beta_f": [_s(v, 12) for v in vals], "band": _s(band, 2),
           "offset": _s(off, 2), "resolved": band < mp.mpf("1e-4"),
           "kind": "finite-size consistency check (cylinders), no proven bound"}
    if not off <= band:
        raise Refusal("no-solution:torus-limit", f"{geom}: the cylinder sequence ends at "
                      f"{_s(est, 12)}, {_s(off, 2)} from the bulk {_s(b, 12)}, outside its band "
                      f"{_s(band, 2)} (10 x the last change)")
    return out


def _torus(geom, L, Ks, bulk, sizes):
    """The L x L torus by the transfer matrix, the enumeration where it is small, the cylinder
    sequence (circumferences `sizes`) towards the bulk."""
    tz = G.torus_free_energy(geom, L, L, Ks)
    t = G.GEOM[geom][0]
    meta = {"torus_size": f"{L} x {L}"}
    if t * L * L <= 16:                 # 2^16 configurations (the 18-site honeycomb: tests)
        E, cnt = G.class_sizes(geom, L, L), G.counts(geom, L, L)
        ez = certify(lambda: G.z_of_counts(cnt, E, [arb_q(k) for k in Ks]).log() / (t * L * L))
        if not tz.overlaps(ez):
            raise Refusal("no-solution:torus-disagreement", f"{geom} {L} x {L}: transfer matrix "
                          f"{tz} and enumeration {ez} differ")
        meta["torus_enumeration"] = "equal (all 2^n configurations)"
    meta["cylinder_sequence"] = cylinder_sequence(geom, sizes, Ks, bulk)
    return tz, meta


# ---- ising_triangular --------------------------------------------------------------------------
def _ferro(p):
    return all(A.couplings(1, (j,), False)[0] > 0 for j in _js(p, False))


def _tri_crit(p):
    Js = _js(p, True)
    beta = p.get("beta", "0.2")
    bc = T.triangular_critical_beta(*Js)
    js = A.couplings(1, Js)
    with baller().hygiene.ctx_guard(prec=600):
        Kc = [bc.arb * arb_q(j) for j in js]
        S = [(2 * k).sinh() for k in Kc]
        crit_min = (2 * Kc[0]).cosh() * (2 * Kc[1]).cosh() * (2 * Kc[2]).cosh() + S[0] * S[1] * S[2] \
            - S[0] - S[1] - S[2]
        fixed = T.triangular_self_dual(Kc)
        ok_fixed = all(f.overlaps(k) for f, k in zip(fixed, Kc))
    if not crit_min.contains(0):
        raise Refusal("no-solution:critical-point", f"the Kac-Ward integrand's minimum at beta_c is "
                      f"{crit_min.str(5)}, not 0")
    if not ok_fixed:
        raise Refusal("no-solution:duality-fixed-point", "star-triangle(dual(K_c)) != K_c: "
                      + ", ".join(f.str(8) for f in fixed))
    ks = A.couplings(beta, Js)
    kap = T.triangular_kappa(beta, *Js)
    sd = [certify(lambda i=i: T.triangular_self_dual([arb_q(k) for k in ks])[i]) for i in range(3)]
    b65 = [certify(lambda i=i: (T._kappa(*[arb_q(k) for k in ks]) * (2 * arb_q(ks[i])).sinh())
                   .asinh() / 2) for i in range(3)]
    if not all(x.overlaps(y) for x, y in zip(sd, b65)):
        raise Refusal("no-solution:duality-fixed-point", "star-triangle(dual(K)) differs from "
                      "Baxter (6.5.2) sinh 2K* = k sinh 2K")
    v, e = _put(beta_c=bc, kappa=kap, self_dual_K1=sd[0], self_dual_K2=sd[1], self_dual_K3=sd[2])
    phase = "T > T_c" if kap.mid() > 1 else "T < T_c"
    return Out({**v, "phase": phase}, f"beta_c from k = 1 (Baxter (6.4.16), (6.5.5)); the Kac-Ward "
               "integrand's minimum vanishes there; K_c is the fixed point of star-triangle o "
               f"duality; at beta: k = {_s(kap.mid())} ({phase}), K* = Baxter (6.5.2)", errors=e)


def _tri_f(p):
    Js, beta = _js(p, False), p.get("beta", "0.2")
    ks = A.couplings(beta, Js, False)
    ferro = _ferro(p)
    if not ferro and max(abs(k) for k in ks) > AF_MAX_K:
        raise Refusal("precondition:beta", f"frustrated couplings with max |K| = "
                      f"{float(max(abs(k) for k in ks)):.4g} > {AF_MAX_K}: the integrand's near-zero "
                      "(the ground-state soft mode) is not certified here; T = 0 is the "
                      "ground_state_entropy (J1 = J2 = J3 < 0)")
    f = T.triangular_free_energy(beta, *Js)
    balls = {"minus_beta_f": f}
    if ferro:
        balls["minus_beta_f_square_route"] = T.triangular_free_energy_square(beta, *Js)
        if Js[0] == Js[1] == Js[2]:
            balls["minus_beta_f_pierre"] = T.triangular_free_energy_pierre(beta, Js[0])
    _agree("triangular -beta f", f, **{k: b for k, b in balls.items() if k != "minus_beta_f"})
    v, e = _put(**balls)
    return Out(v, "-beta f: Kac-Ward single integral" + (", = Baxter (11.7.13) (three square "
               "lattices through the star-triangle map)" if ferro else "") +
               (", = Pierre et al. (19)-(20)" if "minus_beta_f_pierre" in balls else ""), errors=e)


def _tri_dual(p):
    d = G.duality_identity(3, 3)
    if not d.exact:
        raise Refusal("no-solution:duality-identity", f"3 x 3 torus: the four-sector duality "
                      f"identity fails (map {d.mobius})")
    return Out({"duality_identity": {"torus": "3 x 3", "monomials": d.terms, "exact": True,
                                     "periodic_sector_alone_mismatches": d.periodic_only_mismatch}},
               "duality (triangular high-T, four sectors) = dual honeycomb (low-T) as an identity of "
               f"integer polynomials on the 3 x 3 torus ({d.terms} monomials); the periodic sector "
               f"alone misses {d.periodic_only_mismatch} (the topology term)")


def _tri_torus(p):
    Js, beta = _js(p, False), p.get("beta", "0.2")
    L = _side(p, 4, 6)
    ks = A.couplings(beta, Js, False)
    bulk = T.triangular_free_energy(beta, *Js)
    tz, meta = _torus("triangular", L, ks, bulk, (3, 4, 5))
    v, e = _put(torus_minus_beta_f=tz)
    return Out({**v, **meta}, f"{L} x {L} triangular torus (transfer matrix in balls" +
               (", = enumeration" if "torus_enumeration" in meta else "") + "); cylinders of "
               "circumference 3, 4, 5 towards the bulk (consistency check)", errors=e)


def _af(p):
    Js = _js(p, False)
    return Js[0] == Js[1] == Js[2] and A.couplings(1, (Js[0],), False)[0] < 0


def _wannier(p):
    s0 = T.wannier_entropy()
    beta, (J,) = p.get("beta", "0.2"), _js(p, False)[:1]
    (K,) = A.couplings(beta, (J,), False)
    f = T.triangular_free_energy(beta, J, J, J)
    with mp.workdps(45):
        excess = f.mid() - abs(mp.mpf(K.numerator) / K.denominator) - s0.mid()
    if not excess >= -f.rad() - s0.rad():
        raise Refusal("no-solution:ground-state", f"-beta f - beta|e0| < s0 by {_s(-excess, 3)}: "
                      "Z >= (ground states) e^{-beta E0} violated")
    counts = {}
    for Lx, Ly in ((3, 3),):              # 6 x 3, 6 x 6, 9 x 3: tests/test_lattice_geom.py
        c = G.univariate(G.counts("triangular", Lx, Ly), 3 * Lx * Ly)
        jmax = max(j for j, m in enumerate(c) if m)
        with mp.workdps(20):
            counts[f"{Lx} x {Ly}"] = {"ground_states": c[jmax], "frustrated_bonds": 3 * Lx * Ly - jmax,
                                      "ln_count_per_site": _s(mp.log(c[jmax]) / (Lx * Ly), 8)}
    v, e = _put(ground_state_entropy=s0)
    return Out({**v, "ground_state_counts": counts, "entropy_excess_at_beta": _s(excess, 6)},
               "s0 = (2/pi) int_0^{pi/3} ln(2 cos w) dw (Pierre et al. Table 5: 0.3231); -beta f - "
               "beta |e0| >= s0 at the run's beta; the exact ground-state count on the 3 x 3 torus "
               "(finite-size, no bound)", errors=e)


def ising_triangular():
    steps = [fstep("triangular-critical-point", f"{DUAL}; {ST}", _tri_crit, when=_ferro,
                   skip="a coupling is negative: no ferromagnetic critical point computed"),
             fstep("triangular-free-energy", f"{AU}; {BAX}, Eqs. (11.7.13), (11.7.15); {PBM}",
                   _tri_f),
             fstep("triangular-duality-identity", f"{DUAL}; {TORI}", _tri_dual),
             fstep("triangular-torus", TORI, _tri_torus),
             fstep("triangular-antiferromagnet", f"{PBM}, Sec. 5.2, Table 5", _wannier, when=_af,
                   skip="not the isotropic antiferromagnet (J1 = J2 = J3 < 0)")]
    rc = lambda key, k, **pp: RefCheck("lattice", key, (lambda k: lambda v: v[k])(k), params=pp,
                                       route="exact")
    refs = [rc("triangular_beta_c", "beta_c"), rc("triangular_minus_beta_f_beta0_2", "minus_beta_f"),
            rc("triangular_af_ground_state_entropy", "ground_state_entropy", J1=-1, J2=-1, J3=-1)]
    crit = f"{BAX}, Eqs. (6.4.16), (6.5.5), (6.5.6)"
    fin = {"beta_c": LIT(crit, "k = 1: closed form (1/2) asinh(3^-1/2)/J when J1 = J2 = J3, else the "
                         "certified root of Baxter's k(beta) = 1 (k decreasing in beta)"),
           "kappa": LIT(f"{BAX}, Eq. (6.4.16)", "Baxter's k at beta, certified ball"),
           **{f"self_dual_K{i}": LIT(f"{BAX}, Eq. (6.5.2)", "the triangular self-dual coupling, "
                                     "certified ball (= star-triangle of the dual honeycomb)")
              for i in (1, 2, 3)},
           "minus_beta_f": LIT(AU, "Kac-Ward free energy (inner angle in closed form, derived "
                               "here), certified ball"),
           "minus_beta_f_square_route": LIT(f"{BAX}, Eqs. (11.7.13), (11.7.15)", "three square "
                                            "lattices and R, certified ball"),
           "minus_beta_f_pierre": LIT(f"{PBM}, Eqs. (19), (20), Table 2", "isotropic, certified ball "
                                      "(inner angle in closed form, derived here)"),
           "torus_minus_beta_f": LIT(TORI, "exact by construction: ln Z / N of the finite torus, "
                                     "certified ball"),
           "ground_state_entropy": LIT(f"{PBM}, Sec. 5.2, Table 5 (0.3231; Wannier, Phys. Rev. 79, "
                                       "357 (1950), not fetched)", "certified ball of (2/pi) "
                                       "int_0^{pi/3} ln(2 cos w) dw, reduced here from Pierre et "
                                       "al.'s X_k(-1) = 16 (9/4 - xi(k))")}
    spec = ModelSpec("ising_triangular", "ising", "none", constraint="triangular lattice, exact")
    return entry("ising_triangular", spec, f"{AU}; {BAX}, Ch. 6; {PBM}", {"exact": steps}, refs,
                 {"exact": "exact route, certified: Kac-Ward / Baxter / Pierre free energies and "
                           "Baxter's critical point in ball arithmetic; finite tori exact; the torus "
                           "sequence is a consistency check"},
                 "Triangular-lattice Ising model (zero field, three couplings): critical point, free "
                 "energy (three routes), exact duality identity, finite tori, the antiferromagnet's "
                 "ground-state entropy, certified", default_route="exact", finality=fin,
                 meta=("phase", "duality_identity", "torus_size", "torus_enumeration",
                       "cylinder_sequence", "ground_state_counts", "entropy_excess_at_beta"),
                 parameters={"exact": ("beta", "J1", "J2", "J3", "L")}, rigor={"exact": "certified"})


# ---- ising_honeycomb ---------------------------------------------------------------------------
def _hc_crit(p):
    Js, beta = _js(p, True), p.get("beta", "0.4")
    bc = T.honeycomb_critical_beta(*Js)
    js = A.couplings(1, Js)
    with baller().hygiene.ctx_guard(prec=600):
        Lc = [bc.arb * arb_q(j) for j in js]
        fixed = T.honeycomb_self_dual(Lc)
        kap = T._kappa(*T.star_triangle(*Lc)[:3])
        ok = all(f.overlaps(x) for f, x in zip(fixed, Lc))
    if not ok:
        raise Refusal("no-solution:duality-fixed-point", "dual(star-triangle(L_c)) != L_c: "
                      + ", ".join(f.str(8) for f in fixed))
    if not kap.contains(1):
        raise Refusal("no-solution:critical-point", f"the star-triangle image of L_c has k = "
                      f"{kap.str(8)}, not 1")
    lv = T.honeycomb_l(beta, *Js)
    v, e = _put(beta_c=bc, l=lv)
    phase = "T > T_c" if lv.mid() > 1 else "T < T_c"
    return Out({**v, "phase": phase}, "beta_c from l = 1 (Baxter (11.7.6)); L_c is the fixed point "
               "of duality o star-triangle, and its star-triangle image has k = 1 (Baxter (6.4.16)); "
               f"l = {_s(lv.mid())} at beta ({phase})", errors=e)


def _hc_f(p):
    Js, beta = _js(p, True), p.get("beta", "0.4")
    f = T.honeycomb_free_energy(beta, *Js)
    balls = {"minus_beta_f": f, "minus_beta_f_duality": T.honeycomb_free_energy_duality(beta, *Js)}
    if Js[0] == Js[1] == Js[2]:
        balls["minus_beta_f_pierre"] = T.honeycomb_free_energy_pierre(beta, Js[0])
    _agree("honeycomb -beta f", f, **{k: b for k, b in balls.items() if k != "minus_beta_f"})
    v, e = _put(**balls)
    return Out(v, "-beta f: star-triangle from the triangular (Baxter (6.4.7)) = duality from the "
               "triangular at the dual couplings (Baxter (6.3.7))" +
               (" = Pierre et al. (19)-(20)" if "minus_beta_f_pierre" in balls else ""), errors=e)


def _hc_st(p):
    Js, beta = _js(p, True), p.get("beta", "0.4")
    ls = A.couplings(beta, Js)
    # 4 x 3, not L x L: the L x L torus is symmetric under every permutation of the three edge
    # directions, so a K on the wrong edge would pass there
    zh, zt = G.star_triangle_check(4, 3, ls)
    if not zh.overlaps(zt):
        raise Refusal("no-solution:star-triangle-identity", f"4 x 3: Z_honeycomb {zh} != R^N "
                      f"Z_triangular {zt}")
    return Out({"star_triangle_identity": {"torus": "4 x 3 (24 + 12 sites)", "Z_honeycomb": zh.render(25),
                                           "R_N_Z_triangular": zt.render(25), "overlap": True}},
               "Z_H(L) = R^N Z_T(K) on the matched 4 x 3 tori, both by the transfer matrix in balls")


def _hc_torus(p):
    Js, beta = _js(p, True), p.get("beta", "0.4")
    L = _side(p, 3, 4)
    ks = A.couplings(beta, Js)
    tz, meta = _torus("honeycomb", L, ks, T.honeycomb_free_energy(beta, *Js), (2, 3))
    v, e = _put(torus_minus_beta_f=tz)
    return Out({**v, **meta}, f"{L} x {L} honeycomb torus ({2 * L * L} sites; transfer matrix in "
               "balls" + (", = enumeration" if "torus_enumeration" in meta else "") + "); cylinders "
               "of circumference 2, 3 cells towards the bulk (consistency check)", errors=e)


def ising_honeycomb():
    steps = [fstep("honeycomb-critical-point", f"{BAX}, Eqs. (11.7.6), (6.5.7); {DUAL}", _hc_crit),
             fstep("honeycomb-free-energy", f"{ST}; {DUAL}; {PBM}", _hc_f),
             fstep("honeycomb-star-triangle-identity", f"{ST}; {TORI}", _hc_st),
             fstep("honeycomb-torus", TORI, _hc_torus)]
    rc = lambda key, k: RefCheck("lattice", key, (lambda k: lambda v: v[k])(k), route="exact")
    refs = [rc("honeycomb_beta_c", "beta_c"), rc("honeycomb_minus_beta_f_beta0_4", "minus_beta_f")]
    fin = {"beta_c": LIT(f"{BAX}, Eqs. (6.5.7), (11.7.6)", "closed form (1/2) asinh(3^1/2)/J when "
                         "J1 = J2 = J3, else the certified root of l(beta) = 1 (l decreasing)"),
           "l": LIT(f"{BAX}, Eq. (11.7.6)", "Baxter's l at beta, certified ball"),
           "minus_beta_f": LIT(f"{ST}; {AU}", "star-triangle (6.4.7) from the Kac-Ward triangular "
                               "free energy, certified ball"),
           "minus_beta_f_duality": LIT(f"{DUAL}; {AU}", "duality (6.3.7) from the triangular at "
                                       "tanh K = e^{-2L}, certified ball"),
           "minus_beta_f_pierre": LIT(f"{PBM}, Eqs. (19), (20), Table 2", "isotropic, certified "
                                      "ball (inner angle in closed form, derived here)"),
           "torus_minus_beta_f": LIT(TORI, "exact by construction: ln Z / N of the finite torus, "
                                     "certified ball")}
    spec = ModelSpec("ising_honeycomb", "ising", "none", constraint="honeycomb lattice, exact")
    return entry("ising_honeycomb", spec, f"{BAX}, Ch. 6, Sec. 11.7; {PBM}", {"exact": steps}, refs,
                 {"exact": "exact route, certified: the honeycomb free energy through Baxter's "
                           "star-triangle and duality maps from the triangular one (and Pierre et "
                           "al.'s isotropic integral) in ball arithmetic; finite tori exact"},
                 "Honeycomb-lattice Ising model (zero field, three couplings > 0): critical point, "
                 "free energy (three routes), the star-triangle identity on a finite torus, "
                 "certified", default_route="exact", finality=fin,
                 meta=("phase", "star_triangle_identity", "torus_size", "torus_enumeration",
                       "cylinder_sequence"),
                 parameters={"exact": ("beta", "J1", "J2", "J3", "L")}, rigor={"exact": "certified"})


__all__ = ["ising_honeycomb", "ising_triangular", "cylinder_sequence"]
