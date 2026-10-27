"""Catalog entry `ising2d_anisotropic` (route `exact`, certified; I3): the square lattice with
couplings K1 = beta J1 (horizontal), K2 = beta J2 (vertical), J1, J2 > 0.  A new entry rather than
a K2 parameter of ising2d: every ising2d step (Yang, Kaufman, xi, the exponents) is isotropic, and
its runs stay byte-identical.

  aniso-critical-line   beta_c of the J2/J1 ratio (root of sinh 2K1 sinh 2K2 = 1, Baxter (6.2.17)),
                        sinh 2K1 sinh 2K2 at beta, K2 on the line given K1; the Kac-Ward integrand's
                        minimum vanishes at beta_c (else no-solution:critical-point) and the line is
                        fixed by the duality map (else no-solution:duality-fixed-point)
  aniso-free-energy     -beta f (Kac-Ward), = Baxter (11.7.15) (else no-solution:route-disagreement),
                        = Onsager (lattice/onsager.py) when J1 = J2; the duality relation between
                        (K1, K2) and (K1*, K2*) (else no-solution:duality)
  aniso-torus           the L x L torus (transfer matrix; = the bulk at beta = 1e-8 within O(K^L),
                        else no-solution:torus-weight), the cylinder sequence
Run parameters: beta (rational > 0, default 1/5), J1 (default 1), J2 (default 2; nonzero, computed
as |J|: gauge), L (3 .. 6, default 4).
"""
from .catalog import Out, RefCheck, entry, fstep
from .catalog_lattice_common import AU, BAX, LIT, TORI, _agree, _put, _raw, _s, _side, _torus, gauge
from .core import Refusal
from .lattice import anisotropic as A
from .lattice.ball import arb_q, baller
from .lattice.onsager import onsager_free_energy
from .spec import ModelSpec

CRIT = f"{BAX}, Eq. (6.2.17) and p. 89 (k = (sinh 2K sinh 2L)^-1 = 1 is T_c, Ch. 7)"
DUALITY = f"{BAX}, Eqs. (6.2.11)-(6.2.15)"
SQ = f"{BAX}, Eqs. (11.7.15), (11.7.16)"


def _p(p):
    """beta, |J| (every square plaquette has two edges of each class: any sign pattern is gauge-
    equivalent to |J|), K, and the gauge note."""
    beta = p.get("beta", "0.2")
    Js, note = gauge(_raw(p, 2))
    return beta, Js, A.couplings(beta, Js), note


def _crit(p):
    beta, Js, (K1, K2), note = _p(p)
    bc = A.critical_beta(*Js)
    j1, j2 = A.couplings(1, Js)
    with baller().hygiene.ctx_guard(prec=600):
        k1, k2 = bc.arb * arb_q(j1), bc.arb * arb_q(j2)
        m = (2 * k1).cosh() * (2 * k2).cosh() - (2 * k1).sinh() - (2 * k2).sinh()
        d1, d2 = A.dual_pair(k1, k2)                 # the map the entry reports (dual_K1, dual_K2)
        fixed = d1.overlaps(k1) and d2.overlaps(k2)
    if not m.contains(0):
        raise Refusal("no-solution:critical-point", f"cosh 2K1 cosh 2K2 - sinh 2K1 - sinh 2K2 = "
                      f"{m.str(5)} at beta_c, not 0 (the Kac-Ward integrand's minimum)")
    if not fixed:
        raise Refusal("no-solution:duality-fixed-point", f"the duality map moves the critical "
                      f"point: (K1*, K2*) = ({d1.str(8)}, {d2.str(8)}) against ({k1.str(8)}, "
                      f"{k2.str(8)})")
    lam = A.criticality(beta, *Js)
    v, e = _put(beta_c=bc, criticality=lam, critical_K2_given_K1=A.critical_coupling(K1))
    phase = "T < T_c" if lam.mid() > 1 else "T > T_c"
    return Out({**v, "phase": phase, **({"gauge": note} if note else {})}, f"beta_c of J2/J1 = {Js[1]}/{Js[0]}: sinh 2K1 sinh 2K2 = 1 "
               "(certified bisection); the integrand's minimum vanishes there and the duality map "
               f"fixes it; at beta sinh 2K1 sinh 2K2 = {_s(lam.mid())} ({phase})", errors=e)


def _free(p):
    beta, Js, _, _ = _p(p)
    f = A.free_energy(beta, *Js)
    balls = {"minus_beta_f": f, "minus_beta_f_baxter": A.free_energy_baxter(beta, *Js)}
    iso = Js[0] == Js[1]
    if iso:
        balls["minus_beta_f_onsager"] = onsager_free_energy(beta, Js[0])
    _agree("anisotropic -beta f", f, **{k: b for k, b in balls.items() if k != "minus_beta_f"})
    lhs, rhs = A.duality_relation(beta, *Js)
    if not lhs.overlaps(rhs):
        raise Refusal("no-solution:duality", f"-beta f(K*) = {lhs} but -beta f(K) + ln(2 cosh K1* "
                      f"cosh K2*) - K1 - K2 = {rhs}")
    d1, d2 = A.dual_couplings(beta, *Js)
    v, e = _put(**balls, minus_beta_f_dual=lhs, dual_K1=d1, dual_K2=d2)
    return Out(v, "-beta f: the Kac-Ward single integral = Baxter's diagonal form (11.7.15)" +
               (" = Onsager (J1 = J2)" if iso else "") + "; duality: -beta f(K1*, K2*) = -beta "
               "f(K1, K2) + ln(2 cosh K1* cosh K2*) - K1 - K2", errors=e)


def _tor(p):
    beta, Js, ks, _ = _p(p)
    L = _side(p, 4, 6)
    tz, meta = _torus("square", L, ks + [0], A.free_energy(beta, *Js), (3, 4, 5), tuple(Js) + (0,),
                      lambda b: A.free_energy(b, *Js))
    v, e = _put(torus_minus_beta_f=tz)
    return Out({**v, **meta}, f"{L} x {L} anisotropic torus (transfer matrix in balls; = the bulk at "
               "beta = 1e-8 within O(K^L)); cylinders of circumference 3, 4, 5 towards the bulk "
               "(consistency check)", errors=e)


def ising2d_anisotropic():
    steps = [fstep("aniso-critical-line", CRIT, _crit),
             fstep("aniso-free-energy", f"{AU} (J3 = 0); {SQ}; {DUALITY}", _free),
             fstep("aniso-torus", TORI, _tor)]
    rc = lambda key, k: RefCheck("lattice", key, (lambda k: lambda v: v[k])(k), route="exact")
    refs = [rc("aniso_square_beta_c_j1_1_j2_2", "beta_c"),
            rc("aniso_square_minus_beta_f_beta0_2_j1_1_j2_2", "minus_beta_f")]
    dual = lambda which: LIT(f"{BAX}, Eq. (6.2.14a)", f"tanh K{which}* = e^(-2 K{3 - which}), "
                             "certified ball")
    fin = {"beta_c": LIT(CRIT, "certified root of sinh(2 beta J1) sinh(2 beta J2) = 1 (increasing in "
                         "beta); J1 = J2: (1/2) ln(1 + sqrt 2)/J"),
           "criticality": LIT(CRIT, "sinh 2K1 sinh 2K2 = k^-1, certified ball"),
           "critical_K2_given_K1": LIT(CRIT, "(1/2) asinh(1/sinh 2K1), certified ball"),
           "minus_beta_f": LIT(f"{AU} (J3 = 0)", "Kac-Ward free energy (inner angle in closed "
                               "form, derived here), certified ball"),
           "minus_beta_f_baxter": LIT(SQ, "Baxter's single integral, certified ball"),
           "minus_beta_f_onsager": LIT("Onsager (1944), lattice/onsager.py", "J1 = J2: the isotropic "
                                       "certified ball"),
           "minus_beta_f_dual": LIT(f"{DUALITY}; {AU}", "-beta f at the dual couplings, certified "
                                    "ball (equal to the duality relation's right side)"),
           "dual_K1": dual(1), "dual_K2": dual(2),
           "torus_minus_beta_f": LIT(TORI, "exact by construction: ln Z / N of the finite torus, "
                                     "certified ball")}
    spec = ModelSpec("ising2d_anisotropic", "ising", "none",
                     constraint="square lattice, anisotropic couplings, exact")
    return entry("ising2d_anisotropic", spec, f"{AU}; {BAX}, Eqs. (6.2.17), (11.7.15)",
                 {"exact": steps}, refs,
                 {"exact": "exact route, certified: the anisotropic free energy (two independently "
                           "derived single integrals), the critical line and duality in ball "
                           "arithmetic; finite tori exact; the cylinder sequence is a consistency check"},
                 "Anisotropic square-lattice Ising model (J1 != J2, zero field): critical line, "
                 "free energy (two routes), duality, finite tori, certified", default_route="exact",
                 finality=fin, meta=("phase", "gauge", "torus_size", "torus_high_t_identity",
                                     "cylinder_sequence"),
                 parameters={"exact": ("beta", "J1", "J2", "L")}, rigor={"exact": "certified"})


__all__ = ["ising2d_anisotropic"]
