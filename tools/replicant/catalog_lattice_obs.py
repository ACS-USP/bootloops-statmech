"""Catalog steps for the exact observables of `ising1d` and `ising2d` (route `exact`), certified.

Wired into the entries of catalog_lattice.py (ising1d(), ising2d() append STEPS_*, REFS_*, FIN_*,
META_*).  Every physical value is a python-flint arb ball (lattice/), printed as in
catalog_lattice._put: a 40-digit midpoint and in `errors` an upper bound of |value - x| over the ball.
  ising1d: magnetization, susceptibility (two closed forms), correlation (connected, distance r),
           correlation_length; the N-ring's ring_* values, which must overlap closed form + exact
           finite-N correction and lie within the explicit N -> oo bound (else no-solution:ring-limit)
  ising2d: energy (Onsager Eq. (116); -sqrt 2 at beta_c, Eq. (121)), specific_heat off beta_c
           (Eq. (117)); at beta_c no value, the divergence is reported with its amplitude
           specific_heat_amplitude (Eq. (120)) and the log-slope consistency check; the L x Ly
           torus energy and specific heat from Kaufman's formula and from the exact polynomial
           (overlap required, else no-solution:torus-disagreement)
"""
from .catalog import Out, RefCheck, Stability, fstep
from .core import Refusal
from .lattice import BETA_C
from .lattice import observables1d as O1
from .lattice import onsager_thermo as OT
from .lattice import torus_thermo as TT

BAXTER = ("Baxter, Exactly Solved Models in Statistical Mechanics (Academic Press, 1982), Ch. 2 "
          "(read: the ANU PDF)")
ONS44 = "Onsager, Phys. Rev. 65, 117 (1944) (read: journal scan, pp. 140-143)"
KAUF_T = ("Kaufman, Phys. Rev. 76, 1232 (1949) (not fetched), as printed by Izmailian & Hu, Phys. "
          "Rev. E 65 (2002), arXiv:cond-mat/0009024, Eqs. (10)-(16)")
LIT = lambda cite, note: Stability("exact-by-literature", "exact", cite, note)


def _put(**balls):
    vals, errs = {}, {}
    for k, b in balls.items():
        vals[k], errs[k] = b.value_and_error(40)
    return vals, errs


def _int(p, name, default, lo, hi):
    v = p.get(name, default)
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise Refusal(f"precondition:{name}", f"{name} = {v!r}: need an int in {lo} .. {hi}")
    return v


# ---- ising1d ---------------------------------------------------------------------------------
def _obs1d(p):
    beta, h = p.get("beta", 1), p.get("h", 0)
    N = _int(p, "N", 8, 2, 1000)
    r = p.get("r", 1)
    if isinstance(r, bool) or not isinstance(r, int) or not 0 <= r <= max(N - 2, 0):
        raise Refusal("precondition:distance", f"r = {r!r}: need an int in 0 .. {max(N - 2, 0)} "
                      f"(the ring has N = {N} sites)")
    chk = O1.ring_check(N, beta, h, r)
    if not chk.ok:
        bad = [k for k, o in chk.observables.items() if not (o.identity and o.within_bound)]
        raise Refusal("no-solution:ring-limit", f"N = {N}: the ring's {', '.join(bad)} disagree with "
                      "the closed form beyond the exact finite-N correction or the N -> oo bound")
    ob = chk.observables
    v, e = _put(magnetization=O1.ising1d_magnetization(beta, h),
                susceptibility=O1.ising1d_susceptibility(beta, h),
                susceptibility_fluctuation=O1.ising1d_susceptibility_fluctuation(beta, h),
                correlation=O1.ising1d_correlation(beta, h, r),
                correlation_length=O1.ising1d_correlation_length(beta, h),
                ring_magnetization=ob["magnetization"].ring,
                ring_susceptibility=ob["susceptibility"].ring,
                ring_correlation=ob["correlation"].ring,
                ring_correlation_length=ob["correlation_length"].ring)
    bounds = {k: (o.bound.render(3) if o.bound.arb.is_finite() else "inf")
              + ("" if o.limit_tested else " (not tested)") for k, o in ob.items()}
    untested = chk.limit_untested
    return Out({**v, "correlation_r": r, "ring_limit_bounds": bounds,
                "ring_limit_untested": list(untested), "ring_rho": chk.rho.value_and_error(20)[0]},
               f"closed forms (Baxter Ch. 2); ring N = {N}: ring = closed + exact finite-N "
               f"correction, |ring - closed| <= the N -> oo bound (rho^N = "
               f"{(chk.rho.arb ** N).str(3, radius=False)}; the limit check is informative only when "
               "rho^N is small: "
               + (f"not tested for {', '.join(untested)}, whose bound is infinite or not below the "
                  "value's size)" if untested else "every leg tested)"), errors=e)


def _ring_note(what):
    return LIT(f"{BAXTER}, Eqs. (2.1.10), (2.2.7)", f"exact by construction: the N-ring's {what} "
               "(Tr of the transfer matrix, d/dH by exact truncated series) in ball arithmetic")


STEPS_1D = [fstep("ising1d-observables", f"{BAXTER}, Eqs. (2.1.18), (2.2.9), (2.2.14), (2.2.15), "
                  "(1.7.17), (1.7.30)", _obs1d)]
_P1 = {"beta": 1, "h": "1/2"}
REFS_1D = [RefCheck("lattice", f"ising1d_{k}_beta1_h1_2", (lambda k: lambda v: v[k])(k),
                    params=_P1, route="exact")
           for k in ("magnetization", "susceptibility", "correlation", "correlation_length")]
FIN_1D = {
    "magnetization": LIT(f"{BAXTER}, Eq. (2.1.18)", "Baxter's M(H, T), certified ball"),
    "susceptibility": LIT(f"{BAXTER}, Eqs. (1.7.17), (2.1.18)", "dM/dh of Eq. (2.1.18) (the "
                          "derivative taken here), certified ball; equals the fluctuation sum"),
    "susceptibility_fluctuation": LIT(f"{BAXTER}, Eqs. (1.7.30), (2.2.14)", "beta sum_r g(r), the "
                                      "geometric sum done here, certified ball"),
    "correlation": LIT(f"{BAXTER}, Eqs. (2.2.9), (2.2.14)", "sin^2(2 phi) (lambda_2/lambda_1)^r, "
                       "certified ball"),
    "correlation_length": LIT(f"{BAXTER}, Eq. (2.2.15)", "1/ln(lambda_1/lambda_2), certified ball"),
    "ring_magnetization": _ring_note("(1/N) d ln Z_N/dH"),
    "ring_susceptibility": _ring_note("beta (1/N) d^2 ln Z_N/dH^2"),
    "ring_correlation": _ring_note("<s_0 s_r> - m_N^2"),
    "ring_correlation_length": _ring_note("-1/ln[G_N(r + 1)/G_N(r)]"),
}
META_1D = ("correlation_r", "ring_limit_bounds", "ring_limit_untested", "ring_rho")


# ---- ising2d ---------------------------------------------------------------------------------
def _b2(p):
    return p.get("beta", BETA_C)


def _thermo2d(p):
    beta = _b2(p)
    balls = {"energy": OT.onsager_energy(beta), "specific_heat_amplitude": OT.specific_heat_amplitude(),
             "specific_heat_amplitude_loglog": OT.specific_heat_log_slope("1e-6", "1e-8")}
    extra = {}
    if beta == BETA_C:
        extra["specific_heat_divergence"] = ("logarithmic: c = -A ln|beta - beta_c| + O(1) on both "
                                             "sides (Onsager 1944, Eq. (120)); no value at beta_c")
        how = "u = -sqrt 2 (Eq. (121)); c diverges at beta_c (amplitude A, Eq. (120))"
    else:
        balls["specific_heat"] = OT.onsager_specific_heat(beta)
        how = "u (Eq. (116)) and c (Eq. (117)), complete elliptic integrals in arb"
    v, e = _put(**balls)
    return Out({**v, **extra, "loglog_points": "beta - beta_c = 1e-6, 1e-8"}, how + ", certified",
               errors=e)


def _torus_thermo(p):
    L = _int(p, "L", 4, 2, 6)
    Ly = _int(p, "Ly", L, 2, 6)
    beta = _b2(p)
    ku, kc = TT.torus_thermo_kaufman(L, Ly, beta)
    pu, pc = TT.torus_thermo_polynomial(L, Ly, beta)
    mu, mc = TT.torus_thermo_moments(L, Ly, beta)
    if not (ku.overlaps(pu) and kc.overlaps(pc)):
        raise Refusal("no-solution:torus-disagreement", f"{L} x {Ly}: Kaufman's energy / specific "
                      "heat differ from the exact polynomial's")
    # Kaufman and the polynomial share the derivative extraction (torus_thermo._thermo); the
    # energy moments over the integer coefficients do not
    if not (pu.overlaps(mu) and pc.overlaps(mc)):
        raise Refusal("no-solution:torus-disagreement", f"{L} x {Ly}: the K-derivatives of ln Z differ "
                      "from the energy moments <E>/N, K^2 Var(E)/N of the integer polynomial")
    v, e = _put(torus_energy=ku, torus_specific_heat=kc, torus_energy_tm=pu,
                torus_specific_heat_tm=pc)
    return Out(v, f"{L} x {Ly} torus: K-derivatives of Kaufman's Z and of the exact polynomial "
               "agree, and equal the energy moments <E>/N, K^2 Var(E)/N of the integer "
               "coefficients (finite torus, not the bulk)", errors=e)


STEPS_2D = [fstep("ising2d-onsager-thermodynamics", f"{ONS44}, Eqs. (114)-(121)", _thermo2d),
            fstep("ising2d-torus-thermodynamics", KAUF_T, _torus_thermo)]
REFS_2D = [RefCheck("lattice", "onsager_critical_energy", lambda v: v["energy"], route="exact"),
           RefCheck("lattice", "onsager_specific_heat_amplitude",
                    lambda v: v["specific_heat_amplitude"], route="exact")]
_TOR = lambda what: LIT(KAUF_T, f"the finite torus's {what}: K-derivatives of Kaufman's Z (arb_series), "
                        "certified ball")
FIN_2D = {
    "energy": LIT(f"{ONS44}, Eqs. (116), (121)", "Onsager's U/N, certified ball (-sqrt 2 at beta_c)"),
    "specific_heat": LIT(f"{ONS44}, Eq. (117)", "Onsager's C/Nk off beta_c, certified ball"),
    "specific_heat_amplitude": LIT(f"{ONS44}, Eq. (120)", "A = (2/pi) ln^2 cot(pi/8), the "
                                   "coefficient of -ln|beta - beta_c| in c; certified ball"),
    "specific_heat_amplitude_loglog": Stability(
        "not-computed", "exact", f"{ONS44}, Eq. (117)", "[c(beta_c + 1e-8) - c(beta_c + 1e-6)] / "
        "ln 100 from certified values of Eq. (117): a consistency check of A, |slope - A| < 1e-6"),
    "torus_energy": _TOR("energy"), "torus_specific_heat": _TOR("specific heat"),
    "torus_energy_tm": Stability("exact-by-literature", "exact", "Duminil-Copin, arXiv:2208.00864 "
                                 "(2022), Frame 2", "exact by construction: K-derivative of the "
                                 "transfer-matrix integer polynomial"),
    "torus_specific_heat_tm": Stability("exact-by-literature", "exact", "Duminil-Copin, "
                                        "arXiv:2208.00864 (2022), Frame 2", "exact by construction: "
                                        "second K-derivative of the integer polynomial"),
}
META_2D = ("specific_heat_divergence", "loglog_points")

__all__ = ["FIN_1D", "FIN_2D", "META_1D", "META_2D", "REFS_1D", "REFS_2D", "STEPS_1D", "STEPS_2D"]
