"""Catalog entry `series_ising2d` (E8), route `series`: exact series and holonomic recognition.

Model: the square-lattice Ising model, isotropic nearest-neighbour coupling, zero field (no disorder).
Quantity (series/ising2d.py): g(v) = ln Z/N - ln 2 - 2 ln cosh K = sum a_n v^n, v = tanh K, from the
Onsager double integral expanded in exact arithmetic; recognised in u = v^2.  Why a route of its own
(`series`): nothing here is a replica, cavity or AMP calculation; the steps are an exact expansion,
an independent exact finite-lattice route, operator guessing (tools/annihilator) with a 2-prime
confirm and held-out terms, an exact operator identity with the hypergeometric (K(k)) equation, and
the local analysis of the operator at its singular points.  Run parameters:
  order        the v-order of the series (default 200: u-terms 0 .. 100), even, >= 2 fit_terms
  fit_terms    u-coefficients shown to the annihilator (default 62); the rest are held out
  max_order    theta-order bound of the search (default 3)
  max_degree   degree bound (default 9); fit_terms >= (max_degree + 1)(max_order + 1) + 5 + max_degree
  L            torus size of the independent route (default 7: a4 .. a12; 3 .. 8)
"""
from fractions import Fraction

import mpmath as mp

from .catalog import Out, RefCheck, Stability, entry, fstep
from .core import Refusal
from .series import ising2d as S
from .series import (critical_exponent, elliptic_factor_check, recognize_holonomic, singularities,
                     theta_g_identity, torus_free_energy_series)
from .spec import ModelSpec

ONSAGER = ("Onsager, Phys. Rev. 65, 117 (1944) (not fetched), the free energy as printed by "
           "Duminil-Copin, arXiv:2208.00864 (2022), Eq. (3.6)")
KW = ("Kramers & Wannier, Phys. Rev. 60, 252 (1941) and Onsager (1944) (not fetched), via "
      "Duminil-Copin, arXiv:2208.00864 (2022), Sec. 3.1, Eq. (3.3)")
LOG = ("Onsager (1944) (not fetched): logarithmic specific heat, via Duminil-Copin, arXiv:2208.00864 "
       "(2022), Sec. 3.2, and Baxter, arXiv:1010.0710 (2010), Sec. 5.6")
FINCH = "Finch, arXiv:math/9810155v3 (1999), Sec. 4 (Harris & Meir 1987 via Finch)"
ANN = "tools/annihilator (pf_from_series mode='modp' at 2 primes, mode='auto'; rec_to_theta)"
HYP = "DLMF 15.10.1 (hypergeometric equation) and 19.5.1 (K(k) = (pi/2) 2F1(1/2, 1/2; 1; k^2))"
CAVEAT = ("series route: the high-temperature series is the exact expansion of the cited Onsager "
          "integral (exact rational coefficients, checked against the exact L x L torus below v^L and, "
          "for odd L, at even powers through v^(2L - 2): a4 .. a12 at the default L = 7); the operator "
          "is guessed by tools/annihilator (2-prime confirm, held-out terms) and then proved for this "
          "series by an exact identity with the hypergeometric equation of K(k), so its values are "
          "not-computed (proved, not a stability); 'minimal' means within the searched bounds only; "
          "no finite lattice here constrains the operator; the critical point and alpha are the "
          "operator's, final because the literature (Kramers-Wannier, Onsager) establishes them")
_HEAD = tuple(range(4, 21, 2))


def _p(p):
    order, fit = p.get("order", 200), p.get("fit_terms", 62)
    for name, v in (("order", order), ("fit_terms", fit), ("L", p.get("L", 7))):
        if isinstance(v, bool) or not isinstance(v, int):
            raise Refusal(f"precondition:{name}", f"{name}={v!r}: need an int")
    if order < 20 or order % 2:
        raise Refusal("precondition:order", f"order={order}: need an even int >= 20")
    if order // 2 + 1 <= fit:
        raise Refusal("precondition:series-length", f"order={order} gives {order // 2 + 1} "
                      f"u-coefficients, not more than fit_terms={fit}: no held-out terms")
    return order, fit, p.get("max_order", 3), p.get("max_degree", 9), p.get("L", 7)


def _mp(x):
    with mp.workdps(30):
        return mp.mpf(x.numerator) / x.denominator


def _state(p):
    order, fit, so, sd, L = _p(p)
    return Out({"order": order, "fit_terms": fit, "max_order": so, "max_degree": sd, "L": L},
               "square lattice, zero field; v = tanh K, g = ln Z/N - ln 2 - 2 ln cosh K")


def _expansion(p):
    a = S.ising2d_free_energy_series(_p(p)[0])
    return Out({**{f"a{n}": _mp(a[n]) for n in _HEAD},
                "coefficients": [str(x) for x in a[::2]]},
               f"exact rational a_n, n <= {len(a) - 1} (trigonometric moments integrated exactly)")


def _torus(p):
    L = _p(p)[4]
    top = 2 * L if L % 2 else L
    bulk, tor = S.ising2d_free_energy_series(top), torus_free_energy_series(L, top)
    agree = next((k for k in range(top + 1) if tor[k] != bulk[k]), top + 1) - 1
    even = next((k for k in range(0, top + 1, 2) if tor[k] != bulk[k]), top + 2) - 2
    want = S.torus_window(L)
    if (agree, even) != want or tor[L] - bulk[L] != Fraction(2, L) or tor[top] == bulk[top]:
        raise Refusal("no-solution:torus-disagreement", f"L = {L}: agreement through v^{agree} (even "
                      f"k through v^{even}), excess at v^{L} {tor[L] - bulk[L]}; expected {want}, 2/{L}, "
                      f"and a difference at v^{top}")
    return Out({"torus_L": L, "torus_agrees_through": agree, "torus_even_agrees_through": even,
                "torus_excess_at_L": str(tor[L] - bulk[L])},
               f"exact {L} x {L} torus: equal to the bulk through v^{agree} and, for even powers, "
               f"through v^{even} (first even difference at v^{even + 2}); at v^{L} the 2L straight "
               f"winding loops add 2/{L}")


def _rec(p):
    order, fit, so, sd, _ = _p(p)
    b = S.ising2d_free_energy_series(order // 2, variable="u")
    return recognize_holonomic(b, max_order=so, max_degree=sd, fit_terms=fit)


def _recognition(p):
    rec = _rec(p)
    op = rec.operator
    return Out({"operator_order": op.order, "operator_degree": op.degree,
                "heldout_verified": rec.verified_terms,
                "operator": {f"u^{i} theta^{k}": str(c) for (i, k), c in sorted(op.terms.items())},
                "boundary_polynomial": [str(c) for c in op.rhs]},
               f"theta-order {op.order}, degree {op.degree} (none of order < {op.order} within the "
               f"bounds); fitted on u^0 .. u^{rec.fit_terms - 1}, {rec.verified_terms} held-out "
               "terms verified exactly")


def _elliptic(p):
    chk = elliptic_factor_check(_rec(p).operator)
    tg = theta_g_identity(40)
    return Out({"elliptic_factor_order": chk.factor_order, "elliptic_identity": chk.ok,
                "theta_g_identity": tg},
               "L = M theta; M (s F) = 0 with F = 2F1(1/2, 1/2; 1; 16 W) and M(R) = P: "
               + ("holds exactly" if chk.ok else "FAILS") + "; theta g = R + s F through u^40: "
               + ("exact" if tg else "FAILS"))


def _singular(p):
    op = _rec(p).operator
    sing = singularities(op)
    real = [s for s in sing if s.kind == "singular" and s.point is not None and s.point.d > 0
            and s.value > 0]
    if not real:
        raise Refusal("no-solution:critical-point", "no positive real singular point")
    crit = min(real, key=lambda s: s.value)
    rho = critical_exponent(op, crit.point)
    with mp.workdps(30):
        return Out({"u_c": crit.value, "v_c": mp.sqrt(crit.value), "critical_exponent": rho,
                    "alpha": 2 - rho, "u_c_exact": repr(crit.point),
                    "log_at_critical": rho in crit.exponents.log_exponents,
                    "singular_points": [f"{s.point!r}: {s.kind}, exponents "
                                        f"{s.exponents.integer_exponents if s.exponents else '-'}"
                                        for s in sing]},
                   f"dominant singular point u_c = {crit.point!r}; local exponents "
                   f"{crit.exponents.integer_exponents}, logarithm at {crit.exponents.log_exponents}")


def series_ising2d():
    steps = [fstep("series-state-point", ONSAGER, _state),
             fstep("series-onsager-expansion", ONSAGER, _expansion),
             fstep("series-torus-route", "exact L x L torus, parity transfer matrix (series/ising2d.py)",
                   _torus),
             fstep("series-recognition", ANN, _recognition),
             fstep("series-elliptic-identity", HYP, _elliptic),
             fstep("series-singularities", f"{KW}; {LOG}", _singular)]
    refs = [RefCheck("series", f"a{n}", lambda v, n=n: v[f"a{n}"], route="series") for n in (4, 6, 8, 10)]
    refs += [RefCheck("series", "v_c", lambda v: v["v_c"], route="series"),
             RefCheck("series", "alpha", lambda v: v["alpha"], route="series")]
    coef = Stability("exact-by-literature", "exact series", f"{ONSAGER}; {FINCH}",
                     "exact rational expansion of the cited integral (no truncation error); equal to "
                     "the exact 7 x 7 torus through v^12 and to Finch's table through v^10")
    coef_hi = Stability("exact-by-literature", "exact series", ONSAGER,
                        "exact rational expansion of the cited integral (no truncation error); a12 also "
                        "equals the exact 7 x 7 torus; beyond v^12 no finite-lattice check here")

    def op_label(v):
        if v.get("elliptic_identity") and v.get("theta_g_identity"):
            return Stability("not-computed", "holonomic", f"{ANN}; {HYP}",
                             f"L g = P proved exactly (DLMF 15.10.1 + theta g = R + s F); minimality only "
                             f"within the searched bounds (degree <= {v['max_degree']}, order <= "
                             f"{v['max_order']}); held-out terms verified")
        return Stability("not-computed", "holonomic", ANN, "guessed and held-out verified only; the "
                         "exact identity did not hold")
    crit = Stability("exact-by-literature", "holonomic", KW, "the dominant singular point of the "
                     "operator, u_c = 3 - 2 sqrt 2 = (sqrt 2 - 1)^2, v_c = sqrt 2 - 1")
    alpha = Stability("exact-by-literature", "holonomic", LOG, "local exponents of the operator at "
                      "u_c: a logarithm multiplies the exponent-2 solution, (u - u_c)^2 ln|u - u_c| "
                      "in g, i.e. 2 - alpha = 2")
    fin = {**{f"a{n}": (coef if n <= 10 else coef_hi) for n in _HEAD},
           **{k: op_label for k in ("operator_order", "operator_degree", "elliptic_factor_order")},
           "u_c": crit, "v_c": crit, "critical_exponent": alpha, "alpha": alpha}
    meta = ("order", "fit_terms", "max_order", "max_degree", "L", "coefficients", "torus_L",
            "torus_agrees_through", "torus_even_agrees_through", "torus_excess_at_L", "theta_g_identity", "heldout_verified", "operator",
            "boundary_polynomial", "elliptic_identity", "u_c_exact", "log_at_critical", "singular_points")
    spec = ModelSpec("ising2d", "ising", "none", constraint="square lattice, nearest neighbour, zero field")
    return entry("series_ising2d", spec, ONSAGER, {"series": steps}, refs, {"series": CAVEAT},
                 "2D Ising (Onsager) free energy: exact high-temperature series, its holonomic "
                 "operator (order 3 = elliptic order 2 times theta), critical point and log exponent",
                 default_route="series", finality=fin, meta=meta,
                 parameters={"series": ("order", "fit_terms", "max_order", "max_degree", "L")})


def series_entries():
    return [series_ising2d()]


__all__ = ["series_entries", "series_ising2d"]
