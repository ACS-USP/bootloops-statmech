"""Catalog entry `series_ising2d` (E8), route `series`: exact series and holonomic recognition.

Model: the square-lattice Ising model, isotropic nearest-neighbour coupling, zero field (no disorder).
Quantity (series/ising2d.py): g(v) = ln Z/N - ln 2 - 2 ln cosh K = sum a_n v^n, v = tanh K, from the
Onsager double integral expanded in exact arithmetic; recognised in u = v^2.  Why a route of its own
(`series`): nothing here is a replica, cavity or AMP calculation; the steps are an exact expansion,
an independent exact finite-lattice route, operator guessing (tools/annihilator) with a 2-prime
confirm and held-out terms, an exact operator identity with the hypergeometric (K(k)) equation, and
the local analysis of the operator at its singular points.  Run parameters:
  order        the v-order of the series (default 200: u-terms 0 .. 100), even, >= 2 fit_terms
  fit_terms    u-coefficients shown to the annihilator (default 70); the rest are held out
  max_order    theta-order bound of the search (default 4)
  max_degree   degree bound (default 10); fit_terms >= (max_degree + 1)(max_order + 1) + 5 + max_degree
  L            torus size of the independent route (default 8; 3 .. 8)
"""
from fractions import Fraction

import mpmath as mp

from .catalog import Out, RefCheck, Stability, entry, fstep
from .core import Refusal
from .series import ising2d as S
from .series import (critical_exponent, elliptic_factor_check, recognize_holonomic, singularities,
                     torus_free_energy_series)
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
          "integral (exact rational coefficients, checked against the exact L x L torus for powers "
          "below L); the operator is guessed by tools/annihilator (2-prime confirm, held-out terms) "
          "and then proved for this series by an exact identity with the hypergeometric equation "
          "of K(k); 'minimal' means within the searched bounds only; the critical point and alpha are "
          "the operator's, final because the literature (Kramers-Wannier, Onsager) establishes them")
_HEAD = tuple(range(4, 21, 2))


def _p(p):
    order, fit = p.get("order", 200), p.get("fit_terms", 70)
    for name, v in (("order", order), ("fit_terms", fit), ("L", p.get("L", 8))):
        if isinstance(v, bool) or not isinstance(v, int):
            raise Refusal(f"precondition:{name}", f"{name}={v!r}: need an int")
    if order < 20 or order % 2:
        raise Refusal("precondition:order", f"order={order}: need an even int >= 20")
    if order // 2 + 1 <= fit:
        raise Refusal("precondition:series-length", f"order={order} gives {order // 2 + 1} "
                      f"u-coefficients, not more than fit_terms={fit}: no held-out terms")
    return order, fit, p.get("max_order", 4), p.get("max_degree", 10), p.get("L", 8)


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
    bulk, tor = S.ising2d_free_energy_series(L), torus_free_energy_series(L, L)
    agree = next((k for k in range(L + 1) if tor[k] != bulk[k]), L + 1) - 1
    if agree != L - 1 or tor[L] - bulk[L] != Fraction(2, L):
        raise Refusal("no-solution:torus-disagreement", f"L = {L}: agreement through v^{agree}, "
                      f"excess at v^{L} {tor[L] - bulk[L]} (expected through v^{L - 1}, 2/{L})")
    return Out({"torus_L": L, "torus_agrees_through": agree, "torus_excess_at_L": str(tor[L] - bulk[L])},
               f"exact {L} x {L} torus: equal to the bulk through v^{agree}; at v^{L} the 2L straight "
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
    return Out({"elliptic_factor_order": chk.factor_order, "elliptic_identity": chk.ok},
               "L = M theta; M (s F) = 0 with F = 2F1(1/2, 1/2; 1; 16 W) and M(R) = P: "
               + ("holds exactly" if chk.ok else "FAILS"))


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
                     "the exact L x L torus below v^L and to Finch's table through v^10")

    def op_label(v):
        if v.get("elliptic_identity"):
            return Stability("stable", "holonomic", f"{ANN}; {HYP}",
                             "guessed operator verified on held-out terms and proved for this series "
                             "by the exact hypergeometric identity (elliptic_identity); its order is "
                             "minimal only within the searched bounds")
        return Stability("not-computed", "holonomic", ANN, "guessed and held-out verified only")
    crit = Stability("exact-by-literature", "holonomic", KW, "the dominant singular point of the "
                     "operator, u_c = 3 - 2 sqrt 2 = (sqrt 2 - 1)^2, v_c = sqrt 2 - 1")
    alpha = Stability("exact-by-literature", "holonomic", LOG, "local exponents of the operator at "
                      "u_c: a logarithm multiplies the exponent-2 solution, (u - u_c)^2 ln|u - u_c| "
                      "in g, i.e. 2 - alpha = 2")
    fin = {**{f"a{n}": coef for n in _HEAD},
           **{k: op_label for k in ("operator_order", "operator_degree", "elliptic_factor_order")},
           "u_c": crit, "v_c": crit, "critical_exponent": alpha, "alpha": alpha}
    meta = ("order", "fit_terms", "max_order", "max_degree", "L", "coefficients", "torus_L",
            "torus_agrees_through", "torus_excess_at_L", "heldout_verified", "operator",
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
