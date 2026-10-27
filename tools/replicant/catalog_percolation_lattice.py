"""Catalog entry `percolation_lattice` (P2): bond or site percolation on the square, triangular and
honeycomb lattices.  Routes:
  exact   (certified)    the closed-form thresholds in arb balls with their exact identities
                         (percolation/exact2d.py: a second route by the self-dual cubic, the
                         star-triangle polynomials, duality), the 2D exponents as exact rationals
                         (scaling relations checked), Cardy's crossing formula by two routes
                         (percolation/cardy.py), and the exact crossing probability of a small open
                         lattice by exhaustive enumeration with its planar duality identity
                         (percolation/enumerate2d.py)
  sample  (statistical)  Newman-Ziff sweeps (percolation/lattice2d.py) over named seeds: crossing
                         probability, largest cluster and mean cluster size at p; with Ls, the
                         finite-size scaling of percolation/fss2d.py (catalog_percolation_fss.py)
Run parameters: lattice (square | triangular | honeycomb, default square), dilution (bond | site,
default bond), p (default 1/2), aspect (width / height of Cardy's rectangle, default 1), Lx, Ly (the
open lattice: exact default 3 x 3, sample default 16 x 16); sample: seeds (default 0 .. 7), sweeps per
seed (default 16), Ls (FSS sizes, default none).
"""
from fractions import Fraction

from .catalog import Out, RefCheck, Stability, entry, fstep, has
from .catalog_lattice_common import _put
from .catalog_percolation_fss import BAND_P, FSS_FINALITY, FSS_META, fss_step
from .core import Refusal
from .lattice.ball import arb_q, ball_of, baller, rational
from .percolation import cardy as CY
from .percolation import enumerate2d as EN
from .percolation import exact2d as X
from .percolation.lattice2d import check_lattice, check_size, sample_lattice
from .spec import ModelSpec

BR_SQ = ("Bollobas & Riordan, 'A short proof of the Harris-Kesten Theorem', arXiv:math/0410359v3 (PDF read), "
         "Theorems 8, 10 (theta(1/2) = 0; P_p(E_oo) = 1 for p > 1/2), Lemma 3, Corollaries 4, 5 (Kesten, Comm. "
         "Math. Phys. 74, 41 (1980), their ref. [16], not fetched)")
BR_TRI = ("Bollobas & Riordan, 'Sharp thresholds and percolation in the plane', arXiv:math/0412510v2 (PDF "
          "read), Sec. 4.3, Theorem 14 (site percolation on the triangular lattice: theta(p) > 0 for p > 1/2, "
          "exponential decay for p < 1/2)")
GM = ("Grimmett & Manolescu, 'Inhomogeneous bond percolation on square, triangular and hexagonal lattices', "
      "Ann. Probab. 41, 2990 (2013) (as printed on the reprint), arXiv:1105.5535v3 (PDF read), Theorem 1.1, "
      "Eqs. (1.1)-(1.3), Proposition 2.1 (proof in Grimmett, Percolation (1999), Sec. 11.9, and Wierman, Adv. "
      "Appl. Probab. 13, 298 (1981): neither fetched)")
SW = ("Smirnov & Werner, 'Critical exponents for two-dimensional percolation', arXiv:math/0109120v2 (PDF "
      "read), Theorem 1 (i)-(iii), Eqs. (1), (2) (site percolation on the triangular lattice)")
CARDY = ("Cardy, 'Critical percolation in finite geometries', arXiv:hep-th/9111026v1 (PDF read; pages 7-9 "
         "rendered), Eq. (8) and p. 8 (the rectangle)")
SMIRNOV = ("Smirnov, 'Critical percolation in the plane', arXiv:0909.4499v1 (PDF read; the 2001 preprint), "
           "Theorem 1, Corollaries 2, 3")
NZ = ("Newman & Ziff, Phys. Rev. Lett. 85, 4104 (2000), arXiv:cond-mat/0005264v2 (PDF read), the algorithm "
      "(pp. 3-5) and Eq. (1)")
CHECK_MAX = 16
ENUM = "exhaustive enumeration of the open Lx x Ly lattice (percolation/enumerate2d.py; derived here)"
THEOREM = {("square", "bond"): BR_SQ, ("triangular", "site"): BR_TRI, ("triangular", "bond"): GM,
           ("honeycomb", "bond"): GM}
CAVEAT = ("exact route, certified: closed-form thresholds (arb balls, second routes and exact identities), "
          "the 2D exponents (exact rationals; theorems for triangular site, universality elsewhere), Cardy's "
          "formula (two routes; a theorem for triangular site, physics-level elsewhere), and the exact "
          "crossing probability of a small open lattice (exhaustive enumeration)")
SAMPLE_CAVEAT = ("sample route, statistical: Newman-Ziff sweeps over named seeds (open boundaries, horizontal "
                 "crossing), canonical values by NZ Eq. (1); means +- stderr, 99% Student-t half-widths in "
                 "the notes; never certified, never final; the finite-size scaling (Ls) is a consistency "
                 "check with a stated band and an assumed systematic, no proven bound")


def _lat(p):
    lattice, dilution = p.get("lattice", "square"), p.get("dilution", "bond")
    check_lattice(lattice, dilution)
    return lattice, dilution


def _occ(p):
    q = rational(p.get("p", "1/2"), "occupation", positive=False)
    if not 0 <= q <= 1:
        raise Refusal("precondition:occupation", f"p = {p.get('p')!r}: need 0 <= p <= 1")
    return q


def _q(q):
    with baller().hygiene.ctx_guard(prec=256):
        return ball_of(arb_q(Fraction(q)))


def _threshold(p):
    lattice, dilution = _lat(p)
    got = X.threshold(lattice, dilution)
    balls, notes = {"p_c": got[0], **got[1]}, list(got[2])
    if lattice in ("triangular", "honeycomb"):
        X.star_triangle_identity()
        notes.append("star-triangle: triangle - star = kappa_tri, -kappa_tri, 0, 0, 0 on the five "
                     "connection patterns (exact integer polynomials)")
    elif lattice == "square":
        q = X.EXACT_RATIONAL[(lattice, dilution)]
        if q + q - 1 != 0:
            raise Refusal("no-solution:route-disagreement", f"kappa_square({q}, {q}) = {2 * q - 1} != 0")
        notes.append("kappa_square(1/2, 1/2) = 1/2 + 1/2 - 1 = 0 (GM Eq. (1.1); exact rationals)")
    else:
        notes.append("self-matching: the Hex identity R_h(p) + R_v(1 - p) = 1 on enumerated rectangles "
                     "(step exact-small-lattice)")
    v, e = _put(**balls)
    return Out({**v, "identities": notes}, f"p_c = {got[0].render(12)}; " + "; ".join(notes), errors=e)


def _exponents(p):
    ex, rels = X.exponents()
    v, e = _put(**{k: _q(ex[k]) for k in ("beta", "nu", "gamma", "fractal_dimension")})
    return Out({**v, "scaling_relations": rels}, f"beta = {ex['beta']}, nu = {ex['nu']}, gamma = {ex['gamma']}, "
               f"D = {ex['fractal_dimension']}; relations hold exactly: {', '.join(rels)}", errors=e)


def _cardy(p):
    a = CY.aspect_ratio(p.get("aspect", 1))
    b, f = CY.cardy_ball(a), CY.cardy_float(a)
    import mpmath as mp
    with mp.workdps(40):
        if not abs(b.mid() - f) <= mp.mpf(10) ** -25 * max(abs(f), mp.mpf(10) ** -30):
            raise Refusal("no-solution:route-disagreement", f"Cardy: arb {b} vs float route {f}")
    with baller().hygiene.ctx_guard(prec=256):
        v, e = _put(cardy_crossing_h=b, cardy_crossing_v=ball_of(1 - b.arb))
    return Out({**v, "aspect": str(a), "cardy_routes": "lambda(2ri) + 2F1 (arb) = elliptic findroot + "
                "incomplete beta (mpmath) to 1e-25"}, f"aspect {a}: horizontal crossing {b.render(12)}", errors=e)


def _exact_lattice(p):
    lattice, dilution = _lat(p)
    W, H = check_size(p.get("Lx", 3), p.get("Ly", p.get("Lx", 3)), hi=64)
    q = _occ(p)
    counts = EN.crossing_counts(lattice, dilution, W, H)
    holds, detail = EN.duality_identity(lattice, dilution, W, H)
    if holds is False:
        raise Refusal("no-solution:duality-identity", detail)
    v, e = _put(crossing_exact=_q(EN.crossing_probability(counts, q)))
    return Out({**v, "exact_lattice": f"{W} x {H}", "crossing_counts": counts, "duality_identity": detail},
               f"{lattice} {dilution} {W} x {H}: R({q}) = {EN.crossing_probability(counts, q)} (exact); {detail}",
               errors=e)


def _sample(p):
    lattice, dilution = _lat(p)
    W, H = p.get("Lx", 16), p.get("Ly", p.get("Lx", 16))
    check_size(W, H)
    cc = None
    elems = len(EN.graph(lattice, W, H)[1]) if dilution == "bond" and W * H <= 64 else W * H
    if elems <= CHECK_MAX:
        cc = lambda runs, bonds: EN.nz_cross_check(lattice, dilution, W, H, runs, bonds)
    r = sample_lattice(lattice, dilution, _occ(p), W, H, p.get("seeds", tuple(range(8))), p.get("sweeps", 16),
                       cc)
    ren = {"R": "crossing_probability_sampled", "P_inf": "P_inf_sampled", "chi": "mean_cluster_size_sampled"}
    vals = {ren[k]: x for k, x in r.mean.items()}
    errs = {ren[k]: x for k, x in r.stderr.items()}
    hw = ", ".join(f"{ren[k]} +- {x:.3g}" for k, x in r.halfwidth99.items())
    meta = {"Lx": W, "Ly": H, "n_seeds": len(r.seeds), "sweeps": r.sweeps,
            "nz_enumeration_check": "every sweep's n*, |C1|(n) and S2(n) equal the enumeration's" if cc else
            f"not run: more than {CHECK_MAX} elements"}
    return Out({**vals, **meta}, f"{len(r.seeds)} x {r.sweeps} NZ sweeps on {lattice} {dilution} {W} x {H}: "
               f"R({r.p:.6g}) = {r.mean['R']:.4g}", errors=errs, statistical=tuple(vals),
               notes=(f"statistical: means over named seeds, stderr in errors; 99% Student-t half-widths: {hw}",))


def _lit_or_universal(theorem_keys, cite_true, note_true, note_false):
    def fin(v):
        if (v.get("lattice"), v.get("dilution")) in theorem_keys:
            return Stability("exact-by-literature", "exact", cite_true, note_true)
        return Stability("not-computed", "exact", cite_true, note_false)
    return fin


def _threshold_fin(v):
    key = (v.get("lattice"), v.get("dilution"))
    return Stability("exact-by-literature", "exact", THEOREM[key], "closed form in an arb ball; second "
                     "route and identities in meta['identities']")


STAT = Stability("not-computed", "sample", NZ, "finite-size statistical estimate over named seeds: NOT final")
TRI_SITE = {("triangular", "site")}
_UNIV = ("universality: physics-level on this lattice (den Nijs, Nienhuis not fetched); proved only for triangular "
         "site (SW)")
_CARDY_NO = "Cardy's CFT prediction: physics-level on this lattice (no proof: GM Sec. 1.1)"
FINALITY = {
    **{k: _threshold_fin for k in ("p_c", "p_c_cubic_root", "p_c_dual")},
    **{k: _lit_or_universal(TRI_SITE, SW, "a theorem for site percolation on the triangular lattice (SW Theorem "
                            "1; the exponent in the o(1)-sense of their statement)", _UNIV)
       for k in ("beta", "nu", "gamma")},
    "fractal_dimension": Stability("not-computed", "exact", SW, "D = 2 - a1 = 91/48 by hyperscaling (derived here) "
                                   "from the one-arm exponent a1 = 5/48 of SW Eq. (1); not a theorem stated there"),
    **{k: _lit_or_universal(TRI_SITE, f"{SMIRNOV}; {CARDY}", "the scaling limit of the crossing probability, a "
                            "theorem for triangular site (Smirnov Cor. 2, 3)", _CARDY_NO)
       for k in ("cardy_crossing_h", "cardy_crossing_v")},
    "crossing_exact": Stability("exact-by-literature", "exact", ENUM, "exact by construction (the I3 "
                                "convention): this package's own exhaustive enumeration of the finite open "
                                "lattice, evaluated at the rational p; no literature value exists"),
    **{k: STAT for k in ("crossing_probability_sampled", "P_inf_sampled", "mean_cluster_size_sampled")},
    **FSS_FINALITY}
META = ("lattice", "dilution", "identities", "scaling_relations", "aspect", "cardy_routes", "exact_lattice",
        "crossing_counts", "duality_identity", "Lx", "Ly", "n_seeds", "sweeps", "nz_enumeration_check") + FSS_META


def _echo(p):
    lattice, dilution = _lat(p)
    return Out({"lattice": lattice, "dilution": dilution}, f"{lattice} lattice, {dilution} percolation")


def _rc(key, extract, params=None, route="exact", statistical=False):
    if statistical:
        return RefCheck("percolation_lattice", key, extract, statistical=True, route="sample", slow=True,
                        params=params, error=lambda e, k=extract.key: e[k], runs=len(params["seeds"]),
                        systematic=lambda v: v["fss_bands"][BAND_P][3])
    return RefCheck("percolation_lattice", key, extract, route=route, params=params or {})


class _V:
    def __init__(self, key):
        self.key = key

    def __call__(self, v):
        return v[self.key]


FSS_SQ_SITE = {"lattice": "square", "dilution": "site", "Ls": (16, 32, 64, 128), "seeds": tuple(range(48)),
               "sweeps": 25, "Lx": 4}


def percolation_lattice():
    has_cf = lambda p: (p.get("lattice", "square"), p.get("dilution", "bond")) in X.CLOSED_FORM
    exact = [fstep("lattice-echo", NZ, _echo),
             fstep("lattice-threshold", f"{GM}; {BR_SQ}; {BR_TRI}", _threshold, when=has_cf,
                   skip="no closed form for this lattice (square site: banked numerically, NZ 2000 and "
                        "Jacobsen 2015; estimated by the sample route's finite-size scaling)"),
             fstep("critical-exponents", SW, _exponents),
             fstep("cardy-crossing", f"{CARDY}; {SMIRNOV}", _cardy),
             fstep("exact-small-lattice", f"{ENUM}; {BR_SQ}", _exact_lattice)]
    sample = [fstep("lattice-echo", NZ, _echo), fstep("newman-ziff-sweeps", NZ, _sample),
              fss_step(has("Ls"), _lat)]
    tri_site, tri_bond = {"lattice": "triangular", "dilution": "site"}, {"lattice": "triangular", "dilution": "bond"}
    refs = [_rc("square_bond_pc", _V("p_c")), _rc("triangular_site_pc", _V("p_c"), tri_site),
            _rc("triangular_bond_pc", _V("p_c"), tri_bond),
            _rc("honeycomb_bond_pc", _V("p_c"), {"lattice": "honeycomb", "dilution": "bond"}),
            _rc("percolation_2d_nu", _V("nu"), tri_site), _rc("percolation_2d_beta", _V("beta"), tri_site),
            _rc("percolation_2d_gamma", _V("gamma"), tri_site),
            _rc("cardy_crossing_square", _V("cardy_crossing_h"), tri_site),
            _rc("square_bond_self_dual_crossing", _V("crossing_exact"), {"Lx": 3, "Ly": 2}),
            _rc("square_site_pc_nz2000", _V("p_c_crossing"), FSS_SQ_SITE, statistical=True),
            _rc("square_site_pc_jacobsen2015", _V("p_c_crossing"), FSS_SQ_SITE, statistical=True)]
    spec = ModelSpec("percolation", "occupation", "none", constraint="2D lattice (square, triangular, honeycomb)")
    common = ("lattice", "dilution", "p", "Lx", "Ly")
    return entry("percolation_lattice", spec, f"{GM}; {SW}; {CARDY}; {NZ}", {"exact": exact, "sample": sample},
                 refs, {"exact": CAVEAT, "sample": SAMPLE_CAVEAT},
                 "bond / site percolation on the square, triangular and honeycomb lattices: exact thresholds, "
                 "exponents, Cardy's formula, exact small lattices; Newman-Ziff sampling and finite-size scaling",
                 default_route="exact", finality=FINALITY, meta=META,
                 parameters={"exact": common + ("aspect",), "sample": common + ("seeds", "sweeps", "Ls")},
                 rigor={"exact": "certified", "sample": "statistical"})


__all__ = ["percolation_lattice"]
