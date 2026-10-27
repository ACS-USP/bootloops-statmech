"""Shared pieces of the I3 catalog entries (catalog_lattice_geom.py, catalog_lattice_aniso.py): the
source strings, the sign-pattern gauge, the route-agreement check, and the finite-torus checks (the
high-temperature torus = bulk identity and the cylinder sequence towards the bulk).
"""
from fractions import Fraction

import mpmath as mp

from .catalog import Stability
from .core import Refusal
from .lattice import anisotropic as A
from .lattice import geom_tori as G

BAX = "Baxter, Exactly Solved Models (1982) (ANU PDF, page images read)"
AU = ("Athanasopoulos & Ueltschi, Ann. Henri Poincare (2024), arXiv:2407.05303v2 (read), Theorem "
      "2.1(b) (Houtappel, Physica 16, 425 (1950), not fetched)")
PBM = "Pierre, Bernu & Messio, SciPost Phys. 19, 025 (2025), arXiv:2501.18569v2 (PDF read)"
ST = f"{BAX}, Eqs. (6.4.4)-(6.4.16)"
DUAL = f"{BAX}, Eqs. (6.3.6)-(6.3.8), (6.5.1)-(6.5.7)"
TORI = "exact finite tori (lattice/geom_tori.py; derived here)"
ZUK = ("Zukovic, Eur. Phys. J. B 86, 283 (2013), arXiv:1303.0184 (PDF read): S0/N = 0.32306, citing "
       "Wannier, Phys. Rev. 79, 357 (1950) and Phys. Rev. B 7, 5017 (1973) (neither fetched; the "
       "1950 value is the one corrected in 1973, as reported to us, not verified here)")
LIT = lambda cite, note: Stability("exact-by-literature", "exact", cite, note)
CONS = lambda cite, note: Stability("not-computed", "exact", cite, note)


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


def _raw(p, n=3):
    """The run's couplings as nonzero Fractions (refusals precondition:J*)."""
    defaults = (1, 2) if n == 2 else (1, 1, 1)
    return A.couplings(1, tuple(p.get(f"J{i}", d) for i, d in zip(range(1, n + 1), defaults)), False)


def _frustrated(p):
    """A triangle is frustrated iff J1 J2 J3 < 0; with J1 J2 J3 > 0 flipping alternate rows (or
    columns) changes the sign of two coupling classes, so the model is gauge-equivalent to |J|."""
    a, b, c = _raw(p)
    return a * b * c < 0


def gauge(Js):
    """(|J|, meta note or None): the gauge map of an unfrustrated sign pattern (in the bulk; a
    finite torus is mapped when its sides are even, so the entry's tori are those of |J|)."""
    absJ = tuple(abs(j) for j in Js)
    if absJ == tuple(Js):
        return absJ, None
    return absJ, (f"J = {tuple(str(j) for j in Js)} -> |J| = {tuple(str(j) for j in absJ)}: "
                  "unfrustrated sign pattern, gauge-equivalent by flipping a sublattice of rows or "
                  "columns (values, critical point and tori are those of |J|)")


def _tri_js(p):
    """The couplings the triangular entry computes with: |J| unless frustrated."""
    Js = _raw(p)
    return (tuple(Js), None) if _frustrated(p) else gauge(Js)


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
    balls = [G.cylinder_free_energy(geom, m, Ks) for m in widths]
    vals = [c.mid() for c in balls]
    with mp.workdps(45):
        est, b = vals[-1], bulk.mid()
        # never narrower than the enclosures themselves (the cylinders are enclosed to 1e-12)
        floor = 2 * (balls[-1].rad() + balls[-2].rad()) + bulk.rad()
        band = max(10 * abs(vals[-1] - vals[-2]), floor, mp.mpf("1e-30") * max(1, abs(est)))
        off = abs(est - b)
    out = {"widths": list(widths), "minus_beta_f": [_s(v, 12) for v in vals], "band": _s(band, 2),
           "offset": _s(off, 2), "resolved": band < mp.mpf("1e-4"),
           "kind": "finite-size consistency check (cylinders), no proven bound"}
    if not off <= band:
        raise Refusal("no-solution:torus-limit", f"{geom}: the cylinder sequence ends at "
                      f"{_s(est, 12)}, {_s(off, 2)} from the bulk {_s(b, 12)}, outside its band "
                      f"{_s(band, 2)} (10 x the last change)")
    return out


BETA_HT = Fraction(1, 10 ** 8)


def high_t_identity(geom, L, Js, bulk_fn):
    """The torus against the bulk at beta = 1e-8 (couplings K = 1e-8 J), both certified balls.  Their
    high-temperature expansions agree term by term until the first non-contractible polygon, of
    length >= L on the L x L torus: |torus - bulk| = O(K^L).  Tolerance 10 (6 K)^L, K = max |K_j|
    (at most 6^l polygons of length l per site on these lattices; a counting estimate made here, not
    a proof).  A wrong edge weight w shifts -beta f by about (w^2 - 1) K^2 / 2: 2e-18 for 2 %,
    against 1.3e-28 (L = 4) or 2.2e-21 (L = 3).  Refused no-solution:torus-weight."""
    ks = [BETA_HT * Fraction(j) for j in Js]          # J = 0: the square's empty third class
    tz, b = G.torus_free_energy(geom, L, L, ks), bulk_fn(BETA_HT)
    with mp.workdps(60):
        K = max(abs(mp.mpf(k.numerator) / k.denominator) for k in ks)
        tol = 10 * (6 * K) ** L
        diff = abs(tz.mid(60) - b.mid(60))
    out = {"beta": "1e-8", "torus": f"{L} x {L}", "difference": _s(diff, 3), "tolerance": _s(tol, 3),
           "kind": "certified balls; finite size enters at order K^L"}
    if not diff <= tol + tz.rad() + b.rad():
        raise Refusal("no-solution:torus-weight", f"{geom} {L} x {L} at beta = 1e-8: torus {tz} and "
                      f"bulk {b} differ by {_s(diff, 3)} > {_s(tol, 3)} (finite size is O(K^{L})): "
                      "the finite lattice's Boltzmann weights are not the bulk formula's")
    return out


def _torus(geom, L, Ks, bulk, sizes, Js, bulk_fn):
    """The L x L torus by the transfer matrix, its weights checked against the bulk at high
    temperature (high_t_identity), and the cylinder sequence (circumferences `sizes`).  The
    enumeration of all configurations is the tests' cross-check (tests/test_lattice_geom.py)."""
    tz = G.torus_free_energy(geom, L, L, Ks)
    meta = {"torus_size": f"{L} x {L}", "torus_high_t_identity": high_t_identity(geom, L, Js, bulk_fn),
            "cylinder_sequence": cylinder_sequence(geom, sizes, Ks, bulk)}
    return tz, meta


__all__ = ["AU", "BAX", "BETA_HT", "CONS", "DUAL", "LIT", "PBM", "ST", "TORI", "ZUK", "cylinder_sequence",
           "gauge", "high_t_identity"]
