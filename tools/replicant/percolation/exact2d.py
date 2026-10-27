"""Exact thresholds, exponents and their identities for 2D lattice percolation (P2), certified in arb
balls or exact rationals.  No Monte Carlo and no enumeration code is used here.

Thresholds (closed forms; the theorems that establish them are cited by the catalog entry):
  square bond 1/2, triangular site 1/2 (exact rationals);
  triangular bond 2 sin(pi/18), honeycomb bond 1 - 2 sin(pi/18).
  Two routes for the triangular bond value: the trigonometric closed form (arb), and the root in
  (0, 1) of the self-dual polynomial kappa_tri(p, p, p) = 3p - p^3 - 1 (Grimmett & Manolescu,
  arXiv:1105.5535, Eq. (1.2)), isolated by bisection on exact rationals (the sign of 3p - p^3 - 1 at
  rational p decided exactly) and enclosed in a ball: the two balls must overlap
  (no-solution:route-disagreement).  The root is unique in (0, 1): 3p - p^3 - 1 is -1 at p = 0, 1 at
  p = 1, and strictly increasing there (derivative 3 - 3p^2 > 0), so isolation is complete.  The
  trigonometric form is derived here: with x = 2 sin t, x^3 - 3x = -2 sin 3t, so t = pi/18 gives -1.
  The honeycomb value: kappa_hex(p) = -kappa_tri(1 - p) (GM Eq. (1.3)), so p_c(hex bond) = 1 - p_c(tri
  bond): the ball 1 - 2 sin(pi/18) must contain a zero of kappa_hex (checked in arb).
Star-triangle (GM Proposition 2.1; the polynomials derived here): a triangle ABC with edge
  probabilities (p0, p1, p2) and a star with arm probabilities 1 - p_i (the arm to a vertex opposite
  the triangle edge of p_i) give the five connection patterns of {A, B, C} with probabilities that are
  polynomials in (p0, p1, p2).  As exact polynomial identities (integer coefficients, dict-of-monomial
  arithmetic): triangle - star = kappa_tri for "all connected", -kappa_tri for "none connected", and 0
  for each of the three pairs; so the laws agree exactly on kappa_tri = 0 and nowhere else
  (no-solution:star-triangle otherwise).  The homogeneous kappa_tri(p, p, p) = 0 is p^3 - 3p + 1 = 0.
Exponents (2D, rational): beta = 5/36, nu = 4/3, gamma = 43/18 (Smirnov & Werner, arXiv:math/0109120,
  Theorem 1), from the arm exponents of their Eqs. (1), (2), a1 = 5/48 and a4 = 5/4, through the
  scaling relations nu = 1/(2 - a4), beta = nu a1, gamma = nu (2 - 2 a1) (Kesten's relations as
  combined in SW Sec. 2; written out here), and the fractal dimension D = 2 - a1 = 91/48 = d - beta/nu
  (hyperscaling); an exponent that breaks a relation is refused no-solution:scaling-relation.
_HOOKS (tests): "star_prob" "dual" | "same" (planted: the star at p instead of 1 - p); "exponent"
  None | (name, Fraction) (planted: a wrong exponent value).
"""
from fractions import Fraction

from ..core import Refusal
from ..lattice.ball import arb_q, ball_of, baller

_HOOKS = {"star_prob": "dual", "exponent": None}
EXACT_RATIONAL = {("square", "bond"): Fraction(1, 2), ("triangular", "site"): Fraction(1, 2)}
CLOSED_FORM = {("square", "bond"), ("triangular", "site"), ("triangular", "bond"), ("honeycomb", "bond")}
A1, A4 = Fraction(5, 48), Fraction(5, 4)


def _ctx(prec=256):
    return baller().hygiene.ctx_guard(prec=prec)


def _arb():
    baller()
    from flint import arb
    return arb


def kappa_tri_hom(p):
    """3p - p^3 - 1 (GM Eq. (1.2) at p0 = p1 = p2 = p); works on Fraction or arb."""
    return 3 * p - p ** 3 - 1


def cubic_root_ball(bits=200):
    """The root of 3p - p^3 - 1 in (0, 1) by exact rational bisection to width 2^-bits, as a ball."""
    lo, hi = Fraction(0), Fraction(1)
    if not kappa_tri_hom(lo) < 0 < kappa_tri_hom(hi):
        raise Refusal("no-solution:route-disagreement", "3p - p^3 - 1 does not change sign on [0, 1]")
    for _ in range(bits):
        mid = (lo + hi) / 2
        if kappa_tri_hom(mid) < 0:          # kappa increases through 0 at the root (lo side < 0)
            lo = mid
        else:
            hi = mid
    with _ctx(4 * bits):
        a, b = arb_q(lo), arb_q(hi)
        return ball_of((a + b) / 2 + (b - a) / 2 * _arb()("0 +/- 1"))


def trig_ball():
    """2 sin(pi/18) in arb."""
    arb = _arb()
    with _ctx():
        return ball_of(2 * (arb.pi() / 18).sin())


def threshold(lattice, dilution):
    """(Ball p_c, {second-route key: Ball}, [identity notes]) or None when no closed form exists."""
    key = (lattice, dilution)
    if key not in CLOSED_FORM:
        return None
    if key in EXACT_RATIONAL:
        with _ctx():
            return ball_of(arb_q(EXACT_RATIONAL[key])), {}, [f"{lattice} {dilution}: 1/2, exact rational"]
    tri, root = trig_ball(), cubic_root_ball()
    if not tri.overlaps(root):
        raise Refusal("no-solution:route-disagreement", f"2 sin(pi/18) = {tri} vs the cubic's root {root}")
    with _ctx():
        if not kappa_tri_hom(tri.arb).contains(0):
            raise Refusal("no-solution:route-disagreement", "3p - p^3 - 1 at 2 sin(pi/18) excludes 0")
        hexb = ball_of(1 - tri.arb)
        hex_root = ball_of(1 - root.arb)
        if not (-kappa_tri_hom(1 - hexb.arb)).contains(0):
            raise Refusal("no-solution:route-disagreement", "kappa_hex at 1 - 2 sin(pi/18) excludes 0")
    notes = ["2 sin(pi/18) overlaps the certified root of 3p - p^3 - 1 in (0, 1) (exact bisection)",
             "kappa_hex(p) = -kappa_tri(1 - p) vanishes on 1 - 2 sin(pi/18) (ball)"]
    if lattice == "triangular":
        return tri, {"p_c_cubic_root": root, "p_c_dual": hexb}, notes
    return hexb, {"p_c_cubic_root": hex_root, "p_c_dual": tri}, notes


# ---- star-triangle: exact polynomials in (p0, p1, p2), {exponent tuple: int} ----------------------
def _mul(a, b):
    out = {}
    for ea, ca in a.items():
        for eb, cb in b.items():
            e = tuple(x + y for x, y in zip(ea, eb))
            out[e] = out.get(e, 0) + ca * cb
    return {e: c for e, c in out.items() if c}


def _add(*ps, sign=None):
    out = {}
    for i, p in enumerate(ps):
        s = 1 if sign is None else sign[i]
        for e, c in p.items():
            out[e] = out.get(e, 0) + s * c
    return {e: c for e, c in out.items() if c}


ONE = {(0, 0, 0): 1}
P = [{tuple(int(i == j) for j in range(3)): 1} for i in range(3)]
Q = [_add(ONE, P[i], sign=(1, -1)) for i in range(3)]          # 1 - p_i
KAPPA = _add(P[0], P[1], P[2], _mul(_mul(P[0], P[1]), P[2]), ONE, sign=(1, 1, 1, -1, -1))


def _prod(fs):
    out = ONE
    for f in fs:
        out = _mul(out, f)
    return out


def _patterns(probs, star):
    """{pattern: polynomial} over the 8 open/closed states of three elements with open-probabilities
    `probs` (polynomials); pattern = frozenset of connected pairs among A, B, C (indices 0, 1, 2)."""
    out = {}
    for s in range(8):
        opened = [(s >> i) & 1 for i in range(3)]
        w = _prod(probs[i] if opened[i] else _add(ONE, probs[i], sign=(1, -1)) for i in range(3))
        if star:                      # arm i joins the centre to vertex i
            ends = [i for i in range(3) if opened[i]]
            pairs = {frozenset((a, b)) for a in ends for b in ends if a < b}
        else:                         # edge i joins the two vertices other than i
            pairs = {frozenset(j for j in range(3) if j != i) for i in range(3) if opened[i]}
            if len(pairs) >= 2:
                pairs = {frozenset((0, 1)), frozenset((0, 2)), frozenset((1, 2))}
        key = frozenset(pairs)
        out[key] = _add(out.get(key, {}), w)
    return out


def star_triangle_identity():
    """[(pattern, triangle - star)] with the star at 1 - p; refused unless every difference is
    kappa_tri, -kappa_tri or 0 (exact integer polynomials)."""
    star_p = Q if _HOOKS["star_prob"] == "dual" else P
    tri, star = _patterns(P, False), _patterns(star_p, True)
    rows = []
    for key in set(tri) | set(star):
        d = _add(tri.get(key, {}), star.get(key, {}), sign=(1, -1))
        n = len(key)
        want = KAPPA if n == 3 else (_add(KAPPA, sign=(-1,)) if n == 0 else {})
        if d != want:
            raise Refusal("no-solution:star-triangle", f"connection pattern {sorted(map(sorted, key))}: "
                          f"triangle - star = {d}, expected {want}")
        rows.append((n, d))
    return rows


# ---- exponents ----------------------------------------------------------------------------------
def exponents():
    """{name: Fraction} and the scaling relations checked exactly (Refusal otherwise)."""
    nu = 1 / (2 - A4)
    ex = {"beta": nu * A1, "nu": nu, "gamma": nu * (2 - 2 * A1), "fractal_dimension": 2 - A1,
          "one_arm": A1, "four_arm": A4}
    if _HOOKS["exponent"] is not None:
        name, val = _HOOKS["exponent"]
        ex[name] = Fraction(val)
    rel = {"nu = 1/(2 - a4)": ex["nu"] * (2 - ex["four_arm"]) == 1,
           "beta = nu a1": ex["beta"] == ex["nu"] * ex["one_arm"],
           "gamma = nu (2 - 2 a1)": ex["gamma"] == ex["nu"] * (2 - 2 * ex["one_arm"]),
           "D = d - beta/nu": ex["fractal_dimension"] == 2 - ex["beta"] / ex["nu"],
           "gamma + 2 beta = d nu (hyperscaling)": ex["gamma"] + 2 * ex["beta"] == 2 * ex["nu"]}
    bad = [k for k, ok in rel.items() if not ok]
    if bad:
        raise Refusal("no-solution:scaling-relation", f"{', '.join(bad)} fails for {ex}")
    return ex, list(rel)


__all__ = ["CLOSED_FORM", "EXACT_RATIONAL", "cubic_root_ball", "exponents", "kappa_tri_hom",
           "star_triangle_identity", "threshold", "trig_ball"]
