"""Guerra's replica-symmetric bound on the SK pressure and its minimum over q, certified (E10).

Conventions (families/sk.py: H = -sum_{i<j} J_ij s_i s_j - h sum_i s_i, J_ij ~ N(0, 1/N); pressure
P = -beta f = lim (1/N) E ln Z_N).  The RS bound (Guerra, Comm. Math. Phys. 233, 1 (2003),
arXiv:cond-mat/0205123, Theorem 3 with Eq. (9): K = 2, q_1 = q, m_1 = 0, m_2 = 1; the package's
-beta f_RS of families/sk.py):
  P_RS(q) = ln 2 + beta^2 (1 - q)^2 / 4 + E ln cosh(beta sqrt(q) z + beta h)   >=  (1/N) E ln Z_N
for every q in [0, 1] and every N.  Its q-derivative (Gaussian integration by parts):
  P_RS'(q) = (beta^2/2) G(q),  G(q) = q - E tanh^2(beta sqrt(q) z + beta h),
  G'(q) = 1 - (beta^2/2) E (tanh^2)''(beta sqrt(q) z + beta h),  |(tanh^2)''| <= 2, so |G'| <= 1 + beta^2.
All stationary points of P_RS on [0, 1] (lesson 6):
  h = 0: G(0) = 0, so q = 0 is stationary, with G'(0) = 1 - beta^2.
    beta < 1: G' > 0 is certified on an adaptive cover of [0, 1] (G' enclosed over a cell [a, c] with
      the argument beta * hull(sqrt a, sqrt c), sqrt being monotone; cells bisected until the sign is
      decided, else refused undetermined:stationary-points).  G(0) = 0 and G strictly increasing: no
      other root, P_RS increasing, q = 0 the unique minimiser (at beta = 1/2, 9/10, 99/100: 1, 4, 8
      cells).
    beta > 1: G' < 0 is certified on a cover of [0, a] (a = 1/2, 1/4, ... the first that succeeds), so
      G < 0 on (0, a] and q = 0 is a local maximum; [a, 1] is then treated as for h != 0.
    beta = 1 exactly (G'(0) = 0, no cover can decide the sign at 0): the analytic lemma
      tanh^2 x < x^2 for x != 0, so E tanh^2(sqrt(q) z) < E q z^2 = q and G(q) > 0 on (0, 1]; this one
      case is a proof in the docstring, not a computation (BestBound.certificate says so).  Latala and
      Guerra's lemma (E tanh^2(z sqrt q + h)/q decreasing; Talagrand, Mean field models for spin
      glasses, Vol. I, Sec. 1.3) is not used and not verified here (not fetched).
  h != 0: G(0) = -tanh^2(beta h) < 0 < G(1).  Roots are located by Lipschitz exclusion (a cell
    [a, b] has no root when |G((a+b)/2)| > (1 + beta^2)(b - a)/2, certified), each surviving cluster
    is split into runs on which G' has a certified sign (cells bisected until decided; a tiny cell
    around a zero of G' is dropped only if Lipschitz exclusion shows G has no root on it), and each run
    (G strictly monotone) holds one root when G changes sign across it, none when the end signs agree,
    else refused undetermined:stationary-points.
_HOOKS["dG_shift"] (0): planted-error hook added to every cell enclosure of G' in the covers (tests:
a shift of -1 at beta = 1/2 makes the cover fail and the minimum is refused).
Each root is enclosed by interval Newton (Moore: N = x - G(x)/G'(X) inside X proves a unique zero in
N), the minimum value is P_RS evaluated on the ball N (it contains P_RS(q*)), and P_RS''(q*) =
(beta^2/2) G'(N) is the second variation.  Several local minima are compared by their certified
values (ambiguous:minimiser when the balls overlap).

  guerra_rs_bound(beta, h, q, digits=40) -> Ball;  rs_bound_derivative(beta, h, q) -> Ball
  guerra_rs_best_bound(beta, h=0, digits=40) -> BestBound (a Ball: the minimum over q), with
      .q (Ball, the certified minimiser), .second_variation (Ball), .stable (P'' > 0 certified),
      .stationary (Stationary(q, value, second_variation, kind "min" | "max"), ordered in q),
      .certificate (how the list of stationary points was shown complete); memoised per exact input
  rs_bound_mp(beta, h, q, dps=40), rs_stationary_q_mp(beta, h, dps=40): independent mpmath
      (tanh-sinh, findroot; no arb, no shared code)
"""
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

import mpmath as mp

from ..core import Refusal
from ..lattice.ball import Ball, arb_q, ball_of, baller, certify, rational
from .gauss import expect
from .krsb import check_digits, exact

DIGITS = 40
_HOOKS = {"dG_shift": 0}
_PREC_LO, _MAX_PREC, _CELL = 96, 1 << 13, Fraction(1, 1 << 12)


def _beta(beta):
    return rational(beta, "beta")


def _q(q):
    q = exact(q, "order-parameter")
    if not 0 <= q <= 1:
        raise Refusal("precondition:order-parameter", f"q = {q}: the RS bound needs 0 <= q <= 1")
    return q


def _P(b, hb, Q):
    """P_RS at an arb Q (exact or a ball), b = beta, hb = beta h as arbs."""
    from flint import arb
    a = b * Q.sqrt() if not Q.is_zero() else arb(0)
    return arb(2).log() + b * b * (1 - Q) ** 2 / 4 + expect("lncosh", a, hb)


def _G(b, hb, Q):
    from flint import arb
    a = b * Q.sqrt() if not Q.is_zero() else arb(0)
    return Q - expect("tanh2", a, hb)


def _dG(b, hb, Q):
    from flint import arb
    a = b * Q.sqrt() if not Q.is_zero() else arb(0)
    return 1 - b * b / 2 * expect("d2tanh2", a, hb)


def guerra_rs_bound(beta, h, q, digits=DIGITS) -> Ball:
    b, hq, qq, digits = _beta(beta), exact(h, "h"), _q(q), check_digits(digits)
    return certify(lambda: _P(arb_q(b), arb_q(b) * arb_q(hq), arb_q(qq)), digits)


def rs_bound_derivative(beta, h, q, digits=30) -> Ball:
    """P_RS'(q) = (beta^2/2) G(q), certified (the displayed derivative, latex_bounds.py)."""
    b, hq, qq, digits = _beta(beta), exact(h, "h"), _q(q), check_digits(digits)
    return certify(lambda: arb_q(b) ** 2 / 2 * _G(arb_q(b), arb_q(b) * arb_q(hq), arb_q(qq)), digits)


@dataclass(frozen=True)
class Stationary:
    q: Ball
    value: Ball               # P_RS(q)
    second_variation: Ball    # P_RS''(q) = (beta^2/2) G'(q)
    kind: str                 # "min" | "max"


@dataclass(frozen=True, eq=False)
class BestBound(Ball):
    q: Ball = None
    second_variation: Ball = None
    stable: bool = False
    stationary: tuple = ()
    certificate: str = ""


# ---- locating the stationary points (low precision) ----------------------------------------
def _sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0


def _dG_cell(b, hb, a, c):
    """G' enclosed over every q in [a, c] (Fractions): the argument beta * hull(sqrt a, sqrt c)."""
    from flint import arb
    lo, hi = arb_q(a).sqrt(), arb_q(c).sqrt()
    root = arb(lo.lower()).union(arb(hi.upper()))
    return 1 - b * b / 2 * expect("d2tanh2", b * root, hb) + arb_q(_HOOKS["dG_shift"])


def _cover(b, hb, a, c, sign, depth=12):
    """Cells covering [a, c] with sign(G') == sign certified on each; None when a cell has the
    other sign or stays undecided at `depth` bisections."""
    cells, stack = [], [(a, c, 0)]
    while stack:
        x, y, d = stack.pop()
        s = _sign(_dG_cell(b, hb, x, y))
        if s == sign:
            cells.append((x, y))
            continue
        if s == -sign or d >= depth:
            return None
        m = (x + y) / 2
        stack += [(m, y, d + 1), (x, m, d + 1)]
    return sorted(cells)


def _brackets(b, hb, beta, hq):
    """([(lo, hi)], certificate): each bracket holds exactly one root of G (h = 0: q = 0 excluded,
    it is stationary by G(0) = 0); the certificate says how completeness was shown."""
    if hq == 0:
        if beta == 1:
            return [], ("beta = 1: lemma tanh^2 x < x^2 => G(q) > 0 on (0, 1] (analytic proof, not "
                        "computed)")
        if beta < 1:
            cells = _cover(b, hb, Fraction(0), Fraction(1), +1)
            if cells is None:
                raise Refusal("undetermined:stationary-points", f"beta = {beta}, h = 0: G' > 0 not "
                              "certified on [0, 1]")
            return [], f"G' > 0 certified on {len(cells)} cell(s) covering [0, 1]"
        a = Fraction(1, 2)
        while (cells := _cover(b, hb, Fraction(0), a, -1)) is None:
            a /= 2
            if a < Fraction(1, 1 << 40):
                raise Refusal("undetermined:stationary-points", f"beta = {beta}, h = 0: G' < 0 not "
                              "certified on any [0, a]")
        if _sign(_G(b, hb, arb_q(a))) >= 0:
            raise Refusal("undetermined:stationary-points", f"G({a}) < 0 not certified")
        out = _exclude(b, hb, beta, hq, a)
        return out, (f"G' < 0 certified on {len(cells)} cell(s) covering [0, {a}]; [{a}, 1]: "
                     f"Lipschitz exclusion, {len(out)} root(s) on monotone runs")
    out = _exclude(b, hb, beta, hq, Fraction(0))
    return out, f"Lipschitz exclusion on [0, 1]: {len(out)} root(s) on monotone runs"


def _exclude(b, hb, beta, hq, start):
    """Roots of G on [start, 1] by Lipschitz exclusion; each cluster certified to hold one root."""
    G = lambda q: _G(b, hb, arb_q(q))
    L = 1 + beta * beta
    cells, stack = [], [(start, Fraction(1))]
    while stack:
        a, c = stack.pop()
        g = G((a + c) / 2)
        if abs(g) > arb_q(L * (c - a) / 2):
            continue
        if c - a <= _CELL:
            cells.append((a, c))
        else:
            stack += [((a + c) / 2, c), (a, (a + c) / 2)]
    cells.sort()
    clusters = []
    for a, c in cells:
        if clusters and clusters[-1][1] == a:
            clusters[-1][1] = c
        else:
            clusters.append([a, c])
    out = []
    for a, c in clusters:
        for x, y in _monotone_runs(b, hb, a, c, beta, hq):
            sx, sy = _sign(G(x)), _sign(G(y))
            if sx * sy == -1:
                out.append(_bisect(G, x, y))
            elif not (sx == sy != 0):
                raise Refusal("undetermined:stationary-points", f"beta = {beta}, h = {hq}: no certified "
                              f"root count on [{float(x):.6g}, {float(y):.6g}] (signs of G {sx}, {sy})")
    return out


def _monotone_runs(b, hb, a, c, beta, hq, depth=14):
    """The runs of [a, c] on which G' has one certified sign (G strictly monotone).  A cell where G'
    stays undecided at `depth` (around a zero of G') is dropped when Lipschitz exclusion shows G has
    no root on it, else refused."""
    cells, stack = [], [(a, c, 0)]
    while stack:
        x, y, d = stack.pop()
        s = _sign(_dG_cell(b, hb, x, y))
        if s != 0:
            cells.append((x, y, s))
        elif d >= depth and abs(_G(b, hb, arb_q((x + y) / 2))) > arb_q((1 + beta * beta) * (y - x) / 2):
            cells.append((x, y, 0))                    # G' undecided, but no root of G here
        elif d >= depth:
            raise Refusal("undetermined:stationary-points", f"beta = {beta}, h = {hq}: the sign of G' "
                          f"is undecided on [{float(x):.6g}, {float(y):.6g}]")
        else:
            m = (x + y) / 2
            stack += [(m, y, d + 1), (x, m, d + 1)]
    runs = []
    for x, y, s in sorted(cells):
        if runs and runs[-1][2] == s:
            runs[-1][1] = y
        else:
            runs.append([x, y, s])
    return [(x, y) for x, y, s in runs if s != 0]


def _steps(a, c, d):
    x = a
    while x < c:
        yield x
        x += d


def _hull(lo, hi):
    from flint import arb
    l, h = arb_q(lo), arb_q(hi)
    return (l + h) / 2 + arb(0, ((h - l) / 2).upper())


def _bisect(G, lo, hi, width=Fraction(1, 1 << 30)):
    slo = _sign(G(lo))
    while hi - lo > width:
        mid = (lo + hi) / 2
        s = _sign(G(mid))
        if s == 0:
            break
        lo, hi = (mid, hi) if s == slo else (lo, mid)
    return lo, hi


# ---- certification (escalating precision) ---------------------------------------------------
def _newton(b, hb, lo, hi):
    """Interval-Newton enclosure of the unique root of G in [lo, hi] at the current precision."""
    from flint import arb, ctx
    x = arb_q((lo + hi) / 2)
    for _ in range(60):                         # polish the point (quadratic convergence)
        step = (_G(b, hb, x) / _dG(b, hb, x)).mid()
        x = (x - step).mid()
        if abs(step) < arb(2) ** (-(ctx.prec - 8)):
            break
    r = arb(2) ** (-(ctx.prec // 2))
    for _ in range(40):
        X = arb(x, r)
        N = x - _G(b, hb, x) / _dG(b, hb, X)
        if X.contains_interior(N):
            if not _hull(lo, hi).contains(N):       # N must be the root of THIS bracket
                raise Refusal("no-solution:certification", f"interval Newton left its bracket "
                              f"[{float(lo):.9g}, {float(hi):.9g}]")
            return N
        r *= 4
    raise Refusal("no-solution:certification", f"interval Newton did not contract near {x.mid()}")


def _certified_point(b, hb, Q):
    return ball_of(_P(b, hb, Q)), ball_of(b * b / 2 * _dG(b, hb, Q))


def guerra_rs_best_bound(beta, h=0, digits=DIGITS) -> BestBound:
    return _best_cached(_beta(beta), exact(h, "h"), check_digits(digits), Fraction(_HOOKS["dG_shift"]))


@lru_cache(maxsize=128)
def _best_cached(bq, hq, digits, _shift):
    hyg = baller().hygiene
    with hyg.ctx_guard(prec=_PREC_LO):
        brackets, cert = _brackets(arb_q(bq), arb_q(bq) * arb_q(hq), bq, hq)
    prec, last = max(_PREC_LO, int(3.33 * digits) + 48), None
    while prec <= _MAX_PREC:
        with hyg.ctx_guard(prec=prec):
            from flint import arb
            b, hb = arb_q(bq), arb_q(bq) * arb_q(hq)
            pts = []
            if hq == 0:                           # q = 0: G(0) = 0, P''(0) = (beta^2/2)(1 - beta^2)
                Q = arb(0)
                pts.append((Q, *_certified_point(b, hb, Q)))
            for lo, hi in brackets:
                N = _newton(b, hb, lo, hi)
                pts.append((N, *_certified_point(b, hb, N)))
            stat = [_classify(ball_of(Q), v, s, hq, bq) for Q, v, s in pts]
            worst = min(min(s.value.digits, s.q.digits) for s in stat)
            rad = max(s.value.arb.rad() for s in stat)
        if worst >= digits:
            return _best(stat, bq, hq, cert)
        if last is not None and not rad < last / 2:
            raise Refusal("no-solution:radius-stall", f"RS minimum: radius {rad} at {prec} bits did "
                          f"not halve from {last}")
        last, prec = rad, prec * 2
    raise Refusal("no-solution:certification", f"RS minimum not certified to {digits} digits")


def _classify(Q, value, sv, hq, bq):
    if hq == 0 and Q.arb.is_zero() and bq == 1:
        kind = "min"                           # P''(0) = 0: G > 0 on (0, 1] by the lemma (docstring)
    elif sv.arb > 0:
        kind = "min"
    elif sv.arb < 0:
        kind = "max"
    else:
        raise Refusal("undetermined:stationary-points", f"P'' at q = {Q} has no certified sign")
    return Stationary(Q, value, sv, kind)


def _best(stat, bq, hq, cert):
    mins = [s for s in stat if s.kind == "min"]
    best = mins[0]
    for s in mins[1:]:
        if s.value.arb < best.value.arb:
            best = s
        elif not best.value.arb < s.value.arb:
            raise Refusal("ambiguous:minimiser", f"beta = {bq}, h = {hq}: local minima at "
                          f"{best.q} and {s.q} have overlapping certified values")
    stable = bool(best.second_variation.arb > 0)
    return BestBound(best.value.arb, best.value.digits, best.q, best.second_variation, stable,
                     tuple(stat), cert)


# ---- independent mpmath route ------------------------------------------------------------------
def _mpq(x):
    x = Fraction(x) if not isinstance(x, mp.mpf) else x
    return x if isinstance(x, mp.mpf) else mp.mpf(x.numerator) / x.denominator


def _gauss_mp(f):
    """E f(z) by tanh-sinh on [-Z, 0, Z], Z = sqrt(2 ln 10 (dps + 6)): the dropped Gaussian tail
    (times |f| <= C (1 + |z|)) is below 10^-(dps + 4) of the working precision's digits."""
    c = 1 / mp.sqrt(2 * mp.pi)
    Z = mp.ceil(mp.sqrt(2 * mp.log(10) * (mp.mp.dps + 6)))
    return mp.quad(lambda z: c * mp.exp(-z * z / 2) * f(z), [-Z, 0, Z])


def rs_bound_mp(beta, h, q, dps=40):
    with mp.workdps(dps + 15):
        b, hh, qq = _mpq(exact(beta, "beta")), _mpq(exact(h, "h")), _mpq(q)
        v = mp.log(2) + b * b * (1 - qq) ** 2 / 4 + _gauss_mp(
            lambda z: mp.log(mp.cosh(b * (mp.sqrt(qq) * z + hh))))
    with mp.workdps(dps):
        return +v


def rs_stationary_q_mp(beta, h, dps=40):
    """The positive root of q = E tanh^2(beta (sqrt(q) z + h)) by the secant method from (1/2, 3/5)
    (a starting pair, not a package value)."""
    with mp.workdps(dps + 10):
        b, hh = _mpq(exact(beta, "beta")), _mpq(exact(h, "h"))
        G = lambda q: q - _gauss_mp(lambda z: mp.tanh(b * (mp.sqrt(q) * z + hh)) ** 2)
        r = mp.findroot(G, (mp.mpf(1) / 2, mp.mpf(3) / 5), solver="secant",
                        tol=mp.mpf(10) ** (-(dps + 5)))
    with mp.workdps(dps):
        return +r


__all__ = ["BestBound", "DIGITS", "Stationary", "guerra_rs_best_bound", "guerra_rs_bound",
           "rs_bound_derivative", "rs_bound_mp", "rs_stationary_q_mp"]
