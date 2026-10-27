"""Percolation on locally tree-like graphs by generating functions: every fixed point of the cavity
equation, certified in arb balls, with its computed stability; the giant component, the mean
finite-cluster size, the threshold and the critical amplitudes.

Equations (Callaway, Newman, Strogatz & Watts, arXiv:cond-mat/0007300 (CNSW), Eqs. (5)-(7), (11),
(12), (14), (15); Newman, Strogatz & Watts, arXiv:cond-mat/0007235 (NSW), Eqs. (26), (27), (34)-(36);
on the Bethe lattice Malthe-Sorenssen, Percolation Theory Using Python (2024), Eqs. (3.4), (3.5)):
  H1(x) = 1 - t + t x G1(H1(x)),   H0(x) = x G0(H1(x))     (bond t; CNSW (5), (7))
  site t: H1 as above, H0(x) = 1 - t + t x G0(H1(x))       (CNSW (5), (6))
  u = H1(1) is the smallest root in [0, 1] of  u = f(u) := 1 - t + t G1(u)      (CNSW (15), NSW (35))
  bond: S = 1 - G0(u);  site: P_inf = t (1 - G0(u))                           (CNSW (14), NSW (34))
  in the form S = 1 - G0(1 - t w): w = (1 - u)/t, the probability that an occupied edge leads to the
  giant, solves w = 1 - G1(1 - t w) (substitute u = 1 - t w; derived here); Erdos-Renyi, G0 = G1 =
  e^(c(x-1)): S = 1 - exp(-c t S) (NSW (37) at t = 1)
Here S_occ = 1 - G0(u) is the giant fraction among occupied vertices in both cases (for site
percolation H0 restricted to occupied vertices is the bond H0 at the same t: derived here), and the
reported P_inf is the fraction of ALL vertices in the giant (bond: S_occ; site: t S_occ).
  mean finite-cluster size of an occupied vertex (derived here by differentiating H0 at x = 1;
  NSW (36) at t = 1; CNSW (11) is t times it below threshold, averaged over all vertices):
     <s> = 1 + G0'(u) H1'(1) / G0(u),   H1'(1) = t G1(u) / (1 - t G1'(u));
  the audit form uses t G1(u) = u - 1 + t (the fixed point): 1 + G0'(u)(u - 1 + t)/(G0(u)(1 - t G1'(u))).
  threshold t_c = 1/G1'(1) (CNSW (12); NSW (32) at t = 1).  Below it <s> = 1 + <k> t/(1 - t G1'(1)), so
  (t_c - t) <s> -> <k> t_c^2 (the amplitude, gamma = 1; derived here).  Above it u = 1 - eps with
  eps = 2 (t G1'(1) - 1)/(t G1''(1)) + O(eps^2), so S_occ ~ B (t - t_c), B = 2 <k> G1'(1)^2 / G1''(1)
  (beta = 1; derived here; the site P_inf slope is t_c B).
All fixed points.  g(u) = f(u) - u is convex on [0, 1] (G1 has nonnegative coefficients), g(1) = 0
and g'(1) = t G1'(1) - 1.  If t G1'(1) < 1, g is decreasing on [0, 1] (g' <= g'(1) < 0): u = 1 is the
only root in [0, 1], stable (f'(1) < 1).  If t G1'(1) > 1, g(0) = 1 - t + t G1(0) >= 0 and g < 0 just
below 1, so convexity gives exactly two roots: u* in [0, 1), with g'(u*) < 0 (stable), and u = 1
(unstable).  At t G1'(1) = 1 the root u = 1 is marginal (a double root): refused ambiguous:critical-
point.  u* is found by Newton from 0 (monotone for a convex g with g(0) >= 0) and certified by interval
Newton (a unique zero in a ball inside [0, 1), or exactly 0 when f(0) = 0); its stability is the
certified ball t G1'(u*) < 1.  Selection: the smallest root (NSW after (34)); a list whose smallest
root is unstable, or that lacks u = 1, is refused no-solution:dropped-root.
Second route of u* (rigorous, no Newton): f is increasing, so l_{n+1} = lower(f(l_n)) from l_0 = 0
and h_{n+1} = upper(f(h_n)) from any h_0 < 1 with g(h_0) < 0 bracket u*; the bracket must overlap the
Newton ball (else no-solution:route-disagreement); one that has not narrowed to 1e-30 in 4000 steps
(near t_c the rate f'(u*) -> 1; when f'(u*) > 1 - 70/4000 one step is taken) is reported unresolved,
never refused.

  percolation_fixed_points(law, dilution) -> (FixedPoint, ...)     ascending u
  percolation_giant(law, dilution) -> GiantResult
_HOOKS["drop_root"] ("trivial" | "nontrivial" | None): a planted dropped root; "root_shift" (0): a
planted offset added to the certified nontrivial root (refused by the <s> audit or the bracket) (tests).
"""
from dataclasses import dataclass
from fractions import Fraction

from ..core import Refusal
from ..lattice.ball import Ball, arb_q, ball_of, baller
from .laws import occupation, parse_law

DIGITS = 40
_PREC0, _MAX_PREC = 192, 1 << 12
_HOOKS = {"drop_root": None, "root_shift": 0}
BRACKET_STEPS, BRACKET_WIDTH = 4000, Fraction(1, 10 ** 30)


@dataclass(frozen=True)
class FixedPoint:
    branch: str          # "trivial" (u = 1) or "nontrivial" (u* < 1)
    u: Ball
    jacobian: Ball       # f'(u) = t G1'(u)
    stable: bool         # jacobian < 1, certified


@dataclass(frozen=True)
class GiantResult:
    law: object
    dilution: str
    t: Fraction
    threshold: Fraction | None        # t_c = 1/G1'(1) (None: G1'(1) = 0, never percolates)
    branching: Fraction               # G1'(1)
    fixed_points: tuple
    selected: FixedPoint
    S_occupied: Ball                  # 1 - G0(u)
    P_inf: Ball                       # bond: S_occupied; site: t S_occupied
    mean_finite: Ball | None          # <s> over occupied vertices in finite clusters
    mean_finite_audit: Ball | None
    chi_amplitude: Ball | None        # <k> t_c^2
    p_inf_slope: Ball | None          # dP_inf/dt at t_c+
    bracket: dict | None
    phase: str


def _t_lambda(law, dilution):
    L = parse_law(law)
    kind, t = occupation(dilution)
    return L, kind, t, t * L.branching()


class _F:
    def __init__(self, L, t):
        self.L, self.T = L, arb_q(t)

    def f(self, x):
        return 1 - self.T + self.T * self.L.G1(x)

    def g(self, x):
        return self.f(x) - x

    def dg(self, x):
        return self.T * self.L.dG1(x) - 1


def _q0(L, t):
    """f(0) = 1 - t + t G1(0) is exactly 0 iff t = 1 and no vertex has degree 1."""
    return L.kind != "poisson" and t == 1 and not L.excess().get(0, 0)


def _nontrivial(L, t, prec):
    """The certified ball of u* at the context precision, or None (undecided at this precision)."""
    from flint import arb
    F = _F(L, t)
    if _q0(L, t):
        X = arb(0)
    else:
        x, tiny = arb(0), arb(2) ** (-4 * prec)
        for _ in range(400):
            xn = (x - F.g(x) / F.dg(x)).mid()
            done = abs(xn - x) <= arb(2) ** (8 - prec) * (abs(xn) + tiny)
            x = xn
            if done:
                break
        r = (abs(x) + arb(2) ** (-2 * prec)) * arb(2) ** (-(prec // 2))
        X = None
        for _ in range(40):
            box = arb(x, r)
            N = x - F.g(arb(x)) / F.dg(box)
            if box.contains_interior(N):
                X = N
                break
            r *= 4
        if X is None or not (X > 0 and X < 1):
            return None
    jac = F.T * L.dG1(X)
    if not jac < 1:
        return None
    return X, jac


def percolation_fixed_points(law, dilution):
    L, kind, t, lam = _t_lambda(law, dilution)
    if lam == 1 and L.kind != "poisson" and max(L.excess()) <= 1:
        raise Refusal("ambiguous:critical-point", f"t G1'(1) = 1 with a linear G1 (every degree is 0 or 2, "
                      f"t = {t}): f(u) = u identically, so every u in [0, 1] is a fixed point; the graphs are "
                      "unions of cycles (Janson & Luczak, arXiv:0707.1786, Remark 2.7: several "
                      "behaviours are possible); not certified")
    if lam == 1:
        raise Refusal("ambiguous:critical-point", f"t G1'(1) = 1 exactly (t = {t}, G1'(1) = "
                      f"{L.branching()}): u = 1 is a double, marginal fixed point (f'(1) = 1); the giant "
                      "is 0 but the mean cluster size diverges; not certified")
    from flint import arb
    with baller().hygiene.ctx_guard(prec=4 * DIGITS + 128):
        triv = FixedPoint("trivial", ball_of(arb(1)), ball_of(arb_q(lam)), lam < 1)
    roots = [triv]
    if lam > 1:
        hyg, prec, got = baller().hygiene, _PREC0, None
        while prec <= _MAX_PREC and got is None:
            with hyg.ctx_guard(prec=prec):
                r = _nontrivial(L, t, prec)
                if r is not None:
                    u, jac = ball_of(r[0] + arb_q(_HOOKS["root_shift"])), ball_of(r[1])
                    if min(u.digits, jac.digits) >= DIGITS:
                        got = FixedPoint("nontrivial", u, jac, True)
            prec *= 2
        if got is None:
            raise Refusal("undetermined:near-critical", f"t = {t}, t G1'(1) = {lam}: the root u* < 1 "
                          f"is not separated from u = 1, or its stability undecided, at {_MAX_PREC} bits")
        roots.insert(0, got)
    drop = _HOOKS["drop_root"]
    if drop:
        roots = [r for r in roots if r.branch != drop]
    return tuple(roots)


def _select(fps):
    if not any(f.branch == "trivial" for f in fps):
        raise Refusal("no-solution:dropped-root", "u = 1 solves u = 1 - t + t G1(u) for every law "
                      "(G1(1) = 1) and is missing from the list of fixed points")
    sel = fps[0]
    if not sel.stable:
        raise Refusal("no-solution:dropped-root", f"the smallest listed fixed point ({sel.branch}, "
                      f"f'(u) = {sel.jacobian}) is unstable: g = f - u is convex with g(0) >= 0, so a "
                      "stable root lies below it and was dropped")
    return sel


def _bracket(L, t, ball, rate=None):
    """The monotone (Galton-Watson) bracket of u*, at 128 bits; see the module docstring.  `rate`
    (the certified f'(u*), an upper bound of the contraction near u*): when 4000 steps cannot narrow
    the bracket to 1e-30 (about 70 / (1 - rate) steps are needed), only h_0 and l_1 are computed."""
    from flint import arb
    with baller().hygiene.ctx_guard(prec=128):
        F = _F(L, t)
        h = None
        steps = BRACKET_STEPS
        if rate is not None and rate.arb.upper() > 1 - arb(70) / BRACKET_STEPS:
            steps = 1
        for j in range(1, 120):
            cand = 1 - arb(2) ** (-j)
            if F.g(cand) < 0:
                h = cand
                break
        lo, n, width = arb(0), 0, None
        if h is not None:
            eps = arb_q(BRACKET_WIDTH)
            for n in range(1, steps + 1):
                lo, h = F.f(lo).lower(), F.f(h).upper()
                if h - lo < eps:
                    break
            width = h - lo
        enc = lo.union(h) if h is not None else None
        out = {"lower": lo.str(12, radius=False), "upper": h.str(12, radius=False) if h is not None
               else "1 (no h_0 < 1 with g(h_0) < 0 found)", "iterations": n,
               "width": width.str(3, radius=False) if width is not None else "n/a",
               "resolved": bool(width is not None and width < eps),
               "kind": "rigorous monotone bracket (f increasing), second route of u*"}
        if enc is not None and not enc.overlaps(ball.arb):
            raise Refusal("no-solution:route-disagreement", f"the monotone bracket [{out['lower']}, "
                          f"{out['upper']}] excludes the interval-Newton root {ball}")
        if h is None and not ball.arb.lower() >= lo:     # only the lower sequence: u* >= lo
            raise Refusal("no-solution:route-disagreement", f"u* {ball} below the lower bound {lo}")
    return out


def percolation_giant(law, dilution):
    L, kind, t, lam = _t_lambda(law, dilution)
    fps = percolation_fixed_points(law, dilution)
    sel = _select(fps)
    br = L.branching()
    tc = 1 / br if br else None
    z = L.mean()
    from flint import arb
    with baller().hygiene.ctx_guard(prec=4 * DIGITS + 128):
        T = arb_q(t)
        for x in (sel.u.arb, arb_q(Fraction(1, 2))):
            a, b = L.g1_audit(x)
            if not a.overlaps(b):
                raise Refusal("no-solution:generating-function-audit", f"G1({x.str(8)}) = {a.str(10)} "
                              f"but G0'/G0'(1) = {b.str(10)}: G0 and G1 are not one law")
        if sel.branch == "trivial":
            s_occ = arb(0)
            mean = ball_of(arb_q(1 + z * t / (1 - lam)))
            audit = mean
        else:
            u = sel.u.arb
            g0 = L.G0(u)
            s_occ = 1 - g0
            if g0 == 0:                         # u = 0 and p_0 = 0: no finite clusters at all
                mean = audit = None
            else:
                den = g0 * (1 - sel.jacobian.arb)
                mean = ball_of(1 + L.dG0(u) * T * L.G1(u) / den)
                audit = ball_of(1 + L.dG0(u) * (u - 1 + T) / den)
                if not mean.overlaps(audit):
                    raise Refusal("no-solution:audit", f"<s> {mean} and its fixed-point form {audit} "
                                  "disagree")
        p_inf = s_occ if kind == "bond" else T * s_occ
        chi = ball_of(arb_q(z * tc * tc)) if tc is not None else None
        g1pp = L.g1pp()
        slope = None
        if tc is not None and g1pp:
            B = 2 * z * br * br / g1pp
            slope = ball_of(arb_q(B if kind == "bond" else tc * B))
        s_ball, p_ball = ball_of(s_occ), ball_of(p_inf)
    bracket = _bracket(L, t, sel.u, sel.jacobian) if sel.branch == "nontrivial" else None
    phase = "below threshold" if lam < 1 else "above threshold"
    return GiantResult(L, kind, t, tc, br, fps, sel, s_ball, p_ball, mean, audit, chi, slope, bracket,
                       phase)


__all__ = ["FixedPoint", "GiantResult", "percolation_fixed_points", "percolation_giant"]
