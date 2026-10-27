"""The audits of the k-core selection (percolation/kcore.py): the exclusion scan above the selected
root (no-solution:dropped-root; an unfinished scan undetermined:near-critical) and the rigorous
peeling bracket, the second route of the selected root (no-solution:route-disagreement).
"""
from fractions import Fraction

from ..core import Refusal
from ..lattice.ball import arb_q

MIN_WIDTH = Fraction(1, 10 ** 30)


def no_root_above(M, start, xs):
    """Exclusion scan of [start, 1] (start: the selected root's isolating upper end, where F < x is
    certified, or x0): a box is excluded when F(hi) < lo (F increasing), or when F - x < 0 at both
    ends and F' - 1 excludes 0 on the box; a box that is neither, down to width 1e-30, is refused
    no-solution:dropped-root."""
    from flint import arb
    stack = [(start, arb(1))]
    width = arb_q(MIN_WIDTH)
    n = 0
    while stack:
        lo, hi = stack.pop()
        n += 1
        if M.F(hi).upper() < lo:
            continue
        if (M.F(lo) - lo).upper() < 0 and (M.F(hi) - hi).upper() < 0 and \
                not (M.dF(lo.union(hi)) - 1).contains(0):
            continue                                   # F - x < 0 at both ends and monotone between
        if xs.arb == 1:
            break
        if hi == 1 and M.F1_exact() == 1 and M.dF(lo.union(hi)).upper() < 1 and (M.F(lo) - lo) < 0:
            continue
        if n > 20000:
            raise Refusal("undetermined:near-critical", f"the exclusion scan above the selected root "
                          f"{xs.render(12)} did not finish in 20000 boxes (near [{lo.str(10)}, {hi.str(10)}])")
        if hi - lo < width:
            raise Refusal("no-solution:dropped-root", f"a fixed point of F above the selected root "
                          f"{xs.render(12)} cannot be excluded on [{lo.str(10)}, {hi.str(10)}]: the "
                          "largest root was not selected")
        mid = ((lo + hi) / 2).mid()
        stack += [(lo, mid), (mid, hi)]


def bracket(M, iv_lo, x, rate):
    """The peeling iteration from 1 (upper) and from the isolating lower end (lower)."""
    from flint import arb
    h, lo = arb(1), iv_lo
    steps = 1 if rate.upper() > 1 - arb(70) / 2000 else 2000
    eps = arb_q(Fraction(1, 10 ** 30))
    n = 0
    for n in range(1, steps + 1):
        h, lo = M.F(h).upper(), M.F(lo).lower()
        if h - lo < eps:
            break
    if not lo.union(h).overlaps(x) or not (lo <= x.upper() and h >= x.lower()):
        raise Refusal("no-solution:route-disagreement", f"the peeling bracket [{lo.str(12)}, {h.str(12)}] "
                      f"excludes the interval-Newton root {x.str(15)}")
    return {"lower": lo.str(12, radius=False), "upper": h.str(12, radius=False), "iterations": n,
            "resolved": bool(h - lo < eps), "kind": "peeling iteration from x = 1 (upper) and from "
            "the isolating interval (lower): rigorous monotone bracket, second route of x*"}


__all__ = ["bracket", "no_root_above"]
