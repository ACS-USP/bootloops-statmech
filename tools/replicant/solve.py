"""Numerical solvers: Gaussian expectations and a branch-aware fixed point.

  gauss_expect(f, dps=30)              -> E_{z~N(0,1)} f(z)   (mpf)
  fixed_point(F, x0, ...)              -> FixedPoint, one chosen branch plus
                                          the names/values of all branches seen

gauss_expect integrates f(z) exp(-z^2/2)/sqrt(2 pi) over the real line with
mpmath tanh-sinh quadrature on [-inf, -6, 0, 6, inf].  For smooth f of
moderate growth it agrees with exact values to ~25 digits at dps=30 (tests:
z^2, z^4, cos z, exp(a z) at 1e-25) and with independent trapezoid references
for tanh^2 and log cosh at beta=10 (tests, ~1e-10).  NOT validated for
integrands with kinks or singularities (e.g. erfc-type), nor for f growing
faster than exp(z^2/2 - eps).

fixed_point looks for x = F(x), scalar x.  It runs damped iteration
(x <- (x + F(x))/2) from several seeds (x0, 0, +-x0 scaled), polishes with
secant steps once the residual is small, and merges converged points into
distinct branches.  Only attracting points of the damped map can be found, so
an absent branch means "not found from these seeds", not "does not exist".
A point is accepted only if |F(x) - x| < tol; anything else is not reported.
Branch naming: "trivial" is x ~ 0 when F(0) = 0 (returned as exactly 0);
other points are "nontrivial", "nontrivial-2", ... in discovery order
(seeds tried in the order x0, 0, -x0, x0/10, x0*10, -x0/10, -x0*10, 3*x0).

Near a continuous transition (SK beta -> 1+) damped iteration slows, and the
secant polish only starts once the residual is < 1e-3 within maxit=400 steps;
in practice beta=1.05 and 1.001 are found (tests/spot check), but a missing
branch always means "not found from these seeds".  When `branch` is given the
sweep stops as soon as that branch is found, so `branches_seen` then lists
only the branches found up to that point; with branch=None the sweep is full.
`branch_values` pairs each seen name with its value; `converged` is always
True on return (failures raise Refusal) and is kept for the interface.
"""
from dataclasses import dataclass

import mpmath as mp

from .core import Refusal

_SEED_FACTORS = ("1", "0", "-1", "0.1", "10", "-0.1", "-10", "3")
_TRIVIAL_ZERO = "1e-10"   # converged |x| below this (with F(0)=0) is trivial
_SAME_BRANCH = "1e-8"     # converged points closer than this are one branch


def gauss_expect(f, *, dps: int = 30):
    """E[f(z)], z ~ N(0,1), by tanh-sinh quadrature at `dps` digits."""
    with mp.workdps(dps):
        c = 1 / mp.sqrt(2 * mp.pi)
        return mp.quad(lambda z: f(z) * c * mp.exp(-z * z / 2),
                       [-mp.inf, -6, 0, 6, mp.inf])


@dataclass(frozen=True)
class FixedPoint:
    value: object
    residual: object
    converged: bool
    branch: str
    branches_seen: tuple = ()
    branch_values: tuple = ()   # ((name, value), ...) for every branch seen


def _real(v):
    return v if isinstance(v, mp.mpf) else None


def _solve_from(F, x, tol, maxit):
    """Damped iteration + secant polish from seed x. Returns (x, res) or None."""
    try:
        for _ in range(maxit):
            fx = _real(F(x))
            if fx is None or abs(fx) > 1e30:
                return None
            res = abs(fx - x)
            if res < tol:
                return x, res
            if res < mp.mpf("1e-3"):
                break
            x = (x + fx) / 2
        else:
            return None
        G = lambda t: F(t) - t
        root = mp.findroot(G, (x, x * (1 + mp.mpf("1e-3")) + mp.mpf("1e-6")),
                           solver="secant", tol=tol**2, maxsteps=40, verify=False)
        root = _real(root)
        if root is None:
            return None
        res = abs(F(root) - root)
        return (root, res) if res < tol and abs(root - x) < 1 + abs(x) else None
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None


def fixed_point(F, x0, *, dps: int = 30, tol: str = "1e-20", branch: str | None = None,
                maxit: int = 400) -> FixedPoint:
    with mp.workdps(dps):
        tol_ = mp.mpf(tol)
        x0 = mp.mpf(x0)
        zero_ok = False
        try:
            f0 = _real(F(mp.mpf(0)))
            zero_ok = f0 is not None and abs(f0) < tol_
        except (ValueError, TypeError, ZeroDivisionError, OverflowError):
            pass
        found = []   # [(name, value, residual)]
        if zero_ok:
            found.append(("trivial", mp.mpf(0), abs(f0)))
        n_non = 0
        if branch == "trivial" and zero_ok:
            return FixedPoint(found[0][1], found[0][2], True, "trivial",
                              ("trivial",), (("trivial", found[0][1]),))
        for fac in _SEED_FACTORS:
            seed = x0 * mp.mpf(fac)
            if zero_ok and seed == 0:
                continue
            sol = _solve_from(F, seed, tol_, maxit)
            if sol is None:
                continue
            x, res = sol
            if zero_ok and abs(x) < mp.mpf(_TRIVIAL_ZERO):
                continue    # the trivial branch, already recorded as exactly 0
            if any(abs(x - v) < mp.mpf(_SAME_BRANCH) for _, v, _ in found):
                continue
            n_non += 1
            found.append(("nontrivial" if n_non == 1 else f"nontrivial-{n_non}", x, res))
            if found[-1][0] == branch:
                break
        if not found:
            raise Refusal("no-solution", "no seed converged to |F(x)-x| < tol")
        seen = tuple(n for n, _, _ in found)
        vals = tuple((n, v) for n, v, _ in found)
        if branch is None:
            pick = next((t for t in found if t[0] != "trivial"), found[0])
        else:
            pick = next((t for t in found if t[0] == branch), None)
            if pick is None:
                raise Refusal("no-solution",
                              f"branch {branch!r} not found; seen {list(seen)}")
        return FixedPoint(pick[1], pick[2], True, pick[0], seen, vals)
