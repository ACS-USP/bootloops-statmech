"""Regular (l, r) LDPC ensembles on the binary erasure channel BEC(eps): density evolution, the
BP threshold and the MAP threshold (Maxwell construction).

Ensemble.  N variable nodes of degree l, M = N l / r check nodes of degree r (configuration
model, inference/ldpc_finite.py); design rate 1 - l/r.  Allowed: integers 2 <= l < r (r > l
so the design rate is positive); the thresholds need l >= 3 (for l = 2 the BP threshold is
the stability limit 1/(r - 1), MMU Fact 1; not implemented here).

Degree polynomials (the convention of Mezard & Montanari 2009, Sec. 15.3 and of Measson,
Montanari & Urbanke 2008, "MMU").  Edge perspective: lambda(x) = sum_i lambda_i x^(i-1) with
lambda_i the fraction of EDGES attached to a variable of degree i, rho(x) likewise for the
checks; regular: lambda(x) = x^(l-1), rho(x) = x^(r-1).  Node perspective Lambda(y) =
int_0^y lambda / int_0^1 lambda = y^l, Gamma(z) = z^r; Lambda'(1) = 1 / int_0^1 lambda = l.
These are a HOOK: DEGREE_PAIRS[name] (default "regular") returns the coefficient maps of
(lambda, rho) for (l, r); every quantity below is computed from them, results cached per
(l, r, pair object), so planting a different pair object never returns a stale value.

Density evolution (MM Eq. (15.33); BP = peeling on the BEC):
    x_{t+1} = eps lambda(y_t),  y_t = 1 - rho(1 - x_t),  x_0 = eps,
x = P(a variable-to-check message is an erasure), y = P(a check-to-variable message is one).
x_t decreases to the largest fixed point in [0, eps]; the bit erasure probability is
eps Lambda(y).  Iterated in binary64 (DE is a 1-d monotone map; tol 1e-15 on the step).
BP threshold (MM Eq. (15.34)):  eps_BP = min over x in (0, 1] of eps(x) = x / lambda(1 - rho(1-x));
the minimum is interior for l >= 3 (MMU Fact 1) and is located here as the root of the
numerator W(x) = lambda(y) - x lambda'(y) rho'(1 - x) of d eps / dx.
MAP threshold (MMU Definition 4, Lemmas 2-3; KRU Lemma 4).  Trial (Bethe) entropy
    P(x) = Lambda'(1) x rho(1-x) - (Lambda'(1)/Gamma'(1)) [1 - Gamma(1-x)] + eps(x) Lambda(y(x)),
P(1) = design rate, P' = Lambda(y) eps'(x) (MMU Lemma 4.4: P is the area under the EBP EXIT
curve).  eps_MAP = eps(x*) at the root x* of P in (x_BP, 1).  Second route (area theorem,
MMU Lemma 2): the x* where int_{x*}^1 Lambda(y(x)) eps'(x) dx equals the design rate, by
quadrature (bec_map_threshold_area); the two agree to 1e-12 when the code is right.

  bec_density_evolution(l, r, eps) -> DEResult(erasure_prob, converged, edge_erasure, ...)
  bec_thresholds(l, r) -> BecThresholds(design_rate, eps_bp, x_bp, eps_map, x_map)
  bec_bp_threshold(l, r), bec_map_threshold(l, r), bec_map_threshold_area(l, r) -> mpf
  trial_entropy(l, r, x), eps_of_x(l, r, x), de_map(l, r, eps, x) -> mpf
Refusals: precondition:degree-pair, :variable-degree-at-least-3, :erasure-probability,
unsupported:degree-pair, no-solution:bracket, no-solution:audit.
Precision: mp.workdps(DPS) inside every call (global precision untouched).
"""
import numbers
from dataclasses import dataclass
from functools import lru_cache

import mpmath as mp

from ..core import Refusal

DPS = 30
_GRID = 200
MAX_ITER = 200000


class RegularPair:
    """lambda(x) = x^(l-1+var_shift), rho(x) = x^(r-1+check_shift).  The shifts exist so a
    test can plant a wrong degree polynomial (e.g. the node perspective x^l, x^r)."""

    def __init__(self, var_shift=0, check_shift=0, name="regular"):
        self.var_shift, self.check_shift, self.name = var_shift, check_shift, name

    def polynomials(self, l, r):
        return {l - 1 + self.var_shift: 1}, {r - 1 + self.check_shift: 1}


DEGREE_PAIRS = {"regular": RegularPair()}


@dataclass(frozen=True)
class DEResult:
    l: int
    r: int
    eps: float
    erasure_prob: float     # bit erasure probability eps Lambda(y) at the DE limit
    converged: bool         # |x_{t+1} - x_t| <= 1e-15 within MAX_ITER sweeps
    edge_erasure: float     # x: variable-to-check message erasure probability
    check_erasure: float    # y = 1 - rho(1 - x)
    iterations: int


@dataclass(frozen=True)
class BecThresholds:
    l: int
    r: int
    design_rate: object
    eps_bp: object
    x_bp: object
    eps_map: object
    x_map: object


class _Poly:
    """A polynomial from {power: coefficient}, with its derivative and the node-perspective
    normalisation (integral from 0 to 1)."""

    def __init__(self, coeffs, num=mp.mpf):
        self.c = {int(k): num(v) for k, v in coeffs.items()}
        self.integral = sum(v / (k + 1) for k, v in self.c.items())

    def __call__(self, x):
        return sum(v * x**k for k, v in self.c.items())

    def d(self, x):
        return sum(k * v * x**(k - 1) for k, v in self.c.items() if k)

    def node(self, y):
        return sum(v * y**(k + 1) / (k + 1) for k, v in self.c.items()) / self.integral


def _integer(v):
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


def _pair(l, r):
    if not (_integer(l) and _integer(r) and 2 <= l < r):
        raise Refusal("precondition:degree-pair",
                      f"(l, r) = ({l!r}, {r!r}): need integers 2 <= l < r (variable degree l "
                      "at least 2, check degree r above l so the design rate 1 - l/r is "
                      "positive)")
    return int(l), int(r)


def _l3(l, r):
    l, r = _pair(l, r)
    if l < 3:
        raise Refusal("precondition:variable-degree-at-least-3",
                      f"l = {l}: for l = 2 eps(x) has no interior minimum (MMU Fact 1); the BP "
                      "threshold is the stability limit 1/(r - 1) and the transition is "
                      "continuous: not implemented here")
    return l, r


def _eps(eps):
    try:
        e = eps if isinstance(eps, mp.mpf) else mp.mpf(str(eps))
    except (TypeError, ValueError):
        e = None
    if isinstance(eps, bool) or eps is None or e is None or not mp.isfinite(e) or not 0 <= e <= 1:
        raise Refusal("precondition:erasure-probability",
                      f"eps = {eps!r}: the erasure probability must be a number in [0, 1]")
    return e


def _get(name):
    if name not in DEGREE_PAIRS:
        raise Refusal("unsupported:degree-pair",
                      f"degree pair {name!r}; implemented: {', '.join(sorted(DEGREE_PAIRS))}")
    return DEGREE_PAIRS[name]


def _polys(pair, l, r, num=mp.mpf):
    lam, rho = pair.polynomials(l, r)
    return _Poly(lam, num), _Poly(rho, num)


def bec_density_evolution(l, r, eps, pair="regular") -> DEResult:
    l, r = _pair(l, r)
    e = float(_eps(eps))
    lam, rho = _polys(_get(pair), l, r, float)
    x, t, conv = e, 0, False
    while t < MAX_ITER:
        nx = e * lam(1 - rho(1 - x))
        t += 1
        if abs(nx - x) <= 1e-15 or nx < 1e-300:
            x, conv = nx, True
            break
        x = nx
    y = 1 - rho(1 - x)
    return DEResult(l, r, e, e * lam.node(y), conv, x, y, t)


def _eps_x(lam, rho, x):
    return x / lam(1 - rho(1 - x))


def _w(lam, rho, x):
    y = 1 - rho(1 - x)
    return lam(y) - x * lam.d(y) * rho.d(1 - x)


def _trial(lam, rho, x):
    y = 1 - rho(1 - x)
    lp1, gp1 = 1 / lam.integral, 1 / rho.integral
    return lp1 * x * rho(1 - x) - lp1 / gp1 * (1 - rho.node(1 - x)) + _eps_x(lam, rho, x) * lam.node(y)


def _refine(f, a, b, what):
    x = mp.findroot(f, (a, b), solver="anderson")
    if not (a <= x <= b and abs(f(x)) < mp.mpf(10)**(8 - DPS)):
        raise Refusal("no-solution:audit", f"{what}: root {x} outside [{a}, {b}] or residual "
                      f"{f(x)}")
    return x


@lru_cache(maxsize=None)
def _thresholds(l, r, pair):
    with mp.workdps(DPS):
        lam, rho = _polys(pair, l, r)
        grid = [mp.mpf(i) / _GRID for i in range(1, _GRID + 1)]
        ev = [_eps_x(lam, rho, x) for x in grid]
        i = min(range(_GRID), key=lambda j: ev[j])
        if i in (0, _GRID - 1):
            raise Refusal("no-solution:bracket", f"eps(x) has no interior minimum on the grid "
                          f"((l, r) = ({l}, {r}))")
        if any(ev[j + 1] >= ev[j] for j in range(i - 1)) or \
                any(ev[j + 1] <= ev[j] for j in range(i + 1, _GRID - 1)):
            raise Refusal("no-solution:audit", f"eps(x) is not unimodal on the grid ((l, r) = "
                          f"({l}, {r}); MMU Fact 1 says it is for regular pairs)")
        xb = _refine(lambda x: _w(lam, rho, x), grid[i - 1], grid[i + 1], "BP tangency")
        eb = _eps_x(lam, rho, xb)
        g2 = [xb + (1 - xb) * mp.mpf(j) / _GRID for j in range(_GRID + 1)]
        pv = [_trial(lam, rho, x) for x in g2]
        ch = [j for j in range(_GRID) if (pv[j] < 0) != (pv[j + 1] < 0)]
        if pv[0] >= 0 or len(ch) != 1:
            raise Refusal("no-solution:bracket", f"trial entropy: P(x_BP) = {mp.nstr(pv[0], 5)}, "
                          f"{len(ch)} sign changes on (x_BP, 1) ((l, r) = ({l}, {r}))")
        j = ch[0]
        xm = _refine(lambda x: _trial(lam, rho, x), g2[j], g2[j + 1], "trial-entropy root")
        rate = 1 - rho.integral / lam.integral
        return BecThresholds(l, r, +rate, +eb, +xb, +_eps_x(lam, rho, xm), +xm)


def bec_thresholds(l, r, pair="regular") -> BecThresholds:
    l, r = _l3(l, r)
    return _thresholds(l, r, _get(pair))


def bec_bp_threshold(l, r, pair="regular"):
    """eps_BP(l, r): the largest eps at which DE has only the trivial fixed point."""
    return bec_thresholds(l, r, pair).eps_bp


def bec_map_threshold(l, r, pair="regular"):
    """eps_MAP(l, r) = eps(x*), P(x*) = 0 on (x_BP, 1) (Maxwell construction)."""
    return bec_thresholds(l, r, pair).eps_map


def bec_map_threshold_area(l, r, pair="regular"):
    """eps_MAP by the area theorem: int_{x*}^1 Lambda(y(x)) eps'(x) dx = design rate, by
    Gauss-Legendre quadrature (independent of the closed form P)."""
    th = bec_thresholds(l, r, pair)
    with mp.workdps(25):
        lam, rho = _polys(_get(pair), th.l, th.r)

        def integrand(x):
            y = 1 - rho(1 - x)
            return lam.node(y) * _w(lam, rho, x) / lam(y)**2

        area = lambda a: mp.quad(integrand, [a, 1], method="gauss-legendre") - th.design_rate
        lo, hi = th.x_bp, 1 - mp.mpf("1e-6")
        if not (area(lo) > 0 > area(hi)):
            raise Refusal("no-solution:bracket", "area under the BP EXIT curve does not cross "
                          "the design rate on (x_BP, 1)")
        x = mp.findroot(area, (lo, hi), solver="anderson")
        return +_eps_x(lam, rho, x)


def _at(l, r, x, pair):
    l, r = _pair(l, r)
    lam, rho = _polys(_get(pair), l, r)
    return lam, rho, (x if isinstance(x, mp.mpf) else mp.mpf(str(x)))


def trial_entropy(l, r, x, pair="regular"):
    with mp.workdps(DPS):
        lam, rho, x = _at(l, r, x, pair)
        return +_trial(lam, rho, x)


def eps_of_x(l, r, x, pair="regular"):
    with mp.workdps(DPS):
        lam, rho, x = _at(l, r, x, pair)
        return +_eps_x(lam, rho, x)


def de_map(l, r, eps, x, pair="regular"):
    """One DE step eps lambda(1 - rho(1 - x))."""
    with mp.workdps(DPS):
        lam, rho, x = _at(l, r, x, pair)
        return +(_eps(eps) * lam(1 - rho(1 - x)))


__all__ = ["BecThresholds", "DEGREE_PAIRS", "DEResult", "DPS", "RegularPair",
           "bec_bp_threshold", "bec_density_evolution", "bec_map_threshold",
           "bec_map_threshold_area", "bec_thresholds", "de_map", "eps_of_x", "trial_entropy"]
