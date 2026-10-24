"""Cavity model for the Hopfield model (RS, one condensed pattern), used by dense.py.

Conventions (AGS): H = -(1/2N) sum_ij sum_mu xi_i^mu xi_j^mu s_i s_j (diagonal
p/2 subtracted), p = alpha N random +-1 patterns, pattern 1 condensed with
overlap m, beta = 1/T.  The argument is the cavity method of Mezard, Parisi &
Virasoro (1987), ch. V, applied to sites and to patterns; the equations it
closes on are those of Amit, Gutfreund & Sompolinsky (1985, 1987):

  pattern cavity: take out the term of one uncondensed pattern mu.  In that
    system x = sqrt(N) m_mu is Gaussian, thermal mean a (disorder law N(0, q),
    q = (1/N) sum m_i^2) and thermal variance 1 - q.  Putting the term
    beta x^2/2 back reweights the Gaussian by exp(beta x^2/2): a Gaussian of
    variance (1 - q)/(1 - C) and mean a/(1 - C), C = beta(1 - q), normalisable
    only if 1 - C > 0.  So r = E <x>^2 = q/(1 - C)^2: the factor 1/(1 - C) is
    the Onsager reaction of the pattern on its own overlap.
  site cavity: remove s_0.  Its local field is xi_0^1 m (signal) plus
    sum_{mu>1} xi_0^mu <x_mu>/sqrt(N): Gaussian, variance alpha r (crosstalk).
  add s_0: m_0 = tanh(beta(m + sqrt(alpha r) z)) (xi_0^1 = +1 by gauge);
    the reaction on the site is a constant (s_0^2 = 1) and drops out of m_0.
    Response dm_0/du = beta(1 - m_0^2).
  self-consistency: m = E m_0, q = E m_0^2, r = q/(1 - C)^2.
  stability labels: m-mode 1 - dM/dm, M(m) the reduced cavity map with q, r
    re-closed at each m (= 1 - C at m = 0); paramagnet q-mode = slope of the
    linearised q-map at q = 0, 1 - alpha beta^2/(1 - beta)^2; cavity
    susceptibility lambda = 1 - alpha beta^2 E(1 - m_0^2)^2/(1 - C)^2 (site
    response squared, amplified by the pattern reaction), the AT quantity.
  energy (a separate formula): e = alpha/2 - m^2/2 - (alpha/2) E<x_mu^2>, with
    E<x^2> = r + (1 - q)/(1 - C) from the reweighted Gaussian.
  The cavity route does not produce f here (it would need the site + pattern
    free-energy shifts); HopfieldSolution.f is None.

Planted error (reaction=False): r = q, i.e. crosstalk variance alpha q.
Numerics: float64 (cavity/_quad.py); every returned point is re-evaluated
and must satisfy the three equations to < 1e-12, else Refusal.
"""
import math

import mpmath as mp
import numpy as np

from ..core import Refusal
from ..families.common import CAVITY_CAVEAT, HopfieldSolution
from . import _quad as Q

BRANCHES = ("retrieval", "spinglass", "paramagnet")
_MPV = "Mezard, Parisi & Virasoro, Spin Glass Theory and Beyond (1987)"
_TOL = 1e-12


def _moments(b, m, s):
    """(E t, E t^2, E (1 - t^2)^2), t = tanh(b (m + s z))."""
    if s == 0:
        t = math.tanh(b * m)
        return t, t * t, (1 - t * t)**2
    z, w = Q.nodes(Q.kink_cuts(-m / s, 1 / (b * s)))
    t = np.tanh(b * (m + s * z))
    t2 = t * t
    return (0.0 if m == 0 else float(w @ t)), float(w @ t2), float(w @ (1 - t2)**2)


def energy(m, q, r, a, b):
    m, q, r = float(m), float(q), float(r)
    D = 1 - b * (1 - q)
    return a / 2 - m * m / 2 - a / 2 * (r + (1 - q) / D)


class _Law:
    def __init__(self, a, b, reaction):
        self.a, self.b, self.reaction = a, b, reaction
        self.desc = ("local field xi_0 m + N(0, alpha r); pattern overlaps reweighted by "
                     + ("exp(beta x^2/2): r = q/(1 - beta(1 - q))^2" if reaction
                        else "NOTHING (PLANTED ERROR): r = q"))

    def r_of(self, q):
        if not self.reaction:
            return q
        D = 1 - self.b * (1 - q)
        if not D > 0:
            raise ValueError("non-normalisable reweighting")
        return q / D**2

    def sigma_of(self, q):
        return math.sqrt(self.a * self.r_of(q))


class HopfieldCavity:
    name = "hopfield"
    cites = {"remove_site": _MPV + ", ch. V (site cavity)",
             "cavity_field_law": _MPV + ", ch. V (pattern cavity: reweighted overlaps); "
                                 "AGS, PRL 55, 1530 (1985) (crosstalk noise alpha r)",
             "add_site": _MPV + ", ch. V (site marginal and reaction)",
             "self_consistency": "Amit, Gutfreund & Sompolinsky, Ann. Phys. 173, 30 (1987)"}

    def __init__(self, reaction=True):
        self.reaction = reaction

    def check(self, p):
        br = p.get("branch", "retrieval")
        if br not in BRANCHES:
            raise Refusal("precondition:known-branch",
                          f"branch {br!r} is not one of {', '.join(BRANCHES)} "
                          "(the RS Hopfield branches with one condensed pattern)")
        a, b = mp.mpf(p.get("alpha")), mp.mpf(p.get("beta"))
        if not a > 0:
            raise Refusal("precondition:alpha-positive", f"alpha = p/N must be > 0, got {a}")
        if not b > 0:
            raise Refusal("precondition:beta-positive", f"beta = 1/T must be > 0, got {b}")
        return {"alpha": float(a), "beta": float(b), "branch": br}

    def remove(self, p):
        return {"desc": "removed spin s_0 and, per uncondensed pattern, its own energy term"}

    def field_law(self, p, sys_):
        return _Law(p["alpha"], p["beta"], self.reaction)

    def add(self, p, sys_, law):
        b = p["beta"]
        return {"desc": "m_0 = tanh(beta(m + sqrt(alpha r) z)); response beta(1 - m_0^2)",
                "m0": lambda m, s, z: np.tanh(b * (m + s * z))}

    def close(self, p, sys_, law, site):
        return {"retrieval": _retrieval, "spinglass": _spinglass,
                "paramagnet": _paramagnet}[p["branch"]](p["alpha"], p["beta"], law)

    def summary(self, sol):
        return (f"{sol.branch}: m = {mp.nstr(sol.m, 10)}, q = {mp.nstr(sol.q, 10)}, "
                f"r = {mp.nstr(sol.r, 10)}, stable = {sol.stable}")


# ---- equations ------------------------------------------------------------------------
def _F(a, b, law, v):
    """Residuals (m - E t, sigma - sqrt(alpha r(q))), q = E t^2, at v = (m, sigma)."""
    m, s = v
    t1, t2, _ = _moments(b, m, s)
    return np.array([t1 - m, s - law.sigma_of(t2)])


def _ok(b, law):
    def ok(v):
        if not (0 <= v[0] <= 1 and v[1] > 0):
            return False
        return not law.reaction or 1 - b * (1 - _moments(b, v[0], v[1])[1]) > 0
    return ok


def _mode(a, b, law, m, s):
    """1 - dM/dm = -det J / J22 (J the Jacobian of _F in (m, sigma))."""
    J = Q.jacobian(lambda v: _F(a, b, law, v), [m, s])
    return -(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]) / J[1, 1]


def _labels(a, b, law, m, q, r, mode):
    lam = 1 - a * b * b * _moments(b, m, math.sqrt(a * r))[2] / (1 - b * (1 - q))**2
    return dict(mode_stable=bool(mode > 0), mode_eig=mp.mpf(mode), replicon=mp.mpf(lam),
                stable=bool(mode > 0 and lam > 0))


def _solution(a, b, law, branch, m, s, mode, note):
    q = _moments(b, m, s)[1]
    r = law.r_of(q) if q > 0 else 0.0
    t1, t2, _ = _moments(b, m, math.sqrt(a * r))      # re-evaluated at sigma = sqrt(alpha r)
    res = max(abs(t1 - m), abs(t2 - q))
    if res > _TOL:
        raise Refusal("no-solution:audit", f"cavity point (m, q, r) = {(m, q, r)}: residual {res}")
    lab = _labels(a, b, law, m, q, r, mode)
    lam = lab["replicon"]
    note += (f"; cavity susceptibility lambda = {mp.nstr(lam, 6)}: "
             + ("RS locally stable" if lam > 0 else "RS AT-unstable (RS kept as RS)"))
    return HopfieldSolution(mp.mpf(m), mp.mpf(q), mp.mpf(r), branch, None,
                            residual=mp.mpf(res), note="cavity route: " + note,
                            caveat=CAVITY_CAVEAT, **lab)


def _mattis(b):
    return Q.illinois(lambda m: math.tanh(b * m) - m, 1e-3, 1.0)


def _no_hot(b):
    return (f"no retrieval solution at T = {1 / b:.8g} >= 1: even at alpha -> 0 the cavity "
            "closure m = tanh(beta m) has no root m > 0")


# ---- retrieval: Newton, and the fold alpha_c(T) of the curve alpha(m) --------------------
def _sigma_at_m(b, m):
    """sigma > 0 with E tanh(b(m + sigma z)) = m (m below the Mattis value)."""
    g = lambda s: _moments(b, m, s)[0] - m
    hi = 1.0
    while g(hi) > 0:
        hi *= 2
        if hi > 1e6:
            raise Refusal("no-solution", f"no sigma for m = {m}")
    return Q.illinois(g, 1e-12, hi)


def _alpha_of_m(b, law, m):
    """alpha along the closed curve at overlap m: sigma^2 = alpha r(q) (r/alpha-free)."""
    s = _sigma_at_m(b, m)
    q = _moments(b, m, s)[1]
    try:
        return s * s / law.r_of(q)                      # r(q) does not involve alpha
    except ValueError:
        return None


def _fold(b, law):
    mm = _mattis(b)
    prev = None
    for c in (0.995, 0.98, 0.96, 0.93, 0.88, 0.8, 0.65, 0.5, 0.3):
        al = _alpha_of_m(b, law, mm * c)
        if al is None or (prev is not None and al < prev[1]):
            break
        prev = (mm * c, al)
    if prev is None:
        raise Refusal("no-solution:retrieval-fold", f"no retrieval curve at beta = {b}")
    lo, hi = prev[0] * 0.9, min(prev[0] / 0.9, mm * 0.9999)
    gr = (math.sqrt(5) - 1) / 2
    x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
    f = lambda x: -(_alpha_of_m(b, law, x) or 0.0)
    f1, f2 = f(x1), f(x2)
    for _ in range(60):
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - gr * (hi - lo)
            f1 = f(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + gr * (hi - lo)
            f2 = f(x2)
    mf = (lo + hi) / 2
    return _alpha_of_m(b, law, mf), mf, mm


def _retrieval(a, b, law):
    if b <= 1:
        raise Refusal("no-solution:retrieval", _no_hot(b))
    F, ok = (lambda v: _F(a, b, law, v)), _ok(b, law)
    mm = _mattis(b)
    found = None
    for seed in ((mm, math.sqrt(a)), (0.97 * mm, 1.2 * math.sqrt(a))):
        v = Q.newton2(F, seed, ok)
        if v is not None and v[0] > 1e-8 and _mode(a, b, law, *v) > 0:
            found = v
            break
    if found is None:
        ac, mf, mm = _fold(b, law)
        if a > ac:
            raise Refusal("no-solution:retrieval",
                          f"no retrieval solution: alpha = {a:.10g} above alpha_c(T) = {ac:.10g} "
                          f"at T = {1 / b:.8g} (cavity curve alpha(m) ends in a fold)")
        m = Q.illinois(lambda x: _alpha_of_m(b, law, x) - a, mf, mm * (1 - 1e-12), xtol=1e-14)
        found = Q.newton2(F, (m, _sigma_at_m(b, m)), ok)
        if found is None or not _mode(a, b, law, *found) > 0:
            raise Refusal("no-solution:solver", f"alpha = {a} < alpha_c(T) = {ac} but no "
                          "m-stable cavity retrieval point converged")
    m, s = (float(x) for x in found)
    mode = _mode(a, b, law, m, s)
    return _solution(a, b, law, "retrieval", m, s, mode,
                     f"larger-m retrieval solution, m-mode 1 - dM/dm = {mode:.6g} > 0")


def _spinglass(a, b, law):
    tg = 1 + math.sqrt(a)
    qlo = max(0.0, 1 - 1 / b) if law.reaction else 0.0

    def G(q):
        return _moments(b, 0.0, law.sigma_of(q))[1] - q
    lo, hi = qlo + (1 - qlo) * 1e-12, 1 - 1e-14
    if G(lo) > 0 and G(hi) < 0:
        q = Q.illinois(G, lo, hi)
    elif 1 / b < tg:
        raise Refusal("no-solution:solver", f"T = {1 / b:.8g} < T_g = {tg:.8g} but the cavity "
                      "q-map is not bracketed")
    else:
        raise Refusal("no-solution:spinglass",
                      f"no spin-glass solution (m = 0, q > 0) at T = {1 / b:.8g} >= T_g = "
                      f"1 + sqrt(alpha) = {tg:.8g} (cavity: the linearised q-map slope "
                      "alpha beta^2/(1 - beta)^2 is <= 1): only the paramagnet")
    if q < 1e-13:
        raise Refusal("no-solution:solver", f"q root {q} indistinguishable from q = 0")
    s = law.sigma_of(q)
    mode = _mode(a, b, law, 0.0, s)
    return _solution(a, b, law, "spinglass", 0.0, s, mode,
                     f"m-mode 1 - dM/dm = {mode:.6g} (= 1 - beta(1 - q))")


def _paramagnet(a, b, law):
    if b >= 1 and law.reaction:
        raise Refusal("no-solution:paramagnet",
                      f"at T = {1 / b:.8g} <= 1 the point m = q = r = 0 has no cavity "
                      "solution: 1 - beta(1 - q) = 1 - beta <= 0, so the reweighting "
                      "exp(beta x^2/2) of the pattern overlap x ~ N(0, 1 - q) is not "
                      "normalisable")
    slope = a * b * b / (1 - b)**2
    return _solution(a, b, law, "paramagnet", 0.0, 0.0, 1 - slope,
                     f"q-mode 1 - alpha beta^2/(1 - beta)^2 = {1 - slope:.6g}")
