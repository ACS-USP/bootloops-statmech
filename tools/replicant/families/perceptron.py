"""Family (b): Gardner volume of the perceptron, spherical and Ising couplings.

Convention.  P = alpha N patterns xi^mu with i.i.d. N(0, 1) components (+-1
patterns give the same N -> infinity limit), labels absorbed into the
patterns; J stores them if J.xi^mu / sqrt(N) >= kappa for all mu.
spherical: J.J = N (Gardner, J. Phys. A 21, 257 (1988));  Ising: J in
{-1, +1}^N (Gardner & Derrida, J. Phys. A 21, 271 (1988); Krauth & Mezard,
J. Phys. France 50, 3057 (1989)).  H(x) = int_x^inf Dz, phi = N(0,1) density,
E(x) = phi(x)/H(x).  Entropy per coupling: spherical S = (1/N) ln(fraction
of the sphere), Ising S = (1/N) ln #solutions.

RS functional (n -> 0), with x(t) = (kappa - sqrt(q) t)/sqrt(1-q):
  spherical  S(q)       = 1/2 ln(1-q) + q/(2(1-q)) + alpha E_t ln H(x)
  Ising      S(q, qhat) = -qhat(1-q)/2 + E_z ln 2cosh(sqrt(qhat) z) + alpha E_t ln H(x)
(the Ising form is Ding & Sun, arXiv:1809.07742, Eq. (4) with psi = qhat).
Stationarity uses d/dq E_t ln H(x) = -E_t[E(x)^2] / (2(1-q)) (heat-equation
identity: H(x) is a Gaussian smoothing of a step, of variance 1-q):
  spherical  q/(1-q) = alpha E_t E(x)^2                       (Gardner 1988)
  Ising      q = E_z tanh^2(sqrt(qhat) z),  qhat = alpha E_t E(x)^2 / (1-q).
Both are explicit in the order parameter: alpha(q) (spherical) and
alpha(qhat) (Ising, q from qhat), which is how gardner_rs_point
parametrises the saddle; gardner_entropy inverts alpha(.) by a bracketed root.

Capacity (q -> 1).  E(x)^2 -> x^2 for x -> +inf, so E_t E(x)^2 ->
D(kappa)/(1-q) with D(kappa) = int_{-kappa}^inf Dt (t+kappa)^2
= (1+kappa^2) H(-kappa) + kappa phi(kappa).  Spherical: the saddle gives
alpha(q) -> 1/D, and the divergent 1/(2(1-q)) terms of S cancel there:
alpha_c = 1/D (Gardner; = 2 at kappa = 0).  Near alpha_c, 1-q ~ 1 - alpha/alpha_c
and, adding the -ln x tail of ln H on the violated fraction H(-kappa),
S ~ c ln(1 - alpha/alpha_c) with c = (1 + alpha_c H(-kappa))/2 (= 1 at kappa = 0):
the version space shrinks to a point and S -> -inf (what 'entropy vanishes'
means for a continuous volume).
Ising: qhat (1-q)^2 -> alpha D and, from 1-q = E sech^2(sqrt(qhat) z) ~
(int sech^2)/sqrt(2 pi qhat) = 2/sqrt(2 pi qhat), qhat (1-q)^2 -> 2/pi:
the RS endpoint is alpha = (2/pi)/D = 4/pi at kappa = 0 (Gardner-Derrida).

Stability.  Replicon of the RS point (de Almeida-Thouless form; the same
structure as Buhot et al. arXiv:cond-mat/9703183 Eq. (9)):
  spherical  lambda = 1 - alpha E_t[(ln H)''(x)^2]
  Ising      lambda = 1 - alpha E_z[sech^4(sqrt(qhat) z)] E_t[(ln H)''(x)^2]/(1-q)^2
with (ln H)''(x) = -E(x)(E(x)-x).  Spherical at q -> 1: lambda -> 1 -
alpha_c H(-kappa) = kappa (kappa H(-kappa) + phi(kappa)) alpha_c: zero at
kappa = 0 (marginal), positive for kappa > 0 (convex version space, RS
exact), negative for kappa < 0 (RS breaks; Franz & Parisi, arXiv:1501.03397).
Ising at q -> 1: lambda ~ -alpha/(3(1-q)) -> -inf; replicant finds lambda = 0
near alpha = 1.015 at kappa = 0 (Gyorgyi, arXiv:cond-mat/0010025, writes that
RS stays AT-stable up to 4/pi: we do not reproduce that statement).

1RSB (Ising, frozen).  q1 = 1 within blocks of m replicas: each block is one
configuration, so (1/n) ln E Z^n = S_RS(q0, qhat0)/m with the RS functional
taken for the n/m distinct blocks (Z = #solutions).  d/dm (S_RS/m) = 0 has a
solution only where S_RS = 0: the frozen 1RSB keeps S = S_RS while it is >= 0
and puts the capacity at the root of the RS entropy (Krauth & Mezard 1989),
where the RS point must still be AT-stable (checked, refused otherwise).

Honest label: replica n -> 0, RS and the frozen 1RSB are physics-level, not
proven here (the spherical kappa >= 0 threshold is a theorem, Shcherbina-
Tirozzi 2003, as quoted by Ding-Sun; not used).  Numerics: roots are
bracketed in float64 (_perceptron_float), then polished by secant and
audited in mpmath tanh-sinh at 15 digits, or the caller's precision if higher.
"""
import math
from dataclasses import dataclass
from functools import lru_cache

import mpmath as mp
import sympy as sp

from ..core import Refusal
from ..gauss import GaussE
from . import _perceptron_float as F
from .common import CAVEAT, Capacity  # noqa: F401 (re-exported)

_DPS = 15
_VARIABLES = ("spherical", "ising")


def _prec():   # never lower the caller's precision
    return mp.workdps(max(_DPS, mp.mp.dps))


@dataclass(frozen=True)
class RSPoint:
    """One RS saddle: alpha(order), q, qhat (None for spherical), S, replicon."""
    alpha: object
    q: object
    qhat: object
    S: object
    replicon: object
    variables: str
    kappa: object
    caveat: str = CAVEAT


# ---- Gaussian helpers (mpmath) -----------------------------------------------
def _phi(x):
    return mp.exp(-x * x / 2) / mp.sqrt(2 * mp.pi)


def _H(x):
    return mp.erfc(x / mp.sqrt(2)) / 2


def _E(x):
    return _phi(x) / _H(x)


def _gauss(f, pts):
    """E_t f(t), t ~ N(0,1), tanh-sinh on the given breakpoints (those with
    |p| < 10 only: further out the Gaussian weight is < 1e-21)."""
    pts = sorted({p for p in pts if abs(p) < 10} | {-mp.inf, mp.inf})
    return mp.quad(lambda t: f(t) * _phi(t), pts)


def _den(kappa):
    """D(kappa) = int_{-kappa}^inf Dt (t + kappa)^2, in closed form."""
    return (1 + kappa**2) * _H(-kappa) + kappa * _phi(kappa)


def _check(variables, kappa=0):
    if variables not in _VARIABLES:
        raise Refusal("precondition:known-variables",
                      f"variables = {variables!r}: use 'spherical' (J.J = N) or 'ising' (J = +-1)")
    k = mp.mpf(kappa)
    if not mp.isfinite(k):
        raise Refusal("precondition:finite-kappa", f"kappa = {kappa}")
    return k


class _T:
    """t-integrals at overlap q (x = (kappa - sqrt(q) t)/sqrt(1-q))."""

    def __init__(self, q, one_m_q, kappa):
        self.s, self.r, self.k = mp.sqrt(q), mp.sqrt(one_m_q), kappa
        t0, w = kappa / self.s, self.r / self.s
        self.pts = [t0 - 30 * w, t0 - 3 * w, t0, t0 + 3 * w, t0 + 30 * w, 0]

    def x(self, t):
        return (self.k - self.s * t) / self.r

    def eh2(self):
        return _gauss(lambda t: _E(self.x(t))**2, self.pts)

    def elnh(self):
        return _gauss(lambda t: mp.log(_H(self.x(t))), self.pts)

    def d2(self):                       # E_t [(ln H)''(x)]^2, (ln H)'' = -E (E - x)
        def f(t):
            x = self.x(t)
            e = _E(x)
            return (e * (e - x))**2
        return _gauss(f, self.pts)


def _zpts(g):
    return [-30 / g, -3 / g, 0, 3 / g, 30 / g]


def _ising_q(qhat):
    g = mp.sqrt(qhat)
    one_m_q = _gauss(lambda z: mp.sech(g * z)**2, _zpts(g))     # 1 - q, no cancellation at q -> 1
    q = 1 - one_m_q if one_m_q < 0.5 else _gauss(lambda z: mp.tanh(g * z)**2, _zpts(g))
    return q, one_m_q


def _alpha_of(variables, o, k):
    if variables == "spherical":
        return o / ((1 - o) * _T(o, 1 - o, k).eh2())
    q, omq = _ising_q(o)
    return o * omq / _T(q, omq, k).eh2()


def _ising_S(qhat, k):
    g = mp.sqrt(qhat)
    q, omq = _ising_q(qhat)
    T = _T(q, omq, k)
    a = qhat * omq / T.eh2()
    return -qhat * omq / 2 + _gauss(lambda z: mp.log(2 * mp.cosh(g * z)), _zpts(g)) + a * T.elnh()


def rs_equations_symbolic(variables="spherical"):
    """The saddle equations the solver uses, in sympy form with the GaussE
    placeholder; E(x) = phi(x)/H(x) as in _E, x = (kappa - sqrt(q) t)/sqrt(1-q).
      spherical  q/(1-q) = alpha E_t E(x)^2
                 (alpha(q) = q / ((1-q) _T.eh2()) in _alpha_of / gardner_rs_point)
      ising      q = E_z tanh^2(sqrt(qhat) z)            (_ising_q)
                 qhat = alpha E_t E(x)^2 / (1-q)         (alpha(qhat) = qhat (1-q) / _T.eh2())"""
    _check(variables)                         # refuses unknown variables
    q, alpha = sp.Symbol("q", positive=True), sp.Symbol("alpha", positive=True)
    kappa, t = sp.Symbol("kappa", real=True), sp.Symbol("t", real=True)
    x = (kappa - sp.sqrt(q) * t) / sp.sqrt(1 - q)
    E = (sp.exp(-x**2 / 2) / sp.sqrt(2 * sp.pi)) / (sp.erfc(x / sp.sqrt(2)) / 2)
    if variables == "spherical":
        return [sp.Eq(q / (1 - q), alpha * GaussE(E**2, t))]
    qhat, z = sp.Symbol("qhat", positive=True), sp.Symbol("z", real=True)
    return [sp.Eq(q, GaussE(sp.tanh(sp.sqrt(qhat) * z)**2, z)),
            sp.Eq(qhat, alpha * GaussE(E**2, t) / (1 - q))]


# ---- RS saddle, parametrised by the order parameter --------------------------
def gardner_rs_point(variables, order, kappa=0) -> RSPoint:
    """RS saddle at order parameter `order`: q in [0, 1) (spherical) or
    qhat >= 0 (Ising); alpha comes from the stationarity equation."""
    with _prec():
        k = _check(variables, kappa)
        o = mp.mpf(order)
        if variables == "spherical":
            if not 0 <= o < 1:
                raise Refusal("precondition:order-in-range", f"spherical q = {order} must be in [0, 1)")
            if o == 0:
                return RSPoint(mp.mpf(0), o, None, mp.mpf(0), mp.mpf(1), variables, k)
            T = _T(o, 1 - o, k)
            a = o / ((1 - o) * T.eh2())
            S = mp.log(1 - o) / 2 + o / (2 * (1 - o)) + a * T.elnh()
            return RSPoint(a, o, None, S, 1 - a * T.d2(), variables, k)
        if not o >= 0:
            raise Refusal("precondition:order-in-range", f"Ising qhat = {order} must be >= 0")
        if o == 0:
            return RSPoint(mp.mpf(0), mp.mpf(0), o, mp.log(2), mp.mpf(1), variables, k)
        g = mp.sqrt(o)
        q, omq = _ising_q(o)
        T = _T(q, omq, k)
        a = o * omq / T.eh2()
        S = -o * omq / 2 + _gauss(lambda z: mp.log(2 * mp.cosh(g * z)), _zpts(g)) + a * T.elnh()
        s4 = _gauss(lambda z: mp.sech(g * z)**4, _zpts(g))
        return RSPoint(a, q, o, S, 1 - a * s4 * T.d2() / omq**2, variables, k)


# ---- roots: float64 bracket, mpmath secant polish -----------------------------
def _bracket_root(f, lo, hi, what, tol=1e-12):
    """Root of f on [lo, hi] (sign change required), Illinois regula falsi."""
    flo, fhi = f(lo), f(hi)
    if (flo > 0) == (fhi > 0):
        raise Refusal("no-solution", f"{what}: no sign change on [{lo:.6g}, {hi:.6g}]")
    x_old, side = None, 0
    for _ in range(200):
        x = (lo * fhi - hi * flo) / (fhi - flo)
        fx = f(x)
        if fx == 0 or (x_old is not None and abs(x - x_old) < tol * (1 + abs(x))):
            return x
        x_old = x
        if (fx > 0) == (fhi > 0):
            hi, fhi = x, fx
            if side == -1:
                flo /= 2
            side = -1
        else:
            lo, flo = x, fx
            if side == 1:
                fhi /= 2
            side = 1
    raise Refusal("no-solution", f"{what}: root not converged in 200 steps")


def _polish(f, u0, what):
    """Secant in mpmath from the float64 root u0, until the step is below the
    working precision (one or two steps from a 1e-12 start); audited by the caller."""
    u0 = mp.mpf(u0)
    u1 = u0 + mp.mpf("1e-7") * (1 + abs(u0))
    f0, f1 = f(u0), f(u1)
    for _ in range(10):
        if f1 == f0:
            break
        u0, u1 = u1, u1 - f1 * (u1 - u0) / (f1 - f0)
        if abs(u1 - u0) < mp.mpf(10)**(-mp.mp.dps + 2) * (1 + abs(u1)):
            return u1
        f0, f1 = f1, f(u1)
    if abs(u1 - u0) < mp.mpf(10)**(-mp.mp.dps + 4) * (1 + abs(u1)):
        return u1
    raise Refusal("no-solution", f"{what}: secant polish did not converge")


def _solve_alpha(variables, alpha, k):
    """Invert alpha(order) (increasing) in u = logit(q) or ln(qhat)."""
    a, kf = float(alpha), float(k)
    if variables == "spherical":
        order, lo, hi = (lambda u: 1 / (1 + mp.exp(-u))), -30.0, 28.0
        ff = lambda u: math.log(F.spherical_alpha(1 / (1 + math.exp(-u)), kf) / a)
    else:
        order, lo, hi = mp.exp, -30.0, 18.0
        ff = lambda u: math.log(F.ising_alpha(math.exp(u), kf) / a)
    what = f"RS saddle at alpha = {mp.nstr(alpha, 10)}"
    u0 = _bracket_root(ff, lo, hi, what)
    u = _polish(lambda u: mp.log(_alpha_of(variables, order(u), k) / alpha), u0, what)
    pt = gardner_rs_point(variables, order(u), k)
    if abs(pt.alpha / alpha - 1) > mp.mpf(10)**(-mp.mp.dps + 3):
        raise Refusal("no-solution", f"{what}: audit residual {mp.nstr(pt.alpha / alpha - 1, 3)}")
    return pt


def gardner_entropy(alpha, variables, kappa=0):
    """RS entropy density of the version space at load alpha (see conventions)."""
    with _prec():
        k = _check(variables, kappa)
        a = mp.mpf(alpha)
        if not a >= 0:
            raise Refusal("precondition:nonnegative-alpha", f"alpha = {alpha}")
        if a == 0:
            return mp.mpf(0) if variables == "spherical" else mp.log(2)
        ac = gardner_capacity(variables, k).alpha_c
        if variables == "spherical" and a >= ac:
            return mp.ninf          # q -> 1: the version space is empty (prob. -> 1)
        if variables == "ising" and a >= ac:
            raise Refusal("no-solution", f"alpha = {mp.nstr(a, 8)} >= RS endpoint "
                          f"{mp.nstr(ac, 8)}: no RS saddle (the 1RSB capacity is lower)")
        pt = _solve_alpha(variables, a, k)
        if pt.replicon < 0:
            raise Refusal("unstable:rs", f"replicon = {mp.nstr(pt.replicon, 6)} < 0 at alpha = "
                          f"{mp.nstr(a, 8)}: RS is AT-unstable, its entropy is not reported")
        if variables == "ising" and pt.S < 0:
            raise Refusal("no-solution:negative-entropy",
                          f"RS entropy {mp.nstr(pt.S, 6)} < 0 at alpha = {mp.nstr(a, 8)}: beyond "
                          "the frozen-1RSB capacity no solutions remain; see "
                          "gardner_capacity('ising', ansatz='1rsb')")
        return pt.S


# ---- capacity ---------------------------------------------------------------
def gardner_capacity(variables, kappa=0, ansatz="rs") -> Capacity:
    with _prec():
        k = _check(variables, kappa)
        if ansatz not in ("rs", "1rsb"):
            raise Refusal("precondition:known-ansatz", f"ansatz = {ansatz!r}: use 'rs' or '1rsb'")
        D = _den(k)
        if variables == "spherical":
            if ansatz == "1rsb":
                raise Refusal("not-implemented:spherical-rsb",
                              "spherical perceptron: RS is exact for kappa >= 0 (use ansatz='rs'); "
                              "for kappa < 0 the capacity needs full RSB (Franz-Parisi 2015), not in v1")
            ac = 1 / D
            lam = k * (k * _H(-k) + _phi(k)) * ac        # = 1 - alpha_c H(-kappa), sign = sign(kappa)
            if k > 0:
                note = "kappa > 0: replicon > 0 at capacity, convex version space, RS exact"
            elif k == 0:
                note = "kappa = 0: replicon -> 0 at alpha_c (marginal); RS exact (Gardner 1988)"
            else:
                note = ("kappa < 0: replicon < 0 at capacity; RS alpha_c is only an upper bound "
                        "(non-convex, RSB needed)")
            return Capacity(ac, bool(lam >= 0), "rs", variables, k, lam, note)
        if ansatz == "rs":
            note = ("RS q -> 1 endpoint (Gardner-Derrida), not the capacity: RS entropy is negative "
                    "beyond the frozen-1RSB root, the replicon -> -inf as q -> 1, and alpha_c > 1 "
                    "breaks the counting bound E #solutions = 2^N 2^-P")
            return Capacity(2 / (mp.pi * D), False, "rs", variables, k, mp.ninf, note)
        return _ising_1rsb(mp.nstr(k, mp.mp.dps), mp.mp.dps)


@lru_cache(maxsize=None)
def _ising_1rsb(kappa_str, dps):
    with mp.workdps(dps):
        k = mp.mpf(kappa_str)
        what = f"root of the Ising RS entropy, kappa = {kappa_str}"
        u0 = _bracket_root(lambda u: F.ising_entropy(math.exp(u), float(k)),
                           math.log(0.1), math.log(50.0), what)
        u = _polish(lambda u: _ising_S(mp.exp(u), k), u0, what)
        pt = gardner_rs_point("ising", mp.exp(u), k)
        if abs(pt.S) > mp.mpf(10)**(-mp.mp.dps + 3):
            raise Refusal("no-solution", f"audit: |S_RS| = {mp.nstr(pt.S, 3)} at the root")
        if pt.replicon <= 0:
            raise Refusal("unstable:rs", f"frozen 1RSB root at alpha = {mp.nstr(pt.alpha, 8)} has "
                          f"RS replicon {mp.nstr(pt.replicon, 6)} <= 0")
        note = ("frozen 1RSB (q1 = 1): root of the RS entropy (Krauth-Mezard); RS replicon "
                f"{mp.nstr(pt.replicon, 4)} > 0 there")
        return Capacity(pt.alpha, True, "1rsb", "ising", k, pt.replicon, note, pt)


def gardner_capacity_curve(variables, kappas, ansatz="rs"):
    return [gardner_capacity(variables, k, ansatz) for k in kappas]
