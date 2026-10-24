"""Family (b): spherical p-spin glass (Crisanti & Sommers, Z. Phys. B 87, 341 (1992)).

Convention.  H = -sum_{i1<..<ip} J_{i1..ip} s_i1..s_ip, sum_i s_i^2 = N,
J ~ N(0, J^2 p!/(2 N^(p-1))) with J = 1 (Castellani-Cavagna 2005, Eq. 42), so
E H(s) H(t) = N Q(s,t)^p / 2: at p -> infinity, the REM with N_scale = 1/2.

  pspin_functional(p)               -> sympy -beta f(q0, q, m, beta)  (1RSB, n -> 0)
  logdet_n_to_zero()                -> sympy lim (1/n) ln det Q (1RSB)
  pspin_spherical_1rsb(p, beta)     -> RSBSolution(q0 = 0, q1 = q, m, f, e)
  pspin_critical_temperature(p, kind="static" | "dynamic")

Replica functional.  E Z^n = int dQ exp(N n [beta^2/(4n) sum_{ab} Q_ab^p
+ (1/(2n)) ln det Q] + const): the spherical constraint gives Q_aa = 1 and
the ln det from the Gaussian integral over the spins.  1RSB (q0 between
blocks, q in a block of size m), n -> 0:
  (1/n) sum_ab Q^p     = 1 + (m-1) q^p - m q0^p      (ansatz.sum_offdiag_power)
  (1/n) ln det Q       = (1 - 1/m) ln(1-q) + (1/m) ln(1-q + m(q-q0))
                         + q0 / (1-q + m(q-q0))        (eigenvalues; tested
                         against ansatz.rsb_matrix at integer n)
  -beta f = beta^2/4 (1 + (m-1) q^p - m q0^p) + (1/2)(1/n) ln det Q.
This is Castellani-Cavagna Eq. (81) divided by 2.  f omits the
q-independent sphere-entropy constant -(1 + ln 2 pi)/(2 beta); the energy
e = d(beta f)/d beta = -(beta/2)(1 + (m-1) q^p) does not depend on it.
At zero field q0 = 0 solves the q0 equation.

Solution.  The stationarity equations in (q, m) reduce (Crisanti-Sommers;
Cugliandolo-Lozano-Nessi 2017 Eqs. 8-10) to y = (1-q)/(1-(1-m)q) with
    1/p = -y (1 - y + ln y)/(1 - y)^2       (y depends on p only),
    (p/2) q^(p-2) (1-q)^2 = y T^2,           m = (1-y)(1-q)/(y q),
taking the larger root q > (p-2)/p.  The static transition T_s is where
m = 1: q_s = 1 - y, T_s = sqrt(p y / 2) (1-y)^(p/2 - 1).  Above T_s, m > 1 is
outside 0 < m <= 1 and the equilibrium state is the paramagnet (q = 0,
-beta f = beta^2/4): pspin_spherical_1rsb refuses and says so.  Every returned
point is audited on OUR sympy functional: gradient in (q0, q, m) below 1e-15,
Hessian of f negative definite (local maximum, the n -> 0 replica rule).
The dynamic (mode-coupling) transition is the closed form
T_d = sqrt(p (p-2)^(p-2) / (2 (p-1)^(p-1))) (Castellani-Cavagna Eq. 147).

Honest label: replica n -> 0 and 1RSB are physics-level here (the exactness of
1RSB for this model is argued by Crisanti-Sommers; not proven in replicant).
"""
from functools import lru_cache

import mpmath as mp
import sympy as sp

from ..ansatz import n_to_zero, sum_offdiag_power
from ..core import Refusal
from .common import RSBSolution

_DPS = 30


def _prec():   # never lower the caller's precision (mp.diff raises it)
    return mp.workdps(max(_DPS, mp.mp.dps))
_Q0, _Q, _M, _B, _N = sp.symbols("q0 q m beta n")


def logdet_n_to_zero():
    """lim_{n->0} (1/n) ln det Q for the 1RSB matrix (eigenvalues 1-q with
    multiplicity n - n/m, 1-q+m(q-q0) with n/m - 1, and that plus n q0 once)."""
    lam1, lam2 = 1 - _Q, 1 - _Q + _M * (_Q - _Q0)
    expr = ((_N - _N / _M) * sp.log(lam1) + (_N / _M - 1) * sp.log(lam2)
            + sp.log(lam2 + _N * _Q0)) / _N
    return sp.simplify(sp.limit(expr, _N, 0))


def _check_p(p):
    if not (isinstance(p, int) and p >= 3):
        raise Refusal("precondition:p-at-least-3",
                      f"p = {p!r}: the 1RSB entry needs an integer p >= 3 "
                      "(p = 2 has a continuous RS transition, not 1RSB)")


_P = sp.Symbol("p", positive=True)


@lru_cache(maxsize=None)
def _functional_symbolic_p():
    pairs = 1 + n_to_zero(sum_offdiag_power("rsb", [_M], [_Q0, _Q], _P, _N), _N)
    return _B**2 * pairs / 4 + logdet_n_to_zero() / 2


def pspin_functional(p):
    """-beta f per spin, 1RSB, as a sympy expression in q0, q, m, beta."""
    _check_p(p)
    return _functional_symbolic_p().subs(_P, p)


@lru_cache(maxsize=None)
def _compiled():
    """phi, grad, Hessian in (q0, q, m), compiled once with p as an argument."""
    phi = _functional_symbolic_p()
    v = (_Q0, _Q, _M)
    ps = lambda e: sp.powsimp(e, force=True)     # q0**p/q0 -> q0**(p-1): finite at q0 = 0
    grad = [ps(sp.diff(phi, x)) for x in v]
    hess = sp.hessian(phi, v).applyfunc(ps)
    args = (_Q0, _Q, _M, _B, _P)
    return (sp.lambdify(args, phi, "mpmath"), sp.lambdify(args, grad, "mpmath"),
            sp.lambdify(args, hess, "mpmath"))


@lru_cache(maxsize=None)
def _y(p):
    """Root in (0, 1) of 1/p + y (1 - y + ln y)/(1 - y)^2 = 0."""
    with _prec():
        F = lambda y: 1 / mp.mpf(p) + y * (1 - y + mp.log(y)) / (1 - y)**2
        return mp.findroot(F, (mp.mpf("1e-30"), 1 - mp.mpf("1e-12")), solver="illinois")


def pspin_critical_temperature(p, kind="static"):
    """T_s (static 1RSB / Kauzmann, m = 1) or T_d (dynamic, closed form), J = 1."""
    _check_p(p)
    with _prec():
        pm = mp.mpf(p)
        if kind == "static":
            y = _y(p)
            return mp.sqrt(pm * y / 2) * (1 - y)**(pm / 2 - 1)
        if kind == "dynamic":
            return mp.sqrt(pm * (pm - 2)**(pm - 2) / (2 * (pm - 1)**(pm - 1)))
    raise Refusal("precondition:known-transition",
                  f"kind = {kind!r}: use 'static' (T_s, 1RSB) or 'dynamic' (T_d)")


def _hessian_f_eigs(p, beta, q0, q, m):
    """Eigenvalues of the Hessian of f = -phi/beta in (q0, q, m)."""
    with _prec():
        b = mp.mpf(beta)
        H = mp.matrix(_compiled()[2](mp.mpf(q0), mp.mpf(q), mp.mpf(m), b, p)) * (-1 / b)
        return tuple(sorted(mp.re(e) for e in mp.eig(H)[0]))


def pspin_spherical_1rsb(p, beta) -> RSBSolution:
    _check_p(p)
    with _prec():
        b = mp.mpf(beta)
        if not b > 0:
            raise Refusal("precondition:positive-beta", f"beta = {beta} must be > 0")
        T, pm, y = 1 / b, mp.mpf(p), _y(p)
        ts = pspin_critical_temperature(p)
        para = (f"T = {mp.nstr(T, 8)} > T_s = {mp.nstr(ts, 8)}: paramagnet (q = 0, "
                f"-beta f = beta^2/4); the 1RSB")
        F = lambda q: pm / 2 * q**(p - 2) * (1 - q)**2 - y * T**2
        q_lo = (pm - 2) / pm                 # maximum of q^(p-2)(1-q)^2
        if F(q_lo) <= 0:
            raise Refusal("no-solution", f"{para} q-equation has no root at all here")
        q = mp.findroot(F, (q_lo, mp.mpf(1)), solver="illinois")
        m = (1 - y) * (1 - q) / (y * q)
        if m > 1:
            raise Refusal("no-solution", f"{para} branch has m = {mp.nstr(m, 8)} > 1, "
                          "outside 0 < m <= 1")
        phi, grad, _ = _compiled()
        g = max(abs(x) for x in grad(mp.mpf(0), q, m, b, p))
        if g > mp.mpf("1e-15") * (1 + b**2):
            raise Refusal("no-solution", f"audit: |grad| = {mp.nstr(g, 3)} at q = {q}, m = {m}")
        eigs = _hessian_f_eigs(p, b, 0, q, m)
        if max(eigs) >= 0:
            raise Refusal("no-solution", f"audit: 1RSB point (q = {mp.nstr(q, 8)}, m = "
                          f"{mp.nstr(m, 8)}) is not a local maximum of f (eigenvalues {eigs})")
        f = -phi(mp.mpf(0), q, m, b, p) / b
        e = -(b / 2) * (1 + (m - 1) * q**p)
        return RSBSolution(mp.mpf(0), q, m, f, e)
