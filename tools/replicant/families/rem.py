"""Family (b): Random Energy Model (Derrida, Phys. Rep. 67, 29 (1980); PRB 24, 2613 (1981)).

Normalisation.  2^N i.i.d. energies E_i ~ N(0, N_scale * N): N_scale is the
energy variance per spin.  N_scale = 1 (the default) gives
beta_c = sqrt(2 ln 2).  Derrida writes P(E) ~ exp(-E^2/(N J^2)), i.e.
N_scale = J^2/2; at J = 1 (N_scale = 1/2) beta_c = 2 sqrt(ln 2),
T_c = 1/(2 sqrt(ln 2)), and that is also the energy scale of the spherical
p-spin with couplings of variance p!/(2 N^(p-1)) (E H^2 = N/2), so the p-spin
module compares against N_scale = 1/2.

  rem_critical_beta(N_scale=1)            -> sqrt(2 ln 2 / N_scale)
  rem_entropy(e, N_scale=1)               -> s(e) = ln 2 - e^2/(2 N_scale)   (e = E/N)
  rem_free_energy(beta, N_scale=1)        -> f per spin, N -> infinity
  rem_1rsb_functional()                   -> sympy -beta f(m) (symbols m, beta, s)
  rem_1rsb(beta, N_scale=1)               -> RSBSolution(q0=0, q1=1, m=beta_c/beta, f, e)
  rem_quenched_finite_n(beta, N, N_scale) -> (1/N) E ln Z at finite N (float, quadrature)

Results.  Microcanonically s(e) vanishes at e0 = -sqrt(2 N_scale ln 2) and
beta_c = s'(e0).  Above T_c the annealed and quenched free energies agree:
-beta f = ln 2 + N_scale beta^2 / 2.  Below, the system is frozen at e0:
-beta f = beta sqrt(2 N_scale ln 2) (f = e0, entropy 0).  f and e are
continuous at beta_c; the specific heat jumps.

1RSB route (n -> 0).  E Z^n with the n replicas in n/m groups of m that share
a configuration (overlap q1 = 1 inside a group, q0 = 0 between): the choice of
configurations gives (2^N)^(n/m), and E exp(-beta sum_a E_a) gives
exp(N_scale N beta^2 sum_{a,b} q_ab^p / 2), with sum_{a,b} q_ab^p = n + the
off-diagonal ansatz.sum_offdiag_power sum (any p > 0, since q in {0, 1}).
Per replica and spin, via ansatz.n_to_zero:
    -beta f(m) = ln 2 / m + N_scale beta^2 m / 2,
stationary at m* = sqrt(2 ln 2 / N_scale) / beta = beta_c / beta (Parisi
breakpoint; Derrida & Mottishaw 2015, mu = T/T_c).  For beta < beta_c,
m* > 1 leaves 0 < m <= 1 and the answer is the paramagnet (m = 1).  The n -> 0
continuation and the ansatz are physics-level, not proven; the free energy
itself is a theorem here (Derrida's direct computation), which is what makes
the REM a check on the replica route.
"""
import math

import mpmath as mp
import numpy as np
import sympy as sp

from ..ansatz import n_to_zero, sum_offdiag_power
from ..core import Refusal
from .common import RSBSolution

_DPS = 30


def _prec():   # never lower the caller's precision (mp.diff raises it)
    return mp.workdps(max(_DPS, mp.mp.dps))


def _s(N_scale):
    s = mp.mpf(N_scale)
    if not s > 0:
        raise Refusal("precondition:positive-N_scale",
                      f"N_scale = {N_scale} (energy variance per spin) must be > 0")
    return s


def _beta(beta):
    b = mp.mpf(beta)
    if not b > 0:
        raise Refusal("precondition:positive-beta", f"beta = {beta} must be > 0")
    return b


def rem_critical_beta(N_scale=1):
    """beta_c = sqrt(2 ln 2 / N_scale); sqrt(2 ln 2) at the default N_scale = 1."""
    with _prec():
        return mp.sqrt(2 * mp.log(2) / _s(N_scale))


def rem_entropy(e, N_scale=1):
    """Annealed entropy per spin at energy per spin e: ln 2 - e^2 / (2 N_scale)."""
    with _prec():
        return mp.log(2) - mp.mpf(e)**2 / (2 * _s(N_scale))


def rem_free_energy(beta, N_scale=1):
    """Quenched free energy per spin, N -> infinity (Derrida)."""
    with _prec():
        b, s = _beta(beta), _s(N_scale)
        bc = rem_critical_beta(s)
        if b <= bc:
            return -(mp.log(2) + s * b**2 / 2) / b
        return -mp.sqrt(2 * s * mp.log(2))


def rem_1rsb_functional():
    """-beta f(m) per spin from the replica blocks (symbols m, beta, s = N_scale)."""
    n, m, beta, s, p = sp.symbols("n m beta s p", positive=True)
    q0, q1 = sp.Integer(0), sp.Integer(1)
    # (1/n) sum_{a,b} q_ab^p: diagonal n plus the off-diagonal 1RSB sum
    pairs = 1 + n_to_zero(sum_offdiag_power("rsb", [m], [q0, q1], p, n), n)
    entropy = n_to_zero((n / m) * sp.log(2), n)        # ln (2^N)^(n/m), per n and N
    return sp.simplify(entropy + s * beta**2 * pairs / 2)


def rem_1rsb(beta, N_scale=1) -> RSBSolution:
    with _prec():
        b, s = _beta(beta), _s(N_scale)
        bc = rem_critical_beta(s)
        m = bc / b
        if m > 1:
            raise Refusal("no-solution", f"beta = {mp.nstr(b, 8)} < beta_c = {mp.nstr(bc, 8)}: "
                          f"stationary m = beta_c/beta = {mp.nstr(m, 8)} > 1 is outside "
                          "0 < m <= 1; this is the paramagnet (m = 1, RS), use rem_free_energy")
        phi = mp.log(2) / m + s * b**2 * m / 2
        # e = d(beta f)/d beta at fixed m (stationarity) = -N_scale beta m = e0
        return RSBSolution(mp.mpf(0), mp.mpf(1), m, -phi / b, -s * b * m)


def rem_quenched_finite_n(beta, N, N_scale=1, *, de=0.05, nu=1500):
    """(1/N) E ln Z for N finite, without replicas, float64 quadrature.

    E ln Z = int_0^inf dt/t [e^-t - E e^(-t Z)], and since the 2^N energies
    are i.i.d., E e^(-t Z) = g(t)^(2^N), g(t) = E exp(-t e^(-beta E)).  Using
    1 - g = h(t) = E[1 - exp(-t e^(-beta E))] keeps the precision:
    integrand (in u = ln t) = expm1(-t) - expm1(2^N log1p(-h)).
    Energy grid: trapezoid over |E| <= (sqrt(2 N ln 2) + 8) sd, step de/beta;
    u grid: nu points down to -(ln Z_max + 20).  Grid refinement (de 0.05 ->
    0.02, nu 1500 -> 4000) at beta = 2 beta_c changes the result by < 1e-12
    for N <= 80, 1.3e-10 at N = 160, 1.8e-7 at N = 320.  float64, not certified.
    """
    beta, s = float(_beta(beta)), float(_s(N_scale))
    if not (isinstance(N, int) and 1 <= N <= 1000):
        raise Refusal("precondition:integer-N", f"N = {N!r}: need an int in [1, 1000]")
    sd = math.sqrt(s * N)
    L = (math.sqrt(2 * N * math.log(2)) + 8) * sd
    E = np.linspace(-L, L, int(2 * L * beta / de) + 2)
    w = np.exp(-E**2 / (2 * sd * sd)) / (math.sqrt(2 * math.pi) * sd) * (E[1] - E[0])
    u = np.linspace(-(N * math.log(2) + beta * L) - 20, 6, nu)
    with np.errstate(over="ignore"):
        h = (-np.expm1(-np.exp(u[:, None] - beta * E[None, :]))) @ w
    integ = np.expm1(-np.exp(u)) - np.expm1(2.0**N * np.log1p(-h))
    return float(integ.sum() * (u[1] - u[0]) / N)
