"""Offline (batch) learning of the linear perceptron at zero temperature, for comparison with the
online flow of linear.py (O2; Kinouchi & Caticha, Phys. Rev. E 52, 2878 (1995), Sec. 4 -- KC95).

KC minimise E(J, r) = (1/2) sum_mu (sqrt(Q) b_mu / (sqrt(M) r) - h_mu)^2 over the aN examples (Eq.
(39)), on the sphere J . J = Q (constrained) or freely, and report (RS, zero learning temperature)
rho = alpha / r, q~ = alpha / r^2 (Eq. (47)) for alpha < 1: the standard algorithm is
r = sqrt(Q/M) (E = sum (b - h)^2 / 2), the 'optimal' one r = sqrt(alpha) (Eq. (48)), rho = sqrt(alpha)
(Eq. (49)); for alpha > 1 rho = 1.  Unconstrained, the minimiser is the pseudoinverse, rho =
sqrt(alpha) (Sec. 4.4).

Route here (derived, not the replica calculation): the zero-energy set.  The aN inputs span a
random subspace with projector P; |P B|^2 -> alpha M (a random alpha N-dimensional subspace); h_mu
= c b_mu for all mu, c = sqrt(Q) / (sqrt(M) r), means J = c P B + w with w orthogonal to the data,
and |J|^2 = Q fixes |w|^2 = Q (1 - alpha / r^2) >= 0 (so the zero-error set exists iff alpha <= r^2;
outside it the standard algorithm on an unrealizable sphere Q < alpha M is refused).  At T -> 0
the Gibbs measure is uniform on that set: w has a random direction, so rho = c alpha M / sqrt(Q M)
= alpha / r and two such J overlap q~ = c^2 alpha M / Q = alpha / r^2.  Unconstrained, the minimum
norm zero-error J is P B: R = Q = alpha M, rho = sqrt(alpha), e_g = M (1 - alpha) / 2.

ridge_route maps the pseudoinverse onto ridge (E7, learning/ridge.py, HMRT Eq. (8)) as a second
route: n = P examples, p = N weights, gamma = p / n = 1 / alpha, noise sigma2 = 0 (KC's examples
are noiseless), signal r^2 = |beta|^2 = M (both use unit-variance inputs, so the teacher output
has variance M in both); ridge's excess risk |beta_hat - beta|^2 = <(h - b)^2> = 2 e_g; a projection
has R = Q, hence Q = M - risk and rho = sqrt(1 - risk / M).  Ridge's number never enters linear.py.
"""
import math
from collections import namedtuple

import numpy as np

from ..core import Refusal
from . import linear as L

Offline = namedtuple("Offline", "alpha rho q_tilde R Q eg")
RidgeRoute = namedtuple("RidgeRoute", "gamma risk eg rho")
ALGORITHMS = ("standard", "optimal")


def _alpha(alpha):
    if not (L._real(alpha) and alpha > 0):
        raise Refusal("precondition:alpha", f"alpha={alpha!r}: need a finite alpha > 0")
    return float(alpha)


def offline(alpha, algorithm="standard", constraint="sphere", M=1.0, Q=None):
    """Offline(alpha, rho, q_tilde, R, Q, eg) of the T = 0 offline solution (module docstring)."""
    a = _alpha(alpha)
    if algorithm not in ALGORITHMS:
        raise Refusal("unsupported:algorithm", f"algorithm {algorithm!r}: {', '.join(ALGORITHMS)}")
    if constraint not in L.CONSTRAINTS:
        raise Refusal("precondition:constraint", f"constraint={constraint!r}: none | sphere")
    if not (L._real(M) and M > 0):
        raise Refusal("precondition:teacher-norm", f"M={M!r}: need a finite teacher norm > 0")
    if constraint == "none":
        if Q is not None:
            raise Refusal("precondition:known-parameter", "Q is the sphere's radius: the "
                          "unconstrained offline solution chooses its own norm")
        f = min(a, 1.0)
        return Offline(a, math.sqrt(f), None, f * M, f * M, M * (1 - f) / 2)
    Q = float(M if Q is None else Q)
    if not (math.isfinite(Q) and Q > 0):
        raise Refusal("precondition:state-point", f"Q={Q!r}: need a sphere radius Q > 0")
    r2 = Q / M if algorithm == "standard" else min(a, 1.0)
    if algorithm == "standard" and Q != M and (a > r2 or a >= 1):
        raise Refusal("precondition:offline-window", f"alpha = {a:g}, Q/M = {Q / M:g}: the standard "
                      "algorithm has no zero-error point on this sphere (alpha > Q/M, or alpha >= 1 "
                      "where the data fix J = B and Q != M: unrealizable)")
    if a >= 1:
        rho, qt = 1.0, 1.0
    else:
        rho, qt = a / math.sqrt(r2), a / r2
    R = rho * math.sqrt(Q * M)
    return Offline(a, rho, qt, R, Q, (M + Q - 2 * R) / 2)


def ridge_route(alpha, M=1.0):
    """The pseudoinverse's e_g and rho through ridge's ridgeless excess risk (module docstring)."""
    from ..learning import ridge
    a = _alpha(alpha)
    risk = float(ridge.ridgeless_test_error(1 / a, 0, signal=M))
    return RidgeRoute(1 / a, risk, risk / 2, math.sqrt(max(1 - risk / M, 0.0)))


def crossing_alpha(M=1.0, R0=1e-4, grid=(0.55, 0.85, 301)):
    """alpha where optimal ONLINE learning on the sphere Q = M (the ODE of linear.py from
    rho0 = R0 / M) stops beating OFFLINE standard learning (rho = alpha): KC p. 11 'up to
    alpha ~ 0.71'.  Located on the grid, then linearly interpolated (grid step 1e-3)."""
    al = np.linspace(*grid)
    tr = L.integrate_linear(L.linear_flow("optimal", constraint="sphere", M=M), al[-1],
                            (R0 * M, M), alphas=al)
    g = tr.rho - np.array([offline(a, "standard", M=M).rho for a in al])
    k = np.flatnonzero((g[:-1] > 0) & (g[1:] <= 0))
    if len(k) != 1:
        raise Refusal("undetermined:crossing", f"{len(k)} sign changes of rho_online - rho_offline "
                      f"on [{grid[0]}, {grid[1]}]")
    i = int(k[0])
    return float(al[i] + (al[i + 1] - al[i]) * g[i] / (g[i] - g[i + 1]))


__all__ = ["ALGORITHMS", "Offline", "RidgeRoute", "crossing_alpha", "offline", "ridge_route"]
