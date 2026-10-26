"""Finite-N learning of a linear teacher by a linear perceptron over named seeds (O2; an
independent route for linear.py and linear_offline.py).

It imports nothing from linear.py, linear_offline.py, odes.py, fields.py or optimal.py
(tests/test_online_linear.py inspects the imports); the rules are written out here again.
Convention as linear.py: per seed a teacher B uniform on the sphere |B|^2 = M, a student
J0 = (R0 / M) B + sqrt(Q0 - R0^2 / M) v (v a random unit vector orthogonal to B), and examples xi
with i.i.d. N(0, 1) components; b = B . xi, h = J . xi.

online  J <- J + (eta / N) F xi with F = b - h (standard) or F = (Q / R) b - h (optimal, which
        reads the current R and Q exactly: it is the rule that knows rho); eta = "greedy" is
        (Q M - R^2) / (R (M + Q - 2R)) from the current weights.  constraint "sphere" rescales J
        to |J|^2 = Q0 after every example (KC's decay Omega to O(1/N)).
offline the zero-temperature solutions of KC Sec. 4 from the sampled data X (P x N, P = round(alpha N)
        < N), two ways.  method="construct": the projection P B onto the span of the inputs (QR of
        X^T) is the pseudoinverse (unconstrained); on the sphere the optimal solution is
        sqrt(Q) P B / |P B| and the standard one is P B + w, w a uniformly random direction
        orthogonal to the data with |w|^2 = Q - |P B|^2.  That last step IS the derivation's
        assumption (the T -> 0 Gibbs measure is uniform on the zero-error set), so this method checks
        the finite-N concentration of |P B|^2 only.  method="descent" assumes nothing about the
        solution: gradient descent on the standard energy E = |X J - X B|^2 / 2 with step 1 / |X|_2^2, renormalised to |J|^2 = Q after every step from a uniformly
        random start on the sphere (from J = 0 unconstrained); stopped when the gradient tangent to the
        sphere (the gradient, unconstrained) is below 1e-9 |X^T c X B| (refused undetermined:descent
        after 20000 steps).  The gradient lies in the span of the data, so descent keeps the direction
        of the start's component orthogonal to it: for the standard algorithm it lands on the point of
        the zero-error set picked by that random direction.  It is offered for the standard algorithm
        only (refused precondition:method for the optimal one): at finite N the optimal energy with
        r = sqrt(alpha) has a degenerate zero-error set whenever |P B|^2 < alpha M, so its T = 0
        minimiser is not unique and differs from the N -> oo answer by the O(1/sqrt N) fluctuation
        of |P B|^2 (measured: rho 0.67 +- 0.06 at N = 100, against sqrt(1/2)).

e_g = (M + Q - 2R) / 2 and rho = R / sqrt(Q M) are exact for Gaussian inputs given the weights.
Seeds: each seed owns its numpy Generator (default_rng(seed)); 99% Student-t bands over seeds
(amp._stats.student_band): statistical, never certified.
"""
import math
from dataclasses import dataclass

import numpy as np

from ..amp._stats import Band, student_band
from ..core import Refusal
from ..csp import _checks

_BLOCK = 128


@dataclass(frozen=True)
class LinSim:
    alpha: float
    band: Band                 # 99% band of the mean e_g over seeds
    rho_band: Band             # 99% band of the mean rho over seeds
    eg_seeds: tuple
    N: int
    seeds: tuple


def _checked(N, seeds, M, alpha):
    if isinstance(N, bool) or not isinstance(N, int) or N < 10:
        raise Refusal("precondition:size", f"N={N!r}: need an int >= 10")
    if not (isinstance(M, (int, float)) and math.isfinite(M) and M > 0):
        raise Refusal("precondition:teacher-norm", f"M={M!r}: need a teacher norm > 0")
    if not (isinstance(alpha, (int, float)) and math.isfinite(alpha) and alpha > 0):
        raise Refusal("precondition:alpha", f"alpha={alpha!r}: need alpha > 0")
    return _checks.seeds(seeds)


def _teacher(rngs, N, M):
    B = np.stack([g.standard_normal(N) for g in rngs])
    return B * (math.sqrt(M) / np.linalg.norm(B, axis=1, keepdims=True))


def _dot(a, b):
    return np.einsum("sn,sn->s", a, b)


def _result(alpha, J, B, M, N, seeds):
    R, Q = _dot(J, B), _dot(J, J)
    eg = (M + Q - 2 * R) / 2
    return LinSim(float(alpha), student_band(eg), student_band(R / np.sqrt(Q * M)),
                  tuple(float(e) for e in eg), N, seeds)


def simulate_linear(rule, N, alpha_max, seeds, eta=1.0, constraint="none", M=1.0, init=(0.0, 1.0)):
    """LinSim of ONLINE learning at N to alpha_max over the named seeds (module docstring)."""
    if rule not in ("standard", "optimal"):
        raise Refusal("unsupported:rule", f"rule {rule!r}: standard | optimal")
    if constraint not in ("none", "sphere"):
        raise Refusal("precondition:constraint", f"constraint={constraint!r}: none | sphere")
    greedy = eta == "greedy"
    if not greedy and not (isinstance(eta, (int, float)) and math.isfinite(eta) and eta > 0):
        raise Refusal("precondition:eta", f"eta={eta!r}: a finite learning rate > 0 or 'greedy'")
    if rule == "optimal" and eta != 1:
        raise Refusal("precondition:optimal-eta", "the optimal modulation is the optimum at eta = 1 only")
    seeds = _checked(N, seeds, M, alpha_max)
    R0, Q0 = (float(v) for v in init)
    if not (Q0 > 0 and R0 * R0 < Q0 * M) or ((rule == "optimal" or greedy) and R0 <= 0):
        raise Refusal("precondition:initial-overlap", f"(R0, Q0) = ({R0}, {Q0}): need Q0 > 0, "
                      "R0^2 < Q0 M, and R0 > 0 for the optimal rule or the greedy eta")
    rngs = [np.random.default_rng(s) for s in seeds]
    B = _teacher(rngs, N, M)
    V = np.stack([g.standard_normal(N) for g in rngs])
    V -= (_dot(V, B) / M)[:, None] * B
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    J = (R0 / M) * B + math.sqrt(Q0 - R0 * R0 / M) * V
    P, step = round(alpha_max * N), 0
    while step < P:
        k = min(_BLOCK, P - step)
        xs = np.stack([g.standard_normal((k, N)) for g in rngs], axis=1)     # (k, S, N)
        for i in range(k):
            x = xs[i]
            b, h = _dot(B, x), _dot(J, x)
            if rule == "optimal" or greedy:
                R, Q = _dot(J, B), _dot(J, J)
            F = (Q / R) * b - h if rule == "optimal" else b - h
            e = (Q * M - R * R) / (R * (M + Q - 2 * R)) if greedy else eta
            J += (e / N * F)[:, None] * x
            if constraint == "sphere":
                J *= np.sqrt(Q0 / _dot(J, J))[:, None]
            step += 1
    return _result(alpha_max, J, B, M, N, seeds)


def _descent(X, beta, c, Q, g, sphere):
    y = c * (X @ beta)
    step = 1 / np.linalg.norm(X, 2) ** 2
    if sphere:
        J = g.standard_normal(X.shape[1])
        J *= math.sqrt(Q) / np.linalg.norm(J)
    else:
        J = np.zeros(X.shape[1])
    scale = np.linalg.norm(X.T @ y)
    for _ in range(20000):
        grad = X.T @ (X @ J - y)
        tang = grad - (grad @ J) / (J @ J) * J if sphere else grad
        if np.linalg.norm(tang) < 1e-9 * scale:
            return J
        J = J - step * grad
        if sphere:
            J *= math.sqrt(Q) / np.linalg.norm(J)
    raise Refusal("undetermined:descent", "gradient descent did not reach zero training error in 20000 steps")


def simulate_offline(algorithm, N, alpha, seeds, constraint="sphere", M=1.0, Q=None, method="construct"):
    """LinSim of the zero-temperature OFFLINE solutions built from sampled data (module docstring)."""
    if method not in ("construct", "descent"):
        raise Refusal("precondition:method", f"method={method!r}: construct | descent")
    if method == "descent" and algorithm == "optimal" and constraint == "sphere":
        raise Refusal("precondition:method", "descent is offered for the standard algorithm only (the "
                      "finite-N optimal energy has a degenerate zero-error set; module docstring)")
    if algorithm not in ("standard", "optimal"):
        raise Refusal("unsupported:algorithm", f"algorithm {algorithm!r}: standard | optimal")
    if constraint not in ("none", "sphere"):
        raise Refusal("precondition:constraint", f"constraint={constraint!r}: none | sphere")
    seeds = _checked(N, seeds, M, alpha)
    P = round(alpha * N)
    if not 0 < P < N:
        raise Refusal("precondition:alpha", f"alpha N = {P}: the offline simulation needs 0 < P < N")
    Q = float(M if Q is None else Q)
    rngs = [np.random.default_rng(s) for s in seeds]
    B = _teacher(rngs, N, M)
    Js = []
    for g, beta in zip(rngs, B):
        X = g.standard_normal((P, N))
        if method == "descent":
            Js.append(_descent(X, beta, 1.0, Q, g, constraint == "sphere"))
            continue
        U, _ = np.linalg.qr(X.T)                                            # (N, P) span of the data
        pb = U @ (U.T @ beta)
        if constraint == "none":
            Js.append(pb)
        elif algorithm == "optimal":
            Js.append(math.sqrt(Q) * pb / np.linalg.norm(pb))
        else:
            left = Q - pb @ pb
            if left < 0:
                raise Refusal("precondition:offline-window", "|P B|^2 > Q: no zero-error point on the sphere")
            w = g.standard_normal(N)
            w -= U @ (U.T @ w)
            Js.append(pb + math.sqrt(left) * w / np.linalg.norm(w))
    return _result(alpha, np.stack(Js), B, M, N, seeds)


__all__ = ["LinSim", "simulate_linear", "simulate_offline"]
