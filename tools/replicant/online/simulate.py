"""Finite-N online learning of a perceptron teacher over named seeds (an independent route).

It draws the examples and applies the update itself: it imports nothing from fields.py, odes.py
or optimal.py (tests/test_online_noise_sim.py inspects the imports), and its modulation functions
are written out here again.  Convention as odes.py: per seed a teacher B uniform on the unit sphere,
a student J0 = R0 B + sqrt(Q0 - R0^2) v (v a random unit vector orthogonal to B), and P = round(alpha N)
examples xi with i.i.d. N(0, 1) components; label b = sign(B . xi), flipped with probability `noise`;
J <- J + (eta / N) F(h, b, Q) b xi with h = J . xi, Q = J . J.  The optimal rule reads the current
rho = J . B / |J| exactly (it is the rule that knows rho, Kinouchi-Caticha 1992); a seed whose rho
reaches 0 or below (finite-N fluctuations of size 1/sqrt(N) when R0 is small; W* ~ 1/rho) is refused
undetermined:rho-sign, never averaged in.  eps of a student
is exact for Gaussian inputs: arccos(J . B / |J|) / pi, computed as atan2(|J - (J . B) B|, J . B) / pi.

Seeds: each seed owns its numpy Generator (default_rng(seed)); the seeds run side by side.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.special import log_ndtr

from ..amp._stats import Band, student_band
from ..core import Refusal
from ..csp import _checks

NAMES = ("hebb", "perceptron", "adatron", "optimal")
_BLOCK = 128


@dataclass(frozen=True)
class SimResult:
    alpha: float
    eps_mean: float
    band: Band                 # 99% Student-t band of the mean over seeds at alpha
    eps_seeds: tuple
    N: int
    seeds: tuple
    alphas: tuple              # checkpoints
    curve: tuple               # mean eps over seeds at each checkpoint


def _modulation(rule, h, b, Q, rho, est):
    stab = b * h
    if rule == "hebb":
        return np.ones_like(h)
    if rule == "perceptron":
        return (stab < 0).astype(float)
    if rule == "adatron":
        return np.where(stab < 0, -stab, 0.0)
    if rule == "optimal":
        j = np.sqrt(Q)
        s = np.sqrt(np.maximum(1 - rho * rho, 0.0))
        u = rho * (stab / j) / np.where(s > 0, s, 1.0)
        lg = log_ndtr(u) if est == 0 else np.logaddexp(math.log(est), math.log1p(-2 * est) + log_ndtr(u))
        w = (s / rho) * (1 - 2 * est) * np.exp(-0.5 * u * u - 0.5 * math.log(2 * math.pi) - lg)
        return j * np.where(s > 0, w, 0.0)
    return np.asarray(rule(h, b, Q), dtype=float) * np.ones_like(h)


def _eps(J, B):
    R = np.einsum("sn,sn->s", J, B)
    perp = np.linalg.norm(J - R[:, None] * B, axis=1)
    return np.arctan2(perp, R) / math.pi


def simulate_perceptron(rule, N, alpha_max, seeds, eta=1.0, noise=0.0, noise_estimate=None,
                        init=(0.0, 1.0), alphas=None):
    """SimResult of online learning at N over the named seeds (see the module docstring)."""
    if not callable(rule) and rule not in NAMES:
        raise Refusal("unsupported:rule", f"rule {rule!r}: {', '.join(NAMES)} or a callable F(h, b, Q)")
    if isinstance(N, bool) or not isinstance(N, int) or N < 10:
        raise Refusal("precondition:size", f"N={N!r}: need an int >= 10")
    seeds = _checks.seeds(seeds)
    if isinstance(eta, bool) or not (isinstance(eta, (int, float)) and math.isfinite(eta) and eta > 0):
        raise Refusal("precondition:eta", f"eta={eta!r}: need a finite learning rate > 0")
    if not (isinstance(noise, (int, float)) and 0 <= noise < 0.5):
        raise Refusal("precondition:noise", f"noise={noise!r}: a flip probability in [0, 1/2)")
    if noise_estimate is not None and rule != "optimal":
        raise Refusal("precondition:known-parameter", "noise_estimate is read by the optimal rule only")
    if noise_estimate is not None and not (isinstance(noise_estimate, (int, float))
                                           and 0 <= noise_estimate < 0.5):
        raise Refusal("precondition:noise", f"noise_estimate={noise_estimate!r}: in [0, 1/2)")
    if rule == "optimal" and eta != 1:
        raise Refusal("precondition:optimal-eta", "the optimal modulation is the optimum at eta = 1 only")
    est = float(noise if noise_estimate is None else noise_estimate)
    if not (isinstance(alpha_max, (int, float)) and math.isfinite(alpha_max) and alpha_max > 0):
        raise Refusal("precondition:alpha", f"alpha_max={alpha_max!r}: need alpha > 0")
    try:
        R0, Q0 = (float(v) for v in init)
    except (TypeError, ValueError):
        raise Refusal("precondition:state-point", f"init={init!r}: need (R0, Q0)") from None
    if not (math.isfinite(R0) and math.isfinite(Q0)) or not (Q0 > 0 and abs(R0) < math.sqrt(Q0)) or (rule == "optimal" and R0 <= 0):
        raise Refusal("precondition:initial-overlap", f"(R0, Q0) = ({R0}, {Q0}): need Q0 > 0, "
                      "|R0| < sqrt(Q0), and R0 > 0 for the optimal rule")
    marks = sorted({max(1, round(a * N)) for a in (alphas or (alpha_max,))})
    P = round(alpha_max * N)
    if marks[-1] > P:
        raise Refusal("precondition:alpha", "checkpoints beyond alpha_max")
    rngs = [np.random.default_rng(s) for s in seeds]
    B = np.stack([g.standard_normal(N) for g in rngs])
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    V = np.stack([g.standard_normal(N) for g in rngs])
    V -= np.einsum("sn,sn->s", V, B)[:, None] * B
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    J = R0 * B + math.sqrt(Q0 - R0 * R0) * V
    curve, at, step = [], 0, 0
    while step < P:
        k = min(_BLOCK, P - step)
        xs = np.stack([g.standard_normal((k, N)) for g in rngs], axis=1)     # (k, S, N)
        us = np.stack([g.random(k) for g in rngs], axis=1)                    # (k, S)
        for i in range(k):
            x = xs[i]
            y = np.einsum("sn,sn->s", B, x)
            h = np.einsum("sn,sn->s", J, x)
            b = np.where(y >= 0, 1.0, -1.0) * np.where(us[i] < noise, -1.0, 1.0)
            Q = np.einsum("sn,sn->s", J, J)
            rho = None
            if rule == "optimal":
                rho = np.einsum("sn,sn->s", J, B) / np.sqrt(Q)
                if np.any(rho <= 0):
                    bad = [seeds[i] for i in np.flatnonzero(rho <= 0)]
                    raise Refusal("undetermined:rho-sign", f"seeds {bad} reached rho <= 0 at alpha = "
                                  f"{step / N:.4g} (N = {N}): the optimal modulation is defined for "
                                  "rho > 0; start at R0 well above the 1/sqrt(N) fluctuations")
            f = _modulation(rule, h, b, Q, rho, est)
            J += (eta / N) * (f * b)[:, None] * x
            step += 1
            while at < len(marks) and step == marks[at]:
                curve.append(_eps(J, B))
                at += 1
    last = curve[-1]
    band = student_band(last)
    return SimResult(float(alpha_max), band.mean, band, tuple(float(e) for e in last), N, seeds,
                     tuple(m / N for m in marks), tuple(float(np.mean(c)) for c in curve))


__all__ = ["SimResult", "simulate_perceptron"]
