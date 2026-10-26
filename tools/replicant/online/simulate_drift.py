"""Finite-N online learning of a DRIFTING perceptron teacher over named seeds (an independent route).

It draws the examples, applies the update and moves the teacher itself: it imports nothing from
fields.py, odes.py, optimal.py, drift.py or drift_rules.py (tests/test_online_drift_sim.py inspects the
imports).  The O1 rules' modulations come from simulate.py (which shares no code with the ODEs); the
VKC rules (annealed_hebb, step, symmetric; VKC 1998 Eqs. (22)-(24), chi = 2 noise) are written out
here again.  Convention as online/drift.py: per seed a unit teacher B, J0 = R0 B + sqrt(Q0 - R0^2) v,
then for each example xi ~ N(0, I_N): b = sign(B . xi) flipped with probability `noise`,
J <- J + (eta / N) F b xi (VKC rules: F = |J| W(b h / |J|; rho)), and the teacher moves:

  random         B <- (B + (sqrt(2 D) / N) u) / |.|, u ~ N(0, I_N) from the seed's generator
  deterministic  B <- cos(phi) B - sin(phi) e, e the unit vector of J - (J . B) B, 1 - cos(phi) = D / N^2
  jumps          at round(alpha_k N) examples B <- jump_overlap J/|J| + sqrt(1 - jump_overlap^2) v,
                 v a fresh unit vector orthogonal to J (the teacher replaced; the ODE's reset)

D may be a schedule D(alpha).  The rules that read rho (optimal, VKC) read the exact
rho = J . B / |J|; a seed reaching rho <= 0 is refused undetermined:rho-sign.  eps = arccos(rho)/pi
against the current teacher, exact for Gaussian inputs.  Bands: 99% Student-t over the seeds.
"""
import math

import numpy as np

from ..amp._stats import student_band
from ..core import Refusal
from ..csp import _checks
from .simulate import NAMES as O1_NAMES
from .simulate import SimResult, _eps, _modulation

VKC_NAMES = ("annealed_hebb", "step", "symmetric")
KINDS = ("random", "deterministic")
_BLOCK = 64


def _vkc(rule, d, rho, noise):
    """W per unit |J| of a VKC rule at stability d = b h / |J| and overlap rho (arrays over seeds)."""
    s = np.sqrt(np.maximum(1 - rho * rho, 0.0))
    lam = s / rho
    k = 1 - 2 * noise
    if rule == "annealed_hebb":
        return math.sqrt(2 / math.pi) * lam * lam * rho * k
    if rule == "step":
        den = noise + k * np.arccos(-rho * np.where(d > 0, 1.0, -1.0)) / math.pi
        return lam * lam * rho * k / math.sqrt(2 * math.pi) / den
    lam = np.where(lam > 0, lam, 1e-300)
    return math.sqrt(2 / math.pi) * lam * k * np.exp(-d * d / (2 * lam * lam))


def _unit_orthogonal(g, J):
    """A random unit vector orthogonal to each row of J (one draw per seed from its generator)."""
    v = np.stack([gi.standard_normal(J.shape[1]) for gi in g])
    v -= (np.einsum("sn,sn->s", v, J) / np.einsum("sn,sn->s", J, J))[:, None] * J
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def simulate_drift(rule, N, alpha_max, seeds, D=0.0, drift="random", eta=1.0, noise=0.0,
                   noise_estimate=None, init=(0.0, 1.0), alphas=None, jumps=(), jump_overlap=0.0):
    """SimResult of online learning of a drifting teacher at N over the named seeds."""
    if not callable(rule) and rule not in O1_NAMES + VKC_NAMES:
        raise Refusal("unsupported:rule", f"rule {rule!r}: {', '.join(O1_NAMES + VKC_NAMES)} or a callable F(h, b, Q)")
    if isinstance(N, bool) or not isinstance(N, int) or N < 10:
        raise Refusal("precondition:size", f"N={N!r}: need an int >= 10")
    seeds = _checks.seeds(seeds)
    if drift not in KINDS:
        raise Refusal("precondition:drift-kind", f"drift={drift!r}: {' | '.join(KINDS)}")
    if not callable(D) and not (isinstance(D, (int, float)) and math.isfinite(D) and D >= 0):
        raise Refusal("precondition:drift", f"D={D!r}: need D >= 0 or a schedule D(alpha)")
    if isinstance(eta, bool) or not (isinstance(eta, (int, float)) and math.isfinite(eta) and eta > 0):
        raise Refusal("precondition:eta", f"eta={eta!r}: need a finite learning rate > 0")
    if not (isinstance(noise, (int, float)) and 0 <= noise < 0.5):
        raise Refusal("precondition:noise", f"noise={noise!r}: a flip probability in [0, 1/2)")
    if noise_estimate is not None and rule != "optimal":
        raise Refusal("precondition:known-parameter", "noise_estimate is read by the optimal rule only")
    reads_rho = rule == "optimal" or rule in VKC_NAMES
    if reads_rho and eta != 1:
        raise Refusal("precondition:optimal-eta", f"{rule} is the optimum of its class at eta = 1 only")
    est = float(noise if noise_estimate is None else noise_estimate)
    if not (isinstance(alpha_max, (int, float)) and math.isfinite(alpha_max) and alpha_max > 0):
        raise Refusal("precondition:alpha", f"alpha_max={alpha_max!r}: need alpha > 0")
    try:
        R0, Q0 = (float(v) for v in init)
    except (TypeError, ValueError):
        raise Refusal("precondition:state-point", f"init={init!r}: need (R0, Q0)") from None
    if not (math.isfinite(R0) and math.isfinite(Q0) and Q0 > 0 and abs(R0) < math.sqrt(Q0)) or (reads_rho and R0 <= 0):
        raise Refusal("precondition:initial-overlap", f"(R0, Q0) = ({R0}, {Q0}): need Q0 > 0, |R0| < sqrt(Q0), "
                      "and R0 > 0 for a rule that reads rho")
    P = round(alpha_max * N)
    marks = sorted({max(1, round(a * N)) for a in (alphas or (alpha_max,))})
    jump_at = sorted({round(j * N) for j in jumps})
    if marks[-1] > P or any(not 0 < j < P for j in jump_at):
        raise Refusal("precondition:alpha", "checkpoints and jumps must lie in (0, alpha_max]")
    if jump_at and not (abs(jump_overlap) < 1 and (jump_overlap > 0 or not reads_rho)):
        raise Refusal("precondition:initial-overlap", f"jump_overlap={jump_overlap!r}: |rho| < 1, > 0 for a rule "
                      "that reads rho")
    rngs = [np.random.default_rng(s) for s in seeds]
    B = np.stack([g.standard_normal(N) for g in rngs])
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    J = R0 * B + math.sqrt(Q0 - R0 * R0) * _unit_orthogonal(rngs, B)
    curve, at, step = [], 0, 0
    while step < P:
        k = min(_BLOCK, P - step)
        xs = np.stack([g.standard_normal((k, N)) for g in rngs], axis=1)                 # (k, S, N)
        us = np.stack([g.random(k) for g in rngs], axis=1)                                # (k, S)
        ws = np.stack([g.standard_normal((k, N)) for g in rngs], axis=1) if drift == "random" else None
        for i in range(k):
            x = xs[i]
            y = np.einsum("sn,sn->s", B, x)
            h = np.einsum("sn,sn->s", J, x)
            b = np.where(y >= 0, 1.0, -1.0) * np.where(us[i] < noise, -1.0, 1.0)
            Q = np.einsum("sn,sn->s", J, J)
            rho = None
            if reads_rho:
                rho = np.einsum("sn,sn->s", J, B) / np.sqrt(Q)
                if np.any(rho <= 0):
                    bad = [seeds[j] for j in np.flatnonzero(rho <= 0)]
                    raise Refusal("undetermined:rho-sign", f"seeds {bad} reached rho <= 0 at alpha = {step / N:.4g} "
                                  f"(N = {N}): the rule reads 1/rho; start further from rho = 0")
            if rule in VKC_NAMES:
                f = np.sqrt(Q) * _vkc(rule, b * h / np.sqrt(Q), rho, noise)
            else:
                f = _modulation(rule, h, b, Q, rho, est)
            J += (eta / N) * (f * b)[:, None] * x
            d = D(step / N) if callable(D) else D
            if d > 0 and drift == "random":
                B += (math.sqrt(2 * d) / N) * ws[i]
                B /= np.linalg.norm(B, axis=1, keepdims=True)
            elif d > 0:
                e = J - np.einsum("sn,sn->s", J, B)[:, None] * B
                e /= np.linalg.norm(e, axis=1, keepdims=True)
                phi = 2 * math.asin(math.sqrt(d / 2) / N)
                B = math.cos(phi) * B - math.sin(phi) * e
            step += 1
            while at < len(marks) and step == marks[at]:          # checkpoints are taken before a jump
                curve.append(_eps(J, B))
                at += 1
            if step in jump_at:
                jn = J / np.linalg.norm(J, axis=1, keepdims=True)
                B = jump_overlap * jn + math.sqrt(1 - jump_overlap ** 2) * _unit_orthogonal(rngs, J)
    last = curve[-1]
    band = student_band(last)
    return SimResult(float(alpha_max), band.mean, band, tuple(float(e) for e in last), N, seeds,
                     tuple(m / N for m in marks), tuple(float(np.mean(c)) for c in curve))


__all__ = ["simulate_drift"]
