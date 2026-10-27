"""Single-spin-flip Metropolis for the 2D Ising model on the L x L torus (the local-update baseline of the
critical-slowing-down comparison), numpy-vectorised over independent chains, in the style of
numerics/pt.py (E11).

A sweep updates the two checkerboard sublattices in turn ((x + y) even, then odd); every spin of the
current sublattice is proposed for a flip with dE = 2 s h (h the sum of its four neighbours, which all
lie on the other sublattice, so the simultaneous proposals are independent) and accepted with
probability min(1, exp(-beta dE)).  Each sublattice update leaves the Boltzmann measure invariant, so
the sweep does (the argument of pt.py's sequential sweep); L must be even (an odd torus breaks the
checkerboard: precondition:checkerboard).  Nightingale & Blote's z (arXiv:cond-mat/9601059) is for
random-site heat-bath dynamics; the dynamic exponent is taken as universal across these local
single-spin dynamics (their 1998 follow-up, not fetched), a physics-level assumption stated here.

  metropolis_run(L, beta, sweeps, seeds, init=None) -> dict of (runs, sweeps) arrays: "E" (energy,
      measured before each sweep), "M" (sum s), and "final" spins; spins start iid +-1 per seed (numpy
      default_rng(seed)), or at `init` (R, L, L) with one generator default_rng(seeds[0]) for all chains
Exact invariant (no-solution:sampler-invariant): the energy tracked through the accepted dE equals
the energy recomputed from the final spins, in every chain.
_HOOKS["acceptance_scale"] (tests): 1.0; planted 0.5 (accepts with exp(-beta dE / 2): a wrong rule).
"""
import numpy as np

from ..core import Refusal

_HOOKS = {"acceptance_scale": 1.0}


def _energy(s):
    return -(s * (np.roll(s, 1, 1) + np.roll(s, 1, 2))).sum(axis=(1, 2))


def metropolis_run(L, beta, sweeps, seeds, init=None):
    if L % 2 or L < 2:
        raise Refusal("precondition:checkerboard", f"L = {L}: the checkerboard sweep needs an even L >= 2")
    seeds = tuple(int(x) for x in seeds)
    if init is None:
        rngs = [np.random.default_rng(x) for x in seeds]
        s = np.stack([g.choice(np.array([-1, 1], np.int64), size=(L, L)) for g in rngs])
        draw = lambda: np.stack([g.random((L, L)) for g in rngs])
    else:                       # given starting spins (R, L, L): one generator, seeds[0], for all chains
        s = np.asarray(init, np.int64).reshape(-1, L, L).copy()
        g = np.random.default_rng(seeds[0])
        draw = lambda: g.random(s.shape)
    R = s.shape[0]
    x, y = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    masks = [((x + y) % 2 == c)[None] for c in (0, 1)]
    b = _HOOKS["acceptance_scale"] * float(beta)
    acc_of = {d: np.exp(-b * d) for d in (4, 8)}
    E = _energy(s).astype(np.int64)
    Es, Ms = np.empty((R, sweeps), np.int64), np.empty((R, sweeps), np.int64)
    for t in range(sweeps):
        Es[:, t], Ms[:, t] = E, s.sum(axis=(1, 2))
        for m in masks:
            h = np.roll(s, 1, 1) + np.roll(s, -1, 1) + np.roll(s, 1, 2) + np.roll(s, -1, 2)
            dE = 2 * s * h
            u = draw()
            prob = np.where(dE <= 0, 1.0, np.where(dE == 4, acc_of[4], acc_of[8]))
            flip = m & (u < prob)
            E = E + (dE * flip).sum(axis=(1, 2))
            s = np.where(flip, -s, s)
    if not np.array_equal(E, _energy(s)):
        raise Refusal("no-solution:sampler-invariant", "Metropolis: tracked energy != recomputed energy")
    return {"E": Es, "M": Ms, "final": s}


__all__ = ["metropolis_run"]
