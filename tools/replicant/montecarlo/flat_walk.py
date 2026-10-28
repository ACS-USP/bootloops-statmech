"""The single-spin-flip random walk in energy shared by Wang-Landau and the multicanonical run (M2): independent
walkers on the L x L Ising torus, numpy-vectorised over walkers, each walker driven only by its own named seed.

Move.  A site is chosen uniformly (Wang & Landau, arXiv:cond-mat/0011174v1 (2001), PDF read: "flipping spins
randomly"), the flip from level k1 to k2 accepted with probability min(1, g(E1) / g(E2)) (their Eq. (1);
Belardinelli & Pereyra, PRE 75, 046701 (2007), arXiv:cond-mat/0701672, PDF read, Eq. (1)); written with
S = ln g: accept iff ln u < S(k1) - S(k2).  With fixed S this is Metropolis for the weight exp(-S(E)) (the
multicanonical weight of Berg, arXiv:cond-mat/9909236, PDF read, Eqs. (1.5), (1.6), (2.11)).  After the move
(accepted or not) the walker's current level k gets H(k) += 1 and, when learning, S(k) += F (WL: g -> g f,
F = ln f; BP step (iii)): a rejected move counts the old level (Berg below Eq. (2.12): "rejected proposals
have to be counted").
Levels k = n_unsat / 2 (dos_exact.py), E = 4k - 2N; on a flip dk = (# equal neighbours) - 2.

Named seeds.  Walker block b (walkers b W .. b W + W - 1) draws every site and uniform from
numpy default_rng(seeds[b]) only (an int, or a tuple (seed, tag) for a second stream), in chunks whose
size depends only on W, so a seed's walkers do not depend on the other seeds
(tests: a one-seed run reproduces that seed's block).  Starting spins iid +-1 from the same generator.

  Walkers(L, seeds, walkers_per_seed) -> state (spins, level k, magnetisation M per walker)
  acceptance_probability(S1, S2) -> min(1, exp(S1 - S2)) (the rule the walk uses, hook included)
  Walkers.run(steps, S, F=None, H=None, msum=None)    `steps` moves of every walker; S (R, K) flat log weights;
      F (R,) learning increments (None: fixed weights); H (R, K) visit counts; msum (R, K) sum of |M| per level
_HOOKS["acceptance_power"] (1.0; planted e.g. 0.9: accept iff ln u < a (S1 - S2), a wrong rule),
_HOOKS["sites"] ("all"; planted "half": only the first N/2 sites are ever proposed, a non-ergodic walk).
Exact invariant (no-solution:sampler-invariant, checked by Walkers.check): the tracked levels and
magnetisations equal those recomputed from the spins.
"""
import numpy as np

from ..core import Refusal
from .exact_small import torus_bonds

_HOOKS = {"acceptance_power": 1.0, "sites": "all"}
CHUNK = 2048
CHUNK_DRAWS = 1 << 16


def neighbour_array(L):
    u, v = torus_bonds(L)
    nb = [[] for _ in range(L * L)]
    for a, b in zip(u.tolist(), v.tolist()):
        nb[a].append(b)
        nb[b].append(a)
    return np.array(nb, dtype=np.int64)


def acceptance_probability(S1, S2):
    """min(1, g(E1) / g(E2)) with S = ln g (WL Eq. (1)); the walk accepts iff ln u < min(0, a (S1 - S2))."""
    return float(np.exp(min(0.0, float(_HOOKS["acceptance_power"]) * (float(S1) - float(S2)))))


def levels_of(spins, L):
    """k = n_unsat / 2 of each row of spins (R, N)."""
    u, v = torus_bonds(L)
    return (spins[:, u] != spins[:, v]).sum(axis=1) // 2


class Walkers:
    def __init__(self, L, seeds, walkers_per_seed, init=None):
        self.L, self.N, self.K = L, L * L, L * L + 1
        self.seeds, self.W = tuple(seeds), int(walkers_per_seed)       # ints, or tuples for default_rng
        self.R = len(self.seeds) * self.W
        self.rngs = [np.random.default_rng(s) for s in self.seeds]
        if init is None:
            sp = np.concatenate([g.choice(np.array([-1, 1], np.int64), size=(self.W, self.N)) for g in self.rngs])
        else:
            sp = np.asarray(init, np.int64).reshape(self.R, self.N).copy()
        self.nb = neighbour_array(L)
        self.off = np.arange(self.R, dtype=np.int64) * self.N
        self.koff = np.arange(self.R, dtype=np.int64) * self.K
        self.s = sp.ravel()
        self.k = levels_of(sp, L)
        self.M = sp.sum(axis=1)
        self._sites = self._logu = None
        self.chunk = int(min(CHUNK, max(16, CHUNK_DRAWS // self.W)))     # rows per refill: fixed by W
        self._pos = self.chunk

    def _refill(self):
        n_prop = self.N if _HOOKS["sites"] == "all" else self.N // 2
        sites = np.concatenate([g.integers(n_prop, size=(self.chunk, self.W)) for g in self.rngs], axis=1)
        self._logu = np.log(np.concatenate([g.random((self.chunk, self.W)) for g in self.rngs], axis=1))
        self._gi = sites + self.off                      # flat site index per row, walker
        self._nbi = self.nb[sites] + self.off[:, None]   # its four neighbours
        self._pos = 0

    def run(self, steps, S, F=None, H=None, msum=None):
        a = float(_HOOKS["acceptance_power"])
        s = self.s
        kf, M = self.koff + self.k, self.M                # flat index of each walker's current level
        for _ in range(int(steps)):
            if self._pos == self.chunk:
                self._refill()
            p = self._pos
            self._pos += 1
            gi = self._gi[p]
            si = s[gi]
            dk = si * s[self._nbi[p]].sum(axis=1) // 2    # (# equal neighbours) - 2
            acc = self._logu[p] < np.minimum(0.0, a * (S[kf] - S[kf + dk]))
            s[gi] = np.where(acc, -si, si)
            kf = np.where(acc, kf + dk, kf)
            if F is not None:
                S[kf] += F
            if H is not None:
                H[kf] += 1
            if msum is not None:
                M = M - 2 * si * acc
                msum[kf] += np.abs(M)
        self.k = kf - self.koff
        if msum is not None:
            self.M = M
        else:
            self.M = self.spins().sum(axis=1)

    def spins(self):
        return self.s.reshape(self.R, self.N)

    def check(self):
        sp = self.spins()
        if not (np.array_equal(self.k, levels_of(sp, self.L)) and np.array_equal(self.M, sp.sum(axis=1))):
            raise Refusal("no-solution:sampler-invariant", "flat-histogram walk: tracked level or magnetisation "
                          "!= the spins'")


__all__ = ["CHUNK", "Walkers", "acceptance_probability", "levels_of", "neighbour_array"]
