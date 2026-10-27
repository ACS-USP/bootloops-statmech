"""Detailed-balance and ergodicity tests of the samplers against exact Boltzmann weights (small tori), by
Pearson's chi^2 (stats.chi2_gof) with a stated alpha; cells = the classes of exact_small (n_sat and the
sorted colour counts), whose exact probabilities come from the enumeration (and, through the
energy histogram, Kaufman's integer polynomial at q = 2: exact_small.kaufman_histogram_identity).

  stationarity_chi2(algorithm, q, L, beta, samples, seed, alpha) -> stats.Chi2
      `samples` independent exact draws from the Boltzmann distribution (numpy default_rng(seed)
      choice over the q^N states), one update applied to each (SW: one sweep; Wolff: one cluster flip;
      Metropolis: one checkerboard sweep), the classes of the results against the exact class
      probabilities.  The samples are independent by construction, so chi^2 applies exactly; it tests
      pi P = pi (stationarity), which detailed balance implies.
  ergodicity_chi2(algorithm, q, L, beta, chains, steps, seed, alpha) -> stats.Chi2
      `chains` chains from the all-0 colouring (independent generators default_rng((seed, c))), `steps`
      updates each, the final classes against the exact probabilities: a check that pi P^steps from a
      fixed start is pi (it fails for a non-ergodic or non-stationary rule; `steps` must exceed the
      mixing time, which the test states)
Power (tests/test_mc_cluster.py, measured on named seeds 100-139 at beta_c, alpha = 0.01): a +-5% bond-probability
error is rejected 39/40 and 38/40 by SW (L = 3, 5000 samples) and 38/40, 40/40 by Wolff (L = 2, 3000 samples); 0/40
false rejections.  stationarity_chi2 passes any update that leaves the state unchanged; ergodicity_chi2 is the check
that fails it (planted: SW without recolouring, Wolff at p = 1; unplanted: Metropolis on the 2 x 2 torus).
"""
import numpy as np

from ..core import Refusal
from ..percolation.unionfind import UnionFind
from .cluster import bond_lists, neighbours, sw_sweep, wolff_step
from .exact_small import enumerate_torus, exact_distribution
from .fk import beta_value, bond_probability
from .stats import chi2_gof


def _indices(finals, q):
    a = np.asarray(finals, np.int64)
    return a @ (q ** np.arange(a.shape[1], dtype=np.int64))


def _updates(algorithm, q, L, beta, starts, steps, rng_of):
    N = L * L
    if algorithm == "metropolis":
        from .metropolis import metropolis_run
        if q != 2:
            raise Refusal("precondition:algorithm", "metropolis is the Ising (q = 2) single-spin flip")
        out = []
        b = float(beta_value(q, beta))
        spins = 1 - 2 * np.asarray(starts, np.int64)
        fin = metropolis_run(L, b, steps, (rng_of(0).integers(2 ** 31),), init=spins)["final"]
        return [((1 - row.ravel()) // 2).tolist() for row in fin]
    p = bond_probability(q, beta)
    u, v = bond_lists(L)
    nb = neighbours(L)
    uf = UnionFind(N)
    res = []
    for c, st in enumerate(np.asarray(starts).tolist()):
        s = st
        rng = rng_of(c)
        for _ in range(steps):
            if algorithm == "swendsen_wang":
                sw_sweep(s, q, p, u, v, rng, uf)
            elif algorithm == "wolff":
                wolff_step(s, q, p, nb, rng)
            else:
                raise Refusal("precondition:algorithm", f"algorithm = {algorithm!r}")
        res.append(s)
    return res


def _chi2(en, ex, finals, q, alpha):
    cls = en.cls[_indices(finals, q)]
    return chi2_gof(np.bincount(cls, minlength=en.n_cls), ex.class_probs, alpha)


def stationarity_chi2(algorithm, q, L, beta, samples, seed, alpha):
    en, ex = enumerate_torus(L, q), exact_distribution(L, q, beta)
    rng = np.random.default_rng(seed)
    idx = rng.choice(ex.probs.size, size=samples, p=ex.probs)
    finals = _updates(algorithm, q, L, beta, en.states[idx], 1, lambda c: rng)
    return _chi2(en, ex, finals, q, alpha)


def ergodicity_chi2(algorithm, q, L, beta, chains, steps, seed, alpha):
    en, ex = enumerate_torus(L, q), exact_distribution(L, q, beta)
    starts = np.zeros((chains, L * L), np.int64)
    finals = _updates(algorithm, q, L, beta, starts, steps, lambda c: np.random.default_rng((seed, c)))
    return _chi2(en, ex, finals, q, alpha)


__all__ = ["ergodicity_chi2", "stationarity_chi2"]
