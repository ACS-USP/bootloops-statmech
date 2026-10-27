"""The Erdos-Renyi critical window, sampled: the largest component C1 of G(N, p) with
p = 1/N + lambda N^(-4/3) (mean degree c = (N - 1) p) over named seeds, and the finite-size scaling
|C1| ~ N^(2/3).  Statistical; a consistency check, never a proven bound, except where a theorem is
quoted.

Sources: Nachmias & Peres, "The critical random graph, with martingales", arXiv:math/0512201v4 (journal
reference not checked) (NP; PDF read): Theorem 1 (p = 1/n, n > 1000, A > 8:
P(|C1| > A n^(2/3)) <= (4/A) exp(-A^2 (A - 4)/32)), Theorem 2 (P(|C1| < floor(delta n^(2/3))) <=
15 delta^(3/5) for 0 < delta < 1/10, n > 200/delta^(3/5)), and their statement of Luczak, Pittel &
Wierman (1994) (Theorem 3: the window p = 1/n + lambda n^(-4/3), N^(-2/3)|C1| converging in law).
Molloy & Reed, Random Struct. Alg. 6, 161 (1995), p. 161: |C1| = O(n^(2/3)) at M = n/2 + o(n).
Theorem 2 is informative only for delta < (1/15)^(5/3) = 0.011 (else its bound exceeds 1); it is
tested at delta = 1/100 (bound 0.946) on the sizes N > 200/delta^(3/5) = 3170.

  window_scaling(Ns, seeds, lam=0, offset=None) -> dict    (offset: a fixed c = 1 + offset, not scaled
      with N: the contrast to the window)
      scaled_agreement: whether mean |C1|/N^(2/3) at the smallest and the largest N agree within the
      99% Student-t band of their difference plus a stated systematic N_min^(-1/3) x the larger mean
      (relative corrections of order N^(-1/3): an assumption, not a bound).  Inside the window (fixed
      lambda) it should agree; a width other than N^(-1/3) (or a fixed offset) makes it drift with N,
      flagged `drifts` (printed and asserted only in the slow tests).
      per N: mean |C1| / N^(2/3) and its stderr; the log-slope of mean |C1| between the smallest
      and the largest N with its 99% Student-t band (delta method) plus a stated systematic
      N_min^(-1/3) / ln(N_max/N_min) (corrections of relative order N^(-1/3) to E|C1|/N^(2/3): an
      assumption, not a bound); whether 2/3 is inside slope +- (band + systematic), and whether
      1/2 and 3/4 are excluded (printed, not asserted); the NP Theorem 1 check at A = 9: the
      fraction of instances with |C1| > 9 N^(2/3) must not exceed the theorem's bound by more than
      its exact 99% Clopper-Pearson interval allows (else no-solution:np-theorem-1); likewise
      Theorem 2 at delta = 1/100 for |C1| < floor(N^(2/3)/100) (else no-solution:np-theorem-2).
"""
import math

import numpy as np

from ..core import Refusal
from . import sampling as SM
from .unionfind import UnionFind

A_NP, DELTA_NP = 9, 0.01
_HOOKS = {"window_exponent": -4 / 3}      # p = 1/N + lambda N^exponent; planted -5/4: a N^(-1/4) window


def largest_component(N, c, seed):
    rng = np.random.default_rng(seed)
    uf = UnionFind(N)
    uf.add_edges(SM._er_edges(N, c, rng).tolist())
    return uf.largest


def np_bound(A):
    return 4 / A * math.exp(-A * A * (A - 4) / 32)


def np_bound_small(delta):
    return 15 * delta ** 0.6


def window_scaling(Ns, seeds, lam=0.0, offset=None):
    from ..csp.xorsat_finite import clopper_pearson
    seeds = SM._seeds(seeds)
    Ns = sorted(int(n) for n in Ns)
    if len(Ns) < 2 or Ns[0] <= 1000:
        raise Refusal("precondition:n", f"Ns = {Ns}: need >= 2 sizes, each > 1000 (NP Theorem 1)")
    rows, above, total, below, total2 = [], 0, 0, 0, 0
    for N in Ns:
        c = 1 + offset if offset is not None else (N - 1) * (1 / N + lam * N ** _HOOKS["window_exponent"])
        c1 = np.array([largest_component(N, c, s) for s in seeds], dtype=float)
        if lam == 0 and offset is None:
            above += int(np.count_nonzero(c1 > A_NP * N ** (2 / 3)))
            total += c1.size
            if N > 200 / DELTA_NP ** 0.6:
                below += int(np.count_nonzero(c1 < math.floor(DELTA_NP * N ** (2 / 3))))
                total2 += c1.size
        r = c1 / N ** (2 / 3)
        rows.append({"N": N, "c": c, "mean_C1": float(c1.mean()), "se_C1": float(c1.std(ddof=1) / math.sqrt(c1.size)),
                     "scaled_mean": float(r.mean()), "scaled_se": float(r.std(ddof=1) / math.sqrt(r.size))})
    f, l = rows[0], rows[-1]
    ln_ratio = math.log(l["N"] / f["N"])
    slope = math.log(l["mean_C1"] / f["mean_C1"]) / ln_ratio
    se = math.hypot(f["se_C1"] / f["mean_C1"], l["se_C1"] / l["mean_C1"]) / ln_ratio
    tq = SM._t(0.995, len(seeds) - 1)
    band, sys = tq * se, f["N"] ** (-1 / 3) / ln_ratio
    out = {"rows": rows, "slope": slope, "band99": band, "systematic": sys,
           "contains_2_3": abs(slope - 2 / 3) <= band + sys,
           "excludes_1_2": abs(slope - 1 / 2) > band + sys, "excludes_3_4": abs(slope - 3 / 4) > band + sys,
           "kind": "finite-size consistency check (statistical; the systematic is an assumption)"}
    diff = abs(l["scaled_mean"] - f["scaled_mean"])
    aband = tq * math.hypot(f["scaled_se"], l["scaled_se"])
    asys = f["N"] ** (-1 / 3) * max(f["scaled_mean"], l["scaled_mean"])
    out["scaled_agreement"] = {"difference": diff, "band99": aband, "systematic": asys,
                               "agrees": diff <= aband + asys, "drifts": diff > aband + asys}
    if lam == 0 and offset is None:
        lo, _ = clopper_pearson(above, total)
        out["np_theorem_1"] = {"A": A_NP, "fraction_above": above / total, "cp99_lower": lo,
                               "bound": np_bound(A_NP)}
        if lo > np_bound(A_NP):
            raise Refusal("no-solution:np-theorem-1", f"{above}/{total} instances have |C1| > {A_NP} "
                          f"N^(2/3): the exact 99% lower bound {lo:.3g} exceeds Nachmias-Peres Theorem 1's "
                          f"{np_bound(A_NP):.3g} (a sampler error)")
        if total2:
            lo2, _ = clopper_pearson(below, total2)
            out["np_theorem_2"] = {"delta": DELTA_NP, "fraction_below": below / total2, "cp99_lower": lo2,
                                   "bound": np_bound_small(DELTA_NP)}
            if lo2 > np_bound_small(DELTA_NP):
                raise Refusal("no-solution:np-theorem-2", f"{below}/{total2} instances have |C1| < "
                              f"floor({DELTA_NP} N^(2/3)): the exact 99% lower bound {lo2:.3g} exceeds "
                              f"Nachmias-Peres Theorem 2's {np_bound_small(DELTA_NP):.3g}")
    return out


__all__ = ["largest_component", "np_bound", "np_bound_small", "window_scaling"]
