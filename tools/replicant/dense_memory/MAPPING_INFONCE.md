# Dense associative memory and the InfoNCE problem: what the `dense_memory` entry does and does not say

Scope: a note for the free-embedding / top-2 InfoNCE study (`limit-fig2-paper`, not edited). Sources read for this
work: Lucibello & Mézard, *The Exponential Capacity of Dense Associative Memories*, Phys. Rev. Lett. 132, 077301
(2024), arXiv:2304.14964v4 [LM], and Demircigil, Heusel, Löwe, Upgang & Vermet, J. Stat. Phys. 168, 288 (2017),
arXiv:1702.01929v2 [DHLUV] (both PDFs read; journal versions not checked). Every number below comes from
`dense_memory/infonce.py`, so `python3 -c` on that module reproduces it.

## 1. The dictionary (exact)

InfoNCE for a unit query `u` and `n` unit documents `v_x` in R^d,
`l_tau(u) = log sum_x exp(<u,v_x>/tau) - (<u,v_1> + <u,v_2>)/(2 tau)`,
has the softmax denominator of the dense memory of [LM] with

| memory | InfoNCE |
|---|---|
| N neurons | `d` |
| state `x`, `|x|^2 = N` (rho = 1) | `sqrt(d) u` |
| patterns `xi^x`, `|xi|^2 = N` | `sqrt(d) v_x` (spherical ensemble for documents uniform on the sphere) |
| `exp(lambda x.xi)` | `exp(<u,v>/tau)`, so **`lambda = 1/(tau d)`** |
| `P = e^(alpha N)` | `n = e^(alpha d)` |

Your regime (`d` = 4-45, `tau` = 0.02-1, `n` = 10-10^3) is `lambda` = 0.02-12: the finite-`lambda` part of the
phase diagram, not the `lambda -> 0` (annealed) corner it would reach at fixed `tau` and `d -> infinity`.

The two positives enter as a mixture state: `u = (v_1+v_2)/|v_1+v_2|` has `<u,v_a> = sqrt((1+c)/2)` with
`c = <v_1,v_2>`, i.e. `1/sqrt(2)` at `c = 0` (`infonce.mixture_level`).

## 2. What the entry computes under that dictionary

For a query **independent of the competing documents** (the condition under which [LM]'s Eq. (6) holds):

* the competing sum is `log sum_{x>=3} exp(<u,v_x>/tau) = (1/tau) phi_{alpha,1}(lambda)`, with `phi` the free energy of
  an auxiliary Random Energy Model (the 1RSB solution over document labels): annealed (`m* = 1`) for
  `lambda < lambda*(alpha)`, frozen on the largest overlap `eps*(alpha) = sqrt(1 - n^(-2/d))` above it (spherical);
* one positive (`u = v_1`, [LM] Eq. (7)): retrieval iff `phi < 1`, i.e. `n < exp(d alpha_1(lambda))` with
  `alpha_1 = lambda - zeta(lambda)` (spherical); at fixed `tau` and `d -> infinity` this is `n < e^(1/tau)`;
* **extension (not in [LM], labelled `not-computed`)**: replace the signal 1 by `level` (`1/sqrt(2)` for the mixture
  state): `phi < level`. The `tau -> 0` limit is the cap-covering number `n < (1-level^2)^(-d/2) = 2^(d/2)`.

## 3. Numbers, and where they fail

Mixture-state threshold (`level = 1/sqrt(2)`, `c = 0`) against the published critical `n` (Table `tab_tau`,
`n_1/2`). `ratio` = published / predicted.

`tau = 0.1`:

| d | lambda | published | predicted | ratio | phase at the prediction |
|---|---|---|---|---|---|
| 4 | 2.5 | 9.4 | 4.0 | 2.35 | frozen (cap 2^(d/2)) |
| 6 | 1.67 | 19.3 | 8.0 | 2.41 | frozen (cap) |
| 8 | 1.25 | 29 | 15.7 | 1.85 | annealed |
| 10 | 1.0 | 38 | 27.0 | 1.41 | annealed |
| 14 | 0.714 | 65 | 59.0 | 1.10 | annealed |
| 20 | 0.5 | 114 | 123 | 0.93 | annealed |
| 30 | 0.333 | 270 | 241 | 1.12 | annealed |

Dependence on `tau`:

| d | tau | published | predicted | cap 2^(d/2) |
|---|---|---|---|---|
| 30 | 0.3 | 100 | 8.8 | 32768 |
| 30 | 0.1 | 270 | 241 | 32768 |
| 30 | 0.05 | 725 | 4710 | 32768 |
| 10 | 0.3 | 22 | 6.2 | 32 |
| 10 | 0.05 | 90 | 32 (cap) | 32 |
| 10 | 0.02 | 231 | 32 (cap) | 32 |

**Reading.** At `tau = 0.1` and `d >= 14` the prediction is within 12% of the published values, but this is not
evidence that the entry predicts `n_c(d, tau)`: the `tau`-dependence is wrong in the way one expects of a
non-adapted query. The local exponent in `tau` between 0.3 and 0.1 at `d = 30` is 3.0 predicted against 0.90 published,
and between 0.1 and 0.05 it is 4.3 against 1.43; for small `tau` the prediction saturates at `2^(d/2)` while the
trained `n_c` keeps growing as `tau^(-p)`. Treat the `tau = 0.1` agreement as a coincidence of one slice until a
mechanism is shown.

## 4. What it does not say (do not over-claim)

1. **The published critical `n(d, tau)`.** That is the largest `n` at which *optimised* documents *and* queries
   return **every one of the `n(n-1)/2` pairs as a top-2 set** with 100% accuracy. The entry has: random i.i.d.
   documents, a fixed non-adapted query, a *typical* pair, and a criterion on softmax-weight dominance
   (`phi < level`) rather than on ranking. It has no union bound over pairs, no optimisation of the documents, and no
   optimiser efficiency `eta`.
2. **The per-query threshold `c(d) tau`.** That threshold is a property of the *optimised* query `u*(tau)`, which
   bends away from the competitors and ranks the pair correctly when its maximal margin `m` exceeds `c(d) tau`. The
   REM analysis assumes `u` independent of the competitors, the opposite of that adaptation. The right tool for random
   documents is the one already in your Appendix F (Gardner's perceptron with a pair-dependent margin); the REM noise
   is its non-adapted limit (your "extreme-value estimate" `m -> q - sqrt(2 log n/d)`), and the entry replaces
   `sqrt(2 alpha)` by the exact exponent-order value `sqrt(1 - e^(-2 alpha))` (for `d = 30`, `n = 270`: 0.558 instead
   of 0.611; sub-exponential prefactors are not captured, so this is a trend, not a finite-size prediction).
3. **A phase diagram of the memory at finite temperature.** [LM] is a zero-temperature retrieval analysis; the
   "temperature" and "energies" of the REM are auxiliary. There is no RS free energy of the memory, no replicon and
   no spin-glass/paramagnet transition of the memory in it (the paper names the finite-temperature landscape open).
   The REM condensation `lambda*(alpha)` is the only freezing line, and it concerns the competing sum.
4. **Generalisation to non-rotation-invariant or structured documents, or to a `lambda`-dependent embedding
   dimension.** Spherical and Gaussian ensembles only; binary documents need Ising neurons (`dense_memory_ising`).

## 5. What it does say that is usable

* The denominator `log sum_x exp(<u,v_x>/tau)` of random documents is `(1/tau) phi(1/(tau d))`, annealed for
  `lambda < lambda*(alpha)` and frozen on a few competitors above it. This gives the crossover `tau* = 1/(d lambda*(alpha))`
  below which the softmax over *random* documents is dominated by O(1) of them (not the same object as your `k`,
  the number of active constraints of the max-margin problem at the optimum).
* A hard, checkable cap for non-adapted mixture queries: `n < 2^(d/2)` at `c = 0` (`infonce.n_cap`). Your optimised
  arrangements exceed it (published `n_c = 90` at `d = 10, tau = 0.05`; the cap is 32): adaptation of the query and
  the documents is what lifts the capacity above it.
* The `tau_threshold(n, d, level)` inverse (largest `tau` below the REM threshold), refused
  `no-solution:no-retrieval` above the cap.

## 6. Finality

`dense_memory` (Gaussian/spherical, [LM]) values are `exact-by-literature` (non-rigorous REM analysis, as [LM] say);
the `level` extension, the finite-N sampler and everything about `dense_memory_ising` except
`alpha_fixed_points` ([DHLUV] Thm 1.3, proved for `lambda = 1` only) are `not-computed`. Polynomial interactions are
refused `unsupported:interaction`.

Reproduce: `python3 -m replicant run dense_memory --param alpha=0.2 --param lam=1.5`;
`from replicant.dense_memory import infonce as I; I.n_threshold(30, "0.1", I.mixture_level(0))`.
