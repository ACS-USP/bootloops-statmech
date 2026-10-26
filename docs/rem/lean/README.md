# REM: Derrida's free energy, formally verified

Lean 4 + Mathlib proof that for the Random Energy Model

    (1/N) E log Σ_{i ≤ 2^N} exp(a √N g_i)  →  log 2 + a²/2      (a ≤ √(2 log 2))
                                            a √(2 log 2)       (a > √(2 log 2))

for standard Gaussians `g_i` that are pairwise independent, `a = β √s ≥ 0`.

| file | content |
|---|---|
| `REM/Upper.lean` | `upper_bound`: E log Z ≤ log M / m + m t²/2 for every N and m ∈ (0,1] (the 1RSB functional as a rigorous bound, Guerra form; no independence used) |
| `REM/Lower.lean` | `lower_bound`: second-moment / Chebyshev bound; `tail_ge`: Gaussian tail lower bound |
| `REM/Limit.lean` | `tendsto_freeEntropy` (the theorem), `pressure_beta` (Derrida's form with β_c = √(2 log 2 / s)) |
| `REM/Model.lean` | `tendsto_freeEntropy_canonical`: the hypotheses hold on the product of Gaussians |

Build: `lake build`, then `lake env lean Check.lean` prints the axioms
(only `propext`, `Classical.choice`, `Quot.sound`; no `sorry`).

Mathlib is taken from the local build at `~/Projects/learn-lean` (path dependency in
`lakefile.toml`; the other packages are symlinked into `.lake/packages`). For a portable build
replace the path dependency with the git one at the same rev and run `lake exe cache get`.
