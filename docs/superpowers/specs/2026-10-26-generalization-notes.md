# Generalising `replicant`: notes for a design (not yet brainstormed)

Status: notes agreed in conversation on 2026-10-26. This is not an approved design. Brainstorming starts after the online-learning series (O1–O7) ends.

## Why

Today a new model adapts only through the hand-written family hooks (`sk`, `perceptron`, `hopfield`). The aim is a system that adapts pre-existing techniques to new problems, and that says what it reused and where it stops.

## Items, in the order recommended

1. **A generic generalised-linear-model layer (prior × channel).** A model is a prior P0 and a channel P_out(y|z). From these two, the RS free entropy, the AMP state evolution and the online ODEs all follow from the same Gaussian averages over local fields. Barbier et al. 2019 proved the RS formula for GLMs; fetch it when the layer is designed.
2. **Checks generated from the spec.** The spec also yields a sampler, so every composed model gets an independent route for free: exact enumeration at small N, and Monte Carlo or online simulation at finite N.
3. **Finite-size scaling as a standard tool [RV: include].** It covers:
   - estimating x_c and exponents: Binder-cumulant crossings, and data collapse fitting x_c, 1/ν and the peak height, with bootstrap bands over named seeds;
   - corrections to scaling as parameters;
   - transition windows N^(−1/ν) for XORSAT, K-SAT and LDPC (the finite-length scaling of Amraoui–Montanari–Richardson–Urbanke, if fetched);
   - extrapolation with model selection (1/N, N^(−2/3), free N^(−ω)), where the spread between forms becomes the systematic band.

   It is validated on the exact certified 2D Ising tori from E9 (Kaufman), which must return ν = 1, γ/ν = 7/4 and β_c = ½ ln(1 + √2). An unresolvable exponent is reported as unresolved.

   It replaces the ad hoc extrapolations in E7 (ridge), E6a (LDPC crossing) and E11 (SK N^(−2/3), perceptron 1/N). It becomes the default finite-N check of item 2.
4. **Techniques as rewrite rules with preconditions.** Examples: Hubbard–Stratonovich, the Gaussian average, the saddle point, local-field integration, cavity, BP → AMP in the dense limit, AMP → state evolution, Plefka/TAP. The composer searches for a sequence of rules and names the missing precondition when none fits.
5. **An ansatz ladder:** RS → replicon → 1RSB → Gardner, taken automatically.
7. **A library of classical theorems as validators [RV: keep, 2026-10-26].** Each theorem is a check that any result of the right class must pass. Each has a positive control (a planted violation must fail) and is evaluated with balls where the data allow.
   - Thermodynamic identities: e = ∂(βf)/∂β, C = β² Var E ≥ 0, concavity of −βf in β and in h, and entropy ≥ 0 for discrete variables.
   - Inequalities: GKS (Griffiths), GHS (m concave in h ≥ 0), FKG, Jensen/annealed (E ln Z ≤ ln E Z), and the Gibbs–Bogoliubov variational bound.
   - Exact symmetries: Kramers–Wannier duality on the Onsager f and the E9 torus polynomials, and the Lee–Yang circle theorem (zeros of the exact ferromagnetic partition polynomials on |z| = 1).
   - Nishimori identities on the Bayes-optimal line, for teacher–student, spiked Wigner and SBM (for example q = m).
   - I-MMSE (Guo–Shamai–Verdú): the derivative of the mutual information equals the MMSE.
   - Spin-glass identities: Ghirlanda–Guerra and Aizenman–Contucci on finite-N simulations.
   - Existing pieces join it: the Guerra bounds (E10) and the LDPC area theorem (E6a).
   The same theorems can also serve as premise → conclusion rules for item 4.
   **7b. Lean behind the validators [RV: include, 2026-10-26].** The repository has no Lean project today. The machine has elan and Lean 4 (v4.31–4.33).
   - Level 1, instance checks: Python exports the exact data (E9 torus polynomials, exact counts, E8 rational series), and Lean checks the instance by `decide`, `norm_num` or `polyrith`. Examples: Kramers–Wannier duality as a polynomial identity, and the coefficient positivity GKS needs. Low cost; do it for every validator that touches exact data.
   - Level 2, finite-N theorems proved in Lean/Mathlib: Jensen (the annealed bound), C = β² Var E ≥ 0 and the convexity of ln Z in β, Gibbs–Bogoliubov, and the Nishimori identities (gauge symmetry). Medium cost; a small, chosen set.
   - Level 3, out of scope: thermodynamic-limit theorems (Lee–Yang in general, FKG, Guerra–Talagrand, online-ODE concentration, I-MMSE).
   - Each validator carries a provenance label: "cited theorem", "instance verified in Lean", or "proved in Lean for finite N".
8. **Agent-proposed hooks under a verifier.** A new hook is accepted only if an independent derivation agrees and the spec-generated simulation (item 2) confirms it.
