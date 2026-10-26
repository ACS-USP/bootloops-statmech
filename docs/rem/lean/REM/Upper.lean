import Mathlib

/-!
# Random Energy Model: the 1RSB (Guerra-type) upper bound

For `M` standard Gaussians `g i` (no independence needed) and `t : ℝ`,
`Z = ∑ i, exp (t * g i)`. For every `m ∈ (0, 1]`,

  `E log Z ≤ log M / m + m t² / 2`.

With `M = 2^N`, `t = β √(s N)` this reads `(1/N) E log Z_N ≤ log 2 / m + s β² m / 2`,
which is the replica 1RSB functional `Φ(m)`: the replica expression is a rigorous upper bound
at every `N` and every `m`, as in Guerra's broken-replica bound.

Proof: `m log Z = log Z^m ≤ Z^m / K - 1 + log K` (tangent line of `log` at `K`),
`Z^m ≤ ∑ exp (m t g i)` (subadditivity of `x ↦ x^m`, `m ≤ 1`), and
`E exp (m t g) = exp (m² t² / 2)` (Gaussian MGF); take `K = M exp (m² t² / 2)`.
-/

open MeasureTheory ProbabilityTheory Real Finset

namespace REM

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω}

/-- The partition function `∑ i, exp (t * g i ω)`. -/
noncomputable def Z {ι : Type*} [Fintype ι] (t : ℝ) (g : ι → Ω → ℝ) (ω : Ω) : ℝ :=
  ∑ i, exp (t * g i ω)

section Gaussian

variable {X : Ω → ℝ}

lemma integrable_comp_of_law (hX : Measurable X) (hlaw : P.map X = gaussianReal 0 1)
    {f : ℝ → ℝ} (hf : Measurable f) (hint : Integrable f (gaussianReal 0 1)) :
    Integrable (fun ω ↦ f (X ω)) P := by
  have : Integrable f (P.map X) := by rwa [hlaw]
  exact (integrable_map_measure hf.aestronglyMeasurable hX.aemeasurable).1 this

lemma integrable_of_law (hX : Measurable X) (hlaw : P.map X = gaussianReal 0 1) :
    Integrable X P :=
  integrable_comp_of_law hX hlaw measurable_id
    ((memLp_id_gaussianReal (μ := 0) (v := 1) 1).integrable le_rfl)

lemma integrable_exp_of_law (hX : Measurable X) (hlaw : P.map X = gaussianReal 0 1) (s : ℝ) :
    Integrable (fun ω ↦ exp (s * X ω)) P :=
  integrable_comp_of_law hX hlaw (by fun_prop) (integrable_exp_mul_gaussianReal s)

lemma integral_exp_of_law (_hX : Measurable X) (hlaw : P.map X = gaussianReal 0 1) (s : ℝ) :
    ∫ ω, exp (s * X ω) ∂P = exp (s ^ 2 / 2) := by
  have h := mgf_gaussianReal hlaw s
  simp only [mgf, NNReal.coe_one, zero_mul, one_mul, zero_add] at h
  exact h

end Gaussian

/-- `(∑ x i)^m ≤ ∑ (x i)^m` for nonnegative terms and `0 < m ≤ 1`. -/
lemma rpow_sum_le_sum_rpow {ι : Type*} (s : Finset ι) (x : ι → ℝ) (hx : ∀ i, 0 ≤ x i)
    {m : ℝ} (hm0 : 0 < m) (hm1 : m ≤ 1) :
    (∑ i ∈ s, x i) ^ m ≤ ∑ i ∈ s, x i ^ m := by
  classical
  induction s using Finset.induction_on with
  | empty => simp [Real.zero_rpow hm0.ne']
  | insert a s ha ih =>
    rw [sum_insert ha, sum_insert ha]
    calc (x a + ∑ i ∈ s, x i) ^ m ≤ x a ^ m + (∑ i ∈ s, x i) ^ m :=
          Real.rpow_add_le_add_rpow (hx a) (sum_nonneg fun i _ ↦ hx i) hm0.le hm1
      _ ≤ x a ^ m + ∑ i ∈ s, x i ^ m := by gcongr

variable {ι : Type*} [Fintype ι] {g : ι → Ω → ℝ}

lemma Z_pos [Nonempty ι] (t : ℝ) (ω : Ω) : 0 < Z t g ω :=
  sum_pos (fun _ _ ↦ exp_pos _) univ_nonempty

lemma measurable_Z (hmeas : ∀ i, Measurable (g i)) (t : ℝ) : Measurable (Z t g) := by
  unfold Z
  exact Finset.measurable_sum _ fun i _ ↦ (measurable_exp.comp ((hmeas i).const_mul t))

/-- Two-sided pointwise bound `|log Z| ≤ log M + |t| ∑ |g i|`. -/
lemma abs_log_Z_le [Nonempty ι] (t : ℝ) (ω : Ω) :
    |log (Z t g ω)| ≤ log (Fintype.card ι) + |t| * ∑ i, |g i ω| := by
  set S := ∑ i, |g i ω| with hS
  have hS0 : 0 ≤ S := sum_nonneg fun i _ ↦ abs_nonneg _
  have hcard : (1 : ℝ) ≤ Fintype.card ι := by exact_mod_cast Fintype.card_pos
  have hlogc : 0 ≤ log (Fintype.card ι) := log_nonneg hcard
  have hZ := Z_pos (g := g) t ω
  obtain ⟨i0⟩ := ‹Nonempty ι›
  have hgi : ∀ i, |g i ω| ≤ S := fun i ↦
    single_le_sum (f := fun j ↦ |g j ω|) (fun j _ ↦ abs_nonneg _) (mem_univ i)
  rw [abs_le]
  constructor
  · -- lower: log Z ≥ t g i0 ≥ -|t| S
    have h1 : exp (t * g i0 ω) ≤ Z t g ω :=
      single_le_sum (f := fun j ↦ exp (t * g j ω)) (fun j _ ↦ (exp_pos _).le) (mem_univ i0)
    have h2 : t * g i0 ω ≤ log (Z t g ω) := by
      have := log_le_log (exp_pos _) h1
      rwa [log_exp] at this
    have h3 : -(|t| * S) ≤ t * g i0 ω := by
      have : |t * g i0 ω| ≤ |t| * S := by
        rw [abs_mul]; exact mul_le_mul_of_nonneg_left (hgi i0) (abs_nonneg t)
      linarith [neg_abs_le (t * g i0 ω)]
    linarith
  · -- upper: Z ≤ M exp(|t| S)
    have h1 : Z t g ω ≤ Fintype.card ι * exp (|t| * S) := by
      unfold Z
      calc ∑ i, exp (t * g i ω) ≤ ∑ _i : ι, exp (|t| * S) := by
            apply sum_le_sum; intro i _
            apply exp_le_exp.2
            calc t * g i ω ≤ |t * g i ω| := le_abs_self _
              _ = |t| * |g i ω| := abs_mul _ _
              _ ≤ |t| * S := mul_le_mul_of_nonneg_left (hgi i) (abs_nonneg t)
        _ = Fintype.card ι * exp (|t| * S) := by simp
    have h2 := log_le_log hZ h1
    rw [log_mul (by positivity) (exp_pos _).ne', log_exp] at h2
    linarith

lemma integrable_log_Z [Nonempty ι] [IsProbabilityMeasure P] (hmeas : ∀ i, Measurable (g i))
    (hlaw : ∀ i, P.map (g i) = gaussianReal 0 1) (t : ℝ) :
    Integrable (fun ω ↦ log (Z t g ω)) P := by
  refine Integrable.mono' (g := fun ω ↦ log (Fintype.card ι) + |t| * ∑ i, |g i ω|) ?_ ?_ ?_
  · exact (integrable_const _).add
      ((integrable_finsetSum _ fun i _ ↦ (integrable_of_law (hmeas i) (hlaw i)).abs).const_mul _)
  · exact (measurable_log.comp (measurable_Z hmeas t)).aestronglyMeasurable
  · exact Filter.Eventually.of_forall fun ω ↦ by
      rw [Real.norm_eq_abs]; exact abs_log_Z_le t ω

/-- **Upper bound (1RSB / Guerra form).** For every `m ∈ (0,1]`,
`E log Z ≤ log M / m + m t² / 2`. -/
theorem upper_bound [Nonempty ι] [IsProbabilityMeasure P] (hmeas : ∀ i, Measurable (g i))
    (hlaw : ∀ i, P.map (g i) = gaussianReal 0 1) (t : ℝ) {m : ℝ} (hm0 : 0 < m) (hm1 : m ≤ 1) :
    ∫ ω, log (Z t g ω) ∂P ≤ log (Fintype.card ι) / m + m * t ^ 2 / 2 := by
  set M : ℝ := (Fintype.card ι : ℝ) with hM
  have hM0 : 0 < M := by rw [hM]; exact_mod_cast Fintype.card_pos
  set K : ℝ := M * exp ((m * t) ^ 2 / 2) with hK
  have hK0 : 0 < K := by positivity
  -- the dominating sum and its expectation
  set W : Ω → ℝ := fun ω ↦ ∑ i, exp ((m * t) * g i ω) with hW
  have hWint : Integrable W P :=
    integrable_finsetSum _ fun i _ ↦ integrable_exp_of_law (hmeas i) (hlaw i) _
  have hWexp : ∫ ω, W ω ∂P = K := by
    rw [hW, integral_finset_sum _ fun i _ ↦ integrable_exp_of_law (hmeas i) (hlaw i) _]
    simp only [integral_exp_of_law (hmeas _) (hlaw _), sum_const, card_univ, nsmul_eq_mul, hK, hM]
  -- Z^m ≤ W pointwise
  have hZW : ∀ ω, Z t g ω ^ m ≤ W ω := by
    intro ω
    calc Z t g ω ^ m ≤ ∑ i, exp (t * g i ω) ^ m :=
          rpow_sum_le_sum_rpow _ _ (fun i ↦ (exp_pos _).le) hm0 hm1
      _ = W ω := by
          refine sum_congr rfl fun i _ ↦ ?_
          rw [← Real.exp_mul]; ring_nf
  have hZm_meas : Measurable fun ω ↦ Z t g ω ^ m := (measurable_Z hmeas t).pow_const m
  have hZm_int : Integrable (fun ω ↦ Z t g ω ^ m) P := by
    refine Integrable.mono' hWint hZm_meas.aestronglyMeasurable ?_
    exact Filter.Eventually.of_forall fun ω ↦ by
      rw [Real.norm_eq_abs, abs_of_nonneg (Real.rpow_nonneg (Z_pos t ω).le _)]
      exact hZW ω
  have hZm_le : ∫ ω, Z t g ω ^ m ∂P ≤ K := hWexp ▸ integral_mono hZm_int hWint hZW
  -- pointwise tangent-line bound
  have hpt : ∀ ω, m * log (Z t g ω) ≤ Z t g ω ^ m / K - 1 + log K := by
    intro ω
    have hZ := Z_pos (g := g) t ω
    have hZm : 0 < Z t g ω ^ m := Real.rpow_pos_of_pos hZ m
    have h1 : log (Z t g ω ^ m / K) ≤ Z t g ω ^ m / K - 1 := log_le_sub_one_of_pos (by positivity)
    rw [log_div hZm.ne' hK0.ne', Real.log_rpow hZ] at h1
    linarith
  have hlog_int := integrable_log_Z hmeas hlaw t (P := P)
  have hint : m * ∫ ω, log (Z t g ω) ∂P ≤ log K := by
    rw [← integral_const_mul]
    calc ∫ ω, m * log (Z t g ω) ∂P ≤ ∫ ω, (Z t g ω ^ m / K - 1 + log K) ∂P :=
          integral_mono (hlog_int.const_mul m)
            (((hZm_int.div_const K).sub (integrable_const 1)).add (integrable_const _)) hpt
      _ = (∫ ω, Z t g ω ^ m ∂P) / K - 1 + log K := by
          have h1 : ∫ ω, (Z t g ω ^ m / K - 1 + log K) ∂P
              = ∫ ω, (Z t g ω ^ m / K - 1) ∂P + ∫ _ω, log K ∂P :=
            integral_add ((hZm_int.div_const K).sub (integrable_const 1)) (integrable_const _)
          have h2 : ∫ ω, (Z t g ω ^ m / K - 1) ∂P = ∫ ω, Z t g ω ^ m / K ∂P - ∫ _ω, (1 : ℝ) ∂P :=
            integral_sub (hZm_int.div_const K) (integrable_const 1)
          rw [h1, h2, integral_div]
          simp
      _ ≤ K / K - 1 + log K := by gcongr
      _ = log K := by rw [div_self hK0.ne']; ring
  have hlogK : log K = log M + m ^ 2 * t ^ 2 / 2 := by
    rw [hK, log_mul hM0.ne' (exp_pos _).ne', log_exp]; ring
  rw [hlogK] at hint
  rw [div_add' _ _ _ hm0.ne', le_div_iff₀ hm0]
  nlinarith [hint]

end REM
