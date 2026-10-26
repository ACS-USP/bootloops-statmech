import REM.Upper

/-!
# Random Energy Model: the second-moment lower bound

Let `K = #{i : g i ≥ θ}` with the `g i` standard Gaussians, pairwise independent. Then
`E K = M p` with `p = γ[θ, ∞)` and `Var K ≤ M p`, so by Chebyshev `K > M p / 2` except on a set
of probability at most `4 / (M p)`. On the good set `Z ≥ K e^{tθ}`; everywhere `log Z ≥ t g i₀`.
Hence, with `q = M p` and `L = log (q / 2) + t θ ≥ 0`,

  `E log Z ≥ L (1 - 4 / q) - t E|g|`.

The Gaussian tail is bounded below by `p ≥ exp (-(θ+1)²/2) / √(2π)` (the density on `[θ, θ+1]`).
-/

open MeasureTheory ProbabilityTheory Real Finset

namespace REM

/-- The standard Gaussian tail `γ[θ, ∞)`. -/
noncomputable def tail (θ : ℝ) : ℝ := (gaussianReal 0 1).real (Set.Ici θ)

/-- `E|g|` for a standard Gaussian. -/
noncomputable def absMean : ℝ := ∫ x, |x| ∂(gaussianReal 0 1)

lemma tail_ge {θ : ℝ} (hθ : 0 ≤ θ) : exp (-(θ + 1) ^ 2 / 2) / √(2 * π) ≤ tail θ := by
  have hsub : Set.Icc θ (θ + 1) ⊆ Set.Ici θ := fun x hx ↦ hx.1
  have h1 : (gaussianReal 0 1).real (Set.Icc θ (θ + 1)) ≤ tail θ :=
    measureReal_mono hsub
  have h2 : (gaussianReal 0 1).real (Set.Icc θ (θ + 1))
      = ∫ x in Set.Icc θ (θ + 1), gaussianPDFReal 0 1 x := by
    rw [measureReal_def, gaussianReal_apply_eq_integral 0 one_ne_zero, ENNReal.toReal_ofReal]
    exact setIntegral_nonneg measurableSet_Icc fun x _ ↦ gaussianPDFReal_nonneg _ _ _
  have h3 : gaussianPDFReal 0 1 (θ + 1) * volume.real (Set.Icc θ (θ + 1))
      ≤ ∫ x in Set.Icc θ (θ + 1), gaussianPDFReal 0 1 x := by
    refine setIntegral_ge_of_const_le_real measurableSet_Icc (by simp) ?_
      (integrable_gaussianPDFReal 0 1).integrableOn
    intro x hx
    simp only [gaussianPDFReal_def, NNReal.coe_one, mul_one, sub_zero]
    gcongr <;> linarith [hx.1, hx.2]
  have h4 : volume.real (Set.Icc θ (θ + 1)) = 1 := by simp
  have h5 : gaussianPDFReal 0 1 (θ + 1) = exp (-(θ + 1) ^ 2 / 2) / √(2 * π) := by
    simp only [gaussianPDFReal_def, NNReal.coe_one, mul_one, sub_zero]
    ring
  rw [h4, mul_one, h5] at h3
  linarith

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]
variable {ι : Type*} [Fintype ι] {g : ι → Ω → ℝ}

/-- **Lower bound (second moment).** -/
theorem lower_bound [Nonempty ι] (hmeas : ∀ i, Measurable (g i))
    (hlaw : ∀ i, P.map (g i) = gaussianReal 0 1)
    (hind : Pairwise fun i j ↦ IndepFun (g i) (g j) P)
    {t θ : ℝ} (ht : 0 ≤ t)
    (hq : 0 < (Fintype.card ι : ℝ) * tail θ)
    (hL : 0 ≤ log ((Fintype.card ι : ℝ) * tail θ / 2) + t * θ) :
    (log ((Fintype.card ι : ℝ) * tail θ / 2) + t * θ) * (1 - 4 / ((Fintype.card ι : ℝ) * tail θ))
      - t * absMean ≤ ∫ ω, log (Z t g ω) ∂P := by
  set q : ℝ := (Fintype.card ι : ℝ) * tail θ with hqdef
  set L : ℝ := log (q / 2) + t * θ with hLdef
  -- indicators of the high levels
  set f : ℝ → ℝ := (Set.Ici θ).indicator 1 with hf
  have hfmeas : Measurable f := (measurable_const).indicator measurableSet_Ici
  have hf01 : ∀ x, f x = 0 ∨ f x = 1 := by
    intro x; by_cases hx : x ∈ Set.Ici θ <;> simp [hf, hx]
  have hfsq : ∀ x, f x ^ 2 = f x := by
    intro x; rcases hf01 x with h | h <;> simp [h]
  have hfabs : ∀ x, ‖f x‖ ≤ 1 := by
    intro x; rcases hf01 x with h | h <;> simp [h]
  set Y : ι → Ω → ℝ := fun i ω ↦ f (g i ω) with hY
  have hYmeas : ∀ i, Measurable (Y i) := fun i ↦ hfmeas.comp (hmeas i)
  have hYL2 : ∀ i, MemLp (Y i) 2 P := fun i ↦
    MemLp.of_bound (hYmeas i).aestronglyMeasurable 1 (Filter.Eventually.of_forall fun ω ↦ hfabs _)
  have hYint : ∀ i, Integrable (Y i) P := fun i ↦ (hYL2 i).integrable one_le_two
  have hYexp : ∀ i, ∫ ω, Y i ω ∂P = tail θ := by
    intro i
    have : ∫ ω, Y i ω ∂P = ∫ x, f x ∂(P.map (g i)) :=
      (integral_map (hmeas i).aemeasurable hfmeas.aestronglyMeasurable).symm
    rw [this, hlaw i, hf, integral_indicator_one measurableSet_Ici]
    rfl
  set K : Ω → ℝ := fun ω ↦ ∑ i, Y i ω with hK
  have hKmeas : Measurable K := Finset.measurable_sum _ fun i _ ↦ hYmeas i
  have hKL2 : MemLp K 2 P := by
    have := memLp_finsetSum' (s := (univ : Finset ι)) (fun i _ ↦ hYL2 i)
    convert this using 1; ext ω; simp [hK]
  have hKexp : ∫ ω, K ω ∂P = q := by
    rw [hK, integral_finsetSum _ fun i _ ↦ hYint i]
    simp [hYexp, hqdef]
  -- variance
  have hvar : Var[K; P] ≤ q := by
    have hsum : Var[K; P] = ∑ i, Var[Y i; P] := by
      have := IndepFun.variance_sum (μ := P) (X := Y) (s := univ) (fun i _ ↦ hYL2 i)
        (fun i _ j _ hij ↦ (hind hij).comp hfmeas hfmeas)
      convert this using 2; ext ω; simp [hK]
    rw [hsum, hqdef]
    have : ∀ i, Var[Y i; P] ≤ tail θ := by
      intro i
      calc Var[Y i; P] ≤ P[Y i ^ 2] := variance_le_expectation_sq (hYmeas i).aestronglyMeasurable
        _ = tail θ := by
          rw [← hYexp i]; congr 1; funext ω; simp [hY, hfsq]
    calc ∑ i, Var[Y i; P] ≤ ∑ _i : ι, tail θ := sum_le_sum fun i _ ↦ this i
      _ = (Fintype.card ι : ℝ) * tail θ := by simp
  -- Chebyshev
  set B : Set Ω := {ω | q / 2 ≤ |K ω - ∫ ω', K ω' ∂P|} with hB
  have hBmeas : MeasurableSet B :=
    measurableSet_le measurable_const ((hKmeas.sub measurable_const).abs)
  have hBle : P.real B ≤ 4 / q := by
    have h := meas_ge_le_variance_div_sq hKL2 (c := q / 2) (by positivity)
    have h' : Var[K; P] / (q / 2) ^ 2 ≤ 4 / q := by
      rw [div_le_div_iff₀ (by positivity) hq]
      nlinarith [hvar, variance_nonneg K P]
    calc P.real B = (P B).toReal := rfl
      _ ≤ (ENNReal.ofReal (Var[K; P] / (q / 2) ^ 2)).toReal :=
          ENNReal.toReal_mono ENNReal.ofReal_ne_top h
      _ = Var[K; P] / (q / 2) ^ 2 :=
          ENNReal.toReal_ofReal (div_nonneg (variance_nonneg K P) (by positivity))
      _ ≤ 4 / q := h'
  have hGood : P.real Bᶜ ≥ 1 - 4 / q := by
    rw [probReal_compl_eq_one_sub hBmeas]; linarith
  -- pointwise lower bound
  obtain ⟨i0⟩ := ‹Nonempty ι›
  have hpt : ∀ ω, Bᶜ.indicator (fun _ ↦ L) ω - t * |g i0 ω| ≤ log (Z t g ω) := by
    intro ω
    have hZ := Z_pos (g := g) t ω
    have hlow : -(t * |g i0 ω|) ≤ log (Z t g ω) := by
      have h1 : exp (t * g i0 ω) ≤ Z t g ω :=
        single_le_sum (f := fun j ↦ exp (t * g j ω)) (fun j _ ↦ (exp_pos _).le) (mem_univ i0)
      have h2 := log_le_log (exp_pos _) h1
      rw [log_exp] at h2
      have h3 : -(t * |g i0 ω|) ≤ t * g i0 ω := by
        have := neg_abs_le (g i0 ω); nlinarith
      linarith
    by_cases hω : ω ∈ B
    · simp only [Set.indicator_of_notMem (Set.notMem_compl_iff.2 hω)]; linarith
    · rw [Set.indicator_of_mem (Set.mem_compl hω)]
      have hKq : q / 2 < K ω := by
        simp only [hB, Set.mem_setOf_eq, not_le, hKexp] at hω
        have := (abs_lt.1 hω).1; linarith
      have hZK : K ω * exp (t * θ) ≤ Z t g ω := by
        rw [hK, sum_mul]
        refine sum_le_sum fun i _ ↦ ?_
        by_cases hi : θ ≤ g i ω
        · have : Y i ω = 1 := by simp [hY, hf, Set.indicator_of_mem (Set.mem_Ici.2 hi)]
          rw [this, one_mul]; exact exp_le_exp.2 (mul_le_mul_of_nonneg_left hi ht)
        · have : Y i ω = 0 := by
            simp [hY, hf, Set.indicator_of_notMem (show g i ω ∉ Set.Ici θ from hi)]
          rw [this, zero_mul]; exact (exp_pos _).le
      have hq2 : 0 < q / 2 := by positivity
      have h1 : q / 2 * exp (t * θ) ≤ Z t g ω :=
        le_trans (mul_le_mul_of_nonneg_right hKq.le (exp_pos _).le) hZK
      have h2 := log_le_log (by positivity) h1
      rw [log_mul hq2.ne' (exp_pos _).ne', log_exp] at h2
      have : 0 ≤ t * |g i0 ω| := mul_nonneg ht (abs_nonneg _)
      linarith
  -- integrate
  have hgabs : ∫ ω, |g i0 ω| ∂P = absMean := by
    have : ∫ ω, |g i0 ω| ∂P = ∫ x, |x| ∂(P.map (g i0)) :=
      (integral_map (hmeas i0).aemeasurable continuous_abs.aestronglyMeasurable).symm
    rw [this, hlaw i0]; rfl
  have hlowint : Integrable (fun ω ↦ Bᶜ.indicator (fun _ ↦ L) ω - t * |g i0 ω|) P :=
    ((integrable_const L).indicator hBmeas.compl).sub
      ((integrable_of_law (hmeas i0) (hlaw i0)).abs.const_mul t)
  have hmain := integral_mono hlowint (integrable_log_Z hmeas hlaw t) hpt
  rw [integral_sub ((integrable_const L).indicator hBmeas.compl)
      ((integrable_of_law (hmeas i0) (hlaw i0)).abs.const_mul t),
    integral_indicator_const _ hBmeas.compl, integral_const_mul, hgabs, smul_eq_mul] at hmain
  have : L * (1 - 4 / q) ≤ P.real Bᶜ * L := by
    rw [mul_comm]; exact mul_le_mul_of_nonneg_right hGood hL
  linarith

end REM
