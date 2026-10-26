import REM.Lower

/-!
# Random Energy Model: Derrida's free energy

For each `N`, `2^N` standard Gaussians `g N i` (pairwise independent) on a probability space
`(Ω N, P N)`, and `a ≥ 0`. With `Z_N = ∑ i, exp (a √N g N i)`,

  `(1/N) E log Z_N → pressure a`,  `pressure a = log 2 + a²/2` if `a ≤ √(2 log 2)`,
                                    `= a √(2 log 2)` otherwise.

With energies `E_i = -√(s N) g_i` (variance `s N`) and `a = β √s` this is Derrida's result
`-β f = log 2 + s β² / 2` for `β ≤ β_c = √(2 log 2 / s)` and `-β f = β √(2 s log 2)` above.

Upper half: `upper_bound` with `m = 1` or `m = √(2 log 2)/a` (the 1RSB breakpoint `β_c/β`).
Lower half: `lower_bound` at threshold `θ = x √N`, any `0 ≤ x < √(2 log 2)`, then `x → a` or
`x → √(2 log 2)`.
-/

open MeasureTheory ProbabilityTheory Real Finset Filter Topology

namespace REM

/-- `√(2 log 2)`, the critical value of `a = β √s`. -/
noncomputable def crit : ℝ := √(2 * log 2)

/-- Derrida's limiting pressure `lim (1/N) E log Z_N` as a function of `a = β √s`. -/
noncomputable def pressure (a : ℝ) : ℝ := if a ≤ crit then log 2 + a ^ 2 / 2 else a * crit

lemma crit_sq : crit ^ 2 = 2 * log 2 := Real.sq_sqrt (by positivity)

lemma crit_pos : 0 < crit := Real.sqrt_pos.2 (by positivity)

/-- The exponent of the lower bound at threshold `x √N`. -/
noncomputable def psi (a x : ℝ) : ℝ := log 2 - x ^ 2 / 2 + a * x

lemma pressure_eq_inf_m {a : ℝ} (ha : 0 ≤ a) :
    ∃ m, 0 < m ∧ m ≤ 1 ∧ log 2 / m + m * a ^ 2 / 2 = pressure a := by
  unfold pressure
  split_ifs with h
  · exact ⟨1, one_pos, le_rfl, by ring⟩
  · push_neg at h
    have ha0 : 0 < a := lt_trans crit_pos h
    refine ⟨crit / a, div_pos crit_pos ha0, (div_le_one ha0).2 h.le, ?_⟩
    have hc := crit_sq
    have hcp := crit_pos
    field_simp
    nlinarith [hc]

lemma exists_psi_gt {a a' : ℝ} (ha : 0 ≤ a) (h : a' < pressure a) :
    ∃ x, 0 ≤ x ∧ x < crit ∧ a' < psi a x := by
  by_cases hac : a < crit
  · refine ⟨a, ha, hac, ?_⟩
    have : psi a a = pressure a := by
      simp only [psi, pressure, if_pos hac.le]; ring
    rwa [this]
  · push_neg at hac
    have hpc : psi a crit = pressure a := by
      unfold psi pressure
      by_cases h' : a ≤ crit
      · rw [if_pos h']
        have : a = crit := le_antisymm h' hac
        subst this; nlinarith [crit_sq]
      · rw [if_neg h']; nlinarith [crit_sq]
    have hcont : ContinuousAt (psi a) crit := by unfold psi; fun_prop
    have h1 : ∀ᶠ x in 𝓝[<] crit, a' < psi a x :=
      (hcont.tendsto.mono_left nhdsWithin_le_nhds).eventually (lt_mem_nhds (hpc ▸ h))
    have h2 : ∀ᶠ x in 𝓝[<] crit, x ∈ Set.Ioo 0 crit := Ioo_mem_nhdsLT crit_pos
    obtain ⟨x, hx1, hx2⟩ := (h1.and h2).exists
    exact ⟨x, hx2.1.le, hx2.2, hx1⟩

variable {Ω : ℕ → Type*} [∀ N, MeasurableSpace (Ω N)] (P : ∀ N, Measure (Ω N))
  [∀ N, IsProbabilityMeasure (P N)] (g : ∀ N, Fin (2 ^ N) → Ω N → ℝ)

/-- `(1/N) E log Z_N` at `a`. -/
noncomputable def freeEntropy (a : ℝ) (N : ℕ) : ℝ :=
  (1 / (N : ℝ)) * ∫ ω, log (Z (a * √(N : ℝ)) (g N) ω) ∂(P N)

lemma card_fin_pow (N : ℕ) : (Fintype.card (Fin (2 ^ N)) : ℝ) = 2 ^ N := by simp

lemma log_card (N : ℕ) : log (Fintype.card (Fin (2 ^ N)) : ℝ) = N * log 2 := by
  rw [card_fin_pow, Real.log_pow]

instance (N : ℕ) : Nonempty (Fin (2 ^ N)) := ⟨⟨0, by positivity⟩⟩

theorem freeEntropy_le (hmeas : ∀ N i, Measurable (g N i))
    (hlaw : ∀ N i, (P N).map (g N i) = gaussianReal 0 1) {a : ℝ} (ha : 0 ≤ a)
    {N : ℕ} (hN : 1 ≤ N) : freeEntropy P g a N ≤ pressure a := by
  obtain ⟨m, hm0, hm1, hm⟩ := pressure_eq_inf_m ha
  have hNpos : (0 : ℝ) < N := by exact_mod_cast hN
  have hub := upper_bound (hmeas N) (hlaw N) (a * √(N : ℝ)) hm0 hm1
  rw [log_card, mul_pow, Real.sq_sqrt hNpos.le] at hub
  unfold freeEntropy
  rw [← hm, one_div, inv_mul_le_iff₀ hNpos]
  calc ∫ ω, log (Z (a * √(N : ℝ)) (g N) ω) ∂(P N)
      ≤ N * log 2 / m + m * (a ^ 2 * N) / 2 := hub
    _ = N * (log 2 / m + m * a ^ 2 / 2) := by ring

/-- The explicit lower bound `b_N ≤ 2^N γ[x√N, ∞)`. -/
noncomputable def bN (x : ℝ) (N : ℕ) : ℝ := 2 ^ N * (exp (-(x * √(N : ℝ) + 1) ^ 2 / 2) / √(2 * π))

/-- The lower-bound sequence. -/
noncomputable def ell (a x : ℝ) (N : ℕ) : ℝ :=
  (log (bN x N / 2) / N + a * x) * (1 - 4 / bN x N) - a * absMean / √(N : ℝ)

lemma bN_pos (x : ℝ) (N : ℕ) : 0 < bN x N := by unfold bN; positivity

lemma log_bN (x : ℝ) (N : ℕ) :
    log (bN x N) = N * log 2 - (x * √(N : ℝ) + 1) ^ 2 / 2 - log √(2 * π) := by
  unfold bN
  have h1 : (0 : ℝ) < 2 ^ N := by positivity
  have h2 : 0 < √(2 * π) := Real.sqrt_pos.2 (by positivity)
  rw [log_mul h1.ne' (by positivity), log_div (exp_pos _).ne' h2.ne', log_exp, Real.log_pow]
  ring

lemma tendsto_inv_sqrt : Tendsto (fun N : ℕ ↦ 1 / √(N : ℝ)) atTop (𝓝 0) :=
  tendsto_const_nhds.div_atTop (tendsto_sqrt_atTop.comp tendsto_natCast_atTop_atTop)

lemma tendsto_inv_nat : Tendsto (fun N : ℕ ↦ 1 / (N : ℝ)) atTop (𝓝 0) :=
  tendsto_const_nhds.div_atTop tendsto_natCast_atTop_atTop

/-- `log b_N / N → log 2 - x²/2`. -/
lemma tendsto_log_bN_div (x : ℝ) :
    Tendsto (fun N : ℕ ↦ log (bN x N) / N) atTop (𝓝 (log 2 - x ^ 2 / 2)) := by
  have hform : ∀ᶠ N : ℕ in atTop, log (bN x N) / N
      = log 2 - x ^ 2 / 2 - x * (1 / √(N : ℝ)) - (1 / 2 + log √(2 * π)) * (1 / (N : ℝ)) := by
    filter_upwards [eventually_ge_atTop 1] with N hN
    have hNpos : (0 : ℝ) < N := by exact_mod_cast hN
    have hs : √(N : ℝ) ^ 2 = N := Real.sq_sqrt hNpos.le
    have hsp : 0 < √(N : ℝ) := Real.sqrt_pos.2 hNpos
    rw [log_bN]
    field_simp
    nlinarith [hs]
  refine Tendsto.congr' (EventuallyEq.symm hform) ?_
  have := ((tendsto_const_nhds (x := log 2 - x ^ 2 / 2)).sub
    (tendsto_inv_sqrt.const_mul x)).sub (tendsto_inv_nat.const_mul (1 / 2 + log √(2 * π)))
  simpa using this

lemma tendsto_bN {x : ℝ} (hx : x < crit) (hx0 : 0 ≤ x) : Tendsto (bN x) atTop atTop := by
  have hδ : 0 < log 2 - x ^ 2 / 2 := by
    have : x ^ 2 < crit ^ 2 := by nlinarith [crit_pos]
    rw [crit_sq] at this; linarith
  have hlog : Tendsto (fun N : ℕ ↦ log (bN x N)) atTop atTop := by
    have h1 : ∀ᶠ N : ℕ in atTop, (log 2 - x ^ 2 / 2) / 2 * N ≤ log (bN x N) := by
      have := (tendsto_log_bN_div x).eventually (lt_mem_nhds (show (log 2 - x ^ 2 / 2) / 2 <
        log 2 - x ^ 2 / 2 by linarith))
      filter_upwards [this, eventually_ge_atTop 1] with N hN h1
      have hNpos : (0 : ℝ) < N := by exact_mod_cast h1
      rw [lt_div_iff₀ hNpos] at hN; linarith
    exact tendsto_atTop_mono' _ h1
      ((tendsto_natCast_atTop_atTop).const_mul_atTop (by linarith))
  have := tendsto_exp_atTop.comp hlog
  refine this.congr fun N ↦ ?_
  simp [Function.comp, exp_log (bN_pos x N)]

lemma tendsto_ell {a x : ℝ} (hx : x < crit) (hx0 : 0 ≤ x) :
    Tendsto (ell a x) atTop (𝓝 (psi a x)) := by
  have hb := tendsto_bN hx hx0
  have h4 : Tendsto (fun N ↦ 4 / bN x N) atTop (𝓝 0) := tendsto_const_nhds.div_atTop hb
  have hlog2 : Tendsto (fun N : ℕ ↦ log (bN x N / 2) / N) atTop (𝓝 (log 2 - x ^ 2 / 2)) := by
    have hform : ∀ N : ℕ, log (bN x N / 2) / N = log (bN x N) / N - log 2 * (1 / (N : ℝ)) := by
      intro N
      rw [log_div (bN_pos x N).ne' two_ne_zero]; ring
    simp_rw [hform]
    simpa using (tendsto_log_bN_div x).sub (tendsto_inv_nat.const_mul (log 2))
  have := ((hlog2.add_const (a * x)).mul ((tendsto_const_nhds (x := (1 : ℝ))).sub h4)).sub
    (tendsto_inv_sqrt.const_mul (a * absMean))
  unfold ell psi
  simp only [sub_zero, mul_one, mul_zero] at this
  refine this.congr fun N ↦ ?_
  ring

theorem freeEntropy_ge_eventually (hmeas : ∀ N i, Measurable (g N i))
    (hlaw : ∀ N i, (P N).map (g N i) = gaussianReal 0 1)
    (hind : ∀ N, Pairwise fun i j ↦ IndepFun (g N i) (g N j) (P N))
    {a x : ℝ} (ha : 0 ≤ a) (hx : x < crit) (hx0 : 0 ≤ x) :
    ∀ᶠ N in atTop, ell a x N ≤ freeEntropy P g a N := by
  filter_upwards [(tendsto_bN hx hx0).eventually_ge_atTop 4, eventually_ge_atTop 1] with N hb hN
  have hNpos : (0 : ℝ) < N := by exact_mod_cast hN
  have hsN : 0 < √(N : ℝ) := Real.sqrt_pos.2 hNpos
  set θ := x * √(N : ℝ)
  set t := a * √(N : ℝ)
  have ht : 0 ≤ t := mul_nonneg ha hsN.le
  have hθ : 0 ≤ θ := mul_nonneg hx0 hsN.le
  have htθ : t * θ = a * x * N := by
    simp only [t, θ]; rw [show a * √(N : ℝ) * (x * √(N : ℝ)) = a * x * (√(N : ℝ)) ^ 2 by ring,
      Real.sq_sqrt hNpos.le]
  -- b_N ≤ q_N
  have hbq : bN x N ≤ (Fintype.card (Fin (2 ^ N)) : ℝ) * tail θ := by
    rw [card_fin_pow]; unfold bN
    exact mul_le_mul_of_nonneg_left (tail_ge hθ) (by positivity)
  set q := (Fintype.card (Fin (2 ^ N)) : ℝ) * tail θ
  have hb0 : 0 < bN x N := bN_pos x N
  have hq0 : 0 < q := lt_of_lt_of_le hb0 hbq
  have hLb : 0 ≤ log (bN x N / 2) + t * θ := by
    have : 0 ≤ log (bN x N / 2) := log_nonneg (by linarith)
    rw [htθ]; have : 0 ≤ a * x * N := by positivity
    linarith
  have hLq : log (bN x N / 2) + t * θ ≤ log (q / 2) + t * θ := by
    have := log_le_log (by positivity) (div_le_div_of_nonneg_right hbq (by norm_num : (0:ℝ) ≤ 2))
    linarith
  have hlb := lower_bound (hmeas N) (hlaw N) (hind N) ht hq0 (le_trans hLb hLq)
  have hfac : 0 ≤ 1 - 4 / bN x N := by
    rw [sub_nonneg, div_le_one hb0]; exact hb
  have hfacq : 1 - 4 / bN x N ≤ 1 - 4 / q := by
    have : 4 / q ≤ 4 / bN x N := div_le_div_of_nonneg_left (by norm_num) hb0 hbq
    linarith
  have hprod : (log (bN x N / 2) + t * θ) * (1 - 4 / bN x N)
      ≤ (log (q / 2) + t * θ) * (1 - 4 / q) :=
    mul_le_mul hLq hfacq hfac (le_trans hLb hLq)
  unfold freeEntropy ell
  rw [htθ] at hprod hlb
  have key : (log (bN x N / 2) + a * x * N) * (1 - 4 / bN x N) - t * absMean
      ≤ ∫ ω, log (Z t (g N) ω) ∂(P N) := by linarith
  have hexpand : (log (bN x N / 2) / N + a * x) * (1 - 4 / bN x N) - a * absMean / √(N : ℝ)
      = (1 / (N : ℝ)) * ((log (bN x N / 2) + a * x * N) * (1 - 4 / bN x N) - t * absMean) := by
    simp only [t]
    have hsq : (N : ℝ) = √(N : ℝ) ^ 2 := (Real.sq_sqrt hNpos.le).symm
    generalize √(N : ℝ) = r at hsq hsN ⊢
    rw [hsq]
    field_simp
  rw [hexpand]
  exact mul_le_mul_of_nonneg_left key (by positivity)

/-- **Derrida's theorem for the REM.** `(1/N) E log Z_N → pressure a`. -/
theorem tendsto_freeEntropy (hmeas : ∀ N i, Measurable (g N i))
    (hlaw : ∀ N i, (P N).map (g N i) = gaussianReal 0 1)
    (hind : ∀ N, Pairwise fun i j ↦ IndepFun (g N i) (g N j) (P N))
    {a : ℝ} (ha : 0 ≤ a) :
    Tendsto (freeEntropy P g a) atTop (𝓝 (pressure a)) := by
  rw [tendsto_order]
  constructor
  · intro a' ha'
    obtain ⟨x, hx0, hx, hψ⟩ := exists_psi_gt ha ha'
    filter_upwards [(tendsto_ell (a := a) hx hx0).eventually (lt_mem_nhds hψ),
      freeEntropy_ge_eventually P g hmeas hlaw hind ha hx hx0] with N h1 h2
    exact lt_of_lt_of_le h1 h2
  · intro a' ha'
    filter_upwards [eventually_ge_atTop 1] with N hN
    exact lt_of_le_of_lt (freeEntropy_le P g hmeas hlaw ha hN) ha'

/-- Derrida's formula in the physical variables: energies of variance `s N`, inverse
temperature `β`, `a = β √s`, `β_c = √(2 log 2 / s)`. -/
theorem pressure_beta {β s : ℝ} (hs : 0 < s) :
    pressure (β * √s) = if β ≤ √(2 * log 2 / s) then log 2 + s * β ^ 2 / 2
      else β * √(2 * s * log 2) := by
  have hss : 0 < √s := Real.sqrt_pos.2 hs
  have hiff : β * √s ≤ crit ↔ β ≤ √(2 * log 2 / s) := by
    rw [crit, Real.sqrt_div' _ hs.le, le_div_iff₀ hss]
  unfold pressure
  by_cases h : β ≤ √(2 * log 2 / s)
  · rw [if_pos (hiff.2 h), if_pos h, mul_pow, Real.sq_sqrt hs.le]; ring
  · rw [if_neg (fun h' ↦ h (hiff.1 h')), if_neg h, crit, mul_assoc, ← Real.sqrt_mul hs.le]
    ring_nf

end REM
