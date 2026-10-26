import REM.Limit

/-!
# The hypotheses are satisfiable: the canonical product model

`Ω N = Fin (2^N) → ℝ` with the product of standard Gaussians and `g N i ω = ω i`.
-/

open MeasureTheory ProbabilityTheory Filter Topology

namespace REM

/-- The canonical REM probability space at size `N`. -/
noncomputable def canonical (N : ℕ) : Measure (Fin (2 ^ N) → ℝ) :=
  Measure.pi fun _ ↦ gaussianReal 0 1

instance (N : ℕ) : IsProbabilityMeasure (canonical N) := by
  unfold canonical; infer_instance

/-- Derrida's theorem on the canonical product model. -/
theorem tendsto_freeEntropy_canonical {a : ℝ} (ha : 0 ≤ a) :
    Tendsto (freeEntropy canonical (fun N i ω ↦ ω i) a) atTop (𝓝 (pressure a)) := by
  refine tendsto_freeEntropy canonical _ (fun N i ↦ measurable_pi_apply i) ?_ ?_ ha
  · intro N i
    classical
    unfold canonical
    rw [Measure.pi_map_eval]
    simp
  · intro N i j hij
    have := iIndepFun_pi (μ := fun _ : Fin (2 ^ N) ↦ gaussianReal 0 1)
      (X := fun _ x ↦ x) (fun _ ↦ aemeasurable_id)
    exact this.indepFun hij

end REM
