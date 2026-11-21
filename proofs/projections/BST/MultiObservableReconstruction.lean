import BST.LatentEquivalence

namespace BST

universe u v

structure ProjectionFamily (I : Type u) (L : Type v) where
  O : I → Type u
  P : (i : I) → L → O i

theorem multi_observable_reconstruction
    {I : Type u} {L : Type v}
    (F : ProjectionFamily I L)
    (hSurj : ∀ i : I, Function.Surjective (F.P i)) :
    ∀ i : I, Bijective (quotientToObservable (F.P i)) := by
  intro i
  exact observable_reconstruction_bijective (F.P i) (hSurj i)

theorem projections_can_induce_distinct_equivalence
    {L : Type u} {O₁ : Type v} {O₂ : Type v}
    (P₁ : L → O₁)
    (P₂ : L → O₂)
    (a b : L)
    (h₁ : ResolutionEquiv P₁ a b)
    (h₂ : ¬ ResolutionEquiv P₂ a b) :
    ResolutionEquiv P₁ a b ∧ ¬ ResolutionEquiv P₂ a b := by
  exact ⟨h₁, h₂⟩

end BST
