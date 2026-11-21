import BST.LatentEquivalence

namespace BST

universe u v w

theorem resolution_monotonicity
    {L : Type u} {O₁ : Type v} {O₂ : Type w}
    (P₁ : L → O₁)
    (P₂ : L → O₂)
    (C : O₁ → O₂)
    (hFactor : ∀ x : L, P₂ x = C (P₁ x))
    {a b : L}
    (hFine : ResolutionEquiv P₁ a b) :
    ResolutionEquiv P₂ a b := by
  unfold ResolutionEquiv at *
  calc
    P₂ a = C (P₁ a) := hFactor a
    _ = C (P₁ b) := by rw [hFine]
    _ = P₂ b := (hFactor b).symm

theorem finer_class_subset_coarser
    {L : Type u} {O₁ : Type v} {O₂ : Type w}
    (P₁ : L → O₁)
    (P₂ : L → O₂)
    (C : O₁ → O₂)
    (hFactor : ∀ x : L, P₂ x = C (P₁ x))
    (a x : L)
    (h : ResolutionEquiv P₁ x a) :
    ResolutionEquiv P₂ x a := by
  exact resolution_monotonicity P₁ P₂ C hFactor h

end BST
