import BST.ResolutionMonotonicity

namespace BST

universe u v w x

def ClassIncluded
    {L : Type u} {Ofine : Type v} {Ocoarse : Type w}
    (Pfine : L → Ofine)
    (Pcoarse : L → Ocoarse) : Prop :=
  ∀ a x : L, ResolutionEquiv Pfine x a → ResolutionEquiv Pcoarse x a

theorem hierarchical_refinement_two_level
    {L : Type u} {Ofine : Type v} {Ocoarse : Type w}
    (Pfine : L → Ofine)
    (Pcoarse : L → Ocoarse)
    (C : Ofine → Ocoarse)
    (hFactor : ∀ x : L, Pcoarse x = C (Pfine x)) :
    ClassIncluded Pfine Pcoarse := by
  intro a x hFine
  exact finer_class_subset_coarser Pfine Pcoarse C hFactor a x hFine

theorem hierarchical_refinement_three_level
    {L : Type u} {O₁ : Type v} {O₂ : Type w} {O₃ : Type x}
    (P₁ : L → O₁)
    (P₂ : L → O₂)
    (P₃ : L → O₃)
    (C₁₂ : O₁ → O₂)
    (C₂₃ : O₂ → O₃)
    (h12 : ∀ x : L, P₂ x = C₁₂ (P₁ x))
    (h23 : ∀ x : L, P₃ x = C₂₃ (P₂ x)) :
    ClassIncluded P₁ P₂ ∧ ClassIncluded P₂ P₃ := by
  constructor
  · exact hierarchical_refinement_two_level P₁ P₂ C₁₂ h12
  · exact hierarchical_refinement_two_level P₂ P₃ C₂₃ h23

theorem class_inclusion_trans
    {L : Type u} {O₁ : Type v} {O₂ : Type w} {O₃ : Type x}
    (P₁ : L → O₁)
    (P₂ : L → O₂)
    (P₃ : L → O₃)
    (h12 : ClassIncluded P₁ P₂)
    (h23 : ClassIncluded P₂ P₃) :
    ClassIncluded P₁ P₃ := by
  intro a x hFine
  exact h23 a x (h12 a x hFine)

end BST
