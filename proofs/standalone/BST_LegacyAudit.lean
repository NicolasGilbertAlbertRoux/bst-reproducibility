namespace BST.LegacyAudit

theorem empty_list_exclusion_incompatible_with_nonzero_defect
    {D M : Type} (charge : D → Int) (d : D) (nonzero : charge d ≠ 0)
    (exclusion : ∀ z : D, charge z ≠ 0 → ¬ (∃ modes : List M, True)) : False :=
  exclusion d nonzero ⟨[], True.intro⟩

end BST.LegacyAudit
#print axioms BST.LegacyAudit.empty_list_exclusion_incompatible_with_nonzero_defect
