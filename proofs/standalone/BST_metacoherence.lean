namespace BST.Meta

abbrev Coh (S : Type) := S → S

abbrev Meta (S : Type) := Coh S → Coh S

def metaStep {S : Type} (M : Meta S) : Coh (S × Coh S) :=
  fun p => (p.2 p.1, M p.2)

theorem metacoherence_reduces_to_coherence {S : Type} (M : Meta S) :
    ∃ c : Coh (S × Coh S), ∀ (s : S) (c₀ : Coh S), c (s, c₀) = (c₀ s, M c₀) :=
  ⟨metaStep M, fun _ _ => rfl⟩

theorem reduced_simulates_one_step {S : Type} (M : Meta S) (s : S) (c₀ : Coh S) :
    metaStep M (s, c₀) = (c₀ s, M c₀) := rfl

theorem bool_ne_not : ∀ b : Bool, b = !b → False
  | true,  h => Bool.noConfusion h
  | false, h => Bool.noConfusion h

theorem no_universal_self_representation {C : Type} (rep : C → (C → Bool)) :
    ∃ p : C → Bool, ∀ c : C, rep c ≠ p :=
  ⟨fun c => !(rep c c),
   fun c h => bool_ne_not (rep c c) (congrFun h c)⟩

end BST.Meta

#print axioms BST.Meta.metacoherence_reduces_to_coherence
#print axioms BST.Meta.reduced_simulates_one_step
#print axioms BST.Meta.bool_ne_not
#print axioms BST.Meta.no_universal_self_representation
