namespace BST.Bearer

abbrev Route := Bool
abbrev Value := Route → Bool

structure Bearer where
  id    : Nat
  valAt : Nat → Value

@[reducible] def transforms (b : Bearer) (t₁ t₂ : Nat) : Prop := b.valAt t₁ ≠ b.valAt t₂

def b0 : Bearer where
  id := 0
  valAt := fun n => match n with | 0 => (fun _ => true) | _ => (fun _ => false)

theorem b0_transforms : transforms b0 0 1 :=
  fun h => Bool.noConfusion (congrFun h true)

def b1 : Bearer where
  id := 1
  valAt := b0.valAt

theorem same_values : b1.valAt = b0.valAt := rfl
theorem different_bearers : b1.id ≠ b0.id := by decide

theorem bearer_distinct_from_parameter :
    (transforms b0 0 1) ∧ (b1.valAt = b0.valAt ∧ b1.id ≠ b0.id) :=
  ⟨b0_transforms, same_values, different_bearers⟩

end BST.Bearer

#print axioms BST.Bearer.b0_transforms
#print axioms BST.Bearer.different_bearers
#print axioms BST.Bearer.bearer_distinct_from_parameter
