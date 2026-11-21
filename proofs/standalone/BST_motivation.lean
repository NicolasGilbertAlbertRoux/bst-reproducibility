namespace BST.Motivation

abbrev Route := Bool

structure Agent where
  sel : Route
  val : Route → Bool

def akratic : Agent := { sel := true, val := fun r => !r }

theorem akrasia_possible : akratic.val akratic.sel = false := rfl

def agentA : Agent := { sel := true, val := fun _ => true }
def agentB : Agent := { sel := true, val := fun _ => false }

theorem same_selection : agentA.sel = agentB.sel := rfl
theorem different_value : agentA.val ≠ agentB.val :=
  fun h => Bool.noConfusion (congrFun h true)

theorem motivation_not_reducible_to_selection :
    (agentA.sel = agentB.sel ∧ agentA.val ≠ agentB.val)
  ∧ (akratic.val akratic.sel = false) :=
  ⟨⟨same_selection, different_value⟩, akrasia_possible⟩

end BST.Motivation

#print axioms BST.Motivation.akrasia_possible
#print axioms BST.Motivation.different_value
#print axioms BST.Motivation.motivation_not_reducible_to_selection
