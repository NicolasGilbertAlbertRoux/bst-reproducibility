namespace BST.Coherence

structure Cover where
  G  : Type
  A  : Type
  B  : Type
  rA : G → A
  rB : G → B

def restrict (c : Cover) (g : c.G) : c.A × c.B := (c.rA g, c.rB g)

def Separated (c : Cover) : Prop :=
  ∀ g g' : c.G, restrict c g = restrict c g' → g = g'

def Glues (c : Cover) : Prop :=
  ∀ (a : c.A) (b : c.B), ∃ g : c.G, restrict c g = (a, b)

def boolCover : Cover where
  G := Bool; A := Bool; B := Bool; rA := id; rB := id

theorem boolCover_separated : Separated boolCover :=
  fun _ _ h => congrArg Prod.fst h

theorem boolCover_not_glues : ¬ Glues boolCover :=
  fun glue =>
    match glue true false with
    | ⟨_, hg⟩ =>
        Bool.noConfusion ((congrArg Prod.fst hg).symm.trans (congrArg Prod.snd hg))

def collapseCover : Cover where
  G := Bool; A := Unit; B := Unit; rA := fun _ => (); rB := fun _ => ()

theorem collapseCover_glues : Glues collapseCover :=
  fun a b => ⟨true, by cases a; cases b; rfl⟩

theorem collapseCover_not_separated : ¬ Separated collapseCover :=
  fun sep => Bool.noConfusion (sep true false rfl)

theorem coherence_not_reducible_to_distinguishability :
    (Separated boolCover ∧ ¬ Glues boolCover)
  ∧ (Glues collapseCover ∧ ¬ Separated collapseCover) :=
  ⟨⟨boolCover_separated, boolCover_not_glues⟩,
   ⟨collapseCover_glues, collapseCover_not_separated⟩⟩

def prodCover : Cover where
  G := Bool × Bool; A := Bool; B := Bool; rA := Prod.fst; rB := Prod.snd

theorem prodCover_form_varies :
    prodCover.rA (true, false) ≠ prodCover.rB (true, false) :=
  fun h => Bool.noConfusion h

end BST.Coherence

#print axioms BST.Coherence.boolCover_separated
#print axioms BST.Coherence.boolCover_not_glues
#print axioms BST.Coherence.collapseCover_glues
#print axioms BST.Coherence.collapseCover_not_separated
#print axioms BST.Coherence.coherence_not_reducible_to_distinguishability
#print axioms BST.Coherence.prodCover_form_varies
