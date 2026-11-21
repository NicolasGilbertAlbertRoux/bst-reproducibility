namespace BST

universe u v

def NonInjective {L : Type u} {O : Type v} (P : L → O) : Prop :=
  ∃ a b : L, a ≠ b ∧ P a = P b

def ResolutionEquiv {L : Type u} {O : Type v} (P : L → O) (a b : L) : Prop :=
  P a = P b

def Bijective {A : Type u} {B : Type v} (f : A → B) : Prop :=
  Function.Injective f ∧ Function.Surjective f

theorem latent_equivalence_under_resolution
    {L : Type u} {O : Type v}
    (P : L → O)
    (h : NonInjective P) :
    ∃ a b : L, a ≠ b ∧ ResolutionEquiv P a b := by
  exact h

def projectionSetoid {L : Type u} {O : Type v} (P : L → O) : Setoid L where
  r := fun a b => P a = P b
  iseqv := by
    constructor
    · intro a
      rfl
    · intro a b h
      exact h.symm
    · intro a b c hab hbc
      exact hab.trans hbc

def reconstructionClass {L : Type u} {O : Type v} (P : L → O) (a : L) :
    Quotient (projectionSetoid P) :=
  Quotient.mk (projectionSetoid P) a

def quotientToObservable {L : Type u} {O : Type v} (P : L → O) :
    Quotient (projectionSetoid P) → O :=
  Quotient.lift P (by
    intro a b h
    exact h)

theorem quotientToObservable_surjective
    {L : Type u} {O : Type v}
    (P : L → O)
    (hSurj : Function.Surjective P) :
    Function.Surjective (quotientToObservable P) := by
  intro o
  rcases hSurj o with ⟨a, ha⟩
  exists Quotient.mk (projectionSetoid P) a

theorem quotientToObservable_injective
    {L : Type u} {O : Type v}
    (P : L → O) :
    Function.Injective (quotientToObservable P) := by
  intro q₁ q₂ hEq
  induction q₁ using Quotient.inductionOn with
  | h a =>
    induction q₂ using Quotient.inductionOn with
    | h b =>
      apply Quotient.sound
      exact hEq

theorem observable_reconstruction_bijective
    {L : Type u} {O : Type v}
    (P : L → O)
    (hSurj : Function.Surjective P) :
    Bijective (quotientToObservable P) := by
  constructor
  · exact quotientToObservable_injective P
  · exact quotientToObservable_surjective P hSurj

end BST
