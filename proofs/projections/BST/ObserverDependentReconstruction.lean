import BST.LatentEquivalence

namespace BST

universe u v

structure Observer (L : Type u) where
  O : Type v
  R : L → O

def ObserverEquiv {L : Type u} (ω : Observer L) (a b : L) : Prop :=
  ω.R a = ω.R b

def observerSetoid {L : Type u} (ω : Observer L) : Setoid L where
  r := fun a b => ObserverEquiv ω a b
  iseqv := by
    constructor
    · intro a
      rfl
    · intro a b h
      exact h.symm
    · intro a b c hab hbc
      exact hab.trans hbc

def observerClass {L : Type u} (ω : Observer L) (a : L) :
    Quotient (observerSetoid ω) :=
  Quotient.mk (observerSetoid ω) a

theorem observer_dependent_reconstruction
    {L : Type u}
    (ω : Observer L)
    (hSurj : Function.Surjective ω.R) :
    Bijective (quotientToObservable ω.R) := by
  exact observable_reconstruction_bijective ω.R hSurj

theorem observer_equivalence_can_differ
    {L : Type u}
    (ω₁ ω₂ : Observer L)
    (a b : L)
    (h₁ : ObserverEquiv ω₁ a b)
    (h₂ : ¬ ObserverEquiv ω₂ a b) :
    ObserverEquiv ω₁ a b ∧ ¬ ObserverEquiv ω₂ a b := by
  exact ⟨h₁, h₂⟩

end BST
