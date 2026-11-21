namespace BST

universe u v w

structure Substrate where
  State : Type u
  Obs : Type v
  observe : State → Obs

namespace Substrate

variable (S : Substrate)

def ObsEq (x y : S.State) : Prop := S.observe x = S.observe y

def Resolver (α : Type w) := S.Obs → α

def GaugeInvisible (g : S.State → S.State) : Prop :=
  ∀ x, S.observe (g x) = S.observe x

def ObservableInvariant (F : S.State → α) : Prop :=
  ∀ x y, ObsEq S x y → F x = F y

def ClosedUnder (step : S.State → S.State) (P : S.State → Prop) : Prop :=
  ∀ x, P x → P (step x)

def ObservationPreserving (step : S.State → S.State) : Prop :=
  ∀ x, S.observe (step x) = S.observe x

def ProjectsTo (step : S.State → S.State) (ostep : S.Obs → S.Obs) : Prop :=
  ∀ x, S.observe (step x) = ostep (S.observe x)

def Disambiguates (mem : S.State → α) (x y : S.State) : Prop :=
  mem x ≠ mem y

end Substrate

def SelectorSound {α : Type u} (select : α → Bool) (Admissible : α → Prop) : Prop :=
  ∀ a, select a = true → Admissible a

def LeftInverseOn {α : Type u} {β : Type v}
    (encode : α → β) (decode : β → α) (P : α → Prop) : Prop :=
  ∀ x, P x → decode (encode x) = x

def ThresholdSupported {α : Type u} (score : α → Nat) (threshold : Nat) (x : α) : Prop :=
  threshold ≤ score x

end BST
