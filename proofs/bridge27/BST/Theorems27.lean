import BST.Basic

namespace BST

universe u v w z

open Substrate

variable {S : Substrate}

theorem T01_closure_step
    {step : S.State → S.State} {P : S.State → Prop}
    (hclosed : S.ClosedUnder step P) {x : S.State} (hx : P x) :
    P (step x) :=
  hclosed x hx

theorem T02_closure_preserves_observable_class
    {step : S.State → S.State} (hstep : S.ObservationPreserving step) (x : S.State) :
    S.ObsEq (step x) x :=
  hstep x

theorem T03_projection_commutes_with_step
    {step : S.State → S.State} {ostep : S.Obs → S.Obs}
    (h : S.ProjectsTo step ostep) (x : S.State) :
    S.observe (step x) = ostep (S.observe x) :=
  h x

theorem T04_observable_equivalence_refl (x : S.State) :
    S.ObsEq x x :=
  rfl

theorem T05_observable_equivalence_symm {x y : S.State} :
    S.ObsEq x y → S.ObsEq y x := by
  intro h
  exact h.symm

theorem T06_observable_equivalence_trans {x y z : S.State} :
    S.ObsEq x y → S.ObsEq y z → S.ObsEq x z := by
  intro hxy hyz
  exact hxy.trans hyz

theorem T07_resolver_respects_observable_equivalence
    {α : Type w} (r : S.Resolver α) {x y : S.State} (h : S.ObsEq x y) :
    r (S.observe x) = r (S.observe y) := by
  exact congrArg r h

theorem T08_classifier_respects_resolved_equivalence
    {α : Type w} {β : Type z} (r : S.Resolver α) (c : α → β)
    {x y : S.State} (h : S.ObsEq x y) :
    c (r (S.observe x)) = c (r (S.observe y)) := by
  exact congrArg c (congrArg r h)

theorem T09_gauge_invisible_preserves_observation
    {g : S.State → S.State} (hg : S.GaugeInvisible g) (x : S.State) :
    S.observe (g x) = S.observe x :=
  hg x

theorem T10_gauge_invisible_preserves_resolver_output
    {α : Type w} {g : S.State → S.State} (hg : S.GaugeInvisible g)
    (r : S.Resolver α) (x : S.State) :
    r (S.observe (g x)) = r (S.observe x) := by
  exact congrArg r (hg x)

theorem T11_observable_invariant_constant_on_classes
    {α : Type w} {F : S.State → α} (hF : S.ObservableInvariant F)
    {x y : S.State} (hxy : S.ObsEq x y) :
    F x = F y :=
  hF x y hxy

theorem T12_gauge_invisible_preserves_invariant
    {α : Type w} {F : S.State → α} (hF : S.ObservableInvariant F)
    {g : S.State → S.State} (hg : S.GaugeInvisible g) (x : S.State) :
    F (g x) = F x :=
  hF (g x) x (hg x)

theorem T13_coarse_graining_preserves_aliasing
    {α : Type w} (coarse : S.Obs → α) {x y : S.State} (h : S.ObsEq x y) :
    coarse (S.observe x) = coarse (S.observe y) := by
  exact congrArg coarse h

theorem T14_memory_disambiguation_implies_distinct_states
    {α : Type w} {mem : S.State → α} {x y : S.State}
    (hmem : S.Disambiguates mem x y) :
    x ≠ y := by
  intro hxy
  apply hmem
  exact congrArg mem hxy

theorem T15_reconstruction_left_inverse_recovers
    {α : Type u} {β : Type v} {encode : α → β} {decode : β → α} {P : α → Prop}
    (h : LeftInverseOn encode decode P) {x : α} (hx : P x) :
    decode (encode x) = x :=
  h x hx

theorem T16_arbitration_selector_sound
    {α : Type u} {select : α → Bool} {Admissible : α → Prop}
    (hs : SelectorSound select Admissible) {a : α} (ha : select a = true) :
    Admissible a :=
  hs a ha

theorem T17_constrained_selection_sound
    {α : Type u} {select : α → Bool} {A B : α → Prop}
    (hs : SelectorSound select A) (hAB : ∀ a, A a → B a)
    {a : α} (ha : select a = true) :
    B a :=
  hAB a (hs a ha)

theorem T18_non_temporal_selector_constant
    {α : Type u} {Time : Type v} (select : α → Bool) (a : α) (t₁ t₂ : Time) :
    (fun (_ : Time) => select a) t₁ = (fun (_ : Time) => select a) t₂ :=
  rfl

theorem T19_threshold_falsification
    {α : Type u} {score : α → Nat} {threshold : Nat} {x : α}
    (h : ¬ ThresholdSupported score threshold x) :
    ¬ threshold ≤ score x :=
  h

theorem T20_threshold_support_intro
    {α : Type u} {score : α → Nat} {threshold : Nat} {x : α}
    (h : threshold ≤ score x) :
    ThresholdSupported score threshold x :=
  h

theorem T21_resolution_invariant_on_classes
    {α : Type w} (f : S.Obs → α) {x y : S.State} (h : S.ObsEq x y) :
    f (S.observe x) = f (S.observe y) := by
  exact congrArg f h

theorem T22_residual_information_from_memory
    {α : Type w} {mem : S.State → α} {x y : S.State}
    (halias : S.ObsEq x y) (hmem : S.Disambiguates mem x y) :
    S.observe x = S.observe y ∧ mem x ≠ mem y :=
  And.intro halias hmem

theorem T23_projected_dynamics_preserve_equivalence
    {step : S.State → S.State} {ostep : S.Obs → S.Obs}
    (hproj : S.ProjectsTo step ostep) {x y : S.State} (hxy : S.ObsEq x y) :
    S.ObsEq (step x) (step y) := by
  calc
    S.observe (step x) = ostep (S.observe x) := hproj x
    _ = ostep (S.observe y) := congrArg ostep hxy
    _ = S.observe (step y) := (hproj y).symm

theorem T24_topological_selection_sound
    {α : Type u} {select : α → Bool} {TopologicallyAllowed : α → Prop}
    (hs : SelectorSound select TopologicallyAllowed) {a : α} (ha : select a = true) :
    TopologicallyAllowed a :=
  hs a ha

theorem T25_contact_network_property_transport
    {Node : Type u} {EdgeA EdgeB : Node → Node → Prop}
    (hAB : ∀ x y, EdgeA x y → EdgeB x y) {x y : Node} (hxy : EdgeA x y) :
    EdgeB x y :=
  hAB x y hxy

theorem T26_global_identifiability
    {α : Type u} {β : Type v} {encode : α → β} {decode : β → α} {P : α → Prop}
    (h : LeftInverseOn encode decode P) {x y : α} (hx : P x) (hy : P y)
    (hcode : encode x = encode y) :
    x = y := by
  calc
    x = decode (encode x) := (h x hx).symm
    _ = decode (encode y) := congrArg decode hcode
    _ = y := h y hy

theorem T27_localization_stability
    {α : Type w} {perturb : S.State → S.State} {loc : S.State → α}
    (hstable : ∀ x, loc (perturb x) = loc x) (x : S.State) :
    loc (perturb x) = loc x :=
  hstable x

end BST
