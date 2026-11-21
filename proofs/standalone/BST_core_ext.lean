namespace BST

variable {L O O₁ O₂ R : Type}

@[reducible] def ObsEquiv (obs : L → O) (a b : L) : Prop := obs a = obs b

theorem obsEquiv_refl (obs : L → O) (a : L) : ObsEquiv obs a a := rfl
theorem obsEquiv_symm (obs : L → O) {a b : L}
    (h : ObsEquiv obs a b) : ObsEquiv obs b a := h.symm
theorem obsEquiv_trans (obs : L → O) {a b c : L}
    (h₁ : ObsEquiv obs a b) (h₂ : ObsEquiv obs b c) : ObsEquiv obs a c := h₁.trans h₂

@[reducible] def Limited (obs : L → O) : Prop := ∃ a b : L, a ≠ b ∧ obs a = obs b
@[reducible] def LatentAt (obs : L → O) (a b : L) : Prop := a ≠ b ∧ obs a = obs b

theorem latent_necessity (obs : L → O) (h : Limited obs) :
    ∃ a b : L, LatentAt obs a b :=
  match h with
  | ⟨a, b, hne, hobs⟩ => ⟨a, b, hne, hobs⟩

theorem latent_forced_by_persistence
    (obs₁ : L → O₁) (obs₂ : L → O₂) (a b : L)
    (haliased  : obs₁ a = obs₁ b)
    (hdistinct : obs₂ a ≠ obs₂ b)
    : LatentAt obs₁ a b :=
  ⟨fun h => hdistinct (congrArg obs₂ h), haliased⟩

@[reducible] def Stable (closed persists : Prop) : Prop := closed ∧ persists

theorem closure_persistence_imp_stability {closed persists : Prop}
    (hc : closed) (hp : persists) : Stable closed persists := ⟨hc, hp⟩

structure Validation where
  necessity       : Prop
  nonCheating     : Prop
  persistence     : Prop
  closure         : Prop
  reproducibility : Prop

@[reducible] def Admissible (v : Validation) : Prop :=
  v.necessity ∧ v.nonCheating ∧ v.persistence ∧ v.closure ∧ v.reproducibility

theorem admissible_needs_necessity {v : Validation} (h : Admissible v) : v.necessity := h.1
theorem necessity_gates {v : Validation} (h : ¬ v.necessity) : ¬ Admissible v :=
  fun ha => h ha.1
theorem nonCheating_gates {v : Validation} (h : ¬ v.nonCheating) : ¬ Admissible v :=
  fun ha => h ha.2.1

theorem residual_closure {AllInternalForced HierarchyFixed : Prop}
    (forced : AllInternalForced) (hopen : ¬ HierarchyFixed) :
    AllInternalForced ∧ ¬ HierarchyFixed := ⟨forced, hopen⟩

@[reducible] def Separates (obs : R → L → O) : Prop :=
  ∀ a b : L, (∀ r : R, obs r a = obs r b) → a = b

theorem latent_not_absolute (obs : R → L → O) (sep : Separates obs)
    {a b : L} (hne : a ≠ b) : ¬ (∀ r : R, obs r a = obs r b) :=
  fun hall => hne (sep a b hall)

theorem latent_visible_somewhere (obs : R → L → O) (sep : Separates obs)
    {a b : L} (hne : a ≠ b) : ∃ r : R, obs r a ≠ obs r b :=
  Classical.byContradiction fun hnex =>
    hne (sep a b (fun r => Classical.byContradiction (fun hr => hnex ⟨r, hr⟩)))

@[reducible] def DistinguishedSomewhere (obs : R → L → O) (a b : L) : Prop :=
  ∃ r : R, obs r a ≠ obs r b

theorem latent_real_but_hidden (obs : R → L → O) (r₀ : R) {a b : L}
    (hpers : DistinguishedSomewhere obs a b)
    (hmerged : obs r₀ a = obs r₀ b)
    : a ≠ b ∧ obs r₀ a = obs r₀ b :=
  match hpers with
  | ⟨r₁, hr₁⟩ => ⟨fun h => hr₁ (congrArg (obs r₁) h), hmerged⟩

theorem separates_imp_distinguished (obs : R → L → O) (sep : Separates obs)
    {a b : L} (hne : a ≠ b) : DistinguishedSomewhere obs a b :=
  latent_visible_somewhere obs sep hne

inductive Three | x | y | z

def obsEx : Unit → Three → Bool
  | _, Three.x => true
  | _, Three.y => false
  | _, Three.z => false

theorem xy_persists : DistinguishedSomewhere obsEx Three.x Three.y :=
  ⟨(), by decide⟩

theorem obsEx_not_separates : ¬ Separates obsEx :=
  fun sep => Three.noConfusion (sep Three.y Three.z (fun _ => rfl))

theorem weak_does_not_imply_strong :
    (DistinguishedSomewhere obsEx Three.x Three.y) ∧ (¬ Separates obsEx) :=
  ⟨xy_persists, obsEx_not_separates⟩

end BST

#print axioms BST.latent_necessity
#print axioms BST.latent_forced_by_persistence
#print axioms BST.closure_persistence_imp_stability
#print axioms BST.necessity_gates
#print axioms BST.residual_closure
#print axioms BST.latent_not_absolute
#print axioms BST.latent_real_but_hidden
#print axioms BST.latent_visible_somewhere
#print axioms BST.separates_imp_distinguished
#print axioms BST.xy_persists
#print axioms BST.obsEx_not_separates
#print axioms BST.weak_does_not_imply_strong
