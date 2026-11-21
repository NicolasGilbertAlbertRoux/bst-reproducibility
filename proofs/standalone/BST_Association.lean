namespace BST.Association

universe u v w z

theorem required_hidden_distinction
    {X : Type u} {O : Type v} {Z : Type w} {V : Type z}
    (obs : X → O) (latent : X → Z) (task : X → V) (decode : O → Z → V)
    (realizes : ∀ x, task x = decode (obs x) (latent x))
    {x y : X} (same : obs x = obs y) (different : task x ≠ task y) :
    latent x ≠ latent y := by
  intro hz
  apply different
  calc
    task x = decode (obs x) (latent x) := realizes x
    _ = decode (obs y) (latent y) := by rw [same, hz]
    _ = task y := (realizes y).symm

theorem joint_window_equivalence
    {X : Type u} {A : Type v} {B : Type w}
    (p : X → A) (q : X → B) (x y : X) :
    (p x, q x) = (p y, q y) ↔ p x = p y ∧ q x = q y := by
  constructor
  · intro h
    exact ⟨congrArg Prod.fst h, congrArg Prod.snd h⟩
  · intro h
    exact Prod.ext h.1 h.2

theorem realization_respects_encoding
    {X : Type u} {Z : Type v} {V : Type w}
    (encode : X → Z) (read : Z → V) (role : X → V)
    (realizes : ∀ x, read (encode x) = role x)
    {x y : X} (same : encode x = encode y) : role x = role y := by
  calc
    role x = read (encode x) := (realizes x).symm
    _ = read (encode y) := congrArg read same
    _ = role y := realizes y

def iter {X : Type u} (step : X → X) : Nat → X → X
  | 0, x => x
  | n+1, x => step (iter step n x)

theorem commuting_realization_all_steps
    {X : Type u} {Z : Type v} (step : X → X) (hiddenStep : Z → Z)
    (encode : X → Z) (commutes : ∀ x, encode (step x) = hiddenStep (encode x))
    (n : Nat) (x : X) : encode (iter step n x) = iter hiddenStep n (encode x) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    change encode (step (iter step n x)) = hiddenStep (iter hiddenStep n (encode x))
    rw [commutes, ih]

theorem invariant_role_from_invariant_code
    {X : Type u} {Z : Type v} {V : Type w}
    (step : X → X) (encode : X → Z) (read : Z → V)
    (conserved : ∀ x, encode (step x) = encode x) (x : X) :
    read (encode (step x)) = read (encode x) := congrArg read (conserved x)

theorem symmetry_survives_projection
    {G : Type u} {X : Type v} {Y : Type w}
    (actX : G → X → X) (actY : G → Y → Y) (obs : X → Y)
    (equivariant : ∀ g x, obs (actX g x) = actY g (obs x))
    (g : G) (x : X) (fixed : actX g x = x) : actY g (obs x) = obs x := by
  calc
    actY g (obs x) = obs (actX g x) := (equivariant g x).symm
    _ = obs x := congrArg obs fixed

theorem compatible_encodings_need_not_be_identical :
    (∃ decode : Bool → Bool, ∀ x : Bool, decode x = x) ∧
    (∃ decode : Bool → Bool, ∀ x : Bool, decode (!x) = x) ∧
    (fun x : Bool => x) ≠ (fun x : Bool => !x) := by
  refine ⟨⟨id, fun _ => rfl⟩, ⟨fun b => !b, ?_⟩, ?_⟩
  · intro x
    cases x <;> rfl
  · intro h
    exact Bool.noConfusion (congrFun h true)

theorem separate_existence_is_not_identification :
    (∃ b : Bool, b = true) ∧ (∃ b : Bool, b = false) ∧
    ¬ (∃ b : Bool, b = true ∧ b = false) := by
  refine ⟨⟨true, rfl⟩, ⟨false, rfl⟩, ?_⟩
  intro h
  obtain ⟨b, ht, hf⟩ := h
  exact Bool.noConfusion (ht.symm.trans hf)

end BST.Association

#print axioms BST.Association.required_hidden_distinction
#print axioms BST.Association.joint_window_equivalence
#print axioms BST.Association.realization_respects_encoding
#print axioms BST.Association.commuting_realization_all_steps
#print axioms BST.Association.invariant_role_from_invariant_code
#print axioms BST.Association.symmetry_survives_projection
#print axioms BST.Association.compatible_encodings_need_not_be_identical
#print axioms BST.Association.separate_existence_is_not_identification
