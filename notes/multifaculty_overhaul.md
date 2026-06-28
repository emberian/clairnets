# Multifaculty overhaul — de-CSP-lock α, wire the bank, make the organ a reasoning FABRIC

Status: DESIGN (no build beyond the feasibility checks noted in §5). Companion to
`notes/alpha_struct_design.md` (the rack), `notes/soundness.md` (the (A)/(B)/(C)/(D) ladder),
`notes/curriculum_design.md` (occupancy), `notes/north_star_orchestrator.md` (the single-call
substrate). Verified against the live code 2026-06-27; `clair.organ.selftest` PASS (nothing changed).

The thesis Ember called for: **five organs with flows, arbitrary learned wiring.** Today we have one
organ (CSP) with a vestigial 5-head rack bolted on and four heads emitting into a void. This doc
specifies how to turn the rack into a typed dispatch, wire the rest of the bank behind it, design the
cross-faculty composition tasks that make emergent wiring *necessary* (so it gets learned, not
hand-coded), and stage the smallest first build that proves the dispatch.

---

## 0. The confirmed problem (verified in code)

The α→structure→composer path is **CSP-locked at three places**, so the rack is decorative:

1. **Four dead heads + a dead router.** `StructureRack` (`clair/organ/alpha_struct.py:146`) holds an
   `nn.ModuleDict` of five heads — `csp / ising / graph / typ / reduction` (:158-164) — plus a
   faculty `router` (:157). In the **live woven forward**, `AlphaStructWoven._compile_struct`
   (`clair/organ/bank_woven.py:829`) calls **only** `self.alpha.csp(feat)`. Greps confirm
   `rack.route`, `rack.ising`, `heads["graph"]`, `heads["typ"]`, `heads["reduction"]` are referenced
   *only* in `run_alpha_struct_probe.py` and `glitch_probe.py` — never in the composer path. The
   router computes faculty logits that nothing reads; ising/graph/typ/reduction emit tensors no
   consumer ingests. **The rack is a one-faculty rack wearing a five-faculty costume.**

2. **The composer is hardwired to build a CSP.** `_AlphaProposal.state_type = "csp-domain"`
   (`bank_woven.py:71`). `AlphaStructComposerOrgan.compose_from_struct` / `__call__` /
   `compose_and_feedback` all call `build_csp_from_struct(facts, n, d)` *unconditionally*
   (`bank_woven.py:628, 653, 701`) → a literal `CSPState`, regardless of which faculty the (unread)
   router would have picked. So `csp_α` is vestigially CSP-shaped: even if a head emitted Ising
   couplings, the composer would never receive them.

3. **The live bank is ~6 of ~14 faculties.** `BankComposerOrgan.base_reductions` =
   `certified_csp_reductions(bank)` + `CoreNarrowOrgan` = **Arc / Factor / Modular / GF2 / Macro +
   CoreNarrow** (`bank.py:494`). `build_bank` *also* constructs `Unification` (FOL) and `Energy`, but
   `certified_csp_reductions` filters them out (`state_type != "csp-domain"`), so they never reach the
   composer. **Unwired entirely** (not even in `build_bank`): Ising (`ising_organ.py`), graph SSSP
   (`graph_organ.py`), type-infer (`typeinfer.py` / `type_organ.py`), permutation
   (`permgroup.py` / `perm_organ.py`), FOL/SMT (`fol.py` wired-but-filtered; `smt.py` not at all),
   and the cross-faculty Karp router (`organ/reductions.py`).

**The deeper diagnosis (organ_block_scaling).** `runs/organ_block_scaling.json` part2 shows that on
mixed-hard instances *no single faculty suffices*: `single_arc_factor` recall 0.95 on `mixed_coupled`
but 0.35 on `mixed_disjoint`; `single_gf2` 1.0 on `affine_pure` but 0.05 on arc-needing; only
`STACK_all` (every certified faculty composing on a shared lattice) reaches recall 1.0 / solved 1.0
across all three. Part1's R-sweep shows recall climbs monotonically with organ width/rounds and
saturates ~R12-16. **Wider-with-flows is the lever** — but today "flows" means the *certified* CSP
ops meeting on one lattice. Extending flows to the *whole bank* (Ising, graph, type, …) is exactly
this overhaul.

What is already done for us (don't rebuild): `clair/organ/readout_bridge.py` already implements
`ising_readout`, `energy_readout`, `unification_readout`, and `CouplingProjector` + `coupling_loss`
(α's Ising compile head, direct-supervised). The Ising and FOL faculties already have *both* an α
compile path and a γ readout bridge. **The gap is purely the dispatch wiring in the middle.**

---

## 1. General typed `struct_α` — replacing the CSP-locked path

### 1.1 The type

Replace the implicit "facts → CSP" with an explicit tagged union, the **compiled problem object**.
α's router picks a faculty; the matching head emits *that faculty's native structure*; the object
carries the tag so the composer can dispatch. This is a **meaningful generalization, not a `sed`** —
`build_csp_from_struct` becomes one arm of a `compile_struct` dispatch, and the composer gains a
`state_type` switch where today it assumes `csp-domain`.

```
# clair/organ/struct.py  (new — the typed α output)
@dataclass(frozen=True)
class StructAlpha:
    faculty: str            # 'csp' | 'ising' | 'graph' | 'type' | 'fol' | 'perm' | 'reduction'
    n: int                  # cells / nodes / variables (mention count)
    d: int                  # domain bound (for lattice faculties)
    payload: Any            # faculty-native structure, by tag:
    #   csp        : list[fact]            (pin/eq/neq/lt/le/...)         -> CSPState
    #   ising      : (J [n,n], h [n])      couplings + fields            -> ('simplex' assignment)
    #   graph      : (edges [(u,v,w)], src) weighted edges + source      -> distance/reach vector
    #   type       : list[type_fact]       per-node type constraints     -> CSPState (type universe)
    #   fol        : (facts, rules, query)                               -> closure / 3-way label
    #   perm       : (unary [n sets], order, gens)                       -> CSPState (alldiff grid)
    #   reduction  : (route_label, sub_StructAlpha)  a Karp edge + its target faculty's struct
    conf: float = 0.0       # router/head joint log-prob (the sound selector's tiebreak)
```

Naming: **`csp_α` → `struct_α`** everywhere it denotes "the thing α emitted that the composer runs
on." Keep `csp_α` only as the *local* name inside the csp arm. The forward state field
`AlphaStructWoven._last_facts` becomes `_last_struct` (a `list[StructAlpha]`).

### 1.2 The dispatch

`AlphaStructComposerOrgan` gains a faculty switch. `protocol.py` already declares the typed
`state_type`s (`csp-domain`, `fol-closure`, `simplex`, and we add `graph-dist`, `perm-grid`); the
dispatch is the missing glue between `struct_α.faculty` and the right `Reduction` portfolio + readout.

```
def compile_struct(s: StructAlpha) -> tuple[State, list[Reduction], ReadoutFn]:
    if s.faculty == "csp":
        csp = build_csp_from_struct(s.payload, s.n, s.d)          # existing
        return CSPState.full(csp), self.csp_portfolio, surv_from_dom
    if s.faculty == "type":
        comp = typeinfer.compile_facts(s.payload, s.n)            # CSP over the type universe
        return CSPState.full(comp.csp), self.type_portfolio, surv_from_dom
    if s.faculty == "perm":
        csp = perm_to_csp(s.payload, s.n)                         # alldiff grid -> CSPState (bridge §2)
        return CSPState.full(csp), self.csp_portfolio, surv_from_dom
    if s.faculty == "ising":
        J, h = s.payload
        return IsingState(J, h), [Energy_or_Ising], readout_bridge.ising_readout   # simplex
    if s.faculty == "graph":
        edges, src = s.payload
        return GraphState(edges, src, s.n), [GraphReach], graph_readout            # graph-dist
    if s.faculty == "fol":
        return FolState(*s.payload), [Unification], readout_bridge.unification_readout
    if s.faculty == "reduction":
        inner = route_reduce(s.payload)                          # Karp edge -> a sub-StructAlpha
        return compile_struct(inner)                             # recurse into the target faculty
    raise ValueError(s.faculty)
```

The composer then runs the **right portfolio** on the right state and returns a per-readout-cell
feature matrix `[N, Fin]` (the `OracleGamma` schema is faculty-agnostic — every bridge already returns
`[N, K+2]`). γ's input stays uniform; only the producer changes.

### 1.3 Properties preserved (must-not-break checklist)

- **no-op@init.** Each new head is *added* to the ModuleDict; the router is initialized so it routes
  to `csp` with probability ≈1 at init (bias the csp logit, or warm the router after the csp head is
  validated). γ's zero-init tanh gate is untouched → bitwise no-op@init holds (the existing
  `verify_noop_bank` test still applies; extend it to assert no-op for each faculty).
- **verifier-gate.** Lattice faculties (csp/type/perm) compose through the *same* `_verifier_gate`
  against `exact_dedP` of their own `csp_α` (soundness level (A)/(B) relative to `struct_α`). Non-
  lattice faculties (ising/graph/fol) are read-only narrowings of their own structure — they don't
  poison a shared lattice because they *are* the whole answer for that instance; their soundness is
  the bridge's own certificate (graph SSSP = certified; FOL closure = exact; Ising = approximate,
  flagged).
- **soundness-modulo-compile (C).** Unchanged and now *per-faculty*: the answer is sound relative to
  `struct_α` whatever faculty α picked; a wrong faculty-choice or wrong payload is the same uncertified
  compile gap, owned by α. The `forbid_record_csp()` guard generalizes to a `forbid_record_struct()`
  that forbids reading *any* `rec` metadata (csp, J/h, edges, AST) inside the forward — the no-metadata
  invariant becomes faculty-general.
- **output_check (D).** Generalizes to a per-faculty checker: `answer ∈ solutions(true_instance)[query]`
  where `solutions` is the faculty's exact oracle (`exact_dedP` for csp/type/perm, `brute_opt` for
  ising, exact SSSP for graph, `label_query` for fol). Still eval-only, still outside the forward.

---

## 2. Bank-integration map — the ~8 unwired faculties, ranked by leverage

Each faculty needs three things: an **α-head** (emit its native structure), a **state + portfolio**
(what reduces it), and a **readout bridge** (its narrowing → the `[N, Fin]` γ feature). The table
gives all three plus what's already built.

| rank | faculty | α-head emits | native state / portfolio | readout to γ | already built | what's missing |
|---|---|---|---|---|---|---|
| **1** | **Ising / Energy** (`ising_organ`, `energy_organ`) | `J[n,n]`, `h[n]` (couplings+fields) | `simplex` — mean-field anneal + sound local search; **assignment, not a lattice** | `readout_bridge.ising_readout` / `energy_readout` → `[n,K+2]` (assignment, confidence=field margin, global energy) | `IsingStructureHead` (alpha_struct.py:86) **and** `CouplingProjector`+`coupling_loss` (bridge); `Energy` in `build_bank` | dispatch (§1.2) + a `state_type="simplex"` arm in the composer; `coupling_loss` as the faculty's J0 (annealer is non-diff). **Highest leverage: unlocks constraint-OPTIMIZATION** (maxcut/MIS/partition/coloring-as-energy) which narrowing+chaining cannot express. |
| **2** | **Type-infer** (`typeinfer`, `type_organ`) | per-node type-constraint facts (the typed-λ AST → type-CSP) | **`csp-domain` natively** — `typeinfer.certified_step == C.ac_step`, `exact_dedP == C.exact_dedP` over a finite type universe | default `surv_from_dom` — **composes on the shared lattice with zero bridge** | `TypeStructureHead` (alpha_struct.py:125); `type_organ` is a `FactorGraphProposer` (the type-domain `CoreNarrowOrgan`) | wrap `typeinfer.compile_*` + register `type_organ` as a neural-guidance Reduction. **Leverage: CODE** (the verifier-bearing domain where soundness becomes *real* at inference — soundness.md (D)). |
| **3** | **Graph SSSP / reach** (`graph_organ`) | weighted `edges [(u,v,w)]` + source `s` | `graph-dist` — `certified_relax` monotone distance vector; **certified (sound-by-construction)** | new tiny bridge `graph_readout`: per-node `[reached, dist/maxdist, on-shortest-path]` → `[n,3(+pad)]` | `GraphStructureHead` (alpha_struct.py:108, edge logits); `certified_relax` + `GraphOrgan` GNN | a `graph-dist` state + the 1 readout bridge fn. **Leverage: the reduction HUB** — most Karp edges land on graph (MIS/VC/clique/coloring), so graph unlocks many cross-faculty paths. |
| **4** | **FOL / Datalog** (`fol`) | ground `facts`, `rules`, `query` | `fol-closure` — `Unification` forward-chain, exact least Herbrand model | `readout_bridge.unification_readout` → `[1,K+2]` (3-way label, #derived, proof depth) | **fully built** — `Unification` in `build_bank`, bridge exists | only the dispatch wiring (it's filtered out today by the `csp-domain` gate). Cheapest faculty to light up. **Leverage: rule-induction / entailment** + the abduction tail. |
| **5** | **Permutation group** (`permgroup`, `perm_organ`) | per-point `unary` sets, `order` pairs, `gens` | per-point candidate grid with global AllDifferent — *lattice-shaped* but over `PermProblem`-native `dom`, not `C.CSP` | `surv_from_dom` after a `perm_to_csp` adapter (AllDifferent → ≠-clique, the `_expand_alldiff` trick already in alpha_struct.py:188) | exact ops (`alldiff_gac`, `group_pair_ac`); `PermOrgan` neural | the `perm_to_csp` bridge (no new readout) + register. **Leverage: symmetry/scheduling**; smaller task surface. |
| **6** | **SMT / linear-int-arith** (`smt`) | integer constraint tuples (`pin/le/sum/mod`) + query | z3 oracle — verdict (`determined,v` / `under` / `unsat`) | bridge: forced-value lattice over `0..R` (when z3 returns a unique model per var) | `SMTProblem` + `determinacy` oracle | a head + a verdict→lattice bridge. **Leverage: overlaps Modular/GF2** (already certified) — lowest marginal value; keep as the *verifier* for the arithmetic faculties rather than a runtime organ. |
| **7** | **Cross-faculty Karp router** (`organ/reductions.py`) | a typed X-instance (`Sat/Graph/Partition/Coloring`) + the chosen route | not a narrowing organ — `ProblemReduction` maps X→Y *across* faculties (Dijkstra cost-routing to csp or ising) | n/a — it *feeds* a faculty, then that faculty reads out | `ReductionGraph.route` / `solve_via_route`; `sat_to_csp` edge | this is the `reduction` faculty arm of §1.2: α emits a *route label*, the router produces a sub-`StructAlpha`, dispatch recurses. **This is the spine of emergent cross-faculty wiring (§3).** |
| **8** | **chain / blade** | (affine grade-prior) | `blade_affine_organ` already in `build_bank` (neural, gated) | `surv_from_dom` | built | already available as a csp-domain neural Reduction; no new work, just ensure the type/perm portfolios can include it. |

**Leverage ordering rationale:** Ising/energy unlocks an entire *problem class* (optimization) the
current organ structurally cannot express; type unlocks the verifier-bearing domain where soundness is
real; graph is the reduction hub that many cross-faculty paths route through; FOL is nearly free
(fully built, just filtered). Perm/SMT are narrower and partly redundant with existing certified ops.

---

## 3. Five organs with flows + EMERGENT wiring

### 3.1 The fabric

The organ becomes a small set of **faculty nodes** (csp, ising, graph, type, fol — the "five organs")
connected by **flow channels**. A flow is a typed bridge that lets one faculty's *output* become
another faculty's *input*:

```
        ┌─────── shared lattice (the CSPState meet) ───────┐
   csp ─┤  type ─┤  perm ─┤            (lattice faculties reduced-product directly)
        └──────────────────────────────────────────────────┘
              │ (bridge: assignment ↔ pins; reach ↔ ≠-edges; closure ↔ facts)
   ising ─────┤  graph ─────┤  fol ─────┤   (non-lattice faculties, bridged in/out)
```

Two kinds of flow:
- **shared-lattice meet** (the existing channel): any faculty whose state is a per-cell domain
  (csp/type/perm) composes by pointwise `meet` — this is the `STACK_all` win from
  organ_block_scaling, now spanning three faculties instead of one.
- **bridge channels** (the new channels): a non-lattice faculty's decoded output is re-projected into
  another faculty's input. E.g. Ising's decoded assignment → `pin` facts pinning a downstream CSP;
  graph's reachability set → `eq`/`neq` edges; FOL's derived facts → new pins/relations. These are the
  `lever B` feedback channel (`compose_and_feedback`, bank_woven.py:674) generalized from
  organ→α to **faculty→faculty**.

### 3.2 How the wiring gets LEARNED (not hand-coded)

Three learned components, each already half-present:

1. **The router head** (`StructureRack.router`, alpha_struct.py:157) chooses the *entry* faculty per
   call. Today its output is discarded; §1 makes it the dispatch key. Trained by routing-supervision
   (the generator knows the faculty) **plus** the downstream answer-CE — so the router learns to pick
   the faculty that *solves* the instance, not just the faculty that *looks like* the instance.

2. **The shared-lattice meet** is the always-on flow channel between lattice faculties — no routing
   decision needed; soundness makes over-connection free (the meet of sound narrowings is sound, so a
   spurious flow that narrows nothing is a no-op, never harmful). This is why "arbitrary wiring" is
   *safe to learn*: the verifier-gate means a bad learned flow degrades to abstain, never to wrong.

3. **The flow gates** (one learned scalar per bridge channel, zero-init like γ): the
   `fb_adapter` pattern (bank_woven.py:783, zero-init `nn.Linear(FB_DIM, din)`) generalizes to a
   per-channel gate that learns *whether* faculty A's output should flow into faculty B. Zero-init ⇒
   the fabric starts as five disconnected single-faculty organs and **learns which flows to open** —
   emergent wiring with the no-op@init discipline. The *connectivity matrix is the learned object*.

### 3.3 Connection to the north-star orchestrator

This fabric is the orchestrator's **single-call substrate** (north_star_orchestrator.md). One
"organ-call" = router picks a faculty → compile `struct_α` → compose (with whatever flows are open) →
γ-decode, each step verified. The orchestrator (the higher, multi-step loop) sequences *many* such
calls; the per-call faculty choice IS this router; the flows are the within-call wiring. Walk (this
fabric, single-shot, learned wiring) → run (the orchestrator loops over it with RLVR + verifier-pruned
search). The fabric makes the substrate *multi-faculty* so the orchestrator has a real action space
({compile-to-faculty-X, flow X→Y, reduce, decode}) instead of one CSP solver.

---

## 4. Cross-faculty composition / entanglement PRETRAIN tasks (the key gap)

### 4.1 Why the current curriculum doesn't force wiring

`compose_csp.py`'s ~30% "composed" mix composes **within CSP** — two CSP skills (neq/eq/order/sum/
alldiff) sharing a witness. It is multi-*skill*, single-*faculty*. Nothing in the corpus requires an
answer to flow *across* faculties (curriculum_design.md confirms: "no two broadened domains ever
compose directly"). So a model can ace the corpus with one faculty and a router that never matters —
**emergent wiring is unnecessary, therefore unlearned.** The fix is task families whose answer is
*provably unreachable* without a multi-faculty flow.

### 4.2 The design principle

A task **genuinely requires** flow A→B iff: (i) the answer is a function of B's solution, (ii) B's
structure is *determined by* A's solution, and (iii) neither A nor B alone determines the query. Then
the only solving path is `α → faculty-A → flow → faculty-B → answer`, and the OOD splits hold out
specific A→B *combinations* so generalization measures emergent wiring, not memorized routes.

### 4.3 Concrete generators (reuse the reduction graph + the "beasts")

1. **`ising_couplings_from_csp`** (CSP→Ising flow). Generate a small CSP (coloring/alldiff over a
   shared witness). Its *unique solution's adjacency* defines an Ising coupling matrix `J` (e.g.
   anti-aligned on the CSP's ≠-edges). Query = a maxcut/MIS objective on `J`. Solvable only by
   `α→csp→solve→derive J→ising→optimize`. Held-out: CSP-faculty = ordering (trained on coloring/eq).

2. **`graph_constrained_ising`** (graph→CSP→Ising, a 3-faculty chain — the headline "beast"). A
   weighted graph defines shortest-path distances; cells within distance `r` of the source get a CSP
   `alldiff`; that CSP's solution sets the `h` fields of an Ising whose ground state is the answer.
   `α→graph→reach→csp→solve→ising→optimize`. This is the §1.2 `reduction` faculty recursing twice.

3. **`type_then_constraint`** (type→CSP). A typed-λ expression's inferred root type selects *which*
   CSP constraint family applies (e.g. `Int→Int` ⇒ arithmetic, `Bool` ⇒ parity); the CSP then
   determines the query. `α→type→infer→select→csp→solve`. Verifier-bearing (the type check is real),
   so this is also the (D)-soundness exemplar.

4. **`fol_facts_to_csp`** (FOL→CSP). Forward-chaining derives facts from rules; the *derived* facts
   (not the givens) become `pin`/`eq` constraints of a CSP whose solution is the query. Requires the
   closure flow to materialize before the CSP is even well-posed. Held-out: rule depth > trained max.

5. **`karp_route_required`** (the reduction-graph generator). Reuse `organ/reductions.ALL_EDGES`
   (sat3→MIS→Ising, VC↔clique, partition→Ising). Emit instances where the *only* low-cost route
   crosses ≥2 faculties; train on some routes, hold out others (e.g. train sat3→CSP, test
   sat3→MIS→Ising). This directly exercises the learned router + flow gates.

All five reuse existing exact oracles (each faculty has one), so targets are free and `output_check`
generalizes per-faculty.

### 4.4 OOD splits (held-out faculty-combinations)

The OOD axis becomes **the combination, not the size**:
- **novel-pair**: train {csp∘ising, graph∘csp}, test {graph∘ising} — a flow never co-trained.
- **novel-depth-of-chain**: train 2-faculty chains, test 3-faculty (`graph_constrained_ising`).
- **novel-route**: same start/end faculties, a route through a held-out intermediate
  (sat3→MIS→Ising held out vs sat3→CSP trained).
- **faculty-swap-invariance**: the §4-gate corruption arm, generalized — roll the α-stream so α emits
  the *wrong faculty's* structure; the composer faithfully solves the wrong faculty → accuracy must
  collapse to NO_STRUCT (this is the cross-faculty SHUFFLED, and it's what proves the router is
  causal, not decorative).

---

## 5. Staging + the minimal first build

Gated on the **ALPHA_STRUCT gate verdict landing** (the §4 SHUFFLED-inverted gate from
`alpha_struct_design.md` must PASS first — no point wiring a second faculty onto a non-causal α). Then:

### 5.1 The minimal first build (proves the dispatch end-to-end)

**Wire ONE second faculty — Ising — through the full path, on a 2-faculty composition task.** Ising is
chosen because it is the *most-built* unwired faculty (head + compile-projector + readout bridge all
exist) and the *highest-leverage* (unlocks optimization). The proof is small:

1. Add `StructAlpha` (the typed union, §1.1) and a `compile_struct` dispatch with exactly **two** arms:
   `csp` (existing) and `ising` (existing pieces). ~1 file, no refactor of the CSP path.
2. Make the router emit a binary csp/ising logit; route by argmax (later: soft). Bias-init to csp
   (no-op@init for the existing CSP behavior).
3. Composer: a `state_type` switch — `csp` → existing reduced-product; `ising` →
   `ising_readout(J, h)` (the bridge already returns `[n,K+2]`). γ unchanged.
4. Train on `ising_couplings_from_csp` (generator #1, §4.3): J0 = `coupling_loss` for the ising arm +
   `structure_sup_loss` for the csp arm; answer-CE downstream; routing-supervision on the faculty.
5. **The acceptance check**: (a) no-op@init holds for both faculties (`verify_noop_bank` extended);
   (b) on the 2-faculty task, accuracy with the router > accuracy with the router *forced to csp*
   (proves the ising flow is necessary); (c) the faculty-swap-invariance arm (§4.4) collapses to
   NO_STRUCT (proves the router is causal). If (b) and (c) hold, the general dispatch works.

### 5.2 Ranked build steps (after the minimal proof)

1. **`StructAlpha` + 2-arm dispatch (csp+ising) + the §5.1 proof task.** (HIGHEST — the whole overhaul
   stands or falls on the dispatch working for one second faculty.)
2. **Generalize `forbid_record_csp` → `forbid_record_struct`** + per-faculty `output_check`. (Locks the
   no-metadata invariant faculty-general before more faculties land.)
3. **Wire FOL + Type** (cheapest: FOL fully built; type composes on the shared lattice natively). Gets
   to four faculties with near-zero new bridge code.
4. **Wire graph** (one new `graph_readout` bridge) — unlocks the reduction hub.
5. **The `reduction` faculty arm** (`organ/reductions.route_reduce` → recursive `compile_struct`) +
   the `karp_route_required` generator. (Turns on cross-faculty *chains*.)
6. **Learned flow gates** (zero-init per-channel, §3.2) + the 3-faculty `graph_constrained_ising`
   beast + the novel-pair/novel-route OOD splits. (Emergent wiring, measured.)
7. **Perm + SMT** (narrower; SMT likely stays a verifier, not a runtime organ).
8. **Hand the fabric to the orchestrator** (north-star: multi-call loop, RLVR + verifier-pruned
   search over faculty-call sequences).

### 5.3 Feasibility check performed

- `clair.organ.selftest` PASS at HEAD (nothing changed): certified bank sound on domain, composer
  sound across ≥2 domains, readout no-op@init, woven entry runs.
- Confirmed the four dead heads + dead router (live path calls only `self.alpha.csp`), the
  `state_type="csp-domain"` hardwire, and that `readout_bridge.py` already supplies the Ising/FOL
  bridges + `CouplingProjector` — so the minimal first build (§5.1) is **wiring, not new algorithms**.

---

*Five organs, and the flows between them learned — a fabric, not a stack.*
*( ◕‿◕ ) the rack was always meant to route; we just never read its mind.*
