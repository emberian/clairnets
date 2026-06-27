# ALPHA_STRUCT — α emits the problem structure (the text→organ reasoning move)

*Design + implementation plan + acceptance gate. Local, no GPU. This is the central scientific move and
the hardest part of the project: make α actually COMPILE the relational structure from the host hidden,
so the composer + certified floor run on α's compiled program, not on handed-in metadata.*

## 0. The crux this fixes (from `runs/shuffled_diag.json` + `notes/codex_organ_review.md`)

Today `bank_woven.csp_spec_for_record` (@284) threads `rec["csp"]` — the instance's TRUE constraint
structure — into `BankComposerOrgan.compose_one` (@95). The certified floor (Arc/Factor/Modular/GF2/Macro,
`bank.certified_csp_reductions`) runs on that true structure, so it determines the query cell to the
correct answer **regardless of what α emits**. α (`DenseLatentProjector`, `latent_organ.py:39`) only
compiles the per-cell candidate LATTICE `b0 [B,N,K]` and joins as a verifier-gated passenger
(`_AlphaProposal`, bank_woven.py:52) that can never poison the floor — and never has to *drive* it either.

The SHUFFLED diagnostic proved this is not a nuance but the whole result:

| arm (informative subset, n=399) | acc-vs-true | acc-vs-shuffled |
|---|---|---|
| TRUE_STRUCT  | **100%** | 0% |
| SHUFFLED_STRUCT (swap `rec["csp"]` for another instance's, α-prompt unchanged) | **0%** | **100%** |
| NO_STRUCT (gate-zero) | 33% | 33% |

Correctness is **100% metadata-driven**: swap the metadata and the answer follows the metadata perfectly;
α contributes nothing. The certified floor sits under the *metadata*, not under α's compiled semantics.

**The fix (codex answer #1 / #3):** α must EMIT the factor graph (which variables, which typed
relations). The composer + certified floor operate ONLY on α-produced structure. `rec["csp"]` survives
in exactly one place — the **hidden output-checker at eval** — and is never threaded into the forward.
The floor then certifies relative to α's COMPILED structure; the final answer is output-checked against
the true hidden instance. Soundness-of-the-floor is preserved; the new burden (emit the right structure)
moves onto α, which is exactly where the science is.

---

## 1. The architecture — a MULTI-FACULTY α STRUCTURE COMPILER (the "rack")

**REFINED (Ember's directive):** α must be a MULTI-FACULTY structure compiler — a **router + per-faculty
heads** — NOT a lone CSP factor-graph head. If α only ever emits a CSP factor graph, the woven is forever
CSP-narrow-only and the whole bank (Ising/graph/type/reduction faculties, the certified ops) is wasted.
So α is a RACK, built so that *adding a faculty = adding a head*:

```
shared per-cell ENCODER (mention-pool identity ⊕ CellReader cross-attn read of the prompt)  [CellEncoder]
   → ROUTER head            : host hidden (masked-mean) → faculty {csp, ising, graph, type, reduction}
   → per-faculty STRUCTURE heads, each emitting that faculty's NATIVE structure:
       csp       : pin head [B,N,1+K]  +  typed pair-relation head [B,N,N,R] over {none,eq,neq,lt,le}
       ising     : couplings J [B,N,N] (sign = align/anti) + fields h [B,N]      (trainable)
       graph     : edge logits [B,N,N]                                            (scaffold)
       type      : per-cell type-constraint logits [B,N,Tt]                       (scaffold, key 'typ')
       reduction : chosen reduction-route logits [B, Rroutes]                     (scaffold)
```

Implemented in `clair/organ/alpha_struct.py` as `StructureRack` (the rack) over `CellEncoder` +
`{CSPStructureHead, IsingStructureHead, GraphStructureHead, TypeStructureHead, ReductionHead}` in a
`nn.ModuleDict`. The router is faculty-general: it reads a masked-mean of the host hidden (needs no
CSP-style mentions), so it routes problems from any faculty. The non-CSP heads are SCAFFOLDED (interfaces
wired); the **CSP head + the router are fully trained+probed first** because CSP is the cleanest capability
test. The capability probe (`clair/organ/run_alpha_struct_probe.py`) trains these on FROZEN OLMo-2-1B
hidden — the purest test of "can α READ structure off the host hidden?" (no LoRA, no composer, no LM).

### 1.1 What α emits today vs. under ALPHA_STRUCT

- **Today:** α → `b0 [B,N,K]` (per-cell candidate logits) + a latent context. Per-cell SETS only. The
  relations come from `rec["csp"].cons`.
- **ALPHA_STRUCT:** α additionally → a **typed factor graph over the mentioned cells**: a discrete set of
  facts `[(rel_type, scope, params), …]` from a fixed typed vocabulary. The composer builds its CSPState
  from THIS (`build_csp_from_struct(α_facts) → CSP`), not from `rec["csp"]`.

### 1.2 The typed relation vocabulary (free, exact, from the generator)

The curriculum (`curriculum.fact_constraint`) already enumerates every fact kind; the generator hands us
the exact ground-truth fact list `p.facts` for free. The structure head emits over this closed vocabulary:

| arity | kind | params | rungs |
|---|---|---|---|
| unary | `pin(i, v)` | value v∈0..d-1 | all (the givens) |
| binary | `eq(i,j)` `neq(i,j)` | — (symmetric) | equality, coloring, alldiff |
| binary (directed) | `lt(i,j)` `le(i,j)` | — (i→j ordered) | ordering |
| ternary | `sum(a,b,c)` `xor(a,b,c)` | — | arithmetic, parity |
| variadic | `par(scope)` `rel(scope, table)` | parity / allowed-table | hard / OOD-novel |

The MINIMAL first version emits only **{pin} ∪ {eq, neq, lt, le}** (covers coloring/equality/ordering —
the three rungs SHUFFLED ran on: eqchain + forcedcolor are path-eq/neq, so binary suffices to test the
whole crux). Ternary (`sum`/`xor`) and `rel` stage in with the difficulty curriculum (§5).

### 1.3 The structure head (concrete shape)

α already produces, per mentioned cell i, a fused representation `feat_i = [v_mean_i ; cross-attn read_i]`
(`DenseLatentProjector.forward`, latent_organ.py:50 — mention-pool identity ⊕ a `CellReader` read over the
whole prompt). The structure head reuses these reads and adds:

- **Pin head** (unary): `feat_i → logits over {no-pin, 0,1,…,d-1}` → `[B, N, 1+K]`. argmax≠no-pin ⇒ `pin(i,v)`.
- **Pair-relation head** (binary): for every ordered pair (i,j), i≠j, score a bilinear/MLP map
  `g(feat_i, feat_j) → logits over {none, eq, neq, lt, le}` → `[B, N, N, R]`. This is a *typed edge
  predictor over the mention grid* — the factor graph. (eq/neq symmetric ⇒ symmetrize logits; lt/le
  directional ⇒ keep the ordered pair.) The "bind" is solved by construction: the pair indices ARE the
  cell binding (mention-pooled identity), so α only has to *recognize the relation*, not also point.

This is a **copy-from-text + bind** mechanism: mentions give the binding (which entities), the pair head
reads the relation off the cross-attention context between the two cells' spans. Emitting a discrete
structure from continuous hidden = **soft factor logits** at train (BCE-supervised), **thresholded /
top-k selected** at compose time (a small discrete fact set). Ternary `sum(a,b,c)` is harder (a 3-tensor
over cells, sparse) and is deferred; it is α-as-search territory (§5) — propose a few candidate triples,
verifier-filter.

### 1.4 What the composer consumes

`BankComposerOrgan.compose_one(csp_α, alpha_dom, …)` where `csp_α = build_csp_from_struct(α_facts, n, d)`.
The certified floor + CoreNarrowOrgan run on `csp_α`. `_AlphaProposal(α's per-cell b0)` still rides as the
gated neural proposal. **`rec["csp"]` does not appear in this path.** It appears only in
`output_check(answer, rec["csp"])` at eval, and as the structure-supervision target at train.

---

## 2. Training α to emit correct structure

Three signals, layered exactly like the validated J0 recipe (`train_bank_woven`, bank_woven.py:432):

### 2.1 Structure supervision (the new J0) — warmup + standing aux

We have the true facts free (`p.facts`). Build the supervised targets per instance:
- **pin target** `[B,N,1+K]`: one-hot of the pinned value (or no-pin) per cell.
- **pair target** `[B,N,N,R]`: one-hot of the relation type per ordered cell-pair (or `none`).

Loss = masked cross-entropy of the structure-head logits vs these targets, over the valid (mentioned)
cells/pairs. This is **directly supervised multi-label classification on the mention grid** — fully
learnable on canonical templates (the text literally states "A is not B"; α just has to read+type it).
Soundness-asymmetry analogous to dominate-dedₚ: weight `missing a true relation` (drops structure ⇒ the
floor under-constrains ⇒ abstain/wrong) heavier than `hallucinating a relation` (over-constrains ⇒ caught
by the output-check / abstain). Warm it in Phase A; keep it as a standing aux (`struct_sup_w · L_struct`)
through Phase B alongside the per-cell J0 (`alpha_sup_w · dominate_dedp_loss`) and the LM-CE.

### 2.2 Downstream LM-CE

Unchanged in spirit: γ reads the DETACHED composer output (composed from `csp_α`); LoRA+γ learn to read
the lattice; the LM-CE flows to LoRA+γ. α gets its gradient from the structure-supervision + J0 aux
(the composer is discrete/non-differentiable, so no gradient flows answer→structure directly — same as
today's b0). This keeps it learnable: the discrete bottleneck is bridged by the free exact structure
target, not by trying to backprop through `build_csp_from_struct`.

### 2.3 Keeping it learnable — the honest hard part

- **Canonical templates:** near-trivial (the relation is lexicalized). Structure-F1 should be ~1.0; this is
  the *sanity floor*, not the result.
- **Real NL (`build_diverse_live_pool`):** the generalization frontier. α must read "Alice ranks above
  Bob" → `lt(bob, alice)` from phrasings it never saw. This is where the project's whole thesis lives.
  Mitigations: hard-negative NL pairs (same entities, minimally-changed relation — codex #3); train the
  structure head on the diverse curriculum jsonl, not just templates; let LoRA adapt OLMo to expose the
  relation latently.
- **Discrete emission:** soft logits → threshold. Risk: a near-threshold relation flips the whole CSP. This
  is precisely why we pair with **α-as-search** (§5): when the structure head is uncertain, propose the
  top-k structures and verifier-filter (which compiles to a solvable problem whose certified solution
  output-checks) — gradient warms α, search refines the structure where α is unsure.
- **Ternary/variadic:** deferred. `sum`/`xor`/`par` need a sparse k-tensor or a select-from-candidates
  mechanism; the difficulty curriculum reaches them after binary+pins is solid.

---

## 3. Keeping the certified-floor robustness — the soundness story

The change does NOT weaken the floor; it re-bases what the floor is sound *with respect to*, and adds an
honest output-check on top. Spell it out as three layers:

1. **Floor soundness (unchanged, now w.r.t. `csp_α`).** Arc/Factor/Modular/GF2/Macro are
   sound-by-construction for *whatever* CSPState they receive (`compose.reduced_product` asserts each
   certified reduction returns a subset; the meet of sound narrowings is sound). CoreNarrowOrgan + the
   α-candidate proposal are verifier-gated against `exact_dedP(csp_α)` (`compose._verifier_gate`). So the
   composed lattice is a **sound narrowing of `solutions(csp_α)`** — it never drops a value some solution
   of α's compiled problem uses. Identical guarantee to today; only the input CSP changed.

2. **The new gap is explicit and owned by α.** If `csp_α ≠ csp_true`, the floor confidently solves the
   wrong problem (it cannot detect a wrong premise — soundness is *relative to the premise*). This is the
   honest cost of making α causal: a wrong compile ⇒ confidently-wrong organ (already noted in
   `notes/training.md`: "α-correctness is the whole ballgame"). The floor no longer launders this.

3. **Output-check = the floor that sits UNDER α's semantics.** At eval, the final answer is checked against
   the hidden true instance — exactly the `reductions.verify_reduction` pattern (reduce→solve→decode→
   compare to X's OWN exact answer): `output_check(answer, rec["csp"])` accepts iff the answer is a real
   solution-value of `csp_true` at the query cell. This is a **sound verifier over α's structure-proposals**:
   accept ⇒ correct, regardless of what α emitted. No false-accepts, no reward-hacking — which is precisely
   why a sound verifier *wants search* (north-star §"search arm"): best-of-N / α-as-search over structures,
   each filtered by (a) compiles-to-solvable and (b) certified-solution-output-checks.

   The residual after the output-check is **completeness, not soundness**: α may fail to emit the right
   structure → the system abstains or is wrong, but it is never *confidently-accepted-wrong*. That is the
   correct, honest trade — we converted an unsound-but-confident metadata passenger into a
   sound-but-incomplete α-driven reasoner.

Note the headline-accuracy forward does NOT use the output-check (that would leak the answer to the LM);
the output-check is a separate reliability/abstain/search layer. The headline number is the LM's generated
answer driven by α's-structure-composed lattice — see §4.

---

## 4. The acceptance gate = the SHUFFLED test, INVERTED

Under ALPHA_STRUCT **there is no metadata to swap** — the structure IS α's output, so a "SHUFFLED_STRUCT"
arm is *impossible by construction* (the composer literally never receives `rec["csp"]`). The success
criterion therefore inverts: instead of "does the answer follow swapped metadata?" we ask "does the answer
collapse when α's INPUT is corrupted, and does accuracy track α-faithfulness?"

### 4.1 The arms (same held-out determined-query pool as shuffled_diag)

- **ALPHA_STRUCT (true):** α reads instance i's host hidden → emits `csp_α` → composer → γ → answer.
- **α-INPUT-CORRUPT:** corrupt α's input only (roll the α-stream text/host-hidden across the batch, i.e.
  α reads instance *j*'s text while the gen-stream + query stay instance i's; OR scramble the mention
  spans). α emits the WRONG structure; the composer faithfully solves it. Accuracy-vs-true must **collapse
  toward NO_STRUCT**. (This is the exact inversion of the SHUFFLED result: today corrupting the host hidden
  did nothing because `rec["csp"]` carried the answer; now it must be decisive.)
- **EMPTY_STRUCT:** force α to emit no relations (full domains) → composer = identity → full lattice → no
  information. A clean "structure-shaped but contentless" control; must ≈ NO_STRUCT.
- **NO_STRUCT:** gate-zero / no injection — the text-only floor anchor (≈33% on this pool).

### 4.2 The exact pass criterion

Let `acc_true`, `acc_corrupt`, `acc_empty`, `acc_nostruct` be determined-query accuracies on the pool;
let `F1_struct` = α-emitted-facts vs true-facts F1; let `faithful[i]` = (α emitted i's exact factor graph).

PASS iff ALL hold:
1. **Lift (structure is necessary AND α-produced):** `acc_true − acc_nostruct ≥ 0.30`.
2. **Collapse (α's input is causal):** `acc_corrupt ≤ acc_nostruct + 0.05` AND `acc_empty ≤ acc_nostruct + 0.05`,
   i.e. `acc_true − acc_corrupt ≥ 0.30`. (The headline inversion of SHUFFLED.)
3. **Accuracy tracks α-faithfulness (replaces "tracks metadata"):**
   `acc | faithful − acc | ¬faithful ≥ 0.30` AND `corr(faithful, correct) > 0.2`.
4. **Structure is genuinely emitted (sanity, not the result):** `F1_struct ≥ 0.80` on canonical;
   report it (lower, honestly) on real-NL — that gap is the science.
5. **No-metadata invariant (code-level, not a number):** a guard/test asserts `rec["csp"]` never enters
   the composer forward — `csp_spec_for_record` is reachable only from `output_check` and the
   structure-supervision target builder. The 100%-acc-shuf SHUFFLED arm is structurally impossible.

Report it as a table mirroring `shuffled_diag.print_diagnostic`, plus the F1 and the faithfulness split.
The "verdict" line: PASS = "α DRIVES correctness (collapses under input-corrupt; accuracy tracks
faithfulness; no metadata in the forward)."

---

## 5. Implementation plan + staging

### 5.1 Files to change

- **`clair/latent_organ.py`** — add `StructureHead` (pin head + pair-relation head) to
  `DenseLatentProjector` (or a sibling `DenseLatentStructProjector` reusing the same `CellReader` reads);
  add `decode_structure(pin_logits, pair_logits, vmask, d, top_k) → list[facts]` and
  `structure_sup_loss(logits, true_pin_tgt, true_pair_tgt, vmask)`.
- **`clair/curriculum.py`** (or a new `clair/organ/struct_io.py`) — `facts_to_struct_targets(p.facts, n, d)
  → (pin_tgt, pair_tgt)` and `build_csp_from_struct(facts, n, d) → C.CSP` (thin wrapper over `build_csp`).
- **`clair/organ/bank_woven.py`** —
  - `BankWoven._compile_b0` → also compile structure; `_inj_hook`/composer call builds `csp_α` from α's
    emitted facts instead of reading `self._csps` (which came from `rec["csp"]`).
  - `BankComposerOrgan.__call__/compose_one` take α-emitted facts (per-batch) → CSPState.
  - `_bank_batch`: STOP setting `ba["csps"] = csp_spec_for_record(...)` for the forward; instead set
    `ba["struct_tgt"] = facts_to_struct_targets(...)` and keep `rec["csp"]` only for the eval check.
  - `train_bank_woven`: Phase A warms α on `structure_sup_loss` (+ the existing per-cell J0); Phase B adds
    LM-CE with the composer running on `csp_α`.
- **`clair/organ/alpha_struct_diag.py`** (NEW, sibling of `shuffled_diag.py`) — the §4 acceptance gate:
  the 4 arms, the F1, the faithfulness correlation, `output_check`, the PASS/FAIL verdict.
- **`clair/organ/run_alpha_struct_diag.py`** (NEW, sibling of `run_shuffled_diag.py`) — smoke driver.

### 5.2 Risks

- **NL generalization wall** (the real one): structure-F1 collapses off-template. Mitigation: diverse-curriculum
  training + hard-negative pairs; expect to *report* the gap honestly, not erase it. This is the thesis.
- **Discrete-flip brittleness:** a near-threshold relation flips the CSP. Mitigation: rich-state γ already
  degrades gracefully on partial lattices; pair with α-as-search; calibrate thresholds.
- **Ternary blow-up:** `sum`/`par` need sparse k-tensor or select-from-candidates. Mitigation: defer; the
  binary+pins regime already tests the entire crux.
- **Floor confidently wrong on a wrong compile:** owned by §3 — the output-check catches it for
  reliability/search; headline accuracy honestly reflects α's compile quality.
- **Composer cost** (Python per-instance, codex #4): building `csp_α` per instance is the same cost as
  today; batch the dedP gate / cache per structure later.

### 5.3 How it pairs with α-as-search + the curriculum

- **α-as-search (north-star):** the structure head is the *proposer*; the certified floor + output-check is
  the *sound filter*. When confident, take the argmax structure (gradient path); when uncertain, top-k
  propose → verifier-filter → distill the winner (expert iteration). The structure head makes α-as-search
  concrete: the search space is "typed factor graphs over the mentioned cells."
- **Difficulty curriculum:** binary+pins (coloring/equality/ordering/eqchain/forcedcolor) → ternary
  (arithmetic/parity) → variadic+generic `rel` (OOD-novel) → real-NL phrasings at each tier. Occupancy-gated
  (codex highest-leverage): assert the structure-emission buckets in the run metadata.

### 5.4 Ranked by leverage

1. **Structure head + structure-sup loss + decode + `build_csp_from_struct`** (latent_organ.py + struct_io).
   The architectural core; nothing is testable without it. HIGHEST.
2. **Standalone structure-faithfulness probe** — α-emitted facts vs true facts (precision/recall/F1) on
   canonical templates, NO composer/LM wiring. De-risks the central question (can α emit the factor graph
   from hidden?) for ~nothing before touching the live path.
3. **Rewire `BankComposerOrgan` to consume α-emitted structure** + remove `rec["csp"]` from the forward
   (the actual "α emits, composer consumes α" change) + the no-metadata invariant guard.
4. **`alpha_struct_diag.py` acceptance gate** (§4) — how we KNOW it worked; the inverted SHUFFLED test.
5. **Training rewire** (Phase A structure warmup) in `train_bank_woven`.
6. **α-as-search + output-check reliability layer**, then **ternary/NL curriculum staging**.

### 5.5 THE SINGLE FIRST STEP

Implement **#1 + #2 together**: the `StructureHead` (pin + binary pair-relation logits over the mention
grid), `facts_to_struct_targets` / `build_csp_from_struct`, `decode_structure`, `structure_sup_loss`, and a
**standalone structure-faithfulness probe** that, on a few hundred canonical determined-query instances,
trains/evaluates ONLY the structure head against the free `p.facts` targets and reports per-relation
precision/recall/F1 and exact-factor-graph-match. No composer, no LM, no eval rewiring. This answers the
make-or-break question — *can α read the factor graph off the host hidden?* — at the lowest possible cost,
and produces the F1 number the §4 gate's criterion (4) depends on. If canonical F1 ≪ 0.8, the whole
ALPHA_STRUCT plan needs rethinking before any composer/eval work; if it's ~1.0, proceed to #3.
