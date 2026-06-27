# Codex organ-architecture critique (pre-pretrain)

## Verdict
The certified floor is a real strength for reliability, but it's the current CRUX. In the easy bank-woven
regime GLaDOS is **a robust checked solver/readout with metadata structure — NOT yet a demonstrated
text-to-organ reasoner.** Some architecture *prose* slides from "α compiles a lattice from hidden" into
"α compiles the problem constraints" — which the live code does NOT prove.

## Proven
- Sound CSP composition (α/neural gated by exact dedP in compose.py).
- The live woven really uses bank+composer+γ with α as a gated proposal (bank_woven.py).
- Easy-regime engagement/interp real (interp_bank: true 100%, corrupt 0.25%, α F1 0.896, recall 0.982).
- Reduction-graph edges are typed + verified as reductions.

## Over-claimed / aspirational (the honest gaps)
- **α does NOT compile relational structure** — it compiles per-cell candidate SETS; `rec["csp"]` provides
  the structure (bank_woven.csp_spec_for_record). THE CRUX.
- **"solve-by-reduction generalizes" ≠ "LM learned to reduce X→Y"** — records carry the target csp Y directly.
- organ-process-reward is metadata-CSP-only (reads metadata["csp"]).
- **the difficulty-controlled pretrain is NOT the canonical pretrain path** — organ/train.py delegates to the
  OLD RUNGS (run_glados_staged); the fixed stream lives separately in stream.py. (We'd waste the pretrain.)

## The 6 answers
1. **CRUX — make α causal by REMOVING privileged structure** (not just corrupting γ). 3-arm experiment:
   TRUE_STRUCT (current) · **SHUFFLED_STRUCT** (α-prompt correct but rec["csp"] from ANOTHER instance → if
   accuracy follows metadata, α isn't driving) · **ALPHA_STRUCT** (α must EMIT the structure; composer operates
   ONLY on α-produced structure; answer output-checked vs hidden truth). **The floor must sit under α's compiled
   semantics, not under metadata.**
2. **Interreducer/γ**: dumping K views = bandwidth crowding. ROUTE FIRST, read 1-2 deeply + a provenance channel
   (source_type/chosen_path/cert_strength/cost/decode_map); view-dropout + irrelevant-view decoys; learned top-k
   not all-views.
3. **Generalization**: the gap is α SEMANTICS under shift, not solver variety. Add an α STRUCTURE head
   (factors/reduction-path/couplings, not just candidate sets); hard-negative NL pairs (same entities, minimally
   changed relation); occupancy-gated data (assert buckets in run-metadata); query-centric γ compression.
4. **Performance**: composer is Python-per-instance, exact-gated INSIDE the forward — batch the Rust dedP gate,
   cache certified-floor per structure, route only applicable faculties, don't recompute γ per closed-set
   candidate. The Ising readout confidence is HARDCODED fully-confident (readout_bridge:69) — don't sell as calibrated.
5. **LM integration**: weakest link = α's OBJECTIVE. J0 teaches dedP candidate-SETS, not the underlying PROGRAM →
   α becomes an answer-set predictor without learning compilation. Fix: γ provenance+uncertainty + force
   decode-from-route (recognize type → choose reduction → solve target → decode answer).
6. **Paper risks**: (1) "you gave the model the parsed problem" — TRUE for the key results. (2) wasting pretrain on
   the old easy sampler. (3) claiming cross-type reduction learning when the target's handed in.

## HIGHEST-LEVERAGE (before pretrain)
Make `clair.organ.train pretrain` CONSUME the difficulty-controlled streaming spec OR FAIL. Add a PREFLIGHT
OCCUPANCY REPORT (required-level/treewidth/depth/random-relations/composition/reduction share); BLOCK if still the
old RUNGS distribution. Don't spend the pretrain budget through the old run_glados_staged path.
