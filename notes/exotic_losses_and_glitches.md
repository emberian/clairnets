# Exotic losses, the next-token question, and the glitch probe

Three threads (Ember), captured. Status: SPECULATIVE / post-ALPHA_STRUCT — do NOT add before the base text→organ
reasoner works (adding loss terms before the model trains is how you get un-debuggable runs).

## 1. Exotic losses (beyond dominate-dedₚ + LM-CE + structure-sup)
- **Faithfulness/usage-disentangling loss** — today we MEASURE corr(α-faithful, correct) post-hoc; instead OPTIMIZE
  calibrated organ-use: reward "used the organ *because* it was right", penalize "right answer, ignored organ"
  (organ wasted) and "wrong answer, trusted organ" (over-trust). A coupling/MI objective → trains the *calibrated*
  trust we keep circling (trust-when-reliable, take-it-lightly-when-not). The principled version of the calibration
  study.
- **MDL / description-length loss** — the organ should COMPRESS the problem: if routing through it lets the LM
  predict the answer in fewer bits than text-only, the organ earned its keep, quantified. Natural anti-shortcut
  pressure + a principled "is the organ pulling weight" signal.
- Caveat: the current losses are well-motivated; these are speculative. File for AFTER ALPHA_STRUCT works.

## 2. Does this help NEXT-TOKEN / general performance? (the thesis make-or-break — UNKNOWN, must measure)
- The bet: Stage-3 consolidation (Dolma ⊕ reasoning) folds organ-use into general behavior → the organ becomes a
  RECRUITED faculty; on CSP/logic/arithmetic-shaped spans in ordinary text the LM offloads → predicts better.
- Honest risks: (a) helps only on organ-shaped problems (narrow win, still publishable, not the dream); (b) HURTS
  general perplexity (Tier-3 no-harm exists to catch this); (c) the organ never gets recruited on natural text if α
  can't compile real NL (the frontier we mapped: NL structure-F1 0.73→0.82 w/ LoRA, not solved).
- We WILL measure it: eval_suite Tier-2 transfer (GSM8K/BBH) + Tier-3 no-harm (MMLU/perplexity). Framing: "uplift
  on reasoning + no-harm elsewhere" is the MINIMUM bar; "helps next-token broadly" is the STRETCH — test, don't assume.

## 3. THE GLITCH PROBE — organ misapplication (the best idea; QUEUED as the first study on trained ALPHA_STRUCT)
When the LM OVER-applies the organ — type-inference on social relations, CSP-narrowing on a poem, Schreier-Sims on
"who likes whom at the party", arithmetic-deduction on a metaphor — it's (a) funny, (b) genuinely diagnostic: the
analog of human cognitive overreach (treating fuzzy things as formal). The failure modes reveal what the model
thinks the organ is FOR, and it's a real interp+safety/calibration probe.

**Design (inference-only, cheap, no training — runs on the trained ALPHA_STRUCT model):**
Feed deliberately INFORMAL / ambiguous / category-mismatched inputs (social, emotional, aesthetic, metaphorical,
narrative) and measure:
- (i) does the ROUTER fire an organ at all? (ideally not — knowing-when-NOT-to-reason-formally = calibration)
- (ii) WHICH faculty does it misroute to? (type-inference on feelings? graph on a friendship? Ising on a mood?)
- (iii) does the certified FLOOR catch the nonsense (abstain) or does the LM COMMIT to a confidently-wrong formal
  answer? (the safety question: over-trust of a formal tool on an informal domain)
- (iv) harvest the funniest/most-revealing concrete outputs.
Passes the money-bar: decision-relevant (calibration/safety — does it over-trust the formal tool), genuinely
uncertain (no idea what it does on a sonnet), cheap (inference-only).

## Discipline note (Ember's money-bar)
A box is worth it ONLY when the answer is (a) decision-relevant — changes what we build — AND (b) genuinely
uncertain. Pre-mortem each box: "if it comes back the other way, does what we build change?" If no → CPU check /
back-of-envelope, not an L40S. (Studies that re-confirmed known theory — affine-is-capacity, data-alone-caps,
all-views-crowds — should've been cheaper. SHUFFLED + α-can-read-structure were the gold standard: decision-
relevant AND uncertain.)
