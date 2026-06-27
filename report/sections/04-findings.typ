#import "../helpers.typ": *

= Findings — the honest arc

This is the project's spine read in order, negatives included. The shape of the arc: the *coupling* works on a
frozen LLM; RLVR and scale are *not* the levers people assume; the naive end-to-end version *fails*; a staged
version *works*; and the live organ — once it became the *real* consolidated bank+composer rather than a
single-faculty stopgap — *engages causally*, is *legible*, *generalizes by reduction*, and is *robust by
construction*. The one thing that distinguishes all of this from an award-winning artifact that was secretly
reading the labels is the causal controls, run on every result.

== The checked-deductor coupling works (and what RLVR / scale actually buy)

A frozen LLM bolted to a checked deductor gains a capability it structurally lacked: *calibrated abstention*.
Deterministic accuracy rises to *88.8% vs a base 1.4%* — the base essentially cannot say "I can't conclude
that." Three sharpening results, each a "not what you'd guess":

- *RLVR fixes calibration, not capacity.* RL adds +18–20 points of abstain-recall OOD — it tunes *when* to
  defer — but does not raise the underlying solvable set (#arxiv("2504.13837"); see §6).
- *7B $approx$ 1B on the coupling.* Scaling the host 1B$arrow.r$7B does not improve the *reading* of the organ.
  Capacity is not the wall.
- *Grounding fixes extraction robustness, not exactness*, and the *rich-state set-valued readout makes
  degradation graceful and sound* — clean +58 > partial +28.7 > wrong +8 $approx$ corrupted $-3$; the LM mines
  a partial-but-sound output and down-weights confident-wrong cells. Rich beats a collapsed point-readout by
  +27.5 pts on partial states.

== Cold co-training fails — and staging fixes it

The decisive negative: a *learned* organ co-trained from scratch with the LM is *ignored*. The LM settles into
a lazy optimum — shortcut the answer from prompt text — and the causal controls go flat. This is the same
disease as SATNet's grounding collapse (§5) and the reasoning-shortcut theorem (#arxiv("2305.19951")): wherever
the loss admits a cheaper non-deductive optimum, it is taken. The fix is *not* to deny the LM the text; it is to
*decouple* training. Two intermediate results pinned the diagnosis:

- *Oracle readout (de-risk, positive).* Feed the LM the *true* narrowed lattice through structured $gamma$ and
  its LM head generates the right answer *causally* — corrupt the lattice and accuracy goes to 0 *even with the
  full problem in the prompt*, 100% OOD. The "latch gap" was a *bandwidth* problem; structured dense $gamma$
  solves it.
- *Frozen-learned-organ readout (positive, reconciling).* A *frozen, standalone-trained learned* organ reads
  through $gamma$ essentially as well as the oracle. So the failure was *distrust of a moving organ*, not the
  coupling.

== The staged generative readout works (readout-isolation milestone)

Putting the recipe together first yielded a *readout-isolation* deliverable: a *learned, general* organ causally
woven into generation across seven reasoning types, with the organ handed the *true* narrowed lattice. *Woven
98.8% · base 48% · text-LoRA 25%* in-distribution; OOD-phrasing 96.3%; *corrupt$arrow.r$2.7% with the full
problem text present*. This proved the LM reads back and generates *through* a certified state it cannot get
from the prompt — but it left the hard half (where does the lattice come from?) open.

== ⭐ The consolidated woven organ engages — live, on a certified floor

#keynote[
The live woven model is now the *real* consolidated organ (`organ.bank_woven`): $alpha$ latently compiles the
host hidden into a per-cell lattice; the *composer* runs the *certified floor* (Arc/Factor/Modular/GF2/Macro)
+ the pretrained narrower + $alpha$, all *verifier-gated*; $gamma$ reads the *composed* lattice. It *engages,
causally and necessarily*: true $approx$ *100%* $arrow.r$ corrupt $approx$ *0%* (a *~97–100 pt drop*), *lift
+77* over the no-organ floor, *no-op\@init bitwise 0.0*, *false-elim 0* vs the exact verifier. The levers
are the *two-stream* asymmetry ($alpha$ reads full text; the LM generates from a fact-ablated cells prompt) +
*$J_0$ direct $alpha$-supervision* (the SATNet grounding fix). The earlier "gate stays shut, drop = 0" was a
step-count under-training artifact, not a model verdict.
]

The structural payoff: because the certified ops run from the full domain and the meet of sound narrowings is
sound, the neural/$alpha$ proposals can only sharpen *toward* the exact transformer — never below it. The
injected lattice is therefore *miscompile-robust by construction*: a confidently-wrong $alpha$ cannot poison
it. This *subsumes* the earlier "train the organ to be robust" calibration story — robustness is now structural.

== Solve-by-reduction generalizes

The cross-type reduction graph (`organ.reductions`) turns one faculty into many. Every edge is *exact-verified
end-to-end against X's own solver*; cost-routing is Dijkstra with a certified-over-approximate tie-break. On the
*real* bank-woven model: a *route-required* family with *no direct faculty* (XOR-SAT) is solved *100% via the
reduction* (lift +47.5); *held-out reduction-paths 100%* (lift +37.5); base-with-LoRA-off 0% (the answer
exists only through the reduced-then-solved lattice). The LM's *recognize · route · read · decode* transfers
across family and representation — the structural Karp prior, handed to the model nearly for free.

== ⭐ The organ is legible — learned-free

#keynote[
A probe with *no trained decoder anywhere* (`interp_probe_bank`) reads $alpha$'s compile by *direct
set-comparison* to the exact $#raw("ded")_p$: overall *F1 0.90, recall 0.98* (rarely misses a true constraint;
precision 0.82), per-rung alldiff 0.97 > coloring 0.90 > arithmetic 0.89 > equality 0.86 > ordering 0.85;
candidate-set recovery 0.93–1.0 across cardinalities; the query reaches a singleton *100%*. The composer's
reduced-product trace is legible by construction (which faculty fired, which neural proposals were gated). We
can read *what the organ posed and what it concluded* — the interp advantage over a dense bypass is real
(n = 300).
]

*Scope, stated plainly.* This is the *easy* regime, where the problem's true constraint structure is handed to
the composer. There the *certified floor carries answer-correctness on its own*: the probe shows answer-acc
1.0 *whether or not* $alpha$ is faithful, with $"corr"("faithful", "correct") = 0$ — *no variance to
correlate*. So the bottleneck test — *does $alpha$-from-hidden drive correctness?* — is *saturated /
inconclusive* here. The genuinely open piece is *$alpha$-on-real-NL*: $alpha$ compiling from natural language
with *no provided structure*, where correctness drops and failures attribute. The `eval_real_nl` hook is wired;
the result is the next frontier. *Proven: legible, engages, certified-floor-robust, reduction-generalizes.
Open: $alpha$-from-real-text driving correctness.*

== The coupling is residual-stream-agnostic — 7/7 bases

The graft (`organ.graft`) only ever *reads* a layer's output hidden and *adds* to a later one through the
zero-init gate, so it ports to any residual stream. Proven across ~100M$arrow.r$32B and three architecture
families, each *bitwise no-op\@init* + gate-open-moves-logits:

#figure(
  table(columns: (auto, auto, auto, auto), inset: 5pt, align: (left, left, center, center),
    stroke: 0.4pt + luma(180),
    table.header([*base*], [*architecture*], [*layers / hidden*], [*no-op\@init*]),
    [Pythia-160M], [GPT-NeoX], [12 / 768], [✓ 0.0],
    [OLMo-2-1B], [Olmo2], [16 / 2048], [✓ 0.0],
    [SmolLM3-3B], [SmolLM3], [36 / 2048], [✓ 0.0],
    [Gemma-4-12B], [multimodal-wrapped], [48 / 3840], [✓ 0.0],
    [Qwen-3.6-27B], [full / *linear-attn* hybrid], [64 / 5120], [✓ 0.0],
    [Nemotron-H-8B], [*Mamba-2 + attn + MLP* hybrid], [52 / 4096], [✓ 0.0],
    [OLMo-3-32B], [Olmo3 (QLoRA 4-bit)], [64 / 5120], [✓ 0.0],
  ),
  caption: [The 7/7 model-agnostic matrix. The Nemotron-H datapoint is decisive: the inject point is *on a
  Mamba-2 SSM block*, and the residual edit is still clean — every block boundary is a plain residual add,
  regardless of mixer.],
)

== The CPU wall, crushed (Rust hot path)

The pretrain bottleneck was the single-threaded Python exact-$#raw("ded")_p$ ground-truth path, not the GPU (the
organs are under 0.5M params). `clair_fast/` (PyO3 + rayon) ports it — GIL-released in-process parallelism replaces
the pickling ProcessPool — for *29–56× wall-clock* (per-call 6.7× small $arrow.r$ 55× large), *bitwise-identical*
(13,590 comparisons, 0 mismatches; a pure-Python fallback gated by `CLAIR_NO_FAST`). This unblocks the real
organ-pretrain at the $d^n$-explosion budgets that the larger-budget organ needs.

== The causal-control table — the actual contribution

Every woven result is gated behind adversarial controls. A *sound* organ guarantees a *correct* answer if the
organ produced it; it does *not* prove the organ produced it. These controls do.

#figure(
  table(columns: (auto, 1fr, auto), inset: 5pt, align: (left, left, center),
    stroke: 0.4pt + luma(180),
    table.header([*control*], [*what it injects / what it proves*], [*result*]),
    [`true`], [the real composed lattice — the intended path], [pass],
    [`shuffle`], [a *different problem's* lattice; unchanged accuracy $arrow.r$ bypass], [collapses],
    [`permute`], [permute candidate/value labels; tests intended semantics, not an alias], [$arrow.r 0$],
    [`corrupt`], [flip *only the query cell's* survivors to a wrong set; must change the answer], [$arrow.r 0$],
    [`zero`], [no injection at all (the *true* zero) — the no-organ floor], [lift +77],
  ),
  caption: [The causal controls — exactly the ones SATNet did not run. `wired` requires corrupt/shuffle/permute
  to collapse *and* the gate open; `necessary` requires woven $>>$ text-LoRA (the two-stream lever). Both must
  hold for `engages` — and on the consolidated organ they do.],
)

== A (explicit) vs B (latent): the bitter-lesson path that became the default

The latent path B — feed the organ *only* the LM's hidden, no extracted factors — narrows 94% of $#raw("ded")_p$
in-distribution, is *provably equivariant* (perm/size error = 0), and after wide-$N$ + multi-task training is
sound and phrasing-robust (false-elim 8.8%$arrow.r$0.9%; recall 76$arrow.r$88). It carried a ~20-point
OOD-by-size completeness gap vs explicit factors — but it is the path that the consolidated organ adopts and
*de-risks structurally*: the certified floor now backs the latent $alpha$, so the residual $alpha$-leak can no
longer cause an unsound answer, only a less-narrowed one.

== Macro-deduction: log-depth on the bounded-width side

Checked *macros* give sublinear-depth sound reasoning: $ceil(log_2 L)$ macro-applications replace ~$L$ base
steps, *identical exact $#raw("ded")_p$ fixpoint*, *0 false-elim, verified to $L = 256$*. The compression is
*symbolic transition-monoid composition* (graph-squaring, #arxiv("2210.10749")), not the net. Cost is polynomial
in $S = k^w$, so it works *only on the bounded-width side*: *width is the wall; bounded-width depth is a
log-depth shortcut.*

== Blades and the recurring null

Blades (the GA grades) help *affine generalization* specifically (grades$->$arities), and buy *soundness*
off-distribution: zero-shot to a never-seen arity-3 relation, a generic relation-table proposer's false-elim
collapses to 0.36 while the blade's grade-3 path stays 0.07 (5× more sound). The recurring signal across the
bank: the *affine wall (3-way correlation) is the level-0 ceiling* — single ternary constraints need grade-3,
XOR *systems* need the certified GF(2) row-space organ, and neither is beaten by depth alone. The geometric
product as a generic mixer remains a clean null; its value is structural, not magical.
