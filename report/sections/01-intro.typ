#import "../helpers.typ": *

= The bet, in one paragraph

Language models confabulate because nothing in their forward pass is allowed to say "I cannot conclude
that." The bet of GLaDOS is that you can give a transformer a *reasoning channel that can abstain* by
welding a small, learned deduction module into its residual stream — and that the module can stay
*reliable* not because it is a verifier, but because we *check its output*. The check is the whole trick.
A learned module is free to be loose, general, even emergent; if a cheap exact check sits at the boundary,
the only outputs that survive are the correct ones, and the rest become honest abstentions. We call the
module an #term[organ] and the assembled system GLaDOS.

The reason to want this *woven*, rather than as an external SAT/SMT tool the LLM calls in text, is
bandwidth. A text round-trip to a solver is non-differentiable and lossy: the LLM never sees the solver's
internal certified state, gets no gradient on "did I state the problem right," and inherits none of the
solver's $bot$ (don't-know) channel. Weaving keeps the interface differentiable end-to-end and lets the
LLM's *own* head generate from a hidden state already saturated by the deduction.

== The spine

The organ's state $a_t$ is a point in a per-position #term[powerset lattice] — for each variable, the set
of candidate values still alive. One step of deduction proposes, then commits only by *narrowing*:

$ a_(t+1) = a_t inter Pi (f_theta (a_t, p, m)) $

Here $f_theta$ is a *learned* cell (it proposes, loosely), $Pi$ projects the proposal into the lattice, and
the meet $inter$ guarantees $a_(t+1) subset.eq a_t$: the step can only *remove* candidates, never add one.
The bilinear/learned part *proposes*; the lattice meet is the sole *authority* on elimination. That single
algebraic fact — monotone narrowing — is what makes the recurrence sound-by-construction *early in
training* (it is the training-wheels inductive bias), and it is what an exact verifier at the output can
later relax: the endpoint is a general, emergent reasoner that stays sound because the *boundary* checks it,
not because every internal step is provably correct.

== The one-sentence architecture

#keynote[
At layer $k$ of the host LLM, a learned map *$alpha$ compiles* the hidden state into a #term[typed
factor-graph program] — variables, domains, and factors, *not* a pre-solved answer; the *organ narrows*
that program recurrently, *between* host layers (no backprop through repeated host passes); a learned map
*$gamma$ writes* the certified state *densely and structurally* back into the residual stream through a
*zero-init gate*; the host's *LM head generates* the answer from the saturated hidden state; an *exact
verifier* supplies the reward.
]

Two phrasings of the same discipline run through the whole paper. First: *soundness is a permission, not a
prescription.* The output check buys correctness-given-the-organ; it lets us make the organ loose, learned,
and general instead of a hand-built verifier. Second — and this is the lesson that cost us the most —
*a sound organ can still be bypassed.* The check guarantees the answer is right *if the organ produced it*;
it says nothing about whether the organ, versus the prompt text or a LoRA, actually produced it. Only
adversarial causal controls (shuffle / permute / corrupt the organ's state and watch accuracy collapse)
prove the deduction is load-bearing. We run them on every result, and they are the real contribution.