"""clair/organ/compose.py — VERIFIER-GATED certified-reduction composition (the reduced product).

The codex zoo review (notes/codex_zoo_review.md) is explicit about why composition cannot just be
"chain organs and output-verify at the end": output-verification makes ACCEPTED ANSWERS sound, NOT
intermediate organ states — "one unsound reduction poisons the next organ before the final check."
So composition must be over REDUCED PRODUCTS of reductions, each with its own soundness rule:

  * a CERTIFIED reduction (sound-by-construction) is trusted: its output is asserted to be a subset
    of the input (a real narrowing) and met in directly;
  * a NEURAL / approximate reduction is VERIFIER-GATED: we only keep the eliminations the exact
    per-cell verifier (clair.csp.exact_dedP) also makes, so a wrong neural proposal can never poison
    the shared state. (Soundest path #5 in the review: "learned routing only over CERTIFIED outputs".)

Each round applies every APPLICABLE csp-domain reduction to the current state and MEETS (pointwise
intersects) the results; iterate to a fixpoint. The meet of sound narrowings is a sound narrowing,
so the composite stays sound by construction — the final `verify` is a belt-and-braces check that
false-elim vs the exact oracle is 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import csp as C
from .protocol import CSPState, Reduction, exact_oracle, false_elim


@dataclass
class Trace:
    rounds: int = 0
    applied: list = field(default_factory=list)     # [(round, reduction_name, alive_before, alive_after)]
    gated: list = field(default_factory=list)        # neural reductions that were verifier-gated
    sound: bool = True                               # false-elim vs exact oracle == 0
    final_status: str = "open"


def _verifier_gate(state: CSPState, proposed: CSPState) -> CSPState:
    """Keep only the eliminations the exact per-cell verifier also makes — so an unsound neural
    proposal cannot drop a value some real solution uses. proposed' = proposed `join` exact_dedP,
    i.e. we re-admit any value the proposer killed that the verifier keeps."""
    exact = C.exact_dedP(state.csp, state.dom)
    # re-admit any value the proposer killed that the exact verifier keeps, then clamp to the input
    safe = tuple((proposed.dom[i] | (exact[i] & state.dom[i])) & state.dom[i] for i in range(state.csp.n))
    return state.with_dom(safe)


def reduced_product(state: CSPState, reductions, verify: bool = True, max_rounds: int = 64):
    """Compose `reductions` over a shared CSPState by the verifier-gated reduced product.

    Returns (state', trace). Certified reductions are trusted (asserted subset); neural/approximate
    ones are intersected with what the exact verifier confirms. Soundness (false-elim 0 vs the exact
    oracle on the ORIGINAL state) is asserted at the end when verify=True."""
    applicable = [r for r in reductions if r.state_type == "csp-domain"]
    origin = state
    tr = Trace()
    for rnd in range(1, max_rounds + 1):
        tr.rounds = rnd
        cur = state
        changed = False
        for r in applicable:
            if not r.applies(cur):
                continue
            before = cur.alive()
            raw = r.reduce(cur)
            cert = r.certificate()
            if cert.sound:
                assert raw.issub(cur), f"certified reduction {r.name} returned a non-subset (unsound!)"
                out = raw
            else:
                out = _verifier_gate(cur, raw)          # gate neural proposals through the verifier
                if out.dom != raw.dom:
                    tr.gated.append(r.name)
            cur = cur.meet(out)
            after = cur.alive()
            if after != before:
                changed = True
                tr.applied.append((rnd, r.name, before, after))
        if cur.dom == state.dom:
            break
        state = cur
        if not changed:
            break
    tr.final_status = state.status()
    if verify:
        ref = exact_oracle(origin)
        fe = false_elim(state, ref)
        tr.sound = (fe == 0)
        assert fe == 0, f"composition was UNSOUND: dropped {fe} value(s) the exact verifier keeps"
    return state, tr


def reduced_product_batch(states, shared, extra=None, max_rounds: int = 64):
    """BATCHED reduced product: run the verifier-gated reduced product for a LIST of states at once, fanning
    every reduction that exposes `reduce_batch` (e.g. the neural CoreNarrowOrgan) across the whole active set
    in a single call, while keeping the EXACT per-instance round structure of `reduced_product`.

    `shared` are the reductions applied to every instance (the certified floor + the gated neural organ, in
    order); `extra[i]` (optional) are per-instance reductions appended AFTER the shared ones (α's per-cell
    gated proposal, which differs per instance) — this mirrors `reds = base_reductions + [α_i]` in the serial
    composer. The result is BITWISE-IDENTICAL to calling `reduced_product(state_i, shared + extra[i],
    verify=False)` for each i (same reduction order, same verifier gate, same per-instance fixpoint /
    early-stop), because the only change is that independent instances are evaluated together and batchable
    reductions share one forward. Asserted in clair.organ.selftest. Returns (list[state'], list[Trace])."""
    n = len(states)
    extra = extra if extra is not None else [() for _ in range(n)]
    cur_state = list(states)
    traces = [Trace() for _ in range(n)]
    active = [True] * n
    shared = [r for r in shared if r.state_type == "csp-domain"]

    def _apply(r, i, cur):
        """Apply reduction r to instance i's current state `cur` with the serial gate; returns new cur."""
        before = cur.alive()
        raw = r.reduce(cur)
        if r.certificate().sound:
            assert raw.issub(cur), f"certified reduction {r.name} returned a non-subset (unsound!)"
            out = raw
        else:
            out = _verifier_gate(cur, raw)
            if out.dom != raw.dom:
                traces[i].gated.append(r.name)
        cur = cur.meet(out)
        after = cur.alive()
        if after != before:
            traces[i]._changed = True
            traces[i].applied.append((traces[i].rounds, r.name, before, after))
        return cur

    for rnd in range(1, max_rounds + 1):
        cur = [cur_state[i] if active[i] else None for i in range(n)]
        for i in range(n):
            if active[i]:
                traces[i].rounds = rnd
                traces[i]._changed = False
        # SHARED reductions, in order; batch the ones exposing reduce_batch over the applicable active set.
        for r in shared:
            idxs = [i for i in range(n) if active[i] and r.applies(cur[i])]
            if not idxs:
                continue
            if hasattr(r, "reduce_batch") and not r.certificate().sound and len(idxs) > 1:
                # batched neural-guidance proposal -> per-instance verifier gate + meet (gate is exact/cheap)
                raws = r.reduce_batch([cur[i] for i in idxs])
                for j, i in enumerate(idxs):
                    before = cur[i].alive()
                    out = _verifier_gate(cur[i], raws[j])
                    if out.dom != raws[j].dom:
                        traces[i].gated.append(r.name)
                    cur[i] = cur[i].meet(out)
                    after = cur[i].alive()
                    if after != before:
                        traces[i]._changed = True
                        traces[i].applied.append((rnd, r.name, before, after))
            else:
                for i in idxs:
                    cur[i] = _apply(r, i, cur[i])
        # PER-INSTANCE extras (α's gated proposal), appended after the shared reductions (serial order).
        for i in range(n):
            if not active[i]:
                continue
            for r in extra[i]:
                if r.state_type == "csp-domain" and r.applies(cur[i]):
                    cur[i] = _apply(r, i, cur[i])
        # per-instance convergence / state advance (identical to reduced_product's break conditions)
        any_active = False
        for i in range(n):
            if not active[i]:
                continue
            if cur[i].dom == cur_state[i].dom:
                active[i] = False
            else:
                cur_state[i] = cur[i]
                if not traces[i]._changed:
                    active[i] = False
            any_active = any_active or active[i]
        if not any_active:
            break
    for i in range(n):
        traces[i].final_status = cur_state[i].status()
    return cur_state, traces
