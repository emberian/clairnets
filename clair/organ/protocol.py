"""clair/organ/protocol.py — THE TYPED SPINE of the GLaDOS organ.

This is the abstract interface every validated beast implements so they (a) COMPOSE as a
reduced product and (b) plug into the LLM uniformly through one structured-gamma readout. It is
the protocol the codex zoo review (notes/codex_zoo_review.md) named as the precondition for the
whole zoo to hold: "every domain has explicit semantics: alpha, gamma, top/bottom, meet/reduction,
verifier/certificate, abstain. Else 'domain x operation' = a taxonomy of vibes."

All soundness claims here are OPERATOR-level and RELATIVE TO THE GIVEN STATE — see notes/soundness.md
for the (A)/(B)/(C)/(D) hierarchy (operator-sound-rel-given-CSP is (A); it is NOT answer-soundness (D)).

The contract, per beast:
  * an abstract STATE (the lattice element it narrows),
  * reduce(state) -> state' with gamma(state') subset gamma(state)  (a SOUND narrowing OF THAT STATE),
  * a CERTIFICATE saying HOW the soundness holds (sound-by-construction / neural-guidance /
    approximate) and the completeness caveat,
  * abstain (reduce is a no-op when the beast can prove nothing locally),
  * an optional NEURAL-GUIDANCE hook (raw proposer logits, for the LLM / verifier-gated meet),
  * the structured-gamma READOUT signature: survival(state, K) -> [n, K] in {0,1}, the exact
    tensor clair.oracle_readout.OracleGamma projects (zero-init gate => no-op at init) into OLMo's
    residual stream. This is the single uniform channel the LLM reads every organ through.

The COMMON state for the certified narrowing organs is `CSPState` (per-cell domains over a finite
clair.csp.CSP). It is the reduced-product spine: reductions over it compose by pointwise meet
(clair.csp shows pair_strictly_richer_than_cell etc.). Beasts whose native semantics are NOT a
per-cell lattice (forward-chaining closure, energy simplex) declare their own state_type and do not
reduced-product with the CSP organs — composing them needs an explicit bridge, not a silent fusion.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import csp as C


# ============================================================ certificate
@dataclass(frozen=True)
class Certificate:
    """How a reduction's soundness is guaranteed (the thing the composer trusts or must gate).

    SCOPE: `sound` is OPERATOR soundness RELATIVE TO THE GIVEN CSP (level (A) in notes/soundness.md):
    reduce(s) drops no value used by a solution OF THE STATE IT IS HANDED. It is NOT answer-soundness
    — it says nothing about whether that state is the right problem (the uncertified α-compile gap, (C)).
    Do not read `sound=True` as 'certified => correct answer'."""
    sound: bool          # operator-sound REL. given CSP: reduce(s) is a GUARANTEED gamma-subset of s
                         # (drops no solution OF s). NOT answer-soundness — see notes/soundness.md (A) vs (D).
    kind: str            # 'sound-by-construction' | 'neural-guidance' | 'approximate'
    complete: str        # honest completeness note (what it can / cannot fully solve)
    detail: str = ""

    def __str__(self):
        tag = {"sound-by-construction": "CERTIFIED", "neural-guidance": "NEURAL",
               "approximate": "APPROX"}.get(self.kind, self.kind)
        return f"[{tag}] sound={self.sound}  completeness: {self.complete}"


# ============================================================ the composable CSP-domain state
@dataclass
class CSPState:
    """Per-cell lattice over a finite CSP: dom[i] subset {0..csp.d-1} = the still-alive values at
    cell i (== clair.csp's `dom`). The reduced-product spine. `system` optionally carries a native
    algebraic structure (clair.modular.LinSystem, or ('gf2', A, b)) so a SPECIALIZED certified
    reduction can solve it exactly; reductions without their system fall back to (csp, dom) only.
    `tags` mark structure (e.g. 'xor', 'modular', 'alldiff', 'path') for `applies()` dispatch."""
    csp: C.CSP
    dom: tuple
    system: Any = None
    tags: frozenset = field(default_factory=frozenset)

    @staticmethod
    def full(csp: C.CSP, system: Any = None, tags=()) -> "CSPState":
        return CSPState(csp, csp.full(), system, frozenset(tags))

    def status(self) -> str:
        return C.status(self.dom)                       # 'solved' | 'conflict' | 'open'

    def meet(self, other: "CSPState") -> "CSPState":
        """Reduced product: pointwise domain intersection (sound if both are sound narrowings)."""
        return CSPState(self.csp, tuple(self.dom[i] & other.dom[i] for i in range(self.csp.n)),
                        self.system, self.tags)

    def with_dom(self, dom) -> "CSPState":
        return CSPState(self.csp, tuple(dom), self.system, self.tags)

    def issub(self, other: "CSPState") -> bool:
        """self.dom subset other.dom componentwise (the soundness relation: narrowing only)."""
        return all(self.dom[i] <= other.dom[i] for i in range(self.csp.n))

    def alive(self) -> int:
        return sum(len(c) for c in self.dom)


# ============================================================ the abstract organ / reduction
class Reduction(ABC):
    """One organ as a typed, sound(-or-gated) narrowing operator. Implements the spine contract."""

    name: str = "reduction"
    domain: str = ""                  # human label of what it reasons about
    state_type: str = "csp-domain"    # which State it consumes; csp-domain organs reduced-product
    verifier_only: bool = False       # True => the ORACLE/verifier (e.g. exact_dedP), NOT a deployable
                                      #         runtime reduction; excluded from certified_csp_reductions

    # -- core narrowing --------------------------------------------------------------------------
    @abstractmethod
    def applies(self, state: Any) -> bool:
        """Is this reduction relevant to `state`? (right state_type + required structure present)."""

    @abstractmethod
    def reduce(self, state: Any) -> Any:
        """Return a narrowed state' with gamma(state') subset gamma(state). MUST be a no-op
        (return an equal state) when the organ can prove nothing — that is the ABSTAIN."""

    @abstractmethod
    def certificate(self) -> Certificate:
        """How soundness is guaranteed + the completeness caveat."""

    # -- abstain ---------------------------------------------------------------------------------
    def abstained(self, before: Any, after: Any) -> bool:
        """Did reduce narrow nothing? (default: CSPState domain equality)."""
        if isinstance(before, CSPState) and isinstance(after, CSPState):
            return before.dom == after.dom
        return before == after

    # -- optional neural-guidance hook -----------------------------------------------------------
    def guidance(self, state: Any):
        """Raw proposer signal (e.g. per-(cell,value) survival logits) for verifier-gated meet or
        for training. None for purely-symbolic organs."""
        return None

    # -- the uniform structured-gamma readout (the LLM channel) ----------------------------------
    def survival(self, state: Any, K: int) -> np.ndarray:
        """The structured-gamma readout: a per-(cell, candidate) survival matrix surv[n, K] in
        {0,1}, exactly the tensor clair.oracle_readout.OracleGamma projects into OLMo's residual
        stream (gated by a zero-init tanh scalar => bitwise no-op at init). One uniform channel for
        every organ. Default = the CSP-domain readout (clair.run_glados_staged.surv_from_dom)."""
        if not isinstance(state, CSPState):
            raise NotImplementedError(f"{self.name} must override survival() for {self.state_type}")
        n = state.csp.n
        surv = np.zeros((n, K), dtype=np.float32)
        for i in range(n):
            for v in state.dom[i]:
                if v < K:
                    surv[i, v] = 1.0
        return surv

    def __repr__(self):
        return f"<Reduction {self.name} [{self.state_type}] {self.certificate().kind}>"


# ============================================================ the exact CSP-domain verifier
def exact_oracle(state: CSPState) -> CSPState:
    """The independent ground-truth narrowing for the CSP-domain spine: clair.csp.exact_dedP — the
    strongest SOUND per-cell transformer (alpha o gamma). This is the VERIFIER the composer uses to
    gate neural reductions and to certify soundness (false-elim == 0) in the self-tests."""
    return state.with_dom(C.exact_dedP(state.csp, state.dom))


def false_elim(narrowed: CSPState, reference: CSPState) -> int:
    """Count values the `narrowed` state dropped that the sound `reference` keeps (must be 0 for a
    sound narrowing). reference is typically exact_oracle(full)."""
    return sum(len(reference.dom[i] - narrowed.dom[i]) for i in range(narrowed.csp.n))
