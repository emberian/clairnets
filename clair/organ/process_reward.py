"""clair/organ/process_reward.py — the ORGAN-AS-PROCESS-REWARD (GLaDOS blocker 6 / RLVR stage 3).

RLVR with only an OUTCOME reward (exact answer-checker → 1.0/0.0) is SPARSE: on hard instances every
rollout in a group is wrong, the advantage is zero, and GRPO gets no signal. The organ supplies a DENSE
process signal that is meaningful even when the final answer is wrong — exactly the LSRL / RLTT idea:
credit the deduction PROGRESS, not just the destination.

The process reward for an emitted answer on a CSP problem (the organ's native domain) =
    cardinality-drop  : how far the SOUND organ narrowed the candidate space (Σ|dom| from full → fixpoint),
                        normalised so all-cells-singleton = 1.0 — this is dense (>0) the moment ANY
                        deduction happens, independent of the final answer;
    survival-depth     : how pinned the QUERY cell got (singleton = 1.0, untouched = 0.0);
  SOUNDNESS-GATED      : if the emitted value was ELIMINATED by the sound organ (provably impossible), the
                        process reward is 0 — we never credit a deduction the organ refutes.

Mixed with the outcome (`mix_reward`, default 0.7·outcome + 0.3·process) the total stays outcome-dominant
(retention / no reward-hacking) while being dense where outcome is 0. The narrowing is the verifier-gated
reduced product (certified floor + pretrained CoreNarrowOrgan), so the cardinality trajectory is SOUND.
"""
from __future__ import annotations

from dataclasses import dataclass

from .. import csp as C
from .bank import build_bank, certified_csp_reductions
from .compose import reduced_product
from .protocol import CSPState


@dataclass
class ProcessReward:
    process: float          # in [0,1], dense (the organ's deduction progress for this answer)
    cardinality_drop: float # normalised Σ|dom| reduction full→fixpoint
    survival_depth: float   # how pinned the query cell got
    sound: bool             # was the emitted value still alive in the sound narrowed domain?
    rounds: int             # composer rounds (per-step trajectory length)


class OrganProcessReward:
    """Computes the organ-as-process-reward on CSP problems via the verifier-gated reduced product
    (certified floor + pretrained CoreNarrowOrgan). Reusable across an RLVR run (loads the bank once)."""

    def __init__(self, dev="cpu", core_ckpt="runs/general_organ_full.pt", use_core=True):
        bank = build_bank(load_neural=use_core, dev=dev, core_ckpt=core_ckpt)
        core = bank.get("core_narrow_organ")
        self.reductions = certified_csp_reductions(bank) + ([core] if core else [])

    def narrow(self, csp: C.CSP, system=None, tags=()):
        """Sound narrowing trajectory: returns (final_state, trace). trace.applied carries the per-step
        (round, op, alive_before, alive_after) — the per-step cardinality drops."""
        full = CSPState.full(csp, system=system, tags=frozenset(tags))
        return reduced_product(full, self.reductions, verify=False)

    def reward(self, csp: C.CSP, query: int, emitted_value: int, system=None, tags=()) -> ProcessReward:
        """Dense process reward for emitting `emitted_value` at cell `query` of `csp`."""
        full = CSPState.full(csp, system=system, tags=frozenset(tags))
        out, tr = self.narrow(csp, system=system, tags=tags)
        full_alive = full.alive()                       # n*d
        n = csp.n
        narrowed = out.alive()
        # normalised cardinality drop: best case every cell → singleton (alive == n)
        denom = max(1, full_alive - n)
        card_drop = min(1.0, max(0.0, (full_alive - narrowed) / denom))
        # survival depth of the query cell (1 = singleton/solved, 0 = untouched)
        qd = len(out.dom[query])
        depth = 1.0 - (max(1, qd) - 1) / max(1, csp.d - 1)
        sound = emitted_value in out.dom[query]         # did the organ refute the emitted value?
        process = (0.5 * card_drop + 0.5 * depth) if sound else 0.0
        return ProcessReward(process=float(process), cardinality_drop=float(card_drop),
                             survival_depth=float(depth), sound=bool(sound), rounds=int(tr.rounds))


def mix_reward(outcome: float, process: float, w_outcome: float = 0.7, w_process: float = 0.3) -> float:
    """The LSRL-style mix: outcome-dominant (retention) + dense process signal. 0.7·outcome + 0.3·process."""
    return float(w_outcome * outcome + w_process * process)


# ============================================================ smoke (CPU)
def smoke():
    """Prove the process reward (1) computes, (2) is DENSE where the outcome is 0 (a wrong-but-alive
    answer earns cardinality-drop credit when the organ narrows without fully pinning the query — the
    realistic hard/sparse case), and (3) the SOUNDNESS GATE zeroes a value the organ provably refutes."""
    import numpy as np
    from .. import run_glados_staged as G
    opr = OrganProcessReward(core_ckpt="runs/general_organ_full.pt", use_core=True)
    rng = np.random.default_rng(0)
    print("== organ-as-process-reward smoke ==")
    dense_when_wrong = refuted_gated = alive_total = refuted_total = 0
    rows = []
    for rg in ("coloring", "equality", "arithmetic", "alldiff", "ordering"):
        for _ in range(20):
            p = G.make_problem(rng, rg, "id")
            csp = G.organ_csp(p)
            out, _tr = opr.narrow(csp)
            q = p.query
            survivors = sorted(out.dom[q])
            gold = p.answer if p.determined else None
            if len(survivors) >= 2:
                # the organ narrowed but did NOT fully pin q → a wrong-but-ALIVE value still gets credit
                wrong_alive = next((v for v in survivors if v != gold), None)
                if wrong_alive is not None:
                    r = opr.reward(csp, q, wrong_alive)
                    mixed = mix_reward(0.0, r.process)          # outcome 0 (wrong), but process is dense
                    alive_total += 1
                    if mixed > 0.0:
                        dense_when_wrong += 1
                    if len(rows) < 8:
                        rows.append((rg, "alive-wrong", len(survivors), r.cardinality_drop,
                                     r.survival_depth, r.sound, mixed))
            elif len(survivors) == 1 and gold is not None:
                # the organ fully pinned q → emitting a different value is REFUTED → process gated to 0
                wrong = next((v for v in range(csp.d) if v != gold), gold)
                r = opr.reward(csp, q, wrong)
                refuted_total += 1
                if r.process == 0.0 and not r.sound:
                    refuted_gated += 1
                if len(rows) < 8:
                    rows.append((rg, "refuted", 1, r.cardinality_drop, r.survival_depth, r.sound,
                                 mix_reward(0.0, r.process)))
    print(f"  {'rung':9s} {'case':12s} {'|surv_q|':>8} {'card_drop':>10} {'depth':>6} {'sound':>6} {'mix(out=0)':>10}")
    for rg, case, sq, cd, dp, sd, mx in rows:
        print(f"  {rg:9s} {case:12s} {sq:>8} {cd:>10.3f} {dp:>6.2f} {str(sd):>6} {mx:>10.3f}")
    print(f"  DENSE-WHEN-OUTCOME-0: {dense_when_wrong}/{alive_total} alive-but-wrong answers earned >0 "
          f"process credit (cardinality-drop signal)")
    print(f"  SOUNDNESS-GATE: {refuted_gated}/{refuted_total} organ-refuted wrong answers correctly gated to 0")
    assert dense_when_wrong > 0, "process reward must be DENSE (>0) on outcome-0 cases where q is under-pinned"
    assert refuted_total == 0 or refuted_gated == refuted_total, "soundness gate must zero refuted values"
    print("  SMOKE OK: organ-as-process-reward computes, is dense where outcome=0, soundness-gated\n")


if __name__ == "__main__":
    smoke()
