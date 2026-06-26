"""clair/organ/bank.py — THE REGISTRY of validated beasts, each wired as a protocol.Reduction.

Every entry IMPORTS + adapts the already-validated, committed organ code (clair.csp / clair.modular
/ clair.xor_wall / clair.macro_deduct / clair.fol / clair.energy_organ) and the trained checkpoints
(runs/general_organ_full.pt, runs/modular_organ.pt). NOTHING here re-derives an algorithm; it only
adapts each beast to the typed spine in clair.organ.protocol so they compose + read out uniformly.

Status partition (the honest table — see README):
  CERTIFIED (sound-by-construction, CSP-domain, reduced-product): ArcConsistency, FactorConsistency,
      ExactDedP, Modular (SNF), GF2 (row-space), Macro (reach-doubling path solver).
  CERTIFIED (own state type, standalone): Unification (forward-chaining least Herbrand model).
  APPROXIMATE (own state type): Energy (mean-field constraint optimisation; exact brute oracle).
  NEURAL-GUIDANCE (CSP-domain, must be verifier-gated): CoreNarrowOrgan (the union-trained general
      deductor), BladeAffineOrgan (the blade/grade-prior deductor for affine/modular).

  build_bank(load_neural=True) returns {name: Reduction}. Pure-certified entries need no torch and
  no checkpoints; neural entries load lazily on first use.
"""
from __future__ import annotations

import itertools as it
from typing import Any

import numpy as np

from .. import csp as C
from .. import modular as MOD
from .. import xor_wall as XW
from .. import macro_deduct as MAC
from .. import fol as FOL
from .protocol import Certificate, CSPState, Reduction


# ============================================================ CERTIFIED CSP-domain narrowers
class ArcConsistency(Reduction):
    """Generalized-arc-consistency to a fixpoint (clair.csp.ac_step). The cheap local level-0
    operator; sound, abstains on affine/Hall structure (the documented wall)."""
    name = "arc_consistency"
    domain = "generic finite CSP (local consistency)"

    def applies(self, state):
        return isinstance(state, CSPState)

    def reduce(self, state: CSPState) -> CSPState:
        dom, _ = C.to_fixpoint(C.ac_step, state.csp, state.dom)
        return state.with_dom(dom)

    def certificate(self):
        return Certificate(True, "sound-by-construction",
                           "level-0 (per-cell AC); abstains on affine/parity/Hall (needs higher level)")


class FactorConsistency(Reduction):
    """Level-k generalized (factor) consistency (clair.csp.solve_factor / factor_step). Captures
    3-cell structure (affine/arithmetic) the pair/cell levels cannot. Sound; complete up to width k."""
    name = "factor_consistency"
    domain = "finite CSP, k-ary factors (level-2 reduced product)"

    def __init__(self, k: int = 3):
        self.k = k

    def applies(self, state):
        return isinstance(state, CSPState)

    def reduce(self, state: CSPState) -> CSPState:
        factors = C.default_factors(state.csp, self.k)
        st = C.factor_init(state.csp, factors, state.dom)
        for _ in range(state.csp.n * state.csp.d * state.csp.d + 2):
            nxt = C.factor_step(state.csp, st, factors)
            if nxt == st:
                break
            st = nxt
        cells = C.factor_cells(state.csp, st, factors)
        return state.with_dom(tuple(cells[i] & state.dom[i] for i in range(state.csp.n)))

    def certificate(self):
        return Certificate(True, "sound-by-construction",
                           f"level-{self.k - 1} factor consistency; solves bounded-width, not unbounded XOR systems")


class ExactDedP(Reduction):
    """The exact best per-cell transformer dedP (clair.csp.exact_dedP): alpha(gamma(a) cap solutions).
    The strongest SOUND per-cell narrowing and the spine's VERIFIER. Exponential in the worst case
    (backtracking with witness early-stop)."""
    name = "exact_dedP"
    domain = "finite CSP (exact per-cell transformer / verifier)"
    verifier_only = True             # the ORACLE: used to GATE/verify, never a deployed runtime reduction

    def applies(self, state):
        return isinstance(state, CSPState)

    def reduce(self, state: CSPState) -> CSPState:
        return state.with_dom(C.exact_dedP(state.csp, state.dom))

    def certificate(self):
        return Certificate(True, "sound-by-construction",
                           "exact at the per-cell level (the dedP target); exponential worst-case")


class Modular(Reduction):
    """The certified congruence organ (clair.modular). When the state carries a LinSystem, solve it
    EXACTLY via integer Smith Normal Form (sound AND complete over Z_m, incl composite rings); else
    fall back to the local certified residue-AC on the modular CSP. Closes the affine-mod wall."""
    name = "modular_snf"
    domain = "linear systems over Z_m (congruence / parity)"

    def applies(self, state):
        return isinstance(state, CSPState) and isinstance(state.system, MOD.LinSystem)

    def reduce(self, state: CSPState) -> CSPState:
        sysm: MOD.LinSystem = state.system
        res = MOD.solve_mod(sysm)                       # exact per-variable reachable residue sets
        residues = res["residues"]
        # meet the exact residues into the current per-cell domain (sound: residues == exact dedP)
        return state.with_dom(tuple(state.dom[i] & residues[i] for i in range(state.csp.n)))

    def certificate(self):
        return Certificate(True, "sound-by-construction",
                           "complete for linear systems over Z_m (SNF, exact); cross-checked vs brute")


class GF2RowSpace(Reduction):
    """The certified GF(2) linear-algebra organ (clair.xor_wall.gf2_forced): Gaussian elimination
    over GF(2) gives the EXACT affine dedP for an XOR system Ax=b — the unbounded-width affine wall
    the per-cell/factor lattices abstain on. Sound + complete for GF(2)-affine. Needs ('gf2', A, b)."""
    name = "gf2_rowspace"
    domain = "XOR / GF(2) affine systems (row-space)"

    def applies(self, state):
        return (isinstance(state, CSPState) and isinstance(state.system, tuple)
                and len(state.system) == 3 and state.system[0] == "gf2")

    def reduce(self, state: CSPState) -> CSPState:
        _, A, b = state.system
        forced, _rank, _ok = XW.gf2_forced(A, b, state.csp.n)   # exact per-variable forced/free sets
        return state.with_dom(tuple(state.dom[i] & forced[i] for i in range(state.csp.n)))

    def certificate(self):
        return Certificate(True, "sound-by-construction",
                           "complete for GF(2)-affine systems (RREF row-space); beats the affine wall")


class Macro(Reduction):
    """The certified macro-deduction organ (clair.macro_deduct): for PATH-structured CSPs it
    composes per-edge transition maps (transition-monoid product) to reach the exact dedP fixpoint
    in O(log L) depth instead of O(L). Sound-by-construction (asserted == sequential == exact)."""
    name = "macro_reach"
    domain = "path / chain CSPs (automata-shortcut reach-doubling)"

    def __init__(self, w: int = 2):
        self.w = w

    def applies(self, state):
        return isinstance(state, CSPState) and ("path" in state.tags)

    def reduce(self, state: CSPState) -> CSPState:
        w = state.system if isinstance(state.system, int) else self.w
        dom, _depth, _S = MAC.macro_solve(state.csp, w, use_doubling=True)
        return state.with_dom(tuple(dom[i] & state.dom[i] for i in range(state.csp.n)))

    def certificate(self):
        return Certificate(True, "sound-by-construction",
                           "exact dedP for path-structured CSPs in O(log L) depth (== sequential)")


# ============================================================ CERTIFIED, own state type
class Unification(Reduction):
    """The type/relation organ (clair.fol): least Herbrand model by forward chaining with ground
    unification. NOT a per-cell narrowing (it GROWS a closure of derivable atoms), so it has its own
    state type and does not reduced-product with the CSP organs. Sound + exact (the dedP DUAL)."""
    name = "unification_chain"
    domain = "Datalog / FOL entailment (forward-chaining closure)"
    state_type = "fol-closure"

    def applies(self, state):
        return isinstance(state, tuple) and len(state) in (2, 3)  # (facts, rules[, query])

    def reduce(self, state):
        if len(state) == 3:
            facts, rules, query = state
            closure = FOL.forward_chain(facts, rules)        # monotone GROW to the least model
            return (tuple(closure), rules, query)
        facts, rules = state
        closure = FOL.forward_chain(facts, rules)
        return (tuple(closure), rules)

    def certificate(self):
        return Certificate(True, "sound-by-construction",
                           "exact least Herbrand model (forward chaining); 3-way entail/contradict/unknown")

    # 3-way entailment label readout (NOT a per-cell survival matrix — the readout-bridge in
    # clair.organ.readout_bridge.unification_readout exposes the RICH label + derived-fact features to
    # the LM). survival() here returns the minimal [1,K] label one-hot so it plugs the uniform γ channel.
    LABELS = ("entail", "contradict", "unknown")

    def label(self, state) -> str:
        """The 3-way entailment label for a (facts, rules, query) state, via the exact closure."""
        if len(state) != 3:
            raise ValueError("Unification.label needs a (facts, rules, query) state with a query atom")
        facts, rules, query = state
        return FOL.label_query(FOL.forward_chain(facts, rules), query)

    def survival(self, state, K):
        if len(state) != 3:
            raise NotImplementedError(
                "Unification reads out as an ENTAILMENT LABEL — pass a (facts, rules, query) state, or use "
                "clair.organ.readout_bridge.unification_readout for the rich label+derived-fact readout")
        surv = np.zeros((1, K), dtype=np.float32)
        li = self.LABELS.index(self.label(state))
        if li < K:
            surv[0, li] = 1.0                                # entail/contradict/unknown one-hot
        return surv


# ============================================================ APPROXIMATE, own state type
class Energy(Reduction):
    """The optimisation organ (clair.energy_organ): relaxes a CSP to the product of simplices and
    descends a soft-constraint energy by deterministic-annealing mean-field. Handles WEIGHTED
    constraints / constraint optimisation (which narrowing+chaining cannot express). The descent is
    sound (monotone energy), but the decoded assignment is APPROXIMATE; brute_opt is the exact oracle."""
    name = "energy_optimise"
    domain = "constraint optimisation (weighted CSP, mean-field relaxation)"
    state_type = "simplex"

    def applies(self, state):
        return isinstance(state, C.CSP)

    def reduce(self, state: C.CSP, device="cpu", steps=140):
        from .. import energy_organ as EN
        dev = device if isinstance(device, str) else "cpu"
        facs = EN.factor_tensors(state, dev)
        x, _info = EN.relax_decode(facs, state.n, state.d, dev, steps=steps)
        return tuple(int(x[i].argmax()) for i in range(state.n))

    def certificate(self):
        return Certificate(False, "approximate",
                           "mean-field descent (sound energy, approx decode); exact via brute_opt oracle")

    def survival(self, state, K, steps=140):
        """Energy reads out as a (soft) ASSIGNMENT, not a binary candidate set: the per-cell mean-field
        marginals x*[i] (argmax = the answer, spread = confidence). This is a genuine per-cell readout
        (soft values in [0,1], NOT {0,1}); the RICH readout (assignment + its energy + a confidence
        scalar) is clair.organ.readout_bridge.energy_readout."""
        from .. import energy_organ as EN
        facs = EN.factor_tensors(state, "cpu")
        x, _info = EN.relax_decode(facs, state.n, state.d, "cpu", steps=steps)
        xn = x.cpu().numpy()
        surv = np.zeros((state.n, K), dtype=np.float32)
        kd = min(state.d, K)
        surv[:, :kd] = xn[:, :kd]
        return surv


# ============================================================ NEURAL-GUIDANCE CSP-domain organs
def _featurize_budget(csps_doms, dev, N, D, M, A):
    """Self-contained featurization for a FactorGraphProposer / BladeFactorDeductor at an explicit
    (N,D,M,A) budget — adapted from clair.run_glados_staged.featurize but parameterized so loading
    organs of DIFFERENT budgets (core 12/8/80/3, blade 8/9/40/3) never mutate shared module globals."""
    import torch
    B = len(csps_doms)
    REL = D ** A
    var_mask = np.zeros((B, N, D), np.float32)
    given = np.zeros((B, N), np.float32)
    var_valid = np.zeros((B, N), np.float32)
    fac_rel = np.zeros((B, M, REL), np.float32)
    fac_arity = np.zeros((B, M, A), np.float32)
    fac_valid = np.zeros((B, M), np.float32)
    edge_var = np.full((B, M, A), N, np.int64)
    edge_valid = np.zeros((B, M, A), np.float32)
    for bi, (csp, dom) in enumerate(csps_doms):
        for i in range(csp.n):
            var_valid[bi, i] = 1.0
            for v in dom[i]:
                if v < D:
                    var_mask[bi, i, v] = 1.0
            if len(dom[i]) == 1:
                given[bi, i] = 1.0
        for fi, (sc, al) in enumerate(csp.cons):
            if fi >= M:
                break
            fac_valid[bi, fi] = 1.0
            fac_arity[bi, fi, len(sc) - 1] = 1.0
            t = np.zeros((D,) * A, np.float32)
            for tup in al:
                sl = [slice(None)] * A
                for p in range(len(sc)):
                    sl[p] = tup[p]
                t[tuple(sl)] = 1.0
            fac_rel[bi, fi] = t.reshape(-1)
            for p, cell in enumerate(sc):
                edge_var[bi, fi, p] = cell
                edge_valid[bi, fi, p] = 1.0
    t = lambda a: torch.as_tensor(a, device=dev)
    return dict(var_mask=t(var_mask), given=t(given), var_valid=t(var_valid), fac_rel=t(fac_rel),
                fac_arity=t(fac_arity), fac_valid=t(fac_valid), edge_var=t(edge_var),
                edge_valid=t(edge_valid))


def load_core_organ(path="runs/general_organ_full.pt", dev="cpu"):
    """Load the UNION-trained general narrow organ (the default deductor): a FactorGraphProposer
    trained by clair.run_glados_staged.train_organ over the 7-rung MIX (coloring/equality/ordering/
    arithmetic/alldiff/eqchain/forcedcolor). Checkpoint format {'state', 'meta':{N_MAX,...,d,R}}."""
    import torch
    from ..proposer import FactorGraphProposer
    blob = torch.load(path, map_location=dev, weights_only=False)
    m = blob["meta"]
    organ = FactorGraphProposer("full", m["N_MAX"], m["D_MAX"], m["M_MAX"], m["A_MAX"],
                                d=m["d"], R=m["R"], ds=min(4, m["R"])).to(dev)
    organ.load_state_dict(blob["state"])
    organ.eval()
    for p in organ.parameters():
        p.requires_grad_(False)
    return organ, m


def load_blade_organ(path="runs/modular_organ.pt", dev="cpu"):
    """Load the blade/grade-prior deductor (BladeFactorDeductor, msg='blade') — the affine/modular
    specialist trained by clair.run_modular. Checkpoint format {'state_dict', 'config':{msg,...,d,R,ds}}."""
    import torch
    from ..blade_deductor import BladeFactorDeductor
    blob = torch.load(path, map_location=dev, weights_only=False)
    c = blob["config"]
    organ = BladeFactorDeductor(c["msg"], c["n_max"], c["d_max"], c["m_max"], c["a_max"],
                                d=c["d"], R=c["R"], ds=c["ds"]).to(dev)
    organ.load_state_dict(blob["state_dict"])
    organ.eval()
    for p in organ.parameters():
        p.requires_grad_(False)
    return organ, c


class _NeuralOrgan(Reduction):
    """Shared base: a trained proposer run to a monotone-meet fixpoint to propose a per-cell
    narrowing. NOT sound by construction (OOD-soundness can break — the codex review's point) =>
    the COMPOSER must verifier-gate it. guidance() exposes the raw survival logits."""
    state_type = "csp-domain"

    def __init__(self, ckpt: str, dev: str = "cpu", theta: float = 0.5, R_max: int = 64):
        self._ckpt = ckpt
        self._dev = dev
        self._theta = theta
        self._R_max = R_max
        self._organ = None
        self._cfg = None

    def _budget(self):
        raise NotImplementedError

    def _load(self):
        raise NotImplementedError

    def _ensure(self):
        if self._organ is None:
            self._organ, self._cfg = self._load()
        return self._organ

    def _fits(self, state: CSPState) -> bool:
        N, D, M, A = self._budget()
        return state.csp.n <= N and state.csp.d <= D and len(state.csp.cons) <= M

    def applies(self, state):
        return isinstance(state, CSPState) and self._fits(state)

    def reduce(self, state: CSPState) -> CSPState:
        import torch
        organ = self._ensure()
        N, D, M, A = self._budget()
        vm_dom = state.dom
        with torch.no_grad():
            feat = _featurize_budget([(state.csp, vm_dom)], self._dev, N, D, M, A)
            vm = feat["var_mask"].clone()
            for _ in range(self._R_max):
                # #3: recompute `given` each pass from the CURRENT lattice so inference matches the
                # re-featurized training distribution (newly-singleton cells are marked given).
                feat["given"] = (vm.sum(-1) == 1).float() * feat["var_valid"]
                b, cls, _ = organ(vm, feat["given"], feat["fac_rel"], feat["fac_arity"],
                                  feat["edge_var"], feat["edge_valid"], feat["var_valid"],
                                  feat["fac_valid"])
                new_vm = vm * (torch.sigmoid(b) >= self._theta).float()
                if bool((new_vm == vm).all()):
                    break
                vm = new_vm
            nv = vm[0].cpu().numpy()
        dom = tuple(frozenset(v for v in range(state.csp.d) if nv[i, v] > 0.5) & state.dom[i]
                    for i in range(state.csp.n))
        return state.with_dom(dom)

    def guidance(self, state: CSPState):
        import torch
        organ = self._ensure()
        N, D, M, A = self._budget()
        with torch.no_grad():
            feat = _featurize_budget([(state.csp, state.dom)], self._dev, N, D, M, A)
            b, cls, _ = organ(feat["var_mask"], feat["given"], feat["fac_rel"], feat["fac_arity"],
                              feat["edge_var"], feat["edge_valid"], feat["var_valid"], feat["fac_valid"])
        return {"survival_logits": b[0].cpu().numpy(), "conflict_logit": float(cls[0])}


class CoreNarrowOrgan(_NeuralOrgan):
    """THE CORE narrow organ: the union-trained general deductor (runs/general_organ_full.pt),
    default for every reasoning rung. Neural-guidance => verifier-gated by the composer."""
    name = "core_narrow_organ"
    domain = "general narrowing (union-trained over 7 rungs)"

    def __init__(self, ckpt="runs/general_organ_full.pt", dev="cpu", **kw):
        super().__init__(ckpt, dev, **kw)

    def _budget(self):
        return (12, 8, 80, 3)        # the staged budget general_organ_full.pt was trained at

    def _load(self):
        return load_core_organ(self._ckpt, self._dev)

    def certificate(self):
        return Certificate(False, "neural-guidance",
                           "proposes; sound only under the verifier gate (OOD-soundness can break). "
                           "Strong recall on coloring/equality/chains; honest gaps on affine/ordering")


class BladeAffineOrgan(_NeuralOrgan):
    """The blade/grade-PRIOR deductor for affine (runs/modular_organ.pt, BladeFactorDeductor): the
    grade-k wedge message gives the affine/modular inductive bias. Neural-guidance => gated."""
    name = "blade_affine_organ"
    domain = "affine / modular narrowing (grade-prior blade deductor)"

    def __init__(self, ckpt="runs/modular_organ.pt", dev="cpu", **kw):
        super().__init__(ckpt, dev, **kw)

    def _budget(self):
        return (8, 9, 40, 3)         # modular_organ.pt config budget

    def _load(self):
        return load_blade_organ(self._ckpt, self._dev)

    def certificate(self):
        return Certificate(False, "neural-guidance",
                           "blade/grade-prior proposer for affine/modular; gated by the certified Modular/GF2 ops")


# ============================================================ the registry
def build_bank(load_neural: bool = True, dev: str = "cpu",
               core_ckpt: str = "runs/general_organ_full.pt",
               blade_ckpt: str = "runs/modular_organ.pt") -> dict:
    """Return {name: Reduction}. Pure-certified entries always present (no torch / no checkpoints).
    Neural entries (lazy-loading) included iff load_neural and the checkpoints exist."""
    import os
    bank: dict[str, Reduction] = {}
    for r in (ArcConsistency(), FactorConsistency(k=3), ExactDedP(), Modular(), GF2RowSpace(),
              Macro(), Unification(), Energy()):
        bank[r.name] = r
    if load_neural:
        if os.path.exists(core_ckpt):
            bank["core_narrow_organ"] = CoreNarrowOrgan(core_ckpt, dev)
        if os.path.exists(blade_ckpt):
            bank["blade_affine_organ"] = BladeAffineOrgan(blade_ckpt, dev)
    return bank


def certified_csp_reductions(bank: dict) -> list:
    """The DEPLOYABLE sound-by-construction CSP-domain reductions (the reduced-product portfolio).
    EXCLUDES verifier-only oracles (ExactDedP): the exact dedP is the GATE/verifier the composer
    trusts to certify the others, not a runtime reduction we ship — deployment must not call it."""
    return [r for r in bank.values()
            if r.state_type == "csp-domain" and r.certificate().sound and not r.verifier_only]
