"""clair/organ/selftest.py — the CONSOLIDATED organ smoke (the assembly's single proof).

One run that exercises the whole package:
  [1] each registered CERTIFIED reduction is SOUND on its domain (false-elim 0 vs the exact verifier),
      and the specialized ops match their own exact authority (Modular vs brute, GF2 vs gf2_forced,
      Macro vs dedP); the standalone certified/approx beasts (Unification, Energy) pass their checks;
  [2] the COMPOSER chains across >= 2 domains SOUNDLY (modular + GF(2)), and the reduced product
      narrows where any single local op abstains;
  [3] the structured-gamma READOUT is a bitwise NO-OP at init (the zero-init gate);
  [4] the WOVEN entry point runs: the organ-side training entry trains+saves a tiny union organ, it
      loads back as the CoreNarrowOrgan and is verifier-gated sound by the composer; (with --woven the
      full OLMo staged smoke is run too).

  python -m clair.organ.selftest            # core smoke (no OLMo)
  python -m clair.organ.selftest --woven    # + the full staged woven smoke (loads OLMo-2-1B)
"""
from __future__ import annotations

import argparse

import numpy as np

from .. import csp as C
from .. import modular as MOD
from .. import xor_wall as XW
from .. import macro_deduct as MAC
from .. import fol as FOL
from .. import hard_tasks as HT
from . import bank as B
from .protocol import CSPState, exact_oracle, false_elim
from .compose import reduced_product


def _sound(state_after: CSPState, state_full: CSPState) -> int:
    return false_elim(state_after, exact_oracle(state_full))


# ============================================================ [1] per-reduction soundness
def test_certified_reductions():
    print("\n[1] each CERTIFIED reduction sound on its domain")
    rng = np.random.default_rng(0)

    # ArcConsistency on a coloring instance (it solves nothing unsound)
    col = C.coloring(4, [(0, 1), (1, 2), (2, 3), (0, 2)], k=3)
    s0 = CSPState.full(col)
    out = B.ArcConsistency().reduce(s0)
    assert out.issub(s0) and _sound(out, s0) == 0
    print("    arc_consistency      sound (false-elim 0) on coloring")

    # FactorConsistency solves the affine xor_parity at level-2 where AC abstains
    xr = C.xor_parity()
    sx = CSPState.full(xr)
    ac = B.ArcConsistency().reduce(sx)
    fac = B.FactorConsistency(k=3).reduce(sx)
    assert _sound(fac, sx) == 0 and _sound(ac, sx) == 0
    assert fac.alive() < ac.alive(), "factor consistency should out-narrow AC on affine parity"
    print(f"    factor_consistency   sound; narrows xor_parity to alive={fac.alive()} (AC stalls at {ac.alive()})")

    # ExactDedP == the oracle by definition
    assert B.ExactDedP().reduce(sx).dom == exact_oracle(sx).dom
    print("    exact_dedP           == exact verifier (the spine ground truth)")

    # Modular: certified SNF solver vs brute, on the modular parity wall
    bad = 0
    for _ in range(40):
        m = int(rng.choice([3, 5, 6, 7])); n = int(rng.integers(3, 6))
        sysm = MOD.gen_cyclic(rng, m, n)
        cspm = MOD.to_csp(sysm)
        st = CSPState.full(cspm, system=sysm, tags={"modular"})
        out = B.Modular().reduce(st)
        bres, _ = MOD.brute_solve(sysm)
        bad += int(tuple(out.dom) != bres)
    assert bad == 0, f"Modular mismatched brute on {bad}/40 systems"
    print("    modular_snf          == brute residues on 40 modular-parity systems (sound+complete)")

    # GF2: row-space exact affine dedP vs gf2_forced, where AC abstains
    bad = 0; beats = 0
    for _ in range(30):
        n = int(rng.integers(6, 12)); band = int(rng.choice([3, 4, 6]))
        d = XW.gen_xor_system(rng, n, max(2, n // 2), band)
        cspx = d["csp"]
        st = CSPState.full(cspx, system=("gf2", d["A"], d["b"]), tags={"xor"})
        out = B.GF2RowSpace().reduce(st)
        forced = XW.gf2_forced(d["A"], d["b"], n)[0]
        bad += int(tuple(out.dom) != tuple(forced))
        ac = B.ArcConsistency().reduce(st)
        beats += int(out.alive() < ac.alive())
    assert bad == 0, f"GF2 mismatched gf2_forced on {bad}/30 systems"
    print(f"    gf2_rowspace         == gf2_forced on 30 XOR systems; out-narrows AC on {beats}/30")

    # Macro: reach-doubling path solver == exact dedP
    bad = 0
    for L in (3, 4, 5):
        cspc, w = MAC.gen_eqchain(L, k=3, rng=np.random.default_rng(L), determined=True)
        st = CSPState.full(cspc, system=w, tags={"path"})
        out = B.Macro().reduce(st)
        bad += int(tuple(out.dom) != tuple(HT.fast_dedP(cspc)))
    assert bad == 0
    print("    macro_reach          == exact dedP on eqchain L in {3,4,5} (O(log L) depth)")

    # Unification (own state): least Herbrand model exact
    assert FOL.self_check(verbose=False) is True
    print("    unification_chain    == brute least Herbrand model (fol.self_check)")

    # Energy (own state, approximate decode but solves the determined chain)
    from .. import energy_organ as EN
    ch = C.chain_eq(4)
    assign = B.Energy().reduce(ch, device="cpu", steps=140)
    assert EN.feasible(ch, assign) and assign == C.solutions(ch)[0]
    print("    energy_optimise      solves determined chain_eq(4) (mean-field; brute_opt = exact oracle)")


# ============================================================ [2] composer across >= 2 domains
def test_composer():
    print("\n[2] composer (verifier-gated reduced product) chains across >= 2 domains soundly")
    bank = B.build_bank(load_neural=False)
    certified = B.certified_csp_reductions(bank)
    rng = np.random.default_rng(7)

    # domain A: modular parity wall
    okA = 0
    for _ in range(20):
        m = int(rng.choice([3, 5, 7])); n = int(rng.integers(3, 6))
        sysm = MOD.gen_cyclic(rng, m, n)
        st = CSPState.full(MOD.to_csp(sysm), system=sysm, tags={"modular"})
        out, tr = reduced_product(st, certified, verify=True)
        bres, _ = MOD.brute_solve(sysm)
        okA += int(tr.sound and tuple(out.dom) == bres)
    assert okA == 20
    print(f"    modular domain: {okA}/20 composed runs sound AND == brute residues")

    # domain B: GF(2) affine wall
    okB = 0; solved = 0
    for _ in range(20):
        n = int(rng.integers(6, 11)); band = int(rng.choice([3, 4, 6]))
        d = XW.gen_xor_system(rng, n, max(2, n // 2), band)
        st = CSPState.full(d["csp"], system=("gf2", d["A"], d["b"]), tags={"xor"})
        out, tr = reduced_product(st, certified, verify=True)
        okB += int(tr.sound and tuple(out.dom) == tuple(XW.gf2_forced(d["A"], d["b"], n)[0]))
        solved += int(out.status() == "solved")
    assert okB == 20
    print(f"    GF(2) domain:   {okB}/20 composed runs sound AND == gf2_forced ({solved}/20 fully solved)")

    # the reduced product narrows where local AC alone abstains (the point of composing): the
    # modular parity wall is designed so local residue-AC proves nothing, yet the certified SNF solves
    sysm = MOD.gen_cyclic(np.random.default_rng(2), 5, 4)
    st = CSPState.full(MOD.to_csp(sysm), system=sysm, tags={"modular"})
    ac_only = B.ArcConsistency().reduce(st)
    comp, _ = reduced_product(st, certified, verify=True)
    assert comp.alive() < ac_only.alive(), (ac_only.alive(), comp.alive())
    print(f"    composing narrows alive {ac_only.alive()} (AC abstains) -> {comp.alive()} "
          f"(AC+factor+modular), soundly")

    # BATCHED reduced product == serial per-instance, BITWISE (the composer-cost fix: the certified floor
    # — incl. the Rust-ported factor consistency — fans across the batch dim; identity is a hard gate).
    from .compose import reduced_product_batch
    from .. import curriculum as CU
    rng2 = np.random.default_rng(11)
    batch_states = []
    for _ in range(24):
        p = CU.gen_problem(rng2)
        batch_states.append(CSPState.full(CU.build_csp(p.n, p.d, p.facts)))
    ser = [reduced_product(s, certified, verify=False)[0] for s in batch_states]
    bat, _ = reduced_product_batch(batch_states, certified)
    assert all(ser[i].dom == bat[i].dom for i in range(len(ser))), "batched reduced product != serial!"
    print(f"    batched reduced product == serial per-instance: {len(ser)}/{len(ser)} bitwise-identical")


# ============================================================ [3] readout no-op at init
def test_readout_noop():
    print("\n[3] structured-gamma readout is a bitwise no-op at init")
    import torch
    from .. import oracle_readout as O
    D, K, n = 64, 8, 4
    gamma = O.OracleGamma(D, K).float()
    col = C.coloring(n, [(0, 1), (1, 2)], k=3)
    st = CSPState.full(col)
    surv = torch.from_numpy(B.ArcConsistency().survival(st, K)).unsqueeze(0)          # [1,n,K]
    mention = torch.zeros(1, n, 5); mention[:, :, 1] = 1.0
    delta = gamma.delta(surv, mention)
    injected = torch.tanh(gamma.alpha) * delta                                         # the hook math
    assert float(injected.abs().max()) == 0.0, "zero-init gate must make the readout a bitwise no-op"
    with torch.no_grad():
        gamma.alpha.fill_(2.0)
    live = float((torch.tanh(gamma.alpha) * gamma.delta(surv, mention)).abs().max())
    assert live > 0.0, "with the gate open the readout must move the residual stream"
    print(f"    no-op at init (max|delta|=0.0); gate-open moves residual ({live:.3e})  [readout signature]")


# ============================================================ [4] woven entry point runs
def test_woven(full=False):
    print("\n[4] woven entry point runs")
    import os, tempfile, torch
    from . import train as T
    from .. import run_glados_staged as G
    assert callable(G.main), "the staged woven recipe entry must be importable"

    # organ-side training entry: train + save a tiny union organ, load it back, gate it sound
    dev = G.device()
    tmp = os.path.join(tempfile.gettempdir(), "organ_selftest_core.pt")
    organ, meta = T.train_organ(out=tmp, steps=30, target=1.2e5, pool=48, R=6, seed=0, dev=dev)
    core = B.CoreNarrowOrgan(ckpt=tmp, dev=dev)
    # exercise on an in-budget coloring instance, verifier-gated by the composer
    col = C.coloring(5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)], k=3)
    st = CSPState.full(col)
    raw = core.reduce(st)
    assert raw.issub(st), "the loaded core organ must produce a narrowing of the input"
    # the BATCHED neural reduce (the composer-cost fix) is BITWISE-identical to per-instance reduce
    col2 = C.coloring(5, [(0, 1), (1, 3), (2, 4)], k=3)
    sts = [st, CSPState.full(col2), CSPState.full(C.chain_eq(4))]
    rb = core.reduce_batch(sts)
    assert all(rb[i].dom == core.reduce(sts[i]).dom for i in range(len(sts))), \
        "core.reduce_batch must be bitwise-identical to per-instance core.reduce"
    out, tr = reduced_product(st, B.certified_csp_reductions(B.build_bank(load_neural=False)) + [core],
                              verify=True)
    assert tr.sound
    g = core.guidance(st)
    assert "survival_logits" in g
    print(f"    organ trained+saved ({tmp}), reloaded as core_narrow_organ, verifier-gated sound; "
          f"guidance logits shape {g['survival_logits'].shape}")
    os.remove(tmp)

    if full:
        print("    running the FULL staged woven smoke (OLMo-2-1B)...", flush=True)
        T.weave("allenai/OLMo-2-0425-1B", regime="small", smoke=True)
        print("    full staged woven smoke completed")
    else:
        print("    (skip full OLMo staged smoke; pass --woven to run it)")


# ============================================================ [5] readout-bridges (energy / FOL)
def test_readout_bridge():
    print("\n[5] readout-bridges for the non-CSP-lattice beasts (energy/ising assignment + FOL label)")
    import torch
    import numpy as np
    from . import readout_bridge as RB
    from .. import oracle_readout as O
    K = 4

    # Energy.survival = soft marginals that SOLVE the determined chain; energy_readout = rich [n,K+2]
    ch = C.chain_eq(4)
    sv = B.Energy().survival(ch, K)
    assert tuple(int(sv[i].argmax()) for i in range(ch.n)) == C.solutions(ch)[0], "energy survival must solve"
    feat, assign, info = RB.energy_readout(ch, K, steps=140)
    assert feat.shape == (ch.n, K + 2) and tuple(assign) == C.solutions(ch)[0]
    print(f"    energy_optimise   survival solves chain_eq(4); rich feat {feat.shape} "
          f"(assignment + conf {feat[0,K]:.2f} + quality {feat[0,K+1]:.2f})")

    # Ising readout: a max-cut maps -> Ising -> organ -> rich spin readout that recovers the optimum
    try:
        import networkx as nx
        from .. import ising_organ as IS
        G = nx.gnp_random_graph(10, 0.5, seed=2)
        J, h, _, meta = IS.encode_maxcut(G, device="cpu")
        ifeat, spins, E = RB.ising_readout(J, h, K=2, restarts=64, steps=160, seed=3)
        cut = IS.decode_maxcut(G, meta, torch.as_tensor(spins, dtype=torch.float32))
        opt = IS.exact_maxcut(G, device="cpu")
        assert ifeat.shape == (G.number_of_nodes(), 2 + 2) and abs(cut - opt) < 1e-6
        print(f"    ising readout     max-cut n=10 organ cut={cut:.0f} == exact {opt:.0f}; rich feat {ifeat.shape}")
    except ImportError:
        print("    ising readout     (networkx unavailable; skipped)")

    # α→couplings compile head + DIRECT coupling-supervision (the annealer is non-diff): loss must drop
    from .. import ising_organ as IS
    a_part = [3, 1, 4, 1, 5, 9, 2, 6]
    Jt, ht, _, _ = IS.encode_partition(a_part, device="cpu")
    n, D = len(a_part), 32
    torch.manual_seed(0)
    v = torch.randn(1, n, D)
    proj = RB.CouplingProjector(D, dp=64)
    opt = torch.optim.Adam(proj.parameters(), lr=1e-2)
    Jtt = torch.as_tensor(Jt).unsqueeze(0); htt = torch.as_tensor(ht).unsqueeze(0)
    l0 = None
    for step in range(120):
        Jp, hp = proj(v)
        loss = RB.coupling_loss(Jp, hp, Jtt, htt)
        if l0 is None:
            l0 = float(loss)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss) < 0.5 * l0, f"coupling supervision did not reduce loss ({l0:.3f} -> {float(loss):.3f})"
    print(f"    α→couplings       CouplingProjector fits (J,h) under direct supervision: "
          f"loss {l0:.3f} -> {float(loss):.3f}")

    # Unification (FOL): rich label readout matches the exact closure label, and survival one-hots it
    from .. import fol as FOL
    rng = np.random.default_rng(1)
    ok = 0
    for lab in ("entail", "contradict", "unknown"):
        p = FOL.gen_problem(rng, label=lab)
        ufeat, label, closure = RB.unification_readout(p.facts, p.rules, p.query, K)
        usv = B.Unification().survival((p.facts, p.rules, p.query), K)
        ok += int(label == lab and int(ufeat[0, ("entail", "contradict", "unknown").index(lab)]) == 1
                  and int(usv[0].argmax()) == ("entail", "contradict", "unknown").index(lab))
    assert ok == 3, "unification readout/label must match the exact closure for all 3 labels"
    print(f"    unification_chain rich label readout matches exact closure on entail/contradict/unknown (3/3)")

    # the rich feature scatters through the SAME zero-init γ channel (bitwise no-op at init)
    Fin = K + 2
    gamma = O.OracleGamma(32, Fin).float()
    ft = torch.from_numpy(feat).unsqueeze(0)
    mention = torch.zeros(1, ch.n, 5); mention[:, :, 1] = 1.0
    inj0 = float((torch.tanh(gamma.alpha) * gamma.delta(ft, mention)).abs().max())
    with torch.no_grad():
        gamma.alpha.fill_(2.0)
    inj1 = float((torch.tanh(gamma.alpha) * gamma.delta(ft, mention)).abs().max())
    assert inj0 == 0.0 and inj1 > 0.0, "rich readout must be a γ no-op at init and move when opened"
    print(f"    γ-bridge          rich readout no-op@init (0.0); gate-open moves residual ({inj1:.3e})")


# ============================================================ [6] organ-as-process-reward
def test_process_reward():
    print("\n[6] organ-as-process-reward (dense where outcome is 0)")
    from . import process_reward as PR
    PR.smoke()


# ============================================================ [7] MULTI-FACULTY dispatch (de-CSP-lock)
def test_multifaculty():
    print("\n[7] MULTI-FACULTY: general typed struct_α + composer DISPATCH (non-CSP faculty wired)")
    import torch
    import numpy as np
    from . import faculty as FAC
    from . import faculty_tasks as FT
    from .faculty import (GraphReach, GraphReachState, CrossCSPGraph, build_struct_alpha,
                          reachable, LIVE_FACULTIES)
    from .multifaculty import MultiFacultyComposerOrgan
    from .protocol import CSPState
    K = 8
    rng = np.random.default_rng(0)

    # (a) GraphReach certified closure == brute BFS; survival is the per-cell reach one-hot
    bad = 0
    for _ in range(40):
        n = int(rng.integers(4, 8))
        edges = frozenset((i, j) for i in range(n) for j in range(n) if i != j and rng.random() < 0.3)
        s = int(rng.integers(n))
        out = GraphReach().reduce(GraphReachState(n, edges, s))
        truth = reachable(n, edges, s)
        bad += int(tuple(out.reach) != tuple(i in truth for i in range(n)))
    assert bad == 0, f"GraphReach mismatched brute reachability on {bad}/40"
    print("    graph_reach          == brute BFS reachability on 40 graphs (sound+complete closure)")

    # (b) the typed builder dispatches by faculty → the right state_type (de-CSP-lock)
    ts_c = build_struct_alpha("csp", [("pin", 0, 1), ("neq", 0, 1)], 3, 3)
    ts_g = build_struct_alpha("graph", ([(0, 1), (1, 2)], 0, 2), 3, 2)
    ts_x = build_struct_alpha("cross", ([("pin", 0, 0), ("neq", 0, 1)], [(0, 1)], 0, 1), 2, 2)
    assert (ts_c.state_type, ts_g.state_type, ts_x.state_type) == \
        ("csp-domain", "graph-reach", "cross-csp-graph"), "struct_α must carry its faculty state_type"
    print(f"    build_struct_alpha   dispatches csp/graph/cross → "
          f"{ts_c.state_type} / {ts_g.state_type} / {ts_x.state_type}")

    # (c) CrossCSPGraph (the faculty-A→flow→faculty-B composite) == exact ground truth
    bad = 0
    for _ in range(30):
        r = FT.gen_cross_record(rng, want_yes=(rng.random() < 0.5))
        ts = build_struct_alpha("cross", (r["true_facts"], r["cand_edges"], r["source"], r["target"]),
                                r["n"], 2)
        out = CrossCSPGraph().reduce(ts.state)
        bad += int(bool(out.reach[r["target"]]) != (r["gold_idx"] == 0))
    assert bad == 0, f"CrossCSPGraph mismatched exact ground truth on {bad}/30"
    print("    cross_csp_graph      == exact (CSP colours → active edges → reachability) on 30 instances")

    # (d) the DISPATCHING composer routes a NON-CSP (graph) problem to its faculty (no LM): the de-CSP-lock
    comp = MultiFacultyComposerOrgan(dev="cpu", use_core=False)
    rg = FT.gen_graph_record(rng)
    n = rg["n"]
    edge_logits = torch.full((1, n, n), -9.0)
    for (u, v) in rg["true_edges"]:
        edge_logits[0, u, v] = 9.0                          # α "emits" the true edges
    b0 = torch.zeros(1, n, K); vmask = torch.ones(1, n)
    specs = [{"faculty": "graph", "facts": [], "n": n, "d": 2,
              "source": rg["source"], "target": rg["target"]}]
    surv, disp = comp.dispatch(b0, vmask, 0.5, edge_logits, specs, K)
    assert disp[0] == "graph", "a graph problem must dispatch to the graph faculty (NOT csp-locked)"
    truth = reachable(n, frozenset(map(tuple, rg["true_edges"])), rg["source"])
    assert int(surv[0, rg["target"], 0]) == int(rg["target"] in truth), "graph readout must encode reach"
    print(f"    composer.dispatch    routed a graph problem → graph faculty; reach readout correct "
          f"(faculties={comp.faculties()})")

    # (e) the multifaculty NECESSITY proof: cross solved, single-faculty baselines fail
    res = FT.prove_multifaculty(n_inst=120, seed=1)
    assert res["graph_faculty_acc"] > 0.99
    assert res["cross_multifaculty_acc"] > 0.99
    assert res["cross_graphonly_acc"] < 0.95 and res["cross_csponly_majority_acc"] < 0.7
    print(f"    multifaculty win     cross {res['cross_multifaculty_acc']*100:.0f}% vs "
          f"graph-only {res['cross_graphonly_acc']*100:.0f}% / csp-only "
          f"{res['cross_csponly_majority_acc']*100:.0f}% (routing NECESSARY)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--woven", action="store_true", help="also run the full OLMo staged woven smoke")
    a = ap.parse_args()
    print("================ GLaDOS ORGAN — consolidated self-test ================")
    bank = B.build_bank(load_neural=True)
    print(f"registry: {len(bank)} reductions -> {sorted(bank)}")
    test_certified_reductions()
    test_composer()
    test_readout_noop()
    test_readout_bridge()
    test_process_reward()
    test_multifaculty()
    test_woven(full=a.woven)
    print("\nALL CHECKS PASS — clair/organ is the assembled GLaDOS organ: certified bank sound on "
          "domain, composer sound across >=2 domains, readout no-op at init, MULTI-FACULTY dispatch "
          "(graph faculty + cross-faculty composition) wired, woven entry runs.")


if __name__ == "__main__":
    main()
