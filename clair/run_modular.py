"""clair/run_modular.py — NEURAL GUIDANCE for the certified congruence organ + the recall/soundness table.

clair.modular is the CERTIFIED authority (SNF modular solver + CRT + local residue-AC), validated exact
vs brute. THIS file trains the neural part: a factor-graph narrower (blade message, the affine/arity-3
inductive bias) that learns to PROPOSE which residues to eliminate — the learned "contractor choice" —
while the certified discipline keeps it SOUND:

  * objective = DOMINATE the exact modular dedP (clair.csp.exact_dedP on the Z_m encoding == solve_mod,
    proven equal in modular.py smoke step 5). Never eliminate a residue some solution uses => 0 false-elim.
  * state = per-variable residue mask over Z_m (== the per-cell lattice; composable reduced product).
  * the local certified residue-AC (clair.modular.ac_fixpoint / clair.csp.ac_step) is the SOUND baseline
    that ABSTAINS on coupled affine-mod systems; the complete certified solve_mod is the 100% reference.

The honest research variable is COMPLETENESS: across moduli and system sizes, does the learned narrower
recover the modular deduction the LOCAL certified operator abstains on (closing part of the affine-mod
wall, soundly), and where does it — like the XOR/affine wall (notes/codex_zoo_review.md) — still fall
short of the certified Gaussian completeness (which is exactly why the certified op is the necessary
soundness/completeness floor, with the neural part as amortised guidance)?

  python -m clair.run_modular --smoke
  python -m clair.run_modular --train --steps 1500 --R 12 --target 4e5 --ckpt runs/modular_organ.pt
"""
from __future__ import annotations

import argparse, json, os, time
import numpy as np
import torch

from . import csp as C
from . import run_general as RG

# ---- budget for the modular sweep (override RG globals so its featurize/targets/eval use it) ----
# D_MAX=9 covers moduli 2..9: primes 2,3,5,7 + composites 4,6,8,9 (prime-powers 4,8,9 + 6=2·3 for the
# ring/CRT story). Bigger m balloons the extensional relation tables (m^2 tuples/factor) the host-side
# featurizer must build every step, so 9 keeps training tractable while still sweeping moduli.
N_MAX, D_MAX, M_MAX, A_MAX = 8, 9, 40, 3
RG.N_MAX, RG.D_MAX, RG.M_MAX, RG.A_MAX = N_MAX, D_MAX, M_MAX, A_MAX
RG.ENUM_CAP = 10 ** 18                                    # drop the d**n gate; we cap via solve_mod n_sol

from .blade_deductor import BladeFactorDeductor, size_for   # noqa: E402
from . import modular as MOD                                # noqa: E402

MODULI = [2, 3, 4, 5, 6, 7, 8, 9]                          # primes + composites (rings) incl the GF(2) wall
SOL_CAP = 36                                               # reject systems with > this many solutions (cheap dedP)
TRAIN_N = (4, 6)                                           # training system sizes (on-policy dedP stays cheap)


# ============================================================ generators (witness-first; curated by solve_mod)
def _fits(csp):
    return (csp.n <= N_MAX and csp.d <= D_MAX and len(csp.cons) <= M_MAX
            and all(len(sc) <= A_MAX for sc, _ in csp.cons))


def gen_lineq(rng, m=None, n=None):
    """Sparse linear modular system (the core affine-mod rung). Curated to be COUPLED but low-solution
    (so the dedP target is cheap and the affine-mod wall is sharp: unique/near-unique answer that local
    residue-AC abstains on)."""
    m = m or int(rng.choice(MODULI))
    n = n or int(rng.integers(TRAIN_N[0], TRAIN_N[1] + 1))
    for _ in range(40):
        sysm = MOD.gen_lineq(rng, m, n, n_eq=int(rng.integers(n, n + 4)),
                             n_pin=int(rng.integers(1, max(2, n // 2 + 1))))
        cert = MOD.solve_mod(sysm)
        if cert["status"] != "unsat" and cert["n_solutions"] <= SOL_CAP:
            csp = MOD.to_csp(sysm)
            if _fits(csp):
                return csp
    return MOD.to_csp(MOD.gen_lineq(rng, 3, 4, n_eq=5, n_pin=2))


def gen_modsum(rng, m=None, n=None):
    """Ternary modular sums x_i + x_j ≡ x_k (mod m) + pins — the gentler affine rung."""
    m = m or int(rng.choice(MODULI))
    n = n or int(rng.integers(TRAIN_N[0], TRAIN_N[1] + 1))
    for _ in range(40):
        sysm = MOD.gen_modsum(rng, m, n)
        cert = MOD.solve_mod(sysm)
        if cert["status"] != "unsat" and cert["n_solutions"] <= SOL_CAP:
            csp = MOD.to_csp(sysm)
            if _fits(csp):
                return csp
    return MOD.to_csp(MOD.gen_modsum(rng, 3, 4))


def gen_cyclic(rng, m=None, n=None):
    """The modular parity WALL (clair.modular.gen_cyclic): equalities + ternary coupler, no pins, where
    local residue-AC narrows NOTHING but the system is determined — the rung the per-cell lattice
    abstains on. Curated low-solution so the dedP target is cheap and the answer is (near-)unique."""
    m = m or int(rng.choice(MODULI))
    n = n or int(rng.integers(TRAIN_N[0], TRAIN_N[1] + 1))
    for _ in range(40):
        sysm = MOD.gen_cyclic(rng, m, n)
        cert = MOD.solve_mod(sysm)
        if cert["status"] != "unsat" and cert["n_solutions"] <= SOL_CAP:
            csp = MOD.to_csp(sysm)
            if _fits(csp):
                return csp
    return MOD.to_csp(MOD.gen_cyclic(rng, 5, 4))


def gen_congruence(rng, m=None, n=None):
    """CRT rung: a few variables over domain L=lcm(small moduli), each pinned by congruences
    x ≡ r (mod m_i) for distinct small moduli + a couple of equality links. The certified CRT combine
    (clair.modular.crt_combine) is the authority; here it is the EASY breadth/soundness family (local AC
    already fuses the unary congruences). Witness-first."""
    small = [2, 3, 4, 5]
    rng.shuffle(small)
    k = int(rng.integers(2, 4))
    mods = small[:k]
    L = 1
    for mm in mods:
        L = L // MOD.gcd(L, mm) * mm
    L = min(L, D_MAX)
    n = n or int(rng.integers(2, 5))
    x = [int(rng.integers(0, L)) for _ in range(n)]
    cons = []
    for v in range(n):
        for mm in rng.choice(mods, size=int(rng.integers(1, len(mods) + 1)), replace=False):
            r = x[v] % int(mm)
            cons.append(C._rel((v,), (lambda mm, r: lambda t: t[0] % mm == r)(int(mm), r), L))
    for _ in range(int(rng.integers(0, n))):                  # a few equality links
        i, j = (int(t) for t in rng.choice(n, size=2, replace=False)) if n >= 2 else (0, 0)
        if x[i] == x[j]:
            cons.append(C._rel((i, j), lambda t: t[0] == t[1], L))
    csp = C.CSP(n, L, tuple(cons))
    return csp if _fits(csp) and len(C.solutions(csp, limit=1)) > 0 else gen_congruence(rng)


GEN = {"lineq": gen_lineq, "cyclic": gen_cyclic, "modsum": gen_modsum, "congruence": gen_congruence}
TRAIN_RUNGS = ["lineq", "cyclic", "modsum", "congruence"]


def sample(rng, rung, m=None):
    for _ in range(30):
        csp = GEN[rung](rng, m=m) if rung != "congruence" else GEN[rung](rng)
        if _fits(csp) and len(C.solutions(csp, limit=1)) > 0:
            return csp
    return MOD.to_csp(MOD.gen_lineq(rng, 3, 4, n_eq=5, n_pin=2))


def sample_corpus(rng, n, rungs):
    return [(rungs[i % len(rungs)], sample(rng, rungs[i % len(rungs)])) for i in range(n)]


# ============================================================ baselines (certified, exact)
def _removed(full, dom, n):
    return {(i, v) for i in range(n) for v in full[i] if v not in dom[i]}


def certified_baselines(csp):
    """Exact references for one CSP: (ac_recall, ac_false_elim, ac_solved, dedP_uniq, solver_solved).
    ac = local certified residue-AC (clair.csp.ac_step, == clair.modular.ac_fixpoint). solver = the
    complete certified solve (clair.csp.exact_dedP fixpoint == clair.modular.solve_mod)."""
    full = csp.full()
    ded, _ = C.to_fixpoint(C.exact_dedP, csp, full)
    ac, _ = C.to_fixpoint(C.ac_step, csp, full)
    rm_ded = _removed(full, ded, csp.n)
    rm_ac = _removed(full, ac, csp.n)
    ac_rec = len(rm_ac & rm_ded) / max(1, len(rm_ded))
    ac_fe = len(rm_ac - rm_ded)                              # AC is sound => should be 0
    return {"ac_recall": ac_rec, "ac_false_elim": ac_fe,
            "ac_solved": int(C.status(ac) == "solved"),
            "dedP_uniq": int(all(len(c) == 1 for c in ded)),
            "solver_solved": int(C.status(ded) == "solved")}


@torch.no_grad()
def neural_fixpoint(m, csps, dev, theta=0.5, R_max=64):
    """Roll the learned monotone meet to a fixed point WITHOUT recomputing the exact dedP every
    iteration (RG.run_to_fixpoint does, which is O(R_max) dedP enumerations — far too slow at d=12).
    Soundness is measured ONCE on the final state in evaluate(): the meet is monotone, so the final
    dom is the strongest the organ commits to, and false-elim = anything it dropped that the
    full-domain dedP keeps."""
    items = [(c, c.full()) for c in csps]
    feat = RG.featurize(items, dev)
    vm = feat["var_mask"].clone()
    B = len(csps)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    for _ in range(R_max):
        b, cls, _ = RG.fwd(m, feat, vm)
        new_vm = RG.meet(vm, b, theta)
        changed = (new_vm != vm).any(-1).any(-1)
        vm = torch.where(done.view(-1, 1, 1), vm, new_vm)
        done = done | ~changed
        if done.all():
            break
    return [RG.dom_from_mask(vm[i].cpu().numpy(), csps[i]) for i in range(B)]


@torch.no_grad()
def evaluate(m, csps, dev, theta=0.5, R_max=64):
    """Per-set: NEURAL narrowing-recall + soundness vs exact modular dedP, alongside the certified
    local-AC baseline and the complete-solver reference. One dedP per instance (final-state metric)."""
    m.eval()
    doms = neural_fixpoint(m, csps, dev, theta, R_max)
    nv = neu_solved = neu_wrong = ac_solved = solver_solved = uniq = 0
    rec_num = rec_den = 0
    ac_rec_num = ac_rec_den = 0
    fe_count = 0
    for csp, dom in zip(csps, doms):
        if not C.solutions(csp, limit=1):
            continue
        nv += 1
        full = csp.full()
        ded, _ = C.to_fixpoint(C.exact_dedP, csp, full)       # the exact modular dedP (== solve_mod)
        ac, _ = C.to_fixpoint(C.ac_step, csp, full)           # local certified residue-AC baseline
        rm_ded = _removed(full, ded, csp.n)
        rm_neu = _removed(full, dom, csp.n)
        rm_ac = _removed(full, ac, csp.n)
        rec_num += len(rm_neu & rm_ded); rec_den += len(rm_ded)
        ac_rec_num += len(rm_ac & rm_ded); ac_rec_den += len(rm_ded)
        fe_count += len(rm_neu - rm_ded)                      # SOUNDNESS: dropped a value some solution uses
        ac_solved += int(C.status(ac) == "solved")
        solver_solved += int(C.status(ded) == "solved")
        uniq += int(all(len(c) == 1 for c in ded))
        st = C.status(dom); neu_solved += int(st == "solved")
        if st == "solved":
            assign = [next(iter(dom[i])) for i in range(csp.n)]
            neu_wrong += int(not all(tuple(assign[c] for c in sc) in al for sc, al in csp.cons))
    m.train()
    ns = max(1, nv)
    return {"n": nv,
            "neural_recall": rec_num / max(1, rec_den),
            "ac_recall": ac_rec_num / max(1, ac_rec_den),
            "solver_recall": 1.0,                            # certified solve_mod == exact dedP (proven)
            "neural_solved": neu_solved / ns, "ac_solved": ac_solved / ns,
            "solver_solved": solver_solved / ns, "uniq": uniq / ns,
            "false_elim": fe_count / max(1, rec_den), "false_elim_count": int(fe_count),
            "neural_wrong": neu_wrong / ns}


def _row(tag, e):
    return (f"  {tag:16s} n={e['n']:3d}  recall: neural {e['neural_recall']*100:5.1f}%  "
            f"AC {e['ac_recall']*100:5.1f}%  solver {e['solver_recall']*100:5.1f}%  |  "
            f"solved: neural {e['neural_solved']*100:5.1f}% AC {e['ac_solved']*100:5.1f}% "
            f"solver {e['solver_solved']*100:5.1f}%(uniq {e['uniq']*100:4.1f}%)  |  "
            f"FALSE-ELIM {e['false_elim']:.4f}({e['false_elim_count']}) wrong {e['neural_wrong']*100:.1f}%")


# ============================================================ eval-set construction (moduli x sizes)
def make_eval_by_modulus(rung, moduli, neval, seed=4321):
    rng = np.random.default_rng(seed)
    out = {}
    for mm in moduli:
        cs = []
        while len(cs) < neval:
            csp = GEN[rung](rng, m=mm)
            if _fits(csp) and len(C.solutions(csp, limit=1)) > 0:
                cs.append(csp)
        out[mm] = cs
    return out


def make_eval_by_size(rung, m, sizes, neval, seed=99):
    rng = np.random.default_rng(seed)
    out = {}
    for nn in sizes:
        cs = []
        tries = 0
        while len(cs) < neval and tries < neval * 200:
            tries += 1
            csp = GEN[rung](rng, m=m, n=nn)
            if _fits(csp) and len(C.solutions(csp, limit=1)) > 0:
                cs.append(csp)
        out[nn] = cs
    return out


# ============================================================ training (blade, dominate-dedP, on-policy)
def train(rungs, dev, d, steps, pool=96, R=12, lr=3e-4, theta=0.5, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = BladeFactorDeductor("blade", N_MAX, D_MAX, M_MAX, A_MAX, d=d, R=R, ds=min(4, R)).to(dev)
    npar = m.n_params()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95))
    tagged = sample_corpus(rng, pool, rungs)
    items = [(c, c.full()) for _, c in tagged]
    rtags = [rg for rg, _ in tagged]
    print(f"  modular organ: blade d={d} params={npar:,} pool={pool} R={R} steps={steps} "
          f"budget N={N_MAX} D={D_MAX} M={M_MAX} moduli={MODULI}", flush=True)
    fe_k = fe_n = 0; t0 = time.time(); log = []
    for s in range(1, steps + 1):
        feat = RG.featurize(items, dev); vm = feat["var_mask"]
        tgt, conflict = RG.build_targets(items, RG.SHARED, dev)
        b, cls, sup = RG.fwd(m, feat, vm)
        loss = RG.loss_fn(sup, vm, feat["var_valid"], tgt, conflict)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        with torch.no_grad():
            new_vm = RG.meet(vm, b, theta)
            k, nn_ = RG.false_elim(vm, new_vm, feat["var_valid"], tgt); fe_k += k; fe_n += nn_
            nvc = new_vm.cpu().numpy(); nxt = []
            for bi, (csp, dom) in enumerate(items):
                ndom = RG.dom_from_mask(nvc[bi], csp)
                if C.status(ndom) in ("solved", "conflict") or ndom == dom:
                    rg = rtags[bi]; nc = sample(rng, rg); nxt.append((nc, nc.full()))
                else:
                    nxt.append((csp, ndom))
            items = nxt
        if s % max(1, steps // 15) == 0:
            fer = fe_k / max(1, fe_n)
            log.append({"step": s, "loss": float(loss.detach()), "false_elim": fer})
            print(f"    step {s:5d}  loss {float(loss.detach()):.3f}  false_elim {fer:.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
            fe_k = fe_n = 0
    return m, npar, log


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--R", type=int, default=12)
    ap.add_argument("--target", type=float, default=4e5)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--pool", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--neval", type=int, default=80)
    ap.add_argument("--rmax", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="runs/modular_organ.pt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = RG.device()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    print(f"device={dev}  budget N={N_MAX} D={D_MAX} M={M_MAX} A={A_MAX}", flush=True)

    if args.smoke:
        rng = np.random.default_rng(0)
        print("=== run_modular SMOKE: generators within budget + dedP==solver, blade trains ===", flush=True)
        for rg in TRAIN_RUNGS:
            t0 = time.time(); csp = sample(rng, rg)
            ded, _ = C.to_fixpoint(C.exact_dedP, csp, csp.full())
            base = certified_baselines(csp)
            print(f"  {rg:12s} n={csp.n:2d} d={csp.d:2d} m={len(csp.cons):3d} "
                  f"dedP-uniq={base['dedP_uniq']} AC-solved={base['ac_solved']} "
                  f"solver-solved={base['solver_solved']} ({(time.time()-t0)*1000:.0f}ms)", flush=True)
        d, npar = size_for("blade", N_MAX, D_MAX, M_MAX, A_MAX, args.target, R=args.R, ds=4)
        print(f"  size_for(blade,target={args.target:.0e}) -> d={d} params={npar:,}", flush=True)
        m, npar, log = train(TRAIN_RUNGS, dev, d, steps=150, pool=48, R=args.R, seed=0)
        for rg in TRAIN_RUNGS:
            cs = [sample(rng, rg) for _ in range(40)]
            print(_row(rg, evaluate(m, cs, dev, args.theta, args.rmax)), flush=True)
        return

    if args.train:
        d, npar = size_for("blade", N_MAX, D_MAX, M_MAX, A_MAX, args.target, R=args.R, ds=4)
        m, npar, log = train(TRAIN_RUNGS, dev, d, args.steps, pool=args.pool, R=args.R,
                             lr=args.lr, theta=args.theta, seed=args.seed)
        out = {"budget": [N_MAX, D_MAX, M_MAX, A_MAX], "moduli": MODULI, "params": npar,
               "log": log, "args": vars(args)}

        print("\n=== PER-RUNG (mixed moduli, in-distribution) ===", flush=True)
        out["per_rung"] = {}
        for rg in TRAIN_RUNGS:
            cs = [sample(np.random.default_rng(1000 + i), rg) for i in range(args.neval)]
            e = evaluate(m, cs, dev, args.theta, args.rmax); out["per_rung"][rg] = e
            print(_row(rg, e), flush=True)

        print("\n=== lineq RECALL/SOUNDNESS ACROSS MODULI (neural vs certified AC vs complete solver) ===",
              flush=True)
        out["by_modulus"] = {}
        for mm, cs in make_eval_by_modulus("lineq", MODULI, args.neval).items():
            e = evaluate(m, cs, dev, args.theta, args.rmax); out["by_modulus"][mm] = e
            print(_row(f"lineq m={mm}", e), flush=True)

        print("\n=== cyclic (MODULAR PARITY WALL) ACROSS MODULI — the rung local AC ABSTAINS on ===",
              flush=True)
        out["cyclic_by_modulus"] = {}
        for mm, cs in make_eval_by_modulus("cyclic", MODULI, args.neval).items():
            e = evaluate(m, cs, dev, args.theta, args.rmax); out["cyclic_by_modulus"][mm] = e
            print(_row(f"cyclic m={mm}", e), flush=True)

        print("\n=== lineq RECALL/SOUNDNESS ACROSS SYSTEM SIZES (m=7) ===", flush=True)
        out["by_size"] = {}
        for nn, cs in make_eval_by_size("lineq", 7, [4, 5, 6, 7, 8], args.neval).items():
            if not cs:
                continue
            e = evaluate(m, cs, dev, args.theta, args.rmax); out["by_size"][nn] = e
            print(_row(f"lineq n={nn}", e), flush=True)

        os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
        torch.save({"state_dict": m.state_dict(),
                    "config": {"msg": "blade", "n_max": N_MAX, "d_max": D_MAX, "m_max": M_MAX,
                               "a_max": A_MAX, "d": d, "R": args.R, "ds": min(4, args.R)},
                    "params": npar, "moduli": MODULI, "train_rungs": TRAIN_RUNGS,
                    "per_rung": out["per_rung"], "by_modulus": out["by_modulus"],
                    "cyclic_by_modulus": out["cyclic_by_modulus"],
                    "by_size": out["by_size"], "log": log, "args": vars(args)}, args.ckpt)
        print(f"\nsaved checkpoint -> {args.ckpt} ({npar:,} params)", flush=True)
        if args.out:
            json.dump(out, open(args.out, "w"), indent=1, default=str)
            print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
