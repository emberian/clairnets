"""clair/organ/train.py — THE single training + LLM entry point for the GLaDOS organ.

Two clean calls, both delegating to the already-validated recipes (NOT re-prototyped):

  (a) ORGAN-SIDE TRAINING  ("train the organ"): on-policy DOMINATE-dedP of the general narrow organ
      over the 7-rung MIX, via clair.run_glados_staged.train_organ (STAGE 1). Saves the union organ
      in the {'state','meta'} format clair.organ.bank.load_core_organ reads. The MIX of witness-first
      generators IS the corpus (each rung's solvable instances); the diverse Bedrock phrasings at
      data/curriculum/curriculum.jsonl feed the OOD eval. (data/glados_corpus is optional — the
      generators produce the training corpus on the fly.)

  (b) WEAVE INTO THE LLM  ("weave into OLMo for full-training"): the STAGED woven recipe
      bootstrap -> freeze -> structured-gamma readout -> generate, via clair.run_glados_staged.main:
      STAGE 1 trains+freezes the organ, STAGE 2 trains OLMo-2-1B (frozen + LoRA) + the zero-init
      gamma that injects the frozen organ's narrowed lattice into a late residual layer, with the
      causal-control table (shuffle/permute/corrupt/zero) proving the LM causally WIELDS the organ.

Usage:
  python -m clair.organ.train organ  --steps 1500 --out runs/general_organ_full.pt
  python -m clair.organ.train weave  --organ_steps 1500 --steps 2500 --out runs/glados_staged.json
  python -m clair.organ.train weave  --smoke          # tiny end-to-end woven smoke (needs OLMo)
"""
from __future__ import annotations

import argparse
import os
import sys


def train_organ(out="runs/general_organ_full.pt", steps=1500, target=3.0e5, pool=128, R=8,
                lr=3e-4, seed=0, dev=None):
    """STAGE 1: bootstrap the union-trained general narrow organ and save it in load_core_organ's
    format. Returns (organ, meta). Reuses clair.run_glados_staged.train_organ verbatim."""
    import torch
    from .. import run_glados_staged as G
    dev = dev or G.device()
    print(f"[organ] dominate-dedP over rungs={G.RUNGS}  steps={steps} target={target:g} dev={dev}",
          flush=True)
    organ, d, npar, log = G.train_organ(dev, G.RUNGS, target=target, steps=steps, pool=pool,
                                        R=R, lr=lr, seed=seed)
    for p in organ.parameters():
        p.requires_grad_(False)
    organ.eval()
    meta = {"N_MAX": G.N_MAX, "D_MAX": G.D_MAX, "M_MAX": G.M_MAX, "A_MAX": G.A_MAX,
            "d": d, "params": npar, "R": R}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save({"state": organ.state_dict(), "meta": meta}, out)
    print(f"[organ] saved {out}  d={d} params={npar:,} R={R}", flush=True)
    return organ, meta


def weave(argv=None):
    """STAGE 1 + STAGE 2: the full staged woven recipe (organ -> OLMo LM head). Delegates to
    clair.run_glados_staged.main, which owns the bootstrap/freeze/gamma/generate + causal controls."""
    from .. import run_glados_staged as G
    old = sys.argv
    try:
        sys.argv = ["clair.run_glados_staged"] + list(argv or [])
        G.main()
    finally:
        sys.argv = old


def main():
    ap = argparse.ArgumentParser(description="GLaDOS organ training + LLM weaving entry point")
    sub = ap.add_subparsers(dest="cmd", required=True)

    po = sub.add_parser("organ", help="STAGE 1: train + save the union general narrow organ")
    po.add_argument("--out", default="runs/general_organ_full.pt")
    po.add_argument("--steps", type=int, default=1500)
    po.add_argument("--target", type=float, default=3.0e5)
    po.add_argument("--pool", type=int, default=128)
    po.add_argument("--R", type=int, default=8)
    po.add_argument("--lr", type=float, default=3e-4)
    po.add_argument("--seed", type=int, default=0)

    pw = sub.add_parser("weave", help="STAGE 1+2: the staged woven recipe into OLMo (passes args through)")
    pw.add_argument("rest", nargs=argparse.REMAINDER,
                    help="args forwarded to clair.run_glados_staged (e.g. --smoke, --steps N, --out P)")

    a = ap.parse_args()
    if a.cmd == "organ":
        train_organ(out=a.out, steps=a.steps, target=a.target, pool=a.pool, R=a.R, lr=a.lr, seed=a.seed)
    elif a.cmd == "weave":
        weave(a.rest)


if __name__ == "__main__":
    main()
