"""clair/organ/eval.py — THE canonical eval entry (the arbiter).

A thin front-door over clair.eval_suite: the standardized Tier-1/2/3 + causal-controls + pass@k
judge. Tier 1 = exact-verified logic/CSP (the woven model's home turf) + reasoning-gym ceiling; Tier 2
= reasoning transfer (GSM8K / ARC / ...); Tier 3 = no-harm general ability (HellaSwag / WikiText / ...).
Arms: base · text-LoRA · woven · oracle-override. Nothing is re-implemented here — this re-exports the
validated run_eval_suite so "evaluate GLaDOS" has one obvious name.

  from clair.organ.eval import run_eval_suite
  run_eval_suite("allenai/OLMo-2-0425-1B", woven_ckpt="runs/woven.pt")

  python -m clair.organ.eval --base allenai/OLMo-2-0425-1B --woven_ckpt runs/woven.pt
  python -m clair.organ.eval --smoke        # train+save+reload a tiny woven model, eval every tier
"""
from __future__ import annotations

from ..eval_suite import run_eval_suite, load_woven, save_woven, build_smoke_woven, main

__all__ = ["run_eval_suite", "load_woven", "save_woven", "build_smoke_woven", "main"]


if __name__ == "__main__":
    main()
