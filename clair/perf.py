"""Shared performance setup. TF32 is free and safe (only matmul mantissa, keeps fp32 range).
bf16 autocast is opt-in (--amp) because GLaDOS's soundness depends on precise sigmoid thresholds —
we run the model forward in bf16 but cast decision logits back to fp32 before any lattice meet /
threshold / loss, so the sound decisions stay fp32 while the matmuls get the 2x."""
from __future__ import annotations

import contextlib

import torch


def setup():
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def amp(dev, enabled=True):
    if enabled and torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()
