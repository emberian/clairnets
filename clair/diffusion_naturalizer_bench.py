"""Secondary benchmark: is a diffusion LM good enough as the NATURALIZER layer of our datagen corpus
(rule-based bulk facts + diffusion-LM paraphrase into varied English)?

HONEST MODEL CAVEAT: the requested model, DiffusionGemma (google/diffusiongemma-26B-A4B-it), is a
25.2B-total MoE (~50 GB bf16, all experts resident) and does NOT fit one L40S (46 GB) without 4-bit
quantization. So this benchmarks the diffusion LM that DOES fit and is already loaded here,
GSAI-ML/LLaDA-8B-Instruct — a real masked-diffusion LM — as the stand-in for the "diffusion-LM-as-
naturalizer" question. Conclusions about throughput/quality transfer in kind (DiffusionGemma would be
slower per step but stronger per token); the FIT verdict is the headline.

We: take N coloring CSP problems, hand the model the STRUCTURED facts, ask for a short natural English
paraphrase, then (1) round-trip extract the facts back and score faithfulness, (2) time records/sec,
(3) print samples to judge naturalness/variety.
"""
from __future__ import annotations
import re, time, argparse
import numpy as np
import torch

from . import curriculum as Cu
from .diffusion_organ import LLaDAAdapter

COLORS = Cu.COLORS


# --------------------------------------------------------------------- LLaDA masked-diffusion generation
@torch.no_grad()
def llada_generate(model, prompt_ids, gen_len=96, steps=48, mask_id=126336, temperature=0.0):
    """Canonical LLaDA iterative-unmask loop: gen region starts all-MASK; each step commit the highest-
    confidence still-masked positions on an even schedule until none remain."""
    dev = prompt_ids.device
    x = torch.cat([prompt_ids, torch.full((1, gen_len), mask_id, device=dev, dtype=torch.long)], 1)
    plen = prompt_ids.size(1)
    for i in range(steps):
        mask_index = (x == mask_id)
        if not mask_index[:, plen:].any():
            break
        logits = model(x).logits
        if temperature > 0:
            noise = torch.rand_like(logits.float()).clamp_min(1e-9)
            logits = logits.float() + temperature * (-torch.log(-torch.log(noise)))
        x0 = logits.argmax(-1)
        conf = logits.softmax(-1).gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        conf = torch.where(mask_index, conf, torch.full_like(conf, -1e30))
        n_left = int(mask_index[:, plen:].sum())
        budget = max(1, n_left // (steps - i))
        idx = torch.topk(conf[0], k=min(budget, n_left)).indices
        x[0, idx] = x0[0, idx]
    return x[0, plen:]


# --------------------------------------------------------------------- structured facts <-> text
def problem_facts(rng, N=5, k=3):
    n, d, kind, facts, s = Cu.gen_coloring(rng, k=k, n_lo=N, n_hi=N, edge_p=0.5, pin_frac=0.4)
    ents = [Cu.ENTITIES[i] for i in range(n)]
    pins = {ents[f[1]]: COLORS[f[2]] for f in facts if f[0] == "pin"}
    neqs = {frozenset((ents[f[1]], ents[f[2]])) for f in facts if f[0] == "neq"}
    return ents, pins, neqs


def facts_to_bullets(ents, pins, neqs):
    lines = []
    for e, c in pins.items():
        lines.append(f"- {e} is {c}.")
    for pr in neqs:
        a, b = tuple(pr)
        lines.append(f"- {a} and {b} must be different colors.")
    return "\n".join(lines)


def extract_facts(text, ents, k_colors):
    """Approximate round-trip parser. pins: '<E> is/are <color>'. neqs: a sentence mentioning both
    entities together with a difference cue (different/differ/not the same/distinct/can't ... same)."""
    t = text.lower()
    pins = {}
    for e in ents:
        for c in k_colors:
            if re.search(rf"\b{e.lower()}\b[^.]*\b(?:is|are|=|colored|gets?)\b[^.]*\b{c}\b", t):
                pins[e] = c
    neqs = set()
    diff = r"(different|differ|not the same|distinct|cannot be the same|can't be the same|unequal|nor)"
    sents = re.split(r"[.;\n]", t)
    for s in sents:
        present = [e for e in ents if re.search(rf"\b{e.lower()}\b", s)]
        if len(present) >= 2 and re.search(diff, s):
            for a in range(len(present)):
                for b in range(a + 1, len(present)):
                    neqs.add(frozenset((present[a], present[b])))
    return pins, neqs


# --------------------------------------------------------------------- benchmark
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--gen-len", type=int, default=110)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    args = ap.parse_args()

    ad = LLaDAAdapter(args.model)
    tok, model = ad.tok, ad.model
    print(f"loaded {args.model} for naturalizer benchmark\n")

    rng = np.random.default_rng(0)
    samples = []
    t0 = time.time()
    fact_total = fact_recovered = spurious = 0
    pins_total = pins_recovered = 0
    texts = []
    for qi in range(args.n):
        ents, pins, neqs = problem_facts(rng)
        bullets = facts_to_bullets(ents, pins, neqs)
        instr = ("Rewrite the following graph-coloring constraints as ONE short, natural paragraph of "
                 "varied English. Use these exact names and color words; do not add, drop, or change any "
                 "constraint.\n\n" + bullets + "\n\nParagraph:")
        msgs = [{"role": "user", "content": instr}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to(ad.device)
        out = llada_generate(model, ids, gen_len=args.gen_len, steps=args.steps,
                             mask_id=ad.MASK_ID, temperature=args.temperature)
        text = tok.decode(out, skip_special_tokens=True).strip()
        texts.append(text)
        gp, gn = extract_facts(text, ents, COLORS)
        # faithfulness: recall of pins + neqs, and spurious (extracted not in truth)
        pins_total += len(pins)
        pins_recovered += sum(1 for e, c in pins.items() if gp.get(e) == c)
        fact_total += len(neqs)
        fact_recovered += len(neqs & gn)
        spurious += len(gn - neqs) + sum(1 for e, c in gp.items() if pins.get(e) != c)
        if qi < 4:
            samples.append((bullets, text))
    dt = time.time() - t0

    print("=== SAMPLES (structured -> diffusion-LM paraphrase) ===")
    for b, t in samples:
        print("FACTS:\n" + b)
        print("PARAPHRASE:\n" + (t[:400] if t else "<empty>"))
        print("-" * 60)
    uniq = len(set(texts))
    print("\n=== VERDICT ===")
    print(f"records              : {args.n}")
    print(f"throughput           : {args.n/dt:.2f} rec/s  ({dt:.1f}s total, {args.steps} denoise steps/rec)")
    print(f"pin faithfulness     : {pins_recovered}/{pins_total} recovered")
    print(f"neq faithfulness     : {fact_recovered}/{fact_total} recovered")
    print(f"spurious facts       : {spurious}  (added/changed constraints — lower is better)")
    print(f"output variety       : {uniq}/{args.n} distinct paraphrases")
    print(f"mean output chars    : {np.mean([len(t) for t in texts]):.0f}")


if __name__ == "__main__":
    main()
