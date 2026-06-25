"""Generate the curriculum dataset: harness CSP -> Bedrock diverse renderings -> faithfulness
round-trip -> cached jsonl of (diverse_text, exact_CSP, query, answer).

Ground truth is generated FIRST by clair.curriculum (witness-first, exact via clair.csp). Bedrock
(cheap Nova-2-Lite by default) only RENDERS and is then CHECKED by a second extract-and-compare
call; renderings that don't round-trip to the exact facts are DROPPED.

    # pilot: tiny, prints 3 diverse renderings of one CSP + the keep-rate
    PYTHONPATH=/Users/ember/dev/gowexp/src:/Users/ember/dev/clairnets \
        /Users/ember/dev/gowexp/.venv/bin/python -m clair.gen_curriculum --pilot

    # modest real dataset
    PYTHONPATH=... python -m clair.gen_curriculum --n 400 --k 4 --workers 16 \
        --out /Users/ember/dev/clairnets/data/curriculum/curriculum.jsonl

Needs boto3 (run from the gowexp venv) + AWS creds + clair on PYTHONPATH.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from . import curriculum as Cur

# Nova-2-Lite on-demand price, USD per 1e6 tokens (input, output). Estimate only.
_PRICE = {"us.amazon.nova-2-lite-v1:0": (0.06, 0.24)}


def _process(p, k, model, region):
    """Render one problem K ways, faithfulness-check each, return (kept_records, usage, n_render)."""
    texts, ru = Cur.render_diverse(p, k=k, model=model, region=region)
    in_tok, out_tok = ru
    kept = []
    n_seen = len(texts)
    for t in texts:
        ok, _, (ei, eo) = Cur.extract_and_check(p, t, model=model, region=region)
        in_tok += ei
        out_tok += eo
        if ok:
            kept.append(Cur.to_record(p, t, source=model))
    return kept, (in_tok, out_tok), n_seen


def run(n, k, model, region, workers, relations, seed, out_path):
    rng = np.random.default_rng(seed)
    probs = [Cur.gen_problem(rng, relation=(None if relations == "all" else
             str(rng.choice(relations)))) for _ in range(n)]

    lock = threading.Lock()
    tot_in = tot_out = n_render = n_kept = 0
    per_rel = {}                                              # relation -> [kept, rendered]
    t0 = time.time()
    fh = open(out_path, "w")
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_process, p, k, model, region): p for p in probs}
            for i, fut in enumerate(as_completed(futs), 1):
                p = futs[fut]
                try:
                    kept, (ti, to), seen = fut.result()
                except Exception as e:                        # noqa: BLE001 — log + continue
                    print(f"  ! {p.relation}: {type(e).__name__}: {str(e)[:90]}")
                    continue
                with lock:
                    tot_in += ti; tot_out += to; n_render += seen; n_kept += len(kept)
                    r = per_rel.setdefault(p.relation, [0, 0])
                    r[0] += len(kept); r[1] += seen
                    for rec in kept:
                        fh.write(json.dumps(rec) + "\n")
                    if i % 25 == 0 or i == len(probs):
                        kr = n_kept / max(1, n_render)
                        print(f"  [{i:4d}/{len(probs)}] kept {n_kept:5d}/{n_render:5d} "
                              f"(faithful {kr:.1%})  {time.time()-t0:.0f}s", flush=True)
    finally:
        fh.close()

    pin, pout = _PRICE.get(model, (1.0, 5.0))
    cost = tot_in / 1e6 * pin + tot_out / 1e6 * pout
    print(f"\ndataset -> {out_path}")
    print(f"kept {n_kept} / {n_render} renderings  (overall faithfulness keep-rate "
          f"{n_kept/max(1,n_render):.1%})")
    print("per-relation keep-rate:")
    for rel, (kp, sn) in sorted(per_rel.items()):
        print(f"  {rel:11s} {kp:5d}/{sn:5d}  {kp/max(1,sn):.1%}")
    print(f"tokens: in={tot_in:,} out={tot_out:,}  est. cost ${cost:.3f}  "
          f"({time.time()-t0:.0f}s, {model})")
    return n_kept


def pilot(model, region):
    """Tiny demonstration: 3 diverse renderings of ONE CSP + a small keep-rate sample."""
    rng = np.random.default_rng(1)
    print("== PILOT: 3 diverse renderings of one CSP ==\n")
    p = Cur.gen_problem(rng, relation="coloring", det_target=True)
    print("CANONICAL (ground truth):")
    print("  " + Cur.canonical_render(p))
    print(f"  query={p.entity(p.query)}  answer={Cur.canonical_answer(p)!r}\n")
    texts, _ = Cur.render_diverse(p, k=3, model=model, region=region)
    for i, t in enumerate(texts, 1):
        ok, ne, _ = Cur.extract_and_check(p, t, model=model, region=region)
        print(f"RENDERING {i} [faithful={ok}]:\n  {t}\n")

    print("== keep-rate over a small mixed sample (5 problems x 4 renderings) ==")
    kept = seen = 0
    for _ in range(5):
        q = Cur.gen_problem(rng)
        ts, _ = Cur.render_diverse(q, k=4, model=model, region=region)
        for t in ts:
            seen += 1
            ok, _, _ = Cur.extract_and_check(q, t, model=model, region=region)
            kept += int(ok)
    print(f"  kept {kept}/{seen}  ({kept/max(1,seen):.0%} faithful)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="number of base CSPs")
    ap.add_argument("--k", type=int, default=4, help="diverse renderings per CSP")
    ap.add_argument("--model", default=Cur.DEFAULT_MODEL)
    ap.add_argument("--region", default=Cur.DEFAULT_REGION)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--relations", default="all",
                    help="'all' or comma list: coloring,equality,ordering,arithmetic,alldiff")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..",
                    "data", "curriculum", "curriculum.jsonl"))
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()

    if a.pilot:
        pilot(a.model, a.region)
        return
    rels = "all" if a.relations == "all" else a.relations.split(",")
    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"generating {a.n} base CSPs x {a.k} renderings via {a.model} ...", flush=True)
    run(a.n, a.k, a.model, a.region, a.workers, rels, a.seed, out)


if __name__ == "__main__":
    main()
