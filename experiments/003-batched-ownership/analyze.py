"""SPIKE: summarize experiment 003 runs from what the worker actually logged.

    python analyze.py NOPLUGIN_RUN_DIR PLUGIN_RUN_DIR

Per request: how many rows its DiT calls carried, what each call decided, and
how many blocks ran. Then image comparisons: the plugin run against the
no-plugin run request by request, and within the plugin run the pairs that
test ownership (does one item's decision leave the other alone?). Exact
equality first, then max absolute difference and PSNR on 8-bit pixels.
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np
from PIL import Image

# (left, right, what equality would show). Request ids are "<round>-<rep>.<item>".
PAIRS = [
    ("batch-observe-1.a", "batch-observe-0.a", "batched runs repeat"),
    ("batch-observe-0.a", "solo-a-0.a", "batching vs alone, a"),
    ("batch-observe-0.b", "solo-b-0.b", "batching vs alone, b"),
    ("batch-a-identity-0.b", "batch-observe-0.b", "a's identity leaves b alone"),
    ("batch-a-reuse-0.b", "batch-observe-0.b", "a's reuse leaves b alone"),
    ("batch-a-reuse-0.a", "solo-a-reuse-0.a", "a's reuse, batched vs alone"),
    ("batch-both-reuse-0.a", "batch-a-reuse-0.a", "a's reuse, skip vs splice"),
    ("batch-both-reuse-0.b", "batch-observe-0.b", "b's reuse cost"),
    ("batch-a-reuse-0.a", "batch-observe-0.a", "a's reuse cost"),
    ("batch-observe-after-0.a", "batch-observe-0.a", "no state leaks after control, a"),
    ("batch-observe-after-0.b", "batch-observe-0.b", "no state leaks after control, b"),
]


def pixels(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)


def compare(a, b):
    d = np.abs(pixels(a) - pixels(b))
    mse = float((d.astype(np.float64) ** 2).mean())
    if mse == 0:
        return "identical"
    return f"max_abs={int(d.max())} psnr={10 * np.log10(255**2 / mse):.2f}dB"


def load(run):
    summary = json.loads((run / "summary.json").read_text())
    log = run / "probe.jsonl"
    events = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    return summary, [e for e in events if e.get("event") == "trunk" and not e["warmup"]]


def main():
    ref_dir, run_dir = map(pathlib.Path, sys.argv[1:3])
    ref, _ = load(ref_dir)
    run, trunks = load(run_dir)
    images = {r["request_id"]: r["image"] for r in run["rows"]}
    ref_images = {r["request_id"]: r["image"] for r in ref["rows"]}
    by_tag = collections.defaultdict(list)
    for e in trunks:
        by_tag[e["tag"]].append(e)

    print(f"## {run_dir.name}  kwargs={ {k: v for k, v in run['server_kwargs'].items() if k != 'model_path'} }")
    for row in run["rows"]:
        mine = by_tag.get(row["request_id"], [])
        decisions = collections.Counter(e["call_decision"] for e in mine)
        reused = [(e["step_index"], e["branch"]) for e in mine if e.get("decision") == "reuse"]
        exact = [e["identity_exact"] for e in mine if "identity_exact" in e]
        print(f"  {row['request_id']:<26} submitted_with={row['round_size']} "
              f"rows_per_call={sorted({e['rows'] for e in mine})} calls={len(mine)} "
              f"blocks_run={sum(e['blocks_run'] for e in mine)} "
              f"blocks_skipped={sum(e['blocks_skipped'] for e in mine)} "
              f"decisions={dict(decisions)} reused_at={reused or '-'} "
              f"identity_exact={('all' if all(exact) else exact.count(False)) if exact else '-'} "
              f"wall={row['wall_s']:.2f}s")

    print("\n### plugin vs no-plugin, same request")
    for rid, path in images.items():
        if rid in ref_images:
            print(f"  {rid:<26} {compare(path, ref_images[rid])}")

    print("\n### ownership pairs (plugin run)")
    for left, right, what in PAIRS:
        if left in images and right in images:
            print(f"  {what:<34} {left} vs {right}: {compare(images[left], images[right])}")

    print("\n### signal per item: batched vs alone (max |Δ rel_l1| over steps)")
    for item, solo in (("a", "solo-a-0.a"), ("b", "solo-b-0.b")):
        alone = {(e["step_index"], e["branch"]): e.get("signal_rel_l1") for e in by_tag.get(solo, [])}
        batched = {(e["step_index"], e["branch"]): e.get("signal_rel_l1")
                   for e in by_tag.get(f"batch-observe-0.{item}", [])}
        diffs = [abs(batched[k] - alone[k]) for k in alone
                 if alone[k] is not None and batched.get(k) is not None]
        if diffs:
            print(f"  {item}: steps={len(diffs)} max_delta={max(diffs):.3e}")


if __name__ == "__main__":
    main()
