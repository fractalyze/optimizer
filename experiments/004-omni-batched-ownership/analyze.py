"""SPIKE: summarize experiment 004 runs from what the worker actually logged.

    python analyze.py RUN_DIR [NO_PROBE_RUN_DIR]

Per request: the rows its DiT calls carried, its own step sequence (from the
engine, and as counted by the adapter), what each call decided, and blocks
run and skipped. Its *composition* is, per call, its own step and the step of
every other row in that call: batching changes numerics, so two images are
compared bitwise only when their compositions match. Then image comparisons,
exact first, then max absolute difference and PSNR on 8-bit pixels.
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np
from PIL import Image


def compare(a, b):
    pa, pb = (np.asarray(Image.open(p).convert("RGB"), dtype=np.int16) for p in (a, b))
    d = np.abs(pa - pb)
    mse = float((d.astype(np.float64) ** 2).mean())
    return "identical" if mse == 0 else f"max_abs={int(d.max())} psnr={10 * np.log10(255**2 / mse):.2f}dB"


def load(run):
    summary = json.loads((run / "summary.json").read_text())
    log = run / "probe.jsonl"
    events = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    return summary, events


def calls_by_key(events):
    """Group trunk events into DiT calls (consecutive rows) and index them by key."""
    trunks = [e for e in events if e.get("event") == "trunk" and e.get("key")]
    calls, current = [], []
    for e in trunks:
        if e["row"] == 0 and current:
            calls.append(current)
            current = []
        current.append(e)
    if current:
        calls.append(current)
    mine = collections.defaultdict(list)
    for call in calls:
        for e in call:
            others = tuple(sorted((o["key"].rpartition(".")[2], o["step_index"])
                                  for o in call if o is not e))
            mine[e["key"]].append(dict(e, others=others))
    return mine


def composition(entries):
    return tuple((e["step_index"], e["others"]) for e in entries)


def main():
    run = pathlib.Path(sys.argv[1])
    summary, events = load(run)
    mine = calls_by_key(events)
    images = {r["request_id"]: r["image"] for r in summary["rows"]}
    retire = [e for e in events if e.get("event") == "retire"]

    print(f"## {run.name}  engine={ {k: v for k, v in summary['engine_kwargs'].items() if k != 'model'} }")
    for row in summary["rows"]:
        es = mine.get(row["request_id"], [])
        steps = [e["step_index"] for e in es]
        engine_ok = all(e["engine_step"] in (None, e["counted_step"]) for e in es)
        mixed = sorted({(e["step_index"], o) for e in es for _, o in e["others"] if o != e["step_index"]})
        print(f"  {row['request_id']:<30} rows/call={sorted({e['rows'] for e in es})} "
              f"steps={steps} engine==counted={engine_ok} "
              f"decisions={dict(collections.Counter(e['call_decision'] for e in es))} "
              f"blocks_run={sum(e['blocks_run'] for e in es)} skipped={sum(e['blocks_skipped'] for e in es)} "
              f"reused_at={[(e['step_index'], e['others']) for e in es if e.get('decision') == 'reuse'] or '-'} "
              f"identity_exact={'all' if [e for e in es if 'identity_exact' in e] and all(e['identity_exact'] for e in es if 'identity_exact' in e) else '-'} "
              f"steps_shared_with_other_step={mixed or '-'}")
    print(f"  retired owners: {len(retire)}; live owners at last retire: "
          f"{retire[-1]['live_owners'] if retire else '-'}")

    print("\n### same composition, compared bitwise")
    by_comp = collections.defaultdict(list)
    for rid in images:
        by_comp[(rid.rpartition(".")[2], composition(mine.get(rid, [])))].append(rid)
    for (item, _), rids in by_comp.items():
        for other in rids[1:]:
            print(f"  {item}: {rids[0]} vs {other}: {compare(images[rids[0]], images[other])}")

    print("\n### every request against its round's observe partner (quality of control)")
    for rid, path in images.items():
        item = rid.rpartition(".")[2]
        ref = next((r for r in images if r.endswith(f"observe-0.{item}")), None)
        if ref and ref != rid:
            same = composition(mine.get(rid, [])) == composition(mine.get(ref, []))
            print(f"  {rid:<30} vs {ref}: {compare(path, images[ref])}"
                  f"{'' if same else '  (composition differs)'}")

    if len(sys.argv) > 2:
        ref_summary, _ = load(pathlib.Path(sys.argv[2]))
        ref_images = {r["request_id"]: r["image"] for r in ref_summary["rows"]}
        print("\n### probe vs no probe, same request")
        for rid, path in images.items():
            if rid in ref_images:
                print(f"  {rid:<30} {compare(path, ref_images[rid])}")


if __name__ == "__main__":
    main()
