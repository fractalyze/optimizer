"""SPIKE: summarize experiment 001 runs from what the worker actually logged.

    python analyze.py REF_IMAGE RUN_DIR [RUN_DIR ...]

For every request: steps the worker executed, model calls, scheduler-index
anomalies, wall time, and the image compared with REF_IMAGE (exact equality
first, then max/mean absolute difference and PSNR on 8-bit pixels).
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys

import numpy as np
from PIL import Image


def pixels(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)


def compare(a, b):
    if a.shape != b.shape:
        return dict(identical=False, shape_mismatch=True)
    d = np.abs(a - b)
    mse = float((d.astype(np.float64) ** 2).mean())
    return dict(identical=bool(mse == 0.0), max_abs=int(d.max()), mean_abs=round(float(d.mean()), 4),
                psnr_db=None if mse == 0 else round(float(10 * np.log10(255**2 / mse)), 2))


def request_events(run):
    path = run / "probe.jsonl"
    events = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    by_rid = {}
    for e in events:
        if e.get("request_id") is not None:
            by_rid.setdefault(e["request_id"], []).append(e)
    return by_rid


def main():
    ref = pixels(sys.argv[1])
    for run in map(pathlib.Path, sys.argv[2:]):
        summary = json.loads((run / "summary.json").read_text())
        by_rid = request_events(run)
        kwargs = {k: v for k, v in summary["server_kwargs"].items() if k != "model_path"}
        print(f"\n## {run.name}  model={summary['model']}  plugin={summary['plugin']}  "
              f"kwargs={kwargs}  load={summary['load_s']:.0f}s")
        walls = {}
        for row in summary["rows"]:
            steps = [e for e in by_rid.get(row["request_id"], []) if e.get("event") == "step"]
            ran = [e for e in steps if e.get("action") == "ran"]
            calls = sum(e.get("model_calls_this_step", 0) for e in ran)
            # The flow-match scheduler picks its sigma interval from its own
            # counter, not from the timestep it is handed, so the counter must
            # equal the loop's index on entry (None before its first update).
            desync = [e["step_index"] for e in steps
                      if e["scheduler_index_before"] not in (None, e["step_index"])]
            sched = [e for e in by_rid.get(row["request_id"], []) if e.get("event") == "schedule"]
            cmp = compare(pixels(row["image"]), ref)
            walls.setdefault(row["name"], []).append(row["wall_s"])
            n_ts = steps[0]["total_steps"] if steps else "-"
            print(f"  {row['request_id']:<16} mode={row['mode']:<8} k={row['k']!s:<4} "
                  f"steps_seen={len(steps):>3} ran={len(ran):>3} model_calls={calls:>3} "
                  f"len(timesteps)={n_ts!s:>3} desync_at={desync[:3] or '-'} "
                  f"sched_removed={[s['removed'] for s in sched] or '-'} "
                  f"wall={row['wall_s']:.2f}s  vs_ref={cmp}")
        for name, w in walls.items():
            if len(w) > 1:
                print(f"  timing {name}: median={statistics.median(w):.3f}s min={min(w):.3f}s "
                      f"max={max(w):.3f}s n={len(w)}")


if __name__ == "__main__":
    main()
