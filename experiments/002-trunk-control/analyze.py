"""SPIKE: summarize experiment 002 runs from what the worker actually logged.

    python analyze.py REF_RUN_DIR RUN_DIR [RUN_DIR ...]

Each image is compared with the image of the same request name in REF_RUN_DIR
(falling back to REF_RUN_DIR's observe-0), exact equality first, then max
absolute difference and PSNR on 8-bit pixels. Trunk numbers come from the
worker's log, never from the requested configuration.
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
    d = np.abs(a - b)
    mse = float((d.astype(np.float64) ** 2).mean())
    if mse == 0:
        return "identical"
    return f"max_abs={int(d.max())} psnr={10 * np.log10(255**2 / mse):.2f}dB"


def main():
    ref_dir = pathlib.Path(sys.argv[1])
    ref_rows = {r["request_id"]: r["image"] for r in json.loads((ref_dir / "summary.json").read_text())["rows"]}
    for run in map(pathlib.Path, sys.argv[2:]):
        summary = json.loads((run / "summary.json").read_text())
        log = run / "probe.jsonl"
        events = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
        trunks = [e for e in events if e.get("event") == "trunk"]
        warm = [e for e in trunks if e["warmup"]]
        kwargs = {k: v for k, v in summary["server_kwargs"].items() if k != "model_path"}
        print(f"\n## {run.name}  plugin={summary['plugin']}  kwargs={kwargs}  "
              f"warmup trunk calls={len(warm)}")
        walls = {}
        for row in summary["rows"]:
            mine = [e for e in trunks if e["request_id"] == row["request_id"]]
            run_ = sum(e["blocks_run"] for e in mine)
            skipped = sum(e["blocks_skipped"] for e in mine)
            reused = [(e["step_index"], e["branch"]) for e in mine if e.get("decision") == "reuse"]
            exact = [e.get("identity_exact") for e in mine if "identity_exact" in e]
            closed = [e["step_index"] for e in mine if not e["overridable"]]
            rel = [e["signal_rel_l1"] for e in mine if e.get("signal_rel_l1") is not None]
            ref = ref_rows.get(row["request_id"]) or ref_rows["observe-0"]
            walls.setdefault(row["name"], []).append(row["wall_s"])
            print(f"  {row['request_id']:<20} trunk_calls={len(mine):>3} blocks_run={run_:>5} "
                  f"blocks_skipped={skipped:>3} reused_at={reused or '-'} "
                  f"identity_exact={('all' if exact and all(exact) else exact.count(False)) if exact else '-'} "
                  f"not_overridable_steps={closed or '-'} "
                  f"signal_rel_l1[min/med/max]={'%.4f/%.4f/%.4f' % (min(rel), statistics.median(rel), max(rel)) if rel else '-'} "
                  f"wall={row['wall_s']:.2f}s vs_ref={compare(pixels(row['image']), pixels(ref))}")
            if mine:
                first = mine[0]
                skipped_ms = [e["dit_ms"] for e in mine if e["blocks_skipped"]]
                print(f"      entry={first['entry']} signal={first['signal']} "
                      f"dit_ms[med]={statistics.median(e['dit_ms'] for e in mine):.1f}"
                      + (f" dit_ms[trunk skipped]={skipped_ms}" if skipped_ms else ""))
        for name, w in walls.items():
            if len(w) > 1:
                print(f"  timing {name}: median={statistics.median(w):.3f}s n={len(w)}")


if __name__ == "__main__":
    main()
