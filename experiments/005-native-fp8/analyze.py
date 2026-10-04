"""SPIKE: summarize experiment 005 runs from what the engine actually logged.

    python analyze.py REF_RUN_DIR RUN_DIR [RUN_DIR ...]

For each run: the status and reason run.py computed, the layer inventory and
runtime counts it was computed from, then per request the denoise time and peak
memory the probe measured and the image compared with REF_RUN_DIR's image of
the same request name (falling back to its r0): exact equality first, then max
absolute difference and PSNR on 8-bit pixels. A run with trunk.jsonl (FP8 with
experiment 002's trunk probe) also compares its identity image with its own
observe image and sums the trunk blocks it skipped.
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


def jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def main():
    ref_dir = pathlib.Path(sys.argv[1])
    ref_rows = {r["name"]: r["image"] for r in json.loads((ref_dir / "summary.json").read_text())["rows"]}
    for run in map(pathlib.Path, sys.argv[2:]):
        s = json.loads((run / "summary.json").read_text())
        print(f"\n## {run.name}  status={s.get('status')}  impl={s.get('implementation', '-')}  "
              f"env={s.get('env') or '-'}")
        print(f"  reason: {s.get('reason')}")
        if "rows" not in s or not s["rows"]:
            continue  # nothing was launched, or it failed before a request finished
        ev = s.get("evidence", {})
        print(f"  evidence: {ev}")
        for inv in s.get("inventories", []):
            print(f"  inventory {inv['model']}: {inv['layers']} weight_scales={inv.get('weight_scales')}")
        print(f"  load_s={s.get('load_s', 0):.1f}")
        reqs = {r["request_id"]: r for r in s.get("requests", [])}
        rows = {r["name"]: r for r in s["rows"]}
        for name, row in rows.items():
            req = reqs.get(name, {})
            ref = ref_rows.get(name) or ref_rows["r0"]
            print(f"  {name:<10} wall={row['wall_s']:.2f}s denoise={req.get('denoise_s', float('nan')):.2f}s "
                  f"peak_alloc={req.get('peak_allocated_gib', float('nan')):.2f}GiB "
                  f"counts={req.get('counts', {})} vs_ref={compare(pixels(row['image']), pixels(ref))}")
        warm = [rows[n]["wall_s"] for n in rows if n not in ("r0", "observe")]
        if len(warm) > 1:
            print(f"  timing after first request: median wall={statistics.median(warm):.3f}s n={len(warm)}")
        trunk = [e for e in jsonl(run / "trunk.jsonl") if e.get("event") == "trunk" and not e.get("warmup")]
        if trunk:
            for name in rows:
                mine = [e for e in trunk if e.get("request_id") == name]
                print(f"  trunk {name}: calls={len(mine)} blocks_skipped={sum(e['blocks_skipped'] for e in mine)}")
            if "identity" in rows and "observe" in rows:
                print(f"  identity vs observe (same run): "
                      f"{compare(pixels(rows['identity']['image']), pixels(rows['observe']['image']))}")


if __name__ == "__main__":
    main()
