"""SPIKE: drive one SGLang engine configuration through a list of requests.

One invocation = one engine configuration (model, plugin on/off, compile or
graph capture), loaded once. Each request carries its own probe policy, so
request-local control is exercised inside one process.

Machine-specific settings (GPU, weight cache, CUDA toolkit) come from the
environment of whoever runs this, never from this file.

    python run.py --model Qwen/Qwen-Image-2.1 --out OUT --plan plan.json \
        [--no-plugin] [--server-kwarg enable_torch_compile=true]

plan.json: {"defaults": {sampling kwargs}, "requests":
            [{"name": "observe", "mode": "observe", "k": null, "repeat": 3}, ...]}
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time


def _kv(pairs):
    out = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--plan", required=True, type=pathlib.Path)
    ap.add_argument("--no-plugin", action="store_true")
    ap.add_argument("--server-kwarg", action="append", default=[])
    args = ap.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    control_path, log_path = out / "control.json", out / "probe.jsonl"
    log_path.unlink(missing_ok=True)

    # Every request id and its policy are fixed before the engine starts, so
    # the spawned worker reads the same table for the whole run.
    schedule, control = [], {}
    for req in plan["requests"]:
        for i in range(req.get("repeat", 1)):
            rid = f"{req['name']}-{i}"
            schedule.append((rid, req))
            control[rid] = {"mode": req["mode"], "k": req.get("k")}
    control_path.write_text(json.dumps(control, indent=1))

    # Allow-list our plugin by name, or name something that does not exist so
    # SGLang skips every third-party plugin.
    os.environ["SGLANG_PLUGINS"] = "__none__" if args.no_plugin else "opt_step_probe"
    os.environ["OPT_PROBE_CONTROL"] = str(control_path)
    os.environ["OPT_PROBE_LOG"] = str(log_path)

    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator

    server_kwargs = dict(model_path=args.model, **_kv(args.server_kwarg))
    t0 = time.perf_counter()
    gen = DiffGenerator.from_pretrained(**server_kwargs)
    load_s = time.perf_counter() - t0

    rows = []
    try:
        for rid, req in schedule:
            params = dict(plan["defaults"], **req.get("sampling", {}))
            params.update(request_id=rid, output_path=str(out / "images"), output_file_name=rid)
            t = time.perf_counter()
            result = gen.generate(params)
            wall = time.perf_counter() - t
            results = result if isinstance(result, list) else [result]
            path = results[0].output_file_path if results and results[0] else None
            if path is None:
                raise RuntimeError(f"{rid}: generate() produced no image ({result!r})")
            rows.append(dict(request_id=rid, name=req["name"], mode=req["mode"],
                             k=req.get("k"), wall_s=wall, image=path,
                             engine_time_s=getattr(results[0], "generation_time", None)))
            print(f"{rid}: {wall:.3f}s", flush=True)
    finally:
        gen.shutdown()

    events = [json.loads(line) for line in log_path.read_text().splitlines()] if log_path.exists() else []
    witnessed = {e["request_id"] for e in events if e.get("event") == "step"}
    if not args.no_plugin:
        missing = [r["request_id"] for r in rows if r["request_id"] not in witnessed]
        if missing:
            raise RuntimeError(f"plugin never ran inside the worker for {missing}")
    elif events:
        raise RuntimeError("plugin ran although it was disabled")

    summary = dict(model=args.model, plugin=not args.no_plugin, server_kwargs=server_kwargs,
                   load_s=load_s, defaults=plan["defaults"], rows=rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
