"""SPIKE: drive one SGLang engine configuration through rounds of requests (exp 003).

One invocation = one engine configuration, loaded once. A round is a set of
requests submitted at the same moment from separate threads, so SGLang's
scheduler sees them together and may merge them into one DiT call; a round of
one is a solo request. Rounds run one after another.

Each item is keyed by its seed: a merged batch keeps no per-request id, and
the seed is the per-row identity that survives (see opt_batch_probe/adapter.py).
Seeds must therefore be unique within a round.

Machine-specific settings (GPU, weight cache, CUDA toolkit) come from the
environment of whoever runs this, never from this file.

    python run.py --model M --out OUT --plan plan.json [--no-plugin] \
        [--server-kwarg batching_max_size=2 ...]

plan.json: {"defaults": {sampling kwargs},
            "items": {"a": {"prompt": ..., "seed": 101}, ...},
            "rounds": [{"name": "batch-reuse", "repeat": 1,
                        "items": [{"item": "a", "mode": "reuse", "k": 20}, ...]}]}
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import threading
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


def _submit_together(gen, params_list):
    """Call generate() once per params from its own thread, released together."""
    start = threading.Barrier(len(params_list))
    results = [None] * len(params_list)

    def worker(i, params):
        start.wait()
        t = time.perf_counter()
        try:
            out = gen.generate(params)
            results[i] = (out, time.perf_counter() - t, None)
        except Exception as e:  # reported per item, never swallowed
            results[i] = (None, time.perf_counter() - t, e)

    threads = [threading.Thread(target=worker, args=(i, p)) for i, p in enumerate(params_list)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return results


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
    control_path.write_text("{}")

    os.environ["SGLANG_PLUGINS"] = "__none__" if args.no_plugin else "opt_batch_probe"
    os.environ["OPT_BATCH_CONTROL"] = str(control_path)
    os.environ["OPT_BATCH_LOG"] = str(log_path)

    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator

    server_kwargs = dict(model_path=args.model, **_kv(args.server_kwarg))
    t0 = time.perf_counter()
    gen = DiffGenerator.from_pretrained(**server_kwargs)
    load_s = time.perf_counter() - t0

    rows = []
    try:
        for rnd in plan["rounds"]:
            for rep in range(rnd.get("repeat", 1)):
                round_id = f"{rnd['name']}-{rep}"
                specs = [dict(plan["items"][it["item"]], **it) for it in rnd["items"]]
                seeds = [s["seed"] for s in specs]
                if len(set(seeds)) != len(seeds):
                    raise ValueError(f"{round_id}: seeds must be unique within a round")
                # Written before submission; the worker reads it when it first
                # meets each seed in this round's DiT calls.
                control_path.write_text(json.dumps(
                    {str(s["seed"]): {"mode": s["mode"], "k": s.get("k"), "tag": f"{round_id}.{s['item']}"}
                     for s in specs}))
                params_list = []
                for s in specs:
                    rid = f"{round_id}.{s['item']}"
                    params = dict(plan["defaults"], prompt=s["prompt"], seed=s["seed"])
                    params.update(request_id=rid, output_path=str(out / "images"), output_file_name=rid)
                    params_list.append(params)
                for s, params, (result, wall, err) in zip(
                        specs, params_list, _submit_together(gen, params_list)):
                    if err is not None:
                        raise RuntimeError(f"{params['request_id']}: {err!r}") from err
                    results = result if isinstance(result, list) else [result]
                    path = results[0].output_file_path if results and results[0] else None
                    if path is None:
                        raise RuntimeError(f"{params['request_id']}: generate() produced no image")
                    rows.append(dict(round=round_id, request_id=params["request_id"],
                                     item=s["item"], seed=s["seed"], mode=s["mode"], k=s.get("k"),
                                     round_size=len(specs), wall_s=wall, image=path))
                    print(f"{params['request_id']}: {wall:.3f}s", flush=True)
    finally:
        gen.shutdown()

    events = [json.loads(line) for line in log_path.read_text().splitlines()] if log_path.exists() else []
    witnessed = {e["key"] for e in events if e.get("event") == "trunk" and not e["warmup"]}
    if not args.no_plugin:
        missing = sorted({str(r["seed"]) for r in rows} - witnessed)
        if missing:
            raise RuntimeError(f"plugin never saw rows for seeds {missing}")
    elif events:
        raise RuntimeError("plugin ran although it was disabled")

    summary = dict(model=args.model, plugin=not args.no_plugin, server_kwargs=server_kwargs,
                   load_s=load_s, defaults=plan["defaults"], rows=rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
