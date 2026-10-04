"""SPIKE: drive one vLLM-Omni engine configuration through rounds of requests (exp 004).

One invocation = one engine configuration, loaded once. A round is a set of
requests, each submitted after its own delay, so the scheduler can put them in
one DiT call: at the same step (request mode) or at different steps (step
mode, when a request joins while another is running). Rounds run one after
another.

Each item's policy is keyed by its request id, which vLLM-Omni keeps per row
at the runner (see opt_omni_probe/adapter.py).

Machine-specific settings (GPU, weight cache) come from the environment of
whoever runs this, never from this file.

    python run.py --model M --out OUT --plan plan.json [--no-probe] \
        [--engine-kwarg max_num_seqs=2 ...]

plan.json: {"defaults": {sampling kwargs},
            "items": {"a": {"prompt": ..., "seed": 101}, ...},
            "rounds": [{"name": "step-a-reuse",
                        "items": [{"item": "a", "mode": "reuse", "k": 2},
                                  {"item": "b", "mode": "observe", "after": {"item": "a", "step": 2}}]}]}
"""

from __future__ import annotations

import argparse
import asyncio
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


async def _wait_for_step(log_path, key, step):
    """Until the probe has logged `key` reaching `step` (a deterministic join)."""
    while True:
        if log_path.exists():
            for line in log_path.read_text().splitlines():
                e = json.loads(line)
                if e.get("event") == "trunk" and e.get("key") == key and e["step_index"] >= step:
                    return
        await asyncio.sleep(0.05)


async def _one(engine, params_cls, rid, spec, defaults, out_dir, log_path):
    await asyncio.sleep(spec.get("delay_s", 0))
    if "after" in spec:  # {"item": ..., "step": ...}: join once that item passed that step
        await _wait_for_step(log_path, rid.rpartition(".")[0] + "." + spec["after"]["item"],
                             spec["after"]["step"])
    params = params_cls(**dict(defaults, seed=spec["seed"]))
    t0 = time.perf_counter()
    last = None
    async for output in engine.generate(prompt={"prompt": spec["prompt"]}, request_id=rid,
                                        sampling_params_list=[params], output_modalities=["image"]):
        last = output
    if last is None or not last.images:
        raise RuntimeError(f"{rid}: no image")
    path = out_dir / f"{rid}.png"
    last.images[0].save(path)
    return dict(request_id=rid, submitted_after_s=spec.get("delay_s", 0),
                wall_s=time.perf_counter() - t0, image=str(path))


async def _main(args) -> int:
    out = args.out.resolve()
    (out / "images").mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    control_path, log_path = out / "control.json", out / "probe.jsonl"
    log_path.unlink(missing_ok=True)
    control_path.write_text("{}")
    os.environ["OPT_OMNI_PROBE"] = "0" if args.no_probe else "1"
    os.environ["OPT_OMNI_CONTROL"] = str(control_path)
    os.environ["OPT_OMNI_LOG"] = str(log_path)
    if not args.no_probe:
        # The single-GPU engine runs its worker in this process; register
        # before the model is built (the plugin entry point would also do it).
        from opt_omni_probe.plugin import register
        register()

    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    engine_kwargs = dict(model=args.model, **_kv(args.engine_kwarg))
    t0 = time.perf_counter()
    engine = AsyncOmni(**engine_kwargs)
    load_s = time.perf_counter() - t0

    rows = []
    try:
        for rnd in plan["rounds"]:
            specs = {f"{rnd['name']}.{it['item']}": dict(plan["items"][it["item"]], **it)
                     for it in rnd["items"]}
            # Written before submission; the worker reads it when it first
            # meets each request id.
            control_path.write_text(json.dumps(
                {rid: {"mode": s["mode"], "k": s.get("k")} for rid, s in specs.items()}))
            results = await asyncio.gather(*(
                _one(engine, OmniDiffusionSamplingParams, rid, s, plan["defaults"], out / "images",
                     log_path)
                for rid, s in specs.items()))
            for res in results:
                spec = specs[res["request_id"]]
                rows.append(dict(res, round=rnd["name"], item=spec["item"], mode=spec["mode"],
                                 k=spec.get("k"), round_size=len(specs)))
                print(f"{res['request_id']}: {res['wall_s']:.2f}s", flush=True)
    finally:
        engine.shutdown()

    events = [json.loads(x) for x in log_path.read_text().splitlines()] if log_path.exists() else []
    trunks = [e for e in events if e.get("event") == "trunk"]
    if not args.no_probe:
        missing = sorted({r["request_id"] for r in rows} - {e["key"] for e in trunks})
        if missing:
            raise RuntimeError(f"probe never saw rows for {missing}")
    elif trunks:
        raise RuntimeError("probe ran although it was disabled")

    summary = dict(model=args.model, probe=not args.no_probe, engine_kwargs=engine_kwargs,
                   load_s=load_s, defaults=plan["defaults"], rows=rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--plan", required=True, type=pathlib.Path)
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--engine-kwarg", action="append", default=[])
    return asyncio.run(_main(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
