"""SPIKE: one engine configuration through the optimizer path (exp 005).

    python run.py --model M --out OUT --plan plan.json [--technique T]
        [--skip-feasibility] [--env K=V ...] [--server-kwarg k=v ...] [--trunk]

Without --technique this is the baseline: same engine, same requests, no FP8.
With it, optimizer_path decides the implementation, its feasibility and the
native settings; the run then ends in a status computed from the evidence the
engine produced, never from the configuration. --skip-feasibility exists only
to show what engagement verification does when feasibility missed something.
--trunk also loads experiment 002's trunk probe, to test FP8 with it.

Machine-specific settings (GPU, weight cache, CUDA toolkit) come from the
environment of whoever runs this, never from this file.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import optimizer_path as op  # noqa: E402


def _kv(pairs):
    out = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def _target():
    import torch

    try:
        from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config  # noqa: F401
        caps = frozenset({"engine_feature.fp8_w8a8_dynamic_linear"})
    except ImportError:
        caps = frozenset()
    cuda = tuple(int(x) for x in torch.version.cuda.split(".")[:2])
    return op.Target(engine="sglang", gpu_capability=torch.cuda.get_device_capability(0),
                     cuda_version=cuda, capabilities=caps,
                     env_overrides={k: v for k, v in os.environ.items() if k.startswith("SGLANG_")})


def _evidence(events):
    inv = [e for e in events if e["event"] == "inventory"]
    layers = {}
    for e in inv:
        for k, v in e["layers"].items():
            layers[k] = layers.get(k, 0) + v
    reqs = [e for e in events if e["event"] == "request" and not e["warmup"]]
    count = lambda k: sum(e["counts"].get(k, 0) for e in reqs)  # noqa: E731
    quantizable = sum(v for k, v in layers.items() if k != "not_quantizable")
    return op.Evidence(
        quantizable=quantizable, fp8_w8a8=layers.get("fp8_w8a8", 0),
        fp8_weight_only=layers.get("fp8_weight_only", 0),
        unquantized=layers.get("unquantized", 0), not_quantizable=layers.get("not_quantizable", 0),
        runtime_w8a8_gemms=count("apply_w8a8"), runtime_weight_only_gemms=count("apply_weight_only"),
        runtime_activation_quants=count("act_quant_per_token")), inv, reqs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--plan", required=True, type=pathlib.Path)
    ap.add_argument("--technique")
    ap.add_argument("--skip-feasibility", action="store_true")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--server-kwarg", action="append", default=[])
    ap.add_argument("--trunk", action="store_true")
    args = ap.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    log_path = out / "evidence.jsonl"
    log_path.unlink(missing_ok=True)
    os.environ.update({k: str(v) for k, v in _kv(args.env).items()})
    summary = dict(model=args.model, technique=args.technique, env=_kv(args.env))

    impl, settings = None, {}
    if args.technique:
        target = _target()
        impl, settings, status, why = op.plan(args.technique, target, op.SGLangAdapter(),
                                              skip_feasibility=args.skip_feasibility)
        summary.update(implementation=impl and impl.id, target=dict(
            gpu_capability=target.gpu_capability, cuda=target.cuda_version), feasibility=why)
        if status:  # UNSUPPORTED or INFEASIBLE: nothing is launched
            summary.update(status=status, reason=why)
            (out / "summary.json").write_text(json.dumps(summary, indent=1))
            print(status, why)
            return 0

    plugins = ["opt_fp8_probe"] + (["opt_trunk_probe"] if args.trunk else [])
    os.environ["SGLANG_PLUGINS"] = ",".join(plugins)
    os.environ["OPT_FP8_LOG"] = str(log_path)
    if args.trunk:
        control = out / "trunk_control.json"
        control.write_text(json.dumps({r["name"]: r["trunk"] for r in plan["requests"] if "trunk" in r}))
        os.environ["OPT_TRUNK_CONTROL"] = str(control)
        os.environ["OPT_TRUNK_LOG"] = str(out / "trunk.jsonl")
        (out / "trunk.jsonl").unlink(missing_ok=True)

    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator

    server_kwargs = dict(model_path=args.model, **_kv(args.server_kwarg), **settings)
    summary["server_kwargs"] = server_kwargs
    rows = []
    try:
        t0 = time.perf_counter()
        gen = DiffGenerator.from_pretrained(**server_kwargs)
        summary["load_s"] = time.perf_counter() - t0
        try:
            for req in plan["requests"]:
                params = dict(plan["defaults"], request_id=req["name"],
                              output_path=str(out / "images"), output_file_name=req["name"])
                t = time.perf_counter()
                result = gen.generate(params)
                results = result if isinstance(result, list) else [result]
                rows.append(dict(name=req["name"], wall_s=time.perf_counter() - t,
                                 image=results[0].output_file_path))
                print(f"{req['name']}: {rows[-1]['wall_s']:.3f}s", flush=True)
        finally:
            gen.shutdown()
    except Exception as e:  # recorded as a status, then re-raised for the log
        summary.update(status=op.RUNTIME_ERROR, reason=repr(e), rows=rows)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        raise

    events = [json.loads(x) for x in log_path.read_text().splitlines()] if log_path.exists() else []
    evidence, inventories, requests = _evidence(events)
    summary.update(rows=rows, evidence=evidence.__dict__, inventories=inventories,
                   requests=requests)
    if impl:
        status, why = op.verdict(impl, evidence)
        summary.update(status=status, reason=why)
        print(status, why)
    else:
        summary.update(status="BASELINE", reason="no technique selected")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
