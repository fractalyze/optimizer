"""CPU check of the probe's control logic against fake stage objects.

Runs without SGLang or a GPU, so a probe bug is caught before it can be
mistaken for an engine finding.  python -m pytest test_plugin.py
"""

import json
from types import SimpleNamespace

from opt_step_probe import plugin


def _run(tmp_path, monkeypatch, policy, n=5):
    control = tmp_path / "control.json"
    control.write_text(json.dumps({"r": policy}))
    monkeypatch.setenv("OPT_PROBE_CONTROL", str(control))
    monkeypatch.delenv("OPT_PROBE_LOG", raising=False)
    batch = SimpleNamespace(request_id="r", extra={})
    calls, used = [], []

    def predict(self, **kw):
        calls.append(kw["timestep_index"])
        return f"pred{kw['timestep_index']}"

    def step_fn(self, ctx, step, batch, server_args):
        used.append(plugin._around_predict(predict, None, batch=batch, timestep_index=step.step_index))

    ctx = SimpleNamespace(is_warmup=False, timesteps=list(range(n)), num_inference_steps=n, scheduler=None)
    server = SimpleNamespace(model_path="m")
    for i in range(n):
        plugin._around_step(step_fn, None, ctx, SimpleNamespace(step_index=i, t_host=float(i)), batch, server)
    return calls, used


def test_observe_changes_nothing(tmp_path, monkeypatch):
    calls, used = _run(tmp_path, monkeypatch, {"mode": "observe"})
    assert calls == [0, 1, 2, 3, 4] and used == ["pred0", "pred1", "pred2", "pred3", "pred4"]


def test_override_reuses_previous_prediction_without_a_model_call(tmp_path, monkeypatch):
    calls, used = _run(tmp_path, monkeypatch, {"mode": "override", "k": 2})
    assert calls == [0, 1, 3, 4] and used[2] == "pred1"


def test_bypass_skips_the_whole_step(tmp_path, monkeypatch):
    calls, used = _run(tmp_path, monkeypatch, {"mode": "bypass", "k": 2})
    assert calls == [0, 1, 3, 4] and len(used) == 4


def test_each_step_reports_its_graph_replays(tmp_path, monkeypatch):
    log = tmp_path / "probe.jsonl"
    control = tmp_path / "control.json"
    control.write_text("{}")
    monkeypatch.setenv("OPT_PROBE_CONTROL", str(control))
    monkeypatch.setenv("OPT_PROBE_LOG", str(log))
    batch = SimpleNamespace(request_id="r", extra={})
    ctx = SimpleNamespace(is_warmup=False, timesteps=[0, 1, 2], num_inference_steps=3, scheduler=None)

    def step_fn(self, ctx, step, batch, server_args):
        if step.step_index > 0:  # mirrors Qwen-Image-2.1: an eager prefill, then graph replays
            plugin._around_replay(lambda runner: None, None)

    for i in range(3):
        plugin._around_step(step_fn, None, ctx, SimpleNamespace(step_index=i, t_host=float(i)), batch,
                            SimpleNamespace(model_path="m"))
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["graph_replays_this_step"] for r in records] == [0, 1, 1]


def test_unlisted_request_is_only_observed(tmp_path, monkeypatch):
    monkeypatch.setenv("OPT_PROBE_CONTROL", str(tmp_path / "missing.json"))
    assert plugin._policy("anything") == {"mode": "observe"}
