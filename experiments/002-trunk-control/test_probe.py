"""CPU check of the trunk probe's contracts against a fake model and Binding.

Runs without SGLang or a GPU, so a probe bug is caught before it can be
mistaken for an engine finding. Needs torch.  python -m pytest test_probe.py
"""

import json
from types import SimpleNamespace

import pytest
import torch

from opt_trunk_probe import adapter


class FakeBinding:
    """Two blocks that each add 1 to the image stream; the exit is `norm`."""

    name = "fake"

    @staticmethod
    def exit_module(model):
        return model.norm

    @staticmethod
    def overridable(dit_kwargs):
        return not dit_kwargs.get("prefill", False)

    @staticmethod
    def entry(kind, args, kwargs):
        return args[0]

    @staticmethod
    def signal(block, kind, args, kwargs):
        return args[0].float()

    @staticmethod
    def identity(kind, args, kwargs):
        return args[0]


def _model():
    calls = {"blocks": 0}

    def block(self, x):
        calls["blocks"] += 1
        return x + 1

    def norm(self, x):
        return x

    hooked_block = adapter.around_block(FakeBinding, "block")
    hooked_exit = adapter.around_exit(FakeBinding)
    model = SimpleNamespace()
    model.norm = SimpleNamespace()

    def forward(self, x, **kwargs):
        for _ in range(2):
            x = hooked_block(block, None, x)
        return hooked_exit(norm, model.norm, x)

    hooked_dit = adapter.around_dit(FakeBinding)
    return lambda x, **kw: hooked_dit(forward, model, x, **kw), calls


def _run(tmp_path, monkeypatch, policy, steps=4, branches=("cond",), prefill_step=None):
    control = tmp_path / "control.json"
    control.write_text(json.dumps({"r": policy}))
    log = tmp_path / "log.jsonl"
    monkeypatch.setenv("OPT_TRUNK_CONTROL", str(control))
    monkeypatch.setenv("OPT_TRUNK_LOG", str(log))
    batch = SimpleNamespace(request_id="r", extra={}, num_inference_steps=steps, is_warmup=False)
    dit, calls = _model()
    outs = []
    for step in range(steps):
        for branch in branches:
            batch.is_cfg_negative = branch == "uncond"
            monkeypatch.setattr(adapter, "_forward_context",
                                lambda s=step: SimpleNamespace(forward_batch=batch, current_timestep=s))
            x = torch.full((1, 3), float(step * 10), dtype=torch.bfloat16)
            outs.append(dit(x, prefill=(step == prefill_step)))
    records = [json.loads(line) for line in log.read_text().splitlines()]
    return outs, records, calls


def test_observe_runs_every_block_and_changes_nothing(tmp_path, monkeypatch):
    outs, records, calls = _run(tmp_path, monkeypatch, {"mode": "observe"})
    assert calls["blocks"] == 8
    assert [o[0, 0].item() for o in outs] == [2, 12, 22, 32]
    assert all(r["blocks_skipped"] == 0 for r in records)


@pytest.mark.parametrize("mode", ["identity_output", "identity_residual"])
def test_identity_override_is_exact(tmp_path, monkeypatch, mode):
    _, records, _ = _run(tmp_path, monkeypatch, {"mode": mode})
    assert all(r["identity_exact"] for r in records)


def test_residual_roundtrip_is_exact_in_bf16():
    entry = torch.randn(4, 64, 32).to(torch.bfloat16)
    out = (entry.float() * 1.7 + torch.randn(4, 64, 32)).to(torch.bfloat16)
    rebuilt = adapter.apply(entry, adapter.capture("residual", entry, out))
    assert torch.equal(rebuilt, out)


def test_reuse_skips_only_the_step_after_k_and_rebuilds_from_its_own_entry(tmp_path, monkeypatch):
    outs, records, calls = _run(tmp_path, monkeypatch, {"mode": "reuse", "k": 1})
    assert [r["blocks_run"] for r in records] == [2, 2, 0, 2]
    assert records[2]["decision"] == "reuse"
    # The reused residual (+2) is added to the skipped call's own entry (20),
    # not to the entry it was captured from.
    assert outs[2][0, 0].item() == 22


def test_each_cfg_branch_keeps_its_own_payload(tmp_path, monkeypatch):
    _, records, _ = _run(tmp_path, monkeypatch, {"mode": "reuse", "k": 1}, branches=("cond", "uncond"))
    skipped = [(r["step_index"], r["branch"]) for r in records if r["blocks_run"] == 0]
    assert skipped == [(2, "cond"), (2, "uncond")]


def test_a_non_overridable_call_always_computes(tmp_path, monkeypatch):
    _, records, _ = _run(tmp_path, monkeypatch, {"mode": "reuse", "k": 1}, prefill_step=2)
    assert records[2]["blocks_run"] == 2 and not records[2]["overridable"]
