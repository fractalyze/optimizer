"""CPU check of the batch probe's contracts against a fake model and Binding.

Runs without SGLang or a GPU, so a probe bug is caught before it can be
mistaken for an engine finding. Needs torch and experiment 002's probe
installed.  python -m pytest test_probe.py
"""

import json
from types import SimpleNamespace

import torch

from opt_batch_probe import adapter


class FakeBinding:
    """Two blocks that each double the image stream; the exit is `norm`. With a
    non-additive block the residual depends on the entry, so a payload applied
    to the wrong row would show."""

    name = "fake"

    @staticmethod
    def exit_module(model):
        return model.norm

    @staticmethod
    def overridable(dit_kwargs):
        return True

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
        return x * 2

    def norm(self, x):
        return x

    hooked_block = adapter.around_block(FakeBinding, "block")
    hooked_exit = adapter.around_exit(FakeBinding)
    model = SimpleNamespace(norm=SimpleNamespace())

    def forward(self, x, **kwargs):
        for _ in range(2):
            x = hooked_block(block, None, x)
        return hooked_exit(norm, model.norm, x)

    hooked_dit = adapter.around_dit(FakeBinding)
    return lambda x, **kw: hooked_dit(forward, model, x, **kw), calls


def _run(tmp_path, monkeypatch, policies, steps=3, seeds=None):
    """`policies` maps each row's seed to its policy; rows run in that order.
    Row r enters step s with value 10*s + 1 + 100*r."""
    control = tmp_path / "control.json"
    control.write_text(json.dumps({str(k): v for k, v in policies.items()}))
    log = tmp_path / "log.jsonl"
    log.unlink(missing_ok=True)
    monkeypatch.setenv("OPT_BATCH_CONTROL", str(control))
    monkeypatch.setenv("OPT_BATCH_LOG", str(log))
    rows = len(policies)
    batch = SimpleNamespace(request_id="dynamic_batch::a", extra={}, num_inference_steps=steps,
                            is_warmup=False, is_cfg_negative=False,
                            seeds=list(policies) if seeds is None else seeds)
    dit, calls = _model()
    outs = []
    for step in range(steps):
        monkeypatch.setattr(adapter, "_forward_context",
                            lambda s=step: SimpleNamespace(forward_batch=batch, current_timestep=s))
        x = torch.tensor([[10.0 * step + 1 + 100 * r] for r in range(rows)])
        outs.append(dit(x)[:, 0].tolist())
    records = [json.loads(line) for line in log.read_text().splitlines()]
    return outs, records, calls


OBSERVE = {"mode": "observe"}
REUSE = {"mode": "reuse", "k": 1}


def test_observe_gives_every_row_its_own_item_and_changes_nothing(tmp_path, monkeypatch):
    outs, records, calls = _run(tmp_path, monkeypatch, {7: OBSERVE, 9: OBSERVE})
    assert outs == [[4, 404], [44, 444], [84, 484]]
    assert calls["blocks"] == 6
    assert [(r["key"], r["row"], r["rows"]) for r in records[:2]] == [("7", 0, 2), ("9", 1, 2)]


def test_one_row_reusing_still_runs_the_blocks_and_leaves_the_other_row_alone(tmp_path, monkeypatch):
    outs, records, calls = _run(tmp_path, monkeypatch, {7: REUSE, 9: OBSERVE})
    # The reusing row rebuilds from its own entry (21) and its own stored
    # residual (44 - 11); the other row computes as if it were alone.
    assert outs[2] == [54, 484]
    reused = [r for r in records if r["step_index"] == 2]
    assert [r["call_decision"] for r in reused] == ["splice", "splice"]
    assert [r["blocks_run"] for r in reused] == [2, 2]
    assert calls["blocks"] == 6


def test_blocks_are_skipped_only_when_every_row_reuses_and_each_keeps_its_payload(tmp_path, monkeypatch):
    outs, records, calls = _run(tmp_path, monkeypatch, {7: REUSE, 9: REUSE})
    # The second row's residual is 444 - 111; swapped payloads would give 354 / 154.
    assert outs[2] == [54, 454]
    assert [r["blocks_skipped"] for r in records if r["step_index"] == 2] == [2, 2]
    assert calls["blocks"] == 4


def test_identity_on_one_row_is_exact(tmp_path, monkeypatch):
    outs, records, _ = _run(tmp_path, monkeypatch, {7: {"mode": "identity_residual"}, 9: OBSERVE})
    assert outs == [[4, 404], [44, 444], [84, 484]]
    assert all(r["identity_exact"] for r in records if r["key"] == "7")
    assert not any("identity_exact" in r for r in records if r["key"] == "9")


def test_rows_that_cannot_be_attributed_are_never_overridden(tmp_path, monkeypatch):
    # Two rows but one seed, e.g. CFG branches stacked as rows of one request.
    outs, records, calls = _run(tmp_path, monkeypatch, {7: REUSE, 9: REUSE}, seeds=[7])
    assert calls["blocks"] == 6
    assert all(r["key"] is None and not r["overridable"] for r in records)
