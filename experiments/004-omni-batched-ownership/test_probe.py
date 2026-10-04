"""CPU check of the vLLM-Omni probe's contracts against a fake model and Binding.

Runs without vLLM-Omni or a GPU. Needs torch and experiment 002's probe
installed.  python -m pytest test_probe.py
"""

import json
from types import SimpleNamespace

import pytest
import torch

from opt_omni_probe import adapter


class FakeBinding:
    """Two blocks that each double the image stream; the exit is `norm`."""

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


@pytest.fixture
def dit(tmp_path, monkeypatch):
    monkeypatch.setenv("OPT_OMNI_LOG", str(tmp_path / "log.jsonl"))
    monkeypatch.setattr(adapter, "_owners", {})
    calls = {"blocks": 0}

    def block(self, x):
        calls["blocks"] += 1
        return x * 2

    hooked_block = adapter.around_block(FakeBinding, "block")
    hooked_exit = adapter.around_exit(FakeBinding)
    model = SimpleNamespace(norm=SimpleNamespace())

    def forward(self, x, **kwargs):
        for _ in range(2):
            x = hooked_block(block, None, x)
        return hooked_exit(lambda self, y: y, model.norm, x)

    hooked = adapter.around_dit(FakeBinding)

    def run(policies, rows, x):
        (tmp_path / "control.json").write_text(json.dumps(policies))
        monkeypatch.setenv("OPT_OMNI_CONTROL", str(tmp_path / "control.json"))
        with adapter.rows_for_test(rows):
            return hooked(forward, model, torch.tensor([[v] for v in x]))[:, 0].tolist()

    def records():
        return [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]

    return SimpleNamespace(run=run, calls=calls, records=records)


REUSE = {"mode": "reuse", "k": 1}


def test_owners_at_different_steps_keep_their_own_state(dit):
    policies = {"A": REUSE, "B": {"mode": "observe"}}
    dit.run(policies, [("req-A", 0)], [1.0])
    dit.run(policies, [("req-A", 1)], [11.0])  # A stores its residual, 44 - 11
    # A joins a call with a request that has just started: A reuses from its
    # own entry and residual, B computes as if alone.
    out = dit.run(policies, [("req-A", 2), ("req-B", 0)], [21.0, 101.0])
    assert out == [54.0, 404.0]
    last = dit.records()[-2:]
    assert [(r["owner"], r["step_index"], r["call_decision"]) for r in last] == [
        ("req-A", 2, "splice"), ("req-B", 0, "splice")]


def test_blocks_are_skipped_only_when_every_owner_reuses(dit):
    policies = {"A": REUSE, "B": REUSE}
    for step, x in ((0, [1.0, 101.0]), (1, [11.0, 111.0])):
        dit.run(policies, [("req-A", step), ("req-B", step)], x)
    before = dit.calls["blocks"]
    out = dit.run(policies, [("req-A", 2), ("req-B", 2)], [21.0, 121.0])
    assert out == [54.0, 454.0] and dit.calls["blocks"] == before


def test_request_mode_counts_steps_per_owner(dit):
    policies = {"A": {"mode": "observe"}}
    for _ in range(3):
        dit.run(policies, [("req-A", None)], [1.0])
    assert [r["step_index"] for r in dit.records()] == [0, 1, 2]


def test_unattributed_rows_are_never_overridden(dit):
    policies = {"A": REUSE}
    for step in range(3):
        dit.run(policies, [("req-A", step)], [1.0, 2.0])  # one owner, two rows
    assert dit.calls["blocks"] == 6
    assert all(r["owner"] is None and not r["overridable"] for r in dit.records())


def test_retired_owner_state_is_dropped(dit):
    dit.run({"A": REUSE}, [("req-A", 0)], [1.0])
    assert "req-A" in adapter._owners
    adapter._retire("req-A")
    assert "req-A" not in adapter._owners
