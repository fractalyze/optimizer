"""CPU check of experiment 005's optimizer path. Needs only pytest.

    python -m pytest test_optimizer_path.py
"""

import dataclasses

import pytest

import optimizer_path as op

TECH = "fp8_w8a8_dynamic_linear"
CAP = frozenset({"engine_feature.fp8_w8a8_dynamic_linear"})


def target(**kw):
    base = dict(engine="sglang", gpu_capability=(12, 0), cuda_version=(13, 0),
                capabilities=CAP, env_overrides={})
    return op.Target(**{**base, **kw})


ENGAGED = op.Evidence(quantizable=224, fp8_w8a8=224, fp8_weight_only=0, unquantized=0,
                      not_quantizable=8, runtime_w8a8_gemms=8960, runtime_weight_only_gemms=0,
                      runtime_activation_quants=8960)


def test_resolution_picks_the_engines_own_candidate():
    assert [i.id for i in op.resolve(TECH, target())] == ["sglang-native-fp8-w8a8"]
    assert [i.id for i in op.resolve(TECH, target(engine="vllm-omni"))] == ["vllm-omni-native-fp8-w8a8"]


def test_feasible_target_gets_native_settings_and_no_status():
    impl, settings, status, _ = op.plan(TECH, target(), op.SGLangAdapter())
    assert (impl.id, settings, status) == ("sglang-native-fp8-w8a8", {"quantization": "fp8"}, None)


def test_the_flag_never_appears_above_the_adapter():
    technique = op.TECHNIQUES[TECH]
    impl = op.resolve(TECH, target())[0]
    assert "quantization" not in str(technique) and "quantization" not in str(impl.requires)


@pytest.mark.parametrize("kw, reason", [
    (dict(gpu_capability=(8, 6)), "weight-only"),
    (dict(gpu_capability=(8, 9), cuda_version=(12, 1)), "12.4"),
    (dict(env_overrides={"SGLANG_FORCE_FP8_MARLIN": "1"}), "SGLANG_FORCE_FP8_MARLIN"),
])
def test_targets_that_would_silently_change_the_method_are_infeasible(kw, reason):
    _, settings, status, why = op.plan(TECH, target(**kw), op.SGLangAdapter())
    assert (settings, status) == (None, op.INFEASIBLE) and reason in why


def test_missing_capability_is_unsupported():
    _, _, status, _ = op.plan(TECH, target(capabilities=frozenset()), op.SGLangAdapter())
    assert status == op.UNSUPPORTED


def test_full_coverage_with_executed_fp8_gemms_is_valid():
    impl = op.resolve(TECH, target())[0]
    assert op.verdict(impl, ENGAGED)[0] == op.VALID


@pytest.mark.parametrize("change, reason", [
    (dict(fp8_w8a8=0, fp8_weight_only=224, runtime_w8a8_gemms=0,
          runtime_weight_only_gemms=8960, runtime_activation_quants=0), "weight-only"),
    (dict(fp8_w8a8=0, unquantized=224, runtime_w8a8_gemms=0, runtime_activation_quants=0), "partial: 0 of 224"),
    (dict(fp8_w8a8=200, unquantized=24), "partial: 200 of 224"),
    (dict(runtime_w8a8_gemms=0, runtime_activation_quants=0), "no FP8 GEMM"),
    (dict(quantizable=0, fp8_w8a8=0), "no quantizable"),
])
def test_configured_but_not_engaged_fails(change, reason):
    impl = op.resolve(TECH, target())[0]
    status, why = op.verdict(impl, dataclasses.replace(ENGAGED, **change))
    assert status == op.FAILED_TO_ENGAGE and reason in why
