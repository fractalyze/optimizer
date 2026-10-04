"""The capability vocabulary: the semantic operations implementations require
and providers supply, and the check that keeps ids to this vocabulary.

Ids name operations, never hooks or seams; their meanings are defined in
docs/architecture.md ("Capabilities in V1"). Only capabilities with runtime
evidence are listed. Add one only together with the experiment that
validates it.
"""

from __future__ import annotations

STEP_OBSERVE = "step_observe"
STEP_PREDICTION_OVERRIDE = "step_prediction_override"
STEP_SCHEDULE_MUTATE = "step_schedule_mutate"
TIMESTEP_STATE = "timestep_state"
REQUEST_LOCAL_STATE = "request_local_state"
TRUNK_OBSERVE = "trunk_observe"
TRUNK_OUTPUT_OVERRIDE = "trunk_output_override"
SIGNAL_OBSERVE = "signal_observe"

CAPABILITIES = frozenset({
    STEP_OBSERVE, STEP_PREDICTION_OVERRIDE, STEP_SCHEDULE_MUTATE,
    TIMESTEP_STATE, REQUEST_LOCAL_STATE,
    TRUNK_OBSERVE, TRUNK_OUTPUT_OVERRIDE, SIGNAL_OBSERVE,
})

# Engine-native features: the engine implements the method whole (ADR 0016).
# Any id under this prefix is accepted. FP8's id happens to match its
# technique's; that is not a rule, and a native implementation may need
# several such capabilities.
ENGINE_FEATURE_PREFIX = "engine_feature."
ENGINE_FEATURE_FP8_W8A8_DYNAMIC_LINEAR = ENGINE_FEATURE_PREFIX + "fp8_w8a8_dynamic_linear"


def check_capability(capability: str) -> None:
    """Rejects a misspelled or unvalidated capability at registration, where
    it would otherwise surface later as a confusing UNSUPPORTED."""
    if capability in CAPABILITIES:
        return
    if capability.startswith(ENGINE_FEATURE_PREFIX) and len(capability) > len(ENGINE_FEATURE_PREFIX):
        return
    raise ValueError(f"unknown capability {capability!r}")
