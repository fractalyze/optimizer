"""Contract tests for capability resolution, on the validated catalog.

The three paths the experiments validated: engine-common (step), model-bound
(trunk, adapter + Binding) and engine-native (FP8). Then the two failures.
"""

from absl.testing import absltest

from optimizer import catalog
from optimizer.core.registry import ProviderRegistry
from optimizer.core.resolver import Status, resolve
from optimizer.core.specs import (
    REQUEST_LOCAL_STATE, SIGNAL_OBSERVE, TIMESTEP_STATE, TRUNK_OBSERVE, TRUNK_OUTPUT_OVERRIDE,
    ImplementationSpec, ProviderKind, ProviderSpec, Target,
)

SGLANG_QWEN = Target("sglang", "qwen-image-2.1")
SGLANG_FLUX = Target("sglang", "flux.2-klein")


def _resolve(implementation_id, target, providers=None):
    impl = catalog.implementations().get(implementation_id)
    return resolve(impl, target, providers or catalog.providers())


class ValidatedPathsTest(absltest.TestCase):

    def test_engine_common_prediction_reuse_resolves_in_adapter_only(self):
        r = _resolve("fractalyze-prediction-reuse", SGLANG_QWEN)
        self.assertEqual(r.status, Status.RESOLVED)
        self.assertEqual(set(r.providers.values()), {"SGLangAdapter"})

    def test_engine_common_prediction_reuse_needs_no_binding(self):
        r = _resolve("fractalyze-prediction-reuse", Target("sglang", "a-model-with-no-binding"))
        self.assertEqual(r.status, Status.RESOLVED)

    def test_trunk_reuse_splits_across_adapter_and_qwen_binding(self):
        r = _resolve("fractalyze-teacache", SGLANG_QWEN)
        self.assertEqual(r.status, Status.RESOLVED)
        self.assertEqual(dict(r.providers), {
            TIMESTEP_STATE: "SGLangAdapter",
            REQUEST_LOCAL_STATE: "SGLangAdapter",
            TRUNK_OBSERVE: "SGLangQwenImage21Binding",
            TRUNK_OUTPUT_OVERRIDE: "SGLangQwenImage21Binding",
            SIGNAL_OBSERVE: "SGLangQwenImage21Binding",
        })

    def test_same_trunk_implementation_resolves_on_flux_through_its_binding(self):
        r = _resolve("fractalyze-teacache", SGLANG_FLUX)
        self.assertEqual(r.status, Status.RESOLVED)
        self.assertEqual(r.providers[SIGNAL_OBSERVE], "SGLangFlux2Binding")
        self.assertEqual(r.providers[TIMESTEP_STATE], "SGLangAdapter")

    def test_same_trunk_implementation_resolves_on_a_second_engine(self):
        r = _resolve("fractalyze-teacache", Target("vllm-omni", "qwen-image-2512"))
        self.assertEqual(r.status, Status.RESOLVED)
        self.assertEqual(r.providers[REQUEST_LOCAL_STATE], "VllmOmniAdapter")
        self.assertEqual(r.providers[TRUNK_OUTPUT_OVERRIDE], "VllmOmniQwenImageBinding")

    def test_native_fp8_resolves_to_the_engine_feature_in_the_adapter(self):
        for target in (SGLANG_QWEN, SGLANG_FLUX):
            r = _resolve("sglang-native-fp8-w8a8", target)
            self.assertEqual(r.status, Status.RESOLVED)
            self.assertEqual(dict(r.providers), {"engine_feature.fp8_w8a8_dynamic_linear": "SGLangAdapter"})

    def test_native_fp8_is_unsupported_where_the_engine_lacks_the_feature(self):
        r = _resolve("sglang-native-fp8-w8a8", Target("vllm-omni", "qwen-image-2512"))
        self.assertEqual(r.status, Status.UNSUPPORTED)
        self.assertEqual(r.missing, ("engine_feature.fp8_w8a8_dynamic_linear",))


class FailureTest(absltest.TestCase):

    def test_model_without_binding_leaves_trunk_capabilities_missing(self):
        r = _resolve("fractalyze-teacache", Target("sglang", "a-model-with-no-binding"))
        self.assertEqual(r.status, Status.UNSUPPORTED)
        self.assertEqual(r.missing, (SIGNAL_OBSERVE, TRUNK_OBSERVE, TRUNK_OUTPUT_OVERRIDE))

    def test_binding_without_signal_is_unsupported_with_no_partial_result_hidden(self):
        no_signal = ProviderRegistry([
            catalog.providers().get("SGLangAdapter"),
            ProviderSpec("NoSignalBinding", ProviderKind.BINDING, "sglang", "m",
                         frozenset({TRUNK_OBSERVE, TRUNK_OUTPUT_OVERRIDE})),
        ])
        r = _resolve("fractalyze-teacache", Target("sglang", "m"), no_signal)
        self.assertEqual(r.status, Status.UNSUPPORTED)
        self.assertEqual(r.missing, (SIGNAL_OBSERVE,))
        self.assertNotIn(SIGNAL_OBSERVE, r.providers)

    def test_two_providers_claiming_one_capability_is_ambiguous_not_a_choice(self):
        both = ProviderRegistry([
            ProviderSpec("A", ProviderKind.ENGINE_ADAPTER, "e", None, frozenset({SIGNAL_OBSERVE})),
            ProviderSpec("B", ProviderKind.BINDING, "e", "m", frozenset({SIGNAL_OBSERVE})),
        ])
        impl = ImplementationSpec("i", "t", frozenset({SIGNAL_OBSERVE}))
        r = resolve(impl, Target("e", "m"), both)
        self.assertEqual(r.status, Status.AMBIGUOUS)
        self.assertEqual(dict(r.ambiguous), {SIGNAL_OBSERVE: ("A", "B")})
        self.assertNotIn(SIGNAL_OBSERVE, r.providers)


class GenericityTest(absltest.TestCase):

    def test_resolver_needs_no_knowledge_of_the_technique(self):
        impl = ImplementationSpec("never-registered", "invented-technique", frozenset({TIMESTEP_STATE}))
        self.assertEqual(resolve(impl, SGLANG_QWEN, catalog.providers()).status, Status.RESOLVED)

    def test_printable(self):
        text = str(_resolve("fractalyze-teacache", SGLANG_QWEN))
        self.assertIn("fractalyze-teacache on sglang + qwen-image-2.1: RESOLVED", text)
        self.assertIn("signal_observe -> SGLangQwenImage21Binding", text)


if __name__ == "__main__":
    absltest.main()
