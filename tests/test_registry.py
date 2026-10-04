"""Contract tests for the specs and registries: what they accept, reject and list."""

from absl.testing import absltest

from optimizer import catalog
from optimizer.core.registry import ImplementationRegistry, ProviderRegistry, TechniqueRegistry
from optimizer.core.specs import (
    SIGNAL_OBSERVE, TIMESTEP_STATE, ImplementationSpec, ProviderKind, ProviderSpec, Target, TechniqueSpec,
    engine_feature,
)


class TechniqueRegistryTest(absltest.TestCase):

    def test_get_and_list_in_registration_order(self):
        registry = catalog.techniques()
        self.assertEqual(registry.get("teacache").id, "teacache")
        self.assertEqual([t.id for t in registry.list()],
                         ["prediction_reuse", "teacache", "fp8_w8a8_dynamic_linear"])

    def test_duplicate_id_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            TechniqueRegistry([TechniqueSpec("a", "x"), TechniqueSpec("a", "y")])

    def test_unknown_id_is_a_clear_error(self):
        with self.assertRaisesRegex(KeyError, "nope"):
            catalog.techniques().get("nope")


class ImplementationRegistryTest(absltest.TestCase):

    def test_for_technique(self):
        registry = catalog.implementations()
        self.assertEqual([i.id for i in registry.for_technique("teacache")], ["fractalyze-teacache"])
        self.assertEqual([i.id for i in registry.for_technique("fp8_w8a8_dynamic_linear")],
                         ["sglang-native-fp8-w8a8"])

    def test_for_technique_keeps_registration_order(self):
        techniques = TechniqueRegistry([TechniqueSpec("t", "x")])
        registry = ImplementationRegistry(techniques, [
            ImplementationSpec("b", "t", frozenset()), ImplementationSpec("a", "t", frozenset())])
        self.assertEqual([i.id for i in registry.for_technique("t")], ["b", "a"])

    def test_must_reference_a_registered_technique(self):
        with self.assertRaisesRegex(KeyError, "missing"):
            ImplementationRegistry(catalog.techniques(), [ImplementationSpec("i", "missing", frozenset())])

    def test_duplicate_id_rejected(self):
        spec = ImplementationSpec("i", "teacache", frozenset())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ImplementationRegistry(catalog.techniques(), [spec, spec])


class CapabilityIdTest(absltest.TestCase):

    def test_known_and_engine_feature_capabilities_accepted(self):
        ImplementationSpec("i", "t", frozenset({SIGNAL_OBSERVE, engine_feature("anything")}))

    def test_misspelled_capability_rejected(self):
        with self.assertRaisesRegex(ValueError, "signal_observ"):
            ImplementationSpec("i", "t", frozenset({"signal_observ"}))
        with self.assertRaisesRegex(ValueError, "engine_feature."):
            ProviderSpec("p", ProviderKind.ENGINE_ADAPTER, "e", None, frozenset({"engine_feature."}))


class ProviderRegistryTest(absltest.TestCase):

    def test_adapter_then_model_binding(self):
        registry = catalog.providers()
        self.assertEqual([p.id for p in registry.for_target(Target("sglang", "qwen-image-2.1"))],
                         ["SGLangAdapter", "SGLangQwenImage21Binding"])
        self.assertEqual([p.id for p in registry.for_target(Target("sglang", "flux.2-klein"))],
                         ["SGLangAdapter", "SGLangFlux2Binding"])

    def test_model_without_binding_still_gets_its_adapter(self):
        self.assertEqual([p.id for p in catalog.providers().for_target(Target("sglang", "unknown"))],
                         ["SGLangAdapter"])

    def test_unknown_engine_has_no_providers(self):
        self.assertEqual(catalog.providers().for_target(Target("comfyui", "qwen-image-2.1")), ())

    def test_binding_must_name_a_model_and_adapter_must_not(self):
        with self.assertRaises(ValueError):
            ProviderSpec("b", ProviderKind.BINDING, "e", None, frozenset({TIMESTEP_STATE}))
        with self.assertRaises(ValueError):
            ProviderSpec("a", ProviderKind.ENGINE_ADAPTER, "e", "m", frozenset({TIMESTEP_STATE}))

    def test_duplicate_id_rejected(self):
        spec = ProviderSpec("a", ProviderKind.ENGINE_ADAPTER, "e", None, frozenset())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ProviderRegistry([spec, spec])


if __name__ == "__main__":
    absltest.main()
