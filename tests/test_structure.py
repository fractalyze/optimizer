"""Contract tests for ownership: the catalog is atomic, Core depends on no engine
or catalog, registries are isolated, and SGLang's policy is explicit data."""

import subprocess
import sys

from absl.testing import absltest

from optimizer import catalog
from optimizer.core.capabilities import ENGINE_FEATURE_FP8_W8A8_DYNAMIC_LINEAR, TRUNK_OBSERVE
from optimizer.core.feasibility import ReasonKind, RuntimeContext
from optimizer.core.registry import TechniqueRegistry
from optimizer.core.specs import Behavior, ExecutionRequirement, ImplementationSpec, ProviderSpec, TechniqueSpec
from optimizer.engines.sglang.feasibility import EnvTrigger, NativeFallback, SeamRegion, SGLangFeasibility


class CatalogTest(absltest.TestCase):

    def test_catalog_holds_only_atomic_entries(self):
        self.assertTrue(all(isinstance(t, TechniqueSpec) for t in catalog.TECHNIQUES))
        self.assertTrue(all(isinstance(i, ImplementationSpec) for i in catalog.IMPLEMENTATIONS))
        self.assertTrue(all(isinstance(p, ProviderSpec) for p in catalog.PROVIDERS))

    def test_entries_from_every_owner_validate_together(self):
        self.assertLen(catalog.implementations().list(), len(catalog.IMPLEMENTATIONS))
        self.assertLen(catalog.providers().list(), len(catalog.PROVIDERS))

    def test_a_test_registry_leaves_the_default_catalog_untouched(self):
        isolated = TechniqueRegistry([TechniqueSpec("only-here", "x")])
        self.assertEqual([t.id for t in isolated.list()], ["only-here"])
        self.assertNotIn("only-here", [t.id for t in catalog.techniques().list()])


class DependencyDirectionTest(absltest.TestCase):

    def test_core_imports_no_engine_package_catalog_or_engine_runtime(self):
        code = ("import sys, optimizer.core.capabilities, optimizer.core.specs, optimizer.core.registry, "
                "optimizer.core.resolver, optimizer.core.feasibility; "
                "print(sorted(m for m in sys.modules if m.startswith(('optimizer.engines', 'optimizer.catalog', "
                "'optimizer.techniques', 'sglang', 'vllm', 'torch'))))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
        self.assertEqual(out.strip(), "[]")


class SGLangPolicyTest(absltest.TestCase):

    def test_env_triggers_read_values_as_sglang_does(self):
        self.assertTrue(EnvTrigger.BOOL.is_set("TRUE"))
        self.assertTrue(EnvTrigger.BOOL.is_set("1"))
        self.assertFalse(EnvTrigger.BOOL.is_set("0"))
        self.assertFalse(EnvTrigger.BOOL.is_set("yes"))
        self.assertTrue(EnvTrigger.NONEMPTY.is_set("attn"))
        self.assertFalse(EnvTrigger.NONEMPTY.is_set("  "))

    def test_policy_comes_only_from_the_evaluator_it_is_given(self):
        requirement = ExecutionRequirement(Behavior.RUNS_EVERY_INVOCATION, TRUNK_OBSERVE)
        replay = RuntimeContext(graph_replay=True)
        unknown = SGLangFeasibility(seams={}, native_fallbacks=())
        self.assertIn("no evidence", unknown.requirement(requirement, RuntimeContext()).detail)
        in_loop = SGLangFeasibility(seams={TRUNK_OBSERVE: SeamRegion.DENOISING_LOOP}, native_fallbacks=())
        self.assertIsNone(in_loop.requirement(requirement, replay))

    def test_native_fallback_applies_only_to_its_capability(self):
        evaluator = SGLangFeasibility(seams={}, native_fallbacks=(
            NativeFallback(ENGINE_FEATURE_FP8_W8A8_DYNAMIC_LINEAR, "X", EnvTrigger.BOOL, "other method"),))
        context = RuntimeContext(engine_env={"X": "1"})
        self.assertEqual(evaluator.capability(ENGINE_FEATURE_FP8_W8A8_DYNAMIC_LINEAR, context).kind,
                         ReasonKind.KNOWN_NATIVE_FALLBACK)
        self.assertIsNone(evaluator.capability(TRUNK_OBSERVE, context))


if __name__ == "__main__":
    absltest.main()
