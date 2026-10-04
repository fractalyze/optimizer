"""Contract tests for feasibility, on the validated catalog and SGLang's evaluator.

Every runtime here is one an experiment ran: eager, regional and whole-model
compile, and breakable CUDA graphs (exp 001, 002); native FP8 with and without
the settings that make SGLang run another method (exp 005).
"""

from absl.testing import absltest

from optimizer import catalog
from optimizer.core import feasibility
from optimizer.core.feasibility import CompileScope, ReasonKind, RuntimeContext, Status
from optimizer.core.resolver import resolve
from optimizer.core.specs import (
    TIMESTEP_STATE, TRUNK_OBSERVE, Behavior, ExecutionRequirement, ImplementationSpec, Target,
)

SGLANG_QWEN = Target("sglang", "qwen-image-2.1")
SGLANG_FLUX = Target("sglang", "flux.2-klein")
EAGER = RuntimeContext()


def _check(implementation_id, target, context):
    impl = catalog.implementations().get(implementation_id)
    return feasibility.check(impl, resolve(impl, target, catalog.providers()), context, catalog.evaluators())


class TrunkReuseTest(absltest.TestCase):

    def test_eager_is_feasible(self):
        self.assertEqual(_check("fractalyze-teacache", SGLANG_QWEN, EAGER).status, Status.FEASIBLE)

    def test_whole_model_compile_without_an_identity_check_is_infeasible(self):
        r = _check("fractalyze-teacache", SGLANG_QWEN, RuntimeContext(compile_scope=CompileScope.WHOLE_MODEL))
        self.assertEqual(r.status, Status.INFEASIBLE)
        self.assertEqual([(x.kind, x.subject) for x in r.reasons],
                         [(ReasonKind.EXECUTION_REQUIREMENT_UNSATISFIED, "override_exact(trunk_output_override)")])

    def test_whole_model_compile_where_the_identity_check_failed_is_infeasible(self):
        # exp 002: on FLUX.2 an identity override changed the image (32.6 dB).
        context = RuntimeContext(compile_scope=CompileScope.WHOLE_MODEL, identity_check_passed=False)
        r = _check("fractalyze-teacache", SGLANG_FLUX, context)
        self.assertEqual(r.status, Status.INFEASIBLE)
        self.assertIn("identity check failed", r.reasons[0].detail)

    def test_compile_where_the_identity_check_passed_is_feasible(self):
        # exp 002: Qwen-Image-2.1 stayed bitwise exact under both compile scopes.
        for scope in (CompileScope.REGIONAL, CompileScope.WHOLE_MODEL):
            context = RuntimeContext(compile_scope=scope, identity_check_passed=True)
            self.assertEqual(_check("fractalyze-teacache", SGLANG_QWEN, context).status, Status.FEASIBLE)

    def test_graph_replay_fails_every_trunk_requirement(self):
        r = _check("fractalyze-teacache", SGLANG_QWEN, RuntimeContext(graph_replay=True))
        self.assertEqual(r.status, Status.INFEASIBLE)
        self.assertEqual({x.subject for x in r.reasons}, {
            "runs_every_invocation(trunk_observe)",
            "runs_every_invocation(trunk_output_override)",
            "override_exact(trunk_output_override)",
        })
        self.assertTrue(all("does not run" in x.detail for x in r.reasons))


class NativeFp8Test(absltest.TestCase):

    def test_native_w8a8_configuration_is_feasible(self):
        for target in (SGLANG_QWEN, SGLANG_FLUX):
            self.assertEqual(_check("sglang-native-fp8-w8a8", target, EAGER).status, Status.FEASIBLE)

    def test_forced_weight_only_fallback_is_rejected_before_launch(self):
        r = _check("sglang-native-fp8-w8a8", SGLANG_QWEN,
                   RuntimeContext(engine_env={"SGLANG_FORCE_FP8_MARLIN": "1"}))
        self.assertEqual(r.status, Status.INFEASIBLE)
        self.assertEqual(r.reasons[0].kind, ReasonKind.KNOWN_NATIVE_FALLBACK)
        self.assertEqual(r.reasons[0].subject, "engine_feature.fp8_w8a8_dynamic_linear")

    def test_partial_coverage_is_rejected_before_launch(self):
        r = _check("sglang-native-fp8-w8a8", SGLANG_QWEN,
                   RuntimeContext(engine_env={"SGLANG_FP8_IGNORED_LAYERS": "attn"}))
        self.assertEqual(r.reasons[0].kind, ReasonKind.KNOWN_NATIVE_FALLBACK)

    def test_a_setting_that_does_not_trigger_the_fallback_is_feasible(self):
        r = _check("sglang-native-fp8-w8a8", SGLANG_QWEN,
                   RuntimeContext(engine_env={"SGLANG_FORCE_FP8_MARLIN": "0"}))
        self.assertEqual(r.status, Status.FEASIBLE)


class GenericTest(absltest.TestCase):

    def test_implementation_without_execution_requirements_is_feasible_anywhere(self):
        impl = ImplementationSpec("plain", "invented-technique", frozenset({TIMESTEP_STATE}))
        resolution = resolve(impl, SGLANG_QWEN, catalog.providers())
        for context in (EAGER, RuntimeContext(graph_replay=True, compile_scope=CompileScope.WHOLE_MODEL)):
            r = feasibility.check(impl, resolution, context, catalog.evaluators())
            self.assertEqual(r.status, Status.FEASIBLE)

    def test_step_seam_requirements_hold_in_every_tested_mode(self):
        # exp 001: step seams held under compile and graph replay.
        context = RuntimeContext(graph_replay=True, compile_scope=CompileScope.WHOLE_MODEL)
        self.assertEqual(_check("fractalyze-prediction-reuse", SGLANG_QWEN, context).status, Status.FEASIBLE)

    def test_engine_without_an_evaluator_cannot_be_judged(self):
        r = _check("fractalyze-teacache", Target("vllm-omni", "qwen-image-2512"), EAGER)
        self.assertEqual(r.status, Status.INFEASIBLE)
        self.assertEqual(r.reasons[0].kind, ReasonKind.UNSUPPORTED_RUNTIME_CONDITION)

    def test_only_a_resolved_resolution_of_the_same_implementation_is_accepted(self):
        impl = catalog.implementations().get("fractalyze-teacache")
        unsupported = resolve(impl, Target("sglang", "no-binding"), catalog.providers())
        with self.assertRaisesRegex(ValueError, "RESOLVED"):
            feasibility.check(impl, unsupported, EAGER, catalog.evaluators())
        other = resolve(catalog.implementations().get("sglang-native-fp8-w8a8"), SGLANG_QWEN, catalog.providers())
        with self.assertRaisesRegex(ValueError, "RESOLVED"):
            feasibility.check(impl, other, EAGER, catalog.evaluators())

    def test_execution_requirement_must_be_on_a_required_capability(self):
        with self.assertRaisesRegex(ValueError, "does not require"):
            ImplementationSpec("i", "t", frozenset({TIMESTEP_STATE}),
                               frozenset({ExecutionRequirement(Behavior.OVERRIDE_EXACT, TRUNK_OBSERVE)}))

    def test_printable(self):
        text = str(_check("fractalyze-teacache", SGLANG_QWEN, RuntimeContext(graph_replay=True)))
        self.assertIn("fractalyze-teacache: INFEASIBLE", text)
        self.assertIn("execution_requirement_unsatisfied: override_exact(trunk_output_override)", text)


if __name__ == "__main__":
    absltest.main()
