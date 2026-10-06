"""Contract tests for execution planning and SGLang's lowering of a plan.

The central property: the ServerKey changes exactly when a live server would
have to be rebuilt, never when only a request's parameters change.
"""

import dataclasses

from absl.testing import absltest

from optimizer import catalog, engines
from optimizer.core import feasibility
from optimizer.core.capabilities import TIMESTEP_STATE
from optimizer.core.composition import Candidate, compose
from optimizer.core.feasibility import CompileScope, RuntimeContext
from optimizer.core.plan import ExecutionPlan, ServerSettings, plan
from optimizer.core.resolver import resolve
from optimizer.core.specs import ImplementationSpec, ModelRef, ParamType, ParameterSpec, Target, TechniqueConfig, TechniqueSpec
from optimizer.core.registry import TechniqueRegistry
from optimizer.engines.sglang import plan as sglang_plan

SGLANG_QWEN = Target("sglang", "QwenImage21Transformer2DModel", ModelRef("Qwen/Qwen-Image-2.1"))
FP8, TEACACHE, REUSE = "sglang-native-fp8-w8a8", "fractalyze-teacache", "fractalyze-prediction-reuse"


def _plan(ids, configs=(), context=RuntimeContext(), target=SGLANG_QWEN) -> ExecutionPlan:
    candidates = []
    for impl_id in ids:
        impl = catalog.implementations().get(impl_id)
        resolution = resolve(impl, target, catalog.providers())
        candidates.append(Candidate(impl, resolution,
                                    feasibility.check(impl, resolution, context, engines.evaluators())))
    return plan(compose(candidates), candidates, configs, catalog.techniques())


def _tc(threshold):
    return TechniqueConfig.of("teacache", threshold=threshold)


class PlacementTest(absltest.TestCase):

    def test_fp8_with_teacache(self):
        p = _plan([FP8, TEACACHE], [_tc(0.12)])
        self.assertEqual([(c.implementation, c.values) for c in p.server.contributions],
                         [(TEACACHE, ()), (FP8, ())])
        self.assertEqual([(r.implementation, r.values) for r in p.requests], [(TEACACHE, (("threshold", 0.12),))])
        self.assertNotIn("0.12", repr(p.server))

    def test_fp8_is_server_only(self):
        p = _plan([FP8])
        self.assertEqual([c.implementation for c in p.server.contributions], [FP8])
        self.assertEqual(p.requests, ())

    def test_prediction_reuse_is_request_only(self):
        p = _plan([REUSE], [TechniqueConfig.of("prediction_reuse", steps=(10, 20))])
        self.assertEqual(p.server.contributions, ())
        self.assertEqual(p.requests[0].values, (("steps", (10, 20)),))

    def test_values_of_an_implementation_without_request_state_are_fixed_with_the_server(self):
        techniques = TechniqueRegistry([TechniqueSpec("t", "x", (ParameterSpec("p", ParamType.FLOAT),))])
        impl = ImplementationSpec("server-param", "t", frozenset({TIMESTEP_STATE}))
        resolution = resolve(impl, SGLANG_QWEN, catalog.providers())
        candidate = Candidate(impl, resolution,
                              feasibility.check(impl, resolution, RuntimeContext(), engines.evaluators()))
        a = plan(compose([candidate]), [candidate], [TechniqueConfig.of("t", p=1.0)], techniques)
        b = plan(compose([candidate]), [candidate], [TechniqueConfig.of("t", p=2.0)], techniques)
        self.assertNotEqual(a.server, b.server)
        self.assertEqual(a.requests, ())


class ServerKeyTest(absltest.TestCase):

    def test_teacache_threshold_does_not_change_the_server(self):
        a, b = _plan([FP8, TEACACHE], [_tc(0.10)]), _plan([FP8, TEACACHE], [_tc(0.20)])
        self.assertEqual(a.server, b.server)
        self.assertEqual(hash(a.server), hash(b.server))
        self.assertNotEqual(a.requests, b.requests)

    def test_fp8_presence_changes_the_server(self):
        self.assertNotEqual(_plan([TEACACHE], [_tc(0.1)]).server, _plan([FP8, TEACACHE], [_tc(0.1)]).server)

    def test_prediction_reuse_steps_do_not_change_the_server(self):
        a = _plan([REUSE], [TechniqueConfig.of("prediction_reuse", steps=(10,))])
        b = _plan([REUSE], [TechniqueConfig.of("prediction_reuse", steps=(5, 30))])
        self.assertEqual(a.server, b.server)
        self.assertNotEqual(a.requests, b.requests)

    def test_installing_teacache_changes_the_server(self):
        # exp 002: its hooks must be in place before the model is built.
        self.assertNotEqual(_plan([FP8]).server, _plan([FP8, TEACACHE], [_tc(0.1)]).server)

    def test_evidence_in_the_context_does_not_change_the_server(self):
        compiled = dict(compile_scope=CompileScope.REGIONAL)
        a = _plan([TEACACHE], [_tc(0.1)], RuntimeContext(**compiled, identity_check_passed=True))
        b = _plan([REUSE], [TechniqueConfig.of("prediction_reuse", steps=(1,))],
                  RuntimeContext(identity_check_passed=True))
        c = _plan([REUSE], [TechniqueConfig.of("prediction_reuse", steps=(1,))],
                  RuntimeContext(identity_check_passed=False))
        self.assertEqual(b.server, c.server)
        self.assertEqual(a.server.settings, ServerSettings(CompileScope.REGIONAL, False, ()))

    def test_execution_mode_and_engine_env_change_the_server(self):
        steps = [TechniqueConfig.of("prediction_reuse", steps=(1,))]
        eager = _plan([REUSE], steps).server
        self.assertNotEqual(eager, _plan([REUSE], steps, RuntimeContext(graph_replay=True)).server)
        self.assertNotEqual(eager, _plan([REUSE], steps,
                                         RuntimeContext(compile_scope=CompileScope.WHOLE_MODEL)).server)
        self.assertNotEqual(eager, _plan([REUSE], steps, RuntimeContext(engine_env={"X": "1"})).server)

    def test_every_runtime_context_field_is_classified(self):
        # A new field must be decided: server-affecting, or evidence.
        evidence = {"identity_check_passed"}
        server = {f.name for f in dataclasses.fields(ServerSettings)}
        self.assertEqual({f.name for f in dataclasses.fields(RuntimeContext)}, server | evidence)


class ParameterTest(absltest.TestCase):

    def test_wrong_type_unknown_and_missing_are_rejected(self):
        for config in (_tc("0.1"), _tc(True), TechniqueConfig.of("teacache", threshold=0.1, extra=1),
                       TechniqueConfig.of("teacache")):
            with self.assertRaises(ValueError):
                _plan([TEACACHE], [config])

    def test_a_parameterized_technique_must_be_configured(self):
        with self.assertRaisesRegex(ValueError, "missing \\['threshold'\\]"):
            _plan([TEACACHE])

    def test_steps_must_be_a_tuple_of_ints(self):
        with self.assertRaises(ValueError):
            _plan([REUSE], [TechniqueConfig.of("prediction_reuse", steps=[1, 2])])

    def test_config_for_a_technique_not_in_the_plan_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not in the plan"):
            _plan([FP8], [_tc(0.1)])

    def test_integer_threshold_is_a_float(self):
        self.assertEqual(_plan([TEACACHE], [_tc(0)]).requests[0].values, (("threshold", 0),))


class SGLangLoweringTest(absltest.TestCase):

    def setUp(self):
        self.translator = sglang_plan.validated()

    def test_fp8_with_teacache(self):
        server, request = self.translator.lower(_plan([FP8, TEACACHE], [_tc(0.12)]))
        self.assertEqual(server.model_path, "Qwen/Qwen-Image-2.1")
        self.assertEqual(server.server_kwargs, (("quantization", "fp8"),))
        self.assertEqual(server.plugins, ("opt_step_probe", "opt_trunk_probe"))
        self.assertEqual(request.policies,
                         (("opt_trunk_probe", (("implementation", TEACACHE), ("threshold", 0.12))),))

    def test_prediction_reuse_adds_nothing_to_the_server(self):
        server, request = self.translator.lower(
            _plan([REUSE], [TechniqueConfig.of("prediction_reuse", steps=(10,))]))
        self.assertEqual((server.server_kwargs, server.plugins), ((), ("opt_step_probe",)))
        self.assertEqual(request.policies[0][0], "opt_step_probe")

    def test_equal_server_keys_lower_to_equal_server_configs(self):
        a, _ = self.translator.lower(_plan([FP8, TEACACHE], [_tc(0.1)]))
        b, _ = self.translator.lower(_plan([FP8, TEACACHE], [_tc(0.2)]))
        self.assertEqual(a, b)

    def test_execution_mode_and_env_are_lowered(self):
        context = RuntimeContext(compile_scope=CompileScope.REGIONAL, identity_check_passed=True,
                                 engine_env={"SGLANG_FP8_GEMM": "x"})
        server, _ = self.translator.lower(_plan([TEACACHE], [_tc(0.1)], context))
        self.assertEqual(server.server_kwargs, (("enable_torch_compile", True), ("regional_compile", True)))
        self.assertEqual(server.env, (("SGLANG_FP8_GEMM", "x"),))
        steps = [TechniqueConfig.of("prediction_reuse", steps=(1,))]
        graphs, _ = self.translator.lower(_plan([REUSE], steps, RuntimeContext(graph_replay=True)))
        self.assertEqual(graphs.server_kwargs, (("enable_breakable_cuda_graph", True),))

    def test_other_engines_and_unknown_implementations_are_refused(self):
        omni = _plan([REUSE], [TechniqueConfig.of("prediction_reuse", steps=(1,))])
        omni = dataclasses.replace(omni, server=dataclasses.replace(omni.server, engine="vllm-omni"))
        with self.assertRaisesRegex(ValueError, "vllm-omni"):
            self.translator.lower(omni)
        bare = dataclasses.replace(self.translator, request_plugin={})
        with self.assertRaisesRegex(ValueError, "no lowering"):
            bare.lower(_plan([REUSE], [TechniqueConfig.of("prediction_reuse", steps=(1,))]))


class InputContractTest(absltest.TestCase):

    def test_composer_rejects_candidates_from_different_contexts(self):
        impl = catalog.implementations().get(REUSE)
        resolution = resolve(impl, SGLANG_QWEN, catalog.providers())
        a = Candidate(impl, resolution, feasibility.check(impl, resolution, RuntimeContext(), engines.evaluators()))
        other = catalog.implementations().get(FP8)
        r2 = resolve(other, SGLANG_QWEN, catalog.providers())
        b = Candidate(other, r2, feasibility.check(other, r2, RuntimeContext(graph_replay=True),
                                                   engines.evaluators()))
        with self.assertRaisesRegex(ValueError, "different runtime contexts"):
            compose([a, b])

    def test_plan_rejects_candidates_the_composer_did_not_approve(self):
        p_candidates = []
        for impl_id in (FP8, TEACACHE):
            impl = catalog.implementations().get(impl_id)
            res = resolve(impl, SGLANG_QWEN, catalog.providers())
            p_candidates.append(Candidate(impl, res, feasibility.check(impl, res, RuntimeContext(),
                                                                        engines.evaluators())))
        with self.assertRaisesRegex(ValueError, "approved"):
            plan(compose(p_candidates[:1]), p_candidates, [], catalog.techniques())

    def test_printable(self):
        text = str(_plan([FP8, TEACACHE], [_tc(0.12)]))
        self.assertIn("request:\n  fractalyze-teacache {'threshold': 0.12}", text)


if __name__ == "__main__":
    absltest.main()
