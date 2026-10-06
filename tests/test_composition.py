"""Contract tests for composition: coexistence from exclusive claims, ordering
from declared `after` resources, and nothing from implementation names.

Test-only implementations use the real resource vocabulary; the trunk
collision mirrors one the investigation found in SGLang (TeaCache and
cache-dit both want the trunk).
"""

from absl.testing import absltest

from optimizer import catalog, engines
from optimizer.core import feasibility
from optimizer.core.capabilities import TIMESTEP_STATE
from optimizer.core.composition import Candidate, ReasonKind, Status, compose
from optimizer.core.feasibility import RuntimeContext
from optimizer.core.resolver import resolve
from optimizer.core.resources import LINEAR_LAYERS, STEP_PREDICTION, TRUNK
from optimizer.core.specs import ImplementationSpec, ModelRef, Target

SGLANG_QWEN = Target("sglang", "QwenImage21Transformer2DModel", ModelRef("Qwen/Qwen-Image-2.1"))
EAGER = RuntimeContext()


def _candidate(impl, target=SGLANG_QWEN, context=EAGER):
    resolution = resolve(impl, target, catalog.providers())
    return Candidate(impl, resolution, feasibility.check(impl, resolution, context, engines.evaluators()))


def _catalog(*ids):
    return [_candidate(catalog.implementations().get(i)) for i in ids]


def _fixture(id_, owns=(), after=()):
    return _candidate(ImplementationSpec(id_, "test-only", frozenset({TIMESTEP_STATE}),
                                         owns=frozenset(owns), after=frozenset(after)))


class ValidatedCombinationsTest(absltest.TestCase):

    def test_fp8_and_teacache_coexist(self):
        r = compose(_catalog("sglang-native-fp8-w8a8", "fractalyze-teacache"))
        self.assertEqual(r.status, Status.COMPOSABLE)
        self.assertEqual(r.ordering, ())

    def test_teacache_and_prediction_reuse_coexist(self):
        r = compose(_catalog("fractalyze-teacache", "fractalyze-prediction-reuse"))
        self.assertEqual(r.status, Status.COMPOSABLE)

    def test_three_together_compose_whatever_the_input_order(self):
        ids = ("sglang-native-fp8-w8a8", "fractalyze-teacache", "fractalyze-prediction-reuse")
        forward, backward = compose(_catalog(*ids)), compose(_catalog(*reversed(ids)))
        self.assertEqual(forward.status, Status.COMPOSABLE)
        self.assertEqual(forward, backward)
        self.assertEqual(forward.implementations, tuple(sorted(ids)))


class ConflictTest(absltest.TestCase):

    def test_shared_required_capability_is_not_a_conflict(self):
        self.assertEqual(compose([_fixture("a"), _fixture("b")]).status, Status.COMPOSABLE)

    def test_two_exclusive_claims_on_one_resource_conflict(self):
        r = compose(_catalog("fractalyze-teacache") + [_fixture("other-trunk-cache", owns={TRUNK})])
        self.assertEqual(r.status, Status.CONFLICT)
        self.assertEqual([(x.kind, x.subject) for x in r.reasons], [(ReasonKind.EXCLUSIVE_RESOURCE, TRUNK)])
        self.assertIn("fractalyze-teacache, other-trunk-cache", r.reasons[0].detail)
        self.assertEqual(r.ordering, ())

    def test_every_collision_is_reported(self):
        r = compose(_catalog("sglang-native-fp8-w8a8", "fractalyze-teacache")
                    + [_fixture("x", owns={TRUNK, LINEAR_LAYERS})])
        self.assertEqual([x.subject for x in r.reasons], [LINEAR_LAYERS, TRUNK])


class OrderingTest(absltest.TestCase):

    def test_after_a_resource_orders_after_its_owner(self):
        r = compose([_fixture("late", after={STEP_PREDICTION}), _fixture("early", owns={STEP_PREDICTION})])
        self.assertEqual(r.status, Status.COMPOSABLE)
        self.assertEqual([str(o) for o in r.ordering], ["early before late (step_prediction)"])

    def test_after_a_resource_nobody_owns_imposes_nothing(self):
        r = compose([_fixture("late", after={STEP_PREDICTION}), _fixture("other")])
        self.assertEqual((r.status, r.ordering), (Status.COMPOSABLE, ()))

    def test_ordering_cycle_conflicts(self):
        r = compose([_fixture("a", owns={TRUNK}, after={STEP_PREDICTION}),
                     _fixture("b", owns={STEP_PREDICTION}, after={TRUNK}),
                     _fixture("c")])
        self.assertEqual(r.status, Status.CONFLICT)
        self.assertEqual([(x.kind, x.subject) for x in r.reasons], [(ReasonKind.ORDERING_CYCLE, "a, b")])


class InputContractTest(absltest.TestCase):

    def test_a_candidate_must_be_feasible(self):
        impl = catalog.implementations().get("fractalyze-teacache")
        resolution = resolve(impl, SGLANG_QWEN, catalog.providers())
        infeasible = feasibility.check(impl, resolution, RuntimeContext(graph_replay=True), engines.evaluators())
        with self.assertRaisesRegex(ValueError, "FEASIBLE"):
            Candidate(impl, resolution, infeasible)

    def test_an_implementation_may_appear_only_once(self):
        with self.assertRaisesRegex(ValueError, "more than once"):
            compose(_catalog("fractalyze-teacache", "fractalyze-teacache"))

    def test_candidates_must_share_one_target(self):
        flux = _candidate(catalog.implementations().get("fractalyze-teacache"), Target("sglang", "Flux2Transformer2DModel", ModelRef("black-forest-labs/FLUX.2-klein-base-4B")))
        with self.assertRaisesRegex(ValueError, "different targets"):
            compose(_catalog("sglang-native-fp8-w8a8") + [flux])

    def test_resources_are_a_closed_vocabulary(self):
        with self.assertRaisesRegex(ValueError, "unknown resource"):
            ImplementationSpec("i", "t", frozenset(), owns=frozenset({"trunc"}))

    def test_cannot_come_after_a_resource_it_owns(self):
        with self.assertRaisesRegex(ValueError, "owns itself"):
            ImplementationSpec("i", "t", frozenset(), owns=frozenset({TRUNK}), after=frozenset({TRUNK}))

    def test_printable(self):
        text = str(compose(_catalog("fractalyze-teacache") + [_fixture("other-trunk-cache", owns={TRUNK})]))
        self.assertIn("fractalyze-teacache + other-trunk-cache: CONFLICT", text)
        self.assertIn("exclusive_resource: trunk", text)


if __name__ == "__main__":
    absltest.main()
