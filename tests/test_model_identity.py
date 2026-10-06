"""Contract tests for model identity: Bindings follow the engine's model class,
the checkpoint is the user's input, and the class is read from the checkpoint."""

import json
import pathlib
import shutil
import tempfile

from absl.testing import absltest

from optimizer import catalog, engines
from optimizer.core import feasibility
from optimizer.core.composition import Candidate, compose
from optimizer.core.feasibility import RuntimeContext
from optimizer.core.plan import plan
from optimizer.core.resolver import Status, resolve
from optimizer.core.specs import ModelRef, Target, TechniqueConfig
from optimizer.engines.sglang import model as sglang_model
from optimizer.engines.sglang import plan as sglang_plan

FLUX_DIT = "Flux2Transformer2DModel"


def _teacache_plan(target):
    impl = catalog.implementations().get("fractalyze-teacache")
    resolution = resolve(impl, target, catalog.providers())
    candidate = Candidate(impl, resolution,
                          feasibility.check(impl, resolution, RuntimeContext(), engines.evaluators()))
    return plan(compose([candidate]), [candidate], [TechniqueConfig.of("teacache", threshold=0.1)],
                catalog.techniques())


class ArchitectureTest(absltest.TestCase):

    def test_every_checkpoint_of_one_class_shares_its_binding(self):
        teacache = catalog.implementations().get("fractalyze-teacache")
        for checkpoint in ("black-forest-labs/FLUX.2-klein-base-4B", "/models/my-finetuned-flux2"):
            r = resolve(teacache, Target("sglang", FLUX_DIT, ModelRef(checkpoint)), catalog.providers())
            self.assertEqual(r.status, Status.RESOLVED)
            self.assertEqual(r.providers["trunk_observe"], "SGLangFlux2Binding")

    def test_a_class_without_a_binding_is_unsupported_not_mismatched(self):
        teacache = catalog.implementations().get("fractalyze-teacache")
        r = resolve(teacache, Target("sglang", "SomeOtherTransformer2DModel", ModelRef("x")), catalog.providers())
        self.assertEqual(r.status, Status.UNSUPPORTED)

    def _tempdir(self):
        path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, path)
        return pathlib.Path(path)

    def test_architecture_is_read_from_the_checkpoint_config(self):
        snapshot = self._tempdir()
        (snapshot / "transformer").mkdir()
        (snapshot / "transformer" / "config.json").write_text(json.dumps({"_class_name": FLUX_DIT}))
        self.assertEqual(sglang_model.architecture(snapshot), FLUX_DIT)

    def test_a_checkpoint_without_a_transformer_config_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not a diffusers checkpoint"):
            sglang_model.architecture(self._tempdir())


class CheckpointTest(absltest.TestCase):

    def _flux(self, checkpoint, revision=None):
        return _teacache_plan(Target("sglang", FLUX_DIT, ModelRef(checkpoint, revision)))

    def test_checkpoint_and_revision_are_part_of_the_server_key(self):
        base = self._flux("black-forest-labs/FLUX.2-klein-base-4B")
        self.assertNotEqual(base.server, self._flux("/models/my-finetuned-flux2").server)
        self.assertNotEqual(base.server, self._flux("black-forest-labs/FLUX.2-klein-base-4B", "a3b4f48").server)

    def test_any_checkpoint_is_lowered_as_given(self):
        server, _ = sglang_plan.validated().lower(self._flux("/models/my-finetuned-flux2", "a3b4f48"))
        self.assertEqual(server.model_path, "/models/my-finetuned-flux2")
        self.assertIn(("revision", "a3b4f48"), server.server_kwargs)


if __name__ == "__main__":
    absltest.main()
