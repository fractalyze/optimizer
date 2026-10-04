# Experiment 004: does per-owner trunk control survive vLLM-Omni's batching, including requests at different steps in one call?

Status: done · 2026-10-04 · vLLM-Omni @ `68003cf6a` (vLLM 0.30.0) · outcome
recorded in [ADR 0015](../../docs/adr/0015-owners-at-different-steps.md)

## In short

vLLM-Omni put two Qwen-Image-2512 requests into one DiT call, and in **step
mode** put them there **at different denoising steps**: request a at step 4
next to request b at step 0, each row with its own timestep. Experiment 002's
policy ran unchanged, imported; a new vLLM-Omni adapter and a new
(vLLM-Omni × Qwen-Image) Binding were all that changed. The claim survives a
second engine and the harder case:

- **Ownership holds across engines and steps.** Without control, outputs were
  bitwise identical to running without the probe. One request's identity
  override or reuse left the other request's image, and its signal, bitwise
  identical, whether the two shared a step or not.
- **Each owner keeps its own step.** The adapter's step per owner matched the
  engine's per-row step on every call, and a reuse fired at the owner's own
  step, whatever step the other row was at.
- **Skipping works across steps.** When both rows decided to reuse in the
  same call, a at step 6 and b at step 2, all 60 blocks were skipped and each
  row was rebuilt from its own payload. As in SGLang, a call where only one
  row reuses splices instead: exact, image-identical to a skip, no saving.
- **vLLM-Omni keeps per-row identity, but only at the model runner.** Nothing
  per row reaches the transformer, so the adapter reads the rows' requests and
  steps at the runner and carries them to the next DiT call. Unlike SGLang, no
  workaround (seeds) was needed.
- **Nothing model- or engine-specific reached the implementation**, and the
  Binding has the same five answers as experiment 002's, with no batch logic.

## Call path, traced at the pinned commit

Paths under `vllm_omni/diffusion/` unless noted. Qwen-Image-2512 was not assumed
to be Qwen-Image; this is how it resolves.

| Step | Where | What happens |
|---|---|---|
| pipeline class | `data.py:1586`, `registry.py:38` | `model_index.json` `_class_name` = `QwenImagePipeline` → `models/qwen_image/pipeline_qwen_image.py` |
| batching allowed | `pipeline_qwen_image.py:275,280` | `supports_request_batch = True`, `supports_step_execution = True` |
| scheduler | `diffusion_engine.py:366`, `data.py:1106,1112,1126` | request scheduler by default; step scheduler with `step_execution=True`; `max_num_seqs` defaults to 1 |
| request mode | `worker/diffusion_model_runner.py:760,865` → `pipeline_qwen_image.py:1007` → `models/qwen_image/cfg_parallel.py:28` | the runner hands a list of requests to `forward`; `diffuse` runs every step for all rows together |
| step mode | `diffusion_model_runner.py:1357,1142,1397` → `pipeline_qwen_image.py:925` | each step, `_prepare_batch_inputs` builds an `InputBatch` from the live requests; `denoise_step` runs one DiT call |
| per-row step | `worker/input_batch.py:347,392,594` | latents gathered as rows on dim 0; each request's own timestep expanded to its rows; `request_ids` and `states` (with `step_index`) per request |
| inside the DiT | `forward_context.py:26-83` | no per-row request, seed or step |
| trunk | `models/qwen_image/qwen_image_transformer.py:1145,1264,1279` | `transformer_blocks` loop of dual-stream blocks returning `(text, image)`; exit `norm_out(hidden_states, temb)`, a shared diffusers class |
| CFG | `distributed/cfg_parallel.py:155`, `pipeline_qwen_image.py:716` | separate calls per branch; off without a negative prompt, as here |
| compile | `data.py:975,978` | regional torch.compile by default; this experiment used `enforce_eager=True` |
| plugins | `plugins/__init__.py:14`, `worker/diffusion_worker.py:1634` | `vllm_omni.general_plugins` entry points load before the model is built |

FLUX.2-klein, the other model of experiments 002 and 003, cannot batch here
(`pipeline_flux2_klein.py:184` declares neither flag; `diffusion_engine.py:344`
rejects `max_num_seqs > 1`), and Qwen-Image-2.1 is not supported. The 20B
transformer does not fit a 32 GB GPU in bf16, so it ran with
`enable_layerwise_offload=True`.

## What was tested

| Layer | File | Changed from experiment 002? |
|---|---|---|
| implementation | `opt_trunk_probe/policy.py` | no, imported |
| vLLM-Omni adapter | `opt_omni_probe/adapter.py` | new |
| vLLM-Omni × Qwen-Image Binding | `opt_omni_probe/bindings.py` | new; same five answers as the SGLang Bindings |

The adapter records row owners at the runner: in request mode from the
request list (one step for all rows; it counts each owner's own calls), in step
mode from the `InputBatch` (each row's request and engine step). In the next
DiT call it splits the trunk's entry and signal by row, asks the policy once
per owner with that owner's state, and skips blocks only if every owner
overrides; otherwise it splices. Owner state is dropped when the runner
retires the request.

Batching changes numerics in vLLM-Omni too (the same request alone versus
batched: 28–32 dB), so two images are compared bitwise only when their **batch
composition** matched: per DiT call, the owner's own step and the steps of the
other rows. To make step-mode composition reproducible, request b is submitted
only once the probe has logged a passing step 2; it joined at a's step 4 in
every round.

Qwen-Image-2512, 256 × 256, 8 steps, CFG off, two prompts with seeds 101 and
202. Three engines: request mode with and without the probe, step mode with
it.

| Round (request mode) | a | b |
|---|---|---|
| solo-a, solo-a-reuse | observe; reuse at K = 3 | |
| pair-observe (×2), pair-observe-after | observe | observe |
| pair-a-identity | identity-residual | observe |
| pair-a-reuse | reuse at K | observe |
| pair-both-reuse | reuse at K | reuse at K |

| Round (step mode, b joins after a's step 2) | a | b |
|---|---|---|
| stagger-observe (×2), stagger-observe-after | observe | observe |
| stagger-a-identity | identity-residual | observe |
| stagger-a-reuse | reuse at its step 5, next to b at step 1 | observe |
| stagger-b-reuse | observe | reuse at its step 2, next to a at step 6 |
| stagger-both-reuse-d2 / d3 / d4 | reuse at step 4 / 5 / 6 | reuse at step 2 |

The three both-reuse rounds bracket the offset between the requests; d4 is the
one where both reuses land in the same call.

## Results

| | request mode | step mode |
|---|---|---|
| rows per DiT call | 2, same step | 1 or 2; while shared, a at step s + 4 next to b at step s |
| probe vs no probe: observe, identity | identical | (no-probe step run not made; see Limits) |
| repeated round | identical | identical, same composition every round |
| identity-residual on a: every call exact | yes | yes |
| a's identity leaves b alone | identical | identical |
| **a's reuse leaves b alone** (image) | identical | identical |
| **b's reuse leaves a alone** (image) | — | identical |
| other owner's signal unchanged by a reuse | — | identical, both directions |
| adapter step == engine step, every call | — | yes |
| one reuses: blocks skipped | 0 (splice) | 0 (splice) |
| both reuse in one call: blocks skipped | 60, both at step 4 | 60, **a at step 6 and b at step 2** |
| splice vs skip, same reuse | identical | identical |
| observe after control | identical | identical |
| owner states retired | all | all but the last request, see below |

One step mode call from `stagger-both-reuse-d4`, as logged by the worker:

| row | owner | its step | its timestep | decision | call |
|---|---|---|---|---|---|
| 0 | `stagger-both-reuse-d4.a-96e96d2b` | 6 | 0.219 | reuse | skip, 60 blocks |
| 1 | `stagger-both-reuse-d4.b-816754e6` | 2 | 0.797 | reuse | skip, 60 blocks |

Reuse quality is not the point at 8 steps (one reused step costs 17–28 dB here)
and was not tuned.

## Findings

1. **The abstraction held on a second engine and with mixed steps.** The
   implementation never saw a layout, a model or an engine. `signal_observe`
   needed no ownership metadata: each owner got its own slice.
2. **`timestep_state` is per owner, not per call.** In SGLang every row of a
   call shared a step; here they do not. The policy was already reading the
   step from its own item, so nothing changed for it, but the adapter must
   provide the step per row.
3. **Per-row identity lives at different depths in different engines.**
   SGLang drops it when merging (experiment 003); vLLM-Omni keeps it, but only
   in the model runner, outside the DiT call. Both are adapter seams; neither
   needed a Binding or a new capability.
4. **State retirement follows the engine's signal.** In step mode a request's
   state is released when the scheduler reports it finished, which comes with
   the *next* step wave, so the last request's state lived until shutdown.
   Harmless here, but a long-running engine holds it until more work arrives.
5. **Batch composition is a measurement condition.** Since batching changes
   numerics in both engines, a quality or identity result is comparable only
   at the same composition. That is something Core must record with a
   measurement.
6. **The Binding stayed small and batch-free.** Entry, signal, block identity,
   exit and "always overridable" were the whole of it, as in experiment 002.

## Limits

One engine version, one model, two requests, eager mode with layerwise
offload, CFG off, 256 × 256 and 8 steps. No step-mode run without the probe was
made, because its batch composition depends on timing; step-mode identity is
shown within the probe run, at matched composition. CFG branches as stacked
rows (ComfyUI) are still untested. The adapter's runner hooks name private
runner methods (`_execute_request_list`, `_prepare_batch_inputs`,
`_cleanup_finished_step_requests`), which is a spike's shortcut, not a seam to
build on.

## Files

| File | What |
|---|---|
| `opt_omni_probe/adapter.py` | the vLLM-Omni adapter: row owners at the runner, per-owner trunk control |
| `opt_omni_probe/bindings.py` | the vLLM-Omni × Qwen-Image Binding |
| `opt_omni_probe/plugin.py` | the `vllm_omni.general_plugins` entry point; inert unless `OPT_OMNI_PROBE=1` |
| `run.py` | one engine configuration; rounds of requests, joined by delay or after another request's step |
| `analyze.py` | per-request rows, steps and decisions; comparisons at matched batch composition |
| `plans/` | request-mode and step-mode rounds, and the two smokes |
| `test_probe.py` | CPU contract tests with a fake model: `python -m pytest test_probe.py` |

Needs experiment 002's probe installed (`opt-trunk-probe`). Run bundles stay
outside the repository.
