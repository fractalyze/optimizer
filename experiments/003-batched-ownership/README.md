# Experiment 003: do trunk capabilities keep per-owner control when one DiT call serves several requests?

Status: done · 2026-10-02 · SGLang @ `8ca82118e` ([ADR 0003](../../docs/adr/0003-sglang-first-engine.md))
· outcome recorded in [ADR 0014](../../docs/adr/0014-batched-execution-owners.md)

## In short

SGLang's dynamic batching put two requests into one DiT call, on both
Qwen-Image-2.1 and FLUX.2-klein-base-4B. Experiment 002's TeaCache-style
implementation and both Bindings ran **unchanged**, imported rather than
copied; only the adapter learned to split a call into one owner per batch row.
The claim survives, with two limits:

- **Ownership holds.** Observation and an identity override on one request
  were bitwise identical to running without the probe. One request reusing
  its trunk left the other request's image bitwise identical. No state leaked
  into later requests.
- **A per-owner decision is honored, but saves compute only when every row
  agrees.** If one row reuses and the other computes, every block still runs
  and the reusing row's payload is spliced in at the exit: correct, and
  image-identical to a real skip, but no faster. Blocks are skipped only when
  every row of the call reuses.
- **SGLang does not carry per-request identity through a merged batch.** The
  merged request is renamed `dynamic_batch::<first id>` and the other requests'
  `extra` is dropped. The only per-row identity left is the seed, which is
  ambiguous when two merged requests share one.
- **The implementation needed no slice layout and `signal_observe` needed no
  ownership metadata**, because the adapter slices before the implementation
  sees anything. The batch axis is dim 0 at both trunk edges for both models:
  an engine convention, so no Binding was asked.

## What SGLang does with concurrent requests

Paths under `python/sglang/multimodal_gen/`, at the pinned commit.

| Step | Where | What happens |
|---|---|---|
| opt-in | `configs/pipeline_configs/qwen_image21.py:30`, `flux.py:490` | both pipelines allow dynamic batching for text-only requests |
| cap | `runtime/managers/dynamic_batch_admission.py` | with no `--batching-config` rules, `--batching-max-size` is the only cap |
| wait | `runtime/managers/scheduler.py:1126` | the head request waits up to `--batching-delay-ms` for compatible partners |
| compatible? | `scheduler.py:474`, `configs/sample/sampling_params.py:226-281` | sampling parameters must match, except fields marked `batch_sig_exclude`: prompt, seed, request id, output names |
| merge | `scheduler.py:901-915` | `deepcopy` of the first request; `prompt` becomes a list; seeds go to `extra["dynamic_batch_seeds"]`; request id becomes `dynamic_batch::<first id>`; other requests' `extra` is gone |
| rows | `runtime/pipelines_core/stages/input_validation.py:132-161` | `batch.seeds` gets one seed per output row, in merged order |
| submit | `runtime/entrypoints/diffusion_generator.py:319` | `generate()` sends prompt groups one after another, so batching needs concurrent callers; `scheduler_client.py:170` opens a socket per call, so threads work |

Row order is arrival order, not submission order: in the smoke run the second
thread's request became row 0.

## What was tested

| Layer | File | Changed from experiment 002? |
|---|---|---|
| implementation | `opt_trunk_probe/policy.py` | no, imported |
| Bindings | `opt_trunk_probe/bindings.py` | no, imported |
| SGLangAdapter | `opt_batch_probe/adapter.py` | yes: one owner per batch row |

The adapter, per DiT call:

1. reads the rows' owners from `batch.seeds`; if the row count does not match,
   every row is observe-only and nothing is overridden;
2. slices the trunk's entry and signal by row and asks the policy once per
   owner, each with its own `request_local_state` (owner × CFG branch);
3. if every owner returns a payload, suppresses all blocks; if none does, runs
   them; if some do, runs them and **splices** the payloads into those rows at
   the exit;
4. runs the policy's exit step per row and concatenates the rows back.

`run.py` submits each round's requests from separate threads released by a
barrier, with `batching_max_size=2` and `batching_delay_ms=2000`. Each model ran
two engines: without the plugin, and with it.

| Round | Rows | a | b |
|---|---|---|---|
| solo-a, solo-b | 1 | observe | observe |
| batch-observe (×2) | 2 | observe | observe |
| batch-a-identity | 2 | identity-residual | observe |
| batch-a-reuse | 2 | reuse at K | observe |
| batch-both-reuse | 2 | reuse at K | reuse at K |
| solo-a-reuse | 1 | reuse at K | |
| batch-observe-after | 2 | observe | observe |

FLUX.2: 50 steps, CFG 4.0, K = 25. Qwen-Image-2.1: 40 steps, CFG off, K = 20.
Prompts differ between a and b; seeds 101 and 202.

## Results

Every batched round ran as **one DiT call with 2 rows** per step (per CFG branch
on FLUX.2; cond and uncond stay separate calls). Image comparisons are on 8-bit
pixels; "identical" means every pixel matches.

| | FLUX.2 | Qwen-Image-2.1 |
|---|---|---|
| plugin vs no plugin: observe, identity (solo and batched) | identical | identical |
| identity-residual on a: every call exact | yes | yes |
| a's identity leaves b alone | identical | identical |
| **a's reuse leaves b alone** | **identical** | **identical** |
| a's reuse: splice vs skip | identical | identical |
| blocks skipped, batch-a-reuse (splice) | 0 | 0 |
| blocks skipped, batch-both-reuse (skip) | 50 (25 × 2 branches) | 32 |
| reuse quality vs batched observe, a / b | 43.2 / 50.6 dB | 45.3 / 49.3 dB |
| observe after control | identical | identical |
| per-owner signal, batched vs alone (max Δ relative L1) | 1.3e-4 | 5.3e-5 |

Client wall time per round on one RTX 5090 (a solo request includes the 2 s
batching wait):

| | FLUX.2 | Qwen-Image-2.1 |
|---|---|---|
| solo | 19.8 s | 15.6 s |
| batch, observe | 35.3 s | 27.2 s |
| batch, a reuses (splice) | 35.4 s | 27.4 s |
| batch, both reuse (skip) | 34.7 s | 26.7 s |

Two engine facts, independent of the probe:

- **Batching changes the images.** The same request batched versus alone
  differs: FLUX.2 41.5–46.9 dB, Qwen-Image-2.1 30.5–31.8 dB. Batched runs repeat
  exactly. Qwen's larger gap is plausibly the two prompts' different lengths in
  one prefix cache; not investigated.
- **Batching two requests saved no time** at 1024² on this GPU: a batch of two
  costs what two solo requests do without the wait.

## Findings

1. **The implementation stays layout-free.** The falsifier was "a per-slice
   override needs the implementation to know the layout". It did not: the same
   policy file ran per owner, and a payload applied to the wrong row would have
   shown (the CPU test checks exactly that).
2. **`signal_observe` needs no ownership metadata at the implementation.** The
   adapter hands each owner its own slice, so the signal stays a plain tensor
   per owner. Ownership is the adapter's job.
3. **Mixed decisions are honored, not cheap.** Suppressing blocks for some rows
   only would mean slicing every block input (modulation, rotary embeddings,
   prefix caches) by row: that is model-specific and was not attempted. Splice
   costs nothing extra and is exact. So a reuse decided by one owner saves
   compute only if every owner in the call decides the same.
4. **Engagement has to be per owner.** Under splice the decision is honored
   but no block is skipped, so a trunk-reuse speed claim for that owner is not
   engaged. Counting blocks per call would have hidden that.
5. **Per-row identity is an engine requirement.** SGLang keeps none, and the
   seed is a workaround that breaks when two merged requests share a seed.
6. **The batch axis needed no Binding.** Dim 0 at the trunk's entry and exit is
   SGLang's convention for both models.

## Limits

One engine, two requests, eager mode only, one seed and prompt pair per model.
CFG branches were never stacked as rows in SGLang, so ComfyUI's one-call
cond/uncond case is untested; the adapter refuses to override rows it cannot
attribute. vLLM-Omni's request batching is untested.

## Files

| File | What |
|---|---|
| `opt_batch_probe/adapter.py` | the per-row adapter |
| `opt_batch_probe/plugin.py` | registers the hooks at experiment 002's Binding seams |
| `run.py` | one engine configuration, rounds of concurrent requests |
| `analyze.py` | block counts per request, image comparisons, ownership pairs |
| `plans/` | the rounds per model, and an 8-step FLUX.2 smoke |
| `test_probe.py` | CPU contract tests with a fake model: `python -m pytest test_probe.py` |

Needs experiment 002's probe installed (`opt-trunk-probe`). Run bundles stay
outside the repository.
