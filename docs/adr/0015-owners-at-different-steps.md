# ADR 0015: Owners in one model call may be at different steps; step state and batch composition are per owner and per measurement

Status: accepted · 2026-10-04 · extends [ADR 0014](0014-batched-execution-owners.md) to a second engine

## In short

[Experiment 004](../../experiments/004-omni-batched-ownership/README.md) ran
experiment 002's implementation, unchanged, on vLLM-Omni with Qwen-Image-2512.
vLLM-Omni batched two requests into one DiT call and, in step mode, put them
there at **different denoising steps**. ADR 0014 held without change: the
adapter mapped rows to owners, one owner's control left the other bitwise
identical, and blocks were skipped when both owners reused, even at different
steps. Three things are now explicit: `timestep_state` is provided per owner,
not per call; where per-row identity comes from is each adapter's own seam; and
batch composition is part of a measurement's conditions.

## Context

ADR 0014 was validated in one engine, where every row of a call shared a step.
Its open items were a second engine, and the question this experiment added:
does state stay per owner when owners in one call are at different steps?

| Question | Answer in vLLM-Omni |
|---|---|
| One DiT call for two requests? | Yes, in request mode (same step) and step mode (different steps) |
| Implementation or capability changed? | No; the policy was imported; a new adapter and a new Binding |
| Rows at different steps in one call? | Yes: a at step 6 next to b at step 2, each row with its own timestep |
| One owner's reuse leaves the other alone? | Yes, image and signal, both directions |
| Skip with owners at different steps? | Yes, 60 blocks, each row from its own payload |
| Per-row identity available? | Yes, but only in the model runner, not inside the DiT call |
| Does batching change numerics? | Yes, 28–32 dB for the same request alone versus batched |

## Decision

### `timestep_state` is provided per owner

The step index and timestep an implementation reads belong to its execution
owner, not to the model call. In SGLang they coincide; in vLLM-Omni's step mode
they do not. The EngineAdapter supplies them per owner: from the engine where it
tracks each request's step, or by counting the owner's own calls where every
row of a call shares one. The implementation already reads them from its own
call view, so it does not change.

### Row identity is read wherever the engine keeps it

Mapping rows to owners stays the EngineAdapter's job (ADR 0014), and so does
finding the identity. SGLang keeps none past the merge, so its adapter needs a
workaround or must refuse to batch controlled requests whose seeds collide. vLLM-Omni keeps request ids and steps
per row in its model runner, so its adapter reads them there and carries them
into the next DiT call. Neither engine needed a Binding or a new capability for
this. The feasibility condition of ADR 0014 is unchanged: if an adapter cannot
attribute every row, no row is overridden.

### Owner state lives until the engine retires the owner

`request_local_state` is released when the engine reports the request finished.
That signal may lag: vLLM-Omni's step mode reports it with the next step wave.
An adapter must not release earlier than the engine's signal, and must report
live owners so a lag stays visible.

### Batch composition is a measurement condition

Batching changes numerics in both engines. So an identity check, a quality
result or a comparison between two runs holds only at the same **batch
composition**: per model call, which owners shared it and at which steps. Core
records the composition with every measurement, and the identity check of
[ADR 0013](0013-feasibility-and-engagement.md) compares runs at matched
composition.

## Consequences

- No capability, contract field or Binding responsibility changes.
- An EngineAdapter for a batching engine must report, per call, the owners, each
  owner's step, and the composition.
- Reuse savings in step mode need owners whose reuse steps land in the same
  call, which depends on arrival times. Whether the optimizer should shape
  batches for that is a later question.

## What this rests on

**Runtime validated** (experiment 004, vLLM-Omni @ `68003cf6a`, Qwen-Image-2512,
eager with layerwise offload, two requests, 256 × 256, 8 steps, CFG off):
batching in request and step mode; probe versus no probe identical in request
mode; per-owner identity, reuse, splice and skip at matched composition, with
owners at different steps; adapter step equal to engine step on every call;
other owner's signal unchanged; state released when the engine retired the
request.

**Code reading only:** that FLUX.2-klein cannot batch in vLLM-Omni at this
commit, and that Qwen-Image-2.1 is not supported there.

**Open:**
- **CFG branches as stacked rows** (ComfyUI): untested in either engine.
- **Step-mode identity against a run without the probe:** not made, because
  step-mode composition depends on timing; identity was shown within the probe
  run at matched composition.
- **Compiled execution under batching:** both experiments ran eager.
- **A stable adapter seam in vLLM-Omni:** the probe hooked private runner
  methods.

## Alternatives rejected

- **A per-call step in `timestep_state`.** Wrong as soon as rows are at
  different steps.
- **Pass row identity into the Binding or the DiT call.** Identity is an engine
  fact, and both engines kept it, or failed to, above the model.
- **Compare images across batch compositions.** Batching alone moves them by
  28–32 dB; the comparison would measure the batch, not the technique.
