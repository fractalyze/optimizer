# ADR 0012: Trunk control is three capabilities, defined at the trunk's edges

Status: accepted · 2026-10-02 · amends the capability list of [ADR 0010](0010-v1-capabilities-and-first-techniques.md)

## In short

[Experiment 002](../../experiments/002-trunk-control/README.md) ran one
TeaCache-style implementation on Qwen-Image-2.1 and FLUX.2-klein, with every
model difference inside a small per-model Binding. That works, so the Binding
stays, and it must stay small. ADR 0010's `trunk_control` and `signal_tap`
become `trunk_observe`, `trunk_output_override` and `signal_observe`, defined
only at the trunk's two edges, where both models carry image tokens in and
image tokens out. `request_local_state` becomes `execution_local_state`,
because its owner is not always a whole request. When these capabilities can
actually be used, and how a run proves it used them, is
[ADR 0013](0013-feasibility-and-engagement.md).

## Context

ADR 0010 expected TeaCache to need "trunk control" and "a signal tap" from a
Binding, but had not checked that the two models share anything a capability
could name. Experiment 002 read both trunks in SGLang and ran an observe-only
probe, an identity override and a one-step reuse through the same code.

| Question | Answer |
|---|---|
| Is there a trunk function to wrap? | No. In both models it is inline code inside `forward`. |
| Do the two trunks share anything? | Their edges: image tokens enter the first block and leave into the output norm, same shape. FLUX.2's text stream lives inside the trunk; Qwen-Image-2.1's text is a prefix cache. |
| Can one payload serve both? | Yes: the image-stream residual, exact in eager on every call. |
| Did the generic code ever need a model name? | No. The Bindings were about 30 lines each. |
| Is state per request enough? | No. FLUX.2 runs one trunk call per CFG branch, and each branch needs its own payload. |

## Decision

### Capabilities

| Capability | Kind ([ADR 0013](0013-feasibility-and-engagement.md)) | Lets an implementation… | Provided by |
|---|---|---|---|
| `trunk_observe` | observe | see every trunk invocation, on entry and on exit | EngineAdapter (mechanics) + Binding (where the edges are) |
| `trunk_output_override` | decide | on entry, supply a payload instead of running the trunk; on exit, replace the result. A call may be declared not overridable, and the implementation must accept that | EngineAdapter + Binding |
| `signal_observe` | observe | read the model's cache signal for this call | Binding |
| `execution_local_state` (was `request_local_state`) | — | keep state owned by one logical execution item, never shared across items, gone when the item ends | EngineAdapter |

- **Payloads are made below the contract.** The adapter packs a trunk result as
  `output` or `residual` (exit − entry) and re-applies it to a later call's
  entry. The implementation stores payloads and never looks inside them.
- **`signal_observe` returns the raw signal, not a metric.** The metric
  (relative L1 here) is the technique's choice, and a Binding should not know
  which technique is running. Its current shape, one opaque tensor per call,
  is **provisional**: see Open.
- **The execution item is the adapter's call.** For a step-level capability in
  SGLang it is one request; for a trunk-level one it is one request × one CFG
  branch. The name drops "request" because ComfyUI has no request (one sampling
  run plays that role) and because one request may hold several items.

### What a Binding may contain

A Binding answers "where does this model's logical part live in this engine,
and what is true only here". In experiment 002 that was, per model:

- the DiT forward and the block classes that make up the trunk;
- the trunk's exit module;
- what a block returns when told not to compute (its identity);
- the tensor entering the trunk;
- the cache signal and how to compute it from the block's inputs;
- which calls may not be overridden (Qwen-Image-2.1's first call fills the
  prefix cache, so it must always run).

A Binding must **not** contain: worker or request lifecycle, generic request or
branch state, payload arithmetic, compile or graph-capture policy, engagement
counting, or anything a second model in the same engine would repeat. If a
Binding starts growing those, it is becoming a god object, and the code belongs
in the EngineAdapter. The probe kept to this: its payload forms were first
written in the Binding file and were moved to the adapter when the rule was
written down, with no change in behavior.

## Consequences

- The separation ADR 0010 was built to test holds: step skip needs only the
  EngineAdapter, TeaCache needs a Binding, and the Binding stays declarative.
- Signal scales differ per model (median relative change per call 0.063 on
  FLUX.2, 0.050 on Qwen-Image-2.1, with very different ranges). Thresholds
  belong in measurement history ([ADR 0008](0008-capability-layer.md)).
- Trunk capabilities are not usable in every execution mode. Which modes, and
  how a run proves it used them, is [ADR 0013](0013-feasibility-and-engagement.md).

## What this rests on

**Runtime validated** (SGLang @ `8ca82118e`, one prompt and seed per model):

- One generic implementation observed, replaced and skipped the trunk of
  Qwen-Image-2.1 and FLUX.2-klein; it never named a model or engine.
- The Bindings isolated every model difference that came up.
- Observation and identity override were bitwise exact in eager mode, and under
  regional compile on Qwen-Image-2.1.
- Reuse at one step suppressed every block of that call (32 on Qwen, 25 per
  branch on FLUX.2), and the rest of the request ran correctly.
- Separate payloads per CFG branch were needed and worked.
- Whole-model compile changed FLUX.2's output under an identity override, and
  graph replay bypassed the hooks entirely.
- Counting blocks that did not run detected engagement, including its absence.

**Code reading only** (vLLM-Omni `bbee488`, ComfyUI `1b883be`):

- Both engines have plausible seams for every capability here. vLLM-Omni's own
  TeaCache already uses per-model "extractors", a Binding in all but name.
- ComfyUI has no request object and no step index; it has a sampling run and
  sigmas.

**Open:**

- **One model call that serves several execution items.** ComfyUI runs cond and
  uncond in one call; vLLM-Omni batches requests. Override, payload and signal
  would then act per batch slice. Whether `signal_observe` must return a
  structured view that says which slice belongs to which item is unresolved, so
  the single-tensor shape is not yet a permanent contract.
- Whether one payload form keeps serving models whose trunk exit is not a
  single image stream.

## Alternatives rejected

- **A generic `trunk.run(context)`.** There is no trunk call in either model, and
  the two trunks take different arguments. *Revisit if* an engine exposes the
  block stack as one callable.
- **A standard per-block interface.** Whole-trunk observe and override needed
  only each block's identity, which the Binding supplies. *Revisit when* a
  technique needs per-block decisions, through optional `block_access`
  ([ADR 0010](0010-v1-capabilities-and-first-techniques.md)).
- **The Binding returns a computed metric instead of the signal.** It would tie
  the Binding to one technique's metric. *Revisit if* batching makes a raw
  per-call signal unworkable.
- **A multi-dimensional state manager (request × branch × slice).** One owner
  per item, chosen by the adapter, covered every case seen. *Revisit when* a
  batched call is tested.
- **Naming capabilities after TeaCache.** cache-dit, first-block caches and other
  reuse techniques need the same operations.
