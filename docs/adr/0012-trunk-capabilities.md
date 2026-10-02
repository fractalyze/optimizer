# ADR 0012: Trunk control is three capabilities, defined at the trunk's edges

Status: accepted · 2026-10-02 · amends the capability list of [ADR 0010](0010-v1-capabilities-and-first-techniques.md)

## In short

[Experiment 002](../../experiments/002-trunk-control/README.md) ran one
TeaCache-style implementation on Qwen-Image-2.1 and FLUX.2-klein, with every
model difference inside a per-model Binding. That works, so the Binding stays.
ADR 0010's `trunk_control` and `signal_tap` become `trunk_observe`,
`trunk_output_override` and `signal_observe`. All three are defined only at the
trunk's two edges, where both models carry image tokens in and image tokens out.
Trunk control also needs a compile mode that leaves those edges outside
compiled code, and it cannot work under graph replay.

## Context

ADR 0010 expected TeaCache to need "trunk control" and "a signal tap" from a
Binding, but had not checked that the two models share anything a capability
could name. Experiment 002 read both trunks in SGLang and ran an observe-only
probe, an identity override and a one-step reuse through the same code.

| Question | Answer |
|---|---|
| Is there a trunk function to wrap? | No. In both models it is inline code inside `forward`. |
| Do the two trunks share anything? | Their edges: image tokens enter the first block and leave into the output norm, same shape. FLUX.2's text stream is internal to the trunk; Qwen-Image-2.1's text is a prefix cache. |
| Can one payload serve both? | Yes: the image-stream residual, exact in eager on every call. |
| Did the generic code ever need a model name? | No. The Bindings were about 30 lines each. |
| Does it survive compile? | With regional compile (blocks only): exact and cheap. With whole-model compile: graph breaks roughly triple, and on FLUX.2 an identity override changes the image. |
| Does it survive graph replay? | No. The hooks never run, and a requested reuse silently does nothing. |

## Decision

| Capability | Lets an implementation… | Provided by |
|---|---|---|
| `trunk_observe` | see every trunk invocation, on entry and on exit | EngineAdapter (mechanics) + Binding (where the edges are) |
| `trunk_output_override` | on entry, supply a payload instead of running the trunk; on exit, replace the result. A Binding may declare a call not overridable, and the implementation must accept that | EngineAdapter + Binding |
| `signal_observe` | read an opaque per-call tensor, the same shape for every call of one request and branch | Binding |

- **Payloads are made below the contract.** The adapter packs a trunk result as
  `output` or `residual` (exit − entry) and re-applies it to a later call's
  entry. The implementation stores payloads; it never looks inside them.
- **`request_local_state` is per request *and* per CFG branch.** FLUX.2 runs one
  trunk call per branch per step, and each branch needs its own payload.
- **A Binding owns:** the DiT forward, the block classes, the exit module, what a
  block returns when told not to compute, the entry tensor, the signal, and
  which calls may be overridden (Qwen-Image-2.1's first call fills the prefix
  cache and must always run).
- **Constraints.** A trunk implementation is `dynamic_in_forward`
  ([ADR 0009](0009-lifecycle-constraints.md)): refused under graph capture. It
  also requires that the trunk's edges stay outside compiled code. In SGLang
  that means regional compile, which Qwen-Image-2.1 offers and FLUX.2 does not.
  Without it, the adapter runs the target eager or refuses the implementation.
- **`engaged()` counts trunk blocks that did not run.** Neither a flag nor a
  log line proves a reuse happened.

## Consequences

- The separation ADR 0010 was built to test holds: step skip needs only the
  EngineAdapter, while TeaCache needs a Binding, and the Binding stays small and
  declarative.
- Per-model signal scales differ (median relative change 0.063 on FLUX.2 versus
  0.050 on Qwen-Image-2.1, with very different ranges). Thresholds belong in
  measurement history, as ADR 0008 already says.
- The compile-mode requirement is new and lands in the EngineAdapter: it must
  know which models support regional compile, read from the engine rather than
  copied.
- **Still open:** one DiT call that serves several requests or CFG branches.
  ComfyUI batches cond and uncond into one call and vLLM-Omni batches requests,
  so override and signal would have to work per batch slice. SGLang ran one
  request and one branch per call, so this was not tested. A code-reading check
  found the capability names themselves portable to both engines.

## Alternatives rejected

- **A generic `trunk.run(context)`.** There is no trunk call in either model, and
  the two trunks take different arguments. *Revisit if* an engine exposes the
  block stack as one callable.
- **A standard per-block interface.** Whole-trunk observe and override needed
  only each block's identity, which the Binding supplies. *Revisit when* a
  technique needs per-block decisions, through optional `block_access`
  ([ADR 0010](0010-v1-capabilities-and-first-techniques.md)).
- **The Binding computes the cache metric and returns a scalar.** The metric
  (relative L1 here) is the technique's choice, and a Binding should not know
  which technique is running. *Revisit if* a signal cannot be exposed without
  its layout mattering to the metric.
- **Naming the capabilities after TeaCache.** cache-dit, first-block caches and
  other reuse techniques need the same operations.
