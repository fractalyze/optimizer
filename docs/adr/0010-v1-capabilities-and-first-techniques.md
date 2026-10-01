# ADR 0010: V1 capabilities, optional block access, first three techniques

Status: accepted · 2026-10-01 · supersedes [ADR 0007](0007-v1-seams-and-first-techniques.md)

## Context

ADR 0007 listed V1 *seams* and rejected per-block as a seam. The finding
behind it holds: Qwen-Image, FLUX.1 and FLUX.2 have three incompatible trunk
shapes. But the conclusion was worded too strongly. Selective block caching,
block skipping, block-wise precision and layer-wise kernel selection all need
block-level access, and some later technique will want it.

## Decision

- V1 **capabilities**, with their SGLang providers and seams listed in
  [architecture](../architecture.md#capabilities-in-v1):
  - common: `step_control`, `timestep_state`, `request_local_state`
  - common, provided per Binding: `trunk_control`, `signal_tap`
  - per engine: `engine_feature.*`, `compile_control`
  - optional: `block_access`, `linear_access`, `attention_access`
- **A portable, standardized per-block seam is not justified for V1.**
  Block-level access remains an optional, target-specific capability. A
  Binding may expose its blocks in its own signature, and must never be forced
  into one fake universal block signature.
- The first three techniques are unchanged: generic step skip, generic
  TeaCache, native FP8. Their resolution is worked through in
  [architecture](../architecture.md#capability-resolution-examples).
- Models: Qwen-Image and FLUX.2-klein. FLUX.1 is deferred.

## Consequences

- The first build step is a falsification experiment, not framework code.
  The experiments are ordered in the
  [investigation's open items](../architecture-investigation.md#open-items).

## Deferred

- **Implementing `block_access`.** Revisit when a chosen technique requires
  it, starting with a generic block-residual cache.
- **NVFP4 and selective FP8** (via `linear_access`). Revisit after native FP8
  passes the engagement check.
