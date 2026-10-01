# ADR 0008: Capability layer, three target responsibilities, capability resolution

Status: accepted · 2026-10-01 · supersedes [ADR 0005](0005-target-binding-architecture.md)
Definitions and examples: [architecture](../architecture.md).

## Context

ADR 0005 went from Implementation straight to Target. Read literally, it
dropped the standard seam layer, and it presented Binding(engine × model) as a
replacement for model- and engine-specific adaptation. Both readings would
have made generic implementations SGLang code, and would have left nothing
for a vLLM-Omni or ComfyUI target to resolve against.

## Decision

1. **Implementations depend on capabilities, never on seams.** A capability is
   an abstract operation ("observe each step and decide whether to compute the
   prediction"). A seam is where one target exposes it
   (`DenoisingStage._run_denoising_step`). Seams are private to the target.
2. **A target has three responsibilities, and none replaces another.**
   - EngineAdapter: how the engine works, the same for all models.
   - ModelSpec: what the model is, the same in all engines.
   - Binding: where this model lives in this engine; glue between the other two.
3. **The ownership rule** in [architecture](../architecture.md#ownership-rule)
   decides where a fact goes. Facts the engine itself encodes per model are
   read from the engine by the EngineAdapter.
4. **Capability resolution is a first-class step.** Each required capability
   is resolved by whichever layer provides it. Step skip resolves entirely in
   SGLangAdapter. TeaCache splits between SGLangAdapter and a Binding. Native
   FP8 resolves to an `engine_feature.*` capability on the EngineAdapter.
5. **Native and generic implementations share one contract.** Native ones
   require `engine_feature.*`; generic ones require standard capabilities.
   Selection defaults to: validated native, then generic, then unvalidated
   native. The optimizer may still measure both.
6. **Empirical knowledge is not ModelSpec.** Thresholds, coefficients and
   sensitive layers live in Core's measurement history and reach
   implementations as parameters.

## Consequences

- A new engine means an EngineAdapter plus a Binding per model.
  Implementations and ModelSpecs are unchanged as long as the engine provides
  the capabilities they require.
- Per-target support is computed by resolution, so nothing lists supported
  targets by hand.

## Rejected

- **Implementations calling seams directly.** Simpler on day one, but every
  generic implementation becomes engine code.
- **A separate `supported_targets` field.** It would duplicate what resolution
  computes, and drift from it.
- **A `Profile` type now.** Core's trace and frontier already hold measured
  results. Revisit once two consumers need the same empirical query.
