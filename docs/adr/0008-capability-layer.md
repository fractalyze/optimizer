# ADR 0008: Implementations depend on capabilities; EngineAdapter, ModelSpec and Binding are separate

Status: accepted · 2026-10-01 · supersedes [ADR 0005](0005-target-binding-architecture.md)

## In short

An implementation asks for abstract **capabilities**, never for a specific
place in SGLang's code. A target answers those requests from three separate
places: the **EngineAdapter**, the **ModelSpec** and the **Binding**. Each
capability is answered by whichever of them can provide it. The terms are
defined in the [architecture](../architecture.md#terms).

## Context

ADR 0005 went straight from implementation to an "engine × model binding".
Read literally, it said two things that would have hurt us:

- **There is no standard layer between implementations and engines.** Every
  "generic" implementation would then really be SGLang code, and a vLLM-Omni
  or ComfyUI target would have nothing to plug into.
- **The binding replaces engine- and model-specific adaptation.** In reality
  most knowledge is engine-wide (the denoising loop) or model-wide (CFG
  style). Only a little is specific to the pair.

## Decision

1. **Capabilities and seams are different things.** A capability is *what* an
   implementation needs ("control each step"). A seam is *where* one engine
   provides it (`DenoisingStage._run_denoising_step`). Implementations only
   name capabilities; seams stay private to the target.
2. **A target has three parts, and none replaces another:**
   - EngineAdapter: the engine, the same for every model;
   - ModelSpec: the model, the same in every engine;
   - Binding: the glue for one pair.
3. **Where a fact goes** is decided by the
   [ownership rule](../architecture.md#where-knowledge-belongs).
4. **Capability resolution is an explicit step.** Step skip resolves entirely
   in the EngineAdapter; TeaCache needs a Binding as well.
5. **Native and generic implementations share one contract.** A native
   implementation asks for an engine feature; a generic one asks for standard
   capabilities. The default
   [selection order](../architecture.md#choosing-between-implementations)
   prefers a native implementation that has passed the gate on this target.
   It is only a default: SGLang ships features that look present but never
   run for our models (its TeaCache), so measurement decides.
6. **Measured knowledge is not model knowledge.** Thresholds, coefficients
   and sensitive layers go in measurement history, never in ModelSpec.

## Consequences

- **A new engine** means one EngineAdapter plus one Binding per model.
  Implementations and ModelSpecs stay as they are, provided the engine offers
  the capabilities they need.
- **Which targets an implementation supports** is computed, never listed by
  hand.

## Alternatives rejected

- **Let implementations call seams directly.** Simpler on day one, but every
  generic implementation becomes engine code.
- **A `supported_targets` field.** It would repeat what resolution computes,
  and drift from it.
- **A `Profile` type for measured knowledge now.** Core already records
  results. *Revisit when* two different consumers need the same query over
  them.
