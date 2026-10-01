# ADR 0010: V1 capabilities, optional block access, first three techniques

Status: accepted · 2026-10-01 · supersedes [ADR 0007](0007-v1-seams-and-first-techniques.md)
· step control amended by [ADR 0011](0011-split-step-control.md)

## In short

V1 has a small set of common capabilities, and block-level access is an
*optional* capability rather than a rejected idea. The design is validated
with three techniques: step skip, TeaCache and FP8.

## Context

ADR 0007 rejected per-block access because Qwen-Image, FLUX.1 and FLUX.2
chain their blocks in three incompatible ways. That finding still holds. But
the wording went too far: selective block caching, block skipping, per-block
precision and per-layer kernel choice all need *some* block-level access, and
a later technique will want it.

## Decision

- **Capabilities.** V1 uses the capabilities listed in the
  [architecture](../architecture.md#capabilities-in-v1). Some are common
  (step control, trunk control). Some are per engine (built-in features). One
  group is optional: block, linear and attention access.
- **Blocks.** A portable, standard per-block interface is not justified for
  V1. A Binding *may* expose its blocks, in its own signature. No target is
  ever forced into a fake universal block signature.
- **Three validation techniques,** each testing a different part of the
  design (worked through in the
  [architecture](../architecture.md#three-techniques-resolved)):
  - **step skip** (ours): needs only the engine;
  - **TeaCache** (ours): needs the engine *and* a per-model Binding;
  - **FP8** (SGLang's own): just switches on an engine feature.
- **Models:** Qwen-Image and FLUX.2-klein. FLUX.1 is deferred.

## Consequences

- The next step is an experiment that could prove this design wrong, not
  framework code. The experiments are ordered in the
  [investigation](../architecture-investigation.md#open-items).

## Deferred

- **Implementing `block_access`.** *Revisit when* a chosen technique needs
  it; the likely first one is a generic block cache.
- **NVFP4, and FP8 that keeps some layers in full precision.** *Revisit
  after* native FP8 passes its "did it really run" check.
