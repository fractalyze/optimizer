# ADR 0002: Start with image diffusion on one GPU

Status: accepted · 2026-10-01

## In short

Version 1 optimizes text-to-image diffusion (Qwen-Image and FLUX) on a single
GPU. Multi-GPU, video and non-diffusion workloads are kept *possible* but not
built.

## Context

The first workloads worth optimizing are text-to-image diffusion models
served on one GPU. The structure still has to grow later to multi-GPU and to
other kinds of models.

## Decision

- **V1 targets:** Qwen-Image and FLUX text-to-image, batch size 1, one GPU.
- **Kept open, not built:** multi-GPU topology (Sol's parallel-topology
  family), video models, non-diffusion workloads.
- **The rule that keeps them open:** nothing in Core may assume "image",
  "one GPU" or "denoising loop" in a way that would have to be torn out
  later. Those assumptions belong in the diffusion-specific adapters.

## Consequences

- The V1 quality gate can be image-based: perceptual distance over a fixed
  prompt set.
- Searching multi-GPU layouts is out of scope for V1.

## Deferred

- **Multi-GPU topology search.** *Revisit once* one single-GPU line has
  shipped end to end.
