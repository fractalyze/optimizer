# ADR 0002: Scope: image diffusion on one GPU first, extensible later

Status: accepted · 2026-10-01

## Context

The first workloads worth optimizing are text-to-image diffusion models served
on a single GPU. The structure still has to extend later to multi-GPU and to
other workloads.

## Decision

- **V1 targets:** Qwen-Image and FLUX text-to-image, batch 1, on one GPU.
- **Extension points kept open, not built:** multi-GPU topology (Sol's
  `06_parallel_topology`), video models, and non-diffusion workloads. In
  practice no Core type may assume "image", "one GPU", or "denoising loop" in a
  way that would have to be removed later. Those assumptions belong in the
  diffusion-specific adapters.

## Consequences

- The quality gate can be image-based (LPIPS/SSIM over a prompt set) in V1.
- Topology search is out of scope for V1.

## Deferred

- Multi-GPU topology search. Revisit once a single-GPU line has shipped end
  to end.
