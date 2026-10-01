# ADR 0003: SGLang diffusion is the first serving engine; engines are pluggable

Status: accepted · 2026-10-01

## Context

Speedups only matter against a real serving baseline. SGLang's
`multimodal_gen` runtime is what Sol-Engine builds on and what
earlier experiments measured against. vLLM(-Omni) and
ComfyUI as serving engines later.

## Decision

- SGLang diffusion is the first and only engine implemented in V1.
- The engine is behind an adapter boundary. Nothing above that boundary may
  import SGLang. vLLM-Omni and ComfyUI must be addable as new adapters without
  changing Core or the techniques that do not depend on an engine.
- Code reading uses upstream `sgl-project/sglang` main at `8ca82118e`
  (2026-09-24). Forks are not the reference.

## Consequences

- SGLang runs the model in a separate worker process. Anything that runs at
  model load or during the forward pass has to be injected into *that*
  process, not into the process that launches it. SGLang provides an official mechanism for this, the `sglang.multimodal_gen.plugins` entry-point hooks; see the
  [investigation](../architecture-investigation.md).

## Rejected

- **diffusers as the first engine.** It is the easiest to instrument, but a
  speedup over diffusers says little about a production server. It may still
  be useful as a reference implementation.
