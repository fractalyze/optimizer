# ADR 0003: SGLang is the first engine; engines are pluggable

Status: accepted · 2026-10-01

## In short

V1 runs on SGLang's diffusion runtime. Nothing above the engine adapter may
depend on SGLang, so vLLM-Omni and ComfyUI can be added later as new adapters.

## Context

A speedup only matters against a real serving baseline. SGLang's diffusion
runtime is what Sol-Engine builds on and what earlier experiments measured
against. vLLM-Omni and ComfyUI are wanted as engines later.

## Decision

- SGLang is the first and only engine in V1.
- **Boundary rule:** only the engine adapter may import SGLang. Adding
  vLLM-Omni or ComfyUI must not require changes to Core, or to the techniques
  that do not depend on an engine.
- **Pinned source:** code is read at upstream `sgl-project/sglang` main
  `8ca82118e` (2026-09-24), not at any fork. Every `file:line` in these docs
  refers to that commit. Moving the pin is a deliberate change to this ADR.

## Consequences

- SGLang runs the model in a separate GPU worker process. Anything that acts
  at model load or during the forward pass has to run *inside that process*.
  SGLang has an official way to do this, plugins in the
  `sglang.multimodal_gen.plugins` entry-point group; see the
  [investigation](../architecture-investigation.md#3-which-interception-points-are-real).

## Alternatives rejected

- **diffusers as the first engine.** It is the easiest to instrument, but a
  speedup over diffusers says little about a production server. It may still
  serve as a reference implementation.
