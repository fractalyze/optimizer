# ADR 0009: Implementations declare constraints; engine lifecycles stay in the adapter

Status: accepted · 2026-10-01 · supersedes [ADR 0006](0006-lifecycle-phases.md)

## In short

An implementation states three facts about itself: does it change the model,
does it make decisions inside the model's forward pass, and does it keep
per-request state. Each engine adapter turns those facts into the right
moment in its own lifecycle. SGLang's eight lifecycle stages are SGLang's
business.

## Context

ADR 0006 made SGLang's lifecycle stages, from `launch` to
`request_finalize`, the vocabulary every implementation installs into. Those
stages describe *SGLang*: it compiles the model while building the pipeline,
and records graphs during warmup. vLLM-Omni, ComfyUI or TensorRT need not
work that way. A generic implementation, like our TeaCache, should not name
one engine's stages.

## Decision

| Constraint | Meaning | Who sets it | In SGLang this means |
|---|---|---|---|
| `mutates_model` | Changes the model's modules or forward. It must be in place before compiling or graph recording, and changing it needs a fresh model. | FP8 / NVFP4, fused QKV, a trunk wrapper | install at construction or right after load, before the pipeline is built |
| `dynamic_in_forward` | Makes data-dependent Python decisions *inside* the model's forward pass | TeaCache, Spectrum | refused with graph capture; causes graph breaks under torch.compile |
| `request_state` | Needs setup and cleanup for every request | step skip, TeaCache | attach at request start, reset at request end |

SGLang's eight stages are described in the
[investigation](../architecture-investigation.md#7-sglangs-lifecycle), as
SGLang's lifecycle. Only `SGLangAdapter` uses them.

## Consequences

- **Decisions made in the denoising loop** (step skip) are not
  `dynamic_in_forward`. They should survive graph capture, because graphs
  replay once per transformer call. That is still unverified.
- **Adding an engine** means mapping three flags, not reconciling eight stage
  names.

## Alternatives rejected

- **Use SGLang's stage names as the shared vocabulary.** The type is a bit
  smaller, but two of the first three techniques are generic, and it would
  tie them to one engine from day one.
- **A full cross-engine lifecycle model.** No second engine has been studied
  yet. *Revisit when* a second EngineAdapter is written.
