# ADR 0009: Implementations declare lifecycle constraints; engine phases stay in the adapter

Status: accepted · 2026-10-01 · supersedes [ADR 0006](0006-lifecycle-phases.md)

## Context

ADR 0006 made SGLang's eight phases (`launch` … `request_finalize`) the
vocabulary every implementation installs into. Those phases are what *SGLang*
does: it compiles at pipeline build (`denoising.py:380`) and captures graphs
at warmup (`runner.py:575`). vLLM-Omni, ComfyUI or TensorRT need not have the
same phases. A generic implementation, such as our TeaCache, should not name
SGLang's phases.

## Decision

Implementations declare **three constraints**. Each EngineAdapter maps them
onto its own lifecycle.

| Constraint | Meaning | Required by | SGLang mapping |
|---|---|---|---|
| `mutates_model` | Changes the module tree or a module's forward; must be in place before compile or capture, and changing it needs a new model instance | FP8 / NVFP4, QKV fusion, a trunk wrapper | `construct` / `post_load`, before `pipeline_build` |
| `dynamic_in_forward` | Makes data-dependent Python decisions *inside* the model forward | TeaCache, Spectrum | rejected with breakable CUDA graphs; graph break under torch.compile |
| `request_state` | Needs per-request initialization and cleanup | step skip, TeaCache | `request_init` / `request_finalize` |

SGLang's eight phases are documented in the
[investigation](../architecture-investigation.md#part-6-lifecycle) as **SGLang's**
lifecycle. They are used only inside `SGLangAdapter`.

## Consequences

- Policies at the step seam (outside the DiT call) are not
  `dynamic_in_forward`. They are expected to work under graph capture, since
  the graph is replayed per DiT call. Runtime verification is pending.
- Adding an engine means writing its mapping for three flags, not reconciling
  eight phase names.

## Rejected

- **SGLang's phase enum as the shared vocabulary.** It is the smaller type,
  but it couples every generic implementation to one engine, and the first
  three techniques include two generic ones.
- **A full cross-engine lifecycle model.** No second engine has been read
  yet. Revisit when the second EngineAdapter exists.
