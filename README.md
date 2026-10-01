# optimizer

An agent-driven inference optimizer for diffusion models, modeled on NVIDIA's
Sol-Engine. You give it a model served by an engine (SGLang first); it
searches acceleration techniques and returns configurations that are faster
**and** pass a quality gate. Every speedup carries its baseline, its
measurement and its gate evidence.

Status: design phase. No code yet. The architecture has been checked against
SGLang's real Qwen-Image and FLUX code paths. Components are built one at a
time, in the order the ADRs set.

```mermaid
flowchart LR
    Agent["Agent layer<br/>propose · implement"] --> Tech["Technique layer<br/>technique → implementation"]
    Tech --> Target["Target<br/>engine adapter × model binding"]
    Target --> Core["Core<br/>launch · measure · gate · frontier"]
    Core -->|results| Agent
```

## Layout

    docs/                       design docs and decision records (start here)
    docs/adr/NNNN-*.md          one accepted decision per file

## Docs

- [Architecture](docs/architecture.md) — the current design: concepts, where knowledge belongs, and how a technique resolves onto a runtime
- [Sol-Engine](docs/sol-engine.md) — what the system we're modeled on does, and which parts of it are not public
- [Architecture investigation](docs/architecture-investigation.md) — how Qwen-Image and FLUX actually run in SGLang (the evidence), and the falsification experiments to run next
- [ADR 0001: Three layers](docs/adr/0001-three-layer-architecture.md) — why Core / Technique / Agent, and what each owns
- [ADR 0002: Scope](docs/adr/0002-scope-single-gpu-diffusion.md) — why one-GPU image diffusion first, and what is kept open
- [ADR 0003: SGLang first](docs/adr/0003-sglang-first-engine.md) — why SGLang, and the rule that keeps vLLM-Omni/ComfyUI addable
- [ADR 0004: Incremental build](docs/adr/0004-incremental-build-interface-tests.md) — how we build (one agreed piece at a time) and what gets tested
- [ADR 0008: Capability layer](docs/adr/0008-capability-layer.md) — why implementations depend on capabilities, and why EngineAdapter, ModelSpec and Binding are three separate things
- [ADR 0009: Lifecycle constraints](docs/adr/0009-lifecycle-constraints.md) — why implementations declare constraints instead of naming SGLang's phases
- [ADR 0010: V1 capabilities and first techniques](docs/adr/0010-v1-capabilities-and-first-techniques.md) — which capabilities are common or optional, and the three validation techniques
- Superseded, kept for history: [ADR 0005](docs/adr/0005-target-binding-architecture.md) (target binding), [ADR 0006](docs/adr/0006-lifecycle-phases.md) (SGLang phases), [ADR 0007](docs/adr/0007-v1-seams-and-first-techniques.md) (V1 seams)
