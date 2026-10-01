# Decision records

An ADR (architecture decision record) explains **one decision**: what forced
it, what was chosen, what it costs, and which alternatives were turned down
and why. They are written for people: a reviewer today, or a teammate months
from now asking "why is it built like this?"

They do not describe the current design as a whole. That lives in
[architecture](../architecture.md), which links back here for the reasoning.

## How to read one

Every ADR has the same parts:
- **In short:** the decision in two or three sentences.
- **Context:** what forced a choice.
- **Decision:** what was chosen.
- **Consequences:** what got easier and what got harder.
- **Alternatives rejected:** with the condition that would make us revisit
  them.

An accepted ADR is never edited to say something else. When a decision
changes, a new ADR *supersedes* it, and the old one stays, marked, so the
history of the design is visible.

## Index

| # | Decision | Status |
|---|---|---|
| [0001](0001-three-layer-architecture.md) | Split the system into Core, Technique and Agent layers | accepted |
| [0002](0002-scope-single-gpu-diffusion.md) | Start with image diffusion on one GPU, keep the structure extensible | accepted |
| [0003](0003-sglang-first-engine.md) | SGLang is the first engine; engines are pluggable | accepted |
| [0004](0004-incremental-build-interface-tests.md) | Build one agreed piece at a time; test interfaces, not internals | accepted |
| [0005](0005-target-binding-architecture.md) | Technique → Implementation → Target binding | superseded by 0008 |
| [0006](0006-lifecycle-phases.md) | Lifecycle phases follow SGLang's compile and capture points | superseded by 0009 |
| [0007](0007-v1-seams-and-first-techniques.md) | V1 seams and the first three techniques | superseded by 0010 |
| [0008](0008-capability-layer.md) | Implementations depend on capabilities; EngineAdapter, ModelSpec and Binding are separate | accepted |
| [0009](0009-lifecycle-constraints.md) | Implementations declare lifecycle constraints; engine stages stay in the adapter | accepted |
| [0010](0010-v1-capabilities-and-first-techniques.md) | V1 capabilities, optional block access, first three techniques | accepted |
