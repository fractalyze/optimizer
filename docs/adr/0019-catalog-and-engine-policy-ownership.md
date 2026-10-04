# ADR 0019: The catalog is atomic and assembled from its owners; engine policy is explicit data in the engine's package

Status: accepted · 2026-10-04 · amends where entries live in [ADR 0017](0017-registries-and-capability-resolution.md) and where the SGLang evaluator lives in [ADR 0018](0018-feasibility-in-core.md); no behavior change

## In short

Before adding the composer, Core's code was reorganized around ownership, not
features. The catalog declares atomic entries only, techniques,
implementations and providers, and never a combination; the composer will
build combinations. Each engine package declares its own entries and owns its
runtime policy as explicit, inspectable data on its evaluator, not as
module-level tables and anonymous lambdas. Core imports no engine package and
no catalog; everything it needs is passed in.

```text
optimizer/
  core/                 engine-free algorithms and types; imports nothing below
  techniques.py         techniques, and our engine-free implementations
  engines/<engine>/     that engine's providers, native implementations, policy
  catalog.py            gathers the atomic entries from their owners
```

## Context

An audit of the code after ADRs 0017 and 0018 found:

| Item | Was in | Problem | Now |
|---|---|---|---|
| capability ids | `core/specs.py`, mixed with the metadata types | two responsibilities in one file | `core/capabilities.py`, the vocabulary alone |
| techniques and generic implementations | `catalog.py`, with every engine's providers | one file for all owners | `techniques.py` |
| SGLang adapter and Bindings, native FP8 | `catalog.py` | engine entries outside the engine | `engines/sglang/catalog.py` |
| vLLM-Omni adapter and Binding | `catalog.py` | the same | `engines/vllm_omni/catalog.py` |
| `_LOOP`, `_IN_DIT` (seam placement) | module globals in `adapters/sglang.py` | SGLang policy as an unowned set | `SGLangFeasibility.seams`, typed by `SeamRegion`, built by `measured()` |
| `_FP8_FALLBACKS` with lambdas | module global | policy hidden in anonymous callbacks | `SGLangFeasibility.native_fallbacks`, `NativeFallback` with a named `EnvTrigger` |
| engine → evaluator map | `catalog.evaluators()` | not a catalog entry | `engines.evaluators()` |
| engine and model ids | string literals repeated across entries | duplicated source of truth | `ENGINE` and model constants in each engine package |

Nothing in the resolver, registries or feasibility orchestrator branched on a
technique, implementation, engine or model; that was confirmed, not changed.

## Decision

### The catalog declares what exists, and nothing else

It holds `TechniqueSpec`, `ImplementationSpec` and `ProviderSpec` entries.
It never holds a combination such as "SGLang + Qwen + TeaCache + FP8", and it
does not say which implementation should win, whether a target is supported,
whether a runtime is feasible, whether two implementations conflict, in what
order they run, or how fast they are. Those are computed by the stages that
consume it. It therefore grows additively: one entry per technique,
implementation or provider, never one per combination.

### Each owner declares its entries; the catalog only gathers them

Engine-free entries live in `techniques.py`. An engine's providers and native
implementations live in `engines/<engine>/catalog.py`, with that engine's id
and model ids as constants. `catalog.py` concatenates them and builds the
registries. The registries own validation and indexing: duplicate ids,
implementations of unregistered techniques, deterministic order,
`for_technique`, `for_target`.

### Engine policy is explicit data owned by the engine's evaluator

`SGLangFeasibility` is a frozen dataclass holding two pieces of data: where
each capability's seam sits (`SeamRegion.DENOISING_LOOP` or `DIT_CALL`), and
which settings make SGLang run another native method (`NativeFallback`:
capability, environment variable, how SGLang reads it, consequence).
`measured()` builds it with exactly what experiments 001, 002 and 005
measured, each value next to its evidence. A test can build one with any
other policy; nothing global is consulted.

### Core depends on nothing above it

`core/` imports no engine package, no catalog and no engine runtime; the
resolver and the feasibility check take their providers and evaluators as
arguments. A test enforces the direction.

## Consequences

- The composer takes registries, resolutions and feasibility results as
  input and creates combinations itself.
- A new engine is a new package under `engines/`, plus one line in
  `catalog.py` and, once it has an evaluator, one in `engines.evaluators()`.
- Module-level data that remains is vocabulary (capability ids, technique
  ids, engine and model ids) or declared catalog entries, never policy.

## What this rests on

No runtime evidence changed; the 41 existing tests pass unchanged, and new
tests cover the catalog's atomicity, registry isolation, Core's dependency
direction and the evaluator's explicit policy.

**Naming audit, documented rather than renamed:**
- `teacache` / `fractalyze-teacache`: the technique decides from a signal
  threshold, while experiment 002's policy reused the trunk at a fixed step.
  The capabilities and execution requirements are the same; the decision rule
  is the implementation's, not yet built.
- `trunk_observe` and `trunk_output_override` run through the adapter but are
  provided by the Binding (ADR 0017); `request_local_state` keeps its open
  ownership model (request × CFG branch).
- `prediction_reuse` and `step_schedule_mutate` stay distinct: reusing a
  prediction keeps the scheduler step, mutating the schedule removes one.

## Alternatives rejected

- **A `constants.py` for every id and table.** It centralizes without giving
  ownership; engine rules belong with the engine's evaluator.
- **A `Catalog` class beside the registries.** Tuples of entries and the
  existing registries already separate declaration from validation.
- **A rule engine for feasibility policy.** Two small frozen structures say
  everything the evidence supports.
