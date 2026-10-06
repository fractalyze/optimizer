# ADR 0017: Registries are plain metadata; capability resolution matches each requirement to exactly one provider

Status: accepted · 2026-10-04 · first Core component, built under [ADR 0004](0004-incremental-build-interface-tests.md); amends the capability naming in [ADR 0016](0016-native-implementations.md)

## In short

The first piece of Core is a technique registry, an implementation registry,
a registry of target providers (engine adapters and Bindings), and a resolver
that answers one question: **can this target supply every capability this
implementation requires?** Every entry is frozen metadata, written out by
hand and imported without any engine. The resolver assigns each required
capability to exactly one of the target's providers. If none provides it, the
result is `UNSUPPORTED`; if more than one does, it is `AMBIGUOUS`. It never
picks a provider by precedence. Nothing about feasibility, composition or
running is decided here: a `RESOLVED` implementation is *supported*, which
says nothing about whether it is *feasible* or will be *engaged*.

## Context

Experiments 001–005 validated three ways a capability reaches an
implementation:

| Path | Example | Provided by | Evidence |
|---|---|---|---|
| engine-common | prediction reuse at the step seam (`fractalyze-prediction-reuse`) | EngineAdapter only | exp 001 |
| model-bound | trunk reuse | EngineAdapter (timestep, owner state) + Binding (trunk, signal) | exp 002–004, two engines |
| engine-native | FP8 W8A8 | EngineAdapter (`engine_feature.*`) | exp 005 |

Core had to encode these without hard-coding any of them, and without
pulling in feasibility (GPU, execution mode, silent fallbacks), which
[ADR 0013](0013-feasibility-and-engagement.md) keeps as a separate stage.

## Decision

### Entries are frozen metadata, registered explicitly

`TechniqueSpec`, `ImplementationSpec` and `ProviderSpec` are frozen
dataclasses. The validated entries live in one module, `optimizer/catalog.py`,
and each cites the experiment that validated it. There is no discovery, no
decorator and no entry point. Listing order is registration order. Duplicate
ids are rejected, and so is an implementation whose technique is not
registered. Loading the registries imports no engine; when executable code
is needed later it will be referenced from an entry, not imported by it.

### The specs hold only what resolution needs

| Spec | Fields | Not here, and where it goes |
|---|---|---|
| `TechniqueSpec` | `id`, `summary` (the numerical method, per ADR 0016) | conceptual parameters: when a stage consumes them |
| `ImplementationSpec` | `id`, `technique`, `requires` | lifecycle, execution requirements, `owns`, `engaged`: feasibility, composition, engagement |
| `ProviderSpec` | `id`, `kind` (adapter or Binding), `engine`, `model` (Bindings only), `provides` | seams, hooks, execution mode: the adapter's private code |

*2026-10-06: model identity is amended by [ADR 0021](0021-execution-plan.md): `Target` is (engine, architecture, ModelRef) and Bindings are keyed by architecture.*

| `Target` | `engine`, `model` | GPU, compile and graph mode, engine settings: feasibility |

`ImplementationSpec` has no engine or model field. Where it can run is
computed by resolving its `requires`, so `sglang-native-fp8-w8a8` is
unsupported on vLLM-Omni because only SGLang's adapter provides its
capability, not because a list says so. Native and generic implementations
differ only in what they require, and the resolver treats them the same.

### Capabilities are a closed set of semantic ids

A capability id is a string from the vocabulary of the architecture: the
eight with runtime evidence (`step_observe`, `step_prediction_override`,
`step_schedule_mutate`, `timestep_state`, `request_local_state`,
`trunk_observe`, `trunk_output_override`, `signal_observe`), or
an id under `engine_feature.`. An unknown id is rejected when a spec is
built, so a misspelling fails at registration rather than as a puzzling
`UNSUPPORTED`. Adding an id means adding the experiment that validates it.

`engine_feature.fp8_w8a8_dynamic_linear` is the one engine-native capability
with evidence. Its name matches its technique's, but that is not a naming
rule: a future native implementation may need several engine-native
capabilities, or one whose name differs from any technique. This amends
[ADR 0016](0016-native-implementations.md), which described the requirement
as "a single `engine_feature.<technique>` capability"; its decision stands
otherwise.

### A provider lists only what was exercised on it

The SGLang adapter provides the step capabilities, timestep and owner state,
and `engine_feature.fp8_w8a8_dynamic_linear`. The vLLM-Omni adapter provides
only timestep and owner state, because its step seams and FP8 were read, not
run. Each Binding provides `trunk_observe`, `trunk_output_override` and
`signal_observe`.

Trunk capabilities run through the engine's adapter but exist only where a
Binding locates the trunk, so resolution names the Binding as their provider.
That keeps each capability with exactly one provider, and matches the
architecture's rule that what is specific to one engine × model pair belongs
to the Binding.

### Resolution answers "supported", and only that

Resolution answers one question: can this target provide the semantic
capabilities this implementation requires? It does not answer, and must not
be read as answering:

- whether the hardware can run it (FP8 needs sm ≥ 89);
- whether this engine revision behaves as traced;
- whether the applied compile or CUDA graph mode lets the operations execute
  correctly;
- whether the engine will silently fall back to a different method (FP8's
  weight-only Marlin path);
- whether the optimization actually executed at runtime.

The first four are feasibility and the last is engagement, both later
stages ([ADR 0013](0013-feasibility-and-engagement.md),
[ADR 0016](0016-native-implementations.md)). Supported, feasible and engaged
stay three separate states.

### Resolution is one-to-one, or it fails visibly

A target's providers are its engine's adapter and the Binding for its model,
if any. A model with no Binding still gets its adapter, so engine-common
implementations resolve on any model. For each required capability:

| Providers claiming it | Result |
|---|---|
| exactly one | resolved to that provider |
| none | listed as missing; status `UNSUPPORTED` |
| two or more | listed as ambiguous; status `AMBIGUOUS` |

`UNSUPPORTED` wins over `AMBIGUOUS`, since no choice of provider would make
the implementation runnable. There is no precedence rule because no
experiment needed one: in every validated target each capability had one
owner. Two providers claiming one capability means the registry is wrong.

## Consequences

- Feasibility, composition and planning will take a `Resolution` as input;
  they must not re-derive which provider supplies what.
- A new engine or model is supported by adding a `ProviderSpec`; a new
  technique by adding a `TechniqueSpec` and an `ImplementationSpec`. Core
  code does not change.
- `engine_feature.*` support is declared per adapter, not detected from the
  installed engine build. Detecting it belongs with the other target facts
  read at feasibility time.
- Experiments 001–005 are unchanged and remain the evidence; they are not yet
  ported onto these specs.

## What this rests on

The catalog's entries come from experiments 001–005 (SGLang @ `8ca82118e`,
vLLM-Omni @ `68003cf6a`). The resolver itself is tested on a CPU, against the
catalog and against artificial registries for the failure cases.

**Open:**
- **Model identity.** `Target.model` is a short id per checkpoint family
  (`qwen-image-2.1`, `flux.2-klein`, `qwen-image-2512`). Whether one Binding
  covers several checkpoints, as FLUX.2's shared transformer class suggests,
  is untested.
  *2026-10-06: resolved by [ADR 0021](0021-execution-plan.md): Bindings are keyed by the engine's model class read from the checkpoint.*

## Alternatives rejected

- **Registries of live objects with hooks.** Loading the catalog would import
  engines, need a GPU machine to test, and could not be inspected or
  serialized for a future search space.
- **Provider precedence (Binding over adapter, or registration order).** It
  would hide a registry error behind a silent choice; no evidence needs it.
- **An engine or model field on implementations.** A hand-kept support list
  is what resolution exists to replace.
- **Hardware or execution mode in `Target`.** Those decide feasibility, and
  putting them here would merge "supported" with "feasible".
- **A capability class hierarchy.** Matching needs only identity; strings from
  a closed set give that and stay readable in a printed resolution.
