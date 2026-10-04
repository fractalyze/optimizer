# ADR 0018: Feasibility asks the target engine's evaluator about each execution requirement and capability of a resolved implementation

Status: accepted · 2026-10-04 · encodes the feasibility stage of [ADR 0013](0013-feasibility-and-engagement.md) in Core; amends where its execution-mode record lives

## In short

Core's second component takes a `RESOLVED` capability resolution
([ADR 0017](0017-registries-and-capability-resolution.md)) and a
`RuntimeContext`, and answers `FEASIBLE` or `INFEASIBLE` with structured
reasons. Implementations declare execution requirements as behavior
(`runs_every_invocation`, `override_exact` on a capability). The target
engine's evaluator, which knows where each seam sits and which settings make
the engine run another method, judges every requirement and every required
capability; all must pass. Core names no technique. Only what is knowable
before launch is rejected; anything else is left to engagement.

```text
Supported   all required capabilities resolve against the target
    ↓
Feasible    the resolved implementation's known execution requirements can
    ↓       be satisfied by the provided RuntimeContext
[future Execution]
    ↓
Engaged     runtime evidence proves the implementation actually executed
```

## Context

ADR 0013 defined feasibility and the two execution requirements; experiments
001, 002 and 005 measured where they hold in SGLang:

| Runtime (SGLang) | Trunk seams (inside the DiT call) | Step seams | Native FP8 |
|---|---|---|---|
| eager | both hold | both hold | |
| regional or whole-model compile | `runs_every_invocation` holds; `override_exact` held where the identity check passed (Qwen-Image-2.1), failed where it did not (FLUX.2, whole-model) | both hold | |
| breakable CUDA graphs | both fail: hooks ran on 1 of 40 calls | both hold | |
| `SGLANG_FORCE_FP8_MARLIN` set | | | weight-only FP8 ran instead |
| `SGLANG_FP8_IGNORED_LAYERS` set | | | 96 of 224 layers became FP8 |

Two things followed for the code. Whole-model compile is not unsafe for trunk
reuse as such: it was exact on Qwen-Image-2.1 and inexact on FLUX.2, so the
rule must turn on the identity check, not on the model or the mode. And the
two FP8 settings make SGLang accept the flag and run a different method, so
the adapter must reject them before launch.

## Decision

### Implementations declare behavior on a capability

`ImplementationSpec` gains `execution`, a set of `ExecutionRequirement(behavior,
capability)`, each on a capability the implementation requires. The catalog
declares, as ADR 0013 did:

| Implementation | Execution requirements |
|---|---|
| `fractalyze-teacache` | `runs_every_invocation(trunk_observe)`, `runs_every_invocation(trunk_output_override)`, `override_exact(trunk_output_override)` |
| `fractalyze-prediction-reuse` | `runs_every_invocation(step_observe)`, `override_exact(step_prediction_override)` |
| `sglang-native-fp8-w8a8` | none |

No implementation names an engine setting.

### The engine's evaluator judges; Core only collects

`feasibility.check(implementation, resolution, context, evaluators)` refuses
anything but a `RESOLVED` resolution of the same implementation, so
resolution is never repeated. It looks up the evaluator for the target's
engine and asks it about every execution requirement and every capability in
the resolution. All must pass, and every failure is reported. An engine with
no evaluator is `INFEASIBLE`, because nothing can be guaranteed there;
vLLM-Omni has none yet, since experiment 004 ran eager only.

`SGLangFeasibility` holds the table above, keyed by where a capability's seam
sits: step seams in the denoising loop hold in every mode; seams inside the DiT
call fail under graph replay and, under compile, keep `override_exact` only if
the identity check passed. For `engine_feature.fp8_w8a8_dynamic_linear` it
rejects the two settings. It names no technique, implementation or model, and
imports nothing from SGLang.

Bindings take no part, as ADR 0013 decided: the FLUX.2 / Qwen-Image-2.1
difference is a measured identity-check result, not model structure.

### The RuntimeContext holds four facts, each read by a rule

| Field | Read by |
|---|---|
| `compile_scope` (none, regional, whole model) | `override_exact` inside the DiT call |
| `graph_replay` | both requirements inside the DiT call |
| `identity_check_passed` (measured; None if not measured) | `override_exact` under compile |
| `engine_env` | native FP8's silent fallbacks |

`Target` stays `(engine, model)`. The context describes the mode the engine
applied, not the one requested. `identity_check_passed` is measurement
history for this target and compile scope, supplied with the context until a
measurement store exists.

ADR 0013 kept the execution-mode record private to SGLangAdapter and rejected
a shared mode vocabulary. This ADR names compile scope and graph replay in
Core, as typed fields instead of an untyped map. They are facts the engine
reports, and what they imply is still decided only by the engine's evaluator.
vLLM-Omni's compile options (regional by default, whole model optional, by
code reading) fit the same three scopes. If a second engine's modes do not,
this is the decision to revisit.

### Feasibility rejects only what is knowable before launch

A setting that is known to make SGLang run weight-only or partial FP8 is
`INFEASIBLE`. A fallback for a reason the evaluator does not know, or a graph
mode that turns out to replay a call this context did not predict, is caught
after the run by engagement verification as `FAILED_TO_ENGAGE`. Feasibility
is a prediction; engagement is the proof.

## Consequences

- The composer will receive implementations that are both supported and
  feasible, each judged alone; conflicts between them are its question.
- A new engine needs an evaluator before anything on it can be feasible.
- The identity check is now an input the optimizer must measure, per target
  and compile scope, before trunk-control implementations can run compiled.
- Reasons have three kinds: `execution_requirement_unsatisfied`,
  `known_native_fallback`, `unsupported_runtime_condition`.

## What this rests on

Every rule comes from a runtime result: experiments 001 and 002 (SGLang @
`8ca82118e`, Qwen-Image-2.1 and FLUX.2-klein) for the execution requirements,
experiment 005 for the FP8 settings. Tested on a CPU against the catalog.

**Deliberately left out, though known from code reading:**
- SGLang's silent weight-only FP8 below sm_89
  (`srt/layers/quantization/fp8_utils.py:2334`). No experiment ran on such a
  GPU, so the context carries no hardware yet; until then the case is caught
  by engagement.
- The `quantization_ignored_layers` server argument, which drops layers like
  the environment variable does. Untested; the adapter never sets it.
- SGLang offering FLUX.2 no regional compile: the engine's own refusal.

**Open:** whether the identity check belongs per target and compile scope or
also per graph signature; engine-revision checks, once a second revision is
traced.

## Alternatives rejected

- **Engine-shaped requirements on implementations** (`cuda_graph_disabled`).
  Rejected in ADR 0013; still wrong for the next engine.
- **"Whole-model compile is infeasible for trunk reuse."** Contradicted by
  Qwen-Image-2.1, which stayed exact.
- **A per-model compile rule in the Binding.** The difference is a measured
  result, not structure (ADR 0013).
- **A feasibility callback on each implementation.** It would put engine
  knowledge back into implementations and make the registry opaque; the FP8
  fallbacks are knowledge of the engine feature, so they live with the
  adapter.
- **`RuntimeContext(settings: dict[str, Any])`.** It hides which facts a rule
  depends on. `engine_env` is the one map, because environment variables are
  a map of strings.
- **Re-resolving capabilities inside feasibility.** The resolution is the
  source of truth for who provides what.
