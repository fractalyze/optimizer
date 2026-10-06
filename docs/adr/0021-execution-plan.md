# ADR 0021: An execution plan places each implementation by its lifecycle; the ServerKey is the live-server reuse boundary

Status: accepted · 2026-10-06 · applies [ADR 0009](0009-lifecycle-constraints.md) and [ADR 0016](0016-native-implementations.md) to planning; amends [ADR 0017](0017-registries-and-capability-resolution.md) (techniques gain parameters) and [ADR 0018](0018-feasibility-in-core.md) (results record their context)

## In short

Core's fourth stage turns a composable configuration and its parameter values
into an engine-neutral `ExecutionPlan`: what must be fixed when the server is
built, keyed by a `ServerKey`, and what varies per request. An implementation
that mutates the model contributes to the server, one with per-request state
contributes to the request, and TeaCache, which does both, contributes to
both. Two plans with equal ServerKeys can run on the same started server.
SGLang's translator lowers a plan to `quantization="fp8"`, plugin names and
per-request policies; Core knows none of those. Nothing is launched.

```text
Composer (COMPOSABLE)
        ↓
Execution plan
    ├── server   ServerKey: engine, model, server settings, server contributions
    └── request  per-request contributions with their parameter values
        ↓
engine translator (SGLang: server kwargs, env, plugins; request policies)
        ↓
[execution]  not built yet
```

## Context

ADR 0016 found that a technique applied at load is part of the model's
identity, so trials must be grouped by server configuration. ADR 0020 left
lifecycle constraints to this stage. Planning also needed what nothing had
yet: the parameter values a trial chooses, such as TeaCache's threshold.

| Implementation | What the experiments showed | Places |
|---|---|---|
| native FP8 | one server argument, fixed for the server's life (exp 005) | server |
| TeaCache | trunk hooks loaded before the model is built (exp 002); policy and state per request and CFG branch | server and request |
| prediction reuse | step seams in the shared denoising loop, controlled per request (exp 001) | request |

## Decision

### Techniques declare parameters; a trial supplies a TechniqueConfig

`TechniqueSpec.parameters` lists conceptual parameters by name and type:
TeaCache `threshold` (float), prediction reuse `steps` (tuple of ints), FP8
none. A `TechniqueConfig` holds one trial's values; validation rejects unknown
names, missing names and wrong types, nothing more. Ranges, defaults and
priors are the search space's. Engine settings are never parameters.

### Placement follows two lifecycle flags

`ImplementationSpec.lifecycle` carries `mutates_model` and `request_state`
from ADR 0009; its third flag, `dynamic_in_forward`, is expressed by execution
requirements since ADR 0013. The planner, which names no technique:

| Lifecycle | Server contribution | Request contribution |
|---|---|---|
| `mutates_model` | yes | |
| `request_state` | | yes, with the technique's values |
| both | yes, without values | yes, with the values |
| neither, with values | yes, with the values (nothing else could change them) | |

### The ServerKey holds only what decides how a server is built

`ServerKey` = engine, model, `ServerSettings`, and the sorted server
contributions. `ServerSettings` projects the `RuntimeContext` onto the facts
that build a server: compile scope, graph replay and engine environment.
`identity_check_passed` is evidence about a target, not configuration, and is
left out; a test fails if a new context field is not classified either way.
Everything in the key is immutable and hashable. Plans with equal keys can run
on one live server; nothing yet groups or schedules them.

### Each stage's verdict travels with its inputs

`FeasibilityResult` now records the `RuntimeContext` it judged, and the
composer refuses candidates judged in different contexts, so the plan's
server settings are the ones feasibility approved. `RuntimeContext` became
hashable for this; it accepts a mapping for `engine_env` and freezes it.

### Engine translation is engine data

`engines/sglang/plan.py` lowers a plan using explicit data built by
`validated()`: checkpoint paths, the plugin each implementation needs at
server time, the plugin that reads each request policy, and the server
arguments for each execution mode (`enable_torch_compile`,
`regional_compile`, `enable_breakable_cuda_graph`), all as the experiments
used them. Equal ServerKeys lower to equal server configurations.

The step plugin is loaded in every SGLang server, because with no policy for
a request it only observes (exp 001). That is what lets request-only step
techniques leave the ServerKey alone.

## Consequences

- A future scheduler groups plans by `ServerKey` and reuses a server for
  every request-level variation; changing FP8 or adding TeaCache means a new
  server.
- The plugins named are the experiments' probes. Their policies do not yet
  read a TeaCache threshold (experiment 002's policy reused at a fixed step),
  so lowering is correct as configuration but no plugin implements it yet.
- SGLang has no request field for plugin settings; the experiments used a
  control table keyed by request id. How execution delivers policies is the
  next stage's choice.

## What this rests on

Placement and lowering values come from experiments 001, 002 and 005 (SGLang
@ `8ca82118e`). Tested on a CPU: the three lifecycle shapes, ServerKey
equality under parameter changes and inequality under FP8 or TeaCache
presence, execution mode and environment; projection of evidence out of the
key; parameter validation; lowering.

**Left out on purpose:**
- engine and model revisions in the ServerKey: `Target` has none yet;
- machine-specific server arguments the experiments used for memory
  (component residency, warmup resolutions): they belong to the machine's
  execution setup, not the plan;
- ordering in lowering: the plan carries the composer's partial order, and no
  current implementation declares one.

## Alternatives rejected

- **One placement per implementation.** TeaCache needs both.
- **The whole RuntimeContext in the ServerKey.** Identical servers would get
  different keys whenever an identity check was or was not measured.
- **Engine settings as technique parameters.** `quantization="fp8"` is how one
  engine realizes a choice, not the choice.
- **Loading the step plugin only when a request needs it.** The server
  configuration would then depend on requests, and equal ServerKeys could
  lower to different servers.
