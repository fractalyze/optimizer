# ADR 0013: Execution feasibility and engagement verification are stages every implementation passes

Status: accepted · 2026-10-02 · amends the engine mapping of `dynamic_in_forward` in [ADR 0009](0009-lifecycle-constraints.md)

## In short

A target that *provides* a capability may still be unable to use it in the
way it is being run. [Experiment 002](../../experiments/002-trunk-control/README.md)
showed one trunk-reuse implementation that was exact in eager mode, changed
the image under whole-model compile, and silently did nothing under graph
replay. So every implementation, not just TeaCache, now passes two more stages:

```text
Technique → Implementation → required capabilities → capability resolution
  (EngineAdapter + Binding) → EXECUTION FEASIBILITY → install / run
  → ENGAGEMENT VERIFICATION → measurement (only if engaged)
```

- **Execution feasibility:** given this implementation, this target and the
  execution mode the engine actually applied, can the required operations
  execute correctly? The implementation states what behavior it needs, in
  semantic terms; the EngineAdapter decides whether the current mode gives it.
- **Engagement verification:** after the run, runtime evidence must prove the
  implementation actually executed. Each implementation defines its own
  evidence. No benchmark result is accepted without it.

## Context

| Run (experiment 002) | Capabilities resolved? | What actually happened |
|---|---|---|
| Qwen-Image-2.1 and FLUX.2, eager | yes | exact; reuse skipped every block |
| Qwen-Image-2.1, regional compile | yes | exact; a few extra graph breaks |
| Qwen-Image-2.1, whole-model compile | yes | exact; graph breaks roughly tripled |
| FLUX.2, whole-model compile | yes | an override returning the trunk's own output still changed the image (32.6 dB) |
| Qwen-Image-2.1, graph replay | yes | the hooks ran on 1 of 40 calls; a requested reuse did nothing, and nothing reported it |

Experiment 001 adds one more: a graph mode requested for FLUX.2 was turned off
by SGLang with only a warning. In every row, configuration said "on". Only
counting what ran told the cases apart.

[ADR 0009](0009-lifecycle-constraints.md) had mapped `dynamic_in_forward`
straight to engine consequences: "refused with graph capture; graph breaks
under torch.compile". The evidence is finer than that: compile was fine for
one model and not the other, and fine for observing but not always for
overriding. A flag with a fixed engine meaning cannot express that.

## Decision

### Four states, never inferred from one another

| State | Means | Example (TeaCache) |
|---|---|---|
| configured | the optimizer selected it | TeaCache was chosen with a threshold |
| supported | the target resolves every required capability | SGLang + a Binding provide `trunk_output_override` |
| feasible | the applied execution mode lets the required operations execute correctly | the trunk callbacks run on every trunk call, and overrides are used as given |
| engaged | runtime evidence shows the optimization executed | trunk blocks actually did not run on the reused calls |

### The implementation declares three kinds of requirement

| Kind | Examples | Interpreted by |
|---|---|---|
| **A. capabilities** | `timestep_state`, `signal_observe`, `trunk_output_override`, `request_local_state` | capability resolution (EngineAdapter, Binding, ModelSpec) |
| **B. lifecycle** | `mutates_model`, `dynamic_in_forward`, `request_state` ([ADR 0009](0009-lifecycle-constraints.md), unchanged) | the EngineAdapter, onto its own lifecycle |
| **C. execution** | `runs_every_invocation`, `override_exact`, each on a capability it uses | the EngineAdapter, against the applied execution mode |

Execution requirements say what behavior the implementation needs, never which
engine setting provides it. V1 has exactly the two that experiment 002's
failures call for:

| Requirement | Means | Broken in experiment 002 by |
|---|---|---|
| `runs_every_invocation(capability)` | the implementation's code at this capability runs on every logical invocation, e.g. every trunk call | graph replay: the trunk callbacks did not run |
| `override_exact(capability)` | what the implementation supplies through a decide-type capability is exactly what the model continues with | whole-model compile on FLUX.2: an identity override changed the image |

TeaCache declares both on its trunk capabilities. A field such as
`whole_model_compile_supported` is deliberately absent: it is an SGLang
consequence, not a need, and it would be wrong for the next engine.

### The EngineAdapter interprets the execution mode

The EngineAdapter alone knows compile boundaries, graph-capture boundaries,
whether Python callbacks execute, and whether the engine silently fell back
from what was requested. It keeps a record of the mode the engine *applied*,
read back from the engine. For SGLang in V1 that is two fields: compile scope
(off, regional, whole model) and graph mode (off, breakable CUDA graphs). That
record is SGLangAdapter's and is not shared vocabulary.

From it, the adapter answers per requirement:

| SGLang, measured | `runs_every_invocation` (trunk) | `override_exact` (trunk) | step seams |
|---|---|---|---|
| eager | holds | holds | both hold |
| regional compile (Qwen-Image-2.1) | holds | holds, identity check passed | both hold |
| whole-model compile | holds | Qwen-Image-2.1: holds, identity check passed · FLUX.2: **fails** | both hold |
| breakable CUDA graphs (Qwen-Image-2.1) | **fails** | **fails** | both hold |

Where compiled code sits between the implementation and the model, the adapter
establishes `override_exact` with an **identity check**: run the override with
a payload equal to what was computed, and require the final output to be
bitwise identical to a run without the implementation. The result depends on
the target, so it is recorded in measurement history. It must be judged on the
final output, never by checks inside the compiled code: those are compiled too,
and in experiment 002 they reported "not equal" with zero differing elements.

The **Binding gets none of this.** The FLUX.2 / Qwen-Image-2.1 difference under
whole-model compile is a measured fact about a target, held in measurement
history, not model structure. Compile or graph-capture logic moves into a
Binding only if evidence shows it is genuinely model-specific.

Feasibility is then one check before the run:

| Check | Fails when |
|---|---|
| capabilities | a required capability cannot be resolved (`UNSUPPORTED`) |
| lifecycle | `mutates_model` / `dynamic_in_forward` / `request_state` cannot be honored |
| execution | an execution requirement does not hold in the applied mode |
| conflicts | two implementations claim the same exclusive resource (`owns`) |

### Engagement: each implementation brings its own evidence check

Each implementation provides `engaged(evidence) -> (verdict, reason)`. The
EngineAdapter supplies the evidence as counters of what actually ran at the
seams the implementation used, so the implementation stays engine-free. There
is no universal metric.

| Implementation | Engaged when |
|---|---|
| trunk reuse (TeaCache) | it saw every trunk call, and on every reuse it decided, the trunk's blocks did not run |
| prediction reuse (step skip) | model prediction calls fell by the number of reuses it decided |
| timestep reduction | the schedule that ran is as long as it asked for |
| native FP8 | the live model's layers report the FP8 method |
| CUDA graphs (an engine feature) | graph replays actually happened |

Feasibility is a prediction; engagement is the proof. They can disagree: under
graph mode, a request whose shape was never captured runs eager, so the same
configuration can fail `runs_every_invocation` for one request and satisfy it
for the next.

### Every run ends in exactly one status

| Status | Means | Enters measurement history as |
|---|---|---|
| `UNSUPPORTED` | a capability cannot be resolved on this target | a fact about the target |
| `INFEASIBLE` | lifecycle, execution or a conflict rules it out | a fact about the target and mode |
| `RUNTIME_ERROR` | it crashed | a failure |
| `FAILED_TO_ENGAGE` | it ran, but the evidence does not show the optimization executed | a failure, **never a speed or quality point** |
| `VALID` | engaged | a speed and quality data point |

The distinction that matters: "the optimization ran and gave no speedup" is a
`VALID` measurement; "the optimization never executed" is `FAILED_TO_ENGAGE`,
and the optimizer must not learn from it as if it were one.

## Consequences

- The composer is a feasibility check over four inputs, not two.
- A new EngineAdapter must answer the execution requirements for its seams; it
  need not share SGLang's mode record.
- Every implementation must ship `engaged()`. One that cannot prove it ran
  cannot produce `VALID` results.
- TeaCache on FLUX.2 in SGLang is feasible only in eager mode: whole-model
  compile breaks `override_exact`, and SGLang offers FLUX.2 no regional compile.

## What this rests on

**Runtime validated** (experiments 001 and 002, SGLang @ `8ca82118e`,
Qwen-Image-2.1 and FLUX.2-klein):
- eager and regional compile kept the tested trunk interception exact;
- whole-model compile caused concrete problems: tripled graph breaks, and on
  FLUX.2 an identity override that changed the image;
- graph replay bypassed the trunk interception entirely, while the step seams
  held in every mode tested;
- counting calls, steps and blocks that actually ran detected engagement,
  including its absence.

**Code reading only:** vLLM-Omni compiles only the repeated blocks by default and
has a whole-model option; neither vLLM-Omni nor ComfyUI graph-captures these
models.

**Open:**
- whether two execution requirements are enough for a second engine, and what
  that engine's mode record looks like;
- whether `override_exact` may ever be granted under compile without an
  identity check;
- engagement per execution item rather than per call, once one call serves
  several items.

## Alternatives rejected

- **A universal execution-mode enum** (eager, regional compile, whole compile,
  CUDA graph, …). Those terms describe SGLang. *Revisit when* a second
  EngineAdapter shows a shared mode vocabulary pays for itself.
- **Engine-shaped fields on the implementation**, such as
  `cuda_graph_supported` or "requires whole-model compile off". They put engine
  knowledge into generic code and would be wrong for the next engine.
- **Deriving execution needs from a capability's kind** (observe versus
  decide), with seams classified as running live, traced or bypassed. It
  explains experiment 002 but makes the framework guess. What an
  implementation needs is behavior, and two implementations using the same capability can need
  different behavior; the implementation states it.
- **Counting configuration as engagement.** Experiment 002's graph-mode reuse
  reported "on" and did nothing.
- **One engagement metric for all implementations.** A trunk reuse, an FP8 swap
  and a schedule change leave different evidence.
