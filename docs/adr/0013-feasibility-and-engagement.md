# ADR 0013: An implementation must be feasible in the current execution mode, and a result counts only if it engaged

Status: accepted · 2026-10-02 · supersedes the `dynamic_in_forward` constraint of [ADR 0009](0009-lifecycle-constraints.md)

## In short

A target that *provides* a capability may still be unable to use it in the
way it is being run. [Experiment 002](../../experiments/002-trunk-control/README.md)
showed one trunk-reuse implementation that was exact in eager mode, changed
the image under whole-model compile, and silently did nothing under graph
replay. So the framework now asks three questions, in order:

1. **Supported:** can this target provide every capability the implementation
   needs? (capability resolution, as before)
2. **Feasible:** can it provide them *in the execution mode actually applied*,
   given the implementation's lifecycle needs and the other implementations it
   runs with?
3. **Engaged:** did the run prove, from runtime evidence, that the
   optimization path executed?

Configuration alone answers none of these. A result enters measurement history
as a data point only if all three are yes.

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

[ADR 0009](0009-lifecycle-constraints.md) tried to capture this with one flag,
`dynamic_in_forward`, mapped to "refused with graph capture; graph breaks under
torch.compile". The evidence is finer than that: compile was fine for one model
and not the other, and fine for observing but not always for overriding.

## Decision

### Execution mode: shared vocabulary is small, the details stay in the adapter

Each EngineAdapter keeps its own record of the execution mode the engine
*applied* (not requested). For SGLang in V1 that is two fields, both read back
from the engine:

| Field | Values seen |
|---|---|
| compile scope | off · regional (blocks only) · whole model |
| graph mode | off · breakable CUDA graphs |

Implementations never see this record. What crosses the boundary is one fact
per seam, the **interception state**: what happens to code placed at that seam
in this mode.

| Interception state | Meaning | SGLang evidence |
|---|---|---|
| `live` | runs as ordinary Python on every invocation | eager; the step seams in every mode tested |
| `traced` | runs, but inside code the compiler traces; results may differ from eager and graph breaks cost time | trunk edges under regional or whole-model compile |
| `bypassed` | the invocation is replayed from a recording; the seam does not run | trunk edges under graph replay |

### Feasibility follows from the capabilities, not from new flags

Every capability is one of two kinds:

- **observe** (`step_observe`, `trunk_observe`, `signal_observe`,
  `timestep_state`): only reads;
- **decide** (`step_prediction_override`, `step_schedule_mutate`,
  `trunk_output_override`): changes what the model computes.

| Interception state of the seam | observe | decide |
|---|---|---|
| `live` | feasible | feasible |
| `traced` | feasible | feasible **only after an identity check passes** for this target and mode |
| `bypassed` | infeasible | infeasible |

The **identity check** runs the decide capability with an override that returns
exactly what was computed, and requires the final output to be bitwise identical
to a run without the implementation. Its result is measured, so it lives in
measurement history. It must be judged on the final output, never by checks
inside the traced code: those are traced too, and in experiment 002 they
reported "not equal" with zero differing elements.

An implementation therefore declares no execution-mode constraints of its own.
They follow from the capabilities it requires. This replaces ADR 0009's
`dynamic_in_forward`, which said the same thing more coarsely. `mutates_model`
and `request_state` stay as they are.

Feasibility is a check before the run over four things:

| Check | Fails when |
|---|---|
| capabilities | a required capability cannot be resolved on this target |
| lifecycle | `mutates_model` / `request_state` cannot be honored ([ADR 0009](0009-lifecycle-constraints.md)) |
| execution mode | a required capability's seam is `bypassed`, or `traced` without a passed identity check |
| conflicts | two implementations claim the same exclusive resource (`owns`) |

### Engagement is each implementation's own check, on evidence the adapter reports

Each implementation provides `engaged(evidence)`, returning a verdict and a
reason. The adapter supplies the evidence as counters of what actually ran at
each capability's seam, so the implementation stays engine-free. There is no
universal metric; examples:

| Implementation | Engaged when |
|---|---|
| step prediction reuse | DiT calls fell by the number of reuses it decided |
| step schedule mutation | the schedule that ran is as long as it asked for |
| trunk reuse (TeaCache) | it observed every trunk invocation the request made, and for each reuse it decided, the trunk's blocks did not run |
| native FP8 | the live model's linear layers report the FP8 method |
| an engine feature (compile, graph mode) | the engine's applied settings, plus its own counters (graph replays) |

Feasibility is a prediction; engagement is the proof. The two can disagree:
under graph mode, a request whose shape was never captured runs eager, so the
same configuration can be `bypassed` for one request and `live` for the next.

### Every run ends in exactly one status

| Status | Means | Enters measurement history as |
|---|---|---|
| `UNSUPPORTED` | a capability cannot be resolved on this target | a fact about the target |
| `INFEASIBLE` | lifecycle, execution mode or a conflict rules it out | a fact about the target and mode |
| `RUNTIME_ERROR` | it crashed | a failure |
| `NOT_ENGAGED` | it ran, but the evidence does not show the optimization executed | a failure, **never a speed or quality point** |
| `VALID` | engaged | a speed and quality data point |

`NOT_ENGAGED` is what separates "configured but never executed" from "executed
and gave no speedup". Only `VALID` is a measurement.

## Consequences

- The composer becomes a feasibility check over four inputs, not two.
- A new EngineAdapter must report interception states for its seams. It need not
  share SGLang's mode record.
- Every implementation must ship `engaged()`. A native implementation with no
  way to prove it ran cannot produce `VALID` results.
- TeaCache on FLUX.2 in SGLang is feasible only in eager mode: whole-model
  compile failed the identity check, and FLUX.2 has no regional compile there.

## What this rests on

**Runtime validated** (experiments 001 and 002, SGLang, Qwen-Image-2.1 and
FLUX.2-klein): the interception states of the step seams and trunk edges in
eager, regional compile, whole-model compile and breakable graph mode; the
identity check passing for Qwen-Image-2.1 under both compile scopes and failing
for FLUX.2 under whole-model compile; engagement by counting calls, steps and
blocks that ran.

**Code reading only:** that vLLM-Omni compiles only the repeated blocks by
default and has a whole-model option, and that neither vLLM-Omni nor ComfyUI
graph-captures these models.

**Open:** whether three interception states are enough for a second engine;
whether `traced` should ever be accepted for decide capabilities without an
identity check.

## Alternatives rejected

- **A universal execution-mode hierarchy.** One engine has been measured. The
  interception state is the smallest shared fact that explains every result.
  *Revisit when* a second EngineAdapter needs a state these three cannot express.
- **Implementations declare engine-shaped constraints** (e.g. "requires graph
  capture off"). That puts engine knowledge into generic code, and the same
  fact is already implied by the capabilities required.
- **Count configuration as engagement.** Experiment 002's graph-mode reuse
  reported "on" and did nothing.
- **One engagement metric for all implementations.** A trunk reuse, an FP8 swap
  and a schedule change leave different evidence.
