# Architecture

Status: current · last substantive update 2026-10-02

This is the current design, and the place where its terms are defined. Why
each part is shaped this way is in the ADRs
([0008](adr/0008-capability-layer.md), [0009](adr/0009-lifecycle-constraints.md),
[0010](adr/0010-v1-capabilities-and-first-techniques.md),
[0011](adr/0011-split-step-control.md),
[0012](adr/0012-trunk-capabilities.md),
[0013](adr/0013-feasibility-and-engagement.md),
[0014](adr/0014-batched-execution-owners.md)). The evidence from
SGLang's code is in the [investigation](architecture-investigation.md).

## In short

> A Technique selects an Implementation. The Implementation declares semantic
> requirements. The target resolves them through EngineAdapter and Binding.
> The runtime mode must make those operations feasible. And no benchmark result
> is accepted until runtime evidence proves the Implementation actually engaged.

- The optimizer picks **techniques**, like "TeaCache" or "FP8". It never
  needs to know how an engine or a model is built.
- Each technique has one or more **implementations**. An implementation
  declares three kinds of requirement: abstract **capabilities** (e.g.
  "replace the transformer trunk's output"), **lifecycle** needs, and
  **execution** needs (e.g. "my callback runs on every trunk call").
- A **target** is one engine running one model, such as SGLang running
  Qwen-Image. It answers those needs from three places:
  - what the engine does for every model (**EngineAdapter**);
  - what the model is in any engine (**ModelSpec**);
  - where this model lives inside this engine (**Binding**).
- Each capability is answered by whichever place can provide it. This
  **capability resolution** is why a simple technique never touches
  model-specific code.
- Being provided is not enough. **Execution feasibility** asks whether the
  required operations actually execute correctly in the mode the engine
  applied; only the EngineAdapter can judge that. After the run, **engagement
  verification** asks for runtime evidence that the optimization executed.
  Configured, supported, feasible and engaged are four different states, and
  only an engaged run is a measurement.

## The picture

```mermaid
flowchart TB
    OPT["<b>Optimizer / agent</b><br/>chooses techniques + conceptual params"]
    TEC["<b>Technique</b><br/>what: concept + conceptual param schema"]
    IMP["<b>Implementation</b><br/>capability · lifecycle · execution requirements<br/>owns · engaged()"]
    RES["<b>Capability resolution</b><br/>which layer provides each capability, through which seam"]
    subgraph TGT["Target = one engine × one model"]
        EA["<b>EngineAdapter</b><br/>how this engine works, incl. its execution mode<br/>e.g. SGLangAdapter"]
        BD["<b>Binding</b><br/>where this model lives in this engine<br/>e.g. SGLangQwenBinding"]
        MS["<b>ModelSpec</b><br/>what this model is<br/>e.g. QwenImageSpec"]
        EA ~~~ BD ~~~ MS
    end
    FEAS["<b>Execution feasibility</b><br/>lifecycle · execution requirements in the applied mode · conflicts"]
    RUN["<b>Install and run</b><br/>plugin hooks · launch args · request params"]
    ENG{"<b>Engagement verification</b><br/>runtime evidence"}
    MEAS["<b>Measurement</b><br/>speed · quality gate → measurement history"]
    BAD["<b>Failed / invalid, recorded as a status</b><br/>UNSUPPORTED · INFEASIBLE ·<br/>RUNTIME_ERROR · FAILED_TO_ENGAGE"]

    OPT --> TEC --> IMP -- "required capabilities" --> RES
    RES --- TGT
    RES -- resolved --> FEAS
    FEAS -- feasible --> RUN --> ENG
    ENG -- "VALID run" --> MEAS --> OPT
    RES -- unresolved --> BAD
    FEAS -- infeasible --> BAD
    RUN -- crashed --> BAD
    ENG -- "not engaged" --> BAD
    BAD -. "a fact, never a measurement" .-> OPT
```

## A walk through one example

This is the canonical example: **TeaCache in SGLang**, on Qwen-Image-2.1 or
FLUX.2-klein. Everything below except the measured numbers was exercised in
[experiment 002](../experiments/002-trunk-control/README.md).

```text
Technique                TeaCache, with a threshold
   │
Implementation           fractalyze-teacache
   │  requires           timestep_state · signal_observe · trunk_observe ·
   │                     trunk_output_override · request_local_state
   │  lifecycle          mutates_model · dynamic_in_forward · request_state
   │  execution          runs_every_invocation(trunk) · override_exact(trunk)
   ▼
Capability resolution
   SGLangAdapter         timestep state · request state (per request × CFG branch) ·
                         runtime interception (open a trunk call, stop its blocks,
                         replace its output, pack payloads) · execution-mode interpretation
   SGLangQwenBinding     Qwen trunk edges · Qwen signal · block identity ·
                         "the first call fills the prefix cache, never skip it"
   or SGLangFluxBinding  FLUX.2 trunk edges · FLUX.2 signal · block identity
   ModelSpec             FLUX.2 uses true CFG, so state is kept per branch
   │
Execution feasibility    does the applied compile / graph mode run the trunk
   │                     callback on every trunk call, and use overrides exactly?
Run
   │
Engagement verification  did trunk blocks actually not run on the reused calls?
   ├── yes → VALID: measured for speed and quality
   └── no  → FAILED_TO_ENGAGE: recorded, never measured
```

1. **Requirements.** The implementation names behavior, not engine settings:
   it needs its callback to run on every logical trunk invocation, and what it
   supplies as the trunk's result to be what the model continues with. Payloads
   (`output`, or the residual exit − entry) are packed and applied by the
   adapter; the implementation never looks inside them.
2. **Resolution.** Everything model-specific is in the two Bindings, a few
   class paths and four short functions each. Compile and graph behavior is in
   none of them.
3. **Feasibility** ([ADR 0013](adr/0013-feasibility-and-engagement.md)). The
   adapter reads the mode SGLang applied and answers the two execution
   requirements:

   | Applied mode | Qwen-Image-2.1 | FLUX.2 |
   |---|---|---|
   | eager | feasible | feasible |
   | regional compile | feasible (identity check passed) | not offered by SGLang |
   | whole-model compile | feasible (identity check passed) | **infeasible**: `override_exact` fails |
   | breakable CUDA graphs | **infeasible**: `runs_every_invocation` fails | not offered by SGLang |

   Lifecycle: the hooks must be in place before the model is built, and state is
   set up per owner. Conflicts: TeaCache owns the trunk, so cache-dit cannot run
   with it.
4. **Run.** SGLang loads the plugin in its GPU worker; the hooks attach at the
   seams the adapter and Binding named.
5. **Engagement.** TeaCache's `engaged()` checks, from the adapter's counters,
   that it saw every trunk call and that on every reuse it decided, the trunk's
   blocks did not run. Under graph replay it saw 1 call of 40 and skipped no
   block, so the run is `FAILED_TO_ENGAGE`, not a "no speedup" result.
6. **Measure.** Only a `VALID` run is measured; the threshold that worked is
   measurement history, per model.

Step skip, by contrast, needs only `step_observe`,
`step_prediction_override` and per-request state. It resolves entirely in
SGLangAdapter, never touches a Binding, and its execution requirements held in
every mode tested. That difference is the point of resolution.

## Terms

**Technique.** An optimization idea. It is defined by three things: the
signal it decides from, the rule it decides with, and what it reuses or
replaces. It also has a schema of *conceptual* parameters. Two
implementations are the same technique only if all three match. That is why
cache-dit's block cache is not "TeaCache".

**Implementation.** One concrete way to realize a technique. It is either:
- **native:** it switches on a feature the engine already has, e.g.
  `sglang-native-fp8`;
- **generic:** our own code, written against capabilities, e.g.
  `fractalyze-teacache`.

Both kinds follow the same [contract](#implementation-contract).

**Capability.** An abstract operation an implementation needs, described by
what it lets you do and not by where it lives. Example:
`step_prediction_override` means "return your own prediction for a denoising
step instead of calling the model".

**Seam.** The concrete place where one target provides a capability.
Example: SGLang's `DenoisingStage._run_denoising_step`, reached through a
plugin hook. Seams are private to adapters and bindings, and implementations
never name them. If they did, our "generic" code would really be SGLang code.

**EngineAdapter.** Everything about one engine that is the same for all
models: how its worker processes start, how we inject code into them, where
per-request state lives, when it compiles and records graphs, its shared
denoising loop, and its built-in features. Example: `SGLangAdapter`.

**ModelSpec.** What a model family *is*, true in every engine. Examples:
whether it uses true CFG, how it scales timesteps, how its text and image
streams are arranged. It holds no measured numbers. Example: `QwenImageSpec`.

**Binding.** Glue that only makes sense for one engine × model pair. It says
where a logical part, like "the trunk", lives in that engine's version of the
model, and translates the engine's native state into a capability's standard
form. It sits between EngineAdapter and ModelSpec and replaces neither.
Example: `SGLangQwenBinding`.

**Execution mode.** How the engine actually runs the model for this target:
for SGLang, the compile scope and the graph mode it *applied*, read back from
the engine rather than taken from the request. It is private to the
EngineAdapter.

**Execution requirement.** A behavior an implementation needs from the
runtime at one of its capabilities, stated without naming any engine setting.
V1 has two: `runs_every_invocation` (its code runs on every logical invocation,
e.g. every trunk call) and `override_exact` (what it supplies is what the model
continues with). The EngineAdapter decides whether the applied execution mode
gives them.

**Configured / supported / feasible / engaged.** Four states, none implied by
another. *Configured*: the optimizer selected it. *Supported*: every required
capability resolves on the target. *Feasible*: lifecycle and execution
requirements hold in the applied mode, with no ownership conflict. *Engaged*:
the run's own evidence shows the optimization executed.

**Measurement history.** Everything learned by measuring: good thresholds,
fitted coefficients, layers that are sensitive to precision, the quality cost
of a setting. These depend on model, technique, implementation, hardware,
workload and quality gate together, so they are *not* model facts. In V1 this
is simply what Core records; there is no separate type.

## Where knowledge belongs

Ask in this order:

1. Is it **measured** rather than derived from the architecture? It goes in
   **measurement history**.
2. Is it the **same for many models** in one engine? It goes in the
   **EngineAdapter**.
3. Is it **true of the model in every engine**? It goes in the **ModelSpec**.
4. Does it **only make sense for one engine × model pair**? It goes in the
   **Binding**.

One exception: when the engine already records a per-model fact itself (an
allowlist, a registry), the EngineAdapter reads it from the engine instead of
copying it.

The rule, tested on real cases:

| Fact | Belongs in | Because |
|---|---|---|
| SGLang's denoising loop and CFG handling | EngineAdapter | shared engine code for every model |
| where per-request state lives in SGLang | EngineAdapter | an engine mechanism |
| SGLang compiles at pipeline build and records graphs at warmup | EngineAdapter | the engine's lifecycle |
| Qwen scales timesteps by 1/1000 and uses true CFG with rescaling | ModelSpec | true in the reference implementation too |
| FLUX runs double-stream blocks, then single-stream blocks | ModelSpec | an architecture fact |
| Qwen's trunk is the `transformer_blocks` loop in SGLang's model file | Binding | a location inside SGLang's version |
| Qwen-Image-2.1's first trunk call fills the prefix cache, so it must not be skipped | Binding | how *this engine* implements the model's text conditioning |
| SGLang's FLUX.2 joins the streams once and its single blocks return one tensor | Binding | how *this engine* built that architecture |
| TeaCache coefficients, good thresholds, sensitive layers | measurement history | fitted or measured |
| graph capture is allowed for Qwen but not FLUX.2 | EngineAdapter | SGLang keeps the allowlist itself |

**Cases that split across owners:**
- **Finding linear layers.** "Every SGLang linear has a swappable
  `quant_method`" is engine knowledge. "Qwen-Image-2.1 uses plain PyTorch
  linears for a few layers, and FLUX.2 merges two projections into one" is
  Binding knowledge.
- **Block topology.** "FLUX has double then single blocks" is ModelSpec. "In
  SGLang the single blocks run after a one-time join" is Binding.

## Implementation contract

Every field is here because a real problem in SGLang's code needs it.

| Field | What it says | The problem it handles |
|---|---|---|
| `id`, `technique` | which technique this realizes | one technique can have a native and a generic implementation |
| `kind` | native or generic | they are selected and verified differently |
| `params` | the technique's conceptual parameters, plus its own extras | TeaCache's threshold is conceptual; its coefficients are model-specific |
| `requires` | **A.** the capabilities it needs | decides whether it can run on a target at all |
| `constraints` | **B.** lifecycle needs: `mutates_model`, `dynamic_in_forward`, `request_state` ([ADR 0009](adr/0009-lifecycle-constraints.md)) | some changes must precede compilation; some decide inside the forward; some need setup and cleanup per request |
| `execution` | **C.** execution requirements on the capabilities it uses: `runs_every_invocation`, `override_exact` ([ADR 0013](adr/0013-feasibility-and-engagement.md)) | a capability that exists may still not execute correctly in the applied mode |
| `owns` | resources it needs exclusively | FP8 and NVFP4 both want the linear layers; TeaCache and cache-dit both want the trunk |
| `engaged(evidence)` | a verdict and reason, from counters the adapter reports | an optimization that silently did nothing must not report a speedup |

**The implementation says what behavior it needs, the EngineAdapter says
whether this mode gives it.** A generic implementation never names an engine
flag such as "graph capture off" or "whole-model compile unsupported"; those
are one engine's consequences, worked out by its adapter
([ADR 0013](adr/0013-feasibility-and-engagement.md)).

There is deliberately no list of supported targets. Resolution and feasibility
compute that, so nobody maintains it by hand.

### Choosing between implementations

1. Drop every implementation whose capabilities this target cannot provide
   (`UNSUPPORTED`).
2. Drop every implementation that is not feasible in the applied execution
   mode, with its lifecycle needs and the other chosen implementations
   (`INFEASIBLE`).
3. Default order among the rest:
   1. a native implementation that has already **passed the quality gate on
      this target**;
   2. a generic implementation;
   3. any other native implementation.
4. Nothing left means the technique cannot run on this target in this mode.

This is a default, not a law. When both a native and a generic version
remain, the optimizer can measure both.

### Every run ends in one status

| Status | Means |
|---|---|
| `UNSUPPORTED` | a required capability cannot be resolved |
| `INFEASIBLE` | lifecycle, an execution requirement in the applied mode, or an ownership conflict rules it out |
| `RUNTIME_ERROR` | it crashed |
| `FAILED_TO_ENGAGE` | it ran, but its evidence does not show the optimization executed |
| `VALID` | engaged; the only status that becomes a speed and quality data point |

`FAILED_TO_ENGAGE` keeps "the optimization never executed" apart from "it
executed and gave no speedup". The second is a measurement; the first is not,
and the optimizer must never learn from it as if it were.

## Three techniques, resolved

| | Step skip | TeaCache | FP8 linear |
|---|---|---|---|
| **Implementation** | `fractalyze-step-skip` (generic) | `fractalyze-teacache` (generic) | `sglang-native-fp8` (native) |
| **Needs** | step observe, step prediction override, request state | timestep, request state, trunk observe, trunk output override, signal observe | the engine's FP8 feature |
| **Provided by** | SGLangAdapter only | SGLangAdapter + per-model Binding | SGLangAdapter only |
| **Model-specific code** | none | where the trunk and signal are | none |
| **Installed** | per request; no reload | hooks at the trunk's edges before the model is built; state per request and CFG branch | at server launch |
| **Decides** | every step, outside the transformer | every step, inside the transformer | never (static) |
| **Lifecycle constraints** | `request_state` | `mutates_model`, `request_state` | `mutates_model` |
| **Exclusive resource** | the step's prediction | the trunk | the linear layers |
| **Feasible in (SGLang)** | every mode tested ([exp 001](../experiments/001-step-control/README.md)) | eager; compiled where the identity check passed; never under graph replay ([exp 002](../experiments/002-trunk-control/README.md)) | handled by the engine |
| **Engaged when** | DiT calls fell by the reuses decided | every trunk call seen, and blocks did not run on every reuse decided | the live model's linears report FP8 |

The decomposition is natural for all three. It gets harder the moment we
want *selective* FP8, keeping some layers in full precision. That needs a
generic implementation with `linear_access`, which needs a Binding to name
the layer groups. This is why `linear_access` stays a capability rather than
just an engine flag.

## Capabilities in V1

| Capability | Status | In SGLang, provided by | …at this seam |
|---|---|---|---|
| `step_observe` | common | SGLangAdapter | `_run_denoising_step` (`denoising.py:1626`) |
| `step_prediction_override` | common | SGLangAdapter | `_predict_noise_with_cfg` (`denoising.py:2195`) |
| `step_schedule_mutate` | common | SGLangAdapter | `TimestepPreparationStage.forward` (`timestep_preparation.py:78`) |
| `timestep_state` | common | SGLangAdapter | step state; forward context |
| `request_local_state` | common | SGLangAdapter | the request object, keyed by CFG branch where needed; means isolation per logical execution owner, final ownership model open |
| `trunk_observe` | common, per Binding | SGLangAdapter + Binding | first block call (entry) and the output norm's input (exit) |
| `trunk_output_override` | common, per Binding | SGLangAdapter + Binding | the same edges; blocks return their identity while overridden |
| `signal_observe` | common, per Binding | Binding | e.g. the first block's modulated image input |
| `engine_feature.*` | — | per engine | SGLangAdapter | launch flags, environment, per-request switches |
| `compile_control` | — | per engine | SGLangAdapter | compile and graph-capture flags |
| `block_access` | — | **optional** | Binding, if it can | the target's own block list and signature |
| `linear_access` | — | **optional** | SGLangAdapter + Binding | swappable `quant_method` + named layer groups |
| `attention_access` | — | **optional** | SGLangAdapter | attention backend selector |

"Optional" means a target may not provide it, in which case implementations
that need it are unsupported there. It does not mean the idea was rejected.
Block-level access in particular is expected to be needed later, for
selective block caching, block skipping and per-block precision. It just has
no shared signature across models.

**Execution requirements in SGLang**, as SGLangAdapter answers them:

| Capabilities at | eager | regional compile | whole-model compile | breakable CUDA graphs |
|---|---|---|---|---|
| step seams (denoising loop) | both hold | both hold (code reading: only the blocks are compiled) | both hold | both hold |
| trunk edges (inside the DiT forward) | both hold | both hold after an identity check (Qwen-Image-2.1) | `runs_every_invocation` holds; `override_exact` held on Qwen-Image-2.1, failed on FLUX.2 | both fail, except calls that fall back to eager |

Every cell except the one marked was measured in experiments
[001](../experiments/001-step-control/README.md) and
[002](../experiments/002-trunk-control/README.md).

## What a Binding may contain

A Binding is small and declarative, and that is an invariant, not a style
preference. It may contain:

- where a logical part, such as the trunk, starts and ends in this engine's
  version of the model;
- how the model's structures map to and from the generic forms (e.g. what a
  block returns when told not to compute);
- where a signal can be read, and how to compute it from the model's own
  inputs;
- invariants true only of this engine × model (e.g. Qwen-Image-2.1's first
  trunk call must always run).

It must not contain generic worker or request lifecycle, generic request
state, payload arithmetic, compile policy, CUDA graph policy, benchmarking
logic, or generic engagement logic. Compile or graph behavior enters a Binding
only if evidence shows it is genuinely model-specific; so far none has. When a Binding needs something a second model in the same
engine would also need, that code belongs in the EngineAdapter. A Binding that
keeps growing is turning into a god object. In
[experiment 002](../experiments/002-trunk-control/README.md) each Binding was a
list of class paths and four short functions.

## What is settled, what is open

**Settled** (decisions and rejected alternatives are in the ADRs):
- Implementations depend on capabilities, never on seams.
  [ADR 0008](adr/0008-capability-layer.md)
- EngineAdapter, ModelSpec and Binding are separate.
  [ADR 0008](adr/0008-capability-layer.md)
- Measured values live in measurement history, not in ModelSpec.
  [ADR 0008](adr/0008-capability-layer.md)
- Implementations declare three lifecycle constraints; SGLang's lifecycle
  stages stay inside SGLangAdapter. [ADR 0009](adr/0009-lifecycle-constraints.md)
- Block access is optional and target-specific, not rejected.
  [ADR 0010](adr/0010-v1-capabilities-and-first-techniques.md)
- Step control needs no Binding, and is three capabilities: observe, override
  a prediction, mutate the schedule. Skipping a whole step is not offered,
  because it desyncs the scheduler. All three survive graph capture.
  [ADR 0011](adr/0011-split-step-control.md),
  [experiment 001](../experiments/001-step-control/README.md)
- Trunk control needs a small per-model Binding and is three capabilities
  defined at the trunk's edges. Runtime evidence supports, for Qwen-Image-2.1
  and FLUX.2 in SGLang:
  1. one generic implementation works on both models;
  2. Bindings isolate the structural differences;
  3. whole-trunk operations travel better than a shared per-block interface;
  4. capabilities name operations, not hook mechanisms;
  5. the EngineAdapter / Binding split holds.
  [ADR 0012](adr/0012-trunk-capabilities.md),
  [experiment 002](../experiments/002-trunk-control/README.md)
- Execution feasibility and engagement verification are stages every
  implementation passes; implementations declare execution requirements, the
  EngineAdapter interprets the applied mode, and only an engaged run is a
  measurement. [ADR 0013](adr/0013-feasibility-and-engagement.md)
- In a batched call the EngineAdapter maps rows to owners; a decision by some
  owners is spliced, compute is saved only when all agree, and engagement is
  per owner. Runtime-validated in SGLang only.
  [ADR 0014](adr/0014-batched-execution-owners.md)

**Evidence behind the settled items**, kept visibly apart:
- *Runtime validated* (SGLang, Qwen-Image-2.1 and FLUX.2-klein, experiments
  001 and 002): one generic trunk-reuse implementation works on both models;
  the Bindings isolate the structural differences seen; eager and regional
  compile kept the tested interception exact; whole-model compile caused
  concrete problems; graph replay bypassed the trunk interception; block
  counts detect whether trunk reuse engaged.
- *Code reading only:* every mapping onto vLLM-Omni and ComfyUI. None of it is
  runtime-validated portability.

**Open until an experiment answers it:**
- **Logical ownership under batching, beyond SGLang.** In SGLang the adapter
  maps batch rows to owners, the implementation and Bindings stay unchanged,
  and a reuse saves compute only when every row agrees
  ([ADR 0014](adr/0014-batched-execution-owners.md),
  [experiment 003](../experiments/003-batched-ownership/README.md)). Whether
  that holds where rows are CFG branches (ComfyUI) or under vLLM-Omni's request
  batching is untested.
- **The final ownership model of `request_local_state`:** request × CFG branch,
  mapped onto batch rows in SGLang; no request at all (ComfyUI) untested.
- **The final cross-engine execution-mode model:** are two execution
  requirements enough for a second engine?
- Does a real cache policy built on these capabilities pay off, and with what
  thresholds per model?
- Does linear replacement need a Binding in practice?

The experiments that test these are listed, cheapest-to-falsify first, in the
[investigation](architecture-investigation.md#open-items).

**Deliberately not generalized yet:**
- **Lifecycle.** There is no cross-engine lifecycle model, only three
  constraint flags that each adapter maps onto its own lifecycle.
- **Execution mode.** There is no universal mode enum. Each adapter keeps its
  own record of what the engine applied; only the implementations' execution
  requirements are shared.
- **Blocks.** There is no universal block signature, and V1 has no
  block-level technique.
- **vLLM-Omni and ComfyUI.** No adapter exists. A code reading found a
  plausible seam for every capability in both; how many they can provide at
  runtime is an experiment, not an assumption.
