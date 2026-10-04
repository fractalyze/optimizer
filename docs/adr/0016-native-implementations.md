# ADR 0016: A native implementation is configuration plus an engagement check; the adapter owns the engine's silent fallbacks

Status: accepted · 2026-10-04 · applies [ADR 0013](0013-feasibility-and-engagement.md) to native implementations

## In short

[Experiment 005](../../experiments/005-native-fp8/README.md) ran SGLang's
native FP8 through the whole optimizer path on Qwen-Image-2.1 and FLUX.2-klein.
It needed no Binding and no model-specific code, and it engaged on both
(1.57× and 1.50× faster). It also showed that one engine flag can silently run
a different numerical method, or only part of the model. So: a Technique names
a numerical method, not a flag; a native implementation needs one
`engine_feature.*` capability and an engagement check; the EngineAdapter owns
every condition under which the engine accepts the flag but runs something
else; and a technique applied at load is part of the model's identity.

## Context

[ADR 0010](0010-v1-capabilities-and-first-techniques.md) chose native FP8 as a
first technique because it "just switches on an engine feature", and the
architecture took "the live model's linears report FP8" as its engagement
evidence. The investigation's item 4 asked whether a
native implementation really is just configuration plus that check, or needs
glue for models such as Qwen-Image-2.1 that keep some plain `nn.Linear`.

| Question | Answer in SGLang |
|---|---|
| Model-specific glue? | None. The model's code decides which layers are quantizable; the flag quantizes all of them |
| One flag, one method? | No. `quantization="fp8"` runs W8A8, or silently weight-only FP8 below sm_89 or with `SGLANG_FORCE_FP8_MARLIN`, or silently drops layers named by `SGLANG_FP8_IGNORED_LAYERS` |
| Does SGLang report the switch? | A log line, which on this sm_120 GPU said the GPU lacked FP8 support |
| Would the wrong method look like a result? | Yes: weight-only gave 1.03×, partial gave 1.33× at better PSNR |
| Does FP8 disturb a trunk Binding? | No; it changes `quant_method` and weight dtype, not modules |
| When does it apply? | At load, fixed for the server's life |

## Decision

### A Technique names a numerical method

`fp8_w8a8_dynamic_linear` (weights quantized at load, activations per call
with runtime scales) and `fp8_weight_only_linear` are different Techniques,
because they compute different things and pay off differently. An engine
reaching both through one flag is the adapter's concern. Details that do not
change the method, such as per-channel or per-tensor weight scales, are
recorded on the implementation; they become a split only if measurements
show they behave differently.

### A native implementation needs one capability and an engagement check

Its `requires` is a single `engine_feature.<technique>` capability: the engine
implements the whole method, and the capability says this build has it. It
installs nothing and needs no Binding; coverage is the model's own code's
decision, observed rather than configured. Its `engaged` check verifies the
**method and its coverage**: every quantizable layer became the Technique's
kind (none weight-only, none left in 16-bit), and the fast path actually
executed during the measured requests.

### The EngineAdapter owns the engine's silent fallbacks

`feasible` must return `INFEASIBLE` for every known condition under which the
engine accepts the configuration but runs a different method or coverage: GPU
generation, toolkit version, engine environment variables, and settings that
change coverage. Engagement verification remains the backstop for what
feasibility does not know. Engine log lines are not evidence.

### A load-time technique is part of the model's identity

A technique with `mutates_model` that is applied at load changes the model the
server serves. Turning it on or off means a new server, so Core groups trials
by server configuration, and two measurements are comparable only if they were
taken on the same one.

## Consequences

- No new contract field. `engine_feature.*` is a capability like any other,
  resolved by the EngineAdapter alone.
- Each EngineAdapter carries a list of its engine's silent fallbacks per
  native feature, kept at the engine version it was traced against.
- Search over load-time techniques costs a reload per configuration, so the
  optimizer should batch trials sharing a server configuration.
- `VALID` still says nothing about quality: FLUX.2's FP8 image engaged and
  diverged in composition (18.5 dB pixel PSNR). The quality gate needs a
  metric that tolerates sampler divergence; pixel PSNR is not it.

## What this rests on

**Runtime validated** (experiment 005, SGLang @ `8ca82118e`, RTX 5090 sm_120,
CUDA 13.0, eager, one request at a time): W8A8 FP8 engaged on 224/224 and
101/101 quantizable layers with every GEMM in FP8; forced weight-only and
partial runs ended `FAILED_TO_ENGAGE`; the forced weight-only plan was
`INFEASIBLE` before launch; FP8 with experiment 002's trunk probe kept the
identity override exact and skipped blocks on reuse.

**Code reading only:** the sm < 89 fallback (not runnable on this GPU), and
vLLM-Omni's FP8 path at `68003cf6a` (the same method, per-tensor weight
scales, a different evidence collector).

**Open:**
- FP8 under torch.compile and graph replay.
- Selective FP8 (some layers kept in 16-bit on purpose), which needs a generic
  implementation with `linear_access` and a Binding to name layer groups.
- A quality metric for the gate that is not pixel PSNR.

## Alternatives rejected

- **One Technique per engine flag (`fp8`).** It would merge weight-only and
  W8A8 results into one measurement history, and the optimizer would learn
  from a method it did not choose.
- **Trust the configuration once it is accepted.** Both negative runs were
  accepted configurations that ran something else.
- **Engagement as "some layer is FP8".** The partial run had 96 FP8 layers and
  would have passed, as a faster, better-looking FP8 that was not the
  Technique.
- **A Binding for FP8 coverage.** The model already decides coverage by which
  layer classes it builds; a Binding would duplicate that. It becomes needed
  only for selective FP8.
