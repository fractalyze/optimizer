# Experiment 005: is a native implementation just configuration plus an engagement check?

Status: done · 2026-10-04 · SGLang @ `8ca82118e` · RTX 5090 (sm_120), CUDA 13.0 ·
outcome recorded in [ADR 0016](../../docs/adr/0016-native-implementations.md)

## In short

Native FP8 went through the whole optimizer path on SGLang, on Qwen-Image-2.1
and FLUX.2-klein: a Technique, a native Implementation, the capability it
needs, an SGLangAdapter that knows the flag, a feasibility check, a launch, and
a status computed from what the live engine did. The claim survives:

- **No model-specific glue.** One flag, `quantization="fp8"`, and the same
  engine-level probe on both models. No Binding. Each model's own code decides
  which layers can be quantized (Qwen-Image-2.1: 224 of 232 linears; FLUX.2:
  101 of 109), and the engine quantized all of them.
- **It engaged and it paid off.** Every quantizable layer became FP8 W8A8, and
  every one of their GEMMs ran in FP8 with a per-token activation quantization
  (27 552 per Qwen request). Qwen-Image-2.1 got **1.57× faster** and used
  **41 % less peak memory**; FLUX.2-klein got **1.50× faster** and 22 % less.
- **Configuration is not evidence.** With the same flag, SGLang silently ran a
  *different numerical method* (weight-only FP8, no speedup) or quantized only
  part of the model. Feasibility caught the first before launch; engagement
  verification caught both after the run, and neither reached a measurement.
- **"Engaged" is not "good".** FLUX.2's FP8 image is a valid but different
  picture (18.5 dB), Qwen's a close one (30.4 dB). That is the quality gate's
  job, not the engagement check's.
- **FP8 composes with trunk control.** Experiment 002's trunk probe ran
  unchanged on the FP8 model: identity exact, reuse skipped blocks.

## The path, as run

```text
fp8_w8a8_dynamic_linear            Technique: a numerical method, no engine words
  → sglang-native-fp8-w8a8         Implementation: native, needs one capability
  → engine_feature.fp8_w8a8_dynamic_linear
  → SGLangAdapter.supported        does this build provide it?
  → SGLangAdapter.feasible         GPU, CUDA, env vars that change the method
  → SGLangAdapter.configure        {"quantization": "fp8"}
  → launch, requests
  → Evidence (from the live engine) → VALID | FAILED_TO_ENGAGE
```

`optimizer_path.py` is that path and nothing more: pure Python, no SGLang
import, tested on a CPU. The Technique and the Implementation never see a flag,
a GPU rule or a class name; only `SGLangAdapter` does.

## A. What SGLang's native FP8 actually is

Traced at the pinned commit; paths under `python/sglang/` unless noted.

| Question | Answer | Where |
|---|---|---|
| How is it selected? | one server argument, `quantization="fp8"`, for the DiT | `multimodal_gen/runtime/server_args/server_args.py:400-402` |
| What does it compute? | **online W8A8**: weights quantized to FP8 once after loading, per output channel; activations quantized on every call, per token, scales computed at runtime | `multimodal_gen/runtime/layers/quantization/fp8.py:77-100`; `srt/layers/quantization/fp8_utils.py:2063`, per-token at the dynamic branch |
| Which kernel? | CUTLASS `fp8_scaled_mm` (a tuned Triton tile only where a tuned config exists; none was used here) | `fp8_utils.py:2104-2120` |
| Which layers? | every `LinearBase` the model builds; plain `nn.Linear` is never touched | `multimodal_gen/runtime/layers/linear.py:214-219` |
| When? | at load, fixed for the server's life; nothing per request | `multimodal_gen/runtime/utils/quantization_utils.py:60` |
| Does it change module classes? | no: only `quant_method` and the weight's dtype | `linear.py:214-226` |
| GPU check? | none for the DiT. Below sm_89 it **silently** switches to Marlin weight-only FP8 (W8A16); `SGLANG_FORCE_FP8_MARLIN` does the same anywhere | `fp8.py:126-130`, `fp8_utils.py:2334-2340` |
| Coverage knobs? | `SGLANG_FP8_IGNORED_LAYERS` (env) and `quantization_ignored_layers` (server argument) drop layers by name, silently | `srt/layers/quantization/fp8.py:278` |

Other FP8 methods exist and are **different techniques**, not variants: modelopt
FP8 (static scales, pre-quantized checkpoint), mxfp8 (1×32 blocks, sm ≥ 100,
and it silently falls back to plain fp8 below that). `weight_only_fp8` and
`comfy_fp8` exist but are not selectable from the command line.

## B. Taxonomy: what the Technique is

The Technique is `fp8_w8a8_dynamic_linear`: *linear layers compute in FP8;
weights are quantized once at load, activations per call with runtime scales.*
It names a numerical method, not a flag. `fp8_weight_only_linear` is a
separate Technique, because it computes something different, costs different
memory and speeds up different workloads, even though SGLang reaches it with
the same flag. mxfp8 and static modelopt FP8 would be two more.

Weight-scale granularity is an implementation detail and is recorded, not
part of the Technique: SGLang scales weights per channel, vLLM-Omni per tensor
(section H). If the quality gate later shows the two behave differently, that
is the evidence to split them.

## C. The implementation

| Field | Value |
|---|---|
| `id` | `sglang-native-fp8-w8a8` |
| `kind` | native |
| `requires` | `engine_feature.fp8_w8a8_dynamic_linear` |
| `lifecycle` | `mutates_model` (changes weights at load) |
| `execution` | none |
| `owns` | `linear_layers` |
| `engaged` | every quantizable layer is FP8 W8A8, none is weight-only, and FP8 GEMMs and activation quantizations actually ran |

`engine_feature.*` is a new kind of capability: the engine already implements
the whole technique, and the capability says only that this build has it.
The implementation installs nothing and needs no Binding.

## D. Feasibility: the adapter owns the silent fallbacks

`SGLangAdapter.feasible` says no when SGLang would run the flag but not the
Technique:

| Condition | Why it matters | Verdict |
|---|---|---|
| GPU below sm_89 | SGLang silently runs Marlin weight-only | `INFEASIBLE` |
| sm_89 with CUDA < 12.4 | no FP8 GEMM | `INFEASIBLE` |
| `SGLANG_FORCE_FP8_MARLIN` set | weight-only, on any GPU | `INFEASIBLE` |
| `SGLANG_FP8_IGNORED_LAYERS` set | only part of the model is quantized | `INFEASIBLE` |

None of these is an error in SGLang. With Marlin forced on this sm_120 GPU,
SGLang's only signal was a log line claiming the GPU *"does not have native
support for FP8"*, which is false here. A log line is not evidence.

## E. The engagement probe

`opt_fp8_probe` is an observation-only SGLang plugin. Every hook sits on an
engine-level class or function, the same for both models:

| What | Hook | Records |
|---|---|---|
| inventory | after `process_model_weights_after_loading` | each linear layer classified as `fp8_w8a8`, `fp8_weight_only`, `unquantized` (quantizable, left in 16-bit) or `not_quantizable` (plain `nn.Linear`); weight-scale granularity |
| runtime | `Fp8LinearMethod.apply` and the kernels it dispatches to | W8A8 GEMMs, weight-only GEMMs, per-token activation quantizations, which GEMM kernel |
| request | around `DenoisingStage.forward` | denoise time and peak allocated memory per request |

`run.py` turns that into an engine-neutral `Evidence` and the
implementation's `engaged` function decides. The configuration is never read.

## F. Results

Qwen-Image-2.1: 1024 × 1024, 40 steps, CFG off, DiT resident, text encoder
layerwise-offloaded. FLUX.2-klein-base-4B: 1024 × 1024, 50 steps, CFG 4. Eager,
one request at a time, three requests per run; timing is the median of the
second and third. PSNR is against the baseline run's image, same seed.

| Run | Status | Layers (FP8 W8A8 / quantizable) | Per request | Wall | Speedup | Peak alloc | vs baseline |
|---|---|---|---|---|---|---|---|
| q-baseline | baseline | 0 / 0 (232 plain) | | 13.74 s | 1.00× | 15.74 GiB | |
| **q-fp8** | **VALID** | **224 / 224** | 9 184 W8A8 GEMMs, 9 184 act. quant. | **8.77 s** | **1.57×** | **9.30 GiB** | 30.38 dB |
| q-fp8-marlin-planned | INFEASIBLE | not launched | | | | | |
| q-fp8-marlin-forced | FAILED_TO_ENGAGE | 0 / 224 (224 weight-only) | 9 184 weight-only GEMMs | 13.31 s | 1.03× | 9.26 GiB | 31.73 dB |
| q-fp8-partial-forced | FAILED_TO_ENGAGE | 96 / 224 (128 left 16-bit) | 3 936 W8A8 GEMMs | 10.32 s | 1.33× | 11.29 GiB | 34.32 dB |
| f-baseline | baseline | 0 / 0 (109 plain) | | 18.01 s | 1.00× | 15.57 GiB | |
| **f-fp8** | **VALID** | **101 / 101** | 10 100 W8A8 GEMMs, 10 100 act. quant. | **12.00 s** | **1.50×** | **12.19 GiB** | 18.48 dB |

Denoising alone: Qwen 13.05 → 8.12 s (1.61×), FLUX.2 17.71 → 11.72 s (1.51×).
Every FP8 GEMM went to CUTLASS; no Triton or dequantizing fallback ran.
Outputs were deterministic: r0, r1 and r2 of a run were identical.

What the numbers say:

- **The counts add up.** FLUX.2: 101 layers × 50 steps × 2 CFG branches =
  10 100. Qwen: 224 × 40 = 8 960, plus 224 for the text prefix, which runs
  through the same linears once, on the first step
  (`multimodal_gen/runtime/models/dits/qwen_image21.py:316,332`).
- **The two negative runs are exactly the cases configuration cannot see.**
  The Marlin run is FP8 by every configuration measure, saves the same memory
  and gives almost no speedup (it also took 125 s to load instead of ~20 s).
  Had it been measured as `fp8_w8a8_dynamic_linear`, the optimizer would have
  learned that FP8 does not speed up Qwen. The partial run is faster than
  baseline and better in PSNR; measured as the Technique, it would have
  looked like a better FP8.
- **Quality is a separate question.** Qwen's FP8 image is the same picture
  with small differences. FLUX.2's keeps the prompt and legible text but
  shifts the composition (sign style, camera, street furniture); a pixel
  metric scores that 18.5 dB. Whether either passes is for the quality gate,
  and pixel PSNR is the wrong gate for a sampler that diverges this way.
- Load times: the first run of the batch (q-baseline, 38 s) read cold weights;
  the others took 20–26 s. No FP8 load-time claim is made.

## G. Composition with trunk control

`q-fp8-trunk` loaded experiment 002's trunk probe next to FP8, with its
Qwen-Image-2.1 Binding unchanged:

| Request | Trunk calls | Blocks skipped | W8A8 GEMMs | Result |
|---|---|---|---|---|
| observe | 40 | 0 | 9 184 | same image as q-fp8 |
| identity | 40 | 0 | 9 184 | **identical to observe** |
| reuse (K = 20) | 40 | 32 | 8 960 | one trunk call skipped: 224 fewer FP8 GEMMs |

The Binding needed nothing, because FP8 changes `quant_method` and weight
dtype, not the module classes or structure the Binding names. The two
implementations own different resources (`linear_layers`, the trunk) and did
not interfere. One FP8 + reuse run is not a claim about their combined
quality.

## H. The same Technique on a second engine (code reading only)

vLLM-Omni @ `68003cf6a` (vLLM 0.30.0) also selects FP8 with `"fp8"`, and it is
the same Technique with a different implementation:

| | SGLang | vLLM-Omni |
|---|---|---|
| method | online W8A8, dynamic per-token activations | the same |
| weight scale | per channel | **per tensor** (`Fp8PerTensorOnlineLinearMethod`) |
| kernel choice | CUTLASS, or Triton where tuned | first kernel in a priority list that passes; logged as `Selected <Kernel> for Fp8PerTensorOnlineLinearMethod` |
| weight-only fallback | silent below sm_89 or when forced | only when CUTLASS is disabled (`use_marlin`) |
| layers left out | plain `nn.Linear` | Qwen-Image's model code excludes embedders, modulations, `norm_out` and `proj_out` |
| when | load only | load only |
| engagement evidence | `quant_method` class, FP8 weight dtype, `use_marlin`, the kernel-selection log line | |

So a second engine is one more Implementation row with the same `requires`
and the same `engaged` contract; only its adapter's evidence collector and
feasibility rules differ. `optimizer_path.py` carries that row, unused.

## I. What changed in the design

1. **A Technique names a numerical method, not a flag.** One flag reaching
   two methods is the engine's business; the taxonomy splits them.
2. **A native implementation is configuration plus an engagement check,** and
   needs no Binding: the model's own code decides coverage, and the adapter
   observes it.
3. **The adapter owns the engine's silent fallbacks.** Feasibility must know
   every condition under which the engine accepts the flag but runs something
   else; engagement verification catches whatever feasibility missed.
4. **Engagement checks the method and its coverage**, not "FP8 is on": the
   kind of layer each one became, and that the fast path executed.
5. **Load-time techniques are part of the model's identity.** Turning FP8 on
   or off means reloading; the optimizer must group trials by server
   configuration, not by request.

## J. Next

Architecture experiments end here. The next stage builds the real pieces, one
at a time, starting with the technique registry (Technique, Implementation,
capability resolution) and the status contract of ADR 0013, using
`optimizer_path.py` as the reference for the native case.

## Limits

One GPU (sm_120), one engine version, eager mode, three requests per
configuration, one prompt per model. Compiled and graph-replayed FP8 were not
run. vLLM-Omni's FP8 was read, not run. Quality was measured with pixel PSNR
only. Feasibility checks the env-var coverage knob but not
`quantization_ignored_layers`; the adapter never sets it, and engagement
would catch it.

## Files

| File | What |
|---|---|
| `optimizer_path.py` | Technique, Implementation, Evidence, SGLangAdapter, `plan`, `verdict` |
| `test_optimizer_path.py` | CPU contract tests: `python -m pytest test_optimizer_path.py` |
| `opt_fp8_probe/` | the observation-only SGLang plugin (inventory, runtime counts, per-request timing) |
| `run.py` | one engine configuration through the optimizer path, ending in a status |
| `analyze.py` | statuses, evidence, timing, memory and image comparisons per run |
| `plans/` | Qwen-Image-2.1, FLUX.2-klein, and Qwen with trunk control |

Needs experiment 002's probe installed (`opt-trunk-probe`) for the trunk run.
Run bundles stay outside the repository.
