# Investigation: do technique → implementation → seam abstractions match SGLang?

Status: current · last substantive update 2026-10-01

Scope: this is an evidence report. It reads the real Qwen-Image and FLUX code
in SGLang diffusion and tests one hypothesis: optimization techniques can be
separated from their implementations, and those implementations can plug into
models and engines through a small set of standard seams. The decisions drawn
from it are recorded in [ADR 0005](adr/0005-target-binding-architecture.md),
[ADR 0006](adr/0006-lifecycle-phases.md) and
[ADR 0007](adr/0007-v1-seams-and-first-techniques.md).

**Source pinned:** upstream `sgl-project/sglang` main at `8ca82118e`
(2026-09-24). Paths below are relative to
`python/sglang/multimodal_gen/`, and `denoising.py` means
`runtime/pipelines_core/stages/denoising.py`. Line numbers are from that
commit. The trace was done by reading code only; nothing was executed on a GPU.

---

## TL;DR

- **The engine owns almost everything.** Request handling, the stage
  pipeline, the denoising loop, CFG, the scheduler, the attention layer, the
  linear layers and the decode stage are all **shared SGLang code**. Qwen and
  FLUX diverge in only two places:
  1. *config callbacks*: prompt post-processing, latent packing, `mu`, and the
     conditioning kwargs;
  2. *the inside of the DiT forward*: the block topology.
- **"Block" is not a portable seam.** The three DiTs have three different trunk
  shapes (see [1.3](#13-the-block-trunk-is-where-the-models-actually-differ)).
  The portable unit is the **trunk**, meaning the whole block stack. Per-block
  hooks need a model-specific description of the trunk.
- **"Model adapter" and "engine adapter" are not independent axes.** SGLang
  *re-implements* each model. `QwenImageTransformer2DModel` in SGLang is not
  the diffusers class, and neither is ComfyUI's. Where to hook depends on
  **(engine, model)** together. What *is* independent of the engine is a small
  set of **model facts**: token layout, timestep scaling, CFG style and
  sensitive layers.
- **Two lifecycle phases are not enough.** SGLang compiles the DiT while it
  *builds the pipeline*, before any request arrives. Anything that replaces
  modules has to run before that. cache-dit has to mount before a deferred
  compile. Breakable CUDA graphs capture during warmup.
- **Many "techniques" are already native SGLang knobs.** They are selected by
  server args, env vars or per-request sampling params. A native
  implementation is therefore mostly a *configuration binding*, not code.
- **Revised shape:** `Technique → Implementation → Target(engine × model) binding`.
  The binding provides a few **seams** (launch, load, request, step, trunk),
  and composition is checked through **resource ownership** plus
  **lifecycle phase** plus **graph-mode compatibility**.

---

## Part 1: the real inference path

### 1.1 Side-by-side call graph

```mermaid
flowchart TD
    subgraph P1["HTTP process (uvicorn)"]
        A1["entrypoints/openai/image_api.py:262<br/>generations()"] --> A2["entrypoints/utils.py:763<br/>prepare_request() → Req"]
        A2 --> A3["scheduler_client.py:255<br/>AsyncSchedulerClient.forward<br/>(pickle over ZMQ)"]
    end
    A3 -. "process boundary<br/>launch_server.py:179 spawn" .-> B0
    subgraph P2["GPU worker process (one per GPU)"]
        B0["managers/worker_bootstrap.py:131<br/>load_plugins() + apply_plugin_hooks()"] --> B1
        B1["managers/scheduler.py:1240<br/>Scheduler.event_loop → _handle_generation:279"] --> B2["managers/gpu_worker.py:501<br/>GPUWorker.execute_forward"]
        B2 --> B3["composed_pipeline_base.py:1063<br/>ComposedPipelineBase.forward"]
        B3 --> B4["executors/parallel_executor.py:35<br/>loop over stages"]
        B4 --> S1["TextEncodingStage.forward<br/>text_encoding.py:395"]
        S1 --> S2["LatentPreparationStage.forward<br/>latent_preparation.py:105"]
        S2 --> S3["TimestepPreparationStage.forward<br/>timestep_preparation.py:78"]
        S3 --> S4["DenoisingStage._denoise<br/>denoising.py:2037 · loop :2076"]
        S4 --> D1["_run_denoising_step :1626"]
        D1 --> D2["_predict_noise_with_cfg :2195<br/>CFGPolicy (distributed/cfg_policy.py)"]
        D2 --> D3["_predict_noise :2481<br/>current_model(**call_kwargs) :2503"]
        D3 ==> M["<b>DiT forward: MODEL-SPECIFIC</b>"]
        M ==> D4["scheduler.step  denoising.py:1689<br/>FlowMatchEulerDiscreteScheduler.step :446"]
        D4 --> D1
        S4 --> S5["DecodingStage.forward<br/>decoding.py:309 → vae.decode"]
        S5 --> B5["gpu_worker.py:1155 _save_output_paths"]
    end
```

The same graph serves both models. What differs is the configuration fed to
the shared stages and the inside of the DiT:

| Step | Qwen-Image | FLUX.1-dev | FLUX.2 / klein |
|---|---|---|---|
| Stage recipe | `add_standard_t2i_stages` (`pipelines/qwen_image.py:49`) | `add_standard_t2i_stages` (`pipelines/flux.py:45`) | `add_standard_ti2i_stages` (`pipelines/flux_2.py:47`) |
| Text encoder | Qwen2.5-VL, drop first 34 tokens (`pipeline_configs/qwen_image.py:63`) | CLIP (pooled) + T5-512 | Mistral3 (dev) / Qwen3 (klein); stacked hidden layers (`pipeline_configs/flux.py:412,424`) |
| Latent pack | `_pack_latents` 2×2 → `(B, H/16·W/16, 64)` | same `_pack_latents` | `flux2_pack_latents` reshape + 4-D `latent_ids` |
| `mu` | `prepare_mu` (`pipelines/qwen_image.py:25`) | `prepare_mu` (`pipelines/flux.py:19`) | empirical, depends on step count (`pipelines/flux_2.py:16`) |
| Scheduler | **shared** `FlowMatchEulerDiscreteScheduler`, cloned per request | shared | shared |
| Guidance | **true CFG**, 2 sequential forward passes (`denoising.py:2297`), then norm-rescale (`qwen_image.py:237`) | **distilled guidance embedding**; no CFG unless a negative prompt is given | klein: neither; klein-base: true CFG |
| Timestep into the DiT | divided by 1000 inside the DiT (`qwen_image.py:2348`) | raw | raw; guidance ×1000 inside the DiT (`flux_2.py:1665`) |
| DiT forward | `qwen_image.py:2287` | `flux.py:1335` | `flux_2.py:1630` |
| VAE | `AutoencoderKLQwenImage` (Wan-style), `optimize_vae` uses the Wan fast path | AutoencoderKL | `AutoencoderKLFlux2` (BN stats), `flux2_vae_cuda_opt.py` |

**Where the paths diverge and converge.** In terms of *control flow*, the two
paths are identical until `_predict_noise` calls `current_model(**call_kwargs)`
(`denoising.py:2503`), and they rejoin as soon as the noise prediction comes
back. In terms of *data*, they diverge earlier, but only through
`PipelineConfig` callbacks: encoder post-processing, pack/unpack,
`prepare_pos/neg_cond_kwargs`, `mu`, and VAE scale/shift. The engine calls
these callbacks; it does not branch on the model.

### 1.2 Stage ownership

| Stage | Location | Shared? | Owner |
|---|---|---|---|
| Request / IPC | `image_api.py:262`, `scheduler_client.py:255`, `scheduler.py:1240` | shared | engine |
| Build and load | `gpu_worker.py:358` → `build_pipeline` (`pipelines_core/__init__.py:35`) → `TransformerLoader.load_customized` (`loader/component_loaders/transformer_loader.py:246`) | shared, plus a per-model `post_load_weights` (`qwen_image.py:2245`, `flux_2.py:1474`) | engine + model |
| Prompt encoding | `TextEncodingStage.encode_text` (`text_encoding.py:573`) | shared stage, model callbacks | engine + config |
| Latent init | `latent_preparation.py:105` | shared stage, model pack | engine + config |
| Timesteps | `timestep_preparation.py:78`, sigmas from `configs/pipeline_configs/base.py:1225` | shared; `mu` is per model | engine |
| Denoising loop / CFG / scheduler step | `denoising.py:2037`, `:2195`, `:1689` | shared | engine |
| DiT forward and block loop | `qwen_image.py:2443`; `flux.py:1451,1461`; `flux_2.py:1724,1746` | **model-specific** | model (SGLang's reimplementation) |
| Attention | `USPAttention.forward` (`runtime/layers/attention/layer.py:899`), `get_attn_backend` (`selector.py:171`) | shared layer, model call sites | engine + model |
| GEMM | `LinearBase.quant_method.apply` (`runtime/layers/linear.py:219,311,513,1199`) | shared | engine |
| VAE decode | `DecodingStage.decode` (`decoding.py:205`) | shared stage, model VAE | engine + model |

### 1.3 The block trunk is where the models actually differ

```mermaid
flowchart LR
    subgraph Q["Qwen-Image (qwen_image.py:2443)"]
        q1["N × QwenImageTransformerBlock<br/>(enc, hid) → (enc, hid)<br/>dual-stream, per-block img_mod/txt_mod"]
    end
    subgraph F1["FLUX.1 (flux.py:1451, 1461)"]
        f1["19 × FluxTransformerBlock<br/>(enc, hid) → (enc, hid)"] --> f2["38 × FluxSingleTransformerBlock<br/>(enc, hid) → (enc, hid)<br/>join/split *inside* each block"]
    end
    subgraph F2["FLUX.2 (flux_2.py:1724, 1746)"]
        g1["Flux2TransformerBlock<br/>(enc, hid) → (enc, hid)<br/>may return deferred gated-residual tuples"] --> j["join_seqs once<br/>:1744"] --> g2["Flux2SingleTransformerBlock<br/>hid → hid (one tensor)<br/>merged QKV+MLP GEMM"] --> s["slice text off<br/>:1760"]
    end
```

The block signatures disagree on argument names (`temb_img_silu`, `temb`,
`temb_mod_params_img`), on return arity, and on where text and image are
joined. Even the *value* a block returns can be a lazy tuple
(`_materialize_gated_residual`, `flux_2.py:1735`). A block wrapper written
against one model breaks on the next.

What does stay common is the **trunk boundary**. Every model computes some
pre-processing, then a trunk mapping `(hidden, encoder_hidden) → hidden`,
then `norm_out` and `proj_out`. SGLang's own Spectrum cache already treats the
whole trunk as the unit it skips (`flux.py:1446-1475`).

---

## Part 2: candidate seams

| Seam | Qwen location | FLUX location | Common abstraction? | Phase | Verdict and risks |
|---|---|---|---|---|---|
| `model_load` | `TransformerLoader.load_customized` → `post_load_weights` (`qwen_image.py:2245`) | same loader, `flux_2.py:1474` | **yes** | load | **Keep, but split it in two:** quantization and QKV fusion are decided at *module construction* (`linear.py:219`, `qwen_image.py:766`, `flux_2.py:514`). Post-hoc replacement after load fights the engine's own post-load hooks, such as the FP8 norm+quant enabled in `post_load_weights`. |
| `denoise_step` | `_run_denoising_step` (`denoising.py:1626`) | same | **yes**, engine-owned | runtime/step | **Keep.** Visible state: `DenoisingContext` (`:258`: timesteps, latents, cond kwargs, cfg_policy, `extra`), `DenoisingStepState` (`:295`: step_index, t, current_model), and the `Req`. The docstring says the method is meant to be overridden. |
| `transformer_forward` | `_predict_noise` (`:2481`) | same | **yes, if treated as opaque** | runtime/step | **Merge into the step seam** as "the denoiser call". The kwargs are model-specific, so treat them as opaque. Enough for whole-output reuse; not enough for residual caching. |
| `transformer_block` | `qwen_image.py:2443` | `flux.py:1451,1461`; `flux_2.py:1724,1746` | **no** | runtime/block | **Reject as a standard seam.** Replace with the **trunk** seam plus an optional model-bound block description. cache-dit already does this, with per-model `BlockAdapter`s and forward patterns (`cache/cache_dit_integration.py:496`, custom adapters at `:405-420`). |
| `attention` | `USPAttention` (`layer.py:899`) | same | **yes**, engine layer | launch / static | **Do not wrap it; select it.** The backend is chosen by `--attention-backend` (`selector.py:171`); 25 backends exist (`platforms/interface.py:29-54`). Per-request override is refused under compile or breakable CUDA graphs (`denoising.py:704-756`). Sparse patterns also need model facts (text-prefix length, image grid). |
| `linear` | `LinearBase.quant_method` | same | **yes**, engine layer | construct + post_load | **Keep as an engine capability.** Model-specific parts: Qwen-Image-2.1 uses plain `nn.Linear` for `img_in`/`proj_out`/modulation, which `quant_method` skips; FLUX.2 single blocks merge QKV+MLP into one GEMM; dense-guard names are per model. |
| `CFG` | `CFGPolicy` (`distributed/cfg_policy.py:34`), swappable via `pipeline_config.cfg_policy` | same | yes, engine | runtime/step | **Fold into the step seam** as a *branch* dimension. FLUX.1-dev and klein have no true CFG by default, so CFG techniques are model-conditional, not universal. |
| `scheduler` | shared FlowMatchEuler, `scheduler_class_override` (`scheduler_loader.py:50`) | same | yes | request | **Keep as the "schedule" part of the request seam.** Timesteps and sigmas are fixed per request in `TimestepPreparationStage`. `mu` is model-specific. |
| `latent/resolution` | `progressive_resolution/qwen_image.py:53` | `progressive_resolution/flux.py`, `flux_2.py` | **no** | request | **Reject as a seam.** Pack/unpack/repack is model-specific, and SGLang already implements it per model behind `progressive_mode`. Treat it as a native-only technique. |
| `VAE` | `DecodingStage` + `optimize_vae` (`platforms/cuda.py:838`) | same | stage is common; module is not | load | **Keep only as a load-time module target**, not as a runtime seam. |

### Seams the hypothesis missed

| Missing seam | Evidence | Why it is needed |
|---|---|---|
| **launch** (server args and env) | Attention backend, `--enable-torch-compile`, `--enable-breakable-cuda-graph`, `--quantization`, `SGLANG_CACHE_DIT_*` | Most native techniques can *only* be expressed here. |
| **request** (per-request sampling params and plan) | `enable_cache_dit`, `progressive_mode`, `cfg_gate_step`, `quality`, `enable_spectrum` (`configs/sample/sampling_params.py:317-345`) | A large share of SGLang's optimizations are toggled per request. |
| **trunk** | see [1.3](#13-the-block-trunk-is-where-the-models-actually-differ) | This is the portable unit for whole-trunk caching (TeaCache residual variants, first-block cache, Spectrum). |
| **worker injection** (engine-internal, not a technique seam) | `worker_bootstrap.py:131`: `load_plugins()` and `apply_plugin_hooks()` run in **every** spawned worker; entry-point group `sglang.multimodal_gen.plugins`, allow-list `SGLANG_PLUGINS`; hooks are BEFORE/AFTER/AROUND/REPLACE on dotted targets (`runtime/platforms/plugins.py:88`) | Every hook we install has to cross a process boundary. Patching the launching process does nothing. |

---

## Part 3: techniques mapped onto the code

"Native" means SGLang already implements it. Q / F1 / F2 are Qwen-Image,
FLUX.1 and FLUX.2.

| Technique | Concept | Native in SGLang | Q / F1 / F2 wired? | Install | Decides at | Seams | Generic implementation realistic? | Compile / graph | Conflicts found in code |
|---|---|---|---|---|---|---|---|---|---|
| **TeaCache** | Skip the trunk when an accumulated, rescaled rel-L1 of the timestep-modulated input stays under a threshold | `cache/teacache.py`, `TeaCacheMixin` in `CachableDiT` | **no / no / no** (Wan, Hunyuan, LingBot only) | request init | per step | step + **trunk** + a model-specific "modulated input" signal | **partially.** The controller is generic; the signal tap and the rescale coefficients are per model. | `.item()` host sync breaks graphs (`teacache.py:196`) | excludes Spectrum (`sampling_params.py:720`); no guard against breakable CUDA graphs |
| **Block cache (DBCache / FBCache)** | Reuse the residual of block `n..N` when the first-`n`-block residual barely changes | cache-dit (`cache_dit_integration.py:496`) | yes / yes / yes | request init, **before compile** | per step, per block | per-block adapter (cache-dit owns it) | **prefer native.** cache-dit already ships per-model forward patterns. | compile deferred until mounted (`denoising.py:544,612`) | refused under breakable CUDA graphs (`:1025`); FSDP; DMD vs TaylorSeer |
| **DPCache** (schedule-searched cache) | — | **not in this commit** (open upstream PR sgl-project/sglang#40848) | — | — | — | — | — | — | — |
| **Step skip / fixed-step reuse** | Reuse or extrapolate the noise prediction on chosen steps | Spectrum (F1 only); cache-dit SCM step masks | no / yes / no; SCM: yes / yes / yes | request | per step | **step only** | **yes, fully.** The noise prediction is an opaque tensor. | needs eager control flow; breaks under breakable CUDA graphs | Spectrum vs TeaCache |
| **Timestep reduction** | Fewer steps or a different sigma schedule | `num_inference_steps`; `scheduler_class_override` | yes / yes / yes | request | request | request (schedule) | yes | changes shapes for nothing; changes graph count for nothing | invalidates per-step schedules of other caches |
| **CFG skip / gate** | Reuse `uncond = cond − cached delta` after step `k` | CFG gate (`denoising.py:1483-1530`) | Q only meaningful (FLUX.1-dev and klein have no true CFG) | request | per step | step (branch dimension) | yes | eager tensor math | disabled with CFG-parallel (`:1498`) |
| **Block skipping** | Drop blocks entirely | SD3 `skip_layers` only | no / no / no | — | per step | trunk + block description | needs model block description | — | quality-sensitive |
| **Progressive resolution** | Coarse-to-fine latent resolution | `progressive_resolution/*` | yes / yes / yes | pipeline build | request (multi-stage) | not a seam (model-specific repack) | **no, keep native** | shapes miss captured graphs | refuses sequence parallel (`progressive_resolution/denoising.py:464`) |
| **NVFP4 linear** | 4-bit weights and activations on chosen linears | `ModelOptFp4LinearMethod` (`modelopt_quant.py:694,798`); `Flux2NvfpPipeline` | Qwen FP4 path / – / yes | construct + post_load | static | linear (`quant_method`) | **prefer native.** It needs a prequantized checkpoint and a kernel. | before compile | forces a precision-preserving attention backend (`transformer_loader.py:507`) |
| **FP8 linear** | 8-bit GEMM | `Fp8Config` (`layers/quantization/fp8.py:77`), `ModelOptFp8Config` | yes / yes / yes | construct (online FP8 quantizes after load) | static | linear | prefer native | before compile | changes which QKV fusion exists (`qwen_image.py:766`, `flux_2.py:514`) |
| **QKV fusion** | One GEMM for Q/K/V | construction-time key remap (`qwen_image.py:303-330`) | quant-dependent / Nunchaku only / FP4/FP8 | construct | static (Qwen added-QKV is a per-request quality site) | linear + module structure | model-specific | before compile | quality sites refused under breakable CUDA graphs (`denoising.py:786`) |
| **Attention backend** | Different kernel, same math (or sparse) | 25 backends, `selector.py:171` | all | launch | static | launch | select, don't implement | per-request override refused under compile or graphs | sparse default, ring vs skip_softmax |
| **Custom fused kernels** | Fuse norm, modulate, RoPE, SiLU-mul | bit-exact gated (`kernels/ops/diffusion`); quality "sites" (`denoising.py:179-240`) | yes / yes / yes | always on, or per request (`quality`) | per call | inside the model (module code) | **model-specific code** (agent-written) | bypassed while compiling or capturing | quality sites vs breakable CUDA graphs |
| **VAE fusion** | Fused GroupNorm+SiLU, channels_last | `optimize_vae` (`platforms/cuda.py:838`), `flux2_vae_cuda_opt.py`, `wan_vae_cuda_opt.py` | yes / yes / yes | post_load | per decode (quality-gated) | load (VAE module) | per VAE class | VAE compile is separate (`decoding.py:170`) | spatial-parallel decode |
| **torch.compile** | Inductor over the DiT | `DenoisingStage._maybe_torch_compile` (`denoising.py:535`) | whole DiT only (no `_compile_conditions` for Q/F1/F2) | **pipeline build** (or deferred) | static | launch | native | — | skipped under breakable CUDA graphs; refuses per-request attention override |
| **Breakable CUDA graph** | Capture the DiT with eager breaks at attention | `breakable_cuda_graph/runner.py` | yes / yes / **no** (allowlist `server_args.py:234`) | lazy; captured at **warmup** | replay | launch | native | replaces compile | cache-dit, quality fusions, attention override; **TeaCache and Spectrum are unguarded and silently become no-ops** |

### Worked example: TeaCache

- **Concept:** skip the transformer trunk when the timestep-modulated input
  has changed little since the last computed step. Reuse the cached trunk
  residual instead.
- **Install:** at request init. Attach the controller, reset its state at step 0.
- **Execute:** at runtime, once per step and per CFG branch.
- **Seams it needs:**
  1. the step seam, for step index, branch and refresh;
  2. the trunk seam, to run the trunk or replay the residual;
  3. a model-bound **signal tap**: the modulated input of the first block.
- **Qwen:** no native wiring. The modulated input comes from `img_mod` in
  the first block (`qwen_image.py:1306`). CFG has two sequential branches,
  and each needs its own state.
- **FLUX.1:** no native wiring. The `temb`-modulated `norm1` of the first
  double block. No true CFG by default.
- **FLUX.2:** no native wiring. Modulation is shared across blocks and
  computed once (`flux_2.py:1669-1671`), which makes the signal cheap.
- **Can the implementation be shared?** *Partially.* The controller and the
  trunk replay are generic. The signal tap and the polynomial coefficients
  are per model.

---

## Part 4: is the Technique / Implementation split useful?

**Test case: "TeaCache" with implementations `sglang-native`, `ours-generic`,
`model-specific`.**

- **`sglang-native-teacache` does not exist for our models.** SGLang's
  TeaCache runs only where the model calls the mixin, which none of Q, F1 or
  F2 does.
- **What SGLang ships for these models is cache-dit's DBCache/FBCache.** Its
  decision signal is a *block residual* difference, not a *timestep-modulated
  input* difference, and its payload is per-block residuals. That makes it a
  **different technique**, not another implementation of TeaCache.

**Rule (adopted):** two implementations belong to the same technique when they
have the same **decision signal**, **decision rule** and **reused payload**.
They may differ only in *mechanism*: where they hook, which kernel they use,
or which language they are written in. If the signal, rule or payload changes,
it is a different technique.

| Question | Answer from the code |
|---|---|
| Can the optimizer pick "TeaCache" without engine details? | **Yes.** The search space is over techniques and conceptual params. *Feasibility* is a query against the target: is any implementation available for (sglang, qwen-image)? |
| Can implementation selection happen later? | **Yes**, at bind time, once the target is known. |
| One shared parameter schema? | **Yes for conceptual params; no for the rest.** Each implementation declares its own extras. |
| Conceptual vs implementation params | Conceptual: `threshold`, `warmup_steps`, `max_consecutive_reuse`, `tail_protect_steps`. Implementation-specific: rescale coefficients (per model), hook target, sync behaviour. |
| Should native have priority? | **By default, yes.** Native implementations already handle the engine's ordering, for example cache-dit deferring compile and mounting before it (`denoising.py:544`). But it is a default, not a rule; the measurement decides. |
| SGLang has cache-dit, not exactly our algorithm? | Register cache-dit as its **own technique** (block-residual cache) and implement TeaCache ourselves. Never label cache-dit "TeaCache"; a mislabeled result cannot be compared with anything. |

---

## Part 5: are "model" and "engine" orthogonal axes?

| Concern | Model? | Engine? | Evidence |
|---|---|---|---|
| Find transformer blocks | yes | **yes** | Module paths and signatures are SGLang's reimplementation (`runtime/models/dits/*.py`), not diffusers' or ComfyUI's. |
| Intercept the denoise loop | no | yes | `DenoisingStage._run_denoising_step` is shared engine code. |
| Replace Linear | partly (which modules, guards) | yes (`LinearBase.quant_method` is SGLang's) | `linear.py:219`; Qwen-2.1's plain `nn.Linear`. |
| Request-local state | no | yes | `Req`, `forward_context` (`managers/forward_context.py:55`), `ctx.extra`. |
| Timestep semantics | yes | no | Qwen divides by 1000 in the DiT; FLUX.2 multiplies guidance by 1000 inside; the scheduler is shared. |
| Attention replacement | yes (token layout) | yes (backend registry) | `selector.py:171` plus `img_shapes`, `txt_seq_lens`. |

**Conclusion: the axes are not independent at the hooking level.** "Where do
I hook block 0 of Qwen?" has a different answer in SGLang
(`QwenImageTransformer2DModel.transformer_blocks`), in diffusers, and in
ComfyUI. What does separate cleanly is:

- **ModelSpec**: engine-independent *facts* about the model. Token layout
  (text prefix + image grid), timestep scale, CFG style, block count and
  roles, layers that are sensitive to precision.
- **Engine adapter**: engine-wide *mechanics*. How to inject into the worker,
  launch options, request params, the step seam, and the graph and compile
  modes.
- **Binding (engine × model)**: a thin map. *Where* the trunk, the signal tap
  and the module groups live for this model in this engine.

---

## Part 6: lifecycle

The proposed two phases (install, execute) are not enough. The compile point
and the graph-capture point are fixed by the engine:

```mermaid
flowchart LR
    L["launch<br/>server args · env"] --> C["construct<br/>quant_method chosen,<br/>QKV fused"]
    C --> P["post_load<br/>weights processed,<br/>post_load_weights(),<br/>optimize_vae"]
    P --> B["pipeline_build<br/>DenoisingStage.__init__<br/><b>torch.compile here</b>"]
    B --> R["request_init<br/>cache-dit mount,<br/>quality sites,<br/>deferred compile"]
    R --> W["warmup<br/>dynamo trace,<br/><b>graph capture</b>"]
    W --> S["step / trunk<br/>runtime decisions"]
    S --> F["request_finalize<br/>state reset"]
```

Evidence:
- **Order of load steps:** `transformer_loader.py:246` → `fsdp_load.py:485-486` → `transformer_loader.py:549`.
- **Compile:** happens in `DenoisingStage.__init__` (`denoising.py:380-383`), which runs during `create_pipeline_stages` (`composed_pipeline_base.py:172`).
- **Request-init order:** `_maybe_enable_cache_dit_and_torch_compile` (`denoising.py:584`) runs, in order:
  1. attention override
  2. quality sites
  3. cache-dit
  4. deferred compile
- **Graph capture:** happens only when `is_warmup` (`breakable_cuda_graph/runner.py:575-584`).

**Rules this imposes:**
1. Anything that changes the module tree (quantization, fusion, module
   replacement) must be in **construct** or **post_load**, which comes
   *before* `pipeline_build`. Changing it later means reloading the model.
2. Runtime policies that use Python control flow (TeaCache, step skip,
   Spectrum) are not compatible with **graph capture**. Captured graphs
   replay without them, and SGLang does not warn.
3. A technique attached at **request_init** that modifies the forward must
   precede a deferred compile. cache-dit is the only native one, and SGLang
   enforces its ordering itself.

---

## Part 7: composition and conflicts

Every field below exists because of a concrete conflict found in the code.
Generic `before`/`after` edges are not needed; phase order covers ordering.

| Field | Values | Concrete conflict it catches |
|---|---|---|
| `phase` | one of the lifecycle stages above | Module replacement after `pipeline_build` silently runs uncompiled or requires a reload. |
| `owns` | resource names: `linear.quant`, `module.qkv`, `trunk.forward`, `step.prediction`, `cfg.branch`, `attention.backend`, `schedule`, `vae.decode` | Two exclusive owners of one resource: NVFP4 vs FP8 on the same linears; TeaCache vs Spectrum vs block cache on the trunk; CFG gate vs CFG-parallel on the branch. |
| `graph` | `eager_only` / `compile_ok` / `capture_ok` | cache-dit, quality sites and attention override are refused under breakable CUDA graphs; TeaCache and Spectrum silently do nothing under capture. |
| `state` | `none` / `request` (and a reset point) | TeaCache and Spectrum keep state on the *module*, reset at step 0 on the positive branch (`teacache.py:287`); progressive resolution re-refreshes the cache per stage (`progressive_resolution/denoising.py:615`). |
| `requires` | engine, model and capability predicates | Breakable CUDA graphs not allowed for FLUX.2; progressive resolution refuses sequence parallel; CFG techniques need true CFG, which FLUX.1-dev lacks; regional compile needs `_compile_conditions`, which Q, F1 and F2 lack. |

Worked combination: **NVFP4 + QKV fusion + torch.compile + TeaCache + step skip**
- NVFP4 and QKV fusion both belong to `construct`. Under FP4 on FLUX.2 they
  are *coupled*: the fused QKV is chosen *because of* the quant config
  (`flux_2.py:514`). Model them as one implementation, or as `requires`.
- torch.compile belongs to `pipeline_build` and is `compile_ok`. It must come
  after both of the above.
- TeaCache and step skip both claim the per-step prediction. Step skip owns
  `step.prediction`, and TeaCache owns `trunk.forward` + `step` reads. They
  can coexist only if one runs inside the other: a skipped step never reaches
  the trunk. The composer must order them explicitly or reject the pair.
- TeaCache is `eager_only` for its control flow. Under torch.compile with
  `fullgraph=False`, its decision forces a graph break each step, which is
  acceptable. Under breakable CUDA graphs it must be **rejected**, because a
  captured graph would replay without it.

---

## Part 8: revised architecture

```mermaid
flowchart TB
    subgraph Agent["Agent / search layer"]
        O["Orchestrator + per-family executors<br/>propose configs, write implementations"]
    end
    subgraph Tech["Technique layer"]
        TR["Technique registry<br/>concept + conceptual param schema"]
        IR["Implementation registry<br/>native bindings · our generic code · model-specific code<br/>declares: phase · owns · graph · state · requires"]
        CV["Composer<br/>feasibility · ownership conflicts ·<br/>phase order · graph-mode check"]
        TR --> IR --> CV
    end
    subgraph Target["Target = engine × model"]
        MS["ModelSpec (engine-independent facts)<br/>token layout · timestep scale · CFG style ·<br/>block roles · sensitive layers"]
        EA["Engine adapter (SGLang)<br/>worker injection (plugin hooks) ·<br/>launch args · request params ·<br/>step seam · graph/compile modes"]
        BD["Binding (SGLang × Qwen, SGLang × FLUX…)<br/>trunk location · signal taps ·<br/>module groups"]
        MS --- BD --- EA
    end
    subgraph Core["Core"]
        RUN["Launch run bundle"] --> MEAS["Measure<br/>CUDA events · clock · controls"] --> GATE["Gates<br/>OFF-identity · engagement ·<br/>LPIPS/SSIM · VLM"] --> FR["Frontier / trace / tiers"]
    end
    O -->|choose techniques + params| TR
    CV -->|resolved plan| Target
    Target --> RUN
    FR -->|feedback| O
    IR -. "native impl = launch/request knobs" .-> EA
    IR -. "our impl = code on seams" .-> BD
```

Where native SGLang features fit: a native implementation is an
`Implementation` whose mechanism is **configuration**. It sets launch args,
env vars or request params through the engine adapter, and declares
`requires: engine=sglang, model∈{…}`. Our generic implementations are code
attached to seams through the binding. Both kinds go through the same
composer and the same Core gates. An engagement check, meaning "did it
actually run", is mandatory for both.

---

## Part 9: minimal interfaces

Every field below is required by a named technique. A field with no
technique behind it was left out.

```python
class TechniqueSpec:                 # "what"
    name: str                        # "teacache", "step-skip", "fp8-linear"
    params: Schema                   # conceptual params only (TeaCache: threshold, warmup, max_reuse)

class ImplementationSpec:            # "how"
    technique: str
    name: str                        # "ours-generic", "sglang-native"
    phase: Phase                     # NVFP4/FP8 → construct; TeaCache → request_init; compile → pipeline_build
    owns: frozenset[str]             # NVFP4 vs FP8 ("linear.quant"); TeaCache vs block cache ("trunk.forward")
    graph: GraphCompat               # TeaCache/step skip → eager_only (silent no-op under capture)
    requires: Predicate              # breakable CUDA graphs ∌ FLUX.2; CFG gate needs true CFG
    extra_params: Schema             # TeaCache coefficients; cache-dit Fn/Bn
    def install(target, params) -> Installed   # sets knobs or attaches hooks
    def engaged(evidence) -> bool              # authenticity: did it actually skip/quantize?

class ModelSpec:                     # engine-independent facts
    cfg_style: Literal["true", "distilled", "none"]   # CFG gate/skip feasibility
    timestep_scale: float                             # TeaCache signal and schedules
    token_layout: ...                                 # sparse attention, token pruning (later)
    sensitive_layers: ...                             # NVFP4/FP8 dense guards

class StepContext:                   # step seam (engine-owned)
    step_index: int; num_steps: int; t: Tensor; branch: Literal["cond", "uncond"]
    latents: Tensor; request_state: dict      # step skip, TeaCache, CFG gate
    def predict() -> Tensor                    # run the denoiser (or not)

class TrunkSeam:                     # binding-provided
    def run(hidden, encoder, **ctx) -> hidden # TeaCache residual replay, block cache
    def signal(...) -> Tensor                  # TeaCache modulated-input tap
```

`EngineCapabilities` is deliberately absent. Its fields would either be
`requires` predicates or questions to the binding, so a separate type adds
nothing.

---

## Part 10: verdict

### What is truly generic (shared by Qwen-Image and FLUX)
- The request → stage → denoise-loop → scheduler → decode skeleton, which
  SGLang itself owns and shares.
- A **step-level** policy that treats the noise prediction as an opaque
  tensor: step skip, fixed-step reuse, extrapolation, schedule changes.
- **Whole-trunk** reuse controllers (TeaCache, first-block cache), *once a
  binding provides the trunk and the signal tap*.
- `LinearBase.quant_method` and the attention backend registry, as engine
  capabilities.
- Measurement and gating (Core).

### What must stay model-specific
- Block topology, arity and stream joins (three different shapes).
- Signal taps and per-model coefficients (TeaCache).
- CFG style (Qwen: true CFG with norm-rescale; FLUX.1-dev: distilled; klein: none).
- Timestep scaling, latent packing, `mu`.
- Which modules are quantizable or fused: Qwen-2.1 plain `nn.Linear`, FLUX.2
  merged QKV+MLP, dense guards.

### What must stay engine-specific
- Worker-process injection: SGLang plugin hooks.
- Launch args and env, per-request sampling params.
- The `DenoisingStage` step seam and `Req`/forward-context state.
- Compile point (pipeline build) and graph capture (warmup) semantics.
- Native techniques: cache-dit, Spectrum, CFG gate, progressive resolution,
  quantization configs, fused-kernel sites.

### Bad abstractions to avoid
- **A universal per-block seam.** It does not survive FLUX.2's
  join-once/return-one-tensor trunk.
- **"Model adapter" independent of the engine.** Hook locations belong to the
  engine's reimplementation of the model.
- **Calling cache-dit "TeaCache".** Different signal, different payload,
  different technique.
- **Runtime module replacement.** Compile has already happened by the time a
  request arrives.
- **A latent/resolution seam.** Repacking is per model; keep progressive
  resolution native.
- **An attention *wrapper* seam.** Attention is selected from a registry.
  Wrapping it fights compile and graph capture.
- **Generic `before`/`after` edges.** Lifecycle phase plus `owns` covers
  every ordering found.

### Recommended V1 architecture
See [ADR 0005](adr/0005-target-binding-architecture.md) and
[ADR 0007](adr/0007-v1-seams-and-first-techniques.md):
- **Seams:** `launch`, `load` (construct and post_load), `request`, `step`,
  and `trunk` (bound per model).
- **Code:** one engine adapter (SGLang, via plugin hooks); two bindings
  (SGLang×Qwen-Image, SGLang×FLUX.2-klein); a composer that checks `phase`,
  `owns`, `graph` and `requires`.

### Recommended first three techniques
| # | Technique | Exercises | Why it validates the architecture |
|---|---|---|---|
| 1 | **Step skip / fixed-step prediction reuse** (our generic implementation) | `step` seam, request state, `eager_only` | Fully engine-side and model-agnostic. If one implementation runs unchanged on Qwen and FLUX, the step seam is real. Graph-mode rejection gets exercised immediately. |
| 2 | **TeaCache** (our generic implementation; SGLang has none wired for Q/F1/F2) | `trunk` seam, model-bound signal tap, per-branch request state | The hardest test of the binding: the same controller with two different trunk shapes and CFG styles. Pairing it with technique 1 tests `owns` and nesting. |
| 3 | **FP8 linear** (SGLang-native implementation) | `launch`/`construct` phase, `linear.quant` ownership, native-as-configuration | Proves that a native implementation is a configuration binding with an engagement check. Interacts with QKV fusion and compile, which tests phase ordering. NVFP4 follows once FP8 works. |

## Open items

- [ ] Verify at runtime that a plugin registered under
      `sglang.multimodal_gen.plugins` runs in the worker and can wrap
      `DenoisingStage._run_denoising_step` (the first Core/engine spike).
- [ ] Check whether FLUX.2 klein's layer count and FLUX.2 forward kwargs
      differ from FLUX.2-dev in the loaded checkpoint config (unverified,
      from config only).
- [ ] Confirm whether the online `--quantization fp8` path runs fused FP8 GEMMs
      on the target GPU architecture for both models, or falls back.
- [ ] Decide whether to vendor the FLUX.1 path at all. FLUX.2-klein is the
      target, and FLUX.1 differs (Spectrum is wired there, a distilled
      guidance embedding, per-block join).
