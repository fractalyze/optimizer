# ADR 0007: V1 seams and the first three techniques

Status: accepted · 2026-10-01
Evidence: [architecture investigation](../architecture-investigation.md), parts 2, 3, 7 and 10.

## Decision: V1 seams

| Seam | Owner | Visible state | Used by |
|---|---|---|---|
| `launch` | engine adapter | server args, env | native attention / compile / quant |
| `load` (`construct`, `post_load`) | engine adapter + binding module groups | module tree before compile | FP8 / NVFP4, fusions |
| `request` | engine adapter | sampling params, schedule (timesteps) | step-count / schedule changes, native per-request toggles |
| `step` | engine adapter (`DenoisingStage._run_denoising_step`) | step index, t, num steps, CFG branch, latents, request state | step skip, CFG gate, TeaCache |
| `trunk` | binding | run trunk or replay residual; signal tap | TeaCache, first-block cache |

Not seams in V1:
- **Per-block.** Not portable across the three trunk shapes. Reached only
  through cache-dit, natively.
- **Attention.** Selected from a registry at `launch`, never wrapped.
- **Latent / resolution.** Native progressive resolution only.

Composition metadata is exactly `phase`, `owns`, `graph`, `state`,
`requires`. Each one was introduced by a conflict found in the code.

## Decision: first three techniques

1. **Step skip / fixed-step prediction reuse** (ours). Exercises the `step`
   seam and model-agnosticism, and `eager_only` rejection under graph capture.
2. **TeaCache** (ours; SGLang wires none for Qwen-Image or FLUX). Exercises
   the `trunk` seam, model-bound signal taps, per-CFG-branch state, and
   ownership when combined with step skip.
3. **FP8 linear** (SGLang-native). Exercises native-as-configuration, the
   `construct` phase, `linear.quant` ownership, and the engagement check.

Models: Qwen-Image and FLUX.2-klein. FLUX.1 is deferred.

## Consequences

- The first build step is a spike, not a framework: show that a plugin
  registered under `sglang.multimodal_gen.plugins` runs in the worker and can
  wrap `_run_denoising_step`.

## Deferred

- **Block-residual caching of our own.** Revisit after TeaCache shows the
  trunk binding holds on both models.
- **NVFP4.** Revisit after FP8 proves native-as-configuration and the
  engagement check.
- **FLUX.1.** Revisit if a benchmark target needs it.
