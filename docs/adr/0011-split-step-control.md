# ADR 0011: Split step control into observe, prediction override and schedule mutation

Status: accepted · 2026-10-01 · amends the capability list of [ADR 0010](0010-v1-capabilities-and-first-techniques.md)

## In short

`step_control` was one capability: observe each step and decide to compute,
reuse or replace its prediction. [Experiment 001](../../experiments/001-step-control/README.md)
showed that the same engine-level code works on Qwen-Image-2.1 and FLUX.2-klein,
so step control needs no Binding. But "skip this step" turned out not to be one
safe operation. Step control is now three capabilities, all provided by the
EngineAdapter, and skipping a whole step is not offered at all.

## Context

ADR 0010 listed `step_control` as a common capability provided by
SGLangAdapter at `_run_denoising_step`. Experiment 001 tried to falsify that
with one plugin on both models. Four ways to act on step K behaved differently:

| Operation | Result on both models |
|---|---|
| observe the step | bitwise identical to no plugin, eager and compiled |
| replace the step's prediction with the previous one | one fewer DiT call, schedule intact |
| remove the step from the schedule before the loop | one fewer step, schedule intact |
| skip the whole step | **every later step uses the wrong sigma**, and nothing reports it |

The last row fails because SGLang's flow-match scheduler counts its own steps
and picks the sigma from that count. Skipping a step without telling the
scheduler leaves the count one behind. A capability that silently corrupts the
output when used naively is not one we should expose.

## Decision

| Capability | Lets an implementation… | Provided by |
|---|---|---|
| `step_observe` | see each step's index, timestep and schedule length | EngineAdapter |
| `step_prediction_override` | return its own prediction for a step instead of calling the DiT; the scheduler update still runs | EngineAdapter |
| `step_schedule_mutate` | change the timestep schedule before denoising starts, e.g. drop steps | EngineAdapter, which owns the scheduler knowledge |

- **Whole-step bypass is not a capability.** A technique that wants to save a
  step uses `step_prediction_override` (the step still advances the scheduler)
  or `step_schedule_mutate` (the step never exists).
- **Step skip** now requires `step_prediction_override` (plus `step_observe`
  and request state), not `step_control`.
- All three are request-local: the experiment showed control set for one
  request never leaks into the next.

## Consequences

- Step skip still resolves entirely in the EngineAdapter. The separation that
  ADR 0010 wanted to test, techniques with and without a Binding, holds.
- The EngineAdapter must carry per-request control into the worker itself.
  SGLang has no per-request field for plugins; the experiment used a table
  keyed by `request_id`.
- Core cannot assume a plugin loaded: SGLang skips a failing plugin with only a
  log line. `engaged()` must rest on what the worker actually reported.
- An adapter must read the graph and compile settings the engine *applied*.
  SGLang turns a refused graph mode into a warning and runs eager.
- The three capabilities survive graph capture: on Qwen-Image-2.1 with
  breakable CUDA graphs, steps replayed from graphs and every request was
  bitwise identical to eager. This holds because all three act outside the
  DiT call, which is the unit SGLang records.
- "Graph capture on" does not mean a request used it. Qwen-Image-2.1 replays
  only at the warmup prompt's exact length and otherwise runs eager, reported
  once per process. `engaged()` for anything that relies on graph replay must
  count replays, not read the flag.

## Alternatives rejected

- **Keep one `step_control` and document the bypass hazard.** A capability that
  is wrong by default invites the silent failure the experiment found.
  *Revisit if* an engine's scheduler takes the step index from the loop rather
  than counting its own steps, making bypass safe there.
- **Make bypass safe by advancing the scheduler counter ourselves.** That needs
  each scheduler's internals, which is exactly what `step_schedule_mutate`
  already contains in one place. *Revisit if* a technique needs to skip a step
  it can only identify once the loop is running.
- **Move step control into the Binding.** Nothing in the experiment was model
  specific. *Revisit if* a model's pipeline replaces the shared denoising loop.
