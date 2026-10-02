# ADR 0014: In a batched model call, the EngineAdapter maps rows to owners; savings need agreement

Status: accepted · 2026-10-02 · resolves, for SGLang, the batching question left open in [ADR 0012](0012-trunk-capabilities.md)

## In short

[Experiment 003](../../experiments/003-batched-ownership/README.md) ran two
requests through one SGLang DiT call. Experiment 002's implementation and
Bindings worked unchanged once the EngineAdapter split each call into one
**logical execution owner per batch row**. So the capabilities stay as they
are, and three rules are added: the adapter owns the row-to-owner mapping; a
decision made by some owners of a call is honored by splicing and saves compute
only when every owner agrees; and engagement is judged per owner. Mapping rows
to owners needs an identity per row, which SGLang does not keep, so that is an
adapter feasibility condition.

## Context

ADR 0012 left open whether, once one model call serves several owners,
`trunk_output_override` can act on selected slices and `signal_observe` needs
ownership metadata. Experiment 003 tested it on Qwen-Image-2.1 and FLUX.2:

| Question | Answer in SGLang |
|---|---|
| Does one DiT call serve two requests? | Yes, as two rows on dim 0, under `batching_max_size=2` |
| Did the implementation or a Binding change? | No; both were imported from experiment 002 |
| Can one owner reuse while another computes? | Yes; the other owner's image was bitwise identical |
| Does that save compute? | No; the blocks still run for both rows |
| Does the signal need ownership metadata? | Not at the implementation: the adapter slices it per owner |
| Does the engine say which row is which request? | No; the merged request keeps only the first id; seeds survive, but may collide |

## Decision

### The EngineAdapter maps rows to owners

The owner of `request_local_state` (request × CFG branch in SGLang) may be one
row of a batched call. The adapter finds each row's owner, slices the trunk's
entry, signal and result by row, calls the implementation once per owner with
that owner's state, and reassembles the rows. The implementation still sees one
owner per call and never a layout. `signal_observe` keeps its plain-tensor shape;
ownership metadata stays below the contract.

The row axis is the engine's convention (dim 0 at both trunk edges in SGLang,
for both models), so it belongs to the EngineAdapter, not a Binding. A Binding
would answer it only for a model whose trunk edges use a different axis.

### A mixed decision is spliced; compute is saved only on agreement

For each call, `trunk_output_override` resolves as:

| Owners that supply a payload | The adapter | Compute saved |
|---|---|---|
| all | suppresses every block, rebuilds each row from its own payload | yes |
| none | runs the trunk | — |
| some | runs the trunk, replaces those rows at the exit | no |

Splicing is exact: in experiment 003 it gave the same image as a real skip.
Suppressing blocks for some rows only would need every block input sliced by
row, which is model-specific, so it is not offered.

### Engagement is per owner

The adapter reports counters per owner, not per call: decisions made, and how
each override was carried out (skipped or spliced). A trunk-reuse
implementation's speed claim engages only for overrides that skipped blocks,
so an owner whose reuses were all spliced is `FAILED_TO_ENGAGE` for speed
([ADR 0013](0013-feasibility-and-engagement.md)), even though its output did
change as decided.

### Row identity is a feasibility condition of the adapter

To provide `request_local_state` under batching, the adapter must attribute
every row to exactly one owner. Where it cannot, no row of that call may be
overridden, and an implementation needing `override_exact` there is
`INFEASIBLE`. SGLang drops per-request ids when it merges requests; the seed is
the only per-row identity left, and two merged requests may share one. An
SGLangAdapter therefore needs either SGLang to keep per-row request ids, or
must refuse to batch controlled requests whose seeds collide. Implementations
declare nothing new.

## Consequences

- The capability list and implementation contract do not change.
- Batching and reuse pull against each other: a batch saves compute through
  reuse only if its owners decide alike. Whether the optimizer should batch
  owners with aligned schedules is a later question.
- An EngineAdapter for a batching engine must report row ownership and per-owner
  counters.

## What this rests on

**Runtime validated** (experiment 003, SGLang @ `8ca82118e`, eager, two
requests, Qwen-Image-2.1 and FLUX.2-klein): one DiT call per step carried both
requests; per-owner observe and identity were exact; one owner's reuse left the
other bitwise identical; splice and skip gave identical images; blocks were
skipped only when both owners reused; no state leaked after control.

**Code reading only:** SGLang's merge keeps the first request's id and `extra`
and lists the seeds per row.

**Open:**
- **CFG branches stacked as rows** (ComfyUI runs cond and uncond in one call):
  untested; the probe refuses to override rows it cannot attribute.
- **vLLM-Omni request batching:** whether it keeps per-row request identity, and
  where per-request state can live.
- **The final name and shape of `request_local_state`:** request × branch,
  mapped onto rows, held in one engine. A second engine decides it.

## Alternatives rejected

- **Ownership metadata in `signal_observe`** (signal plus slice owners). The
  implementation would have to know the layout, which is what the capability
  layer exists to avoid. *Revisit if* an implementation must decide jointly
  across owners, e.g. to align reuse within a batch.
- **Per-row block suppression.** Needs model-specific slicing of every block
  input. *Revisit if* mixed decisions turn out to be common and costly.
- **Treat a spliced reuse as engaged.** It reports a speed decision that saved
  no compute.
- **Batch axis in the Binding.** Both models used the engine's convention.
