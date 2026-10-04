# ADR 0020: The composer derives coexistence and ordering from each implementation's own exclusive claims

Status: accepted · 2026-10-04 · builds the composer named in [ADR 0013](0013-feasibility-and-engagement.md); amends 0013's statuses with `CONFLICT`, and where [ADR 0009](0009-lifecycle-constraints.md)'s lifecycle constraints are consumed

## In short

Core's third stage takes several implementations that are each supported and
feasible on one target and answers `COMPOSABLE`, with a partial order, or
`CONFLICT`, with reasons. Each implementation declares, about itself alone,
which resources it controls exclusively (`owns`) and which resources' owners
must be in place before it (`after`). Two exclusive claims on one resource
conflict, an ordering cycle conflicts, and sharing a required capability
never does. The composer names no technique, implementation, engine or model,
and no combination is stored anywhere: it is built from atomic entries when
asked.

```text
Catalog / Registry   what exists
        ↓
Resolution           is one implementation supported?   UNSUPPORTED
        ↓
Feasibility          can it run in this RuntimeContext?   INFEASIBLE
        ↓
Composer             can these coexist, in what order?   CONFLICT
        ↓
[execution plan]     not built yet
```

## Context

The architecture's contract already had `owns`, motivated by conflicts found
in SGLang's code (investigation §8): FP8 and NVFP4 both want the linear
layers; TeaCache, Spectrum and cache-dit all want the trunk. Experiment 005
ran FP8 with experiment 002's trunk control on one model and both engaged.
What the composer needed was a way to tell those two situations apart from
metadata, and to stay linear in the size of the catalog: a pairwise matrix or
stored combinations would grow with every pair.

## Decision

### Claims are about the claimant, never about another implementation

`ImplementationSpec` gains two fields:

| Field | Means | In the catalog |
|---|---|---|
| `owns` | resources it controls exclusively | prediction reuse: `step_prediction`; TeaCache: `trunk`; native FP8: `linear_layers` |
| `after` | resources whose owners must be in place before it | none yet |

Resource ids are a closed vocabulary (`core/resources.py`), like capability
ids, because a misspelled resource would silently hide a conflict. A
resource is claimed, never read: reading goes through capabilities, which any
number of implementations share, so two implementations both requiring
`timestep_state` compose.

### The algorithm is three checks over the claims

1. **Exclusive claims:** a resource owned by more than one candidate is a
   conflict, reported with every claimant.
2. **Ordering:** for each `after` resource, every candidate owning it comes
   before the declarer; owners that are not in the configuration impose
   nothing.
3. **Cycles:** a topological sort; candidates it cannot place conflict.

Every reason is reported at once. The result does not depend on the order of
the input. A composable result carries the partial order only, never a
schedule or a sequence of actions.

### A conflict is its own status

ADR 0013 folded ownership conflicts into `INFEASIBLE`. They are now
`CONFLICT`: `INFEASIBLE` says one implementation cannot run in this runtime,
`CONFLICT` that implementations which can each run cannot run together. The
optimizer learns different things from the two.

### Inputs are proven, not re-checked

A `Candidate` carries its implementation, its `RESOLVED` resolution and its
`FEASIBLE` result, and refuses anything else, so the composer never sees an
unsupported or infeasible implementation and never calls an evaluator.
Candidates must share one target; duplicates are refused.

### Lifecycle constraints wait for execution planning

The contract's `constraints` (`mutates_model`, `dynamic_in_forward`,
`request_state`, ADR 0009) decide *when* and *how* an implementation is
installed, not *whether* two coexist: FP8 at load and TeaCache at runtime
composed in experiment 005 without either needing the other's phase. They
join `ImplementationSpec` with the execution plan, which needs them to
separate server configuration from per-request policy (ADR 0016).

## Consequences

- The catalog keeps growing by one entry per technique, implementation or
  provider; the composer evaluates any set of them.
- A future native FP4 implementation that owns `linear_layers` will conflict
  with native FP8 without a rule mentioning either.
- An ordering such as "fuse QKV before quantizing" is expressible as the
  quantizer declaring `after` the resource the fusion owns, once such an
  implementation and its resource exist.

## What this rests on

**Runtime validated:** FP8 W8A8 and trunk control engaged together on
Qwen-Image-2.1 in SGLang (experiment 005).

**Code reading only:** the exclusive-resource conflicts of investigation §8.

**Open:**
- **Prediction reuse with TeaCache.** The investigation noted they coexist
  only if a skipped step never reaches the trunk. Prediction reuse replaces
  the whole DiT call before the trunk is entered, so the condition holds by
  construction and no ordering is declared; the pair has not been run
  together.
- **Ordering with real implementations.** No current implementation declares
  `after`; ordering is tested with test-only implementations.

## Alternatives rejected

- **A compatibility matrix or stored combinations.** Quadratic in the
  catalog, and every new implementation would need entries against all
  others.
- **Conflict on any shared capability.** Reading the same state is how most
  techniques work; only control is exclusive.
- **`before` and `after` naming other implementations.** It couples entries
  to each other; a resource-based `after` keeps each entry about itself.
- **A shared/exclusive mode on each claim.** Shared use already has a home in
  `requires`; no evidence calls for shared control.
- **A constraint solver.** Two set checks and a topological sort cover every
  rule the evidence supports.
