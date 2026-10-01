# ADR 0004: Build incrementally; test interfaces, not internals

Status: accepted · 2026-10-01

## Context

Every component should be understood by its reviewers as it lands, not review a large
finished framework.

## Decision

- Components are built one at a time, in this order: profile schema, engine
  adapter, measurement, gate, and so on. Each one is agreed in discussion
  before it is implemented.
- Tests cover the **contracts that other code relies on**: the engine adapter
  interface, the technique/implementation interface, the composition and
  conflict rules, and the gate verdicts. Internals are not unit-tested for
  their own sake.

## Consequences

- Fewer tests, so each one has to protect a boundary that would otherwise
  break silently.
- Design notes go in `docs/` as decisions are made, not after.
