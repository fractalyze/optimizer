# ADR 0004: Build one agreed piece at a time; test interfaces

Status: accepted · 2026-10-01

## In short

Components land one at a time, each after its interface is agreed. Tests
protect the contracts other code relies on, not internals.

## Context

Every component should be understood by its reviewers as it lands, rather
than reviewed as a large finished framework.

## Decision

- **Order of work:** one component at a time, e.g. config schema, then
  engine adapter, then measurement, then quality gate. Each is agreed in
  discussion before it is written.
- **What gets tested:** the contracts other code depends on:
  - the engine adapter interface;
  - the technique and implementation contract;
  - composition and conflict rules;
  - quality-gate verdicts.
- **What doesn't:** internals, tested for their own sake.

## Consequences

- Fewer tests, so each one must guard a boundary that would otherwise break
  silently.
- Design notes go into `docs/` as decisions are made, not afterwards.
