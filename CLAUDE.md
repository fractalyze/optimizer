# CLAUDE.md

Read README.md once at the start of a session — it carries the project's key
info and indexes docs/. When you need background (design, decisions), follow
README's index and read the specific doc; don't guess and don't grep blind.

Claude Code-specific rules for this repo:

## Working agreement
- Build one component at a time, and only after its interface has been agreed
  with the user (ADR 0004). Do not scaffold ahead.
- Tests cover interfaces and contracts only, not internals.
- A decision made in chat goes into `docs/` (a new ADR or an amendment) in the
  same change. Never silently rewrite an accepted ADR; supersede it.
- This repo is public: no absolute paths, hostnames, or machine-specific
  setup (GPU indices, locks, local checkouts) in committed files.

## Reference sources (read-only)
- SGLang: the upstream commit pinned in `docs/adr/0003-sglang-first-engine.md`.
  Cite file:line at that commit; moving the pin means amending that ADR.
- Sol-Engine: `NVlabs/Sana`, branch `sol-engine`, @ `670482d`.
- vLLM-Omni: `68003cf6a` (vLLM 0.30.0), the commit experiments 004 and 005
  traced; moving it means amending ADR 0015.

## Git
- Work lands through PRs; the user reviews on GitHub.
- Commits follow `/workflow:commit`: `type(scope): summary` with a scope
  (e.g. `optimizer`, `adr`), a body that says why, and no `Co-Authored-By`
  trailer. Earlier history is unscoped; do not copy it.
- Stacked PRs: never delete a base branch while a PR targets it; retarget the
  child to `main`, rebase it onto `main`, and push with `--force-with-lease`.
