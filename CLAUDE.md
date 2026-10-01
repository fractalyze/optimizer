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
- SGLang: upstream `sgl-project/sglang` @ `8ca82118e`. Cite file:line at that
  commit. Re-pin explicitly; do not drift to a fork.
- Sol-Engine: `NVlabs/Sana`, branch `sol-engine`, @ `670482d`.

## Git
- Work lands through PRs; the user reviews on GitHub.
