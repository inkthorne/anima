# Current Work: Opt-in Injection Behavior

**Date:** 2026-02-04
**Status:** Complete ✅

## Summary

Change injection behavior so that creating an always.md means opting into manual control.

## New Behavior

| Scenario | Behavior |
|----------|----------|
| No always.md | Inject tools + memories (sensible defaults) |
| always.md **with** directives | Expand in-place at directive positions |
| always.md **without** directives | Use as-is, no injection (user opted out) |

## Rationale

Creating an always.md signals that the user wants explicit control over what gets injected. If they want tools/memories, they add the directives. If they don't add them, they're intentionally leaving them out.

## Changes

- Modified `build_effective_always()` in daemon.rs
- Updated tests to reflect new behavior
- Added test for "no base always" case

---

## Pending Work (Backlog)

- [ ] **Semantic agent injection** — Inject agents with brief descriptions based on semantic relevancy
- [ ] `anima tasks` CLI commands
- [ ] Fix: `resume` not forwarding queued messages
- [ ] BUG: claude_code fails silently if workdir doesn't exist
- [ ] Empty tool results should show "(no output)" or exit code
