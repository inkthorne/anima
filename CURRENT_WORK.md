# Current Work: Remove Global Always.md Fallback

**Date:** 2026-02-04
**Status:** Complete ✅

## Summary

Remove automatic fallback to global `~/.anima/agents/always.md`. Agents now explicitly include shared files via `{{include:path}}` directives.

## Rationale

- Global always.md is a place for **shared templates**, not a mandatory fallback
- Different agents may want different shared files (e.g., `always-tooluser.md` vs `always-nontooluser.md`)
- Explicit includes are clearer than implicit fallback behavior

## New Behavior

| Scenario | Behavior |
|----------|----------|
| Agent has always.md | Use it (expand includes) |
| Agent has no always.md | Return None (no global fallback) |
| Agent includes shared file | `{{include:../always.md}}` or absolute path |

## Changes

- `agent_dir.rs`: Removed global fallback logic in `load_always()`
- Updated tests to expect None when no agent always.md exists
- Added test for agent including shared global file
- Updated arya + gendry always.md to include global via `{{include:../always.md}}`
- Claude's always.md unchanged (no tools needed for delegate worker)

---

## Pending Work (Backlog)

- [ ] **Semantic agent injection** — Inject agents with brief descriptions based on semantic relevancy
- [ ] `anima tasks` CLI commands
- [ ] Fix: `resume` not forwarding queued messages
- [ ] BUG: claude_code fails silently if workdir doesn't exist
- [ ] Empty tool results should show "(no output)" or exit code
