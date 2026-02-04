# Current Work: Templated Always.md Injection

**Date:** 2026-02-04
**Status:** Complete ✅

## Summary

Add support for injection directives in `always.md` so users can control what gets injected and where, instead of fixed append behavior.

## Directives

```markdown
<!-- @inject:tools -->     <!-- Expand to relevant tools -->
<!-- @inject:memories -->  <!-- Expand to semantic memories -->
<!-- @inject:agents -->    <!-- Expand to relevant agent descriptions (future) -->
```

## Behavior

1. **If directives present**: Replace each directive with its content at that location
2. **If no directives**: Fall back to current behavior (append tools, then memories)
3. **If directive present but empty result**: Remove the directive line (no empty sections)

## Implementation Plan

### Phase 1: Parser
- [x] Add function to detect `<!-- @inject:TYPE -->` directives in always.md content
- [x] `expand_inject_directives()` function added (lines 140-195)

### Phase 2: Injection Logic  
- [x] Modified `build_effective_always()` in daemon.rs (lines 197-235)
- [x] If directives found: expand in-place
- [x] If no directives found: fallback to append (backward compatible)

### Phase 3: Testing
- [x] 11 new tests added covering all cases
- [x] All tests passing

## Files Likely Involved

- `src/daemon.rs` — system prompt building, notify handling
- `src/agent.rs` — if prompt assembly happens here
- `src/prompt.rs` — if there's dedicated prompt logic
- `src/memory.rs` — memory injection formatting

## Notes

- Keep existing `<!-- @include:file.md -->` pattern as reference
- Directive syntax should be HTML-comment style (invisible in rendered markdown)
- Consider: should we support `<!-- @inject:tools:5 -->` for limit? (future)

---

## Pending Work (Backlog)

- [ ] **Semantic agent injection** — Inject agents with brief descriptions based on semantic relevancy
- [ ] `anima tasks` CLI commands
- [ ] Fix: `resume` not forwarding queued messages
- [ ] BUG: claude_code fails silently if workdir doesn't exist
- [ ] Empty tool results should show "(no output)" or exit code
