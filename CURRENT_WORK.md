# Current Work: Anima v3.6.1

**Date:** 2026-02-04
**Status:** Idle ✅

## Recent Releases

- **v3.6.1** — SafeShellTool for native tool mode
- **v3.6.0** — Rename persona.md to system.md (BREAKING)
- **v3.5.9** — Unified logging via AgentLoggerObserver
- **v3.5.8** — Route daemon stdout/stderr to agent.log
- **v3.5.7** — Native tools pass ALL allowed tools

## Pending Work (Backlog)

- [ ] Rename `always.md` to something clearer (preamble.md? context.md?)
- [ ] `find_tool` command — dynamically expand tools array between turns
- [ ] **Semantic agent injection** — Inject agents based on semantic relevancy
- [ ] `anima tasks` CLI commands
- [ ] Fix: `resume` not forwarding queued messages
- [ ] BUG: claude_code fails silently if workdir doesn't exist
- [ ] Empty tool results should show "(no output)" or exit code
