# Current Work: Anima v3.6.10

**Date:** 2026-02-05
**Status:** Idle ✅

## Recent Releases

- **v3.6.10** — Add newlines between messages in streaming output
- **v3.6.9** — Fix recall message ordering in conversation history
- **v3.6.8** — Fix streaming mode tool_calls persistence for KV caching
- **v3.6.1** — SafeShellTool for native tool mode
- **v3.6.0** — Rename persona.md to system.md (BREAKING)
- **v3.5.9** — Unified logging via AgentLoggerObserver
- **v3.5.8** — Route daemon stdout/stderr to agent.log
- **v3.5.7** — Native tools pass ALL allowed tools

## v3.6.8 Details

**Problem:** Streaming mode (`anima ask`) was losing tool_calls and tool_trace data, 
breaking conversation persistence and preventing KV cache hits on subsequent turns.

**Root cause:** `think_streaming_with_options` returned `String` instead of `ThinkResult`.
The daemon's streaming branch then set `tool_calls = None, tool_trace = Vec::new()`.

**Fix:** 
- Changed streaming functions to return `ThinkResult` (matching non-streaming)
- Added tracking for `tool_names_used`, `last_tool_calls`, `tool_trace`, and token usage
- Updated daemon to extract tool data from streaming `ThinkResult`

Files changed:
- `src/agent.rs` — `think_streaming*` now returns `ThinkResult`
- `src/daemon.rs` — Updated both native tool mode and JSON-block mode streaming paths

## Pending Work (Backlog)

- [ ] Rename `always.md` to something clearer (preamble.md? context.md?)
- [ ] `find_tool` command — dynamically expand tools array between turns
- [ ] **Semantic agent injection** — Inject agents based on semantic relevancy
- [ ] `anima tasks` CLI commands
- [ ] Fix: `resume` not forwarding queued messages
- [ ] BUG: claude_code fails silently if workdir doesn't exist
- [ ] Empty tool results should show "(no output)" or exit code
