# Anima Backlog

## Tool Call Limits (2026-02-02)

**Problem:** Tool output can flood context window. Arya made 8 tool calls, read multiple files, then hallucinated.

**Current state:**
- Hardcoded `max_tool_calls = 10` in daemon.rs
- No output size limit
- Not configurable

**Proposed improvements:**
1. Make `max_tool_calls` configurable per agent/model
2. Add `max_tool_output_bytes` — truncate results over threshold
3. Sensible defaults based on context window size

**Priority:** Medium — quality of life, prevents runaway tool loops
