# Anima — Next Task

**Status:** Idle — pick from pending work below

**Build:** `cargo build --release && cargo test`

## Pending Work Queue

1. **`anima tasks` CLI commands** — List/add/complete task management
2. **Fix: `resume` not forwarding queued messages** — Messages sent while agent paused don't deliver on resume
3. **BUG: claude_code fails silently if workdir doesn't exist** — Should error gracefully
4. **Empty tool results UX** — Show "(no output)" or exit code instead of blank

## Recently Completed

- v3.7: Native `remember` tool with daemon persistence
- v3.6.5: Per-agent max_iterations config  
- v3.6.4: restart only affects running agents with glob patterns
- v3.6.3: Context usage tracking "X% full" display
- TUI: display assistant narration alongside tool calls
- Clippy warnings cleanup
- Stop: prompt before SIGKILL on timeout

## Notes

Pick the highest priority item, or ask Chris which to tackle next.
