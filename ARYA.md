# Task: v3.6.3 — Context Usage Tracking

**Goal:** Show users how full their context window is with an "X% full" indicator.

**Build:** `cargo build --release && cargo test`

## Feature Requirements

1. **Track context usage** — Calculate approximate token count of conversation history
2. **Display in chat TUI** — Show "Context: X% full" or similar in the status area
3. **Model awareness** — Use model's context limit (e.g., 200K for Claude, 128K for GPT-4)

## Implementation Ideas

### Token Counting
- Simple approximation: ~4 chars per token (good enough for display)
- Or use tiktoken-rs crate for accurate counts

### Where to Display
- Chat TUI status bar (bottom area with the duration)
- Show something like: `[Context: 23% | 2.3s]` after responses

### Model Context Limits
Could add to config or hardcode sensible defaults:
- claude-*: 200K
- gpt-4*: 128K  
- gemini*: 1M
- Default: 128K

## Checklist

- [ ] Add token counting utility
- [ ] Track cumulative tokens in conversation
- [ ] Get model context limit (config or defaults)
- [ ] Display percentage in chat TUI
- [ ] Test with different conversation lengths

## Notes

Keep it simple — approximate counting is fine for a UX indicator. Users just need a rough sense of "am I running out of context?"
