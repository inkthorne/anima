# Task: Refactor daemon.rs to serialize agent work through an mpsc channel

## Problem

Race conditions between clear_history() and think() in Message/Notify/Heartbeat handlers. See commit be30d93 for context - that fix is incomplete because there is still a gap between clearing and thinking where another task can interleave.

Scenarios:
1. `anima ask arya "..."` while heartbeat fires - heartbeat can sneak into the gap
2. Two conversations @mention arya simultaneously - cross-conversation context leakage
3. Chat TUI + heartbeat - same race

## Design

1. Create enum `AgentWork` with variants:
   - `Message { content: String, conv_name: Option<String>, response_tx: oneshot::Sender<...> }`
   - `Notify { conv_id: String, message_id: i64, depth: u32 }`
   - `Heartbeat`
   - Keep existing simple ones like Status, Clear, Shutdown as direct handlers - they don't need serialization

2. In `run_daemon()`:
   - Create `mpsc::channel<AgentWork>` (unbounded or bounded with backpressure)
   - Spawn single worker task that owns/holds the Agent
   - Worker pulls from channel, processes sequentially

3. Worker task:
   - Owns the Agent directly (no Arc<Mutex<Agent>> needed for the worker)
   - For each work item: clear_history, do the work, respond
   - No race possible since single consumer

4. Connection handlers (`handle_connection`):
   - For Message requests: send `AgentWork::Message` to channel, await response via oneshot
   - For Notify requests: send `AgentWork::Notify` to channel (fire-and-forget is fine since Notify already spawns)

5. Heartbeat loop:
   - Instead of try_lock + heartbeat_pending complexity, just send `AgentWork::Heartbeat` to channel
   - Remove `heartbeat_pending` AtomicBool - no longer needed

6. Response handling:
   - Message needs response back to caller (oneshot channel)
   - Notify can stay fire-and-forget (writes to DB, caller already acked)
   - Heartbeat is internal (writes to DB)

## Files to modify

- `src/daemon.rs` (main changes)

## Build & test

```bash
cargo build --release
cargo test
```

## Important

- Keep streaming working for Message handler (chunks sent via socket, final response via oneshot)
- The existing `handle_notify` and `run_heartbeat` logic can be refactored into the worker or called from it
- Remove the scattered `clear_history()` calls - worker handles it once per work item
- This is a significant refactor but the goal is correctness over minimizing diff size
