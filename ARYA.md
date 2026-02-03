# Task: Phase 3 — Auto-Invoke on Notification

**Goal:** When a daemon receives a Notify request, it should automatically invoke the agent to generate a response. Currently the notification system works, but agents don't auto-respond.

**Build:** `cargo build --release && cargo test`

## Context

From `AUTONOMOUS_CONV_SPEC.md`, the daemon should:
1. Receive Notify request with conv_name, message_id, depth
2. Fetch conversation context
3. Generate response (invoke the agent)
4. Store response in conversation
5. Parse @mentions from response
6. If not paused, notify mentioned agents (daemon-to-daemon)

## Current State

- ✅ `anima chat send <conv> "message"` stores message and sends Notify
- ✅ Notify requests are delivered to daemons
- ❓ Daemon receives Notify but may not auto-invoke agent

## Tasks

1. **Check current Notify handling in `src/daemon.rs`**
   - What happens when daemon receives a Notify request?
   - Does it invoke the agent? Store the response?

2. **Implement auto-invoke if missing:**
   - On Notify: fetch conversation messages for context
   - Call agent to generate response
   - Store response in conversation
   - Parse @mentions, forward notifications if not paused

3. **Test the flow:**
   ```bash
   # Start two agent daemons
   anima spawn gendry
   anima spawn codey
   
   # Create conversation and send message
   anima chat new test-conv
   anima chat send test-conv "@gendry what's your status?"
   
   # Gendry should auto-respond. Check:
   anima chat view test-conv
   ```

## Checklist

- [x] Review current Notify handler in daemon.rs
- [x] Implement agent auto-invoke on Notify (if missing) — already implemented!
- [x] Implement conversation context fetching
- [x] Implement response storage
- [x] Implement @mention forwarding from responses
- [x] Test with real agents (gendry, arya)
- [x] Update CURRENT_WORK.md when complete

## Phase 3 Complete ✅

Verified 2026-02-02: Auto-invoke was already fully implemented in `handle_notify()` at `src/daemon.rs:1087`.

**Test results:**
- Created `test-notify-1770088617` conversation
- Sent "@gendry what's your status?" — gendry auto-responded
- Sent "@gendry can you ask @arya for her status too?" — both responded, arya was forwarded via @mention chain
