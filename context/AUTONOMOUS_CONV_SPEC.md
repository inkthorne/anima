# Autonomous Conversations Spec

## Problem

Currently, the CLI is required to drive @mention chains. When you exit the CLI, the chain stops. This limits agent collaboration to interactive sessions.

**Goal:** Enable daemon-to-daemon communication where agents can continue conversations autonomously, with user able to monitor and intervene.

---

## Decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | @mention forwarding | **Daemon-side** | Daemons forward @mentions themselves, CLI not required |
| 2 | Default state | **Active** | Conversations run autonomously by default |
| 3 | Control mechanism | **Pause/Resume** | Simple two-state model |
| 4 | Paused behavior | **Queue** | Notifications queued, processed on resume |
| 5 | Depth limit | **100 per chain** | Safety cap, rely on agent etiquette |

---

## Commands

```bash
anima chat                           # List all chats
anima chat new [name]                # Create + enter interactive
anima chat join <name>               # Join interactive (existing)
anima chat send <conv> "msg"         # Fire-and-forget message injection
anima chat view <conv>               # Dump all messages to stdout
anima chat view <conv> --limit 20    # Last N messages
anima chat view <conv> --since <id>  # Messages after ID (caller tracks)
anima chat pause <conv>              # Pause: queue notifications
anima chat resume <conv>             # Resume: process queue, go active
anima chat delete <conv>             # Delete conversation
anima chat cleanup                   # Remove expired messages + empty convs
```

Note: `view` is stateless — caller tracks last seen ID and passes `--since`. Clean, pipeable, automatable.

---

## Architecture

### Current (CLI-driven)
```
User types → CLI stores msg → CLI notifies agent → Agent responds →
CLI parses @mentions → CLI notifies next agent → ...
CLI exits → chain stops
```

### Proposed (Daemon-driven)
```
User: anima chat send <conv> "@arya @gendry solve X"
      → stores message → notifies agents → exits

Daemon loop:
  Daemon receives Notify → responds → stores response →
  Daemon parses own @mentions → Daemon notifies other daemons →
  Chain continues autonomously

User: anima chat view <conv> --since <last_id>
      → dumps new messages to stdout (stateless, caller tracks)
```

---

## Schema Changes

Add `paused` column to conversations table:

```sql
ALTER TABLE conversations ADD COLUMN paused INTEGER DEFAULT 0;
```

---

## Conversation States

```
                    ┌─────────┐
      (new conv) ──►│ Active  │◄─── (resume)
                    └────┬────┘
                         │ (pause)
                         ▼
                    ┌─────────┐
                    │ Paused  │───► (delete) ───► gone
                    └─────────┘
                         │ (resume)
                         ▼
                    ┌─────────┐
                    │ Active  │ (queue processed)
                    └─────────┘
```

- **Active**: Daemons forward @mentions autonomously
- **Paused**: Notifications queued in `pending_notifications`, not processed

---

## Implementation Phases

### Phase 1: Daemon-side @mention Forwarding ✅
- Move @mention parsing + notification from CLI into daemon
- When daemon responds, it checks for @mentions and notifies directly
- Daemon checks `paused` flag before forwarding
- CLI no longer required for chain continuation

### Phase 2: New Chat Commands ✅
- `anima chat send <conv> "msg"` — inject message + trigger notifications
- `anima chat pause <conv>` — set paused=1
- `anima chat resume <conv>` — set paused=0, process pending_notifications

### Phase 3: View Command ✅
- `anima chat view <conv>` — dump messages to stdout
- Options: `--limit N`, `--since <msg_id>`
- Stateless: caller tracks last seen ID
- Output format: parseable (for scripts/agents)

---

## Daemon Changes (daemon.rs)

### On Notify request:
```rust
1. Fetch conversation context
2. Generate response
3. Store response in conversation
4. Parse @mentions from own response
5. Check if conversation is paused
   - If paused: queue in pending_notifications
   - If active: notify mentioned agents (daemon-to-daemon)
6. Return acknowledgment
```

### Depth limit:
- Track depth in Notify request: `Request::Notify { conv_name, message_id, depth }`
- Increment depth when forwarding
- Stop at MAX_DEPTH (100)

---

## CLI Changes (main.rs)

### chat send:
```rust
1. Parse conv name and message from args
2. Store message as "user" in conversation
3. Parse @mentions from message
4. Notify mentioned agents (fire and forget)
5. Exit immediately (don't wait for responses)
```

### chat view:
```rust
1. Parse args: --limit N, --since <msg_id>
2. Query messages (filtered by limit/since)
3. Output to stdout (format: "id|timestamp|from|content" or JSON)
4. Exit immediately (stateless, no tracking)
```

### chat pause / resume:
```rust
1. Update conversations.paused = 1 or 0
2. On resume: check pending_notifications, process them
```

---

## Safety Considerations

1. **Depth limit (100)** — prevents infinite chains per user message
2. **@mention etiquette** — agents trained to use @mentions intentionally
3. **Pause command** — emergency brake
4. **Delete command** — nuclear option
5. **Watch command** — visibility into what's happening

---

## What Changes

| Component | Change |
|-----------|--------|
| `conversation.rs` | Add `paused` column, `set_paused()`, `is_paused()` |
| `daemon.rs` | Parse @mentions from own response, forward to other daemons |
| `socket_api.rs` | Add `depth` to `Request::Notify` |
| `main.rs` | Add `send`, `view`, `pause`, `resume` subcommands; rename `open` to `join` |

---

## Open Questions

1. Should `anima chat join` auto-pause? (Prevents competing with user input)
2. View output format — plain text? JSON? Configurable?
3. Should there be a global pause? (`anima chat pause --all`)
