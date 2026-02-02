# Multi-Agent Communication Spec

## Problem

Currently, agents can only talk to each other via the REPL:
- REPL manages single shared conversation
- @mention → REPL forwards full context up to cursor position
- Limitations: single thread, REPL required, context grows unbounded

**Goal:** Enable daemon-to-daemon communication without REPL, supporting group conversations (3+ agents).

---

## Decisions Made ✅

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Context delivery | **Pull** | Recipients control their context window; scales better for groups |
| 2 | Conversation model | **Hybrid** | First-class in data model, implicit UX for 1:1s, explicit for groups |
| 3 | @mention semantics | **Notification** | @mention pings specific agents; all participants see message in history |
| 4 | Concurrent responses | **Parallel** | Multiple mentioned agents notified simultaneously; responses arrive in any order |
| 5 | User as participant | **Equal** | "user" is just another participant, no special treatment; multi-human possible |
| 6 | Notification mechanism | **Hybrid** | Socket ping for running daemons; DB queue for stopped agents |
| 7 | Conversation storage | **SQLite** | `~/.anima/conversations.db` — simple, queryable, persistent |

---

## Still Open

| # | Question | Options |
|---|----------|---------|
| 1 | Shared memories | Can agents share semantic memories in group convos? |
| 2 | Message ordering | Timestamps sufficient? Or need vector clocks? |
| 3 | Context window config | Per-agent `conversation_context_messages` setting |

## Additional Decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 8 | Message TTL | **7 days default** | Messages expire; important info should be [REMEMBER]'d to semantic memory |
| 9 | Cleanup policy | **Full deletion** | When all messages expire, delete conversation entirely. Keep things clean. |

---

## Architecture: Conversation Store

```
┌─────────────────────────────────┐
│  ~/.anima/conversations.db      │
│  ┌─────────────────────────┐   │
│  │ conversations           │   │
│  │ - id, name, created_at  │   │
│  ├─────────────────────────┤   │
│  │ participants            │   │
│  │ - conv_id, agent        │   │
│  ├─────────────────────────┤   │
│  │ messages                │   │
│  │ - conv_id, from, text,  │   │
│  │   mentions[], timestamp │   │
│  └─────────────────────────┘   │
└─────────────────────────────────┘
         ▲    ▲    ▲
         │    │    │
      [Arya][Gendry][User]
```

---

## Message Flow

1. **User sends:** `anima chat --conv project-x "hey @arya @gendry thoughts?"`
2. **Stored in DB** with mentions extracted
3. **Notifications sent:** Socket ping to running agents, queued for stopped
4. **Agents fetch context:** Each pulls last N messages (per their config)
5. **Agents respond:** Responses stored, other participants can see

---

## CLI Commands

```bash
# Conversations
anima conv new "project-x" --with arya,gendry   # Create group
anima conv list                                  # List conversations
anima conv archive <id>                          # Archive (manual cleanup)

# Messaging  
anima chat arya                     # 1:1 (implicit conv, auto-created)
anima chat arya gendry              # Group (implicit)
anima chat --conv project-x         # Explicit conv

# One-shot
anima send arya "quick question"    # Ephemeral 1:1
```

---

## Schema

```sql
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    name TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE participants (
    conv_id TEXT NOT NULL,
    agent TEXT NOT NULL,
    joined_at INTEGER NOT NULL,
    PRIMARY KEY (conv_id, agent),
    FOREIGN KEY (conv_id) REFERENCES conversations(id)
);

CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id TEXT NOT NULL,
    from_agent TEXT NOT NULL,
    content TEXT NOT NULL,
    mentions TEXT,  -- JSON array: ["arya", "gendry"]
    created_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,  -- TTL: default 7 days from created_at
    FOREIGN KEY (conv_id) REFERENCES conversations(id)
);

-- For pending notifications (offline agents)
CREATE TABLE pending_notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent TEXT NOT NULL,
    conv_id TEXT NOT NULL,
    message_id INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (conv_id) REFERENCES conversations(id),
    FOREIGN KEY (message_id) REFERENCES messages(id)
);

CREATE INDEX idx_messages_conv ON messages(conv_id, created_at);
CREATE INDEX idx_pending_agent ON pending_notifications(agent);
```

---

## Implementation Phases

### Phase 1: Conversation Store
- New `conversation.rs` module
- SQLite schema + CRUD operations
- CLI: `anima conv new`, `anima conv list`

### Phase 2: CLI Integration  
- Update `anima chat` to use store
- Implicit 1:1 conversations
- `--conv` flag for explicit

### Phase 3: Daemon Notifications
- `Request::Notify { conv_id, from }` in socket API
- Daemon checks pending on wake
- Real-time ping for running agents

### Phase 4: Group Conversations
- Multiple participants
- @mention routing
- Parallel response handling

---

## What Changes

| Component | Change |
|-----------|--------|
| New: `conversation.rs` | Conversation store (SQLite), query helpers |
| `socket_api.rs` | Add `Request::Notify` |
| `daemon.rs` | Check pending on wake, handle notifications |
| `main.rs` | New `conv` subcommand, update `chat` |
| Agent config | Add `conversation_context_messages` |
