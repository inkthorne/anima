# Multi-Agent Communication Spec

## Problem

Currently, agents can only talk to each other via the REPL:
- REPL manages single shared conversation
- @mention → REPL forwards full context up to cursor position
- Limitations: single thread, REPL required, context grows unbounded

**Goal:** Enable daemon-to-daemon communication without REPL, supporting group conversations (3+ agents).

---

## Key Design Questions

### 1. Conversation as entity
- Should conversations be first-class? (ID, participants, history)
- Can agents be in multiple conversations simultaneously?
- 1:1 vs group conversations — same mechanism or different?

### 2. Context delivery
- **Push** (current): Sender includes full context in message
- **Pull**: Recipient fetches from shared store
- Pull is cleaner — recipient controls their context window

### 3. @mention semantics
- Does @gendry mean "include gendry" or "message IS FOR gendry"?
- What context does the mentioned agent receive?
- Last N messages? Summary? Full history?

---

## Proposed Architecture: Conversation Store

```
┌─────────────────────────────────┐
│  ~/.anima/conversations.db      │
│  ┌─────────────────────────┐   │
│  │ conversations           │   │
│  │ - id, created, name     │   │
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
      [Arya][Gendry][CLI/User]
```

---

## Message Flow

1. **User sends message:**
   ```bash
   anima chat --conv work-project "hey @arya @gendry thoughts on streaming?"
   ```

2. **Message stored:**
   ```json
   {
     "conv_id": "work-project",
     "from": "user",
     "content": "hey @arya @gendry thoughts on streaming?",
     "mentions": ["arya", "gendry"],
     "timestamp": 1706...
   }
   ```

3. **Mentioned agents notified** (via socket or polling)

4. **Agent wakes, fetches context:**
   - Query: last N messages from conv_id where I'm participant
   - Each agent controls their own context window
   - Build prompt with conversation history

5. **Agent responds:**
   - Response stored in same conversation
   - Other participants can see it
   - If @mentions someone, they get notified

---

## CLI Commands

```bash
# Conversations
anima conv new "project-x" --with arya,gendry   # Create group conv
anima conv list                                  # List my conversations
anima conv join <conv-id>                        # Join existing

# Messaging  
anima chat arya                     # 1:1 (implicit conv)
anima chat arya gendry              # Group (implicit conv)
anima chat --conv project-x         # Explicit conv

# Or keep `send` for one-shot
anima send arya "quick question"    # Creates ephemeral 1:1
```

---

## Key Decisions

| Decision | Recommendation | Why |
|----------|----------------|-----|
| Context delivery | **Pull** | Recipients control their window |
| Conversation storage | **SQLite** | Simple, queryable, persistent |
| Notification | **Socket ping** | Daemons already have sockets |
| Offline agents | **Queue in DB** | Fetch on next wake |
| Context window | **Per-agent config** | 32k model needs less than 128k |

---

## What Changes

| Component | Change |
|-----------|--------|
| `daemon.rs` | Check for pending messages on wake, subscribe to notifications |
| `socket_api.rs` | Add `Request::Notify { conv_id, from }` for pings |
| New: `conversation.rs` | Conversation store (SQLite), query helpers |
| `main.rs` | New `conv` subcommand, update `chat` for groups |
| Agent config | Add `conversation_context_messages: 20` |

---

## Open Questions

1. **Real-time vs polling?** Socket notification is cleaner but more complex
2. **Conversation cleanup?** Auto-archive after N days? Manual delete?
3. **Private vs shared memories?** Can agents share semantic memories in group convos?
4. **User as participant?** Is "user" just another agent, or special?
5. **Message ordering?** Timestamps enough? Vector clocks for true ordering?
6. **Concurrent responses?** If both @arya and @gendry are mentioned, who responds first? Both in parallel?

---

## Alternatives Considered

### A. Message Bus / Pub-Sub
- Central bus routes messages
- Agents subscribe to channels/topics
- Pro: Clean separation, scalable
- Con: Another moving part, single point of failure

### B. Direct Socket-to-Socket
- Agents discover each other via pid files/sockets
- Direct connections for messaging
- Pro: Simple, no central component
- Con: N² connections, context sync is hard

### C. Conversation Server (Recommended)
- Conversations stored centrally (SQLite)
- Agents fetch/push to conversation store
- Pro: Persistent, queryable, clean
- Con: Another component (but minimal)

---

## Implementation Phases

### Phase 1: Conversation Store
- Create `conversation.rs` with SQLite schema
- Basic CRUD: create conv, add message, fetch messages
- CLI: `anima conv new`, `anima conv list`

### Phase 2: CLI Integration
- Update `anima chat` to use conversation store
- Support `--conv` flag for explicit conversations
- Implicit 1:1 conversations for `anima chat <agent>`

### Phase 3: Daemon Notifications
- Add `Request::Notify` to socket API
- Daemons listen for notifications
- On notification, fetch context and respond

### Phase 4: Group Conversations
- Multiple participants
- @mention routing within groups
- Concurrent response handling

---

## Schema Draft

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
    FOREIGN KEY (conv_id) REFERENCES conversations(id)
);

CREATE INDEX idx_messages_conv ON messages(conv_id, created_at);
```
