# Per-Agent Chat DB

**Status:** Planning
**Date:** 2026-03-04

## Motivation

The global `~/.anima/conversations.db` supports multi-agent routing (participants, notifications, @mentions, embeddings). For single-agent use cases — `anima ask`, simple daemon chats — this complexity is unnecessary overhead.

**Goal:** Single-agent chats use `AgentStore` (chats.db), multi-agent conversations use `ConversationStore` (conversations.db). Store selection is per-chat, not per-agent — the daemon always opens both stores and routes based on how the chat was created.

## Store Routing

Store selection is determined by how a chat is created:

```
anima chat create <chat> <agent>
  └─ creates chat in ~/.anima/agents/{agent}/chats.db (AgentStore)
  └─ single-agent: no participants, no @mentions, no notifications

anima chat create <chat>
  └─ creates conversation in ~/.anima/conversations.db (ConversationStore)
  └─ multi-agent: participants, @mentions, notifications, embeddings

anima ask <agent> "msg"
  └─ always uses AgentStore (single-agent by definition)
  └─ creates ephemeral chat in ~/.anima/agents/{agent}/chats.db
```

The daemon opens both stores at startup and dispatches to the correct one per-request based on which store owns the chat/conversation name.

## What `AgentStore` Omits

These features exist in the global `ConversationStore` but are **not present** in `AgentStore`:

- **`participants` table** — single agent per DB, no join/leave tracking
- **`pending_notifications` table** — no cross-agent notification queue
- **`message_embeddings` table** — semantic search deferred
- **`mentions` column** — no @mention routing
- **`expires_at` / TTL** — no auto-expiry; chats persist until deleted
- **`ConversationEvent` hooks** — no lifecycle events needed
- **`triggered_by` column** — only one agent writes to the DB
- **Catchup system** — pause/resume with notification replay removed
- **`cleanup_expired()`** — no TTL means no expiry cleanup

## Schema

### `chats`

```sql
CREATE TABLE chats (
    name       TEXT PRIMARY KEY,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
```

Lightweight. No pause state (removed with multi-agent routing). `updated_at` bumped on each new message.

### `messages`

```sql
CREATE TABLE messages (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    chat           TEXT    NOT NULL,
    role           TEXT    NOT NULL,  -- 'user' | 'assistant' | 'tool'
    content        TEXT,              -- extracted text (for display/search)
    FOREIGN KEY (chat) REFERENCES chats(name)
);

CREATE INDEX idx_messages_chat ON messages (chat, id);
```

#### Role values

| Role | Description |
|------|-------------|
| `user` | User message. `content` = user text. |
| `assistant` | Agent response. `content` = extracted text. |
| `tool` | Tool execution result. `content` = tool output. |


## `AgentStore` API

Store for single-agent chats. Lives in `src/agent_store.rs`. Does **not** replace `ConversationStore` — they coexist.

```rust
pub struct AgentStore {
    conn: Connection,
    agent_name: String,  // for context, not used as DB key
}
```

### Construction

```rust
impl AgentStore {
    /// Open or create the agent's chat DB.
    /// Path: ~/.anima/agents/{name}/chats.db
    pub fn open(agent_dir: &Path) -> Result<Self>;

    /// For testing with an in-memory or temp DB.
    pub fn open_path(path: &Path, agent_name: &str) -> Result<Self>;
}
```

No more `ConversationStore::init()` that hardcodes the global path. Each `AgentStore` is scoped to one agent directory.

### Chats

```rust
// Create / lookup
fn create_chat(&self, name: Option<&str>) -> Result<String>;
fn get_chat(&self, name: &str) -> Result<Option<Chat>>;
fn list_chats(&self) -> Result<Vec<Chat>>;
fn match_chats(&self, pattern: &str) -> Result<Vec<String>>;
fn delete_chat(&self, name: &str) -> Result<()>;
fn clear_chat(&self, name: &str) -> Result<usize>;
```

Dropped: `create_chat` no longer takes `participants`. `get_or_create_conversation` removed (was for multi-agent find-by-participants). `find_by_participants` removed.

### Message

```rust
pub struct AgentStoreMessage {
    pub id: i64,
    pub chat: String,
    pub role: String,    // "user" | "assistant" | "tool"
    pub content: String,
}
```

### Messages

```rust
// Write
fn add_user_message(&self, chat: &str, content: &str) -> Result<i64>;
fn add_assistant_message(&self, chat: &str, content: &str) -> Result<i64>;
fn add_tool_message(&self, chat: &str, content: &str) -> Result<i64>;

// Read
fn get_messages(&self, chat: &str, limit: Option<usize>) -> Result<Vec<AgentStoreMessage>>;
fn get_messages_from(&self, chat: &str, from_id: i64) -> Result<Vec<AgentStoreMessage>>;
fn get_message_count(&self, chat: &str) -> Result<i64>;
```

Dropped: `add_message` / `add_message_with_tool_calls` / `add_message_with_tokens` unified into typed `add_*_message` methods. `add_recall_message` removed — recall injection doesn't need persistence. `copy_messages` removed.


## Impact on daemon.rs

The daemon always opens both stores at startup:

```rust
// In run_daemon():
let agent_store = AgentStore::open(&agent_dir_path)?;
let conv_store = ConversationStore::init()?;
```

Per-request routing: the daemon checks which store owns the chat/conversation name and dispatches accordingly.

### Call mapping (when handling a local chat)

When a request targets a local chat (in `AgentStore`), daemon handlers use `AgentStore` instead of `ConversationStore`:

| ConversationStore call | AgentStore equivalent |
|---|---|
| `ConversationStore::init()` (every handler) | `self.agent_store` (opened once at daemon start) |
| `add_message(conv, "user", ...)` | `store.add_user_message(chat, ...)` |
| `add_message_with_tokens(conv, agent, ...)` | `store.add_assistant_message(chat, ...)` |
| `add_tool_result(conv, content, triggered_by)` | `store.add_tool_message(chat, content)` |
| `add_native_tool_result(conv, tool_call_id, ...)` | `store.add_tool_message(chat, content)` |
| `add_recall_message(conv, content, agent)` | Skipped — recall not persisted in local chats |
| `get_messages(conv, limit)` | `store.get_messages(chat, limit)` |
| `get_messages_from_with_pinned(conv, id)` | `store.get_messages_from(chat, id)` |
| `store_message_embedding(...)` | Skipped — no embeddings in local chats |
| `add_participant(conv, agent)` | Skipped — no participants in local chats |
| `get/set_participant_state(conv, agent, ...)` | Skipped — no state table in local chats |
| `get/set_steps_in_state(conv, agent, ...)` | Skipped — no state table in local chats |
| `clear_all_cursors_for_agent(agent)` | Skipped — no state table in local chats |

### What local chats skip

For local chats, the daemon does **not**:
- Handle `Request::Notify` (no cross-agent notifications)
- Call `forward_notify_to_agent` (no @mention forwarding)
- Process pending notifications on startup
- Store message embeddings
- Track participants

These features remain available for multi-agent conversations routed to `ConversationStore`.

## Impact on main.rs

### `anima ask <agent> "msg"`

Always uses `AgentStore` — single-agent by definition, no flag needed. If the daemon is not running, auto-starts it (daemon opens both stores regardless). The ask handler creates a chat in the agent's `chats.db` and sends the message.

### `anima chat create <chat> <agent>`

New form: creates a local chat in the agent's `chats.db` via `AgentStore`. The agent name determines which agent directory holds the chat.

### `anima chat create <chat>`

Existing behavior: creates a conversation in the global `ConversationStore` (`conversations.db`). Used for multi-agent conversations.

### `anima chat` — unified listing

`anima chat` (no subcommand) merges local and global chats into one table sorted by `updated_at`.

**Algorithm:**

1. Query `ConversationStore::list_conversations()` — returns global conversations as today
2. Enumerate agent directories via `list_saved_agents()` (from `discovery.rs`)
3. For each agent with a `chats.db`, open `AgentStore` and call `list_chats()`
4. Merge all entries, sort by `updated_at` descending

**Display format:**

Same columns as today: NAME, MSGS, UPDATED, PARTICIPANTS. Local chats show the agent name in the PARTICIPANTS column with a `(local)` suffix.

```
NAME                           MSGS   UPDATED    PARTICIPANTS
────────────────────────────────────────────────────────────────
my-project                      124    2h ago     claude, coder
debug-session                     8    3h ago     arya          (local)
quick-question                    2    1d ago     arya          (local)
```

### `anima chat` — other subcommands

Other `anima chat` commands (`view`, `join`, `send`, etc.) check both stores to find the named chat/conversation and dispatch accordingly. The unified listing described above ensures that local chats are discoverable — users see all chats regardless of store, then subcommands route to the correct store by name lookup.

### `anima task` — unchanged

`anima task` runs standalone (no daemon). Could optionally support `AgentStore` in a future iteration, but not required for v1.

## History Reconstruction (`get_chat_history`)

`get_chat_history` converts `AgentStore` messages into `Vec<ChatMessage>` for the LLM. It is the local-chat equivalent of `format_conversation_history`, but much simpler — no dedup, no recall merging, no notes stripping, no multi-speaker batching.

**Signature:** `get_chat_history(messages: &[AgentStoreMessage]) -> (Vec<ChatMessage>, String)`

A single linear pass over messages, mapping by `role`:

| `role` | Emitted as | Details |
|---|---|---|
| `user` | `role: "user"` | Content passed through as-is. |
| `assistant` | `role: "assistant"` | Thinking tags stripped. Tool calls deserialized from JSON (synthetic `tb-` IDs filtered out). |
| `tool` (native, real ID) | `role: "tool"` | `tool_call_id` preserved for API compatibility. |
| `tool` (legacy/tool-block, `tb-` ID or none) | `role: "user"` | Content passed through as-is. |

**Return value:** `(history, final_user_content)` — same contract as `format_conversation_history`. If the last message is `user`, it's extracted as `final_user_content` and excluded from `history`. If the last message is `assistant`, `final_user_content` is empty.

**What it does NOT do** (compared to `format_conversation_history`):
- No `triggered_by` filtering — single agent, all messages belong to this agent
- No recall injection — recall is not persisted in local chats
- No dedup pre-scan — no tool call/result pair pruning
- No notes stripping — notes not supported in local chats
- No multi-speaker batching — no JSON-escaping or `<conversation>` wrapping

## Coexistence

The two stores are completely independent:

| | Global (`ConversationStore`) | Local (`AgentStore`) |
|---|---|---|
| **Path** | `~/.anima/conversations.db` | `~/.anima/agents/{name}/chats.db` |
| **Routed by** | `anima chat create <chat>` (no agent) | `anima chat create <chat> <agent>`, `anima ask` |
| **Multi-agent** | Yes (participants, @mentions, notifications) | No |
| **Schema** | 5 tables (conversations, participants, messages, pending_notifications, message_embeddings) | 2 tables (chats, messages) |
| **CLI access** | `anima chat *` commands | `anima chat *` commands (both stores searched) |

No migration needed. No data shared between the two. The daemon always has access to both stores — routing is per-chat, not per-agent.

## Test Strategy

Tests for `AgentStore` are self-contained in `src/agent_store.rs`:

- Schema creation (open creates tables)
- CRUD for chats and messages
- `clear_chat` deletes messages
- Role-specific message insertion (`add_user_message`, `add_assistant_message`, `add_tool_message`)
- Store routing dispatches correctly based on chat origin

Existing `conversation.rs` tests remain untouched — they validate the global store path.
