# Conversation Database

The conversation database (`~/.anima/conversations.db`) is the persistent store for all multi-agent communication in Anima. It uses SQLite via `rusqlite` and is managed by `ConversationStore` in `src/conversation.rs`.

## Schema

### `conversations`

Conversation metadata. Names are auto-generated in `adjective-noun` format (e.g. `wild-screwdriver`) or user-specified.

```sql
CREATE TABLE conversations (
    name            TEXT PRIMARY KEY,
    created_at      INTEGER NOT NULL,       -- Unix timestamp
    updated_at      INTEGER NOT NULL,       -- Updated on every new message
    paused_at_msg_id INTEGER DEFAULT NULL   -- Non-NULL = paused. Max message ID when paused (0 for empty conversations)
);
```

### `participants`

Agents in each conversation. Each agent tracks its own cursor positions, working notes, and state machine state.

```sql
CREATE TABLE participants (
    conv_name       TEXT NOT NULL,
    agent           TEXT NOT NULL,
    joined_at       INTEGER NOT NULL,       -- Unix timestamp
    context_cursor  INTEGER DEFAULT NULL,   -- Last message ID sent to LLM
    dedup_cursor    INTEGER DEFAULT NULL,   -- Messages <= this ID eligible for dedup
    notes           TEXT DEFAULT NULL,       -- Agent scratchpad (DaemonNotesTool)
    state           TEXT DEFAULT NULL,       -- Current state file name (state machine)
    state_vars      TEXT DEFAULT NULL,       -- Persisted state variables (JSON)
    PRIMARY KEY (conv_name, agent),
    FOREIGN KEY (conv_name) REFERENCES conversations(name)
);
```

| Column | Purpose |
|--------|---------|
| `context_cursor` | Tracks the last message ID included in the LLM context. Enables incremental context building — only messages after the cursor are new. NULL triggers a cold start. Cleared on daemon startup to prevent stale cursors. |
| `dedup_cursor` | Messages with `id <= dedup_cursor` are eligible for deduplication (stripping redundant tool results, notes). |
| `notes` | Free-form scratchpad text managed via the `notes` tool. Stripped from older messages during dedup. |
| `state` | Current state file name for agents using the state machine system. |
| `state_vars` | JSON blob of persisted variables for the state machine. |

### `messages`

All messages in all conversations — user input, agent responses, tool results, and recall injections.

```sql
CREATE TABLE messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_name       TEXT NOT NULL,
    from_agent      TEXT NOT NULL,           -- See "Message Roles" below
    content         TEXT NOT NULL,
    mentions        TEXT,                    -- JSON array of @mentioned agent names
    created_at      INTEGER NOT NULL,        -- Unix timestamp
    expires_at      INTEGER NOT NULL,        -- Unix timestamp (created_at + TTL)
    duration_ms     INTEGER DEFAULT NULL,    -- Response time in ms (agent messages)
    tool_calls      TEXT DEFAULT NULL,       -- JSON array of native tool calls
    tokens_in       INTEGER DEFAULT NULL,    -- Input tokens for this response
    tokens_out      INTEGER DEFAULT NULL,    -- Output tokens for this response
    num_ctx         INTEGER DEFAULT NULL,    -- Context window size at generation time
    triggered_by    TEXT DEFAULT NULL,        -- Agent that triggered this message
    pinned          INTEGER DEFAULT 0,       -- Boolean: 1 = pinned in context
    prompt_tokens_est INTEGER DEFAULT NULL,  -- (migration artifact, unused)
    prompt_eval_ns  INTEGER DEFAULT NULL,    -- Prompt eval duration in ns (Ollama)
    tool_call_id    TEXT DEFAULT NULL,        -- Links tool results to their tool calls
    cached_tokens   INTEGER DEFAULT NULL,    -- Prompt tokens from cache (OpenAI/LMStudio)
    FOREIGN KEY (conv_name) REFERENCES conversations(name)
);

CREATE INDEX idx_messages_conv ON messages(conv_name, created_at);
```

### `pending_notifications`

Queued @mention notifications for agents that were offline when mentioned. Delivered when the agent's daemon starts.

```sql
CREATE TABLE pending_notifications (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    agent           TEXT NOT NULL,
    conv_name       TEXT NOT NULL,
    message_id      INTEGER NOT NULL,        -- The message that triggered the notification
    created_at      INTEGER NOT NULL,
    FOREIGN KEY (conv_name) REFERENCES conversations(name),
    FOREIGN KEY (message_id) REFERENCES messages(id)
);

CREATE INDEX idx_pending_agent ON pending_notifications(agent);
```

### `message_embeddings`

Vector embeddings for semantic search over conversation history. Stored as binary blobs (f32 arrays).

```sql
CREATE TABLE message_embeddings (
    message_id      INTEGER PRIMARY KEY,
    conv_name       TEXT NOT NULL,
    embedding       BLOB NOT NULL,           -- f32 array serialized as bytes
    FOREIGN KEY (message_id) REFERENCES messages(id)
);

CREATE INDEX idx_msg_emb_conv ON message_embeddings(conv_name);
```

## Message Roles

The `from_agent` column encodes the message type:

| Value | Meaning |
|-------|---------|
| `user` | Human input (from `anima chat send`, interactive mode, or `anima ask`) |
| `<agent-name>` | Agent response (e.g. `arya`, `scout`) |
| `tool` | Tool execution result. `triggered_by` indicates which agent invoked the tool. |
| `recall` | Recall injection (recall.md contents, memory). `triggered_by` indicates the owning agent. |

When formatting history for the LLM, `tool` and `recall` messages are filtered by `triggered_by` — each agent only sees its own tool results and recall injections.

## Message Lifecycle (TTL)

Messages have a 7-day TTL by default (`DEFAULT_TTL_SECONDS = 604800`).

- `expires_at` is set to `created_at + DEFAULT_TTL_SECONDS` on insertion
- All queries filter with `expires_at > now` — expired messages are invisible
- `cleanup_expired()` physically deletes expired messages and their embeddings
- Conversations with no remaining messages are also deleted during cleanup
- Orphaned participants and pending notifications are cleaned up in the same pass

Run cleanup via: `anima chat cleanup`

## Pinning

Pinned messages (`pinned = 1`) are always included in the LLM context, regardless of the history window or token budget.

**Cascade behavior:** Pinning/unpinning cascades between tool calls and their results:
- Pinning an assistant message with `tool_calls` also pins all matching tool result messages (matched via `tool_call_id`)
- Pinning a tool result message also pins the parent assistant message (matched via `tool_calls` JSON containing the `tool_call_id`)

**Context assembly:** Pinned messages outside the normal window are prepended in chronological order before the recent messages.

CLI: `anima chat pin <conv> <id>` / `anima chat unpin <conv> <id>`

## Pause, Resume & Catchup

Conversations can be paused to queue messages without triggering agent processing.

**Pause** (`anima chat pause <conv>`):
1. Sets `paused_at_msg_id` to the current max message ID (0 if no messages yet)
2. While paused, new messages are stored but @mention notifications are queued (not delivered)

**Stop** (`anima chat stop <conv>`):
- Drops all queued pending notifications for the conversation without processing them

**Resume** (`anima chat resume <conv>`):
1. Retrieves `paused_at_msg_id` and clears pause state
2. Fetches all messages with `id > paused_at_msg_id` (messages added during pause)
3. Builds catchup items from those messages:
   - Agent messages are checked for tool calls (need execution) and @mentions (need forwarding)
   - User messages are checked for @mentions only
   - Tool result messages are skipped (they'll be re-created when tools execute)

## Notifications

When a message contains @mentions, Anima notifies the mentioned agents:

1. **Parse mentions** — `parse_mentions()` extracts `@agent-name` patterns from message content, ignoring mentions inside backticks or code blocks. `@all` expands to all participants (except `user`).
2. **Running agents** — Notification sent via Unix socket (`Request::Notify`). The agent can respond synchronously (`Notified`) or acknowledge for async processing (`Acknowledged`).
3. **Offline agents** — Notification queued in `pending_notifications` table. Delivered when the agent's daemon starts and calls `get_pending_notifications()`.

## Context Windowing

Each agent in a conversation maintains two cursors in the `participants` table:

**`context_cursor`** — The message ID up to which the agent has already sent context to the LLM. On the next turn, only messages after this cursor need to be fetched as new context. When NULL, a cold start is triggered (full history loaded within window limits).

- Cleared on daemon startup (`clear_all_cursors_for_agent()`) to prevent stale cursors
- Cleared on `clear_messages()` since all messages are gone

**`dedup_cursor`** — Messages with `id <= dedup_cursor` are eligible for deduplication. The dedup system strips redundant tool call/result pairs from older context:

- **Read/peek dedup**: If the same file was read multiple times, only the latest read is kept
- **Write dedup**: Superseded by later writes to the same file
- **Edit dedup**: Superseded by later edits to the same file
- **Shell dedup**: Duplicate commands (normalized — strips `2>&1` and trailing pipe filters) keep only the latest
- **Notes dedup**: Only the most recent notes tool call is kept
- **Unknown tools**: All but the latest call are dropped

Tool call/result pairs are linked via `tool_call_id` for precise matching.

**Context retrieval methods:**
- `get_messages(limit)` — Last N messages (expired filtered)
- `get_messages_with_pinned(limit)` — Last N messages merged with pinned messages
- `get_messages_by_token_budget(budget)` — Walk backward until estimated token budget exceeded, merge with pinned
- `get_messages_from_with_pinned(cursor)` — Messages from cursor ID onward, merged with pinned

## Tool Call Tracking

Tool calls are tracked differently depending on the tools mode:

**Native mode** (`tools = true` in model config):
- The LLM returns structured `ToolCall` objects with unique IDs
- `tool_calls` column stores the JSON array: `[{"id": "...", "name": "...", "arguments": {...}}, ...]`
- Tool results reference back via `tool_call_id`
- `add_native_tool_result()` stores results with both `triggered_by` and `tool_call_id`

**Tool-block mode** (`tools = false`):
- The LLM outputs `<tool>{"tool": "name", "params": {...}}</tool>` in message content
- The daemon parses and executes these tool blocks
- Tool results are stored as separate `tool` messages with `triggered_by` set

Both modes use `triggered_by` to attribute tool results to the invoking agent, enabling per-agent filtering in `format_conversation_history()`.

## Semantic Search

Embeddings enable semantic similarity search over conversation history:

1. **Storage** — `store_message_embedding()` saves an f32 vector blob in `message_embeddings`
2. **Search** — `search_similar_messages()` computes cosine similarity against all embeddings in the conversation
3. **Filtering** — Excludes `tool` and `recall` messages, empty content, and messages that are pure tool-call intermediaries (`tool_calls IS NOT NULL` with no content). Messages in the current context window (`exclude_ids`) are also skipped.
4. **Results** — Returns `(message_id, from_agent, content, created_at, similarity)` sorted by similarity descending

Keyword search is also available via `search_messages()` using SQL `LIKE` patterns, with optional sender filtering and content truncation.
