//! Conversation Store for Multi-Agent Communication
//!
//! Provides persistent storage for conversations between agents using SQLite.
//! Located at `~/.anima/conversations.db`.

use rusqlite::{Connection, params};
use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur during conversation operations.
#[derive(Debug, Error)]
pub enum ConversationError {
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Conversation not found: {0}")]
    NotFound(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Pattern error: {0}")]
    Pattern(#[from] regex::Error),
}

/// A conversation between agents.
/// The `name` field is the unique identifier (PRIMARY KEY).
#[derive(Debug, Clone)]
pub struct Conversation {
    pub name: String,
    pub created_at: i64,
    pub updated_at: i64,
    pub paused: bool,
    /// The message ID at which the conversation was paused.
    /// Used during resume to catch up on skipped tool calls and @mentions.
    pub paused_at_msg_id: Option<i64>,
}

/// A participant in a conversation.
#[derive(Debug, Clone)]
pub struct Participant {
    pub conv_name: String,
    pub agent: String,
    pub joined_at: i64,
}

/// A message in a conversation.
#[derive(Debug, Clone)]
pub struct ConversationMessage {
    pub id: i64,
    pub conv_name: String,
    pub from_agent: String,
    pub content: String,
    pub mentions: Vec<String>,
    pub created_at: i64,
    pub expires_at: i64,
    /// Response duration in milliseconds (for agent responses)
    pub duration_ms: Option<i64>,
    /// Native tool calls JSON (for native tool mode persistence)
    pub tool_calls: Option<String>,
    /// Input tokens used for this response
    pub tokens_in: Option<i64>,
    /// Output tokens used for this response
    pub tokens_out: Option<i64>,
    /// Context window size at generation time
    pub num_ctx: Option<i64>,
    /// Which agent triggered this message (for tool results: the agent that invoked the tool)
    pub triggered_by: Option<String>,
    /// Whether this message is pinned (always included in LLM context regardless of history window)
    pub pinned: bool,
    /// Prompt eval duration in nanoseconds (Ollama-specific, drops when KV caching is active)
    pub prompt_eval_ns: Option<i64>,
}

/// A pending notification for an offline agent.
#[derive(Debug, Clone)]
pub struct PendingNotification {
    pub id: i64,
    pub agent: String,
    pub conv_name: String,
    pub message_id: i64,
    pub created_at: i64,
}

/// Represents an item that needs catchup processing after a conversation is resumed.
#[derive(Debug, Clone)]
pub struct CatchupItem {
    /// The message that needs catchup processing.
    pub message: ConversationMessage,
    /// Whether this message contains a tool call that needs execution.
    /// This is only relevant for agent messages (not "user" or "tool").
    pub has_tool_call: bool,
    /// The raw content that may contain a tool call block.
    /// The caller should use daemon::extract_tool_call() to parse this.
    pub raw_content: String,
    /// @mentions found in this message that need forwarding.
    pub mentions: Vec<String>,
}

/// Default message TTL: 7 days in seconds.
pub const DEFAULT_TTL_SECONDS: i64 = 7 * 24 * 60 * 60;

/// SQLite-backed conversation store.
pub struct ConversationStore {
    conn: Connection,
}

impl ConversationStore {
    /// Initialize the conversation store, creating the database if needed.
    /// Database is located at `~/.anima/conversations.db`.
    pub fn init() -> Result<Self, ConversationError> {
        let db_path = Self::db_path()?;

        // Ensure parent directory exists
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(&db_path)?;
        let store = Self { conn };
        store.create_schema()?;
        Ok(store)
    }

    /// Open the conversation store from a specific path (for testing).
    pub fn open(path: &std::path::Path) -> Result<Self, ConversationError> {
        let conn = Connection::open(path)?;
        let store = Self { conn };
        store.create_schema()?;
        Ok(store)
    }

    /// Get the default database path.
    fn db_path() -> Result<PathBuf, ConversationError> {
        let home = dirs::home_dir().ok_or_else(|| {
            ConversationError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Could not determine home directory",
            ))
        })?;
        Ok(home.join(".anima").join("conversations.db"))
    }

    /// Create the database schema, migrating from old schema if needed.
    fn create_schema(&self) -> Result<(), ConversationError> {
        // Check if we need to migrate from old schema (has 'id' column in conversations)
        let needs_migration = self.needs_migration()?;

        if needs_migration {
            self.migrate_from_old_schema()?;
        } else {
            // Create fresh schema
            self.conn.execute_batch(
                r#"
                CREATE TABLE IF NOT EXISTS conversations (
                    name TEXT PRIMARY KEY,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    paused INTEGER DEFAULT 0,
                    paused_at_msg_id INTEGER DEFAULT NULL
                );

                CREATE TABLE IF NOT EXISTS participants (
                    conv_name TEXT NOT NULL,
                    agent TEXT NOT NULL,
                    joined_at INTEGER NOT NULL,
                    PRIMARY KEY (conv_name, agent),
                    FOREIGN KEY (conv_name) REFERENCES conversations(name)
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conv_name TEXT NOT NULL,
                    from_agent TEXT NOT NULL,
                    content TEXT NOT NULL,
                    mentions TEXT,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    FOREIGN KEY (conv_name) REFERENCES conversations(name)
                );

                CREATE TABLE IF NOT EXISTS pending_notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent TEXT NOT NULL,
                    conv_name TEXT NOT NULL,
                    message_id INTEGER NOT NULL,
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY (conv_name) REFERENCES conversations(name),
                    FOREIGN KEY (message_id) REFERENCES messages(id)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conv_name, created_at);
                CREATE INDEX IF NOT EXISTS idx_pending_agent ON pending_notifications(agent);

                CREATE TABLE IF NOT EXISTS message_embeddings (
                    message_id INTEGER PRIMARY KEY,
                    conv_name TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    FOREIGN KEY (message_id) REFERENCES messages(id)
                );
                CREATE INDEX IF NOT EXISTS idx_msg_emb_conv ON message_embeddings(conv_name);
                "#,
            )?;
        }

        self.run_column_migrations()?;

        Ok(())
    }

    /// Check if a table has a specific column.
    fn has_column(&self, table: &str, column: &str) -> Result<bool, ConversationError> {
        let mut stmt = self.conn.prepare(&format!("PRAGMA table_info({})", table))?;
        let found = stmt
            .query_map([], |row| {
                let col_name: String = row.get(1)?;
                Ok(col_name)
            })?
            .any(|r| r.map(|n| n == column).unwrap_or(false));
        Ok(found)
    }

    /// Add a column to a table if it doesn't already exist.
    fn add_column_if_missing(
        &self,
        table: &str,
        column: &str,
        definition: &str,
    ) -> Result<(), ConversationError> {
        if !self.has_column(table, column)? {
            self.conn.execute(
                &format!("ALTER TABLE {} ADD COLUMN {} {}", table, column, definition),
                [],
            )?;
        }
        Ok(())
    }

    /// Run all column migrations for existing databases.
    fn run_column_migrations(&self) -> Result<(), ConversationError> {
        self.add_column_if_missing("conversations", "paused", "INTEGER DEFAULT 0")?;
        self.add_column_if_missing("conversations", "paused_at_msg_id", "INTEGER DEFAULT NULL")?;
        self.add_column_if_missing("messages", "duration_ms", "INTEGER DEFAULT NULL")?;
        self.add_column_if_missing("messages", "tool_calls", "TEXT DEFAULT NULL")?;
        self.add_column_if_missing("messages", "tokens_in", "INTEGER DEFAULT NULL")?;
        self.add_column_if_missing("messages", "tokens_out", "INTEGER DEFAULT NULL")?;
        self.add_column_if_missing("messages", "num_ctx", "INTEGER DEFAULT NULL")?;
        self.add_column_if_missing("messages", "triggered_by", "TEXT DEFAULT NULL")?;
        self.add_column_if_missing("messages", "pinned", "INTEGER DEFAULT 0")?;
        self.add_column_if_missing("messages", "prompt_tokens_est", "INTEGER DEFAULT NULL")?;
        self.add_column_if_missing("messages", "prompt_eval_ns", "INTEGER DEFAULT NULL")?;
        self.add_column_if_missing("participants", "context_cursor", "INTEGER DEFAULT NULL")?;

        // Create message_embeddings table if it doesn't exist (for existing databases)
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS message_embeddings (
                message_id INTEGER PRIMARY KEY,
                conv_name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                FOREIGN KEY (message_id) REFERENCES messages(id)
            );
            CREATE INDEX IF NOT EXISTS idx_msg_emb_conv ON message_embeddings(conv_name);",
        )?;

        Ok(())
    }

    /// Check if the database needs migration from the old schema.
    fn needs_migration(&self) -> Result<bool, ConversationError> {
        let mut stmt = self.conn.prepare(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'",
        )?;
        if !stmt.exists([])? {
            return Ok(false);
        }
        self.has_column("conversations", "id")
    }

    /// Migrate from old schema (id + name) to new schema (name as primary key).
    fn migrate_from_old_schema(&self) -> Result<(), ConversationError> {
        // Start transaction
        self.conn.execute("BEGIN TRANSACTION", [])?;

        // Create new tables with _new suffix
        self.conn.execute_batch(
            r#"
            CREATE TABLE conversations_new (
                name TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                paused INTEGER DEFAULT 0,
                paused_at_msg_id INTEGER DEFAULT NULL
            );

            CREATE TABLE participants_new (
                conv_name TEXT NOT NULL,
                agent TEXT NOT NULL,
                joined_at INTEGER NOT NULL,
                PRIMARY KEY (conv_name, agent),
                FOREIGN KEY (conv_name) REFERENCES conversations_new(name)
            );

            CREATE TABLE messages_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conv_name TEXT NOT NULL,
                from_agent TEXT NOT NULL,
                content TEXT NOT NULL,
                mentions TEXT,
                created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL,
                FOREIGN KEY (conv_name) REFERENCES conversations_new(name)
            );

            CREATE TABLE pending_notifications_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent TEXT NOT NULL,
                conv_name TEXT NOT NULL,
                message_id INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (conv_name) REFERENCES conversations_new(name),
                FOREIGN KEY (message_id) REFERENCES messages_new(id)
            );
            "#,
        )?;

        // Migrate data: use name if set, otherwise use id as name
        self.conn.execute(
            "INSERT INTO conversations_new (name, created_at, updated_at)
             SELECT COALESCE(name, id), created_at, updated_at FROM conversations",
            [],
        )?;

        // Migrate participants: map old conv_id to new conv_name
        self.conn.execute(
            "INSERT INTO participants_new (conv_name, agent, joined_at)
             SELECT COALESCE(c.name, c.id), p.agent, p.joined_at
             FROM participants p
             JOIN conversations c ON p.conv_id = c.id",
            [],
        )?;

        // Migrate messages
        self.conn.execute(
            "INSERT INTO messages_new (id, conv_name, from_agent, content, mentions, created_at, expires_at)
             SELECT m.id, COALESCE(c.name, c.id), m.from_agent, m.content, m.mentions, m.created_at, m.expires_at
             FROM messages m
             JOIN conversations c ON m.conv_id = c.id",
            [],
        )?;

        // Migrate pending notifications
        self.conn.execute(
            "INSERT INTO pending_notifications_new (id, agent, conv_name, message_id, created_at)
             SELECT pn.id, pn.agent, COALESCE(c.name, c.id), pn.message_id, pn.created_at
             FROM pending_notifications pn
             JOIN conversations c ON pn.conv_id = c.id",
            [],
        )?;

        // Drop old tables
        self.conn.execute_batch(
            r#"
            DROP TABLE IF EXISTS pending_notifications;
            DROP TABLE IF EXISTS messages;
            DROP TABLE IF EXISTS participants;
            DROP TABLE IF EXISTS conversations;
            "#,
        )?;

        // Rename new tables
        self.conn.execute_batch(
            r#"
            ALTER TABLE conversations_new RENAME TO conversations;
            ALTER TABLE participants_new RENAME TO participants;
            ALTER TABLE messages_new RENAME TO messages;
            ALTER TABLE pending_notifications_new RENAME TO pending_notifications;

            CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conv_name, created_at);
            CREATE INDEX IF NOT EXISTS idx_pending_agent ON pending_notifications(agent);
            "#,
        )?;

        // Commit transaction
        self.conn.execute("COMMIT", [])?;

        Ok(())
    }

    /// Create a new conversation with the given participants.
    /// If name is None, generates a fun name like "wild-screwdriver".
    /// Returns the conversation name.
    pub fn create_conversation(
        &self,
        name: Option<&str>,
        participants: &[&str],
    ) -> Result<String, ConversationError> {
        let conv_name = match name {
            Some(n) => n.to_string(),
            None => self.generate_unique_fun_name()?,
        };
        let now = current_timestamp();

        self.conn.execute(
            "INSERT INTO conversations (name, created_at, updated_at) VALUES (?1, ?2, ?3)",
            params![conv_name, now, now],
        )?;

        // Add participants
        for agent in participants {
            self.conn.execute(
                "INSERT INTO participants (conv_name, agent, joined_at) VALUES (?1, ?2, ?3)",
                params![conv_name, agent, now],
            )?;
        }

        Ok(conv_name)
    }

    /// Generate a unique fun name (adjective-noun format).
    fn generate_unique_fun_name(&self) -> Result<String, ConversationError> {
        // Try up to 100 times to find a unique name
        for _ in 0..100 {
            let name = generate_fun_name();
            if self.find_by_name(&name)?.is_none() {
                return Ok(name);
            }
        }
        // Fallback: append random suffix
        Ok(format!("{}-{}", generate_fun_name(), &generate_id()[..4]))
    }

    /// List all conversations.
    pub fn list_conversations(&self) -> Result<Vec<Conversation>, ConversationError> {
        let mut stmt = self.conn.prepare(
            "SELECT name, created_at, updated_at, paused, paused_at_msg_id FROM conversations ORDER BY updated_at DESC",
        )?;

        let conversations = stmt
            .query_map([], row_to_conversation)?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(conversations)
    }

    /// Match conversation names against a glob pattern.
    /// Supports * (any chars) and ? (single char).
    /// Returns matching conversation names sorted alphabetically.
    pub fn match_conversations(&self, pattern: &str) -> Result<Vec<String>, ConversationError> {
        let all_convs = self.list_conversations()?;

        // If no wildcards, return exact match only
        if !pattern.contains('*') && !pattern.contains('?') {
            if all_convs.iter().any(|c| c.name == pattern) {
                return Ok(vec![pattern.to_string()]);
            } else {
                return Ok(vec![]);
            }
        }

        // Convert glob pattern to regex
        let regex_pattern = pattern
            .replace('.', "\\.")
            .replace('*', ".*")
            .replace('?', ".");
        let regex = regex::Regex::new(&format!("^{}$", regex_pattern))?;

        let mut matches: Vec<String> = all_convs
            .iter()
            .filter(|c| regex.is_match(&c.name))
            .map(|c| c.name.clone())
            .collect();

        matches.sort();
        Ok(matches)
    }

    /// Get a conversation by name.
    pub fn get_conversation(&self, name: &str) -> Result<Option<Conversation>, ConversationError> {
        let mut stmt = self.conn.prepare(
            "SELECT name, created_at, updated_at, paused, paused_at_msg_id FROM conversations WHERE name = ?1",
        )?;

        let mut rows = stmt.query(params![name])?;
        match rows.next()? {
            Some(row) => Ok(Some(row_to_conversation(row)?)),
            None => Ok(None),
        }
    }

    /// Check if a conversation is paused.
    pub fn is_paused(&self, conv_name: &str) -> Result<bool, ConversationError> {
        match self.get_conversation(conv_name)? {
            Some(conv) => Ok(conv.paused),
            None => Err(ConversationError::NotFound(conv_name.to_string())),
        }
    }

    /// Set the paused state of a conversation.
    /// When pausing, records the current max message ID.
    /// When resuming, returns the paused_at_msg_id for catchup processing.
    pub fn set_paused(
        &self,
        conv_name: &str,
        paused: bool,
    ) -> Result<Option<i64>, ConversationError> {
        // Verify conversation exists and get current state
        let conv = self
            .get_conversation(conv_name)?
            .ok_or_else(|| ConversationError::NotFound(conv_name.to_string()))?;

        if paused {
            // Pausing: record the current max message ID
            let max_msg_id: Option<i64> = self
                .conn
                .query_row(
                    "SELECT MAX(id) FROM messages WHERE conv_name = ?1",
                    params![conv_name],
                    |row| row.get(0),
                )
                .ok();

            self.conn.execute(
                "UPDATE conversations SET paused = 1, paused_at_msg_id = ?1 WHERE name = ?2",
                params![max_msg_id, conv_name],
            )?;

            Ok(None)
        } else {
            // Resuming: get the paused_at_msg_id before clearing
            let paused_at_msg_id = conv.paused_at_msg_id;

            self.conn.execute(
                "UPDATE conversations SET paused = 0, paused_at_msg_id = NULL WHERE name = ?1",
                params![conv_name],
            )?;

            Ok(paused_at_msg_id)
        }
    }

    /// Get messages that need catchup processing after resume.
    /// Returns messages with id > paused_at_msg_id for the given conversation.
    pub fn get_catchup_messages(
        &self,
        conv_name: &str,
        paused_at_msg_id: i64,
    ) -> Result<Vec<ConversationMessage>, ConversationError> {
        self.get_messages_filtered(conv_name, None, Some(paused_at_msg_id))
    }

    /// Build catchup items from messages that were skipped during pause.
    /// Returns a list of CatchupItem structs for processing.
    ///
    /// For agent messages (not "user" or "tool"):
    /// - Checks if message may contain a tool call (has JSON block)
    /// - Extracts @mentions for forwarding
    ///
    /// For user messages:
    /// - Only extracts @mentions for forwarding
    ///
    /// Tool result messages are skipped (they'll be re-created when tools are executed).
    pub fn build_catchup_items(messages: &[ConversationMessage]) -> Vec<CatchupItem> {
        let mut items = Vec::new();

        for msg in messages {
            // Skip tool result messages - they'll be regenerated
            if msg.from_agent == "tool" {
                continue;
            }

            // Check if this is an agent message that might have a tool call
            let is_agent_msg = msg.from_agent != "user" && msg.from_agent != "tool";
            let has_tool_call = is_agent_msg && msg.content.contains("```json");

            // Parse @mentions from the message
            let mentions = parse_mentions(&msg.content);

            // Only include if there's something to process
            if has_tool_call || !mentions.is_empty() {
                items.push(CatchupItem {
                    message: msg.clone(),
                    has_tool_call,
                    raw_content: msg.content.clone(),
                    mentions,
                });
            }
        }

        items
    }

    /// Get participants for a conversation.
    pub fn get_participants(&self, conv_name: &str) -> Result<Vec<Participant>, ConversationError> {
        let mut stmt = self.conn.prepare(
            "SELECT conv_name, agent, joined_at FROM participants WHERE conv_name = ?1 ORDER BY joined_at",
        )?;

        let participants = stmt
            .query_map(params![conv_name], |row| {
                Ok(Participant {
                    conv_name: row.get(0)?,
                    agent: row.get(1)?,
                    joined_at: row.get(2)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(participants)
    }

    /// Add a participant to an existing conversation.
    /// Does nothing if the participant is already in the conversation.
    pub fn add_participant(&self, conv_name: &str, agent: &str) -> Result<(), ConversationError> {
        // Verify conversation exists
        if self.get_conversation(conv_name)?.is_none() {
            return Err(ConversationError::NotFound(conv_name.to_string()));
        }

        let now = current_timestamp();

        // Use INSERT OR IGNORE to handle duplicate participants gracefully
        self.conn.execute(
            "INSERT OR IGNORE INTO participants (conv_name, agent, joined_at) VALUES (?1, ?2, ?3)",
            params![conv_name, agent, now],
        )?;

        Ok(())
    }

    /// Add a message to a conversation.
    /// Returns the message ID.
    pub fn add_message(
        &self,
        conv_name: &str,
        from_agent: &str,
        content: &str,
        mentions: &[&str],
    ) -> Result<i64, ConversationError> {
        self.add_message_with_tokens(
            conv_name, from_agent, content, mentions, None, None, None, None, None, None,
        )
    }

    /// Add a message to a conversation with optional response duration and tool_calls.
    /// Returns the message ID.
    pub fn add_message_with_tool_calls(
        &self,
        conv_name: &str,
        from_agent: &str,
        content: &str,
        mentions: &[&str],
        duration_ms: Option<i64>,
        tool_calls: Option<&str>,
    ) -> Result<i64, ConversationError> {
        self.add_message_with_tokens(
            conv_name,
            from_agent,
            content,
            mentions,
            duration_ms,
            tool_calls,
            None,
            None,
            None,
            None,
        )
    }

    /// Add a message to a conversation with all optional fields including token tracking.
    /// Returns the message ID.
    #[allow(clippy::too_many_arguments)]
    pub fn add_message_with_tokens(
        &self,
        conv_name: &str,
        from_agent: &str,
        content: &str,
        mentions: &[&str],
        duration_ms: Option<i64>,
        tool_calls: Option<&str>,
        tokens_in: Option<i64>,
        tokens_out: Option<i64>,
        num_ctx: Option<i64>,
        prompt_eval_ns: Option<i64>,
    ) -> Result<i64, ConversationError> {
        self.add_message_full(
            conv_name, from_agent, content, mentions,
            duration_ms, tool_calls, tokens_in, tokens_out, num_ctx, None, prompt_eval_ns,
        )
    }

    /// Add a tool result message attributed to the agent that triggered it.
    /// This allows `format_conversation_history` to filter out other agents' tool results.
    pub fn add_tool_result(
        &self,
        conv_name: &str,
        content: &str,
        triggered_by: &str,
    ) -> Result<i64, ConversationError> {
        self.add_message_full(
            conv_name, "tool", content, &[],
            None, None, None, None, None, Some(triggered_by), None,
        )
    }

    /// Add a recall message attributed to the agent that owns it.
    /// This allows `format_conversation_history` to filter out other agents' recall injections.
    pub fn add_recall_message(
        &self,
        conv_name: &str,
        content: &str,
        triggered_by: &str,
    ) -> Result<i64, ConversationError> {
        self.add_message_full(
            conv_name, "recall", content, &[],
            None, None, None, None, None, Some(triggered_by), None,
        )
    }

    /// Internal: add a message with all fields including triggered_by.
    #[allow(clippy::too_many_arguments)]
    fn add_message_full(
        &self,
        conv_name: &str,
        from_agent: &str,
        content: &str,
        mentions: &[&str],
        duration_ms: Option<i64>,
        tool_calls: Option<&str>,
        tokens_in: Option<i64>,
        tokens_out: Option<i64>,
        num_ctx: Option<i64>,
        triggered_by: Option<&str>,
        prompt_eval_ns: Option<i64>,
    ) -> Result<i64, ConversationError> {
        // Verify conversation exists
        if self.get_conversation(conv_name)?.is_none() {
            return Err(ConversationError::NotFound(conv_name.to_string()));
        }

        let now = current_timestamp();
        let expires_at = now + DEFAULT_TTL_SECONDS;
        let mentions_json = serde_json::to_string(mentions).unwrap_or_else(|_| "[]".to_string());

        self.conn.execute(
            "INSERT INTO messages (conv_name, from_agent, content, mentions, created_at, expires_at, duration_ms, tool_calls, tokens_in, tokens_out, num_ctx, triggered_by, prompt_eval_ns) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            params![conv_name, from_agent, content, mentions_json, now, expires_at, duration_ms, tool_calls, tokens_in, tokens_out, num_ctx, triggered_by, prompt_eval_ns],
        )?;

        let message_id = self.conn.last_insert_rowid();

        // Update conversation's updated_at
        self.conn.execute(
            "UPDATE conversations SET updated_at = ?1 WHERE name = ?2",
            params![now, conv_name],
        )?;

        Ok(message_id)
    }

    /// Update token stats on an existing message by ID.
    pub fn update_message_stats(
        &self,
        message_id: i64,
        duration_ms: Option<i64>,
        tokens_in: Option<i64>,
        tokens_out: Option<i64>,
        num_ctx: Option<i64>,
        prompt_eval_ns: Option<i64>,
    ) -> Result<(), ConversationError> {
        self.conn.execute(
            "UPDATE messages SET duration_ms = ?1, tokens_in = ?2, tokens_out = ?3, num_ctx = ?4, prompt_eval_ns = ?5 WHERE id = ?6",
            params![duration_ms, tokens_in, tokens_out, num_ctx, prompt_eval_ns, message_id],
        )?;
        Ok(())
    }

    /// Stamp token stats on all messages from an agent that don't have stats yet.
    pub fn stamp_unstamped_messages(
        &self,
        conv_name: &str,
        from_agent: &str,
        tokens_in: Option<i64>,
        tokens_out: Option<i64>,
        num_ctx: Option<i64>,
        prompt_eval_ns: Option<i64>,
    ) -> Result<usize, ConversationError> {
        let count = self.conn.execute(
            "UPDATE messages SET tokens_in = ?1, tokens_out = ?2, num_ctx = ?3, prompt_eval_ns = ?4
             WHERE conv_name = ?5 AND from_agent = ?6 AND tokens_in IS NULL",
            params![tokens_in, tokens_out, num_ctx, prompt_eval_ns, conv_name, from_agent],
        )?;
        Ok(count)
    }

    /// Get messages for a conversation, optionally limited to the last N messages.
    /// Messages are returned in chronological order (oldest first).
    pub fn get_messages(
        &self,
        conv_name: &str,
        limit: Option<usize>,
    ) -> Result<Vec<ConversationMessage>, ConversationError> {
        let now = current_timestamp();

        let query = match limit {
            Some(n) => format!(
                "SELECT {} FROM messages WHERE conv_name = ?1 AND expires_at > ?2 ORDER BY created_at DESC LIMIT {}",
                MESSAGE_COLUMNS, n
            ),
            None => format!(
                "SELECT {} FROM messages WHERE conv_name = ?1 AND expires_at > ?2 ORDER BY created_at ASC",
                MESSAGE_COLUMNS
            ),
        };

        let mut stmt = self.conn.prepare(&query)?;
        let mut messages: Vec<ConversationMessage> = stmt
            .query_map(params![conv_name, now], row_to_message)?
            .collect::<Result<Vec<_>, _>>()?;

        // If we used LIMIT with DESC, reverse to get chronological order
        if limit.is_some() {
            messages.reverse();
        }

        Ok(messages)
    }

    /// Get messages for a conversation with filtering options.
    ///
    /// - `limit`: Return at most N messages (applied after filtering)
    /// - `since_id`: Only return messages with ID > since_id
    ///
    /// Messages are returned in chronological order (oldest first).
    pub fn get_messages_filtered(
        &self,
        conv_name: &str,
        limit: Option<usize>,
        since_id: Option<i64>,
    ) -> Result<Vec<ConversationMessage>, ConversationError> {
        let now = current_timestamp();
        let since = since_id.unwrap_or(0);

        // When using --since, we want messages after that ID in chronological order.
        // When using --limit without --since, we want the last N messages.
        let base = format!(
            "SELECT {} FROM messages WHERE conv_name = ?1 AND expires_at > ?2 AND id > ?3",
            MESSAGE_COLUMNS
        );
        let query = match (limit, since_id) {
            (Some(n), Some(_)) => format!("{} ORDER BY id ASC LIMIT {}", base, n),
            (Some(n), None) => format!("{} ORDER BY id DESC LIMIT {}", base, n),
            (None, _) => format!("{} ORDER BY id ASC", base),
        };

        let mut stmt = self.conn.prepare(&query)?;
        let mut messages: Vec<ConversationMessage> = stmt
            .query_map(params![conv_name, now, since], row_to_message)?
            .collect::<Result<Vec<_>, _>>()?;

        // If we used LIMIT with DESC (no since_id), reverse to get chronological order
        if limit.is_some() && since_id.is_none() {
            messages.reverse();
        }

        Ok(messages)
    }

    /// Pin or unpin a message. Returns error if the message doesn't exist in the conversation.
    pub fn pin_message(
        &self,
        conv_name: &str,
        msg_id: i64,
        pinned: bool,
    ) -> Result<(), ConversationError> {
        let rows = self.conn.execute(
            "UPDATE messages SET pinned = ?1 WHERE id = ?2 AND conv_name = ?3",
            params![pinned as i64, msg_id, conv_name],
        )?;
        if rows == 0 {
            return Err(ConversationError::NotFound(format!(
                "message {} in conversation {}",
                msg_id, conv_name
            )));
        }
        Ok(())
    }

    /// Unpin all pinned tool result messages in a conversation whose content contains the given text.
    /// Used to release spawn_child tool results after wait_for_children collects the result.
    pub fn unpin_tool_results_for(
        &self,
        conv_name: &str,
        content_match: &str,
    ) -> Result<usize, ConversationError> {
        let pattern = format!("%{}%", content_match);
        let rows = self.conn.execute(
            "UPDATE messages SET pinned = 0 WHERE conv_name = ?1 AND pinned = 1 AND from_agent = 'tool' AND content LIKE ?2",
            params![conv_name, pattern],
        )?;
        Ok(rows)
    }

    /// Get messages with pinned messages always included, regardless of the limit window.
    ///
    /// When `limit` is Some(n): fetches pinned messages + last N messages, merging them
    /// in chronological order without duplicates. Pinned messages that fall outside the
    /// recent window appear first.
    ///
    /// When `limit` is None: delegates to `get_messages()` (all messages returned).
    pub fn get_messages_with_pinned(
        &self,
        conv_name: &str,
        limit: Option<usize>,
    ) -> Result<Vec<ConversationMessage>, ConversationError> {
        let limit = match limit {
            Some(n) => n,
            None => return self.get_messages(conv_name, None),
        };

        let now = current_timestamp();

        // 1. Fetch pinned messages (chronological)
        let pinned_query = format!(
            "SELECT {} FROM messages WHERE conv_name = ?1 AND expires_at > ?2 AND pinned = 1 ORDER BY created_at ASC",
            MESSAGE_COLUMNS
        );
        let mut pinned_stmt = self.conn.prepare(&pinned_query)?;
        let pinned: Vec<ConversationMessage> = pinned_stmt
            .query_map(params![conv_name, now], row_to_message)?
            .collect::<Result<Vec<_>, _>>()?;

        // 2. Fetch last N messages (DESC + reverse)
        let recent_query = format!(
            "SELECT {} FROM messages WHERE conv_name = ?1 AND expires_at > ?2 ORDER BY created_at DESC LIMIT {}",
            MESSAGE_COLUMNS, limit
        );
        let mut recent_stmt = self.conn.prepare(&recent_query)?;
        let mut recent: Vec<ConversationMessage> = recent_stmt
            .query_map(params![conv_name, now], row_to_message)?
            .collect::<Result<Vec<_>, _>>()?;
        recent.reverse();

        // 3. Merge: pinned messages outside the window go first, then recent in order
        let recent_ids: std::collections::HashSet<i64> = recent.iter().map(|m| m.id).collect();
        let mut result: Vec<ConversationMessage> = pinned
            .into_iter()
            .filter(|m| !recent_ids.contains(&m.id))
            .collect();
        result.extend(recent);

        Ok(result)
    }

    /// Get the context cursor for an agent in a conversation.
    /// Returns None if no cursor is set (cold start needed).
    pub fn get_context_cursor(
        &self,
        conv_name: &str,
        agent: &str,
    ) -> Result<Option<i64>, ConversationError> {
        let mut stmt = self.conn.prepare(
            "SELECT context_cursor FROM participants WHERE conv_name = ?1 AND agent = ?2",
        )?;
        let cursor: Option<Option<i64>> = stmt
            .query_row(params![conv_name, agent], |row| row.get(0))
            .ok();
        Ok(cursor.flatten())
    }

    /// Set the context cursor for an agent in a conversation.
    pub fn set_context_cursor(
        &self,
        conv_name: &str,
        agent: &str,
        cursor_id: i64,
    ) -> Result<(), ConversationError> {
        self.conn.execute(
            "UPDATE participants SET context_cursor = ?1 WHERE conv_name = ?2 AND agent = ?3",
            params![cursor_id, conv_name, agent],
        )?;
        Ok(())
    }

    /// Clear the context cursor for an agent (triggers cold start on next turn).
    pub fn clear_context_cursor(
        &self,
        conv_name: &str,
        agent: &str,
    ) -> Result<(), ConversationError> {
        self.conn.execute(
            "UPDATE participants SET context_cursor = NULL WHERE conv_name = ?1 AND agent = ?2",
            params![conv_name, agent],
        )?;
        Ok(())
    }

    /// Count messages in a conversation from a given message ID (inclusive).
    pub fn count_messages_from(
        &self,
        conv_name: &str,
        from_id: i64,
    ) -> Result<i64, ConversationError> {
        let now = current_timestamp();
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM messages WHERE conv_name = ?1 AND id >= ?2 AND expires_at > ?3",
            params![conv_name, from_id, now],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    /// Get messages from a cursor ID (inclusive), merged with pinned messages.
    /// Returns messages in chronological order with deduplication.
    pub fn get_messages_from_with_pinned(
        &self,
        conv_name: &str,
        from_id: i64,
    ) -> Result<Vec<ConversationMessage>, ConversationError> {
        let now = current_timestamp();

        // 1. Fetch pinned messages (chronological)
        let pinned_query = format!(
            "SELECT {} FROM messages WHERE conv_name = ?1 AND expires_at > ?2 AND pinned = 1 ORDER BY created_at ASC",
            MESSAGE_COLUMNS
        );
        let mut pinned_stmt = self.conn.prepare(&pinned_query)?;
        let pinned: Vec<ConversationMessage> = pinned_stmt
            .query_map(params![conv_name, now], row_to_message)?
            .collect::<Result<Vec<_>, _>>()?;

        // 2. Fetch messages from cursor inclusive (chronological)
        let from_query = format!(
            "SELECT {} FROM messages WHERE conv_name = ?1 AND id >= ?2 AND expires_at > ?3 ORDER BY created_at ASC",
            MESSAGE_COLUMNS
        );
        let mut from_stmt = self.conn.prepare(&from_query)?;
        let from_msgs: Vec<ConversationMessage> = from_stmt
            .query_map(params![conv_name, from_id, now], row_to_message)?
            .collect::<Result<Vec<_>, _>>()?;

        // 3. Merge: pinned messages not in from-range go first, then from messages
        let from_ids: std::collections::HashSet<i64> = from_msgs.iter().map(|m| m.id).collect();
        let mut result: Vec<ConversationMessage> = pinned
            .into_iter()
            .filter(|m| !from_ids.contains(&m.id))
            .collect();
        result.extend(from_msgs);

        Ok(result)
    }

    /// Store an embedding for a message (for conversation recall).
    pub fn store_message_embedding(
        &self,
        message_id: i64,
        conv_name: &str,
        embedding: &[f32],
    ) -> Result<(), ConversationError> {
        let blob = crate::memory::embedding_to_blob(embedding);
        self.conn.execute(
            "INSERT OR REPLACE INTO message_embeddings (message_id, conv_name, embedding) VALUES (?1, ?2, ?3)",
            params![message_id, conv_name, blob],
        )?;
        Ok(())
    }

    /// Search for similar messages in a conversation using cosine similarity.
    ///
    /// Returns `(message_id, from_agent, content, created_at, similarity)` tuples sorted by
    /// similarity descending, excluding messages in `exclude_ids` (the current window).
    /// Includes user and agent messages but excludes tool results, recall injections,
    /// empty messages, and tool-call intermediaries.
    pub fn search_similar_messages(
        &self,
        conv_name: &str,
        query_embedding: &[f32],
        exclude_ids: &[i64],
        limit: usize,
    ) -> Result<Vec<(i64, String, String, i64, f32)>, ConversationError> {
        use crate::embedding::cosine_similarity;
        use crate::memory::blob_to_embedding;

        let now = current_timestamp();

        // Load all embeddings for substantive messages in this conversation (non-expired)
        let mut stmt = self.conn.prepare(
            "SELECT me.message_id, m.from_agent, m.content, m.created_at, me.embedding
             FROM message_embeddings me
             JOIN messages m ON me.message_id = m.id
             WHERE me.conv_name = ?1
               AND m.from_agent NOT IN ('tool', 'recall')
               AND m.content != ''
               AND m.tool_calls IS NULL
               AND m.expires_at > ?2
             ORDER BY m.created_at DESC",
        )?;

        let exclude_set: std::collections::HashSet<i64> = exclude_ids.iter().copied().collect();

        let rows = stmt.query_map(params![conv_name, now], |row| {
            let message_id: i64 = row.get(0)?;
            let from_agent: String = row.get(1)?;
            let content: String = row.get(2)?;
            let created_at: i64 = row.get(3)?;
            let embedding_blob: Vec<u8> = row.get(4)?;
            Ok((message_id, from_agent, content, created_at, embedding_blob))
        })?;

        let mut scored: Vec<(i64, String, String, i64, f32)> = Vec::new();

        for row in rows {
            let (message_id, from_agent, content, created_at, embedding_blob) = row?;

            // Skip messages in the current window
            if exclude_set.contains(&message_id) {
                continue;
            }

            let embedding = blob_to_embedding(&embedding_blob);
            let similarity = cosine_similarity(query_embedding, &embedding);

            if similarity > 0.0 {
                scored.push((message_id, from_agent, content, created_at, similarity));
            }
        }

        // Sort by similarity descending
        scored.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        Ok(scored)
    }

    /// Search conversation messages by keyword (SQL LIKE).
    /// Supports optional sender filter, exclusion of internal messages (tool/recall),
    /// and content truncation for long messages.
    pub fn search_messages(
        &self,
        conv_name: &str,
        keyword: &str,
        from_agent: Option<&str>,
        include_internal: bool,
        limit: usize,
        max_content_len: Option<usize>,
    ) -> Result<Vec<ConversationMessage>, ConversationError> {
        let pattern = format!("%{}%", keyword);

        let (sql, has_from) = {
            let mut sql = format!(
                "SELECT {} FROM messages WHERE conv_name = ?1 AND content LIKE ?2",
                MESSAGE_COLUMNS
            );
            let has_from = from_agent.is_some();
            if has_from {
                sql.push_str(" AND from_agent = ?3");
            } else if !include_internal {
                sql.push_str(" AND from_agent NOT IN ('tool', 'recall')");
            }
            sql.push_str(" ORDER BY created_at ASC LIMIT ");
            sql.push_str(&limit.to_string());
            (sql, has_from)
        };

        let mut messages = if has_from {
            let mut stmt = self.conn.prepare(&sql)?;
            stmt.query_map(
                params![conv_name, pattern, from_agent.unwrap()],
                row_to_message,
            )?
            .collect::<Result<Vec<_>, _>>()?
        } else {
            let mut stmt = self.conn.prepare(&sql)?;
            stmt.query_map(params![conv_name, pattern], row_to_message)?
                .collect::<Result<Vec<_>, _>>()?
        };

        // Truncate content if requested
        if let Some(max_len) = max_content_len {
            for msg in &mut messages {
                if msg.content.len() > max_len {
                    let end = msg.content.floor_char_boundary(max_len.saturating_sub(3));
                    msg.content = format!("{}...", &msg.content[..end]);
                }
            }
        }

        Ok(messages)
    }

    /// Add a pending notification for an offline agent.
    pub fn add_pending_notification(
        &self,
        agent: &str,
        conv_name: &str,
        message_id: i64,
    ) -> Result<i64, ConversationError> {
        let now = current_timestamp();

        self.conn.execute(
            "INSERT INTO pending_notifications (agent, conv_name, message_id, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![agent, conv_name, message_id, now],
        )?;

        Ok(self.conn.last_insert_rowid())
    }

    /// Get pending notifications for an agent.
    pub fn get_pending_notifications(
        &self,
        agent: &str,
    ) -> Result<Vec<PendingNotification>, ConversationError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, agent, conv_name, message_id, created_at
             FROM pending_notifications
             WHERE agent = ?1
             ORDER BY created_at",
        )?;

        let notifications = stmt
            .query_map(params![agent], |row| {
                Ok(PendingNotification {
                    id: row.get(0)?,
                    agent: row.get(1)?,
                    conv_name: row.get(2)?,
                    message_id: row.get(3)?,
                    created_at: row.get(4)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(notifications)
    }

    /// Clear pending notifications for an agent.
    pub fn clear_pending_notifications(&self, agent: &str) -> Result<usize, ConversationError> {
        let count = self.conn.execute(
            "DELETE FROM pending_notifications WHERE agent = ?1",
            params![agent],
        )?;
        Ok(count)
    }

    /// Get pending notifications for a specific conversation.
    pub fn get_pending_notifications_for_conversation(
        &self,
        conv_name: &str,
    ) -> Result<Vec<PendingNotification>, ConversationError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, agent, conv_name, message_id, created_at
             FROM pending_notifications
             WHERE conv_name = ?1
             ORDER BY created_at",
        )?;

        let notifications = stmt
            .query_map(params![conv_name], |row| {
                Ok(PendingNotification {
                    id: row.get(0)?,
                    agent: row.get(1)?,
                    conv_name: row.get(2)?,
                    message_id: row.get(3)?,
                    created_at: row.get(4)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(notifications)
    }

    /// Clear all pending notifications for a specific conversation.
    /// Used by stop command to abandon queued work without processing.
    pub fn clear_pending_notifications_for_conversation(&self, conv_name: &str) -> Result<usize, ConversationError> {
        let deleted = self.conn.execute(
            "DELETE FROM pending_notifications WHERE conv_name = ?1",
            params![conv_name],
        )?;
        Ok(deleted)
    }

    /// Delete a single pending notification by ID.
    pub fn delete_pending_notification(
        &self,
        notification_id: i64,
    ) -> Result<(), ConversationError> {
        self.conn.execute(
            "DELETE FROM pending_notifications WHERE id = ?1",
            params![notification_id],
        )?;
        Ok(())
    }

    /// Delete expired messages and clean up empty conversations.
    pub fn cleanup_expired(&self) -> Result<(usize, usize), ConversationError> {
        let now = current_timestamp();

        // Delete embeddings for expired messages
        self.conn.execute(
            "DELETE FROM message_embeddings WHERE message_id IN (SELECT id FROM messages WHERE expires_at <= ?1)",
            params![now],
        )?;

        // Delete expired messages
        let messages_deleted = self
            .conn
            .execute("DELETE FROM messages WHERE expires_at <= ?1", params![now])?;

        // Delete pending notifications for deleted messages
        self.conn.execute(
            "DELETE FROM pending_notifications WHERE message_id NOT IN (SELECT id FROM messages)",
            [],
        )?;

        // Delete conversations with no messages
        let convs_deleted = self.conn.execute(
            "DELETE FROM conversations WHERE name NOT IN (SELECT DISTINCT conv_name FROM messages)",
            [],
        )?;

        // Delete orphaned participants
        self.conn.execute(
            "DELETE FROM participants WHERE conv_name NOT IN (SELECT name FROM conversations)",
            [],
        )?;

        Ok((messages_deleted, convs_deleted))
    }

    /// Clear all messages from a conversation (keeps the conversation and participants).
    pub fn clear_messages(&self, conv_name: &str) -> Result<usize, ConversationError> {
        // Check conversation exists
        self.find_by_name(conv_name)?
            .ok_or_else(|| ConversationError::NotFound(conv_name.to_string()))?;

        // Delete pending notifications for this conversation
        self.conn.execute(
            "DELETE FROM pending_notifications WHERE conv_name = ?1",
            params![conv_name],
        )?;

        // Delete embeddings for this conversation
        self.conn.execute(
            "DELETE FROM message_embeddings WHERE conv_name = ?1",
            params![conv_name],
        )?;

        // Delete all messages
        let deleted = self.conn.execute(
            "DELETE FROM messages WHERE conv_name = ?1",
            params![conv_name],
        )?;

        Ok(deleted)
    }

    /// Delete a conversation and all its related data.
    pub fn delete_conversation(&self, conv_name: &str) -> Result<(), ConversationError> {
        self.conn.execute(
            "DELETE FROM pending_notifications WHERE conv_name = ?1",
            params![conv_name],
        )?;
        self.conn.execute(
            "DELETE FROM message_embeddings WHERE conv_name = ?1",
            params![conv_name],
        )?;
        self.conn.execute(
            "DELETE FROM messages WHERE conv_name = ?1",
            params![conv_name],
        )?;
        self.conn.execute(
            "DELETE FROM participants WHERE conv_name = ?1",
            params![conv_name],
        )?;
        self.conn.execute(
            "DELETE FROM conversations WHERE name = ?1",
            params![conv_name],
        )?;

        Ok(())
    }

    /// Get the count of non-expired messages in a conversation.
    pub fn get_message_count(&self, conv_name: &str) -> Result<i64, ConversationError> {
        let now = current_timestamp();
        let mut stmt = self
            .conn
            .prepare("SELECT COUNT(*) FROM messages WHERE conv_name = ?1 AND expires_at > ?2")?;
        let count: i64 = stmt.query_row(params![conv_name, now], |row| row.get(0))?;
        Ok(count)
    }

    /// Find a conversation by its name.
    pub fn find_by_name(&self, name: &str) -> Result<Option<Conversation>, ConversationError> {
        self.get_conversation(name)
    }

    /// Find a conversation by its exact set of participants.
    /// Participants must match exactly (same set, sorted).
    pub fn find_by_participants(
        &self,
        participants: &[&str],
    ) -> Result<Option<Conversation>, ConversationError> {
        // Get all conversations and check participants
        let conversations = self.list_conversations()?;

        for conv in conversations {
            let conv_participants = self.get_participants(&conv.name)?;
            let mut conv_agents: Vec<&str> =
                conv_participants.iter().map(|p| p.agent.as_str()).collect();
            conv_agents.sort();

            let mut target: Vec<&str> = participants.to_vec();
            target.sort();

            if conv_agents == target {
                return Ok(Some(conv));
            }
        }

        Ok(None)
    }

    /// Get or create a conversation with the given participants.
    /// If a conversation with these exact participants exists, returns it.
    /// Otherwise, creates a new conversation with a generated fun name.
    pub fn get_or_create_conversation(
        &self,
        participants: &[&str],
    ) -> Result<Conversation, ConversationError> {
        // Try to find existing conversation by participants
        if let Some(conv) = self.find_by_participants(participants)? {
            return Ok(conv);
        }

        // Create new conversation with a generated fun name
        let now = current_timestamp();
        let conv_name = self.generate_unique_fun_name()?;

        self.conn.execute(
            "INSERT INTO conversations (name, created_at, updated_at, paused) VALUES (?1, ?2, ?3, 0)",
            params![conv_name, now, now],
        )?;

        // Add participants
        for agent in participants {
            self.conn.execute(
                "INSERT INTO participants (conv_name, agent, joined_at) VALUES (?1, ?2, ?3)",
                params![conv_name, agent, now],
            )?;
        }

        Ok(Conversation {
            name: conv_name,
            created_at: now,
            updated_at: now,
            paused: false,
            paused_at_msg_id: None,
        })
    }
}

/// Map a database row to a Conversation struct.
/// Expects columns: name, created_at, updated_at, paused, paused_at_msg_id.
fn row_to_conversation(row: &rusqlite::Row) -> rusqlite::Result<Conversation> {
    Ok(Conversation {
        name: row.get(0)?,
        created_at: row.get(1)?,
        updated_at: row.get(2)?,
        paused: row.get::<_, i64>(3)? != 0,
        paused_at_msg_id: row.get(4)?,
    })
}

/// Map a database row to a ConversationMessage struct.
/// Expects columns: id, conv_name, from_agent, content, mentions, created_at,
/// expires_at, duration_ms, tool_calls, tokens_in, tokens_out, num_ctx, triggered_by.
fn row_to_message(row: &rusqlite::Row) -> rusqlite::Result<ConversationMessage> {
    let mentions_json: String = row.get(4)?;
    let mentions: Vec<String> = serde_json::from_str(&mentions_json).unwrap_or_default();
    Ok(ConversationMessage {
        id: row.get(0)?,
        conv_name: row.get(1)?,
        from_agent: row.get(2)?,
        content: row.get(3)?,
        mentions,
        created_at: row.get(5)?,
        expires_at: row.get(6)?,
        duration_ms: row.get(7)?,
        tool_calls: row.get(8)?,
        tokens_in: row.get(9)?,
        tokens_out: row.get(10)?,
        num_ctx: row.get(11)?,
        triggered_by: row.get(12)?,
        pinned: row.get::<_, i64>(13).unwrap_or(0) != 0,
        prompt_eval_ns: row.get(14)?,
    })
}

/// SQL column list for message queries.
const MESSAGE_COLUMNS: &str = "id, conv_name, from_agent, content, mentions, created_at, expires_at, duration_ms, tool_calls, tokens_in, tokens_out, num_ctx, triggered_by, pinned, prompt_eval_ns";

/// Generate a fun name in "adjective-noun" format.
/// Example: "wild-screwdriver", "quiet-harbor", "swift-falcon"
pub fn generate_fun_name() -> String {
    use rand::prelude::IndexedRandom;

    const ADJECTIVES: &[&str] = &[
        "swift", "quiet", "wild", "bright", "gentle", "bold", "calm", "clever", "eager", "fancy",
        "grand", "happy", "jolly", "keen", "lucky", "merry", "noble", "proud", "quick", "rapid",
        "sharp", "silent", "smooth", "steady", "sunny", "tender", "vivid", "warm", "witty",
        "zesty", "cosmic", "daring", "frozen", "golden", "hidden", "iron", "jade", "kind", "lazy",
        "mighty", "nimble", "orange", "purple", "royal", "silver", "tiny", "ultra", "velvet",
        "wiry", "young",
    ];

    const NOUNS: &[&str] = &[
        "falcon",
        "harbor",
        "screwdriver",
        "mountain",
        "river",
        "thunder",
        "whisper",
        "arrow",
        "beacon",
        "canyon",
        "dagger",
        "eclipse",
        "forest",
        "glacier",
        "horizon",
        "island",
        "jungle",
        "knight",
        "lantern",
        "meadow",
        "nebula",
        "ocean",
        "phoenix",
        "quartz",
        "raven",
        "shadow",
        "temple",
        "umbrella",
        "valley",
        "waterfall",
        "xenon",
        "yarn",
        "zenith",
        "anchor",
        "bridge",
        "compass",
        "dragon",
        "ember",
        "flame",
        "garden",
        "hammer",
        "iceberg",
        "jewel",
        "kite",
        "lightning",
        "mirror",
        "needle",
        "orbit",
        "prism",
    ];

    let mut rng = rand::rng();
    let adj = ADJECTIVES.choose(&mut rng).unwrap_or(&"swift");
    let noun = NOUNS.choose(&mut rng).unwrap_or(&"falcon");

    format!("{}-{}", adj, noun)
}

/// Generate a unique conversation ID.
fn generate_id() -> String {
    use rand::Rng;
    let mut rng = rand::rng();
    let bytes: [u8; 8] = rng.random();
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Get current Unix timestamp.
fn current_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

/// Parse @mentions from message content.
/// Returns a list of unique agent names mentioned (excluding "user").
///
/// Matches patterns like @agentname where agentname contains alphanumeric chars,
/// hyphens, and underscores.
///
/// Ignores mentions inside:
/// - Single backticks: `@name`
/// - Triple-backtick code blocks: ```...@name...```
pub fn parse_mentions(content: &str) -> Vec<String> {
    use regex::Regex;

    // Remove code blocks and inline code before parsing mentions.
    // Order matters: remove triple-backtick blocks first (they may contain single backticks).
    let code_block_re = Regex::new(r"```[\s\S]*?```").unwrap();
    let inline_code_re = Regex::new(r"`[^`]+`").unwrap();

    let without_code_blocks = code_block_re.replace_all(content, "");
    let without_inline_code = inline_code_re.replace_all(&without_code_blocks, "");

    let mention_re = Regex::new(r"@([a-zA-Z0-9_-]+)").unwrap();
    let mut mentions: Vec<String> = mention_re
        .captures_iter(&without_inline_code)
        .map(|cap| cap[1].to_string())
        .filter(|name| name != "user") // Don't notify "user"
        .collect();

    // Deduplicate
    mentions.sort();
    mentions.dedup();
    mentions
}

/// Expand @all mention to all participants except "user".
///
/// If mentions contains "all", it is replaced with all agents from participants
/// (excluding "user"). Returns the expanded and deduplicated list.
pub fn expand_all_mention(mentions: &[String], participants: &[String]) -> Vec<String> {
    let has_all = mentions.iter().any(|m| m == "all");

    if !has_all {
        return mentions.to_vec();
    }

    // Start with non-"all" mentions
    let mut expanded: Vec<String> = mentions.iter().filter(|m| *m != "all").cloned().collect();

    // Add all participants except "user"
    for p in participants {
        if p != "user" && !expanded.contains(p) {
            expanded.push(p.clone());
        }
    }

    // Sort and deduplicate
    expanded.sort();
    expanded.dedup();
    expanded
}

/// Result of attempting to notify an agent.
#[derive(Debug)]
pub enum NotifyResult {
    /// Agent was notified via socket (running daemon) - synchronous response
    Notified { response_message_id: i64 },
    /// Agent acknowledged notification (fire-and-forget) - will process asynchronously
    Acknowledged,
    /// Agent was not running; notification queued
    Queued { notification_id: i64 },
    /// Agent does not exist (no config.toml found)
    UnknownAgent,
    /// Notification failed
    Failed { reason: String },
}

/// Queue a notification for an offline agent, or return UnknownAgent if the agent doesn't exist.
fn queue_notification(
    store: &ConversationStore,
    agent_name: &str,
    conv_id: &str,
    message_id: i64,
) -> NotifyResult {
    if !crate::discovery::agent_exists(agent_name) {
        return NotifyResult::UnknownAgent;
    }
    match store.add_pending_notification(agent_name, conv_id, message_id) {
        Ok(notification_id) => NotifyResult::Queued { notification_id },
        Err(e) => NotifyResult::Failed {
            reason: format!("Failed to queue notification: {}", e),
        },
    }
}

/// Map a socket response to a NotifyResult.
fn response_to_notify_result(response: Result<Option<crate::socket_api::Response>, crate::socket_api::SocketApiError>) -> NotifyResult {
    use crate::socket_api::Response;

    match response {
        Ok(Some(Response::NotifyReceived)) => NotifyResult::Acknowledged,
        Ok(Some(Response::Notified { response_message_id })) => {
            NotifyResult::Notified { response_message_id }
        }
        Ok(Some(Response::Error { message })) => NotifyResult::Failed { reason: message },
        Ok(Some(_)) => NotifyResult::Failed {
            reason: "Unexpected response type".to_string(),
        },
        Ok(None) => NotifyResult::Failed {
            reason: "Connection closed".to_string(),
        },
        Err(e) => NotifyResult::Failed {
            reason: format!("Failed to read response: {}", e),
        },
    }
}

/// Attempt to notify agents mentioned in a message.
/// For running agents: sends Notify request via socket.
/// For offline agents: queues notification in pending_notifications table.
///
/// Returns a map of agent name to NotifyResult.
pub async fn notify_mentioned_agents(
    store: &ConversationStore,
    conv_id: &str,
    message_id: i64,
    mentions: &[String],
) -> std::collections::HashMap<String, NotifyResult> {
    use crate::discovery;
    use crate::socket_api::{Request, SocketApi};
    use tokio::net::UnixStream;

    let mut results = std::collections::HashMap::new();

    for agent_name in mentions {
        let running_agent = match discovery::get_running_agent(agent_name) {
            Some(agent) => agent,
            None => {
                let result = queue_notification(store, agent_name, conv_id, message_id);
                results.insert(agent_name.clone(), result);
                continue;
            }
        };

        let stream = match UnixStream::connect(&running_agent.socket_path).await {
            Ok(s) => s,
            Err(_) => {
                let result = queue_notification(store, agent_name, conv_id, message_id);
                results.insert(agent_name.clone(), result);
                continue;
            }
        };

        let mut api = SocketApi::new(stream);
        let request = Request::Notify {
            conv_id: conv_id.to_string(),
            message_id,
            depth: 0,
        };

        if let Err(e) = api.write_request(&request).await {
            results.insert(
                agent_name.clone(),
                NotifyResult::Failed {
                    reason: format!("Failed to send request: {}", e),
                },
            );
            continue;
        }

        let result = response_to_notify_result(api.read_response().await);
        results.insert(agent_name.clone(), result);
    }

    results
}

/// Notify a single agent and return its result.
/// This is a helper function for parallel notification.
async fn notify_single_agent(
    agent_name: String,
    conv_id: String,
    message_id: i64,
) -> (String, NotifyResult) {
    use crate::discovery;
    use crate::socket_api::{Request, SocketApi};
    use tokio::net::UnixStream;

    let running_agent = match discovery::get_running_agent(&agent_name) {
        Some(agent) => agent,
        None => {
            return (
                agent_name,
                NotifyResult::Failed {
                    reason: "Agent not running".to_string(),
                },
            );
        }
    };

    let stream = match UnixStream::connect(&running_agent.socket_path).await {
        Ok(s) => s,
        Err(_) => {
            return (
                agent_name,
                NotifyResult::Failed {
                    reason: "Agent not reachable (socket connection failed)".to_string(),
                },
            );
        }
    };

    let mut api = SocketApi::new(stream);
    let request = Request::Notify {
        conv_id,
        message_id,
        depth: 0,
    };

    if let Err(e) = api.write_request(&request).await {
        return (
            agent_name,
            NotifyResult::Failed {
                reason: format!("Failed to send request: {}", e),
            },
        );
    }

    let result = response_to_notify_result(api.read_response().await);
    (agent_name, result)
}

/// Notify multiple agents in parallel.
/// Agents are notified concurrently using tokio::spawn, and responses arrive in any order.
/// For offline agents, notifications are queued for later processing.
///
/// This is the preferred function for group conversations where multiple agents
/// need to be notified simultaneously.
///
/// Returns a map of agent name to NotifyResult.
pub async fn notify_mentioned_agents_parallel(
    _store: &ConversationStore,
    conv_id: &str,
    message_id: i64,
    mentions: &[String],
) -> std::collections::HashMap<String, NotifyResult> {
    // Delegate to implementation that creates its own store for queuing
    // This avoids holding store reference across await points (Send issue)
    notify_mentioned_agents_parallel_owned(conv_id.to_string(), message_id, mentions.to_vec()).await
}

/// Parallel notification that owns all its data (no references held across await).
/// This version is safe to use from tokio::spawn.
pub async fn notify_mentioned_agents_parallel_owned(
    conv_id: String,
    message_id: i64,
    mentions: Vec<String>,
) -> std::collections::HashMap<String, NotifyResult> {
    use futures_util::future::join_all;

    let mut results = std::collections::HashMap::new();

    if mentions.is_empty() {
        return results;
    }

    // Check if conversation is paused -- if so, queue all notifications instead of sending.
    let is_paused = ConversationStore::init()
        .ok()
        .and_then(|store| store.is_paused(&conv_id).ok())
        .unwrap_or(false);

    if is_paused {
        match ConversationStore::init() {
            Ok(store) => {
                for agent_name in &mentions {
                    let result = queue_notification(&store, agent_name, &conv_id, message_id);
                    results.insert(agent_name.clone(), result);
                }
            }
            Err(e) => {
                for agent_name in &mentions {
                    results.insert(
                        agent_name.clone(),
                        NotifyResult::Failed {
                            reason: format!("Conversation paused but failed to queue: {}", e),
                        },
                    );
                }
            }
        }
        return results;
    }

    // Spawn parallel notification tasks
    let futures: Vec<_> = mentions
        .iter()
        .map(|agent_name| {
            let agent = agent_name.clone();
            let cid = conv_id.clone();
            tokio::spawn(async move { notify_single_agent(agent, cid, message_id).await })
        })
        .collect();

    let outcomes = join_all(futures).await;

    // Collect agents that need queuing (not running or not reachable)
    let mut agents_to_queue: Vec<String> = Vec::new();

    for outcome in outcomes {
        match outcome {
            Ok((agent_name, result)) => {
                let final_result = match &result {
                    NotifyResult::Failed { reason }
                        if reason.contains("not running") || reason.contains("not reachable") =>
                    {
                        if !crate::discovery::agent_exists(&agent_name) {
                            NotifyResult::UnknownAgent
                        } else {
                            agents_to_queue.push(agent_name.clone());
                            NotifyResult::Failed {
                                reason: "pending_queue".to_string(),
                            }
                        }
                    }
                    _ => result,
                };
                results.insert(agent_name, final_result);
            }
            Err(e) => {
                eprintln!("Notification task panicked: {}", e);
            }
        }
    }

    // Queue notifications for offline agents
    if !agents_to_queue.is_empty() {
        match ConversationStore::init() {
            Ok(store) => {
                for agent in agents_to_queue {
                    let result = queue_notification(&store, &agent, &conv_id, message_id);
                    results.insert(agent, result);
                }
            }
            Err(e) => {
                for agent in agents_to_queue {
                    results.insert(
                        agent,
                        NotifyResult::Failed {
                            reason: format!("Failed to open store for queuing: {}", e),
                        },
                    );
                }
            }
        }
    }

    results
}

/// Fire-and-forget notification - spawns notification tasks without waiting for responses.
/// This is used for CLI commands like `anima chat send` where we want to trigger
/// notifications but return immediately without blocking.
///
/// Unlike `notify_mentioned_agents_parallel`, this function:
/// - Does NOT wait for responses
/// - Queues notifications if conversation is paused
/// - Returns immediately after spawning tasks
///
/// The daemon will handle the conversation chain autonomously.
pub fn notify_mentioned_agents_fire_and_forget(
    conv_id: &str,
    message_id: i64,
    mentions: &[String],
) {
    let is_paused = ConversationStore::init()
        .ok()
        .and_then(|store| store.is_paused(conv_id).ok())
        .unwrap_or(false);

    if is_paused {
        if let Ok(store) = ConversationStore::init() {
            for agent_name in mentions {
                if crate::discovery::agent_exists(agent_name) {
                    let _ = store.add_pending_notification(agent_name, conv_id, message_id);
                }
            }
        }
        return;
    }

    for agent_name in mentions {
        let agent = agent_name.clone();
        let cid = conv_id.to_string();
        tokio::spawn(async move {
            let _ = notify_single_agent(agent, cid, message_id).await;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn test_store() -> ConversationStore {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_conversations.db");
        // Keep dir alive by leaking it (test only)
        std::mem::forget(dir);
        ConversationStore::open(&db_path).unwrap()
    }

    #[test]
    fn test_create_conversation() {
        let store = test_store();

        let name = store
            .create_conversation(Some("test-conv"), &["arya", "gendry"])
            .unwrap();

        assert_eq!(name, "test-conv");

        // Verify conversation exists
        let conv = store.get_conversation(&name).unwrap().unwrap();
        assert_eq!(conv.name, "test-conv");

        // Verify participants
        let participants = store.get_participants(&name).unwrap();
        assert_eq!(participants.len(), 2);
        assert!(participants.iter().any(|p| p.agent == "arya"));
        assert!(participants.iter().any(|p| p.agent == "gendry"));
    }

    #[test]
    fn test_create_conversation_no_name() {
        let store = test_store();

        let name = store.create_conversation(None, &["user", "bot"]).unwrap();

        // Should generate a fun name (adjective-noun format)
        assert!(
            name.contains('-'),
            "Generated name should have a hyphen: {}",
            name
        );

        let conv = store.get_conversation(&name).unwrap().unwrap();
        assert_eq!(conv.name, name);
    }

    #[test]
    fn test_list_conversations() {
        let store = test_store();

        // Create multiple conversations
        store
            .create_conversation(Some("conv-1"), &["a", "b"])
            .unwrap();
        store
            .create_conversation(Some("conv-2"), &["c", "d"])
            .unwrap();

        let convs = store.list_conversations().unwrap();
        assert_eq!(convs.len(), 2);
    }

    #[test]
    fn test_add_and_get_messages() {
        let store = test_store();

        let conv_name = store
            .create_conversation(Some("chat"), &["alice", "bob"])
            .unwrap();

        // Add messages
        store
            .add_message(&conv_name, "alice", "Hello @bob!", &["bob"])
            .unwrap();
        store
            .add_message(&conv_name, "bob", "Hi @alice!", &["alice"])
            .unwrap();
        store
            .add_message(&conv_name, "alice", "How are you?", &[])
            .unwrap();

        // Get all messages
        let messages = store.get_messages(&conv_name, None).unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].from_agent, "alice");
        assert_eq!(messages[0].content, "Hello @bob!");
        assert_eq!(messages[0].mentions, vec!["bob"]);

        // Get last 2 messages
        let recent = store.get_messages(&conv_name, Some(2)).unwrap();
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].content, "Hi @alice!");
        assert_eq!(recent[1].content, "How are you?");
    }

    #[test]
    fn test_add_message_nonexistent_conversation() {
        let store = test_store();

        let result = store.add_message("nonexistent", "user", "Hello", &[]);
        assert!(matches!(result, Err(ConversationError::NotFound(_))));
    }

    #[test]
    fn test_pending_notifications() {
        let store = test_store();

        let conv_name = store
            .create_conversation(Some("chat"), &["alice", "bob"])
            .unwrap();
        let msg_id = store
            .add_message(&conv_name, "alice", "Hey @bob", &["bob"])
            .unwrap();

        // Add pending notification
        store
            .add_pending_notification("bob", &conv_name, msg_id)
            .unwrap();

        // Get pending notifications
        let pending = store.get_pending_notifications("bob").unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].message_id, msg_id);

        // Clear notifications
        let cleared = store.clear_pending_notifications("bob").unwrap();
        assert_eq!(cleared, 1);

        // Verify cleared
        let pending = store.get_pending_notifications("bob").unwrap();
        assert!(pending.is_empty());
    }

    #[test]
    fn test_delete_conversation() {
        let store = test_store();

        let conv_name = store
            .create_conversation(Some("to-delete"), &["a", "b"])
            .unwrap();
        store.add_message(&conv_name, "a", "Hello", &[]).unwrap();

        // Delete
        store.delete_conversation(&conv_name).unwrap();

        // Verify gone
        assert!(store.get_conversation(&conv_name).unwrap().is_none());
        assert!(store.get_messages(&conv_name, None).unwrap().is_empty());
        assert!(store.get_participants(&conv_name).unwrap().is_empty());
    }

    #[test]
    fn test_add_participant() {
        let store = test_store();

        // Create a conversation with initial participants
        let conv_name = store.create_conversation(Some("chat"), &["user"]).unwrap();

        // Verify initial participant
        let participants = store.get_participants(&conv_name).unwrap();
        assert_eq!(participants.len(), 1);
        assert!(participants.iter().any(|p| p.agent == "user"));

        // Add a new participant
        store.add_participant(&conv_name, "arya").unwrap();

        // Verify new participant was added
        let participants = store.get_participants(&conv_name).unwrap();
        assert_eq!(participants.len(), 2);
        assert!(participants.iter().any(|p| p.agent == "arya"));

        // Adding the same participant again should be idempotent (no error)
        store.add_participant(&conv_name, "arya").unwrap();

        // Still only 2 participants
        let participants = store.get_participants(&conv_name).unwrap();
        assert_eq!(participants.len(), 2);
    }

    #[test]
    fn test_add_participant_nonexistent_conversation() {
        let store = test_store();

        let result = store.add_participant("nonexistent", "arya");
        assert!(matches!(result, Err(ConversationError::NotFound(_))));
    }

    #[test]
    fn test_generate_id_uniqueness() {
        let ids: Vec<String> = (0..100).map(|_| generate_id()).collect();
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(ids.len(), unique.len());
    }

    #[test]
    fn test_generate_fun_name_format() {
        // Generate multiple names and verify format
        for _ in 0..10 {
            let name = generate_fun_name();
            assert!(name.contains('-'), "Name should have hyphen: {}", name);
            let parts: Vec<&str> = name.split('-').collect();
            assert_eq!(parts.len(), 2, "Name should have exactly 2 parts: {}", name);
        }
    }

    #[test]
    fn test_find_by_name() {
        let store = test_store();

        // Create a conversation with a name
        let name = store
            .create_conversation(Some("project-x"), &["arya", "user"])
            .unwrap();

        // Find by name should work
        let found = store.find_by_name("project-x").unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, name);

        // Non-existent name should return None
        let not_found = store.find_by_name("nonexistent").unwrap();
        assert!(not_found.is_none());
    }

    #[test]
    fn test_find_by_participants() {
        let store = test_store();

        // Create a conversation
        let name = store
            .create_conversation(Some("test-conv"), &["arya", "user"])
            .unwrap();

        // Find by exact participants should work
        let found = store.find_by_participants(&["arya", "user"]).unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, name);

        // Different order should also work (normalized)
        let found = store.find_by_participants(&["user", "arya"]).unwrap();
        assert!(found.is_some());

        // Different participants should not match
        let not_found = store.find_by_participants(&["arya", "gendry"]).unwrap();
        assert!(not_found.is_none());
    }

    #[test]
    fn test_get_or_create_conversation_creates_new() {
        let store = test_store();

        let conv = store.get_or_create_conversation(&["arya", "user"]).unwrap();

        // Should have created a new conversation with a fun name
        assert!(
            conv.name.contains('-'),
            "Should have fun name: {}",
            conv.name
        );

        // Participants should be set
        let parts = store.get_participants(&conv.name).unwrap();
        assert_eq!(parts.len(), 2);
        assert!(parts.iter().any(|p| p.agent == "arya"));
        assert!(parts.iter().any(|p| p.agent == "user"));
    }

    #[test]
    fn test_get_or_create_conversation_returns_existing() {
        let store = test_store();

        // Create first
        let conv1 = store.get_or_create_conversation(&["arya", "user"]).unwrap();
        let created_at = conv1.created_at;

        // Get or create again - should return the same
        let conv2 = store.get_or_create_conversation(&["arya", "user"]).unwrap();

        assert_eq!(conv1.name, conv2.name);
        assert_eq!(conv2.created_at, created_at);
    }

    #[test]
    fn test_get_or_create_conversation_idempotent() {
        let store = test_store();

        // Call multiple times
        let conv1 = store
            .get_or_create_conversation(&["arya", "gendry", "user"])
            .unwrap();
        let conv2 = store
            .get_or_create_conversation(&["arya", "gendry", "user"])
            .unwrap();
        let conv3 = store
            .get_or_create_conversation(&["arya", "gendry", "user"])
            .unwrap();

        // All should have the same name
        assert_eq!(conv1.name, conv2.name);
        assert_eq!(conv2.name, conv3.name);

        // Should only be one conversation in the database
        let all = store.list_conversations().unwrap();
        assert_eq!(all.len(), 1);
    }

    #[test]
    fn test_parse_mentions_basic() {
        let mentions = parse_mentions("hey @arya what do you think?");
        assert_eq!(mentions, vec!["arya"]);
    }

    #[test]
    fn test_parse_mentions_multiple() {
        let mentions = parse_mentions("@arya @gendry let's discuss this");
        assert_eq!(mentions, vec!["arya", "gendry"]);
    }

    #[test]
    fn test_parse_mentions_with_hyphens_and_underscores() {
        let mentions = parse_mentions("@my-agent and @another_agent");
        assert_eq!(mentions, vec!["another_agent", "my-agent"]);
    }

    #[test]
    fn test_parse_mentions_duplicates() {
        let mentions = parse_mentions("@arya said @arya should @arya");
        assert_eq!(mentions, vec!["arya"]);
    }

    #[test]
    fn test_parse_mentions_no_mentions() {
        let mentions = parse_mentions("no mentions here");
        assert!(mentions.is_empty());
    }

    #[test]
    fn test_parse_mentions_excludes_user() {
        let mentions = parse_mentions("@user @arya @gendry");
        assert_eq!(mentions, vec!["arya", "gendry"]);
    }

    #[test]
    fn test_parse_mentions_in_sentence() {
        let mentions = parse_mentions("I think @arya is right, but @gendry has a point too.");
        assert_eq!(mentions, vec!["arya", "gendry"]);
    }

    #[test]
    fn test_parse_mentions_ignores_inline_code() {
        // Single backticks should be ignored
        let mentions = parse_mentions("Use `@arya` in your code");
        assert!(mentions.is_empty());

        // Real mention alongside inline code mention
        let mentions = parse_mentions("Hey @gendry, use `@arya` in your code");
        assert_eq!(mentions, vec!["gendry"]);
    }

    #[test]
    fn test_parse_mentions_ignores_code_blocks() {
        // Triple-backtick code blocks should be ignored
        let mentions = parse_mentions("Example:\n```\n@arya\n```");
        assert!(mentions.is_empty());

        // Code block with language specifier
        let mentions = parse_mentions("```rust\nlet agent = \"@arya\";\n```");
        assert!(mentions.is_empty());

        // Real mention before code block
        let mentions = parse_mentions("@gendry check this:\n```\n@arya\n```");
        assert_eq!(mentions, vec!["gendry"]);
    }

    #[test]
    fn test_parse_mentions_mixed_code_and_real() {
        // Mix of real mentions and code-wrapped mentions
        let content =
            "@arya look at `@fake` and also @gendry said:\n```\n@codeonly\n```\ncc @sansa";
        let mentions = parse_mentions(content);
        assert_eq!(mentions, vec!["arya", "gendry", "sansa"]);
    }

    #[test]
    fn test_expand_all_mention_basic() {
        let mentions = vec!["all".to_string()];
        let participants = vec!["arya".to_string(), "gendry".to_string(), "user".to_string()];
        let expanded = expand_all_mention(&mentions, &participants);
        assert_eq!(expanded, vec!["arya", "gendry"]);
    }

    #[test]
    fn test_expand_all_mention_with_other_mentions() {
        let mentions = vec!["all".to_string(), "sansa".to_string()];
        let participants = vec!["arya".to_string(), "gendry".to_string(), "user".to_string()];
        let expanded = expand_all_mention(&mentions, &participants);
        // Should include both participants and the extra mention
        assert_eq!(expanded, vec!["arya", "gendry", "sansa"]);
    }

    #[test]
    fn test_expand_all_mention_no_all() {
        let mentions = vec!["arya".to_string()];
        let participants = vec!["arya".to_string(), "gendry".to_string(), "user".to_string()];
        let expanded = expand_all_mention(&mentions, &participants);
        // Should return unchanged when no @all
        assert_eq!(expanded, vec!["arya"]);
    }

    #[test]
    fn test_expand_all_mention_excludes_user() {
        let mentions = vec!["all".to_string()];
        let participants = vec!["arya".to_string(), "user".to_string()];
        let expanded = expand_all_mention(&mentions, &participants);
        // Should not include "user"
        assert_eq!(expanded, vec!["arya"]);
    }

    #[test]
    fn test_expand_all_mention_deduplicates() {
        let mentions = vec!["all".to_string(), "arya".to_string()];
        let participants = vec!["arya".to_string(), "gendry".to_string(), "user".to_string()];
        let expanded = expand_all_mention(&mentions, &participants);
        // arya should only appear once
        assert_eq!(expanded, vec!["arya", "gendry"]);
    }

    #[test]
    fn test_expand_all_mention_empty_participants() {
        let mentions = vec!["all".to_string()];
        let participants: Vec<String> = vec!["user".to_string()];
        let expanded = expand_all_mention(&mentions, &participants);
        // Only user in participants, so nothing to expand to
        assert!(expanded.is_empty());
    }

    #[test]
    fn test_conversation_paused_default() {
        let store = test_store();

        let conv_name = store
            .create_conversation(Some("test-paused"), &["arya", "user"])
            .unwrap();
        let conv = store.get_conversation(&conv_name).unwrap().unwrap();

        // Conversations are not paused by default
        assert!(!conv.paused);
        assert!(!store.is_paused(&conv_name).unwrap());
    }

    #[test]
    fn test_set_paused() {
        let store = test_store();

        let conv_name = store
            .create_conversation(Some("pause-test"), &["arya", "user"])
            .unwrap();

        // Initially not paused
        assert!(!store.is_paused(&conv_name).unwrap());

        // Pause the conversation
        store.set_paused(&conv_name, true).unwrap();
        assert!(store.is_paused(&conv_name).unwrap());

        // Resume the conversation
        store.set_paused(&conv_name, false).unwrap();
        assert!(!store.is_paused(&conv_name).unwrap());
    }

    #[test]
    fn test_set_paused_nonexistent_conversation() {
        let store = test_store();

        let result = store.set_paused("nonexistent", true);
        assert!(matches!(result, Err(ConversationError::NotFound(_))));
    }

    #[test]
    fn test_is_paused_nonexistent_conversation() {
        let store = test_store();

        let result = store.is_paused("nonexistent");
        assert!(matches!(result, Err(ConversationError::NotFound(_))));
    }

    #[test]
    fn test_match_conversations_exact() {
        let store = test_store();

        store
            .create_conversation(Some("test-conv"), &["arya", "user"])
            .unwrap();
        store
            .create_conversation(Some("other-conv"), &["gendry", "user"])
            .unwrap();

        // Exact match (no wildcards)
        let matches = store.match_conversations("test-conv").unwrap();
        assert_eq!(matches, vec!["test-conv"]);

        // Exact match not found
        let matches = store.match_conversations("nonexistent").unwrap();
        assert!(matches.is_empty());
    }

    #[test]
    fn test_match_conversations_star_wildcard() {
        let store = test_store();

        store
            .create_conversation(Some("test-1"), &["arya", "user"])
            .unwrap();
        store
            .create_conversation(Some("test-2"), &["gendry", "user"])
            .unwrap();
        store
            .create_conversation(Some("other-conv"), &["sansa", "user"])
            .unwrap();

        // Match all starting with "test-"
        let matches = store.match_conversations("test-*").unwrap();
        assert_eq!(matches, vec!["test-1", "test-2"]);

        // Match all conversations
        let matches = store.match_conversations("*").unwrap();
        assert_eq!(matches.len(), 3);

        // Match ending pattern
        let matches = store.match_conversations("*-conv").unwrap();
        assert_eq!(matches, vec!["other-conv"]);
    }

    #[test]
    fn test_match_conversations_question_wildcard() {
        let store = test_store();

        store
            .create_conversation(Some("test-1"), &["arya", "user"])
            .unwrap();
        store
            .create_conversation(Some("test-2"), &["gendry", "user"])
            .unwrap();
        store
            .create_conversation(Some("test-10"), &["sansa", "user"])
            .unwrap();

        // ? matches single character
        let matches = store.match_conversations("test-?").unwrap();
        assert_eq!(matches, vec!["test-1", "test-2"]);
    }

    #[test]
    fn test_match_conversations_no_matches() {
        let store = test_store();

        store
            .create_conversation(Some("test-conv"), &["arya", "user"])
            .unwrap();

        let matches = store.match_conversations("other-*").unwrap();
        assert!(matches.is_empty());
    }

    #[test]
    fn test_match_conversations_special_chars() {
        let store = test_store();

        store
            .create_conversation(Some("test.conv"), &["arya", "user"])
            .unwrap();
        store
            .create_conversation(Some("testXconv"), &["gendry", "user"])
            .unwrap();

        // Dots should be escaped (not treated as regex "any char")
        let matches = store.match_conversations("test.conv").unwrap();
        assert_eq!(matches, vec!["test.conv"]);
    }

    #[test]
    fn test_get_pending_notifications_for_conversation() {
        let store = test_store();

        // Create two conversations
        let conv1 = store
            .create_conversation(Some("conv-1"), &["alice", "bob"])
            .unwrap();
        let conv2 = store
            .create_conversation(Some("conv-2"), &["charlie", "dave"])
            .unwrap();

        // Add messages
        let msg1 = store
            .add_message(&conv1, "alice", "Hey @bob", &["bob"])
            .unwrap();
        let msg2 = store
            .add_message(&conv2, "charlie", "Hey @dave", &["dave"])
            .unwrap();

        // Add pending notifications
        store.add_pending_notification("bob", &conv1, msg1).unwrap();
        store
            .add_pending_notification("dave", &conv2, msg2)
            .unwrap();

        // Get pending notifications for conv1 only
        let pending = store
            .get_pending_notifications_for_conversation(&conv1)
            .unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].agent, "bob");
        assert_eq!(pending[0].conv_name, conv1);

        // Get pending notifications for conv2 only
        let pending = store
            .get_pending_notifications_for_conversation(&conv2)
            .unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].agent, "dave");
        assert_eq!(pending[0].conv_name, conv2);
    }

    #[test]
    fn test_delete_pending_notification() {
        let store = test_store();

        let conv_name = store
            .create_conversation(Some("chat"), &["alice", "bob"])
            .unwrap();
        let msg_id = store
            .add_message(&conv_name, "alice", "Hey @bob", &["bob"])
            .unwrap();

        // Add pending notification
        let notif_id = store
            .add_pending_notification("bob", &conv_name, msg_id)
            .unwrap();

        // Verify it exists
        let pending = store.get_pending_notifications("bob").unwrap();
        assert_eq!(pending.len(), 1);

        // Delete the notification
        store.delete_pending_notification(notif_id).unwrap();

        // Verify it's gone
        let pending = store.get_pending_notifications("bob").unwrap();
        assert!(pending.is_empty());
    }

    #[test]
    fn test_delete_pending_notification_nonexistent() {
        let store = test_store();

        // Deleting a nonexistent notification should not error (just does nothing)
        let result = store.delete_pending_notification(999999);
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_messages_filtered_all() {
        let store = test_store();

        let conv_name = store
            .create_conversation(Some("filter-test"), &["alice", "bob"])
            .unwrap();
        store
            .add_message(&conv_name, "alice", "Message 1", &[])
            .unwrap();
        store
            .add_message(&conv_name, "bob", "Message 2", &[])
            .unwrap();
        store
            .add_message(&conv_name, "alice", "Message 3", &[])
            .unwrap();

        // Get all messages (no limit, no since)
        let messages = store.get_messages_filtered(&conv_name, None, None).unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].content, "Message 1");
        assert_eq!(messages[2].content, "Message 3");
    }

    #[test]
    fn test_get_messages_filtered_limit() {
        let store = test_store();

        let conv_name = store
            .create_conversation(Some("limit-test"), &["alice", "bob"])
            .unwrap();
        store
            .add_message(&conv_name, "alice", "Message 1", &[])
            .unwrap();
        store
            .add_message(&conv_name, "bob", "Message 2", &[])
            .unwrap();
        store
            .add_message(&conv_name, "alice", "Message 3", &[])
            .unwrap();
        store
            .add_message(&conv_name, "bob", "Message 4", &[])
            .unwrap();

        // Get last 2 messages
        let messages = store
            .get_messages_filtered(&conv_name, Some(2), None)
            .unwrap();
        assert_eq!(messages.len(), 2);
        // Should be in chronological order (oldest first among the last 2)
        assert_eq!(messages[0].content, "Message 3");
        assert_eq!(messages[1].content, "Message 4");
    }

    #[test]
    fn test_get_messages_filtered_since() {
        let store = test_store();

        let conv_name = store
            .create_conversation(Some("since-test"), &["alice", "bob"])
            .unwrap();
        let msg1 = store
            .add_message(&conv_name, "alice", "Message 1", &[])
            .unwrap();
        let msg2 = store
            .add_message(&conv_name, "bob", "Message 2", &[])
            .unwrap();
        store
            .add_message(&conv_name, "alice", "Message 3", &[])
            .unwrap();
        store
            .add_message(&conv_name, "bob", "Message 4", &[])
            .unwrap();

        // Get messages after msg2
        let messages = store
            .get_messages_filtered(&conv_name, None, Some(msg2))
            .unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, "Message 3");
        assert_eq!(messages[1].content, "Message 4");

        // Get messages after msg1
        let messages = store
            .get_messages_filtered(&conv_name, None, Some(msg1))
            .unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].content, "Message 2");
    }

    #[test]
    fn test_get_messages_filtered_since_and_limit() {
        let store = test_store();

        let conv_name = store
            .create_conversation(Some("since-limit-test"), &["alice", "bob"])
            .unwrap();
        let msg1 = store
            .add_message(&conv_name, "alice", "Message 1", &[])
            .unwrap();
        store
            .add_message(&conv_name, "bob", "Message 2", &[])
            .unwrap();
        store
            .add_message(&conv_name, "alice", "Message 3", &[])
            .unwrap();
        store
            .add_message(&conv_name, "bob", "Message 4", &[])
            .unwrap();
        store
            .add_message(&conv_name, "alice", "Message 5", &[])
            .unwrap();

        // Get 2 messages after msg1
        let messages = store
            .get_messages_filtered(&conv_name, Some(2), Some(msg1))
            .unwrap();
        assert_eq!(messages.len(), 2);
        // When using --since with --limit, we get the first N messages after since_id
        assert_eq!(messages[0].content, "Message 2");
        assert_eq!(messages[1].content, "Message 3");
    }

    #[test]
    fn test_get_messages_filtered_since_none_after() {
        let store = test_store();

        let conv_name = store
            .create_conversation(Some("since-none-test"), &["alice", "bob"])
            .unwrap();
        store
            .add_message(&conv_name, "alice", "Message 1", &[])
            .unwrap();
        let msg2 = store
            .add_message(&conv_name, "bob", "Message 2", &[])
            .unwrap();

        // Get messages after the last message - should be empty
        let messages = store
            .get_messages_filtered(&conv_name, None, Some(msg2))
            .unwrap();
        assert!(messages.is_empty());
    }

    #[test]
    fn test_add_tool_result() {
        let store = test_store();

        let conv_name = store
            .create_conversation(Some("tool-test"), &["arya", "gendry"])
            .unwrap();

        // Add a tool result attributed to arya
        let msg_id = store
            .add_tool_result(&conv_name, "[Tool Result for read_file]\ncontents", "arya")
            .unwrap();
        assert!(msg_id > 0);

        // Retrieve and verify
        let messages = store.get_messages(&conv_name, None).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].from_agent, "tool");
        assert_eq!(messages[0].triggered_by.as_deref(), Some("arya"));
        assert!(messages[0].content.contains("read_file"));
    }

    #[test]
    fn test_add_message_has_no_triggered_by() {
        let store = test_store();

        let conv_name = store
            .create_conversation(Some("no-trigger-test"), &["arya"])
            .unwrap();

        // Regular messages should have triggered_by = None
        store.add_message(&conv_name, "arya", "hello", &[]).unwrap();

        let messages = store.get_messages(&conv_name, None).unwrap();
        assert_eq!(messages.len(), 1);
        assert!(messages[0].triggered_by.is_none());
    }

    #[test]
    fn test_pin_message() {
        let store = test_store();
        let conv = store.create_conversation(Some("pin-test"), &["arya"]).unwrap();
        let id = store.add_message(&conv, "user", "important task", &[]).unwrap();

        store.pin_message(&conv, id, true).unwrap();

        let msgs = store.get_messages(&conv, None).unwrap();
        assert!(msgs[0].pinned);
    }

    #[test]
    fn test_unpin_message() {
        let store = test_store();
        let conv = store.create_conversation(Some("unpin-test"), &["arya"]).unwrap();
        let id = store.add_message(&conv, "user", "important task", &[]).unwrap();

        store.pin_message(&conv, id, true).unwrap();
        assert!(store.get_messages(&conv, None).unwrap()[0].pinned);

        store.pin_message(&conv, id, false).unwrap();
        assert!(!store.get_messages(&conv, None).unwrap()[0].pinned);
    }

    #[test]
    fn test_pin_nonexistent_message() {
        let store = test_store();
        let conv = store.create_conversation(Some("pin-missing"), &["arya"]).unwrap();

        let result = store.pin_message(&conv, 999, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_messages_with_pinned_no_pins() {
        let store = test_store();
        let conv = store.create_conversation(Some("no-pins"), &["arya"]).unwrap();

        for i in 0..5 {
            store.add_message(&conv, "user", &format!("msg {}", i), &[]).unwrap();
        }

        let normal = store.get_messages(&conv, Some(3)).unwrap();
        let with_pinned = store.get_messages_with_pinned(&conv, Some(3)).unwrap();
        assert_eq!(normal.len(), with_pinned.len());
        for (a, b) in normal.iter().zip(with_pinned.iter()) {
            assert_eq!(a.id, b.id);
        }
    }

    #[test]
    fn test_get_messages_with_pinned_in_window() {
        let store = test_store();
        let conv = store.create_conversation(Some("pin-in-window"), &["arya"]).unwrap();

        store.add_message(&conv, "user", "msg 1", &[]).unwrap();
        store.add_message(&conv, "user", "msg 2", &[]).unwrap();
        let id3 = store.add_message(&conv, "user", "msg 3", &[]).unwrap();

        // Pin a message that's within the window
        store.pin_message(&conv, id3, true).unwrap();

        let msgs = store.get_messages_with_pinned(&conv, Some(3)).unwrap();
        // Should be exactly 3, no duplication
        assert_eq!(msgs.len(), 3);
        // The pinned message should appear exactly once
        assert_eq!(msgs.iter().filter(|m| m.id == id3).count(), 1);
    }

    #[test]
    fn test_get_messages_with_pinned_outside_window() {
        let store = test_store();
        let conv = store.create_conversation(Some("pin-outside"), &["arya"]).unwrap();

        let id1 = store.add_message(&conv, "user", "important task", &[]).unwrap();
        store.add_message(&conv, "user", "msg 2", &[]).unwrap();
        store.add_message(&conv, "user", "msg 3", &[]).unwrap();
        store.add_message(&conv, "user", "msg 4", &[]).unwrap();
        store.add_message(&conv, "user", "msg 5", &[]).unwrap();

        // Pin the first message, which is outside a window of 3
        store.pin_message(&conv, id1, true).unwrap();

        let msgs = store.get_messages_with_pinned(&conv, Some(3)).unwrap();
        // Should be 4: the pinned message + the last 3
        assert_eq!(msgs.len(), 4);
        // Pinned message should come first
        assert_eq!(msgs[0].id, id1);
        assert!(msgs[0].pinned);
        // Last 3 should follow
        assert_eq!(msgs[1].content, "msg 3");
        assert_eq!(msgs[2].content, "msg 4");
        assert_eq!(msgs[3].content, "msg 5");
    }

    #[test]
    fn test_get_messages_with_pinned_multiple() {
        let store = test_store();
        let conv = store.create_conversation(Some("pin-multi"), &["arya"]).unwrap();

        let id1 = store.add_message(&conv, "user", "task A", &[]).unwrap();
        let id2 = store.add_message(&conv, "user", "task B", &[]).unwrap();
        store.add_message(&conv, "user", "msg 3", &[]).unwrap();
        store.add_message(&conv, "user", "msg 4", &[]).unwrap();
        store.add_message(&conv, "user", "msg 5", &[]).unwrap();
        store.add_message(&conv, "user", "msg 6", &[]).unwrap();

        // Pin first two messages (outside a window of 2)
        store.pin_message(&conv, id1, true).unwrap();
        store.pin_message(&conv, id2, true).unwrap();

        let msgs = store.get_messages_with_pinned(&conv, Some(2)).unwrap();
        // Should be 4: 2 pinned + 2 recent
        assert_eq!(msgs.len(), 4);
        // Pinned messages first, in chronological order
        assert_eq!(msgs[0].id, id1);
        assert_eq!(msgs[1].id, id2);
        // Then recent
        assert_eq!(msgs[2].content, "msg 5");
        assert_eq!(msgs[3].content, "msg 6");
    }

    #[test]
    fn test_get_messages_with_pinned_no_limit() {
        let store = test_store();
        let conv = store.create_conversation(Some("pin-no-limit"), &["arya"]).unwrap();

        let id1 = store.add_message(&conv, "user", "task", &[]).unwrap();
        store.add_message(&conv, "user", "msg 2", &[]).unwrap();
        store.add_message(&conv, "user", "msg 3", &[]).unwrap();

        store.pin_message(&conv, id1, true).unwrap();

        // No limit — should return all messages normally
        let msgs = store.get_messages_with_pinned(&conv, None).unwrap();
        assert_eq!(msgs.len(), 3);
    }

    #[test]
    fn test_context_cursor_default_null() {
        let store = test_store();
        let conv = store
            .create_conversation(Some("cursor-test"), &["arya"])
            .unwrap();
        let cursor = store.get_context_cursor(&conv, "arya").unwrap();
        assert!(cursor.is_none());
    }

    #[test]
    fn test_set_and_get_context_cursor() {
        let store = test_store();
        let conv = store
            .create_conversation(Some("cursor-test"), &["arya"])
            .unwrap();

        store.set_context_cursor(&conv, "arya", 42).unwrap();
        let cursor = store.get_context_cursor(&conv, "arya").unwrap();
        assert_eq!(cursor, Some(42));

        // Update to a new value
        store.set_context_cursor(&conv, "arya", 99).unwrap();
        let cursor = store.get_context_cursor(&conv, "arya").unwrap();
        assert_eq!(cursor, Some(99));
    }

    #[test]
    fn test_clear_context_cursor() {
        let store = test_store();
        let conv = store
            .create_conversation(Some("cursor-test"), &["arya"])
            .unwrap();

        store.set_context_cursor(&conv, "arya", 42).unwrap();
        assert_eq!(store.get_context_cursor(&conv, "arya").unwrap(), Some(42));

        store.clear_context_cursor(&conv, "arya").unwrap();
        assert!(store.get_context_cursor(&conv, "arya").unwrap().is_none());
    }

    #[test]
    fn test_get_messages_from_with_pinned() {
        let store = test_store();
        let conv = store
            .create_conversation(Some("from-test"), &["arya"])
            .unwrap();

        let id1 = store.add_message(&conv, "user", "msg 1", &[]).unwrap();
        let _id2 = store.add_message(&conv, "arya", "msg 2", &[]).unwrap();
        let _id3 = store.add_message(&conv, "user", "msg 3", &[]).unwrap();

        // Get messages from id1 (inclusive) — should return all 3
        let msgs = store.get_messages_from_with_pinned(&conv, id1).unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0].content, "msg 1");
        assert_eq!(msgs[1].content, "msg 2");
        assert_eq!(msgs[2].content, "msg 3");
    }

    #[test]
    fn test_get_messages_from_with_pinned_includes_pinned() {
        let store = test_store();
        let conv = store
            .create_conversation(Some("from-pin"), &["arya"])
            .unwrap();

        // Pin an early message
        let id1 = store.add_message(&conv, "user", "pinned task", &[]).unwrap();
        store.pin_message(&conv, id1, true).unwrap();

        let _id2 = store.add_message(&conv, "arya", "response", &[]).unwrap();
        let id3 = store.add_message(&conv, "user", "follow up", &[]).unwrap();

        // Get messages from id3 (inclusive) — should include pinned msg (id1) + msg 3
        let msgs = store.get_messages_from_with_pinned(&conv, id3).unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].content, "pinned task");
        assert!(msgs[0].pinned);
        assert_eq!(msgs[1].content, "follow up");
    }

    #[test]
    fn test_get_messages_from_with_pinned_deduplicates() {
        let store = test_store();
        let conv = store
            .create_conversation(Some("from-dedup"), &["arya"])
            .unwrap();

        let _id1 = store.add_message(&conv, "user", "msg 1", &[]).unwrap();
        let id2 = store.add_message(&conv, "user", "msg 2", &[]).unwrap();

        // Pin msg 2 — it's both pinned AND in from-range, should not be doubled
        store.pin_message(&conv, id2, true).unwrap();

        let msgs = store.get_messages_from_with_pinned(&conv, id2).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content, "msg 2");
    }

    #[test]
    fn test_get_messages_from_self_only() {
        let store = test_store();
        let conv = store
            .create_conversation(Some("from-self"), &["arya"])
            .unwrap();

        let id1 = store.add_message(&conv, "user", "only msg", &[]).unwrap();

        // From id1 inclusive — should get the message itself
        let msgs = store.get_messages_from_with_pinned(&conv, id1).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content, "only msg");
    }

    #[test]
    fn test_count_messages_from() {
        let store = test_store();
        let conv = store
            .create_conversation(Some("count-test"), &["arya"])
            .unwrap();

        let id1 = store.add_message(&conv, "user", "msg 1", &[]).unwrap();
        store.add_message(&conv, "arya", "msg 2", &[]).unwrap();
        store.add_message(&conv, "user", "msg 3", &[]).unwrap();

        // Inclusive: id1 itself + 2 after = 3
        assert_eq!(store.count_messages_from(&conv, id1).unwrap(), 3);
        assert_eq!(store.count_messages_from(&conv, 999).unwrap(), 0);
    }

    #[test]
    fn test_store_message_embedding() {
        let store = test_store();
        let conv = store.create_conversation(Some("emb-test"), &["arya"]).unwrap();
        let msg_id = store.add_message(&conv, "user", "hello world", &[]).unwrap();

        let embedding = vec![0.1, 0.2, 0.3, 0.4];
        store.store_message_embedding(msg_id, &conv, &embedding).unwrap();

        // Verify it was stored by searching
        let query_emb = vec![0.1, 0.2, 0.3, 0.4];
        let results = store.search_similar_messages(&conv, &query_emb, &[], 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, msg_id);
        assert_eq!(results[0].1, "user");
        assert_eq!(results[0].2, "hello world");
    }

    #[test]
    fn test_search_similar_messages_excludes_ids() {
        let store = test_store();
        let conv = store.create_conversation(Some("excl-test"), &["arya"]).unwrap();

        let emb = vec![0.9, 0.1, 0.0, 0.0];
        let id1 = store.add_message(&conv, "user", "first question", &[]).unwrap();
        store.store_message_embedding(id1, &conv, &emb).unwrap();

        let id2 = store.add_message(&conv, "user", "second question", &[]).unwrap();
        store.store_message_embedding(id2, &conv, &emb).unwrap();

        // Exclude id1 — should only return id2
        let results = store.search_similar_messages(&conv, &emb, &[id1], 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id2);
    }

    #[test]
    fn test_search_similar_messages_limit() {
        let store = test_store();
        let conv = store.create_conversation(Some("limit-emb"), &["arya"]).unwrap();

        let emb = vec![0.9, 0.1, 0.0, 0.0];
        for i in 0..5 {
            let id = store.add_message(&conv, "user", &format!("msg {}", i), &[]).unwrap();
            store.store_message_embedding(id, &conv, &emb).unwrap();
        }

        let results = store.search_similar_messages(&conv, &emb, &[], 2).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_similar_messages_empty() {
        let store = test_store();
        let conv = store.create_conversation(Some("empty-emb"), &["arya"]).unwrap();

        let query_emb = vec![0.1, 0.2, 0.3, 0.4];
        let results = store.search_similar_messages(&conv, &query_emb, &[], 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_similar_messages_includes_agent_messages() {
        let store = test_store();
        let conv = store.create_conversation(Some("agent-msg"), &["arya"]).unwrap();

        let emb = vec![0.9, 0.1, 0.0, 0.0];

        // Add a user message with embedding
        let uid = store.add_message(&conv, "user", "user question", &[]).unwrap();
        store.store_message_embedding(uid, &conv, &emb).unwrap();

        // Add an agent message with embedding (should now be included)
        let aid = store.add_message(&conv, "arya", "agent response", &[]).unwrap();
        store.store_message_embedding(aid, &conv, &emb).unwrap();

        let results = store.search_similar_messages(&conv, &emb, &[], 10).unwrap();
        assert_eq!(results.len(), 2);
        // Verify from_agent is returned
        let agents: Vec<&str> = results.iter().map(|r| r.1.as_str()).collect();
        assert!(agents.contains(&"user"));
        assert!(agents.contains(&"arya"));
    }

    #[test]
    fn test_search_similar_messages_excludes_tool_and_recall() {
        let store = test_store();
        let conv = store.create_conversation(Some("filter-test"), &["arya"]).unwrap();

        let emb = vec![0.9, 0.1, 0.0, 0.0];

        // User message — should be included
        let uid = store.add_message(&conv, "user", "a question", &[]).unwrap();
        store.store_message_embedding(uid, &conv, &emb).unwrap();

        // Tool result — should be excluded
        let tid = store.add_message(&conv, "tool", "tool output", &[]).unwrap();
        store.store_message_embedding(tid, &conv, &emb).unwrap();

        // Recall injection — should be excluded
        let rid = store.add_message(&conv, "recall", "recall content", &[]).unwrap();
        store.store_message_embedding(rid, &conv, &emb).unwrap();

        let results = store.search_similar_messages(&conv, &emb, &[], 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, "user");
    }

    #[test]
    fn test_search_similar_messages_excludes_tool_call_intermediaries() {
        let store = test_store();
        let conv = store.create_conversation(Some("tc-test"), &["arya"]).unwrap();

        let emb = vec![0.9, 0.1, 0.0, 0.0];

        // Normal agent response — should be included
        let aid = store.add_message(&conv, "arya", "normal response", &[]).unwrap();
        store.store_message_embedding(aid, &conv, &emb).unwrap();

        // Agent message with tool_calls — should be excluded
        let tcid = store.add_message_with_tool_calls(
            &conv, "arya", "calling tool", &[], None, Some(r#"[{"tool":"shell"}]"#),
        ).unwrap();
        store.store_message_embedding(tcid, &conv, &emb).unwrap();

        let results = store.search_similar_messages(&conv, &emb, &[], 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].2, "normal response");
    }

    #[test]
    fn test_store_message_embedding_replace() {
        let store = test_store();
        let conv = store.create_conversation(Some("replace-emb"), &["arya"]).unwrap();
        let msg_id = store.add_message(&conv, "user", "hello", &[]).unwrap();

        let emb1 = vec![0.1, 0.2, 0.3, 0.4];
        store.store_message_embedding(msg_id, &conv, &emb1).unwrap();

        // Replace with new embedding — should not error
        let emb2 = vec![0.9, 0.8, 0.7, 0.6];
        store.store_message_embedding(msg_id, &conv, &emb2).unwrap();

        // Should still find exactly one result
        let results = store.search_similar_messages(&conv, &emb2, &[], 10).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_search_messages_basic() {
        let store = test_store();
        let conv = store.create_conversation(Some("search-basic"), &["arya"]).unwrap();
        store.add_message(&conv, "arya", "hello world", &[]).unwrap();
        store.add_message(&conv, "arya", "goodbye world", &[]).unwrap();
        store.add_message(&conv, "arya", "something else", &[]).unwrap();

        let results = store.search_messages(&conv, "world", None, false, 50, None).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].content.contains("hello"));
        assert!(results[1].content.contains("goodbye"));
    }

    #[test]
    fn test_search_messages_case_insensitive() {
        let store = test_store();
        let conv = store.create_conversation(Some("search-case"), &["arya"]).unwrap();
        store.add_message(&conv, "arya", "Hello World", &[]).unwrap();
        store.add_message(&conv, "arya", "hello world", &[]).unwrap();

        // SQLite LIKE is case-insensitive for ASCII by default
        let results = store.search_messages(&conv, "hello", None, false, 50, None).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_messages_from_agent_filter() {
        let store = test_store();
        let conv = store.create_conversation(Some("search-from"), &["arya", "bob"]).unwrap();
        store.add_message(&conv, "arya", "hello from arya", &[]).unwrap();
        store.add_message(&conv, "bob", "hello from bob", &[]).unwrap();

        let results = store.search_messages(&conv, "hello", Some("arya"), false, 50, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].from_agent, "arya");
    }

    #[test]
    fn test_search_messages_excludes_tool_and_recall() {
        let store = test_store();
        let conv = store.create_conversation(Some("search-excl"), &["arya"]).unwrap();
        store.add_message(&conv, "arya", "hello agent", &[]).unwrap();
        store.add_message(&conv, "tool", "hello tool result", &[]).unwrap();
        store.add_message(&conv, "recall", "hello recall data", &[]).unwrap();

        // Default: excludes tool/recall
        let results = store.search_messages(&conv, "hello", None, false, 50, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].from_agent, "arya");

        // include_internal: includes them
        let results = store.search_messages(&conv, "hello", None, true, 50, None).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_search_messages_from_agent_overrides_exclusion() {
        let store = test_store();
        let conv = store.create_conversation(Some("search-tool-from"), &["arya"]).unwrap();
        store.add_message(&conv, "tool", "hello from tool", &[]).unwrap();

        // Explicit from_agent="tool" should find it even with include_internal=false
        let results = store.search_messages(&conv, "hello", Some("tool"), false, 50, None).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_search_messages_content_truncation() {
        let store = test_store();
        let conv = store.create_conversation(Some("search-trunc"), &["arya"]).unwrap();
        let long_content = "a".repeat(1000);
        store.add_message(&conv, "arya", &long_content, &[]).unwrap();

        let results = store.search_messages(&conv, "aaa", None, false, 50, Some(100)).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.len() <= 100);
        assert!(results[0].content.ends_with("..."));
    }

    #[test]
    fn test_search_messages_limit() {
        let store = test_store();
        let conv = store.create_conversation(Some("search-limit"), &["arya"]).unwrap();
        for i in 0..10 {
            store.add_message(&conv, "arya", &format!("msg {}", i), &[]).unwrap();
        }

        let results = store.search_messages(&conv, "msg", None, false, 3, None).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_search_messages_no_results() {
        let store = test_store();
        let conv = store.create_conversation(Some("search-none"), &["arya"]).unwrap();
        store.add_message(&conv, "arya", "hello world", &[]).unwrap();

        let results = store.search_messages(&conv, "xyz_not_found", None, false, 50, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_messages_chronological_order() {
        let store = test_store();
        let conv = store.create_conversation(Some("search-order"), &["arya"]).unwrap();
        store.add_message(&conv, "arya", "first match", &[]).unwrap();
        store.add_message(&conv, "arya", "second match", &[]).unwrap();
        store.add_message(&conv, "arya", "third match", &[]).unwrap();

        let results = store.search_messages(&conv, "match", None, false, 50, None).unwrap();
        assert_eq!(results.len(), 3);
        assert!(results[0].created_at <= results[1].created_at);
        assert!(results[1].created_at <= results[2].created_at);
    }
}