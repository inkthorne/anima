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
}

/// A conversation between agents.
#[derive(Debug, Clone)]
pub struct Conversation {
    pub id: String,
    pub name: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

/// A participant in a conversation.
#[derive(Debug, Clone)]
pub struct Participant {
    pub conv_id: String,
    pub agent: String,
    pub joined_at: i64,
}

/// A message in a conversation.
#[derive(Debug, Clone)]
pub struct ConversationMessage {
    pub id: i64,
    pub conv_id: String,
    pub from_agent: String,
    pub content: String,
    pub mentions: Vec<String>,
    pub created_at: i64,
    pub expires_at: i64,
}

/// A pending notification for an offline agent.
#[derive(Debug, Clone)]
pub struct PendingNotification {
    pub id: i64,
    pub agent: String,
    pub conv_id: String,
    pub message_id: i64,
    pub created_at: i64,
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
        let home = dirs::home_dir()
            .ok_or_else(|| ConversationError::Io(
                std::io::Error::new(std::io::ErrorKind::NotFound, "Could not determine home directory")
            ))?;
        Ok(home.join(".anima").join("conversations.db"))
    }

    /// Create the database schema.
    fn create_schema(&self) -> Result<(), ConversationError> {
        self.conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                name TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS participants (
                conv_id TEXT NOT NULL,
                agent TEXT NOT NULL,
                joined_at INTEGER NOT NULL,
                PRIMARY KEY (conv_id, agent),
                FOREIGN KEY (conv_id) REFERENCES conversations(id)
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conv_id TEXT NOT NULL,
                from_agent TEXT NOT NULL,
                content TEXT NOT NULL,
                mentions TEXT,
                created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL,
                FOREIGN KEY (conv_id) REFERENCES conversations(id)
            );

            CREATE TABLE IF NOT EXISTS pending_notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent TEXT NOT NULL,
                conv_id TEXT NOT NULL,
                message_id INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (conv_id) REFERENCES conversations(id),
                FOREIGN KEY (message_id) REFERENCES messages(id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conv_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_pending_agent ON pending_notifications(agent);
            "#,
        )?;
        Ok(())
    }

    /// Create a new conversation with the given participants.
    /// Returns the conversation ID.
    pub fn create_conversation(
        &self,
        name: Option<&str>,
        participants: &[&str],
    ) -> Result<String, ConversationError> {
        let id = generate_id();
        let now = current_timestamp();

        self.conn.execute(
            "INSERT INTO conversations (id, name, created_at, updated_at) VALUES (?1, ?2, ?3, ?4)",
            params![id, name, now, now],
        )?;

        // Add participants
        for agent in participants {
            self.conn.execute(
                "INSERT INTO participants (conv_id, agent, joined_at) VALUES (?1, ?2, ?3)",
                params![id, agent, now],
            )?;
        }

        Ok(id)
    }

    /// List all conversations.
    pub fn list_conversations(&self) -> Result<Vec<Conversation>, ConversationError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, created_at, updated_at FROM conversations ORDER BY updated_at DESC",
        )?;

        let conversations = stmt
            .query_map([], |row| {
                Ok(Conversation {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    created_at: row.get(2)?,
                    updated_at: row.get(3)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(conversations)
    }

    /// Get a conversation by ID.
    pub fn get_conversation(&self, id: &str) -> Result<Option<Conversation>, ConversationError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, created_at, updated_at FROM conversations WHERE id = ?1",
        )?;

        let mut rows = stmt.query(params![id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(Conversation {
                id: row.get(0)?,
                name: row.get(1)?,
                created_at: row.get(2)?,
                updated_at: row.get(3)?,
            }))
        } else {
            Ok(None)
        }
    }

    /// Get participants for a conversation.
    pub fn get_participants(&self, conv_id: &str) -> Result<Vec<Participant>, ConversationError> {
        let mut stmt = self.conn.prepare(
            "SELECT conv_id, agent, joined_at FROM participants WHERE conv_id = ?1 ORDER BY joined_at",
        )?;

        let participants = stmt
            .query_map(params![conv_id], |row| {
                Ok(Participant {
                    conv_id: row.get(0)?,
                    agent: row.get(1)?,
                    joined_at: row.get(2)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(participants)
    }

    /// Add a message to a conversation.
    /// Returns the message ID.
    pub fn add_message(
        &self,
        conv_id: &str,
        from_agent: &str,
        content: &str,
        mentions: &[&str],
    ) -> Result<i64, ConversationError> {
        // Verify conversation exists
        if self.get_conversation(conv_id)?.is_none() {
            return Err(ConversationError::NotFound(conv_id.to_string()));
        }

        let now = current_timestamp();
        let expires_at = now + DEFAULT_TTL_SECONDS;
        let mentions_json = serde_json::to_string(mentions).unwrap_or_else(|_| "[]".to_string());

        self.conn.execute(
            "INSERT INTO messages (conv_id, from_agent, content, mentions, created_at, expires_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![conv_id, from_agent, content, mentions_json, now, expires_at],
        )?;

        let message_id = self.conn.last_insert_rowid();

        // Update conversation's updated_at
        self.conn.execute(
            "UPDATE conversations SET updated_at = ?1 WHERE id = ?2",
            params![now, conv_id],
        )?;

        Ok(message_id)
    }

    /// Get messages for a conversation, optionally limited to the last N messages.
    /// Messages are returned in chronological order (oldest first).
    pub fn get_messages(
        &self,
        conv_id: &str,
        limit: Option<usize>,
    ) -> Result<Vec<ConversationMessage>, ConversationError> {
        let now = current_timestamp();

        let query = match limit {
            Some(n) => format!(
                "SELECT id, conv_id, from_agent, content, mentions, created_at, expires_at
                 FROM messages
                 WHERE conv_id = ?1 AND expires_at > ?2
                 ORDER BY created_at DESC
                 LIMIT {}",
                n
            ),
            None => "SELECT id, conv_id, from_agent, content, mentions, created_at, expires_at
                     FROM messages
                     WHERE conv_id = ?1 AND expires_at > ?2
                     ORDER BY created_at ASC".to_string(),
        };

        let mut stmt = self.conn.prepare(&query)?;
        let mut messages: Vec<ConversationMessage> = stmt
            .query_map(params![conv_id, now], |row| {
                let mentions_json: String = row.get(4)?;
                let mentions: Vec<String> =
                    serde_json::from_str(&mentions_json).unwrap_or_default();
                Ok(ConversationMessage {
                    id: row.get(0)?,
                    conv_id: row.get(1)?,
                    from_agent: row.get(2)?,
                    content: row.get(3)?,
                    mentions,
                    created_at: row.get(5)?,
                    expires_at: row.get(6)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        // If we used LIMIT with DESC, reverse to get chronological order
        if limit.is_some() {
            messages.reverse();
        }

        Ok(messages)
    }

    /// Add a pending notification for an offline agent.
    pub fn add_pending_notification(
        &self,
        agent: &str,
        conv_id: &str,
        message_id: i64,
    ) -> Result<i64, ConversationError> {
        let now = current_timestamp();

        self.conn.execute(
            "INSERT INTO pending_notifications (agent, conv_id, message_id, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![agent, conv_id, message_id, now],
        )?;

        Ok(self.conn.last_insert_rowid())
    }

    /// Get pending notifications for an agent.
    pub fn get_pending_notifications(
        &self,
        agent: &str,
    ) -> Result<Vec<PendingNotification>, ConversationError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, agent, conv_id, message_id, created_at
             FROM pending_notifications
             WHERE agent = ?1
             ORDER BY created_at",
        )?;

        let notifications = stmt
            .query_map(params![agent], |row| {
                Ok(PendingNotification {
                    id: row.get(0)?,
                    agent: row.get(1)?,
                    conv_id: row.get(2)?,
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

    /// Delete expired messages and clean up empty conversations.
    pub fn cleanup_expired(&self) -> Result<(usize, usize), ConversationError> {
        let now = current_timestamp();

        // Delete expired messages
        let messages_deleted = self.conn.execute(
            "DELETE FROM messages WHERE expires_at <= ?1",
            params![now],
        )?;

        // Delete pending notifications for deleted messages
        self.conn.execute(
            "DELETE FROM pending_notifications WHERE message_id NOT IN (SELECT id FROM messages)",
            [],
        )?;

        // Delete conversations with no messages
        let convs_deleted = self.conn.execute(
            "DELETE FROM conversations WHERE id NOT IN (SELECT DISTINCT conv_id FROM messages)",
            [],
        )?;

        // Delete orphaned participants
        self.conn.execute(
            "DELETE FROM participants WHERE conv_id NOT IN (SELECT id FROM conversations)",
            [],
        )?;

        Ok((messages_deleted, convs_deleted))
    }

    /// Delete a conversation and all its messages.
    pub fn delete_conversation(&self, conv_id: &str) -> Result<(), ConversationError> {
        // Delete in order: pending_notifications, messages, participants, conversation
        self.conn.execute(
            "DELETE FROM pending_notifications WHERE conv_id = ?1",
            params![conv_id],
        )?;
        self.conn.execute(
            "DELETE FROM messages WHERE conv_id = ?1",
            params![conv_id],
        )?;
        self.conn.execute(
            "DELETE FROM participants WHERE conv_id = ?1",
            params![conv_id],
        )?;
        self.conn.execute(
            "DELETE FROM conversations WHERE id = ?1",
            params![conv_id],
        )?;

        Ok(())
    }

    /// Find a conversation by its name.
    pub fn find_by_name(&self, name: &str) -> Result<Option<Conversation>, ConversationError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, created_at, updated_at FROM conversations WHERE name = ?1",
        )?;

        let mut rows = stmt.query(params![name])?;
        if let Some(row) = rows.next()? {
            Ok(Some(Conversation {
                id: row.get(0)?,
                name: row.get(1)?,
                created_at: row.get(2)?,
                updated_at: row.get(3)?,
            }))
        } else {
            Ok(None)
        }
    }

    /// Get or create a conversation with the given canonical name and participants.
    /// If a conversation with this name exists, returns it (participants are not updated).
    /// Otherwise, creates a new conversation with the given name and participants.
    pub fn get_or_create_conversation(
        &self,
        name: &str,
        participants: &[&str],
    ) -> Result<Conversation, ConversationError> {
        // Try to find existing conversation by name
        if let Some(conv) = self.find_by_name(name)? {
            return Ok(conv);
        }

        // Create new conversation with the canonical name as both id and name
        let now = current_timestamp();

        self.conn.execute(
            "INSERT INTO conversations (id, name, created_at, updated_at) VALUES (?1, ?2, ?3, ?4)",
            params![name, name, now, now],
        )?;

        // Add participants
        for agent in participants {
            self.conn.execute(
                "INSERT INTO participants (conv_id, agent, joined_at) VALUES (?1, ?2, ?3)",
                params![name, agent, now],
            )?;
        }

        Ok(Conversation {
            id: name.to_string(),
            name: Some(name.to_string()),
            created_at: now,
            updated_at: now,
        })
    }
}

/// Generate a canonical conversation ID for a 1:1 chat with an agent.
/// Format: "1:1:{agent}:user"
pub fn canonical_1to1_id(agent: &str) -> String {
    format!("1:1:{}:user", agent)
}

/// Generate a canonical conversation ID for a group chat.
/// Format: "group:{sorted_participants}" where participants are sorted alphabetically.
/// Always includes "user" as a participant.
pub fn canonical_group_id(agents: &[&str]) -> String {
    let mut participants: Vec<&str> = agents.iter().copied().collect();
    if !participants.contains(&"user") {
        participants.push("user");
    }
    participants.sort();
    format!("group:{}", participants.join(":"))
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
pub fn parse_mentions(content: &str) -> Vec<String> {
    use regex::Regex;

    let re = Regex::new(r"@([a-zA-Z0-9_-]+)").unwrap();
    let mut mentions: Vec<String> = re
        .captures_iter(content)
        .map(|cap| cap[1].to_string())
        .filter(|name| name != "user") // Don't notify "user"
        .collect();

    // Deduplicate
    mentions.sort();
    mentions.dedup();
    mentions
}

/// Result of attempting to notify an agent.
#[derive(Debug)]
pub enum NotifyResult {
    /// Agent was notified via socket (running daemon)
    Notified { response_message_id: i64 },
    /// Agent was not running; notification queued
    Queued { notification_id: i64 },
    /// Notification failed
    Failed { reason: String },
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
    use crate::socket_api::{Request, Response, SocketApi};
    use tokio::net::UnixStream;

    let mut results = std::collections::HashMap::new();

    for agent_name in mentions {
        // Check if agent is running
        if let Some(running_agent) = discovery::get_running_agent(agent_name) {
            // Try to connect and send Notify request
            match UnixStream::connect(&running_agent.socket_path).await {
                Ok(stream) => {
                    let mut api = SocketApi::new(stream);

                    // Send Notify request
                    let request = Request::Notify {
                        conv_id: conv_id.to_string(),
                        message_id,
                    };

                    if let Err(e) = api.write_request(&request).await {
                        results.insert(agent_name.clone(), NotifyResult::Failed {
                            reason: format!("Failed to send request: {}", e),
                        });
                        continue;
                    }

                    // Read response
                    match api.read_response().await {
                        Ok(Some(Response::Notified { response_message_id })) => {
                            results.insert(agent_name.clone(), NotifyResult::Notified {
                                response_message_id,
                            });
                        }
                        Ok(Some(Response::Error { message })) => {
                            results.insert(agent_name.clone(), NotifyResult::Failed {
                                reason: message,
                            });
                        }
                        Ok(Some(_)) => {
                            results.insert(agent_name.clone(), NotifyResult::Failed {
                                reason: "Unexpected response type".to_string(),
                            });
                        }
                        Ok(None) => {
                            results.insert(agent_name.clone(), NotifyResult::Failed {
                                reason: "Connection closed".to_string(),
                            });
                        }
                        Err(e) => {
                            results.insert(agent_name.clone(), NotifyResult::Failed {
                                reason: format!("Failed to read response: {}", e),
                            });
                        }
                    }
                }
                Err(_e) => {
                    // Socket connection failed - agent may have just stopped
                    // Queue the notification instead
                    match store.add_pending_notification(agent_name, conv_id, message_id) {
                        Ok(notification_id) => {
                            results.insert(agent_name.clone(), NotifyResult::Queued {
                                notification_id,
                            });
                        }
                        Err(e) => {
                            results.insert(agent_name.clone(), NotifyResult::Failed {
                                reason: format!("Failed to queue notification: {}", e),
                            });
                        }
                    }
                }
            }
        } else {
            // Agent is not running - queue notification
            match store.add_pending_notification(agent_name, conv_id, message_id) {
                Ok(notification_id) => {
                    results.insert(agent_name.clone(), NotifyResult::Queued {
                        notification_id,
                    });
                }
                Err(e) => {
                    results.insert(agent_name.clone(), NotifyResult::Failed {
                        reason: format!("Failed to queue notification: {}", e),
                    });
                }
            }
        }
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
    use crate::socket_api::{Request, Response, SocketApi};
    use tokio::net::UnixStream;

    // Check if agent is running
    if let Some(running_agent) = discovery::get_running_agent(&agent_name) {
        // Try to connect and send Notify request
        match UnixStream::connect(&running_agent.socket_path).await {
            Ok(stream) => {
                let mut api = SocketApi::new(stream);

                // Send Notify request
                let request = Request::Notify {
                    conv_id: conv_id.clone(),
                    message_id,
                };

                if let Err(e) = api.write_request(&request).await {
                    return (agent_name, NotifyResult::Failed {
                        reason: format!("Failed to send request: {}", e),
                    });
                }

                // Read response
                match api.read_response().await {
                    Ok(Some(Response::Notified { response_message_id })) => {
                        (agent_name, NotifyResult::Notified { response_message_id })
                    }
                    Ok(Some(Response::Error { message })) => {
                        (agent_name, NotifyResult::Failed { reason: message })
                    }
                    Ok(Some(_)) => {
                        (agent_name, NotifyResult::Failed {
                            reason: "Unexpected response type".to_string(),
                        })
                    }
                    Ok(None) => {
                        (agent_name, NotifyResult::Failed {
                            reason: "Connection closed".to_string(),
                        })
                    }
                    Err(e) => {
                        (agent_name, NotifyResult::Failed {
                            reason: format!("Failed to read response: {}", e),
                        })
                    }
                }
            }
            Err(_e) => {
                // Socket connection failed - agent may have just stopped
                // For parallel mode, we can't queue (no store reference)
                // Return failed status - caller can decide to queue
                (agent_name, NotifyResult::Failed {
                    reason: "Agent not reachable (socket connection failed)".to_string(),
                })
            }
        }
    } else {
        // Agent is not running
        (agent_name, NotifyResult::Failed {
            reason: "Agent not running".to_string(),
        })
    }
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
    store: &ConversationStore,
    conv_id: &str,
    message_id: i64,
    mentions: &[String],
) -> std::collections::HashMap<String, NotifyResult> {
    use futures_util::future::join_all;

    let mut results = std::collections::HashMap::new();

    if mentions.is_empty() {
        return results;
    }

    // Spawn parallel notification tasks
    let futures: Vec<_> = mentions.iter().map(|agent_name| {
        let agent = agent_name.clone();
        let cid = conv_id.to_string();
        let mid = message_id;
        tokio::spawn(async move {
            notify_single_agent(agent, cid, mid).await
        })
    }).collect();

    // Wait for all notifications to complete
    let outcomes = join_all(futures).await;

    // Process results and queue failed notifications for offline agents
    for outcome in outcomes {
        match outcome {
            Ok((agent_name, result)) => {
                // If agent wasn't reachable or not running, queue the notification
                let final_result = match &result {
                    NotifyResult::Failed { reason } if reason.contains("not running") || reason.contains("not reachable") => {
                        // Queue the notification for when agent comes online
                        match store.add_pending_notification(&agent_name, conv_id, message_id) {
                            Ok(notification_id) => NotifyResult::Queued { notification_id },
                            Err(e) => NotifyResult::Failed {
                                reason: format!("Failed to queue notification: {}", e),
                            },
                        }
                    }
                    _ => result,
                };
                results.insert(agent_name, final_result);
            }
            Err(e) => {
                // Task panicked - shouldn't happen but handle gracefully
                eprintln!("Notification task panicked: {}", e);
            }
        }
    }

    results
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

        let id = store
            .create_conversation(Some("test-conv"), &["arya", "gendry"])
            .unwrap();

        assert!(!id.is_empty());

        // Verify conversation exists
        let conv = store.get_conversation(&id).unwrap().unwrap();
        assert_eq!(conv.name, Some("test-conv".to_string()));

        // Verify participants
        let participants = store.get_participants(&id).unwrap();
        assert_eq!(participants.len(), 2);
        assert!(participants.iter().any(|p| p.agent == "arya"));
        assert!(participants.iter().any(|p| p.agent == "gendry"));
    }

    #[test]
    fn test_create_conversation_no_name() {
        let store = test_store();

        let id = store.create_conversation(None, &["user", "bot"]).unwrap();

        let conv = store.get_conversation(&id).unwrap().unwrap();
        assert_eq!(conv.name, None);
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

        let conv_id = store
            .create_conversation(Some("chat"), &["alice", "bob"])
            .unwrap();

        // Add messages
        store
            .add_message(&conv_id, "alice", "Hello @bob!", &["bob"])
            .unwrap();
        store
            .add_message(&conv_id, "bob", "Hi @alice!", &["alice"])
            .unwrap();
        store
            .add_message(&conv_id, "alice", "How are you?", &[])
            .unwrap();

        // Get all messages
        let messages = store.get_messages(&conv_id, None).unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].from_agent, "alice");
        assert_eq!(messages[0].content, "Hello @bob!");
        assert_eq!(messages[0].mentions, vec!["bob"]);

        // Get last 2 messages
        let recent = store.get_messages(&conv_id, Some(2)).unwrap();
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

        let conv_id = store
            .create_conversation(Some("chat"), &["alice", "bob"])
            .unwrap();
        let msg_id = store
            .add_message(&conv_id, "alice", "Hey @bob", &["bob"])
            .unwrap();

        // Add pending notification
        store
            .add_pending_notification("bob", &conv_id, msg_id)
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

        let conv_id = store
            .create_conversation(Some("to-delete"), &["a", "b"])
            .unwrap();
        store.add_message(&conv_id, "a", "Hello", &[]).unwrap();

        // Delete
        store.delete_conversation(&conv_id).unwrap();

        // Verify gone
        assert!(store.get_conversation(&conv_id).unwrap().is_none());
        assert!(store.get_messages(&conv_id, None).unwrap().is_empty());
        assert!(store.get_participants(&conv_id).unwrap().is_empty());
    }

    #[test]
    fn test_generate_id_uniqueness() {
        let ids: Vec<String> = (0..100).map(|_| generate_id()).collect();
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(ids.len(), unique.len());
    }

    #[test]
    fn test_canonical_1to1_id() {
        assert_eq!(canonical_1to1_id("arya"), "1:1:arya:user");
        assert_eq!(canonical_1to1_id("gendry"), "1:1:gendry:user");
    }

    #[test]
    fn test_canonical_group_id_sorts_participants() {
        // Agents should be sorted alphabetically
        let id1 = canonical_group_id(&["gendry", "arya"]);
        let id2 = canonical_group_id(&["arya", "gendry"]);
        assert_eq!(id1, id2);
        assert_eq!(id1, "group:arya:gendry:user");
    }

    #[test]
    fn test_canonical_group_id_adds_user() {
        // User should be added if not present
        let id = canonical_group_id(&["arya", "gendry"]);
        assert!(id.contains("user"));
        assert_eq!(id, "group:arya:gendry:user");
    }

    #[test]
    fn test_canonical_group_id_with_user_already_present() {
        // Should not duplicate user
        let id = canonical_group_id(&["arya", "user", "gendry"]);
        assert_eq!(id, "group:arya:gendry:user");
    }

    #[test]
    fn test_find_by_name() {
        let store = test_store();

        // Create a conversation with a name
        let id = store.create_conversation(Some("project-x"), &["arya", "user"]).unwrap();

        // Find by name should work
        let found = store.find_by_name("project-x").unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, id);

        // Non-existent name should return None
        let not_found = store.find_by_name("nonexistent").unwrap();
        assert!(not_found.is_none());
    }

    #[test]
    fn test_get_or_create_conversation_creates_new() {
        let store = test_store();

        let conv = store.get_or_create_conversation("1:1:arya:user", &["arya", "user"]).unwrap();

        // Should have created a new conversation
        assert_eq!(conv.id, "1:1:arya:user");
        assert_eq!(conv.name, Some("1:1:arya:user".to_string()));

        // Participants should be set
        let parts = store.get_participants(&conv.id).unwrap();
        assert_eq!(parts.len(), 2);
        assert!(parts.iter().any(|p| p.agent == "arya"));
        assert!(parts.iter().any(|p| p.agent == "user"));
    }

    #[test]
    fn test_get_or_create_conversation_returns_existing() {
        let store = test_store();

        // Create first
        let conv1 = store.get_or_create_conversation("1:1:arya:user", &["arya", "user"]).unwrap();
        let created_at = conv1.created_at;

        // Get or create again - should return the same
        let conv2 = store.get_or_create_conversation("1:1:arya:user", &["arya", "user"]).unwrap();

        assert_eq!(conv1.id, conv2.id);
        assert_eq!(conv2.created_at, created_at);
    }

    #[test]
    fn test_get_or_create_conversation_idempotent() {
        let store = test_store();

        // Call multiple times
        let conv1 = store.get_or_create_conversation("group:arya:gendry:user", &["arya", "gendry", "user"]).unwrap();
        let conv2 = store.get_or_create_conversation("group:arya:gendry:user", &["arya", "gendry", "user"]).unwrap();
        let conv3 = store.get_or_create_conversation("group:arya:gendry:user", &["arya", "gendry", "user"]).unwrap();

        // All should have the same ID
        assert_eq!(conv1.id, conv2.id);
        assert_eq!(conv2.id, conv3.id);

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
}
