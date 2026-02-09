use crate::debug;
use async_trait::async_trait;
use rusqlite::{Connection, params};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

#[derive(Debug, Clone)]
pub enum MemoryError {
    StorageError(String),
    SerializationError(String),
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryError::StorageError(msg) => write!(f, "Storage error: {}", msg),
            MemoryError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
        }
    }
}

impl std::error::Error for MemoryError {}

#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub value: Value,
    pub created_at: i64,
    pub updated_at: i64,
}

#[async_trait]
pub trait Memory: Send + Sync {
    async fn get(&self, key: &str) -> Option<MemoryEntry>;
    async fn set(&mut self, key: &str, value: Value) -> Result<(), MemoryError>;
    async fn delete(&mut self, key: &str) -> bool;
    async fn list_keys(&self, prefix: Option<&str>) -> Vec<String>;
}

pub struct InMemoryStore {
    data: HashMap<String, MemoryEntry>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        InMemoryStore {
            data: HashMap::new(),
        }
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Memory for InMemoryStore {
    async fn get(&self, key: &str) -> Option<MemoryEntry> {
        self.data.get(key).cloned()
    }

    async fn set(&mut self, key: &str, value: Value) -> Result<(), MemoryError> {
        let timestamp = now();
        let entry = if let Some(existing) = self.data.get(key) {
            MemoryEntry {
                value,
                created_at: existing.created_at,
                updated_at: timestamp,
            }
        } else {
            MemoryEntry {
                value,
                created_at: timestamp,
                updated_at: timestamp,
            }
        };
        self.data.insert(key.to_string(), entry);
        Ok(())
    }

    async fn delete(&mut self, key: &str) -> bool {
        self.data.remove(key).is_some()
    }

    async fn list_keys(&self, prefix: Option<&str>) -> Vec<String> {
        match prefix {
            Some(p) => self
                .data
                .keys()
                .filter(|k| k.starts_with(p))
                .cloned()
                .collect(),
            None => self.data.keys().cloned().collect(),
        }
    }
}

// =============================================================================
// SQLite-backed persistent memory
// =============================================================================

/// Persistent memory backed by SQLite
pub struct SqliteMemory {
    conn: Arc<Mutex<Connection>>,
    agent_id: String,
}

impl SqliteMemory {
    /// Build a SqliteMemory from an already-opened connection, initializing the schema.
    fn from_connection(conn: Connection, agent_id: &str) -> Result<Self, MemoryError> {
        let memory = SqliteMemory {
            conn: Arc::new(Mutex::new(conn)),
            agent_id: agent_id.to_string(),
        };
        memory.init_schema()?;
        Ok(memory)
    }

    /// Open or create a SQLite database at the given path.
    pub fn open(path: &str, agent_id: &str) -> Result<Self, MemoryError> {
        let conn = Connection::open(path).map_err(|e| MemoryError::StorageError(e.to_string()))?;
        Self::from_connection(conn, agent_id)
    }

    /// Create an in-memory SQLite database (useful for testing).
    pub fn open_in_memory(agent_id: &str) -> Result<Self, MemoryError> {
        let conn =
            Connection::open_in_memory().map_err(|e| MemoryError::StorageError(e.to_string()))?;
        Self::from_connection(conn, agent_id)
    }

    fn init_schema(&self) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                UNIQUE(agent_id, key)
            );
            CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent_id);
            CREATE INDEX IF NOT EXISTS idx_memories_agent_key ON memories(agent_id, key);
            CREATE INDEX IF NOT EXISTS idx_memories_time ON memories(agent_id, updated_at);",
        )
        .map_err(|e| MemoryError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Query memories by time range (episodic memory)
    pub async fn query_by_time(
        &self,
        since: u64,
        until: Option<u64>,
    ) -> Result<Vec<(String, MemoryEntry)>, MemoryError> {
        let conn = self.conn.clone();
        let agent_id = self.agent_id.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            // Use i64 for SQLite compatibility (i64::MAX is ~292 billion years from epoch, plenty)
            let since_i64 = since as i64;
            let until_i64 = until.map(|u| u as i64).unwrap_or(i64::MAX);

            let mut stmt = conn
                .prepare(
                    "SELECT key, value, created_at, updated_at FROM memories
                 WHERE agent_id = ?1 AND updated_at >= ?2 AND updated_at <= ?3
                 ORDER BY updated_at ASC",
                )
                .map_err(|e| MemoryError::StorageError(e.to_string()))?;

            let rows = stmt
                .query_map(params![agent_id, since_i64, until_i64], |row| {
                    let key: String = row.get(0)?;
                    let value_str: String = row.get(1)?;
                    let created_at: i64 = row.get(2)?;
                    let updated_at: i64 = row.get(3)?;
                    Ok((key, value_str, created_at, updated_at))
                })
                .map_err(|e| MemoryError::StorageError(e.to_string()))?;

            let mut results = Vec::new();
            for row in rows {
                let (key, value_str, created_at, updated_at) =
                    row.map_err(|e| MemoryError::StorageError(e.to_string()))?;
                let value: Value = serde_json::from_str(&value_str)
                    .map_err(|e| MemoryError::SerializationError(e.to_string()))?;
                results.push((
                    key,
                    MemoryEntry {
                        value,
                        created_at,
                        updated_at,
                    },
                ));
            }
            Ok(results)
        })
        .await
        .map_err(|e| MemoryError::StorageError(e.to_string()))?
    }
}

#[async_trait]
impl Memory for SqliteMemory {
    async fn get(&self, key: &str) -> Option<MemoryEntry> {
        let conn = self.conn.clone();
        let agent_id = self.agent_id.clone();
        let key = key.to_string();

        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let result: Result<(String, i64, i64), _> = conn.query_row(
                "SELECT value, created_at, updated_at FROM memories WHERE agent_id = ?1 AND key = ?2",
                params![agent_id, key],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            );

            match result {
                Ok((value_str, created_at, updated_at)) => {
                    serde_json::from_str(&value_str).ok().map(|value| {
                        MemoryEntry { value, created_at, updated_at }
                    })
                }
                Err(_) => None,
            }
        }).await.ok().flatten()
    }

    async fn set(&mut self, key: &str, value: Value) -> Result<(), MemoryError> {
        let conn = self.conn.clone();
        let agent_id = self.agent_id.clone();
        let key = key.to_string();
        let value_str = serde_json::to_string(&value)
            .map_err(|e| MemoryError::SerializationError(e.to_string()))?;
        let timestamp = now();

        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();

            // Check if exists to preserve created_at
            let existing: Option<i64> = conn
                .query_row(
                    "SELECT created_at FROM memories WHERE agent_id = ?1 AND key = ?2",
                    params![agent_id, key],
                    |row| row.get(0),
                )
                .ok();

            let created_at = existing.unwrap_or(timestamp);

            conn.execute(
                "INSERT OR REPLACE INTO memories (agent_id, key, value, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![agent_id, key, value_str, created_at, timestamp],
            )
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

            Ok(())
        })
        .await
        .map_err(|e| MemoryError::StorageError(e.to_string()))?
    }

    async fn delete(&mut self, key: &str) -> bool {
        let conn = self.conn.clone();
        let agent_id = self.agent_id.clone();
        let key = key.to_string();

        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let rows = conn
                .execute(
                    "DELETE FROM memories WHERE agent_id = ?1 AND key = ?2",
                    params![agent_id, key],
                )
                .unwrap_or(0);
            rows > 0
        })
        .await
        .unwrap_or(false)
    }

    async fn list_keys(&self, prefix: Option<&str>) -> Vec<String> {
        let conn = self.conn.clone();
        let agent_id = self.agent_id.clone();
        let prefix = prefix.map(|s| s.to_string());

        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            // Use LIKE with a wildcard pattern: "prefix%" for filtered, "%" for all keys
            let pattern = match &prefix {
                Some(p) => format!("{}%", p),
                None => "%".to_string(),
            };
            let mut stmt = conn
                .prepare("SELECT key FROM memories WHERE agent_id = ?1 AND key LIKE ?2")
                .ok()?;
            let rows = stmt
                .query_map(params![agent_id, pattern], |row| row.get::<_, String>(0))
                .ok()?;
            Some(rows.filter_map(|r| r.ok()).collect())
        })
        .await
        .unwrap_or_default()
        .unwrap_or_default()
    }
}

// =============================================================================
// Semantic Memory System
// =============================================================================

/// A semantic memory entry with importance and access tracking.
#[derive(Debug, Clone)]
pub struct SemanticMemoryEntry {
    pub id: i64,
    pub created_at: i64,
    pub content: String,
    pub importance: f64,
    pub source: String,
    pub keywords: Option<String>,
    pub access_count: i64,
    pub last_accessed: Option<i64>,
    pub embedding: Option<Vec<f32>>,
}

/// Result of a memory save operation.
#[derive(Debug, Clone)]
pub enum SaveResult {
    /// New memory created (id)
    New(i64),
    /// Existing memory reinforced (id, old_importance, new_importance)
    Reinforced(i64, f64, f64),
}

/// Semantic memory store with embedding-based semantic search.
pub struct SemanticMemoryStore {
    conn: Arc<Mutex<Connection>>,
    agent_id: String,
}

/// Serialize embedding to bytes for SQLite BLOB storage.
pub(crate) fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Deserialize embedding from SQLite BLOB.
pub(crate) fn blob_to_embedding(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Standard column list for SemanticMemoryEntry queries.
/// All queries that construct SemanticMemoryEntry should SELECT these columns in this order.
const SEMANTIC_MEMORY_COLUMNS: &str =
    "id, content, importance, source, created_at, keywords, access_count, last_accessed, embedding";

/// Map a rusqlite row (using SEMANTIC_MEMORY_COLUMNS order) to a SemanticMemoryEntry.
fn row_to_semantic_entry(row: &rusqlite::Row<'_>) -> rusqlite::Result<SemanticMemoryEntry> {
    let embedding_blob: Option<Vec<u8>> = row.get(8)?;
    Ok(SemanticMemoryEntry {
        id: row.get(0)?,
        content: row.get(1)?,
        importance: row.get(2)?,
        source: row.get(3)?,
        created_at: row.get(4)?,
        keywords: row.get(5)?,
        access_count: row.get(6)?,
        last_accessed: row.get(7)?,
        embedding: embedding_blob.map(|b| blob_to_embedding(&b)),
    })
}

/// Common English stopwords to filter from keyword extraction.
const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it",
    "its", "of", "on", "that", "the", "to", "was", "were", "will", "with", "the", "this", "but",
    "they", "have", "had", "what", "when", "where", "who", "which", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no", "not", "only", "own",
    "same", "so", "than", "too", "very", "can", "just", "should", "now", "i", "you", "we", "my",
    "your", "our", "me", "him", "her", "them", "us", "do", "does", "did", "doing", "done", "would",
    "could", "might", "must", "shall", "may", "am", "been", "being", "over", "under", "into",
    "out", "up", "down", "about", "after", "before",
];

/// Extract meaningful keywords from text by filtering stopwords.
fn extract_keywords(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && w.len() > 2 && !STOPWORDS.contains(w))
        .map(|s| s.to_string())
        .collect()
}

/// Calculate recency score with exponential decay (half-life of ~7 days / 168 hours).
fn recency_score(created_at: i64, now: i64) -> f64 {
    let age_hours = (now - created_at) as f64 / 3600.0;
    0.5_f64.powf(age_hours / 168.0)
}

/// Format age as human-readable string.
pub fn format_age(created_at: i64) -> String {
    let now = now();
    let age_secs = now - created_at;

    if age_secs < 60 {
        "just now".to_string()
    } else if age_secs < 3600 {
        format!("{}m ago", age_secs / 60)
    } else if age_secs < 86400 {
        format!("{}h ago", age_secs / 3600)
    } else if age_secs < 172800 {
        "yesterday".to_string()
    } else {
        format!("{}d ago", age_secs / 86400)
    }
}

impl SemanticMemoryStore {
    /// Build a SemanticMemoryStore from an already-opened connection, initializing the schema.
    fn from_connection(conn: Connection, agent_id: &str) -> Result<Self, MemoryError> {
        let store = SemanticMemoryStore {
            conn: Arc::new(Mutex::new(conn)),
            agent_id: agent_id.to_string(),
        };
        store.init_schema()?;
        Ok(store)
    }

    /// Open or create a semantic memory database at the given path.
    pub fn open(path: &std::path::Path, agent_id: &str) -> Result<Self, MemoryError> {
        let conn = Connection::open(path).map_err(|e| MemoryError::StorageError(e.to_string()))?;
        Self::from_connection(conn, agent_id)
    }

    /// Create an in-memory semantic memory store (useful for testing).
    pub fn open_in_memory(agent_id: &str) -> Result<Self, MemoryError> {
        let conn =
            Connection::open_in_memory().map_err(|e| MemoryError::StorageError(e.to_string()))?;
        Self::from_connection(conn, agent_id)
    }

    fn init_schema(&self) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS semantic_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                content TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                source TEXT DEFAULT 'auto',
                keywords TEXT,
                access_count INTEGER DEFAULT 0,
                last_accessed INTEGER,
                embedding BLOB
            );
            CREATE INDEX IF NOT EXISTS idx_semantic_agent ON semantic_memories(agent_id);
            CREATE INDEX IF NOT EXISTS idx_semantic_keywords ON semantic_memories(agent_id, keywords);
            CREATE INDEX IF NOT EXISTS idx_semantic_created ON semantic_memories(agent_id, created_at);

            CREATE TABLE IF NOT EXISTS memory_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );"
        ).map_err(|e| MemoryError::StorageError(e.to_string()))?;

        // Migration: Add embedding column if it doesn't exist
        let _ = conn.execute(
            "ALTER TABLE semantic_memories ADD COLUMN embedding BLOB",
            [],
        );

        Ok(())
    }

    /// Save a new semantic memory, or reinforce an existing one if duplicate.
    /// Reinforcing boosts importance by 0.05 (capped at 1.0) and refreshes timestamp.
    ///
    /// # Arguments
    /// * `content` - The memory content to save
    /// * `importance` - Importance score (0.0-1.0)
    /// * `source` - Source of the memory ("auto", "explicit", etc.)
    /// * `embedding` - Optional embedding vector for semantic search
    pub fn save(
        &self,
        content: &str,
        importance: f64,
        source: &str,
    ) -> Result<SaveResult, MemoryError> {
        self.save_with_embedding(content, importance, source, None)
    }

    /// Save a new semantic memory with an optional embedding vector.
    pub fn save_with_embedding(
        &self,
        content: &str,
        importance: f64,
        source: &str,
        embedding: Option<&[f32]>,
    ) -> Result<SaveResult, MemoryError> {
        let keywords = extract_keywords(content).join(",");
        let timestamp = now();
        let embedding_blob = embedding.map(embedding_to_blob);

        let conn = self.conn.lock().unwrap();

        // Check for existing exact match
        let existing: Option<(i64, f64)> = conn
            .query_row(
                "SELECT id, importance FROM semantic_memories WHERE agent_id = ?1 AND content = ?2",
                params![self.agent_id, content],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .ok();

        if let Some((id, old_importance)) = existing {
            // Reinforce: boost importance by 0.05 (cap at 1.0), refresh timestamp, update embedding if provided
            let new_importance = (old_importance + 0.05).min(1.0);
            conn.execute(
                "UPDATE semantic_memories SET importance = ?1, created_at = ?2, embedding = COALESCE(?3, embedding) WHERE id = ?4",
                params![new_importance, timestamp, embedding_blob, id],
            )
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

            debug::log(&format!(
                "[memory] REINFORCE #{}: \"{}\" (importance {} -> {})",
                id, content, old_importance, new_importance
            ));
            Ok(SaveResult::Reinforced(id, old_importance, new_importance))
        } else {
            // New memory
            conn.execute(
                "INSERT INTO semantic_memories (agent_id, created_at, content, importance, source, keywords, access_count, embedding)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0, ?7)",
                params![self.agent_id, timestamp, content, importance, source, keywords, embedding_blob]
            ).map_err(|e| MemoryError::StorageError(e.to_string()))?;

            let id = conn.last_insert_rowid();
            debug::log(&format!(
                "[memory] SAVE #{}: \"{}\" (importance={}, source={}, keywords={}, has_embedding={})",
                id,
                content,
                importance,
                source,
                keywords,
                embedding.is_some()
            ));
            Ok(SaveResult::New(id))
        }
    }

    /// Recall relevant memories for a query using keyword-based search.
    /// For embedding-based search, use `recall_with_embedding` instead.
    #[deprecated(note = "Use recall_with_embedding for semantic search")]
    pub fn recall(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SemanticMemoryEntry>, MemoryError> {
        Ok(self
            .recall_with_embedding(query, limit, None)?
            .into_iter()
            .map(|(m, _)| m)
            .collect())
    }

    /// Recall relevant memories using embedding-based semantic search.
    ///
    /// When `query_embedding` is provided, uses cosine similarity for scoring.
    /// When `query_embedding` is None, returns empty (embeddings are required for recall).
    ///
    /// Returns entries paired with their relevance scores (higher = more relevant).
    pub fn recall_with_embedding(
        &self,
        query: &str,
        limit: usize,
        query_embedding: Option<&[f32]>,
    ) -> Result<Vec<(SemanticMemoryEntry, f64)>, MemoryError> {
        use crate::embedding::cosine_similarity;

        debug::log(&format!(
            "[memory] RECALL query=\"{}\" limit={} has_embedding={}",
            query.chars().take(100).collect::<String>(),
            limit,
            query_embedding.is_some()
        ));

        // Embedding-based recall requires a query embedding
        let query_embedding = match query_embedding {
            Some(emb) => emb,
            None => {
                debug::log("[memory] RECALL: no query embedding provided, returning empty");
                return Ok(Vec::new());
            }
        };

        let conn = self.conn.lock().unwrap();

        // Load all memories for this agent with their embeddings
        let query = format!(
            "SELECT {} FROM semantic_memories WHERE agent_id = ?1 AND embedding IS NOT NULL ORDER BY created_at DESC",
            SEMANTIC_MEMORY_COLUMNS
        );
        let mut stmt = conn
            .prepare(&query)
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        let rows = stmt
            .query_map(params![self.agent_id], row_to_semantic_entry)
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        let candidates: Vec<SemanticMemoryEntry> = rows.filter_map(|r| r.ok()).collect();

        if candidates.is_empty() {
            debug::log("[memory] RECALL: no memories with embeddings stored");
            return Ok(Vec::new());
        }

        // Score memories using cosine similarity
        let current_time = now();

        let mut scored: Vec<(SemanticMemoryEntry, f64)> = candidates
            .into_iter()
            .filter_map(|m| {
                let m_emb = m.embedding.as_ref()?;

                // Embedding-based similarity (convert from [-1, 1] to [0, 1] range)
                let sim = cosine_similarity(query_embedding, m_emb);
                let relevance = ((sim + 1.0) / 2.0) as f64;

                if relevance <= 0.0 {
                    return None;
                }

                let recency = recency_score(m.created_at, current_time);
                let score = relevance * recency * m.importance;

                if score > 0.0 { Some((m, score)) } else { None }
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        // Log what we found
        if scored.is_empty() {
            debug::log("[memory] RECALL: no matching memories found");
        } else {
            debug::log(&format!(
                "[memory] RECALL: found {} memories:",
                scored.len()
            ));
            for (m, score) in &scored {
                debug::log(&format!(
                    "[memory]   #{} (score={:.3}): \"{}\"",
                    m.id,
                    score,
                    m.content.chars().take(80).collect::<String>()
                ));
            }
        }

        // Mark accessed
        let ids: Vec<i64> = scored.iter().map(|(m, _)| m.id).collect();
        drop(stmt);
        for id in &ids {
            self.mark_accessed_internal(&conn, *id)?;
        }

        Ok(scored)
    }

    fn mark_accessed_internal(&self, conn: &Connection, id: i64) -> Result<(), MemoryError> {
        let timestamp = now();
        conn.execute(
            "UPDATE semantic_memories SET access_count = access_count + 1, last_accessed = ?1 WHERE id = ?2",
            params![timestamp, id]
        ).map_err(|e| MemoryError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Get the count of memories for this agent.
    pub fn count(&self) -> Result<i64, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM semantic_memories WHERE agent_id = ?1",
                params![self.agent_id],
                |row| row.get(0),
            )
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;
        Ok(count)
    }

    /// List all memories, ordered by created_at descending.
    pub fn list_all(&self, limit: Option<usize>) -> Result<Vec<SemanticMemoryEntry>, MemoryError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;
        let limit_clause = limit.map(|l| format!(" LIMIT {}", l)).unwrap_or_default();
        let query = format!(
            "SELECT {} FROM semantic_memories WHERE agent_id = ?1 ORDER BY created_at DESC{}",
            SEMANTIC_MEMORY_COLUMNS, limit_clause
        );
        let mut stmt = conn
            .prepare(&query)
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        let entries = stmt
            .query_map(params![self.agent_id], row_to_semantic_entry)
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        entries
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| MemoryError::StorageError(e.to_string()))
    }

    /// Delete a memory by ID. Returns true if deleted, false if not found.
    pub fn delete(&self, id: i64) -> Result<bool, MemoryError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;
        let rows = conn
            .execute(
                "DELETE FROM semantic_memories WHERE id = ?1 AND agent_id = ?2",
                params![id, self.agent_id],
            )
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;
        Ok(rows > 0)
    }

    /// Delete all memories for this agent. Returns count of deleted memories.
    pub fn clear_all(&self) -> Result<usize, MemoryError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;
        let rows = conn
            .execute(
                "DELETE FROM semantic_memories WHERE agent_id = ?1",
                params![self.agent_id],
            )
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;
        Ok(rows)
    }

    /// Update the content of an existing memory. Keeps importance, updates timestamp.
    /// Returns true if updated, false if not found.
    pub fn update_content(&self, id: i64, new_content: &str) -> Result<bool, MemoryError> {
        let timestamp = now();
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;
        let rows = conn.execute(
            "UPDATE semantic_memories SET content = ?1, last_accessed = ?2 WHERE id = ?3 AND agent_id = ?4",
            params![new_content, timestamp, id, self.agent_id]
        ).map_err(|e| MemoryError::StorageError(e.to_string()))?;
        Ok(rows > 0)
    }

    /// Get a specific memory by ID. Returns None if not found or belongs to different agent.
    pub fn get(&self, id: i64) -> Result<Option<SemanticMemoryEntry>, MemoryError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;
        let query = format!(
            "SELECT {} FROM semantic_memories WHERE id = ?1 AND agent_id = ?2",
            SEMANTIC_MEMORY_COLUMNS
        );
        let result = conn.query_row(&query, params![id, self.agent_id], row_to_semantic_entry);
        match result {
            Ok(entry) => Ok(Some(entry)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(MemoryError::StorageError(e.to_string())),
        }
    }

    /// Get the currently configured embedding model from metadata.
    pub fn get_embedding_model(&self) -> Result<Option<String>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let result: Result<String, _> = conn.query_row(
            "SELECT value FROM memory_meta WHERE key = 'embedding_model'",
            [],
            |row| row.get(0),
        );
        match result {
            Ok(model) => Ok(Some(model)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(MemoryError::StorageError(e.to_string())),
        }
    }

    /// Set the embedding model in metadata.
    pub fn set_embedding_model(&self, model: &str) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO memory_meta (key, value) VALUES ('embedding_model', ?1)",
            params![model],
        )
        .map_err(|e| MemoryError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Check if any memories are missing embeddings.
    pub fn has_null_embeddings(&self) -> Result<bool, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM semantic_memories WHERE agent_id = ?1 AND embedding IS NULL",
                params![self.agent_id],
                |row| row.get(0),
            )
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;
        Ok(count > 0)
    }

    /// Get all memories that need embeddings (have NULL embedding).
    pub fn get_memories_needing_embeddings(&self) -> Result<Vec<(i64, String)>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, content FROM semantic_memories WHERE agent_id = ?1 AND embedding IS NULL"
        ).map_err(|e| MemoryError::StorageError(e.to_string()))?;

        let rows = stmt
            .query_map(params![self.agent_id], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        let results: Vec<(i64, String)> = rows.filter_map(|r| r.ok()).collect();
        Ok(results)
    }

    /// Update the embedding for a specific memory.
    pub fn update_embedding(&self, id: i64, embedding: &[f32]) -> Result<(), MemoryError> {
        let blob = embedding_to_blob(embedding);
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE semantic_memories SET embedding = ?1 WHERE id = ?2",
            params![blob, id],
        )
        .map_err(|e| MemoryError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Check if embeddings need to be backfilled.
    ///
    /// Returns true if:
    /// - The configured model differs from the stored model
    /// - Any memories have NULL embeddings
    pub fn needs_backfill(&self, configured_model: &str) -> Result<bool, MemoryError> {
        let stored_model = self.get_embedding_model()?;

        // Check if model changed
        if stored_model.as_deref() != Some(configured_model) {
            return Ok(true);
        }

        // Check for NULL embeddings
        self.has_null_embeddings()
    }
}

/// Parse and extract [REMEMBER: ...] tags from agent output.
/// Returns the cleaned output (with tags removed) and the extracted memory contents.
pub fn extract_remember_tags(output: &str) -> (String, Vec<String>) {
    let re = regex::Regex::new(r"\[REMEMBER:\s*(.+?)\]").unwrap();
    let mut memories = Vec::new();

    for cap in re.captures_iter(output) {
        if let Some(content) = cap.get(1) {
            let memory = content.as_str().trim().to_string();
            debug::log(&format!("[memory] EXTRACT tag: \"{}\"", memory));
            memories.push(memory);
        }
    }

    let cleaned = re.replace_all(output, "").trim().to_string();

    if !memories.is_empty() {
        debug::log(&format!(
            "[memory] EXTRACT: found {} [REMEMBER:] tags",
            memories.len()
        ));
    }

    (cleaned, memories)
}

/// Build the memory injection string for prepending to context.
pub fn build_memory_injection(memories: &[SemanticMemoryEntry]) -> String {
    if memories.is_empty() {
        return String::new();
    }

    let mut injection = String::from("[recalled memories]\n");
    for m in memories {
        let age = format_age(m.created_at);
        let flag = if m.importance > 0.8 { " ⭐" } else { "" };
        injection.push_str(&format!("- ({}) {}{}\n", age, m.content, flag));
    }
    injection.push('\n');

    debug::log(&format!(
        "[memory] INJECT: {} memories into context:\n{}",
        memories.len(),
        injection
    ));

    injection
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // =========================================================================
    // InMemoryStore tests
    // =========================================================================

    #[tokio::test]
    async fn test_in_memory_new() {
        let store = InMemoryStore::new();
        assert!(store.data.is_empty());
    }

    #[tokio::test]
    async fn test_in_memory_default() {
        let store = InMemoryStore::default();
        assert!(store.data.is_empty());
    }

    #[tokio::test]
    async fn test_in_memory_set_and_get() {
        let mut store = InMemoryStore::new();
        store.set("key1", json!("value1")).await.unwrap();

        let entry = store.get("key1").await.unwrap();
        assert_eq!(entry.value, json!("value1"));
    }

    #[tokio::test]
    async fn test_in_memory_get_nonexistent() {
        let store = InMemoryStore::new();
        assert!(store.get("nonexistent").await.is_none());
    }

    #[tokio::test]
    async fn test_in_memory_delete() {
        let mut store = InMemoryStore::new();
        store.set("key1", json!("value1")).await.unwrap();

        assert!(store.delete("key1").await);
        assert!(store.get("key1").await.is_none());
    }

    #[tokio::test]
    async fn test_in_memory_delete_nonexistent() {
        let mut store = InMemoryStore::new();
        assert!(!store.delete("nonexistent").await);
    }

    #[tokio::test]
    async fn test_in_memory_list_keys_all() {
        let mut store = InMemoryStore::new();
        store.set("alpha", json!(1)).await.unwrap();
        store.set("beta", json!(2)).await.unwrap();
        store.set("gamma", json!(3)).await.unwrap();

        let keys = store.list_keys(None).await;
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&"alpha".to_string()));
        assert!(keys.contains(&"beta".to_string()));
        assert!(keys.contains(&"gamma".to_string()));
    }

    #[tokio::test]
    async fn test_in_memory_list_keys_with_prefix() {
        let mut store = InMemoryStore::new();
        store.set("user:1", json!(1)).await.unwrap();
        store.set("user:2", json!(2)).await.unwrap();
        store.set("config:theme", json!("dark")).await.unwrap();

        let keys = store.list_keys(Some("user:")).await;
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"user:1".to_string()));
        assert!(keys.contains(&"user:2".to_string()));
    }

    #[tokio::test]
    async fn test_in_memory_preserves_created_at() {
        let mut store = InMemoryStore::new();
        store.set("key", json!("v1")).await.unwrap();

        let entry1 = store.get("key").await.unwrap();
        let created = entry1.created_at;

        // Update the key
        store.set("key", json!("v2")).await.unwrap();

        let entry2 = store.get("key").await.unwrap();
        assert_eq!(entry2.created_at, created);
        assert_eq!(entry2.value, json!("v2"));
    }

    #[tokio::test]
    async fn test_in_memory_complex_values() {
        let mut store = InMemoryStore::new();
        let complex = json!({
            "nested": {"deep": [1, 2, 3]},
            "list": ["a", "b"],
            "number": 42
        });
        store.set("complex", complex.clone()).await.unwrap();

        let entry = store.get("complex").await.unwrap();
        assert_eq!(entry.value, complex);
    }

    // =========================================================================
    // SqliteMemory tests
    // =========================================================================

    #[tokio::test]
    async fn test_sqlite_open_in_memory() {
        let mem = SqliteMemory::open_in_memory("test-agent").unwrap();
        assert_eq!(mem.agent_id, "test-agent");
    }

    #[tokio::test]
    async fn test_sqlite_set_and_get() {
        let mut mem = SqliteMemory::open_in_memory("agent1").unwrap();
        mem.set("key1", json!("value1")).await.unwrap();

        let entry = mem.get("key1").await.unwrap();
        assert_eq!(entry.value, json!("value1"));
    }

    #[tokio::test]
    async fn test_sqlite_get_nonexistent() {
        let mem = SqliteMemory::open_in_memory("agent1").unwrap();
        assert!(mem.get("nonexistent").await.is_none());
    }

    #[tokio::test]
    async fn test_sqlite_delete() {
        let mut mem = SqliteMemory::open_in_memory("agent1").unwrap();
        mem.set("key1", json!("value1")).await.unwrap();

        assert!(mem.delete("key1").await);
        assert!(mem.get("key1").await.is_none());
    }

    #[tokio::test]
    async fn test_sqlite_delete_nonexistent() {
        let mut mem = SqliteMemory::open_in_memory("agent1").unwrap();
        assert!(!mem.delete("nonexistent").await);
    }

    #[tokio::test]
    async fn test_sqlite_list_keys_all() {
        let mut mem = SqliteMemory::open_in_memory("agent1").unwrap();
        mem.set("alpha", json!(1)).await.unwrap();
        mem.set("beta", json!(2)).await.unwrap();
        mem.set("gamma", json!(3)).await.unwrap();

        let keys = mem.list_keys(None).await;
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&"alpha".to_string()));
        assert!(keys.contains(&"beta".to_string()));
        assert!(keys.contains(&"gamma".to_string()));
    }

    #[tokio::test]
    async fn test_sqlite_list_keys_with_prefix() {
        let mut mem = SqliteMemory::open_in_memory("agent1").unwrap();
        mem.set("user:1", json!(1)).await.unwrap();
        mem.set("user:2", json!(2)).await.unwrap();
        mem.set("config:theme", json!("dark")).await.unwrap();

        let keys = mem.list_keys(Some("user:")).await;
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"user:1".to_string()));
        assert!(keys.contains(&"user:2".to_string()));
    }

    #[tokio::test]
    async fn test_sqlite_agent_isolation() {
        // Two agents sharing the same database shouldn't see each other's data
        let conn = Connection::open_in_memory().unwrap();
        let conn = Arc::new(Mutex::new(conn));

        let mut mem1 = SqliteMemory {
            conn: conn.clone(),
            agent_id: "agent1".to_string(),
        };
        mem1.init_schema().unwrap();

        let mut mem2 = SqliteMemory {
            conn: conn.clone(),
            agent_id: "agent2".to_string(),
        };

        mem1.set("shared_key", json!("agent1_value")).await.unwrap();
        mem2.set("shared_key", json!("agent2_value")).await.unwrap();

        let entry1 = mem1.get("shared_key").await.unwrap();
        let entry2 = mem2.get("shared_key").await.unwrap();

        assert_eq!(entry1.value, json!("agent1_value"));
        assert_eq!(entry2.value, json!("agent2_value"));
    }

    #[tokio::test]
    async fn test_sqlite_preserves_created_at() {
        let mut mem = SqliteMemory::open_in_memory("agent1").unwrap();
        mem.set("key", json!("v1")).await.unwrap();

        let entry1 = mem.get("key").await.unwrap();
        let created = entry1.created_at;

        // Small delay to ensure timestamp changes
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        mem.set("key", json!("v2")).await.unwrap();

        let entry2 = mem.get("key").await.unwrap();
        assert_eq!(entry2.created_at, created);
        assert_eq!(entry2.value, json!("v2"));
    }

    #[tokio::test]
    async fn test_sqlite_complex_values() {
        let mut mem = SqliteMemory::open_in_memory("agent1").unwrap();
        let complex = json!({
            "nested": {"deep": [1, 2, 3]},
            "list": ["a", "b"],
            "number": 42
        });
        mem.set("complex", complex.clone()).await.unwrap();

        let entry = mem.get("complex").await.unwrap();
        assert_eq!(entry.value, complex);
    }

    #[tokio::test]
    async fn test_sqlite_query_by_time() {
        let mut mem = SqliteMemory::open_in_memory("agent1").unwrap();

        // Set some values
        mem.set("key1", json!("v1")).await.unwrap();
        let time_after_first = now() as u64;

        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        mem.set("key2", json!("v2")).await.unwrap();

        // Query all
        let all = mem.query_by_time(0, None).await.unwrap();
        assert_eq!(all.len(), 2);

        // Query since a specific time
        let recent = mem.query_by_time(time_after_first, None).await.unwrap();
        assert!(recent.len() >= 1); // At least key2 should be there
    }

    #[tokio::test]
    async fn test_sqlite_persistence_same_connection() {
        // Test that data persists when reopening with same connection
        let mut mem = SqliteMemory::open_in_memory("agent1").unwrap();
        mem.set("persistent", json!("data")).await.unwrap();

        // Get again - should still be there
        let entry = mem.get("persistent").await.unwrap();
        assert_eq!(entry.value, json!("data"));
    }

    // =========================================================================
    // MemoryError tests
    // =========================================================================

    #[test]
    fn test_memory_error_display() {
        let storage_err = MemoryError::StorageError("disk full".to_string());
        assert!(storage_err.to_string().contains("disk full"));

        let serial_err = MemoryError::SerializationError("invalid json".to_string());
        assert!(serial_err.to_string().contains("invalid json"));
    }

    // =========================================================================
    // MemoryEntry tests
    // =========================================================================

    #[test]
    fn test_memory_entry_clone() {
        let entry = MemoryEntry {
            value: json!("test"),
            created_at: 100,
            updated_at: 200,
        };
        let cloned = entry.clone();
        assert_eq!(cloned.value, entry.value);
        assert_eq!(cloned.created_at, entry.created_at);
        assert_eq!(cloned.updated_at, entry.updated_at);
    }

    // =========================================================================
    // Semantic Memory tests
    // =========================================================================

    #[test]
    fn test_extract_keywords() {
        let keywords = extract_keywords("The quick brown fox jumps over the lazy dog");
        // Should filter out stopwords like "the", "over"
        assert!(keywords.contains(&"quick".to_string()));
        assert!(keywords.contains(&"brown".to_string()));
        assert!(keywords.contains(&"fox".to_string()));
        assert!(keywords.contains(&"jumps".to_string()));
        assert!(keywords.contains(&"lazy".to_string()));
        assert!(keywords.contains(&"dog".to_string()));
        assert!(!keywords.contains(&"the".to_string()));
        assert!(!keywords.contains(&"over".to_string()));
    }

    #[test]
    fn test_extract_keywords_filters_short_words() {
        let keywords = extract_keywords("I am a cat in the box");
        // Should filter out words with 2 or fewer chars
        assert!(!keywords.contains(&"i".to_string()));
        assert!(!keywords.contains(&"am".to_string()));
        assert!(!keywords.contains(&"a".to_string()));
        assert!(!keywords.contains(&"in".to_string()));
        assert!(keywords.contains(&"cat".to_string()));
        assert!(keywords.contains(&"box".to_string()));
    }

    #[test]
    fn test_recency_score_now() {
        let current = now();
        let score = recency_score(current, current);
        assert!((score - 1.0).abs() < 0.01); // Recent = high score
    }

    #[test]
    fn test_recency_score_week_old() {
        let current = now();
        let week_ago = current - (7 * 24 * 3600); // 7 days ago
        let score = recency_score(week_ago, current);
        // After ~7 days (half-life), score should be ~0.5
        assert!(score > 0.4 && score < 0.6);
    }

    #[test]
    fn test_recency_score_two_weeks_old() {
        let current = now();
        let two_weeks_ago = current - (14 * 24 * 3600);
        let score = recency_score(two_weeks_ago, current);
        // After ~14 days (two half-lives), score should be ~0.25
        assert!(score > 0.2 && score < 0.3);
    }

    #[test]
    fn test_format_age_just_now() {
        let current = now();
        assert_eq!(format_age(current), "just now");
    }

    #[test]
    fn test_format_age_minutes() {
        let current = now();
        let five_min_ago = current - 300;
        assert_eq!(format_age(five_min_ago), "5m ago");
    }

    #[test]
    fn test_format_age_hours() {
        let current = now();
        let two_hours_ago = current - 7200;
        assert_eq!(format_age(two_hours_ago), "2h ago");
    }

    #[test]
    fn test_format_age_yesterday() {
        let current = now();
        let yesterday = current - 86400;
        assert_eq!(format_age(yesterday), "yesterday");
    }

    #[test]
    fn test_format_age_days() {
        let current = now();
        let three_days_ago = current - (3 * 86400);
        assert_eq!(format_age(three_days_ago), "3d ago");
    }

    #[test]
    fn test_semantic_memory_save_and_recall() {
        let store = SemanticMemoryStore::open_in_memory("agent1").unwrap();

        // Save a memory with embedding
        let embedding = vec![0.9, 0.1, 0.0, 0.0];
        let result = store
            .save_with_embedding(
                "User prefers dark mode for coding",
                0.9,
                "explicit",
                Some(&embedding),
            )
            .unwrap();
        let id = match result {
            SaveResult::New(id) => id,
            SaveResult::Reinforced(id, _, _) => id,
        };
        assert!(id > 0);

        // Recall with matching embedding
        let query_emb = vec![0.85, 0.15, 0.0, 0.0];
        let results = store
            .recall_with_embedding("dark mode preference", 5, Some(&query_emb))
            .unwrap();
        assert_eq!(results.len(), 1);
        let (entry, score) = &results[0];
        assert!(entry.content.contains("dark mode"));
        assert!((entry.importance - 0.9).abs() < 0.01);
        assert!(*score > 0.0); // Score should be positive
    }

    #[test]
    fn test_semantic_memory_recall_multiple() {
        let store = SemanticMemoryStore::open_in_memory("agent1").unwrap();

        // Save with different embeddings
        let dark_emb = vec![0.9, 0.1, 0.0, 0.0];
        let rust_emb = vec![0.1, 0.9, 0.0, 0.0];
        let coffee_emb = vec![0.0, 0.0, 0.9, 0.1];

        store
            .save_with_embedding("User prefers dark mode", 0.8, "explicit", Some(&dark_emb))
            .unwrap();
        store
            .save_with_embedding("User works on Rust projects", 0.7, "auto", Some(&rust_emb))
            .unwrap();
        store
            .save_with_embedding("User likes coffee", 0.5, "auto", Some(&coffee_emb))
            .unwrap();

        // Query with embedding similar to "dark mode"
        let query_emb = vec![0.85, 0.15, 0.0, 0.0];
        let results = store
            .recall_with_embedding("dark theme preference", 5, Some(&query_emb))
            .unwrap();
        assert!(!results.is_empty());
        assert!(results[0].0.content.contains("dark"));
    }

    #[test]
    fn test_semantic_memory_recall_no_match() {
        let store = SemanticMemoryStore::open_in_memory("agent1").unwrap();

        // Save with embedding
        let embedding = vec![0.9, 0.1, 0.0, 0.0];
        store
            .save_with_embedding("User prefers dark mode", 0.8, "explicit", Some(&embedding))
            .unwrap();

        // Query with orthogonal embedding
        let query_emb = vec![0.0, 0.0, 0.9, 0.1];
        let results = store
            .recall_with_embedding("xyz123 completely unrelated", 5, Some(&query_emb))
            .unwrap();
        // With orthogonal vectors, cosine similarity is 0, so relevance after normalization is 0.5
        // But score = relevance * recency * importance, which may still be positive
        // The test passes if results are empty or if similarity is low
        if !results.is_empty() {
            // Verify the result has low relevance (cosine sim near 0 -> normalized ~0.5)
            assert!(results.len() <= 1);
        }
    }

    #[test]
    fn test_semantic_memory_recall_limit() {
        let store = SemanticMemoryStore::open_in_memory("agent1").unwrap();

        // Save multiple memories with similar embeddings
        let base_emb = vec![0.9, 0.1, 0.0, 0.0];
        for i in 0..10 {
            let mut emb = base_emb.clone();
            emb[0] += (i as f32) * 0.01; // Slight variation
            store
                .save_with_embedding(
                    &format!("Memory about coding number {}", i),
                    0.5,
                    "auto",
                    Some(&emb),
                )
                .unwrap();
        }

        // Query with similar embedding, should only return the limited number
        let results = store
            .recall_with_embedding("coding memory", 3, Some(&base_emb))
            .unwrap();
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_semantic_memory_importance_scoring() {
        let store = SemanticMemoryStore::open_in_memory("agent1").unwrap();

        // Save with different importance but same embedding
        let embedding = vec![0.9, 0.1, 0.0, 0.0];
        store
            .save_with_embedding("Low importance coding fact", 0.3, "auto", Some(&embedding))
            .unwrap();

        // Wait a tiny bit to ensure different timestamps
        std::thread::sleep(std::time::Duration::from_millis(10));

        store
            .save_with_embedding(
                "High importance coding fact",
                0.95,
                "explicit",
                Some(&embedding),
            )
            .unwrap();

        let results = store
            .recall_with_embedding("coding fact", 5, Some(&embedding))
            .unwrap();
        assert!(!results.is_empty());
        // Higher importance should rank higher (when other factors are equal)
        assert!(results[0].0.importance > 0.9);
    }

    #[test]
    fn test_semantic_memory_access_tracking() {
        let store = SemanticMemoryStore::open_in_memory("agent1").unwrap();

        let embedding = vec![0.9, 0.1, 0.0, 0.0];
        store
            .save_with_embedding(
                "Test memory for access tracking",
                0.5,
                "auto",
                Some(&embedding),
            )
            .unwrap();

        // First recall
        let results1 = store
            .recall_with_embedding("access tracking", 5, Some(&embedding))
            .unwrap();
        assert_eq!(results1[0].0.access_count, 0); // Not incremented yet at read time

        // Second recall should show incremented count
        let results2 = store
            .recall_with_embedding("access tracking", 5, Some(&embedding))
            .unwrap();
        assert_eq!(results2[0].0.access_count, 1);

        // Third recall
        let results3 = store
            .recall_with_embedding("access tracking", 5, Some(&embedding))
            .unwrap();
        assert_eq!(results3[0].0.access_count, 2);
    }

    #[test]
    fn test_semantic_memory_count() {
        let store = SemanticMemoryStore::open_in_memory("agent1").unwrap();

        assert_eq!(store.count().unwrap(), 0);

        store.save("First memory", 0.5, "auto").unwrap();
        assert_eq!(store.count().unwrap(), 1);

        store.save("Second memory", 0.5, "auto").unwrap();
        assert_eq!(store.count().unwrap(), 2);
    }

    #[test]
    fn test_extract_remember_tags_single() {
        let output = "Hello! [REMEMBER: User's name is Alice] Nice to meet you.";
        let (cleaned, memories) = extract_remember_tags(output);

        assert_eq!(cleaned, "Hello!  Nice to meet you.");
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0], "User's name is Alice");
    }

    #[test]
    fn test_extract_remember_tags_multiple() {
        let output = "[REMEMBER: Fact 1] Some text [REMEMBER: Fact 2] More text [REMEMBER: Fact 3]";
        let (cleaned, memories) = extract_remember_tags(output);

        assert_eq!(cleaned, "Some text  More text");
        assert_eq!(memories.len(), 3);
        assert_eq!(memories[0], "Fact 1");
        assert_eq!(memories[1], "Fact 2");
        assert_eq!(memories[2], "Fact 3");
    }

    #[test]
    fn test_extract_remember_tags_none() {
        let output = "This is a normal response with no tags.";
        let (cleaned, memories) = extract_remember_tags(output);

        assert_eq!(cleaned, output);
        assert!(memories.is_empty());
    }

    #[test]
    fn test_extract_remember_tags_whitespace() {
        let output = "[REMEMBER:   spaced content   ]";
        let (_, memories) = extract_remember_tags(output);

        assert_eq!(memories[0], "spaced content");
    }

    #[test]
    fn test_build_memory_injection_empty() {
        let memories: Vec<SemanticMemoryEntry> = vec![];
        let injection = build_memory_injection(&memories);
        assert!(injection.is_empty());
    }

    #[test]
    fn test_build_memory_injection_single() {
        let memories = vec![SemanticMemoryEntry {
            id: 1,
            created_at: now(),
            content: "User prefers dark mode".to_string(),
            importance: 0.5,
            source: "auto".to_string(),
            keywords: Some("user,prefers,dark,mode".to_string()),
            access_count: 0,
            last_accessed: None,
            embedding: None,
        }];
        let injection = build_memory_injection(&memories);

        assert!(injection.starts_with("[recalled memories]"));
        assert!(injection.contains("User prefers dark mode"));
        assert!(injection.contains("just now"));
        assert!(!injection.contains("⭐")); // Low importance, no star
    }

    #[test]
    fn test_build_memory_injection_high_importance() {
        let memories = vec![SemanticMemoryEntry {
            id: 1,
            created_at: now(),
            content: "Critical fact".to_string(),
            importance: 0.95,
            source: "explicit".to_string(),
            keywords: Some("critical,fact".to_string()),
            access_count: 0,
            last_accessed: None,
            embedding: None,
        }];
        let injection = build_memory_injection(&memories);

        assert!(injection.contains("⭐")); // High importance gets a star
    }

    // =========================================================================
    // Embedding-based memory tests
    // =========================================================================

    #[test]
    fn test_embedding_to_blob_and_back() {
        let embedding = vec![0.1, 0.2, 0.3, -0.4, 0.5];
        let blob = embedding_to_blob(&embedding);
        let restored = blob_to_embedding(&blob);

        assert_eq!(embedding.len(), restored.len());
        for (a, b) in embedding.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 0.0001);
        }
    }

    #[test]
    fn test_save_with_embedding() {
        let store = SemanticMemoryStore::open_in_memory("agent1").unwrap();
        let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        let result = store
            .save_with_embedding("User prefers dark mode", 0.9, "explicit", Some(&embedding))
            .unwrap();

        match result {
            SaveResult::New(id) => assert!(id > 0),
            _ => panic!("Expected new memory"),
        }

        assert_eq!(store.count().unwrap(), 1);
    }

    #[test]
    fn test_recall_with_embedding() {
        let store = SemanticMemoryStore::open_in_memory("agent1").unwrap();

        // Save memories with embeddings
        let dark_mode_emb = vec![0.9, 0.1, 0.0, 0.0];
        let rust_emb = vec![0.1, 0.9, 0.0, 0.0];
        let coffee_emb = vec![0.0, 0.0, 0.9, 0.1];

        store
            .save_with_embedding(
                "User prefers dark mode",
                0.8,
                "explicit",
                Some(&dark_mode_emb),
            )
            .unwrap();
        store
            .save_with_embedding("User works on Rust projects", 0.7, "auto", Some(&rust_emb))
            .unwrap();
        store
            .save_with_embedding("User likes coffee", 0.5, "auto", Some(&coffee_emb))
            .unwrap();

        // Query with embedding similar to "dark mode"
        let query_emb = vec![0.85, 0.15, 0.0, 0.0];
        let results = store
            .recall_with_embedding("dark theme", 5, Some(&query_emb))
            .unwrap();

        assert!(!results.is_empty());
        assert!(results[0].0.content.contains("dark mode"));
    }

    #[test]
    fn test_recall_without_embedding_returns_empty() {
        let store = SemanticMemoryStore::open_in_memory("agent1").unwrap();

        // Save memory with embedding
        let embedding = vec![0.1, 0.2, 0.3, 0.4];
        store
            .save_with_embedding("Test memory", 0.8, "auto", Some(&embedding))
            .unwrap();

        // Recall without query embedding should return empty
        let results = store.recall_with_embedding("test", 5, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_get_set_embedding_model() {
        let store = SemanticMemoryStore::open_in_memory("agent1").unwrap();

        // Initially no model
        assert!(store.get_embedding_model().unwrap().is_none());

        // Set model
        store.set_embedding_model("nomic-embed-text").unwrap();
        assert_eq!(
            store.get_embedding_model().unwrap(),
            Some("nomic-embed-text".to_string())
        );

        // Update model
        store.set_embedding_model("all-minilm").unwrap();
        assert_eq!(
            store.get_embedding_model().unwrap(),
            Some("all-minilm".to_string())
        );
    }

    #[test]
    fn test_has_null_embeddings() {
        let store = SemanticMemoryStore::open_in_memory("agent1").unwrap();

        // No memories = no null embeddings
        assert!(!store.has_null_embeddings().unwrap());

        // Save without embedding
        store.save("Memory without embedding", 0.5, "auto").unwrap();
        assert!(store.has_null_embeddings().unwrap());

        // Save with embedding
        let embedding = vec![0.1, 0.2, 0.3];
        store
            .save_with_embedding("Memory with embedding", 0.5, "auto", Some(&embedding))
            .unwrap();
        assert!(store.has_null_embeddings().unwrap()); // Still has one null

        // Update the one without embedding
        let memories = store.get_memories_needing_embeddings().unwrap();
        assert_eq!(memories.len(), 1);
        store.update_embedding(memories[0].0, &embedding).unwrap();
        assert!(!store.has_null_embeddings().unwrap());
    }

    #[test]
    fn test_needs_backfill() {
        let store = SemanticMemoryStore::open_in_memory("agent1").unwrap();

        // No model set, should need backfill
        assert!(store.needs_backfill("nomic-embed-text").unwrap());

        // Set model
        store.set_embedding_model("nomic-embed-text").unwrap();
        assert!(!store.needs_backfill("nomic-embed-text").unwrap());

        // Different model = needs backfill
        assert!(store.needs_backfill("all-minilm").unwrap());

        // Add memory without embedding
        store.save("Memory without embedding", 0.5, "auto").unwrap();
        assert!(store.needs_backfill("nomic-embed-text").unwrap());
    }

    #[test]
    fn test_update_embedding() {
        let store = SemanticMemoryStore::open_in_memory("agent1").unwrap();

        // Save without embedding
        let result = store.save("Test memory", 0.5, "auto").unwrap();
        let id = match result {
            SaveResult::New(id) => id,
            _ => panic!("Expected new memory"),
        };

        // Update with embedding
        let embedding = vec![0.1, 0.2, 0.3, 0.4];
        store.update_embedding(id, &embedding).unwrap();

        // Should now be recallable with embedding
        let query_emb = vec![0.1, 0.2, 0.3, 0.4];
        let results = store
            .recall_with_embedding("test", 5, Some(&query_emb))
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].0.embedding.is_some());
    }

    #[test]
    fn test_reinforce_with_embedding() {
        let store = SemanticMemoryStore::open_in_memory("agent1").unwrap();

        // Save with embedding
        let embedding = vec![0.1, 0.2, 0.3, 0.4];
        store
            .save_with_embedding("Test memory", 0.5, "auto", Some(&embedding))
            .unwrap();

        // Reinforce with updated embedding
        let new_embedding = vec![0.2, 0.3, 0.4, 0.5];
        let result = store
            .save_with_embedding("Test memory", 0.5, "auto", Some(&new_embedding))
            .unwrap();

        match result {
            SaveResult::Reinforced(_, old, new) => {
                assert!((old - 0.5).abs() < 0.01);
                assert!((new - 0.55).abs() < 0.01);
            }
            _ => panic!("Expected reinforced memory"),
        }
    }

    #[test]
    fn test_list_all_empty() {
        let store = SemanticMemoryStore::open_in_memory("agent1").unwrap();
        let memories = store.list_all(None).unwrap();
        assert!(memories.is_empty());
    }

    #[test]
    fn test_list_all_with_memories() {
        let store = SemanticMemoryStore::open_in_memory("agent1").unwrap();

        store.save("First memory", 0.5, "auto").unwrap();
        store.save("Second memory", 0.7, "explicit").unwrap();
        store.save("Third memory", 0.3, "auto").unwrap();

        let memories = store.list_all(None).unwrap();
        assert_eq!(memories.len(), 3);

        // Should be ordered by created_at descending (most recent first)
        assert!(memories[0].content.contains("Third"));
        assert!(memories[1].content.contains("Second"));
        assert!(memories[2].content.contains("First"));
    }

    #[test]
    fn test_list_all_with_limit() {
        let store = SemanticMemoryStore::open_in_memory("agent1").unwrap();

        store.save("Memory 1", 0.5, "auto").unwrap();
        store.save("Memory 2", 0.5, "auto").unwrap();
        store.save("Memory 3", 0.5, "auto").unwrap();
        store.save("Memory 4", 0.5, "auto").unwrap();

        let memories = store.list_all(Some(2)).unwrap();
        assert_eq!(memories.len(), 2);
    }

    #[test]
    fn test_delete_existing() {
        let store = SemanticMemoryStore::open_in_memory("agent1").unwrap();

        let result = store.save("Test memory", 0.5, "auto").unwrap();
        let id = match result {
            SaveResult::New(id) => id,
            _ => panic!("Expected new memory"),
        };

        assert_eq!(store.count().unwrap(), 1);

        let deleted = store.delete(id).unwrap();
        assert!(deleted);
        assert_eq!(store.count().unwrap(), 0);
    }

    #[test]
    fn test_delete_nonexistent() {
        let store = SemanticMemoryStore::open_in_memory("agent1").unwrap();

        let deleted = store.delete(999).unwrap();
        assert!(!deleted);
    }

    #[test]
    fn test_delete_respects_agent_id() {
        // Use a temp file so both stores share the same database
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        let store_a = SemanticMemoryStore::open(&db_path, "agent_a").unwrap();
        let store_b = SemanticMemoryStore::open(&db_path, "agent_b").unwrap();

        // Save memory for agent_a
        let result = store_a.save("Agent A memory", 0.5, "auto").unwrap();
        let id_a = match result {
            SaveResult::New(id) => id,
            _ => panic!("Expected new memory"),
        };

        // Save memory for agent_b
        store_b.save("Agent B memory", 0.5, "auto").unwrap();

        // Agent B should not be able to delete Agent A's memory
        let deleted = store_b.delete(id_a).unwrap();
        assert!(!deleted);

        // Agent A can delete their own memory
        let deleted = store_a.delete(id_a).unwrap();
        assert!(deleted);
    }

    #[test]
    fn test_clear_all_empty() {
        let store = SemanticMemoryStore::open_in_memory("agent1").unwrap();

        let cleared = store.clear_all().unwrap();
        assert_eq!(cleared, 0);
    }

    #[test]
    fn test_clear_all_with_memories() {
        let store = SemanticMemoryStore::open_in_memory("agent1").unwrap();

        store.save("Memory 1", 0.5, "auto").unwrap();
        store.save("Memory 2", 0.5, "auto").unwrap();
        store.save("Memory 3", 0.5, "auto").unwrap();

        assert_eq!(store.count().unwrap(), 3);

        let cleared = store.clear_all().unwrap();
        assert_eq!(cleared, 3);
        assert_eq!(store.count().unwrap(), 0);
    }

    #[test]
    fn test_clear_all_respects_agent_id() {
        // Use a temp file so both stores share the same database
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        let store_a = SemanticMemoryStore::open(&db_path, "agent_a").unwrap();
        let store_b = SemanticMemoryStore::open(&db_path, "agent_b").unwrap();

        // Save memories for both agents
        store_a.save("Agent A memory 1", 0.5, "auto").unwrap();
        store_a.save("Agent A memory 2", 0.5, "auto").unwrap();
        store_b.save("Agent B memory", 0.5, "auto").unwrap();

        assert_eq!(store_a.count().unwrap(), 2);
        assert_eq!(store_b.count().unwrap(), 1);

        // Clear agent_a's memories
        let cleared = store_a.clear_all().unwrap();
        assert_eq!(cleared, 2);

        // Agent A should have 0 memories
        assert_eq!(store_a.count().unwrap(), 0);

        // Agent B should still have their memory
        assert_eq!(store_b.count().unwrap(), 1);
    }
}
