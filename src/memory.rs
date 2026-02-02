use async_trait::async_trait;
use rusqlite::{Connection, params};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use crate::debug;

fn now() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64
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
            Some(p) => self.data.keys().filter(|k| k.starts_with(p)).cloned().collect(),
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
    /// Open or create a SQLite database at the given path
    pub fn open(path: &str, agent_id: &str) -> Result<Self, MemoryError> {
        let conn = Connection::open(path)
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;
        
        let memory = SqliteMemory {
            conn: Arc::new(Mutex::new(conn)),
            agent_id: agent_id.to_string(),
        };
        memory.init_schema()?;
        Ok(memory)
    }

    /// Create an in-memory SQLite database (useful for testing)
    pub fn open_in_memory(agent_id: &str) -> Result<Self, MemoryError> {
        let conn = Connection::open_in_memory()
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;
        
        let memory = SqliteMemory {
            conn: Arc::new(Mutex::new(conn)),
            agent_id: agent_id.to_string(),
        };
        memory.init_schema()?;
        Ok(memory)
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
            CREATE INDEX IF NOT EXISTS idx_memories_time ON memories(agent_id, updated_at);"
        ).map_err(|e| MemoryError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Query memories by time range (episodic memory)
    pub async fn query_by_time(&self, since: u64, until: Option<u64>) -> Result<Vec<(String, MemoryEntry)>, MemoryError> {
        let conn = self.conn.clone();
        let agent_id = self.agent_id.clone();
        
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            // Use i64 for SQLite compatibility (i64::MAX is ~292 billion years from epoch, plenty)
            let since_i64 = since as i64;
            let until_i64 = until.map(|u| u as i64).unwrap_or(i64::MAX);
            
            let mut stmt = conn.prepare(
                "SELECT key, value, created_at, updated_at FROM memories 
                 WHERE agent_id = ?1 AND updated_at >= ?2 AND updated_at <= ?3
                 ORDER BY updated_at ASC"
            ).map_err(|e| MemoryError::StorageError(e.to_string()))?;
            
            let rows = stmt.query_map(params![agent_id, since_i64, until_i64], |row| {
                let key: String = row.get(0)?;
                let value_str: String = row.get(1)?;
                let created_at: i64 = row.get(2)?;
                let updated_at: i64 = row.get(3)?;
                Ok((key, value_str, created_at, updated_at))
            }).map_err(|e| MemoryError::StorageError(e.to_string()))?;
            
            let mut results = Vec::new();
            for row in rows {
                let (key, value_str, created_at, updated_at) = row
                    .map_err(|e| MemoryError::StorageError(e.to_string()))?;
                let value: Value = serde_json::from_str(&value_str)
                    .map_err(|e| MemoryError::SerializationError(e.to_string()))?;
                results.push((key, MemoryEntry { value, created_at, updated_at }));
            }
            Ok(results)
        }).await.map_err(|e| MemoryError::StorageError(e.to_string()))?
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
            let existing: Option<i64> = conn.query_row(
                "SELECT created_at FROM memories WHERE agent_id = ?1 AND key = ?2",
                params![agent_id, key],
                |row| row.get(0)
            ).ok();
            
            let created_at = existing.unwrap_or(timestamp);
            
            conn.execute(
                "INSERT OR REPLACE INTO memories (agent_id, key, value, created_at, updated_at) 
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![agent_id, key, value_str, created_at, timestamp]
            ).map_err(|e| MemoryError::StorageError(e.to_string()))?;
            
            Ok(())
        }).await.map_err(|e| MemoryError::StorageError(e.to_string()))?
    }

    async fn delete(&mut self, key: &str) -> bool {
        let conn = self.conn.clone();
        let agent_id = self.agent_id.clone();
        let key = key.to_string();
        
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let rows = conn.execute(
                "DELETE FROM memories WHERE agent_id = ?1 AND key = ?2",
                params![agent_id, key]
            ).unwrap_or(0);
            rows > 0
        }).await.unwrap_or(false)
    }

    async fn list_keys(&self, prefix: Option<&str>) -> Vec<String> {
        let conn = self.conn.clone();
        let agent_id = self.agent_id.clone();
        let prefix = prefix.map(|s| s.to_string());
        
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let mut keys = Vec::new();
            
            match &prefix {
                Some(p) => {
                    let mut stmt = match conn.prepare(
                        "SELECT key FROM memories WHERE agent_id = ?1 AND key LIKE ?2"
                    ) {
                        Ok(s) => s,
                        Err(_) => return keys,
                    };
                    let pattern = format!("{}%", p);
                    if let Ok(rows) = stmt.query_map(params![agent_id, pattern], |row| row.get::<_, String>(0)) {
                        keys = rows.filter_map(|r| r.ok()).collect();
                    }
                }
                None => {
                    let mut stmt = match conn.prepare(
                        "SELECT key FROM memories WHERE agent_id = ?1"
                    ) {
                        Ok(s) => s,
                        Err(_) => return keys,
                    };
                    if let Ok(rows) = stmt.query_map(params![agent_id], |row| row.get::<_, String>(0)) {
                        keys = rows.filter_map(|r| r.ok()).collect();
                    }
                }
            }
            
            keys
        }).await.unwrap_or_default()
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
}

/// Result of a memory save operation.
#[derive(Debug, Clone)]
pub enum SaveResult {
    /// New memory created (id)
    New(i64),
    /// Existing memory reinforced (id, old_importance, new_importance)
    Reinforced(i64, f64, f64),
}

/// Semantic memory store with keyword-based search and relevance scoring.
pub struct SemanticMemoryStore {
    conn: Arc<Mutex<Connection>>,
    agent_id: String,
}

/// Common English stopwords to filter from keyword extraction.
const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "the", "this", "but", "they",
    "have", "had", "what", "when", "where", "who", "which", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "no", "not", "only", "own", "same", "so", "than", "too", "very",
    "can", "just", "should", "now", "i", "you", "we", "my", "your", "our",
    "me", "him", "her", "them", "us", "do", "does", "did", "doing", "done",
    "would", "could", "might", "must", "shall", "may", "am", "been", "being",
    "over", "under", "into", "out", "up", "down", "about", "after", "before",
];

/// Extract meaningful keywords from text by filtering stopwords.
fn extract_keywords(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && w.len() > 2 && !STOPWORDS.contains(w))
        .map(|s| s.to_string())
        .collect()
}

/// Calculate keyword overlap score between query keywords and memory keywords.
fn keyword_overlap(query_keywords: &[String], memory_keywords: &Option<String>) -> f64 {
    let mem_kws = match memory_keywords {
        Some(kws) => kws.split(',').map(|s| s.trim().to_lowercase()).collect::<Vec<_>>(),
        None => return 0.0,
    };

    if mem_kws.is_empty() || query_keywords.is_empty() {
        return 0.0;
    }

    let matches = query_keywords
        .iter()
        .filter(|qk| mem_kws.iter().any(|mk| mk.contains(qk.as_str()) || qk.contains(mk.as_str())))
        .count();

    matches as f64 / query_keywords.len().max(1) as f64
}

/// Calculate recency score with exponential decay (half-life of ~7 days).
fn recency_score(created_at: i64, now: i64) -> f64 {
    let age_hours = (now - created_at) as f64 / 3600.0;
    // Exponential decay: half-life of ~7 days (168 hours)
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
    /// Open or create a semantic memory database at the given path.
    pub fn open(path: &std::path::Path, agent_id: &str) -> Result<Self, MemoryError> {
        let conn = Connection::open(path)
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        let store = SemanticMemoryStore {
            conn: Arc::new(Mutex::new(conn)),
            agent_id: agent_id.to_string(),
        };
        store.init_schema()?;
        Ok(store)
    }

    /// Create an in-memory semantic memory store (useful for testing).
    pub fn open_in_memory(agent_id: &str) -> Result<Self, MemoryError> {
        let conn = Connection::open_in_memory()
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        let store = SemanticMemoryStore {
            conn: Arc::new(Mutex::new(conn)),
            agent_id: agent_id.to_string(),
        };
        store.init_schema()?;
        Ok(store)
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
                last_accessed INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_semantic_agent ON semantic_memories(agent_id);
            CREATE INDEX IF NOT EXISTS idx_semantic_keywords ON semantic_memories(agent_id, keywords);
            CREATE INDEX IF NOT EXISTS idx_semantic_created ON semantic_memories(agent_id, created_at);"
        ).map_err(|e| MemoryError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Save a new semantic memory, or reinforce an existing one if duplicate.
    /// Reinforcing boosts importance by 0.05 (capped at 1.0) and refreshes timestamp.
    pub fn save(&self, content: &str, importance: f64, source: &str) -> Result<SaveResult, MemoryError> {
        let keywords = extract_keywords(content).join(",");
        let timestamp = now();

        let conn = self.conn.lock().unwrap();
        
        // Check for existing exact match
        let existing: Option<(i64, f64)> = conn.query_row(
            "SELECT id, importance FROM semantic_memories WHERE agent_id = ?1 AND content = ?2",
            params![self.agent_id, content],
            |row| Ok((row.get(0)?, row.get(1)?))
        ).ok();
        
        if let Some((id, old_importance)) = existing {
            // Reinforce: boost importance by 0.05 (cap at 1.0) and refresh timestamp
            let new_importance = (old_importance + 0.05).min(1.0);
            conn.execute(
                "UPDATE semantic_memories SET importance = ?1, created_at = ?2 WHERE id = ?3",
                params![new_importance, timestamp, id]
            ).map_err(|e| MemoryError::StorageError(e.to_string()))?;
            
            debug::log(&format!("[memory] REINFORCE #{}: \"{}\" (importance {} → {})", 
                id, content, old_importance, new_importance));
            Ok(SaveResult::Reinforced(id, old_importance, new_importance))
        } else {
            // New memory
            conn.execute(
                "INSERT INTO semantic_memories (agent_id, created_at, content, importance, source, keywords, access_count)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0)",
                params![self.agent_id, timestamp, content, importance, source, keywords]
            ).map_err(|e| MemoryError::StorageError(e.to_string()))?;

            let id = conn.last_insert_rowid();
            debug::log(&format!("[memory] SAVE #{}: \"{}\" (importance={}, source={}, keywords={})", 
                id, content, importance, source, keywords));
            Ok(SaveResult::New(id))
        }
    }

    /// Recall relevant memories for a query, scored by relevance × recency × importance.
    pub fn recall(&self, query: &str, limit: usize) -> Result<Vec<SemanticMemoryEntry>, MemoryError> {
        let query_keywords = extract_keywords(query);
        debug::log(&format!("[memory] RECALL query=\"{}\" keywords={:?} limit={}", 
            query.chars().take(100).collect::<String>(), query_keywords, limit));

        if query_keywords.is_empty() {
            debug::log("[memory] RECALL: no keywords extracted, returning empty");
            return Ok(Vec::new());
        }

        // Build LIKE patterns for each keyword
        let patterns: Vec<String> = query_keywords
            .iter()
            .map(|k| format!("%{}%", k))
            .collect();

        let conn = self.conn.lock().unwrap();

        // Build a query that matches any keyword
        let placeholders: Vec<String> = (0..patterns.len())
            .map(|i| format!("keywords LIKE ?{}", i + 2))
            .collect();
        let where_clause = placeholders.join(" OR ");

        let sql = format!(
            "SELECT id, created_at, content, importance, source, keywords, access_count, last_accessed
             FROM semantic_memories
             WHERE agent_id = ?1 AND ({})
             ORDER BY created_at DESC
             LIMIT 100",
            where_clause
        );

        let mut stmt = conn.prepare(&sql)
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        // Build params: agent_id + patterns
        let mut param_values: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        param_values.push(Box::new(self.agent_id.clone()));
        for pattern in &patterns {
            param_values.push(Box::new(pattern.clone()));
        }
        let params_ref: Vec<&dyn rusqlite::ToSql> = param_values.iter().map(|b| b.as_ref()).collect();

        let rows = stmt.query_map(params_ref.as_slice(), |row| {
            Ok(SemanticMemoryEntry {
                id: row.get(0)?,
                created_at: row.get(1)?,
                content: row.get(2)?,
                importance: row.get(3)?,
                source: row.get(4)?,
                keywords: row.get(5)?,
                access_count: row.get(6)?,
                last_accessed: row.get(7)?,
            })
        }).map_err(|e| MemoryError::StorageError(e.to_string()))?;

        let candidates: Vec<SemanticMemoryEntry> = rows
            .filter_map(|r| r.ok())
            .collect();

        // Score and sort
        let current_time = now();
        let mut scored: Vec<(SemanticMemoryEntry, f64)> = candidates
            .into_iter()
            .map(|m| {
                let relevance = keyword_overlap(&query_keywords, &m.keywords);
                let recency = recency_score(m.created_at, current_time);
                let score = relevance * recency * m.importance;
                (m, score)
            })
            .filter(|(_, score)| *score > 0.0)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        // Log what we found
        if scored.is_empty() {
            debug::log("[memory] RECALL: no matching memories found");
        } else {
            debug::log(&format!("[memory] RECALL: found {} memories:", scored.len()));
            for (m, score) in &scored {
                debug::log(&format!("[memory]   #{} (score={:.3}): \"{}\"", 
                    m.id, score, m.content.chars().take(80).collect::<String>()));
            }
        }

        // Mark accessed
        let ids: Vec<i64> = scored.iter().map(|(m, _)| m.id).collect();
        drop(stmt);
        for id in &ids {
            self.mark_accessed_internal(&conn, *id)?;
        }

        Ok(scored.into_iter().map(|(m, _)| m).collect())
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
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM semantic_memories WHERE agent_id = ?1",
            params![self.agent_id],
            |row| row.get(0)
        ).map_err(|e| MemoryError::StorageError(e.to_string()))?;
        Ok(count)
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

    let cleaned = re.replace_all(output, "").to_string();
    let cleaned = cleaned.trim().to_string();

    if !memories.is_empty() {
        debug::log(&format!("[memory] EXTRACT: found {} [REMEMBER:] tags", memories.len()));
    }

    (cleaned, memories)
}

/// Build the memory injection string for prepending to context.
pub fn build_memory_injection(memories: &[SemanticMemoryEntry]) -> String {
    if memories.is_empty() {
        return String::new();
    }

    let mut injection = String::from("[Relevant memories]\n");
    for m in memories {
        let age = format_age(m.created_at);
        let flag = if m.importance > 0.8 { " ⭐" } else { "" };
        injection.push_str(&format!("- ({}) {}{}\n", age, m.content, flag));
    }
    injection.push('\n');

    debug::log(&format!("[memory] INJECT: {} memories into context:\n{}", memories.len(), injection));

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
    fn test_keyword_overlap_full_match() {
        let query_kws = vec!["rust".to_string(), "programming".to_string()];
        let mem_kws = Some("rust,programming,language".to_string());
        let score = keyword_overlap(&query_kws, &mem_kws);
        assert!((score - 1.0).abs() < 0.01); // Both query keywords match
    }

    #[test]
    fn test_keyword_overlap_partial_match() {
        let query_kws = vec!["rust".to_string(), "python".to_string()];
        let mem_kws = Some("rust,programming".to_string());
        let score = keyword_overlap(&query_kws, &mem_kws);
        assert!((score - 0.5).abs() < 0.01); // Only "rust" matches
    }

    #[test]
    fn test_keyword_overlap_no_match() {
        let query_kws = vec!["rust".to_string()];
        let mem_kws = Some("python,javascript".to_string());
        let score = keyword_overlap(&query_kws, &mem_kws);
        assert!(score < 0.01);
    }

    #[test]
    fn test_keyword_overlap_empty() {
        let query_kws: Vec<String> = vec![];
        let mem_kws = Some("rust".to_string());
        assert!(keyword_overlap(&query_kws, &mem_kws) < 0.01);

        let query_kws = vec!["rust".to_string()];
        let mem_kws: Option<String> = None;
        assert!(keyword_overlap(&query_kws, &mem_kws) < 0.01);
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

        // Save a memory
        let result = store.save("User prefers dark mode for coding", 0.9, "explicit").unwrap();
        let id = match result {
            SaveResult::New(id) => id,
            SaveResult::Reinforced(id, _, _) => id,
        };
        assert!(id > 0);

        // Recall with matching query
        let results = store.recall("dark mode preference", 5).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("dark mode"));
        assert!((results[0].importance - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_semantic_memory_recall_multiple() {
        let store = SemanticMemoryStore::open_in_memory("agent1").unwrap();

        store.save("User prefers dark mode", 0.8, "explicit").unwrap();
        store.save("User works on Rust projects", 0.7, "auto").unwrap();
        store.save("User likes coffee", 0.5, "auto").unwrap();

        // Query should match the first memory
        let results = store.recall("dark theme preference", 5).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].content.contains("dark"));
    }

    #[test]
    fn test_semantic_memory_recall_no_match() {
        let store = SemanticMemoryStore::open_in_memory("agent1").unwrap();

        store.save("User prefers dark mode", 0.8, "explicit").unwrap();

        // Query with no matching keywords
        let results = store.recall("xyz123 completely unrelated", 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_semantic_memory_recall_limit() {
        let store = SemanticMemoryStore::open_in_memory("agent1").unwrap();

        // Save multiple memories with the same keyword
        for i in 0..10 {
            store.save(&format!("Memory about coding number {}", i), 0.5, "auto").unwrap();
        }

        // Should only return the limited number
        let results = store.recall("coding memory", 3).unwrap();
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_semantic_memory_importance_scoring() {
        let store = SemanticMemoryStore::open_in_memory("agent1").unwrap();

        // Save with different importance
        store.save("Low importance coding fact", 0.3, "auto").unwrap();
        store.save("High importance coding fact", 0.95, "explicit").unwrap();

        let results = store.recall("coding fact", 5).unwrap();
        assert!(!results.is_empty());
        // Higher importance should rank higher (when other factors are equal)
        assert!(results[0].importance > 0.9);
    }

    #[test]
    fn test_semantic_memory_access_tracking() {
        let store = SemanticMemoryStore::open_in_memory("agent1").unwrap();

        store.save("Test memory for access tracking", 0.5, "auto").unwrap();

        // First recall
        let results1 = store.recall("access tracking", 5).unwrap();
        assert_eq!(results1[0].access_count, 0); // Not incremented yet at read time

        // Second recall should show incremented count
        let results2 = store.recall("access tracking", 5).unwrap();
        assert_eq!(results2[0].access_count, 1);

        // Third recall
        let results3 = store.recall("access tracking", 5).unwrap();
        assert_eq!(results3[0].access_count, 2);
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
        let memories = vec![
            SemanticMemoryEntry {
                id: 1,
                created_at: now(),
                content: "User prefers dark mode".to_string(),
                importance: 0.5,
                source: "auto".to_string(),
                keywords: Some("user,prefers,dark,mode".to_string()),
                access_count: 0,
                last_accessed: None,
            },
        ];
        let injection = build_memory_injection(&memories);

        assert!(injection.starts_with("[Relevant memories]"));
        assert!(injection.contains("User prefers dark mode"));
        assert!(injection.contains("just now"));
        assert!(!injection.contains("⭐")); // Low importance, no star
    }

    #[test]
    fn test_build_memory_injection_high_importance() {
        let memories = vec![
            SemanticMemoryEntry {
                id: 1,
                created_at: now(),
                content: "Critical fact".to_string(),
                importance: 0.95,
                source: "explicit".to_string(),
                keywords: Some("critical,fact".to_string()),
                access_count: 0,
                last_accessed: None,
            },
        ];
        let injection = build_memory_injection(&memories);

        assert!(injection.contains("⭐")); // High importance gets a star
    }
}