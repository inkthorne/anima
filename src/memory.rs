use async_trait::async_trait;
use rusqlite::{Connection, params};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
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
    pub created_at: u64,
    pub updated_at: u64,
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
                Ok((key, value_str, created_at as u64, updated_at as u64))
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
            let result: Result<(String, u64, u64), _> = conn.query_row(
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
            let existing: Option<u64> = conn.query_row(
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