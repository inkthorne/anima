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
        let time_after_first = now();

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
}