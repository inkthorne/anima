use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
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