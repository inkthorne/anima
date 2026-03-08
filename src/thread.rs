use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::agent_dir::{AgentDir, AgentDirError, create_llm_from_config};
use crate::llm::{ChatMessage, LLM, LLMError, LLMResponse};

#[derive(Debug, thiserror::Error)]
pub enum ThreadError {
    #[error("Agent directory error: {0}")]
    AgentDir(#[from] AgentDirError),
    #[error("LLM error: {0}")]
    Llm(#[from] LLMError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Thread '{0}' already exists")]
    AlreadyExists(String),
    #[error("Thread '{0}' not found — use `anima thread create {0} <agent>` first")]
    NotFound(String),
}

/// On-disk format for a persistent thread.
#[derive(Debug, Serialize, Deserialize)]
struct ThreadFile {
    agent: String,
    history: Vec<ChatMessage>,
}

/// Return `~/.anima/threads/`.
fn threads_dir() -> PathBuf {
    dirs::home_dir()
        .expect("Could not determine home directory")
        .join(".anima")
        .join("threads")
}

/// Return `~/.anima/threads/<name>.json`.
fn thread_path(name: &str) -> PathBuf {
    threads_dir().join(format!("{}.json", name))
}

pub struct AnimaThread {
    name: String,
    agent_name: String,
    llm: Arc<dyn LLM>,
    system_prompt: Option<String>,
    history: Vec<ChatMessage>,
}

impl AnimaThread {
    /// Create a new persistent thread bound to an agent.
    /// Errors if the thread already exists.
    pub fn create(name: &str, agent_dir: &AgentDir) -> Result<(), ThreadError> {
        let path = thread_path(name);
        if path.exists() {
            return Err(ThreadError::AlreadyExists(name.to_string()));
        }
        let dir = threads_dir();
        std::fs::create_dir_all(&dir)?;
        let file = ThreadFile {
            agent: agent_dir.config.agent.name.clone(),
            history: Vec::new(),
        };
        atomic_write(&path, &file)?;
        Ok(())
    }

    /// Load an existing thread from disk and build the LLM client.
    pub async fn load(name: &str, resolve_agent: impl FnOnce(&str) -> PathBuf) -> Result<Self, ThreadError> {
        let path = thread_path(name);
        if !path.exists() {
            return Err(ThreadError::NotFound(name.to_string()));
        }
        let data = std::fs::read_to_string(&path)?;
        let file: ThreadFile = serde_json::from_str(&data)?;

        let agent_path = resolve_agent(&file.agent);
        let agent_dir = AgentDir::load(&agent_path)?;
        let llm_config = agent_dir.resolve_llm_config()?;
        let api_key = AgentDir::api_key_for_config(&llm_config)?;
        let llm = create_llm_from_config(&llm_config, api_key).await?;
        let system_prompt = agent_dir.load_system()?;

        Ok(Self {
            name: name.to_string(),
            agent_name: file.agent,
            llm,
            system_prompt,
            history: file.history,
        })
    }

    /// Delete a thread's JSON file.
    pub fn clear(name: &str) -> Result<(), ThreadError> {
        let path = thread_path(name);
        if !path.exists() {
            return Err(ThreadError::NotFound(name.to_string()));
        }
        std::fs::remove_file(&path)?;
        Ok(())
    }

    /// Send a message and get a response. Persists history to disk.
    pub async fn send(&mut self, message: &str) -> Result<LLMResponse, ThreadError> {
        let mut messages = Vec::new();
        if let Some(ref system) = self.system_prompt {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: Some(system.clone()),
                tool_call_id: None,
                tool_calls: None,
            });
        }
        messages.extend(self.history.clone());
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: Some(message.to_string()),
            tool_call_id: None,
            tool_calls: None,
        });

        let response = self.llm.chat_complete(messages, None).await?;
        let content = response.content.clone().unwrap_or_default();

        self.history.push(ChatMessage {
            role: "user".to_string(),
            content: Some(message.to_string()),
            tool_call_id: None,
            tool_calls: None,
        });
        self.history.push(ChatMessage {
            role: "assistant".to_string(),
            content: Some(content),
            tool_call_id: None,
            tool_calls: None,
        });

        self.save()?;
        Ok(response)
    }

    /// Send a message and stream tokens through the channel as they arrive.
    /// Returns the full `LLMResponse` (including usage). Persists history to disk.
    pub async fn send_stream(
        &mut self,
        message: &str,
        tx: mpsc::Sender<String>,
    ) -> Result<LLMResponse, ThreadError> {
        let mut messages = Vec::new();
        if let Some(ref system) = self.system_prompt {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: Some(system.clone()),
                tool_call_id: None,
                tool_calls: None,
            });
        }
        messages.extend(self.history.clone());
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: Some(message.to_string()),
            tool_call_id: None,
            tool_calls: None,
        });

        let response = self.llm.chat_complete_stream(messages, None, tx).await?;
        let content = response.content.clone().unwrap_or_default();

        self.history.push(ChatMessage {
            role: "user".to_string(),
            content: Some(message.to_string()),
            tool_call_id: None,
            tool_calls: None,
        });
        self.history.push(ChatMessage {
            role: "assistant".to_string(),
            content: Some(content),
            tool_call_id: None,
            tool_calls: None,
        });

        self.save()?;
        Ok(response)
    }

    /// Persist current history to disk with atomic write.
    fn save(&self) -> Result<(), ThreadError> {
        let file = ThreadFile {
            agent: self.agent_name.clone(),
            history: self.history.clone(),
        };
        atomic_write(&thread_path(&self.name), &file)?;
        Ok(())
    }

    pub fn agent_name(&self) -> &str {
        &self.agent_name
    }

    /// List all threads from `~/.anima/threads/`.
    /// Returns `Vec<(name, agent, message_count)>` sorted by name.
    pub fn list_all() -> Result<Vec<(String, String, usize)>, ThreadError> {
        let dir = threads_dir();
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let mut results = Vec::new();
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                let name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();
                let data = std::fs::read_to_string(&path)?;
                let file: ThreadFile = serde_json::from_str(&data)?;
                results.push((name, file.agent, file.history.len()));
            }
        }
        results.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(results)
    }
}

/// Write JSON atomically: write to `.tmp` then rename.
fn atomic_write(path: &PathBuf, file: &ThreadFile) -> Result<(), ThreadError> {
    let tmp = path.with_extension("json.tmp");
    let json = serde_json::to_string_pretty(file)?;
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_error_display() {
        let err = ThreadError::Llm(LLMError::permanent("test"));
        assert!(err.to_string().contains("LLM error"));
    }

    #[test]
    fn test_threads_dir() {
        let dir = threads_dir();
        assert!(dir.ends_with(".anima/threads"));
    }

    #[test]
    fn test_thread_path() {
        let path = thread_path("mythread");
        assert!(path.ends_with("threads/mythread.json"));
    }

    #[test]
    fn test_thread_file_serialization() {
        let file = ThreadFile {
            agent: "hal".to_string(),
            history: vec![
                ChatMessage {
                    role: "user".to_string(),
                    content: Some("hello".to_string()),
                    tool_call_id: None,
                    tool_calls: None,
                },
            ],
        };
        let json = serde_json::to_string(&file).unwrap();
        let parsed: ThreadFile = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.agent, "hal");
        assert_eq!(parsed.history.len(), 1);
    }

    #[test]
    fn test_not_found_error() {
        let result = AnimaThread::clear("nonexistent_thread_xyz_123");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("not found"));
    }
}
