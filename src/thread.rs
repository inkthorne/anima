use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use regex::Regex;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::agent_dir::{AgentDir, AgentDirError, create_llm_from_config};
use crate::llm::{ChatMessage, LLM, LLMError, LLMResponse};

const STREAM_MAX_RETRIES: u32 = 3;
const STREAM_RETRY_BASE_DELAY_SECS: u64 = 2;
use crate::tools::python::run_python;
use crate::tools::shell::DEFAULT_MEM_LIMIT_BYTES;

/// Regex for extracting `<python>...</python>` blocks (with optional `label` attribute).
static PYTHON_BLOCK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(?s)<python(?:\s+label="([^"]*)")?>\s*\n?(.*?)\n?\s*</python>"#).unwrap());

/// A parsed `<python>` block with optional label.
struct PythonBlock {
    code: String,
    label: Option<String>,
}

/// Extract `<python>...</python>` blocks from text.
/// Returns Vec of parsed blocks.
fn extract_python_blocks(text: &str) -> Vec<PythonBlock> {
    PYTHON_BLOCK_RE
        .captures_iter(text)
        .map(|cap| PythonBlock {
            label: cap.get(1).map(|m| m.as_str().to_string()),
            code: cap[2].to_string(),
        })
        .collect()
}

/// Execute python blocks and format results as `<python-output>` XML.
async fn execute_and_format_python(
    blocks: &[PythonBlock],
    agent_dir: Option<&Path>,
) -> String {
    let timeout = Duration::from_secs(30);
    let mem_limit = Some(DEFAULT_MEM_LIMIT_BYTES);
    let mut parts = Vec::new();
    for block in blocks {
        let open_tag = match &block.label {
            Some(label) => format!("<python-output label=\"{}\">", label),
            None => "<python-output>".to_string(),
        };
        match run_python(&block.code, timeout, mem_limit, agent_dir).await {
            Ok(val) => {
                let stdout = val["stdout"].as_str().unwrap_or("");
                let stderr = val["stderr"].as_str().unwrap_or("");
                let exit_code = val["exit_code"].as_i64().unwrap_or(0);
                let mut part = format!("{}\n{}", open_tag, stdout);
                if !stderr.is_empty() {
                    part.push_str(&format!("<stderr>{}</stderr>\n", stderr));
                }
                if exit_code != 0 {
                    part.push_str(&format!("<exit-code>{}</exit-code>\n", exit_code));
                }
                part.push_str("</python-output>");
                parts.push(part);
            }
            Err(e) => {
                parts.push(format!("{}\n<error>{}</error>\n</python-output>", open_tag, e));
            }
        }
    }
    parts.join("\n---\n")
}

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

    /// Clear a thread's history but preserve the agent binding.
    pub fn clear(name: &str) -> Result<(), ThreadError> {
        let path = thread_path(name);
        if !path.exists() {
            return Err(ThreadError::NotFound(name.to_string()));
        }
        let data = std::fs::read_to_string(&path)?;
        let mut file: ThreadFile = serde_json::from_str(&data)?;
        file.history.clear();
        atomic_write(&path, &file)?;
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

    /// Send a message, stream tokens, and automatically execute `<python>` blocks.
    /// If the LLM response contains `<python>` blocks, they are executed and the
    /// results fed back as a tool message, then the LLM is called again in a loop.
    /// Returns the final `LLMResponse` (from the last LLM call).
    pub async fn send_stream_with_python(
        &mut self,
        message: &str,
        tx: mpsc::Sender<String>,
        agent_dir: Option<PathBuf>,
    ) -> Result<LLMResponse, ThreadError> {
        let started = std::time::Instant::now();
        let mut turn = 0u32;
        let mut message_opt = Some(message.to_string());

        loop {
            turn += 1;
            let response = self.step_stream(
                message_opt.as_deref(),
                tx.clone(),
                agent_dir.clone(),
            ).await?;

            // Send usage with elapsed/turn info
            Self::send_usage(&tx, &response, started.elapsed(), turn).await;

            // If step_stream didn't find python blocks, we're done
            let content = response.content.as_deref().unwrap_or_default();
            if extract_python_blocks(content).is_empty() {
                return Ok(response);
            }

            message_opt = None; // continuation from here
        }
    }

    /// Format and send usage stats through the channel with elapsed time and turn number.
    async fn send_usage(
        tx: &mpsc::Sender<String>,
        response: &LLMResponse,
        elapsed: std::time::Duration,
        turn: u32,
    ) {
        if let Some(ref usage) = response.usage {
            if usage.prompt_tokens > 0 || usage.completion_tokens > 0 {
                let cached_str = if let Some(cached) = usage.cached_tokens {
                    format!(", {} cached", cached)
                } else {
                    String::new()
                };
                let _ = tx.send(format!(
                    "<usage>{} in \u{2192} {} out{} | {:.1}s | turn {}</usage>",
                    usage.prompt_tokens, usage.completion_tokens, cached_str,
                    elapsed.as_secs_f64(), turn
                )).await;
            }
        }
    }

    /// Execute one step: single LLM call (with retry) + python execution if response contains blocks.
    /// If `message` is Some, appends it as a user message first.
    /// If `message` is None, continues from current history (e.g., after a tool result).
    /// Does NOT send usage stats — caller is responsible for formatting/sending usage.
    pub async fn step_stream(
        &mut self,
        message: Option<&str>,
        tx: mpsc::Sender<String>,
        agent_dir: Option<PathBuf>,
    ) -> Result<LLMResponse, ThreadError> {
        // LLM call with retry on retryable errors
        let response = {
            let mut attempt = 0u32;
            loop {
                let result = if let Some(msg) = message {
                    self.send_stream(msg, tx.clone()).await
                } else {
                    self.send_stream_continuation(tx.clone()).await
                };
                match result {
                    Ok(resp) => break resp,
                    Err(ThreadError::Llm(ref llm_err)) if llm_err.is_retryable && attempt < STREAM_MAX_RETRIES => {
                        attempt += 1;
                        let delay = STREAM_RETRY_BASE_DELAY_SECS * (1 << (attempt - 1));
                        let _ = tx.send(format!(
                            "<retry>Stream error, retrying in {}s (attempt {}/{})...</retry>",
                            delay, attempt, STREAM_MAX_RETRIES
                        )).await;
                        tokio::time::sleep(Duration::from_secs(delay)).await;
                        continue;
                    }
                    Err(e) => return Err(e),
                }
            }
        };

        // Execute python blocks if any
        let content = response.content.clone().unwrap_or_default();
        let blocks = extract_python_blocks(&content);

        if !blocks.is_empty() {
            let result = execute_and_format_python(&blocks, agent_dir.as_deref()).await;
            let _ = tx.send(format!("\n{}\n", result)).await;

            self.history.push(ChatMessage {
                role: "tool".to_string(),
                content: Some(result),
                tool_call_id: None,
                tool_calls: None,
            });
            self.save()?;
        }

        Ok(response)
    }

    /// Continue a conversation without adding a new user message.
    /// Used for python-block loop: history already has the tool result,
    /// just need to get the next LLM response.
    async fn send_stream_continuation(
        &mut self,
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

        let response = self.llm.chat_complete_stream(messages, None, tx).await?;
        let content = response.content.clone().unwrap_or_default();

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

    /// Fork a thread: copy its history and agent binding to a new thread name.
    /// Returns the agent name on success.
    pub fn fork(source: &str, new_name: &str) -> Result<String, ThreadError> {
        let src_path = thread_path(source);
        if !src_path.exists() {
            return Err(ThreadError::NotFound(source.to_string()));
        }
        let dst_path = thread_path(new_name);
        if dst_path.exists() {
            return Err(ThreadError::AlreadyExists(new_name.to_string()));
        }
        let data = std::fs::read_to_string(&src_path)?;
        let file: ThreadFile = serde_json::from_str(&data)?;
        std::fs::create_dir_all(threads_dir())?;
        atomic_write(&dst_path, &file)?;
        Ok(file.agent)
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

    #[test]
    fn test_extract_python_blocks_single() {
        let input = "Here is the result:\n<python>\nprint('hello')\n</python>\nDone.";
        let blocks = extract_python_blocks(input);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].code, "print('hello')");
        assert!(blocks[0].label.is_none());
    }

    #[test]
    fn test_extract_python_blocks_multiple() {
        let input = "<python>\nx = 1\n</python>\ntext\n<python>\ny = 2\n</python>";
        let blocks = extract_python_blocks(input);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].code, "x = 1");
        assert_eq!(blocks[1].code, "y = 2");
    }

    #[test]
    fn test_extract_python_blocks_none() {
        let input = "Just plain text, no python blocks here.";
        let blocks = extract_python_blocks(input);
        assert!(blocks.is_empty());
    }

    #[test]
    fn test_extract_python_blocks_with_label() {
        let input = r#"<python label="henry">print('hi')</python>"#;
        let blocks = extract_python_blocks(input);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].code, "print('hi')");
        assert_eq!(blocks[0].label.as_deref(), Some("henry"));
    }

    #[test]
    fn test_thread_error_retryable_llm() {
        let err = ThreadError::Llm(LLMError::retryable("transient network error"));
        match &err {
            ThreadError::Llm(llm_err) => assert!(llm_err.is_retryable),
            _ => panic!("expected ThreadError::Llm"),
        }
    }

    #[test]
    fn test_thread_error_permanent_llm() {
        let err = ThreadError::Llm(LLMError::permanent("bad request"));
        match &err {
            ThreadError::Llm(llm_err) => assert!(!llm_err.is_retryable),
            _ => panic!("expected ThreadError::Llm"),
        }
    }

    #[test]
    fn test_thread_error_non_llm_not_retryable() {
        let err = ThreadError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "file not found"));
        assert!(!matches!(err, ThreadError::Llm(_)));
    }

    #[test]
    fn test_retry_constants() {
        assert_eq!(STREAM_MAX_RETRIES, 3);
        assert_eq!(STREAM_RETRY_BASE_DELAY_SECS, 2);
        // Verify exponential backoff: 2s, 4s, 8s
        for attempt in 1..=STREAM_MAX_RETRIES {
            let delay = STREAM_RETRY_BASE_DELAY_SECS * (1 << (attempt - 1));
            assert_eq!(delay, 2u64.pow(attempt));
        }
    }

    #[test]
    fn test_extract_python_blocks_mixed_labeled_unlabeled() {
        let input = "<python>\nx = 1\n</python>\nmiddle\n<python label=\"foo\">\ny = 2\n</python>";
        let blocks = extract_python_blocks(input);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].code, "x = 1");
        assert!(blocks[0].label.is_none());
        assert_eq!(blocks[1].code, "y = 2");
        assert_eq!(blocks[1].label.as_deref(), Some("foo"));
    }
}
