//! Daemon mode for running agents headlessly.
//!
//! This module provides the infrastructure for running agents as background daemons,
//! with Unix socket API for communication and timer trigger support.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use chrono::Local;
use tokio::net::UnixListener;
use tokio::sync::{mpsc, Mutex};

/// Agent-specific logger that writes to {agent_dir}/agent.log
pub struct AgentLogger {
    file: std::sync::Mutex<File>,
    agent_name: String,
}

impl AgentLogger {
    /// Create a new logger for the agent, writing to agent_dir/agent.log
    /// Truncates the log on each daemon restart for cleaner debugging.
    pub fn new(agent_dir: &Path, agent_name: &str) -> std::io::Result<Self> {
        let log_path = agent_dir.join("agent.log");
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)  // Fresh log each restart
            .open(&log_path)?;
        
        Ok(Self {
            file: std::sync::Mutex::new(file),
            agent_name: agent_name.to_string(),
        })
    }
    
    /// Log a message with timestamp
    pub fn log(&self, msg: &str) {
        let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S%.3f");
        let line = format!("[{}] [{}] {}\n", timestamp, self.agent_name, msg);
        
        // Write to file
        if let Ok(mut file) = self.file.lock() {
            let _ = file.write_all(line.as_bytes());
            let _ = file.flush();
        }
        
        // Also print to stdout for interactive use
        print!("{}", line);
    }
    
    /// Log a memory-related event
    pub fn memory(&self, msg: &str) {
        self.log(&format!("[memory] {}", msg));
    }
    
    /// Log a tool-related event
    pub fn tool(&self, msg: &str) {
        self.log(&format!("[tool] {}", msg));
    }
}

/// Parsed tool call from agent output
#[derive(Debug)]
pub struct ToolCall {
    pub tool: String,
    pub params: serde_json::Value,
}

/// Extract a JSON tool block from agent output.
/// Returns (cleaned_output, Option<ToolCall>)
pub fn extract_tool_call(output: &str) -> (String, Option<ToolCall>) {
    // Only accept fenced ```json ... ``` blocks for tool calls
    // This is unambiguous and handles nested braces correctly
    let fenced_re = regex::Regex::new(r"```json\s*\n?\s*(\{[^`]*\})\s*\n?\s*```").unwrap();
    
    if let Some(cap) = fenced_re.captures(output) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&cap[1]) {
            if let Some(tool_name) = json.get("tool").and_then(|t| t.as_str()) {
                let params = json.get("params").cloned().unwrap_or(serde_json::json!({}));
                let cleaned = fenced_re.replace(output, "").trim().to_string();
                return (cleaned, Some(ToolCall {
                    tool: tool_name.to_string(),
                    params,
                }));
            }
        }
    }
    
    (output.to_string(), None)
}

use crate::agent::{Agent, ThinkOptions};
use crate::agent_dir::{AgentDir, AgentDirError, SemanticMemorySection, ResolvedLlmConfig};
use crate::conversation::ConversationStore;
use crate::discovery;
use crate::embedding::EmbeddingClient;
use crate::llm::{LLM, OpenAIClient, AnthropicClient, OllamaClient, ToolSpec, ChatMessage, strip_thinking_tags};
use crate::conversation::ConversationMessage;
use crate::memory::{Memory, SqliteMemory, InMemoryStore, SemanticMemoryStore, SaveResult, extract_remember_tags, build_memory_injection};
use crate::observe::ConsoleObserver;
use crate::runtime::Runtime;
use crate::socket_api::{SocketApi, Request, Response};
use crate::tool::Tool;
use crate::tool_registry::{ToolRegistry, ToolDefinition};
use crate::tools::{AddTool, EchoTool, ReadFileTool, WriteFileTool, HttpTool, ShellTool};
use crate::tools::send_message::DaemonSendMessageTool;
use crate::tools::list_agents::DaemonListAgentsTool;
use crate::tools::claude_code::{ClaudeCodeTool, TaskStore, TaskStatus, is_process_running};

/// Build the effective always prompt by combining tools, memory, base always, and model always.
fn build_effective_always(
    tools_injection: &str,
    memory_injection: &str,
    base_always: &Option<String>,
    model_always: &Option<String>,
) -> Option<String> {
    let mut parts = Vec::new();

    if !tools_injection.is_empty() {
        parts.push(tools_injection.to_string());
    }
    if !memory_injection.is_empty() {
        parts.push(memory_injection.to_string());
    }
    if let Some(base) = base_always {
        parts.push(base.clone());
    }
    // Model-specific always is appended after agent always
    if let Some(model) = model_always {
        parts.push(model.clone());
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n\n"))
    }
}

/// Format conversation history into proper ChatMessage roles for LLM consumption.
///
/// # Design Reasoning
///
/// 1. **Role mapping by speaker:**
///    - Messages from current agent → `assistant` role, RAW text (no JSON wrapper)
///    - Messages from others (user, other agents) → `user` role, JSON wrapped
///
/// 2. **Why this asymmetry?**
///    - Models are trained on user/assistant alternation - this matches their training
///    - Current agent's past responses as 'assistant' helps model understand conversation flow
///    - JSON wrapper on others' messages tells model WHO said what (multi-party awareness)
///    - Model outputs raw text, so seeing its own past output as raw maintains consistency
///
/// 3. **Batching for alternation:**
///    - Consecutive messages from non-self speakers get batched into a single user message
///    - This maintains strict user/assistant alternation that models expect
///    - Prevents back-to-back user messages which can confuse models
///
/// Returns `(history: Vec<ChatMessage>, final_user_content: String)` where:
/// - `history` contains all but the last user turn, properly formatted
/// - `final_user_content` is the last user turn's content (for always_prompt prepending)
fn format_conversation_history(
    messages: &[ConversationMessage],
    current_agent: &str,
) -> (Vec<ChatMessage>, String) {
    if messages.is_empty() {
        return (Vec::new(), String::new());
    }

    let mut history: Vec<ChatMessage> = Vec::new();
    let mut pending_user_batch: Vec<String> = Vec::new();

    // Helper to flush pending user messages into a single ChatMessage
    let flush_user_batch = |batch: &mut Vec<String>, hist: &mut Vec<ChatMessage>| {
        if !batch.is_empty() {
            hist.push(ChatMessage {
                role: "user".to_string(),
                content: Some(batch.join("\n")),
                tool_call_id: None,
                tool_calls: None,
            });
            batch.clear();
        }
    };

    // Process all messages
    for msg in messages {
        if msg.from_agent == current_agent {
            // Current agent's message → assistant role, raw text
            // First, flush any pending user messages
            flush_user_batch(&mut pending_user_batch, &mut history);

            history.push(ChatMessage {
                role: "assistant".to_string(),
                content: Some(msg.content.clone()),
                tool_call_id: None,
                tool_calls: None,
            });
        } else {
            // Other speaker → accumulate for user batch with JSON wrapper
            let escaped = msg.content
                .replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', "\\n");
            pending_user_batch.push(format!(
                "{{\"from\": \"{}\", \"text\": \"{}\"}}",
                msg.from_agent, escaped
            ));
        }
    }

    // After processing all messages, we need to extract the final user content
    // The final user turn is what gets the always_prompt prepended
    if !pending_user_batch.is_empty() {
        // Last turn is from non-self (user/other agent) - this is the current query
        let final_content = pending_user_batch.join("\n");
        (history, final_content)
    } else if !history.is_empty() {
        // Last message was from self - unusual but handle it
        // Pop the last assistant message and treat it as context
        // The "task" will be empty, which think_with_options will handle
        (history, String::new())
    } else {
        (Vec::new(), String::new())
    }
}

/// Context for executing tools that need daemon state.
pub struct ToolExecutionContext {
    pub agent_name: String,
    pub task_store: Option<Arc<Mutex<TaskStore>>>,
    pub conv_id: Option<String>,
    pub semantic_memory_store: Option<Arc<Mutex<SemanticMemoryStore>>>,
    pub embedding_client: Option<Arc<EmbeddingClient>>,
}

/// Execute a tool call and return the result as a string.
/// If a tool_def is provided, command validation is performed for shell tools.
async fn execute_tool_call(
    tool_call: &ToolCall,
    tool_def: Option<&ToolDefinition>,
    context: Option<&ToolExecutionContext>,
) -> Result<String, String> {
    match tool_call.tool.as_str() {
        "read_file" => {
            let tool = ReadFileTool::default();
            match tool.execute(tool_call.params.clone()).await {
                Ok(result) => {
                    if let Some(contents) = result.get("contents").and_then(|c| c.as_str()) {
                        Ok(contents.to_string())
                    } else {
                        Ok(result.to_string())
                    }
                }
                Err(e) => Err(format!("Tool error: {}", e))
            }
        }
        "write_file" => {
            let tool = WriteFileTool::default();
            match tool.execute(tool_call.params.clone()).await {
                Ok(result) => {
                    if let Some(msg) = result.get("message").and_then(|m| m.as_str()) {
                        Ok(msg.to_string())
                    } else {
                        Ok(result.to_string())
                    }
                }
                Err(e) => Err(format!("Tool error: {}", e))
            }
        }
        "shell" | "safe_shell" => {
            // Validate command against allowed_commands if set
            if let Some(def) = tool_def {
                if let Some(ref allowed) = def.allowed_commands {
                    let command = tool_call.params.get("command")
                        .and_then(|c| c.as_str())
                        .unwrap_or("");
                    let first_word = command.split_whitespace().next().unwrap_or("");
                    if !allowed.iter().any(|a| a == first_word) {
                        return Err(format!(
                            "Command '{}' not in allowed list. Allowed: {:?}",
                            first_word, allowed
                        ));
                    }
                }
            }

            let tool = ShellTool::default();
            match tool.execute(tool_call.params.clone()).await {
                Ok(result) => {
                    // Shell returns stdout, stderr, and exit_code
                    let stdout = result.get("stdout").and_then(|s| s.as_str()).unwrap_or("");
                    let stderr = result.get("stderr").and_then(|s| s.as_str()).unwrap_or("");
                    let exit_code = result.get("exit_code").and_then(|c| c.as_i64()).unwrap_or(0);

                    let mut output = String::new();
                    if !stdout.is_empty() {
                        output.push_str(stdout);
                    }
                    if !stderr.is_empty() {
                        if !output.is_empty() {
                            output.push_str("\n");
                        }
                        output.push_str("[stderr] ");
                        output.push_str(stderr);
                    }
                    if exit_code != 0 {
                        output.push_str(&format!("\n[exit code: {}]", exit_code));
                    }
                    Ok(output)
                }
                Err(e) => Err(format!("Tool error: {}", e))
            }
        }
        "http" => {
            let tool = HttpTool::new();
            match tool.execute(tool_call.params.clone()).await {
                Ok(result) => {
                    let status = result.get("status").and_then(|s| s.as_u64()).unwrap_or(0);
                    let body = result.get("body").and_then(|b| b.as_str()).unwrap_or("");
                    Ok(format!("[HTTP {}]\n{}", status, body))
                }
                Err(e) => Err(format!("Tool error: {}", e))
            }
        }
        "claude_code" => {
            // Claude Code tool requires agent context and task store
            let ctx = context.ok_or("claude_code tool requires execution context")?;
            let task_store = ctx.task_store.as_ref()
                .ok_or("claude_code tool requires task store to be initialized")?;

            let tool = ClaudeCodeTool::with_conv_id(ctx.agent_name.clone(), task_store.clone(), ctx.conv_id.clone());
            match tool.execute(tool_call.params.clone()).await {
                Ok(result) => {
                    if let Some(msg) = result.get("message").and_then(|m| m.as_str()) {
                        Ok(msg.to_string())
                    } else {
                        Ok(result.to_string())
                    }
                }
                Err(e) => Err(format!("Tool error: {}", e))
            }
        }
        "remember" => {
            // Remember tool requires semantic memory store
            let ctx = context.ok_or("remember tool requires execution context")?;
            let mem_store = ctx.semantic_memory_store.as_ref()
                .ok_or("remember tool requires semantic memory to be enabled")?;

            let content = tool_call.params.get("content")
                .and_then(|c| c.as_str())
                .ok_or("remember tool requires 'content' parameter")?;

            // Generate embedding if client is available
            let embedding = if let Some(emb_client) = &ctx.embedding_client {
                emb_client.embed(content).await.ok()
            } else {
                None
            };

            // Save with high importance (0.9) as "explicit" source, same as [REMEMBER:] tags
            let store_guard = mem_store.lock().await;
            match store_guard.save_with_embedding(content, 0.9, "explicit", embedding.as_deref()) {
                Ok(result) => {
                    let msg = match result {
                        SaveResult::New(id) => format!("Remembered: {} (id={})", content, id),
                        SaveResult::Reinforced(id, old_imp, new_imp) => {
                            format!("Reinforced memory: {} (id={}, importance {:.2} → {:.2})", content, id, old_imp, new_imp)
                        }
                    };
                    Ok(msg)
                }
                Err(e) => Err(format!("Failed to save memory: {}", e))
            }
        }
        _ => Err(format!("Unknown tool: {}", tool_call.tool))
    }
}

/// Convert ToolDefinition params to JSON Schema format for native tool calling.
fn convert_params_to_json_schema(params: &serde_json::Value) -> serde_json::Value {
    match params {
        serde_json::Value::Object(map) => {
            let mut properties = serde_json::Map::new();
            let mut required = Vec::new();

            for (name, type_info) in map {
                let type_str = match type_info {
                    serde_json::Value::String(s) => s.as_str(),
                    _ => "string",
                };

                let is_optional = type_str.to_lowercase().contains("optional");
                let base_type = type_str.split_whitespace().next().unwrap_or("string");

                properties.insert(name.clone(), serde_json::json!({"type": base_type}));

                if !is_optional {
                    required.push(serde_json::Value::String(name.clone()));
                }
            }

            serde_json::json!({
                "type": "object",
                "properties": properties,
                "required": required
            })
        }
        _ => serde_json::json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }
}

/// Convert ToolDefinitions from registry to ToolSpecs for native LLM tool calling.
fn tool_definitions_to_specs(definitions: &[&ToolDefinition]) -> Vec<ToolSpec> {
    definitions.iter().map(|def| {
        ToolSpec {
            name: def.name.clone(),
            description: def.description.clone(),
            parameters: convert_params_to_json_schema(&def.params),
        }
    }).collect()
}

/// Filter tools by allowed_tools list. If allowed_tools is None, no tools allowed (safe default).
fn filter_by_allowlist<'a>(
    tools: Vec<&'a ToolDefinition>,
    allowed_tools: &Option<Vec<String>>,
) -> Vec<&'a ToolDefinition> {
    match allowed_tools {
        Some(allowed) => tools.into_iter().filter(|t| allowed.contains(&t.name)).collect(),
        None => Vec::new(),  // No allowlist = no tools
    }
}

/// Configuration for the daemon, derived from AgentDir.
#[derive(Debug, Clone)]
pub struct DaemonConfig {
    /// Agent name
    pub name: String,
    /// Path to the agent directory
    pub agent_dir: PathBuf,
    /// Path to the Unix socket
    pub socket_path: PathBuf,
    /// Path to the PID file
    pub pid_path: PathBuf,
    /// Timer configuration (if enabled)
    pub timer: Option<TimerConfig>,
    /// Heartbeat configuration (if enabled)
    pub heartbeat: Option<HeartbeatDaemonConfig>,
    /// Persona (system prompt)
    pub persona: Option<String>,
    /// Always content (injected before user messages for recency bias)
    pub always: Option<String>,
    /// Model-specific always text (appended to agent always)
    pub model_always: Option<String>,
    /// Semantic memory configuration
    pub semantic_memory: SemanticMemorySection,
    /// Allowlist of tool names. If set, only these tools are available.
    pub allowed_tools: Option<Vec<String>>,
}

/// Timer configuration for periodic triggers.
#[derive(Debug, Clone)]
pub struct TimerConfig {
    /// How often the timer fires
    pub interval: Duration,
    /// Message to send when timer fires
    pub message: String,
}

/// Heartbeat configuration for the daemon.
#[derive(Debug, Clone)]
pub struct HeartbeatDaemonConfig {
    /// How often the heartbeat fires
    pub interval: Duration,
    /// Path to the heartbeat.md file
    pub heartbeat_path: PathBuf,
}

impl DaemonConfig {
    /// Create a DaemonConfig from an AgentDir.
    pub fn from_agent_dir(agent_dir: &AgentDir) -> Result<Self, AgentDirError> {
        let name = agent_dir.config.agent.name.clone();
        let dir_path = agent_dir.path.clone();

        // Socket and PID files live in the agent directory
        let socket_path = dir_path.join("agent.sock");
        let pid_path = dir_path.join("daemon.pid");

        // Parse timer config if present and enabled
        let timer = agent_dir.config.timer.as_ref().and_then(|t| {
            if t.enabled {
                parse_duration(&t.interval).map(|interval| TimerConfig {
                    interval,
                    message: t.message.clone().unwrap_or_else(|| "Timer trigger".to_string()),
                })
            } else {
                None
            }
        });

        // Parse heartbeat config if enabled and interval set
        let heartbeat_path = dir_path.join("heartbeat.md");
        let heartbeat = if agent_dir.config.heartbeat.enabled {
            agent_dir.config.heartbeat.interval.as_ref().and_then(|interval_str| {
                parse_duration(interval_str).map(|interval| HeartbeatDaemonConfig {
                    interval,
                    heartbeat_path: heartbeat_path.clone(),
                })
            })
        } else {
            None
        };

        // Load persona
        let persona = agent_dir.load_persona()?;

        // Load always content
        let always = agent_dir.load_always()?;

        // Load model-specific config from resolved LLM config
        let llm_config = agent_dir.resolve_llm_config()?;
        let model_always = llm_config.always;
        let allowed_tools = llm_config.allowed_tools;

        // Build runtime context and append to persona
        let host = gethostname::gethostname()
            .to_string_lossy()
            .to_string();
        let runtime_context = build_runtime_context(&name, &llm_config.model, &host, llm_config.tools);

        let persona = match persona {
            Some(p) => Some(format!("{}\n\n{}", p, runtime_context)),
            None => Some(runtime_context),
        };

        Ok(Self {
            name,
            agent_dir: dir_path,
            socket_path,
            pid_path,
            timer,
            heartbeat,
            persona,
            always,
            model_always,
            semantic_memory: agent_dir.config.semantic_memory.clone(),
            allowed_tools,
        })
    }
}

/// Build the runtime context string that is appended to the persona.
/// This gives agents self-awareness about their environment.
fn build_runtime_context(agent: &str, model: &str, host: &str, tools_native: bool) -> String {
    let tools_mode = if tools_native { "native" } else { "json-block" };
    format!(
        "You are running inside Anima, a multi-agent runtime.\n\nRuntime: agent={} | model={} | host={} | tools={}",
        agent, model, host, tools_mode
    )
}

/// Parse a duration string like "30s", "5m", "1h", or compound "2h30m" into a Duration.
fn parse_duration(s: &str) -> Option<Duration> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    let mut total_secs: u64 = 0;
    let mut remaining = s;

    while !remaining.is_empty() {
        // Find the end of the numeric part
        let num_end = remaining.find(|c: char| !c.is_ascii_digit()).unwrap_or(remaining.len());
        if num_end == 0 {
            return None;
        }

        let num: u64 = remaining[..num_end].parse().ok()?;
        remaining = &remaining[num_end..];

        // Find the end of the unit part
        let unit_end = remaining.find(|c: char| c.is_ascii_digit()).unwrap_or(remaining.len());
        let unit = &remaining[..unit_end];
        remaining = &remaining[unit_end..];

        let secs = match unit {
            "s" | "sec" | "secs" | "second" | "seconds" => num,
            "m" | "min" | "mins" | "minute" | "minutes" => num * 60,
            "h" | "hr" | "hrs" | "hour" | "hours" => num * 3600,
            "" if remaining.is_empty() => num, // Bare number at end means seconds
            _ => return None,
        };

        total_secs += secs;
    }

    if total_secs > 0 {
        Some(Duration::from_secs(total_secs))
    } else {
        None
    }
}

/// PID file manager for daemon lifecycle.
pub struct PidFile {
    path: PathBuf,
}

impl PidFile {
    /// Create a new PID file at the given path.
    /// Writes the current process ID to the file.
    pub fn create(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let pid = std::process::id();
        std::fs::write(&path, pid.to_string())?;
        Ok(Self { path })
    }

    /// Read the PID from an existing PID file.
    pub fn read(path: impl AsRef<Path>) -> std::io::Result<u32> {
        let content = std::fs::read_to_string(path)?;
        content.trim().parse().map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })
    }

    /// Check if a PID file exists and the process is still running.
    pub fn is_running(path: impl AsRef<Path>) -> bool {
        if let Ok(pid) = Self::read(&path) {
            // Check if process exists by sending signal 0
            unsafe {
                libc::kill(pid as i32, 0) == 0
            }
        } else {
            false
        }
    }

    /// Remove the PID file.
    pub fn remove(&self) -> std::io::Result<()> {
        if self.path.exists() {
            std::fs::remove_file(&self.path)?;
        }
        Ok(())
    }
}

impl Drop for PidFile {
    fn drop(&mut self) {
        let _ = self.remove();
    }
}

/// Resolve an agent path from a name or path string.
fn resolve_agent_path(agent: &str) -> PathBuf {
    if agent.contains('/') || agent.contains('\\') || agent.starts_with('.') {
        PathBuf::from(agent)
    } else {
        dirs::home_dir()
            .expect("Could not determine home directory")
            .join(".anima")
            .join("agents")
            .join(agent)
    }
}

/// Run an agent as a daemon.
pub async fn run_daemon(agent: &str) -> Result<(), Box<dyn std::error::Error>> {
    let agent_path = resolve_agent_path(agent);

    // Load the agent directory
    let agent_dir = AgentDir::load(&agent_path)?;
    let config = DaemonConfig::from_agent_dir(&agent_dir)?;

    // Check if already running
    if PidFile::is_running(&config.pid_path) {
        return Err(format!(
            "Agent '{}' is already running (PID file: {})",
            config.name,
            config.pid_path.display()
        ).into());
    }

    // Clean up stale socket file if it exists
    if config.socket_path.exists() {
        std::fs::remove_file(&config.socket_path)?;
    }

    // Create PID file
    let _pid_file = PidFile::create(&config.pid_path)?;

    // Create agent-specific logger
    let logger = Arc::new(AgentLogger::new(&config.agent_dir, &config.name)?);
    
    logger.log(&format!("Starting daemon for agent '{}'", config.name));
    logger.log(&format!("  PID file: {}", config.pid_path.display()));
    logger.log(&format!("  Socket: {}", config.socket_path.display()));
    if let Some(ref timer) = config.timer {
        logger.log(&format!("  Timer: every {:?}, message: \"{}\"", timer.interval, timer.message));
    }
    if let Some(ref hb) = config.heartbeat {
        logger.log(&format!("  Heartbeat: every {:?}, file: {}", hb.interval, hb.heartbeat_path.display()));
    }

    // Create the agent
    let (agent, use_native_tools) = create_agent_from_dir(&agent_dir).await?;
    let agent = Arc::new(Mutex::new(agent));
    logger.log(&format!("  Native tools: {}", use_native_tools));

    // Create embedding client if configured
    let embedding_client: Option<Arc<EmbeddingClient>> = if let Some(ref emb_config) = config.semantic_memory.embedding {
        if emb_config.provider == "ollama" {
            let client = EmbeddingClient::new(&emb_config.model, Some(&emb_config.url));
            logger.log(&format!("  Embedding client: {} via {} at {}",
                emb_config.model, emb_config.provider, emb_config.url));
            Some(Arc::new(client))
        } else {
            logger.log(&format!("  Embedding client: unsupported provider '{}'", emb_config.provider));
            None
        }
    } else {
        None
    };

    // Create semantic memory store if enabled
    let semantic_memory_store: Option<Arc<Mutex<SemanticMemoryStore>>> = if config.semantic_memory.enabled {
        let mem_path = config.agent_dir.join(&config.semantic_memory.path);
        let store = SemanticMemoryStore::open(&mem_path, &config.name)?;
        logger.log(&format!("  Semantic memory: {}", mem_path.display()));

        // Backfill embeddings if needed
        if let Some(ref emb_client) = embedding_client {
            if store.needs_backfill(emb_client.model())? {
                logger.log("  Backfilling embeddings...");
                let memories = store.get_memories_needing_embeddings()?;
                logger.log(&format!("    {} memories need embeddings", memories.len()));

                for (id, content) in memories {
                    match emb_client.embed(&content).await {
                        Ok(embedding) => {
                            if let Err(e) = store.update_embedding(id, &embedding) {
                                logger.log(&format!("    Failed to update embedding for #{}: {}", id, e));
                            }
                        }
                        Err(e) => {
                            logger.log(&format!("    Failed to generate embedding for #{}: {}", id, e));
                        }
                    }
                }

                // Update the stored model
                store.set_embedding_model(emb_client.model())?;
                logger.log("  Backfill complete");
            }
        }

        Some(Arc::new(Mutex::new(store)))
    } else {
        None
    };

    // Load tool registry (if available)
    let tool_registry: Option<Arc<ToolRegistry>> = match ToolRegistry::load_global() {
        Ok(registry) => {
            logger.log(&format!("  Tool registry: {} tools loaded", registry.all_tools().len()));
            Some(Arc::new(registry))
        }
        Err(e) => {
            logger.log(&format!("  Tool registry: not loaded ({})", e));
            None
        }
    };

    // Initialize task store for Claude Code tasks
    let task_store: Option<Arc<Mutex<TaskStore>>> = match TaskStore::init() {
        Ok(store) => {
            logger.log("  Task store: initialized");
            Some(Arc::new(Mutex::new(store)))
        }
        Err(e) => {
            logger.log(&format!("  Task store: not initialized ({})", e));
            None
        }
    };

    // Process pending notifications (queued while agent was offline)
    if let Ok(conv_store) = ConversationStore::init() {
        match conv_store.get_pending_notifications(&config.name) {
            Ok(pending) => {
                if !pending.is_empty() {
                    logger.log(&format!("Processing {} pending notifications...", pending.len()));
                    for notification in &pending {
                        logger.log(&format!("  Processing notification: conv={} msg_id={}",
                            notification.conv_name, notification.message_id));

                        let response = handle_notify(
                            &notification.conv_name,
                            notification.message_id,
                            0, // Start at depth 0 for pending notifications
                            &agent,
                            &config.name,
                            &config.persona,
                            &config.always,
                            &config.model_always,
                            &config.allowed_tools,
                            &semantic_memory_store,
                            &embedding_client,
                            &tool_registry,
                            use_native_tools,
                            &logger,
                            config.semantic_memory.recall_limit,
                            &task_store,
                        ).await;

                        match response {
                            Response::Notified { response_message_id } => {
                                logger.log(&format!("  Responded with msg_id={}", response_message_id));
                            }
                            Response::Error { message } => {
                                logger.log(&format!("  Failed: {}", message));
                            }
                            _ => {}
                        }
                    }
                    // Clear processed notifications
                    if let Err(e) = conv_store.clear_pending_notifications(&config.name) {
                        logger.log(&format!("Failed to clear pending notifications: {}", e));
                    } else {
                        logger.log("Pending notifications cleared");
                    }
                }
            }
            Err(e) => {
                logger.log(&format!("Failed to get pending notifications: {}", e));
            }
        }
    }

    // Create Unix socket listener
    let listener = UnixListener::bind(&config.socket_path)?;
    logger.log(&format!("Listening on {}", config.socket_path.display()));

    // Set up signal handling for graceful shutdown
    let shutdown = Arc::new(tokio::sync::Notify::new());
    let shutdown_clone = shutdown.clone();

    // Handle SIGTERM and SIGINT
    tokio::spawn(async move {
        let mut sigterm = tokio::signal::unix::signal(
            tokio::signal::unix::SignalKind::terminate()
        ).expect("Failed to create SIGTERM handler");

        let mut sigint = tokio::signal::unix::signal(
            tokio::signal::unix::SignalKind::interrupt()
        ).expect("Failed to create SIGINT handler");

        tokio::select! {
            _ = sigterm.recv() => {
                println!("\nReceived SIGTERM, shutting down...");
            }
            _ = sigint.recv() => {
                println!("\nReceived SIGINT, shutting down...");
            }
        }

        shutdown_clone.notify_waiters();
    });

    // Set up timer if configured
    let timer_handle = if let Some(ref timer_config) = config.timer {
        let agent_clone = agent.clone();
        let persona = config.persona.clone();
        let always = config.always.clone();
        let model_always = config.model_always.clone();
        let allowed_tools = config.allowed_tools.clone();
        let timer_config = timer_config.clone();
        let shutdown_clone = shutdown.clone();
        let semantic_memory = semantic_memory_store.clone();
        let timer_embedding_client = embedding_client.clone();
        let timer_registry = tool_registry.clone();
        let timer_logger = logger.clone();
        let timer_recall_limit = config.semantic_memory.recall_limit;

        Some(tokio::spawn(async move {
            let mut interval = tokio::time::interval(timer_config.interval);
            // Skip the first immediate tick
            interval.tick().await;

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        timer_logger.log("[timer] Firing timer trigger...");

                        // Inject relevant memories if enabled
                        let memory_injection = if let Some(ref mem_store) = semantic_memory {
                            // Generate query embedding if we have an embedding client
                            let query_embedding = if let Some(ref emb_client) = timer_embedding_client {
                                match emb_client.embed(&timer_config.message).await {
                                    Ok(emb) => Some(emb),
                                    Err(e) => {
                                        timer_logger.log(&format!("[timer] Failed to generate query embedding: {}", e));
                                        None
                                    }
                                }
                            } else {
                                None
                            };

                            let store = mem_store.lock().await;
                            match store.recall_with_embedding(&timer_config.message, timer_recall_limit, query_embedding.as_deref()) {
                                Ok(memories) => {
                                    if !memories.is_empty() {
                                        timer_logger.memory(&format!("Recall: {} memories for timer", memories.len()));
                                        for (m, score) in &memories {
                                            timer_logger.memory(&format!("  ({:.3}) \"{}\" [#{}]", score, m.content, m.id));
                                        }
                                    }
                                    let entries: Vec<_> = memories.iter().map(|(m, _)| m.clone()).collect();
                                    build_memory_injection(&entries)
                                }
                                Err(e) => {
                                    timer_logger.log(&format!("[timer] Memory recall error: {}", e));
                                    String::new()
                                }
                            }
                        } else {
                            String::new()
                        };

                        // Inject relevant tools if registry is available
                        let tools_injection = if let Some(ref registry) = timer_registry {
                            let relevant = registry.find_relevant(&timer_config.message, timer_recall_limit);
                            let relevant = filter_by_allowlist(relevant, &allowed_tools);
                            if !relevant.is_empty() {
                                timer_logger.tool(&format!("Recall: {} tools for timer", relevant.len()));
                                for t in &relevant {
                                    timer_logger.tool(&format!("  - {}", t.name));
                                }
                            }
                            ToolRegistry::format_for_prompt(&relevant)
                        } else {
                            String::new()
                        };

                        // Combine tools, memory injection, and always prompt
                        let effective_always = build_effective_always(&tools_injection, &memory_injection, &always, &model_always);

                        let mut agent_guard = agent_clone.lock().await;

                        let options = ThinkOptions {
                            system_prompt: persona.clone(),
                            always_prompt: effective_always,
                            ..Default::default()
                        };

                        match agent_guard.think_with_options(&timer_config.message, options).await {
                            Ok(result) => {
                                // Strip thinking tags and extract [REMEMBER: ...] tags
                                let without_thinking = strip_thinking_tags(&result.response);
                                let (cleaned_response, memories_to_save) = extract_remember_tags(&without_thinking);

                                if !memories_to_save.is_empty() {
                                    if let Some(ref mem_store) = semantic_memory {
                                        let store = mem_store.lock().await;
                                        for memory in &memories_to_save {
                                            // Generate embedding if we have a client
                                            let embedding = if let Some(ref emb_client) = timer_embedding_client {
                                                match emb_client.embed(memory).await {
                                                    Ok(emb) => Some(emb),
                                                    Err(e) => {
                                                        timer_logger.log(&format!("[timer] Failed to generate embedding: {}", e));
                                                        None
                                                    }
                                                }
                                            } else {
                                                None
                                            };

                                            match store.save_with_embedding(memory, 0.9, "explicit", embedding.as_deref()) {
                                                Ok(SaveResult::New(id)) => timer_logger.memory(&format!("Save #{}: \"{}\"", id, memory)),
                                                Ok(SaveResult::Reinforced(id, old, new)) => timer_logger.memory(&format!("Reinforce #{}: \"{}\" ({:.2} → {:.2})", id, memory, old, new)),
                                                Err(e) => timer_logger.log(&format!("[timer] Failed to save memory: {}", e)),
                                            }
                                        }
                                    }
                                }

                                timer_logger.log(&format!("[timer] Response: {}", cleaned_response));
                            }
                            Err(e) => {
                                timer_logger.log(&format!("[timer] Error: {}", e));
                            }
                        }
                    }
                    _ = shutdown_clone.notified() => {
                        break;
                    }
                }
            }
        }))
    } else {
        None
    };

    // Set up heartbeat if configured
    // heartbeat_pending tracks if a heartbeat should fire after current think completes
    let heartbeat_pending = Arc::new(AtomicBool::new(false));
    let heartbeat_config = config.heartbeat.clone();

    let heartbeat_handle = if let Some(ref hb_config) = heartbeat_config {
        // Only start heartbeat loop if heartbeat.md exists
        if hb_config.heartbeat_path.exists() {
            let agent_clone = agent.clone();
            let agent_name = config.name.clone();
            let persona = config.persona.clone();
            let always = config.always.clone();
            let model_always = config.model_always.clone();
            let allowed_tools = config.allowed_tools.clone();
            let hb_config = hb_config.clone();
            let shutdown_clone = shutdown.clone();
            let semantic_memory = semantic_memory_store.clone();
            let hb_embedding_client = embedding_client.clone();
            let hb_registry = tool_registry.clone();
            let hb_logger = logger.clone();
            let hb_recall_limit = config.semantic_memory.recall_limit;
            let hb_pending = heartbeat_pending.clone();

            Some(tokio::spawn(async move {
                heartbeat_loop(
                    hb_config,
                    agent_clone,
                    agent_name,
                    persona,
                    always,
                    model_always,
                    allowed_tools,
                    semantic_memory,
                    hb_embedding_client,
                    hb_registry,
                    use_native_tools,
                    hb_logger,
                    hb_recall_limit,
                    hb_pending,
                    shutdown_clone,
                ).await
            }))
        } else {
            logger.log("  Heartbeat: heartbeat.md not found, skipping");
            None
        }
    } else {
        None
    };

    // Set up task watcher for Claude Code tasks (if task store is available)
    let task_watcher_handle = if let Some(ref ts) = task_store {
        let ts_clone = ts.clone();
        let agent_name = config.name.clone();
        let tw_logger = logger.clone();
        let shutdown_clone = shutdown.clone();

        logger.log("  Task watcher: started");
        Some(tokio::spawn(async move {
            task_watcher_loop(
                ts_clone,
                agent_name,
                tw_logger,
                shutdown_clone,
            ).await
        }))
    } else {
        None
    };

    // Main loop: accept connections
    loop {
        tokio::select! {
            result = listener.accept() => {
                match result {
                    Ok((stream, _)) => {
                        let agent_clone = agent.clone();
                        let agent_name = config.name.clone();
                        let persona = config.persona.clone();
                        let always = config.always.clone();
                        let model_always = config.model_always.clone();
                        let allowed_tools = config.allowed_tools.clone();
                        let shutdown_clone = shutdown.clone();
                        let semantic_memory = semantic_memory_store.clone();
                        let conn_embedding_client = embedding_client.clone();
                        let conn_registry = tool_registry.clone();
                        let conn_logger = logger.clone();
                        let conn_heartbeat_config = heartbeat_config.clone();
                        let conn_heartbeat_pending = heartbeat_pending.clone();
                        let conn_task_store = task_store.clone();

                        let recall_limit = config.semantic_memory.recall_limit;
                        tokio::spawn(async move {
                            let api = SocketApi::new(stream);
                            if let Err(e) = handle_connection(
                                api,
                                agent_clone,
                                agent_name,
                                persona,
                                always,
                                model_always,
                                allowed_tools,
                                semantic_memory,
                                conn_embedding_client,
                                conn_registry,
                                use_native_tools,
                                shutdown_clone,
                                conn_logger,
                                recall_limit,
                                conn_heartbeat_config,
                                conn_heartbeat_pending,
                                conn_task_store,
                            ).await {
                                eprintln!("Connection error: {}", e);
                            }
                        });
                    }
                    Err(e) => {
                        eprintln!("Accept error: {}", e);
                    }
                }
            }
            _ = shutdown.notified() => {
                logger.log("Shutting down daemon...");
                break;
            }
        }
    }

    // Wait for timer task to finish
    if let Some(handle) = timer_handle {
        let _ = handle.await;
    }

    // Wait for heartbeat task to finish
    if let Some(handle) = heartbeat_handle {
        let _ = handle.await;
    }

    // Wait for task watcher to finish
    if let Some(handle) = task_watcher_handle {
        let _ = handle.await;
    }

    // Clean up socket file
    if config.socket_path.exists() {
        let _ = std::fs::remove_file(&config.socket_path);
    }

    logger.log("Daemon stopped.");
    Ok(())
}

/// Create an agent from an AgentDir configuration.
/// Returns (Agent, use_native_tools) where use_native_tools indicates hybrid tool calling mode.
async fn create_agent_from_dir(agent_dir: &AgentDir) -> Result<(Agent, bool), Box<dyn std::error::Error>> {
    let agent_name = agent_dir.config.agent.name.clone();

    // Resolve LLM config (loads model file if specified, applies overrides)
    let llm_config = agent_dir.resolve_llm_config()?;
    let use_native_tools = llm_config.tools;

    // Get API key using resolved config
    let api_key = AgentDir::api_key_for_config(&llm_config)?;

    // Create LLM from resolved config
    let llm: Arc<dyn LLM> = create_llm_from_config(&llm_config, api_key)?;

    // Create memory from config
    let memory: Box<dyn Memory> = if let Some(mem_path) = agent_dir.memory_path() {
        if let Some(parent) = mem_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Box::new(SqliteMemory::open(
            mem_path.to_str().ok_or("Invalid memory path")?,
            &agent_name,
        )?)
    } else {
        Box::new(InMemoryStore::new())
    };

    // Create runtime and agent
    let mut runtime = Runtime::new();
    let mut agent = runtime.spawn_agent(agent_name.clone()).await;

    // Register native tools only if model supports tool calling API
    if llm_config.tools {
        agent.register_tool(Arc::new(AddTool));
        agent.register_tool(Arc::new(EchoTool));
        agent.register_tool(Arc::new(ReadFileTool));
        agent.register_tool(Arc::new(WriteFileTool));
        agent.register_tool(Arc::new(HttpTool::new()));
        agent.register_tool(Arc::new(ShellTool::new()));

        // Register daemon-aware messaging tools
        agent.register_tool(Arc::new(DaemonSendMessageTool::new(agent_name.clone())));
        agent.register_tool(Arc::new(DaemonListAgentsTool::new(agent_name.clone())));
    }

    // Apply LLM, memory, and agent_dir
    agent = agent.with_llm(llm);
    agent = agent.with_memory(memory);
    agent = agent.with_agent_dir(agent_dir.path.clone());

    // Add observer (verbose for daemon since there's no REPL)
    let observer = Arc::new(ConsoleObserver::new(true));
    agent = agent.with_observer(observer);

    Ok((agent, use_native_tools))
}

/// Create an LLM client from resolved configuration.
fn create_llm_from_config(
    config: &ResolvedLlmConfig,
    api_key: Option<String>,
) -> Result<Arc<dyn LLM>, Box<dyn std::error::Error>> {
    let llm: Arc<dyn LLM> = match config.provider.as_str() {
        "openai" => {
            let key = api_key.ok_or("OpenAI API key not configured")?;
            let mut client = OpenAIClient::new(key).with_model(&config.model);
            if let Some(ref base_url) = config.base_url {
                client = client.with_base_url(base_url);
            }
            Arc::new(client)
        }
        "anthropic" => {
            let key = api_key.ok_or("Anthropic API key not configured")?;
            let mut client = AnthropicClient::new(key).with_model(&config.model);
            if let Some(ref base_url) = config.base_url {
                client = client.with_base_url(base_url);
            }
            Arc::new(client)
        }
        "ollama" => {
            let mut client = OllamaClient::new()
                .with_model(&config.model)
                .with_thinking(config.thinking)
                .with_num_ctx(config.num_ctx);
            if let Some(ref base_url) = config.base_url {
                client = client.with_base_url(base_url);
            }
            Arc::new(client)
        }
        other => return Err(format!("Unsupported LLM provider: {}", other).into()),
    };
    Ok(llm)
}

/// Maximum depth for @mention chains to prevent infinite loops
const MAX_MENTION_DEPTH: u32 = 100;

/// Handle a Notify request: fetch conversation context, generate response, store it,
/// and forward @mentions to other agents (daemon-to-daemon).
async fn handle_notify(
    conv_id: &str,
    _message_id: i64,
    depth: u32,
    agent: &Arc<Mutex<Agent>>,
    agent_name: &str,
    persona: &Option<String>,
    always: &Option<String>,
    model_always: &Option<String>,
    allowed_tools: &Option<Vec<String>>,
    semantic_memory: &Option<Arc<Mutex<SemanticMemoryStore>>>,
    embedding_client: &Option<Arc<EmbeddingClient>>,
    tool_registry: &Option<Arc<ToolRegistry>>,
    use_native_tools: bool,
    logger: &Arc<AgentLogger>,
    recall_limit: usize,
    task_store: &Option<Arc<Mutex<TaskStore>>>,
) -> Response {
    // Open conversation store
    let store = match ConversationStore::init() {
        Ok(s) => s,
        Err(e) => {
            logger.log(&format!("[notify] Failed to open conversation store: {}", e));
            return Response::Error { message: format!("Failed to open conversation store: {}", e) };
        }
    };

    // Fetch recent messages from conversation for context
    let context_messages = match store.get_messages(conv_id, Some(recall_limit)) {
        Ok(msgs) => msgs,
        Err(e) => {
            logger.log(&format!("[notify] Failed to get messages: {}", e));
            return Response::Error { message: format!("Failed to get messages: {}", e) };
        }
    };

    if context_messages.is_empty() {
        logger.log("[notify] No messages in conversation");
        return Response::Error { message: "No messages in conversation".to_string() };
    }

    // Format conversation into proper user/assistant ChatMessages
    // This maps self→assistant (raw), others→user (JSON wrapped), with batching for alternation
    let (conversation_history, final_user_content) = format_conversation_history(&context_messages, agent_name);

    logger.log(&format!("[notify] Context: {} messages → {} history + final user turn",
        context_messages.len(), conversation_history.len()));

    // Get relevant tools from registry (based on final user content)
    let relevant_tools = if let Some(registry) = tool_registry {
        let relevant = registry.find_relevant(&final_user_content, recall_limit);
        filter_by_allowlist(relevant, allowed_tools)
    } else {
        Vec::new()
    };

    // Build tools injection
    let tools_injection = if use_native_tools {
        String::new()
    } else {
        ToolRegistry::format_for_prompt(&relevant_tools)
    };

    // Get memory injection (based on final user content)
    let memory_injection = if let Some(mem_store) = semantic_memory {
        let query_embedding = if let Some(emb_client) = embedding_client {
            emb_client.embed(&final_user_content).await.ok()
        } else {
            None
        };

        let store_guard = mem_store.lock().await;
        match store_guard.recall_with_embedding(&final_user_content, recall_limit, query_embedding.as_deref()) {
            Ok(memories) => {
                let entries: Vec<_> = memories.iter().map(|(m, _)| m.clone()).collect();
                build_memory_injection(&entries)
            }
            Err(_) => String::new()
        }
    } else {
        String::new()
    };

    let effective_always = build_effective_always(&tools_injection, &memory_injection, always, model_always);
    // Version without memories for tool continuation turns - still has tools + always.md
    let effective_always_no_memory = build_effective_always(&tools_injection, "", always, model_always);

    // Create tool execution context for tools that need daemon state
    let tool_context = ToolExecutionContext {
        agent_name: agent_name.to_string(),
        task_store: task_store.clone(),
        conv_id: Some(conv_id.to_string()),
        semantic_memory_store: semantic_memory.clone(),
        embedding_client: embedding_client.clone(),
    };

    // Tool execution loop - similar to Request::Message handler
    let max_tool_calls = 10;
    let mut tool_call_count = 0;
    let mut current_message = final_user_content.clone();
    #[allow(unused_assignments)]
    let mut final_response: Option<String> = None;

    // Mutable conversation history - refreshed from DB after each tool result
    let mut conversation_history = conversation_history;

    // First iteration gets full always (tools + memories + always.md).
    // Subsequent iterations get tools + always.md but NO memories (memories are query-specific).
    let mut is_first_iteration = true;

    loop {
        // Clear agent's internal history EACH iteration to avoid duplication.
        // think_with_options adds to self.history, and agent.rs injects self.history
        // BEFORE options.conversation_history (lines 655-660). Without clearing each
        // iteration, the growing internal history appears before DB-backed history.
        agent.lock().await.clear_history();

        let options = ThinkOptions {
            system_prompt: persona.clone(),
            always_prompt: if is_first_iteration { effective_always.clone() } else { effective_always_no_memory.clone() },
            conversation_history: Some(conversation_history.clone()),
            ..Default::default()
        };

        // Generate response
        let mut agent_guard = agent.lock().await;
        let result = agent_guard.think_with_options(&current_message, options).await;
        drop(agent_guard); // Release lock before potential tool execution

        match result {
            Ok(think_result) => {
                // Strip thinking tags and extract [REMEMBER: ...] tags
                let without_thinking = strip_thinking_tags(&think_result.response);
                let (after_remember, memories_to_save) = extract_remember_tags(&without_thinking);

                // Save memories
                if !memories_to_save.is_empty() {
                    if let Some(mem_store) = semantic_memory {
                        let store_guard = mem_store.lock().await;
                        for memory in &memories_to_save {
                            let embedding = if let Some(emb_client) = embedding_client {
                                emb_client.embed(memory).await.ok()
                            } else {
                                None
                            };
                            let _ = store_guard.save_with_embedding(memory, 0.9, "explicit", embedding.as_deref());
                        }
                    }
                }

                // Check for tool calls (JSON-block parsing)
                let (cleaned_response, tool_call) = extract_tool_call(&after_remember);

                if let Some(tc) = tool_call {
                    // Check if conversation is paused - skip tool execution if so
                    if store.is_paused(conv_id).unwrap_or(false) {
                        logger.log("[notify] Conversation paused, skipping tool execution");
                        // Store the FULL response (including tool call block) so it can be
                        // re-extracted and executed during catchup on resume
                        final_response = Some(after_remember.clone());
                        break;
                    }

                    tool_call_count += 1;
                    if tool_call_count > max_tool_calls {
                        logger.tool("[notify] Max tool calls reached, stopping");
                        final_response = Some(cleaned_response);
                        break;
                    }

                    // Store this assistant response before executing the tool
                    // (if there's content beyond just the tool call)
                    if !cleaned_response.trim().is_empty() {
                        if let Err(e) = store.add_message(conv_id, agent_name, &cleaned_response, &[]) {
                            logger.log(&format!("[notify] Failed to store intermediate response: {}", e));
                        } else {
                            logger.log(&format!("[notify] Stored intermediate response: {} bytes", cleaned_response.len()));
                        }
                    }

                    logger.tool(&format!("[notify] Executing: {} with params {}", tc.tool, tc.params));

                    // Look up the tool definition to get allowed_commands
                    let tool_def = tool_registry.as_ref()
                        .and_then(|r| r.find_by_name(&tc.tool));

                    match execute_tool_call(&tc, tool_def, Some(&tool_context)).await {
                        Ok(tool_result) => {
                            logger.tool(&format!("[notify] Result: {} bytes", tool_result.len()));
                            current_message = format!("[Tool Result for {}]\n{}", tc.tool, tool_result);
                            // Store tool result in conversation
                            if let Err(e) = store.add_message(conv_id, "tool", &current_message, &[]) {
                                logger.log(&format!("[notify] Failed to store tool result: {}", e));
                            }
                            // Refresh conversation_history from DB to include the tool result we just stored.
                            // This ensures we use ONLY DB-backed history and avoids duplication with agent's
                            // internal history (which we cleared at the start of handle_notify).
                            if let Ok(msgs) = store.get_messages(conv_id, Some(recall_limit)) {
                                let (refreshed_history, _) = format_conversation_history(&msgs, agent_name);
                                conversation_history = refreshed_history;
                            }
                            // Subsequent iterations are tool continuations - no need to re-inject always
                            is_first_iteration = false;
                            // Continue to next iteration
                        }
                        Err(e) => {
                            logger.tool(&format!("[notify] Error: {}", e));
                            current_message = format!("[Tool Error for {}]\n{}", tc.tool, e);
                            // Store tool error in conversation
                            if let Err(e) = store.add_message(conv_id, "tool", &current_message, &[]) {
                                logger.log(&format!("[notify] Failed to store tool error: {}", e));
                            }
                            // Refresh conversation_history from DB (same as success case)
                            if let Ok(msgs) = store.get_messages(conv_id, Some(recall_limit)) {
                                let (refreshed_history, _) = format_conversation_history(&msgs, agent_name);
                                conversation_history = refreshed_history;
                            }
                            // Subsequent iterations are tool continuations - no need to re-inject always
                            is_first_iteration = false;
                            // Continue to next iteration to let LLM handle the error
                        }
                    }
                } else {
                    // No more tool calls - done
                    final_response = Some(cleaned_response);
                    break;
                }
            }
            Err(e) => {
                logger.log(&format!("[notify] Agent error: {}", e));
                return Response::Error { message: e.to_string() };
            }
        }
    }

    // Store final response in conversation
    let cleaned_response = final_response.unwrap_or_default();
    match store.add_message(conv_id, agent_name, &cleaned_response, &[]) {
        Ok(response_msg_id) => {
            logger.log(&format!("[notify] Stored response as msg_id={}", response_msg_id));

            // Parse @mentions from our response and forward to other agents
            // Only forward if not paused and not at depth limit
            let should_forward = depth < MAX_MENTION_DEPTH
                && !store.is_paused(conv_id).unwrap_or(false);

            if should_forward {
                // Parse @mentions from our response
                let mentions = crate::conversation::parse_mentions(&cleaned_response);

                // Filter out self, "user", and non-existent agents
                let valid_mentions: Vec<String> = mentions
                    .into_iter()
                    .filter(|m| m != agent_name && m != "user" && m != "all")
                    .filter(|m| discovery::agent_exists(m))
                    .collect();

                if !valid_mentions.is_empty() {
                    logger.log(&format!("[notify] Forwarding to {} agents at depth {}: {:?}",
                        valid_mentions.len(), depth + 1, valid_mentions));

                    // Add mentioned agents as participants if not already
                    for mention in &valid_mentions {
                        if let Err(e) = store.add_participant(conv_id, mention) {
                            logger.log(&format!("[notify] Warning: Could not add {} as participant: {}", mention, e));
                        }
                    }

                    // Forward to each mentioned agent
                    for mention in valid_mentions {
                        forward_notify_to_agent(
                            &mention,
                            conv_id,
                            response_msg_id,
                            depth + 1,
                            logger,
                        ).await;
                    }
                }
            } else if depth >= MAX_MENTION_DEPTH {
                logger.log(&format!("[notify] Skipping forwarding: depth limit reached ({})", depth));
            }

            Response::Notified { response_message_id: response_msg_id }
        }
        Err(e) => {
            logger.log(&format!("[notify] Failed to store response: {}", e));
            Response::Error { message: format!("Failed to store response: {}", e) }
        }
    }
}

/// Forward a Notify request to another agent daemon.
/// If the agent is not running, queues the notification for later.
///
/// Note: This function creates its own ConversationStore internally when needed
/// to avoid threading issues with SQLite.
async fn forward_notify_to_agent(
    agent_name: &str,
    conv_id: &str,
    message_id: i64,
    depth: u32,
    logger: &Arc<AgentLogger>,
) {
    use tokio::net::UnixStream;

    // Helper function to queue notification using a fresh store
    let queue_notification = |agent: &str, cid: &str, mid: i64, log: &AgentLogger| {
        if let Ok(store) = ConversationStore::init() {
            if let Err(e) = store.add_pending_notification(agent, cid, mid) {
                log.log(&format!("[notify] Failed to queue notification for @{}: {}", agent, e));
            }
        } else {
            log.log(&format!("[notify] Failed to open store to queue notification for @{}", agent));
        }
    };

    // Check if agent is running
    if let Some(running_agent) = discovery::get_running_agent(agent_name) {
        // Try to connect and send Notify request
        match UnixStream::connect(&running_agent.socket_path).await {
            Ok(stream) => {
                let mut api = SocketApi::new(stream);

                let request = Request::Notify {
                    conv_id: conv_id.to_string(),
                    message_id,
                    depth,
                };

                if let Err(e) = api.write_request(&request).await {
                    logger.log(&format!("[notify] Failed to send to @{}: {}", agent_name, e));
                    // Queue notification for later
                    queue_notification(agent_name, conv_id, message_id, logger);
                    return;
                }

                // Read response (but don't wait too long)
                match tokio::time::timeout(
                    std::time::Duration::from_secs(300), // 5 min timeout for agent response
                    api.read_response()
                ).await {
                    Ok(Ok(Some(Response::Notified { response_message_id }))) => {
                        logger.log(&format!("[notify] @{} responded with msg_id={}", agent_name, response_message_id));
                    }
                    Ok(Ok(Some(Response::Error { message }))) => {
                        logger.log(&format!("[notify] @{} error: {}", agent_name, message));
                    }
                    Ok(Ok(Some(Response::NotifyReceived))) => {
                        logger.log(&format!("[notify] @{} acknowledged (async)", agent_name));
                    }
                    Ok(Ok(Some(_))) => {
                        logger.log(&format!("[notify] @{} unexpected response", agent_name));
                    }
                    Ok(Ok(None)) => {
                        logger.log(&format!("[notify] @{} connection closed", agent_name));
                    }
                    Ok(Err(e)) => {
                        logger.log(&format!("[notify] @{} read error: {}", agent_name, e));
                    }
                    Err(_) => {
                        logger.log(&format!("[notify] @{} timeout waiting for response", agent_name));
                    }
                }
            }
            Err(e) => {
                logger.log(&format!("[notify] @{} not reachable ({}), queuing", agent_name, e));
                queue_notification(agent_name, conv_id, message_id, logger);
            }
        }
    } else {
        // Agent not running - queue notification
        logger.log(&format!("[notify] @{} not running, queuing notification", agent_name));
        queue_notification(agent_name, conv_id, message_id, logger);
    }
}

/// Heartbeat loop that fires periodically.
/// Queues heartbeat if agent is busy (thinking lock held).
async fn heartbeat_loop(
    config: HeartbeatDaemonConfig,
    agent: Arc<Mutex<Agent>>,
    agent_name: String,
    persona: Option<String>,
    always: Option<String>,
    model_always: Option<String>,
    allowed_tools: Option<Vec<String>>,
    semantic_memory: Option<Arc<Mutex<SemanticMemoryStore>>>,
    embedding_client: Option<Arc<EmbeddingClient>>,
    tool_registry: Option<Arc<ToolRegistry>>,
    use_native_tools: bool,
    logger: Arc<AgentLogger>,
    recall_limit: usize,
    heartbeat_pending: Arc<AtomicBool>,
    shutdown: Arc<tokio::sync::Notify>,
) {
    let mut interval = tokio::time::interval(config.interval);
    // Skip the first immediate tick - first heartbeat fires AFTER first interval
    interval.tick().await;

    // Polling interval for checking pending heartbeats (1 second)
    let mut pending_check = tokio::time::interval(Duration::from_secs(1));

    loop {
        tokio::select! {
            _ = interval.tick() => {
                logger.log("[heartbeat] Timer fired");

                // Try to acquire the agent lock (non-blocking check)
                // We use try_lock to see if agent is busy
                match agent.try_lock() {
                    Ok(guard) => {
                        // Agent is free, drop the guard and run heartbeat
                        drop(guard);
                        run_heartbeat(
                            &config,
                            &agent,
                            &agent_name,
                            &persona,
                            &always,
                            &model_always,
                            &allowed_tools,
                            &semantic_memory,
                            &embedding_client,
                            &tool_registry,
                            use_native_tools,
                            &logger,
                            recall_limit,
                        ).await;
                    }
                    Err(_) => {
                        // Agent is busy, queue the heartbeat
                        logger.log("[heartbeat] Agent busy, queuing heartbeat");
                        heartbeat_pending.store(true, Ordering::SeqCst);
                    }
                }
            }
            _ = pending_check.tick() => {
                // Check if there's a pending heartbeat and agent is now free
                if heartbeat_pending.load(Ordering::SeqCst) {
                    if let Ok(guard) = agent.try_lock() {
                        // Agent is free now, clear pending and run heartbeat
                        heartbeat_pending.store(false, Ordering::SeqCst);
                        drop(guard);
                        logger.log("[heartbeat] Running pending heartbeat");
                        run_heartbeat(
                            &config,
                            &agent,
                            &agent_name,
                            &persona,
                            &always,
                            &model_always,
                            &allowed_tools,
                            &semantic_memory,
                            &embedding_client,
                            &tool_registry,
                            use_native_tools,
                            &logger,
                            recall_limit,
                        ).await;
                    }
                }
            }
            _ = shutdown.notified() => {
                logger.log("[heartbeat] Shutting down heartbeat loop");
                break;
            }
        }
    }
}

/// Execute a heartbeat: load heartbeat.md, think, store response in <agent>-heartbeat conversation.
async fn run_heartbeat(
    config: &HeartbeatDaemonConfig,
    agent: &Arc<Mutex<Agent>>,
    agent_name: &str,
    persona: &Option<String>,
    always: &Option<String>,
    model_always: &Option<String>,
    allowed_tools: &Option<Vec<String>>,
    semantic_memory: &Option<Arc<Mutex<SemanticMemoryStore>>>,
    embedding_client: &Option<Arc<EmbeddingClient>>,
    tool_registry: &Option<Arc<ToolRegistry>>,
    use_native_tools: bool,
    logger: &Arc<AgentLogger>,
    recall_limit: usize,
) {
    // 1. Load heartbeat.md content
    let heartbeat_prompt = match std::fs::read_to_string(&config.heartbeat_path) {
        Ok(content) if !content.trim().is_empty() => content,
        Ok(_) => {
            logger.log("[heartbeat] heartbeat.md is empty, skipping");
            return;
        }
        Err(e) => {
            logger.log(&format!("[heartbeat] Failed to read heartbeat.md: {}", e));
            return;
        }
    };

    logger.log(&format!("[heartbeat] Running with prompt: {} chars", heartbeat_prompt.len()));

    // 2. Get or create <agent>-heartbeat conversation
    let conv_name = format!("{}-heartbeat", agent_name);
    let store = match ConversationStore::init() {
        Ok(s) => s,
        Err(e) => {
            logger.log(&format!("[heartbeat] Failed to open conversation store: {}", e));
            return;
        }
    };

    // Create conversation if it doesn't exist (agent is only participant)
    if store.find_by_name(&conv_name).ok().flatten().is_none() {
        if let Err(e) = store.create_conversation(Some(&conv_name), &[agent_name]) {
            logger.log(&format!("[heartbeat] Failed to create conversation: {}", e));
            return;
        }
        logger.log(&format!("[heartbeat] Created conversation '{}'", conv_name));
    }

    // 3. Get conversation context (recent heartbeat outputs for continuity)
    // Heartbeat is a self-conversation - all messages are from this agent
    // Format them as assistant messages to show the model its previous outputs
    let context_messages = store.get_messages(&conv_name, Some(recall_limit)).unwrap_or_default();
    let (conversation_history, _) = format_conversation_history(&context_messages, agent_name);

    logger.log(&format!("[heartbeat] Context: {} previous outputs", conversation_history.len()));

    // 4. Build effective always with tools and memory injection
    let tools_injection = if use_native_tools {
        String::new()
    } else if let Some(registry) = tool_registry {
        let relevant = registry.find_relevant(&heartbeat_prompt, recall_limit);
        let relevant = filter_by_allowlist(relevant, allowed_tools);
        ToolRegistry::format_for_prompt(&relevant)
    } else {
        String::new()
    };

    let memory_injection = if let Some(mem_store) = semantic_memory {
        let query_embedding = if let Some(emb_client) = embedding_client {
            emb_client.embed(&heartbeat_prompt).await.ok()
        } else {
            None
        };

        let store_guard = mem_store.lock().await;
        match store_guard.recall_with_embedding(&heartbeat_prompt, recall_limit, query_embedding.as_deref()) {
            Ok(memories) => {
                if !memories.is_empty() {
                    logger.memory(&format!("[heartbeat] Recall: {} memories", memories.len()));
                }
                let entries: Vec<_> = memories.iter().map(|(m, _)| m.clone()).collect();
                build_memory_injection(&entries)
            }
            Err(_) => String::new()
        }
    } else {
        String::new()
    };

    let effective_always = build_effective_always(&tools_injection, &memory_injection, always, model_always);

    // 5. Think - heartbeat_prompt is the user message, previous outputs are conversation_history
    let options = ThinkOptions {
        system_prompt: persona.clone(),
        always_prompt: effective_always,
        conversation_history: if conversation_history.is_empty() { None } else { Some(conversation_history) },
        ..Default::default()
    };

    let mut agent_guard = agent.lock().await;
    let result = match agent_guard.think_with_options(&heartbeat_prompt, options).await {
        Ok(r) => r,
        Err(e) => {
            logger.log(&format!("[heartbeat] Think error: {}", e));
            return;
        }
    };
    drop(agent_guard);

    // 6. Process response: strip thinking tags, extract memories
    let without_thinking = strip_thinking_tags(&result.response);
    let (cleaned_response, memories_to_save) = extract_remember_tags(&without_thinking);

    // Save memories
    if !memories_to_save.is_empty() {
        if let Some(mem_store) = semantic_memory {
            let store_guard = mem_store.lock().await;
            for memory in &memories_to_save {
                let embedding = if let Some(emb_client) = embedding_client {
                    emb_client.embed(memory).await.ok()
                } else {
                    None
                };
                match store_guard.save_with_embedding(memory, 0.9, "explicit", embedding.as_deref()) {
                    Ok(SaveResult::New(id)) => logger.memory(&format!("[heartbeat] Save #{}: \"{}\"", id, memory)),
                    Ok(SaveResult::Reinforced(id, old, new)) => logger.memory(&format!("[heartbeat] Reinforce #{}: ({:.2} → {:.2})", id, old, new)),
                    Err(e) => logger.log(&format!("[heartbeat] Failed to save memory: {}", e)),
                }
            }
        }
    }

    // 7. Store response in <agent>-heartbeat conversation (just the response, not the prompt)
    match store.add_message(&conv_name, agent_name, &cleaned_response, &[]) {
        Ok(msg_id) => {
            logger.log(&format!("[heartbeat] Stored response as msg_id={}", msg_id));
        }
        Err(e) => {
            logger.log(&format!("[heartbeat] Failed to store response: {}", e));
        }
    }

    // 8. Parse @mentions from response and notify (reuse existing logic)
    let mentions = crate::conversation::parse_mentions(&cleaned_response);
    let valid_mentions: Vec<String> = mentions
        .into_iter()
        .filter(|m| m != agent_name && m != "user" && m != "all")
        .filter(|m| discovery::agent_exists(m))
        .collect();

    if !valid_mentions.is_empty() {
        logger.log(&format!("[heartbeat] Notifying {} agents: {:?}", valid_mentions.len(), valid_mentions));

        // Get the message ID we just stored
        if let Ok(msgs) = store.get_messages(&conv_name, Some(1)) {
            if let Some(last_msg) = msgs.last() {
                for mention in valid_mentions {
                    // Add mentioned agent as participant
                    let _ = store.add_participant(&conv_name, &mention);

                    forward_notify_to_agent(
                        &mention,
                        &conv_name,
                        last_msg.id,
                        0, // Start at depth 0
                        logger,
                    ).await;
                }
            }
        }
    }

    logger.log(&format!("[heartbeat] Complete. Response: {} chars", cleaned_response.len()));
}

/// Task watcher loop for Claude Code tasks.
/// Checks running tasks every 10 seconds and notifies the agent when they complete.
async fn task_watcher_loop(
    task_store: Arc<Mutex<TaskStore>>,
    agent_name: String,
    logger: Arc<AgentLogger>,
    shutdown: Arc<tokio::sync::Notify>,
) {
    let mut interval = tokio::time::interval(Duration::from_secs(10));
    // Skip the first immediate tick
    interval.tick().await;

    loop {
        tokio::select! {
            _ = interval.tick() => {
                // Get running tasks for this agent
                let running_tasks = {
                    let store = task_store.lock().await;
                    match store.get_running_tasks() {
                        Ok(tasks) => tasks.into_iter().filter(|t| t.agent == agent_name).collect::<Vec<_>>(),
                        Err(e) => {
                            logger.log(&format!("[task-watcher] Error getting running tasks: {}", e));
                            continue;
                        }
                    }
                };

                for task in running_tasks {
                    if let Some(pid) = task.pid {
                        if !is_process_running(pid) {
                            // Task has completed - read log and update status
                            logger.log(&format!("[task-watcher] Task {} (pid {}) completed", task.id, pid));

                            let log_path = format!("/tmp/claude-{}.log", task.id);
                            let (exit_code, output_summary) = read_task_output(&log_path, &logger);

                            let status = if exit_code == 0 {
                                TaskStatus::Completed
                            } else {
                                TaskStatus::Failed
                            };

                            // Update task in store
                            {
                                let store = task_store.lock().await;
                                if let Err(e) = store.complete_task(&task.id, status.clone(), exit_code, &output_summary) {
                                    logger.log(&format!("[task-watcher] Error updating task {}: {}", task.id, e));
                                    continue;
                                }
                            }

                            // Notify the agent via a conversation message
                            notify_task_complete(&task.agent, &task.id, task.conv_id.as_deref(), status, exit_code, &output_summary, &logger).await;
                        }
                    }
                }
            }
            _ = shutdown.notified() => {
                logger.log("[task-watcher] Shutting down task watcher");
                break;
            }
        }
    }
}

/// Read the output from a Claude Code task log file.
/// Returns (exit_code, summary).
fn read_task_output(log_path: &str, logger: &AgentLogger) -> (i32, String) {
    // Read the log file
    let log_content = match std::fs::read_to_string(log_path) {
        Ok(content) => content,
        Err(e) => {
            logger.log(&format!("[task-watcher] Error reading log {}: {}", log_path, e));
            return (-1, format!("Error reading log: {}", e));
        }
    };

    // Get last 100 lines for the summary
    let lines: Vec<&str> = log_content.lines().collect();
    let last_lines: String = if lines.len() > 100 {
        lines[lines.len() - 100..].join("\n")
    } else {
        lines.join("\n")
    };

    // Try to extract exit code from the log
    // Look for patterns like "exit code: N" or "exited with N"
    let exit_code = if log_content.contains("exit code: 0") || log_content.contains("exited with 0") {
        0
    } else if log_content.contains("exit code: 1") || log_content.contains("exited with 1") {
        1
    } else if log_content.contains("error") || log_content.contains("Error") || log_content.contains("ERROR") {
        // Assume failure if errors present
        1
    } else {
        // Default to success if no clear indicators
        0
    };

    (exit_code, last_lines)
}

/// Notify an agent that a Claude Code task has completed.
/// Posts completion notification to the source conversation with @mention.
async fn notify_task_complete(
    agent_name: &str,
    task_id: &str,
    conv_id: Option<&str>,
    status: TaskStatus,
    exit_code: i32,
    output_summary: &str,
    logger: &Arc<AgentLogger>,
) {
    // conv_id is required - task must have been invoked from a conversation
    let conv_name = match conv_id {
        Some(id) => id.to_string(),
        None => {
            logger.log(&format!("[task-watcher] Task {} has no conv_id, cannot notify", task_id));
            return;
        }
    };

    // Open conversation store
    let store = match ConversationStore::init() {
        Ok(s) => s,
        Err(e) => {
            logger.log(&format!("[task-watcher] Failed to open conversation store: {}", e));
            return;
        }
    };

    // Format notification message
    let status_str = match status {
        TaskStatus::Completed => "completed",
        TaskStatus::Failed => "failed",
        TaskStatus::Running => "running", // shouldn't happen
    };

    // Include @mention when posting to source conversation so the agent gets notified
    let message = format!(
        "@{} Claude Code task {} {} (exit code {}).\n\nOutput summary:\n```\n{}\n```",
        agent_name, task_id, status_str, exit_code, output_summary
    );

    // Add message to conversation
    match store.add_message(&conv_name, "system", &message, &[agent_name]) {
        Ok(msg_id) => {
            logger.log(&format!("[task-watcher] Stored notification as msg_id={}", msg_id));

            // Forward the notification to the agent daemon if it's running
            forward_notify_to_agent(
                agent_name,
                &conv_name,
                msg_id,
                0,
                logger,
            ).await;

            logger.log(&format!("[task-watcher] Notified @{} about task {}", agent_name, task_id));
        }
        Err(e) => {
            logger.log(&format!("[task-watcher] Failed to store notification: {}", e));
        }
    }
}

/// Handle a single connection from a client.
async fn handle_connection(
    mut api: SocketApi,
    agent: Arc<Mutex<Agent>>,
    agent_name: String,
    persona: Option<String>,
    always: Option<String>,
    model_always: Option<String>,
    allowed_tools: Option<Vec<String>>,
    semantic_memory: Option<Arc<Mutex<SemanticMemoryStore>>>,
    embedding_client: Option<Arc<EmbeddingClient>>,
    tool_registry: Option<Arc<ToolRegistry>>,
    use_native_tools: bool,
    shutdown: Arc<tokio::sync::Notify>,
    logger: Arc<AgentLogger>,
    recall_limit: usize,
    heartbeat_config: Option<HeartbeatDaemonConfig>,
    _heartbeat_pending: Arc<AtomicBool>,
    task_store: Option<Arc<Mutex<TaskStore>>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Create tool execution context for tools that need daemon state
    // Note: conv_id is None in handle_connection context (direct socket messages)
    let tool_context = ToolExecutionContext {
        agent_name: agent_name.clone(),
        task_store: task_store.clone(),
        conv_id: None,
        semantic_memory_store: semantic_memory.clone(),
        embedding_client: embedding_client.clone(),
    };
    loop {
        // Read request with a timeout
        let request = tokio::select! {
            result = api.read_request() => {
                match result {
                    Ok(Some(req)) => req,
                    Ok(None) => break, // Connection closed
                    Err(e) => {
                        eprintln!("Error reading request: {}", e);
                        break;
                    }
                }
            }
            _ = shutdown.notified() => {
                break;
            }
        };

        let response = match request {
            Request::Message { ref content } => {
                logger.log(&format!("[socket] Received message: {}", content));

                // Get relevant tools from registry (used for both modes)
                let relevant_tools = if let Some(ref registry) = tool_registry {
                    let relevant = registry.find_relevant(content, recall_limit);
                    let relevant = filter_by_allowlist(relevant, &allowed_tools);
                    if !relevant.is_empty() {
                        logger.tool(&format!("Recall: {} tools for query", relevant.len()));
                        for t in &relevant {
                            logger.tool(&format!("  - {}", t.name));
                        }
                    }
                    relevant
                } else {
                    Vec::new()
                };

                // Build tools injection for JSON-block mode, or tool specs for native mode
                let (tools_injection, external_tools) = if use_native_tools {
                    let specs = if !relevant_tools.is_empty() {
                        Some(tool_definitions_to_specs(&relevant_tools))
                    } else {
                        None
                    };
                    (String::new(), specs)
                } else {
                    (ToolRegistry::format_for_prompt(&relevant_tools), None)
                };

                // Inject relevant memories if enabled
                let memory_injection = if let Some(ref mem_store) = semantic_memory {
                    // Generate query embedding if we have an embedding client
                    let query_embedding = if let Some(ref emb_client) = embedding_client {
                        match emb_client.embed(content).await {
                            Ok(emb) => Some(emb),
                            Err(e) => {
                                logger.log(&format!("[socket] Failed to generate query embedding: {}", e));
                                None
                            }
                        }
                    } else {
                        None
                    };

                    let store = mem_store.lock().await;
                    match store.recall_with_embedding(content, recall_limit, query_embedding.as_deref()) {
                        Ok(memories) => {
                            if !memories.is_empty() {
                                logger.memory(&format!("Recall: {} memories for query", memories.len()));
                                for (m, score) in &memories {
                                    logger.memory(&format!("  ({:.3}) \"{}\" [#{}]", score, m.content, m.id));
                                }
                            }
                            let entries: Vec<_> = memories.iter().map(|(m, _)| m.clone()).collect();
                            build_memory_injection(&entries)
                        }
                        Err(e) => {
                            logger.log(&format!("[socket] Memory recall error: {}", e));
                            String::new()
                        }
                    }
                } else {
                    String::new()
                };

                // Combine tools, memory injection, and always prompt
                let effective_always = build_effective_always(&tools_injection, &memory_injection, &always, &model_always);

                let agent_guard = agent.lock().await;

                if use_native_tools {
                    // Native tool mode with streaming: Agent handles tool calling internally
                    let options = ThinkOptions {
                        system_prompt: persona.clone(),
                        always_prompt: effective_always.clone(),
                        external_tools: external_tools.clone(),
                        ..Default::default()
                    };

                    // Create channel for streaming tokens
                    let (token_tx, mut token_rx) = mpsc::channel::<String>(100);

                    // We need to drop the agent_guard so we can spawn the thinking task
                    drop(agent_guard);

                    // Clone things needed in the spawned task
                    let agent_clone = agent.clone();
                    let content_clone = content.clone();

                    // Spawn thinking in background
                    let think_handle = tokio::spawn(async move {
                        let mut agent_guard = agent_clone.lock().await;
                        agent_guard.think_streaming_with_options(&content_clone, options, token_tx).await
                    });

                    // Forward tokens to socket as they arrive
                    while let Some(token) = token_rx.recv().await {
                        if let Err(e) = api.write_response(&Response::Chunk { text: token }).await {
                            logger.log(&format!("[socket] Error writing chunk: {}", e));
                            break;
                        }
                    }

                    // Wait for completion
                    let result = think_handle.await;

                    // Handle the result
                    let final_response = match result {
                        Ok(Ok(response)) => {
                            // Strip thinking tags and extract [REMEMBER: ...] tags
                            let without_thinking = strip_thinking_tags(&response);
                            let (after_remember, memories_to_save) = extract_remember_tags(&without_thinking);
                            if !memories_to_save.is_empty() {
                                if let Some(ref mem_store) = semantic_memory {
                                    let store = mem_store.lock().await;
                                    for memory in &memories_to_save {
                                        // Generate embedding if we have a client
                                        let embedding = if let Some(ref emb_client) = embedding_client {
                                            match emb_client.embed(memory).await {
                                                Ok(emb) => Some(emb),
                                                Err(e) => {
                                                    logger.log(&format!("[socket] Failed to generate embedding: {}", e));
                                                    None
                                                }
                                            }
                                        } else {
                                            None
                                        };

                                        match store.save_with_embedding(memory, 0.9, "explicit", embedding.as_deref()) {
                                            Ok(SaveResult::New(id)) => logger.memory(&format!("Save #{}: \"{}\"", id, memory)),
                                            Ok(SaveResult::Reinforced(id, old, new)) => logger.memory(&format!("Reinforce #{}: \"{}\" ({:.2} → {:.2})", id, memory, old, new)),
                                            Err(e) => logger.log(&format!("[socket] Failed to save memory: {}", e)),
                                        }
                                    }
                                }
                            }
                            after_remember
                        }
                        Ok(Err(e)) => format!("Error: {}", e),
                        Err(e) => format!("Error: task panicked: {}", e),
                    };

                    // Send Done to signal stream complete
                    if let Err(e) = api.write_response(&Response::Done).await {
                        logger.log(&format!("[socket] Error writing Done: {}", e));
                    }

                    // Log the final response (but don't send it - already streamed)
                    logger.log(&format!("[socket] Final response: {} bytes", final_response.len()));

                    // Continue to next request (don't send Response::Message)
                    continue;
                } else {
                    // JSON-block mode with streaming: Daemon parses and executes tool calls
                    // We need to drop the agent_guard so we can re-acquire it in the loop
                    drop(agent_guard);

                    let mut current_message = content.clone();
                    let max_tool_calls = 10;
                    let mut tool_call_count = 0;

                    loop {
                        let options = ThinkOptions {
                            system_prompt: persona.clone(),
                            always_prompt: effective_always.clone(),
                            external_tools: None,
                            ..Default::default()
                        };

                        // Create channel for streaming tokens
                        let (token_tx, mut token_rx) = mpsc::channel::<String>(100);

                        // Clone things needed in the spawned task
                        let agent_clone = agent.clone();
                        let current_message_clone = current_message.clone();

                        // Spawn thinking in background
                        let think_handle = tokio::spawn(async move {
                            let mut agent_guard = agent_clone.lock().await;
                            agent_guard.think_streaming_with_options(&current_message_clone, options, token_tx).await
                        });

                        // Forward tokens to socket as they arrive, but suppress tool call blocks
                        // Tool blocks look like: ```json\n{"tool": "...", "params": {...}}\n```
                        let mut in_code_block = false;
                        let mut code_block_buffer = String::new();

                        while let Some(token) = token_rx.recv().await {
                            if !in_code_block {
                                // Check if this token starts or contains a code block
                                if token.contains("```") {
                                    in_code_block = true;
                                    code_block_buffer = token;
                                    // Check if the code block is already complete in this token
                                    if code_block_buffer.matches("```").count() >= 2 {
                                        // Complete block in single token - check if it's a tool call
                                        if code_block_buffer.contains("\"tool\"") && code_block_buffer.contains("\"params\"") {
                                            // Tool call - suppress
                                        } else {
                                            // Not a tool call - send it
                                            if let Err(e) = api.write_response(&Response::Chunk { text: code_block_buffer.clone() }).await {
                                                logger.log(&format!("[socket] Error writing chunk: {}", e));
                                                break;
                                            }
                                        }
                                        in_code_block = false;
                                        code_block_buffer.clear();
                                    }
                                    continue;
                                }
                                // Normal token - send it
                                if let Err(e) = api.write_response(&Response::Chunk { text: token }).await {
                                    logger.log(&format!("[socket] Error writing chunk: {}", e));
                                    break;
                                }
                            } else {
                                // Inside code block - buffer
                                code_block_buffer.push_str(&token);

                                // Check if code block ended
                                if code_block_buffer.matches("```").count() >= 2 {
                                    // Code block complete - check if it's a tool call
                                    if code_block_buffer.contains("\"tool\"") && code_block_buffer.contains("\"params\"") {
                                        // Tool call - suppress (don't send)
                                    } else {
                                        // Not a tool call - send the buffered content
                                        if let Err(e) = api.write_response(&Response::Chunk { text: code_block_buffer.clone() }).await {
                                            logger.log(&format!("[socket] Error writing chunk: {}", e));
                                            break;
                                        }
                                    }
                                    in_code_block = false;
                                    code_block_buffer.clear();
                                }
                            }
                        }

                        // Handle any remaining buffer at end (incomplete code block)
                        if !code_block_buffer.is_empty() {
                            // If it doesn't look like a tool call, send it
                            if !(code_block_buffer.contains("\"tool\"") && code_block_buffer.contains("\"params\"")) {
                                if let Err(e) = api.write_response(&Response::Chunk { text: code_block_buffer }).await {
                                    logger.log(&format!("[socket] Error writing final buffer: {}", e));
                                }
                            }
                        }

                        // Wait for completion and get the full response
                        let result = think_handle.await;

                        let llm_response = match result {
                            Ok(Ok(response)) => response,
                            Ok(Err(e)) => {
                                logger.log(&format!("[socket] LLM error: {}", e));
                                if let Err(e) = api.write_response(&Response::Done).await {
                                    logger.log(&format!("[socket] Error writing Done: {}", e));
                                }
                                continue; // Move to next request
                            }
                            Err(e) => {
                                logger.log(&format!("[socket] Task panic: {}", e));
                                if let Err(e) = api.write_response(&Response::Done).await {
                                    logger.log(&format!("[socket] Error writing Done: {}", e));
                                }
                                continue; // Move to next request
                            }
                        };

                        // Strip thinking tags and extract [REMEMBER: ...] tags
                        let without_thinking = strip_thinking_tags(&llm_response);
                        let (after_remember, memories_to_save) = extract_remember_tags(&without_thinking);

                        if !memories_to_save.is_empty() {
                            if let Some(ref mem_store) = semantic_memory {
                                let store = mem_store.lock().await;
                                for memory in &memories_to_save {
                                    // Generate embedding if we have a client
                                    let embedding = if let Some(ref emb_client) = embedding_client {
                                        match emb_client.embed(memory).await {
                                            Ok(emb) => Some(emb),
                                            Err(e) => {
                                                logger.log(&format!("[socket] Failed to generate embedding: {}", e));
                                                None
                                            }
                                        }
                                    } else {
                                        None
                                    };

                                    match store.save_with_embedding(memory, 0.9, "explicit", embedding.as_deref()) {
                                        Ok(SaveResult::New(id)) => logger.memory(&format!("Save #{}: \"{}\"", id, memory)),
                                        Ok(SaveResult::Reinforced(id, old, new)) => logger.memory(&format!("Reinforce #{}: \"{}\" ({:.2} → {:.2})", id, memory, old, new)),
                                        Err(e) => logger.log(&format!("[socket] Failed to save memory: {}", e)),
                                    }
                                }
                            }
                        }

                        // Check for tool calls (JSON-block parsing)
                        let (cleaned_response, tool_call) = extract_tool_call(&after_remember);

                        if let Some(tc) = tool_call {
                            tool_call_count += 1;
                            if tool_call_count > max_tool_calls {
                                logger.tool("Max tool calls reached, stopping");
                                // Send Done to signal stream complete
                                if let Err(e) = api.write_response(&Response::Done).await {
                                    logger.log(&format!("[socket] Error writing Done: {}", e));
                                }
                                break;
                            }

                            // Send ToolCall event to client before executing
                            if let Err(e) = api.write_response(&Response::ToolCall {
                                tool: tc.tool.clone(),
                                params: tc.params.clone(),
                            }).await {
                                logger.log(&format!("[socket] Error writing ToolCall: {}", e));
                            }

                            logger.tool(&format!("Executing: {} with params {}", tc.tool, tc.params));

                            // Look up the tool definition to get allowed_commands
                            let tool_def = tool_registry.as_ref()
                                .and_then(|r| r.find_by_name(&tc.tool));

                            match execute_tool_call(&tc, tool_def, Some(&tool_context)).await {
                                Ok(tool_result) => {
                                    logger.tool(&format!("Result: {} bytes", tool_result.len()));
                                    current_message = format!("[Tool Result for {}]\n{}", tc.tool, tool_result);
                                    // Continue to next iteration - don't send Done yet
                                }
                                Err(e) => {
                                    logger.tool(&format!("Error: {}", e));
                                    current_message = format!("[Tool Error for {}]\n{}", tc.tool, e);
                                    // Continue to next iteration to let LLM handle the error
                                }
                            }
                        } else {
                            // No more tool calls - done
                            if let Err(e) = api.write_response(&Response::Done).await {
                                logger.log(&format!("[socket] Error writing Done: {}", e));
                            }
                            logger.log(&format!("[socket] Final response: {} bytes", cleaned_response.len()));
                            break;
                        }
                    }

                    // Continue to next request (don't send Response::Message - already streamed)
                    continue;
                }
            }

            Request::IncomingMessage { ref from, ref content } => {
                logger.log(&format!("[socket] Incoming message from {}: {}", from, content));

                // Format the message with [sender] prefix for the agent
                let formatted_message = format!("[{}] {}", from, content);

                // Inject relevant tools if registry is available
                let tools_injection = if let Some(ref registry) = tool_registry {
                    let relevant = registry.find_relevant(content, recall_limit);
                    let relevant = filter_by_allowlist(relevant, &allowed_tools);
                    if !relevant.is_empty() {
                        logger.tool(&format!("Recall: {} tools for incoming", relevant.len()));
                        for t in &relevant {
                            logger.tool(&format!("  - {}", t.name));
                        }
                    }
                    ToolRegistry::format_for_prompt(&relevant)
                } else {
                    String::new()
                };

                // Inject relevant memories if enabled
                let memory_injection = if let Some(ref mem_store) = semantic_memory {
                    // Generate query embedding if we have an embedding client
                    let query_embedding = if let Some(ref emb_client) = embedding_client {
                        match emb_client.embed(content).await {
                            Ok(emb) => Some(emb),
                            Err(e) => {
                                logger.log(&format!("[socket] Failed to generate query embedding: {}", e));
                                None
                            }
                        }
                    } else {
                        None
                    };

                    let store = mem_store.lock().await;
                    match store.recall_with_embedding(content, recall_limit, query_embedding.as_deref()) {
                        Ok(memories) => {
                            if !memories.is_empty() {
                                logger.memory(&format!("Recall: {} memories for incoming", memories.len()));
                                for (m, score) in &memories {
                                    logger.memory(&format!("  ({:.3}) \"{}\" [#{}]", score, m.content, m.id));
                                }
                            }
                            let entries: Vec<_> = memories.iter().map(|(m, _)| m.clone()).collect();
                            build_memory_injection(&entries)
                        }
                        Err(e) => {
                            logger.log(&format!("[socket] Memory recall error: {}", e));
                            String::new()
                        }
                    }
                } else {
                    String::new()
                };

                // Combine tools, memory injection, and always prompt
                let effective_always = build_effective_always(&tools_injection, &memory_injection, &always, &model_always);

                let options = ThinkOptions {
                    system_prompt: persona.clone(),
                    always_prompt: effective_always,
                    ..Default::default()
                };

                let mut agent_guard = agent.lock().await;
                match agent_guard.think_with_options(&formatted_message, options).await {
                    Ok(result) => {
                        // Strip thinking tags and extract [REMEMBER: ...] tags
                        let without_thinking = strip_thinking_tags(&result.response);
                        let (cleaned_response, memories_to_save) = extract_remember_tags(&without_thinking);

                        if !memories_to_save.is_empty() {
                            if let Some(ref mem_store) = semantic_memory {
                                let store = mem_store.lock().await;
                                for memory in &memories_to_save {
                                    // Generate embedding if we have a client
                                    let embedding = if let Some(ref emb_client) = embedding_client {
                                        match emb_client.embed(memory).await {
                                            Ok(emb) => Some(emb),
                                            Err(e) => {
                                                logger.log(&format!("[socket] Failed to generate embedding: {}", e));
                                                None
                                            }
                                        }
                                    } else {
                                        None
                                    };

                                    match store.save_with_embedding(memory, 0.9, "explicit", embedding.as_deref()) {
                                        Ok(SaveResult::New(id)) => logger.memory(&format!("Save #{}: \"{}\"", id, memory)),
                                        Ok(SaveResult::Reinforced(id, old, new)) => logger.memory(&format!("Reinforce #{}: \"{}\" ({:.2} → {:.2})", id, memory, old, new)),
                                        Err(e) => logger.log(&format!("[socket] Failed to save memory: {}", e)),
                                    }
                                }
                            }
                        }

                        logger.log(&format!("[socket] Response to {}: {}", from, cleaned_response));
                        Response::Message { content: cleaned_response }
                    }
                    Err(e) => {
                        Response::Error { message: e.to_string() }
                    }
                }
            }

            Request::Notify { ref conv_id, message_id, depth } => {
                logger.log(&format!("[socket] Notify: conv={} msg_id={} depth={}", conv_id, message_id, depth));

                // Send immediate ack - fire-and-forget semantics
                if let Err(e) = api.write_response(&Response::NotifyReceived).await {
                    logger.log(&format!("[socket] Error writing NotifyReceived: {}", e));
                    continue;
                }

                // Spawn async processing - don't block the caller
                let conv_id = conv_id.clone();
                let agent = agent.clone();
                let agent_name = agent_name.clone();
                let persona = persona.clone();
                let always = always.clone();
                let model_always = model_always.clone();
                let allowed_tools = allowed_tools.clone();
                let semantic_memory = semantic_memory.clone();
                let embedding_client = embedding_client.clone();
                let tool_registry = tool_registry.clone();
                let logger = logger.clone();
                let task_store = task_store.clone();

                tokio::spawn(async move {
                    let _response = handle_notify(
                        &conv_id,
                        message_id,
                        depth,
                        &agent,
                        &agent_name,
                        &persona,
                        &always,
                        &model_always,
                        &allowed_tools,
                        &semantic_memory,
                        &embedding_client,
                        &tool_registry,
                        use_native_tools,
                        &logger,
                        recall_limit,
                        &task_store,
                    ).await;
                    // Response is logged inside handle_notify, no need to do anything with it
                });

                // Continue to next request - we already sent the ack
                continue;
            }

            Request::Status => {
                let agent_guard = agent.lock().await;
                Response::Status {
                    running: true,
                    history_len: agent_guard.history_len(),
                }
            }

            Request::Shutdown => {
                println!("[socket] Received shutdown request");
                shutdown.notify_waiters();
                Response::Ok
            }

            Request::Clear => {
                println!("[socket] Clearing conversation history");
                let mut agent_guard = agent.lock().await;
                agent_guard.clear_history();
                Response::Ok
            }

            Request::ListAgents => {
                println!("[socket] Listing agents");
                let agents: Vec<String> = discovery::discover_running_agents()
                    .into_iter()
                    .map(|a| a.name)
                    .collect();
                Response::Agents { agents }
            }

            Request::System => {
                logger.log("[socket] System prompt requested");
                let persona_content = persona.clone().unwrap_or_else(|| "(no persona configured)".to_string());
                Response::System { persona: persona_content }
            }

            Request::Heartbeat => {
                logger.log("[socket] Manual heartbeat requested");

                // Check if heartbeat is configured
                if let Some(ref hb_config) = heartbeat_config {
                    // Check if heartbeat.md exists
                    if !hb_config.heartbeat_path.exists() {
                        logger.log("[socket] heartbeat.md not found");
                        Response::Error { message: "heartbeat.md not found".to_string() }
                    } else {
                        // Run heartbeat immediately (skip busy check for manual trigger)
                        run_heartbeat(
                            hb_config,
                            &agent,
                            &agent_name,
                            &persona,
                            &always,
                            &model_always,
                            &allowed_tools,
                            &semantic_memory,
                            &embedding_client,
                            &tool_registry,
                            use_native_tools,
                            &logger,
                            recall_limit,
                        ).await;
                        Response::HeartbeatTriggered
                    }
                } else {
                    logger.log("[socket] Heartbeat not configured");
                    Response::HeartbeatNotConfigured
                }
            }
        };

        if let Err(e) = api.write_response(&response).await {
            eprintln!("Error writing response: {}", e);
            break;
        }

        // If we just processed a shutdown, break out
        if matches!(request, Request::Shutdown) {
            break;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use tempfile::tempdir;

    #[test]
    fn test_parse_duration_seconds() {
        assert_eq!(parse_duration("30s"), Some(Duration::from_secs(30)));
        assert_eq!(parse_duration("1sec"), Some(Duration::from_secs(1)));
    }

    #[test]
    fn test_parse_duration_minutes() {
        assert_eq!(parse_duration("5m"), Some(Duration::from_secs(300)));
        assert_eq!(parse_duration("1min"), Some(Duration::from_secs(60)));
    }

    #[test]
    fn test_parse_duration_hours() {
        assert_eq!(parse_duration("1h"), Some(Duration::from_secs(3600)));
        assert_eq!(parse_duration("2hrs"), Some(Duration::from_secs(7200)));
    }

    #[test]
    fn test_parse_duration_invalid() {
        assert_eq!(parse_duration(""), None);
        assert_eq!(parse_duration("abc"), None);
        assert_eq!(parse_duration("5x"), None);
    }

    #[test]
    fn test_parse_duration_compound() {
        // Compound durations like "2h30m", "1h30m15s"
        assert_eq!(parse_duration("2h30m"), Some(Duration::from_secs(2 * 3600 + 30 * 60)));
        assert_eq!(parse_duration("1h30m15s"), Some(Duration::from_secs(3600 + 30 * 60 + 15)));
        assert_eq!(parse_duration("1m30s"), Some(Duration::from_secs(60 + 30)));
    }

    #[test]
    fn test_pid_file_lifecycle() {
        let dir = tempdir().unwrap();
        let pid_path = dir.path().join("test.pid");

        // Create PID file
        {
            let pid_file = PidFile::create(&pid_path).unwrap();
            assert!(pid_path.exists());

            // Read back the PID
            let read_pid = PidFile::read(&pid_path).unwrap();
            assert_eq!(read_pid, std::process::id());

            // is_running should return true for current process
            assert!(PidFile::is_running(&pid_path));

            // Explicit remove
            pid_file.remove().unwrap();
            assert!(!pid_path.exists());
        }
    }

    #[test]
    fn test_pid_file_drop() {
        let dir = tempdir().unwrap();
        let pid_path = dir.path().join("test.pid");

        {
            let _pid_file = PidFile::create(&pid_path).unwrap();
            assert!(pid_path.exists());
        }
        // PID file should be removed on drop
        assert!(!pid_path.exists());
    }

    #[test]
    fn test_pid_file_is_running_nonexistent() {
        let dir = tempdir().unwrap();
        let pid_path = dir.path().join("nonexistent.pid");
        assert!(!PidFile::is_running(&pid_path));
    }

    #[test]
    fn test_pid_file_is_running_invalid_pid() {
        let dir = tempdir().unwrap();
        let pid_path = dir.path().join("invalid.pid");

        // Write an invalid PID (unlikely to exist)
        std::fs::write(&pid_path, "999999999").unwrap();
        assert!(!PidFile::is_running(&pid_path));
    }

    #[test]
    #[serial]
    fn test_daemon_config_from_agent_dir() {
        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test-agent"
persona_file = "persona.md"

[llm]
provider = "openai"
model = "gpt-4"
api_key = "sk-test"

[memory]
path = "memory.db"

[timer]
enabled = true
interval = "5m"
message = "heartbeat"
"#;
        std::fs::write(dir.path().join("config.toml"), config_content).unwrap();
        std::fs::write(dir.path().join("persona.md"), "Test persona").unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        let daemon_config = DaemonConfig::from_agent_dir(&agent_dir).unwrap();

        assert_eq!(daemon_config.name, "test-agent");
        assert_eq!(daemon_config.socket_path, dir.path().join("agent.sock"));
        assert_eq!(daemon_config.pid_path, dir.path().join("daemon.pid"));
        assert!(daemon_config.timer.is_some());

        let timer = daemon_config.timer.unwrap();
        assert_eq!(timer.interval, Duration::from_secs(300));
        assert_eq!(timer.message, "heartbeat");

        // Persona should start with original content and include runtime context
        let persona = daemon_config.persona.unwrap();
        assert!(persona.starts_with("Test persona"));
        assert!(persona.contains("You are running inside Anima, a multi-agent runtime."));
        assert!(persona.contains("agent=test-agent"));
        assert!(persona.contains("model=gpt-4"));
        assert!(persona.contains("tools=native"));
    }

    #[test]
    #[serial]
    fn test_daemon_config_no_timer() {
        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test-agent"

[llm]
provider = "openai"
model = "gpt-4"
api_key = "sk-test"
"#;
        std::fs::write(dir.path().join("config.toml"), config_content).unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        let daemon_config = DaemonConfig::from_agent_dir(&agent_dir).unwrap();

        assert!(daemon_config.timer.is_none());
        // Even without a persona file, runtime context is injected
        let persona = daemon_config.persona.unwrap();
        assert!(persona.contains("You are running inside Anima, a multi-agent runtime."));
        assert!(persona.contains("agent=test-agent"));
    }

    #[test]
    #[serial]
    fn test_daemon_config_timer_disabled() {
        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test-agent"

[llm]
provider = "openai"
model = "gpt-4"
api_key = "sk-test"

[timer]
enabled = false
interval = "5m"
"#;
        std::fs::write(dir.path().join("config.toml"), config_content).unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        let daemon_config = DaemonConfig::from_agent_dir(&agent_dir).unwrap();

        assert!(daemon_config.timer.is_none());
    }

    #[test]
    #[serial]
    fn test_daemon_config_with_always() {
        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test-agent"
persona_file = "persona.md"
always_file = "always.md"

[llm]
provider = "openai"
model = "gpt-4"
api_key = "sk-test"
"#;
        std::fs::write(dir.path().join("config.toml"), config_content).unwrap();
        std::fs::write(dir.path().join("persona.md"), "Test persona").unwrap();
        std::fs::write(dir.path().join("always.md"), "Always be concise.").unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        let daemon_config = DaemonConfig::from_agent_dir(&agent_dir).unwrap();

        // Persona should start with original content and include runtime context
        let persona = daemon_config.persona.unwrap();
        assert!(persona.starts_with("Test persona"));
        assert!(persona.contains("You are running inside Anima, a multi-agent runtime."));
        assert_eq!(daemon_config.always, Some("Always be concise.".to_string()));
    }

    #[test]
    #[serial]
    fn test_daemon_config_always_file_missing() {
        // Use a fake HOME so the global ~/.anima/agents/always.md fallback doesn't interfere
        let fake_home = tempdir().unwrap();
        let original_home = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", fake_home.path()) };

        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test-agent"
always_file = "always.md"

[llm]
provider = "openai"
model = "gpt-4"
api_key = "sk-test"
"#;
        std::fs::write(dir.path().join("config.toml"), config_content).unwrap();
        // Note: always.md file is NOT created

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        let daemon_config = DaemonConfig::from_agent_dir(&agent_dir).unwrap();

        // Restore HOME
        match original_home {
            Some(h) => unsafe { std::env::set_var("HOME", h) },
            None => unsafe { std::env::remove_var("HOME") },
        }

        // Should be None when file is missing (backward compatible)
        assert!(daemon_config.always.is_none());
    }

    #[test]
    fn test_resolve_agent_path_name() {
        let path = resolve_agent_path("myagent");
        assert!(path.ends_with(".anima/agents/myagent"));
    }

    #[test]
    fn test_resolve_agent_path_absolute() {
        let path = resolve_agent_path("/some/absolute/path");
        assert_eq!(path, PathBuf::from("/some/absolute/path"));
    }

    #[test]
    fn test_resolve_agent_path_relative() {
        let path = resolve_agent_path("./myagent");
        assert_eq!(path, PathBuf::from("./myagent"));
    }

    #[tokio::test]
    async fn test_safe_shell_allowed_command() {
        use crate::tool_registry::ToolDefinition;

        let tool_def = ToolDefinition {
            name: "safe_shell".to_string(),
            description: "Safe shell".to_string(),
            params: serde_json::json!({"command": "string"}),
            keywords: vec!["shell".to_string()],
            category: Some("system".to_string()),
            allowed_commands: Some(vec!["ls".to_string(), "grep".to_string(), "cat".to_string()]),
        };

        let tool_call = ToolCall {
            tool: "safe_shell".to_string(),
            params: serde_json::json!({"command": "ls -la"}),
        };

        // Should succeed - "ls" is in allowed list
        let result = execute_tool_call(&tool_call, Some(&tool_def), None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_safe_shell_blocked_command() {
        use crate::tool_registry::ToolDefinition;

        let tool_def = ToolDefinition {
            name: "safe_shell".to_string(),
            description: "Safe shell".to_string(),
            params: serde_json::json!({"command": "string"}),
            keywords: vec!["shell".to_string()],
            category: Some("system".to_string()),
            allowed_commands: Some(vec!["ls".to_string(), "grep".to_string(), "cat".to_string()]),
        };

        let tool_call = ToolCall {
            tool: "safe_shell".to_string(),
            params: serde_json::json!({"command": "rm -rf /"}),
        };

        // Should fail - "rm" is not in allowed list
        let result = execute_tool_call(&tool_call, Some(&tool_def), None).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("not in allowed list"));
        assert!(err.contains("rm"));
    }

    #[tokio::test]
    async fn test_regular_shell_no_restrictions() {
        use crate::tool_registry::ToolDefinition;

        let tool_def = ToolDefinition {
            name: "shell".to_string(),
            description: "Regular shell".to_string(),
            params: serde_json::json!({"command": "string"}),
            keywords: vec!["shell".to_string()],
            category: Some("system".to_string()),
            allowed_commands: None,  // No restrictions
        };

        let tool_call = ToolCall {
            tool: "shell".to_string(),
            params: serde_json::json!({"command": "echo hello"}),
        };

        // Should succeed - no allowed_commands restriction
        let result = execute_tool_call(&tool_call, Some(&tool_def), None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_shell_without_tool_def() {
        let tool_call = ToolCall {
            tool: "shell".to_string(),
            params: serde_json::json!({"command": "echo hello"}),
        };

        // Should succeed - no tool_def means no restrictions
        let result = execute_tool_call(&tool_call, None, None).await;
        assert!(result.is_ok());
    }

    // =========================================================================
    // format_conversation_history tests
    // =========================================================================

    fn make_conv_msg(from: &str, content: &str) -> ConversationMessage {
        ConversationMessage {
            id: 1,
            conv_name: "test".to_string(),
            from_agent: from.to_string(),
            content: content.to_string(),
            mentions: vec![],
            created_at: 0,
            expires_at: i64::MAX,
        }
    }

    #[test]
    fn test_format_conversation_history_empty() {
        let (history, final_content) = format_conversation_history(&[], "arya");
        assert!(history.is_empty());
        assert!(final_content.is_empty());
    }

    #[test]
    fn test_format_conversation_history_single_user_message() {
        // Single message from user → should become final_content
        let msgs = vec![make_conv_msg("user", "hello")];
        let (history, final_content) = format_conversation_history(&msgs, "arya");

        assert!(history.is_empty());
        assert!(final_content.contains("\"from\": \"user\""));
        assert!(final_content.contains("\"text\": \"hello\""));
    }

    #[test]
    fn test_format_conversation_history_user_then_self() {
        // user → arya → user
        // Should map: [user→user JSON], [arya→assistant raw], final = [user→user JSON]
        let msgs = vec![
            make_conv_msg("user", "hi arya"),
            make_conv_msg("arya", "hey there!"),
            make_conv_msg("user", "what's up?"),
        ];
        let (history, final_content) = format_conversation_history(&msgs, "arya");

        // History should have 2 messages: user JSON, assistant raw
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].role, "user");
        assert!(history[0].content.as_ref().unwrap().contains("\"from\": \"user\""));
        assert_eq!(history[1].role, "assistant");
        assert_eq!(history[1].content.as_ref().unwrap(), "hey there!");

        // Final content should be the last user message in JSON format
        assert!(final_content.contains("\"from\": \"user\""));
        assert!(final_content.contains("what's up?"));
    }

    #[test]
    fn test_format_conversation_history_batches_consecutive_users() {
        // user → claude → user (should batch claude and user before arya responds)
        // For arya, both "user" and "claude" are non-self, so they batch together
        let msgs = vec![
            make_conv_msg("user", "hi @arya"),
            make_conv_msg("arya", "hey!"),
            make_conv_msg("claude", "I can help"),
            make_conv_msg("user", "thanks"),
        ];
        let (history, final_content) = format_conversation_history(&msgs, "arya");

        // History: [user JSON], [assistant raw]
        // Final: [claude JSON + user JSON batched]
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].role, "user");
        assert_eq!(history[1].role, "assistant");
        assert_eq!(history[1].content.as_ref().unwrap(), "hey!");

        // Final content should batch claude and user messages
        assert!(final_content.contains("\"from\": \"claude\""));
        assert!(final_content.contains("\"from\": \"user\""));
        assert!(final_content.contains("I can help"));
        assert!(final_content.contains("thanks"));
    }

    #[test]
    fn test_format_conversation_history_self_raw_others_json() {
        // Verify self messages are raw, others are JSON wrapped
        let msgs = vec![
            make_conv_msg("gendry", "need help"),
            make_conv_msg("arya", "what do you need?"),
        ];
        let (history, final_content) = format_conversation_history(&msgs, "arya");

        // Both messages go into history since last message is from self
        // gendry → user JSON, arya → assistant raw
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].role, "user");
        assert!(history[0].content.as_ref().unwrap().contains("\"from\": \"gendry\""));
        assert_eq!(history[1].role, "assistant");
        assert_eq!(history[1].content.as_ref().unwrap(), "what do you need?");

        // Last message from arya is from self, so final_content is empty (unusual case)
        // This typically shouldn't happen in NotifyReceived - the notification only
        // fires when there's a new message addressed to this agent
        assert!(final_content.is_empty());
    }

    #[test]
    fn test_format_conversation_history_escapes_special_chars() {
        // Verify special characters are escaped in JSON wrapper
        let msgs = vec![make_conv_msg("user", "line1\nline2\"quote\\backslash")];
        let (_, final_content) = format_conversation_history(&msgs, "arya");

        // Newlines should be escaped as \n (literal)
        assert!(final_content.contains("\\n"));
        // Quotes should be escaped
        assert!(final_content.contains("\\\""));
        // Backslashes should be escaped
        assert!(final_content.contains("\\\\"));
    }

    #[test]
    fn test_format_conversation_history_maintains_alternation() {
        // Complex conversation: user → arya → user → arya → user
        // Should maintain strict user/assistant alternation
        let msgs = vec![
            make_conv_msg("user", "msg1"),
            make_conv_msg("arya", "resp1"),
            make_conv_msg("user", "msg2"),
            make_conv_msg("arya", "resp2"),
            make_conv_msg("user", "msg3"),
        ];
        let (history, final_content) = format_conversation_history(&msgs, "arya");

        // Should be: user, assistant, user, assistant (4 messages)
        // Final: user (msg3)
        assert_eq!(history.len(), 4);
        assert_eq!(history[0].role, "user");
        assert_eq!(history[1].role, "assistant");
        assert_eq!(history[2].role, "user");
        assert_eq!(history[3].role, "assistant");

        assert!(final_content.contains("msg3"));
    }
}
