//! Daemon mode for running agents headlessly.
//!
//! This module provides the infrastructure for running agents as background daemons,
//! with Unix socket API for communication and timer trigger support.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use chrono::Local;
use tokio::net::UnixListener;
use tokio::sync::Mutex;

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
    // Look for ```json ... ``` blocks containing a tool call
    let re = regex::Regex::new(r"```json\s*\n?\s*(\{[^`]*\})\s*\n?\s*```").unwrap();
    
    if let Some(cap) = re.captures(output) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&cap[1]) {
            // Check if it's a tool call (has "tool" field)
            if let Some(tool_name) = json.get("tool").and_then(|t| t.as_str()) {
                let params = json.get("params").cloned().unwrap_or(serde_json::json!({}));
                let cleaned = re.replace(output, "").trim().to_string();
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
use crate::discovery;
use crate::llm::{LLM, OpenAIClient, AnthropicClient, OllamaClient};
use crate::memory::{Memory, SqliteMemory, InMemoryStore, SemanticMemoryStore, SaveResult, extract_remember_tags, build_memory_injection};
use crate::observe::ConsoleObserver;
use crate::runtime::Runtime;
use crate::socket_api::{SocketApi, Request, Response};
use crate::tool::Tool;
use crate::tools::{AddTool, EchoTool, ReadFileTool, WriteFileTool, HttpTool, ShellTool};
use crate::tools::send_message::DaemonSendMessageTool;
use crate::tools::list_agents::DaemonListAgentsTool;

/// Execute a tool call and return the result as a string
async fn execute_tool_call(tool_call: &ToolCall) -> Result<String, String> {
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
        _ => Err(format!("Unknown tool: {}", tool_call.tool))
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
    /// Persona (system prompt)
    pub persona: Option<String>,
    /// Always content (injected before user messages for recency bias)
    pub always: Option<String>,
    /// Semantic memory configuration
    pub semantic_memory: SemanticMemorySection,
}

/// Timer configuration for periodic triggers.
#[derive(Debug, Clone)]
pub struct TimerConfig {
    /// How often the timer fires
    pub interval: Duration,
    /// Message to send when timer fires
    pub message: String,
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

        // Load persona
        let persona = agent_dir.load_persona()?;

        // Load always content
        let always = agent_dir.load_always()?;

        Ok(Self {
            name,
            agent_dir: dir_path,
            socket_path,
            pid_path,
            timer,
            persona,
            always,
            semantic_memory: agent_dir.config.semantic_memory.clone(),
        })
    }
}

/// Parse a duration string like "30s", "5m", "1h" into a Duration.
fn parse_duration(s: &str) -> Option<Duration> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    let num_end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    if num_end == 0 {
        return None;
    }

    let num: u64 = s[..num_end].parse().ok()?;
    let unit = &s[num_end..];

    match unit {
        "s" | "sec" | "secs" | "second" | "seconds" => Some(Duration::from_secs(num)),
        "m" | "min" | "mins" | "minute" | "minutes" => Some(Duration::from_secs(num * 60)),
        "h" | "hr" | "hrs" | "hour" | "hours" => Some(Duration::from_secs(num * 3600)),
        "" => Some(Duration::from_secs(num)),
        _ => None,
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

    // Create the agent
    let agent = create_agent_from_dir(&agent_dir).await?;
    let agent = Arc::new(Mutex::new(agent));

    // Create semantic memory store if enabled
    let semantic_memory_store: Option<Arc<Mutex<SemanticMemoryStore>>> = if config.semantic_memory.enabled {
        let mem_path = config.agent_dir.join(&config.semantic_memory.path);
        let store = SemanticMemoryStore::open(&mem_path, &config.name)?;
        logger.log(&format!("  Semantic memory: {}", mem_path.display()));
        Some(Arc::new(Mutex::new(store)))
    } else {
        None
    };

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
        let timer_config = timer_config.clone();
        let shutdown_clone = shutdown.clone();
        let semantic_memory = semantic_memory_store.clone();
        let timer_logger = logger.clone();

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
                            let store = mem_store.lock().await;
                            match store.recall(&timer_config.message, 5) {
                                Ok(memories) => {
                                    if !memories.is_empty() {
                                        timer_logger.memory(&format!("Recall: {} memories for timer", memories.len()));
                                        for m in &memories {
                                            timer_logger.memory(&format!("  #{}: \"{}\"", m.id, m.content));
                                        }
                                    }
                                    build_memory_injection(&memories)
                                }
                                Err(e) => {
                                    timer_logger.log(&format!("[timer] Memory recall error: {}", e));
                                    String::new()
                                }
                            }
                        } else {
                            String::new()
                        };

                        // Combine memory injection with always prompt
                        let effective_always = match (&always, memory_injection.is_empty()) {
                            (Some(a), false) => Some(format!("{}\n{}", memory_injection, a)),
                            (Some(a), true) => Some(a.clone()),
                            (None, false) => Some(memory_injection),
                            (None, true) => None,
                        };

                        let mut agent_guard = agent_clone.lock().await;

                        let options = ThinkOptions {
                            system_prompt: persona.clone(),
                            always_prompt: effective_always,
                            ..Default::default()
                        };

                        match agent_guard.think_with_options(&timer_config.message, options).await {
                            Ok(result) => {
                                // Extract and store [REMEMBER: ...] tags
                                let (cleaned_response, memories_to_save) = extract_remember_tags(&result.response);

                                if !memories_to_save.is_empty() {
                                    if let Some(ref mem_store) = semantic_memory {
                                        let store = mem_store.lock().await;
                                        for memory in &memories_to_save {
                                            match store.save(memory, 0.9, "explicit") {
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

    // Main loop: accept connections
    loop {
        tokio::select! {
            result = listener.accept() => {
                match result {
                    Ok((stream, _)) => {
                        let agent_clone = agent.clone();
                        let persona = config.persona.clone();
                        let always = config.always.clone();
                        let shutdown_clone = shutdown.clone();
                        let semantic_memory = semantic_memory_store.clone();
                        let conn_logger = logger.clone();

                        tokio::spawn(async move {
                            let api = SocketApi::new(stream);
                            if let Err(e) = handle_connection(
                                api,
                                agent_clone,
                                persona,
                                always,
                                semantic_memory,
                                shutdown_clone,
                                conn_logger,
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

    // Clean up socket file
    if config.socket_path.exists() {
        let _ = std::fs::remove_file(&config.socket_path);
    }

    logger.log("Daemon stopped.");
    Ok(())
}

/// Create an agent from an AgentDir configuration.
async fn create_agent_from_dir(agent_dir: &AgentDir) -> Result<Agent, Box<dyn std::error::Error>> {
    let agent_name = agent_dir.config.agent.name.clone();

    // Resolve LLM config (loads model file if specified, applies overrides)
    let llm_config = agent_dir.resolve_llm_config()?;

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

    // Register tools only if enabled in config (default: true)
    if llm_config.tools {
        agent.register_tool(Arc::new(AddTool));
        agent.register_tool(Arc::new(EchoTool));
        agent.register_tool(Arc::new(ReadFileTool));
        agent.register_tool(Arc::new(WriteFileTool));
        agent.register_tool(Arc::new(HttpTool::new()));
        agent.register_tool(Arc::new(ShellTool::new()));

        // Register daemon-aware messaging tools (use socket communication instead of in-memory router)
        agent.register_tool(Arc::new(DaemonSendMessageTool::new(agent_name.clone())));
        agent.register_tool(Arc::new(DaemonListAgentsTool::new(agent_name.clone())));
    }

    // Apply LLM and memory
    agent = agent.with_llm(llm);
    agent = agent.with_memory(memory);

    // Add observer (verbose for daemon since there's no REPL)
    let observer = Arc::new(ConsoleObserver::new(true));
    agent = agent.with_observer(observer);

    Ok(agent)
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

/// Handle a single connection from a client.
async fn handle_connection(
    mut api: SocketApi,
    agent: Arc<Mutex<Agent>>,
    persona: Option<String>,
    always: Option<String>,
    semantic_memory: Option<Arc<Mutex<SemanticMemoryStore>>>,
    shutdown: Arc<tokio::sync::Notify>,
    logger: Arc<AgentLogger>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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

                // Inject relevant memories if enabled
                let memory_injection = if let Some(ref mem_store) = semantic_memory {
                    let store = mem_store.lock().await;
                    match store.recall(content, 5) {
                        Ok(memories) => {
                            if !memories.is_empty() {
                                logger.memory(&format!("Recall: {} memories for query", memories.len()));
                                for m in &memories {
                                    logger.memory(&format!("  #{}: \"{}\"", m.id, m.content));
                                }
                            }
                            build_memory_injection(&memories)
                        }
                        Err(e) => {
                            logger.log(&format!("[socket] Memory recall error: {}", e));
                            String::new()
                        }
                    }
                } else {
                    String::new()
                };

                // Combine memory injection with always prompt
                let effective_always = match (&always, memory_injection.is_empty()) {
                    (Some(a), false) => Some(format!("{}\n{}", memory_injection, a)),
                    (Some(a), true) => Some(a.clone()),
                    (None, false) => Some(memory_injection),
                    (None, true) => None,
                };

                let mut agent_guard = agent.lock().await;
                // Tool execution loop - keep going while agent makes tool calls
                let mut current_message = content.clone();
                let mut final_response = String::new();
                let max_tool_calls = 10; // Prevent infinite loops
                let mut tool_call_count = 0;
                
                loop {
                    let options = ThinkOptions {
                        system_prompt: persona.clone(),
                        always_prompt: effective_always.clone(),
                        ..Default::default()
                    };
                    
                    match agent_guard.think_with_options(&current_message, options).await {
                        Ok(result) => {
                            // Extract and store [REMEMBER: ...] tags
                            let (after_remember, memories_to_save) = extract_remember_tags(&result.response);

                            if !memories_to_save.is_empty() {
                                if let Some(ref mem_store) = semantic_memory {
                                    let store = mem_store.lock().await;
                                    for memory in &memories_to_save {
                                        match store.save(memory, 0.9, "explicit") {
                                            Ok(SaveResult::New(id)) => logger.memory(&format!("Save #{}: \"{}\"", id, memory)),
                                            Ok(SaveResult::Reinforced(id, old, new)) => logger.memory(&format!("Reinforce #{}: \"{}\" ({:.2} → {:.2})", id, memory, old, new)),
                                            Err(e) => logger.log(&format!("[socket] Failed to save memory: {}", e)),
                                        }
                                    }
                                }
                            }

                            // Check for tool calls
                            let (cleaned_response, tool_call) = extract_tool_call(&after_remember);
                            
                            if let Some(tc) = tool_call {
                                tool_call_count += 1;
                                if tool_call_count > max_tool_calls {
                                    logger.tool("Max tool calls reached, stopping");
                                    final_response = format!("{}\n\n[Tool limit reached]", cleaned_response);
                                    break;
                                }
                                
                                logger.tool(&format!("Executing: {} with params {}", tc.tool, tc.params));
                                
                                match execute_tool_call(&tc).await {
                                    Ok(tool_result) => {
                                        logger.tool(&format!("Result: {} bytes", tool_result.len()));
                                        // Send result back to agent
                                        current_message = format!("[Tool Result for {}]\n{}", tc.tool, tool_result);
                                        // Accumulate non-tool text
                                        if !cleaned_response.is_empty() {
                                            if !final_response.is_empty() {
                                                final_response.push_str("\n");
                                            }
                                            final_response.push_str(&cleaned_response);
                                        }
                                        // Continue loop
                                    }
                                    Err(e) => {
                                        logger.tool(&format!("Error: {}", e));
                                        current_message = format!("[Tool Error for {}]\n{}", tc.tool, e);
                                        // Continue loop so agent can handle error
                                    }
                                }
                            } else {
                                // No tool call, we're done
                                if !final_response.is_empty() {
                                    final_response.push_str("\n");
                                }
                                final_response.push_str(&cleaned_response);
                                break;
                            }
                        }
                        Err(e) => {
                            final_response = format!("Error: {}", e);
                            break;
                        }
                    }
                }
                
                Response::Message { content: final_response }
            }

            Request::IncomingMessage { ref from, ref content } => {
                logger.log(&format!("[socket] Incoming message from {}: {}", from, content));

                // Format the message with [sender] prefix for the agent
                let formatted_message = format!("[{}] {}", from, content);

                // Inject relevant memories if enabled
                let memory_injection = if let Some(ref mem_store) = semantic_memory {
                    let store = mem_store.lock().await;
                    match store.recall(content, 5) {
                        Ok(memories) => {
                            if !memories.is_empty() {
                                logger.memory(&format!("Recall: {} memories for incoming", memories.len()));
                                for m in &memories {
                                    logger.memory(&format!("  #{}: \"{}\"", m.id, m.content));
                                }
                            }
                            build_memory_injection(&memories)
                        }
                        Err(e) => {
                            logger.log(&format!("[socket] Memory recall error: {}", e));
                            String::new()
                        }
                    }
                } else {
                    String::new()
                };

                // Combine memory injection with always prompt
                let effective_always = match (&always, memory_injection.is_empty()) {
                    (Some(a), false) => Some(format!("{}\n{}", memory_injection, a)),
                    (Some(a), true) => Some(a.clone()),
                    (None, false) => Some(memory_injection),
                    (None, true) => None,
                };

                let options = ThinkOptions {
                    system_prompt: persona.clone(),
                    always_prompt: effective_always,
                    ..Default::default()
                };

                let mut agent_guard = agent.lock().await;
                match agent_guard.think_with_options(&formatted_message, options).await {
                    Ok(result) => {
                        // Extract and store [REMEMBER: ...] tags
                        let (cleaned_response, memories_to_save) = extract_remember_tags(&result.response);

                        if !memories_to_save.is_empty() {
                            if let Some(ref mem_store) = semantic_memory {
                                let store = mem_store.lock().await;
                                for memory in &memories_to_save {
                                    match store.save(memory, 0.9, "explicit") {
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

        assert_eq!(daemon_config.persona, Some("Test persona".to_string()));
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
        assert!(daemon_config.persona.is_none());
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

        assert_eq!(daemon_config.persona, Some("Test persona".to_string()));
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
}
