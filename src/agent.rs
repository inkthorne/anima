use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use regex::Regex;

use crate::error::ToolError;
use crate::llm::{ChatMessage, LLM, LLMError, ToolSpec};
use crate::memory::{Memory, MemoryError};
use crate::message::Message;
use crate::messaging::{AgentMessage, MessageRouter, MessagingError};
use crate::observe::{Event, Observer};
use crate::retry::{RetryPolicy, with_retry};
use crate::supervision::{ChildConfig, ChildHandle, ChildStatus};
use crate::tool::Tool;
use serde_json::Value;
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

/// Strip thinking tags from LLM response content for conversation history storage.
/// Removes `<think>...</think>` and `<thinking>...</thinking>` blocks (case-insensitive).
/// Also removes stray opening/closing tags that may appear without their pair.
pub fn strip_thinking(content: &str) -> String {
    // First, remove complete think blocks
    let re = Regex::new(r"(?si)<think(?:ing)?>.*?</think(?:ing)?>").unwrap();
    let result = re.replace_all(content, "");

    // Then remove any stray opening or closing tags
    let stray_tags = Regex::new(r"(?i)</?\s*think(?:ing)?\s*/?>").unwrap();
    stray_tags.replace_all(&result, "").trim().to_string()
}

/// Generate a brief summary of a tool result for logging.
/// Extracts key metadata based on tool type.
fn tool_result_summary(tool_name: &str, params: &Value, result: Option<&Value>) -> Option<String> {
    match tool_name {
        "write_file" => {
            let path = params.get("path").and_then(|v| v.as_str()).unwrap_or("?");
            let bytes = params
                .get("content")
                .and_then(|v| v.as_str())
                .map(|s| s.len())
                .unwrap_or(0);
            Some(format!("path={} bytes={}", path, bytes))
        }
        "read_file" => {
            let path = params.get("path").and_then(|v| v.as_str()).unwrap_or("?");
            let bytes = result
                .and_then(|r| r.get("content"))
                .and_then(|v| v.as_str())
                .map(|s| s.len());
            match bytes {
                Some(b) => Some(format!("path={} bytes={}", path, b)),
                None => Some(format!("path={}", path)),
            }
        }
        "shell" | "safe_shell" => {
            let cmd = params
                .get("command")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let cmd_short = if cmd.len() > 40 { &cmd[..40] } else { cmd };
            let exit_code = result
                .and_then(|r| r.get("exit_code"))
                .and_then(|v| v.as_i64());
            match exit_code {
                Some(code) => Some(format!("cmd='{}' exit={}", cmd_short, code)),
                None => Some(format!("cmd='{}'", cmd_short)),
            }
        }
        "http" => {
            let url = params.get("url").and_then(|v| v.as_str()).unwrap_or("?");
            let method = params
                .get("method")
                .and_then(|v| v.as_str())
                .unwrap_or("GET");
            Some(format!("{} {}", method.to_uppercase(), url))
        }
        "remember" => {
            let content = params.get("content").and_then(|v| v.as_str()).unwrap_or("");
            let preview = if content.len() > 50 {
                &content[..50]
            } else {
                content
            };
            Some(format!("'{}'", preview))
        }
        _ => None, // No special summary for other tools
    }
}

/// Options for the think() agentic loop
pub struct ThinkOptions {
    /// Maximum iterations before giving up (default: 10)
    pub max_iterations: usize,
    /// Optional system prompt to set agent behavior
    pub system_prompt: Option<String>,
    /// Optional reflection configuration for self-evaluation
    pub reflection: Option<ReflectionConfig>,
    /// Optional auto-memory configuration for injecting memories into context
    pub auto_memory: Option<AutoMemoryConfig>,
    /// Enable streaming output (default: false)
    pub stream: bool,
    /// Optional retry policy for LLM calls (default: RetryPolicy::default())
    pub retry_policy: Option<RetryPolicy>,
    /// Optional conversation history to inject before the user task
    pub conversation_history: Option<Vec<ChatMessage>>,
    /// Optional external tools to pass to the LLM. When set, these override the agent's
    /// registered tools for this call. Used for hybrid tool calling where tools are
    /// dynamically selected via keyword recall.
    pub external_tools: Option<Vec<ToolSpec>>,
}

impl Default for ThinkOptions {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            system_prompt: None,
            reflection: None,
            auto_memory: None,
            stream: false,
            retry_policy: Some(RetryPolicy::default()),
            conversation_history: None,
            external_tools: None,
        }
    }
}

/// Configuration for self-reflection after generating a response
#[derive(Debug, Clone)]
pub struct ReflectionConfig {
    /// Prompt to use when asking the LLM to evaluate its response
    pub prompt: String,
    /// Maximum number of revision cycles allowed
    pub max_revisions: usize,
}

impl Default for ReflectionConfig {
    fn default() -> Self {
        Self {
            prompt: String::from(
                "Evaluate your response. Is it complete and correct? If not, explain what needs to change.",
            ),
            max_revisions: 1,
        }
    }
}

/// Configuration for automatic memory injection during thinking
#[derive(Debug, Clone)]
pub struct AutoMemoryConfig {
    /// Maximum number of memory entries to include
    pub max_entries: usize,
    /// Include most recent memories
    pub include_recent: bool,
    /// Only include memories with keys matching these prefixes (empty = all)
    pub key_prefixes: Vec<String>,
}

impl Default for AutoMemoryConfig {
    fn default() -> Self {
        Self {
            max_entries: 10,
            include_recent: true,
            key_prefixes: vec![],
        }
    }
}

/// Result of a reflection evaluation
#[derive(Debug, Clone)]
pub struct ReflectionResult {
    /// Whether the response was accepted as-is
    pub accepted: bool,
    /// Feedback for revision if not accepted
    pub feedback: Option<String>,
}

/// A single tool execution: the call and its result.
#[derive(Debug, Clone)]
pub struct ToolExecution {
    /// The tool call (id, name, arguments)
    pub call: crate::llm::ToolCall,
    /// The result of executing the tool
    pub result: String,
    /// Assistant content (narration) that accompanied the tool call, if any.
    /// When the LLM returns both content AND tool_calls, this captures that content.
    pub content: Option<String>,
}

/// Result from a think operation, including tool usage information.
#[derive(Debug, Clone)]
pub struct ThinkResult {
    /// The final text response from the agent
    pub response: String,
    /// Whether any tools were called during this think operation
    pub tools_used: bool,
    /// Names of tools that were called (for logging/debugging)
    pub tool_names: Vec<String>,
    /// The last set of tool_calls made before the final response (for conversation persistence)
    pub last_tool_calls: Option<Vec<crate::llm::ToolCall>>,
    /// Complete trace of all tool executions during this think operation.
    /// Each entry contains the assistant's tool call and the tool's result.
    /// Used for persisting the full conversation to history.
    pub tool_trace: Vec<ToolExecution>,
    /// Total input tokens used across all LLM calls in this think operation
    pub tokens_in: Option<u32>,
    /// Total output tokens used across all LLM calls in this think operation
    pub tokens_out: Option<u32>,
}

/// Maximum number of messages to retain in conversation history.
/// When exceeded, oldest messages are removed.
const MAX_HISTORY_LEN: usize = 50;

pub struct Agent {
    pub id: String,
    tools: HashMap<String, Arc<dyn Tool>>,
    #[allow(dead_code)]
    inbox: mpsc::Receiver<Message>,
    memory: Option<Box<dyn Memory>>,
    llm: Option<Arc<dyn LLM>>,
    pub children: HashMap<String, ChildHandle>,
    observer: Option<Arc<dyn Observer>>,
    /// Receiver for agent-to-agent messages
    message_rx: Option<mpsc::Receiver<AgentMessage>>,
    /// Router for sending messages to other agents
    router: Option<Arc<Mutex<MessageRouter>>>,
    /// Conversation history for multi-turn interactions
    history: Vec<ChatMessage>,
    /// Agent directory path for context dumping
    agent_dir: Option<PathBuf>,
    /// Current conversation name for debug file naming (turns/{name}.json)
    current_conversation: Option<String>,
}

impl Agent {
    pub fn new(id: String, inbox: mpsc::Receiver<Message>) -> Self {
        Agent {
            id,
            tools: HashMap::new(),
            inbox,
            memory: None,
            llm: None,
            children: HashMap::new(),
            observer: None,
            message_rx: None,
            router: None,
            history: Vec::new(),
            agent_dir: None,
            current_conversation: None,
        }
    }

    /// Set the agent directory for context dumping
    pub fn with_agent_dir(mut self, dir: PathBuf) -> Self {
        self.agent_dir = Some(dir);
        self
    }

    /// Set the current conversation name for debug file naming.
    /// Used to write debug files to turns/{name}.json instead of last_turn.json.
    pub fn set_current_conversation(&mut self, name: Option<String>) {
        self.current_conversation = name;
    }

    /// Attach an observer for monitoring agent activity.
    pub fn with_observer(mut self, observer: Arc<dyn Observer>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Emit an event to the observer if one is attached.
    async fn emit(&self, event: Event) {
        if let Some(obs) = &self.observer {
            obs.observe(event).await;
        }
    }

    pub fn register_tool(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub async fn call_tool(&self, name: &str, input: &str) -> Result<String, ToolError> {
        let start = Instant::now();

        if let Some(tool) = self.tools.get(name) {
            let input_value: Value = serde_json::from_str(input).map_err(|e| {
                // Note: We can't emit here since it's sync context within map_err
                ToolError::InvalidInput(e.to_string())
            })?;

            let result = (*tool).execute(input_value.clone()).await;
            let duration_ms = start.elapsed().as_millis() as u64;

            match &result {
                Ok(val) => {
                    let summary = tool_result_summary(name, &input_value, Some(val));
                    self.emit(Event::ToolCall {
                        tool_name: name.to_string(),
                        duration_ms,
                        success: true,
                        error: None,
                        params: Some(input_value),
                        result_summary: summary,
                    })
                    .await;
                    Ok(val.to_string())
                }
                Err(e) => {
                    self.emit(Event::ToolCall {
                        tool_name: name.to_string(),
                        duration_ms,
                        success: false,
                        error: Some(e.to_string()),
                        params: Some(input_value),
                        result_summary: None,
                    })
                    .await;
                    result.map(|v| v.to_string())
                }
            }
        } else {
            let err = ToolError::ExecutionFailed(format!("Tool '{}' not found", name));
            self.emit(Event::ToolCall {
                tool_name: name.to_string(),
                duration_ms: 0,
                success: false,
                error: Some(err.to_string()),
                params: None,
                result_summary: None,
            })
            .await;
            Err(err)
        }
    }

    pub fn with_memory(mut self, memory: Box<dyn Memory>) -> Self {
        self.memory = Some(memory);
        self
    }

    pub fn with_llm(mut self, llm: Arc<dyn LLM>) -> Self {
        self.llm = Some(llm);
        self
    }

    /// Attach a message router and register this agent for messaging
    pub fn with_router(mut self, router: Arc<Mutex<MessageRouter>>) -> Self {
        // Register with router and get the message receiver
        let rx = {
            let mut router_guard = router.blocking_lock();
            router_guard.register(&self.id)
        };
        self.message_rx = Some(rx);
        self.router = Some(router);
        self
    }

    /// Attach a message router and receiver (for when receiver is created externally)
    pub fn with_router_and_rx(
        mut self,
        router: Arc<Mutex<MessageRouter>>,
        rx: mpsc::Receiver<AgentMessage>,
    ) -> Self {
        self.message_rx = Some(rx);
        self.router = Some(router);
        self
    }

    /// Get a reference to the router if attached
    pub fn router(&self) -> Option<&Arc<Mutex<MessageRouter>>> {
        self.router.as_ref()
    }

    /// Drain all pending messages from the inbox (non-blocking)
    fn drain_inbox(&mut self) -> Vec<AgentMessage> {
        let mut messages = Vec::new();
        if let Some(ref mut rx) = self.message_rx {
            loop {
                match rx.try_recv() {
                    Ok(msg) => messages.push(msg),
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => break,
                }
            }
        }
        messages
    }

    /// Send a message to another agent (fire and forget)
    pub async fn send_message(&self, to: &str, content: &str) -> Result<(), MessagingError> {
        let router = self.router.as_ref().ok_or(MessagingError::NotRegistered)?;
        let msg = AgentMessage::new(&self.id, to, content);
        let router_guard = router.lock().await;
        router_guard.send(msg).await
    }

    /// Send a message and wait for a reply (request-response pattern)
    pub async fn ask(&self, to: &str, content: &str) -> Result<String, MessagingError> {
        let router = self.router.as_ref().ok_or(MessagingError::NotRegistered)?;

        // Create a oneshot channel for the reply
        let (tx, rx) = oneshot::channel();

        // Generate a unique reply ID and register the pending reply
        let reply_id = {
            let mut router_guard = router.lock().await;
            let reply_id = router_guard.generate_reply_id();
            router_guard.register_reply(reply_id.clone(), tx);
            reply_id
        };

        // Send the message with reply_to set
        let msg = AgentMessage {
            from: self.id.clone(),
            to: to.to_string(),
            content: content.to_string(),
            reply_to: Some(reply_id),
        };

        {
            let router_guard = router.lock().await;
            router_guard.send(msg).await?;
        }

        // Wait for the reply
        rx.await
            .map(|m| m.content)
            .map_err(|_| MessagingError::ChannelClosed)
    }

    /// Receive the next message (non-blocking)
    pub async fn receive_message(&mut self) -> Option<AgentMessage> {
        if let Some(rx) = &mut self.message_rx {
            rx.try_recv().ok()
        } else {
            None
        }
    }

    /// Receive a message with timeout
    pub async fn receive_message_timeout(
        &mut self,
        timeout: std::time::Duration,
    ) -> Option<AgentMessage> {
        if let Some(rx) = &mut self.message_rx {
            tokio::time::timeout(timeout, rx.recv())
                .await
                .ok()
                .flatten()
        } else {
            None
        }
    }

    /// Reply to a message (used for request-response pattern)
    pub async fn reply_to(
        &self,
        original: &AgentMessage,
        content: &str,
    ) -> Result<(), MessagingError> {
        let router = self.router.as_ref().ok_or(MessagingError::NotRegistered)?;

        if let Some(reply_id) = &original.reply_to {
            // This is a request that expects a reply - complete it directly
            let reply_msg = AgentMessage::new(&self.id, &original.from, content);
            let mut router_guard = router.lock().await;
            if router_guard.complete_reply(reply_id, reply_msg) {
                Ok(())
            } else {
                // Reply channel was already used or expired, send as regular message
                drop(router_guard);
                self.send_message(&original.from, content).await
            }
        } else {
            // No reply_to, just send as regular message
            self.send_message(&original.from, content).await
        }
    }

    /// List all agents registered with the router
    pub async fn list_peers(&self) -> Result<Vec<String>, MessagingError> {
        let router = self.router.as_ref().ok_or(MessagingError::NotRegistered)?;
        let router_guard = router.lock().await;
        Ok(router_guard.list_agents())
    }

    /// Get a reference to the current conversation history
    pub fn history(&self) -> &[ChatMessage] {
        &self.history
    }

    /// Clear conversation history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Get the number of messages in history
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Trim history to stay within MAX_HISTORY_LEN.
    /// Removes oldest messages when limit is exceeded.
    fn trim_history(&mut self) {
        while self.history.len() > MAX_HISTORY_LEN {
            self.history.remove(0);
        }
    }

    pub async fn remember(
        &mut self,
        key: &str,
        value: serde_json::Value,
    ) -> Result<(), MemoryError> {
        match &mut self.memory {
            Some(mem) => mem.set(key, value).await,
            None => Err(MemoryError::StorageError("No memory attached".to_string())),
        }
    }

    pub async fn recall(&self, key: &str) -> Option<serde_json::Value> {
        match &self.memory {
            Some(mem) => mem.get(key).await.map(|e| e.value),
            None => None,
        }
    }

    pub async fn forget(&mut self, key: &str) -> bool {
        match &mut self.memory {
            Some(mem) => mem.delete(key).await,
            None => false,
        }
    }

    pub fn list_tools_for_llm(&self) -> Vec<ToolSpec> {
        self.tools
            .values()
            .map(|t| ToolSpec {
                name: t.name().to_string(),
                description: t.description().to_string(),
                parameters: t.schema(),
            })
            .collect()
    }

    /// Build memory context string for auto-injection into thinking
    async fn build_memory_context(&self, config: &Option<AutoMemoryConfig>) -> Option<String> {
        let config = config.as_ref()?;
        let memory = self.memory.as_ref()?;

        // Get keys (filtered by prefixes if specified)
        let all_keys: Vec<String> = if config.key_prefixes.is_empty() {
            memory.list_keys(None).await
        } else {
            let mut keys = Vec::new();
            for prefix in &config.key_prefixes {
                keys.extend(memory.list_keys(Some(prefix)).await);
            }
            keys
        };

        if all_keys.is_empty() {
            return None;
        }

        // Get entries with their timestamps for sorting
        let mut entries: Vec<(String, crate::memory::MemoryEntry)> = Vec::new();
        for key in &all_keys {
            if let Some(entry) = memory.get(key).await {
                entries.push((key.clone(), entry));
            }
        }

        // Sort by updated_at (recent first if include_recent, oldest first otherwise)
        if config.include_recent {
            entries.sort_by(|a, b| b.1.updated_at.cmp(&a.1.updated_at));
        } else {
            entries.sort_by(|a, b| a.1.updated_at.cmp(&b.1.updated_at));
        }

        // Limit to max_entries
        entries.truncate(config.max_entries);

        if entries.is_empty() {
            return None;
        }

        // Format as context string
        let mut context = String::from("Your memories:\n");
        for (key, entry) in entries {
            // Format value - stringify JSON nicely
            let value_str = match &entry.value {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            context.push_str(&format!("- {}: {}\n", key, value_str));
        }

        Some(context)
    }

    /// Dump the raw LLM request payload to turns/{conv_name}.json in the agent directory.
    /// This is called before each LLM request to provide the exact JSON for debugging/reproduction.
    fn dump_context(
        &self,
        _system_prompt: &Option<String>,
        _memory_context: &Option<String>,
        tools: &Option<Vec<ToolSpec>>,
        messages: &[ChatMessage],
    ) {
        let agent_dir = match &self.agent_dir {
            Some(dir) => dir,
            None => return, // No agent dir configured, skip dump
        };

        // Determine conversation name for the file
        let conv_name = self.current_conversation.as_deref().unwrap_or("direct");

        // Create turns/ directory if it doesn't exist
        let turns_dir = agent_dir.join("turns");
        if let Err(e) = std::fs::create_dir_all(&turns_dir) {
            eprintln!(
                "[agent:{}] Failed to create turns directory: {}",
                self.id, e
            );
            return;
        }

        // Create .gitignore to make turns/ self-ignoring (debug files shouldn't be committed)
        let gitignore_path = turns_dir.join(".gitignore");
        if !gitignore_path.exists() {
            let _ = std::fs::write(&gitignore_path, "*\n!.gitignore\n");
        }

        let file_path = turns_dir.join(format!("{}.json", conv_name));

        // Get model name from LLM if available
        let model = self
            .llm
            .as_ref()
            .map(|l| l.model_name().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        // Format messages for Ollama API (tool_calls arguments as objects, not strings)
        let formatted_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|msg| {
                let mut formatted = serde_json::json!({
                    "role": msg.role,
                });
                if let Some(ref content) = msg.content {
                    formatted["content"] = serde_json::Value::String(content.clone());
                }
                if let Some(ref tool_call_id) = msg.tool_call_id {
                    formatted["tool_call_id"] = serde_json::Value::String(tool_call_id.clone());
                }
                if let Some(ref tool_calls) = msg.tool_calls
                    && !tool_calls.is_empty()
                {
                    let formatted_tool_calls: Vec<serde_json::Value> = tool_calls
                        .iter()
                        .map(|tc| {
                            serde_json::json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": tc.arguments  // Object, not string (Ollama native format)
                                }
                            })
                        })
                        .collect();
                    formatted["tool_calls"] = serde_json::Value::Array(formatted_tool_calls);
                }
                formatted
            })
            .collect();

        // Build request body matching Ollama /api/chat format
        let mut request_body = serde_json::json!({
            "model": model,
            "messages": formatted_messages,
            "stream": false
        });

        // Add tools if present (Ollama format)
        if let Some(tool_list) = tools
            && !tool_list.is_empty()
        {
            let formatted_tools: Vec<serde_json::Value> = tool_list
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters
                        }
                    })
                })
                .collect();
            request_body["tools"] = serde_json::Value::Array(formatted_tools);
            request_body["tool_choice"] = serde_json::json!("auto");
        }

        // Write pretty-printed JSON for readability while maintaining valid JSON
        let content =
            serde_json::to_string_pretty(&request_body).unwrap_or_else(|_| "{}".to_string());

        // Write to file (best-effort, don't fail the agent on IO errors)
        if let Err(e) = std::fs::write(&file_path, content) {
            eprintln!("[agent:{}] Failed to write context dump: {}", self.id, e);
        }
    }

    pub async fn think_with_options(
        &mut self,
        task: &str,
        options: ThinkOptions,
    ) -> Result<ThinkResult, crate::error::AgentError> {
        let agent_start = Instant::now();

        // Emit agent start event
        self.emit(Event::AgentStart {
            agent_id: self.id.clone(),
            task: task.to_string(),
        })
        .await;

        let result = self.think_with_options_inner(task, options).await;

        // Emit agent complete event
        let duration_ms = agent_start.elapsed().as_millis() as u64;
        self.emit(Event::AgentComplete {
            agent_id: self.id.clone(),
            duration_ms,
            success: result.is_ok(),
        })
        .await;

        if let Err(ref e) = result {
            self.emit(Event::Error {
                context: format!("agent:{}", self.id),
                message: e.to_string(),
            })
            .await;
        }

        result
    }

    /// Inner implementation of think_with_options (without event wrapper).
    async fn think_with_options_inner(
        &mut self,
        task: &str,
        options: ThinkOptions,
    ) -> Result<ThinkResult, crate::error::AgentError> {
        // Drain any pending messages from inbox (must happen before borrowing llm)
        let pending_messages = self.drain_inbox();
        let effective_task = if pending_messages.is_empty() {
            task.to_string()
        } else {
            let inbox_text = pending_messages
                .iter()
                .map(|msg| format!("[from: {}] {}", msg.from, msg.content))
                .collect::<Vec<_>>()
                .join("\n");
            format!(
                "You have messages from other agents:\n{}\n\nTo reply, use the send_message tool with the sender's name.\n\nCurrent task: {}",
                inbox_text, task
            )
        };

        let llm = self
            .llm
            .as_ref()
            .ok_or_else(|| crate::error::AgentError::LlmError("No LLM attached".to_string()))?;

        // Use external_tools if provided (hybrid mode), otherwise use registered tools
        let tools_list = options
            .external_tools
            .clone()
            .unwrap_or_else(|| self.list_tools_for_llm());
        // Send None instead of Some(empty_vec) for models that don't support tools
        let tools: Option<Vec<ToolSpec>> = if tools_list.is_empty() {
            None
        } else {
            Some(tools_list)
        };

        // Build auto-memory context if configured
        let memory_context = self.build_memory_context(&options.auto_memory).await;

        // Combine memory context with system prompt
        let effective_system_prompt = match (&memory_context, &options.system_prompt) {
            (Some(mem), Some(sys)) => Some(format!("{}\n\n{}", mem, sys)),
            (Some(mem), None) => Some(mem.clone()),
            (None, Some(sys)) => Some(sys.clone()),
            (None, None) => None,
        };

        // Build initial messages for LLM call
        let mut messages: Vec<ChatMessage> = Vec::new();

        // Optional system prompt (with memory context prepended)
        if let Some(system) = &effective_system_prompt {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: Some(system.clone()),
                tool_call_id: None,
                tool_calls: None,
            });
        }

        // Inject internal conversation history
        messages.extend(self.history.clone());

        // Also inject any additional context from options (additive, not replacement)
        if let Some(extra_history) = &options.conversation_history {
            messages.extend(extra_history.clone());
        }

        // Create and add user message (unless empty with existing history)
        // Note: Recall content (tools, memories, recall.md) is now injected as an assistant
        // message by the daemon BEFORE the user message, not prepended to user content.
        // This improves KV caching since user messages stay unchanged.
        //
        // When final_user_content is empty (after storing recall and re-fetching history),
        // the user message is already in conversation_history. Skip adding an empty user message.
        if !effective_task.is_empty() {
            let user_message = ChatMessage {
                role: "user".to_string(),
                content: Some(effective_task.clone()),
                tool_call_id: None,
                tool_calls: None,
            };

            // Add user message to internal history
            self.history.push(user_message.clone());

            // Add user message to LLM messages
            messages.push(user_message);
        }

        // Track tool usage across the agentic loop
        let mut tool_names_used: Vec<String> = Vec::new();
        // Track the last tool_calls made before final response (for conversation persistence)
        let mut last_tool_calls: Option<Vec<crate::llm::ToolCall>> = None;
        // Track all tool executions for conversation persistence
        let mut tool_trace: Vec<ToolExecution> = Vec::new();
        // Track total token usage across all LLM calls in this think operation
        let mut total_tokens_in: u32 = 0;
        let mut total_tokens_out: u32 = 0;
        let mut has_token_data = false;

        // Agentic loop
        for _iteration in 0..options.max_iterations {
            // Dump context before each LLM call
            self.dump_context(&effective_system_prompt, &memory_context, &tools, &messages);

            // Call LLM with retry if policy is configured
            let llm_start = Instant::now();
            let response = if let Some(ref policy) = options.retry_policy {
                let llm_ref = llm.clone();
                let msgs = messages.clone();
                let tls = tools.clone();
                let observer = self.observer.clone();
                let result = with_retry(
                    policy,
                    || {
                        let llm = llm_ref.clone();
                        let m = msgs.clone();
                        let t = tls.clone();
                        async move { llm.chat_complete(m, t).await }
                    },
                    |e: &LLMError| e.is_retryable,
                )
                .await;

                // Emit retry events if attempts > 1
                if result.attempts > 1
                    && let Some(obs) = &observer
                {
                    for attempt in 1..result.attempts {
                        let delay_ms = policy.delay_for_attempt(attempt).as_millis() as u64;
                        obs.observe(Event::Retry {
                            operation: "llm_call".to_string(),
                            attempt,
                            delay_ms,
                        })
                        .await;
                    }
                }

                result
                    .result
                    .map_err(|e| crate::error::AgentError::LlmError(e.message))?
            } else {
                llm.chat_complete(messages.clone(), tools.clone())
                    .await
                    .map_err(|e| crate::error::AgentError::LlmError(e.message))?
            };

            // Emit LLM call event
            let llm_duration_ms = llm_start.elapsed().as_millis() as u64;
            self.emit(Event::LlmCall {
                model: llm.model_name().to_string(),
                tokens_in: response.usage.as_ref().map(|u| u.prompt_tokens),
                tokens_out: response.usage.as_ref().map(|u| u.completion_tokens),
                duration_ms: llm_duration_ms,
            })
            .await;

            // Accumulate token usage
            if let Some(ref usage) = response.usage {
                total_tokens_in += usage.prompt_tokens;
                total_tokens_out += usage.completion_tokens;
                has_token_data = true;
            }

            // If no tool calls, we have a final response
            if response.tool_calls.is_empty() {
                let final_response = response
                    .content
                    .unwrap_or_else(|| "No response".to_string());

                // Add final assistant response to internal history (strip thinking tags)
                let stripped_response = strip_thinking(&final_response);
                self.history.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: Some(stripped_response),
                    tool_call_id: None,
                    tool_calls: None,
                });

                // Trim history to stay within limit
                self.trim_history();

                // Apply reflection if configured
                if let Some(ref config) = options.reflection {
                    let reflected_response = self
                        .reflect_and_revise(task, &final_response, config, &options)
                        .await?;
                    return Ok(ThinkResult {
                        response: reflected_response,
                        tools_used: !tool_names_used.is_empty(),
                        tool_names: tool_names_used,
                        last_tool_calls,
                        tool_trace,
                        tokens_in: if has_token_data {
                            Some(total_tokens_in)
                        } else {
                            None
                        },
                        tokens_out: if has_token_data {
                            Some(total_tokens_out)
                        } else {
                            None
                        },
                    });
                }

                return Ok(ThinkResult {
                    response: final_response,
                    tools_used: !tool_names_used.is_empty(),
                    tool_names: tool_names_used,
                    last_tool_calls,
                    tool_trace,
                    tokens_in: if has_token_data {
                        Some(total_tokens_in)
                    } else {
                        None
                    },
                    tokens_out: if has_token_data {
                        Some(total_tokens_out)
                    } else {
                        None
                    },
                });
            }

            // Track the tool_calls for conversation persistence
            last_tool_calls = Some(response.tool_calls.clone());

            // Create assistant message with tool calls (strip thinking tags for storage)
            let assistant_message = ChatMessage {
                role: "assistant".to_string(),
                content: response.content.as_ref().map(|c| strip_thinking(c)),
                tool_call_id: None,
                tool_calls: Some(response.tool_calls.clone()),
            };

            // Add to both internal history and LLM messages
            self.history.push(assistant_message.clone());
            messages.push(assistant_message);

            // Capture content that accompanied the tool calls (for display alongside tool execution)
            let assistant_content = response.content.as_ref().map(|c| strip_thinking(c));

            // Execute each tool and add results
            for (i, tool_call) in response.tool_calls.iter().enumerate() {
                // Track tool name
                tool_names_used.push(tool_call.name.clone());

                let result = self
                    .call_tool(&tool_call.name, &tool_call.arguments.to_string())
                    .await
                    .unwrap_or_else(|e| format!("Error: {}", e));

                // Record tool execution for persistence.
                // Only attach content to the first tool call to avoid duplication.
                tool_trace.push(ToolExecution {
                    call: tool_call.clone(),
                    result: result.clone(),
                    content: if i == 0 { assistant_content.clone() } else { None },
                });

                let tool_message = ChatMessage {
                    role: "tool".to_string(),
                    content: Some(result),
                    tool_call_id: Some(tool_call.id.clone()),
                    tool_calls: None,
                };

                // Add to both internal history and LLM messages
                self.history.push(tool_message.clone());
                messages.push(tool_message);
            }
        }

        // Trim history even on error path
        self.trim_history();

        Err(crate::error::AgentError::MaxIterationsExceeded(
            options.max_iterations,
        ))
    }

    pub async fn think(&mut self, task: &str) -> Result<ThinkResult, crate::error::AgentError> {
        self.think_with_options(task, ThinkOptions::default()).await
    }

    /// Think with streaming output, sending tokens through the channel as they arrive.
    /// Note: Streaming does not support reflection or tool calls mid-stream well,
    /// so this is best used for simple query/response patterns.
    pub async fn think_streaming(
        &mut self,
        task: &str,
        token_tx: mpsc::Sender<String>,
    ) -> Result<ThinkResult, crate::error::AgentError> {
        self.think_streaming_with_options(task, ThinkOptions::default(), token_tx)
            .await
    }

    /// Think with streaming output and custom options.
    pub async fn think_streaming_with_options(
        &mut self,
        task: &str,
        options: ThinkOptions,
        token_tx: mpsc::Sender<String>,
    ) -> Result<ThinkResult, crate::error::AgentError> {
        let agent_start = Instant::now();

        // Emit agent start event
        self.emit(Event::AgentStart {
            agent_id: self.id.clone(),
            task: task.to_string(),
        })
        .await;

        let result = self
            .think_streaming_with_options_inner(task, options, token_tx)
            .await;

        // Emit agent complete event
        let duration_ms = agent_start.elapsed().as_millis() as u64;
        self.emit(Event::AgentComplete {
            agent_id: self.id.clone(),
            duration_ms,
            success: result.is_ok(),
        })
        .await;

        if let Err(ref e) = result {
            self.emit(Event::Error {
                context: format!("agent:{}", self.id),
                message: e.to_string(),
            })
            .await;
        }

        result
    }

    /// Inner implementation of streaming think (without event wrapper).
    async fn think_streaming_with_options_inner(
        &mut self,
        task: &str,
        options: ThinkOptions,
        token_tx: mpsc::Sender<String>,
    ) -> Result<ThinkResult, crate::error::AgentError> {
        // Drain any pending messages from inbox (must happen before borrowing llm)
        let pending_messages = self.drain_inbox();
        let effective_task = if pending_messages.is_empty() {
            task.to_string()
        } else {
            let inbox_text = pending_messages
                .iter()
                .map(|msg| format!("[from: {}] {}", msg.from, msg.content))
                .collect::<Vec<_>>()
                .join("\n");
            format!(
                "You have messages from other agents:\n{}\n\nTo reply, use the send_message tool with the sender's name.\n\nCurrent task: {}",
                inbox_text, task
            )
        };

        let llm = self
            .llm
            .as_ref()
            .ok_or_else(|| crate::error::AgentError::LlmError("No LLM attached".to_string()))?;

        // Use external_tools if provided (hybrid mode), otherwise use registered tools
        let tools_list = options
            .external_tools
            .clone()
            .unwrap_or_else(|| self.list_tools_for_llm());
        // Send None instead of Some(empty_vec) for models that don't support tools
        let tools: Option<Vec<ToolSpec>> = if tools_list.is_empty() {
            None
        } else {
            Some(tools_list)
        };

        // Build auto-memory context if configured
        let memory_context = self.build_memory_context(&options.auto_memory).await;

        // Combine memory context with system prompt
        let effective_system_prompt = match (&memory_context, &options.system_prompt) {
            (Some(mem), Some(sys)) => Some(format!("{}\n\n{}", mem, sys)),
            (Some(mem), None) => Some(mem.clone()),
            (None, Some(sys)) => Some(sys.clone()),
            (None, None) => None,
        };

        // Build initial messages for LLM call
        let mut messages: Vec<ChatMessage> = Vec::new();

        if let Some(system) = &effective_system_prompt {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: Some(system.clone()),
                tool_call_id: None,
                tool_calls: None,
            });
        }

        // Inject internal conversation history
        messages.extend(self.history.clone());

        // Also inject any additional context from options (additive, not replacement)
        if let Some(extra_history) = &options.conversation_history {
            messages.extend(extra_history.clone());
        }

        // Create and add user message
        // Note: Recall content (tools, memories, recall.md) is now injected as an assistant
        // message by the daemon BEFORE the user message, not prepended to user content.
        // This improves KV caching since user messages stay unchanged.
        let user_message = ChatMessage {
            role: "user".to_string(),
            content: Some(effective_task.clone()),
            tool_call_id: None,
            tool_calls: None,
        };

        // Add user message to internal history
        self.history.push(user_message.clone());

        // Add user message to LLM messages
        messages.push(user_message);

        // Track tool usage across the agentic loop (same as non-streaming)
        let mut tool_names_used: Vec<String> = Vec::new();
        let mut last_tool_calls: Option<Vec<crate::llm::ToolCall>> = None;
        let mut tool_trace: Vec<ToolExecution> = Vec::new();
        let mut total_tokens_in: u32 = 0;
        let mut total_tokens_out: u32 = 0;
        let mut has_token_data = false;

        // Agentic loop with streaming
        for _iteration in 0..options.max_iterations {
            // Dump context before each LLM call
            self.dump_context(&effective_system_prompt, &memory_context, &tools, &messages);

            // Call LLM with retry if policy is configured
            // Note: streaming with retry will restart the entire stream on failure
            let llm_start = Instant::now();
            let response = if let Some(ref policy) = options.retry_policy {
                let llm_ref = llm.clone();
                let msgs = messages.clone();
                let tls = tools.clone();
                let tx = token_tx.clone();
                let observer = self.observer.clone();
                let result = with_retry(
                    policy,
                    || {
                        let llm = llm_ref.clone();
                        let m = msgs.clone();
                        let t = tls.clone();
                        let tx_clone = tx.clone();
                        async move { llm.chat_complete_stream(m, t, tx_clone).await }
                    },
                    |e: &LLMError| e.is_retryable,
                )
                .await;

                // Emit retry events if attempts > 1
                if result.attempts > 1
                    && let Some(obs) = &observer
                {
                    for attempt in 1..result.attempts {
                        let delay_ms = policy.delay_for_attempt(attempt).as_millis() as u64;
                        obs.observe(Event::Retry {
                            operation: "llm_call_stream".to_string(),
                            attempt,
                            delay_ms,
                        })
                        .await;
                    }
                }

                result
                    .result
                    .map_err(|e| crate::error::AgentError::LlmError(e.message))?
            } else {
                llm.chat_complete_stream(messages.clone(), tools.clone(), token_tx.clone())
                    .await
                    .map_err(|e| crate::error::AgentError::LlmError(e.message))?
            };

            // Emit LLM call event
            let llm_duration_ms = llm_start.elapsed().as_millis() as u64;
            self.emit(Event::LlmCall {
                model: llm.model_name().to_string(),
                tokens_in: response.usage.as_ref().map(|u| u.prompt_tokens),
                tokens_out: response.usage.as_ref().map(|u| u.completion_tokens),
                duration_ms: llm_duration_ms,
            })
            .await;

            // Track token usage
            if let Some(ref usage) = response.usage {
                total_tokens_in += usage.prompt_tokens;
                total_tokens_out += usage.completion_tokens;
                has_token_data = true;
            }

            // If no tool calls, we have a final response
            if response.tool_calls.is_empty() {
                let final_response = response
                    .content
                    .unwrap_or_else(|| "No response".to_string());

                // Add final assistant response to internal history (strip thinking tags)
                let stripped_response = strip_thinking(&final_response);
                self.history.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: Some(stripped_response),
                    tool_call_id: None,
                    tool_calls: None,
                });

                // Trim history to stay within limit
                self.trim_history();

                return Ok(ThinkResult {
                    response: final_response,
                    tools_used: !tool_names_used.is_empty(),
                    tool_names: tool_names_used,
                    last_tool_calls,
                    tool_trace,
                    tokens_in: if has_token_data {
                        Some(total_tokens_in)
                    } else {
                        None
                    },
                    tokens_out: if has_token_data {
                        Some(total_tokens_out)
                    } else {
                        None
                    },
                });
            }

            // Track the tool_calls for conversation persistence
            last_tool_calls = Some(response.tool_calls.clone());

            // Create assistant message with tool calls (strip thinking tags for storage)
            let assistant_message = ChatMessage {
                role: "assistant".to_string(),
                content: response.content.as_ref().map(|c| strip_thinking(c)),
                tool_call_id: None,
                tool_calls: Some(response.tool_calls.clone()),
            };

            // Add to both internal history and LLM messages
            self.history.push(assistant_message.clone());
            messages.push(assistant_message);

            // Capture content that accompanied the tool calls (for display alongside tool execution)
            let assistant_content = response.content.as_ref().map(|c| strip_thinking(c));

            // Execute each tool and add results
            for (i, tool_call) in response.tool_calls.iter().enumerate() {
                // Track tool name
                tool_names_used.push(tool_call.name.clone());

                let result = self
                    .call_tool(&tool_call.name, &tool_call.arguments.to_string())
                    .await
                    .unwrap_or_else(|e| format!("Error: {}", e));

                // Record tool execution for persistence.
                // Only attach content to the first tool call to avoid duplication.
                tool_trace.push(ToolExecution {
                    call: tool_call.clone(),
                    result: result.clone(),
                    content: if i == 0 { assistant_content.clone() } else { None },
                });

                let tool_message = ChatMessage {
                    role: "tool".to_string(),
                    content: Some(result),
                    tool_call_id: Some(tool_call.id.clone()),
                    tool_calls: None,
                };

                // Add to both internal history and LLM messages
                self.history.push(tool_message.clone());
                messages.push(tool_message);
            }

            // Send newline to separate messages in streaming output
            let _ = token_tx.send("\n".to_string()).await;
        }

        // Trim history even on error path
        self.trim_history();

        Err(crate::error::AgentError::MaxIterationsExceeded(
            options.max_iterations,
        ))
    }

    /// Reflect on a response and potentially revise it
    async fn reflect_and_revise(
        &mut self,
        original_task: &str,
        response: &str,
        config: &ReflectionConfig,
        options: &ThinkOptions,
    ) -> Result<String, crate::error::AgentError> {
        let mut current_response = response.to_string();

        for _revision in 0..config.max_revisions {
            // Clone LLM for this scope to avoid borrow issues
            let llm = self
                .llm
                .clone()
                .ok_or_else(|| crate::error::AgentError::LlmError("No LLM attached".to_string()))?;

            // Ask LLM to reflect on the response
            let reflection_prompt = format!(
                "{}\n\nOriginal task: {}\n\nResponse to evaluate:\n{}\n\nRespond with either:\n- ACCEPTED: if the response is complete and correct\n- REVISE: <feedback> if changes are needed",
                config.prompt, original_task, current_response
            );

            let reflection_messages = vec![ChatMessage {
                role: "user".to_string(),
                content: Some(reflection_prompt),
                tool_call_id: None,
                tool_calls: None,
            }];

            // Call with retry if policy configured
            let reflection = if let Some(ref policy) = options.retry_policy {
                let llm_ref = llm.clone();
                let msgs = reflection_messages.clone();
                let result = with_retry(
                    policy,
                    || {
                        let llm = llm_ref.clone();
                        let m = msgs.clone();
                        async move { llm.chat_complete(m, None).await }
                    },
                    |e: &LLMError| e.is_retryable,
                )
                .await;
                result
                    .result
                    .map_err(|e| crate::error::AgentError::LlmError(e.message))?
            } else {
                llm.chat_complete(reflection_messages, None)
                    .await
                    .map_err(|e| crate::error::AgentError::LlmError(e.message))?
            };

            let reflection_text = reflection.content.unwrap_or_default();

            // Parse reflection result
            if reflection_text.to_uppercase().starts_with("ACCEPTED") {
                return Ok(current_response);
            }

            // Extract feedback and revise
            let feedback = if reflection_text.to_uppercase().starts_with("REVISE:") {
                reflection_text[7..].trim().to_string()
            } else {
                reflection_text.clone()
            };

            // Generate revised response
            let revision_prompt = format!(
                "Original task: {}\n\nYour previous response:\n{}\n\nFeedback:\n{}\n\nPlease provide an improved response.",
                original_task, current_response, feedback
            );

            let revision_options = ThinkOptions {
                max_iterations: options.max_iterations,
                system_prompt: options.system_prompt.clone(),
                reflection: None, // Don't recurse
                auto_memory: options.auto_memory.clone(),
                stream: false, // Reflection doesn't use streaming
                retry_policy: options.retry_policy.clone(),
                conversation_history: options.conversation_history.clone(),
                external_tools: options.external_tools.clone(),
            };

            current_response = self
                .run_agentic_loop(&revision_prompt, &revision_options)
                .await?;
        }

        // Return final response after max revisions
        Ok(current_response)
    }

    /// Core agentic loop extracted for reuse
    async fn run_agentic_loop(
        &mut self,
        task: &str,
        options: &ThinkOptions,
    ) -> Result<String, crate::error::AgentError> {
        let llm = self
            .llm
            .as_ref()
            .ok_or_else(|| crate::error::AgentError::LlmError("No LLM attached".to_string()))?;

        // Use external_tools if provided (hybrid mode), otherwise use registered tools
        let tools_list = options
            .external_tools
            .clone()
            .unwrap_or_else(|| self.list_tools_for_llm());
        // Send None instead of Some(empty_vec) for models that don't support tools
        let tools: Option<Vec<ToolSpec>> = if tools_list.is_empty() {
            None
        } else {
            Some(tools_list)
        };

        // Build auto-memory context if configured
        let memory_context = self.build_memory_context(&options.auto_memory).await;

        // Combine memory context with system prompt
        let effective_system_prompt = match (&memory_context, &options.system_prompt) {
            (Some(mem), Some(sys)) => Some(format!("{}\n\n{}", mem, sys)),
            (Some(mem), None) => Some(mem.clone()),
            (None, Some(sys)) => Some(sys.clone()),
            (None, None) => None,
        };

        let mut messages: Vec<ChatMessage> = Vec::new();

        if let Some(system) = &effective_system_prompt {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: Some(system.clone()),
                tool_call_id: None,
                tool_calls: None,
            });
        }

        // Note: Recall content (tools, memories, recall.md) is now injected as an assistant
        // message by the daemon BEFORE the user message via conversation_history, not here.
        // Inject conversation history if present (may include recall as first message)
        if let Some(history) = &options.conversation_history {
            messages.extend(history.clone());
        }

        messages.push(ChatMessage {
            role: "user".to_string(),
            content: Some(task.to_string()),
            tool_call_id: None,
            tool_calls: None,
        });

        for _iteration in 0..options.max_iterations {
            // Call LLM with retry if policy is configured
            let response = if let Some(ref policy) = options.retry_policy {
                let llm_ref = llm.clone();
                let msgs = messages.clone();
                let tls = tools.clone();
                let result = with_retry(
                    policy,
                    || {
                        let llm = llm_ref.clone();
                        let m = msgs.clone();
                        let t = tls.clone();
                        async move { llm.chat_complete(m, t).await }
                    },
                    |e: &LLMError| e.is_retryable,
                )
                .await;
                result
                    .result
                    .map_err(|e| crate::error::AgentError::LlmError(e.message))?
            } else {
                llm.chat_complete(messages.clone(), tools.clone())
                    .await
                    .map_err(|e| crate::error::AgentError::LlmError(e.message))?
            };

            if response.tool_calls.is_empty() {
                return Ok(response
                    .content
                    .unwrap_or_else(|| "No response".to_string()));
            }

            messages.push(ChatMessage {
                role: "assistant".to_string(),
                content: response.content.clone(),
                tool_call_id: None,
                tool_calls: Some(response.tool_calls.clone()),
            });

            for tool_call in &response.tool_calls {
                let result = self
                    .call_tool(&tool_call.name, &tool_call.arguments.to_string())
                    .await
                    .unwrap_or_else(|e| format!("Error: {}", e));

                messages.push(ChatMessage {
                    role: "tool".to_string(),
                    content: Some(result),
                    tool_call_id: Some(tool_call.id.clone()),
                    tool_calls: None,
                });
            }
        }

        Err(crate::error::AgentError::MaxIterationsExceeded(
            options.max_iterations,
        ))
    }

    /// Helper to create a child agent with cloned tools/LLM/observer
    fn create_child_agent(parent: &Agent, child_id: String) -> Agent {
        let (_, rx) = tokio::sync::mpsc::channel(32);

        // If parent has a router, register the child agent with it
        let (message_rx, router) = if let Some(router) = &parent.router {
            let rx = {
                let mut router_guard = router.blocking_lock();
                router_guard.register(&child_id)
            };
            (Some(rx), Some(router.clone()))
        } else {
            (None, None)
        };

        Agent {
            id: child_id,
            tools: parent.tools.clone(), // Arc clones are cheap
            inbox: rx,
            memory: None,
            llm: parent.llm.clone(), // Arc clone
            children: std::collections::HashMap::new(),
            observer: parent.observer.clone(), // Inherit observer
            message_rx,
            router,
            history: Vec::new(), // Child starts with fresh history
            agent_dir: parent.agent_dir.clone(), // Inherit agent_dir for context dumps
            current_conversation: parent.current_conversation.clone(), // Inherit conversation context
        }
    }

    /// Spawn a child agent for a subtask
    pub fn spawn_child(&mut self, config: ChildConfig) -> String {
        let child_id = format!("{}-child-{}", self.id, self.children.len());
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();

        // Create child agent with inherited tools/LLM
        let mut child = Self::create_child_agent(self, child_id.clone());
        let task = config.task.clone();

        // Spawn the child task in background
        tokio::spawn(async move {
            let result = child.think(&task).await;
            let _ = result_tx.send(result.map(|r| r.response).map_err(|e| e.to_string()));
        });

        // Store handle for parent to wait on
        let handle = ChildHandle::new(child_id.clone(), config.task, result_rx);
        self.children.insert(child_id.clone(), handle);

        child_id
    }

    /// Wait for a specific child to complete
    pub async fn wait_for_child(&mut self, child_id: &str) -> Result<String, String> {
        let handle = self
            .children
            .get_mut(child_id)
            .ok_or_else(|| format!("Child {} not found", child_id))?;

        if let Some(rx) = handle.result_rx.take() {
            match rx.await {
                Ok(Ok(result)) => {
                    handle.status = ChildStatus::Completed(result.clone());
                    Ok(result)
                }
                Ok(Err(e)) => {
                    handle.status = ChildStatus::Failed(e.clone());
                    Err(e)
                }
                Err(_) => {
                    handle.status = ChildStatus::Failed("Channel closed".to_string());
                    Err("Channel closed".to_string())
                }
            }
        } else {
            match &handle.status {
                ChildStatus::Completed(r) => Ok(r.clone()),
                ChildStatus::Failed(e) => Err(e.clone()),
                ChildStatus::Running => Err("Already waiting".to_string()),
            }
        }
    }

    /// Non-blocking check of child status
    pub fn poll_child(&self, child_id: &str) -> Option<&ChildStatus> {
        self.children.get(child_id).map(|h| &h.status)
    }

    /// Wait for all children to complete
    pub async fn wait_for_all_children(&mut self) -> Vec<Result<String, String>> {
        let child_ids: Vec<String> = self.children.keys().cloned().collect();
        let mut results = Vec::new();
        for id in child_ids {
            results.push(self.wait_for_child(&id).await);
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::InMemoryStore;
    use crate::tools::{AddTool, EchoTool};
    use serde_json::json;

    fn create_test_agent(id: &str) -> Agent {
        let (_tx, rx) = mpsc::channel(32);
        Agent::new(id.to_string(), rx)
    }

    // =========================================================================
    // Tool registration tests
    // =========================================================================

    #[test]
    fn test_agent_new() {
        let agent = create_test_agent("test-agent");
        assert_eq!(agent.id, "test-agent");
        assert!(agent.tools.is_empty());
        assert!(agent.memory.is_none());
        assert!(agent.llm.is_none());
        assert!(agent.children.is_empty());
    }

    #[test]
    fn test_register_tool() {
        let mut agent = create_test_agent("test-agent");
        agent.register_tool(Arc::new(AddTool));

        assert_eq!(agent.tools.len(), 1);
        assert!(agent.tools.contains_key("add"));
    }

    #[test]
    fn test_register_multiple_tools() {
        let mut agent = create_test_agent("test-agent");
        agent.register_tool(Arc::new(AddTool));
        agent.register_tool(Arc::new(EchoTool));

        assert_eq!(agent.tools.len(), 2);
        assert!(agent.tools.contains_key("add"));
        assert!(agent.tools.contains_key("echo"));
    }

    #[test]
    fn test_list_tools_for_llm() {
        let mut agent = create_test_agent("test-agent");
        agent.register_tool(Arc::new(AddTool));
        agent.register_tool(Arc::new(EchoTool));

        let specs = agent.list_tools_for_llm();
        assert_eq!(specs.len(), 2);

        let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"add"));
        assert!(names.contains(&"echo"));
    }

    // =========================================================================
    // Tool calling tests
    // =========================================================================

    #[tokio::test]
    async fn test_call_tool_success() {
        let mut agent = create_test_agent("test-agent");
        agent.register_tool(Arc::new(AddTool));

        let result = agent.call_tool("add", r#"{"a": 2, "b": 3}"#).await.unwrap();
        assert!(result.contains("5"));
    }

    #[tokio::test]
    async fn test_call_tool_not_found() {
        let agent = create_test_agent("test-agent");

        let result = agent.call_tool("nonexistent", "{}").await;
        assert!(matches!(result, Err(ToolError::ExecutionFailed(_))));
        if let Err(ToolError::ExecutionFailed(msg)) = result {
            assert!(msg.contains("not found"));
        }
    }

    #[tokio::test]
    async fn test_call_tool_invalid_json() {
        let mut agent = create_test_agent("test-agent");
        agent.register_tool(Arc::new(AddTool));

        let result = agent.call_tool("add", "not valid json").await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_call_echo_tool() {
        let mut agent = create_test_agent("test-agent");
        agent.register_tool(Arc::new(EchoTool));

        let result = agent
            .call_tool("echo", r#"{"message": "hello"}"#)
            .await
            .unwrap();
        assert!(result.contains("hello"));
    }

    // =========================================================================
    // Memory tests
    // =========================================================================

    #[tokio::test]
    async fn test_agent_with_memory() {
        let agent = create_test_agent("test-agent");
        let memory = Box::new(InMemoryStore::new());
        let agent = agent.with_memory(memory);

        assert!(agent.memory.is_some());
    }

    #[tokio::test]
    async fn test_remember_and_recall() {
        let agent = create_test_agent("test-agent");
        let memory = Box::new(InMemoryStore::new());
        let mut agent = agent.with_memory(memory);

        agent.remember("key1", json!("value1")).await.unwrap();

        let value = agent.recall("key1").await;
        assert_eq!(value, Some(json!("value1")));
    }

    #[tokio::test]
    async fn test_recall_nonexistent() {
        let agent = create_test_agent("test-agent");
        let memory = Box::new(InMemoryStore::new());
        let agent = agent.with_memory(memory);

        let value = agent.recall("nonexistent").await;
        assert!(value.is_none());
    }

    #[tokio::test]
    async fn test_recall_without_memory() {
        let agent = create_test_agent("test-agent");
        let value = agent.recall("key").await;
        assert!(value.is_none());
    }

    #[tokio::test]
    async fn test_remember_without_memory() {
        let mut agent = create_test_agent("test-agent");
        let result = agent.remember("key", json!("value")).await;
        assert!(matches!(result, Err(MemoryError::StorageError(_))));
    }

    #[tokio::test]
    async fn test_forget() {
        let agent = create_test_agent("test-agent");
        let memory = Box::new(InMemoryStore::new());
        let mut agent = agent.with_memory(memory);

        agent.remember("key1", json!("value1")).await.unwrap();
        assert!(agent.forget("key1").await);
        assert!(agent.recall("key1").await.is_none());
    }

    #[tokio::test]
    async fn test_forget_nonexistent() {
        let agent = create_test_agent("test-agent");
        let memory = Box::new(InMemoryStore::new());
        let mut agent = agent.with_memory(memory);

        assert!(!agent.forget("nonexistent").await);
    }

    #[tokio::test]
    async fn test_forget_without_memory() {
        let mut agent = create_test_agent("test-agent");
        assert!(!agent.forget("key").await);
    }

    // =========================================================================
    // ThinkOptions tests
    // =========================================================================

    #[test]
    fn test_think_options_default() {
        let opts = ThinkOptions::default();
        assert_eq!(opts.max_iterations, 10);
        assert!(opts.system_prompt.is_none());
        assert!(opts.reflection.is_none());
        assert!(opts.retry_policy.is_some()); // Retry enabled by default
    }

    #[test]
    fn test_reflection_config_default() {
        let config = ReflectionConfig::default();
        assert!(!config.prompt.is_empty());
        assert_eq!(config.max_revisions, 1);
    }

    // =========================================================================
    // AutoMemoryConfig tests
    // =========================================================================

    #[test]
    fn test_auto_memory_config_default() {
        let config = AutoMemoryConfig::default();
        assert_eq!(config.max_entries, 10);
        assert!(config.include_recent);
        assert!(config.key_prefixes.is_empty());
    }

    #[test]
    fn test_auto_memory_config_custom() {
        let config = AutoMemoryConfig {
            max_entries: 5,
            include_recent: false,
            key_prefixes: vec!["user:".to_string(), "task:".to_string()],
        };
        assert_eq!(config.max_entries, 5);
        assert!(!config.include_recent);
        assert_eq!(config.key_prefixes.len(), 2);
    }

    #[test]
    fn test_think_options_with_auto_memory() {
        let opts = ThinkOptions {
            max_iterations: 5,
            system_prompt: Some("Be helpful".to_string()),
            reflection: None,
            auto_memory: Some(AutoMemoryConfig::default()),
            stream: false,
            retry_policy: None,
            conversation_history: None,
            external_tools: None,
        };
        assert!(opts.auto_memory.is_some());
        assert_eq!(opts.auto_memory.as_ref().unwrap().max_entries, 10);
    }

    #[test]
    fn test_think_options_default_has_no_auto_memory() {
        let opts = ThinkOptions::default();
        assert!(opts.auto_memory.is_none());
    }

    #[tokio::test]
    async fn test_build_memory_context_no_config() {
        let agent = create_test_agent("test-agent");
        let memory = Box::new(InMemoryStore::new());
        let mut agent = agent.with_memory(memory);

        // Store some memories
        agent.remember("key1", json!("value1")).await.unwrap();

        // No config - should return None
        let context = agent.build_memory_context(&None).await;
        assert!(context.is_none());
    }

    #[tokio::test]
    async fn test_build_memory_context_no_memory() {
        let agent = create_test_agent("test-agent");

        // Agent has no memory attached
        let config = Some(AutoMemoryConfig::default());
        let context = agent.build_memory_context(&config).await;
        assert!(context.is_none());
    }

    #[tokio::test]
    async fn test_build_memory_context_empty_memory() {
        let agent = create_test_agent("test-agent");
        let memory = Box::new(InMemoryStore::new());
        let agent = agent.with_memory(memory);

        let config = Some(AutoMemoryConfig::default());
        let context = agent.build_memory_context(&config).await;
        assert!(context.is_none());
    }

    #[tokio::test]
    async fn test_build_memory_context_formats_correctly() {
        let agent = create_test_agent("test-agent");
        let memory = Box::new(InMemoryStore::new());
        let mut agent = agent.with_memory(memory);

        agent.remember("name", json!("Alice")).await.unwrap();
        agent.remember("role", json!("Engineer")).await.unwrap();

        let config = Some(AutoMemoryConfig::default());
        let context = agent.build_memory_context(&config).await;

        assert!(context.is_some());
        let ctx = context.unwrap();
        assert!(ctx.starts_with("Your memories:\n"));
        assert!(ctx.contains("name: Alice"));
        assert!(ctx.contains("role: Engineer"));
    }

    #[tokio::test]
    async fn test_build_memory_context_respects_max_entries() {
        let agent = create_test_agent("test-agent");
        let memory = Box::new(InMemoryStore::new());
        let mut agent = agent.with_memory(memory);

        // Create 5 memories
        for i in 0..5 {
            agent
                .remember(&format!("key{}", i), json!(i))
                .await
                .unwrap();
        }

        // Limit to 2 entries
        let config = Some(AutoMemoryConfig {
            max_entries: 2,
            include_recent: true,
            key_prefixes: vec![],
        });
        let context = agent.build_memory_context(&config).await.unwrap();

        // Count the entries (lines starting with "- ")
        let entry_count = context.lines().filter(|l| l.starts_with("- ")).count();
        assert_eq!(entry_count, 2);
    }

    #[tokio::test]
    async fn test_build_memory_context_filters_by_prefix() {
        let agent = create_test_agent("test-agent");
        let memory = Box::new(InMemoryStore::new());
        let mut agent = agent.with_memory(memory);

        agent.remember("user:name", json!("Bob")).await.unwrap();
        agent
            .remember("user:email", json!("bob@example.com"))
            .await
            .unwrap();
        agent.remember("config:theme", json!("dark")).await.unwrap();

        // Only get user: prefixed keys
        let config = Some(AutoMemoryConfig {
            max_entries: 10,
            include_recent: true,
            key_prefixes: vec!["user:".to_string()],
        });
        let context = agent.build_memory_context(&config).await.unwrap();

        assert!(context.contains("user:name"));
        assert!(context.contains("user:email"));
        assert!(!context.contains("config:theme"));
    }

    // =========================================================================
    // Think without LLM (error case)
    // =========================================================================

    #[tokio::test]
    async fn test_think_without_llm() {
        let mut agent = create_test_agent("test-agent");
        let result = agent.think("do something").await;
        assert!(matches!(result, Err(crate::error::AgentError::LlmError(_))));
    }

    #[tokio::test]
    async fn test_think_with_options_without_llm() {
        let mut agent = create_test_agent("test-agent");
        let result = agent
            .think_with_options("do something", ThinkOptions::default())
            .await;
        assert!(matches!(result, Err(crate::error::AgentError::LlmError(_))));
    }

    // =========================================================================
    // Child agent tests
    // =========================================================================

    #[test]
    fn test_poll_child_nonexistent() {
        let agent = create_test_agent("test-agent");
        assert!(agent.poll_child("nonexistent").is_none());
    }

    #[tokio::test]
    async fn test_wait_for_child_nonexistent() {
        let mut agent = create_test_agent("test-agent");
        let result = agent.wait_for_child("nonexistent").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[tokio::test]
    async fn test_wait_for_all_children_empty() {
        let mut agent = create_test_agent("test-agent");
        let results = agent.wait_for_all_children().await;
        assert!(results.is_empty());
    }

    // =========================================================================
    // Observer tests
    // =========================================================================

    #[test]
    fn test_agent_with_observer() {
        use crate::observe::MetricsCollector;
        let agent = create_test_agent("test-agent");
        let observer = Arc::new(MetricsCollector::new());
        let agent = agent.with_observer(observer.clone());
        assert!(agent.observer.is_some());
    }

    #[tokio::test]
    async fn test_observer_receives_tool_call_events() {
        use crate::observe::MetricsCollector;
        let mut agent = create_test_agent("test-agent");
        let observer = Arc::new(MetricsCollector::new());
        agent = agent.with_observer(observer.clone());
        agent.register_tool(Arc::new(AddTool));

        // Call tool successfully
        let _ = agent.call_tool("add", r#"{"a": 2, "b": 3}"#).await;

        let snapshot = observer.snapshot();
        assert_eq!(snapshot.tool_calls, 1);
        assert_eq!(snapshot.errors, 0);
    }

    #[tokio::test]
    async fn test_observer_receives_tool_error_events() {
        use crate::observe::MetricsCollector;
        let mut agent = create_test_agent("test-agent");
        let observer = Arc::new(MetricsCollector::new());
        agent = agent.with_observer(observer.clone());
        agent.register_tool(Arc::new(AddTool));

        // Call tool with invalid input
        let _ = agent
            .call_tool("add", r#"{"a": "not a number", "b": 3}"#)
            .await;

        let snapshot = observer.snapshot();
        assert_eq!(snapshot.tool_calls, 1);
        assert_eq!(snapshot.errors, 1); // Should count the error
    }

    #[tokio::test]
    async fn test_observer_receives_tool_not_found_events() {
        use crate::observe::MetricsCollector;
        let mut agent = create_test_agent("test-agent");
        let observer = Arc::new(MetricsCollector::new());
        agent = agent.with_observer(observer.clone());

        // Call nonexistent tool
        let _ = agent.call_tool("nonexistent", "{}").await;

        let snapshot = observer.snapshot();
        assert_eq!(snapshot.tool_calls, 1);
        assert_eq!(snapshot.errors, 1);
    }

    // =========================================================================
    // Agent-to-Agent Messaging tests
    // =========================================================================

    async fn create_test_agent_with_router(id: &str, router: Arc<Mutex<MessageRouter>>) -> Agent {
        let (_tx, rx) = mpsc::channel(32);
        let message_rx = {
            let mut router_guard = router.lock().await;
            router_guard.register(id)
        };
        Agent::new(id.to_string(), rx).with_router_and_rx(router, message_rx)
    }

    #[tokio::test]
    async fn test_agent_send_message() {
        let router = Arc::new(Mutex::new(MessageRouter::new()));
        let agent1 = create_test_agent_with_router("agent-1", router.clone()).await;
        let mut agent2 = create_test_agent_with_router("agent-2", router.clone()).await;

        // Agent 1 sends message to Agent 2
        agent1
            .send_message("agent-2", "hello from agent 1")
            .await
            .unwrap();

        // Agent 2 receives the message
        let msg = agent2.receive_message().await;
        assert!(msg.is_some());
        let msg = msg.unwrap();
        assert_eq!(msg.from, "agent-1");
        assert_eq!(msg.to, "agent-2");
        assert_eq!(msg.content, "hello from agent 1");
    }

    #[tokio::test]
    async fn test_agent_send_message_not_found() {
        let router = Arc::new(Mutex::new(MessageRouter::new()));
        let agent1 = create_test_agent_with_router("agent-1", router.clone()).await;

        // Try to send to non-existent agent
        let result = agent1.send_message("nonexistent", "hello").await;
        assert!(matches!(result, Err(MessagingError::AgentNotFound(_))));
    }

    #[tokio::test]
    async fn test_agent_send_message_not_registered() {
        let agent = create_test_agent("test-agent");

        // Try to send without being registered with a router
        let result = agent.send_message("other", "hello").await;
        assert!(matches!(result, Err(MessagingError::NotRegistered)));
    }

    #[tokio::test]
    async fn test_agent_receive_message_empty() {
        let router = Arc::new(Mutex::new(MessageRouter::new()));
        let mut agent = create_test_agent_with_router("agent-1", router.clone()).await;

        // No messages in inbox
        let msg = agent.receive_message().await;
        assert!(msg.is_none());
    }

    #[tokio::test]
    async fn test_agent_receive_message_without_router() {
        let mut agent = create_test_agent("test-agent");

        // No router attached
        let msg = agent.receive_message().await;
        assert!(msg.is_none());
    }

    #[tokio::test]
    async fn test_agent_list_peers() {
        let router = Arc::new(Mutex::new(MessageRouter::new()));
        let agent1 = create_test_agent_with_router("agent-1", router.clone()).await;
        let _agent2 = create_test_agent_with_router("agent-2", router.clone()).await;
        let _agent3 = create_test_agent_with_router("agent-3", router.clone()).await;

        let peers = agent1.list_peers().await.unwrap();
        assert_eq!(peers.len(), 3);
        assert!(peers.contains(&"agent-1".to_string()));
        assert!(peers.contains(&"agent-2".to_string()));
        assert!(peers.contains(&"agent-3".to_string()));
    }

    #[tokio::test]
    async fn test_agent_list_peers_not_registered() {
        let agent = create_test_agent("test-agent");
        let result = agent.list_peers().await;
        assert!(matches!(result, Err(MessagingError::NotRegistered)));
    }

    #[tokio::test]
    async fn test_agent_ask_and_reply() {
        let router = Arc::new(Mutex::new(MessageRouter::new()));
        let agent1 = create_test_agent_with_router("agent-1", router.clone()).await;
        let mut agent2 = create_test_agent_with_router("agent-2", router.clone()).await;

        // Spawn a task to handle the request
        let handle = tokio::spawn(async move {
            // Wait for the message
            let msg = tokio::time::timeout(std::time::Duration::from_secs(1), async {
                loop {
                    if let Some(m) = agent2.receive_message().await {
                        return m;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
            })
            .await
            .unwrap();

            assert_eq!(msg.content, "what is 2+2?");
            assert!(msg.reply_to.is_some());

            // Reply to the message
            agent2.reply_to(&msg, "4").await.unwrap();
        });

        // Agent 1 asks and waits for reply
        let response = agent1.ask("agent-2", "what is 2+2?").await.unwrap();
        assert_eq!(response, "4");

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn test_agent_receive_message_timeout() {
        let router = Arc::new(Mutex::new(MessageRouter::new()));
        let mut agent = create_test_agent_with_router("agent-1", router.clone()).await;

        // Should timeout since no messages
        let msg = agent
            .receive_message_timeout(std::time::Duration::from_millis(50))
            .await;
        assert!(msg.is_none());
    }

    #[tokio::test]
    async fn test_agent_receive_message_timeout_success() {
        let router = Arc::new(Mutex::new(MessageRouter::new()));
        let agent1 = create_test_agent_with_router("agent-1", router.clone()).await;
        let mut agent2 = create_test_agent_with_router("agent-2", router.clone()).await;

        // Send a message in a background task
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            agent1.send_message("agent-2", "hello").await.unwrap();
        });

        // Should receive the message within timeout
        let msg = agent2
            .receive_message_timeout(std::time::Duration::from_secs(1))
            .await;
        assert!(msg.is_some());
        assert_eq!(msg.unwrap().content, "hello");
    }

    #[tokio::test]
    async fn test_multiple_messages_between_agents() {
        let router = Arc::new(Mutex::new(MessageRouter::new()));
        let agent1 = create_test_agent_with_router("agent-1", router.clone()).await;
        let mut agent2 = create_test_agent_with_router("agent-2", router.clone()).await;

        // Send multiple messages
        agent1.send_message("agent-2", "message 1").await.unwrap();
        agent1.send_message("agent-2", "message 2").await.unwrap();
        agent1.send_message("agent-2", "message 3").await.unwrap();

        // Receive all messages
        let msg1 = agent2.receive_message().await.unwrap();
        let msg2 = agent2.receive_message().await.unwrap();
        let msg3 = agent2.receive_message().await.unwrap();

        assert_eq!(msg1.content, "message 1");
        assert_eq!(msg2.content, "message 2");
        assert_eq!(msg3.content, "message 3");

        // No more messages
        assert!(agent2.receive_message().await.is_none());
    }

    // =========================================================================
    // strip_thinking tests
    // =========================================================================

    #[test]
    fn test_strip_thinking_single_block() {
        let input = "<think>\nreasoning here\n</think>\n\nActual response";
        let result = strip_thinking(input);
        assert_eq!(result, "Actual response");
    }

    #[test]
    fn test_strip_thinking_multiple_blocks() {
        let input =
            "<think>first thought</think>Some text<thinking>second thought</thinking>More text";
        let result = strip_thinking(input);
        assert_eq!(result, "Some textMore text");
    }

    #[test]
    fn test_strip_thinking_no_block() {
        let input = "Just a regular response with no thinking tags";
        let result = strip_thinking(input);
        assert_eq!(result, "Just a regular response with no thinking tags");
    }

    #[test]
    fn test_strip_thinking_mixed_case_tags() {
        let input = "<THINK>uppercase</THINK>\n\nResponse\n\n<ThInKiNg>mixed</ThInKiNg>";
        let result = strip_thinking(input);
        assert_eq!(result, "Response");
    }

    #[test]
    fn test_strip_thinking_empty_result() {
        let input = "<think>only thinking, nothing else</think>";
        let result = strip_thinking(input);
        assert_eq!(result, "");
    }

    // =========================================================================
    // recall (recall.md) injection tests
    // =========================================================================

    /// Helper function to build initial messages array for testing.
    /// This mirrors the message building logic in think_with_options_inner.
    ///
    /// Recall content (tools, memories, recall.md) is now injected as an assistant message
    /// BEFORE the user message in the conversation history. This improves KV caching because
    /// user messages remain unchanged across requests.
    ///
    /// Pattern: [system] → [assistant: recall] → [user] → [assistant]
    fn build_test_messages(
        system_prompt: Option<&str>,
        conversation_history: Option<Vec<ChatMessage>>,
        user_task: &str,
    ) -> Vec<ChatMessage> {
        use crate::llm::ChatMessage;

        let mut messages: Vec<ChatMessage> = Vec::new();

        // Optional system prompt
        if let Some(system) = system_prompt {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: Some(system.to_string()),
                tool_call_id: None,
                tool_calls: None,
            });
        }

        // Inject conversation history if present
        // Note: Recall content is now injected by the daemon as an assistant message
        // at the start of conversation_history, not prepended to user message
        if let Some(history) = conversation_history {
            messages.extend(history);
        }

        // User message - clean, no prepended content
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: Some(user_task.to_string()),
            tool_call_id: None,
            tool_calls: None,
        });

        messages
    }

    /// Test 1: Verify recall content is injected as an assistant message BEFORE the user message.
    /// Expected order: [system] → [assistant: recall] → [user]
    /// This improves KV caching since user messages remain unchanged.
    #[test]
    fn test_recall_injection_as_assistant_message() {
        // Simulate recall being injected as assistant message in conversation_history
        let recall_content = "Always be concise and helpful.";
        let history_with_recall = vec![
            ChatMessage {
                role: "assistant".to_string(),
                content: Some(recall_content.to_string()),
                tool_call_id: None,
                tool_calls: None,
            },
        ];

        let messages = build_test_messages(
            Some("You are a helpful assistant."),
            Some(history_with_recall),
            "Current question",
        );

        // Verify message count: system + recall (assistant) + user = 3
        assert_eq!(messages.len(), 3);

        // Verify order: system prompt at index 0
        assert_eq!(messages[0].role, "system");
        assert_eq!(
            messages[0].content.as_ref().unwrap(),
            "You are a helpful assistant."
        );

        // Recall as assistant message at index 1
        assert_eq!(messages[1].role, "assistant");
        assert_eq!(
            messages[1].content.as_ref().unwrap(),
            recall_content
        );

        // User message at last position - clean, no prepended content
        assert_eq!(messages[2].role, "user");
        assert_eq!(messages[2].content.as_ref().unwrap(), "Current question");
    }

    /// Test 2: Verify recall is NOT stored in persisted conversation history.
    /// It should be dynamically injected each turn, not repeated from previous turns.
    #[test]
    fn test_recall_not_in_persisted_history() {
        let recall_content = "Always be concise.";

        // After the first turn, history would contain user message + assistant response.
        // The recall content should NOT be persisted in history.
        let simulated_persisted_history = vec![
            ChatMessage {
                role: "user".to_string(),
                content: Some("First question".to_string()),
                tool_call_id: None,
                tool_calls: None,
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: Some("First answer".to_string()),
                tool_call_id: None,
                tool_calls: None,
            },
        ];

        // Verify persisted history does not contain recall content
        for msg in &simulated_persisted_history {
            assert!(
                !msg.content.as_ref().unwrap().contains(recall_content),
                "recall content should not be stored in persisted conversation history"
            );
        }

        // Build messages for second turn - daemon prepends recall as assistant message
        let mut history_with_recall = vec![
            ChatMessage {
                role: "assistant".to_string(),
                content: Some(recall_content.to_string()),
                tool_call_id: None,
                tool_calls: None,
            },
        ];
        history_with_recall.extend(simulated_persisted_history);

        let messages = build_test_messages(
            Some("System prompt"),
            Some(history_with_recall),
            "Second question",
        );

        // Count how many messages contain recall_content
        let recall_count = messages
            .iter()
            .filter(|m| {
                m.content
                    .as_ref()
                    .map(|c| c.contains(recall_content))
                    .unwrap_or(false)
            })
            .count();

        // Should appear exactly once (injected by daemon before current turn)
        assert_eq!(
            recall_count, 1,
            "recall should appear exactly once per request, not duplicated"
        );

        // Verify the recall is in the first assistant message (after system)
        assert_eq!(messages[1].role, "assistant");
        assert!(
            messages[1]
                .content
                .as_ref()
                .unwrap()
                .contains(recall_content)
        );

        // Verify user message is clean
        let last_msg = &messages[messages.len() - 1];
        assert_eq!(last_msg.role, "user");
        assert_eq!(last_msg.content.as_ref().unwrap(), "Second question");
    }

    /// Test 3: Verify behavior when recall is not configured or missing.
    #[test]
    fn test_no_recall_configured() {
        // Test with no recall in conversation_history
        let messages = build_test_messages(
            Some("You are a helpful assistant."),
            Some(vec![
                ChatMessage {
                    role: "user".to_string(),
                    content: Some("Previous question".to_string()),
                    tool_call_id: None,
                    tool_calls: None,
                },
                ChatMessage {
                    role: "assistant".to_string(),
                    content: Some("Previous answer".to_string()),
                    tool_call_id: None,
                    tool_calls: None,
                },
            ]),
            "Current question",
        );

        // Should have system + 2 history + user = 4 messages
        assert_eq!(messages.len(), 4);

        // Verify last message is user with just the task
        let last = &messages[messages.len() - 1];
        assert_eq!(last.role, "user");
        assert_eq!(last.content.as_ref().unwrap(), "Current question");

        // Verify second to last is the assistant from history
        let second_to_last = &messages[messages.len() - 2];
        assert_eq!(second_to_last.role, "assistant");
        assert_eq!(second_to_last.content.as_ref().unwrap(), "Previous answer");
    }

    /// Test that ThinkOptions default has conversation_history as None
    #[test]
    fn test_think_options_default_has_no_conversation_history() {
        let opts = ThinkOptions::default();
        assert!(opts.conversation_history.is_none());
    }

    /// Test ThinkOptions with conversation_history containing recall
    #[test]
    fn test_think_options_with_conversation_history() {
        let opts = ThinkOptions {
            max_iterations: 5,
            system_prompt: Some("Be helpful".to_string()),
            reflection: None,
            auto_memory: None,
            stream: false,
            retry_policy: None,
            conversation_history: Some(vec![
                ChatMessage {
                    role: "assistant".to_string(),
                    content: Some("Recall content here.".to_string()),
                    tool_call_id: None,
                    tool_calls: None,
                },
            ]),
            external_tools: None,
        };
        assert!(opts.conversation_history.is_some());
        let history = opts.conversation_history.as_ref().unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, "assistant");
        assert_eq!(
            history[0].content.as_ref().unwrap(),
            "Recall content here."
        );
    }
}
