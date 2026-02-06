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
use tokio::sync::{Mutex, mpsc, oneshot};

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
            .truncate(true) // Fresh log each restart
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

    if let Some(cap) = fenced_re.captures(output)
        && let Ok(json) = serde_json::from_str::<serde_json::Value>(&cap[1])
        && let Some(tool_name) = json.get("tool").and_then(|t| t.as_str())
    {
        let params = json.get("params").cloned().unwrap_or(serde_json::json!({}));
        let cleaned = fenced_re.replace(output, "").trim().to_string();
        return (
            cleaned,
            Some(ToolCall {
                tool: tool_name.to_string(),
                params,
            }),
        );
    }

    (output.to_string(), None)
}

use crate::agent::{Agent, ThinkOptions};
use crate::agent_dir::{AgentDir, AgentDirError, ResolvedLlmConfig, SemanticMemorySection};
use crate::conversation::ConversationMessage;
use crate::conversation::ConversationStore;
use crate::discovery;
use crate::embedding::EmbeddingClient;
use crate::auth;
use crate::llm::{
    AnthropicClient, ChatMessage, LLM, LLMError, LLMResponse, OllamaClient, OpenAIClient, ToolSpec,
    strip_thinking_tags,
};
use crate::memory::{
    InMemoryStore, Memory, SaveResult, SemanticMemoryStore, SqliteMemory, build_memory_injection,
    extract_remember_tags,
};
use crate::observe::AgentLoggerObserver;
use crate::runtime::Runtime;
use crate::socket_api::{Request, Response, SocketApi};
use crate::tool::Tool;
use crate::tool_registry::{ToolDefinition, ToolRegistry};
use crate::tools::claude_code::{ClaudeCodeTool, TaskStatus, TaskStore, is_process_running};
use crate::tools::list_agents::DaemonListAgentsTool;
use crate::tools::send_message::DaemonSendMessageTool;
use crate::tools::{
    AddTool, DaemonRememberTool, EchoTool, HttpTool, ReadFileTool, SafeShellTool, ShellTool,
    WriteFileTool,
};

/// Work items that are serialized through the agent worker.
/// These operations require exclusive access to the Agent to prevent race conditions.
pub enum AgentWork {
    /// Process a message from the socket API (Request::Message)
    Message {
        content: String,
        conv_name: Option<String>,
        response_tx: oneshot::Sender<MessageWorkResult>,
        /// Token stream sender for streaming responses
        token_tx: Option<mpsc::Sender<String>>,
    },
    /// Process a notify request (conversation @mention)
    Notify {
        conv_id: String,
        message_id: i64,
        depth: u32,
    },
    /// Process a heartbeat
    Heartbeat,
}

/// Result from processing a Message work item
pub struct MessageWorkResult {
    pub response: String,
    pub error: Option<String>,
}

/// Result from processing in native tool mode, includes tool_calls for persistence
struct NativeToolModeResult {
    response: String,
    tool_calls: Option<Vec<crate::llm::ToolCall>>,
}

/// Expand injection directives in always.md content.
///
/// Supported directives:
/// - `<!-- @inject:tools -->` — replaced with formatted tools list
/// - `<!-- @inject:memories -->` — replaced with formatted memories
///
/// If directives are found, they are replaced in-place.
/// If no directives are found, returns None to signal fallback to append behavior.
/// If a directive is found but its content is empty, the directive line is removed.
fn expand_inject_directives(
    content: &str,
    tools_injection: &str,
    memory_injection: &str,
) -> Option<String> {
    const TOOLS_DIRECTIVE: &str = "<!-- @inject:tools -->";
    const MEMORIES_DIRECTIVE: &str = "<!-- @inject:memories -->";

    let has_tools_directive = content.contains(TOOLS_DIRECTIVE);
    let has_memories_directive = content.contains(MEMORIES_DIRECTIVE);

    // No directives found — signal that user opted out of auto-injection
    if !has_tools_directive && !has_memories_directive {
        return None;
    }

    let mut result = content.to_string();

    // Replace or remove each directive based on whether its content is empty
    let directives: &[(&str, &str)] = &[
        (TOOLS_DIRECTIVE, tools_injection),
        (MEMORIES_DIRECTIVE, memory_injection),
    ];

    for &(directive, injection) in directives {
        if !result.contains(directive) {
            continue;
        }
        if injection.is_empty() {
            result = result
                .lines()
                .filter(|line| line.trim() != directive)
                .collect::<Vec<_>>()
                .join("\n");
        } else {
            result = result.replace(directive, injection);
        }
    }

    Some(result)
}

/// Build the recall content by combining tools, memory, base recall, and model recall.
///
/// Injection behavior:
/// - If base_recall contains directives (`<!-- @inject:tools -->`, `<!-- @inject:memories -->`),
///   tools and memories are expanded in-place at those positions.
/// - If base_recall exists but has no directives, it's used as-is (user opted out of injection).
/// - If base_recall is None, tools and memories are injected as sensible defaults.
fn build_recall_content(
    tools_injection: &str,
    memory_injection: &str,
    base_recall: &Option<String>,
    model_recall: &Option<String>,
) -> Option<String> {
    let mut parts = Vec::new();

    // Try directive expansion first
    let expanded_base = base_recall
        .as_ref()
        .and_then(|base| expand_inject_directives(base, tools_injection, memory_injection));

    if let Some(expanded) = expanded_base {
        // Directives were found and expanded — use the expanded content
        parts.push(expanded);
    } else if let Some(base) = base_recall {
        // recall.md exists but has no directives — user opted out of auto-injection
        // Just use the recall.md content as-is
        parts.push(base.clone());
    } else {
        // No recall.md at all — inject tools/memories as sensible defaults
        if !tools_injection.is_empty() {
            parts.push(tools_injection.to_string());
        }
        if !memory_injection.is_empty() {
            parts.push(memory_injection.to_string());
        }
    }

    // Model-specific recall is always appended after agent recall
    if let Some(model) = model_recall {
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
/// - `final_user_content` is the last user turn's content
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
        if msg.from_agent == "recall" {
            // Recall goes BEFORE pending user messages to maintain:
            // [recall] [user] ordering (recall precedes the query it supports)
            history.push(ChatMessage {
                role: "assistant".to_string(),
                content: Some(msg.content.clone()),
                tool_call_id: None,
                tool_calls: None,
            });
        } else if msg.from_agent == current_agent {
            // Agent's own response — flush pending user messages first
            flush_user_batch(&mut pending_user_batch, &mut history);

            let tool_calls: Option<Vec<crate::llm::ToolCall>> = msg
                .tool_calls
                .as_ref()
                .and_then(|json| serde_json::from_str(json).ok());

            history.push(ChatMessage {
                role: "assistant".to_string(),
                content: Some(msg.content.clone()),
                tool_call_id: None,
                tool_calls,
            });
        } else if msg.from_agent == "tool" {
            // Skip tool results triggered by other agents — they're irrelevant noise.
            // Unattributed (triggered_by = None) are included for backward compatibility.
            match &msg.triggered_by {
                Some(owner) if owner != current_agent => continue,
                _ => {}
            }
            // Tool results/errors should NOT be batched with user messages.
            // Flush any pending user messages first, then add tool message as its own user message.
            flush_user_batch(&mut pending_user_batch, &mut history);
            history.push(ChatMessage {
                role: "user".to_string(),
                content: Some(msg.content.clone()), // Raw content, not JSON-wrapped
                tool_call_id: None,
                tool_calls: None,
            });
        } else {
            // Other speaker → accumulate for user batch with JSON wrapper
            let escaped = msg
                .content
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
    pub allowed_tools: Option<Vec<String>>,
}

/// Result of building recall content (tools + memories).
struct RecallResult {
    /// Combined recall content (tools injection + memory injection + recall.md + model recall)
    recall_content: Option<String>,
    /// Tool specs for native tool mode (None in JSON-block mode)
    external_tools: Option<Vec<ToolSpec>>,
    /// Relevant tool definitions for JSON-block mode (empty in native mode)
    relevant_tools: Vec<ToolDefinition>,
}

/// Build tools and memory injection for a query.
///
/// This shared helper handles:
/// - Tool recall (native mode: all allowed tools, JSON-block mode: keyword-matched tools)
/// - Memory recall (semantic search with embeddings)
/// - Combining with recall.md and model-specific recall
#[allow(clippy::too_many_arguments)]
async fn build_recall_for_query(
    query: &str,
    allowed_tools: &Option<Vec<String>>,
    semantic_memory: &Option<Arc<Mutex<SemanticMemoryStore>>>,
    embedding_client: &Option<Arc<EmbeddingClient>>,
    tool_registry: &Option<Arc<ToolRegistry>>,
    use_native_tools: bool,
    recall: &Option<String>,
    model_recall: &Option<String>,
    recall_limit: usize,
    logger: &Arc<AgentLogger>,
) -> RecallResult {
    // Build tools injection or tool specs
    let (tools_injection, external_tools, relevant_tools) = if use_native_tools {
        // Native tool calling: pass ALL allowed tools from registry
        let all_tools = if let Some(registry) = tool_registry {
            let all: Vec<&ToolDefinition> = registry.all_tools().iter().collect();
            filter_by_allowlist(all, allowed_tools)
        } else {
            Vec::new()
        };
        if !all_tools.is_empty() {
            logger.tool(&format!("Native tools: {} allowed", all_tools.len()));
        }
        let specs = if !all_tools.is_empty() {
            Some(tool_definitions_to_specs(&all_tools))
        } else {
            None
        };
        (String::new(), specs, Vec::new())
    } else {
        // JSON-block mode: keyword-match relevant tools
        let relevant_tools = if let Some(registry) = tool_registry {
            let relevant = registry.find_relevant(query, recall_limit);
            let relevant = filter_by_allowlist(relevant, allowed_tools);
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
        let injection = ToolRegistry::format_for_prompt(&relevant_tools);
        // Clone tool definitions since we need to return owned values
        let owned_tools: Vec<ToolDefinition> = relevant_tools.iter().map(|t| (*t).clone()).collect();
        (injection, None, owned_tools)
    };

    // Inject relevant memories
    let memory_injection = if let Some(mem_store) = semantic_memory {
        let query_embedding = if let Some(emb_client) = embedding_client {
            match emb_client.embed(query).await {
                Ok(emb) => Some(emb),
                Err(e) => {
                    logger.log(&format!("Failed to generate query embedding: {}", e));
                    None
                }
            }
        } else {
            None
        };

        let store = mem_store.lock().await;
        match store.recall_with_embedding(query, recall_limit, query_embedding.as_deref()) {
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
                logger.log(&format!("Memory recall error: {}", e));
                String::new()
            }
        }
    } else {
        String::new()
    };

    let recall_content =
        build_recall_content(&tools_injection, &memory_injection, recall, model_recall);

    RecallResult {
        recall_content,
        external_tools,
        relevant_tools,
    }
}
/// Save extracted [REMEMBER: ...] tags to semantic memory.
async fn save_memories(
    memories_to_save: &[String],
    semantic_memory: &Option<Arc<Mutex<SemanticMemoryStore>>>,
    embedding_client: &Option<Arc<EmbeddingClient>>,
    logger: &Arc<AgentLogger>,
) {
    if memories_to_save.is_empty() {
        return;
    }
    let Some(mem_store) = semantic_memory else {
        return;
    };
    let store = mem_store.lock().await;
    for memory in memories_to_save {
        let embedding = if let Some(emb_client) = embedding_client {
            emb_client.embed(memory).await.ok()
        } else {
            None
        };
        match store.save_with_embedding(memory, 0.9, "explicit", embedding.as_deref()) {
            Ok(SaveResult::New(id)) => {
                logger.memory(&format!("Save #{}: \"{}\"", id, memory))
            }
            Ok(SaveResult::Reinforced(id, old, new)) => logger.memory(&format!(
                "Reinforce #{}: \"{}\" ({:.2} → {:.2})",
                id, memory, old, new
            )),
            Err(e) => logger.log(&format!("Failed to save memory: {}", e)),
        }
    }
}

/// Prepend recall content as an assistant message to conversation history.
/// Pattern: [history...] [assistant: recall] [user: task]
fn inject_recall_into_history(
    recall_content: &Option<String>,
    conversation_history: &mut Vec<ChatMessage>,
) {
    if let Some(recall_text) = recall_content {
        if !recall_text.is_empty() {
            conversation_history.push(ChatMessage {
                role: "assistant".to_string(),
                content: Some(recall_text.clone()),
                tool_call_id: None,
                tool_calls: None,
            });
        }
    }
}

/// Spawn a background task that persists tool trace executions to the conversation store.
/// Returns the join handle. The caller should drop `trace_tx` when done to signal completion.
fn spawn_tool_trace_persister(
    conv_name: Option<String>,
    agent_name: String,
    logger: Arc<AgentLogger>,
    trace_rx: tokio::sync::mpsc::Receiver<crate::agent::ToolExecution>,
) -> tokio::task::JoinHandle<()> {
    let mut trace_rx = trace_rx;
    tokio::spawn(async move {
        let Some(cname) = conv_name else {
            // No conversation to persist to -- just drain
            while trace_rx.recv().await.is_some() {}
            return;
        };
        let store = match ConversationStore::init() {
            Ok(s) => s,
            Err(e) => {
                logger.log(&format!("[trace] Failed to init trace store: {}", e));
                while trace_rx.recv().await.is_some() {} // drain
                return;
            }
        };
        while let Some(exec) = trace_rx.recv().await {
            let tool_calls_json = serde_json::to_string(&vec![&exec.call]).ok();
            let content = exec.content.as_deref().unwrap_or("");
            if let Err(e) = store.add_message_with_tool_calls(
                &cname, &agent_name, content, &[], None,
                tool_calls_json.as_deref(),
            ) {
                logger.log(&format!("[trace] Failed to store tool call: {}", e));
            }
            let tool_result_msg = format!("[Tool Result for {}]\n{}", exec.call.name, exec.result);
            if let Err(e) = store.add_tool_result(&cname, &tool_result_msg, &agent_name) {
                logger.log(&format!("[trace] Failed to store tool result: {}", e));
            }
        }
    })
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
            let tool = ReadFileTool;
            match tool.execute(tool_call.params.clone()).await {
                Ok(result) => {
                    if let Some(contents) = result.get("contents").and_then(|c| c.as_str()) {
                        Ok(contents.to_string())
                    } else {
                        Ok(result.to_string())
                    }
                }
                Err(e) => Err(format!("Tool error: {}", e)),
            }
        }
        "write_file" => {
            let tool = WriteFileTool;
            match tool.execute(tool_call.params.clone()).await {
                Ok(result) => {
                    if let Some(msg) = result.get("message").and_then(|m| m.as_str()) {
                        Ok(msg.to_string())
                    } else {
                        Ok(result.to_string())
                    }
                }
                Err(e) => Err(format!("Tool error: {}", e)),
            }
        }
        "shell" | "safe_shell" => {
            // Validate command against allowed_commands if set
            if let Some(def) = tool_def
                && let Some(ref allowed) = def.allowed_commands
            {
                let command = tool_call
                    .params
                    .get("command")
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

            let tool = ShellTool::default();
            match tool.execute(tool_call.params.clone()).await {
                Ok(result) => {
                    // Shell returns stdout, stderr, and exit_code
                    let stdout = result.get("stdout").and_then(|s| s.as_str()).unwrap_or("");
                    let stderr = result.get("stderr").and_then(|s| s.as_str()).unwrap_or("");
                    let exit_code = result
                        .get("exit_code")
                        .and_then(|c| c.as_i64())
                        .unwrap_or(0);

                    let mut output = String::new();
                    if !stdout.is_empty() {
                        output.push_str(stdout);
                    }
                    if !stderr.is_empty() {
                        if !output.is_empty() {
                            output.push('\n');
                        }
                        output.push_str("[stderr] ");
                        output.push_str(stderr);
                    }
                    if exit_code != 0 {
                        output.push_str(&format!("\n[exit code: {}]", exit_code));
                    }
                    Ok(output)
                }
                Err(e) => Err(format!("Tool error: {}", e)),
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
                Err(e) => Err(format!("Tool error: {}", e)),
            }
        }
        "claude_code" => {
            // Claude Code tool requires agent context and task store
            let ctx = context.ok_or("claude_code tool requires execution context")?;
            let task_store = ctx
                .task_store
                .as_ref()
                .ok_or("claude_code tool requires task store to be initialized")?;

            let tool = ClaudeCodeTool::with_conv_id(
                ctx.agent_name.clone(),
                task_store.clone(),
                ctx.conv_id.clone(),
            );
            match tool.execute(tool_call.params.clone()).await {
                Ok(result) => {
                    if let Some(msg) = result.get("message").and_then(|m| m.as_str()) {
                        Ok(msg.to_string())
                    } else {
                        Ok(result.to_string())
                    }
                }
                Err(e) => Err(format!("Tool error: {}", e)),
            }
        }
        "remember" => {
            // Remember tool requires semantic memory store
            let ctx = context.ok_or("remember tool requires execution context")?;
            let mem_store = ctx
                .semantic_memory_store
                .as_ref()
                .ok_or("remember tool requires semantic memory to be enabled")?;

            let content = tool_call
                .params
                .get("content")
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
                            format!(
                                "Reinforced memory: {} (id={}, importance {:.2} → {:.2})",
                                content, id, old_imp, new_imp
                            )
                        }
                    };
                    Ok(msg)
                }
                Err(e) => Err(format!("Failed to save memory: {}", e)),
            }
        }
        "list_agents" => {
            let ctx = context.ok_or("list_agents tool requires execution context")?;
            let tool = DaemonListAgentsTool::new(ctx.agent_name.clone());
            match tool.execute(tool_call.params.clone()).await {
                Ok(result) => {
                    // Use the pre-formatted summary if available, otherwise format from agents array
                    if let Some(summary) = result.get("summary").and_then(|s| s.as_str()) {
                        Ok(summary.to_string())
                    } else {
                        // Fallback: format from agents array (handles both string and object formats)
                        let agents = result
                            .get("agents")
                            .and_then(|a| a.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| {
                                        // Handle both string format and object format
                                        v.as_str().map(|s| s.to_string()).or_else(|| {
                                            v.get("name")
                                                .and_then(|n| n.as_str())
                                                .map(|s| s.to_string())
                                        })
                                    })
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            })
                            .unwrap_or_default();
                        let count = result.get("count").and_then(|c| c.as_u64()).unwrap_or(0);
                        if count == 0 {
                            Ok("No other agents are currently running.".to_string())
                        } else {
                            Ok(format!("Available agents: {}", agents))
                        }
                    }
                }
                Err(e) => Err(format!("Tool error: {}", e)),
            }
        }
        "list_tools" => {
            // Load tool registry and return tool names filtered by agent's allowlist
            let tools_path = dirs::home_dir()
                .map(|h| h.join(".anima").join("tools.toml"))
                .ok_or("Could not determine home directory")?;

            // Get allowed tools from context (if available)
            let allowed = context.and_then(|ctx| ctx.allowed_tools.as_ref());

            match ToolRegistry::load_from_file(&tools_path) {
                Ok(registry) => {
                    let all_tools: Vec<&str> = registry
                        .all_tools()
                        .iter()
                        .map(|t| t.name.as_str())
                        .collect();

                    // Filter by allowlist if present
                    let tool_names: Vec<&str> = match allowed {
                        Some(allowlist) => all_tools
                            .into_iter()
                            .filter(|t| allowlist.iter().any(|a| a == *t))
                            .collect(),
                        None => all_tools, // No allowlist = show all (for testing)
                    };

                    // Always include built-in tools that are typically allowed
                    let mut result: Vec<&str> = tool_names;
                    for builtin in &["list_tools", "list_agents", "remember", "send_message"] {
                        if !result.contains(builtin)
                            && allowed
                                .map(|a| a.iter().any(|x| x == *builtin))
                                .unwrap_or(true)
                        {
                            result.push(builtin);
                        }
                    }

                    if result.is_empty() {
                        Ok("No tools available for this agent.".to_string())
                    } else {
                        Ok(format!("Available tools: {}", result.join(", ")))
                    }
                }
                Err(e) => {
                    // Fallback: return built-in tools if registry fails to load
                    Ok(format!(
                        "Built-in tools: list_tools, list_agents, remember, send_message\n(Note: tools.toml failed to load: {})",
                        e
                    ))
                }
            }
        }
        _ => Err(format!("Unknown tool: {}", tool_call.tool)),
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
        }),
    }
}

/// Convert ToolDefinitions from registry to ToolSpecs for native LLM tool calling.
fn tool_definitions_to_specs(definitions: &[&ToolDefinition]) -> Vec<ToolSpec> {
    definitions
        .iter()
        .map(|def| ToolSpec {
            name: def.name.clone(),
            description: def.description.clone(),
            parameters: convert_params_to_json_schema(&def.params),
        })
        .collect()
}

/// Filter tools by allowed_tools list. If allowed_tools is None, no tools allowed (safe default).
fn filter_by_allowlist<'a>(
    tools: Vec<&'a ToolDefinition>,
    allowed_tools: &Option<Vec<String>>,
) -> Vec<&'a ToolDefinition> {
    match allowed_tools {
        Some(allowed) => tools
            .into_iter()
            .filter(|t| allowed.contains(&t.name))
            .collect(),
        None => Vec::new(), // No allowlist = no tools
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
    /// System prompt
    pub system_prompt: Option<String>,
    /// Recall content (injected as assistant message before user messages)
    pub recall: Option<String>,
    /// Model-specific recall text (appended to agent recall)
    pub model_recall: Option<String>,
    /// Semantic memory configuration
    pub semantic_memory: SemanticMemorySection,
    /// Allowlist of tool names. If set, only these tools are available.
    pub allowed_tools: Option<Vec<String>>,
    /// Context window size (num_ctx) for token tracking
    pub num_ctx: Option<u32>,
    /// Maximum tool call iterations per turn
    pub max_iterations: Option<usize>,
    /// Tool calls per checkpoint window (None = disabled)
    pub checkpoint_interval: Option<usize>,
    /// Maximum number of checkpoint restarts (default: 5)
    pub max_checkpoints: Option<usize>,
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
                    message: t
                        .message
                        .clone()
                        .unwrap_or_else(|| "Timer trigger".to_string()),
                })
            } else {
                None
            }
        });

        // Parse heartbeat config if enabled and interval set
        let heartbeat_path = dir_path.join("heartbeat.md");
        let heartbeat = if agent_dir.config.heartbeat.enabled {
            agent_dir
                .config
                .heartbeat
                .interval
                .as_ref()
                .and_then(|interval_str| {
                    parse_duration(interval_str).map(|interval| HeartbeatDaemonConfig {
                        interval,
                        heartbeat_path: heartbeat_path.clone(),
                    })
                })
        } else {
            None
        };

        // Load system prompt
        let system_prompt = agent_dir.load_system()?;

        // Load recall content
        let recall = agent_dir.load_recall()?;

        // Load model-specific config from resolved LLM config
        let llm_config = agent_dir.resolve_llm_config()?;
        let model_recall = llm_config.recall;
        let allowed_tools = llm_config.allowed_tools;

        // Build runtime context and append to system prompt
        let host = gethostname::gethostname().to_string_lossy().to_string();
        let runtime_context =
            build_runtime_context(&name, &llm_config.model, &host, llm_config.tools);

        let system_prompt = match system_prompt {
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
            system_prompt,
            recall,
            model_recall,
            semantic_memory: agent_dir.config.semantic_memory.clone(),
            allowed_tools,
            num_ctx: llm_config.num_ctx,
            max_iterations: agent_dir.config.think.max_iterations,
            checkpoint_interval: agent_dir.config.think.checkpoint_interval,
            max_checkpoints: agent_dir.config.think.max_checkpoints,
        })
    }
}

/// Build the runtime context string that is appended to the system prompt.
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
        let num_end = remaining
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(remaining.len());
        if num_end == 0 {
            return None;
        }

        let num: u64 = remaining[..num_end].parse().ok()?;
        remaining = &remaining[num_end..];

        // Find the end of the unit part
        let unit_end = remaining
            .find(|c: char| c.is_ascii_digit())
            .unwrap_or(remaining.len());
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
        content
            .trim()
            .parse()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Check if a PID file exists and the process is still running.
    pub fn is_running(path: impl AsRef<Path>) -> bool {
        if let Ok(pid) = Self::read(&path) {
            // Check if process exists by sending signal 0
            unsafe { libc::kill(pid as i32, 0) == 0 }
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
        )
        .into());
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
        logger.log(&format!(
            "  Timer: every {:?}, message: \"{}\"",
            timer.interval, timer.message
        ));
    }
    if let Some(ref hb) = config.heartbeat {
        logger.log(&format!(
            "  Heartbeat: every {:?}, file: {}",
            hb.interval,
            hb.heartbeat_path.display()
        ));
    }

    // Create the agent
    let (agent, use_native_tools) = create_agent_from_dir(&agent_dir, logger.clone()).await?;
    let agent = Arc::new(Mutex::new(agent));
    logger.log(&format!("  Native tools: {}", use_native_tools));

    // Create embedding client if configured
    let embedding_client: Option<Arc<EmbeddingClient>> =
        if let Some(ref emb_config) = config.semantic_memory.embedding {
            if emb_config.provider == "ollama" {
                let client = EmbeddingClient::new(&emb_config.model, Some(&emb_config.url));
                logger.log(&format!(
                    "  Embedding client: {} via {} at {}",
                    emb_config.model, emb_config.provider, emb_config.url
                ));
                Some(Arc::new(client))
            } else {
                logger.log(&format!(
                    "  Embedding client: unsupported provider '{}'",
                    emb_config.provider
                ));
                None
            }
        } else {
            None
        };

    // Create semantic memory store if enabled
    let semantic_memory_store: Option<Arc<Mutex<SemanticMemoryStore>>> =
        if config.semantic_memory.enabled {
            let mem_path = config.agent_dir.join(&config.semantic_memory.path);
            let store = SemanticMemoryStore::open(&mem_path, &config.name)?;
            logger.log(&format!("  Semantic memory: {}", mem_path.display()));

            // Backfill embeddings if needed
            if let Some(ref emb_client) = embedding_client
                && store.needs_backfill(emb_client.model())?
            {
                logger.log("  Backfilling embeddings...");
                let memories = store.get_memories_needing_embeddings()?;
                logger.log(&format!("    {} memories need embeddings", memories.len()));

                for (id, content) in memories {
                    match emb_client.embed(&content).await {
                        Ok(embedding) => {
                            if let Err(e) = store.update_embedding(id, &embedding) {
                                logger.log(&format!(
                                    "    Failed to update embedding for #{}: {}",
                                    id, e
                                ));
                            }
                        }
                        Err(e) => {
                            logger.log(&format!(
                                "    Failed to generate embedding for #{}: {}",
                                id, e
                            ));
                        }
                    }
                }

                // Update the stored model
                store.set_embedding_model(emb_client.model())?;
                logger.log("  Backfill complete");
            }

            Some(Arc::new(Mutex::new(store)))
        } else {
            None
        };

    // Register DaemonRememberTool if semantic memory is enabled and native tools are supported
    if use_native_tools {
        if let Some(ref mem_store) = semantic_memory_store {
            let remember_tool =
                DaemonRememberTool::new(Arc::clone(mem_store), embedding_client.clone());
            agent.lock().await.register_tool(Arc::new(remember_tool));
            logger.log("  Registered DaemonRememberTool");
        } else {
            logger.log("  Skipped DaemonRememberTool (no semantic memory)");
        }
    }

    // Load tool registry (if available)
    let tool_registry: Option<Arc<ToolRegistry>> = match ToolRegistry::load_global() {
        Ok(registry) => {
            logger.log(&format!(
                "  Tool registry: {} tools loaded",
                registry.all_tools().len()
            ));
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
                    logger.log(&format!(
                        "Processing {} pending notifications...",
                        pending.len()
                    ));
                    for notification in &pending {
                        logger.log(&format!(
                            "  Processing notification: conv={} msg_id={}",
                            notification.conv_name, notification.message_id
                        ));

                        let response = handle_notify(
                            &notification.conv_name,
                            notification.message_id,
                            0, // Start at depth 0 for pending notifications
                            &agent,
                            &config.name,
                            &config.system_prompt,
                            &config.recall,
                            &config.model_recall,
                            &config.allowed_tools,
                            &semantic_memory_store,
                            &embedding_client,
                            &tool_registry,
                            use_native_tools,
                            &logger,
                            config.semantic_memory.recall_limit,
                            config.semantic_memory.history_limit,
                            &task_store,
                            config.num_ctx,
                            config.max_iterations,
                            config.checkpoint_interval,
                            config.max_checkpoints,
                        )
                        .await;

                        match response {
                            Response::Notified {
                                response_message_id,
                            } => {
                                logger.log(&format!(
                                    "  Responded with msg_id={}",
                                    response_message_id
                                ));
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
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to create SIGTERM handler");

        let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
            .expect("Failed to create SIGINT handler");

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

    // Create mpsc channel for serializing agent work
    // All Message, Notify, and Heartbeat work goes through this channel to prevent race conditions
    let (work_tx, work_rx) = mpsc::unbounded_channel::<AgentWork>();

    // Spawn the worker task that owns the agent and processes work sequentially
    let worker_handle = {
        let worker_agent = agent.clone();
        let worker_name = config.name.clone();
        let worker_system_prompt = config.system_prompt.clone();
        let worker_recall = config.recall.clone();
        let worker_model_recall = config.model_recall.clone();
        let worker_allowed_tools = config.allowed_tools.clone();
        let worker_semantic_memory = semantic_memory_store.clone();
        let worker_embedding_client = embedding_client.clone();
        let worker_tool_registry = tool_registry.clone();
        let worker_logger = logger.clone();
        let worker_recall_limit = config.semantic_memory.recall_limit;
        let worker_history_limit = config.semantic_memory.history_limit;
        let worker_task_store = task_store.clone();
        let worker_heartbeat_config = config.heartbeat.clone();

        let worker_num_ctx = config.num_ctx;
        let worker_max_iterations = config.max_iterations;
        let worker_checkpoint_interval = config.checkpoint_interval;
        let worker_max_checkpoints = config.max_checkpoints;
        tokio::spawn(async move {
            agent_worker(
                work_rx,
                worker_agent,
                worker_name,
                worker_system_prompt,
                worker_recall,
                worker_model_recall,
                worker_allowed_tools,
                worker_semantic_memory,
                worker_embedding_client,
                worker_tool_registry,
                use_native_tools,
                worker_logger,
                worker_recall_limit,
                worker_history_limit,
                worker_task_store,
                worker_heartbeat_config,
                worker_num_ctx,
                worker_max_iterations,
                worker_checkpoint_interval,
                worker_max_checkpoints,
            )
            .await
        })
    };

    logger.log("  Worker task: started");

    // Set up timer if configured (timer still uses its own agent access since it's a simple single-turn)
    let timer_handle = if let Some(ref timer_config) = config.timer {
        let timer_work_tx = work_tx.clone();
        let timer_config = timer_config.clone();
        let shutdown_clone = shutdown.clone();
        let timer_logger = logger.clone();

        Some(tokio::spawn(async move {
            let mut interval = tokio::time::interval(timer_config.interval);
            // Skip the first immediate tick
            interval.tick().await;

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        timer_logger.log("[timer] Firing timer trigger...");
                        // Timer uses a simple Message work item (no streaming needed)
                        let (response_tx, response_rx) = oneshot::channel();
                        if timer_work_tx.send(AgentWork::Message {
                            content: timer_config.message.clone(),
                            conv_name: None,
                            response_tx,
                            token_tx: None,
                        }).is_ok() {
                            match response_rx.await {
                                Ok(result) => {
                                    if let Some(err) = result.error {
                                        timer_logger.log(&format!("[timer] Error: {}", err));
                                    } else {
                                        timer_logger.log(&format!("[timer] Response: {}", result.response));
                                    }
                                }
                                Err(_) => {
                                    timer_logger.log("[timer] Worker dropped response channel");
                                }
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
    let heartbeat_config = config.heartbeat.clone();

    let heartbeat_handle = if let Some(ref hb_config) = heartbeat_config {
        // Only start heartbeat loop if heartbeat.md exists
        if hb_config.heartbeat_path.exists() {
            let hb_work_tx = work_tx.clone();
            let hb_config = hb_config.clone();
            let shutdown_clone = shutdown.clone();
            let hb_logger = logger.clone();

            Some(tokio::spawn(async move {
                let mut interval = tokio::time::interval(hb_config.interval);
                // Skip the first immediate tick
                interval.tick().await;

                loop {
                    tokio::select! {
                        _ = interval.tick() => {
                            hb_logger.log("[heartbeat] Timer fired, sending work");
                            // Simply send heartbeat work - no need for try_lock or pending flag
                            // The worker will process it in order
                            if hb_work_tx.send(AgentWork::Heartbeat).is_err() {
                                hb_logger.log("[heartbeat] Worker channel closed");
                                break;
                            }
                        }
                        _ = shutdown_clone.notified() => {
                            hb_logger.log("[heartbeat] Shutting down heartbeat loop");
                            break;
                        }
                    }
                }
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
            task_watcher_loop(ts_clone, agent_name, tw_logger, shutdown_clone).await
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
                        let system_prompt = config.system_prompt.clone();
                        let always = config.recall.clone();
                        let model_always = config.model_recall.clone();
                        let allowed_tools = config.allowed_tools.clone();
                        let shutdown_clone = shutdown.clone();
                        let semantic_memory = semantic_memory_store.clone();
                        let conn_embedding_client = embedding_client.clone();
                        let conn_registry = tool_registry.clone();
                        let conn_logger = logger.clone();
                        let conn_heartbeat_config = heartbeat_config.clone();
                        let conn_task_store = task_store.clone();
                        let conn_work_tx = work_tx.clone();

                        let recall_limit = config.semantic_memory.recall_limit;
                        let history_limit = config.semantic_memory.history_limit;
                        tokio::spawn(async move {
                            let api = SocketApi::new(stream);
                            if let Err(e) = handle_connection(
                                api,
                                agent_clone,
                                agent_name,
                                system_prompt,
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
                                history_limit,
                                conn_heartbeat_config,
                                conn_task_store,
                                conn_work_tx,
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

    // Drop the work_tx so the worker knows to stop
    drop(work_tx);

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

    // Wait for worker task to finish
    let _ = worker_handle.await;

    // Clean up socket file
    if config.socket_path.exists() {
        let _ = std::fs::remove_file(&config.socket_path);
    }

    logger.log("Daemon stopped.");
    Ok(())
}

/// Create an agent from an AgentDir configuration.
/// Returns (Agent, use_native_tools) where use_native_tools indicates hybrid tool calling mode.
async fn create_agent_from_dir(
    agent_dir: &AgentDir,
    logger: Arc<AgentLogger>,
) -> Result<(Agent, bool), Box<dyn std::error::Error>> {
    let agent_name = agent_dir.config.agent.name.clone();

    // Resolve LLM config (loads model file if specified, applies overrides)
    let llm_config = agent_dir.resolve_llm_config()?;
    let use_native_tools = llm_config.tools;

    // Get API key using resolved config
    let api_key = AgentDir::api_key_for_config(&llm_config)?;

    // Create LLM from resolved config
    let llm: Arc<dyn LLM> = create_llm_from_config(&llm_config, api_key).await?;

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
        agent.register_tool(Arc::new(SafeShellTool::new()));
        // Note: DaemonRememberTool is registered later after semantic memory is created

        // Register daemon-aware messaging tools
        agent.register_tool(Arc::new(DaemonSendMessageTool::new(agent_name.clone())));
        agent.register_tool(Arc::new(DaemonListAgentsTool::new(agent_name.clone())));
    }

    // Apply LLM, memory, and agent_dir
    agent = agent.with_llm(llm);
    agent = agent.with_memory(memory);
    agent = agent.with_agent_dir(agent_dir.path.clone());

    // Add observer that logs to agent.log via AgentLogger
    let observer = Arc::new(AgentLoggerObserver::new(logger));
    agent = agent.with_observer(observer);

    Ok((agent, use_native_tools))
}

// ---------------------------------------------------------------------------
// Refreshing Anthropic client (auto-refreshes OAuth tokens for daemons)
// ---------------------------------------------------------------------------

/// An LLM wrapper that auto-refreshes OAuth Bearer tokens before expiry.
/// For API key auth, this is a transparent passthrough.
struct RefreshingAnthropicClient {
    inner: tokio::sync::RwLock<AnthropicClient>,
    refresh_token: String,
    expires_at: std::sync::atomic::AtomicI64,
    model: String,
    base_url: Option<String>,
    max_tokens: Option<u32>,
}

impl RefreshingAnthropicClient {
    fn new(
        tokens: auth::StoredTokens,
        model: String,
        base_url: Option<String>,
        max_tokens: Option<u32>,
    ) -> Self {
        let mut client = AnthropicClient::with_bearer(&tokens.access_token).with_model(&model);
        if let Some(ref url) = base_url {
            client = client.with_base_url(url);
        }
        if let Some(mt) = max_tokens {
            client = client.with_max_tokens(mt);
        }
        Self {
            inner: tokio::sync::RwLock::new(client),
            refresh_token: tokens.refresh_token,
            expires_at: std::sync::atomic::AtomicI64::new(tokens.expires_at),
            model,
            base_url,
            max_tokens,
        }
    }

    async fn ensure_fresh(&self) {
        let now = chrono::Utc::now().timestamp();
        let exp = self.expires_at.load(std::sync::atomic::Ordering::Relaxed);
        if now < exp - 300 {
            return; // still valid
        }

        // Try to refresh — extract what we need before the write lock
        let refreshed = auth::refresh_tokens(&self.refresh_token).await;
        if let Ok(new_tokens) = refreshed {
            let _ = auth::save_tokens(&new_tokens);
            self.expires_at
                .store(new_tokens.expires_at, std::sync::atomic::Ordering::Relaxed);

            let mut client =
                AnthropicClient::with_bearer(&new_tokens.access_token).with_model(&self.model);
            if let Some(ref url) = self.base_url {
                client = client.with_base_url(url);
            }
            if let Some(mt) = self.max_tokens {
                client = client.with_max_tokens(mt);
            }
            *self.inner.write().await = client;
        }
    }
}

#[async_trait::async_trait]
impl LLM for RefreshingAnthropicClient {
    fn model_name(&self) -> &str {
        &self.model
    }

    async fn chat_complete(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<ToolSpec>>,
    ) -> Result<LLMResponse, LLMError> {
        self.ensure_fresh().await;
        self.inner.read().await.chat_complete(messages, tools).await
    }

    async fn chat_complete_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<ToolSpec>>,
        tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<LLMResponse, LLMError> {
        self.ensure_fresh().await;
        self.inner
            .read()
            .await
            .chat_complete_stream(messages, tools, tx)
            .await
    }
}

/// Create an LLM client from resolved configuration.
async fn create_llm_from_config(
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
            if let Some(key) = api_key {
                let mut client = AnthropicClient::new(key).with_model(&config.model);
                if let Some(ref base_url) = config.base_url {
                    client = client.with_base_url(base_url);
                }
                if let Some(mt) = config.max_tokens {
                    client = client.with_max_tokens(mt);
                }
                Arc::new(client)
            } else if let Some(tokens) = auth::load_tokens() {
                // Use RefreshingAnthropicClient for daemon longevity
                Arc::new(RefreshingAnthropicClient::new(
                    tokens,
                    config.model.clone(),
                    config.base_url.clone(),
                    config.max_tokens,
                ))
            } else {
                return Err("No Anthropic API key or subscription login found. Run `anima login` or set ANTHROPIC_API_KEY.".into());
            }
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

/// Worker task that owns the agent and processes work items sequentially.
/// This ensures no race conditions between Message, Notify, and Heartbeat handlers.
#[allow(clippy::too_many_arguments)]
async fn agent_worker(
    mut work_rx: mpsc::UnboundedReceiver<AgentWork>,
    agent: Arc<Mutex<Agent>>,
    agent_name: String,
    system_prompt: Option<String>,
    recall: Option<String>,
    model_recall: Option<String>,
    allowed_tools: Option<Vec<String>>,
    semantic_memory: Option<Arc<Mutex<SemanticMemoryStore>>>,
    embedding_client: Option<Arc<EmbeddingClient>>,
    tool_registry: Option<Arc<ToolRegistry>>,
    use_native_tools: bool,
    logger: Arc<AgentLogger>,
    recall_limit: usize,
    history_limit: usize,
    task_store: Option<Arc<Mutex<TaskStore>>>,
    heartbeat_config: Option<HeartbeatDaemonConfig>,
    num_ctx: Option<u32>,
    max_iterations: Option<usize>,
    checkpoint_interval: Option<usize>,
    max_checkpoints: Option<usize>,
) {
    logger.log("[worker] Agent worker started");

    while let Some(work) = work_rx.recv().await {
        // Clear agent history before processing ANY work item
        // This prevents context bleed between different requests
        {
            let mut agent_guard = agent.lock().await;
            agent_guard.clear_history();
        }

        match work {
            AgentWork::Message {
                content,
                conv_name,
                response_tx,
                token_tx,
            } => {
                logger.log(&format!("[worker] Processing message: {}", content));
                let result = process_message_work(
                    &content,
                    conv_name.as_deref(),
                    token_tx,
                    &agent,
                    &agent_name,
                    &system_prompt,
                    &recall,
                    &model_recall,
                    &allowed_tools,
                    &semantic_memory,
                    &embedding_client,
                    &tool_registry,
                    use_native_tools,
                    &logger,
                    recall_limit,
                    history_limit,
                    &task_store,
                )
                .await;
                let _ = response_tx.send(result);
            }
            AgentWork::Notify {
                conv_id,
                message_id,
                depth,
            } => {
                logger.log(&format!(
                    "[worker] Processing notify: conv={} msg_id={} depth={}",
                    conv_id, message_id, depth
                ));
                let _response = handle_notify(
                    &conv_id,
                    message_id,
                    depth,
                    &agent,
                    &agent_name,
                    &system_prompt,
                    &recall,
                    &model_recall,
                    &allowed_tools,
                    &semantic_memory,
                    &embedding_client,
                    &tool_registry,
                    use_native_tools,
                    &logger,
                    recall_limit,
                    history_limit,
                    &task_store,
                    num_ctx,
                    max_iterations,
                    checkpoint_interval,
                    max_checkpoints,
                )
                .await;
            }
            AgentWork::Heartbeat => {
                logger.log("[worker] Processing heartbeat");
                if let Some(ref hb_config) = heartbeat_config {
                    run_heartbeat(
                        hb_config,
                        &agent,
                        &agent_name,
                        &system_prompt,
                        &recall,
                        &model_recall,
                        &allowed_tools,
                        &semantic_memory,
                        &embedding_client,
                        &tool_registry,
                        use_native_tools,
                        &logger,
                        recall_limit,
                        history_limit,
                    )
                    .await;
                } else {
                    logger.log("[worker] Heartbeat work received but no heartbeat config");
                }
            }
        }
    }

    logger.log("[worker] Agent worker stopped");
}

/// Process a Message work item: handle memory/tools injection, streaming, and tool execution.
/// Returns the final response (or error) for the oneshot channel.
#[allow(clippy::too_many_arguments)]
async fn process_message_work(
    content: &str,
    conv_name: Option<&str>,
    token_tx: Option<mpsc::Sender<String>>,
    agent: &Arc<Mutex<Agent>>,
    agent_name: &str,
    system_prompt: &Option<String>,
    recall: &Option<String>,
    model_recall: &Option<String>,
    allowed_tools: &Option<Vec<String>>,
    semantic_memory: &Option<Arc<Mutex<SemanticMemoryStore>>>,
    embedding_client: &Option<Arc<EmbeddingClient>>,
    tool_registry: &Option<Arc<ToolRegistry>>,
    use_native_tools: bool,
    logger: &Arc<AgentLogger>,
    recall_limit: usize,
    history_limit: usize,
    task_store: &Option<Arc<Mutex<TaskStore>>>,
) -> MessageWorkResult {
    let start_time = std::time::Instant::now();

    // Set current conversation for debug file naming
    {
        let mut agent_guard = agent.lock().await;
        agent_guard.set_current_conversation(conv_name.map(|s| s.to_string()));
    }

    // Load conversation history first
    let (mut conversation_history, final_user_content): (Vec<ChatMessage>, String) =
        if let Some(cname) = conv_name {
            match ConversationStore::init() {
                Ok(store) => match store.get_messages(cname, Some(history_limit)) {
                    Ok(msgs) if !msgs.is_empty() => {
                        let (history, final_content) =
                            format_conversation_history(&msgs, agent_name);
                        logger.log(&format!(
                            "[worker] Loaded {} history messages for conversation {}",
                            history.len(),
                            cname
                        ));
                        (history, final_content)
                    }
                    Ok(_) => (Vec::new(), content.to_string()),
                    Err(e) => {
                        logger.log(&format!(
                            "[worker] Failed to load conversation history: {}",
                            e
                        ));
                        (Vec::new(), content.to_string())
                    }
                },
                Err(e) => {
                    logger.log(&format!(
                        "[worker] Failed to init conversation store for history: {}",
                        e
                    ));
                    (Vec::new(), content.to_string())
                }
            }
        } else {
            (Vec::new(), content.to_string())
        };

    // Build recall (tools + memories) based on user message
    let recall_result = build_recall_for_query(
        content,
        allowed_tools,
        semantic_memory,
        embedding_client,
        tool_registry,
        use_native_tools,
        recall,
        model_recall,
        recall_limit,
        logger,
    )
    .await;

    let recall_content_for_storage = recall_result.recall_content.clone();
    inject_recall_into_history(&recall_result.recall_content, &mut conversation_history);

    // Create tool execution context
    let tool_context = ToolExecutionContext {
        agent_name: agent_name.to_string(),
        task_store: task_store.clone(),
        conv_id: conv_name.map(|s| s.to_string()),
        semantic_memory_store: semantic_memory.clone(),
        embedding_client: embedding_client.clone(),
        allowed_tools: allowed_tools.clone(),
    };

    // Create channel for incremental tool trace persistence
    let (trace_tx, trace_rx) =
        tokio::sync::mpsc::channel::<crate::agent::ToolExecution>(32);
    let trace_handle = spawn_tool_trace_persister(
        conv_name.map(|s| s.to_string()),
        agent_name.to_string(),
        logger.clone(),
        trace_rx,
    );

    // Process with or without streaming
    let (final_response, _tool_calls) = if use_native_tools {
        // Native tool mode with optional streaming
        let result = process_native_tool_mode(
            &final_user_content,
            recall_result.external_tools,
            token_tx,
            agent,
            system_prompt,
            semantic_memory,
            embedding_client,
            logger,
            conversation_history,
            Some(trace_tx.clone()),
        )
        .await;
        (result.response, result.tool_calls)
    } else {
        // JSON-block mode with optional streaming (no native tool_calls)
        let response = process_json_block_mode(
            &final_user_content,
            &recall_result.relevant_tools,
            token_tx,
            agent,
            system_prompt,
            semantic_memory,
            embedding_client,
            tool_registry,
            logger,
            &tool_context,
            conversation_history,
            None, // checkpoint_interval — not used for direct messages
            None, // max_checkpoints
        )
        .await;
        (response, None)
    };

    // Wait for incremental tool trace storage to complete
    drop(trace_tx);
    let _ = trace_handle.await;

    // Store response in conversation if conv_name was provided
    if let Some(cname) = conv_name {
        let duration_ms = start_time.elapsed().as_millis() as i64;
        match ConversationStore::init() {
            Ok(store) => {
                // Store recall AFTER response for persistence (but recall was already injected in memory)
                // This positions recall in DB right before the response, preserving context for future turns
                if let Some(ref recall_text) = recall_content_for_storage {
                    if !recall_text.is_empty() {
                        if let Err(e) = store.add_message(cname, "recall", recall_text, &[]) {
                            logger.log(&format!(
                                "[worker] Failed to store recall in conversation: {}",
                                e
                            ));
                        }
                    }
                }
                // Store the final response (no tool_calls since this is the terminal response)
                if let Err(e) = store.add_message_with_tool_calls(
                    cname,
                    agent_name,
                    &final_response,
                    &[],
                    Some(duration_ms),
                    None,
                ) {
                    logger.log(&format!(
                        "[worker] Failed to store response in conversation: {}",
                        e
                    ));
                }
            }
            Err(e) => {
                logger.log(&format!(
                    "[worker] Failed to init conversation store: {}",
                    e
                ));
            }
        }
    }

    MessageWorkResult {
        response: final_response,
        error: None,
    }
}

/// Process message in native tool mode (tools = true in model config)
#[allow(clippy::too_many_arguments)]
async fn process_native_tool_mode(
    content: &str,
    external_tools: Option<Vec<ToolSpec>>,
    token_tx: Option<mpsc::Sender<String>>,
    agent: &Arc<Mutex<Agent>>,
    system_prompt: &Option<String>,
    semantic_memory: &Option<Arc<Mutex<SemanticMemoryStore>>>,
    embedding_client: &Option<Arc<EmbeddingClient>>,
    logger: &Arc<AgentLogger>,
    conversation_history: Vec<ChatMessage>,
    tool_trace_tx: Option<tokio::sync::mpsc::Sender<crate::agent::ToolExecution>>,
) -> NativeToolModeResult {
    let options = ThinkOptions {
        system_prompt: system_prompt.clone(),
        conversation_history: if conversation_history.is_empty() {
            None
        } else {
            Some(conversation_history)
        },
        external_tools,
        tool_trace_tx,
        ..Default::default()
    };

    let (result, tool_calls) = if let Some(tx) = token_tx {
        // Streaming mode - now returns ThinkResult with tool_calls
        let agent_clone = agent.clone();
        let content_clone = content.to_string();

        let handle = tokio::spawn(async move {
            let mut agent_guard = agent_clone.lock().await;
            agent_guard
                .think_streaming_with_options(&content_clone, options, tx)
                .await
        });

        match handle.await {
            Ok(Ok(result)) => (result.response, result.last_tool_calls),
            Ok(Err(e)) => (format!("Error: {}", e), None),
            Err(e) => (format!("Error: task panicked: {}", e), None),
        }
    } else {
        // Non-streaming mode - can capture tool_calls
        let mut agent_guard = agent.lock().await;
        match agent_guard.think_with_options(content, options).await {
            Ok(result) => (result.response, result.last_tool_calls),
            Err(e) => (format!("Error: {}", e), None),
        }
    };

    // Strip thinking tags and extract [REMEMBER: ...] tags
    let without_thinking = strip_thinking_tags(&result);
    let (after_remember, memories_to_save) = extract_remember_tags(&without_thinking);

    // Save memories
    save_memories(&memories_to_save, semantic_memory, embedding_client, logger).await;

    NativeToolModeResult {
        response: after_remember,
        tool_calls,
    }
}

/// Process message in JSON-block tool mode (tools = false in model config)
#[allow(clippy::too_many_arguments)]
async fn process_json_block_mode(
    content: &str,
    _relevant_tools: &[ToolDefinition],
    token_tx: Option<mpsc::Sender<String>>,
    agent: &Arc<Mutex<Agent>>,
    system_prompt: &Option<String>,
    semantic_memory: &Option<Arc<Mutex<SemanticMemoryStore>>>,
    embedding_client: &Option<Arc<EmbeddingClient>>,
    tool_registry: &Option<Arc<ToolRegistry>>,
    logger: &Arc<AgentLogger>,
    tool_context: &ToolExecutionContext,
    conversation_history: Vec<ChatMessage>,
    checkpoint_interval: Option<usize>,
    max_checkpoints: Option<usize>,
) -> String {
    let mut current_message = content.to_string();
    let max_tool_calls = checkpoint_interval.unwrap_or(10);
    let use_checkpoints = checkpoint_interval.is_some();
    let json_max_checkpoints = max_checkpoints.unwrap_or(5);
    let mut json_checkpoint_count: usize = 0;
    let mut checkpoint_trace: Vec<crate::agent::CheckpointTraceEntry> = Vec::new();
    let mut tool_call_count = 0;

    loop {
        let options = ThinkOptions {
            system_prompt: system_prompt.clone(),
            conversation_history: if conversation_history.is_empty() {
                None
            } else {
                Some(conversation_history.clone())
            },
            external_tools: None,
            ..Default::default()
        };

        let llm_response = if let Some(ref tx) = token_tx {
            // Streaming mode - but suppress tool call blocks
            let (internal_tx, mut internal_rx) = mpsc::channel::<String>(100);
            let agent_clone = agent.clone();
            let current_message_clone = current_message.clone();

            let handle = tokio::spawn(async move {
                let mut agent_guard = agent_clone.lock().await;
                agent_guard
                    .think_streaming_with_options(&current_message_clone, options, internal_tx)
                    .await
            });

            // Forward tokens, suppressing tool call blocks
            let tx_clone = tx.clone();
            let mut in_code_block = false;
            let mut code_block_buffer = String::new();

            while let Some(token) = internal_rx.recv().await {
                if !in_code_block {
                    if token.contains("```") {
                        in_code_block = true;
                        code_block_buffer = token;
                        if code_block_buffer.matches("```").count() >= 2 {
                            if !(code_block_buffer.contains("\"tool\"")
                                && code_block_buffer.contains("\"params\""))
                            {
                                let _ = tx_clone.send(code_block_buffer.clone()).await;
                            }
                            in_code_block = false;
                            code_block_buffer.clear();
                        }
                        continue;
                    }
                    let _ = tx_clone.send(token).await;
                } else {
                    code_block_buffer.push_str(&token);
                    if code_block_buffer.matches("```").count() >= 2 {
                        if !(code_block_buffer.contains("\"tool\"")
                            && code_block_buffer.contains("\"params\""))
                        {
                            let _ = tx_clone.send(code_block_buffer.clone()).await;
                        }
                        in_code_block = false;
                        code_block_buffer.clear();
                    }
                }
            }

            if !(code_block_buffer.is_empty()
                || code_block_buffer.contains("\"tool\"")
                    && code_block_buffer.contains("\"params\""))
            {
                let _ = tx_clone.send(code_block_buffer).await;
            }

            match handle.await {
                Ok(Ok(result)) => result.response,
                Ok(Err(e)) => return format!("Error: {}", e),
                Err(e) => return format!("Error: task panicked: {}", e),
            }
        } else {
            // Non-streaming mode
            let mut agent_guard = agent.lock().await;
            match agent_guard
                .think_with_options(&current_message, options)
                .await
            {
                Ok(result) => result.response,
                Err(e) => return format!("Error: {}", e),
            }
        };

        // Strip thinking tags and extract [REMEMBER: ...] tags
        let without_thinking = strip_thinking_tags(&llm_response);
        let (after_remember, memories_to_save) = extract_remember_tags(&without_thinking);

        // Save memories
        save_memories(&memories_to_save, semantic_memory, embedding_client, logger).await;

        // Check for tool calls
        let (cleaned_response, tool_call) = extract_tool_call(&after_remember);

        if let Some(tc) = tool_call {
            tool_call_count += 1;

            // Check checkpoint boundary
            if use_checkpoints && tool_call_count >= max_tool_calls {
                json_checkpoint_count += 1;
                if json_checkpoint_count > json_max_checkpoints {
                    logger.tool("[worker] Max checkpoints exhausted, stopping");
                    return cleaned_response;
                }
                let summary = crate::agent::build_checkpoint_summary(
                    &checkpoint_trace, json_checkpoint_count,
                );
                logger.log(&format!(
                    "[worker] Checkpoint {} reached ({} tool calls), restarting with summary",
                    json_checkpoint_count, tool_call_count
                ));
                tool_call_count = 0;
                current_message = summary;
                continue;
            } else if !use_checkpoints && tool_call_count > max_tool_calls {
                logger.tool("[worker] Max tool calls reached, stopping");
                return cleaned_response;
            }

            logger.tool(&format!(
                "[worker] Executing: {} with params {}",
                tc.tool, tc.params
            ));

            let tool_def = tool_registry
                .as_ref()
                .and_then(|r| r.find_by_name(&tc.tool));

            match execute_tool_call(&tc, tool_def, Some(tool_context)).await {
                Ok(tool_result) => {
                    logger.tool(&format!("[worker] Result: {} bytes", tool_result.len()));
                    checkpoint_trace.push(crate::agent::CheckpointTraceEntry {
                        tool_name: tc.tool.clone(),
                        params_summary: crate::agent::summarize_tool_params(&tc.tool, &tc.params),
                        result_summary: crate::agent::truncate(&tool_result, 100),
                        content: None,
                    });
                    current_message = format!("[Tool Result for {}]\n{}", tc.tool, tool_result);
                }
                Err(e) => {
                    logger.tool(&format!("[worker] Error: {}", e));
                    checkpoint_trace.push(crate::agent::CheckpointTraceEntry {
                        tool_name: tc.tool.clone(),
                        params_summary: crate::agent::summarize_tool_params(&tc.tool, &tc.params),
                        result_summary: format!("ERROR: {}", crate::agent::truncate(&e.to_string(), 80)),
                        content: None,
                    });
                    current_message = format!("[Tool Error for {}]\n{}", tc.tool, e);
                }
            }

            if let Some(nudge) = crate::agent::tool_budget_nudge(tool_call_count, max_tool_calls) {
                current_message.push_str(&format!("\n\n---\n{}", nudge));
            }
        } else {
            return cleaned_response;
        }
    }
}

/// Maximum depth for @mention chains to prevent infinite loops
const MAX_MENTION_DEPTH: u32 = 100;

/// Handle a Notify request: fetch conversation context, generate response, store it,
/// and forward @mentions to other agents (daemon-to-daemon).
#[allow(clippy::too_many_arguments)]
async fn handle_notify(
    conv_id: &str,
    _message_id: i64,
    depth: u32,
    agent: &Arc<Mutex<Agent>>,
    agent_name: &str,
    system_prompt: &Option<String>,
    recall: &Option<String>,
    model_recall: &Option<String>,
    allowed_tools: &Option<Vec<String>>,
    semantic_memory: &Option<Arc<Mutex<SemanticMemoryStore>>>,
    embedding_client: &Option<Arc<EmbeddingClient>>,
    tool_registry: &Option<Arc<ToolRegistry>>,
    use_native_tools: bool,
    logger: &Arc<AgentLogger>,
    recall_limit: usize,
    history_limit: usize,
    task_store: &Option<Arc<Mutex<TaskStore>>>,
    num_ctx: Option<u32>,
    max_iterations: Option<usize>,
    checkpoint_interval: Option<usize>,
    max_checkpoints: Option<usize>,
) -> Response {
    // Track start time for response duration
    let start_time = std::time::Instant::now();

    // Set current conversation for debug file naming
    {
        let mut agent_guard = agent.lock().await;
        agent_guard.set_current_conversation(Some(conv_id.to_string()));
    }

    // Open conversation store
    let store = match ConversationStore::init() {
        Ok(s) => s,
        Err(e) => {
            logger.log(&format!(
                "[notify] Failed to open conversation store: {}",
                e
            ));
            return Response::Error {
                message: format!("Failed to open conversation store: {}", e),
            };
        }
    };

    // First fetch: get messages to extract final_user_content for recall query
    let context_messages = match store.get_messages(conv_id, Some(history_limit)) {
        Ok(msgs) => msgs,
        Err(e) => {
            logger.log(&format!("[notify] Failed to get messages: {}", e));
            return Response::Error {
                message: format!("Failed to get messages: {}", e),
            };
        }
    };

    if context_messages.is_empty() {
        logger.log("[notify] No messages in conversation");
        return Response::Error {
            message: "No messages in conversation".to_string(),
        };
    }

    // Extract final_user_content for building recall
    let (mut conversation_history, final_user_content) =
        format_conversation_history(&context_messages, agent_name);

    // Build recall (tools + memories) based on current user message
    let recall_result = build_recall_for_query(
        &final_user_content,
        allowed_tools,
        semantic_memory,
        embedding_client,
        tool_registry,
        use_native_tools,
        recall,
        model_recall,
        recall_limit,
        logger,
    )
    .await;

    let recall_content_for_storage = recall_result.recall_content.clone();
    inject_recall_into_history(&recall_result.recall_content, &mut conversation_history);

    logger.log(&format!(
        "[notify] Context: {} messages → {} history + final user turn",
        context_messages.len(),
        conversation_history.len()
    ));

    // Create tool execution context for tools that need daemon state
    let tool_context = ToolExecutionContext {
        agent_name: agent_name.to_string(),
        task_store: task_store.clone(),
        conv_id: Some(conv_id.to_string()),
        semantic_memory_store: semantic_memory.clone(),
        embedding_client: embedding_client.clone(),
        allowed_tools: allowed_tools.clone(),
    };

    let external_tools = recall_result.external_tools;

    let json_block_budget = checkpoint_interval.unwrap_or(max_iterations.unwrap_or(10));
    let mut tool_call_count = 0;
    let mut current_message = final_user_content.clone();
    #[allow(unused_assignments)]
    let mut final_response: Option<String> = None;
    let mut last_tool_calls: Option<Vec<crate::llm::ToolCall>> = None;
    let mut total_tokens_in: u32 = 0;
    let mut total_tokens_out: u32 = 0;
    let mut checkpoint_trace: Vec<crate::agent::CheckpointTraceEntry> = Vec::new();
    let mut json_checkpoint_count: usize = 0;
    let json_max_checkpoints = max_checkpoints.unwrap_or(5);

    // Channel for incremental tool trace persistence (native tool mode)
    let (tool_trace_tx, trace_rx) =
        tokio::sync::mpsc::channel::<crate::agent::ToolExecution>(32);
    let trace_handle = spawn_tool_trace_persister(
        Some(conv_id.to_string()),
        agent_name.to_string(),
        logger.clone(),
        trace_rx,
    );

    // Cancellation flag for native tool mode — checked by agent.rs at each iteration boundary
    let cancel_flag = Arc::new(AtomicBool::new(false));

    // Spawn a watcher that polls is_paused every 500ms and sets the cancel flag
    let cancel_for_watcher = cancel_flag.clone();
    let conv_id_for_watcher = conv_id.to_string();
    let pause_watcher = tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(500)).await;
            if let Ok(store) = ConversationStore::init() {
                if store.is_paused(&conv_id_for_watcher).unwrap_or(false) {
                    cancel_for_watcher.store(true, Ordering::Relaxed);
                    break;
                }
            }
        }
    });

    loop {
        // Clear agent's internal history EACH iteration to avoid duplication.
        // think_with_options adds to self.history, and agent.rs injects self.history
        // BEFORE options.conversation_history (lines 655-660). Without clearing each
        // iteration, the growing internal history appears before DB-backed history.
        agent.lock().await.clear_history();

        // No fresh recall injection - recall was injected in the initial conversation_history
        // on first iteration. After tool execution, we refresh from DB which won't include
        // recall (stored after response), but that's fine - LLM has already seen it.
        let options = ThinkOptions {
            system_prompt: system_prompt.clone(),
            conversation_history: if conversation_history.is_empty() {
                None
            } else {
                Some(conversation_history.clone())
            },
            external_tools: external_tools.clone(),
            max_iterations: max_iterations.unwrap_or(10),
            tool_trace_tx: Some(tool_trace_tx.clone()),
            cancel: Some(cancel_flag.clone()),
            checkpoint_interval,
            max_checkpoints: max_checkpoints.unwrap_or(5),
            ..Default::default()
        };

        // Generate response (use streaming to avoid Ollama non-streaming issues)
        let (token_tx, mut token_rx) = mpsc::channel::<String>(100);
        let drain_handle = tokio::spawn(async move {
            while token_rx.recv().await.is_some() {}
        });

        let mut agent_guard = agent.lock().await;
        let result = agent_guard
            .think_streaming_with_options(&current_message, options, token_tx)
            .await;
        drop(agent_guard); // Release lock before potential tool execution

        // Drain completes when token_tx is dropped
        let _ = drain_handle.await;

        match result {
            Ok(think_result) => {
                // Capture tool_calls from native tool mode for persistence
                if think_result.last_tool_calls.is_some() {
                    last_tool_calls = think_result.last_tool_calls.clone();
                }

                // Accumulate token usage
                if let Some(tokens) = think_result.tokens_in {
                    total_tokens_in += tokens;
                }
                if let Some(tokens) = think_result.tokens_out {
                    total_tokens_out += tokens;
                }

                // Strip thinking tags and extract [REMEMBER: ...] tags
                let without_thinking = strip_thinking_tags(&think_result.response);
                let (after_remember, memories_to_save) = extract_remember_tags(&without_thinking);

                // Save memories
                save_memories(&memories_to_save, semantic_memory, embedding_client, logger).await;

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

                    // Check if we've hit the checkpoint boundary (JSON-block mode)
                    if checkpoint_interval.is_some() && tool_call_count >= json_block_budget {
                        json_checkpoint_count += 1;
                        if json_checkpoint_count > json_max_checkpoints {
                            logger.tool("[notify] Max checkpoints exhausted, stopping");
                            final_response = Some(after_remember.clone());
                            break;
                        }
                        // Build checkpoint summary and reset counter
                        let summary = crate::agent::build_checkpoint_summary(
                            &checkpoint_trace, json_checkpoint_count,
                        );
                        logger.log(&format!(
                            "[notify] Checkpoint {} reached ({} tool calls), restarting with summary",
                            json_checkpoint_count, tool_call_count
                        ));
                        tool_call_count = 0;
                        current_message = summary;
                        // Refresh conversation_history from DB for fresh context
                        if let Ok(msgs) = store.get_messages(conv_id, Some(history_limit)) {
                            let (refreshed_history, _) =
                                format_conversation_history(&msgs, agent_name);
                            conversation_history = refreshed_history;
                        }
                        continue;
                    } else if checkpoint_interval.is_none() && tool_call_count > json_block_budget {
                        logger.tool("[notify] Max tool calls reached, stopping");
                        final_response = Some(after_remember.clone());
                        break;
                    }

                    // Store this assistant response before executing the tool
                    // Use after_remember (not cleaned_response) to preserve the tool call JSON block
                    // so the model can see what it called in conversation history
                    if !after_remember.trim().is_empty() {
                        if let Err(e) = store.add_message(conv_id, agent_name, &after_remember, &[])
                        {
                            logger.log(&format!(
                                "[notify] Failed to store intermediate response: {}",
                                e
                            ));
                        } else {
                            logger.log(&format!(
                                "[notify] Stored intermediate response: {} bytes",
                                after_remember.len()
                            ));
                        }
                    }

                    logger.tool(&format!(
                        "[notify] Executing: {} with params {}",
                        tc.tool, tc.params
                    ));

                    // Look up the tool definition to get allowed_commands
                    let tool_def = tool_registry
                        .as_ref()
                        .and_then(|r| r.find_by_name(&tc.tool));

                    // Execute tool and format result message
                    let tool_message = match execute_tool_call(&tc, tool_def, Some(&tool_context)).await {
                        Ok(tool_result) => {
                            logger.tool(&format!("[notify] Result: {} bytes", tool_result.len()));
                            // Track for checkpoint summary
                            checkpoint_trace.push(crate::agent::CheckpointTraceEntry {
                                tool_name: tc.tool.clone(),
                                params_summary: crate::agent::summarize_tool_params(&tc.tool, &tc.params),
                                result_summary: crate::agent::truncate(&tool_result, 100),
                                content: None,
                            });
                            format!("[Tool Result for {}]\n{}", tc.tool, tool_result)
                        }
                        Err(e) => {
                            logger.tool(&format!("[notify] Error: {}", e));
                            checkpoint_trace.push(crate::agent::CheckpointTraceEntry {
                                tool_name: tc.tool.clone(),
                                params_summary: crate::agent::summarize_tool_params(&tc.tool, &tc.params),
                                result_summary: format!("ERROR: {}", crate::agent::truncate(&e.to_string(), 80)),
                                content: None,
                            });
                            format!("[Tool Error for {}]\n{}", tc.tool, e)
                        }
                    };

                    // Store tool result/error in conversation (attributed to this agent)
                    if let Err(e) = store.add_tool_result(conv_id, &tool_message, agent_name) {
                        logger.log(&format!("[notify] Failed to store tool message: {}", e));
                    }

                    // Refresh conversation_history from DB to include the tool result we just stored.
                    // This ensures we use ONLY DB-backed history and avoids duplication with agent's
                    // internal history (which we cleared at the start of each iteration).
                    // Also update current_message from refreshed_final to avoid passing
                    // the tool result twice (once in history, once as the task).
                    current_message = tool_message;
                    if let Ok(msgs) = store.get_messages(conv_id, Some(history_limit)) {
                        let (refreshed_history, refreshed_final) =
                            format_conversation_history(&msgs, agent_name);
                        conversation_history = refreshed_history;
                        current_message = refreshed_final;
                    }

                    if let Some(nudge) = crate::agent::tool_budget_nudge(tool_call_count, json_block_budget) {
                        current_message.push_str(&format!("\n\n---\n{}", nudge));
                    }
                } else {
                    // No more tool calls - done
                    final_response = Some(cleaned_response);
                    break;
                }
            }
            Err(crate::error::AgentError::Cancelled) => {
                logger.log("[notify] Agent cancelled (conversation paused)");
                final_response = Some("[Paused]".to_string());
                break;
            }
            Err(e) => {
                logger.log(&format!("[notify] Agent error: {}", e));
                pause_watcher.abort();
                return Response::Error {
                    message: e.to_string(),
                };
            }
        }
    }

    // Clean up pause watcher
    pause_watcher.abort();

    // Store final response in conversation with duration, tool_calls, and token usage
    let cleaned_response = final_response.unwrap_or_default();
    let duration_ms = start_time.elapsed().as_millis() as i64;
    let tool_calls_json = last_tool_calls
        .as_ref()
        .and_then(|tc| serde_json::to_string(tc).ok());
    let tokens_in = (total_tokens_in > 0).then_some(total_tokens_in as i64);
    let tokens_out = (total_tokens_out > 0).then_some(total_tokens_out as i64);
    let num_ctx_i64 = num_ctx.map(|n| n as i64);

    // Flush tool trace channel — drop sender so consumer finishes
    drop(tool_trace_tx);
    let _ = trace_handle.await;

    // Store recall AFTER response for persistence (but recall was already injected in memory)
    // This positions recall in DB right before the response, preserving context for future turns
    if let Some(ref recall_text) = recall_content_for_storage {
        if !recall_text.is_empty() {
            if let Err(e) = store.add_message(conv_id, "recall", recall_text, &[]) {
                logger.log(&format!(
                    "[notify] Failed to store recall in conversation: {}",
                    e
                ));
            }
        }
    }

    match store.add_message_with_tokens(
        conv_id,
        agent_name,
        &cleaned_response,
        &[],
        Some(duration_ms),
        tool_calls_json.as_deref(),
        tokens_in,
        tokens_out,
        num_ctx_i64,
    ) {
        Ok(response_msg_id) => {
            logger.log(&format!(
                "[notify] Stored response as msg_id={} ({}ms)",
                response_msg_id, duration_ms
            ));

            // Parse @mentions from our response and forward to other agents
            // Only forward if not paused and not at depth limit
            let should_forward =
                depth < MAX_MENTION_DEPTH && !store.is_paused(conv_id).unwrap_or(false);

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
                    logger.log(&format!(
                        "[notify] Forwarding to {} agents at depth {}: {:?}",
                        valid_mentions.len(),
                        depth + 1,
                        valid_mentions
                    ));

                    // Add mentioned agents as participants if not already
                    for mention in &valid_mentions {
                        if let Err(e) = store.add_participant(conv_id, mention) {
                            logger.log(&format!(
                                "[notify] Warning: Could not add {} as participant: {}",
                                mention, e
                            ));
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
                        )
                        .await;
                    }
                }
            } else if depth >= MAX_MENTION_DEPTH {
                logger.log(&format!(
                    "[notify] Skipping forwarding: depth limit reached ({})",
                    depth
                ));
            }

            Response::Notified {
                response_message_id: response_msg_id,
            }
        }
        Err(e) => {
            logger.log(&format!("[notify] Failed to store response: {}", e));
            Response::Error {
                message: format!("Failed to store response: {}", e),
            }
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
                log.log(&format!(
                    "[notify] Failed to queue notification for @{}: {}",
                    agent, e
                ));
            }
        } else {
            log.log(&format!(
                "[notify] Failed to open store to queue notification for @{}",
                agent
            ));
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
                    logger.log(&format!(
                        "[notify] Failed to send to @{}: {}",
                        agent_name, e
                    ));
                    // Queue notification for later
                    queue_notification(agent_name, conv_id, message_id, logger);
                    return;
                }

                // Read response (but don't wait too long)
                match tokio::time::timeout(
                    std::time::Duration::from_secs(300), // 5 min timeout for agent response
                    api.read_response(),
                )
                .await
                {
                    Ok(Ok(Some(Response::Notified {
                        response_message_id,
                    }))) => {
                        logger.log(&format!(
                            "[notify] @{} responded with msg_id={}",
                            agent_name, response_message_id
                        ));
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
                        logger.log(&format!(
                            "[notify] @{} timeout waiting for response",
                            agent_name
                        ));
                    }
                }
            }
            Err(e) => {
                logger.log(&format!(
                    "[notify] @{} not reachable ({}), queuing",
                    agent_name, e
                ));
                queue_notification(agent_name, conv_id, message_id, logger);
            }
        }
    } else {
        // Agent not running - queue notification
        logger.log(&format!(
            "[notify] @{} not running, queuing notification",
            agent_name
        ));
        queue_notification(agent_name, conv_id, message_id, logger);
    }
}

/// Execute a heartbeat: load heartbeat.md, think, store response in <agent>-heartbeat conversation.
#[allow(clippy::too_many_arguments)]
async fn run_heartbeat(
    config: &HeartbeatDaemonConfig,
    agent: &Arc<Mutex<Agent>>,
    agent_name: &str,
    system_prompt: &Option<String>,
    recall: &Option<String>,
    model_recall: &Option<String>,
    allowed_tools: &Option<Vec<String>>,
    semantic_memory: &Option<Arc<Mutex<SemanticMemoryStore>>>,
    embedding_client: &Option<Arc<EmbeddingClient>>,
    tool_registry: &Option<Arc<ToolRegistry>>,
    use_native_tools: bool,
    logger: &Arc<AgentLogger>,
    recall_limit: usize,
    history_limit: usize,
) {
    // Set current conversation for debug file naming
    {
        let mut agent_guard = agent.lock().await;
        agent_guard.set_current_conversation(Some("heartbeat".to_string()));
    }

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

    logger.log(&format!(
        "[heartbeat] Running with prompt: {} chars",
        heartbeat_prompt.len()
    ));

    // 2. Get or create <agent>-heartbeat conversation
    let conv_name = format!("{}-heartbeat", agent_name);
    let store = match ConversationStore::init() {
        Ok(s) => s,
        Err(e) => {
            logger.log(&format!(
                "[heartbeat] Failed to open conversation store: {}",
                e
            ));
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
    let context_messages = store
        .get_messages(&conv_name, Some(history_limit))
        .unwrap_or_default();
    let (conversation_history, _) = format_conversation_history(&context_messages, agent_name);

    logger.log(&format!(
        "[heartbeat] Context: {} previous outputs",
        conversation_history.len()
    ));

    // 4. Build recall (tools + memories) for the heartbeat prompt
    let recall_result = build_recall_for_query(
        &heartbeat_prompt,
        allowed_tools,
        semantic_memory,
        embedding_client,
        tool_registry,
        use_native_tools,
        recall,
        model_recall,
        recall_limit,
        logger,
    )
    .await;

    let recall_content = recall_result.recall_content;

    // 5. Think - heartbeat_prompt is the user message, previous outputs are conversation_history
    let options = ThinkOptions {
        system_prompt: system_prompt.clone(),
        conversation_history: if conversation_history.is_empty() {
            None
        } else {
            Some(conversation_history)
        },
        external_tools: recall_result.external_tools,
        ..Default::default()
    };

    let mut agent_guard = agent.lock().await;
    let result = match agent_guard
        .think_with_options(&heartbeat_prompt, options)
        .await
    {
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
    save_memories(&memories_to_save, semantic_memory, embedding_client, logger).await;

    // 7. Store response in <agent>-heartbeat conversation (just the response, not the prompt)
    match store.add_message(&conv_name, agent_name, &cleaned_response, &[]) {
        Ok(msg_id) => {
            logger.log(&format!("[heartbeat] Stored response as msg_id={}", msg_id));
        }
        Err(e) => {
            logger.log(&format!("[heartbeat] Failed to store response: {}", e));
        }
    }

    // Store recall AFTER response so it appears before next user message in history
    if let Some(ref recall_text) = recall_content
        && !recall_text.is_empty()
    {
        if let Err(e) = store.add_message(&conv_name, "recall", recall_text, &[]) {
            logger.log(&format!(
                "[heartbeat] Failed to store recall in conversation: {}",
                e
            ));
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
        logger.log(&format!(
            "[heartbeat] Notifying {} agents: {:?}",
            valid_mentions.len(),
            valid_mentions
        ));

        // Get the message ID we just stored
        if let Ok(msgs) = store.get_messages(&conv_name, Some(1))
            && let Some(last_msg) = msgs.last()
        {
            for mention in valid_mentions {
                // Add mentioned agent as participant
                let _ = store.add_participant(&conv_name, &mention);

                forward_notify_to_agent(
                    &mention,
                    &conv_name,
                    last_msg.id,
                    0, // Start at depth 0
                    logger,
                )
                .await;
            }
        }
    }

    logger.log(&format!(
        "[heartbeat] Complete. Response: {} chars",
        cleaned_response.len()
    ));
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
                    if let Some(pid) = task.pid
                        && !is_process_running(pid) {
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
            logger.log(&format!(
                "[task-watcher] Error reading log {}: {}",
                log_path, e
            ));
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
    let exit_code = if log_content.contains("exit code: 0") || log_content.contains("exited with 0")
    {
        0
    } else if log_content.contains("exit code: 1") || log_content.contains("exited with 1") {
        1
    } else if log_content.contains("error")
        || log_content.contains("Error")
        || log_content.contains("ERROR")
    {
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
            logger.log(&format!(
                "[task-watcher] Task {} has no conv_id, cannot notify",
                task_id
            ));
            return;
        }
    };

    // Open conversation store
    let store = match ConversationStore::init() {
        Ok(s) => s,
        Err(e) => {
            logger.log(&format!(
                "[task-watcher] Failed to open conversation store: {}",
                e
            ));
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
            logger.log(&format!(
                "[task-watcher] Stored notification as msg_id={}",
                msg_id
            ));

            // Forward the notification to the agent daemon if it's running
            forward_notify_to_agent(agent_name, &conv_name, msg_id, 0, logger).await;

            logger.log(&format!(
                "[task-watcher] Notified @{} about task {}",
                agent_name, task_id
            ));
        }
        Err(e) => {
            logger.log(&format!(
                "[task-watcher] Failed to store notification: {}",
                e
            ));
        }
    }
}

/// Handle a single connection from a client.
#[allow(clippy::too_many_arguments)]
async fn handle_connection(
    mut api: SocketApi,
    agent: Arc<Mutex<Agent>>,
    _agent_name: String,
    system_prompt: Option<String>,
    recall: Option<String>,
    model_recall: Option<String>,
    allowed_tools: Option<Vec<String>>,
    semantic_memory: Option<Arc<Mutex<SemanticMemoryStore>>>,
    embedding_client: Option<Arc<EmbeddingClient>>,
    tool_registry: Option<Arc<ToolRegistry>>,
    use_native_tools: bool,
    shutdown: Arc<tokio::sync::Notify>,
    logger: Arc<AgentLogger>,
    recall_limit: usize,
    _history_limit: usize,
    heartbeat_config: Option<HeartbeatDaemonConfig>,
    _task_store: Option<Arc<Mutex<TaskStore>>>,
    work_tx: mpsc::UnboundedSender<AgentWork>,
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
            Request::Message {
                ref content,
                ref conv_name,
            } => {
                logger.log(&format!("[socket] Received message: {}", content));

                // Create streaming channel for tokens
                let (token_tx, mut token_rx) = mpsc::channel::<String>(100);

                // Create oneshot channel for final result
                let (response_tx, response_rx) = oneshot::channel();

                // Send work to the worker
                if work_tx
                    .send(AgentWork::Message {
                        content: content.clone(),
                        conv_name: conv_name.clone(),
                        response_tx,
                        token_tx: Some(token_tx),
                    })
                    .is_err()
                {
                    logger.log("[socket] Worker channel closed");
                    break;
                }

                // Forward tokens to socket as they arrive
                while let Some(token) = token_rx.recv().await {
                    if let Err(e) = api.write_response(&Response::Chunk { text: token }).await {
                        logger.log(&format!("[socket] Error writing chunk: {}", e));
                        break;
                    }
                }

                // Wait for final result
                match response_rx.await {
                    Ok(result) => {
                        // Send Done to signal stream complete
                        if let Err(e) = api.write_response(&Response::Done).await {
                            logger.log(&format!("[socket] Error writing Done: {}", e));
                        }
                        logger.log(&format!(
                            "[socket] Final response: {} bytes",
                            result.response.len()
                        ));
                    }
                    Err(_) => {
                        logger.log("[socket] Worker dropped response channel");
                        if let Err(e) = api.write_response(&Response::Done).await {
                            logger.log(&format!("[socket] Error writing Done: {}", e));
                        }
                    }
                }

                // Continue to next request (don't send Response::Message - already streamed)
                continue;
            }

            Request::IncomingMessage {
                ref from,
                ref content,
            } => {
                logger.log(&format!(
                    "[socket] Incoming message from {}: {}",
                    from, content
                ));

                // Format the message with [sender] prefix for the agent
                let formatted_message = format!("[{}] {}", from, content);

                // Build recall (tools + memories) for the incoming message
                let recall_result = build_recall_for_query(
                    content,
                    &allowed_tools,
                    &semantic_memory,
                    &embedding_client,
                    &tool_registry,
                    use_native_tools,
                    &recall,
                    &model_recall,
                    recall_limit,
                    &logger,
                )
                .await;

                // Note: recall injection is skipped for IncomingMessage - recall should be
                // stored in DB by the caller and included in conversation history when fetched.
                let _ = &recall_result.recall_content;

                let options = ThinkOptions {
                    system_prompt: system_prompt.clone(),
                    conversation_history: None,
                    external_tools: recall_result.external_tools,
                    ..Default::default()
                };

                let mut agent_guard = agent.lock().await;
                match agent_guard
                    .think_with_options(&formatted_message, options)
                    .await
                {
                    Ok(result) => {
                        let without_thinking = strip_thinking_tags(&result.response);
                        let (cleaned_response, memories_to_save) =
                            extract_remember_tags(&without_thinking);

                        save_memories(
                            &memories_to_save,
                            &semantic_memory,
                            &embedding_client,
                            &logger,
                        )
                        .await;

                        logger.log(&format!(
                            "[socket] Response to {}: {}",
                            from, cleaned_response
                        ));
                        Response::Message {
                            content: cleaned_response,
                        }
                    }
                    Err(e) => Response::Error {
                        message: e.to_string(),
                    },
                }
            }

            Request::Notify {
                ref conv_id,
                message_id,
                depth,
            } => {
                logger.log(&format!(
                    "[socket] Notify: conv={} msg_id={} depth={}",
                    conv_id, message_id, depth
                ));

                // Send immediate ack - fire-and-forget semantics
                if let Err(e) = api.write_response(&Response::NotifyReceived).await {
                    logger.log(&format!("[socket] Error writing NotifyReceived: {}", e));
                    continue;
                }

                // Send Notify work to the worker - serialized with other agent work
                if work_tx
                    .send(AgentWork::Notify {
                        conv_id: conv_id.clone(),
                        message_id,
                        depth,
                    })
                    .is_err()
                {
                    logger.log("[socket] Worker channel closed");
                }

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
                let system_content = system_prompt
                    .clone()
                    .unwrap_or_else(|| "(no system prompt configured)".to_string());
                Response::System {
                    system_prompt: system_content,
                }
            }

            Request::Heartbeat => {
                logger.log("[socket] Manual heartbeat requested");

                // Check if heartbeat is configured
                if let Some(ref hb_config) = heartbeat_config {
                    // Check if heartbeat.md exists
                    if !hb_config.heartbeat_path.exists() {
                        logger.log("[socket] heartbeat.md not found");
                        Response::Error {
                            message: "heartbeat.md not found".to_string(),
                        }
                    } else {
                        // Send heartbeat work to the worker - serialized with other agent work
                        if work_tx.send(AgentWork::Heartbeat).is_err() {
                            logger.log("[socket] Worker channel closed");
                            Response::Error {
                                message: "Worker channel closed".to_string(),
                            }
                        } else {
                            Response::HeartbeatTriggered
                        }
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
        assert_eq!(
            parse_duration("2h30m"),
            Some(Duration::from_secs(2 * 3600 + 30 * 60))
        );
        assert_eq!(
            parse_duration("1h30m15s"),
            Some(Duration::from_secs(3600 + 30 * 60 + 15))
        );
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
system_file = "system.md"

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
        std::fs::write(dir.path().join("system.md"), "Test system prompt").unwrap();

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
        let system_prompt = daemon_config.system_prompt.unwrap();
        assert!(system_prompt.starts_with("Test system prompt"));
        assert!(system_prompt.contains("You are running inside Anima, a multi-agent runtime."));
        assert!(system_prompt.contains("agent=test-agent"));
        assert!(system_prompt.contains("model=gpt-4"));
        assert!(system_prompt.contains("tools=native"));
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
        // Even without a system file, runtime context is injected
        let system_prompt = daemon_config.system_prompt.unwrap();
        assert!(system_prompt.contains("You are running inside Anima, a multi-agent runtime."));
        assert!(system_prompt.contains("agent=test-agent"));
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
    fn test_daemon_config_with_recall() {
        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test-agent"
system_file = "system.md"
recall_file = "recall.md"

[llm]
provider = "openai"
model = "gpt-4"
api_key = "sk-test"
"#;
        std::fs::write(dir.path().join("config.toml"), config_content).unwrap();
        std::fs::write(dir.path().join("system.md"), "Test system prompt").unwrap();
        std::fs::write(dir.path().join("recall.md"), "Always be concise.").unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        let daemon_config = DaemonConfig::from_agent_dir(&agent_dir).unwrap();

        // Persona should start with original content and include runtime context
        let system_prompt = daemon_config.system_prompt.unwrap();
        assert!(system_prompt.starts_with("Test system prompt"));
        assert!(system_prompt.contains("You are running inside Anima, a multi-agent runtime."));
        assert_eq!(daemon_config.recall, Some("Always be concise.".to_string()));
    }

    #[test]
    #[serial]
    fn test_daemon_config_recall_file_missing() {
        // Use a fake HOME so the global ~/.anima/agents/recall.md fallback doesn't interfere
        let fake_home = tempdir().unwrap();
        let original_home = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", fake_home.path()) };

        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test-agent"
recall_file = "recall.md"

[llm]
provider = "openai"
model = "gpt-4"
api_key = "sk-test"
"#;
        std::fs::write(dir.path().join("config.toml"), config_content).unwrap();
        // Note: recall.md file is NOT created

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        let daemon_config = DaemonConfig::from_agent_dir(&agent_dir).unwrap();

        // Restore HOME
        match original_home {
            Some(h) => unsafe { std::env::set_var("HOME", h) },
            None => unsafe { std::env::remove_var("HOME") },
        }

        // Should be None when file is missing (backward compatible)
        assert!(daemon_config.recall.is_none());
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
            allowed_commands: Some(vec![
                "ls".to_string(),
                "grep".to_string(),
                "cat".to_string(),
            ]),
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
            allowed_commands: Some(vec![
                "ls".to_string(),
                "grep".to_string(),
                "cat".to_string(),
            ]),
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
            allowed_commands: None, // No restrictions
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
            duration_ms: None,
            tool_calls: None,
            tokens_in: None,
            tokens_out: None,
            num_ctx: None,
            triggered_by: None,
        }
    }

    fn make_tool_msg(content: &str, triggered_by: Option<&str>) -> ConversationMessage {
        ConversationMessage {
            id: 1,
            conv_name: "test".to_string(),
            from_agent: "tool".to_string(),
            content: content.to_string(),
            mentions: vec![],
            created_at: 0,
            expires_at: i64::MAX,
            duration_ms: None,
            tool_calls: None,
            tokens_in: None,
            tokens_out: None,
            num_ctx: None,
            triggered_by: triggered_by.map(|s| s.to_string()),
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
        assert!(
            history[0]
                .content
                .as_ref()
                .unwrap()
                .contains("\"from\": \"user\"")
        );
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
        assert!(
            history[0]
                .content
                .as_ref()
                .unwrap()
                .contains("\"from\": \"gendry\"")
        );
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

    #[test]
    fn test_format_conversation_history_recall_becomes_assistant() {
        // Recall messages should become standalone assistant messages, NOT JSON-wrapped user content.
        // DB order after a completed turn: user₁, recall, assistant₁, user₂
        // Correct history: [recall], [user₁-JSON], [assistant₁], final = [user₂-JSON]
        // Recall is placed BEFORE the user message it was associated with, matching
        // the in-memory ordering during the original turn.
        let msgs = vec![
            make_conv_msg("user", "hello"),
            make_conv_msg("recall", "[Relevant memories]\n- Memory 1\n- Memory 2"),
            make_conv_msg("arya", "Hi! How can I help?"),
            make_conv_msg("user", "what's up?"),
        ];
        let (history, final_content) = format_conversation_history(&msgs, "arya");

        // Should have 3 messages in history: assistant (recall), user, assistant (arya)
        assert_eq!(history.len(), 3);
        // Recall should be first — assistant role with raw content (not JSON-wrapped)
        assert_eq!(history[0].role, "assistant");
        assert!(history[0].content.as_ref().unwrap().contains("[Relevant memories]"));
        assert!(!history[0].content.as_ref().unwrap().contains("\"from\": \"recall\""));
        // User message comes after recall
        assert_eq!(history[1].role, "user");
        assert!(
            history[1]
                .content
                .as_ref()
                .unwrap()
                .contains("\"from\": \"user\"")
        );
        // Arya's response should also be assistant role
        assert_eq!(history[2].role, "assistant");
        assert_eq!(history[2].content.as_ref().unwrap(), "Hi! How can I help?");

        // Final content should be the last user message
        assert!(final_content.contains("\"from\": \"user\""));
        assert!(final_content.contains("what's up?"));
        // Final content should NOT contain recall (would indicate double injection bug)
        assert!(!final_content.contains("Relevant memories"));
    }

    // =========================================================================
    // Tool result filtering tests (triggered_by)
    // =========================================================================

    #[test]
    fn test_format_conversation_history_own_tool_results_included() {
        // Tool results triggered by the current agent should be included
        let msgs = vec![
            make_conv_msg("user", "read my file"),
            make_conv_msg("arya", "Sure, let me read it."),
            make_tool_msg("[Tool Result for read_file]\nfile contents here", Some("arya")),
            make_conv_msg("user", "thanks"),
        ];
        let (history, _final_content) = format_conversation_history(&msgs, "arya");

        // History should include: user, assistant, tool result (3 messages)
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].role, "user");
        assert_eq!(history[1].role, "assistant");
        assert_eq!(history[2].role, "user"); // tool results become user role
        assert!(history[2].content.as_ref().unwrap().contains("file contents here"));
    }

    #[test]
    fn test_format_conversation_history_other_agent_tool_results_filtered() {
        // Tool results triggered by a different agent should be filtered out
        let msgs = vec![
            make_conv_msg("user", "hello @arya @gendry"),
            make_conv_msg("gendry", "Let me check something."),
            make_tool_msg("[Tool Result for shell]\nsome output", Some("gendry")),
            make_conv_msg("user", "what do you think @arya?"),
        ];
        let (history, final_content) = format_conversation_history(&msgs, "arya");

        // Gendry's tool result should be filtered out.
        // History: user batch (user + gendry), no tool result
        // Final: user asking arya
        assert!(history.len() <= 2); // user batch + possibly gendry's message
        for msg in &history {
            if let Some(content) = &msg.content {
                assert!(!content.contains("some output"), "Should not contain gendry's tool result");
            }
        }
        assert!(final_content.contains("what do you think"));
    }

    #[test]
    fn test_format_conversation_history_unattributed_tool_results_included() {
        // Old tool results (triggered_by = None) should be included for all agents
        let msgs = vec![
            make_conv_msg("user", "do something"),
            make_conv_msg("arya", "Ok."),
            make_tool_msg("[Tool Result for read_file]\nold data", None), // unattributed
            make_conv_msg("user", "next"),
        ];
        let (history, _final_content) = format_conversation_history(&msgs, "arya");

        // Unattributed tool result should be included
        assert_eq!(history.len(), 3);
        assert!(history[2].content.as_ref().unwrap().contains("old data"));
    }

    // Tests for expand_inject_directives

    #[test]
    fn test_expand_inject_directives_no_directives() {
        let content = "Some always content\nwith no directives";
        let result = expand_inject_directives(content, "tools here", "memories here");
        // Should return None to signal fallback behavior
        assert!(result.is_none());
    }

    #[test]
    fn test_expand_inject_directives_tools_only() {
        let content = "Before\n<!-- @inject:tools -->\nAfter";
        let result = expand_inject_directives(content, "**Tools:**\n- tool1", "");
        assert!(result.is_some());
        let expanded = result.unwrap();
        assert!(expanded.contains("Before"));
        assert!(expanded.contains("**Tools:**\n- tool1"));
        assert!(expanded.contains("After"));
        assert!(!expanded.contains("<!-- @inject:tools -->"));
    }

    #[test]
    fn test_expand_inject_directives_memories_only() {
        let content = "Before\n<!-- @inject:memories -->\nAfter";
        let result = expand_inject_directives(content, "", "[Memories]\n- mem1");
        assert!(result.is_some());
        let expanded = result.unwrap();
        assert!(expanded.contains("Before"));
        assert!(expanded.contains("[Memories]\n- mem1"));
        assert!(expanded.contains("After"));
        assert!(!expanded.contains("<!-- @inject:memories -->"));
    }

    #[test]
    fn test_expand_inject_directives_both() {
        let content = "Header\n<!-- @inject:tools -->\nMiddle\n<!-- @inject:memories -->\nFooter";
        let result = expand_inject_directives(content, "TOOLS", "MEMORIES");
        assert!(result.is_some());
        let expanded = result.unwrap();
        assert!(expanded.contains("Header"));
        assert!(expanded.contains("TOOLS"));
        assert!(expanded.contains("Middle"));
        assert!(expanded.contains("MEMORIES"));
        assert!(expanded.contains("Footer"));
        // Verify order is preserved
        let tools_pos = expanded.find("TOOLS").unwrap();
        let memories_pos = expanded.find("MEMORIES").unwrap();
        assert!(tools_pos < memories_pos);
    }

    #[test]
    fn test_expand_inject_directives_empty_tools_removes_line() {
        let content = "Line1\n<!-- @inject:tools -->\nLine2";
        let result = expand_inject_directives(content, "", "");
        assert!(result.is_some());
        let expanded = result.unwrap();
        assert_eq!(expanded, "Line1\nLine2");
    }

    #[test]
    fn test_expand_inject_directives_empty_memories_removes_line() {
        let content = "Line1\n<!-- @inject:memories -->\nLine2";
        let result = expand_inject_directives(content, "", "");
        assert!(result.is_some());
        let expanded = result.unwrap();
        assert_eq!(expanded, "Line1\nLine2");
    }

    #[test]
    fn test_expand_inject_directives_both_empty() {
        let content = "Line1\n<!-- @inject:tools -->\nLine2\n<!-- @inject:memories -->\nLine3";
        let result = expand_inject_directives(content, "", "");
        assert!(result.is_some());
        let expanded = result.unwrap();
        assert_eq!(expanded, "Line1\nLine2\nLine3");
    }

    #[test]
    fn test_expand_inject_directives_whitespace_tolerance() {
        // Directive with surrounding whitespace on the line
        let content = "Before\n  <!-- @inject:tools -->  \nAfter";
        let result = expand_inject_directives(content, "TOOLS", "");
        assert!(result.is_some());
        // The line with only the directive should be filtered out when empty
        // But with content, it replaces the directive text
        let expanded = result.unwrap();
        assert!(expanded.contains("TOOLS"));
    }

    #[test]
    fn test_build_recall_content_with_directives() {
        let base =
            Some("Header\n<!-- @inject:tools -->\n<!-- @inject:memories -->\nFooter".to_string());
        let result = build_recall_content("TOOLS", "MEMORIES", &base, &None);
        assert!(result.is_some());
        let effective = result.unwrap();
        // Tools and memories should be in their directive positions
        let tools_pos = effective.find("TOOLS").unwrap();
        let memories_pos = effective.find("MEMORIES").unwrap();
        let footer_pos = effective.find("Footer").unwrap();
        assert!(tools_pos < memories_pos);
        assert!(memories_pos < footer_pos);
    }

    #[test]
    fn test_build_recall_content_without_directives_no_injection() {
        // If always.md exists but has no directives, user opted out of injection
        let base = Some("Just content, no directives".to_string());
        let result = build_recall_content("TOOLS", "MEMORIES", &base, &None);
        assert!(result.is_some());
        let effective = result.unwrap();
        // Should NOT contain tools or memories - user opted out
        assert!(!effective.contains("TOOLS"));
        assert!(!effective.contains("MEMORIES"));
        assert!(effective.contains("Just content"));
    }

    #[test]
    fn test_build_recall_content_no_base_injects_defaults() {
        // If no always.md at all, inject tools/memories as sensible defaults
        let result = build_recall_content("TOOLS", "MEMORIES", &None, &None);
        assert!(result.is_some());
        let effective = result.unwrap();
        assert!(effective.contains("TOOLS"));
        assert!(effective.contains("MEMORIES"));
    }

    #[test]
    fn test_build_recall_content_model_always_appended() {
        let base = Some("Base\n<!-- @inject:tools -->".to_string());
        let model = Some("Model always".to_string());
        let result = build_recall_content("TOOLS", "", &base, &model);
        assert!(result.is_some());
        let effective = result.unwrap();
        // Model always should be at the end
        assert!(effective.ends_with("Model always") || effective.contains("Model always"));
        let base_pos = effective.find("Base").unwrap();
        let model_pos = effective.find("Model always").unwrap();
        assert!(base_pos < model_pos);
    }
}
