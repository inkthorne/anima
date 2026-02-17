//! Daemon mode for running agents headlessly.
//!
//! This module provides the infrastructure for running agents as background daemons,
//! with Unix socket API for communication and timer trigger support.

use std::borrow::Cow;
use std::fs::{File, OpenOptions};
use std::io::{IsTerminal, Write};
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

        // Also print to stdout for interactive use (skip when stdout is redirected to log file)
        if std::io::stdout().is_terminal() {
            print!("{}", line);
        }
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
use crate::conversation::ConversationError;
use crate::discovery;
use crate::embedding::EmbeddingClient;
use crate::auth;
use crate::llm::{
    AnthropicClient, ChatMessage, ClaudeCodeClient, LLM, LLMError, LLMResponse, OllamaClient,
    OpenAIClient, ToolSpec, strip_thinking_tags,
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
use crate::tools::notes::DaemonNotesTool;
use crate::tools::task::DaemonTaskTool;
use crate::tools::{
    AddTool, CopyLinesTool, DaemonRememberTool, DaemonSearchConversationTool, EchoTool,
    EditFileTool, HttpTool, ListFilesTool, PeekFileTool, ReadFileTool, SafeShellTool, ShellTool,
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

/// Result from processing in native tool mode (standalone, no conversation)
struct NativeToolModeResult {
    response: String,
    duration_ms: Option<u64>,
    tokens_in: Option<u32>,
    tokens_out: Option<u32>,
    prompt_eval_duration_ns: Option<u64>,
    cached_tokens: Option<u32>,
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
    conversation_injection: &str,
) -> Option<String> {
    const TOOLS_DIRECTIVE: &str = "<!-- @inject:tools -->";
    const MEMORIES_DIRECTIVE: &str = "<!-- @inject:memories -->";
    const CONVERSATION_DIRECTIVE: &str = "<!-- @inject:conversation -->";

    let has_tools_directive = content.contains(TOOLS_DIRECTIVE);
    let has_memories_directive = content.contains(MEMORIES_DIRECTIVE);
    let has_conversation_directive = content.contains(CONVERSATION_DIRECTIVE);

    // No directives found — signal that user opted out of auto-injection
    if !has_tools_directive && !has_memories_directive && !has_conversation_directive {
        return None;
    }

    let mut result = content.to_string();

    // Replace or remove each directive based on whether its content is empty
    let directives: &[(&str, &str)] = &[
        (TOOLS_DIRECTIVE, tools_injection),
        (MEMORIES_DIRECTIVE, memory_injection),
        (CONVERSATION_DIRECTIVE, conversation_injection),
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
/// - If base_recall exists but has no directives, recall.md content comes first,
///   then tools/memories/conversation are appended after it.
/// - If base_recall is None, tools and memories are injected as sensible defaults.
fn build_recall_content(
    tools_injection: &str,
    memory_injection: &str,
    conversation_injection: &str,
    base_recall: &Option<String>,
    model_recall: &Option<String>,
) -> Option<String> {
    let mut parts = Vec::new();

    // Try directive expansion first
    let expanded_base = base_recall.as_ref().and_then(|base| {
        expand_inject_directives(base, tools_injection, memory_injection, conversation_injection)
    });

    if let Some(expanded) = expanded_base {
        // Directives were found and expanded — use the expanded content
        parts.push(expanded);
    } else {
        // No directives — include recall.md content first (if any), then auto-inject
        if let Some(base) = base_recall {
            parts.push(base.clone());
        }
        if !tools_injection.is_empty() {
            parts.push(tools_injection.to_string());
        }
        if !memory_injection.is_empty() {
            parts.push(memory_injection.to_string());
        }
        if !conversation_injection.is_empty() {
            parts.push(conversation_injection.to_string());
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
    dedup_up_to: Option<i64>,
) -> (Vec<ChatMessage>, String) {
    if messages.is_empty() {
        return (Vec::new(), String::new());
    }

    let mut history: Vec<ChatMessage> = Vec::new();
    let mut pending_user_batch: Vec<String> = Vec::new();

    // Pre-scan: identify superseded tool pairs for dedup using tool_call_id matching.
    // Since v3.10.9, every tool result has a tool_call_id linking to its tool call.
    // This replaces the old VecDeque positional matching and naturally handles
    // [Tool Error for ...] messages (which also have tool_call_id).
    let drop_ids: std::collections::HashSet<i64>;
    {
        use std::collections::{HashMap, HashSet};

        #[derive(Clone)]
        struct ToolPair {
            assistant_msg_id: i64,
            tool_result_msg_id: i64,
        }

        // Normalize a shell command for dedup: strip 2>&1 and trailing output filters
        let normalize_shell_for_dedup = |cmd: &str| -> String {
            let mut s = cmd.trim().replace(" 2>&1", "");
            // Strip trailing pipe filter (tail/head/grep/wc/sort/tee)
            // only if the filter segment has no further chaining (&&, ||, |)
            if let Some(pos) = s.rfind(" | ") {
                let after = s[pos + 3..].trim_start();
                let filters = ["tail", "head", "grep", "wc", "sort", "tee"];
                if filters.iter().any(|f| after.starts_with(f))
                    && !after.contains(" | ")
                    && !after.contains(" && ")
                    && !after.contains(" || ")
                {
                    s.truncate(pos);
                }
            }
            s.trim().to_string()
        };

        // Pass 1: scan assistant messages, build tool_call_id → (kind, path, assistant_msg_id)
        #[derive(Clone)]
        enum DedupKind { ReadFull, ReadRange, Write, EditFile, Shell, Notes, UnknownTool }

        let mut tc_index: HashMap<String, (DedupKind, Option<String>, i64)> = HashMap::new();
        let mut assistant_tc_count: HashMap<i64, usize> = HashMap::new();

        for msg in messages.iter() {
            if msg.from_agent != current_agent { continue; }
            if let Some(ref tc_json) = msg.tool_calls {
                if let Ok(tcs) = serde_json::from_str::<Vec<crate::llm::ToolCall>>(tc_json) {
                    assistant_tc_count.insert(msg.id, tcs.len());
                    for tc in &tcs {
                        match tc.name.as_str() {
                            "read_file" => {
                                if let Some(path) = tc.arguments.get("path").and_then(|v| v.as_str()) {
                                    tc_index.insert(tc.id.clone(), (DedupKind::ReadFull, Some(path.to_string()), msg.id));
                                }
                            }
                            "peek_file" => {
                                if let Some(path) = tc.arguments.get("path").and_then(|v| v.as_str()) {
                                    tc_index.insert(tc.id.clone(), (DedupKind::ReadRange, Some(path.to_string()), msg.id));
                                }
                            }
                            "write_file" => {
                                if let Some(path) = tc.arguments.get("path").and_then(|v| v.as_str()) {
                                    tc_index.insert(tc.id.clone(), (DedupKind::Write, Some(path.to_string()), msg.id));
                                }
                            }
                            "edit_file" => {
                                if let Some(path) = tc.arguments.get("path").and_then(|v| v.as_str()) {
                                    tc_index.insert(tc.id.clone(), (DedupKind::EditFile, Some(path.to_string()), msg.id));
                                }
                            }
                            "shell" | "safe_shell" => {
                                if let Some(cmd) = tc.arguments.get("command").and_then(|v| v.as_str()) {
                                    tc_index.insert(tc.id.clone(), (DedupKind::Shell, Some(normalize_shell_for_dedup(cmd)), msg.id));
                                }
                            }
                            "notes" => {
                                tc_index.insert(tc.id.clone(), (DedupKind::Notes, None, msg.id));
                            }
                            _ => {
                                tc_index.insert(tc.id.clone(), (DedupKind::UnknownTool, None, msg.id));
                            }
                        }
                    }
                }
            }
        }

        // Pass 2: scan tool result messages, pair via tool_call_id
        let mut fresh_view_pairs: HashMap<String, Vec<ToolPair>> = HashMap::new();
        let mut read_file_range: Vec<(String, ToolPair)> = Vec::new();
        let mut edit_file_pairs: Vec<(String, ToolPair)> = Vec::new();
        let mut shell_pairs: HashMap<String, Vec<ToolPair>> = HashMap::new();
        let mut unknown_tool_pairs: Vec<ToolPair> = Vec::new();
        let mut notes_pairs: Vec<ToolPair> = Vec::new();

        for msg in messages.iter() {
            if msg.from_agent != "tool" { continue; }
            match &msg.triggered_by {
                Some(owner) if owner != current_agent => continue,
                _ => {}
            }
            if let Some(ref tcid) = msg.tool_call_id {
                if let Some((kind, path, asst_id)) = tc_index.get(tcid) {
                    // Error results: unknown-tool errors are collected for dedup;
                    // all other errors are skipped (small and useful to keep).
                    if msg.content.starts_with("[Tool Error") {
                        if matches!(kind, DedupKind::UnknownTool) && msg.content.contains("Unknown tool:") {
                            let pair = ToolPair { assistant_msg_id: *asst_id, tool_result_msg_id: msg.id };
                            unknown_tool_pairs.push(pair);
                        }
                        continue;
                    }
                    let pair = ToolPair { assistant_msg_id: *asst_id, tool_result_msg_id: msg.id };
                    match kind {
                        DedupKind::ReadFull | DedupKind::Write => {
                            if let Some(p) = path { fresh_view_pairs.entry(p.clone()).or_default().push(pair); }
                        }
                        DedupKind::ReadRange => {
                            if let Some(p) = path { read_file_range.push((p.clone(), pair)); }
                        }
                        DedupKind::EditFile => {
                            if let Some(p) = path { edit_file_pairs.push((p.clone(), pair)); }
                        }
                        DedupKind::Shell => {
                            if let Some(c) = path { shell_pairs.entry(c.clone()).or_default().push(pair); }
                        }
                        DedupKind::Notes => {
                            notes_pairs.push(pair);
                        }
                        DedupKind::UnknownTool => {} // handled above in the error check
                    }
                }
            }
        }

        // Pass 3: apply dedup rules — collect tool_result IDs and assistant IDs to drop
        let mut dropped_tool_results: HashSet<i64> = HashSet::new();
        let mut asst_dropped: HashMap<i64, usize> = HashMap::new();

        let mark_drop = |pair: &ToolPair, dropped: &mut HashSet<i64>, ad: &mut HashMap<i64, usize>| {
            if let Some(cutoff) = dedup_up_to {
                if pair.tool_result_msg_id > cutoff { return; }
            } else {
                return; // No pointer set → no dedup
            }
            dropped.insert(pair.tool_result_msg_id);
            *ad.entry(pair.assistant_msg_id).or_insert(0) += 1;
        };

        // Fresh views (full-read + write): keep only the last per path
        for (_path, pairs) in &fresh_view_pairs {
            for pair in pairs.iter().rev().skip(1) {
                mark_drop(pair, &mut dropped_tool_results, &mut asst_dropped);
            }
        }

        // Build latest fresh view per path
        let mut latest_fresh_view: HashMap<&str, i64> = HashMap::new();
        for (path, pairs) in &fresh_view_pairs {
            if let Some(last) = pairs.last() {
                let entry = latest_fresh_view.entry(path.as_str()).or_insert(0);
                if last.tool_result_msg_id > *entry { *entry = last.tool_result_msg_id; }
            }
        }

        // read_file_range: drop if a later full-read or write_file exists for same path
        for (path, pair) in &read_file_range {
            if let Some(&latest_id) = latest_fresh_view.get(path.as_str()) {
                if latest_id > pair.tool_result_msg_id {
                    mark_drop(pair, &mut dropped_tool_results, &mut asst_dropped);
                }
            }
        }

        // edit_file: drop if a later full-read or write_file exists for same path
        for (path, pair) in &edit_file_pairs {
            if let Some(&latest_id) = latest_fresh_view.get(path.as_str()) {
                if latest_id > pair.tool_result_msg_id {
                    mark_drop(pair, &mut dropped_tool_results, &mut asst_dropped);
                }
            }
        }

        // Identical shell commands: keep only the last pair per command
        for (_cmd, pairs) in &shell_pairs {
            for pair in pairs.iter().rev().skip(1) {
                mark_drop(pair, &mut dropped_tool_results, &mut asst_dropped);
            }
        }

        // Notes: keep only the last (each call replaces the scratchpad)
        for pair in notes_pairs.iter().rev().skip(1) {
            mark_drop(pair, &mut dropped_tool_results, &mut asst_dropped);
        }

        // Unknown tool errors: keep only the last pair
        for pair in unknown_tool_pairs.iter().rev().skip(1) {
            mark_drop(pair, &mut dropped_tool_results, &mut asst_dropped);
        }

        // Build final drop_ids: always drop superseded tool results,
        // only drop assistant messages if ALL their tool_calls are dropped
        let mut ids: HashSet<i64> = HashSet::new();
        ids.extend(&dropped_tool_results);
        for (&asst_id, &drop_count) in &asst_dropped {
            if let Some(&total) = assistant_tc_count.get(&asst_id) {
                if drop_count >= total {
                    ids.insert(asst_id);
                }
            }
        }
        drop_ids = ids;
    }

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

    // Track the last history index where the original message id <= dedup_up_to cutoff.
    // Used to limit notes stripping to the frozen prefix region.
    let mut dedup_boundary_idx: Option<usize> = None;

    // Process all messages
    for msg in messages {
        if drop_ids.contains(&msg.id) {
            continue;
        }
        if msg.from_agent == "recall" {
            // Skip recall injected by other agents — they're irrelevant noise.
            // Unattributed (triggered_by = None) are included for backward compatibility.
            match &msg.triggered_by {
                Some(owner) if owner != current_agent => continue,
                _ => {}
            }
            // Recall goes BEFORE pending user messages to maintain:
            // [recall] [user] ordering (recall precedes the query it supports)
            history.push(ChatMessage {
                role: "assistant".to_string(),
                content: Some(msg.content.clone()),
                tool_call_id: None,
                tool_calls: None,
            });
            if dedup_up_to.is_some_and(|cutoff| msg.id <= cutoff) {
                dedup_boundary_idx = Some(history.len() - 1);
            }
        } else if msg.from_agent == current_agent {
            // Agent's own response — flush pending user messages first
            flush_user_batch(&mut pending_user_batch, &mut history);

            let tool_calls: Option<Vec<crate::llm::ToolCall>> = msg
                .tool_calls
                .as_ref()
                .and_then(|json| serde_json::from_str(json).ok());

            history.push(ChatMessage {
                role: "assistant".to_string(),
                content: Some(strip_thinking_tags(&msg.content)),
                tool_call_id: None,
                tool_calls,
            });
            if dedup_up_to.is_some_and(|cutoff| msg.id <= cutoff) {
                dedup_boundary_idx = Some(history.len() - 1);
            }
        } else if msg.from_agent == "tool" {
            // Skip tool results triggered by other agents — they're irrelevant noise.
            // Unattributed (triggered_by = None) are included for backward compatibility.
            match &msg.triggered_by {
                Some(owner) if owner != current_agent => continue,
                _ => {}
            }
            // Tool results/errors should NOT be batched with user messages.
            // Flush any pending user messages first, then add tool result.
            flush_user_batch(&mut pending_user_batch, &mut history);
            if msg.tool_call_id.is_some() {
                // Native tool result — emit with role "tool" and tool_call_id for API compatibility
                history.push(ChatMessage {
                    role: "tool".to_string(),
                    content: Some(msg.content.clone()),
                    tool_call_id: msg.tool_call_id.clone(),
                    tool_calls: None,
                });
            } else {
                // Legacy/JSON-block tool result — emit as role "user"
                history.push(ChatMessage {
                    role: "user".to_string(),
                    content: Some(msg.content.clone()),
                    tool_call_id: None,
                    tool_calls: None,
                });
            }
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

    // Strip <notes> from assistant messages at or before the dedup boundary.
    // Messages after the boundary are part of the frozen prefix — don't touch them.
    // The last assistant message always keeps its notes (most recent copy is useful).
    let last_asst_idx = history.iter().rposition(|m| m.role == "assistant");
    for (i, msg) in history.iter_mut().enumerate() {
        if msg.role == "assistant" && Some(i) != last_asst_idx {
            // Only strip if this message is within the dedup boundary
            let within_boundary = match dedup_boundary_idx {
                Some(boundary) => i <= boundary,
                None => false, // No boundary → no stripping
            };
            if within_boundary {
                if let Some(ref content) = msg.content {
                    if let Some(stripped) = strip_notes(content) {
                        msg.content = Some(stripped);
                    }
                }
            }
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
    /// Query embedding (reusable for conversation recall and user message embedding)
    query_embedding: Option<Vec<f32>>,
}

/// Build tools and memory injection for a query.
///
/// This shared helper handles:
/// - Tool recall (native mode: all allowed tools, JSON-block mode: keyword-matched tools)
/// - Memory recall (semantic search with embeddings)
/// - Conversation recall (semantic search over past user messages)
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
    conv_name: Option<&str>,
    window_message_ids: &[i64],
    conversation_recall_limit: usize,
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

    // Compute query embedding once (shared by memory recall + conversation recall)
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

    // Inject relevant memories
    let memory_injection = if let Some(mem_store) = semantic_memory {
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

    // Conversation recall: search past user messages that have scrolled out of the window
    let conversation_injection = if conversation_recall_limit > 0
        && conv_name.is_some()
        && query_embedding.is_some()
    {
        let cname = conv_name.unwrap();
        match ConversationStore::init() {
            Ok(store) => {
                match store.search_similar_messages(
                    cname,
                    query_embedding.as_deref().unwrap(),
                    window_message_ids,
                    conversation_recall_limit,
                ) {
                    Ok(results) => {
                        if !results.is_empty() {
                            logger.memory(&format!(
                                "Conversation recall: {} messages for query",
                                results.len()
                            ));
                            for (id, from, content, _, sim) in &results {
                                logger.memory(&format!(
                                    "  ({:.3}) [{}] \"{}\" [msg#{}]",
                                    sim,
                                    from,
                                    content.chars().take(80).collect::<String>(),
                                    id
                                ));
                            }
                        }
                        build_conversation_recall_injection(&results)
                    }
                    Err(e) => {
                        logger.log(&format!("Conversation recall error: {}", e));
                        String::new()
                    }
                }
            }
            Err(e) => {
                logger.log(&format!("Conversation recall store error: {}", e));
                String::new()
            }
        }
    } else {
        String::new()
    };

    let recall_content = build_recall_content(
        &tools_injection,
        &memory_injection,
        &conversation_injection,
        recall,
        model_recall,
    );

    RecallResult {
        recall_content,
        external_tools,
        relevant_tools,
        query_embedding,
    }
}

/// Build the conversation recall injection string for prepending to context.
fn build_conversation_recall_injection(results: &[(i64, String, String, i64, f32)]) -> String {
    if results.is_empty() {
        return String::new();
    }

    let mut injection = String::from("[recalled messages]\n");
    for (_id, from_agent, content, created_at, _sim) in results {
        let age = crate::memory::format_age(*created_at);
        // Truncate long messages to 200 chars
        let display: String = if content.len() > 200 {
            format!("{}...", content.chars().take(200).collect::<String>())
        } else {
            content.clone()
        };
        injection.push_str(&format!("- ({}) [{}] {}\n", age, from_agent, display));
    }
    injection.push('\n');
    injection
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

/// Inject a notes prompt when the agent has no notes yet.
/// When notes exist, they are already embedded in stored assistant messages
/// (via `prepend_notes()`) and stripped from all but the last by `format_conversation_history()`.
/// Prepend `<notes>` block to an assistant response before storing in DB.
/// Returns content unchanged if the agent has no notes.
fn prepend_notes(
    content: &str,
    store: &ConversationStore,
    conv_name: &str,
    agent_name: &str,
) -> String {
    let notes = match store.get_participant_notes(conv_name, agent_name) {
        Ok(Some(n)) if !n.is_empty() => n,
        _ => return content.to_string(),
    };
    format!(
        "<notes>\n{}\n</notes>\n\n{}",
        notes, content
    )
}

/// Strip a `<notes>...</notes>` block (and trailing newlines) from content.
/// Returns `None` if no block is found.
fn strip_notes(content: &str) -> Option<String> {
    let start = content.find("<notes>")?;
    let end_tag = "</notes>";
    let end = content.find(end_tag)? + end_tag.len();
    let after = content[end..].trim_start_matches('\n');
    let before = &content[..start];
    let result = format!("{}{}", before, after);
    Some(if result.is_empty() { String::new() } else { result })
}

/// Extract the **last** `<notes>` block from LLM output.
/// Returns `(cleaned_content, Option<notes_content>)`.
/// The LLM may emit inline notes without calling the `notes` tool — this captures them.
fn extract_llm_notes(content: &str) -> (String, Option<String>) {
    let start_tag = "<notes>";
    let end_tag = "</notes>";
    let start = match content.rfind(start_tag) {
        Some(s) => s,
        None => return (content.to_string(), None),
    };
    let end = match content[start..].find(end_tag) {
        Some(e) => start + e + end_tag.len(),
        None => return (content.to_string(), None),
    };
    let inner = content[start + start_tag.len()..end - end_tag.len()]
        .trim()
        .to_string();
    let before = &content[..start];
    let after = content[end..].trim_start_matches('\n');
    let cleaned = format!("{}{}", before, after);
    if inner.is_empty() {
        (cleaned, None)
    } else {
        (cleaned, Some(inner))
    }
}

/// Spawn a background task that forwards log messages to the AgentLogger.
/// Returns the sender and join handle. Drop the sender when done.
fn spawn_log_forwarder(
    logger: Arc<AgentLogger>,
) -> (mpsc::Sender<String>, tokio::task::JoinHandle<()>) {
    let (tx, mut rx) = mpsc::channel::<String>(64);
    let handle = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            logger.log(&msg);
        }
    });
    (tx, handle)
}

/// Spawn a background task that persists tool trace executions to the conversation store.
/// Returns the join handle. The caller should drop `trace_tx` when done to signal completion.
fn spawn_tool_trace_persister(
    conv_name: Option<String>,
    agent_name: String,
    num_ctx: Option<u32>,
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
        let num_ctx_i64 = num_ctx.map(|n| n as i64);
        while let Some(exec) = trace_rx.recv().await {
            let tool_calls_json = serde_json::to_string(&vec![&exec.call]).ok();
            let content = exec.content.as_deref().unwrap_or("");
            let tokens_in = exec.iter_tokens_in.map(|t| t as i64);
            let tokens_out = exec.iter_tokens_out.map(|t| t as i64);
            let eval_ns = exec.iter_prompt_eval_ns.map(|t| t as i64);
            let cached = exec.iter_cached_tokens.map(|t| t as i64);
            if let Err(e) = store.add_message_with_tokens(
                &cname, &agent_name, content, &[],
                exec.iter_duration_ms.map(|d| d as i64),
                tool_calls_json.as_deref(),
                tokens_in, tokens_out, num_ctx_i64, eval_ns, cached,
            ) {
                logger.log(&format!("[trace] Failed to store tool call: {}", e));
            }
            let tool_result_msg = format!("[Tool Result for {}]\n{}", exec.call.name, exec.result);
            if let Err(e) = store.add_native_tool_result(&cname, &exec.call.id, &tool_result_msg, &agent_name) {
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
        "peek_file" => {
            let tool = PeekFileTool;
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
        "edit_file" => {
            let tool = EditFileTool;
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
        "list_files" => {
            let tool = ListFilesTool;
            match tool.execute(tool_call.params.clone()).await {
                Ok(result) => Ok(result.to_string()),
                Err(e) => Err(format!("Tool error: {}", e)),
            }
        }
        "copy_lines" => {
            let tool = CopyLinesTool;
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
        "search_conversation" => {
            let tool = DaemonSearchConversationTool;
            match tool.execute(tool_call.params.clone()).await {
                Ok(result) => {
                    if let Some(summary) = result.get("summary").and_then(|s| s.as_str()) {
                        Ok(summary.to_string())
                    } else {
                        Ok(result.to_string())
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
                    for builtin in &["list_tools", "list_agents", "remember", "send_message", "task", "search_conversation", "notes"] {
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
                        "Built-in tools: list_tools, list_agents, remember, send_message, task, search_conversation, notes\n(Note: tools.toml failed to load: {})",
                        e
                    ))
                }
            }
        }
        "task" => {
            let ctx = context.ok_or("task tool requires execution context")?;
            let tool = DaemonTaskTool::new(ctx.agent_name.clone(), ctx.conv_id.clone());
            match tool.execute(tool_call.params.clone()).await {
                Ok(result) => {
                    let task_conv = result.get("task_conv").and_then(|s| s.as_str()).unwrap_or("");
                    let agent = result.get("agent").and_then(|s| s.as_str()).unwrap_or("");
                    Ok(format!("Task dispatched to '{}' ({}). Result will be posted here when complete.", agent, task_conv))
                }
                Err(e) => Err(format!("Tool error: {}", e)),
            }
        }
        "notes" => {
            let ctx = context.ok_or("notes tool requires execution context")?;
            let tool = DaemonNotesTool::new(ctx.agent_name.clone(), ctx.conv_id.clone());
            match tool.execute(tool_call.params.clone()).await {
                Ok(_) => Ok("Notes updated.".to_string()),
                Err(e) => Err(format!("Tool error: {}", e)),
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

/// Tools that bypass the allowlist and are always available to every agent.
const ALWAYS_ALLOWED_TOOLS: &[&str] = &[];

/// Filter tools by allowed_tools list. If allowed_tools is None, no tools allowed (safe default).
fn filter_by_allowlist<'a>(
    tools: Vec<&'a ToolDefinition>,
    allowed_tools: &Option<Vec<String>>,
) -> Vec<&'a ToolDefinition> {
    match allowed_tools {
        Some(allowed) => tools
            .into_iter()
            .filter(|t| {
                allowed.contains(&t.name)
                    || ALWAYS_ALLOWED_TOOLS.contains(&t.name.as_str())
            })
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
    /// Maximum wall-clock time for a single notify response (e.g. "10m", "1h")
    pub max_response_time: Option<String>,
    /// Whether dedup runs in lazy mode (only at context fill threshold)
    pub dedup_lazy: bool,
    /// Whether outbound @mention forwarding is enabled (default: true)
    pub mentions: bool,
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
            max_response_time: agent_dir.config.think.max_response_time.clone(),
            dedup_lazy: llm_config.dedup_lazy,
            mentions: agent_dir.config.agent.mentions,
        })
    }
}

/// Build the runtime context string that is appended to the system prompt.
/// This gives agents self-awareness about their environment.
fn build_runtime_context(agent: &str, model: &str, host: &str, tools_native: bool) -> String {
    let tools_mode = if tools_native { "native" } else { "json-block" };
    let home = dirs::home_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    format!(
        "You are running inside Anima, a multi-agent runtime.\n\nRuntime: agent={} | model={} | host={} | tools={} | home={}",
        agent, model, host, tools_mode, home
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
    let mut config = DaemonConfig::from_agent_dir(&agent_dir)?;

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

    // Auto-detect num_ctx from OpenAI-compatible /v1/models endpoint
    if config.num_ctx.is_none() {
        let llm_config = agent_dir.resolve_llm_config()?;
        if llm_config.provider == "openai" {
            if let Some(ref base_url) = llm_config.base_url {
                if let Some(ctx) = query_openai_model_ctx(base_url, &llm_config.model).await {
                    logger.log(&format!("  Auto-detected num_ctx: {} from {}", ctx, base_url));
                    config.num_ctx = Some(ctx);
                }
            }
        }
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

    // Set up signal handling for graceful shutdown (before pending notification processing
    // so SIGTERM/SIGINT can abort in-flight LLM calls during startup)
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

    // Clear stale context cursors from previous sessions
    if let Ok(store) = ConversationStore::init() {
        if let Err(e) = store.clear_all_cursors_for_agent(&config.name) {
            logger.log(&format!("Failed to clear stale cursors: {}", e));
        } else {
            logger.log("Cleared stale context cursors from previous session");
        }
    }

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
                            &task_store,
                            config.num_ctx,
                            config.max_iterations,
                            config.max_response_time.as_deref(),
                            config.mentions,
                            &shutdown,
                            config.semantic_memory.conversation_recall_limit,
                            config.dedup_lazy,
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

                        // Clear this notification immediately so it won't replay on restart
                        if let Err(e) = conv_store.delete_pending_notification(notification.id) {
                            logger.log(&format!("Failed to clear notification {}: {}", notification.id, e));
                        }
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
        let worker_task_store = task_store.clone();
        let worker_heartbeat_config = config.heartbeat.clone();

        let worker_num_ctx = config.num_ctx;
        let worker_max_iterations = config.max_iterations;
        let worker_max_response_time = config.max_response_time.clone();
        let worker_mentions = config.mentions;
        let worker_shutdown = shutdown.clone();
        let worker_conversation_recall_limit = config.semantic_memory.conversation_recall_limit;
        let worker_dedup_lazy = config.dedup_lazy;
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
                worker_task_store,
                worker_heartbeat_config,
                worker_num_ctx,
                worker_max_iterations,
                worker_max_response_time,
                worker_mentions,
                worker_shutdown,
                worker_conversation_recall_limit,
                worker_dedup_lazy,
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
                        let conn_max_iterations = config.max_iterations;
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
                                conn_heartbeat_config,
                                conn_task_store,
                                conn_work_tx,
                                conn_max_iterations,
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
        agent.register_tool(Arc::new(PeekFileTool));
        agent.register_tool(Arc::new(WriteFileTool));
        agent.register_tool(Arc::new(EditFileTool));
        agent.register_tool(Arc::new(ListFilesTool));
        agent.register_tool(Arc::new(CopyLinesTool));
        agent.register_tool(Arc::new(HttpTool::new()));
        agent.register_tool(Arc::new(ShellTool::new()));
        agent.register_tool(Arc::new(SafeShellTool::new()));
        // Note: DaemonRememberTool is registered later after semantic memory is created

        // Register daemon-aware messaging tools
        agent.register_tool(Arc::new(DaemonSendMessageTool::new(agent_name.clone())));
        agent.register_tool(Arc::new(DaemonListAgentsTool::new(agent_name.clone())));
        agent.register_tool(Arc::new(DaemonTaskTool::new(agent_name.clone(), None)));
        agent.register_tool(Arc::new(DaemonNotesTool::new(agent_name.clone(), None)));
        agent.register_tool(Arc::new(DaemonSearchConversationTool));
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

/// Query an OpenAI-compatible /v1/models/{model} endpoint for max_context_length.
/// Returns None on any failure (network, parse, missing field).
async fn query_openai_model_ctx(base_url: &str, model: &str) -> Option<u32> {
    let url = format!("{}/models/{}", base_url.trim_end_matches('/'), model);
    let client = reqwest::Client::new();
    let resp = client.get(&url).send().await.ok()?;
    let json: serde_json::Value = resp.json().await.ok()?;
    json["max_context_length"].as_u64().map(|v| v as u32)
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
            if let Some(ref style) = config.api_style {
                client = client.with_api_style(style);
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
        "claude-code" => Arc::new(ClaudeCodeClient::new(&config.model)),
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
    task_store: Option<Arc<Mutex<TaskStore>>>,
    heartbeat_config: Option<HeartbeatDaemonConfig>,
    num_ctx: Option<u32>,
    max_iterations: Option<usize>,
    max_response_time: Option<String>,
    mentions_enabled: bool,
    shutdown: Arc<tokio::sync::Notify>,
    conversation_recall_limit: usize,
    dedup_lazy: bool,
) {
    logger.log("[worker] Agent worker started");

    loop {
        let work = tokio::select! {
            item = work_rx.recv() => {
                match item {
                    Some(w) => w,
                    None => break, // channel closed
                }
            }
            _ = shutdown.notified() => {
                logger.log("[worker] Shutdown signal received, stopping");
                break;
            }
        };

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
                    &task_store,
                    mentions_enabled,
                    &shutdown,
                    conversation_recall_limit,
                    max_iterations,
                    num_ctx,
                    dedup_lazy,
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
                    &task_store,
                    num_ctx,
                    max_iterations,
                    max_response_time.as_deref(),
                    mentions_enabled,
                    &shutdown,
                    conversation_recall_limit,
                    dedup_lazy,
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
                        mentions_enabled,
                        &shutdown,
                        max_iterations,
                        num_ctx,
                        dedup_lazy,
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

/// Execute a batch of native tool calls, storing each result to the DB with tool_call_id.
/// Returns a list of tool names executed.
#[allow(clippy::too_many_arguments)]
async fn execute_native_tool_calls(
    tool_calls: &[crate::llm::ToolCall],
    tool_registry: &Option<Arc<ToolRegistry>>,
    tool_context: &ToolExecutionContext,
    conv_name: &str,
    agent_name: &str,
    logger: &Arc<AgentLogger>,
    num_ctx: Option<u32>,
) -> Vec<String> {
    let store = match ConversationStore::init() {
        Ok(s) => s,
        Err(e) => {
            logger.log(&format!("[loop] Failed to init store for tool results: {}", e));
            return Vec::new();
        }
    };
    let mut results = Vec::new();
    for tc in tool_calls {
        let tool_def = tool_registry.as_ref().and_then(|r| r.find_by_name(&tc.name));
        // Convert llm::ToolCall to daemon::ToolCall for execute_tool_call
        let daemon_tc = ToolCall {
            tool: tc.name.clone(),
            params: tc.arguments.clone(),
        };
        let tool_result = match execute_tool_call(&daemon_tc, tool_def, Some(tool_context)).await {
            Ok(result) => {
                let result = truncate_tool_result(&result, num_ctx);
                logger.tool(&format!("[loop] {} → {} bytes", tc.name, result.len()));
                format!("[Tool Result for {}]\n{}", tc.name, result)
            }
            Err(e) => {
                logger.tool(&format!("[loop] {} → error: {}", tc.name, e));
                format!("[Tool Error for {}]\n{}", tc.name, e)
            }
        };
        // Store tool result with tool_call_id for native protocol round-trip
        if let Err(e) = store.add_native_tool_result(conv_name, &tc.id, &tool_result, agent_name) {
            logger.log(&format!("[loop] Failed to store tool result: {}", e));
        }
        results.push(tc.name.clone());
    }
    results
}

/// Result from the unified tool loop.
struct ToolLoopResult {
    /// Final response text (thinking stripped, memories extracted).
    response: String,
    /// Full response for DB storage (with thinking tags, memories extracted).
    db_response: String,
    /// Duration of the last LLM call in milliseconds.
    duration_ms: Option<u64>,
    /// Token usage from the last LLM call.
    tokens_in: Option<u32>,
    tokens_out: Option<u32>,
    prompt_eval_duration_ns: Option<u64>,
    cached_tokens: Option<u32>,
    /// Sequential turn number from the last LLM call's debug dump.
    /// Used by callers to rename dump files to the DB message ID.
    dump_turn_n: Option<u64>,
}

/// Unified tool loop for both native and JSON-block modes.
///
/// Makes single-turn LLM calls (via `think_single_turn_streaming`), stores results to DB,
/// and refreshes context from DB between iterations. The daemon owns the loop; agent.rs
/// only provides the single LLM call.
#[allow(clippy::too_many_arguments)]
async fn run_tool_loop(
    // Initial state
    initial_message: &str,
    conversation_history: Vec<ChatMessage>,
    // Agent & config
    agent: &Arc<Mutex<Agent>>,
    agent_name: &str,
    system_prompt: &Option<String>,
    external_tools: Option<Vec<ToolSpec>>,
    use_native_tools: bool,
    max_iterations: usize,
    num_ctx: Option<u32>,
    // Tool execution
    tool_registry: &Option<Arc<ToolRegistry>>,
    tool_context: &ToolExecutionContext,
    // Memory
    semantic_memory: &Option<Arc<Mutex<SemanticMemoryStore>>>,
    embedding_client: &Option<Arc<EmbeddingClient>>,
    // DB
    conv_name: &str,
    // Streaming
    token_tx: Option<mpsc::Sender<String>>,
    // Control
    cancel: Option<Arc<AtomicBool>>,
    response_deadline: Option<Duration>,
    start_time: std::time::Instant,
    shutdown: &Arc<tokio::sync::Notify>,
    logger: &Arc<AgentLogger>,
    log_tx: mpsc::Sender<String>,
    dedup_lazy: bool,
) -> ToolLoopResult {
    // Open a store for DB operations (pause checks, message storage, context refresh).
    // Each tool batch opens its own store (via execute_native_tool_calls) for isolation.
    let store = match ConversationStore::init() {
        Ok(s) => s,
        Err(e) => {
            logger.log(&format!("[loop] Failed to init store: {}", e));
            let msg = format!("[Error: failed to init conversation store: {}]", e);
            return ToolLoopResult {
                response: msg.clone(),
                db_response: msg,
                duration_ms: None,
                tokens_in: None,
                tokens_out: None,
                prompt_eval_duration_ns: None,
                cached_tokens: None,
                dump_turn_n: None,
            };
        }
    };

    let mut conversation_history = conversation_history;
    let mut current_message = initial_message.to_string();
    let mut tool_call_count = 0usize;
    let mut last_duration_ms: Option<u64> = None;
    let mut last_tokens_in: Option<u32> = None;
    let mut last_tokens_out: Option<u32> = None;
    let mut last_prompt_eval_ns: Option<u64> = None;
    let mut last_cached_tokens: Option<u32> = None;
    let mut last_dump_turn_n: Option<u64> = None;

    for _iteration in 0..max_iterations {
        // Check wall-clock time limit
        if let Some(deadline) = response_deadline {
            if start_time.elapsed() >= deadline {
                logger.log(&format!("[loop] Response time limit exceeded ({:?})", deadline));
                let msg = format!("[Response terminated: exceeded time limit of {:?}]", deadline);
                return ToolLoopResult {
                    response: msg.clone(),
                    db_response: msg,
                    duration_ms: last_duration_ms,
                    tokens_in: last_tokens_in,
                    tokens_out: last_tokens_out,
                    prompt_eval_duration_ns: last_prompt_eval_ns,
                    cached_tokens: last_cached_tokens,
                    dump_turn_n: last_dump_turn_n,
                };
            }
        }

        // Check cancellation (pause)
        if let Some(ref flag) = cancel {
            if flag.load(Ordering::Relaxed) {
                let msg = "[Paused]".to_string();
                return ToolLoopResult {
                    response: msg.clone(),
                    db_response: msg,
                    duration_ms: last_duration_ms,
                    tokens_in: last_tokens_in,
                    tokens_out: last_tokens_out,
                    prompt_eval_duration_ns: last_prompt_eval_ns,
                    cached_tokens: last_cached_tokens,
                    dump_turn_n: last_dump_turn_n,
                };
            }
        }

        // Clear agent's internal history to avoid duplication with DB-backed history
        agent.lock().await.clear_history();

        let options = ThinkOptions {
            system_prompt: system_prompt.clone(),
            conversation_history: if conversation_history.is_empty() {
                None
            } else {
                Some(conversation_history.clone())
            },
            external_tools: external_tools.clone(),
            num_ctx,
            log_tx: Some(log_tx.clone()),
            ..Default::default()
        };

        // Make a single LLM call via streaming (avoids Ollama non-streaming issues)
        let (iter_token_tx, mut iter_token_rx) = mpsc::channel::<String>(100);
        let outer_tx = token_tx.clone();
        let drain_handle = tokio::spawn(async move {
            while let Some(tok) = iter_token_rx.recv().await {
                // Forward tokens to the outer stream if present (REPL streaming)
                if let Some(ref tx) = outer_tx {
                    let _ = tx.send(tok).await;
                }
            }
        });

        let llm_future = async {
            if let Some(deadline) = response_deadline {
                let remaining = deadline.saturating_sub(start_time.elapsed());
                let mut agent_guard = agent.lock().await;
                match tokio::time::timeout(
                    remaining,
                    agent_guard.think_single_turn_streaming(&current_message, options, iter_token_tx),
                ).await {
                    Ok(r) => { drop(agent_guard); Ok(r) }
                    Err(_elapsed) => {
                        drop(agent_guard);
                        Err(format!("[Response terminated: exceeded time limit of {:?}]", deadline))
                    }
                }
            } else {
                let mut agent_guard = agent.lock().await;
                let r = agent_guard
                    .think_single_turn_streaming(&current_message, options, iter_token_tx)
                    .await;
                drop(agent_guard);
                Ok(r)
            }
        };

        let result = tokio::select! {
            r = llm_future => {
                match r {
                    Ok(think_result) => think_result,
                    Err(timeout_msg) => {
                        let _ = drain_handle.await;
                        return ToolLoopResult {
                            response: timeout_msg.clone(),
                            db_response: timeout_msg,
                            duration_ms: last_duration_ms,
                            tokens_in: last_tokens_in,
                            tokens_out: last_tokens_out,
                            prompt_eval_duration_ns: last_prompt_eval_ns,
                            cached_tokens: last_cached_tokens,
                            dump_turn_n: last_dump_turn_n,
                        };
                    }
                }
            }
            _ = shutdown.notified() => {
                let _ = drain_handle.await;
                logger.log("[loop] Shutdown during LLM call, aborting");
                return ToolLoopResult {
                    response: String::new(),
                    db_response: String::new(),
                    duration_ms: None,
                    tokens_in: None,
                    tokens_out: None,
                    prompt_eval_duration_ns: None,
                    cached_tokens: None,
                    dump_turn_n: None,
                };
            }
        };

        let _ = drain_handle.await;

        match result {
            Ok(think_result) => {
                last_duration_ms = think_result.duration_ms;
                last_tokens_in = think_result.tokens_in;
                last_tokens_out = think_result.tokens_out;
                last_prompt_eval_ns = think_result.prompt_eval_duration_ns;
                last_cached_tokens = think_result.cached_tokens;
                last_dump_turn_n = think_result.dump_turn_n;

                let iter_tokens_in = think_result.tokens_in.map(|t| t as i64);
                let iter_tokens_out = think_result.tokens_out.map(|t| t as i64);
                let iter_eval_ns = think_result.prompt_eval_duration_ns.map(|t| t as i64);
                let iter_cached = think_result.cached_tokens.map(|t| t as i64);
                let num_ctx_i64 = num_ctx.map(|n| n as i64);

                // Strip thinking tags and extract [REMEMBER:...] tags
                let without_thinking = strip_thinking_tags(&think_result.response);
                let (after_remember, memories_to_save) = extract_remember_tags(&without_thinking);
                let (after_remember, llm_notes) = extract_llm_notes(&after_remember);

                // Full response for DB: preserve thinking tags, strip REMEMBER tags
                let (db_content, _) = extract_remember_tags(&think_result.response);
                let (db_content, _) = extract_llm_notes(&db_content);

                // Save LLM-generated inline notes to DB (before prepend_notes reads them back)
                if let Some(ref notes) = llm_notes {
                    let _ = store.set_participant_notes(conv_name, agent_name, notes);
                }

                let db_content = prepend_notes(&db_content, &store, conv_name, agent_name);

                // Save memories
                save_memories(&memories_to_save, semantic_memory, embedding_client, logger).await;

                if use_native_tools {
                    // === Native tool mode ===
                    if let Some(ref tool_calls) = think_result.last_tool_calls {
                        // Check pause before executing tools
                        if store.is_paused(conv_name).unwrap_or(false) {
                            logger.log("[loop] Conversation paused, skipping tool execution");
                            return ToolLoopResult {
                                response: after_remember,
                                db_response: db_content,
                                duration_ms: last_duration_ms,
                                tokens_in: last_tokens_in,
                                tokens_out: last_tokens_out,
                                prompt_eval_duration_ns: last_prompt_eval_ns,
                                cached_tokens: last_cached_tokens,
                                dump_turn_n: last_dump_turn_n,
                            };
                        }

                        tool_call_count += tool_calls.len();

                        // Store assistant message with tool_calls (preserve thinking in DB)
                        let tool_calls_json = serde_json::to_string(tool_calls).ok();
                        match store.add_message_with_tokens(
                            conv_name, agent_name, &db_content, &[],
                            think_result.duration_ms.map(|d| d as i64),
                            tool_calls_json.as_deref(),
                            iter_tokens_in, iter_tokens_out, num_ctx_i64, iter_eval_ns, iter_cached,
                        ) {
                            Ok(msg_id) => {
                                if let Some(turn_n) = think_result.dump_turn_n {
                                    agent.lock().await.rename_turn_files(turn_n, msg_id);
                                }
                            }
                            Err(e) => {
                                logger.log(&format!("[loop] Failed to store assistant message: {}", e));
                            }
                        }

                        // Execute all tool calls in the batch
                        let results = execute_native_tool_calls(
                            tool_calls, tool_registry, tool_context,
                            conv_name, agent_name, logger, num_ctx,
                        ).await;

                        current_message = String::new();
                        let _ = results; // consumed
                    } else {
                        // No tool calls — final response
                        return ToolLoopResult {
                            response: after_remember,
                            db_response: db_content,
                            duration_ms: last_duration_ms,
                            tokens_in: last_tokens_in,
                            tokens_out: last_tokens_out,
                            prompt_eval_duration_ns: last_prompt_eval_ns,
                            cached_tokens: last_cached_tokens,
                            dump_turn_n: last_dump_turn_n,
                        };
                    }
                } else {
                    // === JSON-block mode ===
                    let (cleaned_response, tool_call) = extract_tool_call(&after_remember);

                    if let Some(tc) = tool_call {
                        // Check pause before executing tools
                        if store.is_paused(conv_name).unwrap_or(false) {
                            logger.log("[loop] Conversation paused, skipping tool execution");
                            return ToolLoopResult {
                                response: after_remember,
                                db_response: db_content,
                                duration_ms: last_duration_ms,
                                tokens_in: last_tokens_in,
                                tokens_out: last_tokens_out,
                                prompt_eval_duration_ns: last_prompt_eval_ns,
                                cached_tokens: last_cached_tokens,
                                dump_turn_n: last_dump_turn_n,
                            };
                        }

                        tool_call_count += 1;

                        // Store intermediate response (preserve thinking in DB)
                        if !db_content.trim().is_empty() {
                            match store.add_message_with_tokens(
                                conv_name, agent_name, &db_content, &[],
                                think_result.duration_ms.map(|d| d as i64), None,
                                iter_tokens_in, iter_tokens_out, num_ctx_i64, iter_eval_ns, iter_cached,
                            ) {
                                Ok(msg_id) => {
                                    if let Some(turn_n) = think_result.dump_turn_n {
                                        agent.lock().await.rename_turn_files(turn_n, msg_id);
                                    }
                                }
                                Err(e) => {
                                    logger.log(&format!("[loop] Failed to store intermediate response: {}", e));
                                }
                            }
                        }

                        logger.tool(&format!("[loop] Executing: {} with params {}", tc.tool, tc.params));

                        let tool_def = tool_registry.as_ref().and_then(|r| r.find_by_name(&tc.tool));
                        let tool_message = match execute_tool_call(&tc, tool_def, Some(tool_context)).await {
                            Ok(tool_result) => {
                                let tool_result = truncate_tool_result(&tool_result, num_ctx);
                                logger.tool(&format!("[loop] Result: {} bytes", tool_result.len()));
                                format!("[Tool Result for {}]\n{}", tc.tool, tool_result)
                            }
                            Err(e) => {
                                logger.tool(&format!("[loop] Error: {}", e));
                                format!("[Tool Error for {}]\n{}", tc.tool, e)
                            }
                        };

                        // Store tool result
                        if let Err(e) = store.add_tool_result(conv_name, &tool_message, agent_name) {
                            logger.log(&format!("[loop] Failed to store tool message: {}", e));
                        }

                        current_message = tool_message;
                    } else {
                        // No tool call — final response
                        return ToolLoopResult {
                            response: cleaned_response,
                            db_response: db_content,
                            duration_ms: last_duration_ms,
                            tokens_in: last_tokens_in,
                            tokens_out: last_tokens_out,
                            prompt_eval_duration_ns: last_prompt_eval_ns,
                            cached_tokens: last_cached_tokens,
                            dump_turn_n: last_dump_turn_n,
                        };
                    }
                }

                // Refresh context from DB using cursor-based append system
                if let Ok(msgs) = load_agent_context(&store, conv_name, agent_name, logger, num_ctx) {
                    let latest_msg_id = msgs.last().map(|m| m.id);
                    check_fill_and_maybe_reset(
                        &store,
                        conv_name,
                        agent_name,
                        last_tokens_in.map(|t| t as i64),
                        last_tokens_out.map(|t| t as i64),
                        num_ctx.map(|n| n as i64),
                        dedup_lazy,
                        latest_msg_id,
                        logger,
                    );
                    let dedup_up_to = if dedup_lazy {
                        store.get_dedup_cursor(conv_name, agent_name).unwrap_or(None)
                    } else {
                        latest_msg_id
                    };
                    let (refreshed_history, refreshed_final) =
                        format_conversation_history(&msgs, agent_name, dedup_up_to);
                    conversation_history = refreshed_history;
                    if current_message.is_empty() {
                        current_message = refreshed_final;
                    }
                }

                // Inject tool budget nudge
                if let Some(nudge) = crate::agent::tool_budget_nudge(tool_call_count, max_iterations) {
                    current_message.push_str(&format!("\n\n---\n{}", nudge));
                }
            }
            Err(crate::error::AgentError::Cancelled) => {
                logger.log("[loop] Agent cancelled (conversation paused)");
                let msg = "[Paused]".to_string();
                return ToolLoopResult {
                    response: msg.clone(),
                    db_response: msg,
                    duration_ms: last_duration_ms,
                    tokens_in: last_tokens_in,
                    tokens_out: last_tokens_out,
                    prompt_eval_duration_ns: last_prompt_eval_ns,
                    cached_tokens: last_cached_tokens,
                    dump_turn_n: last_dump_turn_n,
                };
            }
            Err(e) => {
                logger.log(&format!("[loop] Agent error: {}", e));
                let error_msg = format!("[Error: {}]", e);
                let _ = store.add_message(conv_name, agent_name, &error_msg, &[]);
                return ToolLoopResult {
                    response: error_msg.clone(),
                    db_response: error_msg,
                    duration_ms: last_duration_ms,
                    tokens_in: last_tokens_in,
                    tokens_out: last_tokens_out,
                    prompt_eval_duration_ns: last_prompt_eval_ns,
                    cached_tokens: last_cached_tokens,
                    dump_turn_n: last_dump_turn_n,
                };
            }
        }
    }

    logger.log(&format!("[loop] Max iterations reached: {}", max_iterations));
    let msg = format!("[Max iterations reached: {}]", max_iterations);
    ToolLoopResult {
        response: msg.clone(),
        db_response: msg,
        duration_ms: last_duration_ms,
        tokens_in: last_tokens_in,
        tokens_out: last_tokens_out,
        prompt_eval_duration_ns: last_prompt_eval_ns,
        cached_tokens: last_cached_tokens,
        dump_turn_n: last_dump_turn_n,
    }
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
    task_store: &Option<Arc<Mutex<TaskStore>>>,
    mentions_enabled: bool,
    shutdown: &Arc<tokio::sync::Notify>,
    conversation_recall_limit: usize,
    max_iterations: Option<usize>,
    num_ctx: Option<u32>,
    dedup_lazy: bool,
) -> MessageWorkResult {
    // Set current conversation for debug file naming
    {
        let mut agent_guard = agent.lock().await;
        agent_guard.set_current_conversation(conv_name.map(|s| s.to_string()));
    }

    // Append conversation name to system prompt so the agent knows where it is
    let system_prompt = &conv_name
        .map(|c| {
            let base = system_prompt.clone().unwrap_or_default();
            Some(format!("{}\nConversation: {}", base, c))
        })
        .unwrap_or_else(|| system_prompt.clone());

    // Load conversation history first (using append-only context cursors)
    // Also capture window message IDs for conversation recall exclusion
    let (mut conversation_history, final_user_content, window_message_ids, last_user_msg_id): (Vec<ChatMessage>, String, Vec<i64>, Option<i64>) =
        if let Some(cname) = conv_name {
            match ConversationStore::init() {
                Ok(store) => match load_agent_context(&store, cname, agent_name, &logger, num_ctx) {
                    Ok(msgs) if !msgs.is_empty() => {
                        let ids: Vec<i64> = msgs.iter().map(|m| m.id).collect();
                        let last_uid = msgs.iter().rev().find(|m| m.from_agent == "user").map(|m| m.id);
                        let dedup_up_to = if dedup_lazy {
                            store.get_dedup_cursor(cname, agent_name).unwrap_or(None)
                        } else {
                            msgs.last().map(|m| m.id)
                        };
                        let (history, final_content) =
                            format_conversation_history(&msgs, agent_name, dedup_up_to);
                        logger.log(&format!(
                            "[worker] Loaded {} history messages for conversation {}",
                            history.len(),
                            cname
                        ));
                        (history, final_content, ids, last_uid)
                    }
                    Ok(_) => (Vec::new(), content.to_string(), Vec::new(), None),
                    Err(e) => {
                        logger.log(&format!(
                            "[worker] Failed to load conversation history: {}",
                            e
                        ));
                        (Vec::new(), content.to_string(), Vec::new(), None)
                    }
                },
                Err(e) => {
                    logger.log(&format!(
                        "[worker] Failed to init conversation store for history: {}",
                        e
                    ));
                    (Vec::new(), content.to_string(), Vec::new(), None)
                }
            }
        } else {
            (Vec::new(), content.to_string(), Vec::new(), None)
        };



    // Build recall (tools + memories + conversation recall) based on user message
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
        conv_name,
        &window_message_ids,
        conversation_recall_limit,
    )
    .await;

    // Embed last user message for future conversation recall
    if let (Some(cname), Some(uid), Some(emb)) = (conv_name, last_user_msg_id, &recall_result.query_embedding) {
        if let Ok(store) = ConversationStore::init() {
            if let Err(e) = store.store_message_embedding(uid, cname, emb) {
                logger.log(&format!("[worker] Failed to store user message embedding: {}", e));
            }
        }
    }

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

    // With conversation: use unified tool loop (DB-backed context management)
    // Without conversation: fall back to agent.rs tool loop (anima ask without --conversation)
    let (final_response, db_final_response, last_duration_ms, last_tokens_in, last_tokens_out, last_prompt_eval_ns, last_cached_tokens, last_dump_turn_n) = if let Some(cname) = conv_name {
        let (log_tx, log_fwd_handle) = spawn_log_forwarder(logger.clone());
        let start_time = std::time::Instant::now();
        let loop_budget = max_iterations.unwrap_or(25);

        let loop_result = run_tool_loop(
            &final_user_content,
            conversation_history,
            agent,
            agent_name,
            system_prompt,
            recall_result.external_tools,
            use_native_tools,
            loop_budget,
            num_ctx,
            tool_registry,
            &tool_context,
            semantic_memory,
            embedding_client,
            cname,
            token_tx,
            None, // no cancellation for REPL messages
            None, // no response deadline for REPL messages
            start_time,
            shutdown,
            logger,
            log_tx.clone(),
            dedup_lazy,
        ).await;

        drop(log_tx);
        let _ = log_fwd_handle.await;

        (loop_result.response, loop_result.db_response, loop_result.duration_ms, loop_result.tokens_in, loop_result.tokens_out, loop_result.prompt_eval_duration_ns, loop_result.cached_tokens, loop_result.dump_turn_n)
    } else {
        // No conversation — use agent.rs tool loop directly (standalone mode)
        let (trace_tx, trace_rx) =
            tokio::sync::mpsc::channel::<crate::agent::ToolExecution>(32);
        let trace_handle = spawn_tool_trace_persister(
            None,
            agent_name.to_string(),
            num_ctx,
            logger.clone(),
            trace_rx,
        );

        let llm_future = async {
            if use_native_tools {
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
                    max_iterations,
                    num_ctx,
                ).await;
                (result.response.clone(), result.response, result.duration_ms, result.tokens_in, result.tokens_out, result.prompt_eval_duration_ns, result.cached_tokens, None)
            } else {
                let (response, duration_ms, tokens_in, tokens_out, prompt_eval_ns, cached_tokens) = process_json_block_mode(
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
                    max_iterations,
                    num_ctx,
                ).await;
                (response.clone(), response, duration_ms, tokens_in, tokens_out, prompt_eval_ns, cached_tokens, None)
            }
        };

        let result = tokio::select! {
            r = llm_future => r,
            _ = shutdown.notified() => {
                logger.log("[worker] Shutdown during LLM call in process_message_work, aborting");
                return MessageWorkResult {
                    response: String::new(),
                    error: Some("Shutdown during LLM call".to_string()),
                };
            }
        };

        drop(trace_tx);
        let _ = trace_handle.await;

        result
    };

    // Store response in conversation if conv_name was provided
    if let Some(cname) = conv_name {
        let duration_ms = last_duration_ms.map(|d| d as i64);
        let tokens_in = last_tokens_in.map(|t| t as i64);
        let tokens_out = last_tokens_out.map(|t| t as i64);
        let num_ctx_i64 = num_ctx.map(|n| n as i64);
        let prompt_eval_ns = last_prompt_eval_ns.map(|t| t as i64);
        let cached_tokens_i64 = last_cached_tokens.map(|t| t as i64);
        match ConversationStore::init() {
            Ok(store) => {
                // Store recall AFTER response for persistence
                if let Some(ref recall_text) = recall_content_for_storage {
                    if !recall_text.is_empty() {
                        if let Err(e) = store.add_recall_message(cname, recall_text, &agent_name) {
                            logger.log(&format!(
                                "[worker] Failed to store recall in conversation: {}",
                                e
                            ));
                        }
                    }
                }
                // Store the final response with token stats (preserve thinking in DB)
                // Notes already prepended by run_tool_loop — no second prepend_notes here
                match store.add_message_with_tokens(
                    cname,
                    agent_name,
                    &db_final_response,
                    &[],
                    duration_ms,
                    None, // tool_calls stored by run_tool_loop during iterations
                    tokens_in,
                    tokens_out,
                    num_ctx_i64,
                    prompt_eval_ns,
                    cached_tokens_i64,
                ) {
                    Ok(response_msg_id) => {
                        // Rename debug dump files from sequential number to DB message ID
                        if let Some(turn_n) = last_dump_turn_n {
                            agent.lock().await.rename_turn_files(turn_n, response_msg_id);
                        }

                        // Embed agent response for future conversation recall
                        if !final_response.is_empty() {
                            if let Some(emb_client) = embedding_client {
                                match emb_client.embed(&final_response).await {
                                    Ok(emb) => {
                                        if let Err(e) = store.store_message_embedding(response_msg_id, cname, &emb) {
                                            logger.log(&format!("[worker] Failed to store response embedding: {}", e));
                                        }
                                    }
                                    Err(e) => {
                                        logger.log(&format!("[worker] Failed to embed response: {}", e));
                                    }
                                }
                            }
                        }

                        // Parse @mentions from response and forward to other agents
                        if mentions_enabled {
                            let mentions = crate::conversation::parse_mentions(&final_response);
                            let valid_mentions: Vec<String> = mentions
                                .into_iter()
                                .filter(|m| m != agent_name && m != "user" && m != "all")
                                .filter(|m| discovery::agent_exists(m))
                                .collect();

                            if !valid_mentions.is_empty() {
                                logger.log(&format!(
                                    "[worker] Forwarding to {} agents: {:?}",
                                    valid_mentions.len(),
                                    valid_mentions
                                ));

                                for mention in valid_mentions {
                                    let _ = store.add_participant(cname, &mention);
                                    forward_notify_to_agent(
                                        &mention,
                                        cname,
                                        response_msg_id,
                                        0,
                                        logger,
                                    )
                                    .await;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        logger.log(&format!(
                            "[worker] Failed to store response in conversation: {}",
                            e
                        ));
                    }
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
    max_iterations: Option<usize>,
    num_ctx: Option<u32>,
) -> NativeToolModeResult {
    let (log_tx, log_handle) = spawn_log_forwarder(logger.clone());
    let options = ThinkOptions {
        system_prompt: system_prompt.clone(),
        conversation_history: if conversation_history.is_empty() {
            None
        } else {
            Some(conversation_history)
        },
        external_tools,
        tool_trace_tx,
        max_iterations: max_iterations.unwrap_or(25),
        num_ctx,
        log_tx: Some(log_tx),
        ..Default::default()
    };

    let (result, duration_ms, tokens_in, tokens_out, prompt_eval_duration_ns, cached_tokens) = if let Some(tx) = token_tx {
        // Streaming mode - call directly (not spawned) so dropping the future cancels the LLM call
        let mut agent_guard = agent.lock().await;
        match agent_guard
            .think_streaming_with_options(content, options, tx)
            .await
        {
            Ok(result) => {
                drop(agent_guard);
                (result.response, result.duration_ms, result.tokens_in, result.tokens_out, result.prompt_eval_duration_ns, result.cached_tokens)
            }
            Err(e) => {
                drop(agent_guard);
                (format!("Error: {}", e), None, None, None, None, None)
            }
        }
    } else {
        // Non-streaming mode
        let mut agent_guard = agent.lock().await;
        match agent_guard.think_with_options(content, options).await {
            Ok(result) => (result.response, result.duration_ms, result.tokens_in, result.tokens_out, result.prompt_eval_duration_ns, result.cached_tokens),
            Err(e) => (format!("Error: {}", e), None, None, None, None, None),
        }
    };

    // Strip thinking tags and extract [REMEMBER: ...] tags
    let without_thinking = strip_thinking_tags(&result);
    let (after_remember, memories_to_save) = extract_remember_tags(&without_thinking);

    // Save memories
    save_memories(&memories_to_save, semantic_memory, embedding_client, logger).await;

    // Flush log forwarder
    let _ = log_handle.await;

    NativeToolModeResult {
        response: after_remember,
        duration_ms,
        tokens_in,
        tokens_out,
        prompt_eval_duration_ns,
        cached_tokens,
    }
}

/// Guard that aborts a spawned task when dropped, ensuring the LLM call
/// is cancelled on shutdown instead of running to completion in the background.
struct AbortOnDrop<T>(Option<tokio::task::JoinHandle<T>>);
impl<T> AbortOnDrop<T> {
    fn new(handle: tokio::task::JoinHandle<T>) -> Self {
        Self(Some(handle))
    }
    async fn join(&mut self) -> Result<T, tokio::task::JoinError> {
        self.0.take().expect("join called twice").await
    }
}
impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        if let Some(h) = &self.0 {
            h.abort();
        }
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
    max_iterations: Option<usize>,
    num_ctx: Option<u32>,
) -> (String, Option<u64>, Option<u32>, Option<u32>, Option<u64>, Option<u32>) {
    let mut current_message = content.to_string();
    let max_tool_calls = max_iterations.unwrap_or(25);
    let mut tool_call_count = 0;
    #[allow(unused_assignments)]
    let mut last_duration_ms: Option<u64> = None;
    #[allow(unused_assignments)]
    let mut last_tokens_in: Option<u32> = None;
    #[allow(unused_assignments)]
    let mut last_tokens_out: Option<u32> = None;
    #[allow(unused_assignments)]
    let mut last_prompt_eval_ns: Option<u64> = None;
    #[allow(unused_assignments)]
    let mut last_cached_tokens: Option<u32> = None;

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

            let mut handle = AbortOnDrop::new(tokio::spawn(async move {
                let mut agent_guard = agent_clone.lock().await;
                agent_guard
                    .think_streaming_with_options(&current_message_clone, options, internal_tx)
                    .await
            }));

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

            match handle.join().await {
                Ok(Ok(result)) => {
                    last_duration_ms = result.duration_ms;
                    last_tokens_in = result.tokens_in;
                    last_tokens_out = result.tokens_out;
                    last_prompt_eval_ns = result.prompt_eval_duration_ns;
                    last_cached_tokens = result.cached_tokens;
                    result.response
                }
                Ok(Err(e)) => return (format!("Error: {}", e), None, None, None, None, None),
                Err(e) => return (format!("Error: task panicked: {}", e), None, None, None, None, None),
            }
        } else {
            // Non-streaming mode
            let mut agent_guard = agent.lock().await;
            match agent_guard
                .think_with_options(&current_message, options)
                .await
            {
                Ok(result) => {
                    last_duration_ms = result.duration_ms;
                    last_tokens_in = result.tokens_in;
                    last_tokens_out = result.tokens_out;
                    last_prompt_eval_ns = result.prompt_eval_duration_ns;
                    last_cached_tokens = result.cached_tokens;
                    result.response
                }
                Err(e) => return (format!("Error: {}", e), None, None, None, None, None),
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

            if tool_call_count > max_tool_calls {
                logger.tool("[worker] Max tool calls reached, stopping");
                return (cleaned_response, last_duration_ms, last_tokens_in, last_tokens_out, last_prompt_eval_ns, last_cached_tokens);
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
                    let tool_result = truncate_tool_result(&tool_result, num_ctx);
                    logger.tool(&format!("[worker] Result: {} bytes", tool_result.len()));
                    current_message = format!("[Tool Result for {}]\n{}", tc.tool, tool_result);
                }
                Err(e) => {
                    logger.tool(&format!("[worker] Error: {}", e));
                    current_message = format!("[Tool Error for {}]\n{}", tc.tool, e);
                }
            }

            if let Some(nudge) = crate::agent::tool_budget_nudge(tool_call_count, max_tool_calls) {
                current_message.push_str(&format!("\n\n---\n{}", nudge));
            }
        } else {
            return (cleaned_response, last_duration_ms, last_tokens_in, last_tokens_out, last_prompt_eval_ns, last_cached_tokens);
        }
    }
}

/// Maximum depth for @mention chains to prevent infinite loops
const MAX_MENTION_DEPTH: u32 = 100;

/// Default maximum tool result size in bytes before truncation (128 KB).
/// Used as fallback when num_ctx is unknown.
const DEFAULT_MAX_TOOL_RESULT_BYTES: usize = 131_072;

/// Context fill threshold — if (tokens_in + tokens_out) / num_ctx >= this, reset the cursor
const CONTEXT_FILL_THRESHOLD: f64 = 0.90;

/// Truncate a tool result string if it exceeds the budget.
/// Budget = 10% of num_ctx × 4 chars/token, or DEFAULT_MAX_TOOL_RESULT_BYTES as fallback.
/// Keeps the **tail** (where errors and results live) and prepends a truncation notice.
fn truncate_tool_result(result: &str, num_ctx: Option<u32>) -> Cow<'_, str> {
    let max_bytes = num_ctx
        .map(|n| (n as usize / 10) * 4)
        .unwrap_or(DEFAULT_MAX_TOOL_RESULT_BYTES);
    if result.len() <= max_bytes {
        return Cow::Borrowed(result);
    }
    let total = result.len();
    let start = total - max_bytes;
    let safe_start = result.ceil_char_boundary(start);
    let kept = &result[safe_start..];
    Cow::Owned(format!(
        "[output truncated: {}, showing last {}]\n\n{}",
        format_byte_size(total),
        format_byte_size(kept.len()),
        kept,
    ))
}

fn format_byte_size(bytes: usize) -> String {
    if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    }
}

/// Load agent context using append-only cursors for KV cache stability.
///
/// - If cursor is NULL (cold start): load last N messages + pins (sliding window).
/// - If cursor is set: load messages since cursor + pins (append mode).
/// - If zero messages from cursor: fall back to cold start.
fn load_agent_context(
    store: &ConversationStore,
    conv_name: &str,
    agent_name: &str,
    logger: &AgentLogger,
    num_ctx: Option<u32>,
) -> Result<Vec<ConversationMessage>, ConversationError> {
    // Cold start: use token budget (30% of num_ctx) when available, else fall back to message count
    let cold_start = |store: &ConversationStore| -> Result<Vec<ConversationMessage>, ConversationError> {
        if let Some(ctx) = num_ctx {
            let budget = ctx as usize * 30 / 100;
            logger.log(&format!(
                "[context] Cold start with token budget {} (30% of {}) for {} in {}",
                budget, ctx, agent_name, conv_name
            ));
            store.get_messages_by_token_budget(conv_name, budget)
        } else {
            // No num_ctx available — use a conservative default token budget
            store.get_messages_by_token_budget(conv_name, 2048)
        }
    };

    match store.get_context_cursor(conv_name, agent_name)? {
        Some(cursor_id) => {
            let count = store.count_messages_from(conv_name, cursor_id)?;
            if count == 0 {
                // Stale cursor or conversation cleared — cold start with new anchor
                logger.log(&format!(
                    "[context] No messages from cursor for {} in {}, falling back to cold start",
                    agent_name, conv_name
                ));
                store.clear_context_cursor(conv_name, agent_name)?;
                let msgs = cold_start(store)?;
                logger.log(&format!(
                    "[context] Cold start loaded {} messages (id {}..{}) for {} in {}",
                    msgs.len(),
                    msgs.first().map(|m| m.id).unwrap_or(0),
                    msgs.last().map(|m| m.id).unwrap_or(0),
                    agent_name, conv_name
                ));
                set_anchor_cursor(store, conv_name, agent_name, &msgs, logger);
                Ok(msgs)
            } else {
                // Append mode — fetch messages from cursor (inclusive)
                logger.log(&format!(
                    "[context] Append mode for {} in {}: {} messages from cursor {}",
                    agent_name, conv_name, count, cursor_id
                ));
                let msgs = store.get_messages_from_with_pinned(conv_name, cursor_id)?;
                logger.log(&format!(
                    "[context] Append loaded {} messages (id {}..{}) for {} in {}",
                    msgs.len(),
                    msgs.first().map(|m| m.id).unwrap_or(0),
                    msgs.last().map(|m| m.id).unwrap_or(0),
                    agent_name, conv_name
                ));
                Ok(msgs)
            }
        }
        None => {
            // Cold start — no cursor set
            logger.log(&format!(
                "[context] Cold start for {} in {} (no cursor)",
                agent_name, conv_name
            ));
            let msgs = cold_start(store)?;
            logger.log(&format!(
                "[context] Cold start loaded {} messages (id {}..{}) for {} in {}",
                msgs.len(),
                msgs.first().map(|m| m.id).unwrap_or(0),
                msgs.last().map(|m| m.id).unwrap_or(0),
                agent_name, conv_name
            ));
            set_anchor_cursor(store, conv_name, agent_name, &msgs, logger);
            Ok(msgs)
        }
    }
}

/// Set the context cursor to the anchor (oldest non-pinned message) in the window.
fn set_anchor_cursor(
    store: &ConversationStore,
    conv_name: &str,
    agent_name: &str,
    msgs: &[ConversationMessage],
    logger: &AgentLogger,
) {
    if msgs.is_empty() {
        return;
    }
    let anchor_id = msgs
        .iter()
        .filter(|m| !m.pinned)
        .map(|m| m.id)
        .min()
        .unwrap_or(msgs[0].id);
    if let Err(e) = store.set_context_cursor(conv_name, agent_name, anchor_id) {
        logger.log(&format!("[context] Failed to set anchor cursor: {}", e));
    } else {
        logger.log(&format!(
            "[context] Set anchor for {} in {} to msg_id={}",
            agent_name, conv_name, anchor_id
        ));
    }
}

/// Check context fill ratio and reset cursor if threshold exceeded.
///
/// In lazy dedup mode (`dedup_lazy = true`), uses a two-phase approach:
/// 1. First time fill >= threshold: advance dedup_cursor to latest_msg_id
///    (next turn will dedup + strip notes for messages up to that point)
/// 2. If still over threshold after dedup: reset context_cursor + clear dedup_cursor
///
/// In eager mode (`dedup_lazy = false`): always reset context_cursor immediately.
fn check_fill_and_maybe_reset(
    store: &ConversationStore,
    conv_name: &str,
    agent_name: &str,
    tokens_in: Option<i64>,
    tokens_out: Option<i64>,
    num_ctx: Option<i64>,
    dedup_lazy: bool,
    latest_msg_id: Option<i64>,
    logger: &AgentLogger,
) {
    if let (Some(t_in), Some(t_out), Some(ctx)) = (tokens_in, tokens_out, num_ctx) {
        if ctx > 0 {
            let fill = (t_in + t_out) as f64 / ctx as f64;
            if fill >= CONTEXT_FILL_THRESHOLD {
                if dedup_lazy {
                    // Two-phase: try dedup first, then reset cursor if insufficient
                    let current_dedup = store.get_dedup_cursor(conv_name, agent_name).unwrap_or(None);
                    let latest = latest_msg_id.unwrap_or(0);

                    if current_dedup.is_none() {
                        // Phase 1: no dedup cursor yet — set it so next turn applies dedup
                        logger.log(&format!(
                            "[context] Fill {:.0}% >= threshold, advancing dedup cursor to msg_id={} for {} in {}",
                            fill * 100.0, latest, agent_name, conv_name
                        ));
                        if let Err(e) = store.set_dedup_cursor(conv_name, agent_name, latest) {
                            logger.log(&format!("[context] Failed to set dedup cursor: {}", e));
                        }
                    } else {
                        // Phase 2: dedup wasn't enough — reset context cursor
                        logger.log(&format!(
                            "[context] Fill {:.0}% still >= threshold after dedup, resetting cursor for {} in {}",
                            fill * 100.0, agent_name, conv_name
                        ));
                        if let Err(e) = store.clear_context_cursor(conv_name, agent_name) {
                            logger.log(&format!("[context] Failed to clear cursor: {}", e));
                        }
                        if let Err(e) = store.clear_dedup_cursor(conv_name, agent_name) {
                            logger.log(&format!("[context] Failed to clear dedup cursor: {}", e));
                        }
                    }
                } else {
                    // Eager mode: reset cursor immediately
                    logger.log(&format!(
                        "[context] Fill {:.0}% >= threshold {:.0}% for {} in {}, resetting cursor",
                        fill * 100.0,
                        CONTEXT_FILL_THRESHOLD * 100.0,
                        agent_name,
                        conv_name
                    ));
                    if let Err(e) = store.clear_context_cursor(conv_name, agent_name) {
                        logger.log(&format!("[context] Failed to clear cursor: {}", e));
                    }
                }
            }
        }
    }
}

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
    task_store: &Option<Arc<Mutex<TaskStore>>>,
    num_ctx: Option<u32>,
    max_iterations: Option<usize>,
    max_response_time: Option<&str>,
    mentions_enabled: bool,
    shutdown: &Arc<tokio::sync::Notify>,
    conversation_recall_limit: usize,
    dedup_lazy: bool,
) -> Response {
    // Track start time for response duration
    let start_time = std::time::Instant::now();

    // Set current conversation for debug file naming
    {
        let mut agent_guard = agent.lock().await;
        agent_guard.set_current_conversation(Some(conv_id.to_string()));
    }

    // Append conversation name to system prompt so the agent knows where it is
    let system_prompt = &{
        let base = system_prompt.clone().unwrap_or_default();
        Some(format!("{}\nConversation: {}", base, conv_id))
    };

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

    // First fetch: get messages using append-only context cursors
    let context_messages = match load_agent_context(&store, conv_id, agent_name, logger, num_ctx) {
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
    let window_message_ids: Vec<i64> = context_messages.iter().map(|m| m.id).collect();
    let last_user_msg_id = context_messages.iter().rev()
        .find(|m| m.from_agent == "user")
        .map(|m| m.id);
    let dedup_up_to = if dedup_lazy {
        store.get_dedup_cursor(conv_id, agent_name).unwrap_or(None)
    } else {
        context_messages.last().map(|m| m.id)
    };
    let (mut conversation_history, final_user_content) =
        format_conversation_history(&context_messages, agent_name, dedup_up_to);

    // Build recall (tools + memories + conversation recall) based on current user message
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
        Some(conv_id),
        &window_message_ids,
        conversation_recall_limit,
    )
    .await;

    // Embed last user message for future conversation recall
    if let (Some(uid), Some(emb)) = (last_user_msg_id, &recall_result.query_embedding) {
        if let Err(e) = store.store_message_embedding(uid, conv_id, emb) {
            logger.log(&format!("[notify] Failed to store user message embedding: {}", e));
        }
    }

    let recall_content_for_storage = recall_result.recall_content.clone();
    inject_recall_into_history(&recall_result.recall_content, &mut conversation_history);

    logger.log(&format!(
        "[notify] Context: {} messages → {} history + final user turn",
        context_messages.len(),
        conversation_history.len()
    ));

    let external_tools = recall_result.external_tools;

    // Create tool execution context for tools that need daemon state
    let tool_context = ToolExecutionContext {
        agent_name: agent_name.to_string(),
        task_store: task_store.clone(),
        conv_id: Some(conv_id.to_string()),
        semantic_memory_store: semantic_memory.clone(),
        embedding_client: embedding_client.clone(),
        allowed_tools: allowed_tools.clone(),
    };

    // Channel for forwarding log messages to the daemon logger
    let (log_tx, log_fwd_handle) = spawn_log_forwarder(logger.clone());

    // Cancellation flag — checked by run_tool_loop at each iteration boundary
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

    // Parse max_response_time from config
    let response_deadline = max_response_time.and_then(parse_duration);
    let loop_budget = max_iterations.unwrap_or(25);

    // Run unified tool loop
    let loop_result = run_tool_loop(
        &final_user_content,
        conversation_history,
        agent,
        agent_name,
        system_prompt,
        external_tools,
        use_native_tools,
        loop_budget,
        num_ctx,
        tool_registry,
        &tool_context,
        semantic_memory,
        embedding_client,
        conv_id,
        None, // no streaming for daemon notifications
        Some(cancel_flag),
        response_deadline,
        start_time,
        shutdown,
        logger,
        log_tx.clone(),
        dedup_lazy,
    ).await;

    // Clean up pause watcher
    pause_watcher.abort();

    // Flush log forwarder
    drop(log_tx);
    let _ = log_fwd_handle.await;

    let cleaned_response = loop_result.response;
    let db_cleaned_response = loop_result.db_response;
    let duration_ms = loop_result.duration_ms.map(|d| d as i64);
    let tokens_in = loop_result.tokens_in.map(|t| t as i64);
    let tokens_out = loop_result.tokens_out.map(|t| t as i64);
    let num_ctx_i64 = num_ctx.map(|n| n as i64);
    let prompt_eval_ns_i64 = loop_result.prompt_eval_duration_ns.map(|t| t as i64);
    let cached_tokens_i64 = loop_result.cached_tokens.map(|t| t as i64);
    let last_dump_turn_n = loop_result.dump_turn_n;

    // Stamp stats on intermediate messages
    if tokens_in.is_some() || tokens_out.is_some() {
        let _ = store.stamp_unstamped_messages(
            conv_id, agent_name, tokens_in, tokens_out, num_ctx_i64, prompt_eval_ns_i64, cached_tokens_i64,
        );
    }

    // Store recall AFTER response for persistence (but recall was already injected in memory)
    // This positions recall in DB right before the response, preserving context for future turns
    if let Some(ref recall_text) = recall_content_for_storage {
        if !recall_text.is_empty() {
            if let Err(e) = store.add_recall_message(conv_id, recall_text, agent_name) {
                logger.log(&format!(
                    "[notify] Failed to store recall in conversation: {}",
                    e
                ));
            }
        }
    }

    // Notes already prepended by run_tool_loop — no second prepend_notes here
    match store.add_message_with_tokens(
        conv_id,
        agent_name,
        &db_cleaned_response,
        &[],
        duration_ms,
        None, // tool_calls stored by run_tool_loop during iterations
        tokens_in,
        tokens_out,
        num_ctx_i64,
        prompt_eval_ns_i64,
        cached_tokens_i64,
    ) {
        Ok(response_msg_id) => {
            logger.log(&format!(
                "[notify] Stored response as msg_id={} ({}ms)",
                response_msg_id, duration_ms.unwrap_or(0)
            ));

            // Rename debug dump files from sequential number to DB message ID
            if let Some(turn_n) = last_dump_turn_n {
                agent.lock().await.rename_turn_files(turn_n, response_msg_id);
            }

            // Cursor is anchored at cold start; no update needed here.
            // Check fill and reset cursor if context is too full
            check_fill_and_maybe_reset(
                &store,
                conv_id,
                agent_name,
                tokens_in,
                tokens_out,
                num_ctx_i64,
                dedup_lazy,
                Some(response_msg_id),
                logger,
            );

            // Embed agent response for future conversation recall
            if !cleaned_response.is_empty() {
                if let Some(emb_client) = embedding_client {
                    match emb_client.embed(&cleaned_response).await {
                        Ok(emb) => {
                            if let Err(e) = store.store_message_embedding(response_msg_id, conv_id, &emb) {
                                logger.log(&format!("[notify] Failed to store response embedding: {}", e));
                            }
                        }
                        Err(e) => {
                            logger.log(&format!("[notify] Failed to embed response: {}", e));
                        }
                    }
                }
            }

            // Parse @mentions from our response and forward to other agents
            // Only forward if mentions enabled, not paused, and not at depth limit
            let should_forward = mentions_enabled
                && depth < MAX_MENTION_DEPTH
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
    mentions_enabled: bool,
    shutdown: &Arc<tokio::sync::Notify>,
    max_iterations: Option<usize>,
    num_ctx: Option<u32>,
    dedup_lazy: bool,
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

    // 3. Get conversation context using append-only cursors
    // Heartbeat is a self-conversation - all messages are from this agent
    // Format them as assistant messages to show the model its previous outputs
    let context_messages = load_agent_context(&store, &conv_name, agent_name, logger, num_ctx)
        .unwrap_or_default();
    let dedup_up_to = if dedup_lazy {
        store.get_dedup_cursor(&conv_name, agent_name).unwrap_or(None)
    } else {
        context_messages.last().map(|m| m.id)
    };
    let (conversation_history, _) = format_conversation_history(&context_messages, agent_name, dedup_up_to);

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
        None,
        &[],
        0,
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
        max_iterations: max_iterations.unwrap_or(25),
        ..Default::default()
    };

    let llm_future = async {
        let mut agent_guard = agent.lock().await;
        let r = agent_guard
            .think_with_options(&heartbeat_prompt, options)
            .await;
        drop(agent_guard);
        r
    };

    let result = tokio::select! {
        r = llm_future => {
            match r {
                Ok(r) => r,
                Err(e) => {
                    logger.log(&format!("[heartbeat] Think error: {}", e));
                    // Store error in conversation so user can see it via chat view
                    let error_msg = format!("[Error: {}]", e);
                    let _ = store.add_message(&conv_name, agent_name, &error_msg, &[]);
                    return;
                }
            }
        }
        _ = shutdown.notified() => {
            logger.log("[worker] Shutdown during LLM call in run_heartbeat, aborting");
            return;
        }
    };

    // 6. Process response: strip thinking tags, extract memories
    let without_thinking = strip_thinking_tags(&result.response);
    let (cleaned_response, memories_to_save) = extract_remember_tags(&without_thinking);
    let (cleaned_response, llm_notes) = extract_llm_notes(&cleaned_response);
    save_memories(&memories_to_save, semantic_memory, embedding_client, logger).await;

    // Save LLM-generated inline notes to DB
    if let Some(ref notes) = llm_notes {
        let _ = store.set_participant_notes(&conv_name, agent_name, notes);
    }

    // Full response for DB: preserve thinking tags, strip REMEMBER tags
    let (db_content, _) = extract_remember_tags(&result.response);
    let (db_content, _) = extract_llm_notes(&db_content);
    let db_content = prepend_notes(&db_content, &store, &conv_name, agent_name);

    // 7. Store response in <agent>-heartbeat conversation (preserve thinking in DB)
    match store.add_message(&conv_name, agent_name, &db_content, &[]) {
        Ok(msg_id) => {
            logger.log(&format!("[heartbeat] Stored response as msg_id={}", msg_id));
            // Cursor is anchored at cold start; no update needed here
        }
        Err(e) => {
            logger.log(&format!("[heartbeat] Failed to store response: {}", e));
        }
    }

    // Store recall AFTER response so it appears before next user message in history
    if let Some(ref recall_text) = recall_content
        && !recall_text.is_empty()
    {
        if let Err(e) = store.add_recall_message(&conv_name, recall_text, &agent_name) {
            logger.log(&format!(
                "[heartbeat] Failed to store recall in conversation: {}",
                e
            ));
        }
    }

    // 8. Parse @mentions from response and notify (reuse existing logic)
    if mentions_enabled {
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
    heartbeat_config: Option<HeartbeatDaemonConfig>,
    _task_store: Option<Arc<Mutex<TaskStore>>>,
    work_tx: mpsc::UnboundedSender<AgentWork>,
    max_iterations: Option<usize>,
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
                    None,
                    &[],
                    0,
                )
                .await;

                // Note: recall injection is skipped for IncomingMessage - recall should be
                // stored in DB by the caller and included in conversation history when fetched.
                let _ = &recall_result.recall_content;

                let options = ThinkOptions {
                    system_prompt: system_prompt.clone(),
                    conversation_history: None,
                    external_tools: recall_result.external_tools,
                    max_iterations: max_iterations.unwrap_or(25),
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
            pinned: false,
            prompt_eval_ns: None,
            tool_call_id: None,
            cached_tokens: None,
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
            pinned: false,
            prompt_eval_ns: None,
            tool_call_id: None,
            cached_tokens: None,
        }
    }

    #[test]
    fn test_format_conversation_history_empty() {
        let (history, final_content) = format_conversation_history(&[], "arya", Some(i64::MAX));
        assert!(history.is_empty());
        assert!(final_content.is_empty());
    }

    #[test]
    fn test_format_conversation_history_single_user_message() {
        // Single message from user → should become final_content
        let msgs = vec![make_conv_msg("user", "hello")];
        let (history, final_content) = format_conversation_history(&msgs, "arya", Some(i64::MAX));

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
        let (history, final_content) = format_conversation_history(&msgs, "arya", Some(i64::MAX));

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
        let (history, final_content) = format_conversation_history(&msgs, "arya", Some(i64::MAX));

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
        let (history, final_content) = format_conversation_history(&msgs, "arya", Some(i64::MAX));

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
        let (_, final_content) = format_conversation_history(&msgs, "arya", Some(i64::MAX));

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
        let (history, final_content) = format_conversation_history(&msgs, "arya", Some(i64::MAX));

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
            make_conv_msg("recall", "[recalled memories]\n- Memory 1\n- Memory 2"),
            make_conv_msg("arya", "Hi! How can I help?"),
            make_conv_msg("user", "what's up?"),
        ];
        let (history, final_content) = format_conversation_history(&msgs, "arya", Some(i64::MAX));

        // Should have 3 messages in history: assistant (recall), user, assistant (arya)
        assert_eq!(history.len(), 3);
        // Recall should be first — assistant role with raw content (not JSON-wrapped)
        assert_eq!(history[0].role, "assistant");
        assert!(history[0].content.as_ref().unwrap().contains("[recalled memories]"));
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

    #[test]
    fn test_format_conversation_history_other_agent_recall_filtered() {
        // Recall messages triggered by another agent should be excluded.
        // Unattributed recall (triggered_by = None) should still be included for backward compat.
        let mut own_recall = make_conv_msg("recall", "[recalled memories]\n- My memory");
        own_recall.triggered_by = Some("arya".to_string());

        let mut other_recall = make_conv_msg("recall", "[recalled memories]\n- Gendry's memory");
        other_recall.triggered_by = Some("gendry".to_string());

        let unattributed_recall = make_conv_msg("recall", "[recalled memories]\n- Old memory");

        let msgs = vec![
            make_conv_msg("user", "hello"),
            own_recall,
            other_recall,
            unattributed_recall,
            make_conv_msg("arya", "Hi!"),
            make_conv_msg("user", "what's up?"),
        ];
        let (history, _final_content) = format_conversation_history(&msgs, "arya", Some(i64::MAX));

        // Should have 4 messages: own recall, unattributed recall, user, assistant
        // (other_recall from gendry is filtered out)
        assert_eq!(history.len(), 4);
        // First two are recall (assistant role)
        assert_eq!(history[0].role, "assistant");
        assert!(history[0].content.as_ref().unwrap().contains("My memory"));
        assert_eq!(history[1].role, "assistant");
        assert!(history[1].content.as_ref().unwrap().contains("Old memory"));
        // Gendry's recall should NOT appear anywhere
        let all_content: String = history
            .iter()
            .filter_map(|m| m.content.as_ref())
            .cloned()
            .collect();
        assert!(!all_content.contains("Gendry's memory"));
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
        let (history, _final_content) = format_conversation_history(&msgs, "arya", Some(i64::MAX));

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
        let (history, final_content) = format_conversation_history(&msgs, "arya", Some(i64::MAX));

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
        let (history, _final_content) = format_conversation_history(&msgs, "arya", Some(i64::MAX));

        // Unattributed tool result should be included
        assert_eq!(history.len(), 3);
        assert!(history[2].content.as_ref().unwrap().contains("old data"));
    }

    #[test]
    fn test_format_conversation_history_dedup_read_file() {
        // Agent reads /a.rs twice — first pair (assistant + tool result) should be dropped entirely.
        let read_a_tc = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc1".into(),
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "/a.rs"}),
        }])
        .unwrap();
        let read_a_tc2 = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc2".into(),
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "/a.rs"}),
        }])
        .unwrap();

        let mut msg_agent1 = make_conv_msg("arya", "Reading file.");
        msg_agent1.id = 10;
        msg_agent1.tool_calls = Some(read_a_tc);

        let mut msg_tool1 = make_tool_msg(
            "[Tool Result for read_file]\nfn main() { old version }",
            Some("arya"),
        );
        msg_tool1.id = 11;
        msg_tool1.tool_call_id = Some("tc1".into());

        let mut msg_agent2 = make_conv_msg("arya", "Re-reading file.");
        msg_agent2.id = 12;
        msg_agent2.tool_calls = Some(read_a_tc2);

        let mut msg_tool2 = make_tool_msg(
            "[Tool Result for read_file]\nfn main() { new version }",
            Some("arya"),
        );
        msg_tool2.id = 13;
        msg_tool2.tool_call_id = Some("tc2".into());

        let msgs = vec![
            make_conv_msg("user", "read /a.rs"),
            msg_agent1,
            msg_tool1,
            make_conv_msg("user", "read it again"),
            msg_agent2,
            msg_tool2,
            make_conv_msg("user", "thanks"),
        ];

        let (history, _final_content) = format_conversation_history(&msgs, "arya", Some(i64::MAX));

        // First pair should be dropped entirely — only one read_file result in output
        let tool_results: Vec<&str> = history
            .iter()
            .filter(|m| {
                m.role == "tool"
                    && m.content
                        .as_ref()
                        .map_or(false, |c| c.contains("read_file"))
            })
            .map(|m| m.content.as_ref().unwrap().as_str())
            .collect();

        assert_eq!(
            tool_results.len(),
            1,
            "Should have only one read_file result (first pair dropped)"
        );
        assert!(
            tool_results[0].contains("new version"),
            "Kept result should be the latest: {}",
            tool_results[0]
        );

        // The first assistant message (id=10) should also be dropped
        let assistant_msgs: Vec<&str> = history
            .iter()
            .filter(|m| m.role == "assistant")
            .filter_map(|m| m.content.as_deref())
            .collect();
        assert!(
            !assistant_msgs.contains(&"Reading file."),
            "First assistant tool_call message should be dropped"
        );
        assert!(
            assistant_msgs.contains(&"Re-reading file."),
            "Second assistant tool_call message should be kept"
        );
    }

    #[test]
    fn test_format_conversation_history_dedup_different_paths_untouched() {
        // Agent reads /a.rs then /b.rs — both should be kept.
        let read_a_tc = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc1".into(),
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "/a.rs"}),
        }])
        .unwrap();
        let read_b_tc = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc2".into(),
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "/b.rs"}),
        }])
        .unwrap();

        let mut msg_agent1 = make_conv_msg("arya", "Reading a.");
        msg_agent1.id = 10;
        msg_agent1.tool_calls = Some(read_a_tc);

        let mut msg_tool1 = make_tool_msg(
            "[Tool Result for read_file]\ncontents of a",
            Some("arya"),
        );
        msg_tool1.id = 11;
        msg_tool1.tool_call_id = Some("tc1".into());

        let mut msg_agent2 = make_conv_msg("arya", "Reading b.");
        msg_agent2.id = 12;
        msg_agent2.tool_calls = Some(read_b_tc);

        let mut msg_tool2 = make_tool_msg(
            "[Tool Result for read_file]\ncontents of b",
            Some("arya"),
        );
        msg_tool2.id = 13;
        msg_tool2.tool_call_id = Some("tc2".into());

        let msgs = vec![
            make_conv_msg("user", "read both"),
            msg_agent1,
            msg_tool1,
            msg_agent2,
            msg_tool2,
            make_conv_msg("user", "thanks"),
        ];

        let (history, _) = format_conversation_history(&msgs, "arya", Some(i64::MAX));

        // Both should be present (different paths, no dedup)
        let tool_results: Vec<&str> = history
            .iter()
            .filter(|m| {
                m.role == "tool"
                    && m.content
                        .as_ref()
                        .map_or(false, |c| c.starts_with("[Tool Result for read_file]"))
            })
            .map(|m| m.content.as_ref().unwrap().as_str())
            .collect();

        assert_eq!(tool_results.len(), 2, "Both reads should be kept");
        assert!(tool_results[0].contains("contents of a"));
        assert!(tool_results[1].contains("contents of b"));
    }

    #[test]
    fn test_format_conversation_history_dedup_write_file() {
        // Two writes to same path — first pair dropped, second kept.
        let write_tc1 = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc1".into(),
            name: "write_file".into(),
            arguments: serde_json::json!({"path": "/a.rs", "content": "v1"}),
        }])
        .unwrap();
        let write_tc2 = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc2".into(),
            name: "write_file".into(),
            arguments: serde_json::json!({"path": "/a.rs", "content": "v2"}),
        }])
        .unwrap();

        let mut msg_agent1 = make_conv_msg("arya", "Writing v1.");
        msg_agent1.id = 10;
        msg_agent1.tool_calls = Some(write_tc1);

        let mut msg_tool1 = make_tool_msg(
            "[Tool Result for write_file]\nWrote 1 lines to /a.rs",
            Some("arya"),
        );
        msg_tool1.id = 11;
        msg_tool1.tool_call_id = Some("tc1".into());

        let mut msg_agent2 = make_conv_msg("arya", "Writing v2.");
        msg_agent2.id = 12;
        msg_agent2.tool_calls = Some(write_tc2);

        let mut msg_tool2 = make_tool_msg(
            "[Tool Result for write_file]\nWrote 2 lines to /a.rs",
            Some("arya"),
        );
        msg_tool2.id = 13;
        msg_tool2.tool_call_id = Some("tc2".into());

        let msgs = vec![
            make_conv_msg("user", "write /a.rs"),
            msg_agent1,
            msg_tool1,
            msg_agent2,
            msg_tool2,
            make_conv_msg("user", "thanks"),
        ];

        let (history, _) = format_conversation_history(&msgs, "arya", Some(i64::MAX));

        let tool_results: Vec<&str> = history
            .iter()
            .filter(|m| {
                m.role == "tool"
                    && m.content
                        .as_ref()
                        .map_or(false, |c| c.starts_with("[Tool Result for write_file]"))
            })
            .map(|m| m.content.as_ref().unwrap().as_str())
            .collect();

        assert_eq!(tool_results.len(), 1, "First write pair should be dropped");
        assert!(tool_results[0].contains("2 lines"));
    }

    #[test]
    fn test_format_conversation_history_dedup_edit_file_dropped_after_read() {
        // edit → edit → full read of same path → both edit pairs dropped
        let edit_tc1 = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc1".into(),
            name: "edit_file".into(),
            arguments: serde_json::json!({"path": "/a.rs", "old_str": "a", "new_str": "b"}),
        }])
        .unwrap();
        let edit_tc2 = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc2".into(),
            name: "edit_file".into(),
            arguments: serde_json::json!({"path": "/a.rs", "old_str": "b", "new_str": "c"}),
        }])
        .unwrap();
        let read_tc = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc3".into(),
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "/a.rs"}),
        }])
        .unwrap();

        let mut msg_a1 = make_conv_msg("arya", "Edit 1.");
        msg_a1.id = 10;
        msg_a1.tool_calls = Some(edit_tc1);
        let mut msg_t1 =
            make_tool_msg("[Tool Result for edit_file]\ndiff1", Some("arya"));
        msg_t1.id = 11;
        msg_t1.tool_call_id = Some("tc1".into());

        let mut msg_a2 = make_conv_msg("arya", "Edit 2.");
        msg_a2.id = 12;
        msg_a2.tool_calls = Some(edit_tc2);
        let mut msg_t2 =
            make_tool_msg("[Tool Result for edit_file]\ndiff2", Some("arya"));
        msg_t2.id = 13;
        msg_t2.tool_call_id = Some("tc2".into());

        let mut msg_a3 = make_conv_msg("arya", "Reading file.");
        msg_a3.id = 14;
        msg_a3.tool_calls = Some(read_tc);
        let mut msg_t3 = make_tool_msg(
            "[Tool Result for read_file]\nfinal contents",
            Some("arya"),
        );
        msg_t3.id = 15;
        msg_t3.tool_call_id = Some("tc3".into());

        let msgs = vec![
            make_conv_msg("user", "fix /a.rs"),
            msg_a1, msg_t1, msg_a2, msg_t2, msg_a3, msg_t3,
            make_conv_msg("user", "thanks"),
        ];

        let (history, _) = format_conversation_history(&msgs, "arya", Some(i64::MAX));

        // Both edit pairs dropped; read kept
        let edit_results: Vec<_> = history
            .iter()
            .filter(|m| {
                m.content
                    .as_ref()
                    .map_or(false, |c| c.contains("edit_file"))
            })
            .collect();
        assert_eq!(edit_results.len(), 0, "Both edit pairs should be dropped");

        let read_results: Vec<_> = history
            .iter()
            .filter(|m| {
                m.content
                    .as_ref()
                    .map_or(false, |c| c.contains("read_file"))
            })
            .collect();
        assert_eq!(read_results.len(), 1, "Read should be kept");
        assert!(read_results[0]
            .content
            .as_ref()
            .unwrap()
            .contains("final contents"));
    }

    #[test]
    fn test_format_conversation_history_dedup_edit_file_kept_when_no_later_read() {
        // edit → edit → no read → both edits kept
        let edit_tc1 = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc1".into(),
            name: "edit_file".into(),
            arguments: serde_json::json!({"path": "/a.rs", "old_str": "a", "new_str": "b"}),
        }])
        .unwrap();
        let edit_tc2 = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc2".into(),
            name: "edit_file".into(),
            arguments: serde_json::json!({"path": "/a.rs", "old_str": "b", "new_str": "c"}),
        }])
        .unwrap();

        let mut msg_a1 = make_conv_msg("arya", "Edit 1.");
        msg_a1.id = 10;
        msg_a1.tool_calls = Some(edit_tc1);
        let mut msg_t1 =
            make_tool_msg("[Tool Result for edit_file]\ndiff1", Some("arya"));
        msg_t1.id = 11;
        msg_t1.tool_call_id = Some("tc1".into());

        let mut msg_a2 = make_conv_msg("arya", "Edit 2.");
        msg_a2.id = 12;
        msg_a2.tool_calls = Some(edit_tc2);
        let mut msg_t2 =
            make_tool_msg("[Tool Result for edit_file]\ndiff2", Some("arya"));
        msg_t2.id = 13;
        msg_t2.tool_call_id = Some("tc2".into());

        let msgs = vec![
            make_conv_msg("user", "fix /a.rs"),
            msg_a1, msg_t1, msg_a2, msg_t2,
            make_conv_msg("user", "thanks"),
        ];

        let (history, _) = format_conversation_history(&msgs, "arya", Some(i64::MAX));

        let edit_results: Vec<_> = history
            .iter()
            .filter(|m| {
                m.content
                    .as_ref()
                    .map_or(false, |c| c.contains("edit_file"))
            })
            .collect();
        assert_eq!(edit_results.len(), 2, "Both edits should be kept (no later read)");
    }

    #[test]
    fn test_format_conversation_history_dedup_identical_shell() {
        // Two shell commands differing only by tail suffix — first pair dropped, second kept.
        let cargo_tc1 = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc1".into(),
            name: "shell".into(),
            arguments: serde_json::json!({"command": "cd ~/dev/minilang && cargo check 2>&1 | tail -5"}),
        }])
        .unwrap();
        let cargo_tc2 = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc2".into(),
            name: "shell".into(),
            arguments: serde_json::json!({"command": "cd ~/dev/minilang && cargo check 2>&1"}),
        }])
        .unwrap();

        let mut msg_a1 = make_conv_msg("arya", "Checking.");
        msg_a1.id = 10;
        msg_a1.tool_calls = Some(cargo_tc1);
        let mut msg_t1 = make_tool_msg(
            "[Tool Result for shell]\nerror[E0308]: mismatched types",
            Some("arya"),
        );
        msg_t1.id = 11;
        msg_t1.tool_call_id = Some("tc1".into());

        let mut msg_a2 = make_conv_msg("arya", "Re-checking.");
        msg_a2.id = 12;
        msg_a2.tool_calls = Some(cargo_tc2);
        let mut msg_t2 = make_tool_msg(
            "[Tool Result for shell]\nCompiling OK",
            Some("arya"),
        );
        msg_t2.id = 13;
        msg_t2.tool_call_id = Some("tc2".into());

        let msgs = vec![
            make_conv_msg("user", "fix it"),
            msg_a1, msg_t1, msg_a2, msg_t2,
            make_conv_msg("user", "thanks"),
        ];

        let (history, _) = format_conversation_history(&msgs, "arya", Some(i64::MAX));

        let shell_results: Vec<&str> = history
            .iter()
            .filter(|m| {
                m.content
                    .as_ref()
                    .map_or(false, |c| c.starts_with("[Tool Result for shell]"))
            })
            .map(|m| m.content.as_ref().unwrap().as_str())
            .collect();

        assert_eq!(shell_results.len(), 1, "First cargo check pair should be dropped");
        assert!(shell_results[0].contains("Compiling OK"));
    }

    #[test]
    fn test_format_conversation_history_dedup_identical_non_cargo_shell() {
        // Two identical non-cargo shell commands — first pair dropped, second kept.
        let ls_tc1 = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc1".into(),
            name: "shell".into(),
            arguments: serde_json::json!({"command": "ls"}),
        }])
        .unwrap();
        let ls_tc2 = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc2".into(),
            name: "shell".into(),
            arguments: serde_json::json!({"command": "ls"}),
        }])
        .unwrap();

        let mut msg_a1 = make_conv_msg("arya", "Listing.");
        msg_a1.id = 10;
        msg_a1.tool_calls = Some(ls_tc1);
        let mut msg_t1 =
            make_tool_msg("[Tool Result for shell]\nfile1.rs", Some("arya"));
        msg_t1.id = 11;
        msg_t1.tool_call_id = Some("tc1".into());

        let mut msg_a2 = make_conv_msg("arya", "Listing again.");
        msg_a2.id = 12;
        msg_a2.tool_calls = Some(ls_tc2);
        let mut msg_t2 =
            make_tool_msg("[Tool Result for shell]\nfile1.rs file2.rs", Some("arya"));
        msg_t2.id = 13;
        msg_t2.tool_call_id = Some("tc2".into());

        let msgs = vec![
            make_conv_msg("user", "list files"),
            msg_a1, msg_t1, msg_a2, msg_t2,
            make_conv_msg("user", "thanks"),
        ];

        let (history, _) = format_conversation_history(&msgs, "arya", Some(i64::MAX));

        let shell_results: Vec<_> = history
            .iter()
            .filter(|m| {
                m.content
                    .as_ref()
                    .map_or(false, |c| c.starts_with("[Tool Result for shell]"))
            })
            .collect();

        assert_eq!(shell_results.len(), 1, "Identical shell commands should be deduped to last");
        assert!(shell_results[0].content.as_ref().unwrap().contains("file2.rs"));
    }

    #[test]
    fn test_format_conversation_history_dedup_different_shell_commands_kept() {
        // Two different shell commands — both survive.
        let ls_tc = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc1".into(),
            name: "shell".into(),
            arguments: serde_json::json!({"command": "ls"}),
        }])
        .unwrap();
        let git_tc = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc2".into(),
            name: "shell".into(),
            arguments: serde_json::json!({"command": "git status"}),
        }])
        .unwrap();

        let mut msg_a1 = make_conv_msg("arya", "Listing.");
        msg_a1.id = 10;
        msg_a1.tool_calls = Some(ls_tc);
        let mut msg_t1 =
            make_tool_msg("[Tool Result for shell]\nfile1.rs", Some("arya"));
        msg_t1.id = 11;
        msg_t1.tool_call_id = Some("tc1".into());

        let mut msg_a2 = make_conv_msg("arya", "Checking status.");
        msg_a2.id = 12;
        msg_a2.tool_calls = Some(git_tc);
        let mut msg_t2 =
            make_tool_msg("[Tool Result for shell]\nOn branch main", Some("arya"));
        msg_t2.id = 13;
        msg_t2.tool_call_id = Some("tc2".into());

        let msgs = vec![
            make_conv_msg("user", "check things"),
            msg_a1, msg_t1, msg_a2, msg_t2,
            make_conv_msg("user", "thanks"),
        ];

        let (history, _) = format_conversation_history(&msgs, "arya", Some(i64::MAX));

        let shell_results: Vec<_> = history
            .iter()
            .filter(|m| {
                m.content
                    .as_ref()
                    .map_or(false, |c| c.starts_with("[Tool Result for shell]"))
            })
            .collect();

        assert_eq!(shell_results.len(), 2, "Different shell commands should both survive");
    }

    #[test]
    fn test_format_conversation_history_dedup_shell_normalization() {
        // Two commands differing only by | head -N suffix — first pair dropped via normalization.
        let tc1_json = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc1".into(),
            name: "shell".into(),
            arguments: serde_json::json!({"command": "./target/release/minilang examples/demo.mini | head -20"}),
        }])
        .unwrap();
        let tc2_json = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc2".into(),
            name: "shell".into(),
            arguments: serde_json::json!({"command": "./target/release/minilang examples/demo.mini | head -50"}),
        }])
        .unwrap();

        let mut msg_a1 = make_conv_msg("arya", "Running.");
        msg_a1.id = 10;
        msg_a1.tool_calls = Some(tc1_json);
        let mut msg_t1 = make_tool_msg(
            "[Tool Result for shell]\noutput truncated",
            Some("arya"),
        );
        msg_t1.id = 11;
        msg_t1.tool_call_id = Some("tc1".into());

        let mut msg_a2 = make_conv_msg("arya", "Running again.");
        msg_a2.id = 12;
        msg_a2.tool_calls = Some(tc2_json);
        let mut msg_t2 = make_tool_msg(
            "[Tool Result for shell]\nfull output here",
            Some("arya"),
        );
        msg_t2.id = 13;
        msg_t2.tool_call_id = Some("tc2".into());

        let msgs = vec![
            make_conv_msg("user", "run it"),
            msg_a1, msg_t1, msg_a2, msg_t2,
            make_conv_msg("user", "thanks"),
        ];

        let (history, _) = format_conversation_history(&msgs, "arya", Some(i64::MAX));

        let shell_results: Vec<&str> = history
            .iter()
            .filter(|m| {
                m.content
                    .as_ref()
                    .map_or(false, |c| c.starts_with("[Tool Result for shell]"))
            })
            .map(|m| m.content.as_ref().unwrap().as_str())
            .collect();

        assert_eq!(shell_results.len(), 1, "Same base command with different | head should be deduped");
        assert!(shell_results[0].contains("full output here"));
    }

    #[test]
    fn test_format_conversation_history_dedup_unknown_tool() {
        // Three calls to a hallucinated "run" tool — first two pairs dropped, last kept.
        let run_tc1 = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc1".into(),
            name: "run".into(),
            arguments: serde_json::json!({"command": "cargo test"}),
        }])
        .unwrap();
        let run_tc2 = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc2".into(),
            name: "run".into(),
            arguments: serde_json::json!({"command": "cargo build"}),
        }])
        .unwrap();
        let run_tc3 = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc3".into(),
            name: "run".into(),
            arguments: serde_json::json!({"command": "cargo check"}),
        }])
        .unwrap();

        let mut msg_a1 = make_conv_msg("arya", "Running tests.");
        msg_a1.id = 10;
        msg_a1.tool_calls = Some(run_tc1);
        let mut msg_t1 = make_tool_msg(
            "[Tool Error for run]\nUnknown tool: run",
            Some("arya"),
        );
        msg_t1.id = 11;
        msg_t1.tool_call_id = Some("tc1".into());

        let mut msg_a2 = make_conv_msg("arya", "Building.");
        msg_a2.id = 12;
        msg_a2.tool_calls = Some(run_tc2);
        let mut msg_t2 = make_tool_msg(
            "[Tool Error for run]\nUnknown tool: run",
            Some("arya"),
        );
        msg_t2.id = 13;
        msg_t2.tool_call_id = Some("tc2".into());

        let mut msg_a3 = make_conv_msg("arya", "Checking.");
        msg_a3.id = 14;
        msg_a3.tool_calls = Some(run_tc3);
        let mut msg_t3 = make_tool_msg(
            "[Tool Error for run]\nUnknown tool: run",
            Some("arya"),
        );
        msg_t3.id = 15;
        msg_t3.tool_call_id = Some("tc3".into());

        let msgs = vec![
            make_conv_msg("user", "run the tests"),
            msg_a1, msg_t1, msg_a2, msg_t2, msg_a3, msg_t3,
            make_conv_msg("user", "use shell instead"),
        ];

        let (history, _) = format_conversation_history(&msgs, "arya", Some(i64::MAX));

        // Only the last unknown-tool error pair should survive
        let error_results: Vec<&str> = history
            .iter()
            .filter(|m| {
                m.content
                    .as_ref()
                    .map_or(false, |c| c.contains("Unknown tool: run"))
            })
            .map(|m| m.content.as_ref().unwrap().as_str())
            .collect();

        assert_eq!(error_results.len(), 1, "Only last unknown-tool error should survive");

        // The assistant messages for the first two calls should also be dropped
        let assistant_msgs: Vec<&str> = history
            .iter()
            .filter(|m| m.role == "assistant")
            .filter_map(|m| m.content.as_deref())
            .collect();

        assert!(!assistant_msgs.contains(&"Running tests."), "First hallucinated call should be dropped");
        assert!(!assistant_msgs.contains(&"Building."), "Second hallucinated call should be dropped");
        assert!(assistant_msgs.contains(&"Checking."), "Last hallucinated call should be kept");
    }

    #[test]
    fn test_format_conversation_history_dedup_notes() {
        // Three notes calls — first two pairs dropped, last kept.
        let notes_tc1 = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc1".into(),
            name: "notes".into(),
            arguments: serde_json::json!({"content": "first notes"}),
        }])
        .unwrap();
        let notes_tc2 = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc2".into(),
            name: "notes".into(),
            arguments: serde_json::json!({"content": "second notes"}),
        }])
        .unwrap();
        let notes_tc3 = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc3".into(),
            name: "notes".into(),
            arguments: serde_json::json!({"content": "third notes"}),
        }])
        .unwrap();

        let mut msg_a1 = make_conv_msg("arya", "Saving notes 1.");
        msg_a1.id = 10;
        msg_a1.tool_calls = Some(notes_tc1);
        let mut msg_t1 = make_tool_msg("Notes updated.", Some("arya"));
        msg_t1.id = 11;
        msg_t1.tool_call_id = Some("tc1".into());

        let mut msg_a2 = make_conv_msg("arya", "Saving notes 2.");
        msg_a2.id = 12;
        msg_a2.tool_calls = Some(notes_tc2);
        let mut msg_t2 = make_tool_msg("Notes updated.", Some("arya"));
        msg_t2.id = 13;
        msg_t2.tool_call_id = Some("tc2".into());

        let mut msg_a3 = make_conv_msg("arya", "Saving notes 3.");
        msg_a3.id = 14;
        msg_a3.tool_calls = Some(notes_tc3);
        let mut msg_t3 = make_tool_msg("Notes updated.", Some("arya"));
        msg_t3.id = 15;
        msg_t3.tool_call_id = Some("tc3".into());

        let msgs = vec![
            make_conv_msg("user", "do some work"),
            msg_a1, msg_t1, msg_a2, msg_t2, msg_a3, msg_t3,
        ];

        let (history, _) = format_conversation_history(&msgs, "arya", Some(i64::MAX));

        // Only the last notes result should survive
        let notes_results: Vec<_> = history
            .iter()
            .filter(|m| m.role == "tool" && m.content.as_deref() == Some("Notes updated."))
            .collect();
        assert_eq!(notes_results.len(), 1, "Only last notes result should survive");

        // The assistant messages for the first two calls should also be dropped
        let assistant_msgs: Vec<&str> = history
            .iter()
            .filter(|m| m.role == "assistant")
            .filter_map(|m| m.content.as_deref())
            .collect();
        assert!(!assistant_msgs.contains(&"Saving notes 1."), "First notes call should be dropped");
        assert!(!assistant_msgs.contains(&"Saving notes 2."), "Second notes call should be dropped");
        assert!(assistant_msgs.contains(&"Saving notes 3."), "Last notes call should be kept");
    }

    #[test]
    fn test_format_conversation_history_dedup_read_range_superseded_by_full() {
        // Range read → full read of same path → range pair dropped.
        let range_tc = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc1".into(),
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "/a.rs", "start_line": 1, "end_line": 10}),
        }])
        .unwrap();
        let full_tc = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc2".into(),
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "/a.rs"}),
        }])
        .unwrap();

        let mut msg_a1 = make_conv_msg("arya", "Range read.");
        msg_a1.id = 10;
        msg_a1.tool_calls = Some(range_tc);
        let mut msg_t1 = make_tool_msg(
            "[Tool Result for read_file]\nlines 1-10",
            Some("arya"),
        );
        msg_t1.id = 11;
        msg_t1.tool_call_id = Some("tc1".into());

        let mut msg_a2 = make_conv_msg("arya", "Full read.");
        msg_a2.id = 12;
        msg_a2.tool_calls = Some(full_tc);
        let mut msg_t2 = make_tool_msg(
            "[Tool Result for read_file]\nfull contents",
            Some("arya"),
        );
        msg_t2.id = 13;
        msg_t2.tool_call_id = Some("tc2".into());

        let msgs = vec![
            make_conv_msg("user", "read /a.rs"),
            msg_a1, msg_t1, msg_a2, msg_t2,
            make_conv_msg("user", "thanks"),
        ];

        let (history, _) = format_conversation_history(&msgs, "arya", Some(i64::MAX));

        let read_results: Vec<&str> = history
            .iter()
            .filter(|m| {
                m.content
                    .as_ref()
                    .map_or(false, |c| c.starts_with("[Tool Result for read_file]"))
            })
            .map(|m| m.content.as_ref().unwrap().as_str())
            .collect();

        assert_eq!(read_results.len(), 1, "Range read should be dropped");
        assert!(read_results[0].contains("full contents"));
    }

    #[test]
    fn test_format_conversation_history_dedup_write_supersedes_range_read() {
        // Range read → write_file to same path → range pair dropped.
        let range_tc = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc1".into(),
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "/a.rs", "start_line": 5}),
        }])
        .unwrap();
        let write_tc = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc2".into(),
            name: "write_file".into(),
            arguments: serde_json::json!({"path": "/a.rs", "content": "new"}),
        }])
        .unwrap();

        let mut msg_a1 = make_conv_msg("arya", "Range read.");
        msg_a1.id = 10;
        msg_a1.tool_calls = Some(range_tc);
        let mut msg_t1 = make_tool_msg(
            "[Tool Result for read_file]\npartial contents",
            Some("arya"),
        );
        msg_t1.id = 11;
        msg_t1.tool_call_id = Some("tc1".into());

        let mut msg_a2 = make_conv_msg("arya", "Writing file.");
        msg_a2.id = 12;
        msg_a2.tool_calls = Some(write_tc);
        let mut msg_t2 = make_tool_msg(
            "[Tool Result for write_file]\nWrote 1 lines to /a.rs",
            Some("arya"),
        );
        msg_t2.id = 13;
        msg_t2.tool_call_id = Some("tc2".into());

        let msgs = vec![
            make_conv_msg("user", "update /a.rs"),
            msg_a1, msg_t1, msg_a2, msg_t2,
            make_conv_msg("user", "thanks"),
        ];

        let (history, _) = format_conversation_history(&msgs, "arya", Some(i64::MAX));

        let read_results: Vec<_> = history
            .iter()
            .filter(|m| {
                m.content
                    .as_ref()
                    .map_or(false, |c| c.starts_with("[Tool Result for read_file]"))
            })
            .collect();
        assert_eq!(read_results.len(), 0, "Range read should be dropped (superseded by write)");

        let write_results: Vec<_> = history
            .iter()
            .filter(|m| {
                m.content
                    .as_ref()
                    .map_or(false, |c| c.starts_with("[Tool Result for write_file]"))
            })
            .collect();
        assert_eq!(write_results.len(), 1, "Write should be kept");
    }

    #[test]
    fn test_format_conversation_history_dedup_write_supersedes_read() {
        // write_file to path X supersedes earlier read_file of path X.
        // Both are "fresh views" — the write is newer, so the read is dropped.
        let read_tc = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc1".into(),
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "/a.rs"}),
        }])
        .unwrap();
        let write_tc = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc2".into(),
            name: "write_file".into(),
            arguments: serde_json::json!({"path": "/a.rs", "content": "new"}),
        }])
        .unwrap();

        let mut msg_a1 = make_conv_msg("arya", "Reading.");
        msg_a1.id = 10;
        msg_a1.tool_calls = Some(read_tc);
        let mut msg_t1 = make_tool_msg(
            "[Tool Result for read_file]\nold contents",
            Some("arya"),
        );
        msg_t1.id = 11;
        msg_t1.tool_call_id = Some("tc1".into());

        let mut msg_a2 = make_conv_msg("arya", "Writing.");
        msg_a2.id = 12;
        msg_a2.tool_calls = Some(write_tc);
        let mut msg_t2 = make_tool_msg(
            "[Tool Result for write_file]\nWrote 1 lines",
            Some("arya"),
        );
        msg_t2.id = 13;
        msg_t2.tool_call_id = Some("tc2".into());

        let msgs = vec![
            make_conv_msg("user", "update /a.rs"),
            msg_a1, msg_t1, msg_a2, msg_t2,
            make_conv_msg("user", "thanks"),
        ];

        let (history, _) = format_conversation_history(&msgs, "arya", Some(i64::MAX));

        // Unified fresh-view dedup: read and write both provide a full view of /a.rs.
        // The write is newer, so the earlier read is superseded and dropped.
        // The assistant message for the read (id=10, single tool_call) is also dropped.
        let read_results: Vec<_> = history
            .iter()
            .filter(|m| {
                m.content
                    .as_ref()
                    .map_or(false, |c| c.starts_with("[Tool Result for read_file]"))
            })
            .collect();
        assert_eq!(read_results.len(), 0, "Read before write is superseded");

        let write_results: Vec<_> = history
            .iter()
            .filter(|m| {
                m.content
                    .as_ref()
                    .map_or(false, |c| c.starts_with("[Tool Result for write_file]"))
            })
            .collect();
        assert_eq!(write_results.len(), 1, "Write should be kept");

        // Assistant message for the read (id=10) should also be dropped
        let asst_contents: Vec<&str> = history
            .iter()
            .filter(|m| m.role == "assistant")
            .filter_map(|m| m.content.as_deref())
            .collect();
        assert!(!asst_contents.contains(&"Reading."), "Read's assistant msg should be dropped");
        assert!(asst_contents.contains(&"Writing."), "Write's assistant msg should be kept");
    }

    #[test]
    fn test_format_conversation_history_dedup_read_supersedes_write() {
        // read_file of path X supersedes earlier write_file of path X.
        // Both are "fresh views" — the read is newer, so the write is dropped.
        let write_tc = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc1".into(),
            name: "write_file".into(),
            arguments: serde_json::json!({"path": "/a.rs", "content": "new"}),
        }])
        .unwrap();
        let read_tc = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc2".into(),
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "/a.rs"}),
        }])
        .unwrap();

        let mut msg_a1 = make_conv_msg("arya", "Writing.");
        msg_a1.id = 10;
        msg_a1.tool_calls = Some(write_tc);
        let mut msg_t1 = make_tool_msg(
            "[Tool Result for write_file]\nWrote 1 lines",
            Some("arya"),
        );
        msg_t1.id = 11;
        msg_t1.tool_call_id = Some("tc1".into());

        let mut msg_a2 = make_conv_msg("arya", "Reading.");
        msg_a2.id = 12;
        msg_a2.tool_calls = Some(read_tc);
        let mut msg_t2 = make_tool_msg(
            "[Tool Result for read_file]\nnew contents",
            Some("arya"),
        );
        msg_t2.id = 13;
        msg_t2.tool_call_id = Some("tc2".into());

        let msgs = vec![
            make_conv_msg("user", "check /a.rs"),
            msg_a1, msg_t1, msg_a2, msg_t2,
            make_conv_msg("user", "thanks"),
        ];

        let (history, _) = format_conversation_history(&msgs, "arya", Some(i64::MAX));

        // Write is superseded by later read — both are fresh views, read is newer
        let write_results: Vec<_> = history
            .iter()
            .filter(|m| {
                m.content
                    .as_ref()
                    .map_or(false, |c| c.starts_with("[Tool Result for write_file]"))
            })
            .collect();
        assert_eq!(write_results.len(), 0, "Write before read is superseded");

        let read_results: Vec<_> = history
            .iter()
            .filter(|m| {
                m.content
                    .as_ref()
                    .map_or(false, |c| c.starts_with("[Tool Result for read_file]"))
            })
            .collect();
        assert_eq!(read_results.len(), 1, "Read should be kept");

        // Assistant message for the write (id=10) should also be dropped
        let asst_contents: Vec<&str> = history
            .iter()
            .filter(|m| m.role == "assistant")
            .filter_map(|m| m.content.as_deref())
            .collect();
        assert!(!asst_contents.contains(&"Writing."), "Write's assistant msg should be dropped");
        assert!(asst_contents.contains(&"Reading."), "Read's assistant msg should be kept");
    }

    #[test]
    fn test_format_conversation_history_dedup_multi_tool_call_partial_keep() {
        // Assistant message with 2 tool_calls: one read_file superseded, one shell (ls) kept.
        // Entire assistant message should be kept because not ALL tool_calls are dropped.
        let multi_tc = serde_json::to_string(&vec![
            crate::llm::ToolCall {
                id: "tc1".into(),
                name: "read_file".into(),
                arguments: serde_json::json!({"path": "/a.rs"}),
            },
            crate::llm::ToolCall {
                id: "tc2".into(),
                name: "shell".into(),
                arguments: serde_json::json!({"command": "ls"}),
            },
        ])
        .unwrap();
        let read_tc2 = serde_json::to_string(&vec![crate::llm::ToolCall {
            id: "tc3".into(),
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "/a.rs"}),
        }])
        .unwrap();

        let mut msg_a1 = make_conv_msg("arya", "Multi-tool.");
        msg_a1.id = 10;
        msg_a1.tool_calls = Some(multi_tc);

        let mut msg_t1 = make_tool_msg(
            "[Tool Result for read_file]\nold version",
            Some("arya"),
        );
        msg_t1.id = 11;
        msg_t1.tool_call_id = Some("tc1".into());
        let mut msg_t_shell = make_tool_msg(
            "[Tool Result for shell]\nfile1.rs",
            Some("arya"),
        );
        msg_t_shell.id = 12;
        msg_t_shell.tool_call_id = Some("tc2".into());

        let mut msg_a2 = make_conv_msg("arya", "Re-reading.");
        msg_a2.id = 13;
        msg_a2.tool_calls = Some(read_tc2);
        let mut msg_t2 = make_tool_msg(
            "[Tool Result for read_file]\nnew version",
            Some("arya"),
        );
        msg_t2.id = 14;
        msg_t2.tool_call_id = Some("tc3".into());

        let msgs = vec![
            make_conv_msg("user", "check /a.rs"),
            msg_a1, msg_t1, msg_t_shell, msg_a2, msg_t2,
            make_conv_msg("user", "thanks"),
        ];

        let (history, _) = format_conversation_history(&msgs, "arya", Some(i64::MAX));

        // The multi-tool assistant message (id=10) should be kept (not all tool_calls dropped)
        let assistant_msgs: Vec<&str> = history
            .iter()
            .filter(|m| m.role == "assistant")
            .filter_map(|m| m.content.as_deref())
            .collect();
        assert!(
            assistant_msgs.contains(&"Multi-tool."),
            "Multi-tool assistant message should be kept"
        );

        // The old read_file tool result (id=11) IS dropped
        let read_results: Vec<&str> = history
            .iter()
            .filter(|m| {
                m.content
                    .as_ref()
                    .map_or(false, |c| c.starts_with("[Tool Result for read_file]"))
            })
            .map(|m| m.content.as_ref().unwrap().as_str())
            .collect();
        assert_eq!(read_results.len(), 1, "Only latest read kept");
        assert!(read_results[0].contains("new version"));

        // The shell result (id=12) is kept
        let shell_results: Vec<_> = history
            .iter()
            .filter(|m| {
                m.content
                    .as_ref()
                    .map_or(false, |c| c.starts_with("[Tool Result for shell]"))
            })
            .collect();
        assert_eq!(shell_results.len(), 1, "Shell result should be kept");
    }

    // Tests for expand_inject_directives

    #[test]
    fn test_expand_inject_directives_no_directives() {
        let content = "Some always content\nwith no directives";
        let result = expand_inject_directives(content, "tools here", "memories here", "");
        // Should return None to signal fallback behavior
        assert!(result.is_none());
    }

    #[test]
    fn test_expand_inject_directives_tools_only() {
        let content = "Before\n<!-- @inject:tools -->\nAfter";
        let result = expand_inject_directives(content, "**Tools:**\n- tool1", "", "");
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
        let result = expand_inject_directives(content, "", "[Memories]\n- mem1", "");
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
        let result = expand_inject_directives(content, "TOOLS", "MEMORIES", "");
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
        let result = expand_inject_directives(content, "", "", "");
        assert!(result.is_some());
        let expanded = result.unwrap();
        assert_eq!(expanded, "Line1\nLine2");
    }

    #[test]
    fn test_expand_inject_directives_empty_memories_removes_line() {
        let content = "Line1\n<!-- @inject:memories -->\nLine2";
        let result = expand_inject_directives(content, "", "", "");
        assert!(result.is_some());
        let expanded = result.unwrap();
        assert_eq!(expanded, "Line1\nLine2");
    }

    #[test]
    fn test_expand_inject_directives_both_empty() {
        let content = "Line1\n<!-- @inject:tools -->\nLine2\n<!-- @inject:memories -->\nLine3";
        let result = expand_inject_directives(content, "", "", "");
        assert!(result.is_some());
        let expanded = result.unwrap();
        assert_eq!(expanded, "Line1\nLine2\nLine3");
    }

    #[test]
    fn test_expand_inject_directives_whitespace_tolerance() {
        // Directive with surrounding whitespace on the line
        let content = "Before\n  <!-- @inject:tools -->  \nAfter";
        let result = expand_inject_directives(content, "TOOLS", "", "");
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
        let result = build_recall_content("TOOLS", "MEMORIES", "", &base, &None);
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
    fn test_build_recall_content_without_directives_still_injects() {
        // If recall.md exists but has no directives, auto-inject after recall content
        let base = Some("Just content, no directives".to_string());
        let result = build_recall_content("TOOLS", "MEMORIES", "", &base, &None);
        assert!(result.is_some());
        let effective = result.unwrap();
        // Should contain recall.md content AND auto-injected tools/memories
        assert!(effective.contains("Just content"));
        assert!(effective.contains("TOOLS"));
        assert!(effective.contains("MEMORIES"));
        // recall.md content should come before injections
        let content_pos = effective.find("Just content").unwrap();
        let tools_pos = effective.find("TOOLS").unwrap();
        assert!(content_pos < tools_pos);
    }

    #[test]
    fn test_build_recall_content_no_base_injects_defaults() {
        // If no always.md at all, inject tools/memories as sensible defaults
        let result = build_recall_content("TOOLS", "MEMORIES", "", &None, &None);
        assert!(result.is_some());
        let effective = result.unwrap();
        assert!(effective.contains("TOOLS"));
        assert!(effective.contains("MEMORIES"));
    }

    #[test]
    fn test_build_recall_content_model_always_appended() {
        let base = Some("Base\n<!-- @inject:tools -->".to_string());
        let model = Some("Model always".to_string());
        let result = build_recall_content("TOOLS", "", "", &base, &model);
        assert!(result.is_some());
        let effective = result.unwrap();
        // Model always should be at the end
        assert!(effective.ends_with("Model always") || effective.contains("Model always"));
        let base_pos = effective.find("Base").unwrap();
        let model_pos = effective.find("Model always").unwrap();
        assert!(base_pos < model_pos);
    }

    #[test]
    fn test_build_conversation_recall_injection_empty() {
        let results: Vec<(i64, String, String, i64, f32)> = vec![];
        let injection = build_conversation_recall_injection(&results);
        assert!(injection.is_empty());
    }

    #[test]
    fn test_build_conversation_recall_injection_formats() {
        let now = chrono::Utc::now().timestamp();
        let results = vec![
            (1, "user".to_string(), "What about using Redis?".to_string(), now - 7200, 0.85),
            (2, "arya".to_string(), "The API should support pagination".to_string(), now - 86400, 0.72),
        ];
        let injection = build_conversation_recall_injection(&results);
        assert!(injection.starts_with("[recalled messages]\n"));
        assert!(injection.contains("[user] What about using Redis?"));
        assert!(injection.contains("[arya] The API should support pagination"));
        // Each entry should be on its own line with age prefix
        let lines: Vec<&str> = injection.lines().collect();
        assert_eq!(lines[0], "[recalled messages]");
        assert!(lines[1].starts_with("- ("));
        assert!(lines[2].starts_with("- ("));
    }

    #[test]
    fn test_build_conversation_recall_injection_truncates_long() {
        let now = chrono::Utc::now().timestamp();
        let long_msg = "x".repeat(300);
        let results = vec![(1, "arya".to_string(), long_msg, now - 60, 0.9)];
        let injection = build_conversation_recall_injection(&results);
        // Should truncate to 200 chars + "..."
        assert!(injection.contains("..."));
        // The full 300-char message should NOT appear
        assert!(!injection.contains(&"x".repeat(300)));
    }

    #[test]
    fn test_expand_inject_directives_conversation() {
        let content = "Before\n<!-- @inject:conversation -->\nAfter";
        let result = expand_inject_directives(content, "", "", "CONVERSATION");
        assert!(result.is_some());
        let expanded = result.unwrap();
        assert!(expanded.contains("CONVERSATION"));
        assert!(!expanded.contains("<!-- @inject:conversation -->"));
    }

    #[test]
    fn test_build_recall_content_no_base_with_conversation() {
        let result = build_recall_content("TOOLS", "MEMORIES", "CONV", &None, &None);
        assert!(result.is_some());
        let effective = result.unwrap();
        assert!(effective.contains("TOOLS"));
        assert!(effective.contains("MEMORIES"));
        assert!(effective.contains("CONV"));
    }

    fn test_logger() -> AgentLogger {
        let dir = tempdir().unwrap();
        let logger = AgentLogger::new(dir.path(), "test-agent").unwrap();
        std::mem::forget(dir);
        logger
    }

    #[test]
    fn test_load_agent_context_cold_start() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = ConversationStore::open(&db_path).unwrap();
        let logger = test_logger();

        let conv = store
            .create_conversation(Some("ctx-cold"), &["arya"])
            .unwrap();
        let id1 = store.add_message(&conv, "user", "hello", &[]).unwrap();
        store.add_message(&conv, "arya", "hi", &[]).unwrap();
        store.add_message(&conv, "user", "how are you?", &[]).unwrap();

        // No cursor set → cold start → should get last N messages
        let msgs = load_agent_context(&store, &conv, "arya", &logger, None).unwrap();
        assert_eq!(msgs.len(), 3);

        // Cursor should be set to anchor (oldest non-pinned msg)
        let cursor = store.get_context_cursor(&conv, "arya").unwrap();
        assert_eq!(cursor, Some(id1));
    }

    #[test]
    fn test_load_agent_context_append() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = ConversationStore::open(&db_path).unwrap();
        let logger = test_logger();

        let conv = store
            .create_conversation(Some("ctx-append"), &["arya"])
            .unwrap();
        let anchor_id = store.add_message(&conv, "user", "msg 1", &[]).unwrap();
        store.add_message(&conv, "arya", "response 1", &[]).unwrap();
        store.add_message(&conv, "user", "msg 2", &[]).unwrap();

        // Set cursor to anchor (oldest message in window)
        store.set_context_cursor(&conv, "arya", anchor_id).unwrap();

        // Should get all messages from anchor (inclusive) — full window + new
        let msgs = load_agent_context(&store, &conv, "arya", &logger, None).unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0].content, "msg 1");
        assert_eq!(msgs[1].content, "response 1");
        assert_eq!(msgs[2].content, "msg 2");
    }

    #[test]
    fn test_load_agent_context_stale_cursor() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = ConversationStore::open(&db_path).unwrap();
        let logger = test_logger();

        let conv = store
            .create_conversation(Some("ctx-stale"), &["arya"])
            .unwrap();
        let id1 = store.add_message(&conv, "user", "msg 1", &[]).unwrap();
        store.add_message(&conv, "arya", "response 1", &[]).unwrap();
        store.add_message(&conv, "user", "msg 2", &[]).unwrap();

        // Set cursor to a very high ID (no messages from it)
        store.set_context_cursor(&conv, "arya", 99999).unwrap();

        // count == 0 → falls back to cold start with new anchor
        let msgs = load_agent_context(&store, &conv, "arya", &logger, None).unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0].content, "msg 1");

        // Cursor should be set to new anchor (oldest msg)
        let cursor = store.get_context_cursor(&conv, "arya").unwrap();
        assert_eq!(cursor, Some(id1));
    }

    #[test]
    fn test_load_agent_context_many_messages_append() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = ConversationStore::open(&db_path).unwrap();
        let logger = test_logger();

        let conv = store
            .create_conversation(Some("ctx-many"), &["arya"])
            .unwrap();
        let cursor_id = store.add_message(&conv, "arya", "old response", &[]).unwrap();
        store.set_context_cursor(&conv, "arya", cursor_id).unwrap();

        // Add many messages — append mode should return all from cursor
        for i in 0..16 {
            store.add_message(&conv, "user", &format!("msg {}", i), &[]).unwrap();
        }

        // No overflow guard — append mode returns all 17 messages from cursor
        let msgs = load_agent_context(&store, &conv, "arya", &logger, None).unwrap();
        assert_eq!(msgs.len(), 17); // 1 (old response) + 16 new
    }

    #[test]
    fn test_tool_call_id_round_trip() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = ConversationStore::open(&db_path).unwrap();

        let conv = store.create_conversation(Some("tcid-test"), &["dash"]).unwrap();

        // Store a native tool result with tool_call_id
        let id = store
            .add_native_tool_result(&conv, "call_abc123", "[Tool Result for read_file]\ncontents here", "dash")
            .unwrap();
        assert!(id > 0);

        // Read it back
        let msgs = store.get_messages(&conv, None).unwrap();
        let tool_msg = msgs.iter().find(|m| m.from_agent == "tool").unwrap();
        assert_eq!(tool_msg.tool_call_id.as_deref(), Some("call_abc123"));
        assert_eq!(tool_msg.triggered_by.as_deref(), Some("dash"));
    }

    #[test]
    fn test_tool_call_id_null_backward_compat() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = ConversationStore::open(&db_path).unwrap();

        let conv = store.create_conversation(Some("tcid-compat"), &["dash"]).unwrap();

        // Store a legacy tool result (no tool_call_id)
        store.add_tool_result(&conv, "[Tool Result for shell]\nok", "dash").unwrap();

        // Read it back — tool_call_id should be None
        let msgs = store.get_messages(&conv, None).unwrap();
        let tool_msg = msgs.iter().find(|m| m.from_agent == "tool").unwrap();
        assert!(tool_msg.tool_call_id.is_none());
    }

    #[test]
    fn test_format_conversation_history_native_tool_result() {
        // Native tool result (with tool_call_id) should become role "tool"
        let mut msg = make_tool_msg("[Tool Result for read_file]\nfile contents", Some("dash"));
        msg.tool_call_id = Some("call_xyz".to_string());

        let msgs = vec![
            {
                let mut m = make_conv_msg("dash", "reading file");
                m.id = 1;
                m.tool_calls = Some(
                    serde_json::to_string(&vec![crate::llm::ToolCall {
                        id: "call_xyz".to_string(),
                        name: "read_file".to_string(),
                        arguments: serde_json::json!({"path": "test.rs"}),
                    }])
                    .unwrap(),
                );
                m
            },
            {
                let mut m = msg;
                m.id = 2;
                m
            },
            {
                let mut m = make_conv_msg("user", "thanks");
                m.id = 3;
                m
            },
        ];

        let (history, _final_content) = format_conversation_history(&msgs, "dash", Some(i64::MAX));

        // Should have: assistant (with tool_calls), tool (role="tool" with tool_call_id)
        assert!(history.len() >= 2);
        assert_eq!(history[0].role, "assistant");
        assert_eq!(history[1].role, "tool");
        assert_eq!(history[1].tool_call_id.as_deref(), Some("call_xyz"));
    }

    #[test]
    fn test_format_conversation_history_legacy_tool_result() {
        // Legacy tool result (no tool_call_id) should become role "user"
        let msgs = vec![
            {
                let mut m = make_conv_msg("dash", "calling tool");
                m.id = 1;
                m
            },
            {
                let mut m = make_tool_msg("[Tool Result for shell]\nok", Some("dash"));
                m.id = 2;
                m
            },
            {
                let mut m = make_conv_msg("user", "great");
                m.id = 3;
                m
            },
        ];

        let (history, _final_content) = format_conversation_history(&msgs, "dash", Some(i64::MAX));

        // Legacy tool result should use role "user"
        assert!(history.len() >= 2);
        assert_eq!(history[0].role, "assistant");
        assert_eq!(history[1].role, "user");
        assert!(history[1].tool_call_id.is_none());
    }

    #[test]
    fn test_truncate_tool_result() {
        // Under limit — returned unchanged (no num_ctx → 128 KB default)
        let small = "hello world";
        let result = truncate_tool_result(small, None);
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(&*result, small);

        // Exactly at limit — returned unchanged
        // Use a num_ctx that gives max_bytes = 100: num_ctx/10*4 = 100 → num_ctx = 250
        let exact = "a".repeat(100);
        let result = truncate_tool_result(&exact, Some(250));
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(&*result, exact);

        // Over limit — truncated, keeps TAIL with notice at start
        let big: String = (0..200).map(|i| char::from(b'A' + (i % 26) as u8)).collect();
        let result = truncate_tool_result(&big, Some(250)); // max_bytes = 100
        assert!(matches!(result, Cow::Owned(_)));
        assert!(result.contains("[output truncated:"));
        assert!(result.contains("showing last"));
        // The tail of the original should be present
        assert!(result.ends_with(&big[100..]));
    }

    #[test]
    fn test_truncate_tool_result_multibyte() {
        // Multi-byte chars: 'é' is 2 bytes, cutting mid-char should find safe boundary
        // num_ctx=375 → max_bytes = 375/10*4 = 150
        let multi = "é".repeat(100); // 200 bytes
        let result = truncate_tool_result(&multi, Some(375));
        assert!(matches!(result, Cow::Owned(_)));
        // Should not panic and should be valid UTF-8
        assert!(result.contains("[output truncated:"));
        // The kept portion should be valid — ceil_char_boundary ensures this
        let kept = result.split("[output truncated:").last().unwrap();
        let tail_part = kept.split("\n\n").skip(1).collect::<Vec<_>>().join("\n\n");
        assert!(tail_part.len() <= 150);
        // Should be a whole number of 'é' chars (each 2 bytes)
        assert_eq!(tail_part.len() % 2, 0);
    }

    #[test]
    fn test_truncate_tool_result_with_num_ctx() {
        // num_ctx = 1000 → budget = 1000/10*4 = 400 bytes
        let data = "x".repeat(500);
        let result = truncate_tool_result(&data, Some(1000));
        assert!(matches!(result, Cow::Owned(_)));
        // Should keep last 400 bytes
        assert!(result.ends_with(&"x".repeat(400)));
        assert!(result.contains("showing last"));
    }

    #[test]
    fn test_truncate_tool_result_no_num_ctx_fallback() {
        // Without num_ctx, falls back to DEFAULT_MAX_TOOL_RESULT_BYTES (128 KB)
        let small = "y".repeat(131_072); // exactly at limit
        let result = truncate_tool_result(&small, None);
        assert!(matches!(result, Cow::Borrowed(_)));

        let big = "z".repeat(131_073); // 1 byte over
        let result = truncate_tool_result(&big, None);
        assert!(matches!(result, Cow::Owned(_)));
        assert!(result.contains("[output truncated:"));
        assert!(result.ends_with(&"z".repeat(131_072)));
    }

    #[test]
    fn test_strip_notes() {
        // Basic stripping
        let content = "<notes>\nmy notes\n</notes>\n\nHello world";
        let stripped = strip_notes(content).unwrap();
        assert_eq!(stripped, "Hello world");

        // No notes block → returns None
        assert!(strip_notes("just plain text").is_none());

        // Notes only (no response after)
        let content = "<notes>\nmy notes\n</notes>";
        let stripped = strip_notes(content).unwrap();
        assert_eq!(stripped, "");

        // Notes with thinking tags before
        let content = "<think>reasoning</think>\n<notes>\nnotes\n</notes>\n\nresponse";
        let stripped = strip_notes(content).unwrap();
        assert_eq!(stripped, "<think>reasoning</think>\nresponse");
    }

    #[test]
    fn test_extract_llm_notes() {
        // Basic extraction
        let (cleaned, notes) =
            extract_llm_notes("Hello\n<notes>\nmy plan\n</notes>\n\nworld");
        assert_eq!(cleaned, "Hello\nworld");
        assert_eq!(notes.unwrap(), "my plan");

        // No block → passthrough
        let (cleaned, notes) = extract_llm_notes("just plain text");
        assert_eq!(cleaned, "just plain text");
        assert!(notes.is_none());

        // Empty block → stripped but no notes returned
        let (cleaned, notes) =
            extract_llm_notes("before\n<notes>\n\n</notes>\nafter");
        assert_eq!(cleaned, "before\nafter");
        assert!(notes.is_none());

        // Content after block preserved
        let (cleaned, notes) =
            extract_llm_notes("<notes>\ntodo list\n</notes>\nHere is my answer.");
        assert_eq!(cleaned, "Here is my answer.");
        assert_eq!(notes.unwrap(), "todo list");

        // Only notes, no surrounding content
        let (cleaned, notes) =
            extract_llm_notes("<notes>\nsolitary\n</notes>");
        assert_eq!(cleaned, "");
        assert_eq!(notes.unwrap(), "solitary");
    }

    #[test]
    fn test_extract_llm_notes_with_thinking() {
        // Thinking tags + notes — both extracted correctly
        let content = "<think>reasoning here</think>\nSome text\n<notes>\nmy notes\n</notes>\n\nfinal answer";
        let (cleaned, notes) = extract_llm_notes(content);
        assert_eq!(cleaned, "<think>reasoning here</think>\nSome text\nfinal answer");
        assert_eq!(notes.unwrap(), "my notes");
    }

    #[test]
    fn test_prepend_then_extract_roundtrip() {
        // prepend_notes produces a <notes> block at the start;
        // extract_llm_notes should extract those notes and return the original content
        let original = "Here is my response.";
        let notes_text = "step 1: check files\nstep 2: edit code";
        let prepended = format!(
            "<notes>\n{}\n</notes>\n\n{}",
            notes_text, original
        );
        let (cleaned, extracted) = extract_llm_notes(&prepended);
        assert_eq!(cleaned, original);
        assert_eq!(extracted.unwrap(), notes_text);
    }

    #[test]
    fn test_format_conversation_history_strips_old_notes() {
        // Two assistant messages with notes — only the last should keep its notes
        let msgs = vec![
            make_conv_msg("user", "hi"),
            {
                let mut m = make_conv_msg("arya", "<notes>\nold plan\n</notes>\n\nfirst response");
                m.id = 2;
                m
            },
            make_conv_msg("user", "next"),
            {
                let mut m = make_conv_msg("arya", "<notes>\nnew plan\n</notes>\n\nsecond response");
                m.id = 4;
                m
            },
            make_conv_msg("user", "thanks"),
        ];

        let (history, _) = format_conversation_history(&msgs, "arya", Some(i64::MAX));

        // Find assistant messages
        let asst_msgs: Vec<&ChatMessage> = history.iter().filter(|m| m.role == "assistant").collect();
        assert_eq!(asst_msgs.len(), 2);

        // First assistant message should have notes stripped
        let first = asst_msgs[0].content.as_ref().unwrap();
        assert!(!first.contains("<notes>"), "old notes should be stripped");
        assert!(first.contains("first response"));

        // Last assistant message should keep notes
        let last = asst_msgs[1].content.as_ref().unwrap();
        assert!(last.contains("<notes>"), "last notes should be kept");
        assert!(last.contains("new plan"));
        assert!(last.contains("second response"));
    }

    #[test]
    fn test_format_conversation_history_lazy_dedup_skips_new_messages() {
        // With cutoff before any tool results, no dedup should happen at all
        let msgs = vec![
            { let mut m = make_conv_msg("user", "read a.rs"); m.id = 1; m },
            {
                let mut m = make_conv_msg("arya", "");
                m.id = 2;
                m.tool_calls = Some(serde_json::to_string(&vec![
                    crate::llm::ToolCall { id: "tc1".into(), name: "read_file".into(), arguments: serde_json::json!({"path": "/a.rs"}) },
                ]).unwrap());
                m
            },
            {
                let mut m = make_tool_msg("content of a.rs v1", Some("arya"));
                m.id = 3;
                m.tool_call_id = Some("tc1".into());
                m
            },
            { let mut m = make_conv_msg("user", "read a.rs again"); m.id = 4; m },
            {
                let mut m = make_conv_msg("arya", "");
                m.id = 5;
                m.tool_calls = Some(serde_json::to_string(&vec![
                    crate::llm::ToolCall { id: "tc2".into(), name: "read_file".into(), arguments: serde_json::json!({"path": "/a.rs"}) },
                ]).unwrap());
                m
            },
            {
                let mut m = make_tool_msg("content of a.rs v2", Some("arya"));
                m.id = 6;
                m.tool_call_id = Some("tc2".into());
                m
            },
            { let mut m = make_conv_msg("user", "thanks"); m.id = 7; m },
        ];

        // Cutoff at id=1 (before any tool results) — nothing eligible for dedup
        let (history, _) = format_conversation_history(&msgs, "arya", Some(1));

        let tool_results: Vec<&str> = history.iter()
            .filter(|m| m.role == "tool")
            .filter_map(|m| m.content.as_deref())
            .collect();

        // Both tool results survive — neither is within cutoff
        assert_eq!(tool_results.len(), 2);
        assert!(tool_results.iter().any(|c| c.contains("v1")));
        assert!(tool_results.iter().any(|c| c.contains("v2")));
    }

    #[test]
    fn test_format_conversation_history_lazy_dedup_drops_old_messages() {
        // Messages at/before the dedup cutoff ARE deduped normally
        let msgs = vec![
            { let mut m = make_conv_msg("user", "read a.rs"); m.id = 1; m },
            {
                let mut m = make_conv_msg("arya", "");
                m.id = 2;
                m.tool_calls = Some(serde_json::to_string(&vec![
                    crate::llm::ToolCall { id: "tc1".into(), name: "read_file".into(), arguments: serde_json::json!({"path": "/a.rs"}) },
                ]).unwrap());
                m
            },
            {
                let mut m = make_tool_msg("content of a.rs v1", Some("arya"));
                m.id = 3;
                m.tool_call_id = Some("tc1".into());
                m
            },
            { let mut m = make_conv_msg("user", "read a.rs again"); m.id = 4; m },
            {
                let mut m = make_conv_msg("arya", "");
                m.id = 5;
                m.tool_calls = Some(serde_json::to_string(&vec![
                    crate::llm::ToolCall { id: "tc2".into(), name: "read_file".into(), arguments: serde_json::json!({"path": "/a.rs"}) },
                ]).unwrap());
                m
            },
            {
                let mut m = make_tool_msg("content of a.rs v2", Some("arya"));
                m.id = 6;
                m.tool_call_id = Some("tc2".into());
                m
            },
            // -- cutoff at id=7 covers all messages --
            { let mut m = make_conv_msg("user", "thanks"); m.id = 7; m },
        ];

        // Cutoff at 7 — all messages eligible, behaves like eager mode
        let (history, _) = format_conversation_history(&msgs, "arya", Some(7));

        let tool_results: Vec<&str> = history.iter()
            .filter(|m| m.role == "tool")
            .filter_map(|m| m.content.as_deref())
            .collect();

        // Only second (latest) read should survive — first is deduped
        assert_eq!(tool_results.len(), 1);
        assert!(tool_results[0].contains("v2"));
    }

    #[test]
    fn test_format_conversation_history_lazy_notes_stripping() {
        // Notes should only be stripped from messages at/before the dedup boundary
        let msgs = vec![
            { let mut m = make_conv_msg("user", "hi"); m.id = 1; m },
            {
                let mut m = make_conv_msg("arya", "<notes>\nold plan\n</notes>\n\nfirst response");
                m.id = 2;
                m
            },
            // -- dedup cutoff at id=2 --
            { let mut m = make_conv_msg("user", "next"); m.id = 3; m },
            {
                let mut m = make_conv_msg("arya", "<notes>\nmid plan\n</notes>\n\nsecond response");
                m.id = 4;
                m
            },
            { let mut m = make_conv_msg("user", "more"); m.id = 5; m },
            {
                let mut m = make_conv_msg("arya", "<notes>\nnew plan\n</notes>\n\nthird response");
                m.id = 6;
                m
            },
            { let mut m = make_conv_msg("user", "thanks"); m.id = 7; m },
        ];

        // Cutoff at id=2: only first assistant message is within boundary
        let (history, _) = format_conversation_history(&msgs, "arya", Some(2));

        let asst_msgs: Vec<&ChatMessage> = history.iter().filter(|m| m.role == "assistant").collect();
        assert_eq!(asst_msgs.len(), 3);

        // First assistant (id=2, within boundary) — notes stripped
        let first = asst_msgs[0].content.as_ref().unwrap();
        assert!(!first.contains("<notes>"), "notes within boundary should be stripped");
        assert!(first.contains("first response"));

        // Second assistant (id=4, after boundary) — notes kept (frozen prefix)
        let second = asst_msgs[1].content.as_ref().unwrap();
        assert!(second.contains("<notes>"), "notes after boundary should be kept");
        assert!(second.contains("mid plan"));

        // Third/last assistant (id=6) — notes always kept (last assistant)
        let third = asst_msgs[2].content.as_ref().unwrap();
        assert!(third.contains("<notes>"), "last assistant notes always kept");
        assert!(third.contains("new plan"));
    }

    #[test]
    fn test_format_conversation_history_eager_unchanged() {
        // With dedup_up_to = latest msg id, behavior is identical to old "always dedup"
        let msgs = vec![
            { let mut m = make_conv_msg("user", "hi"); m.id = 1; m },
            {
                let mut m = make_conv_msg("arya", "<notes>\nold plan\n</notes>\n\nfirst response");
                m.id = 2;
                m
            },
            { let mut m = make_conv_msg("user", "next"); m.id = 3; m },
            {
                let mut m = make_conv_msg("arya", "<notes>\nnew plan\n</notes>\n\nsecond response");
                m.id = 4;
                m
            },
            { let mut m = make_conv_msg("user", "thanks"); m.id = 5; m },
        ];

        // Eager: pointer at latest message (5) — all messages eligible
        let (history, _) = format_conversation_history(&msgs, "arya", Some(5));

        let asst_msgs: Vec<&ChatMessage> = history.iter().filter(|m| m.role == "assistant").collect();
        assert_eq!(asst_msgs.len(), 2);

        // First assistant — notes stripped (within boundary, not last)
        let first = asst_msgs[0].content.as_ref().unwrap();
        assert!(!first.contains("<notes>"), "old notes stripped in eager mode");
        assert!(first.contains("first response"));

        // Last assistant — notes kept
        let last = asst_msgs[1].content.as_ref().unwrap();
        assert!(last.contains("<notes>"), "last notes kept in eager mode");
    }

    #[test]
    fn test_format_conversation_history_no_dedup_pointer() {
        // With dedup_up_to = None, no dedup or notes stripping happens at all
        let msgs = vec![
            { let mut m = make_conv_msg("user", "read a.rs"); m.id = 1; m },
            {
                let mut m = make_conv_msg("arya", "");
                m.id = 2;
                m.tool_calls = Some(serde_json::to_string(&vec![
                    crate::llm::ToolCall { id: "tc1".into(), name: "read_file".into(), arguments: serde_json::json!({"path": "/a.rs"}) },
                ]).unwrap());
                m
            },
            {
                let mut m = make_tool_msg("content v1", Some("arya"));
                m.id = 3;
                m.tool_call_id = Some("tc1".into());
                m
            },
            { let mut m = make_conv_msg("user", "read again"); m.id = 4; m },
            {
                let mut m = make_conv_msg("arya", "");
                m.id = 5;
                m.tool_calls = Some(serde_json::to_string(&vec![
                    crate::llm::ToolCall { id: "tc2".into(), name: "read_file".into(), arguments: serde_json::json!({"path": "/a.rs"}) },
                ]).unwrap());
                m
            },
            {
                let mut m = make_tool_msg("content v2", Some("arya"));
                m.id = 6;
                m.tool_call_id = Some("tc2".into());
                m
            },
            { let mut m = make_conv_msg("user", "thanks"); m.id = 7; m },
        ];

        // No dedup pointer — nothing should be deduped
        let (history, _) = format_conversation_history(&msgs, "arya", None);

        let tool_results: Vec<&str> = history.iter()
            .filter(|m| m.role == "tool")
            .filter_map(|m| m.content.as_deref())
            .collect();

        // Both tool results survive — no dedup
        assert_eq!(tool_results.len(), 2);
        assert!(tool_results.iter().any(|c| c.contains("v1")));
        assert!(tool_results.iter().any(|c| c.contains("v2")));
    }

    #[test]
    fn test_check_fill_advances_dedup_cursor() {
        use crate::conversation::ConversationStore;
        let dir2 = tempdir().unwrap();
        let store = ConversationStore::open(&dir2.path().join("test.db")).unwrap();
        let conv = store.create_conversation(Some("fill-test"), &["dash"]).unwrap();
        let dir = tempdir().unwrap();
        let logger = AgentLogger::new(dir.path(), "test-agent").unwrap();

        // No dedup cursor initially
        assert!(store.get_dedup_cursor(&conv, "dash").unwrap().is_none());

        // Simulate fill >= 90% with lazy mode
        check_fill_and_maybe_reset(
            &store, &conv, "dash",
            Some(900), Some(100), Some(1000), // 100% fill
            true,       // dedup_lazy
            Some(42),   // latest_msg_id
            &logger,
        );

        // Dedup cursor should be set to latest_msg_id
        assert_eq!(store.get_dedup_cursor(&conv, "dash").unwrap(), Some(42));
        // Context cursor should NOT be cleared (still set to whatever it was)
    }

    #[test]
    fn test_check_fill_resets_after_dedup_insufficient() {
        use crate::conversation::ConversationStore;
        let dir2 = tempdir().unwrap();
        let store = ConversationStore::open(&dir2.path().join("test.db")).unwrap();
        let conv = store.create_conversation(Some("fill-test2"), &["dash"]).unwrap();
        let dir = tempdir().unwrap();
        let logger = AgentLogger::new(dir.path(), "test-agent").unwrap();

        // Set context cursor so we can verify it gets cleared
        store.set_context_cursor(&conv, "dash", 10).unwrap();

        // First call: advances dedup cursor
        check_fill_and_maybe_reset(
            &store, &conv, "dash",
            Some(900), Some(100), Some(1000),
            true, Some(42), &logger,
        );
        assert_eq!(store.get_dedup_cursor(&conv, "dash").unwrap(), Some(42));

        // Second call: dedup cursor already at latest — triggers phase 2 (reset)
        check_fill_and_maybe_reset(
            &store, &conv, "dash",
            Some(900), Some(100), Some(1000),
            true, Some(42), &logger,
        );

        // Both cursors should be cleared
        assert!(store.get_dedup_cursor(&conv, "dash").unwrap().is_none());
        assert!(store.get_context_cursor(&conv, "dash").unwrap().is_none());
    }

    #[test]
    fn test_check_fill_phase2_triggers_even_when_latest_advances() {
        // Regression test: phase 2 must fire even when latest_msg_id grows between calls
        // (new messages added between turns). Before fix, phase 1 always re-triggered
        // because current_dedup < latest was always true.
        use crate::conversation::ConversationStore;
        let dir2 = tempdir().unwrap();
        let store = ConversationStore::open(&dir2.path().join("test.db")).unwrap();
        let conv = store.create_conversation(Some("fill-phase2"), &["dash"]).unwrap();
        let dir = tempdir().unwrap();
        let logger = AgentLogger::new(dir.path(), "test-agent").unwrap();

        store.set_context_cursor(&conv, "dash", 5).unwrap();

        // Phase 1: no dedup cursor → sets it to 42
        check_fill_and_maybe_reset(
            &store, &conv, "dash",
            Some(900), Some(100), Some(1000),
            true, Some(42), &logger,
        );
        assert_eq!(store.get_dedup_cursor(&conv, "dash").unwrap(), Some(42));
        // Context cursor untouched
        assert_eq!(store.get_context_cursor(&conv, "dash").unwrap(), Some(5));

        // Phase 2: dedup cursor exists (42), latest advanced to 50 (new messages added).
        // Still over threshold → should reset both cursors.
        check_fill_and_maybe_reset(
            &store, &conv, "dash",
            Some(900), Some(100), Some(1000),
            true, Some(50), &logger,
        );
        assert!(store.get_dedup_cursor(&conv, "dash").unwrap().is_none());
        assert!(store.get_context_cursor(&conv, "dash").unwrap().is_none());
    }

}
