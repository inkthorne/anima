use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
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

/// Returns a tool-budget nudge string when the agent has used ≥50% of its
/// tool iterations, or None if below the threshold.
pub fn tool_budget_nudge(iteration: usize, max: usize) -> Option<String> {
    let pct = (iteration * 100) / max.max(1);
    if pct >= 80 {
        Some(format!(
            "[Tool call {} of {} — approaching limit, provide your response now]",
            iteration, max
        ))
    } else if pct >= 50 {
        Some(format!(
            "[Tool call {} of {} — consider responding if you have sufficient information]",
            iteration, max
        ))
    } else {
        None
    }
}

/// Expand tilde (~) in path to home directory.
pub fn expand_tilde(path: &str) -> PathBuf {
    if let Some(suffix) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(suffix);
        }
    } else if path == "~" {
        if let Some(home) = dirs::home_dir() {
            return home;
        }
    }
    PathBuf::from(path)
}

/// Walk up from a file path looking for project root markers.
/// Returns the first ancestor directory containing a marker file.
pub fn detect_project_root(file_path: &str) -> Option<PathBuf> {
    const MARKERS: &[&str] = &[
        "Cargo.toml", "package.json", "go.mod", "pyproject.toml", "Makefile", ".git",
    ];
    let expanded = expand_tilde(file_path);
    let mut dir = expanded.parent()?;
    loop {
        for marker in MARKERS {
            if dir.join(marker).exists() {
                return Some(dir.to_path_buf());
            }
        }
        match dir.parent() {
            Some(parent) => dir = parent,
            None => return None,
        }
    }
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
        "search_conversation" => {
            let keyword = params.get("keyword").and_then(|v| v.as_str()).unwrap_or("?");
            let conv = params.get("conversation").and_then(|v| v.as_str()).unwrap_or("?");
            let from = params.get("from").and_then(|v| v.as_str());
            let count = result.and_then(|r| r.get("count")).and_then(|v| v.as_u64());
            match (from, count) {
                (Some(f), Some(c)) => Some(format!("'{}' from={} in={} count={}", keyword, f, conv, c)),
                (None, Some(c)) => Some(format!("'{}' in={} count={}", keyword, conv, c)),
                _ => Some(format!("'{}' in={}", keyword, conv)),
            }
        }
        "list_files" => {
            let path = params.get("path").and_then(|v| v.as_str()).unwrap_or("?");
            let entries = result
                .and_then(|r| r.get("entries"))
                .and_then(|v| v.as_u64());
            match entries {
                Some(e) => Some(format!("path={} entries={}", path, e)),
                None => Some(format!("path={}", path)),
            }
        }
        _ => None, // No special summary for other tools
    }
}

/// Truncate a string to max_len, appending "..." if truncated.
pub fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}

/// Summarize tool params by type for logging.
pub fn summarize_tool_params(tool_name: &str, params: &Value) -> String {
    match tool_name {
        "read_file" | "peek_file" | "write_file" => {
            params.get("path").and_then(|v| v.as_str()).unwrap_or("?").to_string()
        }
        "shell" | "safe_shell" => {
            let cmd = params.get("command").and_then(|v| v.as_str()).unwrap_or("?");
            truncate(cmd, 60)
        }
        "http" => {
            let method = params.get("method").and_then(|v| v.as_str()).unwrap_or("GET");
            let url = params.get("url").and_then(|v| v.as_str()).unwrap_or("?");
            format!("{} {}", method.to_uppercase(), url)
        }
        _ => truncate(&params.to_string(), 60),
    }
}


/// Context fill threshold for mid-loop dump (90%)
const CONTEXT_FILL_THRESHOLD: f64 = 0.90;
/// Target fill after hard trim — keep newest messages up to this fraction of num_ctx
const CONTEXT_TRIM_TARGET: f64 = 0.30;

/// Tool kinds tracked for dedup.
#[derive(Debug, Clone)]
enum DedupToolKind {
    ReadFull,
    ReadRange,
    Write,
    EditFile,
    CargoCheck,
}

/// Replace older tool results with stubs when superseded by later results.
///
/// Covers: read_file (full & range), write_file, edit_file, and cargo check/build shell commands.
/// Rules:
/// - read_file (full): keep only the last result per path
/// - read_file (range): drop if a later full-read or write exists for same path
/// Check if a shell command contains cargo check/build/test/run (for dedup).
/// Uses `contains()` so timeout prefixes, cd prefixes, etc. all match naturally.
pub(crate) fn is_cargo_dedup_command(cmd: &str) -> bool {
    cmd.contains("cargo check")
        || cmd.contains("cargo build")
        || cmd.contains("cargo test")
        || cmd.contains("cargo run")
}

/// - write_file: keep only the last result per path
/// - edit_file: drop if a later full-read or write exists for same path
/// - shell (cargo check/build/test only): keep only the last result globally
fn dedup_tool_results(messages: &mut Vec<ChatMessage>) {
    use std::collections::{HashMap, HashSet};

    // Pass 1: scan assistant messages, build tool_call_id → (kind, path) map
    // Also track each assistant message's index and its tool_call ids
    let mut tool_info: HashMap<String, (DedupToolKind, Option<String>)> = HashMap::new();
    let mut assistant_tc_ids: Vec<(usize, Vec<String>)> = Vec::new();
    for (i, msg) in messages.iter().enumerate() {
        if let Some(ref tcs) = msg.tool_calls {
            let mut ids = Vec::new();
            for tc in tcs {
                ids.push(tc.id.clone());
                match tc.name.as_str() {
                    "read_file" => {
                        if let Some(path) = tc.arguments.get("path").and_then(|v| v.as_str()) {
                            tool_info
                                .insert(tc.id.clone(), (DedupToolKind::ReadFull, Some(path.to_string())));
                        }
                    }
                    "peek_file" => {
                        if let Some(path) = tc.arguments.get("path").and_then(|v| v.as_str()) {
                            tool_info
                                .insert(tc.id.clone(), (DedupToolKind::ReadRange, Some(path.to_string())));
                        }
                    }
                    "write_file" => {
                        if let Some(path) = tc.arguments.get("path").and_then(|v| v.as_str()) {
                            tool_info
                                .insert(tc.id.clone(), (DedupToolKind::Write, Some(path.to_string())));
                        }
                    }
                    "edit_file" => {
                        if let Some(path) = tc.arguments.get("path").and_then(|v| v.as_str()) {
                            tool_info
                                .insert(tc.id.clone(), (DedupToolKind::EditFile, Some(path.to_string())));
                        }
                    }
                    "shell" | "safe_shell" => {
                        let cmd = tc
                            .arguments
                            .get("command")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        if is_cargo_dedup_command(cmd) {
                            tool_info.insert(tc.id.clone(), (DedupToolKind::CargoCheck, None));
                        }
                    }
                    _ => {}
                }
            }
            assistant_tc_ids.push((i, ids));
        }
    }

    if tool_info.is_empty() {
        return;
    }

    // Pass 2: scan tool result messages, collect per-path indices and cargo indices
    let mut read_full: HashMap<String, Vec<usize>> = HashMap::new();
    let mut write_results: HashMap<String, Vec<usize>> = HashMap::new();
    let mut edit_results: HashMap<String, Vec<usize>> = HashMap::new();
    let mut read_range: HashMap<String, Vec<usize>> = HashMap::new();
    let mut cargo_results: Vec<usize> = Vec::new();
    let mut result_idx_to_tc_id: HashMap<usize, String> = HashMap::new();

    for (i, msg) in messages.iter().enumerate() {
        if let Some(ref tcid) = msg.tool_call_id {
            if let Some(info) = tool_info.get(tcid) {
                result_idx_to_tc_id.insert(i, tcid.clone());
                match info.0 {
                    DedupToolKind::ReadFull => {
                        if let Some(ref path) = info.1 {
                            read_full.entry(path.clone()).or_default().push(i);
                        }
                    }
                    DedupToolKind::ReadRange => {
                        if let Some(ref path) = info.1 {
                            read_range.entry(path.clone()).or_default().push(i);
                        }
                    }
                    DedupToolKind::Write => {
                        if let Some(ref path) = info.1 {
                            write_results.entry(path.clone()).or_default().push(i);
                        }
                    }
                    DedupToolKind::EditFile => {
                        if let Some(ref path) = info.1 {
                            edit_results.entry(path.clone()).or_default().push(i);
                        }
                    }
                    DedupToolKind::CargoCheck => {
                        cargo_results.push(i);
                    }
                }
            }
        }
    }

    // Pass 3: apply dedup rules, collect tool result indices to remove
    let mut to_remove: HashSet<usize> = HashSet::new();

    // read_file (full): keep last per path, remove earlier
    for (_path, indices) in &read_full {
        if indices.len() > 1 {
            let last = *indices.last().unwrap();
            for &idx in indices {
                if idx != last {
                    to_remove.insert(idx);
                }
            }
        }
    }

    // write_file: keep last per path, remove earlier
    for (_path, indices) in &write_results {
        if indices.len() > 1 {
            let last = *indices.last().unwrap();
            for &idx in indices {
                if idx != last {
                    to_remove.insert(idx);
                }
            }
        }
    }

    // cargo check/build/test: keep last globally, remove earlier
    if cargo_results.len() > 1 {
        let last = *cargo_results.last().unwrap();
        for &idx in &cargo_results {
            if idx != last {
                to_remove.insert(idx);
            }
        }
    }

    // Build latest "fresh view" per path: last full-read or write index
    let mut latest_fresh: HashMap<String, usize> = HashMap::new();
    for (path, indices) in &read_full {
        let last = *indices.last().unwrap();
        latest_fresh
            .entry(path.clone())
            .and_modify(|v| {
                if last > *v {
                    *v = last;
                }
            })
            .or_insert(last);
    }
    for (path, indices) in &write_results {
        let last = *indices.last().unwrap();
        latest_fresh
            .entry(path.clone())
            .and_modify(|v| {
                if last > *v {
                    *v = last;
                }
            })
            .or_insert(last);
    }

    // read_file (range): remove if a later fresh view exists for same path
    for (path, indices) in &read_range {
        if let Some(&fresh_idx) = latest_fresh.get(path) {
            for &idx in indices {
                if idx < fresh_idx {
                    to_remove.insert(idx);
                }
            }
        }
    }

    // edit_file: remove if a later fresh view exists for same path
    for (path, indices) in &edit_results {
        if let Some(&fresh_idx) = latest_fresh.get(path) {
            for &idx in indices {
                if idx < fresh_idx {
                    to_remove.insert(idx);
                }
            }
        }
    }

    // Pass 4: remove assistant messages whose tool_calls are ALL removed
    let removed_tc_ids: HashSet<String> = to_remove
        .iter()
        .filter_map(|idx| result_idx_to_tc_id.get(idx).cloned())
        .collect();
    for (asst_idx, tc_ids) in &assistant_tc_ids {
        if !tc_ids.is_empty() && tc_ids.iter().all(|id| removed_tc_ids.contains(id)) {
            to_remove.insert(*asst_idx);
        }
    }

    // Apply removals
    if !to_remove.is_empty() {
        let mut idx = 0;
        messages.retain(|_| {
            let keep = !to_remove.contains(&idx);
            idx += 1;
            keep
        });
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
    /// Optional channel to stream tool executions incrementally as they complete.
    /// When set, each ToolExecution is sent through the channel immediately after the tool
    /// finishes, enabling real-time persistence to the conversation DB.
    pub tool_trace_tx: Option<tokio::sync::mpsc::Sender<ToolExecution>>,
    /// Optional cancellation flag. When set to true, the agent stops the tool loop
    /// at the next iteration boundary and returns a Cancelled error.
    pub cancel: Option<Arc<AtomicBool>>,
    /// Context window size (tokens). When set, the tool loop will trim context at 80% fill.
    pub num_ctx: Option<u32>,
    /// Optional channel to forward log messages to the daemon logger.
    /// When None, falls back to eprintln.
    pub log_tx: Option<mpsc::Sender<String>>,
}

impl Default for ThinkOptions {
    fn default() -> Self {
        Self {
            max_iterations: 25,
            system_prompt: None,
            reflection: None,
            auto_memory: None,
            stream: false,
            retry_policy: Some(RetryPolicy::default()),
            conversation_history: None,
            external_tools: None,
            tool_trace_tx: None,
            cancel: None,
            num_ctx: None,
            log_tx: None,
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
    /// Per-iteration LLM token stats for inline persistence by the trace persister.
    pub iter_tokens_in: Option<u32>,
    pub iter_tokens_out: Option<u32>,
    pub iter_prompt_eval_ns: Option<u64>,
    /// Per-iteration LLM wall-clock duration in milliseconds.
    pub iter_duration_ms: Option<u64>,
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
    /// Input tokens from the last LLM call in this think operation
    pub tokens_in: Option<u32>,
    /// Output tokens from the last LLM call in this think operation
    pub tokens_out: Option<u32>,
    /// Prompt eval duration from the last LLM call in nanoseconds (Ollama-specific).
    /// When KV caching is active, this drops significantly.
    pub prompt_eval_duration_ns: Option<u64>,
    /// Wall-clock duration of the last LLM call in milliseconds.
    pub duration_ms: Option<u64>,
}

/// Maximum number of messages to retain in conversation history.
/// When exceeded, oldest messages are removed.
const MAX_HISTORY_LEN: usize = 50;

/// Maximum tool result size in bytes before truncation (128 KB).
const MAX_TOOL_RESULT_BYTES: usize = 131_072;

/// Truncate a tool result if it exceeds `max_bytes`, keeping the beginning.
fn truncate_tool_result(result: String, max_bytes: usize) -> String {
    if result.len() <= max_bytes {
        return result;
    }
    let total = result.len();
    let cut = result.floor_char_boundary(max_bytes);
    let size = |b: usize| -> String {
        if b >= 1_048_576 {
            format!("{:.1} MB", b as f64 / 1_048_576.0)
        } else {
            format!("{:.1} KB", b as f64 / 1024.0)
        }
    };
    format!(
        "{}\n\n[output truncated: {}, showing first {}]",
        &result[..cut],
        size(total),
        size(cut),
    )
}

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

    /// Resolve which tools to send to the LLM. Uses external_tools when provided (hybrid mode),
    /// otherwise falls back to registered tools. Returns None for empty lists since some
    /// models don't support empty tool arrays.
    fn resolve_tools(&self, external_tools: &Option<Vec<ToolSpec>>) -> Option<Vec<ToolSpec>> {
        let tools_list = external_tools
            .clone()
            .unwrap_or_else(|| self.list_tools_for_llm());
        if tools_list.is_empty() {
            None
        } else {
            Some(tools_list)
        }
    }

    /// Combine memory context and system prompt into a single effective system prompt.
    fn combine_prompts(
        memory_context: &Option<String>,
        system_prompt: &Option<String>,
    ) -> Option<String> {
        match (memory_context, system_prompt) {
            (Some(mem), Some(sys)) => Some(format!("{}\n\n{}", mem, sys)),
            (Some(mem), None) => Some(mem.clone()),
            (None, Some(sys)) => Some(sys.clone()),
            (None, None) => None,
        }
    }

    /// Build the task string, prepending any pending inbox messages.
    fn build_effective_task(&mut self, task: &str) -> String {
        let pending_messages = self.drain_inbox();
        if pending_messages.is_empty() {
            return task.to_string();
        }
        let inbox_text = pending_messages
            .iter()
            .map(|msg| format!("[from: {}] {}", msg.from, msg.content))
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "You have messages from other agents:\n{}\n\nTo reply, use the send_message tool with the sender's name.\n\nCurrent task: {}",
            inbox_text, task
        )
    }

    /// Build the initial message list for an LLM call: system prompt, internal history,
    /// conversation history from options, and (optionally) the user message.
    fn build_messages(
        &self,
        effective_system_prompt: &Option<String>,
        conversation_history: &Option<Vec<ChatMessage>>,
        user_content: Option<&str>,
    ) -> Vec<ChatMessage> {
        let mut messages: Vec<ChatMessage> = Vec::new();
        if let Some(system) = effective_system_prompt {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: Some(system.clone()),
                tool_call_id: None,
                tool_calls: None,
            });
        }
        messages.extend(self.history.clone());
        if let Some(extra_history) = conversation_history {
            messages.extend(extra_history.clone());
        }
        if let Some(content) = user_content {
            messages.push(ChatMessage {
                role: "user".to_string(),
                content: Some(content.to_string()),
                tool_call_id: None,
                tool_calls: None,
            });
        }
        messages
    }

    /// Call the LLM with optional retry policy.
    async fn call_llm(
        &self,
        llm: &Arc<dyn LLM>,
        messages: &[ChatMessage],
        tools: &Option<Vec<ToolSpec>>,
        retry_policy: &Option<RetryPolicy>,
    ) -> Result<crate::llm::LLMResponse, crate::error::AgentError> {
        if let Some(policy) = retry_policy {
            let llm_ref = llm.clone();
            let msgs = messages.to_vec();
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
            self.emit_retry_events(&observer, policy, result.attempts)
                .await;
            result
                .result
                .map_err(|e| crate::error::AgentError::LlmError(e.message))
        } else {
            llm.chat_complete(messages.to_vec(), tools.clone())
                .await
                .map_err(|e| crate::error::AgentError::LlmError(e.message))
        }
    }

    /// Call the LLM with streaming and optional retry policy.
    async fn call_llm_stream(
        &self,
        llm: &Arc<dyn LLM>,
        messages: &[ChatMessage],
        tools: &Option<Vec<ToolSpec>>,
        retry_policy: &Option<RetryPolicy>,
        token_tx: &mpsc::Sender<String>,
    ) -> Result<crate::llm::LLMResponse, crate::error::AgentError> {
        if let Some(policy) = retry_policy {
            let llm_ref = llm.clone();
            let msgs = messages.to_vec();
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
            self.emit_retry_events(&observer, policy, result.attempts)
                .await;
            result
                .result
                .map_err(|e| crate::error::AgentError::LlmError(e.message))
        } else {
            llm.chat_complete_stream(messages.to_vec(), tools.clone(), token_tx.clone())
                .await
                .map_err(|e| crate::error::AgentError::LlmError(e.message))
        }
    }

    /// Emit retry events when an LLM call required more than one attempt.
    async fn emit_retry_events(
        &self,
        observer: &Option<Arc<dyn Observer>>,
        policy: &RetryPolicy,
        attempts: usize,
    ) {
        if attempts > 1 {
            if let Some(obs) = observer {
                for attempt in 1..attempts {
                    let delay_ms = policy.delay_for_attempt(attempt).as_millis() as u64;
                    obs.observe(Event::Retry {
                        operation: "llm_call".to_string(),
                        attempt,
                        delay_ms,
                    })
                    .await;
                }
            }
        }
    }

    /// Execute tool calls from an LLM response, recording results in the tool trace
    /// and appending messages to both the LLM message list and internal history.
    async fn execute_tool_calls(
        &mut self,
        tool_calls: &[crate::llm::ToolCall],
        assistant_content: &Option<String>,
        tool_names_used: &mut Vec<String>,
        tool_trace: &mut Vec<ToolExecution>,
        tool_trace_tx: &Option<mpsc::Sender<ToolExecution>>,
        messages: &mut Vec<ChatMessage>,
        iter_usage: Option<&crate::llm::UsageInfo>,
        iter_duration_ms: Option<u64>,
    ) {
        for (i, tool_call) in tool_calls.iter().enumerate() {
            tool_names_used.push(tool_call.name.clone());

            let result = self
                .call_tool(&tool_call.name, &tool_call.arguments.to_string())
                .await
                .unwrap_or_else(|e| format!("Error: {}", e));
            let result = truncate_tool_result(result, MAX_TOOL_RESULT_BYTES);

            // Only attach assistant content to the first tool call to avoid duplication
            // Only attach iter stats to the first tool call (one LLM call → one set of stats)
            let execution = ToolExecution {
                call: tool_call.clone(),
                result: result.clone(),
                content: if i == 0 {
                    assistant_content.clone()
                } else {
                    None
                },
                iter_tokens_in: if i == 0 { iter_usage.map(|u| u.prompt_tokens) } else { None },
                iter_tokens_out: if i == 0 { iter_usage.map(|u| u.completion_tokens) } else { None },
                iter_prompt_eval_ns: if i == 0 { iter_usage.and_then(|u| u.prompt_eval_duration_ns) } else { None },
                iter_duration_ms: if i == 0 { iter_duration_ms } else { None },
            };
            tool_trace.push(execution.clone());

            if let Some(tx) = tool_trace_tx {
                let _ = tx.send(execution).await;
            }

            let tool_message = ChatMessage {
                role: "tool".to_string(),
                content: Some(result),
                tool_call_id: Some(tool_call.id.clone()),
                tool_calls: None,
            };
            self.history.push(tool_message.clone());
            messages.push(tool_message);
        }
    }

    /// Emit AgentComplete and (on failure) Error events after a think operation.
    async fn emit_completion(
        &self,
        start: Instant,
        result: &Result<ThinkResult, crate::error::AgentError>,
    ) {
        let duration_ms = start.elapsed().as_millis() as u64;
        self.emit(Event::AgentComplete {
            agent_id: self.id.clone(),
            duration_ms,
            success: result.is_ok(),
        })
        .await;
        if let Err(e) = result {
            self.emit(Event::Error {
                context: format!("agent:{}", self.id),
                message: e.to_string(),
            })
            .await;
        }
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
        let start = Instant::now();
        self.emit(Event::AgentStart {
            agent_id: self.id.clone(),
            task: task.to_string(),
        })
        .await;

        let result = self.think_with_options_inner(task, options).await;

        self.emit_completion(start, &result).await;
        result
    }

    /// Inner implementation of think_with_options (without event wrapper).
    async fn think_with_options_inner(
        &mut self,
        task: &str,
        options: ThinkOptions,
    ) -> Result<ThinkResult, crate::error::AgentError> {
        let effective_task = self.build_effective_task(task);
        let llm = self
            .llm
            .clone()
            .ok_or_else(|| crate::error::AgentError::LlmError("No LLM attached".to_string()))?;
        let tools = self.resolve_tools(&options.external_tools);
        let memory_context = self.build_memory_context(&options.auto_memory).await;
        let effective_system_prompt = Self::combine_prompts(&memory_context, &options.system_prompt);

        // Build initial messages: system prompt, internal history, conversation history,
        // and the user message (skipped when empty -- already in conversation_history).
        let user_content = if effective_task.is_empty() {
            None
        } else {
            Some(effective_task.as_str())
        };
        let mut messages = self.build_messages(
            &effective_system_prompt,
            &options.conversation_history,
            user_content,
        );
        if let Some(content) = user_content {
            self.history.push(ChatMessage {
                role: "user".to_string(),
                content: Some(content.to_string()),
                tool_call_id: None,
                tool_calls: None,
            });
        }

        let mut tool_names_used: Vec<String> = Vec::new();
        let mut last_tool_calls: Option<Vec<crate::llm::ToolCall>> = None;
        let mut tool_trace: Vec<ToolExecution> = Vec::new();
        let mut last_tokens_in: Option<u32> = None;
        let mut last_tokens_out: Option<u32> = None;
        let mut last_prompt_eval_ns: Option<u64> = None;

        for _iteration in 0..options.max_iterations {
            // Check cancellation before each LLM call
            if let Some(ref cancel) = options.cancel {
                if cancel.load(Ordering::Relaxed) {
                    self.trim_history();
                    return Err(crate::error::AgentError::Cancelled);
                }
            }

            self.dump_context(&tools, &messages);

            let llm_start = Instant::now();
            let response = self
                .call_llm(&llm, &messages, &tools, &options.retry_policy)
                .await?;

            let llm_duration_ms = llm_start.elapsed().as_millis() as u64;
            self.emit(Event::LlmCall {
                model: llm.model_name().to_string(),
                tokens_in: response.usage.as_ref().map(|u| u.prompt_tokens),
                tokens_out: response.usage.as_ref().map(|u| u.completion_tokens),
                duration_ms: llm_duration_ms,
            })
            .await;

            if let Some(ref usage) = response.usage {
                last_tokens_in = Some(usage.prompt_tokens);
                last_tokens_out = Some(usage.completion_tokens);
                last_prompt_eval_ns = usage.prompt_eval_duration_ns;
            }

            if response.tool_calls.is_empty() {
                let final_response = response
                    .content
                    .unwrap_or_else(|| "No response".to_string());

                // Store stripped version in history (thinking tags removed)
                let stripped_response = strip_thinking(&final_response);
                self.history.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: Some(stripped_response),
                    tool_call_id: None,
                    tool_calls: None,
                });

                self.trim_history();

                let response_text = if let Some(ref config) = options.reflection {
                    self.reflect_and_revise(task, &final_response, config, &options)
                        .await?
                } else {
                    final_response
                };

                return Ok(ThinkResult {
                    response: response_text,
                    tools_used: !tool_names_used.is_empty(),
                    tool_names: tool_names_used,
                    last_tool_calls,
                    tool_trace,
                    tokens_in: last_tokens_in,
                    tokens_out: last_tokens_out,
                    prompt_eval_duration_ns: last_prompt_eval_ns,
                    duration_ms: Some(llm_duration_ms),
                });
            }

            last_tool_calls = Some(response.tool_calls.clone());

            let assistant_content = response.content.as_ref().map(|c| strip_thinking(c));
            let assistant_message = ChatMessage {
                role: "assistant".to_string(),
                content: assistant_content.clone(),
                tool_call_id: None,
                tool_calls: Some(response.tool_calls.clone()),
            };
            self.history.push(assistant_message.clone());
            messages.push(assistant_message);

            self.execute_tool_calls(
                &response.tool_calls,
                &assistant_content,
                &mut tool_names_used,
                &mut tool_trace,
                &options.tool_trace_tx,
                &mut messages,
                response.usage.as_ref(),
                Some(llm_duration_ms),
            )
            .await;

            // If spawn_child was called this iteration, nudge the LLM to wait for results
            if response.tool_calls.iter().any(|tc| tc.name == "spawn_child") {
                messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: Some("[System: You have pending child tasks. Call wait_for_children to collect results before responding.]".to_string()),
                    tool_call_id: None,
                    tool_calls: None,
                });
            }

            // Check context fill and dump if approaching capacity
            if let Some(ctx) = options.num_ctx {
                if let (Some(t_in), Some(t_out)) = (last_tokens_in, last_tokens_out) {
                    let fill = (t_in + t_out) as f64 / ctx as f64;
                    if fill >= CONTEXT_FILL_THRESHOLD {
                        // First resort: dedup stale read_file results
                        dedup_tool_results(&mut messages);

                        // Re-estimate fill after dedup (chars/4 as rough token proxy)
                        let est_tokens: usize = messages
                            .iter()
                            .map(|m| m.content.as_ref().map_or(0, |c| c.len()) / 4)
                            .sum();
                        let est_fill = est_tokens as f64 / ctx as f64;

                        // Hard trim only if dedup wasn't enough
                        if est_fill >= CONTEXT_FILL_THRESHOLD {
                            let target_tokens = (ctx as f64 * CONTEXT_TRIM_TARGET) as usize;
                            let mut budget = 0usize;
                            let mut keep_from = messages.len();
                            for i in (1..messages.len()).rev() {
                                let msg_tokens =
                                    messages[i].content.as_ref().map_or(0, |c| c.len()) / 4;
                                if budget + msg_tokens > target_tokens {
                                    break;
                                }
                                budget += msg_tokens;
                                keep_from = i;
                            }
                            if keep_from > 1 {
                                let mut trimmed = vec![messages[0].clone()];
                                trimmed.extend_from_slice(&messages[keep_from..]);
                                messages = trimmed;
                                self.history.clear();
                            }
                        }
                    }
                }
            }

            // Inject tool budget nudge
            if let Some(nudge) = tool_budget_nudge(_iteration + 1, options.max_iterations) {
                messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: Some(nudge),
                    tool_call_id: None,
                    tool_calls: None,
                });
            }
        }

        self.trim_history();
        Ok(ThinkResult {
            response: format!("[Max iterations reached: {}]", options.max_iterations),
            tools_used: !tool_names_used.is_empty(),
            tool_names: tool_names_used,
            last_tool_calls,
            tool_trace,
            tokens_in: last_tokens_in,
            tokens_out: last_tokens_out,
            prompt_eval_duration_ns: last_prompt_eval_ns,
            duration_ms: None,
        })
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
        let start = Instant::now();
        self.emit(Event::AgentStart {
            agent_id: self.id.clone(),
            task: task.to_string(),
        })
        .await;

        let result = self
            .think_streaming_with_options_inner(task, options, token_tx)
            .await;

        self.emit_completion(start, &result).await;
        result
    }

    /// Make exactly one LLM call and return without executing tools or looping.
    /// Returns ThinkResult with `last_tool_calls` populated if the LLM requested tools.
    /// The caller (daemon's run_tool_loop) is responsible for tool execution and iteration.
    pub async fn think_single_turn(
        &mut self,
        task: &str,
        options: ThinkOptions,
    ) -> Result<ThinkResult, crate::error::AgentError> {
        let effective_task = self.build_effective_task(task);
        let llm = self
            .llm
            .clone()
            .ok_or_else(|| crate::error::AgentError::LlmError("No LLM attached".to_string()))?;
        let tools = self.resolve_tools(&options.external_tools);
        let memory_context = self.build_memory_context(&options.auto_memory).await;
        let effective_system_prompt =
            Self::combine_prompts(&memory_context, &options.system_prompt);

        let user_content = if effective_task.is_empty() {
            None
        } else {
            Some(effective_task.as_str())
        };
        let messages =
            self.build_messages(&effective_system_prompt, &options.conversation_history, user_content);

        self.dump_context(&tools, &messages);

        let llm_start = Instant::now();
        let response = self
            .call_llm(&llm, &messages, &tools, &options.retry_policy)
            .await?;
        let llm_duration_ms = llm_start.elapsed().as_millis() as u64;

        let tokens_in = response.usage.as_ref().map(|u| u.prompt_tokens);
        let tokens_out = response.usage.as_ref().map(|u| u.completion_tokens);
        let prompt_eval_ns = response.usage.as_ref().and_then(|u| u.prompt_eval_duration_ns);

        let last_tool_calls = if response.tool_calls.is_empty() {
            None
        } else {
            Some(response.tool_calls)
        };

        Ok(ThinkResult {
            response: response.content.unwrap_or_default(),
            tools_used: last_tool_calls.is_some(),
            tool_names: last_tool_calls
                .as_ref()
                .map(|tcs| tcs.iter().map(|tc| tc.name.clone()).collect())
                .unwrap_or_default(),
            last_tool_calls,
            tool_trace: Vec::new(),
            tokens_in,
            tokens_out,
            prompt_eval_duration_ns: prompt_eval_ns,
            duration_ms: Some(llm_duration_ms),
        })
    }

    /// Streaming variant of think_single_turn.
    /// Makes exactly one streaming LLM call, sends tokens through channel, returns without
    /// executing tools or looping.
    pub async fn think_single_turn_streaming(
        &mut self,
        task: &str,
        options: ThinkOptions,
        token_tx: mpsc::Sender<String>,
    ) -> Result<ThinkResult, crate::error::AgentError> {
        let effective_task = self.build_effective_task(task);
        let llm = self
            .llm
            .clone()
            .ok_or_else(|| crate::error::AgentError::LlmError("No LLM attached".to_string()))?;
        let tools = self.resolve_tools(&options.external_tools);
        let memory_context = self.build_memory_context(&options.auto_memory).await;
        let effective_system_prompt =
            Self::combine_prompts(&memory_context, &options.system_prompt);

        let user_content = if effective_task.is_empty() {
            None
        } else {
            Some(effective_task.as_str())
        };
        let messages =
            self.build_messages(&effective_system_prompt, &options.conversation_history, user_content);

        self.dump_context(&tools, &messages);

        let llm_start = Instant::now();
        let response = self
            .call_llm_stream(&llm, &messages, &tools, &options.retry_policy, &token_tx)
            .await?;
        let llm_duration_ms = llm_start.elapsed().as_millis() as u64;

        let tokens_in = response.usage.as_ref().map(|u| u.prompt_tokens);
        let tokens_out = response.usage.as_ref().map(|u| u.completion_tokens);
        let prompt_eval_ns = response.usage.as_ref().and_then(|u| u.prompt_eval_duration_ns);

        let last_tool_calls = if response.tool_calls.is_empty() {
            None
        } else {
            Some(response.tool_calls)
        };

        Ok(ThinkResult {
            response: response.content.unwrap_or_default(),
            tools_used: last_tool_calls.is_some(),
            tool_names: last_tool_calls
                .as_ref()
                .map(|tcs| tcs.iter().map(|tc| tc.name.clone()).collect())
                .unwrap_or_default(),
            last_tool_calls,
            tool_trace: Vec::new(),
            tokens_in,
            tokens_out,
            prompt_eval_duration_ns: prompt_eval_ns,
            duration_ms: Some(llm_duration_ms),
        })
    }

    /// Inner implementation of streaming think (without event wrapper).
    async fn think_streaming_with_options_inner(
        &mut self,
        task: &str,
        options: ThinkOptions,
        token_tx: mpsc::Sender<String>,
    ) -> Result<ThinkResult, crate::error::AgentError> {
        let effective_task = self.build_effective_task(task);
        let llm = self
            .llm
            .clone()
            .ok_or_else(|| crate::error::AgentError::LlmError("No LLM attached".to_string()))?;
        let tools = self.resolve_tools(&options.external_tools);
        let memory_context = self.build_memory_context(&options.auto_memory).await;
        let effective_system_prompt =
            Self::combine_prompts(&memory_context, &options.system_prompt);

        let user_content = if effective_task.is_empty() {
            None
        } else {
            Some(effective_task.as_str())
        };
        let mut messages = self.build_messages(
            &effective_system_prompt,
            &options.conversation_history,
            user_content,
        );
        if let Some(content) = user_content {
            self.history.push(ChatMessage {
                role: "user".to_string(),
                content: Some(content.to_string()),
                tool_call_id: None,
                tool_calls: None,
            });
        }

        let mut tool_names_used: Vec<String> = Vec::new();
        let mut last_tool_calls: Option<Vec<crate::llm::ToolCall>> = None;
        let mut tool_trace: Vec<ToolExecution> = Vec::new();
        let mut last_tokens_in: Option<u32> = None;
        let mut last_tokens_out: Option<u32> = None;
        let mut last_prompt_eval_ns: Option<u64> = None;

        for _iteration in 0..options.max_iterations {
            // Check cancellation before each LLM call
            if let Some(ref cancel) = options.cancel {
                if cancel.load(Ordering::Relaxed) {
                    self.trim_history();
                    return Err(crate::error::AgentError::Cancelled);
                }
            }

            self.dump_context(&tools, &messages);

            let llm_start = Instant::now();
            let response = self
                .call_llm_stream(&llm, &messages, &tools, &options.retry_policy, &token_tx)
                .await?;

            let llm_duration_ms = llm_start.elapsed().as_millis() as u64;
            self.emit(Event::LlmCall {
                model: llm.model_name().to_string(),
                tokens_in: response.usage.as_ref().map(|u| u.prompt_tokens),
                tokens_out: response.usage.as_ref().map(|u| u.completion_tokens),
                duration_ms: llm_duration_ms,
            })
            .await;

            if let Some(ref usage) = response.usage {
                last_tokens_in = Some(usage.prompt_tokens);
                last_tokens_out = Some(usage.completion_tokens);
                last_prompt_eval_ns = usage.prompt_eval_duration_ns;
            }

            if response.tool_calls.is_empty() {
                let final_response = response
                    .content
                    .unwrap_or_else(|| "No response".to_string());

                let stripped_response = strip_thinking(&final_response);
                self.history.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: Some(stripped_response),
                    tool_call_id: None,
                    tool_calls: None,
                });

                self.trim_history();

                return Ok(ThinkResult {
                    response: final_response,
                    tools_used: !tool_names_used.is_empty(),
                    tool_names: tool_names_used,
                    last_tool_calls,
                    tool_trace,
                    tokens_in: last_tokens_in,
                    tokens_out: last_tokens_out,
                    prompt_eval_duration_ns: last_prompt_eval_ns,
                    duration_ms: Some(llm_duration_ms),
                });
            }

            last_tool_calls = Some(response.tool_calls.clone());

            let assistant_content = response.content.as_ref().map(|c| strip_thinking(c));
            let assistant_message = ChatMessage {
                role: "assistant".to_string(),
                content: assistant_content.clone(),
                tool_call_id: None,
                tool_calls: Some(response.tool_calls.clone()),
            };
            self.history.push(assistant_message.clone());
            messages.push(assistant_message);

            self.execute_tool_calls(
                &response.tool_calls,
                &assistant_content,
                &mut tool_names_used,
                &mut tool_trace,
                &options.tool_trace_tx,
                &mut messages,
                response.usage.as_ref(),
                Some(llm_duration_ms),
            )
            .await;

            // If spawn_child was called this iteration, nudge the LLM to wait for results
            if response.tool_calls.iter().any(|tc| tc.name == "spawn_child") {
                messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: Some("[System: You have pending child tasks. Call wait_for_children to collect results before responding.]".to_string()),
                    tool_call_id: None,
                    tool_calls: None,
                });
            }

            // Check context fill and dump if approaching capacity
            if let Some(ctx) = options.num_ctx {
                if let (Some(t_in), Some(t_out)) = (last_tokens_in, last_tokens_out) {
                    let fill = (t_in + t_out) as f64 / ctx as f64;
                    if fill >= CONTEXT_FILL_THRESHOLD {
                        // First resort: dedup stale read_file results
                        dedup_tool_results(&mut messages);

                        // Re-estimate fill after dedup (chars/4 as rough token proxy)
                        let est_tokens: usize = messages
                            .iter()
                            .map(|m| m.content.as_ref().map_or(0, |c| c.len()) / 4)
                            .sum();
                        let est_fill = est_tokens as f64 / ctx as f64;

                        // Hard trim only if dedup wasn't enough
                        if est_fill >= CONTEXT_FILL_THRESHOLD {
                            let target_tokens = (ctx as f64 * CONTEXT_TRIM_TARGET) as usize;
                            let mut budget = 0usize;
                            let mut keep_from = messages.len();
                            for i in (1..messages.len()).rev() {
                                let msg_tokens =
                                    messages[i].content.as_ref().map_or(0, |c| c.len()) / 4;
                                if budget + msg_tokens > target_tokens {
                                    break;
                                }
                                budget += msg_tokens;
                                keep_from = i;
                            }
                            if keep_from > 1 {
                                let mut trimmed = vec![messages[0].clone()];
                                trimmed.extend_from_slice(&messages[keep_from..]);
                                messages = trimmed;
                                self.history.clear();
                            }
                        }
                    }
                }
            }

            let _ = token_tx.send("\n".to_string()).await;
        }

        self.trim_history();
        Ok(ThinkResult {
            response: format!("[Max iterations reached: {}]", options.max_iterations),
            tools_used: !tool_names_used.is_empty(),
            tool_names: tool_names_used,
            last_tool_calls,
            tool_trace,
            tokens_in: last_tokens_in,
            tokens_out: last_tokens_out,
            prompt_eval_duration_ns: last_prompt_eval_ns,
            duration_ms: None,
        })
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
            let llm = self
                .llm
                .clone()
                .ok_or_else(|| crate::error::AgentError::LlmError("No LLM attached".to_string()))?;

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

            let reflection = self
                .call_llm(&llm, &reflection_messages, &None, &options.retry_policy)
                .await?;

            let reflection_text = reflection.content.unwrap_or_default();

            if reflection_text.to_uppercase().starts_with("ACCEPTED") {
                return Ok(current_response);
            }

            let feedback = if reflection_text.to_uppercase().starts_with("REVISE:") {
                reflection_text[7..].trim().to_string()
            } else {
                reflection_text.clone()
            };

            let revision_prompt = format!(
                "Original task: {}\n\nYour previous response:\n{}\n\nFeedback:\n{}\n\nPlease provide an improved response.",
                original_task, current_response, feedback
            );

            let revision_options = ThinkOptions {
                max_iterations: options.max_iterations,
                system_prompt: options.system_prompt.clone(),
                reflection: None, // Don't recurse
                auto_memory: options.auto_memory.clone(),
                stream: false,
                retry_policy: options.retry_policy.clone(),
                conversation_history: options.conversation_history.clone(),
                external_tools: options.external_tools.clone(),
                tool_trace_tx: None,
                cancel: options.cancel.clone(),
                num_ctx: options.num_ctx,
                log_tx: options.log_tx.clone(),
            };

            current_response = self
                .run_agentic_loop(&revision_prompt, &revision_options)
                .await?;
        }

        Ok(current_response)
    }

    /// Core agentic loop extracted for reuse (used by reflect_and_revise).
    /// Unlike think_with_options_inner, this does not manage internal history or emit events.
    async fn run_agentic_loop(
        &mut self,
        task: &str,
        options: &ThinkOptions,
    ) -> Result<String, crate::error::AgentError> {
        let llm = self
            .llm
            .clone()
            .ok_or_else(|| crate::error::AgentError::LlmError("No LLM attached".to_string()))?;
        let tools = self.resolve_tools(&options.external_tools);
        let memory_context = self.build_memory_context(&options.auto_memory).await;
        let effective_system_prompt =
            Self::combine_prompts(&memory_context, &options.system_prompt);

        let mut messages =
            self.build_messages(&effective_system_prompt, &options.conversation_history, Some(task));

        for _iteration in 0..options.max_iterations {
            let response = self
                .call_llm(&llm, &messages, &tools, &options.retry_policy)
                .await?;

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
                let result = truncate_tool_result(result, MAX_TOOL_RESULT_BYTES);

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
        assert_eq!(opts.max_iterations, 25);
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
            tool_trace_tx: None,
            cancel: None,
            num_ctx: None,
            log_tx: None,
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
            tool_trace_tx: None,
            cancel: None,
            num_ctx: None,
            log_tx: None,
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

    // =========================================================================
    // Checkpoint summary tests
    // =========================================================================

    #[test]
    fn test_truncate_short_string() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_exact_length() {
        assert_eq!(truncate("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_long_string() {
        assert_eq!(truncate("hello world", 5), "hello...");
    }

    #[test]
    fn test_truncate_empty() {
        assert_eq!(truncate("", 5), "");
    }

    #[test]
    fn test_summarize_tool_params_read_file() {
        let params = json!({"path": "/tmp/test.rs"});
        assert_eq!(summarize_tool_params("read_file", &params), "/tmp/test.rs");
    }

    #[test]
    fn test_summarize_tool_params_write_file() {
        let params = json!({"path": "src/main.rs", "content": "fn main() {}"});
        assert_eq!(summarize_tool_params("write_file", &params), "src/main.rs");
    }

    #[test]
    fn test_summarize_tool_params_shell() {
        let params = json!({"command": "cargo check"});
        assert_eq!(summarize_tool_params("shell", &params), "cargo check");
    }

    #[test]
    fn test_summarize_tool_params_shell_long() {
        let long_cmd = "a".repeat(100);
        let params = json!({"command": long_cmd});
        let result = summarize_tool_params("shell", &params);
        assert!(result.len() <= 63); // 60 + "..."
        assert!(result.ends_with("..."));
    }

    #[test]
    fn test_summarize_tool_params_http() {
        let params = json!({"method": "post", "url": "https://api.example.com/data"});
        assert_eq!(
            summarize_tool_params("http", &params),
            "POST https://api.example.com/data"
        );
    }

    #[test]
    fn test_summarize_tool_params_unknown() {
        let params = json!({"key": "value"});
        let result = summarize_tool_params("unknown_tool", &params);
        assert!(result.contains("key"));
    }

    // =========================================================================
    // expand_tilde tests
    // =========================================================================

    #[test]
    fn test_expand_tilde_with_prefix() {
        let result = expand_tilde("~/foo/bar");
        assert!(result.ends_with("foo/bar"), "got: {:?}", result);
        assert!(!result.to_string_lossy().contains('~'));
    }

    #[test]
    fn test_expand_tilde_bare() {
        let result = expand_tilde("~");
        assert!(!result.to_string_lossy().contains('~'));
    }

    #[test]
    fn test_expand_tilde_absolute() {
        let result = expand_tilde("/usr/local/bin");
        assert_eq!(result, PathBuf::from("/usr/local/bin"));
    }

    #[test]
    fn test_expand_tilde_relative() {
        let result = expand_tilde("src/main.rs");
        assert_eq!(result, PathBuf::from("src/main.rs"));
    }

    // =========================================================================
    // detect_project_root tests
    // =========================================================================

    #[test]
    fn test_detect_project_root_cargo() {
        // This file is inside the anima project which has Cargo.toml
        let root = detect_project_root(file!());
        assert!(root.is_some());
        let root = root.unwrap();
        assert!(root.join("Cargo.toml").exists(), "got: {:?}", root);
    }

    #[test]
    fn test_detect_project_root_none() {
        let root = detect_project_root("/");
        assert!(root.is_none());
    }

    // =========================================================================
    // dedup_tool_results tests
    // =========================================================================

    #[test]
    fn test_dedup_read_file_replaces_older_reads() {
        use crate::llm::{ChatMessage, ToolCall};

        let mut messages = vec![
            // System prompt
            ChatMessage {
                role: "system".into(),
                content: Some("You are an agent.".into()),
                tool_call_id: None,
                tool_calls: None,
            },
            // First read_file call for /a.rs
            ChatMessage {
                role: "assistant".into(),
                content: Some("Reading file.".into()),
                tool_call_id: None,
                tool_calls: Some(vec![ToolCall {
                    id: "tc1".into(),
                    name: "read_file".into(),
                    arguments: json!({"path": "/a.rs"}),
                }]),
            },
            // First result
            ChatMessage {
                role: "user".into(),
                content: Some("fn main() { old version }".into()),
                tool_call_id: Some("tc1".into()),
                tool_calls: None,
            },
            // Second read_file call for /a.rs
            ChatMessage {
                role: "assistant".into(),
                content: Some("Re-reading file.".into()),
                tool_call_id: None,
                tool_calls: Some(vec![ToolCall {
                    id: "tc2".into(),
                    name: "read_file".into(),
                    arguments: json!({"path": "/a.rs"}),
                }]),
            },
            // Second result
            ChatMessage {
                role: "user".into(),
                content: Some("fn main() { new version }".into()),
                tool_call_id: Some("tc2".into()),
                tool_calls: None,
            },
        ];

        dedup_tool_results(&mut messages);

        // First pair removed, 3 messages remain: system + second assistant + second result
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].content.as_ref().unwrap(), "You are an agent.");
        assert_eq!(messages[1].content.as_ref().unwrap(), "Re-reading file.");
        assert_eq!(
            messages[2].content.as_ref().unwrap(),
            "fn main() { new version }"
        );
    }

    #[test]
    fn test_dedup_read_file_different_paths_untouched() {
        use crate::llm::{ChatMessage, ToolCall};

        let mut messages = vec![
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_call_id: None,
                tool_calls: Some(vec![ToolCall {
                    id: "tc1".into(),
                    name: "read_file".into(),
                    arguments: json!({"path": "/a.rs"}),
                }]),
            },
            ChatMessage {
                role: "user".into(),
                content: Some("contents of a".into()),
                tool_call_id: Some("tc1".into()),
                tool_calls: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_call_id: None,
                tool_calls: Some(vec![ToolCall {
                    id: "tc2".into(),
                    name: "read_file".into(),
                    arguments: json!({"path": "/b.rs"}),
                }]),
            },
            ChatMessage {
                role: "user".into(),
                content: Some("contents of b".into()),
                tool_call_id: Some("tc2".into()),
                tool_calls: None,
            },
        ];

        dedup_tool_results(&mut messages);

        // Different paths — both kept
        assert_eq!(messages[1].content.as_ref().unwrap(), "contents of a");
        assert_eq!(messages[3].content.as_ref().unwrap(), "contents of b");
    }

    #[test]
    fn test_dedup_no_tracked_tools() {
        use crate::llm::{ChatMessage, ToolCall};

        let mut messages = vec![
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_call_id: None,
                tool_calls: Some(vec![ToolCall {
                    id: "tc1".into(),
                    name: "shell".into(),
                    arguments: json!({"command": "ls"}),
                }]),
            },
            ChatMessage {
                role: "user".into(),
                content: Some("file1.rs file2.rs".into()),
                tool_call_id: Some("tc1".into()),
                tool_calls: None,
            },
        ];

        let original_content = messages[1].content.clone();
        dedup_tool_results(&mut messages);

        // No tracked tool calls — nothing changed
        assert_eq!(messages[1].content, original_content);
    }

    #[test]
    fn test_dedup_write_file_results() {
        use crate::llm::{ChatMessage, ToolCall};

        let mut messages = vec![
            // First write_file to /a.rs
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_call_id: None,
                tool_calls: Some(vec![ToolCall {
                    id: "tc1".into(),
                    name: "write_file".into(),
                    arguments: json!({"path": "/a.rs", "content": "old"}),
                }]),
            },
            ChatMessage {
                role: "user".into(),
                content: Some("Wrote 10 lines (200 B) to '/a.rs'".into()),
                tool_call_id: Some("tc1".into()),
                tool_calls: None,
            },
            // Second write_file to /a.rs
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_call_id: None,
                tool_calls: Some(vec![ToolCall {
                    id: "tc2".into(),
                    name: "write_file".into(),
                    arguments: json!({"path": "/a.rs", "content": "new"}),
                }]),
            },
            ChatMessage {
                role: "user".into(),
                content: Some("Wrote 15 lines (300 B) to '/a.rs'".into()),
                tool_call_id: Some("tc2".into()),
                tool_calls: None,
            },
        ];

        dedup_tool_results(&mut messages);

        // First pair removed, 2 messages remain
        assert_eq!(messages.len(), 2);
        assert!(messages[1].content.as_ref().unwrap().contains("Wrote 15 lines"));
    }

    #[test]
    fn test_dedup_edit_file_superseded_by_read() {
        use crate::llm::{ChatMessage, ToolCall};

        let mut messages = vec![
            // edit_file on /a.rs
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_call_id: None,
                tool_calls: Some(vec![ToolCall {
                    id: "tc1".into(),
                    name: "edit_file".into(),
                    arguments: json!({"path": "/a.rs", "old_string": "x", "new_string": "y"}),
                }]),
            },
            ChatMessage {
                role: "user".into(),
                content: Some("Replaced 'x' with 'y' in /a.rs\n--- context ---".into()),
                tool_call_id: Some("tc1".into()),
                tool_calls: None,
            },
            // Later full read_file of /a.rs
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_call_id: None,
                tool_calls: Some(vec![ToolCall {
                    id: "tc2".into(),
                    name: "read_file".into(),
                    arguments: json!({"path": "/a.rs"}),
                }]),
            },
            ChatMessage {
                role: "user".into(),
                content: Some("full file contents here".into()),
                tool_call_id: Some("tc2".into()),
                tool_calls: None,
            },
        ];

        dedup_tool_results(&mut messages);

        // Edit pair removed, 2 messages remain: second assistant + second result
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].content.as_ref().unwrap(), "full file contents here");
    }

    #[test]
    fn test_dedup_edit_file_kept_when_no_later_read() {
        use crate::llm::{ChatMessage, ToolCall};

        let mut messages = vec![
            // edit_file on /a.rs — no later full read or write
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_call_id: None,
                tool_calls: Some(vec![ToolCall {
                    id: "tc1".into(),
                    name: "edit_file".into(),
                    arguments: json!({"path": "/a.rs", "old_string": "x", "new_string": "y"}),
                }]),
            },
            ChatMessage {
                role: "user".into(),
                content: Some("Replaced 'x' with 'y' in /a.rs".into()),
                tool_call_id: Some("tc1".into()),
                tool_calls: None,
            },
        ];

        dedup_tool_results(&mut messages);

        // Edit kept — no later fresh view
        assert_eq!(
            messages[1].content.as_ref().unwrap(),
            "Replaced 'x' with 'y' in /a.rs"
        );
    }

    #[test]
    fn test_dedup_cargo_check() {
        use crate::llm::{ChatMessage, ToolCall};

        let mut messages = vec![
            // First cargo check (cd-prefixed, matching real LLM output)
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_call_id: None,
                tool_calls: Some(vec![ToolCall {
                    id: "tc1".into(),
                    name: "shell".into(),
                    arguments: json!({"command": "cd ~/dev/minilang && cargo check 2>&1"}),
                }]),
            },
            ChatMessage {
                role: "user".into(),
                content: Some("error[E0308]: mismatched types\n  --> src/main.rs:10:5".into()),
                tool_call_id: Some("tc1".into()),
                tool_calls: None,
            },
            // Second cargo check (cd-prefixed)
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_call_id: None,
                tool_calls: Some(vec![ToolCall {
                    id: "tc2".into(),
                    name: "shell".into(),
                    arguments: json!({"command": "cd ~/dev/minilang && cargo check 2>&1"}),
                }]),
            },
            ChatMessage {
                role: "user".into(),
                content: Some("Finished `dev` profile".into()),
                tool_call_id: Some("tc2".into()),
                tool_calls: None,
            },
        ];

        dedup_tool_results(&mut messages);

        // First pair removed, 2 messages remain
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].content.as_ref().unwrap(), "Finished `dev` profile");
    }

    #[test]
    fn test_dedup_non_cargo_shell_kept() {
        use crate::llm::{ChatMessage, ToolCall};

        let mut messages = vec![
            // Non-cargo shell command
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_call_id: None,
                tool_calls: Some(vec![ToolCall {
                    id: "tc1".into(),
                    name: "shell".into(),
                    arguments: json!({"command": "git status"}),
                }]),
            },
            ChatMessage {
                role: "user".into(),
                content: Some("On branch main".into()),
                tool_call_id: Some("tc1".into()),
                tool_calls: None,
            },
            // Another non-cargo shell
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_call_id: None,
                tool_calls: Some(vec![ToolCall {
                    id: "tc2".into(),
                    name: "shell".into(),
                    arguments: json!({"command": "git diff"}),
                }]),
            },
            ChatMessage {
                role: "user".into(),
                content: Some("diff --git a/file ...".into()),
                tool_call_id: Some("tc2".into()),
                tool_calls: None,
            },
        ];

        dedup_tool_results(&mut messages);

        // Non-cargo shell commands are never deduped
        assert_eq!(messages[1].content.as_ref().unwrap(), "On branch main");
        assert_eq!(messages[3].content.as_ref().unwrap(), "diff --git a/file ...");
    }

    #[test]
    fn test_dedup_read_range_superseded_by_full_read() {
        use crate::llm::{ChatMessage, ToolCall};

        let mut messages = vec![
            // Range read of /a.rs
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_call_id: None,
                tool_calls: Some(vec![ToolCall {
                    id: "tc1".into(),
                    name: "read_file".into(),
                    arguments: json!({"path": "/a.rs", "start_line": 1, "end_line": 10}),
                }]),
            },
            ChatMessage {
                role: "user".into(),
                content: Some("lines 1-10 of /a.rs".into()),
                tool_call_id: Some("tc1".into()),
                tool_calls: None,
            },
            // Later full read of /a.rs
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_call_id: None,
                tool_calls: Some(vec![ToolCall {
                    id: "tc2".into(),
                    name: "read_file".into(),
                    arguments: json!({"path": "/a.rs"}),
                }]),
            },
            ChatMessage {
                role: "user".into(),
                content: Some("full contents of /a.rs".into()),
                tool_call_id: Some("tc2".into()),
                tool_calls: None,
            },
        ];

        dedup_tool_results(&mut messages);

        // Range pair removed, 2 messages remain
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].content.as_ref().unwrap(), "full contents of /a.rs");
    }

    #[test]
    fn test_dedup_edit_file_superseded_by_write() {
        use crate::llm::{ChatMessage, ToolCall};

        let mut messages = vec![
            // edit_file on /a.rs
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_call_id: None,
                tool_calls: Some(vec![ToolCall {
                    id: "tc1".into(),
                    name: "edit_file".into(),
                    arguments: json!({"path": "/a.rs", "old_string": "x", "new_string": "y"}),
                }]),
            },
            ChatMessage {
                role: "user".into(),
                content: Some("Replaced in /a.rs".into()),
                tool_call_id: Some("tc1".into()),
                tool_calls: None,
            },
            // Later write_file to /a.rs
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_call_id: None,
                tool_calls: Some(vec![ToolCall {
                    id: "tc2".into(),
                    name: "write_file".into(),
                    arguments: json!({"path": "/a.rs", "content": "new content"}),
                }]),
            },
            ChatMessage {
                role: "user".into(),
                content: Some("Wrote 5 lines (100 B) to '/a.rs'".into()),
                tool_call_id: Some("tc2".into()),
                tool_calls: None,
            },
        ];

        dedup_tool_results(&mut messages);

        // Edit pair removed, 2 messages remain
        assert_eq!(messages.len(), 2);
        assert!(messages[1].content.as_ref().unwrap().contains("Wrote 5 lines"));
    }

    #[test]
    fn test_dedup_cargo_build_also_deduped() {
        use crate::llm::{ChatMessage, ToolCall};

        let mut messages = vec![
            // cargo build (cd-prefixed)
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_call_id: None,
                tool_calls: Some(vec![ToolCall {
                    id: "tc1".into(),
                    name: "shell".into(),
                    arguments: json!({"command": "cd ~/dev/minilang && cargo build 2>&1"}),
                }]),
            },
            ChatMessage {
                role: "user".into(),
                content: Some("error: build failed".into()),
                tool_call_id: Some("tc1".into()),
                tool_calls: None,
            },
            // Later cargo check (cd-prefixed)
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_call_id: None,
                tool_calls: Some(vec![ToolCall {
                    id: "tc2".into(),
                    name: "shell".into(),
                    arguments: json!({"command": "cd ~/dev/minilang && cargo check 2>&1"}),
                }]),
            },
            ChatMessage {
                role: "user".into(),
                content: Some("Finished `dev` profile".into()),
                tool_call_id: Some("tc2".into()),
                tool_calls: None,
            },
        ];

        dedup_tool_results(&mut messages);

        // Build pair removed, 2 messages remain
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].content.as_ref().unwrap(), "Finished `dev` profile");
    }

    #[test]
    fn test_is_cargo_dedup_command() {
        // Plain commands
        assert!(is_cargo_dedup_command("cargo check"));
        assert!(is_cargo_dedup_command("cargo check 2>&1"));
        assert!(is_cargo_dedup_command("cargo build --release"));
        assert!(is_cargo_dedup_command("cargo test"));
        assert!(is_cargo_dedup_command("cargo test -- --nocapture"));

        // cargo run
        assert!(is_cargo_dedup_command("cargo run -- args"));
        assert!(is_cargo_dedup_command("cargo run --bin demo"));

        // cd-prefixed (real LLM output)
        assert!(is_cargo_dedup_command("cd ~/dev/minilang && cargo check 2>&1"));
        assert!(is_cargo_dedup_command("cd /tmp && cargo build 2>&1"));
        assert!(is_cargo_dedup_command("cd ~/dev && cargo test 2>&1"));

        // timeout-prefixed (real LLM output)
        assert!(is_cargo_dedup_command("timeout 10 cargo test test_something"));
        assert!(is_cargo_dedup_command("timeout 5 cargo run -- examples/demo.mini"));
        assert!(is_cargo_dedup_command("cd ~/dev && timeout 10 cargo test 2>&1"));

        // Non-cargo commands
        assert!(!is_cargo_dedup_command("git status"));
        assert!(!is_cargo_dedup_command("ls -la"));
        assert!(!is_cargo_dedup_command("cd ~/dev && ls"));
    }

    #[test]
    fn test_dedup_plain_cargo_check() {
        use crate::llm::{ChatMessage, ToolCall};

        let mut messages = vec![
            // Plain cargo check (no cd prefix)
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_call_id: None,
                tool_calls: Some(vec![ToolCall {
                    id: "tc1".into(),
                    name: "shell".into(),
                    arguments: json!({"command": "cargo check"}),
                }]),
            },
            ChatMessage {
                role: "user".into(),
                content: Some("error[E0308]: mismatched types".into()),
                tool_call_id: Some("tc1".into()),
                tool_calls: None,
            },
            // Second plain cargo check
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_call_id: None,
                tool_calls: Some(vec![ToolCall {
                    id: "tc2".into(),
                    name: "shell".into(),
                    arguments: json!({"command": "cargo check"}),
                }]),
            },
            ChatMessage {
                role: "user".into(),
                content: Some("Finished `dev` profile".into()),
                tool_call_id: Some("tc2".into()),
                tool_calls: None,
            },
        ];

        dedup_tool_results(&mut messages);

        // First pair removed, 2 messages remain
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].content.as_ref().unwrap(), "Finished `dev` profile");
    }

    #[test]
    fn test_dedup_cargo_test_also_deduped() {
        use crate::llm::{ChatMessage, ToolCall};

        let mut messages = vec![
            // cargo test
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_call_id: None,
                tool_calls: Some(vec![ToolCall {
                    id: "tc1".into(),
                    name: "shell".into(),
                    arguments: json!({"command": "cd ~/dev/proj && cargo test 2>&1"}),
                }]),
            },
            ChatMessage {
                role: "user".into(),
                content: Some("test result: 5 passed".into()),
                tool_call_id: Some("tc1".into()),
                tool_calls: None,
            },
            // Later cargo check
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_call_id: None,
                tool_calls: Some(vec![ToolCall {
                    id: "tc2".into(),
                    name: "shell".into(),
                    arguments: json!({"command": "cd ~/dev/proj && cargo check 2>&1"}),
                }]),
            },
            ChatMessage {
                role: "user".into(),
                content: Some("Finished `dev` profile".into()),
                tool_call_id: Some("tc2".into()),
                tool_calls: None,
            },
        ];

        dedup_tool_results(&mut messages);

        // cargo test pair removed, 2 messages remain
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].content.as_ref().unwrap(), "Finished `dev` profile");
    }

}
