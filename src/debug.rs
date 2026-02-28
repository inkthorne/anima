//! Debug logging for Anima library diagnostics.
//!
//! Enable with `anima --log` to write to ~/.anima/anima.log

use std::path::Path;

use chrono::Local;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::llm::{ChatMessage, LLMResponse, ToolSpec};

/// Global flag for debug logging (enabled via --log flag)
static LOGGING_ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable debug logging (truncates existing log file)
pub fn enable() {
    LOGGING_ENABLED.store(true, Ordering::SeqCst);

    // Truncate the log file on new session
    if let Some(home) = dirs::home_dir() {
        let log_path = home.join(".anima").join("anima.log");
        if let Some(parent) = log_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&log_path, ""); // Truncate
    }

    log("=== Anima debug logging enabled ===");
}

/// Check if logging is enabled
pub fn is_enabled() -> bool {
    LOGGING_ENABLED.load(Ordering::SeqCst)
}

/// Log a message to ~/.anima/anima.log (only if logging is enabled)
pub fn log(msg: &str) {
    if !is_enabled() {
        return;
    }

    let log_path = match dirs::home_dir() {
        Some(home) => home.join(".anima").join("anima.log"),
        None => return,
    };

    // Ensure directory exists
    if let Some(parent) = log_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S%.3f");
    let line = format!("[{}] {}\n", timestamp, msg);

    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&log_path) {
        let _ = file.write_all(line.as_bytes());
    }
}

/// Log a formatted message
#[macro_export]
macro_rules! debug_log {
    ($($arg:tt)*) => {
        $crate::debug::log(&format!($($arg)*))
    };
}

/// Log JSON payload (pretty-printed, truncated if too long)
pub fn log_json(label: &str, json: &serde_json::Value) {
    if !is_enabled() {
        return;
    }

    let pretty = serde_json::to_string_pretty(json).unwrap_or_else(|_| json.to_string());
    let truncated = if pretty.len() > 5000 {
        format!(
            "{}...\n[truncated, {} bytes total]",
            &pretty[..5000],
            pretty.len()
        )
    } else {
        pretty
    };

    log(&format!("{}:\n{}", label, truncated));
}

/// Log request/response pair for LLM calls
pub fn log_llm_request(provider: &str, model: &str, messages: &[crate::llm::ChatMessage]) {
    if !is_enabled() {
        return;
    }

    log(&format!("=== LLM REQUEST: {} / {} ===", provider, model));
    log(&format!("Messages ({}):", messages.len()));

    for (i, msg) in messages.iter().enumerate() {
        let content = msg.content.as_deref().unwrap_or("<none>");
        let preview = if content.len() > 500 {
            format!(
                "{}... [truncated, {} chars]",
                &content[..500],
                content.len()
            )
        } else {
            content.to_string()
        };
        log(&format!("  [{}] {}: {}", i, msg.role, preview));
    }
}

pub fn log_llm_response(provider: &str, content: &str, thinking: Option<&str>) {
    if !is_enabled() {
        return;
    }

    log(&format!("=== LLM RESPONSE: {} ===", provider));

    if let Some(think) = thinking {
        let preview = if think.len() > 1000 {
            format!("{}... [truncated, {} chars]", &think[..1000], think.len())
        } else {
            think.to_string()
        };
        log(&format!("Thinking: {}", preview));
    }

    let preview = if content.len() > 1000 {
        format!(
            "{}... [truncated, {} chars]",
            &content[..1000],
            content.len()
        )
    } else {
        content.to_string()
    };
    log(&format!("Content: {}", preview));
}

// ---------------------------------------------------------------------------
// Turn dump functions — shared by Agent and Pipeline
// ---------------------------------------------------------------------------

/// Dump an LLM request payload to turns/{conv_name}/req-{n}.json.
/// Returns the turn number for pairing with dump_response.
pub fn dump_request(
    agent_dir: &Path,
    conv_name: &str,
    model: &str,
    tools: &Option<Vec<ToolSpec>>,
    messages: &[ChatMessage],
) -> Option<u64> {
    // Create turns/{conv_name}/ directory if it doesn't exist
    let conv_dir = agent_dir.join("turns").join(conv_name);
    if let Err(e) = std::fs::create_dir_all(&conv_dir) {
        eprintln!("Failed to create turns directory: {}", e);
        return None;
    }

    // Create .gitignore in turns/ to make it self-ignoring (debug files shouldn't be committed)
    let gitignore_path = agent_dir.join("turns").join(".gitignore");
    if !gitignore_path.exists() {
        let _ = std::fs::write(&gitignore_path, "*\n!.gitignore\n");
    }

    // Find next sequential number: scan for req-*.json, parse numeric part, take max + 1
    let next_n = std::fs::read_dir(&conv_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let stem = e.path().file_stem()?.to_str()?.to_string();
            stem.strip_prefix("req-")?.parse::<u64>().ok()
        })
        .max()
        .map_or(1, |m| m + 1);

    let file_path = conv_dir.join(format!("req-{}.json", next_n));

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
                                "arguments": tc.arguments
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
        "stream": true
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

    // Write to file (best-effort, don't fail on IO errors)
    if let Err(e) = std::fs::write(&file_path, content) {
        eprintln!("Failed to write request dump: {}", e);
        return None;
    }

    Some(next_n)
}

/// Dump an LLM response to turns/{conv_name}/resp-{n}.json.
pub fn dump_response(
    agent_dir: &Path,
    conv_name: &str,
    turn_n: Option<u64>,
    response: &LLMResponse,
) {
    let turn_n = match turn_n {
        Some(n) => n,
        None => return,
    };
    let conv_dir = agent_dir.join("turns").join(conv_name);
    let file_path = conv_dir.join(format!("resp-{}.json", turn_n));

    let content =
        serde_json::to_string_pretty(response).unwrap_or_else(|_| "{}".to_string());
    if let Err(e) = std::fs::write(&file_path, content) {
        eprintln!("Failed to write response dump: {}", e);
    }

    // Write raw HTTP response body for debugging parsing issues
    if let Some(ref raw) = response.raw_body {
        let raw_path = conv_dir.join(format!("raw-{}.json", turn_n));
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(raw) {
            let pretty =
                serde_json::to_string_pretty(&parsed).unwrap_or_else(|_| raw.clone());
            let _ = std::fs::write(&raw_path, pretty);
        } else {
            let _ = std::fs::write(&raw_path, raw);
        }
    }

    // Write raw SSE stream capture for debugging streaming issues
    if let Some(ref stream) = response.raw_stream {
        let stream_path = conv_dir.join(format!("stream-{}.json", turn_n));
        let _ = std::fs::write(&stream_path, stream);
    }
}

/// Dump raw stream capture to turns/{conv_name}/stream-{n}.json on error.
pub fn dump_stream(
    agent_dir: &Path,
    conv_name: &str,
    turn_n: Option<u64>,
    raw_stream: &str,
) {
    let turn_n = match turn_n {
        Some(n) => n,
        None => return,
    };
    let conv_dir = agent_dir.join("turns").join(conv_name);
    let stream_path = conv_dir.join(format!("stream-{}.json", turn_n));
    let _ = std::fs::write(&stream_path, raw_stream);
}

/// Rename debug dump files from sequential number to DB message ID.
pub fn rename_turn_files(
    agent_dir: &Path,
    conv_name: &str,
    from_n: u64,
    to_id: i64,
) {
    let conv_dir = agent_dir.join("turns").join(conv_name);

    for prefix in &["req", "resp", "raw", "stream"] {
        let from_path = conv_dir.join(format!("{}-{}.json", prefix, from_n));
        let to_path = conv_dir.join(format!("{}-{}.json", prefix, to_id));
        if from_path.exists() {
            if let Err(e) = std::fs::rename(&from_path, &to_path) {
                eprintln!(
                    "Failed to rename {} -> {}: {}",
                    from_path.display(),
                    to_path.display(),
                    e
                );
            }
        }
    }
}
