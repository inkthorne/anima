//! Debug logging for Anima library diagnostics.
//! 
//! Enable with `anima --log` to write to ~/.anima/anima.log

use std::fs::OpenOptions;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use chrono::Local;

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
    
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
    {
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
        format!("{}...\n[truncated, {} bytes total]", &pretty[..5000], pretty.len())
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
            format!("{}... [truncated, {} chars]", &content[..500], content.len())
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
        format!("{}... [truncated, {} chars]", &content[..1000], content.len())
    } else {
        content.to_string()
    };
    log(&format!("Content: {}", preview));
}
