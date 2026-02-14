use serde_json::Value;

pub mod add;
pub mod claude_code;
pub mod echo;
pub mod edit_file;
pub mod http;
pub mod list_agents;
pub mod list_files;
pub mod peek_file;
pub mod read_file;
pub mod remember;
pub mod safe_shell;
pub mod search_conversation;
pub mod send_message;
pub mod shell;
pub mod task;
pub mod write_file;

/// Parse a JSON value as bool, with string fallback for Ollama models that send bools as strings.
pub fn json_to_bool(v: &Value) -> Option<bool> {
    v.as_bool().or_else(|| match v.as_str()? {
        "true" | "True" | "1" => Some(true),
        "false" | "False" | "0" => Some(false),
        _ => None,
    })
}

pub use add::AddTool;
pub use claude_code::{
    ClaudeCodeTask, ClaudeCodeTool, TaskError, TaskStatus, TaskStore, is_process_running,
};
pub use echo::EchoTool;
pub use edit_file::EditFileTool;
pub use http::HttpTool;
pub use list_agents::DaemonListAgentsTool;
pub use list_files::ListFilesTool;
pub use peek_file::PeekFileTool;
pub use read_file::ReadFileTool;
pub use remember::{DaemonRememberTool, RememberTool};
pub use safe_shell::SafeShellTool;
pub use search_conversation::DaemonSearchConversationTool;
pub use send_message::{DaemonSendMessageTool, ListAgentsTool, SendMessageTool};
pub use shell::ShellTool;
pub use task::{DaemonTaskTool, parse_task_origin};
pub use write_file::WriteFileTool;
