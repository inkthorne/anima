use super::shell::run_shell_with_kill;
use crate::error::ToolError;
use crate::tool::Tool;
use async_trait::async_trait;
use serde_json::Value;
use std::time::Duration;

/// Default list of safe commands that can be executed.
const DEFAULT_SAFE_COMMANDS: &[&str] = &[
    "ls", "grep", "find", "cat", "head", "tail", "wc", "pwd", "echo", "which", "file", "stat",
    "du", "df", "env", "date", "whoami", "hostname", "uname",
];

/// Tool for executing shell commands with command allowlist validation.
///
/// Only commands in the allowlist can be executed. For pipelines, ALL
/// commands must be in the allowlist.
#[derive(Debug)]
pub struct SafeShellTool {
    timeout: Duration,
    allowed_commands: Vec<String>,
    mem_limit_bytes: Option<u64>,
}

impl SafeShellTool {
    pub fn new() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            allowed_commands: DEFAULT_SAFE_COMMANDS
                .iter()
                .map(|s| s.to_string())
                .collect(),
            mem_limit_bytes: Some(super::shell::DEFAULT_MEM_LIMIT_BYTES),
        }
    }

    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            timeout,
            allowed_commands: DEFAULT_SAFE_COMMANDS
                .iter()
                .map(|s| s.to_string())
                .collect(),
            mem_limit_bytes: Some(super::shell::DEFAULT_MEM_LIMIT_BYTES),
        }
    }

    pub fn with_allowed_commands(allowed: Vec<String>) -> Self {
        Self {
            timeout: Duration::from_secs(30),
            allowed_commands: allowed,
            mem_limit_bytes: Some(super::shell::DEFAULT_MEM_LIMIT_BYTES),
        }
    }

    pub fn with_mem_limit(mut self, limit: Option<u64>) -> Self {
        self.mem_limit_bytes = limit;
        self
    }

    /// Extract all base commands from a shell command string.
    /// Handles pipes, semicolons, &&, ||, and command substitution.
    fn extract_commands(command: &str) -> Vec<String> {
        let mut commands = Vec::new();

        // Split on common shell operators
        // This is a simplified parser - it won't handle all edge cases
        // but covers the common usage patterns
        for segment in command.split(['|', ';']) {
            // Also handle && and || by splitting on those
            for part in segment.split("&&") {
                for subpart in part.split("||") {
                    let trimmed = subpart.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    // Get the first word (the command)
                    if let Some(first_word) = trimmed.split_whitespace().next() {
                        // Handle command substitution $(...) by ignoring the $( prefix
                        let cmd = first_word
                            .trim_start_matches("$(")
                            .trim_start_matches('`')
                            .trim_end_matches(')')
                            .trim_end_matches('`');

                        if !cmd.is_empty() {
                            commands.push(cmd.to_string());
                        }
                    }
                }
            }
        }

        commands
    }

    /// Check if all commands in the input are allowed.
    fn validate_command(&self, command: &str) -> Result<(), String> {
        let commands = Self::extract_commands(command);

        if commands.is_empty() {
            return Err("No command found".to_string());
        }

        for cmd in &commands {
            if !self.allowed_commands.iter().any(|a| a == cmd) {
                return Err(format!(
                    "Command '{}' not in allowed list. Allowed: {:?}",
                    cmd, self.allowed_commands
                ));
            }
        }

        Ok(())
    }
}

impl Default for SafeShellTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for SafeShellTool {
    fn name(&self) -> &str {
        "safe_shell"
    }

    fn description(&self) -> &str {
        "Executes a shell command from a restricted set of safe commands. \
         Only read-only commands like ls, grep, find, cat, etc. are allowed."
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute (must use only allowed commands)"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let command = input
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ToolError::InvalidInput("Missing or invalid 'command' field".to_string())
            })?;

        // Validate all commands in the input
        self.validate_command(command)
            .map_err(ToolError::InvalidInput)?;

        run_shell_with_kill(command, self.timeout, self.mem_limit_bytes).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_safe_shell_tool_name() {
        let tool = SafeShellTool::new();
        assert_eq!(tool.name(), "safe_shell");
    }

    #[test]
    fn test_safe_shell_tool_description() {
        let tool = SafeShellTool::new();
        assert!(tool.description().contains("safe"));
    }

    #[test]
    fn test_safe_shell_tool_schema() {
        let tool = SafeShellTool::new();
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["command"].is_object());
        assert!(
            schema["required"]
                .as_array()
                .unwrap()
                .contains(&json!("command"))
        );
    }

    #[test]
    fn test_safe_shell_tool_default() {
        let tool = SafeShellTool::default();
        assert_eq!(tool.timeout, Duration::from_secs(30));
        assert!(tool.allowed_commands.contains(&"ls".to_string()));
    }

    #[test]
    fn test_extract_simple_command() {
        let commands = SafeShellTool::extract_commands("ls -la");
        assert_eq!(commands, vec!["ls"]);
    }

    #[test]
    fn test_extract_piped_commands() {
        let commands = SafeShellTool::extract_commands("ls -la | grep foo | wc -l");
        assert_eq!(commands, vec!["ls", "grep", "wc"]);
    }

    #[test]
    fn test_extract_chained_commands() {
        let commands = SafeShellTool::extract_commands("pwd && ls && echo done");
        assert_eq!(commands, vec!["pwd", "ls", "echo"]);
    }

    #[test]
    fn test_extract_semicolon_commands() {
        let commands = SafeShellTool::extract_commands("pwd; ls; echo done");
        assert_eq!(commands, vec!["pwd", "ls", "echo"]);
    }

    #[test]
    fn test_validate_allowed_command() {
        let tool = SafeShellTool::new();
        assert!(tool.validate_command("ls -la").is_ok());
        assert!(tool.validate_command("grep pattern file.txt").is_ok());
    }

    #[test]
    fn test_validate_disallowed_command() {
        let tool = SafeShellTool::new();
        assert!(tool.validate_command("rm -rf /").is_err());
        assert!(tool.validate_command("curl http://example.com").is_err());
    }

    #[test]
    fn test_validate_piped_all_allowed() {
        let tool = SafeShellTool::new();
        assert!(tool.validate_command("ls | grep foo | wc -l").is_ok());
    }

    #[test]
    fn test_validate_piped_one_disallowed() {
        let tool = SafeShellTool::new();
        // rm is not in the allowed list
        assert!(tool.validate_command("ls | rm").is_err());
    }

    #[test]
    fn test_custom_allowed_commands() {
        let tool = SafeShellTool::with_allowed_commands(vec!["custom".to_string()]);
        assert!(tool.validate_command("custom arg").is_ok());
        assert!(tool.validate_command("ls").is_err());
    }

    #[tokio::test]
    async fn test_safe_shell_simple_command() {
        let tool = SafeShellTool::new();
        let result = tool
            .execute(json!({"command": "echo hello"}))
            .await
            .unwrap();
        assert_eq!(result["stdout"].as_str().unwrap().trim(), "hello");
        assert_eq!(result["exit_code"], 0);
    }

    #[tokio::test]
    async fn test_safe_shell_piped_command() {
        let tool = SafeShellTool::new();
        let result = tool
            .execute(json!({"command": "echo hello | cat"}))
            .await
            .unwrap();
        assert_eq!(result["stdout"].as_str().unwrap().trim(), "hello");
    }

    #[tokio::test]
    async fn test_safe_shell_blocked_command() {
        let tool = SafeShellTool::new();
        let result = tool.execute(json!({"command": "rm -rf /"})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
        if let Err(ToolError::InvalidInput(msg)) = result {
            assert!(msg.contains("rm"));
            assert!(msg.contains("not in allowed list"));
        }
    }

    #[tokio::test]
    async fn test_safe_shell_missing_command_field() {
        let tool = SafeShellTool::new();
        let result = tool.execute(json!({})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_safe_shell_ls_command() {
        let tool = SafeShellTool::new();
        let result = tool.execute(json!({"command": "ls /tmp"})).await.unwrap();
        assert_eq!(result["exit_code"], 0);
    }

    #[tokio::test]
    async fn test_safe_shell_pwd_command() {
        let tool = SafeShellTool::new();
        let result = tool.execute(json!({"command": "pwd"})).await.unwrap();
        assert_eq!(result["exit_code"], 0);
        assert!(!result["stdout"].as_str().unwrap().is_empty());
    }
}
