use async_trait::async_trait;
use crate::error::ToolError;
use crate::tool::Tool;
use serde_json::Value;
use std::time::Duration;
use tokio::process::Command;

/// Tool for executing shell commands.
///
/// # Security Warning
///
/// This tool executes arbitrary shell commands. Use with extreme caution:
/// - Never pass untrusted user input directly to commands
/// - Consider sandboxing or restricting available commands
/// - Audit all command invocations in production
#[derive(Debug)]
pub struct ShellTool {
    timeout: Duration,
}

impl ShellTool {
    pub fn new() -> Self {
        Self {
            timeout: Duration::from_secs(30),
        }
    }

    pub fn with_timeout(timeout: Duration) -> Self {
        Self { timeout }
    }
}

impl Default for ShellTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        "Executes a shell command and returns the output. Use with caution."
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let command = input
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing or invalid 'command' field".to_string()))?;

        let output = tokio::time::timeout(
            self.timeout,
            Command::new("sh")
                .arg("-c")
                .arg(command)
                .output()
        )
        .await
        .map_err(|_| ToolError::ExecutionFailed(format!("Command timed out after {:?}", self.timeout)))?
        .map_err(|e| ToolError::ExecutionFailed(format!("Failed to execute command: {}", e)))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1);

        Ok(serde_json::json!({
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code
        }))
    }
}
