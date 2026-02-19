use crate::error::ToolError;
use crate::tool::Tool;
use async_trait::async_trait;
use serde_json::Value;
use std::process::Stdio;
use std::time::Duration;

/// Spawn a shell command in its own process group and kill the entire group on timeout.
///
/// Uses `setpgid(0, 0)` so the child (and any grandchildren) form an isolated process group.
/// On timeout, `kill(-pgid, SIGKILL)` reaps the whole tree, then `wait()` cleans up the zombie.
/// `kill_on_drop(true)` acts as a safety net if the `Child` handle is dropped unexpectedly.
pub(crate) async fn run_shell_with_kill(
    command: &str,
    timeout: Duration,
) -> Result<Value, ToolError> {
    use std::os::unix::process::CommandExt as _;

    let mut child = {
        let mut cmd = std::process::Command::new("sh");
        cmd.arg("-c").arg(command);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        // SAFETY: setpgid is async-signal-safe and called before exec.
        unsafe {
            cmd.pre_exec(|| {
                libc::setpgid(0, 0);
                Ok(())
            });
        }

        tokio::process::Command::from(cmd)
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| {
                ToolError::ExecutionFailed(format!("Failed to spawn command: {}", e))
            })?
    };

    let pid = child.id();

    // Take stdout/stderr handles so we can read them without consuming `child`.
    let mut stdout_handle = child.stdout.take();
    let mut stderr_handle = child.stderr.take();

    let read_output = async {
        let status = child.wait().await?;

        let mut stdout_buf = Vec::new();
        if let Some(ref mut h) = stdout_handle {
            tokio::io::AsyncReadExt::read_to_end(h, &mut stdout_buf).await?;
        }
        let mut stderr_buf = Vec::new();
        if let Some(ref mut h) = stderr_handle {
            tokio::io::AsyncReadExt::read_to_end(h, &mut stderr_buf).await?;
        }

        Ok::<_, std::io::Error>((status, stdout_buf, stderr_buf))
    };

    match tokio::time::timeout(timeout, read_output).await {
        Ok(Ok((status, stdout_buf, stderr_buf))) => {
            let stdout = String::from_utf8_lossy(&stdout_buf).to_string();
            let stderr = String::from_utf8_lossy(&stderr_buf).to_string();
            let exit_code = status.code().unwrap_or(-1);
            Ok(serde_json::json!({
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code
            }))
        }
        Ok(Err(e)) => Err(ToolError::ExecutionFailed(format!(
            "Failed to execute command: {}",
            e
        ))),
        Err(_) => {
            // Timeout — kill the entire process group
            if let Some(pid) = pid {
                // SAFETY: Sending SIGKILL to the process group we created.
                unsafe {
                    libc::kill(-(pid as i32), libc::SIGKILL);
                }
            }
            // kill_on_drop will handle cleanup when `child` is dropped here
            Err(ToolError::ExecutionFailed(format!(
                "Command timed out after {:?} (process killed)",
                timeout
            )))
        }
    }
}

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
            .ok_or_else(|| {
                ToolError::InvalidInput("Missing or invalid 'command' field".to_string())
            })?;

        run_shell_with_kill(command, self.timeout).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_shell_tool_name() {
        let tool = ShellTool::new();
        assert_eq!(tool.name(), "shell");
    }

    #[test]
    fn test_shell_tool_description() {
        let tool = ShellTool::new();
        assert!(tool.description().contains("shell"));
    }

    #[test]
    fn test_shell_tool_schema() {
        let tool = ShellTool::new();
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
    fn test_shell_tool_default() {
        let tool = ShellTool::default();
        assert_eq!(tool.timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_shell_tool_with_timeout() {
        let tool = ShellTool::with_timeout(Duration::from_secs(60));
        assert_eq!(tool.timeout, Duration::from_secs(60));
    }

    #[tokio::test]
    async fn test_shell_simple_command() {
        let tool = ShellTool::new();
        let result = tool
            .execute(json!({"command": "echo hello"}))
            .await
            .unwrap();
        assert_eq!(result["stdout"].as_str().unwrap().trim(), "hello");
        assert_eq!(result["exit_code"], 0);
    }

    #[tokio::test]
    async fn test_shell_exit_code() {
        let tool = ShellTool::new();
        let result = tool.execute(json!({"command": "exit 42"})).await.unwrap();
        assert_eq!(result["exit_code"], 42);
    }

    #[tokio::test]
    async fn test_shell_stderr() {
        let tool = ShellTool::new();
        let result = tool
            .execute(json!({"command": "echo error >&2"}))
            .await
            .unwrap();
        assert!(result["stderr"].as_str().unwrap().contains("error"));
    }

    #[tokio::test]
    async fn test_shell_missing_command_field() {
        let tool = ShellTool::new();
        let result = tool.execute(json!({})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_shell_timeout() {
        let tool = ShellTool::with_timeout(Duration::from_millis(100));
        let result = tool.execute(json!({"command": "sleep 10"})).await;
        assert!(matches!(result, Err(ToolError::ExecutionFailed(_))));
        if let Err(ToolError::ExecutionFailed(msg)) = result {
            assert!(msg.contains("timed out"));
        }
    }

    #[tokio::test]
    async fn test_shell_command_with_args() {
        let tool = ShellTool::new();
        let result = tool
            .execute(json!({"command": "printf '%s' test"}))
            .await
            .unwrap();
        assert_eq!(result["stdout"].as_str().unwrap(), "test");
    }

    #[tokio::test]
    async fn test_shell_piped_command() {
        let tool = ShellTool::new();
        let result = tool
            .execute(json!({"command": "echo hello world | tr a-z A-Z"}))
            .await
            .unwrap();
        assert_eq!(result["stdout"].as_str().unwrap().trim(), "HELLO WORLD");
    }
}
