use crate::error::ToolError;
use crate::tool::Tool;
use crate::tools::shell::DEFAULT_MEM_LIMIT_BYTES;
use async_trait::async_trait;
use serde_json::Value;
use std::process::Stdio;
use std::time::Duration;

/// Tool for executing Python code.
///
/// Spawns `python3 -` and passes code via stdin, avoiding shell quoting issues.
#[derive(Debug)]
pub struct PythonTool {
    timeout: Duration,
    mem_limit_bytes: Option<u64>,
}

impl PythonTool {
    pub fn new() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            mem_limit_bytes: Some(DEFAULT_MEM_LIMIT_BYTES),
        }
    }

    pub fn with_mem_limit(mut self, limit: Option<u64>) -> Self {
        self.mem_limit_bytes = limit;
        self
    }
}

impl Default for PythonTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for PythonTool {
    fn name(&self) -> &str {
        "python"
    }

    fn description(&self) -> &str {
        "Execute Python code and return stdout, stderr, and exit code"
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let code = input
            .get("code")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ToolError::InvalidInput("Missing or invalid 'code' field".to_string())
            })?;

        run_python(code, self.timeout, self.mem_limit_bytes, None).await
    }
}

/// Spawn `python3 -` with code on stdin, using process group isolation and timeout.
///
/// If `agent_dir` is provided and contains a `python/` subdirectory, it is added
/// to the child process's `PYTHONPATH` so the agent can `import` modules from there.
pub async fn run_python(
    code: &str,
    timeout: Duration,
    mem_limit_bytes: Option<u64>,
    agent_dir: Option<&std::path::Path>,
) -> Result<Value, ToolError> {
    use std::os::unix::process::CommandExt as _;

    let mut child = {
        let mut cmd = std::process::Command::new("python3");
        cmd.arg("-");
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        // Auto-inject PYTHONPATH if agent has a python/ directory
        if let Some(dir) = agent_dir {
            let python_dir = dir.join("python");
            if python_dir.is_dir() {
                let new_path = if let Ok(existing) = std::env::var("PYTHONPATH") {
                    format!("{}:{}", python_dir.display(), existing)
                } else {
                    python_dir.display().to_string()
                };
                cmd.env("PYTHONPATH", &new_path);
            }
        }
        // SAFETY: setpgid and setrlimit are async-signal-safe and called before exec.
        unsafe {
            cmd.pre_exec(move || {
                libc::setpgid(0, 0);
                if let Some(limit) = mem_limit_bytes {
                    let rlim = libc::rlimit {
                        rlim_cur: limit as libc::rlim_t,
                        rlim_max: limit as libc::rlim_t,
                    };
                    if libc::setrlimit(libc::RLIMIT_AS, &rlim) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                }
                Ok(())
            });
        }

        tokio::process::Command::from(cmd)
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| {
                ToolError::ExecutionFailed(format!("Failed to spawn python3: {}", e))
            })?
    };

    // Write code to stdin, then close it so python reads EOF and executes.
    if let Some(mut stdin) = child.stdin.take() {
        use tokio::io::AsyncWriteExt;
        let _ = stdin.write_all(code.as_bytes()).await;
        drop(stdin);
    }

    let pid = child.id();
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
            "Failed to execute python3: {}",
            e
        ))),
        Err(_) => {
            if let Some(pid) = pid {
                // SAFETY: Sending SIGKILL to the process group we created.
                unsafe {
                    libc::kill(-(pid as i32), libc::SIGKILL);
                }
            }
            Err(ToolError::ExecutionFailed(format!(
                "Python timed out after {:?} (process killed)",
                timeout
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_python_tool_name() {
        let tool = PythonTool::new();
        assert_eq!(tool.name(), "python");
    }

    #[test]
    fn test_python_tool_schema() {
        let tool = PythonTool::new();
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["code"].is_object());
        assert!(schema["required"].as_array().unwrap().contains(&json!("code")));
    }

    #[tokio::test]
    async fn test_python_basic_execution() {
        let tool = PythonTool::new();
        let result = tool
            .execute(json!({"code": "print('hello')"}))
            .await
            .unwrap();
        assert_eq!(result["stdout"].as_str().unwrap().trim(), "hello");
        assert_eq!(result["exit_code"], 0);
    }

    #[tokio::test]
    async fn test_python_missing_code() {
        let tool = PythonTool::new();
        let result = tool.execute(json!({})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_python_stderr() {
        let tool = PythonTool::new();
        let result = tool
            .execute(json!({"code": "import sys; print('err', file=sys.stderr)"}))
            .await
            .unwrap();
        assert!(result["stderr"].as_str().unwrap().contains("err"));
    }

    #[tokio::test]
    async fn test_python_exit_code() {
        let tool = PythonTool::new();
        let result = tool
            .execute(json!({"code": "import sys; sys.exit(42)"}))
            .await
            .unwrap();
        assert_eq!(result["exit_code"], 42);
    }

    #[tokio::test]
    async fn test_python_multiline() {
        let tool = PythonTool::new();
        let result = tool
            .execute(json!({"code": "x = 2\ny = 3\nprint(x + y)"}))
            .await
            .unwrap();
        assert_eq!(result["stdout"].as_str().unwrap().trim(), "5");
    }

    #[tokio::test]
    async fn test_python_quoting() {
        let tool = PythonTool::new();
        let result = tool
            .execute(json!({"code": "print(\"it's a \\\"test\\\"\")" }))
            .await
            .unwrap();
        assert!(result["stdout"].as_str().unwrap().contains("it's a \"test\""));
    }
}
