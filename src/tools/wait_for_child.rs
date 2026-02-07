use crate::error::ToolError;
use crate::tool::Tool;
use async_trait::async_trait;
use serde_json::Value;

#[derive(Debug)]
pub struct WaitForChildTool;

#[async_trait]
impl Tool for WaitForChildTool {
    fn name(&self) -> &str {
        "wait_for_child"
    }

    fn description(&self) -> &str {
        "Wait for a child agent to complete and get its result. Default timeout is 300s — do not reduce this unless you have a specific reason. Agent responses typically take 30-120s."
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "child_id": {
                    "type": "string",
                    "description": "The child_id returned by spawn_child"
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Max seconds to wait (default: 300, minimum: 30). Agent responses typically take 30-120s."
                }
            },
            "required": ["child_id"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let child_id = input
            .get("child_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("child_id is required".to_string()))?;
        Ok(serde_json::json!({"status": "wait_requested", "child_id": child_id}))
    }
}

/// Daemon-aware tool that waits for a child agent to complete a delegated task.
/// Polls the task conversation for a response from the child agent.
pub struct DaemonWaitForChildTool {
    agent_name: String,
}

impl DaemonWaitForChildTool {
    pub fn new(agent_name: String) -> Self {
        DaemonWaitForChildTool { agent_name }
    }
}

impl std::fmt::Debug for DaemonWaitForChildTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DaemonWaitForChildTool")
            .field("agent_name", &self.agent_name)
            .finish()
    }
}

#[async_trait]
impl Tool for DaemonWaitForChildTool {
    fn name(&self) -> &str {
        "wait_for_child"
    }

    fn description(&self) -> &str {
        "Wait for a spawned child agent to complete its task and return the result. Use the child_id from spawn_child. Default timeout is 300s — do not reduce this unless you have a specific reason. Agent responses typically take 30-120s."
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "child_id": {
                    "type": "string",
                    "description": "The child_id returned by spawn_child"
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Max seconds to wait (default: 300, minimum: 30). Agent responses typically take 30-120s."
                }
            },
            "required": ["child_id"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        use crate::conversation::ConversationStore;

        let child_id = input
            .get("child_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("child_id is required".to_string()))?;

        let timeout_secs = input
            .get("timeout_secs")
            .and_then(|v| v.as_u64())
            .unwrap_or(300)
            .max(30);

        let poll_interval = std::time::Duration::from_secs(1);
        let max_polls = timeout_secs;

        for _ in 0..max_polls {
            // Open a fresh connection each poll to see latest data
            let store = ConversationStore::init().map_err(|e| {
                ToolError::ExecutionFailed(format!("Failed to open conversation store: {}", e))
            })?;

            let messages = store.get_messages(child_id, Some(50)).map_err(|e| {
                ToolError::ExecutionFailed(format!("Failed to read task conversation: {}", e))
            })?;

            // Look for a completed response: must be from the child agent (not ourselves,
            // not a tool result, not recall injection) AND must have duration_ms set,
            // which indicates a final LLM response rather than an intermediate message.
            for msg in &messages {
                if msg.from_agent != self.agent_name
                    && msg.from_agent != "tool"
                    && msg.from_agent != "recall"
                    && msg.duration_ms.is_some()
                {
                    return Ok(serde_json::json!({
                        "status": "completed",
                        "result": msg.content,
                        "child_id": child_id,
                        "from_agent": msg.from_agent
                    }));
                }
            }

            tokio::time::sleep(poll_interval).await;
        }

        Ok(serde_json::json!({
            "status": "timeout",
            "message": format!("Child task '{}' has not completed yet after {}s. The child agent is still working. Call wait_for_child again to keep waiting.", child_id, timeout_secs),
            "child_id": child_id
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_wait_for_child_tool_name() {
        let tool = WaitForChildTool;
        assert_eq!(tool.name(), "wait_for_child");
    }

    #[tokio::test]
    async fn test_wait_for_child_schema() {
        let tool = WaitForChildTool;
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["child_id"].is_object());
        assert!(schema["properties"]["timeout_secs"].is_object());
    }

    #[tokio::test]
    async fn test_wait_for_child_missing_child_id() {
        let tool = WaitForChildTool;
        let result = tool.execute(json!({})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_daemon_wait_for_child_tool_name() {
        let tool = DaemonWaitForChildTool::new("parent".to_string());
        assert_eq!(tool.name(), "wait_for_child");
    }

    #[tokio::test]
    async fn test_daemon_wait_for_child_schema() {
        let tool = DaemonWaitForChildTool::new("parent".to_string());
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["child_id"].is_object());
        assert!(schema["properties"]["timeout_secs"].is_object());
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("child_id")));
    }

    #[tokio::test]
    async fn test_daemon_wait_for_child_missing_child_id() {
        let tool = DaemonWaitForChildTool::new("parent".to_string());
        let result = tool.execute(json!({})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }
}
