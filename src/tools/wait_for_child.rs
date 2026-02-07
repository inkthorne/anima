use crate::error::ToolError;
use crate::tool::Tool;
use async_trait::async_trait;
use serde_json::Value;

/// Shared polling logic: polls a task conversation for a child agent's completed response.
/// Returns a JSON value with status "completed" (with result/from_agent) or "timeout".
async fn poll_for_child_result(
    parent_agent: &str,
    child_id: &str,
    timeout_secs: u64,
) -> Result<Value, ToolError> {
    use crate::conversation::ConversationStore;

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
            if msg.from_agent != parent_agent
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
        "message": format!("Child task '{}' has not completed yet after {}s. The child agent is still working. Call wait_for_children again to keep waiting.", child_id, timeout_secs),
        "child_id": child_id
    }))
}

#[derive(Debug)]
pub struct WaitForChildrenTool;

#[async_trait]
impl Tool for WaitForChildrenTool {
    fn name(&self) -> &str {
        "wait_for_children"
    }

    fn description(&self) -> &str {
        "Wait for multiple child agents to complete and get all results at once. Default timeout is 300s."
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "child_ids": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Array of child_ids returned by spawn_child"
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Max seconds to wait per child (default: 300, minimum: 30). Agent responses typically take 30-120s."
                }
            },
            "required": ["child_ids"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let child_ids = input
            .get("child_ids")
            .and_then(|v| v.as_array())
            .ok_or_else(|| ToolError::InvalidInput("child_ids is required and must be an array".to_string()))?;

        if child_ids.is_empty() {
            return Err(ToolError::InvalidInput("child_ids must not be empty".to_string()));
        }

        // Validate all are strings
        for id in child_ids {
            if id.as_str().is_none() {
                return Err(ToolError::InvalidInput("All child_ids must be strings".to_string()));
            }
        }

        let ids: Vec<&str> = child_ids.iter().filter_map(|v| v.as_str()).collect();
        Ok(serde_json::json!({"status": "wait_requested", "child_ids": ids}))
    }
}

/// Daemon-aware tool that waits for multiple child agents to complete in parallel.
/// Polls all task conversations concurrently and returns all results at once.
pub struct DaemonWaitForChildrenTool {
    agent_name: String,
}

impl DaemonWaitForChildrenTool {
    pub fn new(agent_name: String) -> Self {
        DaemonWaitForChildrenTool { agent_name }
    }
}

impl std::fmt::Debug for DaemonWaitForChildrenTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DaemonWaitForChildrenTool")
            .field("agent_name", &self.agent_name)
            .finish()
    }
}

#[async_trait]
impl Tool for DaemonWaitForChildrenTool {
    fn name(&self) -> &str {
        "wait_for_children"
    }

    fn description(&self) -> &str {
        "Wait for multiple spawned child agents to complete in parallel and get all results at once. Use the child_ids from spawn_child. Default timeout is 300s — do not reduce this unless you have a specific reason."
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "child_ids": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Array of child_ids returned by spawn_child"
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Max seconds to wait per child (default: 300, minimum: 30). Agent responses typically take 30-120s."
                }
            },
            "required": ["child_ids"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let child_ids = input
            .get("child_ids")
            .and_then(|v| v.as_array())
            .ok_or_else(|| ToolError::InvalidInput("child_ids is required and must be an array".to_string()))?;

        if child_ids.is_empty() {
            return Err(ToolError::InvalidInput("child_ids must not be empty".to_string()));
        }

        let ids: Vec<String> = child_ids
            .iter()
            .map(|v| {
                v.as_str()
                    .ok_or_else(|| ToolError::InvalidInput("All child_ids must be strings".to_string()))
                    .map(|s| s.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;

        let timeout_secs = input
            .get("timeout_secs")
            .and_then(|v| v.as_u64())
            .unwrap_or(300)
            .max(30);

        let agent_name = self.agent_name.clone();

        // Poll all children concurrently
        let futures: Vec<_> = ids
            .iter()
            .map(|child_id| {
                let agent = agent_name.clone();
                let id = child_id.clone();
                async move { poll_for_child_result(&agent, &id, timeout_secs).await }
            })
            .collect();

        let results: Vec<Result<Value, ToolError>> =
            futures_util::future::join_all(futures).await;

        let mut result_values = Vec::new();
        let mut all_completed = true;

        for res in results {
            match res {
                Ok(val) => {
                    let status = val.get("status").and_then(|s| s.as_str()).unwrap_or("unknown");
                    if status != "completed" {
                        all_completed = false;
                    }
                    result_values.push(val);
                }
                Err(e) => {
                    all_completed = false;
                    result_values.push(serde_json::json!({
                        "status": "error",
                        "message": format!("{}", e)
                    }));
                }
            }
        }

        let overall_status = if all_completed { "completed" } else { "partial" };

        Ok(serde_json::json!({
            "status": overall_status,
            "results": result_values
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_wait_for_children_tool_name() {
        let tool = WaitForChildrenTool;
        assert_eq!(tool.name(), "wait_for_children");
    }

    #[tokio::test]
    async fn test_wait_for_children_schema() {
        let tool = WaitForChildrenTool;
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["child_ids"].is_object());
        assert_eq!(schema["properties"]["child_ids"]["type"], "array");
        assert!(schema["properties"]["timeout_secs"].is_object());
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("child_ids")));
    }

    #[tokio::test]
    async fn test_wait_for_children_missing_child_ids() {
        let tool = WaitForChildrenTool;
        let result = tool.execute(json!({})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_wait_for_children_empty_child_ids() {
        let tool = WaitForChildrenTool;
        let result = tool.execute(json!({"child_ids": []})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_wait_for_children_non_string_child_ids() {
        let tool = WaitForChildrenTool;
        let result = tool.execute(json!({"child_ids": [123, 456]})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_wait_for_children_valid_input() {
        let tool = WaitForChildrenTool;
        let result = tool
            .execute(json!({"child_ids": ["task:a:b:1", "task:a:c:2"]}))
            .await
            .unwrap();
        assert_eq!(result["status"], "wait_requested");
        let ids = result["child_ids"].as_array().unwrap();
        assert_eq!(ids.len(), 2);
    }

    // --- DaemonWaitForChildrenTool tests ---

    #[tokio::test]
    async fn test_daemon_wait_for_children_tool_name() {
        let tool = DaemonWaitForChildrenTool::new("parent".to_string());
        assert_eq!(tool.name(), "wait_for_children");
    }

    #[tokio::test]
    async fn test_daemon_wait_for_children_schema() {
        let tool = DaemonWaitForChildrenTool::new("parent".to_string());
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["child_ids"].is_object());
        assert_eq!(schema["properties"]["child_ids"]["type"], "array");
        assert!(schema["properties"]["timeout_secs"].is_object());
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("child_ids")));
    }

    #[tokio::test]
    async fn test_daemon_wait_for_children_missing_child_ids() {
        let tool = DaemonWaitForChildrenTool::new("parent".to_string());
        let result = tool.execute(json!({})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_daemon_wait_for_children_empty_child_ids() {
        let tool = DaemonWaitForChildrenTool::new("parent".to_string());
        let result = tool.execute(json!({"child_ids": []})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_daemon_wait_for_children_non_string_child_ids() {
        let tool = DaemonWaitForChildrenTool::new("parent".to_string());
        let result = tool.execute(json!({"child_ids": [123]})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }
}
