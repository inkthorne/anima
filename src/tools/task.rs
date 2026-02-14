use crate::error::ToolError;
use crate::tool::Tool;
use async_trait::async_trait;
use serde_json::Value;

/// Daemon-aware tool that dispatches a task to an agent asynchronously.
/// Creates a task conversation, posts the task with origin metadata, and notifies the child agent.
/// The result is automatically relayed back to the originating conversation when the child finishes.
pub struct DaemonTaskTool {
    agent_name: String,
    conv_id: Option<String>,
}

impl DaemonTaskTool {
    pub fn new(agent_name: String, conv_id: Option<String>) -> Self {
        DaemonTaskTool {
            agent_name,
            conv_id,
        }
    }
}

impl std::fmt::Debug for DaemonTaskTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DaemonTaskTool")
            .field("agent_name", &self.agent_name)
            .finish()
    }
}

#[async_trait]
impl Tool for DaemonTaskTool {
    fn name(&self) -> &str {
        "task"
    }

    fn description(&self) -> &str {
        "Dispatch a task to an agent asynchronously. The agent works in a separate conversation and the result is automatically posted back here when complete. Returns immediately — do not wait for the result."
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "Name of the agent to delegate to"
                },
                "task": {
                    "type": "string",
                    "description": "The task for the agent to complete"
                }
            },
            "required": ["agent", "task"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        use crate::conversation::ConversationStore;
        use crate::discovery;
        use crate::socket_api::{Request, SocketApi};
        use rand::RngExt;
        use tokio::net::UnixStream;

        let agent = input
            .get("agent")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("agent is required".to_string()))?;

        let task = input
            .get("task")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("task is required".to_string()))?;

        // Validate agent exists
        if !discovery::agent_exists(agent) {
            return Err(ToolError::ExecutionFailed(format!(
                "Agent '{}' does not exist",
                agent
            )));
        }

        // Start agent if not running
        if !discovery::is_agent_running(agent) {
            discovery::start_agent_daemon(agent).map_err(|e| {
                ToolError::ExecutionFailed(format!("Failed to start agent '{}': {}", agent, e))
            })?;
        }

        // Generate short random ID for task conversation
        let short_id = {
            let mut rng = rand::rng();
            let id_bytes: [u8; 4] = rng.random();
            id_bytes.iter().map(|b| format!("{:02x}", b)).collect::<String>()
        };
        let conv_name = format!("task:{}:{}:{}", self.agent_name, agent, short_id);

        // Create task conversation
        let store = ConversationStore::init().map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to open conversation store: {}", e))
        })?;

        store
            .create_conversation(
                Some(&conv_name),
                &[&self.agent_name, agent],
            )
            .map_err(|e| {
                ToolError::ExecutionFailed(format!("Failed to create task conversation: {}", e))
            })?;

        // Build task message with origin metadata prefix
        let origin_conv = self.conv_id.as_deref().unwrap_or("unknown");
        let task_content = format!(
            "[task origin={} by={}]\n{}",
            origin_conv, self.agent_name, task
        );

        // Post task message and pin it so it stays in context
        let message_id = store
            .add_message(&conv_name, &self.agent_name, &task_content, &[agent])
            .map_err(|e| {
                ToolError::ExecutionFailed(format!("Failed to post task message: {}", e))
            })?;

        store.pin_message(&conv_name, message_id, true).map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to pin task message: {}", e))
        })?;

        // Connect to child agent with retry (handles both fresh-start and already-running)
        let socket_path = discovery::agents_dir().join(agent).join("agent.sock");
        let mut stream_opt = None;
        for _ in 0..50 {
            match UnixStream::connect(&socket_path).await {
                Ok(s) => {
                    stream_opt = Some(s);
                    break;
                }
                Err(_) => {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            }
        }
        let stream = stream_opt.ok_or_else(|| {
            ToolError::ExecutionFailed(format!(
                "Failed to connect to agent '{}' after retries",
                agent
            ))
        })?;

        let mut api = SocketApi::new(stream);
        let request = Request::Notify {
            conv_id: conv_name.clone(),
            message_id,
            depth: 0,
        };

        api.write_request(&request).await.map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to notify agent '{}': {}", agent, e))
        })?;

        // Read response (don't need to wait for processing, just acknowledgement)
        let _response = api.read_response().await.ok();

        Ok(serde_json::json!({
            "status": "dispatched",
            "task_conv": conv_name,
            "agent": agent,
            "note": "Task dispatched. End your turn now — the result will be posted back to this conversation automatically."
        }))
    }
}

/// Parse task origin metadata from a pinned task message content.
/// Expected format on the first line: `[task origin=<conv_id> by=<agent_name>]`
/// Returns `(origin_conv_id, origin_agent_name)` if found.
pub fn parse_task_origin(content: &str) -> Option<(String, String)> {
    let first_line = content.lines().next()?;
    // Match [task origin=<value> by=<value>]
    let rest = first_line.strip_prefix("[task origin=")?;
    let by_pos = rest.find(" by=")?;
    let origin = &rest[..by_pos];
    let after_by = &rest[by_pos + 4..]; // skip " by="
    let end_bracket = after_by.find(']')?;
    let agent = &after_by[..end_bracket];
    if origin.is_empty() || agent.is_empty() {
        return None;
    }
    Some((origin.to_string(), agent.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parse_task_origin_valid() {
        let content = "[task origin=chat-123 by=arya]\nRefactor the auth module";
        let (origin, agent) = parse_task_origin(content).unwrap();
        assert_eq!(origin, "chat-123");
        assert_eq!(agent, "arya");
    }

    #[test]
    fn test_parse_task_origin_no_body() {
        let content = "[task origin=my-conv by=dash]";
        let (origin, agent) = parse_task_origin(content).unwrap();
        assert_eq!(origin, "my-conv");
        assert_eq!(agent, "dash");
    }

    #[test]
    fn test_parse_task_origin_complex_names() {
        let content = "[task origin=task:arya:dash:abc1 by=arya]\nNested task";
        let (origin, agent) = parse_task_origin(content).unwrap();
        assert_eq!(origin, "task:arya:dash:abc1");
        assert_eq!(agent, "arya");
    }

    #[test]
    fn test_parse_task_origin_missing_prefix() {
        assert!(parse_task_origin("Just some text").is_none());
    }

    #[test]
    fn test_parse_task_origin_malformed() {
        assert!(parse_task_origin("[task origin= by=]").is_none());
        assert!(parse_task_origin("[task origin=x]").is_none());
        assert!(parse_task_origin("").is_none());
    }

    #[tokio::test]
    async fn test_daemon_task_tool_name() {
        let tool = DaemonTaskTool::new("parent".to_string(), Some("conv-1".to_string()));
        assert_eq!(tool.name(), "task");
    }

    #[tokio::test]
    async fn test_daemon_task_schema() {
        let tool = DaemonTaskTool::new("parent".to_string(), None);
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["agent"].is_object());
        assert!(schema["properties"]["task"].is_object());
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("agent")));
        assert!(required.contains(&json!("task")));
    }

    #[tokio::test]
    async fn test_daemon_task_missing_agent() {
        let tool = DaemonTaskTool::new("parent".to_string(), None);
        let result = tool.execute(json!({"task": "do something"})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_daemon_task_missing_task() {
        let tool = DaemonTaskTool::new("parent".to_string(), None);
        let result = tool.execute(json!({"agent": "test-agent"})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_daemon_task_nonexistent_agent() {
        let tool = DaemonTaskTool::new("parent".to_string(), Some("conv-1".to_string()));
        let result = tool
            .execute(json!({"agent": "nonexistent-xyz-999", "task": "hello"}))
            .await;
        assert!(matches!(result, Err(ToolError::ExecutionFailed(_))));
    }
}
