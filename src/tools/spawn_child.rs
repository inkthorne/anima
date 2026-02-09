use crate::error::ToolError;
use crate::tool::Tool;
use async_trait::async_trait;
use serde_json::Value;

#[derive(Debug)]
pub struct SpawnChildTool;

#[async_trait]
impl Tool for SpawnChildTool {
    fn name(&self) -> &str {
        "spawn_child"
    }

    fn description(&self) -> &str {
        "Spawn a child agent to handle a subtask. Returns the child_id."
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
                    "description": "The task for the child agent to complete"
                }
            },
            "required": ["agent", "task"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        // Note: actual spawning requires agent context, handled separately
        let _agent = input
            .get("agent")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("agent is required".to_string()))?;
        let task = input
            .get("task")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("task is required".to_string()))?;
        Ok(serde_json::json!({"status": "spawn_requested", "task": task}))
    }
}

/// Daemon-aware tool that spawns a child agent to handle a subtask.
/// Creates a task conversation, posts the task, and notifies the child agent.
pub struct DaemonSpawnChildTool {
    agent_name: String,
}

impl DaemonSpawnChildTool {
    pub fn new(agent_name: String) -> Self {
        DaemonSpawnChildTool { agent_name }
    }
}

impl std::fmt::Debug for DaemonSpawnChildTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DaemonSpawnChildTool")
            .field("agent_name", &self.agent_name)
            .finish()
    }
}

#[async_trait]
impl Tool for DaemonSpawnChildTool {
    fn name(&self) -> &str {
        "spawn_child"
    }

    fn description(&self) -> &str {
        "Delegate a task to another agent. Starts the agent if needed, creates a task conversation, and sends the task. Returns a child_id to use with wait_for_children."
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
                    "description": "The task for the child agent to complete"
                }
            },
            "required": ["agent", "task"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        use crate::conversation::ConversationStore;
        use crate::discovery;
        use crate::socket_api::{Request, SocketApi};
        use rand::Rng;
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

        // Post task message and pin it so it stays in context
        let message_id = store
            .add_message(&conv_name, &self.agent_name, task, &[agent])
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
            "status": "spawned",
            "child_id": conv_name,
            "agent": agent
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_spawn_child_tool_name() {
        let tool = SpawnChildTool;
        assert_eq!(tool.name(), "spawn_child");
    }

    #[tokio::test]
    async fn test_spawn_child_schema() {
        let tool = SpawnChildTool;
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["agent"].is_object());
        assert!(schema["properties"]["task"].is_object());
    }

    #[tokio::test]
    async fn test_spawn_child_missing_agent() {
        let tool = SpawnChildTool;
        let result = tool.execute(json!({"task": "do something"})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_spawn_child_missing_task() {
        let tool = SpawnChildTool;
        let result = tool.execute(json!({"agent": "test-agent"})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_daemon_spawn_child_tool_name() {
        let tool = DaemonSpawnChildTool::new("parent".to_string());
        assert_eq!(tool.name(), "spawn_child");
    }

    #[tokio::test]
    async fn test_daemon_spawn_child_schema() {
        let tool = DaemonSpawnChildTool::new("parent".to_string());
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["agent"].is_object());
        assert!(schema["properties"]["task"].is_object());
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("agent")));
        assert!(required.contains(&json!("task")));
    }

    #[tokio::test]
    async fn test_daemon_spawn_child_missing_agent() {
        let tool = DaemonSpawnChildTool::new("parent".to_string());
        let result = tool.execute(json!({"task": "do something"})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_daemon_spawn_child_missing_task() {
        let tool = DaemonSpawnChildTool::new("parent".to_string());
        let result = tool.execute(json!({"agent": "test-agent"})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_daemon_spawn_child_nonexistent_agent() {
        let tool = DaemonSpawnChildTool::new("parent".to_string());
        let result = tool
            .execute(json!({"agent": "nonexistent-xyz-999", "task": "hello"}))
            .await;
        assert!(matches!(result, Err(ToolError::ExecutionFailed(_))));
    }
}
