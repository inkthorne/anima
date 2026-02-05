use async_trait::async_trait;
use std::sync::Arc;
use tokio::net::UnixStream;
use tokio::sync::Mutex;

use crate::discovery;
use crate::error::ToolError;
use crate::messaging::{AgentMessage, MessageRouter};
use crate::socket_api::{Request, Response, SocketApi};
use crate::tool::Tool;
use serde_json::Value;

/// Tool that lets agents send messages to other agents
pub struct SendMessageTool {
    /// The router for sending messages
    router: Arc<Mutex<MessageRouter>>,
    /// The ID of the agent using this tool
    agent_id: String,
}

impl SendMessageTool {
    /// Create a new SendMessageTool
    pub fn new(router: Arc<Mutex<MessageRouter>>, agent_id: String) -> Self {
        SendMessageTool { router, agent_id }
    }
}

impl std::fmt::Debug for SendMessageTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SendMessageTool")
            .field("agent_id", &self.agent_id)
            .finish()
    }
}

#[async_trait]
impl Tool for SendMessageTool {
    fn name(&self) -> &str {
        "send_message"
    }

    fn description(&self) -> &str {
        "Send a message to another agent. Use this to communicate with peer agents."
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "The ID of the agent to send the message to"
                },
                "message": {
                    "type": "string",
                    "description": "The content of the message to send"
                }
            },
            "required": ["to", "message"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let to = input
            .get("to")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing or invalid 'to' field".to_string()))?;

        let message = input
            .get("message")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ToolError::InvalidInput("Missing or invalid 'message' field".to_string())
            })?;

        let msg = AgentMessage::new(&self.agent_id, to, message);

        let router = self.router.lock().await;
        router
            .send(msg)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        Ok(serde_json::json!({
            "sent": true,
            "to": to
        }))
    }
}

/// Tool that lets agents list available peer agents
pub struct ListAgentsTool {
    /// The router for querying agents
    router: Arc<Mutex<MessageRouter>>,
}

impl ListAgentsTool {
    /// Create a new ListAgentsTool
    pub fn new(router: Arc<Mutex<MessageRouter>>) -> Self {
        ListAgentsTool { router }
    }
}

impl std::fmt::Debug for ListAgentsTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ListAgentsTool").finish()
    }
}

#[async_trait]
impl Tool for ListAgentsTool {
    fn name(&self) -> &str {
        "list_agents"
    }

    fn description(&self) -> &str {
        "List all available agents that you can send messages to."
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }

    async fn execute(&self, _input: Value) -> Result<Value, ToolError> {
        let router = self.router.lock().await;
        let agents = router.list_agents();

        Ok(serde_json::json!({
            "agents": agents,
            "count": agents.len()
        }))
    }
}

/// Daemon-aware tool that sends messages to other agents via Unix sockets.
/// Used by daemons to communicate with other daemons.
pub struct DaemonSendMessageTool {
    /// The ID of the agent using this tool
    agent_id: String,
}

impl DaemonSendMessageTool {
    /// Create a new DaemonSendMessageTool
    pub fn new(agent_id: String) -> Self {
        DaemonSendMessageTool { agent_id }
    }
}

impl std::fmt::Debug for DaemonSendMessageTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DaemonSendMessageTool")
            .field("agent_id", &self.agent_id)
            .finish()
    }
}

#[async_trait]
impl Tool for DaemonSendMessageTool {
    fn name(&self) -> &str {
        "send_message"
    }

    fn description(&self) -> &str {
        "Send a message to another agent. Use this to communicate with peer agents running as daemons."
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "The name of the agent to send the message to"
                },
                "message": {
                    "type": "string",
                    "description": "The content of the message to send"
                }
            },
            "required": ["to", "message"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let to = input
            .get("to")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing or invalid 'to' field".to_string()))?;

        let message = input
            .get("message")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ToolError::InvalidInput("Missing or invalid 'message' field".to_string())
            })?;

        // Check if the target agent is running
        if !discovery::is_agent_running(to) {
            return Err(ToolError::ExecutionFailed(format!(
                "Agent '{}' is not running. Start it with 'anima start {}'",
                to, to
            )));
        }

        // Get socket path
        let socket_path = discovery::agent_socket_path(to).ok_or_else(|| {
            ToolError::ExecutionFailed("Could not determine agent socket path".to_string())
        })?;

        // Connect to the target daemon
        let stream = UnixStream::connect(&socket_path).await.map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to connect to agent '{}': {}", to, e))
        })?;

        let mut api = SocketApi::new(stream);

        // Send the incoming message request
        let request = Request::IncomingMessage {
            from: self.agent_id.clone(),
            content: message.to_string(),
        };

        api.write_request(&request)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to send message: {}", e)))?;

        // Read the response
        let response = api
            .read_response()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read response: {}", e)))?;

        match response {
            Some(Response::Ok) => Ok(serde_json::json!({
                "sent": true,
                "to": to
            })),
            Some(Response::Message { content }) => Ok(serde_json::json!({
                "sent": true,
                "to": to,
                "response": content
            })),
            Some(Response::Error { message }) => Err(ToolError::ExecutionFailed(message)),
            None => Err(ToolError::ExecutionFailed(
                "Connection closed unexpectedly".to_string(),
            )),
            _ => Err(ToolError::ExecutionFailed(
                "Unexpected response from agent".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_send_message_tool_name() {
        let router = Arc::new(Mutex::new(MessageRouter::new()));
        let tool = SendMessageTool::new(router, "test-agent".to_string());
        assert_eq!(tool.name(), "send_message");
    }

    #[tokio::test]
    async fn test_send_message_tool_schema() {
        let router = Arc::new(Mutex::new(MessageRouter::new()));
        let tool = SendMessageTool::new(router, "test-agent".to_string());
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["to"].is_object());
        assert!(schema["properties"]["message"].is_object());
    }

    #[tokio::test]
    async fn test_send_message_success() {
        let router = Arc::new(Mutex::new(MessageRouter::new()));

        // Register recipient agent
        let mut rx = {
            let mut r = router.lock().await;
            r.register("recipient")
        };

        let tool = SendMessageTool::new(router, "sender".to_string());
        let result = tool
            .execute(json!({
                "to": "recipient",
                "message": "hello"
            }))
            .await
            .unwrap();

        assert_eq!(result["sent"], true);
        assert_eq!(result["to"], "recipient");

        // Verify message was received
        let msg = rx.try_recv().unwrap();
        assert_eq!(msg.from, "sender");
        assert_eq!(msg.to, "recipient");
        assert_eq!(msg.content, "hello");
    }

    #[tokio::test]
    async fn test_send_message_agent_not_found() {
        let router = Arc::new(Mutex::new(MessageRouter::new()));
        let tool = SendMessageTool::new(router, "sender".to_string());

        let result = tool
            .execute(json!({
                "to": "nonexistent",
                "message": "hello"
            }))
            .await;

        assert!(matches!(result, Err(ToolError::ExecutionFailed(_))));
    }

    #[tokio::test]
    async fn test_send_message_missing_to() {
        let router = Arc::new(Mutex::new(MessageRouter::new()));
        let tool = SendMessageTool::new(router, "sender".to_string());

        let result = tool
            .execute(json!({
                "message": "hello"
            }))
            .await;

        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_send_message_missing_message() {
        let router = Arc::new(Mutex::new(MessageRouter::new()));
        let tool = SendMessageTool::new(router, "sender".to_string());

        let result = tool
            .execute(json!({
                "to": "recipient"
            }))
            .await;

        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_list_agents_tool_name() {
        let router = Arc::new(Mutex::new(MessageRouter::new()));
        let tool = ListAgentsTool::new(router);
        assert_eq!(tool.name(), "list_agents");
    }

    #[tokio::test]
    async fn test_list_agents_empty() {
        let router = Arc::new(Mutex::new(MessageRouter::new()));
        let tool = ListAgentsTool::new(router);

        let result = tool.execute(json!({})).await.unwrap();
        assert_eq!(result["count"], 0);
        assert!(result["agents"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_list_agents_with_agents() {
        let router = Arc::new(Mutex::new(MessageRouter::new()));

        // Register some agents
        {
            let mut r = router.lock().await;
            let _ = r.register("agent-1");
            let _ = r.register("agent-2");
        }

        let tool = ListAgentsTool::new(router);
        let result = tool.execute(json!({})).await.unwrap();

        assert_eq!(result["count"], 2);
        let agents = result["agents"].as_array().unwrap();
        assert!(agents.contains(&json!("agent-1")));
        assert!(agents.contains(&json!("agent-2")));
    }
}
