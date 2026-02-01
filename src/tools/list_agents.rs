//! Tool for listing available agents.

use async_trait::async_trait;

use crate::discovery;
use crate::error::ToolError;
use crate::tool::Tool;
use serde_json::Value;

/// Daemon-aware tool that lists available agents by discovering running daemons.
pub struct DaemonListAgentsTool {
    /// The ID of the agent using this tool (excluded from results)
    agent_id: String,
}

impl DaemonListAgentsTool {
    /// Create a new DaemonListAgentsTool
    pub fn new(agent_id: String) -> Self {
        DaemonListAgentsTool { agent_id }
    }
}

impl std::fmt::Debug for DaemonListAgentsTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DaemonListAgentsTool")
            .field("agent_id", &self.agent_id)
            .finish()
    }
}

#[async_trait]
impl Tool for DaemonListAgentsTool {
    fn name(&self) -> &str {
        "list_agents"
    }

    fn description(&self) -> &str {
        "List all available agents that you can send messages to. Returns agents running as daemons."
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }

    async fn execute(&self, _input: Value) -> Result<Value, ToolError> {
        // Discover running agents
        let running = discovery::discover_running_agents();

        // Filter out ourselves
        let agents: Vec<String> = running
            .into_iter()
            .filter(|a| a.name != self.agent_id)
            .map(|a| a.name)
            .collect();

        Ok(serde_json::json!({
            "agents": agents,
            "count": agents.len()
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_daemon_list_agents_tool_name() {
        let tool = DaemonListAgentsTool::new("test-agent".to_string());
        assert_eq!(tool.name(), "list_agents");
    }

    #[tokio::test]
    async fn test_daemon_list_agents_tool_description() {
        let tool = DaemonListAgentsTool::new("test-agent".to_string());
        assert!(tool.description().contains("agents"));
    }

    #[tokio::test]
    async fn test_daemon_list_agents_tool_schema() {
        let tool = DaemonListAgentsTool::new("test-agent".to_string());
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
    }

    #[tokio::test]
    async fn test_daemon_list_agents_empty() {
        // Use a fake HOME so no agents are found
        let fake_home = tempdir().unwrap();
        let original_home = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", fake_home.path()) };

        let tool = DaemonListAgentsTool::new("test-agent".to_string());
        let result = tool.execute(json!({})).await.unwrap();

        // Restore HOME
        match original_home {
            Some(h) => unsafe { std::env::set_var("HOME", h) },
            None => unsafe { std::env::remove_var("HOME") },
        }

        assert_eq!(result["count"], 0);
        assert!(result["agents"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_daemon_list_agents_debug() {
        let tool = DaemonListAgentsTool::new("test-agent".to_string());
        let debug_str = format!("{:?}", tool);
        assert!(debug_str.contains("DaemonListAgentsTool"));
        assert!(debug_str.contains("test-agent"));
    }
}
