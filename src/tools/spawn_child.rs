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
                "task": {
                    "type": "string",
                    "description": "The task for the child agent to complete"
                }
            },
            "required": ["task"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        // Note: actual spawning requires agent context, handled separately
        let task = input
            .get("task")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("task is required".to_string()))?;
        Ok(serde_json::json!({"status": "spawn_requested", "task": task}))
    }
}
