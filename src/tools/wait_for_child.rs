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
        "Wait for a child agent to complete and get its result."
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "child_id": {
                    "type": "string",
                    "description": "The ID of the child agent to wait for"
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
