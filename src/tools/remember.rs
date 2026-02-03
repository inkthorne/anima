use async_trait::async_trait;
use crate::error::ToolError;
use crate::tool::Tool;
use serde_json::Value;

/// Tool that allows agents to explicitly save memories via tool call.
/// This provides an alternative to the [REMEMBER: ...] tag syntax.
#[derive(Debug, Default)]
pub struct RememberTool;

#[async_trait]
impl Tool for RememberTool {
    fn name(&self) -> &str {
        "remember"
    }

    fn description(&self) -> &str {
        "Saves a memory to persistent storage for future recall"
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to remember"
                }
            },
            "required": ["content"]
        })
    }

    async fn execute(&self, _input: Value) -> Result<Value, ToolError> {
        // Note: Actual memory saving happens in daemon.rs execute_tool_call
        // because it requires access to SemanticMemoryStore and EmbeddingClient.
        // This method is not called directly.
        Err(ToolError::ExecutionFailed(
            "RememberTool must be executed through daemon context".to_string()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_remember_tool_name() {
        let tool = RememberTool;
        assert_eq!(tool.name(), "remember");
    }

    #[test]
    fn test_remember_tool_description() {
        let tool = RememberTool;
        assert!(tool.description().contains("memory"));
    }

    #[test]
    fn test_remember_tool_schema() {
        let tool = RememberTool;
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["content"].is_object());
        assert!(schema["required"].as_array().unwrap().contains(&json!("content")));
    }
}
