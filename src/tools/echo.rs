use async_trait::async_trait;
use crate::error::ToolError;
use crate::tool::Tool;
use serde_json::Value;

#[derive(Debug)]
pub struct EchoTool;

impl Default for EchoTool {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "Returns the input unchanged"
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to echo back"
                }
            },
            "required": ["message"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        Ok(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_echo_tool_name() {
        let tool = EchoTool;
        assert_eq!(tool.name(), "echo");
    }

    #[test]
    fn test_echo_tool_description() {
        let tool = EchoTool;
        assert!(tool.description().contains("unchanged"));
    }

    #[test]
    fn test_echo_tool_schema() {
        let tool = EchoTool;
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["message"].is_object());
        assert!(schema["required"].as_array().unwrap().contains(&json!("message")));
    }

    #[tokio::test]
    async fn test_echo_simple_message() {
        let tool = EchoTool;
        let input = json!({"message": "hello"});
        let result = tool.execute(input.clone()).await.unwrap();
        assert_eq!(result, input);
    }

    #[tokio::test]
    async fn test_echo_empty_message() {
        let tool = EchoTool;
        let input = json!({"message": ""});
        let result = tool.execute(input.clone()).await.unwrap();
        assert_eq!(result, input);
    }

    #[tokio::test]
    async fn test_echo_complex_input() {
        let tool = EchoTool;
        let input = json!({"message": "test", "extra": 123, "nested": {"a": 1}});
        let result = tool.execute(input.clone()).await.unwrap();
        assert_eq!(result, input);
    }

    #[tokio::test]
    async fn test_echo_preserves_all_fields() {
        let tool = EchoTool;
        let input = json!({"message": "hello", "other": "data"});
        let result = tool.execute(input.clone()).await.unwrap();
        assert_eq!(result["message"], "hello");
        assert_eq!(result["other"], "data");
    }
}
