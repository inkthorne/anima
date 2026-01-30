use async_trait::async_trait;
use crate::error::ToolError;
use crate::tool::Tool;
use serde_json::Value;

#[derive(Debug)]
pub struct AddTool;

impl Default for AddTool {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for AddTool {
    fn name(&self) -> &str {
        "add"
    }

    fn description(&self) -> &str {
        "Adds two numbers from input.a and input.b"
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "First number to add"
                },
                "b": {
                    "type": "number",
                    "description": "Second number to add"
                }
            },
            "required": ["a", "b"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let a = input
            .get("a")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| ToolError::InvalidInput("Missing or invalid 'a' field".to_string()))?;

        let b = input
            .get("b")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| ToolError::InvalidInput("Missing or invalid 'b' field".to_string()))?;

        let sum = a + b;
        Ok(serde_json::json!(sum))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_add_tool_name() {
        let tool = AddTool;
        assert_eq!(tool.name(), "add");
    }

    #[test]
    fn test_add_tool_description() {
        let tool = AddTool;
        assert!(tool.description().contains("two numbers"));
    }

    #[test]
    fn test_add_tool_schema() {
        let tool = AddTool;
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["a"].is_object());
        assert!(schema["properties"]["b"].is_object());
        assert!(schema["required"].as_array().unwrap().contains(&json!("a")));
        assert!(schema["required"].as_array().unwrap().contains(&json!("b")));
    }

    #[tokio::test]
    async fn test_add_positive_numbers() {
        let tool = AddTool;
        let result = tool.execute(json!({"a": 2, "b": 3})).await.unwrap();
        assert_eq!(result, json!(5.0));
    }

    #[tokio::test]
    async fn test_add_negative_numbers() {
        let tool = AddTool;
        let result = tool.execute(json!({"a": -5, "b": -3})).await.unwrap();
        assert_eq!(result, json!(-8.0));
    }

    #[tokio::test]
    async fn test_add_floats() {
        let tool = AddTool;
        let result = tool.execute(json!({"a": 1.5, "b": 2.5})).await.unwrap();
        assert_eq!(result, json!(4.0));
    }

    #[tokio::test]
    async fn test_add_mixed_signs() {
        let tool = AddTool;
        let result = tool.execute(json!({"a": -10, "b": 25})).await.unwrap();
        assert_eq!(result, json!(15.0));
    }

    #[tokio::test]
    async fn test_add_missing_a() {
        let tool = AddTool;
        let result = tool.execute(json!({"b": 3})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_add_missing_b() {
        let tool = AddTool;
        let result = tool.execute(json!({"a": 3})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_add_invalid_type() {
        let tool = AddTool;
        let result = tool.execute(json!({"a": "not a number", "b": 3})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }
}
