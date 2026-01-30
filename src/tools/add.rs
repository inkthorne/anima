use async_trait::async_trait;
use crate::error::ToolError;
use crate::tool::Tool;
use serde_json::Value;

#[derive(Debug)]
pub struct AddTool;

#[async_trait]
impl Tool for AddTool {
    fn name(&self) -> &str {
        "add"
    }

    fn description(&self) -> &str {
        "Adds two numbers from input.a and input.b"
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
