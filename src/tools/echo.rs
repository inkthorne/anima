use async_trait::async_trait;
use crate::error::ToolError;
use crate::tool::Tool;
use serde_json::Value;

#[derive(Debug)]
pub struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "Returns the input unchanged"
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        Ok(input)
    }
}
