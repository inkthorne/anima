use crate::error::ToolError;
use crate::tool::Tool;
use serde_json::Value;

#[derive(Debug)]
pub struct EchoTool;

impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "Returns the input unchanged"
    }

    fn execute(&self, input: Value) -> Result<Value, ToolError> {
        Ok(input)
    }
}
