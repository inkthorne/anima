use crate::error::ToolError;
use serde_json::Value;

/// A tool that an agent can use to interact with the world.
pub trait Tool: Send + Sync {
    /// Returns the unique name of this tool.
    fn name(&self) -> &str;

    /// Returns a human-readable description of what this tool does.
    fn description(&self) -> &str;

    /// Executes the tool with the given JSON input.
    fn execute(&self, input: Value) -> Result<Value, ToolError>;
}

/// Information about a tool for introspection.
#[derive(Debug, Clone)]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
}
