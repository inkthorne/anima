use async_trait::async_trait;
use crate::error::ToolError;
use crate::tool::Tool;
use serde_json::Value;
use std::path::Path;

/// Tool for reading file contents from the filesystem.
#[derive(Debug)]
pub struct ReadFileTool;

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Reads the contents of a file at the given path"
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to read"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let path = input
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing or invalid 'path' field".to_string()))?;

        let path = Path::new(path);

        let contents = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read file '{}': {}", path.display(), e)))?;

        Ok(serde_json::json!({ "contents": contents }))
    }
}
