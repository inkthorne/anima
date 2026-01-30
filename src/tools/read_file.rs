use async_trait::async_trait;
use crate::error::ToolError;
use crate::tool::Tool;
use serde_json::Value;
use std::path::Path;

/// Tool for reading file contents from the filesystem.
#[derive(Debug, Default)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_read_file_tool_name() {
        let tool = ReadFileTool;
        assert_eq!(tool.name(), "read_file");
    }

    #[test]
    fn test_read_file_tool_description() {
        let tool = ReadFileTool;
        assert!(tool.description().contains("file"));
    }

    #[test]
    fn test_read_file_tool_schema() {
        let tool = ReadFileTool;
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["path"].is_object());
        assert!(schema["required"].as_array().unwrap().contains(&json!("path")));
    }

    #[tokio::test]
    async fn test_read_existing_file() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "test content").unwrap();
        let path = file.path().to_str().unwrap();

        let tool = ReadFileTool;
        let result = tool.execute(json!({"path": path})).await.unwrap();
        assert_eq!(result["contents"], "test content");
    }

    #[tokio::test]
    async fn test_read_missing_file() {
        let tool = ReadFileTool;
        let result = tool.execute(json!({"path": "/nonexistent/file/path/abc123.txt"})).await;
        assert!(matches!(result, Err(ToolError::ExecutionFailed(_))));
    }

    #[tokio::test]
    async fn test_read_missing_path_field() {
        let tool = ReadFileTool;
        let result = tool.execute(json!({})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_read_invalid_path_type() {
        let tool = ReadFileTool;
        let result = tool.execute(json!({"path": 123})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_read_empty_file() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap();

        let tool = ReadFileTool;
        let result = tool.execute(json!({"path": path})).await.unwrap();
        assert_eq!(result["contents"], "");
    }

    #[tokio::test]
    async fn test_read_multiline_file() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "line 1").unwrap();
        writeln!(file, "line 2").unwrap();
        writeln!(file, "line 3").unwrap();
        let path = file.path().to_str().unwrap();

        let tool = ReadFileTool;
        let result = tool.execute(json!({"path": path})).await.unwrap();
        let contents = result["contents"].as_str().unwrap();
        assert!(contents.contains("line 1"));
        assert!(contents.contains("line 2"));
        assert!(contents.contains("line 3"));
    }
}
