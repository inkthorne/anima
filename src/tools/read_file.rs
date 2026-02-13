use crate::error::ToolError;
use crate::tool::Tool;
use async_trait::async_trait;
use serde_json::Value;
use std::path::PathBuf;

/// Expand tilde (~) in path to home directory
fn expand_tilde(path: &str) -> PathBuf {
    if let Some(suffix) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(suffix);
        }
    } else if path == "~"
        && let Some(home) = dirs::home_dir()
    {
        return home;
    }
    PathBuf::from(path)
}

/// Parse a JSON value as usize, with float fallback for Ollama models that send integers as floats.
fn json_to_usize(v: &Value) -> Option<usize> {
    v.as_u64()
        .or_else(|| v.as_f64().map(|f| f as u64))
        .map(|n| n as usize)
}

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
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-based)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to return"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let path_str = input.get("path").and_then(|v| v.as_str()).ok_or_else(|| {
            ToolError::InvalidInput("Missing or invalid 'path' field".to_string())
        })?;

        let path = expand_tilde(path_str);

        let contents = tokio::fs::read_to_string(&path).await.map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to read file '{}': {}", path.display(), e))
        })?;

        let offset = input.get("offset").and_then(json_to_usize);
        let limit = input.get("limit").and_then(json_to_usize);

        let result = if offset.is_some() || limit.is_some() {
            let lines: Vec<&str> = contents.lines().collect();
            let start = offset.map(|o| o.saturating_sub(1)).unwrap_or(0).min(lines.len());
            let end = if let Some(lim) = limit {
                (start + lim).min(lines.len())
            } else {
                lines.len()
            };
            lines[start..end]
                .iter()
                .enumerate()
                .map(|(i, line)| format!("{:>8}  {}", start + i + 1, line))
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            contents
        };

        Ok(serde_json::json!({ "contents": result }))
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
        assert!(
            schema["required"]
                .as_array()
                .unwrap()
                .contains(&json!("path"))
        );
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
        let result = tool
            .execute(json!({"path": "/nonexistent/file/path/abc123.txt"}))
            .await;
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

    #[tokio::test]
    async fn test_read_with_offset() {
        let mut file = NamedTempFile::new().unwrap();
        for i in 1..=5 {
            writeln!(file, "line {}", i).unwrap();
        }
        let path = file.path().to_str().unwrap();

        let tool = ReadFileTool;
        let result = tool.execute(json!({"path": path, "offset": 3})).await.unwrap();
        let contents = result["contents"].as_str().unwrap();
        assert_eq!(contents, "       3  line 3\n       4  line 4\n       5  line 5");
    }

    #[tokio::test]
    async fn test_read_with_limit() {
        let mut file = NamedTempFile::new().unwrap();
        for i in 1..=5 {
            writeln!(file, "line {}", i).unwrap();
        }
        let path = file.path().to_str().unwrap();

        let tool = ReadFileTool;
        let result = tool.execute(json!({"path": path, "limit": 2})).await.unwrap();
        let contents = result["contents"].as_str().unwrap();
        assert_eq!(contents, "       1  line 1\n       2  line 2");
    }

    #[tokio::test]
    async fn test_read_with_offset_and_limit() {
        let mut file = NamedTempFile::new().unwrap();
        for i in 1..=10 {
            writeln!(file, "line {}", i).unwrap();
        }
        let path = file.path().to_str().unwrap();

        let tool = ReadFileTool;
        let result = tool.execute(json!({"path": path, "offset": 3, "limit": 4})).await.unwrap();
        let contents = result["contents"].as_str().unwrap();
        assert_eq!(contents, "       3  line 3\n       4  line 4\n       5  line 5\n       6  line 6");
    }

    #[tokio::test]
    async fn test_read_offset_beyond_file() {
        let mut file = NamedTempFile::new().unwrap();
        for i in 1..=5 {
            writeln!(file, "line {}", i).unwrap();
        }
        let path = file.path().to_str().unwrap();

        let tool = ReadFileTool;
        let result = tool.execute(json!({"path": path, "offset": 100})).await.unwrap();
        let contents = result["contents"].as_str().unwrap();
        assert_eq!(contents, "");
    }

    #[tokio::test]
    async fn test_read_offset_as_float() {
        let mut file = NamedTempFile::new().unwrap();
        for i in 1..=5 {
            writeln!(file, "line {}", i).unwrap();
        }
        let path = file.path().to_str().unwrap();

        let tool = ReadFileTool;
        let result = tool.execute(json!({"path": path, "offset": 3.0, "limit": 2.0})).await.unwrap();
        let contents = result["contents"].as_str().unwrap();
        assert_eq!(contents, "       3  line 3\n       4  line 4");
    }

    #[tokio::test]
    async fn test_read_with_offset_has_line_numbers() {
        let mut file = NamedTempFile::new().unwrap();
        for i in 1..=100 {
            writeln!(file, "content {}", i).unwrap();
        }
        let path = file.path().to_str().unwrap();

        let tool = ReadFileTool;
        let result = tool.execute(json!({"path": path, "offset": 98, "limit": 3})).await.unwrap();
        let contents = result["contents"].as_str().unwrap();
        // Verify 1-based numbering and {:>8} format with double-digit and triple-digit line numbers
        assert_eq!(contents, "      98  content 98\n      99  content 99\n     100  content 100");
    }
}
