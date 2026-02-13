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

/// Tool for reading a range of lines from a file, with formatted line numbers.
#[derive(Debug, Default)]
pub struct PeekFileTool;

#[async_trait]
impl Tool for PeekFileTool {
    fn name(&self) -> &str {
        "peek_file"
    }

    fn description(&self) -> &str {
        "Read a range of lines from a file (returns formatted line numbers)"
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to read"
                },
                "start_line": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-based)"
                },
                "end_line": {
                    "type": "integer",
                    "description": "Line number to stop reading at (inclusive, defaults to end of file)"
                }
            },
            "required": ["path", "start_line"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let path_str = input.get("path").and_then(|v| v.as_str()).ok_or_else(|| {
            ToolError::InvalidInput("Missing or invalid 'path' field".to_string())
        })?;

        let start_line = input.get("start_line").and_then(json_to_usize).ok_or_else(|| {
            ToolError::InvalidInput("Missing or invalid 'start_line' field".to_string())
        })?;

        let end_line = input.get("end_line").and_then(json_to_usize);

        let path = expand_tilde(path_str);

        let contents = tokio::fs::read_to_string(&path).await.map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to read file '{}': {}", path.display(), e))
        })?;

        let lines: Vec<&str> = contents.lines().collect();
        let start = start_line.saturating_sub(1).min(lines.len());
        let end = if let Some(el) = end_line {
            el.min(lines.len())
        } else {
            lines.len()
        };

        let result = lines[start..end]
            .iter()
            .enumerate()
            .map(|(i, line)| format!("{:>8}  {}", start + i + 1, line))
            .collect::<Vec<_>>()
            .join("\n");

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
    fn test_peek_file_tool_name() {
        let tool = PeekFileTool;
        assert_eq!(tool.name(), "peek_file");
    }

    #[test]
    fn test_peek_file_tool_description() {
        let tool = PeekFileTool;
        assert!(tool.description().contains("range"));
    }

    #[test]
    fn test_peek_file_tool_schema() {
        let tool = PeekFileTool;
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["path"].is_object());
        assert!(schema["properties"]["start_line"].is_object());
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("path")));
        assert!(required.contains(&json!("start_line")));
    }

    #[tokio::test]
    async fn test_peek_with_start_line() {
        let mut file = NamedTempFile::new().unwrap();
        for i in 1..=5 {
            writeln!(file, "line {}", i).unwrap();
        }
        let path = file.path().to_str().unwrap();

        let tool = PeekFileTool;
        let result = tool.execute(json!({"path": path, "start_line": 3})).await.unwrap();
        let contents = result["contents"].as_str().unwrap();
        assert_eq!(contents, "       3  line 3\n       4  line 4\n       5  line 5");
    }

    #[tokio::test]
    async fn test_peek_with_start_and_end_line() {
        let mut file = NamedTempFile::new().unwrap();
        for i in 1..=10 {
            writeln!(file, "line {}", i).unwrap();
        }
        let path = file.path().to_str().unwrap();

        let tool = PeekFileTool;
        let result = tool.execute(json!({"path": path, "start_line": 3, "end_line": 6})).await.unwrap();
        let contents = result["contents"].as_str().unwrap();
        assert_eq!(contents, "       3  line 3\n       4  line 4\n       5  line 5\n       6  line 6");
    }

    #[tokio::test]
    async fn test_peek_start_line_beyond_file() {
        let mut file = NamedTempFile::new().unwrap();
        for i in 1..=5 {
            writeln!(file, "line {}", i).unwrap();
        }
        let path = file.path().to_str().unwrap();

        let tool = PeekFileTool;
        let result = tool.execute(json!({"path": path, "start_line": 100})).await.unwrap();
        let contents = result["contents"].as_str().unwrap();
        assert_eq!(contents, "");
    }

    #[tokio::test]
    async fn test_peek_as_float() {
        let mut file = NamedTempFile::new().unwrap();
        for i in 1..=5 {
            writeln!(file, "line {}", i).unwrap();
        }
        let path = file.path().to_str().unwrap();

        let tool = PeekFileTool;
        let result = tool.execute(json!({"path": path, "start_line": 3.0, "end_line": 4.0})).await.unwrap();
        let contents = result["contents"].as_str().unwrap();
        assert_eq!(contents, "       3  line 3\n       4  line 4");
    }

    #[tokio::test]
    async fn test_peek_line_numbers_format() {
        let mut file = NamedTempFile::new().unwrap();
        for i in 1..=100 {
            writeln!(file, "content {}", i).unwrap();
        }
        let path = file.path().to_str().unwrap();

        let tool = PeekFileTool;
        let result = tool.execute(json!({"path": path, "start_line": 98, "end_line": 100})).await.unwrap();
        let contents = result["contents"].as_str().unwrap();
        assert_eq!(contents, "      98  content 98\n      99  content 99\n     100  content 100");
    }

    #[tokio::test]
    async fn test_peek_missing_start_line() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "hello").unwrap();
        let path = file.path().to_str().unwrap();

        let tool = PeekFileTool;
        let result = tool.execute(json!({"path": path})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_peek_missing_file() {
        let tool = PeekFileTool;
        let result = tool
            .execute(json!({"path": "/nonexistent/file/path/abc123.txt", "start_line": 1}))
            .await;
        assert!(matches!(result, Err(ToolError::ExecutionFailed(_))));
    }
}
