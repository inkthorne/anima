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

/// Tool for writing content to a file on the filesystem.
#[derive(Debug, Default)]
pub struct WriteFileTool;

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Writes content to a file at the given path, creating parent directories if needed"
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file"
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let path_str = input.get("path").and_then(|v| v.as_str()).ok_or_else(|| {
            ToolError::InvalidInput("Missing or invalid 'path' field".to_string())
        })?;

        let content = input
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ToolError::InvalidInput("Missing or invalid 'content' field".to_string())
            })?;

        let path = expand_tilde(path_str);

        let existed = path.exists();
        let old_line_count = if existed {
            tokio::fs::read_to_string(&path)
                .await
                .ok()
                .map(|s| s.lines().count())
        } else {
            None
        };

        // Create parent directories if they don't exist
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                ToolError::ExecutionFailed(format!("Failed to create parent directories: {}", e))
            })?;
        }

        tokio::fs::write(&path, content).await.map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to write file '{}': {}", path.display(), e))
        })?;

        let line_count = content.lines().count();
        let byte_len = content.len();
        let size_str = if byte_len >= 1024 {
            format!("{:.1} KB", byte_len as f64 / 1024.0)
        } else {
            format!("{} bytes", byte_len)
        };
        let mut message = format!(
            "Wrote {} lines ({}) to '{}'",
            line_count,
            size_str,
            path.display()
        );

        if let Some(old_lines) = old_line_count {
            message.push_str(&format!(
                "\nNote: overwrote existing file ({} lines). For targeted changes, prefer edit_file.",
                old_lines
            ));
        }

        Ok(serde_json::json!({ "success": true, "message": message }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::{NamedTempFile, tempdir};

    #[test]
    fn test_write_file_tool_name() {
        let tool = WriteFileTool;
        assert_eq!(tool.name(), "write_file");
    }

    #[test]
    fn test_write_file_tool_description() {
        let tool = WriteFileTool;
        assert!(tool.description().contains("file"));
    }

    #[test]
    fn test_write_file_tool_schema() {
        let tool = WriteFileTool;
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["path"].is_object());
        assert!(schema["properties"]["content"].is_object());
        assert!(
            schema["required"]
                .as_array()
                .unwrap()
                .contains(&json!("path"))
        );
        assert!(
            schema["required"]
                .as_array()
                .unwrap()
                .contains(&json!("content"))
        );
    }

    #[tokio::test]
    async fn test_write_new_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.txt");
        let path_str = path.to_str().unwrap();

        let tool = WriteFileTool;
        let result = tool
            .execute(json!({"path": path_str, "content": "hello world"}))
            .await
            .unwrap();
        assert!(result["success"].as_bool().unwrap());
        let msg = result["message"].as_str().unwrap();
        assert!(msg.contains("1 lines"), "Should report line count");
        assert!(msg.contains("11 bytes"), "Should report byte size");

        let contents = std::fs::read_to_string(&path).unwrap();
        assert_eq!(contents, "hello world");
    }

    #[tokio::test]
    async fn test_overwrite_existing_file() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap();

        let tool = WriteFileTool;
        tool.execute(json!({"path": path, "content": "original"}))
            .await
            .unwrap();
        let result = tool
            .execute(json!({"path": path, "content": "updated"}))
            .await
            .unwrap();

        let contents = std::fs::read_to_string(file.path()).unwrap();
        assert_eq!(contents, "updated");

        let msg = result["message"].as_str().unwrap();
        assert!(
            msg.contains("overwrote existing file"),
            "Should warn about overwriting"
        );
        assert!(
            msg.contains("edit_file"),
            "Should suggest edit_file"
        );
    }

    #[tokio::test]
    async fn test_new_file_no_overwrite_warning() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("brand_new.txt");
        let path_str = path.to_str().unwrap();

        let tool = WriteFileTool;
        let result = tool
            .execute(json!({"path": path_str, "content": "fresh content"}))
            .await
            .unwrap();

        let msg = result["message"].as_str().unwrap();
        assert!(
            !msg.contains("overwrote"),
            "New file should not have overwrite warning"
        );
    }

    #[tokio::test]
    async fn test_write_creates_parent_dirs() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nested/deep/dir/file.txt");
        let path_str = path.to_str().unwrap();

        let tool = WriteFileTool;
        let result = tool
            .execute(json!({"path": path_str, "content": "nested content"}))
            .await
            .unwrap();
        assert!(result["success"].as_bool().unwrap());

        let contents = std::fs::read_to_string(&path).unwrap();
        assert_eq!(contents, "nested content");
    }

    #[tokio::test]
    async fn test_write_missing_path_field() {
        let tool = WriteFileTool;
        let result = tool.execute(json!({"content": "hello"})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_write_missing_content_field() {
        let tool = WriteFileTool;
        let result = tool.execute(json!({"path": "/tmp/test.txt"})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_write_reports_kb_for_large_content() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("large.txt");
        let path_str = path.to_str().unwrap();
        let content: String = (1..=100).map(|i| format!("line number {}\n", i)).collect();

        let tool = WriteFileTool;
        let result = tool
            .execute(json!({"path": path_str, "content": content}))
            .await
            .unwrap();
        let msg = result["message"].as_str().unwrap();
        assert!(msg.contains("100 lines"), "Should report 100 lines");
        assert!(msg.contains("KB"), "Should use KB for larger files");
    }

    #[tokio::test]
    async fn test_write_empty_content() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.txt");
        let path_str = path.to_str().unwrap();

        let tool = WriteFileTool;
        let result = tool
            .execute(json!({"path": path_str, "content": ""}))
            .await
            .unwrap();
        assert!(result["success"].as_bool().unwrap());

        let contents = std::fs::read_to_string(&path).unwrap();
        assert_eq!(contents, "");
    }
}
