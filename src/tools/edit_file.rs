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

/// Tool for replacing exact text in a file. Faster than write_file for small edits.
#[derive(Debug, Default)]
pub struct EditFileTool;

#[async_trait]
impl Tool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Replace exact text in a file. Faster than write_file for small edits."
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to edit"
                },
                "old_text": {
                    "type": "string",
                    "description": "The exact text to find and replace"
                },
                "new_text": {
                    "type": "string",
                    "description": "The replacement text"
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences (default: false, requires unique match)"
                }
            },
            "required": ["path", "old_text", "new_text"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let path_str = input.get("path").and_then(|v| v.as_str()).ok_or_else(|| {
            ToolError::InvalidInput("Missing or invalid 'path' field".to_string())
        })?;

        let old_text = input
            .get("old_text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ToolError::InvalidInput("Missing or invalid 'old_text' field".to_string())
            })?;

        let new_text = input
            .get("new_text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ToolError::InvalidInput("Missing or invalid 'new_text' field".to_string())
            })?;

        let replace_all = input
            .get("replace_all")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let path = expand_tilde(path_str);

        let content = tokio::fs::read_to_string(&path).await.map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to read file '{}': {}", path.display(), e))
        })?;

        let match_count = content.matches(old_text).count();

        if match_count == 0 {
            // Include first 50 lines so the agent can see what's actually in the file
            let preview: String = content
                .lines()
                .take(50)
                .enumerate()
                .map(|(i, line)| format!("{:>4}| {}", i + 1, line))
                .collect::<Vec<_>>()
                .join("\n");
            let total_lines = content.lines().count();
            let shown = total_lines.min(50);
            return Err(ToolError::ExecutionFailed(format!(
                "old_text not found in '{}'. File contents ({}/{} lines):\n{}",
                path.display(),
                shown,
                total_lines,
                preview
            )));
        }

        if !replace_all && match_count > 1 {
            return Err(ToolError::ExecutionFailed(format!(
                "old_text found {} times in '{}'. Include more surrounding context to make the match unique, or set replace_all to true.",
                match_count,
                path.display()
            )));
        }

        let new_content = if replace_all {
            content.replace(old_text, new_text)
        } else {
            content.replacen(old_text, new_text, 1)
        };

        tokio::fs::write(&path, &new_content).await.map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to write file '{}': {}", path.display(), e))
        })?;

        // Build context around the replacement site(s)
        let context_lines = 3;
        let new_lines: Vec<&str> = new_content.lines().collect();
        let new_text_lines: Vec<&str> = new_text.lines().collect();
        let mut context_snippets = Vec::new();

        for (i, line) in new_lines.iter().enumerate() {
            // Find lines that contain the first line of new_text
            if let Some(first_new_line) = new_text_lines.first() {
                if line.contains(first_new_line) {
                    let start = i.saturating_sub(context_lines);
                    let end = (i + new_text_lines.len() + context_lines).min(new_lines.len());
                    let snippet: String = new_lines[start..end]
                        .iter()
                        .enumerate()
                        .map(|(j, l)| format!("{:>4}| {}", start + j + 1, l))
                        .collect::<Vec<_>>()
                        .join("\n");
                    context_snippets.push(snippet);
                    if !replace_all {
                        break;
                    }
                }
            }
        }

        let context_str = context_snippets.join("\n---\n");
        let message = format!(
            "Replaced {} occurrence(s) in '{}'. Context:\n{}",
            match_count,
            path.display(),
            context_str
        );

        Ok(serde_json::json!({
            "success": true,
            "message": message,
            "replacements": match_count
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn test_edit_file_tool_name() {
        let tool = EditFileTool;
        assert_eq!(tool.name(), "edit_file");
    }

    #[test]
    fn test_edit_file_tool_schema() {
        let tool = EditFileTool;
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["path"].is_object());
        assert!(schema["properties"]["old_text"].is_object());
        assert!(schema["properties"]["new_text"].is_object());
        assert!(schema["properties"]["replace_all"].is_object());
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("path")));
        assert!(required.contains(&json!("old_text")));
        assert!(required.contains(&json!("new_text")));
        assert!(!required.contains(&json!("replace_all")));
    }

    #[tokio::test]
    async fn test_basic_replacement() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "hello world").unwrap();

        let tool = EditFileTool;
        let result = tool
            .execute(json!({
                "path": path.to_str().unwrap(),
                "old_text": "hello",
                "new_text": "goodbye"
            }))
            .await
            .unwrap();

        assert!(result["success"].as_bool().unwrap());
        assert_eq!(result["replacements"], 1);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "goodbye world");
    }

    #[tokio::test]
    async fn test_replace_all() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "aaa bbb aaa ccc aaa").unwrap();

        let tool = EditFileTool;
        let result = tool
            .execute(json!({
                "path": path.to_str().unwrap(),
                "old_text": "aaa",
                "new_text": "xxx",
                "replace_all": true
            }))
            .await
            .unwrap();

        assert!(result["success"].as_bool().unwrap());
        assert_eq!(result["replacements"], 3);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "xxx bbb xxx ccc xxx");
    }

    #[tokio::test]
    async fn test_old_text_not_found() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "hello world").unwrap();

        let tool = EditFileTool;
        let result = tool
            .execute(json!({
                "path": path.to_str().unwrap(),
                "old_text": "nonexistent",
                "new_text": "replacement"
            }))
            .await;

        assert!(matches!(result, Err(ToolError::ExecutionFailed(ref msg)) if msg.contains("not found")));
        // Should include file preview
        if let Err(ToolError::ExecutionFailed(msg)) = result {
            assert!(msg.contains("hello world"), "Error should include file contents");
            assert!(msg.contains("1/1 lines"), "Error should show line count");
        }
    }

    #[tokio::test]
    async fn test_old_text_not_unique() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "foo bar foo baz").unwrap();

        let tool = EditFileTool;
        let result = tool
            .execute(json!({
                "path": path.to_str().unwrap(),
                "old_text": "foo",
                "new_text": "qux"
            }))
            .await;

        assert!(
            matches!(result, Err(ToolError::ExecutionFailed(msg)) if msg.contains("2 times"))
        );
        // File should be unchanged
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "foo bar foo baz");
    }

    #[tokio::test]
    async fn test_missing_required_fields() {
        let tool = EditFileTool;

        let r = tool.execute(json!({"old_text": "a", "new_text": "b"})).await;
        assert!(matches!(r, Err(ToolError::InvalidInput(_))));

        let r = tool.execute(json!({"path": "/tmp/x", "new_text": "b"})).await;
        assert!(matches!(r, Err(ToolError::InvalidInput(_))));

        let r = tool.execute(json!({"path": "/tmp/x", "old_text": "a"})).await;
        assert!(matches!(r, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_multiline_replacement() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "line1\nline2\nline3\n").unwrap();

        let tool = EditFileTool;
        let result = tool
            .execute(json!({
                "path": path.to_str().unwrap(),
                "old_text": "line2\nline3",
                "new_text": "replaced2\nreplaced3\nreplaced4"
            }))
            .await
            .unwrap();

        assert!(result["success"].as_bool().unwrap());
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "line1\nreplaced2\nreplaced3\nreplaced4\n"
        );
    }

    #[tokio::test]
    async fn test_basic_replacement_includes_context() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(
            &path,
            "line1\nline2\nline3\ntarget\nline5\nline6\nline7\n",
        )
        .unwrap();

        let tool = EditFileTool;
        let result = tool
            .execute(json!({
                "path": path.to_str().unwrap(),
                "old_text": "target",
                "new_text": "replaced"
            }))
            .await
            .unwrap();

        let msg = result["message"].as_str().unwrap();
        assert!(msg.contains("Context:"), "Should include context header");
        assert!(msg.contains("replaced"), "Should show the new text");
        assert!(msg.contains("line2") || msg.contains("line3"), "Should show surrounding lines");
    }

    #[tokio::test]
    async fn test_not_found_shows_file_preview() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.txt");
        let content: String = (1..=60).map(|i| format!("line {}\n", i)).collect();
        std::fs::write(&path, &content).unwrap();

        let tool = EditFileTool;
        let result = tool
            .execute(json!({
                "path": path.to_str().unwrap(),
                "old_text": "nonexistent text",
                "new_text": "replacement"
            }))
            .await;

        if let Err(ToolError::ExecutionFailed(msg)) = result {
            assert!(msg.contains("50/60 lines"), "Should show 50 of 60 lines");
            assert!(msg.contains("line 1"), "Should include first line");
            assert!(msg.contains("line 50"), "Should include line 50");
            assert!(!msg.contains("line 51"), "Should not include line 51");
        } else {
            panic!("Expected ExecutionFailed error");
        }
    }

    #[tokio::test]
    async fn test_file_does_not_exist() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonexistent.txt");

        let tool = EditFileTool;
        let result = tool
            .execute(json!({
                "path": path.to_str().unwrap(),
                "old_text": "hello",
                "new_text": "goodbye"
            }))
            .await;

        assert!(matches!(result, Err(ToolError::ExecutionFailed(msg)) if msg.contains("Failed to read")));
    }
}
