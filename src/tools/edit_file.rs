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

fn format_diff_summary(added: usize, removed: usize) -> String {
    let a = if added == 1 {
        "Added 1 line".to_string()
    } else {
        format!("Added {} lines", added)
    };
    let r = if removed == 1 {
        "removed 1 line".to_string()
    } else {
        format!("removed {} lines", removed)
    };
    format!("{}, {}", a, r)
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
            let total_lines = content.lines().count();
            return Err(ToolError::ExecutionFailed(format!(
                "old_text not found in '{}' ({} lines). Use read_file to check the file contents.",
                path.display(),
                total_lines,
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

        // Build unified diff around each replacement site
        let context_lines = 3;
        let old_lines_all: Vec<&str> = content.lines().collect();
        let old_text_lines: Vec<&str> = old_text.lines().collect();
        let new_text_lines: Vec<&str> = new_text.lines().collect();
        let old_count = old_text_lines.len();
        let new_count = new_text_lines.len();
        let line_shift = new_count as isize - old_count as isize;
        let mut diff_snippets = Vec::new();
        let mut total_added = 0usize;
        let mut total_removed = 0usize;
        let mut cumulative_shift: isize = 0;

        let mut search_from = 0;
        let match_limit = if replace_all { match_count } else { 1 };
        for _ in 0..match_limit {
            if let Some(byte_pos) = content[search_from..].find(old_text) {
                let abs_byte_pos = search_from + byte_pos;
                let start_line = content[..abs_byte_pos].matches('\n').count();

                let ctx_start = start_line.saturating_sub(context_lines);
                let ctx_end =
                    (start_line + old_count + context_lines).min(old_lines_all.len());
                let new_start = start_line as isize + cumulative_shift;

                let mut snippet = String::new();
                // Context lines before
                for i in ctx_start..start_line {
                    let num = (i as isize + 1 + cumulative_shift) as usize;
                    snippet.push_str(&format!("{:>8}  {}\n", num, old_lines_all[i]));
                }
                // Old lines (removed)
                for i in 0..old_count {
                    let line_idx = start_line + i;
                    if line_idx < old_lines_all.len() {
                        let num = (new_start + i as isize + 1) as usize;
                        snippet.push_str(&format!("{:>8} -{}\n", num, old_lines_all[line_idx]));
                    }
                }
                total_removed += old_count;
                // New lines (added)
                for (i, line) in new_text_lines.iter().enumerate() {
                    let num = (new_start + i as isize + 1) as usize;
                    snippet.push_str(&format!("{:>8} +{}\n", num, line));
                }
                total_added += new_count;
                // Context lines after
                cumulative_shift += line_shift;
                let after_start = start_line + old_count;
                for i in after_start..ctx_end {
                    let num = (i as isize + 1 + cumulative_shift) as usize;
                    snippet.push_str(&format!("{:>8}  {}\n", num, old_lines_all[i]));
                }

                diff_snippets.push(snippet.trim_end().to_string());
                search_from = abs_byte_pos + old_text.len();
            }
        }

        let diff_str = diff_snippets.join("\n...\n");
        let summary = format_diff_summary(total_added, total_removed);
        let message = format!("{} in '{}'\n{}", summary, path.display(), diff_str);

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
        if let Err(ToolError::ExecutionFailed(msg)) = result {
            assert!(msg.contains("1 lines"), "Error should show line count");
            assert!(!msg.contains("hello world"), "Error should not include file contents");
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
        assert!(msg.contains("Added 1 line, removed 1 line"), "Should include summary");
        assert!(msg.contains(" +replaced"), "Should show added line");
        assert!(msg.contains(" -target"), "Should show removed line");
        assert!(msg.contains("line2") || msg.contains("line3"), "Should show surrounding lines");
    }

    #[tokio::test]
    async fn test_not_found_shows_line_count() {
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
            assert!(msg.contains("60 lines"), "Should show line count");
            assert!(msg.contains("read_file"), "Should suggest read_file");
            assert!(!msg.contains("line 1\n"), "Should not dump file contents");
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
