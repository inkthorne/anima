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

/// Find the best near-miss match for old_text in the file.
/// Returns (start_line_index_0based, matched_count, total_old_lines) if above threshold.
fn find_near_miss(file_lines: &[&str], old_lines: &[&str]) -> Option<(usize, usize, usize)> {
    let n = old_lines.len();
    if n == 0 || file_lines.is_empty() {
        return None;
    }
    let mut best_matches = 0usize;
    let mut best_pos = 0usize;
    for start in 0..file_lines.len() {
        let window_end = (start + n).min(file_lines.len());
        let mut matches = 0usize;
        for i in 0..(window_end - start) {
            if file_lines[start + i].trim() == old_lines[i].trim() {
                matches += 1;
            }
        }
        if matches > best_matches {
            best_matches = matches;
            best_pos = start;
        }
    }
    let threshold = ((n as f64 * 0.3).ceil() as usize).max(1);
    if best_matches >= threshold {
        Some((best_pos, best_matches, n))
    } else {
        None
    }
}

fn format_near_miss(
    file_lines: &[&str],
    match_start: usize,
    old_line_count: usize,
    matched_count: usize,
    context: usize,
) -> String {
    let match_end = (match_start + old_line_count).min(file_lines.len());
    let display_start = match_start.saturating_sub(context);
    let display_end = (match_end + context).min(file_lines.len());
    let pct = (matched_count as f64 / old_line_count as f64 * 100.0) as usize;
    let mut out = format!(
        "Closest match ({}/{} lines, {}%) at lines {}-{}:\n\n",
        matched_count,
        old_line_count,
        pct,
        match_start + 1,
        match_end,
    );
    for i in display_start..display_end {
        let marker = if i >= match_start && i < match_end {
            "|"
        } else {
            " "
        };
        out.push_str(&format!("{:>8} {} {}\n", i + 1, marker, file_lines[i]));
    }
    out
}

/// Format lines around a known match position with context.
fn format_context_around(
    file_lines: &[&str],
    start_line: usize,
    line_count: usize,
    context: usize,
) -> String {
    let match_end = (start_line + line_count).min(file_lines.len());
    let display_start = start_line.saturating_sub(context);
    let display_end = (match_end + context).min(file_lines.len());
    let mut out = format!("Text found at lines {}-{}:\n\n", start_line + 1, match_end);
    for i in display_start..display_end {
        let marker = if i >= start_line && i < match_end {
            "|"
        } else {
            " "
        };
        out.push_str(&format!("{:>8} {} {}\n", i + 1, marker, file_lines[i]));
    }
    out
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
            .and_then(super::json_to_bool)
            .unwrap_or(false);

        let path = expand_tilde(path_str);

        let content = tokio::fs::read_to_string(&path).await.map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to read file '{}': {}", path.display(), e))
        })?;

        // Reject no-op edits — now after file read so we can show context
        if old_text == new_text {
            let file_lines: Vec<&str> = content.lines().collect();
            let old_lines: Vec<&str> = old_text.lines().collect();
            let message = if let Some(byte_pos) = content.find(old_text) {
                let start_line = content[..byte_pos].matches('\n').count();
                let ctx = format_context_around(&file_lines, start_line, old_lines.len(), 5);
                format!(
                    "No edit needed — old_text and new_text are identical. {}",
                    ctx
                )
            } else {
                "No edit needed — old_text and new_text are identical. Text not found in file — the edit may have been applied with different content. Use read_file to check.".to_string()
            };
            return Err(ToolError::InvalidInput(message));
        }

        let match_count = content.matches(old_text).count();

        if match_count == 0 {
            let file_lines: Vec<&str> = content.lines().collect();
            let old_lines: Vec<&str> = old_text.lines().collect();
            let total_lines = file_lines.len();
            let message = if let Some((pos, matched, total)) =
                find_near_miss(&file_lines, &old_lines)
            {
                let near = format_near_miss(&file_lines, pos, total, matched, 5);
                format!(
                    "old_text not found in '{}' ({} lines). {}",
                    path.display(),
                    total_lines,
                    near
                )
            } else {
                format!(
                    "old_text not found in '{}' ({} lines). No similar region found. Use read_file to check the file contents.",
                    path.display(),
                    total_lines,
                )
            };
            return Err(ToolError::ExecutionFailed(message));
        }

        if !replace_all && match_count > 1 {
            // Collect line numbers for the error message
            let mut line_nums = Vec::new();
            let mut search_from = 0;
            for _ in 0..match_count {
                if let Some(pos) = content[search_from..].find(old_text) {
                    let abs = search_from + pos;
                    line_nums.push(content[..abs].matches('\n').count() + 1);
                    search_from = abs + old_text.len();
                }
            }
            let lines_str = line_nums
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(ToolError::ExecutionFailed(format!(
                "old_text found {} times in '{}' (lines {}). Include more surrounding context to make the match unique, or set replace_all to true.",
                match_count, path.display(), lines_str
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
            assert!(
                msg.contains("No similar region found"),
                "Should say no similar region for unrelated content"
            );
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
    async fn test_not_unique_shows_line_numbers() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "aaa\nbbb\naaa\nccc\nddd\naaa\n").unwrap();

        let tool = EditFileTool;
        let result = tool
            .execute(json!({
                "path": path.to_str().unwrap(),
                "old_text": "aaa",
                "new_text": "zzz"
            }))
            .await;

        if let Err(ToolError::ExecutionFailed(msg)) = result {
            assert!(msg.contains("3 times"), "Should report match count");
            assert!(msg.contains("lines 1, 3, 6"), "Should list line numbers, got: {}", msg);
        } else {
            panic!("Expected ExecutionFailed error");
        }
    }

    #[tokio::test]
    async fn test_replace_all_string_true() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "aaa bbb aaa ccc aaa").unwrap();

        let tool = EditFileTool;
        let result = tool
            .execute(json!({
                "path": path.to_str().unwrap(),
                "old_text": "aaa",
                "new_text": "xxx",
                "replace_all": "True"
            }))
            .await
            .unwrap();

        assert!(result["success"].as_bool().unwrap());
        assert_eq!(result["replacements"], 3);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "xxx bbb xxx ccc xxx");
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
            assert!(
                msg.contains("No similar region found"),
                "Should say no similar region for unrelated content"
            );
            assert!(msg.contains("read_file"), "Should suggest read_file");
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

    #[tokio::test]
    async fn test_noop_edit_rejected() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "line1\nline2\nsame content\nline4\nline5\n").unwrap();

        let tool = EditFileTool;
        let result = tool
            .execute(json!({
                "path": path.to_str().unwrap(),
                "old_text": "same content",
                "new_text": "same content"
            }))
            .await;

        assert!(matches!(result, Err(ToolError::InvalidInput(ref msg)) if msg.contains("identical")));
    }

    #[tokio::test]
    async fn test_noop_edit_shows_context() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(
            &path,
            "aaa\nbbb\nccc\ntarget line\neee\nfff\nggg\n",
        )
        .unwrap();

        let tool = EditFileTool;
        let result = tool
            .execute(json!({
                "path": path.to_str().unwrap(),
                "old_text": "target line",
                "new_text": "target line"
            }))
            .await;

        if let Err(ToolError::InvalidInput(msg)) = result {
            assert!(msg.contains("identical"), "Should mention identical");
            assert!(msg.contains("Text found at lines"), "Should show location, got: {}", msg);
            assert!(msg.contains("target line"), "Should show the matched text, got: {}", msg);
            assert!(msg.contains("bbb") || msg.contains("ccc"), "Should show context above, got: {}", msg);
            assert!(msg.contains("eee") || msg.contains("fff"), "Should show context below, got: {}", msg);
        } else {
            panic!("Expected InvalidInput error");
        }
    }

    #[tokio::test]
    async fn test_noop_edit_text_not_in_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "completely different content\n").unwrap();

        let tool = EditFileTool;
        let result = tool
            .execute(json!({
                "path": path.to_str().unwrap(),
                "old_text": "not in file",
                "new_text": "not in file"
            }))
            .await;

        if let Err(ToolError::InvalidInput(msg)) = result {
            assert!(msg.contains("identical"), "Should mention identical");
            assert!(msg.contains("Text not found in file"), "Should say text not found, got: {}", msg);
            assert!(msg.contains("read_file"), "Should suggest read_file, got: {}", msg);
        } else {
            panic!("Expected InvalidInput error");
        }
    }

    #[test]
    fn test_find_near_miss_one_line_changed() {
        let file_lines = vec!["aaa", "bbb", "ccc", "ddd", "eee"];
        // old_text has 1 line different (CHANGED instead of ccc)
        let old_lines = vec!["aaa", "bbb", "CHANGED", "ddd", "eee"];
        let result = find_near_miss(&file_lines, &old_lines);
        assert!(result.is_some());
        let (pos, matched, total) = result.unwrap();
        assert_eq!(pos, 0);
        assert_eq!(matched, 4);
        assert_eq!(total, 5);
    }

    #[test]
    fn test_find_near_miss_no_match() {
        let file_lines = vec!["aaa", "bbb", "ccc"];
        let old_lines = vec!["xxx", "yyy", "zzz"];
        let result = find_near_miss(&file_lines, &old_lines);
        assert!(result.is_none());
    }

    #[test]
    fn test_find_near_miss_whitespace_diff() {
        let file_lines = vec!["  aaa", "    bbb", "  ccc"];
        // Same content with different indentation
        let old_lines = vec!["aaa", "  bbb", "    ccc"];
        let result = find_near_miss(&file_lines, &old_lines);
        assert!(result.is_some());
        let (pos, matched, total) = result.unwrap();
        assert_eq!(pos, 0);
        assert_eq!(matched, 3, "Trimmed comparison should match all lines");
        assert_eq!(total, 3);
    }

    #[tokio::test]
    async fn test_near_miss_shown_in_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "line1\nline2\nline3\nline4\nline5\n").unwrap();

        let tool = EditFileTool;
        // old_text matches 4/5 lines — only line3 is different
        let result = tool
            .execute(json!({
                "path": path.to_str().unwrap(),
                "old_text": "line1\nline2\nCHANGED\nline4\nline5",
                "new_text": "replacement"
            }))
            .await;

        if let Err(ToolError::ExecutionFailed(msg)) = result {
            assert!(msg.contains("Closest match"), "Should show near-miss, got: {}", msg);
            assert!(msg.contains("4/5 lines"), "Should show match count, got: {}", msg);
        } else {
            panic!("Expected ExecutionFailed error");
        }
    }

    #[tokio::test]
    async fn test_near_miss_not_shown_when_no_match() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "aaa\nbbb\nccc\n").unwrap();

        let tool = EditFileTool;
        let result = tool
            .execute(json!({
                "path": path.to_str().unwrap(),
                "old_text": "xxx\nyyy\nzzz",
                "new_text": "replacement"
            }))
            .await;

        if let Err(ToolError::ExecutionFailed(msg)) = result {
            assert!(
                msg.contains("No similar region found"),
                "Should say no similar region, got: {}",
                msg
            );
            assert!(!msg.contains("Closest match"), "Should not show near-miss");
        } else {
            panic!("Expected ExecutionFailed error");
        }
    }
}
