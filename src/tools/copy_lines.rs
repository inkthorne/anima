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

/// Tool for copying lines from one file to another by line number.
/// Avoids LLM transcription errors with repetitive/symbolic content.
#[derive(Debug, Default)]
pub struct CopyLinesTool;

#[async_trait]
impl Tool for CopyLinesTool {
    fn name(&self) -> &str {
        "copy_lines"
    }

    fn description(&self) -> &str {
        "Copy lines from a source file into a destination file by line number"
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Path to the source file"
                },
                "dest": {
                    "type": "string",
                    "description": "Path to the destination file"
                },
                "start_line": {
                    "type": "integer",
                    "description": "First line to copy (1-based)"
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last line to copy (inclusive, defaults to start_line)"
                },
                "insert_after": {
                    "type": "integer",
                    "description": "Insert copied lines after this line number in dest (0 = prepend)"
                },
                "replace_start": {
                    "type": "integer",
                    "description": "Replace dest lines starting from this line number (1-based)"
                },
                "replace_end": {
                    "type": "integer",
                    "description": "Replace dest lines through this line number (inclusive)"
                }
            },
            "required": ["source", "dest", "start_line"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let source_str = input
            .get("source")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing or invalid 'source' field".to_string()))?;

        let dest_str = input
            .get("dest")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing or invalid 'dest' field".to_string()))?;

        let start_line = input
            .get("start_line")
            .and_then(json_to_usize)
            .ok_or_else(|| ToolError::InvalidInput("Missing or invalid 'start_line' field".to_string()))?;

        if start_line == 0 {
            return Err(ToolError::InvalidInput("start_line must be >= 1".to_string()));
        }

        let end_line = input.get("end_line").and_then(json_to_usize).unwrap_or(start_line);

        if end_line < start_line {
            return Err(ToolError::InvalidInput(format!(
                "end_line ({}) must be >= start_line ({})",
                end_line, start_line
            )));
        }

        let insert_after = input.get("insert_after").and_then(json_to_usize);
        let replace_start = input.get("replace_start").and_then(json_to_usize);
        let replace_end = input.get("replace_end").and_then(json_to_usize);

        // Validate mode: insert vs replace (mutually exclusive)
        let mode = match (insert_after, replace_start, replace_end) {
            (Some(pos), None, None) => Mode::Insert(pos),
            (None, Some(rs), Some(re)) => {
                if rs == 0 {
                    return Err(ToolError::InvalidInput("replace_start must be >= 1".to_string()));
                }
                if re < rs {
                    return Err(ToolError::InvalidInput(format!(
                        "replace_end ({}) must be >= replace_start ({})",
                        re, rs
                    )));
                }
                Mode::Replace(rs, re)
            }
            (None, Some(_), None) | (None, None, Some(_)) => {
                return Err(ToolError::InvalidInput(
                    "Both replace_start and replace_end are required for replace mode".to_string(),
                ));
            }
            (Some(_), Some(_), _) | (Some(_), _, Some(_)) => {
                return Err(ToolError::InvalidInput(
                    "Cannot use insert_after together with replace_start/replace_end".to_string(),
                ));
            }
            (None, None, None) => {
                return Err(ToolError::InvalidInput(
                    "Must specify either insert_after or replace_start+replace_end".to_string(),
                ));
            }
        };

        // Read source file
        let source_path = expand_tilde(source_str);
        let source_contents = tokio::fs::read_to_string(&source_path).await.map_err(|e| {
            ToolError::ExecutionFailed(format!(
                "Failed to read source '{}': {}",
                source_path.display(),
                e
            ))
        })?;

        let source_lines: Vec<&str> = source_contents.lines().collect();

        // Validate source line range
        if start_line > source_lines.len() {
            return Err(ToolError::InvalidInput(format!(
                "start_line ({}) exceeds source file length ({} lines)",
                start_line,
                source_lines.len()
            )));
        }
        let clamped_end = end_line.min(source_lines.len());
        let copied = &source_lines[start_line - 1..clamped_end];
        let lines_copied = copied.len();

        // Read dest file
        let dest_path = expand_tilde(dest_str);
        let dest_contents = tokio::fs::read_to_string(&dest_path).await.map_err(|e| {
            ToolError::ExecutionFailed(format!(
                "Failed to read dest '{}': {}",
                dest_path.display(),
                e
            ))
        })?;

        let mut dest_lines: Vec<&str> = dest_contents.lines().collect();
        // Preserve trailing newline state
        let had_trailing_newline = dest_contents.ends_with('\n');

        match mode {
            Mode::Insert(pos) => {
                let insert_at = pos.min(dest_lines.len());
                for (i, line) in copied.iter().enumerate() {
                    dest_lines.insert(insert_at + i, line);
                }
            }
            Mode::Replace(rs, re) => {
                if rs > dest_lines.len() {
                    return Err(ToolError::InvalidInput(format!(
                        "replace_start ({}) exceeds dest file length ({} lines)",
                        rs,
                        dest_lines.len()
                    )));
                }
                let clamped_re = re.min(dest_lines.len());
                // Remove the range, then insert copied lines at that position
                dest_lines.splice(rs - 1..clamped_re, copied.iter().copied());
            }
        }

        // Rebuild file content
        let mut new_contents = dest_lines.join("\n");
        if had_trailing_newline || dest_contents.is_empty() {
            new_contents.push('\n');
        }

        tokio::fs::write(&dest_path, &new_contents).await.map_err(|e| {
            ToolError::ExecutionFailed(format!(
                "Failed to write dest '{}': {}",
                dest_path.display(),
                e
            ))
        })?;

        let message = format!(
            "Copied {} lines from '{}' to '{}'",
            lines_copied,
            source_path.display(),
            dest_path.display()
        );

        Ok(serde_json::json!({
            "message": message,
            "lines_copied": lines_copied,
        }))
    }
}

enum Mode {
    Insert(usize),
    Replace(usize, usize),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn make_file(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        write!(f, "{}", content).unwrap();
        f
    }

    #[test]
    fn test_copy_lines_tool_name() {
        let tool = CopyLinesTool;
        assert_eq!(tool.name(), "copy_lines");
    }

    #[test]
    fn test_copy_lines_tool_schema() {
        let tool = CopyLinesTool;
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("source")));
        assert!(required.contains(&json!("dest")));
        assert!(required.contains(&json!("start_line")));
    }

    #[tokio::test]
    async fn test_insert_after_line() {
        let src = make_file("alpha\nbeta\ngamma\n");
        let dest = make_file("line1\nline2\nline3\n");

        let tool = CopyLinesTool;
        let result = tool
            .execute(json!({
                "source": src.path().to_str().unwrap(),
                "dest": dest.path().to_str().unwrap(),
                "start_line": 2,
                "end_line": 3,
                "insert_after": 1,
            }))
            .await
            .unwrap();

        assert_eq!(result["lines_copied"], 2);
        let contents = std::fs::read_to_string(dest.path()).unwrap();
        assert_eq!(contents, "line1\nbeta\ngamma\nline2\nline3\n");
    }

    #[tokio::test]
    async fn test_insert_prepend() {
        let src = make_file("header\n");
        let dest = make_file("body\n");

        let tool = CopyLinesTool;
        tool.execute(json!({
            "source": src.path().to_str().unwrap(),
            "dest": dest.path().to_str().unwrap(),
            "start_line": 1,
            "insert_after": 0,
        }))
        .await
        .unwrap();

        let contents = std::fs::read_to_string(dest.path()).unwrap();
        assert_eq!(contents, "header\nbody\n");
    }

    #[tokio::test]
    async fn test_insert_append() {
        let src = make_file("footer\n");
        let dest = make_file("line1\nline2\n");

        let tool = CopyLinesTool;
        tool.execute(json!({
            "source": src.path().to_str().unwrap(),
            "dest": dest.path().to_str().unwrap(),
            "start_line": 1,
            "insert_after": 2,
        }))
        .await
        .unwrap();

        let contents = std::fs::read_to_string(dest.path()).unwrap();
        assert_eq!(contents, "line1\nline2\nfooter\n");
    }

    #[tokio::test]
    async fn test_replace_lines() {
        let src = make_file("new1\nnew2\n");
        let dest = make_file("a\nb\nc\nd\n");

        let tool = CopyLinesTool;
        let result = tool
            .execute(json!({
                "source": src.path().to_str().unwrap(),
                "dest": dest.path().to_str().unwrap(),
                "start_line": 1,
                "end_line": 2,
                "replace_start": 2,
                "replace_end": 3,
            }))
            .await
            .unwrap();

        assert_eq!(result["lines_copied"], 2);
        let contents = std::fs::read_to_string(dest.path()).unwrap();
        assert_eq!(contents, "a\nnew1\nnew2\nd\n");
    }

    #[tokio::test]
    async fn test_single_line_default_end() {
        let src = make_file("one\ntwo\nthree\n");
        let dest = make_file("x\ny\n");

        let tool = CopyLinesTool;
        let result = tool
            .execute(json!({
                "source": src.path().to_str().unwrap(),
                "dest": dest.path().to_str().unwrap(),
                "start_line": 2,
                "insert_after": 1,
            }))
            .await
            .unwrap();

        assert_eq!(result["lines_copied"], 1);
        let contents = std::fs::read_to_string(dest.path()).unwrap();
        assert_eq!(contents, "x\ntwo\ny\n");
    }

    #[tokio::test]
    async fn test_replace_expand() {
        // Replace 1 line with 3 lines
        let src = make_file("r1\nr2\nr3\n");
        let dest = make_file("a\nb\nc\n");

        let tool = CopyLinesTool;
        tool.execute(json!({
            "source": src.path().to_str().unwrap(),
            "dest": dest.path().to_str().unwrap(),
            "start_line": 1,
            "end_line": 3,
            "replace_start": 2,
            "replace_end": 2,
        }))
        .await
        .unwrap();

        let contents = std::fs::read_to_string(dest.path()).unwrap();
        assert_eq!(contents, "a\nr1\nr2\nr3\nc\n");
    }

    #[tokio::test]
    async fn test_replace_shrink() {
        // Replace 3 lines with 1 line
        let src = make_file("only\n");
        let dest = make_file("a\nb\nc\nd\n");

        let tool = CopyLinesTool;
        tool.execute(json!({
            "source": src.path().to_str().unwrap(),
            "dest": dest.path().to_str().unwrap(),
            "start_line": 1,
            "end_line": 1,
            "replace_start": 1,
            "replace_end": 3,
        }))
        .await
        .unwrap();

        let contents = std::fs::read_to_string(dest.path()).unwrap();
        assert_eq!(contents, "only\nd\n");
    }

    #[tokio::test]
    async fn test_error_no_mode() {
        let src = make_file("x\n");
        let dest = make_file("y\n");

        let tool = CopyLinesTool;
        let result = tool
            .execute(json!({
                "source": src.path().to_str().unwrap(),
                "dest": dest.path().to_str().unwrap(),
                "start_line": 1,
            }))
            .await;

        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_error_conflicting_modes() {
        let src = make_file("x\n");
        let dest = make_file("y\n");

        let tool = CopyLinesTool;
        let result = tool
            .execute(json!({
                "source": src.path().to_str().unwrap(),
                "dest": dest.path().to_str().unwrap(),
                "start_line": 1,
                "insert_after": 0,
                "replace_start": 1,
                "replace_end": 1,
            }))
            .await;

        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_error_end_before_start() {
        let src = make_file("x\ny\n");
        let dest = make_file("a\n");

        let tool = CopyLinesTool;
        let result = tool
            .execute(json!({
                "source": src.path().to_str().unwrap(),
                "dest": dest.path().to_str().unwrap(),
                "start_line": 3,
                "end_line": 1,
                "insert_after": 0,
            }))
            .await;

        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_error_start_line_zero() {
        let src = make_file("x\n");
        let dest = make_file("y\n");

        let tool = CopyLinesTool;
        let result = tool
            .execute(json!({
                "source": src.path().to_str().unwrap(),
                "dest": dest.path().to_str().unwrap(),
                "start_line": 0,
                "insert_after": 0,
            }))
            .await;

        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_error_start_line_beyond_source() {
        let src = make_file("one\ntwo\n");
        let dest = make_file("a\n");

        let tool = CopyLinesTool;
        let result = tool
            .execute(json!({
                "source": src.path().to_str().unwrap(),
                "dest": dest.path().to_str().unwrap(),
                "start_line": 10,
                "insert_after": 0,
            }))
            .await;

        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_error_missing_source() {
        let dest = make_file("a\n");

        let tool = CopyLinesTool;
        let result = tool
            .execute(json!({
                "source": "/nonexistent/path/abc.txt",
                "dest": dest.path().to_str().unwrap(),
                "start_line": 1,
                "insert_after": 0,
            }))
            .await;

        assert!(matches!(result, Err(ToolError::ExecutionFailed(_))));
    }

    #[tokio::test]
    async fn test_float_line_numbers() {
        let src = make_file("alpha\nbeta\ngamma\n");
        let dest = make_file("x\ny\n");

        let tool = CopyLinesTool;
        let result = tool
            .execute(json!({
                "source": src.path().to_str().unwrap(),
                "dest": dest.path().to_str().unwrap(),
                "start_line": 1.0,
                "end_line": 2.0,
                "insert_after": 1.0,
            }))
            .await
            .unwrap();

        assert_eq!(result["lines_copied"], 2);
        let contents = std::fs::read_to_string(dest.path()).unwrap();
        assert_eq!(contents, "x\nalpha\nbeta\ny\n");
    }

    #[tokio::test]
    async fn test_error_partial_replace_params() {
        let src = make_file("x\n");
        let dest = make_file("y\n");

        let tool = CopyLinesTool;
        let result = tool
            .execute(json!({
                "source": src.path().to_str().unwrap(),
                "dest": dest.path().to_str().unwrap(),
                "start_line": 1,
                "replace_start": 1,
            }))
            .await;

        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }
}
