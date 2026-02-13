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

/// Tool for listing files in a directory.
#[derive(Debug, Default)]
pub struct ListFilesTool;

#[async_trait]
impl Tool for ListFilesTool {
    fn name(&self) -> &str {
        "list_files"
    }

    fn description(&self) -> &str {
        "List files and directories at the given path, optionally recursive"
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The directory path to list"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "List files recursively (default: false)"
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum depth for recursive listing (default: 3)"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let path_str = input.get("path").and_then(|v| v.as_str()).ok_or_else(|| {
            ToolError::InvalidInput("Missing or invalid 'path' field".to_string())
        })?;

        let recursive = input
            .get("recursive")
            .and_then(super::json_to_bool)
            .unwrap_or(false);

        let max_depth = input
            .get("max_depth")
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as usize;

        let path = expand_tilde(path_str);

        if !path.is_dir() {
            return Err(ToolError::ExecutionFailed(format!(
                "'{}' is not a directory",
                path.display()
            )));
        }

        let mut entries = Vec::new();
        list_dir(&path, &path, recursive, max_depth, 0, &mut entries).map_err(|e| {
            ToolError::ExecutionFailed(format!(
                "Failed to list directory '{}': {}",
                path.display(),
                e
            ))
        })?;

        let total = entries.len();

        Ok(serde_json::json!({
            "success": true,
            "entries": total,
            "listing": entries
        }))
    }
}

fn list_dir(
    base: &std::path::Path,
    dir: &std::path::Path,
    recursive: bool,
    max_depth: usize,
    current_depth: usize,
    entries: &mut Vec<Value>,
) -> std::io::Result<()> {
    let mut items: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .collect();
    items.sort_by_key(|e| e.file_name());

    for entry in items {
        let path = entry.path();
        let relative = path.strip_prefix(base).unwrap_or(&path);
        let is_dir = path.is_dir();
        let suffix = if is_dir { "/" } else { "" };
        let name = format!("{}{}", relative.display(), suffix);

        if is_dir {
            entries.push(serde_json::json!({"name": name, "type": "dir", "size": null}));
        } else {
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            entries.push(serde_json::json!({"name": name, "type": "file", "size": size}));
        }

        if is_dir && recursive && current_depth < max_depth {
            list_dir(base, &path, recursive, max_depth, current_depth + 1, entries)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn test_list_files_tool_name() {
        let tool = ListFilesTool;
        assert_eq!(tool.name(), "list_files");
    }

    #[test]
    fn test_list_files_tool_schema() {
        let tool = ListFilesTool;
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["path"].is_object());
        assert!(schema["properties"]["recursive"].is_object());
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("path")));
    }

    #[tokio::test]
    async fn test_list_flat() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "hello").unwrap();
        std::fs::write(dir.path().join("b.rs"), "fn main() {}").unwrap();
        std::fs::create_dir(dir.path().join("subdir")).unwrap();

        let tool = ListFilesTool;
        let result = tool
            .execute(json!({
                "path": dir.path().to_str().unwrap()
            }))
            .await
            .unwrap();

        assert!(result["success"].as_bool().unwrap());
        assert_eq!(result["entries"].as_u64().unwrap(), 3);
        let listing = result["listing"].as_array().unwrap();
        assert_eq!(listing.len(), 3);

        // Entries are sorted by name
        assert_eq!(listing[0]["name"], "a.txt");
        assert_eq!(listing[0]["type"], "file");
        assert_eq!(listing[0]["size"], 5); // "hello"

        assert_eq!(listing[1]["name"], "b.rs");
        assert_eq!(listing[1]["type"], "file");
        assert_eq!(listing[1]["size"], 12); // "fn main() {}"

        assert_eq!(listing[2]["name"], "subdir/");
        assert_eq!(listing[2]["type"], "dir");
        assert!(listing[2]["size"].is_null());
    }

    #[tokio::test]
    async fn test_list_recursive() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("top.txt"), "").unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub/nested.txt"), "data").unwrap();

        let tool = ListFilesTool;
        let result = tool
            .execute(json!({
                "path": dir.path().to_str().unwrap(),
                "recursive": true
            }))
            .await
            .unwrap();

        assert!(result["success"].as_bool().unwrap());
        let listing = result["listing"].as_array().unwrap();
        let names: Vec<&str> = listing.iter().map(|e| e["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"top.txt"));
        assert!(names.contains(&"sub/"));
        assert!(names.contains(&"sub/nested.txt"));

        // Check nested file has correct size
        let nested = listing.iter().find(|e| e["name"] == "sub/nested.txt").unwrap();
        assert_eq!(nested["type"], "file");
        assert_eq!(nested["size"], 4); // "data"
    }

    #[tokio::test]
    async fn test_list_max_depth() {
        let dir = tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("a/b/c/d")).unwrap();
        std::fs::write(dir.path().join("a/b/c/d/deep.txt"), "").unwrap();

        let tool = ListFilesTool;
        let result = tool
            .execute(json!({
                "path": dir.path().to_str().unwrap(),
                "recursive": true,
                "max_depth": 1
            }))
            .await
            .unwrap();

        let listing = result["listing"].as_array().unwrap();
        let names: Vec<&str> = listing.iter().map(|e| e["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"a/"));
        assert!(names.contains(&"a/b/"));
        // c/ is at depth 2, should not be listed with max_depth=1
        assert!(!names.iter().any(|n| n.contains("deep.txt")));
    }

    #[tokio::test]
    async fn test_list_not_a_directory() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("file.txt");
        std::fs::write(&file, "").unwrap();

        let tool = ListFilesTool;
        let result = tool
            .execute(json!({
                "path": file.to_str().unwrap()
            }))
            .await;

        assert!(matches!(result, Err(ToolError::ExecutionFailed(msg)) if msg.contains("not a directory")));
    }

    #[tokio::test]
    async fn test_list_missing_path_field() {
        let tool = ListFilesTool;
        let result = tool.execute(json!({})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_list_nonexistent_directory() {
        let tool = ListFilesTool;
        let result = tool
            .execute(json!({"path": "/nonexistent/dir/abc123"}))
            .await;
        assert!(result.is_err());
    }
}
