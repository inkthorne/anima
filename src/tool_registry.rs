//! Tool registry with semantic recall.
//!
//! This module provides a registry for storing tool definitions and using
//! keyword matching to recall only relevant tools for each query.

use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

/// A tool definition loaded from tools.toml.
#[derive(Debug, Clone, Deserialize)]
pub struct ToolDefinition {
    /// Tool name (e.g., "read_file")
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// Parameter schema as JSON
    pub params: serde_json::Value,
    /// Keywords for semantic matching
    pub keywords: Vec<String>,
    /// Optional category (e.g., "filesystem", "system")
    pub category: Option<String>,
}

/// TOML file structure for tools.toml.
#[derive(Debug, Deserialize)]
struct ToolsFile {
    tool: Vec<ToolDefinition>,
}

/// Registry that stores tool definitions and provides semantic recall.
#[derive(Debug)]
pub struct ToolRegistry {
    tools: Vec<ToolDefinition>,
}

/// Error type for tool registry operations.
#[derive(Debug)]
pub enum ToolRegistryError {
    /// Home directory not found
    HomeDirNotFound,
    /// File I/O error
    IoError(std::io::Error),
    /// TOML parsing error
    ParseError(toml::de::Error),
}

impl std::fmt::Display for ToolRegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolRegistryError::HomeDirNotFound => write!(f, "Home directory not found"),
            ToolRegistryError::IoError(e) => write!(f, "IO error: {}", e),
            ToolRegistryError::ParseError(e) => write!(f, "Parse error: {}", e),
        }
    }
}

impl std::error::Error for ToolRegistryError {}

impl From<std::io::Error> for ToolRegistryError {
    fn from(e: std::io::Error) -> Self {
        ToolRegistryError::IoError(e)
    }
}

impl From<toml::de::Error> for ToolRegistryError {
    fn from(e: toml::de::Error) -> Self {
        ToolRegistryError::ParseError(e)
    }
}

impl ToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    /// Load tools from ~/.anima/tools.toml.
    pub fn load_global() -> Result<Self, ToolRegistryError> {
        let home = dirs::home_dir().ok_or(ToolRegistryError::HomeDirNotFound)?;
        let path = home.join(".anima").join("tools.toml");
        Self::load_from_file(&path)
    }

    /// Load from a specific file path.
    pub fn load_from_file(path: &Path) -> Result<Self, ToolRegistryError> {
        let content = std::fs::read_to_string(path)?;
        let tools_file: ToolsFile = toml::from_str(&content)?;
        Ok(Self {
            tools: tools_file.tool,
        })
    }

    /// Get all tools in the registry.
    pub fn all_tools(&self) -> &[ToolDefinition] {
        &self.tools
    }

    /// Find tools relevant to a query using keyword matching.
    /// Returns up to `limit` tools, sorted by relevance score.
    pub fn find_relevant(&self, query: &str, limit: usize) -> Vec<&ToolDefinition> {
        let query_keywords = extract_keywords(query);

        if query_keywords.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(&ToolDefinition, f64)> = self
            .tools
            .iter()
            .map(|t| {
                let score = keyword_overlap(&query_keywords, &t.keywords);
                (t, score)
            })
            .filter(|(_, score)| *score > 0.0)
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        scored.into_iter().map(|(t, _)| t).collect()
    }

    /// Format tool definitions for injection into the agent's prompt.
    pub fn format_for_prompt(tools: &[&ToolDefinition]) -> String {
        if tools.is_empty() {
            return String::new();
        }

        let mut output = String::from("**Available tools:**\n");
        for tool in tools {
            output.push_str(&format!(
                "- `{}` — {}. Params: {}\n",
                tool.name,
                tool.description,
                format_params(&tool.params)
            ));
        }
        output
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract keywords from a query string.
/// Lowercases and splits on whitespace/punctuation.
fn extract_keywords(query: &str) -> HashSet<String> {
    query
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() >= 2) // Skip single chars
        .map(|s| s.to_string())
        .collect()
}

/// Calculate the overlap score between query keywords and tool keywords.
/// Returns 0.0 to 1.0 based on how many tool keywords appear in the query.
fn keyword_overlap(query_keywords: &HashSet<String>, tool_keywords: &[String]) -> f64 {
    if tool_keywords.is_empty() {
        return 0.0;
    }

    let tool_keywords_lower: Vec<String> = tool_keywords.iter().map(|k| k.to_lowercase()).collect();

    let matches = query_keywords
        .iter()
        .filter(|qk| {
            tool_keywords_lower
                .iter()
                .any(|tk| tk.contains(qk.as_str()) || qk.contains(tk.as_str()))
        })
        .count();

    matches as f64 / tool_keywords.len() as f64
}

/// Format parameter schema as a human-readable string.
/// Converts {"path": "string", "content": "string"} to "`path` (string), `content` (string)"
fn format_params(params: &serde_json::Value) -> String {
    match params {
        serde_json::Value::Object(map) => {
            let parts: Vec<String> = map
                .iter()
                .map(|(key, val)| {
                    let type_str = match val {
                        serde_json::Value::String(s) => s.clone(),
                        _ => val.to_string(),
                    };
                    format!("`{}` ({})", key, type_str)
                })
                .collect();
            parts.join(", ")
        }
        _ => params.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn sample_tools_toml() -> &'static str {
        r#"
[[tool]]
name = "read_file"
description = "Read a file's contents"
params = { path = "string" }
keywords = ["read", "file", "contents", "open", "load", "cat", "view"]
category = "filesystem"

[[tool]]
name = "write_file"
description = "Write content to a file"
params = { path = "string", content = "string" }
keywords = ["write", "file", "save", "create", "output", "store"]
category = "filesystem"

[[tool]]
name = "shell"
description = "Run a shell command"
params = { command = "string" }
keywords = ["shell", "command", "run", "execute", "bash", "terminal", "build", "test"]
category = "system"
"#
    }

    #[test]
    fn test_load_from_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("tools.toml");
        std::fs::write(&path, sample_tools_toml()).unwrap();

        let registry = ToolRegistry::load_from_file(&path).unwrap();
        assert_eq!(registry.tools.len(), 3);
        assert_eq!(registry.tools[0].name, "read_file");
        assert_eq!(registry.tools[1].name, "write_file");
        assert_eq!(registry.tools[2].name, "shell");
    }

    #[test]
    fn test_load_missing_file() {
        let path = Path::new("/nonexistent/tools.toml");
        let result = ToolRegistry::load_from_file(path);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_keywords() {
        let keywords = extract_keywords("read the config file");
        assert!(keywords.contains("read"));
        assert!(keywords.contains("the"));
        assert!(keywords.contains("config"));
        assert!(keywords.contains("file"));
    }

    #[test]
    fn test_extract_keywords_punctuation() {
        let keywords = extract_keywords("file.txt, open it!");
        assert!(keywords.contains("file"));
        assert!(keywords.contains("txt"));
        assert!(keywords.contains("open"));
        assert!(keywords.contains("it"));
    }

    #[test]
    fn test_keyword_overlap_full_match() {
        let query: HashSet<String> = ["read", "file"].iter().map(|s| s.to_string()).collect();
        let tool_keywords = vec!["read".to_string(), "file".to_string()];
        let score = keyword_overlap(&query, &tool_keywords);
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_keyword_overlap_partial_match() {
        let query: HashSet<String> = ["read", "something"].iter().map(|s| s.to_string()).collect();
        let tool_keywords = vec!["read".to_string(), "file".to_string()];
        let score = keyword_overlap(&query, &tool_keywords);
        assert!((score - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_keyword_overlap_no_match() {
        let query: HashSet<String> = ["network", "http"].iter().map(|s| s.to_string()).collect();
        let tool_keywords = vec!["read".to_string(), "file".to_string()];
        let score = keyword_overlap(&query, &tool_keywords);
        assert!((score - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_find_relevant_read_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("tools.toml");
        std::fs::write(&path, sample_tools_toml()).unwrap();

        let registry = ToolRegistry::load_from_file(&path).unwrap();
        let results = registry.find_relevant("read the config file", 5);

        assert!(!results.is_empty());
        assert_eq!(results[0].name, "read_file");
    }

    #[test]
    fn test_find_relevant_shell() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("tools.toml");
        std::fs::write(&path, sample_tools_toml()).unwrap();

        let registry = ToolRegistry::load_from_file(&path).unwrap();
        let results = registry.find_relevant("run the build command", 5);

        assert!(!results.is_empty());
        assert_eq!(results[0].name, "shell");
    }

    #[test]
    fn test_find_relevant_limit() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("tools.toml");
        std::fs::write(&path, sample_tools_toml()).unwrap();

        let registry = ToolRegistry::load_from_file(&path).unwrap();
        // "file" matches both read_file and write_file
        let results = registry.find_relevant("file operations", 1);

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_find_relevant_empty_query() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("tools.toml");
        std::fs::write(&path, sample_tools_toml()).unwrap();

        let registry = ToolRegistry::load_from_file(&path).unwrap();
        let results = registry.find_relevant("", 5);

        assert!(results.is_empty());
    }

    #[test]
    fn test_find_relevant_no_match() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("tools.toml");
        std::fs::write(&path, sample_tools_toml()).unwrap();

        let registry = ToolRegistry::load_from_file(&path).unwrap();
        let results = registry.find_relevant("quantum entanglement", 5);

        assert!(results.is_empty());
    }

    #[test]
    fn test_format_params_object() {
        let params = serde_json::json!({"path": "string", "content": "string"});
        let formatted = format_params(&params);
        assert!(formatted.contains("`path` (string)"));
        assert!(formatted.contains("`content` (string)"));
    }

    #[test]
    fn test_format_params_single() {
        let params = serde_json::json!({"command": "string"});
        let formatted = format_params(&params);
        assert_eq!(formatted, "`command` (string)");
    }

    #[test]
    fn test_format_for_prompt() {
        let tool = ToolDefinition {
            name: "read_file".to_string(),
            description: "Read a file".to_string(),
            params: serde_json::json!({"path": "string"}),
            keywords: vec!["read".to_string()],
            category: Some("filesystem".to_string()),
        };

        let formatted = ToolRegistry::format_for_prompt(&[&tool]);
        assert!(formatted.contains("**Available tools:**"));
        assert!(formatted.contains("`read_file`"));
        assert!(formatted.contains("Read a file"));
        assert!(formatted.contains("`path` (string)"));
    }

    #[test]
    fn test_format_for_prompt_empty() {
        let formatted = ToolRegistry::format_for_prompt(&[]);
        assert!(formatted.is_empty());
    }

    #[test]
    fn test_all_tools() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("tools.toml");
        std::fs::write(&path, sample_tools_toml()).unwrap();

        let registry = ToolRegistry::load_from_file(&path).unwrap();
        let all = registry.all_tools();

        assert_eq!(all.len(), 3);
    }
}
