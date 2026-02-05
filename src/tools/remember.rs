use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::error::ToolError;
use crate::tool::Tool;
use crate::memory::SemanticMemoryStore;
use crate::embedding::EmbeddingClient;
use serde_json::Value;

/// Tool that allows agents to explicitly save memories via tool call.
/// This provides an alternative to the [REMEMBER: ...] tag syntax.
#[derive(Debug, Default)]
pub struct RememberTool;

/// Daemon-aware version of RememberTool that has access to semantic memory.
/// Use this in daemon mode where we have access to the memory store.
pub struct DaemonRememberTool {
    semantic_memory: Arc<Mutex<SemanticMemoryStore>>,
    embedding_client: Option<Arc<EmbeddingClient>>,
}

impl DaemonRememberTool {
    pub fn new(
        semantic_memory: Arc<Mutex<SemanticMemoryStore>>,
        embedding_client: Option<Arc<EmbeddingClient>>,
    ) -> Self {
        DaemonRememberTool {
            semantic_memory,
            embedding_client,
        }
    }
}

impl std::fmt::Debug for DaemonRememberTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DaemonRememberTool").finish()
    }
}

#[async_trait]
impl Tool for DaemonRememberTool {
    fn name(&self) -> &str {
        "remember"
    }

    fn description(&self) -> &str {
        "Saves a memory to persistent storage for future recall"
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to remember"
                }
            },
            "required": ["content"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let content = input
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing or invalid 'content' field".to_string()))?;

        // Generate embedding if client is available
        let embedding = if let Some(ref client) = self.embedding_client {
            client.embed(content).await.ok()
        } else {
            None
        };

        // Save to semantic memory
        let store = self.semantic_memory.lock().await;
        store.save_with_embedding(content, 0.9, "explicit", embedding.as_deref())
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to save memory: {}", e)))?;

        Ok(serde_json::json!({
            "success": true,
            "message": format!("Saved memory: '{}'", if content.len() > 50 { &content[..50] } else { content })
        }))
    }
}

#[async_trait]
impl Tool for RememberTool {
    fn name(&self) -> &str {
        "remember"
    }

    fn description(&self) -> &str {
        "Saves a memory to persistent storage for future recall"
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to remember"
                }
            },
            "required": ["content"]
        })
    }

    async fn execute(&self, _input: Value) -> Result<Value, ToolError> {
        // Note: Actual memory saving happens in daemon.rs execute_tool_call
        // because it requires access to SemanticMemoryStore and EmbeddingClient.
        // This method is not called directly.
        Err(ToolError::ExecutionFailed(
            "RememberTool must be executed through daemon context".to_string()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_remember_tool_name() {
        let tool = RememberTool;
        assert_eq!(tool.name(), "remember");
    }

    #[test]
    fn test_remember_tool_description() {
        let tool = RememberTool;
        assert!(tool.description().contains("memory"));
    }

    #[test]
    fn test_remember_tool_schema() {
        let tool = RememberTool;
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["content"].is_object());
        assert!(schema["required"].as_array().unwrap().contains(&json!("content")));
    }
}
