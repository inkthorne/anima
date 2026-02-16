use crate::error::ToolError;
use crate::tool::Tool;
use async_trait::async_trait;
use serde_json::Value;

/// Daemon-aware tool that lets agents update their working scratchpad.
/// Notes are stored in the participants table and injected at the end of context
/// each turn, providing persistent working state across tool-loop iterations.
pub struct DaemonNotesTool {
    agent_name: String,
    conv_id: Option<String>,
}

impl DaemonNotesTool {
    pub fn new(agent_name: String, conv_id: Option<String>) -> Self {
        DaemonNotesTool {
            agent_name,
            conv_id,
        }
    }
}

impl std::fmt::Debug for DaemonNotesTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DaemonNotesTool")
            .field("agent_name", &self.agent_name)
            .finish()
    }
}

#[async_trait]
impl Tool for DaemonNotesTool {
    fn name(&self) -> &str {
        "notes"
    }

    fn description(&self) -> &str {
        "Update your working scratchpad. Use this to record your current hypothesis, findings, and next steps. Notes persist across turns and are shown to you automatically. Each call replaces previous notes."
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Your working notes (replaces any previous notes)"
                }
            },
            "required": ["content"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        use crate::conversation::ConversationStore;

        let content = input
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("content is required".to_string()))?;

        let conv_id = self.conv_id.as_deref().ok_or_else(|| {
            ToolError::ExecutionFailed("notes tool requires a conversation context".to_string())
        })?;

        let store = ConversationStore::init().map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to open conversation store: {}", e))
        })?;

        store
            .set_participant_notes(conv_id, &self.agent_name, content)
            .map_err(|e| {
                ToolError::ExecutionFailed(format!("Failed to save notes: {}", e))
            })?;

        Ok(serde_json::json!({
            "status": "ok",
            "size": content.len()
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_daemon_notes_tool_name() {
        let tool = DaemonNotesTool::new("dash".to_string(), Some("conv-1".to_string()));
        assert_eq!(tool.name(), "notes");
    }

    #[test]
    fn test_daemon_notes_schema() {
        let tool = DaemonNotesTool::new("dash".to_string(), None);
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["content"].is_object());
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("content")));
    }

    #[tokio::test]
    async fn test_daemon_notes_missing_content() {
        let tool = DaemonNotesTool::new("dash".to_string(), Some("conv-1".to_string()));
        let result = tool.execute(json!({})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_daemon_notes_no_conversation() {
        let tool = DaemonNotesTool::new("dash".to_string(), None);
        let result = tool.execute(json!({"content": "test notes"})).await;
        assert!(matches!(result, Err(ToolError::ExecutionFailed(_))));
    }

    #[tokio::test]
    async fn test_daemon_notes_roundtrip() {
        use crate::conversation::ConversationStore;

        let store = ConversationStore::init().unwrap();
        let conv = format!("notes-roundtrip-{}", std::process::id());
        store.create_conversation(Some(&conv), &["dash"]).unwrap();

        let tool = DaemonNotesTool::new("dash".to_string(), Some(conv.clone()));
        let result = tool
            .execute(json!({"content": "hypothesis: the bug is in parser"}))
            .await
            .unwrap();
        assert_eq!(result["status"], "ok");

        let notes = store.get_participant_notes(&conv, "dash").unwrap();
        assert_eq!(notes.as_deref(), Some("hypothesis: the bug is in parser"));

        // Second call replaces
        tool.execute(json!({"content": "confirmed: parser bug at line 42"}))
            .await
            .unwrap();
        let notes2 = store.get_participant_notes(&conv, "dash").unwrap();
        assert_eq!(notes2.as_deref(), Some("confirmed: parser bug at line 42"));

        store.delete_conversation(&conv).unwrap();
    }
}
