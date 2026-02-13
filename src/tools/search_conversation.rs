use crate::conversation::ConversationStore;
use crate::error::ToolError;
use crate::tool::Tool;
use async_trait::async_trait;
use serde_json::Value;

/// Tool that allows agents to search conversation messages by keyword.
/// Provides grep-like functionality over conversation history.
#[derive(Debug, Default)]
pub struct DaemonSearchConversationTool;

#[async_trait]
impl Tool for DaemonSearchConversationTool {
    fn name(&self) -> &str {
        "search_conversation"
    }

    fn description(&self) -> &str {
        "Search conversation messages by keyword with optional sender filter"
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "conversation": {
                    "type": "string",
                    "description": "The conversation name to search"
                },
                "keyword": {
                    "type": "string",
                    "description": "The keyword or phrase to search for"
                },
                "from": {
                    "type": "string",
                    "description": "Filter results to messages from this sender (optional)"
                },
                "include_internal": {
                    "type": "boolean",
                    "description": "Include tool and recall messages (default false)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default 20, max 50)"
                }
            },
            "required": ["conversation", "keyword"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let conv_name = input
            .get("conversation")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ToolError::InvalidInput("Missing or invalid 'conversation' field".to_string())
            })?;

        let keyword = input
            .get("keyword")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ToolError::InvalidInput("Missing or invalid 'keyword' field".to_string())
            })?;

        if keyword.is_empty() {
            return Err(ToolError::InvalidInput(
                "Keyword cannot be empty".to_string(),
            ));
        }

        let from_agent = input.get("from").and_then(|v| v.as_str());
        let include_internal = input
            .get("include_internal")
            .and_then(super::json_to_bool)
            .unwrap_or(false);
        let limit = input
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|v| v.min(50) as usize)
            .unwrap_or(20);

        let store = ConversationStore::init().map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to open conversation store: {}", e))
        })?;

        let messages = store
            .search_messages(
                conv_name,
                keyword,
                from_agent,
                include_internal,
                limit,
                Some(500),
            )
            .map_err(|e| ToolError::ExecutionFailed(format!("Search failed: {}", e)))?;

        let count = messages.len();

        let results: Vec<Value> = messages
            .into_iter()
            .map(|msg| {
                serde_json::json!({
                    "id": msg.id,
                    "from": msg.from_agent,
                    "content": msg.content,
                    "created_at": msg.created_at,
                })
            })
            .collect();

        let summary = if count == 0 {
            format!("No messages found matching '{}'", keyword)
        } else {
            let mut lines = Vec::new();
            for r in &results {
                let from = r.get("from").and_then(|s| s.as_str()).unwrap_or("unknown");
                let content = r.get("content").and_then(|s| s.as_str()).unwrap_or("");
                lines.push(format!("[{}] {}", from, content));
            }
            format!(
                "Found {} message(s) matching '{}':\n{}",
                count,
                keyword,
                lines.join("\n")
            )
        };

        Ok(serde_json::json!({
            "results": results,
            "count": count,
            "summary": summary,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_search_conversation_tool_name() {
        let tool = DaemonSearchConversationTool;
        assert_eq!(tool.name(), "search_conversation");
    }

    #[test]
    fn test_search_conversation_tool_description() {
        let tool = DaemonSearchConversationTool;
        assert!(tool.description().contains("Search"));
    }

    #[test]
    fn test_search_conversation_tool_schema() {
        let tool = DaemonSearchConversationTool;
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["conversation"].is_object());
        assert!(schema["properties"]["keyword"].is_object());
        assert!(schema["properties"]["from"].is_object());
        assert!(schema["properties"]["limit"].is_object());
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("conversation")));
        assert!(required.contains(&json!("keyword")));
    }

    #[tokio::test]
    async fn test_search_conversation_missing_conversation() {
        let tool = DaemonSearchConversationTool;
        let result = tool.execute(json!({"keyword": "hello"})).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::InvalidInput(msg) => assert!(msg.contains("conversation")),
            e => panic!("Expected InvalidInput, got {:?}", e),
        }
    }

    #[tokio::test]
    async fn test_search_conversation_missing_keyword() {
        let tool = DaemonSearchConversationTool;
        let result = tool.execute(json!({"conversation": "test"})).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::InvalidInput(msg) => assert!(msg.contains("keyword")),
            e => panic!("Expected InvalidInput, got {:?}", e),
        }
    }

    #[tokio::test]
    async fn test_search_conversation_empty_keyword() {
        let tool = DaemonSearchConversationTool;
        let result = tool
            .execute(json!({"conversation": "test", "keyword": ""}))
            .await;
        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::InvalidInput(msg) => assert!(msg.contains("empty")),
            e => panic!("Expected InvalidInput, got {:?}", e),
        }
    }
}
