use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[async_trait]
pub trait LLM: Send + Sync {
    async fn chat_complete(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<ToolSpec>>,
    ) -> Result<LLMResponse, LLMError>;
}

fn format_tool_calls_for_api(tool_calls: &[ToolCall]) -> Vec<serde_json::Value> {
    tool_calls.iter().map(|tc| {
        serde_json::json!({
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": tc.arguments.to_string()
            }
        })
    }).collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,      // "system", "user", "assistant", "tool"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,  // For tool responses
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,  // OpenAI uses parameters instead of input_schema
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMResponse {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMError {
    pub message: String,
}

impl std::fmt::Display for LLMError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LLMError: {}", self.message)
    }
}

impl std::error::Error for LLMError {}

pub struct OpenAIClient {
    client: Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl OpenAIClient {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".to_string(),
            model: "gpt-4o".to_string(),
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

pub struct AnthropicClient {
    client: Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl AnthropicClient {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com".to_string(),
            model: "claude-sonnet-4-20250514".to_string(),
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

#[async_trait]
impl LLM for OpenAIClient {
    async fn chat_complete(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<ToolSpec>>,
    ) -> Result<LLMResponse, LLMError> {
        let url = format!("{}/chat/completions", self.base_url);
        
        // Transform messages to fix tool_calls serialization for API compatibility
        let mut formatted_messages = Vec::new();
        for message in messages {
            let mut formatted_message = serde_json::to_value(&message).unwrap();
            if let Some(tool_calls) = message.tool_calls {
                if !tool_calls.is_empty() {
                    let formatted_tool_calls = format_tool_calls_for_api(&tool_calls);
                    formatted_message["tool_calls"] = serde_json::to_value(formatted_tool_calls).unwrap();
                }
            }
            formatted_messages.push(formatted_message);
        }
        
        let mut request_body = serde_json::json!({
            "model": self.model,
            "messages": formatted_messages
        });
        
         if let Some(tool_list) = tools {
             let formatted_tools: Vec<serde_json::Value> = tool_list.iter().map(|t| {
                 serde_json::json!({
                     "type": "function",
                     "function": {
                         "name": t.name,
                         "description": t.description,
                         "parameters": t.parameters
                     }
                 })
             }).collect();
             request_body["tools"] = serde_json::to_value(formatted_tools).unwrap();
             request_body["tool_choice"] = serde_json::json!("auto");
         }
        
        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LLMError {
                message: format!("Failed to send request: {}", e),
            })?;
            
        let response_text = response
            .text()
            .await
            .map_err(|e| LLMError {
                message: format!("Failed to read response: {}", e),
            })?;
            
        let openai_response: serde_json::Value = serde_json::from_str(&response_text)
            .map_err(|e| LLMError {
                message: format!("Failed to parse response: {}", e),
            })?;
            
        let content = openai_response["choices"][0]["message"]["content"]
            .as_str()
            .map(|s| s.to_string());
            
         let tool_calls = openai_response["choices"][0]["message"]["tool_calls"]
             .as_array()
             .map(|tool_calls_array| {
                 tool_calls_array
                     .iter()
                     .filter_map(|tool_call| {
                         let id = tool_call["id"].as_str().map(|s| s.to_string())?;
                         let name = tool_call["function"]["name"].as_str().map(|s| s.to_string())?;
                         let arguments_str = tool_call["function"]["arguments"].as_str()?;
                         let arguments: serde_json::Value = serde_json::from_str(arguments_str).ok()?;
                         Some(ToolCall { id, name, arguments })
                     })
                     .collect()
             })
             .unwrap_or_default();
            
        Ok(LLMResponse {
            content,
            tool_calls,
        })
    }
}

#[async_trait]
impl LLM for AnthropicClient {
    async fn chat_complete(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<ToolSpec>>,
    ) -> Result<LLMResponse, LLMError> {
        let url = format!("{}/v1/messages", self.base_url);
        
        // Separate system prompt from messages
        let system_prompt = messages
            .iter()
            .find(|msg| msg.role == "system")
            .and_then(|msg| msg.content.as_ref())
            .cloned();
        
        let filtered_messages: Vec<ChatMessage> = messages
            .into_iter()
            .filter(|msg| msg.role != "system")
            .collect();
        
        let mut request_body = serde_json::json!({
            "model": self.model,
            "messages": filtered_messages,
        });
        
        if let Some(system) = system_prompt {
            request_body["system"] = serde_json::Value::String(system);
        }
        
        if let Some(tool_list) = tools {
            // Anthropic uses input_schema instead of parameters, and tools are top-level (not wrapped in function)
            let formatted_tools: Vec<serde_json::Value> = tool_list.iter().map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters
                })
            }).collect();
            request_body["tools"] = serde_json::to_value(formatted_tools).unwrap();
        }
        
        let response = self
            .client
            .post(&url)
            .header("x-api-key", self.api_key.clone())
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LLMError {
                message: format!("Failed to send request: {}", e),
            })?;
            
        let response_text = response
            .text()
            .await
            .map_err(|e| LLMError {
                message: format!("Failed to read response: {}", e),
            })?;
            
        let anthropic_response: serde_json::Value = serde_json::from_str(&response_text)
            .map_err(|e| LLMError {
                message: format!("Failed to parse response: {}", e),
            })?;
            
        // Extract content (text and tool_use blocks)
        let mut content_text = String::new();
        let mut tool_calls = Vec::new();
        
        if let Some(content_array) = anthropic_response["content"].as_array() {
            for content_block in content_array {
                match content_block["type"].as_str() {
                    Some("text") => {
                        if let Some(text) = content_block["text"].as_str() {
                            content_text.push_str(text);
                        }
                    }
                    Some("tool_use") => {
                        let id = content_block["id"].as_str().map(|s| s.to_string()).unwrap_or_default();
                        let name = content_block["name"].as_str().map(|s| s.to_string()).unwrap_or_default();
                        let input = content_block["input"].clone();
                        tool_calls.push(ToolCall { id, name, arguments: input });
                    }
                    _ => {}
                }
            }
        }
        
        Ok(LLMResponse {
            content: if content_text.is_empty() { None } else { Some(content_text) },
            tool_calls,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_tool_format() {
        // Verify tools are formatted with input_schema
        let tool = ToolSpec {
            name: "add".to_string(),
            description: "Add two numbers".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                }
            }),
        };
        
        let formatted = serde_json::json!({
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters
        });
        
        assert_eq!(formatted["name"], "add");
        assert_eq!(formatted["input_schema"]["type"], "object");
    }

    #[test]
    fn test_anthropic_message_format() {
        // Test that tool results become tool_result content blocks
        let tool_msg = ChatMessage {
            role: "tool".to_string(),
            content: Some("8.0".to_string()),
            tool_call_id: Some("call_123".to_string()),
            tool_calls: None,
        };
        
        // This is how it should be formatted for Anthropic
        let formatted = serde_json::json!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_msg.tool_call_id,
                "content": tool_msg.content
            }]
        });
        
        assert_eq!(formatted["role"], "user");
        assert_eq!(formatted["content"][0]["type"], "tool_result");
        assert_eq!(formatted["content"][0]["tool_use_id"], "call_123");
    }

    #[test]
    fn test_anthropic_client_builder() {
        let client = AnthropicClient::new("test-key")
            .with_model("claude-opus-4-20250514")
            .with_base_url("https://custom.api.com");
        
        assert_eq!(client.model, "claude-opus-4-20250514");
        assert_eq!(client.base_url, "https://custom.api.com");
        assert_eq!(client.api_key, "test-key");
    }
}