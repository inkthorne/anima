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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,      // "system", "user", "assistant", "tool"
    pub content: String,
    pub tool_call_id: Option<String>,  // For tool responses
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,  // JSON Schema
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
            model: "gpt-4o-mini".to_string(),
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
        
        let mut request_body = serde_json::json!({
            "model": self.model,
            "messages": messages
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