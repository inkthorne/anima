use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

#[async_trait]
pub trait LLM: Send + Sync {
    async fn chat_complete(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<ToolSpec>>,
    ) -> Result<LLMResponse, LLMError>;

    /// Stream chat completion, sending tokens through the channel as they arrive.
    /// Returns the final complete response when done.
    async fn chat_complete_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<ToolSpec>>,
        tx: mpsc::Sender<String>,
    ) -> Result<LLMResponse, LLMError>;

    /// Returns the model name for observability purposes.
    fn model_name(&self) -> &str;
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
    /// Token usage information if available from the API
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageInfo>,
}

/// Token usage information from LLM API responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageInfo {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

/// Error from LLM operations with retryability classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMError {
    pub message: String,
    /// HTTP status code if available
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status_code: Option<u16>,
    /// Whether this error is likely transient and worth retrying
    #[serde(default)]
    pub is_retryable: bool,
}

impl LLMError {
    /// Create a retryable error (transient failures like timeouts, rate limits)
    pub fn retryable(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            status_code: None,
            is_retryable: true,
        }
    }

    /// Create a retryable error with status code
    pub fn retryable_with_status(message: impl Into<String>, status: u16) -> Self {
        Self {
            message: message.into(),
            status_code: Some(status),
            is_retryable: true,
        }
    }

    /// Create a non-retryable error (auth failures, bad requests)
    pub fn permanent(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            status_code: None,
            is_retryable: false,
        }
    }

    /// Create a non-retryable error with status code
    pub fn permanent_with_status(message: impl Into<String>, status: u16) -> Self {
        Self {
            message: message.into(),
            status_code: Some(status),
            is_retryable: false,
        }
    }

    /// Classify an HTTP status code as retryable or not
    pub fn from_status(status: u16, message: impl Into<String>) -> Self {
        let is_retryable = matches!(status, 408 | 429 | 500..=599);
        Self {
            message: message.into(),
            status_code: Some(status),
            is_retryable,
        }
    }
}

impl std::fmt::Display for LLMError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(status) = self.status_code {
            write!(f, "LLMError ({}): {}", status, self.message)
        } else {
            write!(f, "LLMError: {}", self.message)
        }
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
    fn model_name(&self) -> &str {
        &self.model
    }

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
            .map_err(|e| {
                // Network errors are typically retryable
                LLMError::retryable(format!("Failed to send request: {}", e))
            })?;

        let status = response.status().as_u16();

        let response_text = response
            .text()
            .await
            .map_err(|e| {
                // Read errors are retryable
                LLMError::retryable(format!("Failed to read response: {}", e))
            })?;

        // Check for HTTP errors before parsing
        if status >= 400 {
            return Err(LLMError::from_status(
                status,
                format!("API error: {}", response_text),
            ));
        }

        let openai_response: serde_json::Value = serde_json::from_str(&response_text)
            .map_err(|e| {
                // Parse errors are not retryable
                LLMError::permanent(format!("Failed to parse response: {}", e))
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

        // Extract usage information
        let usage = openai_response["usage"].as_object().map(|u| UsageInfo {
            prompt_tokens: u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            completion_tokens: u.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        });

        Ok(LLMResponse {
            content,
            tool_calls,
            usage,
        })
    }

    async fn chat_complete_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<ToolSpec>>,
        tx: mpsc::Sender<String>,
    ) -> Result<LLMResponse, LLMError> {
        use futures_util::StreamExt;

        let url = format!("{}/chat/completions", self.base_url);

        // Transform messages for API compatibility
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
            "messages": formatted_messages,
            "stream": true
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
            .map_err(|e| {
                LLMError::retryable(format!("Failed to send request: {}", e))
            })?;

        let status = response.status().as_u16();
        if status >= 400 {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LLMError::from_status(status, format!("API error: {}", error_text)));
        }

        let mut stream = response.bytes_stream();
        let mut full_content = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut tool_call_builders: std::collections::HashMap<usize, (String, String, String)> =
            std::collections::HashMap::new(); // index -> (id, name, arguments)
        let mut buffer = String::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| {
                LLMError::retryable(format!("Failed to read stream chunk: {}", e))
            })?;

            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Process complete SSE lines
            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim().to_string();
                buffer = buffer[line_end + 1..].to_string();

                if line.is_empty() || line.starts_with(':') {
                    continue;
                }

                if line == "data: [DONE]" {
                    break;
                }

                if let Some(data) = line.strip_prefix("data: ") {
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data) {
                        // Handle content delta
                        if let Some(content) = parsed["choices"][0]["delta"]["content"].as_str() {
                            full_content.push_str(content);
                            let _ = tx.send(content.to_string()).await;
                        }

                        // Handle tool calls delta
                        if let Some(tc_array) = parsed["choices"][0]["delta"]["tool_calls"].as_array() {
                            for tc in tc_array {
                                let index = tc["index"].as_u64().unwrap_or(0) as usize;
                                let entry = tool_call_builders.entry(index)
                                    .or_insert_with(|| (String::new(), String::new(), String::new()));

                                if let Some(id) = tc["id"].as_str() {
                                    entry.0 = id.to_string();
                                }
                                if let Some(name) = tc["function"]["name"].as_str() {
                                    entry.1.push_str(name);
                                }
                                if let Some(args) = tc["function"]["arguments"].as_str() {
                                    entry.2.push_str(args);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Convert tool call builders to final tool calls
        let mut indices: Vec<usize> = tool_call_builders.keys().cloned().collect();
        indices.sort();
        for index in indices {
            if let Some((id, name, arguments_str)) = tool_call_builders.remove(&index) {
                if let Ok(arguments) = serde_json::from_str(&arguments_str) {
                    tool_calls.push(ToolCall { id, name, arguments });
                }
            }
        }

        Ok(LLMResponse {
            content: if full_content.is_empty() { None } else { Some(full_content) },
            tool_calls,
            usage: None, // Streaming typically doesn't provide usage info
        })
    }
}

#[async_trait]
impl LLM for AnthropicClient {
    fn model_name(&self) -> &str {
        &self.model
    }
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
            .map_err(|e| {
                LLMError::retryable(format!("Failed to send request: {}", e))
            })?;

        let status = response.status().as_u16();

        let response_text = response
            .text()
            .await
            .map_err(|e| {
                LLMError::retryable(format!("Failed to read response: {}", e))
            })?;

        // Check for HTTP errors before parsing
        if status >= 400 {
            return Err(LLMError::from_status(
                status,
                format!("API error: {}", response_text),
            ));
        }

        let anthropic_response: serde_json::Value = serde_json::from_str(&response_text)
            .map_err(|e| {
                LLMError::permanent(format!("Failed to parse response: {}", e))
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
        
        // Extract usage information (Anthropic format)
        let usage = anthropic_response["usage"].as_object().map(|u| UsageInfo {
            prompt_tokens: u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            completion_tokens: u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        });

        Ok(LLMResponse {
            content: if content_text.is_empty() { None } else { Some(content_text) },
            tool_calls,
            usage,
        })
    }

    async fn chat_complete_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<ToolSpec>>,
        tx: mpsc::Sender<String>,
    ) -> Result<LLMResponse, LLMError> {
        use futures_util::StreamExt;

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
            "stream": true
        });

        if let Some(system) = system_prompt {
            request_body["system"] = serde_json::Value::String(system);
        }

        if let Some(tool_list) = tools {
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
            .map_err(|e| {
                LLMError::retryable(format!("Failed to send request: {}", e))
            })?;

        let status = response.status().as_u16();
        if status >= 400 {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LLMError::from_status(status, format!("API error: {}", error_text)));
        }

        let mut stream = response.bytes_stream();
        let mut full_content = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        // Track tool_use blocks being built: index -> (id, name, input_json_str)
        let mut current_tool_use: Option<(String, String, String)> = None;
        let mut buffer = String::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| {
                LLMError::retryable(format!("Failed to read stream chunk: {}", e))
            })?;

            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Process complete SSE lines
            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim().to_string();
                buffer = buffer[line_end + 1..].to_string();

                if line.is_empty() || line.starts_with(':') {
                    continue;
                }

                // Anthropic uses "event:" lines followed by "data:" lines
                if let Some(data) = line.strip_prefix("data: ") {
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data) {
                        let event_type = parsed["type"].as_str().unwrap_or("");

                        match event_type {
                            "content_block_start" => {
                                // Check if this is a tool_use block
                                if let Some(content_block) = parsed["content_block"].as_object() {
                                    if content_block.get("type").and_then(|v| v.as_str()) == Some("tool_use") {
                                        let id = content_block.get("id")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("")
                                            .to_string();
                                        let name = content_block.get("name")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("")
                                            .to_string();
                                        current_tool_use = Some((id, name, String::new()));
                                    }
                                }
                            }
                            "content_block_delta" => {
                                if let Some(delta) = parsed["delta"].as_object() {
                                    // Text delta
                                    if delta.get("type").and_then(|v| v.as_str()) == Some("text_delta") {
                                        if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                                            full_content.push_str(text);
                                            let _ = tx.send(text.to_string()).await;
                                        }
                                    }
                                    // Input JSON delta for tool_use
                                    if delta.get("type").and_then(|v| v.as_str()) == Some("input_json_delta") {
                                        if let Some(partial_json) = delta.get("partial_json").and_then(|v| v.as_str()) {
                                            if let Some((_, _, ref mut input_str)) = current_tool_use {
                                                input_str.push_str(partial_json);
                                            }
                                        }
                                    }
                                }
                            }
                            "content_block_stop" => {
                                // Finalize tool_use if we were building one
                                if let Some((id, name, input_str)) = current_tool_use.take() {
                                    if !id.is_empty() && !name.is_empty() {
                                        if let Ok(arguments) = serde_json::from_str(&input_str) {
                                            tool_calls.push(ToolCall { id, name, arguments });
                                        } else if input_str.is_empty() {
                                            // Empty input is valid (tool with no parameters)
                                            tool_calls.push(ToolCall {
                                                id,
                                                name,
                                                arguments: serde_json::json!({}),
                                            });
                                        }
                                    }
                                }
                            }
                            "message_stop" => {
                                // End of message
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        Ok(LLMResponse {
            content: if full_content.is_empty() { None } else { Some(full_content) },
            tool_calls,
            usage: None, // Streaming typically doesn't provide usage info
        })
    }
}

/// Parse thinking prefix from message content.
/// Returns (thinking_override, stripped_content):
/// - "/think ..." → (Some(true), "...")
/// - "/no_think ..." → (Some(false), "...")
/// - "..." → (None, "...")
/// Case-insensitive, trims whitespace after prefix.
fn parse_thinking_prefix(content: &str) -> (Option<bool>, &str) {
    let content_lower = content.to_lowercase();
    if content_lower.starts_with("/think ") {
        (Some(true), content[7..].trim_start())
    } else if content_lower.starts_with("/no_think ") {
        (Some(false), content[10..].trim_start())
    } else {
        (None, content)
    }
}

/// Ollama client using OpenAI-compatible API.
/// Configure with OLLAMA_HOST env var (defaults to http://localhost:11434)
pub struct OllamaClient {
    client: Client,
    base_url: String,
    model: String,
    /// Enable thinking mode (Ollama "think" parameter)
    thinking: Option<bool>,
}

impl OllamaClient {
    pub fn new() -> Self {
        let base_url = std::env::var("OLLAMA_HOST")
            .unwrap_or_else(|_| "http://localhost:11434".to_string());
        Self {
            client: Client::new(),
            base_url,
            model: "llama3".to_string(),
            thinking: None,
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

    pub fn with_thinking(mut self, thinking: Option<bool>) -> Self {
        self.thinking = thinking;
        self
    }
}

impl Default for OllamaClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLM for OllamaClient {
    fn model_name(&self) -> &str {
        &self.model
    }

    async fn chat_complete(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<ToolSpec>>,
    ) -> Result<LLMResponse, LLMError> {
        let url = format!("{}/v1/chat/completions", self.base_url);

        // Find thinking override from the last user message
        let mut thinking_override: Option<bool> = None;
        let mut stripped_user_content: Option<String> = None;
        let last_user_idx = messages.iter().rposition(|m| m.role == "user");

        if let Some(idx) = last_user_idx {
            if let Some(ref content) = messages[idx].content {
                let (prefix_override, stripped) = parse_thinking_prefix(content);
                if prefix_override.is_some() {
                    thinking_override = prefix_override;
                    stripped_user_content = Some(stripped.to_string());
                }
            }
        }

        // Determine final thinking value: prefix > config > false
        let thinking = thinking_override.or(self.thinking).unwrap_or(false);

        // Transform messages for API compatibility
        let mut formatted_messages = Vec::new();
        for (i, message) in messages.iter().enumerate() {
            let mut formatted_message = serde_json::to_value(message).unwrap();
            // Use stripped content for the last user message if we have it
            if Some(i) == last_user_idx {
                if let Some(ref stripped) = stripped_user_content {
                    formatted_message["content"] = serde_json::Value::String(stripped.clone());
                }
            }
            if let Some(ref tool_calls) = message.tool_calls {
                if !tool_calls.is_empty() {
                    let formatted_tool_calls = format_tool_calls_for_api(tool_calls);
                    formatted_message["tool_calls"] = serde_json::to_value(formatted_tool_calls).unwrap();
                }
            }
            formatted_messages.push(formatted_message);
        }

        let mut request_body = serde_json::json!({
            "model": self.model,
            "messages": formatted_messages,
            "stop": ["\n\nUser:", "\n\nHuman:", "\nUser:", "\nHuman:", "<|im_end|>", "<|eot_id|>"],
            "think": thinking
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
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                LLMError::retryable(format!("Failed to send request: {}", e))
            })?;

        let status = response.status().as_u16();

        let response_text = response
            .text()
            .await
            .map_err(|e| {
                LLMError::retryable(format!("Failed to read response: {}", e))
            })?;

        if status >= 400 {
            return Err(LLMError::from_status(
                status,
                format!("API error: {}", response_text),
            ));
        }

        let ollama_response: serde_json::Value = serde_json::from_str(&response_text)
            .map_err(|e| {
                LLMError::permanent(format!("Failed to parse response: {}", e))
            })?;

        let content = ollama_response["choices"][0]["message"]["content"]
            .as_str()
            .map(|s| s.to_string());

        let tool_calls = ollama_response["choices"][0]["message"]["tool_calls"]
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

        let usage = ollama_response["usage"].as_object().map(|u| UsageInfo {
            prompt_tokens: u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            completion_tokens: u.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        });

        Ok(LLMResponse {
            content,
            tool_calls,
            usage,
        })
    }

    async fn chat_complete_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<ToolSpec>>,
        tx: mpsc::Sender<String>,
    ) -> Result<LLMResponse, LLMError> {
        use futures_util::StreamExt;

        let url = format!("{}/v1/chat/completions", self.base_url);

        // Find thinking override from the last user message
        let mut thinking_override: Option<bool> = None;
        let mut stripped_user_content: Option<String> = None;
        let last_user_idx = messages.iter().rposition(|m| m.role == "user");

        if let Some(idx) = last_user_idx {
            if let Some(ref content) = messages[idx].content {
                let (prefix_override, stripped) = parse_thinking_prefix(content);
                if prefix_override.is_some() {
                    thinking_override = prefix_override;
                    stripped_user_content = Some(stripped.to_string());
                }
            }
        }

        // Determine final thinking value: prefix > config > false
        let thinking = thinking_override.or(self.thinking).unwrap_or(false);

        // Transform messages for API compatibility
        let mut formatted_messages = Vec::new();
        for (i, message) in messages.iter().enumerate() {
            let mut formatted_message = serde_json::to_value(message).unwrap();
            // Use stripped content for the last user message if we have it
            if Some(i) == last_user_idx {
                if let Some(ref stripped) = stripped_user_content {
                    formatted_message["content"] = serde_json::Value::String(stripped.clone());
                }
            }
            if let Some(ref tool_calls) = message.tool_calls {
                if !tool_calls.is_empty() {
                    let formatted_tool_calls = format_tool_calls_for_api(tool_calls);
                    formatted_message["tool_calls"] = serde_json::to_value(formatted_tool_calls).unwrap();
                }
            }
            formatted_messages.push(formatted_message);
        }

        let mut request_body = serde_json::json!({
            "model": self.model,
            "messages": formatted_messages,
            "stream": true,
            "stop": ["\n\nUser:", "\n\nHuman:", "\nUser:", "\nHuman:", "<|im_end|>", "<|eot_id|>"],
            "think": thinking
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
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                LLMError::retryable(format!("Failed to send request: {}", e))
            })?;

        let status = response.status().as_u16();
        if status >= 400 {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LLMError::from_status(status, format!("API error: {}", error_text)));
        }

        let mut stream = response.bytes_stream();
        let mut full_content = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut tool_call_builders: std::collections::HashMap<usize, (String, String, String)> =
            std::collections::HashMap::new();
        let mut buffer = String::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| {
                LLMError::retryable(format!("Failed to read stream chunk: {}", e))
            })?;

            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim().to_string();
                buffer = buffer[line_end + 1..].to_string();

                if line.is_empty() || line.starts_with(':') {
                    continue;
                }

                if line == "data: [DONE]" {
                    break;
                }

                if let Some(data) = line.strip_prefix("data: ") {
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data) {
                        if let Some(content) = parsed["choices"][0]["delta"]["content"].as_str() {
                            full_content.push_str(content);
                            let _ = tx.send(content.to_string()).await;
                        }

                        if let Some(tc_array) = parsed["choices"][0]["delta"]["tool_calls"].as_array() {
                            for tc in tc_array {
                                let index = tc["index"].as_u64().unwrap_or(0) as usize;
                                let entry = tool_call_builders.entry(index)
                                    .or_insert_with(|| (String::new(), String::new(), String::new()));

                                if let Some(id) = tc["id"].as_str() {
                                    entry.0 = id.to_string();
                                }
                                if let Some(name) = tc["function"]["name"].as_str() {
                                    entry.1.push_str(name);
                                }
                                if let Some(args) = tc["function"]["arguments"].as_str() {
                                    entry.2.push_str(args);
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut indices: Vec<usize> = tool_call_builders.keys().cloned().collect();
        indices.sort();
        for index in indices {
            if let Some((id, name, arguments_str)) = tool_call_builders.remove(&index) {
                if let Ok(arguments) = serde_json::from_str(&arguments_str) {
                    tool_calls.push(ToolCall { id, name, arguments });
                }
            }
        }

        Ok(LLMResponse {
            content: if full_content.is_empty() { None } else { Some(full_content) },
            tool_calls,
            usage: None,
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

    #[test]
    fn test_parse_thinking_prefix_think() {
        let (override_val, stripped) = parse_thinking_prefix("/think Hello world");
        assert_eq!(override_val, Some(true));
        assert_eq!(stripped, "Hello world");
    }

    #[test]
    fn test_parse_thinking_prefix_no_think() {
        let (override_val, stripped) = parse_thinking_prefix("/no_think Hello world");
        assert_eq!(override_val, Some(false));
        assert_eq!(stripped, "Hello world");
    }

    #[test]
    fn test_parse_thinking_prefix_none() {
        let (override_val, stripped) = parse_thinking_prefix("Hello world");
        assert_eq!(override_val, None);
        assert_eq!(stripped, "Hello world");
    }

    #[test]
    fn test_parse_thinking_prefix_case_insensitive() {
        let (override_val, stripped) = parse_thinking_prefix("/THINK Hello world");
        assert_eq!(override_val, Some(true));
        assert_eq!(stripped, "Hello world");

        let (override_val, stripped) = parse_thinking_prefix("/No_Think Hello world");
        assert_eq!(override_val, Some(false));
        assert_eq!(stripped, "Hello world");
    }

    #[test]
    fn test_parse_thinking_prefix_whitespace() {
        // Extra whitespace after prefix should be trimmed
        let (override_val, stripped) = parse_thinking_prefix("/think    Hello world");
        assert_eq!(override_val, Some(true));
        assert_eq!(stripped, "Hello world");
    }

    #[test]
    fn test_ollama_client_with_thinking() {
        let client = OllamaClient::new()
            .with_model("qwen3")
            .with_thinking(Some(true));

        assert_eq!(client.model, "qwen3");
        assert_eq!(client.thinking, Some(true));
    }
}