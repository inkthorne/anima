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

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Convert a potentially empty `String` to `Option<String>`, returning `None`
/// when the string is empty.
fn none_if_empty(s: String) -> Option<String> {
    if s.is_empty() { None } else { Some(s) }
}

/// Format tool calls for OpenAI-compatible API (arguments as JSON string).
fn format_tool_calls_for_api(tool_calls: &[ToolCall]) -> Vec<Value> {
    tool_calls
        .iter()
        .map(|tc| {
            serde_json::json!({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments.to_string()
                }
            })
        })
        .collect()
}

/// Format tool calls for native Ollama API (arguments as object, not string).
fn format_tool_calls_for_ollama(tool_calls: &[ToolCall]) -> Vec<Value> {
    tool_calls
        .iter()
        .map(|tc| {
            serde_json::json!({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments
                }
            })
        })
        .collect()
}

/// Format messages for the OpenAI API, serialising tool_calls arguments as
/// JSON strings (the format OpenAI expects).
fn format_messages_openai(messages: Vec<ChatMessage>) -> Vec<Value> {
    messages
        .into_iter()
        .map(|msg| {
            let tool_calls = msg.tool_calls.clone();
            let mut val = serde_json::to_value(&msg).unwrap();
            if let Some(tcs) = tool_calls
                && !tcs.is_empty()
            {
                val["tool_calls"] = serde_json::to_value(format_tool_calls_for_api(&tcs)).unwrap();
            }
            val
        })
        .collect()
}

/// Format messages for the native Ollama API.
///
/// Handles two Ollama-specific concerns:
/// - Tool call arguments are kept as objects (not stringified).
/// - The last user message may have its content replaced with a stripped
///   version (after removing a `/think` or `/no_think` prefix).
fn format_messages_ollama(
    messages: &[ChatMessage],
    last_user_idx: Option<usize>,
    stripped_user_content: &Option<String>,
) -> Vec<Value> {
    messages
        .iter()
        .enumerate()
        .map(|(i, msg)| {
            let mut val = serde_json::to_value(msg).unwrap();
            if Some(i) == last_user_idx {
                if let Some(stripped) = stripped_user_content {
                    val["content"] = Value::String(stripped.clone());
                }
            }
            if let Some(tcs) = &msg.tool_calls
                && !tcs.is_empty()
            {
                val["tool_calls"] =
                    serde_json::to_value(format_tool_calls_for_ollama(tcs)).unwrap();
            }
            val
        })
        .collect()
}

/// Format `ToolSpec`s into the OpenAI / Ollama function-tool JSON shape.
fn format_tools_openai(tools: &[ToolSpec]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters
                }
            })
        })
        .collect()
}

/// Format `ToolSpec`s into the Anthropic tool JSON shape (`input_schema`).
fn format_tools_anthropic(tools: &[ToolSpec]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters
            })
        })
        .collect()
}

/// Finalize tool calls that were accumulated during streaming.
///
/// `builders` maps an index to `(id, name, arguments_json_string)`.
/// Entries are emitted in index order; entries whose arguments fail to parse
/// as JSON are silently dropped (matching the original behaviour).
fn finalize_tool_call_builders(
    mut builders: std::collections::HashMap<usize, (String, String, String)>,
) -> Vec<ToolCall> {
    let mut indices: Vec<usize> = builders.keys().cloned().collect();
    indices.sort();

    let mut out = Vec::with_capacity(indices.len());
    for index in indices {
        if let Some((id, name, arguments_str)) = builders.remove(&index)
            && let Ok(arguments) = serde_json::from_str(&arguments_str)
        {
            out.push(ToolCall {
                id,
                name,
                arguments,
            });
        }
    }
    out
}

/// Apply the XML tool-call fallback for Ollama models.
///
/// When the model returned no native tool calls but the content text contains
/// XML-encoded tool invocations, extract them and strip the XML from the
/// content.
fn apply_xml_tool_fallback(
    content: Option<String>,
    tool_calls: Vec<ToolCall>,
) -> (Option<String>, Vec<ToolCall>) {
    if !tool_calls.is_empty() {
        return (content, tool_calls);
    }
    if let Some(text) = &content {
        let (cleaned, xml_calls) = extract_xml_tool_calls(text);
        if !xml_calls.is_empty() {
            return (none_if_empty(cleaned), xml_calls);
        }
    }
    (content, tool_calls)
}

/// Extract the system prompt and non-system messages for the Anthropic API,
/// which requires the system prompt to be a separate top-level field.
fn split_system_prompt(messages: Vec<ChatMessage>) -> (Option<String>, Vec<ChatMessage>) {
    let system_prompt = messages
        .iter()
        .find(|msg| msg.role == "system")
        .and_then(|msg| msg.content.clone());

    let filtered: Vec<ChatMessage> = messages
        .into_iter()
        .filter(|msg| msg.role != "system")
        .collect();

    (system_prompt, filtered)
}

/// Detect a `/think` or `/no_think` prefix in the last user message.
///
/// Returns `(thinking_override, last_user_index, stripped_content)`.
fn detect_thinking_override(
    messages: &[ChatMessage],
) -> (Option<bool>, Option<usize>, Option<String>) {
    let last_user_idx = messages.iter().rposition(|m| m.role == "user");

    if let Some(idx) = last_user_idx
        && let Some(content) = &messages[idx].content
    {
        let (prefix_override, stripped) = parse_thinking_prefix(content);
        if prefix_override.is_some() {
            return (prefix_override, last_user_idx, Some(stripped.to_string()));
        }
    }

    (None, last_user_idx, None)
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String, // "system", "user", "assistant", "tool"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value, // OpenAI uses parameters instead of input_schema
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageInfo>,
    /// Raw HTTP response body for non-streaming calls (debugging parsing issues).
    #[serde(skip)]
    pub raw_body: Option<String>,
}

/// Token usage information from LLM API responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageInfo {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// Prompt evaluation duration in nanoseconds (Ollama-specific).
    /// When KV caching is active, this drops significantly.
    pub prompt_eval_duration_ns: Option<u64>,
    /// Number of prompt tokens served from cache (OpenAI Responses API, LMStudio).
    pub cached_tokens: Option<u32>,
}

// ---------------------------------------------------------------------------
// LLMError
// ---------------------------------------------------------------------------

/// Error from LLM operations with retryability classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMError {
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status_code: Option<u16>,
    #[serde(default)]
    pub is_retryable: bool,
}

impl LLMError {
    pub fn retryable(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            status_code: None,
            is_retryable: true,
        }
    }

    pub fn retryable_with_status(message: impl Into<String>, status: u16) -> Self {
        Self {
            message: message.into(),
            status_code: Some(status),
            is_retryable: true,
        }
    }

    pub fn permanent(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            status_code: None,
            is_retryable: false,
        }
    }

    pub fn permanent_with_status(message: impl Into<String>, status: u16) -> Self {
        Self {
            message: message.into(),
            status_code: Some(status),
            is_retryable: false,
        }
    }

    /// Classify an HTTP status code as retryable or not.
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

// ---------------------------------------------------------------------------
// OpenAI client
// ---------------------------------------------------------------------------

/// Which OpenAI-compatible endpoint to use.
#[derive(Debug, Clone, Default)]
enum ApiStyle {
    #[default]
    Chat,      // /chat/completions
    Responses, // /responses
}

pub struct OpenAIClient {
    client: Client,
    api_key: String,
    base_url: String,
    model: String,
    api_style: ApiStyle,
    temperature: Option<f64>,
    top_p: Option<f64>,
    frequency_penalty: Option<f64>,
}

impl OpenAIClient {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::builder()
                .connect_timeout(std::time::Duration::from_secs(30))
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .expect("failed to build HTTP client"),
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".to_string(),
            model: "gpt-4o".to_string(),
            api_style: ApiStyle::default(),
            temperature: None,
            top_p: None,
            frequency_penalty: None,
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

    pub fn with_api_style(mut self, style: &str) -> Self {
        self.api_style = match style {
            "responses" => ApiStyle::Responses,
            _ => ApiStyle::Chat,
        };
        self
    }

    pub fn with_temperature(mut self, temperature: Option<f64>) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_top_p(mut self, top_p: Option<f64>) -> Self {
        self.top_p = top_p;
        self
    }

    pub fn with_frequency_penalty(mut self, frequency_penalty: Option<f64>) -> Self {
        self.frequency_penalty = frequency_penalty;
        self
    }

    fn inject_sampling_params(&self, request_body: &mut Value) {
        if let Some(t) = self.temperature {
            request_body["temperature"] = serde_json::json!(t);
        }
        if let Some(p) = self.top_p {
            request_body["top_p"] = serde_json::json!(p);
        }
        if let Some(fp) = self.frequency_penalty {
            request_body["frequency_penalty"] = serde_json::json!(fp);
        }
    }
}

/// Parse an OpenAI tool_calls JSON array into `Vec<ToolCall>`.
fn parse_openai_tool_calls(value: &Value) -> Vec<ToolCall> {
    value
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|tc| {
                    let id = tc["id"].as_str()?.to_string();
                    let name = tc["function"]["name"].as_str()?.to_string();
                    let arguments: Value =
                        serde_json::from_str(tc["function"]["arguments"].as_str()?).ok()?;
                    Some(ToolCall {
                        id,
                        name,
                        arguments,
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Extract `UsageInfo` from an OpenAI-shaped usage object.
fn parse_openai_usage(value: &Value) -> Option<UsageInfo> {
    value.as_object().map(|u| UsageInfo {
        prompt_tokens: u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        completion_tokens: u
            .get("completion_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32,
        prompt_eval_duration_ns: None,
        cached_tokens: u
            .get("prompt_tokens_details")
            .and_then(|d| d.get("cached_tokens"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
    })
}

/// Convert `Vec<ChatMessage>` to the Responses API `input` array format.
/// Returns `(instructions, input_array)` — system messages become top-level `instructions`.
fn format_messages_responses(messages: Vec<ChatMessage>) -> (Option<String>, Vec<Value>) {
    let mut instructions: Option<String> = None;
    let mut input = Vec::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                instructions = msg.content;
            }
            "user" | "assistant" => {
                let mut item = serde_json::json!({
                    "type": "message",
                    "role": msg.role,
                });
                if let Some(content) = &msg.content {
                    item["content"] = Value::String(content.clone());
                }
                input.push(item);

                // Emit tool_calls as separate function_call items
                if msg.role == "assistant" {
                    if let Some(tcs) = &msg.tool_calls {
                        for tc in tcs {
                            input.push(serde_json::json!({
                                "type": "function_call",
                                "call_id": tc.id,
                                "name": tc.name,
                                "arguments": tc.arguments.to_string()
                            }));
                        }
                    }
                }
            }
            "tool" => {
                input.push(serde_json::json!({
                    "type": "function_call_output",
                    "call_id": msg.tool_call_id.as_deref().unwrap_or(""),
                    "output": msg.content.as_deref().unwrap_or("")
                }));
            }
            _ => {}
        }
    }

    (instructions, input)
}

/// Format `ToolSpec`s for the Responses API (flat, no nested `function` wrapper).
fn format_tools_responses(tools: &[ToolSpec]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters
            })
        })
        .collect()
}

/// Parse the Responses API `output` array into content + tool calls.
fn parse_responses_output(output: &Value) -> (Option<String>, Vec<ToolCall>) {
    let mut content_parts = Vec::new();
    let mut tool_calls = Vec::new();

    if let Some(items) = output.as_array() {
        for item in items {
            match item["type"].as_str() {
                Some("message") => {
                    if let Some(content_arr) = item["content"].as_array() {
                        for block in content_arr {
                            if let Some(text) = block["text"].as_str() {
                                content_parts.push(text.to_string());
                            }
                        }
                    }
                }
                Some("function_call") => {
                    let id = item["call_id"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string();
                    let name = item["name"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string();
                    let arguments: Value = item["arguments"]
                        .as_str()
                        .and_then(|s| serde_json::from_str(s).ok())
                        .unwrap_or(serde_json::json!({}));
                    tool_calls.push(ToolCall {
                        id,
                        name,
                        arguments,
                    });
                }
                _ => {}
            }
        }
    }

    let content = if content_parts.is_empty() {
        None
    } else {
        Some(content_parts.join(""))
    };
    (content, tool_calls)
}

/// Parse usage from a Responses API response.
fn parse_responses_usage(value: &Value) -> Option<UsageInfo> {
    let u = value.as_object()?;
    Some(UsageInfo {
        prompt_tokens: u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        completion_tokens: u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        prompt_eval_duration_ns: None,
        cached_tokens: u
            .get("input_tokens_details")
            .and_then(|d| d.get("cached_tokens"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
    })
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
        match self.api_style {
            ApiStyle::Responses => self.chat_complete_responses(messages, tools).await,
            ApiStyle::Chat => self.chat_complete_chat(messages, tools).await,
        }
    }

    async fn chat_complete_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<ToolSpec>>,
        tx: mpsc::Sender<String>,
    ) -> Result<LLMResponse, LLMError> {
        match self.api_style {
            ApiStyle::Responses => {
                self.chat_complete_stream_responses(messages, tools, tx)
                    .await
            }
            ApiStyle::Chat => self.chat_complete_stream_chat(messages, tools, tx).await,
        }
    }
}

// -- OpenAI Chat Completions (existing) --
impl OpenAIClient {
    async fn chat_complete_chat(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<ToolSpec>>,
    ) -> Result<LLMResponse, LLMError> {
        let url = format!("{}/chat/completions", self.base_url);

        let formatted_messages = format_messages_openai(messages);

        let mut request_body = serde_json::json!({
            "model": self.model,
            "messages": formatted_messages
        });

        if let Some(tool_list) = tools {
            request_body["tools"] = serde_json::to_value(format_tools_openai(&tool_list)).unwrap();
            request_body["tool_choice"] = serde_json::json!("auto");
        }

        self.inject_sampling_params(&mut request_body);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LLMError::retryable(format!("Failed to send request: {}", e)))?;

        let status = response.status().as_u16();

        let response_text = response
            .text()
            .await
            .map_err(|e| LLMError::retryable(format!("Failed to read response: {}", e)))?;

        if status >= 400 {
            return Err(LLMError::from_status(
                status,
                format!("API error: {}", response_text),
            ));
        }

        let parsed: Value = serde_json::from_str(&response_text)
            .map_err(|e| LLMError::permanent(format!("Failed to parse response: {}", e)))?;

        let message = &parsed["choices"][0]["message"];

        Ok(LLMResponse {
            content: message["content"].as_str().map(|s| s.to_string()),
            tool_calls: parse_openai_tool_calls(&message["tool_calls"]),
            usage: parse_openai_usage(&parsed["usage"]),
            raw_body: Some(response_text),
        })
    }

    async fn chat_complete_stream_chat(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<ToolSpec>>,
        tx: mpsc::Sender<String>,
    ) -> Result<LLMResponse, LLMError> {
        use futures_util::StreamExt;

        let url = format!("{}/chat/completions", self.base_url);

        let formatted_messages = format_messages_openai(messages);

        let mut request_body = serde_json::json!({
            "model": self.model,
            "messages": formatted_messages,
            "stream": true,
            "stream_options": { "include_usage": true }
        });

        if let Some(tool_list) = tools {
            request_body["tools"] = serde_json::to_value(format_tools_openai(&tool_list)).unwrap();
            request_body["tool_choice"] = serde_json::json!("auto");
        }

        self.inject_sampling_params(&mut request_body);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LLMError::retryable(format!("Failed to send request: {}", e)))?;

        let status = response.status().as_u16();
        if status >= 400 {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LLMError::from_status(
                status,
                format!("API error: {}", error_text),
            ));
        }

        let mut stream = response.bytes_stream();
        let mut full_content = String::new();
        let mut tool_call_builders: std::collections::HashMap<usize, (String, String, String)> =
            std::collections::HashMap::new();
        let mut buffer = String::new();
        let mut usage: Option<UsageInfo> = None;
        let mut raw_lines: Vec<String> = Vec::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result
                .map_err(|e| LLMError::retryable(format!("Failed to read stream chunk: {}", e)))?;

            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim().to_string();
                buffer = buffer[line_end + 1..].to_string();

                if !line.is_empty() {
                    raw_lines.push(line.clone());
                }

                if line.is_empty() || line.starts_with(':') {
                    continue;
                }

                if line == "data: [DONE]" {
                    break;
                }

                if let Some(data) = line.strip_prefix("data: ")
                    && let Ok(parsed) = serde_json::from_str::<Value>(data)
                {
                    let delta = &parsed["choices"][0]["delta"];

                    if let Some(content) = delta["content"].as_str() {
                        full_content.push_str(content);
                        let _ = tx.send(content.to_string()).await;
                    }

                    if let Some(tc_array) = delta["tool_calls"].as_array() {
                        for tc in tc_array {
                            let index = tc["index"].as_u64().unwrap_or(0) as usize;
                            let entry = tool_call_builders
                                .entry(index)
                                .or_insert_with(|| (String::new(), String::new(), String::new()));

                            if let Some(id) = tc["id"].as_str() {
                                entry.0 = id.to_string();
                            }
                            if let Some(name) = tc["function"]["name"].as_str() {
                                entry.1 = name.to_string();
                            }
                            if let Some(args) = tc["function"]["arguments"].as_str() {
                                entry.2.push_str(args);
                            }
                        }
                    }

                    if let Some(u) = parse_openai_usage(&parsed["usage"]) {
                        usage = Some(u);
                    }
                }
            }
        }

        Ok(LLMResponse {
            content: none_if_empty(full_content),
            tool_calls: finalize_tool_call_builders(tool_call_builders),
            usage,
            raw_body: if raw_lines.is_empty() { None } else { Some(raw_lines.join("\n")) },
        })
    }
}

// -- OpenAI Responses API --
impl OpenAIClient {
    async fn chat_complete_responses(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<ToolSpec>>,
    ) -> Result<LLMResponse, LLMError> {
        let url = format!("{}/responses", self.base_url);
        let (instructions, input) = format_messages_responses(messages);

        let mut request_body = serde_json::json!({
            "model": self.model,
            "input": input
        });

        if let Some(inst) = instructions {
            request_body["instructions"] = Value::String(inst);
        }

        if let Some(tool_list) = tools {
            request_body["tools"] =
                serde_json::to_value(format_tools_responses(&tool_list)).unwrap();
        }

        self.inject_sampling_params(&mut request_body);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LLMError::retryable(format!("Failed to send request: {}", e)))?;

        let status = response.status().as_u16();
        let response_text = response
            .text()
            .await
            .map_err(|e| LLMError::retryable(format!("Failed to read response: {}", e)))?;

        if status >= 400 {
            return Err(LLMError::from_status(
                status,
                format!("API error: {}", response_text),
            ));
        }

        let parsed: Value = serde_json::from_str(&response_text)
            .map_err(|e| LLMError::permanent(format!("Failed to parse response: {}", e)))?;

        let (content, tool_calls) = parse_responses_output(&parsed["output"]);

        Ok(LLMResponse {
            content,
            tool_calls,
            usage: parse_responses_usage(&parsed["usage"]),
            raw_body: Some(response_text),
        })
    }

    async fn chat_complete_stream_responses(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<ToolSpec>>,
        tx: mpsc::Sender<String>,
    ) -> Result<LLMResponse, LLMError> {
        use futures_util::StreamExt;

        let url = format!("{}/responses", self.base_url);
        let (instructions, input) = format_messages_responses(messages);

        let mut request_body = serde_json::json!({
            "model": self.model,
            "input": input,
            "stream": true
        });

        if let Some(inst) = instructions {
            request_body["instructions"] = Value::String(inst);
        }

        if let Some(tool_list) = tools {
            request_body["tools"] =
                serde_json::to_value(format_tools_responses(&tool_list)).unwrap();
        }

        self.inject_sampling_params(&mut request_body);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LLMError::retryable(format!("Failed to send request: {}", e)))?;

        let status = response.status().as_u16();
        if status >= 400 {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LLMError::from_status(
                status,
                format!("API error: {}", error_text),
            ));
        }

        let mut stream = response.bytes_stream();
        let mut full_content = String::new();
        // Map call_id → (name, arguments_string) for accumulating function calls
        let mut fn_call_builders: std::collections::HashMap<String, (String, String)> =
            std::collections::HashMap::new();
        let mut current_fn_call_id: Option<String> = None;
        let mut buffer = String::new();
        let mut usage: Option<UsageInfo> = None;
        let mut raw_lines: Vec<String> = Vec::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result
                .map_err(|e| LLMError::retryable(format!("Failed to read stream chunk: {}", e)))?;

            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim().to_string();
                buffer = buffer[line_end + 1..].to_string();

                if !line.is_empty() {
                    raw_lines.push(line.clone());
                }

                if line.is_empty() || line.starts_with(':') {
                    continue;
                }

                // Extract event type and data from SSE
                if let Some(data) = line.strip_prefix("data: ")
                    && let Ok(parsed) = serde_json::from_str::<Value>(data)
                {
                    let event_type = parsed["type"].as_str().unwrap_or("");

                    match event_type {
                        "response.output_text.delta" => {
                            if let Some(delta) = parsed["delta"].as_str() {
                                full_content.push_str(delta);
                                let _ = tx.send(delta.to_string()).await;
                            }
                        }
                        "response.function_call_arguments.delta" => {
                            if let Some(delta) = parsed["delta"].as_str() {
                                if let Some(ref call_id) = current_fn_call_id {
                                    if let Some(entry) = fn_call_builders.get_mut(call_id) {
                                        entry.1.push_str(delta);
                                    }
                                }
                            }
                        }
                        "response.function_call_arguments.done" => {
                            if let Some(args) = parsed["arguments"].as_str() {
                                if let Some(ref call_id) = current_fn_call_id {
                                    if let Some(entry) = fn_call_builders.get_mut(call_id) {
                                        entry.1 = args.to_string();
                                    }
                                }
                            }
                        }
                        "response.output_item.added" => {
                            if let Some(item) = parsed["item"].as_object() {
                                if item.get("type").and_then(|v| v.as_str())
                                    == Some("function_call")
                                {
                                    let call_id = item
                                        .get("call_id")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    let name = item
                                        .get("name")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    fn_call_builders
                                        .insert(call_id.clone(), (name, String::new()));
                                    current_fn_call_id = Some(call_id);
                                }
                            }
                        }
                        "response.output_item.done" => {
                            current_fn_call_id = None;
                        }
                        "response.completed" => {
                            if let Some(resp) = parsed["response"].as_object() {
                                if let Some(u) = resp.get("usage") {
                                    usage = parse_responses_usage(u);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Build tool calls from accumulated function call builders
        let tool_calls: Vec<ToolCall> = fn_call_builders
            .into_iter()
            .filter_map(|(call_id, (name, args_str))| {
                let arguments: Value = serde_json::from_str(&args_str).ok()?;
                Some(ToolCall {
                    id: call_id,
                    name,
                    arguments,
                })
            })
            .collect();

        Ok(LLMResponse {
            content: none_if_empty(full_content),
            tool_calls,
            usage,
            raw_body: if raw_lines.is_empty() { None } else { Some(raw_lines.join("\n")) },
        })
    }
}

// ---------------------------------------------------------------------------
// Anthropic client
// ---------------------------------------------------------------------------

pub enum AnthropicAuth {
    ApiKey(String),
    Bearer(String),
}

pub struct AnthropicClient {
    client: Client,
    auth: AnthropicAuth,
    base_url: String,
    model: String,
    max_tokens: u32,
}

impl AnthropicClient {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::builder()
                .connect_timeout(std::time::Duration::from_secs(30))
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .expect("failed to build HTTP client"),
            auth: AnthropicAuth::ApiKey(api_key.into()),
            base_url: "https://api.anthropic.com".to_string(),
            model: "claude-sonnet-4-20250514".to_string(),
            max_tokens: 16384,
        }
    }

    pub fn with_bearer(token: impl Into<String>) -> Self {
        Self {
            client: Client::builder()
                .connect_timeout(std::time::Duration::from_secs(30))
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .expect("failed to build HTTP client"),
            auth: AnthropicAuth::Bearer(token.into()),
            base_url: "https://api.anthropic.com".to_string(),
            model: "claude-sonnet-4-20250514".to_string(),
            max_tokens: 16384,
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

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    fn apply_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.auth {
            AnthropicAuth::ApiKey(key) => req.header("x-api-key", key),
            AnthropicAuth::Bearer(tok) => req
                .header("Authorization", format!("Bearer {}", tok))
                .header("anthropic-beta", "oauth-2025-04-20"),
        }
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

        let (system_prompt, filtered_messages) = split_system_prompt(messages);

        let mut request_body = serde_json::json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": filtered_messages,
        });

        if let Some(system) = system_prompt {
            request_body["system"] = Value::String(system);
        }

        if let Some(tool_list) = tools {
            request_body["tools"] =
                serde_json::to_value(format_tools_anthropic(&tool_list)).unwrap();
        }

        let req = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .json(&request_body);
        let response = self
            .apply_auth(req)
            .send()
            .await
            .map_err(|e| LLMError::retryable(format!("Failed to send request: {}", e)))?;

        let status = response.status().as_u16();

        let response_text = response
            .text()
            .await
            .map_err(|e| LLMError::retryable(format!("Failed to read response: {}", e)))?;

        if status >= 400 {
            return Err(LLMError::from_status(
                status,
                format!("API error: {}", response_text),
            ));
        }

        let parsed: Value = serde_json::from_str(&response_text)
            .map_err(|e| LLMError::permanent(format!("Failed to parse response: {}", e)))?;

        let mut content_text = String::new();
        let mut tool_calls = Vec::new();

        if let Some(content_array) = parsed["content"].as_array() {
            for block in content_array {
                match block["type"].as_str() {
                    Some("text") => {
                        if let Some(text) = block["text"].as_str() {
                            content_text.push_str(text);
                        }
                    }
                    Some("tool_use") => {
                        tool_calls.push(ToolCall {
                            id: block["id"].as_str().unwrap_or_default().to_string(),
                            name: block["name"].as_str().unwrap_or_default().to_string(),
                            arguments: block["input"].clone(),
                        });
                    }
                    _ => {}
                }
            }
        }

        let usage = parsed["usage"].as_object().map(|u| UsageInfo {
            prompt_tokens: u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            completion_tokens: u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            prompt_eval_duration_ns: None,
            cached_tokens: None,
        });

        Ok(LLMResponse {
            content: none_if_empty(content_text),
            tool_calls,
            usage,
            raw_body: Some(response_text),
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

        let (system_prompt, filtered_messages) = split_system_prompt(messages);

        let mut request_body = serde_json::json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": filtered_messages,
            "stream": true
        });

        if let Some(system) = system_prompt {
            request_body["system"] = Value::String(system);
        }

        if let Some(tool_list) = tools {
            request_body["tools"] =
                serde_json::to_value(format_tools_anthropic(&tool_list)).unwrap();
        }

        let req = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .json(&request_body);
        let response = self
            .apply_auth(req)
            .send()
            .await
            .map_err(|e| LLMError::retryable(format!("Failed to send request: {}", e)))?;

        let status = response.status().as_u16();
        if status >= 400 {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LLMError::from_status(
                status,
                format!("API error: {}", error_text),
            ));
        }

        let mut stream = response.bytes_stream();
        let mut full_content = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut current_tool_use: Option<(String, String, String)> = None;
        let mut buffer = String::new();
        let mut raw_lines: Vec<String> = Vec::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result
                .map_err(|e| LLMError::retryable(format!("Failed to read stream chunk: {}", e)))?;

            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim().to_string();
                buffer = buffer[line_end + 1..].to_string();

                if !line.is_empty() {
                    raw_lines.push(line.clone());
                }

                if line.is_empty() || line.starts_with(':') {
                    continue;
                }

                if let Some(data) = line.strip_prefix("data: ")
                    && let Ok(parsed) = serde_json::from_str::<Value>(data)
                {
                    match parsed["type"].as_str().unwrap_or("") {
                        "content_block_start" => {
                            if let Some(block) = parsed["content_block"].as_object()
                                && block.get("type").and_then(|v| v.as_str()) == Some("tool_use")
                            {
                                let id = block
                                    .get("id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                let name = block
                                    .get("name")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                current_tool_use = Some((id, name, String::new()));
                            }
                        }
                        "content_block_delta" => {
                            if let Some(delta) = parsed["delta"].as_object() {
                                let delta_type = delta.get("type").and_then(|v| v.as_str());

                                if delta_type == Some("text_delta")
                                    && let Some(text) = delta.get("text").and_then(|v| v.as_str())
                                {
                                    full_content.push_str(text);
                                    let _ = tx.send(text.to_string()).await;
                                }

                                if delta_type == Some("input_json_delta")
                                    && let Some(partial) =
                                        delta.get("partial_json").and_then(|v| v.as_str())
                                {
                                    if let Some((_, _, ref mut input_str)) = current_tool_use {
                                        input_str.push_str(partial);
                                    }
                                }
                            }
                        }
                        "content_block_stop" => {
                            if let Some((id, name, input_str)) = current_tool_use.take()
                                && !id.is_empty()
                                && !name.is_empty()
                            {
                                let arguments = if input_str.is_empty() {
                                    serde_json::json!({})
                                } else if let Ok(parsed) = serde_json::from_str(&input_str) {
                                    parsed
                                } else {
                                    continue;
                                };
                                tool_calls.push(ToolCall {
                                    id,
                                    name,
                                    arguments,
                                });
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(LLMResponse {
            content: none_if_empty(full_content),
            tool_calls,
            usage: None,
            raw_body: if raw_lines.is_empty() { None } else { Some(raw_lines.join("\n")) },
        })
    }
}

// ---------------------------------------------------------------------------
// Thinking-prefix parsing
// ---------------------------------------------------------------------------

/// Parse thinking prefix from message content.
///
/// Returns (thinking_override, stripped_content):
/// - "/think ..." -> (Some(true), "...")
/// - "/no_think ..." -> (Some(false), "...")
/// - "..." -> (None, "...")
///
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

/// Strip <think>...</think> blocks from model output.
pub fn strip_thinking_tags(content: &str) -> String {
    let re = regex::Regex::new(r"(?s)<think>.*?</think>\s*").unwrap();
    re.replace_all(content, "").trim().to_string()
}

// ---------------------------------------------------------------------------
// Ollama client
// ---------------------------------------------------------------------------

/// Ollama client using native Ollama API.
/// Configure with OLLAMA_HOST env var (defaults to http://localhost:11434)
pub struct OllamaClient {
    client: Client,
    base_url: String,
    model: String,
    thinking: Option<bool>,
    num_ctx: Option<u32>,
}

impl OllamaClient {
    pub fn new() -> Self {
        let base_url =
            std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
        Self {
            client: Client::builder()
                .connect_timeout(std::time::Duration::from_secs(30))
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .expect("failed to build HTTP client"),
            base_url,
            model: "llama3".to_string(),
            thinking: None,
            num_ctx: None,
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

    pub fn with_num_ctx(mut self, num_ctx: Option<u32>) -> Self {
        self.num_ctx = num_ctx;
        self
    }

    /// Build the common parts of an Ollama request body.
    fn build_request_body(
        &self,
        formatted_messages: Vec<Value>,
        stream: bool,
        thinking: bool,
    ) -> Value {
        let mut body = serde_json::json!({
            "model": self.model,
            "messages": formatted_messages,
            "stream": stream,
            "think": thinking
        });

        if let Some(ctx) = self.num_ctx {
            body["options"] = serde_json::json!({ "num_ctx": ctx });
        }

        body
    }
}

impl Default for OllamaClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse tool calls from the native Ollama response format.
///
/// Handles both string and object argument formats.
fn parse_ollama_tool_calls(value: &Value) -> Vec<ToolCall> {
    value
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|tc| {
                    let id = tc["id"]
                        .as_str()
                        .or_else(|| tc["function"]["name"].as_str())
                        .map(|s| s.to_string())?;
                    let name = tc["function"]["name"].as_str()?.to_string();
                    let arguments =
                        if let Some(args_str) = tc["function"]["arguments"].as_str() {
                            serde_json::from_str(args_str).ok()?
                        } else {
                            tc["function"]["arguments"].clone()
                        };
                    Some(ToolCall {
                        id,
                        name,
                        arguments,
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Extract `UsageInfo` from an Ollama response, trying the OpenAI-style
/// `usage` object first, then falling back to native Ollama fields
/// (`prompt_eval_count`, `eval_count`).
fn parse_ollama_usage(response: &Value) -> Option<UsageInfo> {
    let prompt_eval_duration_ns = response["prompt_eval_duration"].as_u64();
    if let Some(usage_obj) = response["usage"].as_object() {
        Some(UsageInfo {
            prompt_tokens: usage_obj
                .get("prompt_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            completion_tokens: usage_obj
                .get("completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            prompt_eval_duration_ns,
            cached_tokens: None,
        })
    } else {
        Some(UsageInfo {
            prompt_tokens: response["prompt_eval_count"].as_u64().unwrap_or(0) as u32,
            completion_tokens: response["eval_count"].as_u64().unwrap_or(0) as u32,
            prompt_eval_duration_ns,
            cached_tokens: None,
        })
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
        let url = format!("{}/api/chat", self.base_url);

        let (thinking_override, last_user_idx, stripped_user_content) =
            detect_thinking_override(&messages);
        let thinking = thinking_override.or(self.thinking).unwrap_or(false);

        let formatted_messages =
            format_messages_ollama(&messages, last_user_idx, &stripped_user_content);

        let mut request_body = self.build_request_body(formatted_messages, false, thinking);

        crate::debug::log_json("OLLAMA REQUEST (chat_complete)", &request_body);

        if let Some(tool_list) = tools {
            request_body["tools"] = serde_json::to_value(format_tools_openai(&tool_list)).unwrap();
            request_body["tool_choice"] = serde_json::json!("auto");
        }

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LLMError::retryable(format!("Failed to send request: {}", e)))?;

        let status = response.status().as_u16();

        let response_text = response
            .text()
            .await
            .map_err(|e| LLMError::retryable(format!("Failed to read response: {}", e)))?;

        if status >= 400 {
            return Err(LLMError::from_status(
                status,
                format!("API error: {}", response_text),
            ));
        }

        let parsed: Value = serde_json::from_str(&response_text)
            .map_err(|e| LLMError::permanent(format!("Failed to parse response: {}", e)))?;

        crate::debug::log_json("OLLAMA RESPONSE (chat_complete)", &parsed);

        let content = parsed["message"]["content"]
            .as_str()
            .map(|s| s.to_string());
        let tool_calls = parse_ollama_tool_calls(&parsed["message"]["tool_calls"]);
        let usage = parse_ollama_usage(&parsed);

        let (content, tool_calls) = apply_xml_tool_fallback(content, tool_calls);

        Ok(LLMResponse {
            content,
            tool_calls,
            usage,
            raw_body: Some(response_text),
        })
    }

    async fn chat_complete_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<ToolSpec>>,
        tx: mpsc::Sender<String>,
    ) -> Result<LLMResponse, LLMError> {
        use futures_util::StreamExt;

        let url = format!("{}/api/chat", self.base_url);

        let (thinking_override, last_user_idx, stripped_user_content) =
            detect_thinking_override(&messages);
        let thinking = thinking_override.or(self.thinking).unwrap_or(false);

        let formatted_messages =
            format_messages_ollama(&messages, last_user_idx, &stripped_user_content);

        let mut request_body = self.build_request_body(formatted_messages, true, thinking);

        crate::debug::log_json("OLLAMA REQUEST (chat_complete_stream)", &request_body);

        if let Some(tool_list) = tools {
            request_body["tools"] = serde_json::to_value(format_tools_openai(&tool_list)).unwrap();
            request_body["tool_choice"] = serde_json::json!("auto");
        }

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LLMError::retryable(format!("Failed to send request: {}", e)))?;

        let status = response.status().as_u16();
        if status >= 400 {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LLMError::from_status(
                status,
                format!("API error: {}", error_text),
            ));
        }

        let mut stream = response.bytes_stream();
        let mut full_content = String::new();
        let mut tool_call_builders: std::collections::HashMap<usize, (String, String, String)> =
            std::collections::HashMap::new();
        let mut buffer = String::new();
        let mut usage: Option<UsageInfo> = None;
        let mut raw_lines: Vec<String> = Vec::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result
                .map_err(|e| LLMError::retryable(format!("Failed to read stream chunk: {}", e)))?;

            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim().to_string();
                buffer = buffer[line_end + 1..].to_string();

                if !line.is_empty() {
                    raw_lines.push(line.clone());
                }

                if line.is_empty() {
                    continue;
                }

                if let Ok(parsed) = serde_json::from_str::<Value>(&line) {
                    if parsed["done"].as_bool() == Some(true) {
                        usage = parse_ollama_usage(&parsed);
                        break;
                    }

                    if let Some(content) = parsed["message"]["content"].as_str()
                        && !content.is_empty()
                    {
                        full_content.push_str(content);
                        let _ = tx.send(content.to_string()).await;
                    }

                    if let Some(tc_array) = parsed["message"]["tool_calls"].as_array() {
                        for tc in tc_array {
                            let index = tc["index"].as_u64().unwrap_or(0) as usize;
                            let entry = tool_call_builders
                                .entry(index)
                                .or_insert_with(|| (String::new(), String::new(), String::new()));

                            if let Some(id) = tc["id"].as_str() {
                                entry.0 = id.to_string();
                            }
                            if let Some(name) = tc["function"]["name"].as_str() {
                                entry.1 = name.to_string();
                            }
                            if let Some(args) = tc["function"]["arguments"].as_str() {
                                entry.2.push_str(args);
                            } else if !tc["function"]["arguments"].is_null() {
                                entry.2 = tc["function"]["arguments"].to_string();
                            }
                        }
                    }
                }
            }
        }

        let tool_calls = finalize_tool_call_builders(tool_call_builders);

        let (content, tool_calls) =
            apply_xml_tool_fallback(none_if_empty(full_content), tool_calls);

        Ok(LLMResponse {
            content,
            tool_calls,
            usage,
            raw_body: if raw_lines.is_empty() { None } else { Some(raw_lines.join("\n")) },
        })
    }
}

// ---------------------------------------------------------------------------
// XML tool-call extraction (fallback for some Ollama models)
// ---------------------------------------------------------------------------

/// Extract tool calls from XML-format text that some models output.
/// Handles the `<function=name><parameter=key>value</parameter></function>` format.
/// Returns extracted tool calls and content with XML blocks stripped.
fn extract_xml_tool_calls(content: &str) -> (String, Vec<ToolCall>) {
    let fn_re =
        regex::Regex::new(r"(?s)<function=(\w+)>(.*?)</function>(?:\s*</tool_call>)?").unwrap();
    let param_re = regex::Regex::new(r"(?s)<parameter=(\w+)>\s*(.*?)\s*</parameter>").unwrap();

    let mut tool_calls = Vec::new();

    for (i, cap) in fn_re.captures_iter(content).enumerate() {
        let name = cap[1].to_string();
        let body = &cap[2];

        let mut params = serde_json::Map::new();
        for param_cap in param_re.captures_iter(body) {
            params.insert(
                param_cap[1].to_string(),
                Value::String(param_cap[2].to_string()),
            );
        }

        tool_calls.push(ToolCall {
            id: format!("xmlcall_{}", i),
            name,
            arguments: Value::Object(params),
        });
    }

    let cleaned = fn_re.replace_all(content, "").to_string();
    let leftover_re = regex::Regex::new(r"\s*</tool_call>\s*").unwrap();
    let cleaned = leftover_re.replace_all(&cleaned, "").trim().to_string();

    (cleaned, tool_calls)
}

// ---------------------------------------------------------------------------
// Claude Code client (routes through `claude -p` CLI)
// ---------------------------------------------------------------------------

/// LLM client that routes calls through the `claude` CLI.
///
/// This lets Anima agents use Anthropic subscription credentials (API keys
/// or OAuth tokens from `anima login`) that are restricted to Claude Code.
/// The CLI handles its own auth, so no API key is needed here.
///
/// **Important:** model configs using this provider must set `tools = false`
/// (JSON-block mode) because native tool calling is not available via the CLI.
pub struct ClaudeCodeClient {
    model: String,
}

impl ClaudeCodeClient {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
        }
    }
}

/// Flatten `ChatMessage`s into a plain-text prompt for the `claude -p` CLI.
///
/// Returns `(system_prompt, user_prompt)`.  The system prompt is extracted
/// via `split_system_prompt()`.  The remaining messages are formatted as:
///
/// - Single user message → returned as-is.
/// - Multi-turn → `[Previous conversation]\nrole: content\n...\n\n[Current message]\ncontent`
fn format_messages_for_cli(messages: Vec<ChatMessage>) -> (Option<String>, String) {
    let (system, msgs) = split_system_prompt(messages);

    if msgs.is_empty() {
        return (system, String::new());
    }

    if msgs.len() == 1 {
        let prompt = msgs[0].content.clone().unwrap_or_default();
        return (system, prompt);
    }

    // Multi-turn: format previous messages as context, last as current
    let last = &msgs[msgs.len() - 1];
    let previous = &msgs[..msgs.len() - 1];

    let mut out = String::from("[Previous conversation]\n");
    for msg in previous {
        let content = msg.content.as_deref().unwrap_or("");
        out.push_str(&format!("{}: {}\n", msg.role, content));
    }
    out.push_str(&format!(
        "\n[Current message]\n{}",
        last.content.as_deref().unwrap_or("")
    ));

    (system, out)
}

#[async_trait]
impl LLM for ClaudeCodeClient {
    fn model_name(&self) -> &str {
        &self.model
    }

    async fn chat_complete(
        &self,
        messages: Vec<ChatMessage>,
        _tools: Option<Vec<ToolSpec>>,
    ) -> Result<LLMResponse, LLMError> {
        use tokio::process::Command;

        let (system, prompt) = format_messages_for_cli(messages);

        let mut cmd = Command::new("claude");
        cmd.args(["-p", "--output-format", "json", "--no-session-persistence"]);
        cmd.args(["--model", &self.model]);
        cmd.args(["--tools", ""]);

        if let Some(ref sys) = system {
            cmd.args(["--system-prompt", sys]);
        }

        cmd.stdin(std::process::Stdio::piped());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let mut child = cmd
            .spawn()
            .map_err(|e| LLMError::permanent(format!("Failed to spawn claude CLI: {}", e)))?;

        // Write prompt to stdin
        if let Some(mut stdin) = child.stdin.take() {
            use tokio::io::AsyncWriteExt;
            let _ = stdin.write_all(prompt.as_bytes()).await;
            let _ = stdin.shutdown().await;
        }

        let output = child
            .wait_with_output()
            .await
            .map_err(|e| LLMError::permanent(format!("Failed to wait for claude CLI: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(LLMError::permanent(format!(
                "claude CLI exited with {}: {}{}",
                output.status,
                stderr,
                if stdout.is_empty() {
                    String::new()
                } else {
                    format!("\nstdout: {}", stdout)
                }
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);

        // Parse JSON response: {"type":"result","result":"...","usage":{...}}
        let parsed: Value = serde_json::from_str(&stdout)
            .map_err(|e| LLMError::permanent(format!("Failed to parse claude CLI output: {}", e)))?;

        let content = parsed["result"].as_str().map(|s| s.to_string());

        let usage = parsed["usage"].as_object().map(|u| UsageInfo {
            prompt_tokens: u
                .get("input_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            completion_tokens: u
                .get("output_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            prompt_eval_duration_ns: None,
            cached_tokens: None,
        });

        if parsed["is_error"].as_bool() == Some(true) {
            return Err(LLMError::permanent(format!(
                "claude CLI returned error: {}",
                content.as_deref().unwrap_or("unknown error")
            )));
        }

        Ok(LLMResponse {
            content,
            tool_calls: Vec::new(),
            usage,
            raw_body: None,
        })
    }

    async fn chat_complete_stream(
        &self,
        messages: Vec<ChatMessage>,
        _tools: Option<Vec<ToolSpec>>,
        tx: mpsc::Sender<String>,
    ) -> Result<LLMResponse, LLMError> {
        use tokio::io::AsyncBufReadExt;
        use tokio::process::Command;

        let (system, prompt) = format_messages_for_cli(messages);

        let mut cmd = Command::new("claude");
        cmd.args([
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--no-session-persistence",
        ]);
        cmd.args(["--model", &self.model]);
        cmd.args(["--tools", ""]);

        if let Some(ref sys) = system {
            cmd.args(["--system-prompt", sys]);
        }

        cmd.stdin(std::process::Stdio::piped());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let mut child = cmd
            .spawn()
            .map_err(|e| LLMError::permanent(format!("Failed to spawn claude CLI: {}", e)))?;

        // Write prompt to stdin
        if let Some(mut stdin) = child.stdin.take() {
            use tokio::io::AsyncWriteExt;
            let _ = stdin.write_all(prompt.as_bytes()).await;
            let _ = stdin.shutdown().await;
        }

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| LLMError::permanent("Failed to capture claude CLI stdout"))?;

        let mut reader = tokio::io::BufReader::new(stdout).lines();
        let mut full_content = String::new();
        let mut usage: Option<UsageInfo> = None;
        let mut is_error = false;

        while let Ok(Some(line)) = reader.next_line().await {
            if line.is_empty() {
                continue;
            }
            if let Ok(parsed) = serde_json::from_str::<Value>(&line) {
                match parsed["type"].as_str() {
                    Some("content_block_delta") => {
                        if let Some(text) = parsed["delta"]["text"].as_str() {
                            full_content.push_str(text);
                            let _ = tx.send(text.to_string()).await;
                        }
                    }
                    Some("result") => {
                        // Final result event — capture usage and any content we missed
                        if let Some(result_text) = parsed["result"].as_str() {
                            if full_content.is_empty() {
                                full_content = result_text.to_string();
                            }
                        }
                        is_error = parsed["is_error"].as_bool() == Some(true);
                        usage = parsed["usage"].as_object().map(|u| UsageInfo {
                            prompt_tokens: u
                                .get("input_tokens")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0)
                                as u32,
                            completion_tokens: u
                                .get("output_tokens")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0)
                                as u32,
                            prompt_eval_duration_ns: None,
                            cached_tokens: None,
                        });
                    }
                    _ => {}
                }
            }
        }

        // Wait for process to finish
        let _ = child.wait().await;

        if is_error {
            return Err(LLMError::permanent(format!(
                "claude CLI returned error: {}",
                if full_content.is_empty() {
                    "unknown error"
                } else {
                    &full_content
                }
            )));
        }

        Ok(LLMResponse {
            content: none_if_empty(full_content),
            tool_calls: Vec::new(),
            usage,
            raw_body: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_tool_format() {
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
        let tool_msg = ChatMessage {
            role: "tool".to_string(),
            content: Some("8.0".to_string()),
            tool_call_id: Some("call_123".to_string()),
            tool_calls: None,
        };

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
        assert!(matches!(client.auth, AnthropicAuth::ApiKey(ref k) if k == "test-key"));
    }

    #[test]
    fn test_anthropic_client_bearer() {
        let client = AnthropicClient::with_bearer("my-token")
            .with_model("claude-sonnet-4-20250514");

        assert_eq!(client.model, "claude-sonnet-4-20250514");
        assert!(matches!(client.auth, AnthropicAuth::Bearer(ref t) if t == "my-token"));
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

    #[test]
    fn test_extract_xml_tool_calls_single() {
        let content = "Let me read that file. \u{26a1}\n\n<function=read_file>\n<parameter=path>\nclaude.md\n</parameter>\n</function>\n</tool_call>";
        let (cleaned, calls) = extract_xml_tool_calls(content);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments["path"], "claude.md");
        assert_eq!(cleaned, "Let me read that file. \u{26a1}");
    }

    #[test]
    fn test_extract_xml_tool_calls_multiple_params() {
        let content = "<function=write_file>\n<parameter=path>test.txt</parameter>\n<parameter=content>hello world</parameter>\n</function>";
        let (cleaned, calls) = extract_xml_tool_calls(content);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "write_file");
        assert_eq!(calls[0].arguments["path"], "test.txt");
        assert_eq!(calls[0].arguments["content"], "hello world");
        assert!(cleaned.is_empty());
    }

    #[test]
    fn test_extract_xml_tool_calls_no_xml() {
        let content = "Just a normal response with no tool calls.";
        let (cleaned, calls) = extract_xml_tool_calls(content);
        assert!(calls.is_empty());
        assert_eq!(cleaned, content);
    }

    #[test]
    fn test_extract_xml_tool_calls_multiple_calls() {
        let content = "<function=read_file>\n<parameter=path>a.txt</parameter>\n</function>\nSome text\n<function=read_file>\n<parameter=path>b.txt</parameter>\n</function>";
        let (_, calls) = extract_xml_tool_calls(content);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].arguments["path"], "a.txt");
        assert_eq!(calls[1].arguments["path"], "b.txt");
    }

    #[test]
    fn test_claude_code_client_new() {
        let client = ClaudeCodeClient::new("sonnet");
        assert_eq!(client.model_name(), "sonnet");
    }

    #[test]
    fn test_format_messages_for_cli_single_user() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some("Hello".to_string()),
            tool_call_id: None,
            tool_calls: None,
        }];
        let (system, prompt) = format_messages_for_cli(messages);
        assert!(system.is_none());
        assert_eq!(prompt, "Hello");
    }

    #[test]
    fn test_format_messages_for_cli_with_system() {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: Some("You are helpful.".to_string()),
                tool_call_id: None,
                tool_calls: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: Some("Hi".to_string()),
                tool_call_id: None,
                tool_calls: None,
            },
        ];
        let (system, prompt) = format_messages_for_cli(messages);
        assert_eq!(system, Some("You are helpful.".to_string()));
        assert_eq!(prompt, "Hi");
    }

    #[test]
    fn test_format_messages_for_cli_multi_turn() {
        let messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: Some("What is 2+2?".to_string()),
                tool_call_id: None,
                tool_calls: None,
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: Some("4".to_string()),
                tool_call_id: None,
                tool_calls: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: Some("And 3+3?".to_string()),
                tool_call_id: None,
                tool_calls: None,
            },
        ];
        let (system, prompt) = format_messages_for_cli(messages);
        assert!(system.is_none());
        assert!(prompt.contains("[Previous conversation]"));
        assert!(prompt.contains("user: What is 2+2?"));
        assert!(prompt.contains("assistant: 4"));
        assert!(prompt.contains("[Current message]"));
        assert!(prompt.contains("And 3+3?"));
    }

    #[test]
    fn test_format_messages_for_cli_empty() {
        let messages: Vec<ChatMessage> = vec![];
        let (system, prompt) = format_messages_for_cli(messages);
        assert!(system.is_none());
        assert!(prompt.is_empty());
    }

    #[test]
    fn test_parse_openai_usage_from_stream_chunk() {
        // Simulates the final SSE chunk when stream_options.include_usage is set
        let chunk: Value = serde_json::from_str(r#"{
            "id": "chatcmpl-abc",
            "choices": [],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 45,
                "total_tokens": 165
            }
        }"#).unwrap();

        let usage = parse_openai_usage(&chunk["usage"]);
        assert!(usage.is_some());
        let u = usage.unwrap();
        assert_eq!(u.prompt_tokens, 120);
        assert_eq!(u.completion_tokens, 45);
    }

    #[test]
    fn test_parse_openai_usage_absent_in_stream_chunk() {
        // Regular content chunk — no usage field
        let chunk: Value = serde_json::from_str(r#"{
            "id": "chatcmpl-abc",
            "choices": [{"delta": {"content": "hi"}}]
        }"#).unwrap();

        let usage = parse_openai_usage(&chunk["usage"]);
        assert!(usage.is_none());
    }
}
