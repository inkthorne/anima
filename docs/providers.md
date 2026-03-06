# Providers & the LLM Trait

Anima abstracts LLM backends behind a single async trait. Each provider implements this trait, and the daemon selects the right one based on model configuration.

## LLM Trait

```rust
#[async_trait]
pub trait LLM: Send + Sync {
    async fn chat_complete(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<ToolSpec>>,
    ) -> Result<LLMResponse, LLMError>;

    async fn chat_complete_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<ToolSpec>>,
        tx: mpsc::Sender<String>,
    ) -> Result<LLMResponse, LLMError>;

    fn model_name(&self) -> &str;
}
```

- **`chat_complete()`** — Non-streaming completion. Returns the full response when done.
- **`chat_complete_stream()`** — Streaming completion. Sends tokens through the `mpsc::Sender<String>` channel as they arrive, then returns the final assembled `LLMResponse`.
- **`model_name()`** — Returns the model identifier string for observability/logging.

## Core Types

### ChatMessage

```rust
pub struct ChatMessage {
    pub role: String,                       // "system", "user", "assistant", "tool"
    pub content: Option<String>,
    pub tool_call_id: Option<String>,       // Links tool results back to the call
    pub tool_calls: Option<Vec<ToolCall>>,  // Tool calls made by the assistant
}
```

### ToolSpec

Describes a tool the LLM can invoke. Parameters use JSON Schema.

```rust
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,  // JSON Schema object
}
```

### ToolCall

A tool invocation returned by the model.

```rust
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
    pub parse_error: Option<String>,  // Set when arguments failed to parse as JSON
}
```

When `parse_error` is `Some(...)`, the `arguments` field is `Value::Null` and the error message contains both the parse error and the raw string.

### LLMResponse

```rust
pub struct LLMResponse {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<UsageInfo>,
    pub raw_body: Option<String>,    // Raw HTTP response (non-streaming, debugging)
    pub raw_stream: Option<String>,  // Raw SSE byte stream (streaming, debugging)
}
```

### UsageInfo

```rust
pub struct UsageInfo {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub prompt_eval_duration_ns: Option<u64>,  // Ollama-specific (nanoseconds)
    pub cached_tokens: Option<u32>,            // OpenAI Responses API / LMStudio
}
```

### LLMError

```rust
pub struct LLMError {
    pub message: String,
    pub status_code: Option<u16>,
    pub is_retryable: bool,
    pub raw_stream: Option<String>,
}
```

Retryability is classified by `from_status()`: HTTP 408, 429, and 5xx are retryable; everything else is permanent. Constructors: `retryable()`, `permanent()`, `retryable_with_status()`, `permanent_with_status()`, `from_status()`. Use `.with_stream()` to attach raw SSE data for debugging.

## Providers

| Provider | Backend | API Key Env Var | Native Tools |
|---|---|---|---|
| `OpenAIClient` | OpenAI API (or compatible) | `OPENAI_API_KEY` | Yes |
| `AnthropicClient` | Anthropic API | `ANTHROPIC_API_KEY` | Yes |
| `OllamaClient` | Local Ollama | None (`OLLAMA_HOST` for URL) | Yes |
| `ClaudeCodeClient` | `claude` CLI subprocess | None | No (tool-block only) |

---

### OpenAIClient

```rust
pub struct OpenAIClient {
    client: Client,
    api_key: String,
    base_url: String,           // Default: "https://api.openai.com/v1"
    model: String,              // Default: "gpt-4o"
    api_style: ApiStyle,        // Chat or Responses
    temperature: Option<f64>,
    top_p: Option<f64>,
    frequency_penalty: Option<f64>,
}
```

**Constructor & builders:**

```rust
fn new(api_key: impl Into<String>) -> Self
fn with_base_url(self, url: impl Into<String>) -> Self
fn with_model(self, model: impl Into<String>) -> Self
fn with_api_style(self, style: &str) -> Self      // "responses" or "chat" (default)
fn with_temperature(self, temperature: Option<f64>) -> Self
fn with_top_p(self, top_p: Option<f64>) -> Self
fn with_frequency_penalty(self, frequency_penalty: Option<f64>) -> Self
```

**Two API styles:**
- **Chat** (`/chat/completions`) — Standard OpenAI chat completions endpoint. Default.
- **Responses** (`/responses`) — OpenAI Responses API. System messages become top-level `instructions`. Tool calls are emitted as `function_call` items. Streaming handles `response.output_text.delta`, `response.function_call_arguments.delta`, reasoning events, etc.

Both styles support streaming and non-streaming. Sampling parameters (`temperature`, `top_p`, `frequency_penalty`) are injected into every request. The `base_url` can be overridden to point at any OpenAI-compatible API (LMStudio, vLLM, etc.).

---

### AnthropicClient

```rust
pub struct AnthropicClient {
    client: Client,
    auth: AnthropicAuth,        // ApiKey(String) or Bearer(String)
    base_url: String,           // Default: "https://api.anthropic.com"
    model: String,              // Default: "claude-sonnet-4-20250514"
    max_tokens: u32,            // Default: 16384
}
```

**Constructor & builders:**

```rust
fn new(api_key: impl Into<String>) -> Self         // Uses x-api-key header
fn with_bearer(token: impl Into<String>) -> Self    // Uses Authorization: Bearer (OAuth)
fn with_base_url(self, url: impl Into<String>) -> Self
fn with_model(self, model: impl Into<String>) -> Self
fn with_max_tokens(self, max_tokens: u32) -> Self
```

**Provider-specific behavior:**
- System prompts are extracted from messages and sent as the top-level `system` field (Anthropic API requirement via `split_system_prompt()`).
- Tools use `input_schema` instead of `parameters` in the tool definition.
- Tool results come back as `tool_use` content blocks; text comes back as `text` blocks.
- Bearer auth includes the `anthropic-beta: oauth-2025-04-20` header.

**RefreshingAnthropicClient** — An internal wrapper (`pub(crate)`) used when the daemon runs with OAuth tokens from `anima login`. It wraps an `AnthropicClient` behind a `RwLock` and auto-refreshes the access token 5 minutes before expiry.

---

### OllamaClient

```rust
pub struct OllamaClient {
    client: Client,
    base_url: String,           // Default: OLLAMA_HOST env var or "http://localhost:11434"
    model: String,              // Default: "llama3"
    thinking: Option<bool>,     // Enable thinking mode
    num_ctx: Option<u32>,       // Context window size override
}
```

**Constructor & builders:**

```rust
fn new() -> Self
fn with_base_url(self, url: impl Into<String>) -> Self
fn with_model(self, model: impl Into<String>) -> Self
fn with_thinking(self, thinking: Option<bool>) -> Self
fn with_num_ctx(self, num_ctx: Option<u32>) -> Self
```

**Provider-specific behavior:**
- Uses the native Ollama API (`/api/chat`), not the OpenAI-compatible endpoint.
- Tool call arguments are kept as JSON objects (not stringified) in requests.
- **Thinking mode:** Controlled via `thinking` field or per-message `/think` and `/no_think` prefixes. Thinking content is returned as `<think>...</think>` blocks prepended to the response content.
- **XML tool fallback:** If the model returns no native tool calls but the content contains XML-encoded tool invocations (`<function=name>...</function>`), they are automatically extracted via `apply_xml_tool_fallback()`.
- `num_ctx` sets the context window size via `options.num_ctx` in the request body.
- `prompt_eval_duration_ns` is captured in `UsageInfo` for cache-hit diagnostics.

---

### ClaudeCodeClient

```rust
pub struct ClaudeCodeClient {
    model: String,
}
```

**Constructor:**

```rust
fn new(model: impl Into<String>) -> Self
```

**Provider-specific behavior:**
- Routes all calls through the `claude` CLI as a subprocess (`claude -p --output-format json`).
- No API key needed — the CLI handles its own auth (Anthropic subscription credentials).
- **Must use `tools = false`** (tool-block mode) because the CLI does not support native tool calling.
- Multi-turn conversations are flattened into a plain-text prompt: previous messages become `[Previous conversation]\nrole: content\n...`, and the last message is `[Current message]\ncontent`.
- `chat_complete_stream()` is not implemented (returns `LLMError::permanent`).

## Provider Selection

The daemon selects a provider in `create_llm_from_config()` (`daemon.rs`) by matching the `provider` string from the resolved model config:

```rust
match config.provider.as_str() {
    "openai"     => OpenAIClient::new(key).with_model(...).with_base_url(...)...
    "anthropic"  => AnthropicClient::new(key).with_model(...)...
                    // Falls back to RefreshingAnthropicClient with OAuth tokens
    "ollama"     => OllamaClient::new().with_model(...).with_thinking(...)...
    "claude-code"=> ClaudeCodeClient::new(model)
    _            => error
}
```

Configuration lives in model TOML files (`~/.anima/models/*.toml`). Each agent's `config.toml` references a `model_file`, and per-agent overrides (e.g., `base_url`, `temperature`) are merged at resolve time.

## Tool Calling Modes

### Native tools (`tools = true`)

The LLM receives `ToolSpec` objects in the request and returns `ToolCall` entries in the response. Each provider formats tools for its own API:
- OpenAI Chat: `tools[].function.parameters`
- OpenAI Responses: flat `tools[].parameters`
- Anthropic: `tools[].input_schema`
- Ollama: same as OpenAI Chat format

### Tool-block (`tools = false`)

The model outputs tool invocations as XML in its text response:

```xml
<function=tool_name>
<parameter=key>value</parameter>
</function>
```

The daemon parses these with `extract_xml_tool_calls()` and executes them. This is the only mode available for `ClaudeCodeClient`, and it also serves as an automatic fallback for Ollama models that don't support native tool calling.

## Helper Functions

**Message formatting:**
- `format_messages_openai()` — OpenAI Chat format (stringified tool call arguments)
- `format_messages_ollama()` — Ollama native format (object arguments, thinking prefix handling)
- `format_messages_responses()` — OpenAI Responses API format (input/output text blocks, function_call items)
- `format_messages_for_cli()` — Plain-text flattening for `claude -p`

**Tool formatting:**
- `format_tools_openai()` — `function.parameters` wrapper (OpenAI/Ollama)
- `format_tools_anthropic()` — `input_schema` wrapper (Anthropic)
- `format_tools_responses()` — Flat format (OpenAI Responses)

**Parsing:**
- `parse_openai_tool_calls()` — Parse OpenAI `tool_calls` array
- `parse_ollama_tool_calls()` — Parse Ollama tool calls (handles string and object arguments)
- `extract_xml_tool_calls()` — Parse `<function=name>...</function>` XML blocks
- `parse_responses_output()` — Parse Responses API `output` array
