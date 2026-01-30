# Task: v0.6 - Native Anthropic/Claude Client

**Status:** ✅ COMPLETE

Build: `cargo build` passes with only warnings

## What was done

- [x] AnthropicClient struct with client, api_key, base_url, model fields
- [x] Builder methods: new(), with_base_url(), with_model()
- [x] System prompt extracted to top-level field (not in messages)
- [x] Tools formatted with `input_schema` (Anthropic format), reading from `parameters` field
- [x] LLM trait impl with x-api-key auth, anthropic-version header
- [x] Response parsing: extracts text and tool_use blocks, converts to LLMResponse
- [x] AnthropicClient exported from crate
- [x] OpenAIClient struct added (was missing, causing build failure)

## Key Differences from OpenAI (documented)

- Auth: `x-api-key` header (not Bearer)
- Endpoint: `api.anthropic.com/v1/messages`
- System prompt: top-level field, not a message
- Tools: use `input_schema` not `parameters` in API request
- Response: `tool_use` content blocks, not `tool_calls` array
- Tool results: `tool_result` content blocks in user messages
- Requires `anthropic-version` header

## Next Steps (v0.7 ideas)

- Add streaming response support
- Add tool result message handling for Anthropic
- Write integration tests
- Better error handling with retries
