# Task: Implement v0.4 - LLM Integration

Build: `cargo build`

## Steps

- [x] 1. [src/error.rs] Add `LlmError` — done in llm.rs as simple struct (good enough for now)
- [x] 2. [src/llm.rs] Create `Llm` trait with async `chat_complete()` method
- [x] 3. [src/llm.rs] Define message types: `ChatMessage`, `ToolCall`, `LlmResponse`
- [x] 4. [src/llm/mod.rs] Convert to module directory — SKIPPED, flat structure works
- [x] 5. [src/llm.rs] Implement `OpenAIClient` struct with reqwest HTTP client
- [x] 6. [src/llm.rs] Implement `Llm` trait for `OpenAIClient`
- [x] 7. [src/tool.rs] Add `schema()` method to `Tool` trait returning JSON Schema
- [x] 8. [src/tools/echo.rs] Implement `schema()` for EchoTool
- [x] 9. [src/tools/add.rs] Implement `schema()` for AddTool with properties a, b
- [x] 10. [src/agent.rs] Add `llm: Option<Box<dyn Llm>>` field and `with_llm()` builder method
- [x] 11. [src/agent.rs] Add `list_tools_for_llm()` method returning Vec<ToolSpec>
- [x] 12. [src/lib.rs] Exports — already done
- [x] 13. [src/agent.rs] Implement `think(&mut self, task: &str) -> Result<String, AgentError>` - agentic loop
- [x] 14. [src/main.rs] Update demo: create OpenAIClient, attach to agent, call `think("What is 5 + 3?")`

## Plan

Claude Code analyzed the codebase and designed a modular approach.
Early Qwen run completed LLM trait + OpenAIClient before being killed.
Remaining: Tool schema(), Agent LLM integration, think() loop, demo.
