# Task: Implement v0.5 - Multi-turn Agentic Loop

Build: `cargo build`

## Steps

- [x] 1. [src/llm.rs] Update `ChatMessage` to support assistant tool_calls - add optional `tool_calls: Option<Vec<ToolCall>>` field, make `content` optional
- [x] 2. [src/llm.rs] Update OpenAI serialization - assistant messages with tool_calls, tool response messages with role "tool", use skip_serializing_if
- [x] 3. [src/agent.rs] Add `ThinkOptions` struct - max_iterations (default 10), optional system_prompt
- [x] 4. [src/agent.rs] Rewrite `think()` for agentic loop - message history, loop until no tool_calls or max iterations
- [x] 5. [src/error.rs] Add `MaxIterationsExceeded` variant to AgentError
- [x] 6. [src/agent.rs] Add `think_with_options()` method, keep `think()` as simple wrapper
- [x] 7. [src/main.rs] Update demo for multi-step reasoning ("Add 5+3, then add 10 to that result")

## Plan

Multi-turn agentic loop: LLM calls tools, sees results, decides next action, repeats until done.
Key changes: ChatMessage supports tool_calls on assistant messages, think() loops with message history.
