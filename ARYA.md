# Task: Implement anima v0.2 - Async & Messages

Build: `cargo build`

## Steps

- [x] 1. [Cargo.toml] Add tokio dependency with rt-multi-thread and macros features
- [x] 2. [src/error.rs] Add AgentError enum for agent-level errors
- [x] 3. [src/tool.rs] Make Tool::execute async (used async-trait)
- [x] 4. [src/tools/echo.rs] Update EchoTool for async execute
- [x] 5. [src/tools/add.rs] Update AddTool for async execute
- [x] 6. [src/message.rs] Create Message struct for agent communication
- [x] 7. [src/agent.rs] Add inbox (channel receiver), make call_tool async
- [x] 8. [src/runtime.rs] Add message routing, senders stored in Runtime
- [x] 9. [src/main.rs] Update demo to use async (tokio::main)
- [x] 10. Test and verify everything works

## ✅ v0.2 COMPLETE!

## Plan

v0.2 adds:
- Async tool execution using Tokio
- Message type for inter-agent communication
- Basic inbox for agents to receive messages
- Runtime message routing

Keep backwards compatible where possible.
