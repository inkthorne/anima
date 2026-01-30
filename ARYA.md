# Task: Implement anima v0.2 - Async & Messages

Build: `cargo build`

## Steps

- [ ] 1. [Cargo.toml] Add tokio dependency with rt-multi-thread and macros features
- [ ] 2. [src/error.rs] Add AgentError enum for agent-level errors
- [ ] 3. [src/tool.rs] Make Tool::execute async (return BoxFuture)
- [ ] 4. [src/tools/echo.rs] Update EchoTool for async execute
- [ ] 5. [src/tools/add.rs] Update AddTool for async execute
- [ ] 6. [src/message.rs] Create Message enum for agent communication
- [ ] 7. [src/agent.rs] Add inbox (channel receiver), make call_tool async
- [ ] 8. [src/runtime.rs] Add message routing, make methods async
- [ ] 9. [src/main.rs] Update demo to use async (tokio::main)
- [ ] 10. Test and verify everything works

## Plan

v0.2 adds:
- Async tool execution using Tokio
- Message type for inter-agent communication
- Basic inbox for agents to receive messages
- Runtime message routing

Keep backwards compatible where possible.
