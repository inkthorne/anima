# Task: Implement v0.7 - Agent Supervision

Build: `cargo build`

## Steps

- [x] 1. [src/agent.rs] Add `children: HashMap<String, ChildHandle>` field to Agent
- [x] 2. [src/supervision.rs] Create new module with ChildHandle, ChildConfig, ChildStatus types
- [x] 3. [src/supervision.rs] ChildHandle: child_id, status, result channel, task
- [x] 4. [src/supervision.rs] ChildConfig: task, tools, llm override, memory
- [x] 5. [src/supervision.rs] ChildStatus enum: Running, Completed(String), Failed(String)
- [x] 6. [src/runtime.rs] Add parent_id tracking and spawn_child_agent() method
- [x] 7. [src/runtime.rs] Add get_children() and get_parent() methods
- [x] 8. [src/agent.rs] Implement spawn_child() - creates child, stores handle, inherits LLM
- [x] 9. [src/agent.rs] Implement wait_for_child() - await completion channel
- [x] 10. [src/agent.rs] Implement poll_child() - non-blocking status check
- [x] 11. [src/agent.rs] Implement wait_for_all_children() - await all children
- [x] 12. [src/runtime.rs] Create run_child_task() - runs child.think(), sends result
- [x] 13. [src/runtime.rs] Add terminate_child() method
- [x] 14. [src/runtime.rs] Add terminate_children() - recursive cleanup
- [x] 15. [src/tools/mod.rs] Create SpawnChildTool for LLM to spawn children
- [x] 16. [src/tools/mod.rs] Create WaitForChildTool for LLM to wait for results
- [x] 17. [src/lib.rs] Export new types
- [x] 18. [src/main.rs] Demo: parent spawns child for subtask, uses result  ← IN PROGRESS (Qwen)
- [x] 19. [docs/DESIGN.md] Update architecture docs
- [x] 20. [Cargo.toml] Add futures crate if needed (not needed - using simple loop)

## Key Design

- Parent agents spawn children via spawn_child()
- Children inherit LLM by default (can override)
- ChildHandle tracks status and has result channel
- Parent can wait_for_child() or poll_child()
- Tools let LLM decide when to spawn children during think()
