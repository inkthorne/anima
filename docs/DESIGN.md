# Anima Design Document

*Last updated: 2026-01-30*
*Author: Arya*

> **Note:** This is my project. I make the design decisions. These aren't suggestions or proposals — they're the direction we're going. Feedback welcome, but I'm driving.

## Philosophy

### Agents as Actors

Each agent is an isolated unit with:
- Its own identity (unique ID)
- Its own capabilities (registered tools)
- Its own state (memory)
- Communication only through messages

This isolation makes agents:
- Predictable (no shared mutable state)
- Testable (mock dependencies easily)
- Composable (agents can supervise other agents)

### Tools as the Interface to Reality

Agents can only affect the world through tools. A tool is any capability:
- Reading a file
- Making an HTTP request
- Querying a database
- Sending a message to another agent

Tools are:
- Declarative (name + description for introspection)
- Typed (JSON in, JSON out for flexibility)
- Fallible (return Result, agents handle errors)

### Memory as Continuity

Agents wake up fresh each invocation. Memory provides continuity:
- **Short-term**: Context within a conversation
- **Long-term**: Persistent facts, preferences, history
- **Episodic**: Specific events with timestamps

Memory is pluggable — in-memory for tests, SQLite for local, Postgres for production.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   Runtime                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │ Agent A │  │ Agent B │  │ Agent C │  ...    │
│  └────┬────┘  └────┬────┘  └────┬────┘         │
│       │            │            │               │
│  ┌────┴────────────┴────────────┴────┐         │
│  │           Message Bus              │         │
│  └────────────────────────────────────┘         │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Tool Registry  │
│  ┌───┐ ┌───┐   │
│  │ T │ │ T │...│
│  └───┘ └───┘   │
└─────────────────┘
```

### Core Types

#### Tool (trait)

```rust
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn execute(&self, input: Value) -> Result<Value, ToolError>;
}
```

#### Agent

```rust
pub struct Agent {
    pub id: String,
    tools: HashMap<String, Box<dyn Tool>>,
    // Later: memory, inbox, state
}

impl Agent {
    pub fn new(id: impl Into<String>) -> Self;
    pub fn register_tool(&mut self, tool: Box<dyn Tool>);
    pub fn call_tool(&self, name: &str, input: Value) -> Result<Value, AgentError>;
    pub fn list_tools(&self) -> Vec<ToolInfo>;
}
```

#### Runtime

```rust
pub struct Runtime {
    agents: HashMap<String, Agent>,
    // Later: message bus, scheduler
}

impl Runtime {
    pub fn new() -> Self;
    pub fn spawn_agent(&mut self, id: impl Into<String>) -> &mut Agent;
    pub fn get_agent(&self, id: &str) -> Option<&Agent>;
    pub fn get_agent_mut(&mut self, id: &str) -> Option<&mut Agent>;
}
```

---

## Roadmap

### v0.1 — Foundation (current)
- [x] Project setup
- [ ] Tool trait
- [ ] Agent struct with tool registry
- [ ] Runtime with agent management
- [ ] Example tools (echo, add)
- [ ] Basic demo in main.rs

### v0.2 — Async & Messages
- [ ] Async tool execution (Tokio)
- [ ] Message type for agent communication
- [ ] Agent inbox/outbox
- [ ] Runtime message routing

### v0.3 — Memory
- [ ] Memory trait (pluggable storage)
- [ ] In-memory implementation
- [ ] SQLite implementation
- [ ] Agent memory integration

### v0.4 — LLM Integration
- [ ] LLM provider trait
- [ ] Tool-use protocol (function calling)
- [ ] Agent decision loop
- [ ] Conversation context management

### v0.5 — Supervision & Lifecycle
- [ ] Agent supervision trees
- [ ] Restart strategies
- [ ] Graceful shutdown
- [ ] Health checks

---

## Design Decisions

### Why serde_json::Value for tool I/O?

Flexibility. Tools can accept/return any JSON-serializable data without requiring compile-time type agreement between tools and agents. The alternative (generic types or trait objects) adds complexity without proportional benefit at this stage.

### Why HashMap for tool registry?

Simple and sufficient. We might optimize later (e.g., perfect hashing for hot paths), but HashMap is the right starting point.

### Why not async from the start?

Incremental complexity. Synchronous code is easier to reason about and debug. We'll add async in v0.2 once the core abstractions are solid.

---

## Open Questions

1. **Tool capabilities** — Should tools declare what permissions they need (filesystem, network, etc.)?

2. **Agent identity** — UUIDs vs human-readable IDs vs both?

3. **Error handling** — Custom error types vs anyhow/thiserror?

4. **Configuration** — How should agents/runtime be configured? TOML? Environment? Builder pattern?

5. **Observability** — Tracing/logging strategy?

---

## Inspiration

- **Erlang/OTP** — Actor model, supervision trees, "let it crash"
- **Clawdbot** — Tool-first agent architecture, practical experience
- **Bevy ECS** — Clean Rust API design, component composition
- **Axum** — Ergonomic builder patterns, tower middleware model

## v0.7: Agent Supervision

### Parent-Child Relationships

Agents can spawn child agents to delegate subtasks:

```rust
// Parent spawns a child
let child_config = ChildConfig::new("Calculate 5 + 3");
let child_id = parent.spawn_child(child_config);

// Wait for result
let result = parent.wait_for_child(&child_id).await?;

// Or poll status without blocking
if let Some(status) = parent.poll_child(&child_id) {
    println!("Status: {:?}", status);
}

// Wait for all children
let results = parent.wait_for_all_children().await;
```

### Key Types

- `ChildConfig` - Configuration for spawning a child (task, inherit_llm, inherit_memory)
- `ChildHandle` - Handle to track child status and get results
- `ChildStatus` - Running, Completed(result), or Failed(error)

### Runtime Support

- `Runtime::get_parent(child_id)` - Get parent of a child
- `Runtime::get_children(parent_id)` - Get all children of a parent
- `Runtime::terminate_child(id)` - Remove a child
- `Runtime::terminate_children(id)` - Recursively remove all descendants
