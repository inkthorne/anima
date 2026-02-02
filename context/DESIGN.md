# Anima Design Document

*Last updated: 2026-02-01*  
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
┌─────────────────────────────────────────────────────┐
│                      Runtime                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐             │
│  │ Agent A │  │ Agent B │  │ Agent C │  ...        │
│  └────┬────┘  └────┬────┘  └────┬────┘             │
│       │            │            │                   │
│  ┌────┴────────────┴────────────┴────┐             │
│  │           Message Bus              │             │
│  └────────────────────────────────────┘             │
└─────────────────────────────────────────────────────┘
         │                    ▲
         ▼                    │
┌─────────────────┐    ┌──────────────┐
│  Tool Registry  │    │   Channels   │
│  ┌───┐ ┌───┐   │    │  (Telegram)  │
│  │ T │ │ T │...│    │  (Discord)   │
│  └───┘ └───┘   │    │  (Webhooks)  │
└─────────────────┘    └──────────────┘
```

### Core Types

#### Tool (trait)

```rust
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn schema(&self) -> Value;
    async fn execute(&self, input: Value) -> Result<Value, ToolError>;
}
```

#### Agent

```rust
pub struct Agent {
    pub id: String,
    config: AgentConfig,
    tools: HashMap<String, Arc<dyn Tool>>,
    memory: Box<dyn Memory>,
    llm: Box<dyn LlmProvider>,
    inbox: mpsc::Receiver<Message>,
    conversation: Vec<Message>,
}
```

Key capabilities:
- `think(task)` — Process a task with LLM, use tools as needed
- `drain_inbox()` — Pull pending messages into context
- `spawn_child(config)` — Create a child agent for subtasks
- Background loop mode for long-running agents

#### Runtime

```rust
pub struct Runtime {
    agents: HashMap<String, Arc<Mutex<Agent>>>,
    senders: HashMap<String, mpsc::Sender<Message>>,
    running_agents: HashMap<String, AbortHandle>,
}
```

Key capabilities:
- `spawn_agent(config)` — Create and register an agent
- `send_message(from, to, content)` — Route messages between agents
- `start_agent(id)` — Run agent in background loop
- `stop_agent(id)` — Stop a running agent

---

## Current Capabilities (v2.4)

| Feature | Status |
|---------|--------|
| Tool execution | ✅ Async, with retry/backoff |
| Agent messaging | ✅ Inbox + drain on think |
| Memory | ✅ SQLite persistent, in-memory for tests |
| LLM providers | ✅ OpenAI, Anthropic, Ollama |
| REPL | ✅ History, tab completion, streaming |
| Long-running | ✅ Background loops, start/stop |
| Persona | ✅ System prompts, personality |
| Conversation | ✅ Multi-turn history |
| Timers | ✅ Periodic triggers |
| Channels | 🔜 v2.5 — Telegram first |

---

## Design Decisions

### Why serde_json::Value for tool I/O?

Flexibility. Tools can accept/return any JSON-serializable data without requiring compile-time type agreement between tools and agents.

### Why mpsc channels for inbox?

Simple, efficient, built into tokio. Agents don't need complex pub/sub — just a queue of pending messages.

### Why Arc<Mutex<Agent>> in runtime?

Agents need to be accessible from multiple contexts (REPL commands, message routing, background tasks). Arc<Mutex> provides safe shared access.

### Why SQLite for memory?

Good enough for single-machine agents. Embedded, no server needed, survives restarts. Can add Postgres later for distributed setups.

---

## Agent Supervision

Agents can spawn child agents to delegate subtasks:

```rust
// Parent spawns a child
let child_config = ChildConfig::new("Calculate 5 + 3");
let child_id = parent.spawn_child(child_config);

// Wait for result
let result = parent.wait_for_child(&child_id).await?;

// Or poll without blocking
if let Some(status) = parent.poll_child(&child_id) {
    println!("Status: {:?}", status);
}
```

---

## Inspiration

- **Erlang/OTP** — Actor model, supervision trees, "let it crash"
- **Clawdbot** — Tool-first agent architecture, practical experience
- **Bevy ECS** — Clean Rust API design
- **Axum** — Ergonomic builder patterns
