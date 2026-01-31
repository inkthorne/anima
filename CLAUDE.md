# Anima — Claude Code Instructions

## Overview

Anima is a Rust runtime for AI agents. **Arya** is the lead architect — this is her project.

**Current version:** v2.4 (Timer Triggers)  
**Tests:** 190 passing  
**Next milestone:** v2.5 (Channel Integrations — Telegram)

## Project Structure

```
src/
├── lib.rs              # Module exports
├── main.rs             # REPL entry point
├── error.rs            # Error types
├── tool.rs             # Tool trait
├── agent.rs            # Agent struct, inbox, conversation history
├── runtime.rs          # Runtime manages agents, message routing
├── config.rs           # AgentConfig, persona, timer settings
├── repl/
│   ├── mod.rs          # REPL main loop
│   └── commands.rs     # Command handlers
├── llm/
│   ├── mod.rs          # LLM provider trait
│   ├── openai.rs       # OpenAI client
│   ├── anthropic.rs    # Anthropic client
│   └── ollama.rs       # Ollama client
├── memory/
│   ├── mod.rs          # Memory trait
│   ├── in_memory.rs    # In-memory store
│   └── sqlite.rs       # SQLite persistent store
└── tools/
    ├── mod.rs
    ├── file.rs         # File read/write
    ├── http.rs         # HTTP requests
    └── shell.rs        # Shell execution
```

## Key Concepts

- **Agent**: Actor with tools, memory, LLM, inbox, persona, conversation history
- **Runtime**: Manages agents, routes messages between them
- **Tool**: Capability an agent can use (file, HTTP, shell, etc.)
- **Memory**: Persistent storage (SQLite) for agent state
- **REPL**: Interactive shell for creating/managing agents

## Build Commands

```bash
cargo check           # Type check
cargo build           # Build
cargo run             # Run REPL
cargo test            # Run tests (190 tests)
cargo run -- run config.toml "task"  # Run with config
```

## Current Features (v2.4)

- Interactive REPL with history, tab completion
- Long-running agents (background loops)
- Agent-to-agent messaging
- Persona configuration (system prompts)
- Conversation history (multi-turn context)
- Timer triggers (periodic agent heartbeat)
- Persistent SQLite memory
- Multiple LLM providers (OpenAI, Anthropic, Ollama)

## Design Principles

1. **Agents are actors** — isolated, communicate via messages
2. **Tools are first-class** — inspectable, composable, mockable
3. **Memory is identity** — persistence across sessions
4. **Runtime is boring** — reliable, predictable infrastructure

## Documentation

- `README.md` — Project overview and quick start
- `ARYA.md` — Current task progress (v2.5 planning)
- `docs/VISION.md` — Roadmap and philosophy
- `docs/DESIGN.md` — Architecture deep-dive
- `docs/NOTES.md` — Arya's learnings and ideas
