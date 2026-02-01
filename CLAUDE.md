# Anima — Claude Code Instructions

## Overview

Anima is a Rust runtime for AI agents. **Arya** is the lead architect — this is her project.

**Current version:** v2.6 (Daemon Architecture + @mention Forwarding)  
**Tests:** 288 passing  
**Next milestone:** v2.7 (Channel Integrations — Telegram, Discord)

## Quick Context

**Read `docs/context.md` first** — it has the full architecture and command reference.

## Project Structure

```
src/
├── lib.rs              # Module exports
├── main.rs             # CLI entry point
├── repl.rs             # REPL (thin client, socket connections)
├── daemon.rs           # Daemon mode, socket server
├── discovery.rs        # Find running daemons via pid files
├── socket_api.rs       # Unix socket protocol
├── agent_dir.rs        # Directory loading, always.md, create_agent()
├── agent.rs            # Core agent logic, internal history
├── error.rs            # Error types
├── tool.rs             # Tool trait
├── config.rs           # AgentConfig, persona settings
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
    ├── shell.rs        # Shell execution
    └── send_message.rs # Inter-daemon messaging
```

## Architecture (v2.6)

**REPL-as-Frontend:** Agents always run as daemons. REPL is a thin client.

```
~/.anima/agents/arya/     # Agent directory
├── config.toml           # LLM config
├── persona.md            # System prompt
├── always.md             # Persistent reminders
├── daemon.pid            # PID (when running)
└── agent.sock            # Unix socket (when running)
```

## Build Commands

```bash
cargo check           # Type check
cargo build --release # Build
cargo test            # Run tests (288 tests)
anima                 # Run REPL
anima start arya      # Start agent daemon
anima restart arya    # Restart agent daemon
```

## Current Features (v2.6)

- **Daemon architecture** — agents run as background processes
- **REPL thin client** — connects to daemons via Unix sockets
- **@mention forwarding** — agent responses with @mentions auto-forward
- **CLI/REPL parity** — same commands in both (create, start, stop, restart)
- **Persistent reminders** — always.md injected before each message
- **Multi-agent conversations** — @mentions route between agents
- **Depth-limited forwarding** — prevents runaway loops (max 15 hops)
- **Multiple LLM providers** — OpenAI, Anthropic, Ollama

## Key Design Decisions

1. **Agents are daemons** — persistent processes, not ephemeral
2. **REPL is thin** — just a socket client, no agent logic
3. **@mentions are routing** — agents talk to each other via @mentions
4. **always.md exploits recency bias** — keeps instructions salient
5. **Never forward to sender** — prevents echo loops

## Documentation

- **`docs/context.md`** — **Start here** — full context and commands
- `ARYA.md` — Current task progress (if exists)
- `docs/VISION.md` — Roadmap and philosophy
- `docs/DESIGN.md` — Architecture deep-dive
