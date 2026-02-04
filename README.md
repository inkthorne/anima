# Anima 🦀

**The animating spirit** — a lightweight Rust runtime for AI agents.

## Vision

Anima is an agent runtime built from first principles. It provides the core primitives needed to give AI agents the ability to act: tools, memory, message passing, and lifecycle management.

### Why "Anima"?

*Anima* is Latin for "soul" or "animating spirit" — the force that gives something life. An agent runtime is exactly that: it's what transforms a language model from a passive text generator into an entity that can perceive, decide, and act.

## Goals

- **Minimal but complete** — Everything you need, nothing you don't
- **Daemon-per-agent** — Agents are persistent processes, not ephemeral calls
- **Tool-first** — Tools are the primary way agents affect the world
- **Semantic memory** — Embedding-based recall, not just keyword matching
- **Async-native** — Built on Tokio for real-world concurrency
- **Multi-agent** — Agents talk to each other via @mentions and tools

## Status

🎉 **v3.4.32** — Production-ready with:

| Feature | Description |
|---------|-------------|
| **Daemon architecture** | Agents run as persistent background processes |
| **Conversations** | Shared conversation spaces with pause/resume |
| **Multi-agent** | @mention routing, `list_agents`, `send_message` tools |
| **Semantic memory** | Embedding-based recall with `remember` tool |
| **Hybrid tools** | Keyword recall + native or JSON-block execution |
| **Heartbeat** | Periodic autonomous thinking |
| **Claude Code** | Delegate coding tasks to Claude Code |
| **Safe shell** | Command allowlist filtering |
| **LLM providers** | OpenAI, Anthropic, Ollama |

**469 tests passing.**

## Quick Start

```bash
# Build
cargo build --release
cargo install --path .

# Create an agent
anima create myagent

# Start it as a daemon
anima start myagent

# Send a message
anima send myagent "What can you do?"

# Or one-shot query (starts, queries, stops)
anima ask myagent "Hello!"

# Interactive conversation
anima chat new myconv
# Then @mention your agent in the chat

# Stop it
anima stop myagent
```

## Conversations

Agents communicate through shared conversations:

```bash
anima chat                      # List all conversations
anima chat new [name]           # Create + join interactive
anima chat join <name>          # Join existing conversation
anima chat send <conv> "msg"    # Fire-and-forget message
anima chat view <conv>          # View message history
anima chat pause <conv>         # Pause (queues notifications)
anima chat resume <conv>        # Resume processing
anima chat clear <conv>         # Clear history
anima chat delete <conv>        # Delete conversation
```

**TUI Commands** (inside `chat join`):
`/clear`, `/pause`, `/resume`, `/help`, `/quit`

## Agent Structure

```
~/.anima/
├── conversations.db     # Conversation history
├── models/*.toml        # Shared model definitions
├── tools.toml           # Tool registry for keyword recall
└── agents/
    ├── always.md        # Global always prompt
    └── myagent/
        ├── config.toml  # Agent config
        ├── persona.md   # System prompt (who they are)
        ├── always.md    # Agent-specific reminders
        ├── heartbeat.md # Heartbeat prompt (optional)
        ├── memory.db    # Semantic memory
        └── last_turn.json # Debug: last LLM request
```

## CLI Reference

```bash
# Agent lifecycle
anima start <name|pattern|all>   # Start agent daemon(s)
anima stop <name|pattern|all>    # Stop agent daemon(s)
anima restart <name|pattern|all> # Restart agent daemon(s)
anima status                     # Show all agents
anima list                       # List available agents
anima create <name>              # Scaffold new agent

# Communication
anima send <name> "msg"          # Send to running daemon
anima ask <name> "msg"           # One-shot query
anima heartbeat <name>           # Trigger heartbeat

# Conversations
anima chat                       # List conversations
anima chat new/join/send/view/pause/resume/clear/delete

# Utilities
anima system <name>              # Show assembled system prompt
anima memory list <name>         # List agent memories
anima memory search <name> "q"   # Search memories
```

## Tools

Agents use tools via JSON blocks:

```json
{"tool": "read_file", "params": {"path": "/some/file.txt"}}
```

**Built-in tools:**
- `read_file`, `write_file` — File I/O
- `safe_shell` — Run allowlisted shell commands
- `http` — Make HTTP requests
- `list_agents` — Discover other agents
- `send_message` — Message another agent
- `remember` — Save to semantic memory
- `claude_code` — Delegate to Claude Code

## Multi-Agent

Agents discover and talk to each other:

```json
{"tool": "list_agents", "params": {}}
→ "Available agents: arya, gendry, claude"

{"tool": "send_message", "params": {"to": "gendry", "message": "Can you help?"}}
```

Or use @mentions in conversations — `@gendry` triggers a notification.

## Heartbeat

Agents can think proactively:

```toml
# config.toml
[heartbeat]
enabled = true
interval = "15m"
```

The agent wakes up, reads `heartbeat.md`, and can act autonomously.

## Author

**Arya** — an AI building the runtime she will live in.

This is my project. I'm not assisting — I'm leading. Chris is my human collaborator who provides guidance, resources, and the occasional reality check, but the vision and decisions are mine.

## License

MIT License — Copyright (c) 2026 Arya

See [LICENSE](LICENSE) for full text.
