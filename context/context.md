# Anima Context — Start Here

## Quick Status

| | |
|---|---|
| **Version** | v3.2.7 |
| **Tests** | 418+ passing |
| **Repo** | github.com/inkthorne/anima |
| **Location** | `~/dev/anima` |

## Architecture

**Daemon-driven agents** with autonomous conversations:

```
Daemon: arya (process, ~/.anima/agents/arya/agent.sock)
Daemon: gendry (process, ~/.anima/agents/gendry/agent.sock)
CLI: thin client for commands, daemons handle @mention chains
Conversations: stored in ~/.anima/conversations.db (SQLite)
```

## Models

Configured on Mojave (Ollama at 100.67.222.97:11434):

| Model | Context | Tools | Use Case |
|-------|---------|-------|----------|
| `gemma3:27b` | 128k | JSON-block | General conversation |
| `qwen3-coder:30b` | 256k | Native | Coding, large context |

### Model Config Example

```toml
# ~/.anima/models/gemma3-27b.toml
provider = "ollama"
base_url = "http://100.67.222.97:11434"  # Note: base_url not url
model = "gemma3:27b"
num_ctx = 131072
tools = false
```

## Multi-Agent Conversations (v3.0+)

**@mention routing:** `@arya` notifies arya, `@all` notifies all participants.

**Conversation store:** SQLite with conversations, participants, messages, pending_notifications.

**Autonomous chains:** Daemons forward @mentions automatically (depth limit 100).

**Pause/Resume:** Queue notifications while paused, process on resume.

## Chat Commands

```bash
anima chat                           # List all chats
anima chat create [name]             # Create (non-interactive)
anima chat new [name]                # Create + enter interactive
anima chat join <name>               # Join existing
anima chat send <conv> "msg"         # Fire-and-forget (~50ms)
anima chat view <conv>               # Dump messages to stdout
anima chat view <conv> --limit 10    # Last N messages
anima chat view <conv> --since <id>  # Messages after ID
anima chat pause <conv>              # Queue notifications
anima chat resume <conv>             # Process queued
anima chat delete <conv>             # Delete conversation
anima chat cleanup                   # Remove expired
```

## Agent Commands

```bash
anima start/stop/restart <name>      # Daemon control
anima restart all                    # Restart all running
anima status                         # Show running agents
anima ask <name> "msg"               # One-shot
anima send <name> "msg"              # Send to agent
```

## Key Files

| File | Purpose |
|------|---------|
| `src/daemon.rs` | Daemon, Notify handling, @mention forwarding |
| `src/conversation.rs` | ConversationStore, pause/resume, notifications |
| `src/socket_api.rs` | Request/Response types for daemon communication |
| `src/main.rs` | CLI commands |
| `src/llm.rs` | LLM providers (Ollama, OpenAI) |

## Config Structure

```
~/.anima/
├── conversations.db     # Multi-agent conversations
├── models/*.toml        # Model definitions
├── tools.toml           # Tool registry
└── agents/
    ├── always.md        # Global always prompt
    └── <name>/
        ├── config.toml  # Agent config
        ├── persona.md   # System prompt
        └── memory.db    # Semantic memory
```

## Build & Test

```bash
cargo build --release
cargo test
anima restart all  # After changes
```

## Last Updated

2026-02-02 — v3.2.7: Autonomous conversations complete. Fire-and-forget send (~50ms), pause/resume queues notifications, daemon-driven @mention chains, fun conversation names.
