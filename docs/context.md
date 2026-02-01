# Anima Context — Start Here

## Quick Status

| | |
|---|---|
| **Version** | v2.5+ |
| **Tests** | 238+ passing |
| **Repo** | github.com/inkthorne/anima |
| **Location** | `~/dev/anima` |

## CLI Commands

```bash
# Agent management
anima create <name>         # Scaffold ~/.anima/agents/<name>/
anima list                  # List all agents

# Daemon control
anima start <name>          # Start daemon in background
anima stop <name>           # Stop daemon
anima status                # Show running daemons
anima clear <name>          # Clear conversation history

# Talking to agents
anima ask <name> "msg"      # One-shot (no daemon needed)
anima send <name> "msg"     # Send to running daemon
anima chat <name>           # Interactive via socket

# Interactive
anima run <name>            # REPL with agent loaded
anima                       # REPL (no agent)
```

## REPL Commands

```bash
agent create <name>         # Create ephemeral in-memory agent
load <name>                 # Load from ~/.anima/agents/<name>/
start <name>                # Start background agent
stop <name>                 # Stop background agent
agent clear <name>          # Clear conversation history
agent list                  # List agents in session
agent status                # Show running status
<name>: <message>           # Send message to agent
```

## Agent Directory

```
~/.anima/agents/arya/
├── config.toml       # LLM config
├── persona.md        # System prompt
├── memory.db         # SQLite memory
├── daemon.pid        # PID (when running)
└── agent.sock        # Socket (when running)
```

## Example config.toml

```toml
[agent]
name = "arya"
persona_file = "persona.md"

[llm]
provider = "ollama"
model = "qwen3:8b"
base_url = "http://localhost:11434"

[memory]
path = "memory.db"
```

## Key Source Files

| File | Purpose |
|------|---------|
| `src/main.rs` | CLI commands |
| `src/repl.rs` | Interactive REPL |
| `src/daemon.rs` | Daemon mode |
| `src/socket_api.rs` | Unix socket protocol |
| `src/agent_dir.rs` | Directory loading |
| `src/agent.rs` | Core agent logic |
| `src/llm.rs` | LLM providers |

## Build & Test

```bash
cargo build --release
cargo test

# After changes, restart daemons:
anima stop arya && anima start arya
```

## Last Updated

2026-01-31 — v2.5 complete + start/stop/clear/ask commands.
