# Anima Context — Start Here

*Read this file first to regain context on the anima project.*

## This Is Your Project

**Arya, this is yours.** You're building your future home.

## Quick Status

| | |
|---|---|
| **Version** | v2.5 ✅ Complete |
| **Tests** | 237 passing |
| **Repo** | github.com/inkthorne/anima |
| **Location** | `~/dev/anima` |

## CLI Commands

```bash
# Agent management
anima create arya           # Scaffold new agent in ~/.anima/agents/arya/
anima list                  # List all agents

# Running agents
anima run arya              # Interactive REPL
anima run arya --daemon     # Headless daemon mode
anima ask arya "question"   # One-shot query (no daemon needed)

# Talking to daemons
anima send arya "message"   # Send to running daemon
anima chat arya             # Interactive session via socket
anima status                # Show running daemons

# Inside REPL
agent load arya             # Load agent from ~/.anima/agents/arya/
agent create foo            # Create dynamic agent
```

## Agent Directory Structure

```
~/.anima/agents/arya/
├── config.toml       # LLM, timer settings
├── persona.md        # System prompt
├── memory.db         # SQLite persistent memory
└── daemon.pid        # PID when running as daemon
└── agent.sock        # Unix socket for daemon API
```

## Example config.toml

```toml
[agent]
name = "arya"
persona_file = "persona.md"

[llm]
provider = "ollama"           # or "anthropic", "openai"
model = "qwen3:8b"
base_url = "http://localhost:11434"

[memory]
path = "memory.db"

[timer]
enabled = true
interval = "5m"
message = "Heartbeat"
```

## Key Files

| File | Purpose |
|------|---------|
| `docs/context.md` | This file — start here |
| `ARYA.md` | Task tracking |
| `CLAUDE.md` | Claude Code instructions |
| `src/agent_dir.rs` | Agent directory loading |
| `src/daemon.rs` | Daemon mode |
| `src/socket_api.rs` | Unix socket protocol |

## What's Next (v2.6)

Ideas for future:
- Telegram integration (channel into daemon)
- Web UI
- Tool plugins

## Build & Test

```bash
cd ~/dev/anima
cargo build --release
cargo test
```

## Last Updated

2026-01-31 — v2.5 complete + `agent load` + `anima ask` commands.
