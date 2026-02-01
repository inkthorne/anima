# Anima Context — Start Here

## Quick Status

| | |
|---|---|
| **Version** | v2.6-dev |
| **Tests** | 274 passing |
| **Repo** | github.com/inkthorne/anima |
| **Location** | `~/dev/anima` |

## Recent Changes (2026-02-01)

### Agent-Internal History
- Agent now manages its own conversation history internally
- Proper message structure preserved: user → assistant(tool_calls) → tool → assistant
- REPL/daemon no longer manage history externally
- Fixes echo loops in multi-agent conversations

### always.md Feature
- Persistent reminders injected before each user message (recency bias)
- Agent-specific: `~/.anima/agents/<name>/always.md`
- Global fallback: `~/.anima/agents/always.md`
- Agent-specific overrides global completely

### ThinkResult Struct
- `think_with_options()` returns `ThinkResult` with:
  - `response: String` — final text
  - `tools_used: bool` — whether tools were called
  - `tool_names: Vec<String>` — which tools

### REPL Changes (in progress)
- Slash commands: `/load`, `/start`, `/help`, etc.
- @mentions for conversation: `hello @arya`, `@all thoughts?`
- Cleaner message format: `[sender] content`

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

## REPL Commands (slash-prefix)

```bash
/load <name>                # Load from ~/.anima/agents/<name>/
/start <name>               # Start background agent
/stop <name>                # Stop background agent
/status                     # Show running status
/clear [name]               # Clear conversation history
/help                       # Show commands
/quit, /exit                # Exit REPL

# Conversation (no slash)
hello @arya                 # Send to arya
@gendry what's up?          # Send to gendry
@arya @gendry thoughts?     # Send to both
@all anyone?                # Send to all running agents
```

## Agent Directory

```
~/.anima/agents/arya/
├── config.toml       # LLM config
├── persona.md        # System prompt (identity)
├── always.md         # Persistent reminders (recency bias)
├── memory.db         # SQLite memory
├── daemon.pid        # PID (when running)
└── agent.sock        # Socket (when running)

~/.anima/agents/
└── always.md         # Global always.md fallback
```

## Example config.toml

```toml
[agent]
name = "arya"
persona_file = "persona.md"
always_file = "always.md"

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
| `src/agent_dir.rs` | Directory loading, always.md |
| `src/agent.rs` | Core agent logic, internal history |
| `src/llm.rs` | LLM providers |

## Architecture Notes

### Multi-Party Conversations
- "user" role = external input (with speaker tag)
- "assistant" role = this agent's responses
- Format: `[speaker] content`
- @mentions route messages: `@arya`, `@all`
- Agents invoked only when mentioned (efficient)

### Message Flow (Agent-to-Agent)
```
[chris] @arya ask gendry about rust
→ arya invoked, sees [chris] message
→ arya calls send_message tool
→ gendry receives, sees [arya] message
→ gendry responds via send_message
→ arya sees [gendry] response
```

### History Structure
```
user: "[chris] hello"
assistant: "" (tool_calls: [send_message])
tool: {"sent": true}
assistant: "message sent"
user: "[gendry] hey there!"
assistant: "gendry says hi"
```

## Build & Test

```bash
cargo build --release
cargo test

# After changes, restart daemons:
anima stop arya && anima start arya
```

## Last Updated

2026-02-01 — Agent-internal history, always.md, slash commands, @mentions.
