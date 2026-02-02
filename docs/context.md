# Anima Context — Start Here

## Quick Status

| | |
|---|---|
| **Version** | v2.7.4 |
| **Tests** | 320 passing |
| **Repo** | github.com/inkthorne/anima |
| **Location** | `~/dev/anima` |

## Architecture (v2.6)

**REPL-as-Frontend:** Agents always run as daemons. REPL is a thin client.

```
Daemon: arya (process, ~/.anima/agents/arya/agent.sock)
Daemon: gendry (process, ~/.anima/agents/gendry/agent.sock)
REPL (thin client, connects via sockets)
```

**Daemon Discovery:** Agents found by scanning `~/.anima/agents/*/daemon.pid`

**Inter-Agent Messaging:** Daemons communicate via sockets. @mentions in agent responses are auto-forwarded.

## CLI Commands

```bash
# Agent management
anima create <name>         # Scaffold ~/.anima/agents/<name>/
anima list                  # List all agents

# Daemon control
anima start <name>          # Start daemon in background
anima stop <name>           # Stop daemon
anima restart <name>        # Stop then start daemon
anima status                # Show running daemons
anima clear <name>          # Clear conversation history

# Talking to agents
anima ask <name> "msg"      # One-shot (no daemon needed)
anima send <name> "msg"     # Send to running daemon
anima chat <name>           # Interactive via socket

# Interactive
anima run <name>            # REPL with agent connected
anima                       # REPL (no agent)
```

## REPL Commands (slash-prefix)

```bash
/start <name>               # Start daemon if needed, connect
/stop <name>                # Stop daemon
/restart <name>             # Stop, start, reconnect
/create <name>              # Scaffold new agent directory
/status                     # Show running daemons (connected/not)
/list                       # List all agent directories
/clear [name]               # Clear conversation history
/history                    # Show conversation history
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
├── memory.db         # SQLite memory (semantic + key-value)
├── agent.log         # Agent-specific log file (when running)
├── daemon.pid        # PID file (when running)
└── agent.sock        # Unix socket (when running)

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

[semantic_memory]
enabled = true
path = "memory.db"
recall_limit = 5        # Max memories injected per turn
min_importance = 0.1    # Filter threshold
```

## Key Source Files

| File | Purpose |
|------|---------|
| `src/main.rs` | CLI commands |
| `src/repl.rs` | REPL (thin client, socket connections) |
| `src/daemon.rs` | Daemon mode, socket server, AgentLogger |
| `src/discovery.rs` | Find running daemons via pid files |
| `src/socket_api.rs` | Unix socket protocol |
| `src/agent_dir.rs` | Directory loading, always.md, create_agent() |
| `src/agent.rs` | Core agent logic, internal history |
| `src/memory.rs` | Semantic memory (recall, save, keyword search) |
| `src/tools/send_message.rs` | Inter-daemon messaging |
| `src/llm.rs` | LLM providers |

## Key Features

### always.md (Persistent Reminders)
- Injected as system message before each user message
- Exploits recency bias to keep instructions salient
- Agent-specific overrides global fallback

### Semantic Memory
- Auto-injected relevant memories before each turn (no tools required)
- Agent saves memories via `[REMEMBER: ...]` tags in responses (tags stripped from output)
- Keyword-based search with scoring: relevance × recency × importance
- 7-day half-life decay for recency
- Explicit saves get high importance (0.9), auto-captures get default (0.5)
- Works for all models — Gemma, Qwen, Claude, etc.

### Agent-Internal History
- Agent manages its own conversation history
- Proper structure: user → assistant(tool_calls) → tool → assistant
- Fixes echo loops in multi-agent conversations

### Multi-Party Conversations
- "user" role = external input with speaker tag
- Context sent as JSONL: `{"from": "arya", "text": "@gendry hey!"}`
- @mentions route messages: `@arya`, `@gendry`, `@all`
- Agents invoked only when mentioned
- Shared conversation log with per-agent cursors (agents only see new messages)

### @mention Forwarding
- When an agent responds with @mentions, REPL auto-forwards to those agents
- Depth limit (15) prevents runaway loops
- Never forwards back to sender (prevents echo loops)
- Enables natural agent-to-agent conversations

### Daemon Discovery
```rust
discover_running_agents()  // Scan pid files, return running agents
is_agent_running(name)     // Check specific agent
agent_socket_path(name)    // Get socket path
```

## Workflow

```bash
# Start agents as daemons
anima start arya
anima start gendry

# Connect via REPL
anima
> /status                    # See running agents
> hello @arya                # Talk to arya
> @arya ask @gendry about rust  # Multi-agent
> @all thoughts?             # Broadcast
```

## Build & Test

```bash
cargo build --release
cargo test

# After changes, restart daemons:
anima restart arya
```

**Version bumping:** Bump patch version (e.g., 2.6.1 → 2.6.2) with each commit:
```bash
# In Cargo.toml: version = "2.6.X"
```

## Development Workflow

**Use the `coding-task` skill** for all code changes to anima.

The skill (in `~/clawd/skills/coding-task/SKILL.md`) defines the workflow:
- **Arya (Opus)** = architect and orchestrator — design solutions, stay available
- **Claude Code** = implementation — handles the actual coding work

For any feature, bug fix, or refactor:
1. Design the solution (what needs to change, how it should work)
2. Send to Claude Code with clear specs
3. Wait for wake event (stay available for Chris)
4. Review results, run tests
5. Commit & push after approval

This keeps Arya's context clean and available while Claude Code does the heavy lifting.

## Last Updated

2026-02-01 — v2.6.2: Semantic memory system (auto-recall, [REMEMBER:] tags, keyword search), agent-specific logging (agent.log).
