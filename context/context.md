# Anima Context — Start Here

## Quick Status

| | |
|---|---|
| **Version** | v3.4.32 |
| **Tests** | 469 passing |
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
| `qwen3-next:latest` | 128k+ | JSON-block | General + coding |
| `qwen3:4b` | 32k | Native | Light tasks, tool execution |

## Chat Commands

```bash
anima chat                           # List all chats
anima chat create [name]             # Create (non-interactive)
anima chat new [name]                # Create + enter interactive
anima chat join <name>               # Join existing (shows last 25 msgs)
anima chat send <conv> "msg"         # Fire-and-forget (~50ms)
anima chat view <conv>               # Dump messages to stdout
anima chat view <conv> --limit 10    # Last N messages
anima chat clear <conv>              # Clear all messages (keep conv)
anima chat pause <conv>              # Queue notifications, freeze tools
anima chat resume <conv>             # Process queued
anima chat delete <conv>             # Delete conversation
anima chat cleanup                   # Remove expired
```

**TUI Slash Commands** (inside `chat join`):
- `/clear` — Clear conversation history
- `/pause` — Pause conversation
- `/resume` — Resume conversation
- `/help` — Show available commands
- `/quit` or `/q` — Exit chat

## Agent Commands

```bash
anima start <name|pattern|all>       # Start agents (supports globs)
anima stop <name|pattern|all>        # Stop agents
anima restart <name|pattern|all>     # Restart agents
anima status                         # Show running agents
anima ask <name> "msg"               # One-shot
anima send <name> "msg"              # Send to agent (daemon mode)
anima heartbeat <name>               # Manually trigger heartbeat
```

**Glob examples:** `anima restart all`, `anima start gend*`, `anima stop *`

## Heartbeat (v3.3.0+)

Agents wake up periodically and think proactively.

```toml
# ~/.anima/agents/<name>/config.toml
[heartbeat]
enabled = true
interval = "15m"
```

**Prompt:** `~/.anima/agents/<name>/heartbeat.md`
**Output:** `<agent>-heartbeat` conversation

## Tool Format (JSON-block)

Agents use tools via JSON blocks:
```json
{"tool": "tool_name", "params": {"param1": "value"}}
```

**Core tools:**
- `read_file` — Read file contents
- `write_file` — Write/create files
- `safe_shell` — Run read-only shell commands (ls, grep, find, cat, etc.)
- `http` — Make HTTP requests
- `list_agents` — Discover other agents in the system
- `remember` — Save to persistent semantic memory
- `send_message` — Send message to another agent
- `claude_code` — Delegate coding tasks to Claude Code

## Multi-Agent Communication

Agents can discover and message each other:

```json
{"tool": "list_agents", "params": {}}
→ "Available agents: claude, gendry"

{"tool": "send_message", "params": {"to": "gendry", "message": "..."}}
```

**@mentions:** `@gendry` in a response triggers notification to that agent.
- @mentions inside backticks or code blocks are ignored
- Pause freezes both tools AND @mention forwarding

## Claude Code Tool

Agents can delegate coding tasks to Claude Code.

```json
{"tool": "claude_code", "params": {"task": "Create a hello world", "workdir": "/path"}}
```

**Flow:**
1. Agent uses `claude_code` tool in a conversation
2. Tool launches Claude Code in background, returns task ID
3. Task watcher monitors for completion (every 10s)
4. On completion, posts `@agent` notification to source conversation
5. Agent wakes via @mention, sees result, responds

## Debugging

**`last_turn.json`** — Each agent dumps the exact messages sent to the LLM:
```bash
cat ~/.anima/agents/arya/last_turn.json | jq .
```

Useful for debugging tool loops, history issues, and prompt injection.

## Config Structure

```
~/.anima/
├── conversations.db     # Conversations + claude_code_tasks
├── models/*.toml        # Model definitions
├── tools.toml           # Tool registry
└── agents/
    ├── always.md        # Global always prompt
    └── <name>/
        ├── config.toml  # Agent config
        ├── persona.md   # System prompt
        ├── always.md    # Agent-specific always prompt
        ├── heartbeat.md # Heartbeat prompt (optional)
        ├── memory.db    # Semantic memory
        └── last_turn.json # Debug: last LLM request
```

## Build & Test

```bash
cargo build --release
cargo test
cargo install --path . --force  # Install to ~/.cargo/bin
anima restart all               # After changes
```

## Recent Changes (v3.4.6 → v3.4.32)

- **v3.4.32** — Fix duplicate tool results in conversation history
- **v3.4.29** — `list_agents` tool for agent discovery
- **v3.4.28** — Split always_prompt: memories only on first turn (reduces noise in tool loops)
- **v3.4.27** — Fix history duplication in tool loop
- **v3.4.18** — Store tool results in conversation history
- **v3.4.16** — `remember` tool for explicit memory saving
- **v3.4.15** — /pause and /resume in chat TUI
- **v3.4.12** — Ignore @mentions inside code blocks
- **v3.4.11** — `last_turn.json` debug output
- **v3.4.10** — Slash commands in chat TUI
- **v3.4.7-8** — `anima start/stop all` with glob patterns
- **Enhanced pause** — Full freeze of tools and @mentions
- **Memory CLI** — `anima memory list/search/delete`

## Last Updated

2026-02-03 — v3.4.32: Comprehensive tool loop fixes, multi-agent communication, enhanced debugging.
