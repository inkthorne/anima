# Anima Context — Start Here

## Quick Status

| | |
|---|---|
| **Version** | v3.4.5 |
| **Tests** | 428+ passing |
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
anima heartbeat <name>               # Manually trigger heartbeat
```

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

## Claude Code Tool (v3.4.0+)

Agents can delegate coding tasks to Claude Code.

**Tool call format:**
```json
{"tool": "claude_code", "params": {"task": "Create a hello world", "workdir": "/path"}}
```

**Flow:**
1. Agent uses `claude_code` tool in a conversation
2. Tool launches Claude Code in background, returns task ID
3. Task watcher monitors for completion (every 10s)
4. On completion, posts `@agent` notification to source conversation
5. Agent wakes via @mention, sees result, responds

**Agent config:**
```toml
[tools]
allowed = ["read_file", "write_file", "safe_shell", "claude_code"]
```

## Tool Format (JSON-block)

Agents use tools via JSON blocks:
```json
{"tool": "tool_name", "params": {"param1": "value"}}
```

**Available tools:** `read_file`, `write_file`, `safe_shell`, `http`, `claude_code`

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
        ├── heartbeat.md # Heartbeat prompt (optional)
        └── memory.db    # Semantic memory
```

## Build & Test

```bash
cargo build --release
cargo test
anima restart all  # After changes
```

## Last Updated

2026-02-02 — v3.4.5: Claude Code tool with source conversation notifications, chat join shows history, chat clear command, real-time message display in interactive chat.
