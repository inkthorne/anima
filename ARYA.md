# ARYA.md — Current Task

## Status: v2.5.1 Complete ✅

All core features shipped:
- Agent directories (`~/.anima/agents/<name>/`)
- Daemon mode with Unix socket API
- Full CLI: create/start/stop/status/clear/ask/send/chat
- REPL with slash commands and @mentions (59551fa)

## Recent Work (v2.5.1)

REPL refactor completed:
- All commands require `/` prefix (`/load`, `/start`, `/stop`, `/status`, `/list`, `/history`, `/clear`, `/help`, `/quit`)
- Conversation uses `@mentions` (`@arya hello`, `@all hello`)
- Message format: `[sender] content`
- 288 tests passing

## Test Status

288 tests passing.

## What's Next (v2.6)

Pick a direction:
1. **Telegram** — Channel into daemon
2. **Web UI** — Browser interface
3. **Tool plugins** — Extensible tools
4. **Streaming** — Real-time response output

## Key Commands

```bash
# CLI
anima create/start/stop/status/clear
anima ask/send/chat/run

# REPL (slash commands)
/load <name>     - Load agent from ~/.anima/agents/
/start <name>    - Start background loop
/stop <name>     - Stop background agent
/status          - Show agent status
/list            - List active agents
/history         - Show conversation history
/clear [name]    - Clear history
/help            - Show help
/quit            - Exit REPL

# REPL (conversation)
@arya hello      - Message specific agent
@all hello       - Message all running agents
hello            - Message single loaded agent
```
