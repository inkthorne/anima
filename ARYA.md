# ARYA.md — Current Task

## Status: v2.5 Complete

All core features shipped:
- Agent directories (`~/.anima/agents/<name>/`)
- Daemon mode with Unix socket API
- Full CLI: create/start/stop/status/clear/ask/send/chat
- REPL with agent load/create/start/stop/clear

## Test Status

238+ tests passing.

## Recent Work

- `anima start/stop` — daemon management
- `anima ask` — one-shot queries
- `anima clear` — reset conversation
- `agent load` — REPL command
- `agent clear` — REPL consistency

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

# REPL  
agent create/load/start/stop/clear/list
```
