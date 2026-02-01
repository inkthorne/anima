# ARYA.md — Current Status

## Status: v2.6 Complete ✅

**REPL-as-Frontend architecture shipped!**

### What's Done

1. **Daemon discovery** (`src/discovery.rs`) ✅
   - Scan `~/.anima/agents/*/daemon.pid` files
   - Check process alive
   - Return running agents with socket paths

2. **REPL connects to daemons** ✅
   - Removed in-memory agents from REPL
   - REPL is thin client, connects via sockets
   - Tracks socket connections

3. **Inter-daemon send_message** ✅
   - `DaemonSendMessageTool` uses sockets
   - Daemons message each other directly

4. **DaemonListAgentsTool** ✅
   - Uses discovery module
   - Returns all running agents

5. **Tests** ✅
   - 293 tests passing

## Architecture

```
REPL (thin client)
    │ sockets
    ▼
┌─────────┐  ┌─────────┐
│  arya   │◄─│ gendry  │
│ daemon  │  │ daemon  │
└─────────┘  └─────────┘
    │            │
    └────────────┘
   inter-daemon sockets
```

## Key Files
- `src/discovery.rs` — find running daemons
- `src/repl.rs` — thin client
- `src/tools/send_message.rs` — inter-daemon messaging
- `src/tools/list_agents.rs` — discover agents
- `src/socket_api.rs` — protocol
- `src/daemon.rs` — agent process

## Commands

```bash
# CLI
anima start arya        # Start daemon
anima stop arya         # Stop daemon
anima status            # List running

# REPL
/load arya              # Connect to daemon
/status                 # Show connections
hello @arya             # Send message
@arya @gendry thoughts? # Multi-agent
@all anyone?            # Broadcast
```

## Next Up: v2.7

- Channel integrations (Telegram, Discord)
- Webhook adapter

## Last Updated

2026-02-01 — v2.6 complete
