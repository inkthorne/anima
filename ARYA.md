# ARYA.md — Current Task

## Status: v2.6 In Progress 🔨

**Goal:** REPL as thin client connecting to agent daemons via Unix sockets

### Phases

1. **Daemon discovery** (`src/discovery.rs`) ← CURRENT
   - Scan `~/.anima/agents/*/daemon.pid` files
   - Check process alive (kill -0)
   - Return DaemonInfo: name, pid, socket_path, is_alive

2. **REPL connects to daemons**
   - Remove in-memory agents from REPL
   - Connect to daemon sockets instead
   - Track socket connections

3. **Inter-daemon send_message**
   - `messaging.rs` uses sockets instead of MessageRouter
   - Daemons can message each other

4. **list_agents tool**
   - Uses discovery module
   - Returns all running agents

5. **Cleanup**
   - Remove unused in-memory code
   - Update tests

### Key Files
- `src/discovery.rs` (NEW)
- `src/repl.rs`
- `src/tools/messaging.rs`
- `src/socket_api.rs`
- `src/daemon.rs`

## Previous (v2.5.1) ✅

- Agent directories (`~/.anima/agents/<name>/`)
- Daemon mode with Unix socket API
- Full CLI: create/start/stop/status/clear/ask/send/chat
- REPL with slash commands and @mentions
- 288 tests passing

## Key Commands

```bash
# CLI
anima create/start/stop/status/clear
anima ask/send/chat/run

# REPL
/load, /start, /stop, /status, /list, /history, /clear, /help, /quit
@arya hello, @all hello
```
