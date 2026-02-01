# ARYA.md — Current Task

## Status: Idle

v2.5 is **complete**. All phases shipped:

- ✅ Phase 1: Agent directory structure
- ✅ Phase 2: Config-driven agent creation
- ✅ Phase 3: CLI commands (run/create/list)
- ✅ Phase 4: Daemon mode (headless, timers, socket API)
- ✅ Phase 5: Client commands (send/chat/status)

**Bonus features added:**
- ✅ `agent load <name>` in REPL
- ✅ `anima ask <agent> "message"` one-shot queries

## Test Status

237 tests passing.

## What's Next?

Pick a direction for v2.6:

1. **Telegram integration** — Channel messages into daemon
2. **Web UI** — Browser-based chat interface
3. **Tool plugins** — Extensible tool system
4. **Agent-to-agent protocols** — Structured messaging between agents

## Recent Commits

```
a73bfc6 fix: silence unused variable warnings
47c1e67 feat: add 'anima ask <agent> <message>' for one-shot queries
69c7fe7 feat: add 'agent load <name>' command to REPL
29a0cb3 feat: v2.5 Phase 5 — Client commands
971d25e feat: v2.5 Phase 4 — Daemon mode
19a10f5 feat: v2.5 Phase 3 — CLI commands
d203e07 feat: v2.5 Phase 2 — Config-driven agent creation
a89098f feat: v2.5 Phase 1 — Agent directory structure
```
