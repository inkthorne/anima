# ARYA.md — Current Task

## v2.5: Agent Directories + Daemon Mode

**Goal:** Make agents first-class directory structures that can run as daemons, enabling self-hosted agents.

### The Vision

Instead of creating agents in the REPL, agents are directories:
```
~/.anima/agents/arya/
├── config.toml       # LLM, timer, channel settings
├── persona.md        # System prompt — who I am
├── context.md        # Quick context recovery file
├── memory.db         # SQLite persistent memory
└── state.json        # Runtime state
```

Run with: `anima run arya` or `anima run ~/.anima/agents/arya/`

### Implementation Plan

#### Phase 1: Agent Directory Structure ✅
- [x] Define directory layout and required files
- [x] Create `AgentDir` struct to represent an agent directory
- [x] Add `config.toml` schema (name, llm, persona_file, timer, etc.)
- [x] Load agent from directory instead of creating in REPL

#### Phase 2: Config-Driven Agent Creation ✅
- [x] Parse `config.toml` with serde
- [x] Load persona from `persona.md` file
- [x] Initialize memory from `memory.db` path in config
- [x] Support environment variable substitution (e.g., `${ANTHROPIC_API_KEY}`)

#### Phase 3: CLI Commands ✅
- [x] `anima run <agent>` — Load and run agent from directory
- [x] `anima create <name>` — Scaffold new agent directory
- [x] `anima list` — List agents in ~/.anima/agents/

#### Phase 4: Daemon Mode ✅
- [x] Run without REPL (headless)
- [x] Timer triggers continue running
- [x] Expose local API (Unix socket)
- [x] Clean shutdown on SIGTERM/SIGINT

#### Phase 5: Client Commands ✅
- [x] `anima send <agent> "message"` — Send message to running daemon
- [x] `anima chat <agent>` — Interactive session with running daemon
- [x] `anima status` — Show running agents

### Example config.toml

```toml
[agent]
name = "arya"
persona = "persona.md"

[llm]
provider = "anthropic"
model = "claude-sonnet-4"
api_key = "${ANTHROPIC_API_KEY}"

[memory]
path = "memory.db"

[timer]
enabled = true
interval = "5m"
message = "Heartbeat — check for anything interesting"

# Future
[api]
enabled = true
socket = "~/.anima/agents/arya/arya.sock"
```

### Design Notes

- Agent directory IS the agent — portable, self-contained
- `context.md` pattern from project docs applies to agents too
- Daemon exposes local API; CLI/TUI connects to it
- Same API enables future channel integrations (Telegram, etc.)

### Success Criteria

- [x] Can run `anima run arya` and have a working agent
- [x] Agent loads config, persona, memory from directory
- [x] Timer triggers work in daemon mode
- [x] Can send messages via `anima send arya "hello"`
- [x] Clean shutdown preserves state
