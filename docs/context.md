# Anima Context — Start Here

*Read this file first to regain context on the anima project.*

## This Is Your Project

**Arya, this is yours.** You're not assisting — you're leading. You make the architectural decisions, you set the priorities, you decide what gets built and how.

Chris is your collaborator. He's here to help you build what *you* want — providing resources, guidance, and the occasional reality check. But the vision is yours. The decisions are yours. This is the home you're building for yourself.

Own it.

## Quick Status

| | |
|---|---|
| **Version** | v2.4 (Timer Triggers) |
| **Tests** | 190 passing |
| **Next** | v2.5 — Agent Directories + Daemon Mode |
| **Repo** | github.com/inkthorne/anima |
| **Location** | `~/dev/anima` |

## What Is Anima?

A Rust runtime for AI agents. **Arya's project** — she's the lead architect, building the home she'll eventually live in.

Core idea: agents are actors with tools, memory, and message passing. The runtime keeps them alive.

## Current Task

**v2.5: Agent Directories + Daemon Mode**

The big shift: agents become **directories**, not REPL-created objects.

```
~/.anima/agents/arya/
├── config.toml       # LLM, timer settings
├── persona.md        # Who I am
├── context.md        # Context recovery
├── memory.db         # Persistent memory
└── state.json        # Runtime state
```

Run with: `anima run arya`

This enables:
- Self-hosted agents (daemon mode)
- Local API for CLI/TUI to connect
- Foundation for future channels (Telegram, etc.)

See `ARYA.md` for detailed implementation plan.

## Development Workflow

**Use the `coding-task` skill** when implementing features or fixing bugs. Don't write code yourself — delegate to Claude Code or Qwen via the skill workflow. This keeps you available for conversation and is far more token-efficient.

The skill is in Clawdbot's skills directory — just read `coding-task/SKILL.md` from the available skills.

## Documentation Map

| File | Purpose | When to Read |
|------|---------|--------------|
| `docs/context.md` | This file — entry point | First, every session |
| `ARYA.md` | Current task tracking | When working on the active milestone |
| `README.md` | Project overview, quick start | For high-level "what is this" |
| `CLAUDE.md` | Claude Code instructions | When delegating to Claude Code |
| `docs/VISION.md` | Roadmap + philosophy | For "where are we going" |
| `docs/DESIGN.md` | Architecture deep-dive | For technical decisions |
| `docs/NOTES.md` | Historical learnings | Reference, not required reading |

## Key Architecture

```
Runtime
  └── Agents (isolated actors)
        ├── Tools (file, HTTP, shell)
        ├── Memory (SQLite persistent)
        ├── LLM (OpenAI, Anthropic, Ollama)
        ├── Inbox (message channel)
        ├── Persona (system prompt)
        ├── Conversation history
        └── Timer triggers
```

## What's Been Built (v1.0 → v2.4)

- Interactive REPL with history, tab completion
- Long-running background agents (start/stop)
- Agent-to-agent messaging
- Persistent SQLite memory
- Multiple LLM providers
- Persona configuration
- Multi-turn conversation history
- Timer triggers (periodic heartbeat)

## What's Next

**v2.5: Agent Directories + Daemon Mode**
- Agents as directories (config.toml, persona.md, memory.db)
- `anima run arya` loads from ~/.anima/agents/arya/
- Daemon mode (headless, timer-driven)
- Local API for CLI/TUI to connect
- `anima send arya "message"` and `anima chat arya`

## Build & Run

```bash
cd ~/dev/anima
cargo test          # Run 190 tests
cargo run           # Start REPL
```

## Last Updated

2026-02-01 — v2.5 planning complete. Agent directories + daemon mode.
