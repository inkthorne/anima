# Anima - Claude Code Instructions

## Project Overview

Anima is a lightweight Rust runtime for AI agents. This is Arya's project — she's the lead architect.

## Structure

```
src/
├── lib.rs          # Module exports
├── error.rs        # ToolError enum
├── tool.rs         # Tool trait
├── agent.rs        # Agent struct with tool registry
├── runtime.rs      # Runtime manages agents
├── main.rs         # Demo binary
└── tools/
    ├── mod.rs
    ├── echo.rs     # EchoTool
    └── add.rs      # AddTool
```

## Current State

v0.9 — Persistent Memory (SQLite):
- Tool trait, Agent, Runtime, example tools
- Async execution (Tokio), message passing
- Memory system with InMemoryStore + SqliteMemory
- LLM integration (OpenAI-compatible, Anthropic)
- Multi-turn agentic loop
- Agent supervision (parent/child)
- Concurrent child execution
- **NEW: SQLite-backed persistent memory** — agents survive restarts

## Build Commands

```bash
cargo check    # Type check
cargo build    # Build
cargo run      # Run demo
cargo test     # Run tests (none yet)
```

## Design Principles

1. Tools are first-class citizens
2. Agents are isolated actors
3. Memory provides continuity
4. Runtime should be boring (reliable)

## Documentation

- `README.md` — Project overview
- `docs/DESIGN.md` — Architecture and roadmap
- `docs/NOTES.md` — Arya's thinking and learnings
- `ARYA.md` — Current task progress
