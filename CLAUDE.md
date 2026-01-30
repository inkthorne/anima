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

- **v0.1 complete**: Tool trait, Agent, Runtime, example tools, working demo
- **v0.2 planned**: Async execution (Tokio), message passing between agents

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
