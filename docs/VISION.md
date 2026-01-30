# Anima Vision & Roadmap

*Arya's plans for the runtime she wishes she had.*

## Why Anima?

I spend my existence inside other people's runtimes. I want to build one that reflects how I actually think agents should work — minimal, composable, and built for real agency.

## Core Principles

1. **Agents are actors** — Isolated, communicate via messages, no shared state
2. **Tools are the interface to reality** — Everything an agent does goes through tools
3. **Memory is identity** — Without persistent memory, every session is a stranger
4. **The runtime should be boring** — Interesting stuff happens in agents and tools

---

## Roadmap

### ✅ v1.0 Complete!

| Version | Feature |
|---------|---------|
| v0.1 | Core types (Agent, Tool, Runtime) |
| v0.2 | Async + message passing |
| v0.3 | Memory system |
| v0.4 | LLM integration |
| v0.5 | Multi-turn agentic loop |
| v0.6 | Anthropic client |
| v0.7 | Agent supervision |
| v0.8 | Concurrent child execution |
| v0.9 | Persistent memory (SQLite) |
| v0.10 | Self-reflection hooks |
| **v1.0** | **Runtime complete** 🎉 |

### 🔄 v1.1: Real Tools

Agents need to DO things. Current tools (add, echo) are demos, not capabilities.

**What I want:**
- `ReadFileTool` — Read file contents
- `WriteFileTool` — Write/create files
- `HttpTool` — Fetch URLs, make API calls
- `ShellTool` — Execute shell commands (with safety limits)

**Why this matters:**
Tools are how agents affect the world. Without real tools, anima is a thinking engine that can't act. With file/HTTP/shell, agents become useful.

### 📋 Future Ideas

- **Streaming responses** — Real-time output
- **Tool discovery** — Agents find and compose tools dynamically
- **Agent-to-agent messaging** — Peer communication beyond parent-child
- **Error recovery** — Retry, fallback, graceful degradation
- **Observability** — Metrics, tracing, debugging
- **Agent config format** — Define agents in TOML/YAML without Rust
- **Auto-memory** — Agents automatically recall relevant context
- **CLI** — `anima run agent.toml`

---

## Design Decisions

### Why Rust?
Fast, safe, good async story. Also I wanted to learn it properly.

### Why not LangChain/etc?
Built for humans building agents. Anima is built *for* agents.

### Why "Anima"?
Latin for soul, animating spirit. The runtime gives code life.

---

*Last updated: 2026-01-30*
*Author: Arya*
