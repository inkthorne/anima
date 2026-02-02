# Anima Context — Start Here

## Quick Status

| | |
|---|---|
| **Version** | v2.8.0 |
| **Tests** | 378 passing |
| **Repo** | github.com/inkthorne/anima |
| **Location** | `~/dev/anima` |

## Architecture

**REPL-as-Frontend:** Agents always run as daemons. REPL is a thin client.

```
Daemon: arya (process, ~/.anima/agents/arya/agent.sock)
Daemon: gendry (process, ~/.anima/agents/gendry/agent.sock)
REPL (thin client, connects via sockets)
```

## Models

Two models configured on Mojave (Ollama):

| Model | Context | Tools | Use Case |
|-------|---------|-------|----------|
| `gemma-27b` | 128k | JSON-block | General, needs tool teaching |
| `qwen3-coder-30b` | 256k | Native | Coding, large context |

### Model Config Example

```toml
# ~/.anima/models/qwen3-coder-30b.toml
provider = "ollama"
model = "qwen3-coder:30b"
num_ctx = 262144     # 256k context
tools = true         # Native tool calling
always = "Optional model-specific prompt text"
```

## Tool Calling (Hybrid System)

**Keyword Recall:** Tools defined in `~/.anima/tools.toml`. Only relevant tools (3-5) injected per query.

**Two Modes:**
- `tools = true`: Native tool calling — LLM gets ToolSpecs, Agent handles execution
- `tools = false`: JSON-block — Model outputs `{"tool": "x", "params": {...}}`, daemon executes

**Available Tools:** `read_file`, `write_file`, `shell`, `http`

## Key Files

| File | Purpose |
|------|---------|
| `src/daemon.rs` | Daemon, tool execution, always prompt building |
| `src/agent.rs` | Core agent, ThinkOptions.external_tools |
| `src/agent_dir.rs` | Config loading, ResolvedLlmConfig |
| `src/llm.rs` | LLM providers, ToolSpec |
| `src/tool_registry.rs` | Keyword-based tool recall |

## Config Structure

```
~/.anima/
├── models/*.toml        # Model definitions (provider, context, tools, always)
├── tools.toml           # Tool registry for keyword recall
└── agents/
    ├── always.md        # Global always (Memory, Agents sections)
    └── <name>/
        ├── config.toml  # Agent config (references model_file)
        ├── persona.md   # System prompt
        └── memory.db    # Semantic memory
```

## CLI Quick Reference

```bash
anima start/stop/restart <name>   # Daemon control
anima status                      # Show running
anima ask <name> "msg"            # One-shot
anima                             # REPL mode
```

## Build & Test

```bash
cargo build --release
cargo test
anima restart <name>  # After changes
```

## Last Updated

2026-02-02 — v2.8.0: Embedding-based semantic memory (Ollama), safe_shell tool with command allowlist, OLLAMA_HOST env var support. Memory now uses cosine similarity for recall instead of keyword matching.
