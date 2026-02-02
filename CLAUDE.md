# Anima — Claude Code Instructions

## Overview

Anima is a Rust runtime for AI agents. **Arya** is the lead architect — this is her project.

**Version:** v2.9.0
**Tests:** 378 passing
**Repo:** github.com/inkthorne/anima

## Quick Context

**Read `docs/context.md` first** — full architecture and command reference.

## Project Structure

```
src/
├── lib.rs              # Module exports, re-exports main types
├── main.rs             # CLI entry point (clap-based)
├── repl.rs             # REPL (thin client, socket connections)
├── daemon.rs           # Daemon mode, socket server, tool execution
├── discovery.rs        # Find running daemons via pid files
├── socket_api.rs       # Unix socket protocol (Request/Response)
├── agent_dir.rs        # Agent directory loading, create_agent()
├── agent.rs            # Core agent logic, ThinkOptions, history
├── config.rs           # AgentConfig, persona settings
├── error.rs            # AgentError, ToolError, ErrorContext
├── tool.rs             # Tool trait, ToolInfo
├── tool_registry.rs    # Keyword-based tool recall from tools.toml
├── llm.rs              # LLM trait, ToolSpec, providers
├── memory.rs           # Memory trait, InMemoryStore, SqliteMemory, SemanticMemoryStore
├── embedding.rs        # EmbeddingClient (Ollama), cosine_similarity
├── message.rs          # Message type
├── messaging.rs        # AgentMailbox, MessageRouter
├── runtime.rs          # Runtime for spawning agents
├── observe.rs          # Observer trait, ConsoleObserver, MetricsCollector
├── retry.rs            # RetryPolicy, exponential backoff
├── supervision.rs      # ChildHandle, ChildConfig, ChildStatus
├── debug.rs            # Debug logging utilities
└── tools/
    ├── mod.rs
    ├── read_file.rs    # ReadFileTool
    ├── write_file.rs   # WriteFileTool
    ├── shell.rs        # ShellTool
    ├── http.rs         # HttpTool
    ├── send_message.rs # SendMessageTool, DaemonSendMessageTool
    ├── list_agents.rs  # ListAgentsTool, DaemonListAgentsTool
    ├── spawn_child.rs  # SpawnChildTool
    ├── wait_for_child.rs # WaitForChildTool
    ├── add.rs          # AddTool (demo)
    └── echo.rs         # EchoTool (demo)
```

## Architecture (v2.9)

**REPL-as-Frontend:** Agents always run as daemons. REPL is a thin client.

```
~/.anima/
├── models/*.toml        # Model definitions (provider, context, tools, always)
├── tools.toml           # Tool registry for keyword recall
└── agents/
    ├── always.md        # Global always (shared across agents)
    └── <name>/
        ├── config.toml  # Agent config (references model_file)
        ├── persona.md   # System prompt
        ├── always.md    # Agent-specific always
        ├── memory.db    # Semantic memory (SQLite)
        ├── daemon.pid   # PID (when running)
        └── agent.sock   # Unix socket (when running)
```

## Tool System (Hybrid)

**Keyword Recall:** Tools defined in `~/.anima/tools.toml`. Only relevant tools (3-5) injected per query based on keyword matching.

**Two Modes:**
- `tools = true` in model config: Native tool calling — LLM gets ToolSpecs
- `tools = false`: JSON-block — Model outputs `{"tool": "x", "params": {...}}`, daemon parses and executes

**Available Tools:** `read_file`, `write_file`, `shell`, `http`, `send_message`, `list_agents`

**Safe Shell:** Tools can have `allowed_commands` list for command allowlist filtering.

## Memory System

**Semantic Memory with Embeddings:**
- `SemanticMemoryStore` in SQLite with vector embeddings
- `EmbeddingClient` uses Ollama for embeddings (e.g., `nomic-embed-text`)
- Cosine similarity for recall, weighted by recency and importance
- `[REMEMBER: ...]` tags in agent output auto-save to memory
- Memory injection prepends relevant memories to context

**Key Functions:**
- `extract_remember_tags()` — Parse and extract memory tags
- `build_memory_injection()` — Format memories for context injection
- `recall_with_embedding()` — Semantic search using embeddings

## Runtime Context Injection (v2.9)

Agents now know their runtime context — injected into system prompt:
- Agent name, model, host
- Tools mode (native vs JSON-block)
- Available tools

**`anima system <agent>`** — Inspect the assembled system prompt for debugging.

## CLI Commands

```bash
# Daemon control
anima start <name>      # Start agent daemon in background
anima stop <name>       # Stop running daemon
anima restart <name>    # Stop then start

# Interaction
anima                   # REPL mode (connects to daemons)
anima chat <name>       # Interactive chat session
anima ask <name> "msg"  # One-shot query (no daemon required)
anima send <name> "msg" # Send message to running daemon

# Management
anima status            # Show running/stopped agents
anima list              # List available agents
anima create <name>     # Scaffold new agent directory
anima clear <name>      # Clear conversation history
anima system <name>     # Show assembled system prompt

# Development
anima run <name>        # Run with REPL (starts daemon if needed)
anima run <name> --daemon # Run as daemon directly
anima task <config> "task" [--stream] [-v] # One-shot with config file
```

## Build & Test

```bash
cargo check           # Type check
cargo build --release # Build
cargo test            # Run tests (378 tests)
```

## Key Design Decisions

1. **Agents are daemons** — persistent processes, not ephemeral
2. **REPL is thin** — just a socket client, no agent logic
3. **@mentions are routing** — agents talk to each other via @mentions
4. **always.md exploits recency bias** — keeps instructions salient
5. **Keyword tool recall** — only inject relevant tools per query
6. **Semantic memory** — embedding-based recall, not just keyword matching
7. **Runtime context** — agents know their name, model, and capabilities

## LLM Providers

- **OpenAI** — `OpenAIClient`, env: `OPENAI_API_KEY`
- **Anthropic** — `AnthropicClient`, env: `ANTHROPIC_API_KEY`
- **Ollama** — `OllamaClient`, local, supports `num_ctx`, `thinking` mode

## Documentation

- **`docs/context.md`** — Start here — architecture and commands
- `docs/VISION.md` — Roadmap and philosophy
- `docs/DESIGN.md` — Architecture deep-dive
