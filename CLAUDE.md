# Anima — Claude Code Instructions

## Overview

Anima is a Rust runtime for AI agents. **Arya** is the lead architect — this is her project.

**Version:** v3.10.10
**Tests:** 660 passing
**Repo:** github.com/inkthorne/anima

## Quick Context

**Read `context/context.md` first** — full architecture and command reference.

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
├── conversation.rs     # ConversationStore, multi-agent messaging (SQLite)
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
    ├── edit_file.rs    # EditFileTool
    ├── write_file.rs   # WriteFileTool
    ├── list_files.rs   # ListFilesTool
    ├── shell.rs        # ShellTool
    ├── safe_shell.rs   # SafeShellTool (command allowlist filtering)
    ├── http.rs         # HttpTool
    ├── send_message.rs # SendMessageTool, DaemonSendMessageTool
    ├── list_agents.rs  # ListAgentsTool, DaemonListAgentsTool
    ├── remember.rs     # RememberTool, DaemonRememberTool
    ├── search_conversation.rs # DaemonSearchConversationTool
    ├── claude_code.rs  # ClaudeCodeTool (delegate tasks to Claude Code)
    ├── spawn_child.rs  # SpawnChildTool
    ├── wait_for_child.rs # WaitForChildTool, WaitForChildrenTool
    ├── add.rs          # AddTool (demo)
    └── echo.rs         # EchoTool (demo)
```

## Architecture (v3.6)

**REPL-as-Frontend:** Agents always run as daemons. REPL is a thin client.

```
~/.anima/
├── models/*.toml        # Model definitions (provider, context, tools, recall)
├── tools.toml           # Tool registry for keyword recall
├── conversations.db     # Multi-agent conversation store (SQLite)
└── agents/
    ├── recall.md        # Global recall (shared across agents)
    └── <name>/
        ├── config.toml  # Agent config (references model_file)
        ├── system.md    # System prompt
        ├── recall.md    # Agent-specific recall (injected each turn)
        ├── memory.db    # Semantic memory (SQLite)
        ├── daemon.pid   # PID (when running)
        ├── agent.sock   # Unix socket (when running)
        ├── agent.log    # Daemon log file
        └── turns/       # Debug dumps of raw LLM request payloads
            └── {conv}.json  # Named by conversation
```

## Tool System (Hybrid)

**Keyword Recall:** Tools defined in `~/.anima/tools.toml`. Only relevant tools (3-5) injected per query based on keyword matching.

**Two Modes:**
- `tools = true` in model config: Native tool calling — LLM gets ToolSpecs
- `tools = false`: JSON-block — Model outputs `{"tool": "x", "params": {...}}`, daemon parses and executes

**Available Tools:** `read_file`, `edit_file`, `write_file`, `list_files`, `shell`, `safe_shell`, `http`, `send_message`, `list_agents`, `remember`, `search_conversation`, `claude_code`, `spawn_child`, `wait_for_children`

**Safe Shell:** `SafeShellTool` has a command allowlist — only approved commands can be executed, including in pipelines.

## Memory System

**Semantic Memory with Embeddings:**
- `SemanticMemoryStore` in SQLite with vector embeddings
- `EmbeddingClient` uses Ollama for embeddings (e.g., `nomic-embed-text`)
- Cosine similarity for recall, weighted by recency and importance
- `[REMEMBER: ...]` tags in agent output auto-save to memory
- `RememberTool` provides an alternative to tags for explicit memory saves
- Memory injection prepends relevant memories to context

**Key Functions:**
- `extract_remember_tags()` — Parse and extract memory tags
- `build_memory_injection()` — Format memories for context injection
- `recall_with_embedding()` — Semantic search using embeddings

## Runtime Context Injection

Agents know their runtime context — injected into system prompt:
- Agent name, model, host
- Tools mode (native vs JSON-block)
- Available tools

**`anima system <agent>`** — Inspect the assembled system prompt for debugging.

## CLI Commands

```bash
# Daemon control
anima start <name>      # Start agent daemon (supports glob patterns)
anima stop <name>       # Stop running daemon (supports glob patterns)
anima restart <name>    # Stop then start (supports glob patterns)
anima heartbeat <name>  # Trigger heartbeat for running daemon

# Interaction
anima                   # REPL mode (connects to daemons)
anima ask <name> "msg"  # One-shot query (no daemon required)

# Conversations
anima chat              # List conversations (default)
anima chat new [name]   # Create conversation + enter interactive mode
anima chat create [name]# Create conversation (no interactive mode)
anima chat join <name>  # Join existing conversation
anima chat send <conv> "msg"  # Send message (fire-and-forget, @mentions notify agents)
anima chat view <conv>  # View messages (--limit, --since, --json)
anima chat pause <conv> # Pause conversation (queues notifications)
anima chat stop <conv>  # Stop paused conversation (drops queued notifications)
anima chat resume <conv># Resume paused conversation
anima chat delete <name># Delete conversation
anima chat clear <conv> # Clear messages (keeps conversation)
anima chat cleanup      # Delete expired messages and empty conversations

# Memory management
anima memory list <agent>      # List memories (--limit)
anima memory search <agent> "q"# Semantic search (--limit)
anima memory show <agent> <id> # Show full memory details
anima memory add <agent> "text"# Add memory (--importance)
anima memory replace <agent> <id> "text" # Replace memory content
anima memory delete <agent> <id># Delete specific memory
anima memory clear <agent>     # Clear all memories (--force)
anima memory count <agent>     # Count memories

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
cargo install --path . # Install to PATH
cargo test            # Run tests (660 tests)
```

## Key Design Decisions

1. **Agents are daemons** — persistent processes, not ephemeral
2. **REPL is thin** — just a socket client, no agent logic
3. **@mentions are routing** — agents talk to each other via @mentions
4. **recall.md exploits recency bias** — keeps instructions salient
5. **Keyword tool recall** — only inject relevant tools per query
6. **Semantic memory** — embedding-based recall, not just keyword matching
7. **Runtime context** — agents know their name, model, and capabilities

## LLM Providers

- **OpenAI** — `OpenAIClient`, env: `OPENAI_API_KEY`
- **Anthropic** — `AnthropicClient`, env: `ANTHROPIC_API_KEY`
- **Ollama** — `OllamaClient`, local, supports `num_ctx`, `thinking` mode

## Documentation

- **`context/context.md`** — Start here — architecture and commands
- `context/VISION.md` — Roadmap and philosophy
- `context/DESIGN.md` — Architecture deep-dive
- `context/MULTI_AGENT_SPEC.md` — Multi-agent conversation system
- `context/CLAUDE_CODE_TOOL_SPEC.md` — Claude Code tool integration
- `context/HEARTBEAT_SPEC.md` — Heartbeat system
- `context/STREAMING_SPEC.md` — Streaming specification
