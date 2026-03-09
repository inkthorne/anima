use anima::agent_dir::AgentDir;
use anima::config::AgentConfig;
use anima::daemon::PidFile;
use anima::observe::ConsoleObserver;
use anima::socket_api::{Request, Response, SocketApi};
use anima::tools::{AddTool, CopyLinesTool, EchoTool, EditFileTool, HttpTool, ListFilesTool, PeekFileTool, ReadFileTool, SafeShellTool, ShellTool, WriteFileTool};
use anima::{
    AnthropicClient, ConversationStore, InMemoryStore, LLM, NotifyResult,
    OpenAIClient, Runtime, SemanticMemoryStore, SqliteMemory,
    expand_all_mention, format_age, notify_mentioned_agents_parallel, on_conversation_event,
    parse_mentions,
};
use clap::{Parser, Subcommand};
use crossterm::event::{Event, EventStream, KeyCode, KeyEvent, KeyModifiers};
use futures_util::StreamExt;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::time::Duration;
use tokio::net::UnixStream;

/// Format tool params for display in a brief, readable way.
fn format_tool_params(params: &serde_json::Value) -> String {
    if let Some(cmd) = params.get("command").and_then(|c| c.as_str()) {
        return cmd.to_string();
    }
    if let Some(path) = params.get("path").and_then(|p| p.as_str()) {
        return path.to_string();
    }
    if let Some(url) = params.get("url").and_then(|u| u.as_str()) {
        let method = params
            .get("method")
            .and_then(|m| m.as_str())
            .unwrap_or("GET");
        return format!("{} {}", method, url);
    }
    let json = serde_json::to_string(params).unwrap_or_default();
    if json.len() > 60 {
        format!("{}...", &json[..57])
    } else {
        json
    }
}

#[derive(Parser)]
#[command(name = "anima", about = "The animating spirit - AI agent runtime")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

}

#[derive(Subcommand)]
enum Commands {
    /// Run an agent daemon in the foreground
    Run {
        /// Agent name (from ~/.anima/agents/) or path to agent directory
        agent: String,
    },
    /// Scaffold a new agent directory
    Create {
        /// Name for the new agent
        name: String,
        /// Directory to create agent in (default: ~/.anima/agents/<name>)
        #[arg(long)]
        path: Option<PathBuf>,
    },
    /// List agents in ~/.anima/agents/
    List,
    /// Run an agent with a single task (non-interactive)
    Task {
        /// Path to agent config file
        config: String,
        /// Task for the agent
        task: String,
        /// Enable streaming output
        #[arg(long)]
        stream: bool,
        /// Enable verbose observability output (all events, not just errors/completions)
        #[arg(long, short)]
        verbose: bool,
    },
    /// Manage conversations - list, create, join, or delete chats
    Chat {
        #[command(subcommand)]
        command: Option<ChatCommands>,
    },
    /// Show status of all agents (running/stopped)
    #[clap(alias = "ps")]
    Status,
    /// One-shot query to an agent (no daemon required)
    Ask {
        /// Agent name (from ~/.anima/agents/) or path to agent directory
        agent: String,
        /// Message to send to the agent
        message: String,
        /// Enable verbose output (thinking, usage, tool calls, memory) on stderr
        #[arg(long, short = 'v')]
        verbose: bool,
    },
    /// Start an agent daemon in the background. Supports glob patterns (*, ?).
    Start {
        /// Agent name or glob pattern (from ~/.anima/agents/)
        agent: String,
    },
    /// Agent debugging tools
    Agent {
        #[command(subcommand)]
        command: AgentCommands,
    },
    /// Stop a running agent daemon. Supports glob patterns (*, ?).
    Stop {
        /// Agent name or glob pattern (from ~/.anima/agents/)
        agent: String,
    },
    /// Clear conversation history for a running agent daemon
    Clear {
        /// Agent name (from ~/.anima/agents/) or path to agent directory
        agent: String,
    },
    /// Restart a running agent daemon (stop then start). Supports glob patterns (*, ?).
    Restart {
        /// Agent name or glob pattern (from ~/.anima/agents/)
        agent: String,
    },
    /// Show the system prompt for a running agent daemon
    System {
        /// Agent name (from ~/.anima/agents/) or path to agent directory
        agent: String,
    },
    /// Trigger a heartbeat for a running agent daemon
    Heartbeat {
        /// Agent name (from ~/.anima/agents/) or path to agent directory
        agent: String,
    },
    /// Log in with your Anthropic subscription (OAuth)
    Login,
    /// Log out (remove stored subscription tokens)
    Logout,
    /// Show current authentication status
    Whoami,
    /// Manage agent memories
    Memory {
        #[command(subcommand)]
        command: MemoryCommands,
    },
    /// Persistent named threads (no daemon, no tools)
    Thread {
        #[command(subcommand)]
        command: Option<ThreadSubcommands>,
        /// Thread name (when sending a message)
        #[arg(requires = "message")]
        name: Option<String>,
        /// Message to send
        message: Option<String>,
    },
}

#[derive(Subcommand)]
enum ThreadSubcommands {
    /// Create a new thread bound to an agent
    Create {
        /// Thread name
        name: String,
        /// Agent name (from ~/.anima/agents/) or path to agent directory
        agent: String,
    },
    /// Clear (delete) a thread
    Clear {
        /// Thread name
        name: String,
    },
    /// Fork a thread (copy history to a new thread)
    Fork {
        /// Source thread name
        source: String,
        /// New thread name
        new_name: String,
    },
}

#[derive(Subcommand)]
enum AgentCommands {
    /// Step-debug: fresh start (restart daemon, new conversation, one step)
    Debug {
        /// Agent name
        agent: String,
        /// Message to send
        message: String,
    },
    /// Step-debug: continue one more step from current state
    Step {
        /// Agent name
        agent: String,
    },
}

#[derive(Subcommand)]
enum MemoryCommands {
    /// List all memories for an agent
    List {
        /// Agent name
        agent: String,
        /// Maximum memories to show
        #[arg(short, long, default_value = "50")]
        limit: usize,
    },
    /// Search memories using semantic similarity
    Search {
        /// Agent name
        agent: String,
        /// Search query
        query: String,
        /// Maximum results
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },
    /// Delete a specific memory by ID
    Delete {
        /// Agent name
        agent: String,
        /// Memory ID to delete
        id: i64,
    },
    /// Clear all memories for an agent
    Clear {
        /// Agent name
        agent: String,
        /// Skip confirmation
        #[arg(short, long)]
        force: bool,
    },
    /// Count memories for an agent
    Count {
        /// Agent name
        agent: String,
    },
    /// Show full details of a specific memory
    Show {
        /// Agent name
        agent: String,
        /// Memory ID to show
        id: i64,
    },
    /// Add a new memory for an agent
    Add {
        /// Agent name
        agent: String,
        /// Memory content
        content: String,
        /// Importance (0.0-1.0, default 0.9)
        #[arg(short, long, default_value = "0.9")]
        importance: f64,
    },
    /// Replace the content of an existing memory (keeps importance)
    Replace {
        /// Agent name
        agent: String,
        /// Memory ID to replace
        id: i64,
        /// New content
        content: String,
    },
}

#[derive(Subcommand)]
enum ChatCommands {
    /// Create a new conversation with a random fun name (or specify a name)
    New {
        /// Name for the conversation (optional, generates fun name if not provided)
        name: Option<String>,
    },
    /// Create a conversation without entering interactive mode
    Create {
        /// Name for the conversation (optional, generates fun name if not provided)
        name: Option<String>,
    },
    /// Join an existing conversation by name
    #[clap(alias = "open")]
    Join {
        /// Name of the conversation to join
        name: String,
    },
    /// Send a message to a conversation (fire-and-forget, notifies @mentioned agents)
    Send {
        /// Name of the conversation
        conv: String,
        /// Message to send (use @mentions to notify agents)
        message: String,
    },
    /// View messages in a conversation
    View {
        /// Name of the conversation
        conv: String,
        /// Show only the last N messages
        #[arg(long)]
        limit: Option<usize>,
        /// Show only messages with ID greater than this value
        #[arg(long)]
        since: Option<i64>,
        /// Output messages as JSON array
        #[arg(long)]
        json: bool,
        /// Output messages as pretty-printed JSON (content unescaped for readability)
        #[arg(long)]
        pretty: bool,
    },
    /// Pause a conversation (notifications will be queued). Supports glob patterns (*, ?).
    Pause {
        /// Name or pattern of the conversation(s) to pause
        conv: String,
        /// Skip confirmation prompt for multiple matches
        #[arg(short, long)]
        force: bool,
    },
    /// Stop a paused conversation without processing queued notifications. Supports glob patterns.
    Stop {
        /// Name or pattern of the conversation(s) to stop
        conv: String,
        /// Skip confirmation for multiple matches
        #[arg(short, long)]
        force: bool,
    },
    /// Resume a paused conversation (processes queued notifications). Supports glob patterns (*, ?).
    Resume {
        /// Name or pattern of the conversation(s) to resume
        conv: String,
        /// Skip confirmation prompt for multiple matches
        #[arg(short, long)]
        force: bool,
    },
    /// Delete a conversation completely. Supports glob patterns (*, ?).
    Delete {
        /// Name or pattern of the conversation(s) to delete
        name: String,
        /// Skip confirmation prompt for multiple matches
        #[arg(short, long)]
        force: bool,
    },
    /// Clear all messages from a conversation (keeps the conversation and participants). Supports glob patterns (*, ?).
    Clear {
        /// Name or pattern of the conversation(s) to clear
        conv: String,
        /// Skip confirmation prompt for multiple matches
        #[arg(short, long)]
        force: bool,
    },
    /// Pin a message so it always appears in agent context
    Pin {
        /// Name of the conversation
        conv: String,
        /// Message ID to pin
        id: i64,
    },
    /// Unpin a previously pinned message
    Unpin {
        /// Name of the conversation
        conv: String,
        /// Message ID to unpin
        id: i64,
    },
    /// Run cleanup to delete expired messages and empty conversations
    Cleanup,
}

#[tokio::main]
async fn main() {
    register_steps_hook();

    let cli = Cli::parse();
    let command = cli.command.unwrap_or(Commands::Status);

    match command {
        Commands::Run { agent } => {
            if let Err(e) = anima::daemon::run_daemon(&agent).await {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Create { name, path } => {
            if let Err(e) = create_agent(&name, path) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::List => {
            list_agents();
        }
        Commands::Task {
            config,
            task,
            stream,
            verbose,
        } => {
            if let Err(e) = run_agent_task(&config, &task, stream, verbose).await {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Chat { command } => {
            if let Err(e) = handle_chat_command(command).await {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Status => {
            show_status().await;
        }
        Commands::Ask { agent, message, verbose } => {
            if let Err(e) = ask_agent(&agent, &message, verbose).await {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Start { agent } => {
            if let Err(e) = start_agent(&agent) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Agent { command } => match command {
            AgentCommands::Debug { agent, message } => {
                if let Err(e) = start_agent_debug(&agent, &message).await {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
            AgentCommands::Step { agent } => {
                if let Err(e) = step_agent(&agent).await {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
        },
        Commands::Stop { agent } => {
            if let Err(e) = stop_agent(&agent).await {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Clear { agent } => {
            if let Err(e) = clear_agent(&agent).await {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Restart { agent } => {
            if let Err(e) = restart_agent(&agent).await {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::System { agent } => {
            if let Err(e) = show_system_prompt(&agent).await {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Heartbeat { agent } => {
            if let Err(e) = trigger_heartbeat(&agent).await {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Login => {
            if let Err(e) = handle_login().await {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Logout => {
            anima::auth::clear_tokens();
            println!("Logged out — stored subscription tokens removed.");
        }
        Commands::Whoami => {
            handle_whoami();
        }
        Commands::Memory { command } => {
            if let Err(e) = handle_memory_command(command).await {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Thread { command, name, message } => {
            if let Err(e) = handle_thread_command(command, name, message).await {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Conversation event hooks
// ---------------------------------------------------------------------------

/// Register steps-directory lifecycle hooks for conversation events.
fn register_steps_hook() {
    use anima::ConversationEvent;

    on_conversation_event(|event| {
        match event {
            ConversationEvent::Cleared { conv_name, participants } => {
                for agent in participants.iter().filter(|a| a.as_str() != "user") {
                    let dir = agents_dir().join(agent).join("steps").join(conv_name);
                    if dir.is_dir() {
                        for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
                            let _ = std::fs::remove_file(entry.path());
                        }
                    }
                }
            }
            ConversationEvent::Deleted { conv_name, participants } => {
                for agent in participants.iter().filter(|a| a.as_str() != "user") {
                    let dir = agents_dir().join(agent).join("steps").join(conv_name);
                    let _ = std::fs::remove_dir_all(&dir);
                }
            }
            _ => {} // Created/ParticipantAdded — dirs created lazily by dump_request()
        }
    });
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

/// Get the agents directory path (~/.anima/agents/)
fn agents_dir() -> PathBuf {
    dirs::home_dir()
        .expect("Could not determine home directory")
        .join(".anima")
        .join("agents")
}

/// RAII guard to restore terminal from raw mode on drop.
struct RawModeGuard;
impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = crossterm::terminal::disable_raw_mode();
    }
}

/// Watch for ESC or Ctrl+C keypresses (used in raw mode).
async fn watch_for_abort() {
    let mut reader = EventStream::new();
    loop {
        match reader.next().await {
            Some(Ok(Event::Key(KeyEvent { code: KeyCode::Esc, .. }))) => return,
            Some(Ok(Event::Key(KeyEvent {
                code: KeyCode::Char('c'),
                modifiers,
                ..
            }))) if modifiers.contains(KeyModifiers::CONTROL) => return,
            _ => continue,
        }
    }
}

async fn handle_thread_command(
    command: Option<ThreadSubcommands>,
    name: Option<String>,
    message: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Some(ThreadSubcommands::Create { name, agent }) => {
            let agent_path = resolve_agent_path(&agent);
            let agent_dir = AgentDir::load(&agent_path)?;
            anima::AnimaThread::create(&name, &agent_dir)?;
            println!("Thread '{}' created (agent: {})", name, agent_dir.config.agent.name);
        }
        Some(ThreadSubcommands::Clear { name }) => {
            anima::AnimaThread::clear(&name)?;
            println!("Thread '{}' cleared.", name);
        }
        Some(ThreadSubcommands::Fork { source, new_name }) => {
            let agent = anima::AnimaThread::fork(&source, &new_name)?;
            println!("Thread '{}' forked to '{}' (agent: {})", source, new_name, agent);
        }
        None => {
            match (name, message) {
                (Some(name), Some(message)) => {
                    let mut thread = anima::AnimaThread::load(&name, |a| resolve_agent_path(a)).await?;
                    let agent_dir = Some(resolve_agent_path(thread.agent_name()));
                    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(32);

                    // Enable raw mode for ESC/Ctrl+C abort detection (fails gracefully in pipes)
                    let raw_mode = crossterm::terminal::enable_raw_mode().is_ok();
                    let _raw_guard = if raw_mode { Some(RawModeGuard) } else { None };

                    let print_handle = tokio::spawn(async move {
                        let mut had_chunks = false;
                        let mut in_thinking = false;
                        let mut in_python_output = false;
                        while let Some(token) = rx.recv().await {
                            // Detect <usage>...</usage> tags (before \r\n conversion)
                            if token.starts_with("<usage>") && token.ends_with("</usage>") {
                                let inner = &token[7..token.len() - 8];
                                let nl = if raw_mode { "\r\n" } else { "\n" };
                                print!("\x1b[0m{}\x1b[2m  [{}]\x1b[0m\x1b[33m", nl, inner);
                                had_chunks = true;
                                let _ = io::stdout().flush();
                                continue;
                            }

                            // Detect <retry>...</retry> tags (before \r\n conversion)
                            if token.starts_with("<retry>") && token.ends_with("</retry>") {
                                let inner = &token[7..token.len() - 8];
                                let nl = if raw_mode { "\r\n" } else { "\n" };
                                print!("\x1b[0m{}\x1b[2;33m  [{}]\x1b[0m\x1b[33m", nl, inner);
                                had_chunks = true;
                                let _ = io::stdout().flush();
                                continue;
                            }

                            // In raw mode, \n must become \r\n for proper display
                            let token = if raw_mode {
                                token.replace('\n', "\r\n")
                            } else {
                                token
                            };

                            // Detect <think> / </think> boundaries
                            if token == "<think>" {
                                in_thinking = true;
                                if had_chunks {
                                    print!("\x1b[0m");
                                }
                                print!("\x1b[2m<think>\r\n");
                                had_chunks = true;
                                let _ = io::stdout().flush();
                                continue;
                            }
                            if token.starts_with("</think>") {
                                in_thinking = false;
                                print!("</think>\x1b[0m\x1b[33m");
                                let remainder = token.trim_start_matches("</think>");
                                if !remainder.is_empty() {
                                    print!("{}", remainder);
                                }
                                let _ = io::stdout().flush();
                                continue;
                            }

                            // Detect <python-output> / </python-output> boundaries
                            if token.contains("<python-output") {
                                in_python_output = true;
                                print!("\x1b[36m");
                            }
                            if token.contains("</python-output>") {
                                // Will reset after printing
                            }

                            // Set initial color on first chunk
                            if !had_chunks {
                                if in_thinking {
                                    print!("\x1b[2m");
                                } else if in_python_output {
                                    print!("\x1b[36m");
                                } else {
                                    print!("\x1b[33m");
                                }
                            }
                            had_chunks = true;
                            print!("{}", token);
                            let _ = io::stdout().flush();

                            if token.contains("</python-output>") {
                                in_python_output = false;
                                print!("\x1b[33m");
                                let _ = io::stdout().flush();
                            }
                        }
                        // Reset colors at end
                        if had_chunks {
                            print!("\x1b[0m");
                        }
                    });

                    // Race send_stream_with_python against abort keypress (only in raw mode)
                    let result = if raw_mode {
                        tokio::select! {
                            res = thread.send_stream_with_python(&message, tx, agent_dir) => Some(res),
                            _ = watch_for_abort() => None,
                        }
                    } else {
                        Some(thread.send_stream_with_python(&message, tx, agent_dir).await)
                    };

                    // tx is dropped by either path; print_handle drains remaining tokens
                    print_handle.await.unwrap();

                    // Restore terminal before printing final output
                    drop(_raw_guard);

                    match result {
                        Some(Ok(_response)) => {
                            println!();
                        }
                        Some(Err(e)) => return Err(Box::new(e)),
                        None => {
                            println!();
                            println!("\x1b[2m[aborted]\x1b[0m");
                        }
                    }
                }
                _ => {
                    let threads = anima::AnimaThread::list_all()?;
                    if threads.is_empty() {
                        println!("No threads found.");
                        println!("Create one with: anima thread create <name> <agent>");
                    } else {
                        println!("{:<30} {:<20} {:>4}", "NAME", "AGENT", "MSGS");
                        println!("{}", "-".repeat(80));
                        for (name, agent, count) in &threads {
                            println!("{:<30} {:<20} {:>4}", name, agent, count);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Resolve an agent path from a name or path string.
/// If it's a path (contains / or \), use it directly.
/// Otherwise, treat it as an agent name and look in ~/.anima/agents/<name>/
fn resolve_agent_path(agent: &str) -> PathBuf {
    if agent.contains('/') || agent.contains('\\') || agent.starts_with('.') {
        PathBuf::from(agent)
    } else {
        agents_dir().join(agent)
    }
}

/// Resolve the socket path for an agent.
fn resolve_agent_socket(agent: &str) -> PathBuf {
    resolve_agent_path(agent).join("agent.sock")
}

/// Connect to a running agent daemon and return a SocketApi.
async fn connect_to_agent(agent: &str) -> Result<SocketApi, Box<dyn std::error::Error>> {
    let socket_path = resolve_agent_socket(agent);

    if !socket_path.exists() {
        return Err(format!(
            "Agent '{}' is not running (socket not found: {})",
            agent,
            socket_path.display()
        )
        .into());
    }

    let stream = UnixStream::connect(&socket_path)
        .await
        .map_err(|e| format!("Failed to connect to agent '{}': {}", agent, e))?;

    Ok(SocketApi::new(stream))
}

// ---------------------------------------------------------------------------
// Shared helpers (format, display, directory listing, memory, catchup)
// ---------------------------------------------------------------------------

/// Default number of context messages to load from conversation history.
const DEFAULT_CONTEXT_MESSAGES: usize = 20;

/// Format the header line for a conversation message.
/// Example: `[2419] 2026-02-08 15:03 • dash • 2m 51s (23.7s) • 37k/131k (28%)`
fn format_message_header(msg: &anima::conversation::ConversationMessage) -> String {
    let color = if msg.from_agent == "user" { "33" } else { "36" };
    let datetime = format_timestamp_pretty(msg.created_at);

    let is_agent = msg.from_agent != "user" && msg.from_agent != "tool";
    let duration_str = if is_agent {
        msg.duration_ms
            .map(|d| {
                let eval_part = msg.prompt_eval_ns
                    .filter(|&ns| ns > 0)
                    .map(|ns| {
                        let secs = ns as f64 / 1_000_000_000.0;
                        if secs >= 1.0 {
                            format!(" ({:.1}s)", secs)
                        } else {
                            format!(" ({:.0}ms)", secs * 1000.0)
                        }
                    })
                    .unwrap_or_default();
                format!(" \x1b[90m•\x1b[0m \x1b[33m{}{}\x1b[0m", format_duration_ms(d), eval_part)
            })
            .unwrap_or_default()
    } else {
        String::new()
    };
    let ctx_str = if is_agent {
        format_context_usage(msg)
    } else {
        String::new()
    };

    let pin_str = if msg.pinned { " \x1b[33m[pinned]\x1b[0m" } else { "" };

    format!(
        "\x1b[90m[{}] {}\x1b[0m \x1b[90m•\x1b[0m \x1b[{}m{}\x1b[0m{}{}{}",
        msg.id, datetime, color, msg.from_agent, pin_str, duration_str, ctx_str
    )
}

/// Format a conversation message header and content for pretty display.
fn format_message_display(msg: &anima::conversation::ConversationMessage) -> String {
    let header = format_message_header(msg);
    let content = msg.content.clone();

    let display_content = if content.is_empty() {
        if let Some(names) = extract_tool_names(msg) {
            let tools = names.iter().map(|n| format!("🛠️ {}", n)).collect::<Vec<_>>().join("\n");
            format!("\x1b[90m{}\x1b[0m", tools)
        } else {
            content
        }
    } else if let Some(names) = extract_tool_names(msg) {
        let tools = names.iter().map(|n| format!("🛠️ {}", n)).collect::<Vec<_>>().join("\n");
        format!("{}\n\x1b[90m{}\x1b[0m", content, tools)
    } else {
        content
    };

    format!("{}\n{}\n\n", header, display_content)
}

/// Extract tool call summaries from a message's tool_calls JSON, if present.
fn extract_tool_names(msg: &anima::conversation::ConversationMessage) -> Option<Vec<String>> {
    let json = msg.tool_calls.as_ref()?;
    let calls: Vec<serde_json::Value> = serde_json::from_str(json).ok()?;
    let summaries: Vec<String> = calls
        .iter()
        .filter_map(|c| {
            let name = c.get("name").and_then(|n| n.as_str())?;
            let args = c.get("arguments")
                .and_then(|a| {
                    if a.is_string() {
                        serde_json::from_str(a.as_str().unwrap()).ok()
                    } else {
                        Some(a.clone())
                    }
                })
                .unwrap_or(serde_json::Value::Null);
            Some(tool_call_summary(name, &args))
        })
        .collect();
    if summaries.is_empty() { None } else { Some(summaries) }
}

/// Produce a short summary string for a tool call, e.g. "shell: ls -la ~/dev".
fn tool_call_summary(name: &str, args: &serde_json::Value) -> String {
    let truncate = |s: &str, max: usize| -> String {
        if s.len() <= max {
            s.to_string()
        } else {
            format!("{}...", &s[..s.floor_char_boundary(max)])
        }
    };

    let detail = match name {
        "shell" | "safe_shell" => {
            args.get("command")
                .and_then(|v| v.as_str())
                .map(|s| s.lines().next().unwrap_or(s))
                .map(|s| truncate(s, 60))
        }
        "read_file" | "peek_file" | "write_file" | "edit_file" => {
            args.get("path").and_then(|v| v.as_str()).map(String::from)
        }
        "copy_lines" => {
            args.get("source").and_then(|v| v.as_str()).map(String::from)
        }
        "http" => {
            let method = args.get("method").and_then(|v| v.as_str()).unwrap_or("GET");
            args.get("url").and_then(|v| v.as_str()).map(|url| format!("{} {}", method, url))
        }
        "send_message" => {
            args.get("to").and_then(|v| v.as_str()).map(|to| format!("@{}", to))
        }
        "start_task" => {
            args.get("agent").and_then(|v| v.as_str()).map(String::from)
        }
        "wait_task" | "stop_task" => {
            args.get("task_conv").and_then(|v| v.as_str()).map(|s| truncate(s, 40))
        }
        "claude_code" => {
            args.get("task").and_then(|v| v.as_str()).map(|s| truncate(s, 60))
        }
        "remember" => {
            args.get("content").and_then(|v| v.as_str()).map(|s| truncate(s, 60))
        }
        "list_agents" => None,
        "add" => {
            let a = args.get("a").and_then(|v| v.as_f64());
            let b = args.get("b").and_then(|v| v.as_f64());
            match (a, b) {
                (Some(a), Some(b)) => Some(format!("{} + {}", a, b)),
                _ => None,
            }
        }
        "echo" => {
            args.get("message").and_then(|v| v.as_str()).map(|s| truncate(s, 60))
        }
        "search_conversation" => {
            let keyword = args.get("keyword").and_then(|v| v.as_str()).unwrap_or("?");
            let conv = args.get("conversation").and_then(|v| v.as_str()).unwrap_or("?");
            let from = args.get("from").and_then(|v| v.as_str());
            match from {
                Some(f) => Some(format!("'{}' from={} in={}", keyword, f, conv)),
                None => Some(format!("'{}' in={}", keyword, conv)),
            }
        }
        "list_files" => {
            args.get("path").and_then(|v| v.as_str()).map(String::from)
        }
        "notes" => {
            args.get("content").and_then(|v| v.as_str()).map(|s| truncate(s, 60))
        }
        "subtask" => {
            args.get("task").and_then(|v| v.as_str()).map(|s| truncate(s, 80))
        }
        "python" => {
            args.get("code")
                .and_then(|v| v.as_str())
                .map(|s| s.lines().next().unwrap_or(s))
                .map(|s| truncate(s, 60))
        }
        _ => None,
    };

    match detail {
        Some(d) => format!("{}: {}", name, d),
        None => name.to_string(),
    }
}

/// Produce a readable preview for step trace display.
///
/// Extracts meaningful content from tool-block (`<tool>`) and state transition
/// (`<set-vars>`) responses instead of showing raw XML tags.
fn step_trace_preview(content: &str) -> String {
    let trimmed = content.trim();

    // Tool-block: extract tool name and params
    if let Some(start) = trimmed.find("<tool>") {
        if let Some(end) = trimmed.find("</tool>") {
            let json_str = trimmed[start + 6..end].trim();
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
                let name = val["tool"].as_str().unwrap_or("?");
                let params = &val["params"];
                let detail = format_tool_params(params);
                return format!("<tool> {}: {}", name, detail);
            }
        }
        // <tool> tag present but JSON didn't parse — show what we can
        let inner = trimmed[trimmed.find("<tool>").unwrap() + 6..].trim();
        let first = inner.lines().next().unwrap_or(inner);
        let truncated = if first.len() > 120 { format!("{}...", &first[..first.floor_char_boundary(120)]) } else { first.to_string() };
        return format!("<tool> {}", truncated);
    }

    // State transition: extract state name from <set-vars><state>X</state></set-vars>
    if let Some(state_start) = trimmed.find("<state>") {
        if let Some(state_end) = trimmed.find("</state>") {
            let state_name = &trimmed[state_start + 7..state_end];
            // Show any text before the tag as context
            let before = trimmed[..state_start].trim().trim_start_matches("<set-vars>").trim();
            if before.is_empty() {
                return format!("-> {}", state_name);
            } else {
                let before_preview = if before.len() > 100 {
                    format!("{}...", &before[..before.floor_char_boundary(100)])
                } else {
                    before.to_string()
                };
                return format!("{} -> {}", before_preview, state_name);
            }
        }
    }

    // Default: first line, truncated
    let first_line = trimmed.lines().next().unwrap_or("");
    if first_line.len() > 200 {
        format!("{}...", &first_line[..first_line.floor_char_boundary(200)])
    } else {
        first_line.to_string()
    }
}

/// Format a Unix timestamp as "YYYY-MM-DD HH:MM:SS" for pretty display.
fn format_timestamp_pretty(timestamp: i64) -> String {
    use std::time::{Duration, UNIX_EPOCH};
    let datetime = UNIX_EPOCH + Duration::from_secs(timestamp as u64);
    let datetime: chrono::DateTime<chrono::Local> = datetime.into();
    datetime.format("%Y-%m-%d %H:%M:%S").to_string()
}

/// Format a duration in milliseconds as human-readable (e.g., "1m 32s", "450ms").
fn format_duration_ms(ms: i64) -> String {
    if ms < 1000 {
        format!("{}ms", ms)
    } else if ms < 60_000 {
        let secs = ms / 1000;
        let millis = ms % 1000;
        if millis > 0 {
            format!("{}.{}s", secs, millis / 100)
        } else {
            format!("{}s", secs)
        }
    } else {
        let mins = ms / 60_000;
        let secs = (ms % 60_000) / 1000;
        if secs > 0 {
            format!("{}m {}s", mins, secs)
        } else {
            format!("{}m", mins)
        }
    }
}

/// Format context usage from message token data.
/// Returns formatted string like "2k/32k (26%)".
fn format_context_usage(msg: &anima::conversation::ConversationMessage) -> String {
    match (msg.tokens_in, msg.tokens_out, msg.num_ctx) {
        (Some(tokens_in), Some(tokens_out), Some(num_ctx)) if num_ctx > 0 => {
            let total_tokens = tokens_in + tokens_out;
            let percentage = (total_tokens as f64 / num_ctx as f64 * 100.0) as u32;
            let used_str = format_tokens_short(total_tokens);
            let ctx_str = format_tokens_short(num_ctx);

            let cache_str = if let Some(cached) = msg.cached_tokens {
                if tokens_in > 0 {
                    let cache_pct = (cached as f64 / tokens_in as f64 * 100.0) as u32;
                    format!(" ({}% cached)", cache_pct)
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            format!(
                " \x1b[90m•\x1b[0m \x1b[35m{}/{} ({}% full){}\x1b[0m",
                used_str, ctx_str, percentage, cache_str
            )
        }
        (Some(tokens_in), Some(tokens_out), _) => {
            let total_tokens = tokens_in + tokens_out;
            let total_str = format_tokens_short(total_tokens);
            format!(" \x1b[90m•\x1b[0m \x1b[35m{} tokens\x1b[0m", total_str)
        }
        _ => String::new(),
    }
}

/// Format token count as short string (e.g., 2048 -> "2k", 65536 -> "64k", 500 -> "500")
fn format_tokens_short(tokens: i64) -> String {
    if tokens >= 1024 {
        let k = (tokens + 512) / 1024;
        format!("{}k", k)
    } else {
        format!("{}", tokens)
    }
}

/// Format a Unix timestamp as relative time (e.g., "2h ago", "3d ago").
fn format_relative_time(timestamp: i64) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    let diff = now - timestamp;

    if diff < 60 {
        "just now".to_string()
    } else if diff < 3600 {
        format!("{}m ago", diff / 60)
    } else if diff < 86400 {
        format!("{}h ago", diff / 3600)
    } else if diff < 604800 {
        format!("{}d ago", diff / 86400)
    } else {
        format!("{}w ago", diff / 604800)
    }
}

/// Check if a pattern contains glob wildcards (* or ?).
fn has_wildcards(pattern: &str) -> bool {
    pattern.contains('*') || pattern.contains('?')
}

/// Ask for user confirmation before a destructive operation.
/// Returns true if user confirms (enters 'y' or 'Y').
fn confirm_action(action: &str, items: &[String]) -> Result<bool, Box<dyn std::error::Error>> {
    println!("This will {} {} conversation(s):", action, items.len());
    for name in items {
        println!("  - {}", name);
    }
    print!("Continue? [y/N] ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().eq_ignore_ascii_case("y"))
}

/// List agent directory entries from ~/.anima/agents/ that contain config.toml.
/// Prints a message and returns None if no agents are found.
fn list_agent_dirs() -> Option<Vec<std::fs::DirEntry>> {
    let agents_path = agents_dir();

    if !agents_path.exists() {
        println!(
            "\x1b[33mNo agents directory found at {}\x1b[0m",
            agents_path.display()
        );
        println!("Create an agent with: \x1b[36manima create <name>\x1b[0m");
        return None;
    }

    let entries: Vec<_> = match std::fs::read_dir(&agents_path) {
        Ok(dir) => dir
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .filter(|e| e.path().join("config.toml").exists())
            .collect(),
        Err(e) => {
            eprintln!("\x1b[31mCould not read agents directory: {}\x1b[0m", e);
            return None;
        }
    };

    if entries.is_empty() {
        println!(
            "\x1b[33mNo agents found in {}\x1b[0m",
            agents_path.display()
        );
        println!("Create an agent with: \x1b[36manima create <name>\x1b[0m");
        return None;
    }

    Some(entries)
}

/// Open a SemanticMemoryStore for the given agent, returning an error if memory.db doesn't exist.
fn open_memory_store(agent: &str) -> Result<SemanticMemoryStore, Box<dyn std::error::Error>> {
    let agent_path = resolve_agent_path(agent);
    let memory_path = agent_path.join("memory.db");

    if !memory_path.exists() {
        return Err(format!(
            "No memory database found for agent '{}' at {}",
            agent,
            memory_path.display()
        )
        .into());
    }

    Ok(SemanticMemoryStore::open(&memory_path, agent)?)
}

/// Process catchup items and pending notifications after resuming a conversation.
/// Returns the number of items processed.
async fn process_catchup(
    store: &ConversationStore,
    conv_name: &str,
    paused_at_msg_id: Option<i64>,
) -> usize {
    let mut catchup_count = 0;

    if let Some(paused_at) = paused_at_msg_id
        && let Ok(catchup_msgs) = store.get_catchup_messages(conv_name, paused_at)
    {
        let catchup_items = ConversationStore::build_catchup_items(&catchup_msgs);
        for item in &catchup_items {
            if !item.mentions.is_empty() {
                let _ = notify_mentioned_agents_parallel(
                    store,
                    conv_name,
                    item.message.id,
                    &item.mentions,
                )
                .await;
                catchup_count += item.mentions.len();
            }
        }
    }

    if let Ok(pending) = store.get_pending_notifications_for_conversation(conv_name) {
        for notification in &pending {
            let mentions = vec![notification.agent.clone()];
            let _ = notify_mentioned_agents_parallel(
                store,
                conv_name,
                notification.message_id,
                &mentions,
            )
            .await;
            let _ = store.delete_pending_notification(notification.id);
        }
        catchup_count += pending.len();
    }

    catchup_count
}

/// Try to get an embedding for a query string using Ollama.
async fn get_query_embedding(query: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use anima::EmbeddingClient;

    let client = EmbeddingClient::new("nomic-embed-text", None);
    Ok(client.embed(query).await?)
}

// ---------------------------------------------------------------------------
// Chat session
// ---------------------------------------------------------------------------

/// Start an interactive chat session with a conversation by name.
///
/// Messages are sent to @mentioned agents. When a user @mentions an agent
/// for the first time, they are automatically added as a participant.
///
/// Note: The daemon now handles @mention forwarding autonomously. The CLI only
/// sends initial notifications and displays results - follow-up chains continue
/// in the background via daemon-to-daemon communication.
///
/// A background polling task displays incoming agent messages in real-time.
async fn chat_with_conversation(conv_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let store = ConversationStore::init()?;

    let parts = store.get_participants(conv_name)?;
    let mut participants: Vec<String> = parts
        .iter()
        .filter(|p| p.agent != "user")
        .map(|p| p.agent.clone())
        .collect();

    let last_seen_id = Arc::new(AtomicI64::new(0));

    if let Ok(msgs) = store.get_messages(conv_name, Some(1))
        && let Some(last) = msgs.last()
    {
        last_seen_id.store(last.id, Ordering::SeqCst);
    }

    println!("\x1b[32m✓ Joined conversation '{}'\x1b[0m", conv_name);
    if participants.is_empty() {
        println!("\x1b[90mNo agents yet - use @mentions to invite them\x1b[0m");
    } else {
        println!("\x1b[90mParticipants: {}\x1b[0m", participants.join(", "));
    }

    if let Ok(recent_msgs) = store.get_messages(conv_name, Some(25))
        && !recent_msgs.is_empty()
    {
        println!("\x1b[90m--- Recent messages ---\x1b[0m");
        for msg in &recent_msgs {
            if msg.from_agent == "tool" || msg.from_agent == "recall" {
                continue;
            }
            print!("{}", format_message_display(msg));
        }

        if let Some(last) = recent_msgs.last() {
            last_seen_id.store(last.id, Ordering::SeqCst);
        }
    }

    println!("Type your messages. Press Ctrl+D or Ctrl+C to exit.\n");

    // Spawn background message polling task
    let poll_conv_name = conv_name.to_string();
    let poll_last_seen = last_seen_id.clone();
    let shutdown = Arc::new(AtomicBool::new(false));
    let poll_shutdown = shutdown.clone();

    let poll_handle = tokio::spawn(async move {
        let poll_store = match ConversationStore::init() {
            Ok(s) => s,
            Err(_) => return,
        };

        loop {
            if poll_shutdown.load(Ordering::SeqCst) {
                break;
            }

            tokio::time::sleep(Duration::from_millis(500)).await;

            let last_id = poll_last_seen.load(Ordering::SeqCst);
            if let Ok(msgs) = poll_store.get_messages_filtered(&poll_conv_name, None, Some(last_id))
            {
                for msg in msgs {
                    if msg.from_agent != "user" && msg.from_agent != "tool" && msg.from_agent != "recall" {
                        print!("\r\x1b[K");
                        print!("{}", format_message_display(&msg));
                        print!("\x1b[36m#{}\x1b[90m›\x1b[0m ", poll_conv_name);
                        let _ = io::stdout().flush();
                    }
                    poll_last_seen.store(msg.id, Ordering::SeqCst);
                }
            }
        }
    });

    let mut rl = rustyline::DefaultEditor::new()?;
    let prompt = format!("\x1b[36m#{}\x1b[90m›\x1b[0m ", conv_name);

    loop {
        match rl.readline(&prompt) {
            Ok(line) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                let _ = rl.add_history_entry(line);

                // Handle slash commands
                if line.starts_with('/') {
                    match line {
                        "/clear" => {
                            store.clear_messages(conv_name)?;
                            println!("\x1b[90mConversation cleared.\x1b[0m");
                            continue;
                        }
                        "/quit" | "/exit" => {
                            println!("\x1b[33mGoodbye!\x1b[0m");
                            break;
                        }
                        "/pause" => {
                            match store.set_paused(conv_name, true) {
                                Ok(_) => println!(
                                    "\x1b[90mConversation paused. Notifications will be queued.\x1b[0m"
                                ),
                                Err(e) => eprintln!("\x1b[31mFailed to pause: {}\x1b[0m", e),
                            }
                            continue;
                        }
                        "/resume" => {
                            match store.set_paused(conv_name, false) {
                                Ok(paused_at_msg_id) => {
                                    let catchup_count =
                                        process_catchup(&store, conv_name, paused_at_msg_id).await;
                                    if catchup_count > 0 {
                                        println!(
                                            "\x1b[90mConversation resumed. {} pending item(s) processed.\x1b[0m",
                                            catchup_count
                                        );
                                    } else {
                                        println!("\x1b[90mConversation resumed.\x1b[0m");
                                    }
                                }
                                Err(e) => eprintln!("\x1b[31mFailed to resume: {}\x1b[0m", e),
                            }
                            continue;
                        }
                        "/help" => {
                            println!(
                                "\x1b[90mCommands: /clear, /pause, /resume, /pin <id>, /unpin <id>, /quit, /exit, /help\x1b[0m"
                            );
                            continue;
                        }
                        _ if line.starts_with("/pin ") => {
                            let arg = line.strip_prefix("/pin ").unwrap().trim();
                            match arg.parse::<i64>() {
                                Ok(id) => match store.pin_message(conv_name, id, true) {
                                    Ok(_) => println!("\x1b[90mPinned message {}.\x1b[0m", id),
                                    Err(e) => eprintln!("\x1b[31mFailed to pin: {}\x1b[0m", e),
                                },
                                Err(_) => eprintln!("\x1b[33mUsage: /pin <message_id>\x1b[0m"),
                            }
                            continue;
                        }
                        _ if line.starts_with("/unpin ") => {
                            let arg = line.strip_prefix("/unpin ").unwrap().trim();
                            match arg.parse::<i64>() {
                                Ok(id) => match store.pin_message(conv_name, id, false) {
                                    Ok(_) => println!("\x1b[90mUnpinned message {}.\x1b[0m", id),
                                    Err(e) => eprintln!("\x1b[31mFailed to unpin: {}\x1b[0m", e),
                                },
                                Err(_) => eprintln!("\x1b[33mUsage: /unpin <message_id>\x1b[0m"),
                            }
                            continue;
                        }
                        _ => {
                            eprintln!(
                                "\x1b[33mUnknown command. Type /help for available commands.\x1b[0m"
                            );
                            continue;
                        }
                    }
                }

                let mentions = parse_mentions(line);

                if mentions.is_empty() {
                    eprintln!(
                        "\x1b[33m⚠ Use @mentions to notify agents (e.g., @arya what do you think?)\x1b[0m"
                    );
                    continue;
                }

                // @mention-as-invite: Add new mentions as participants
                for agent_name in &mentions {
                    if agent_name != "all" && !participants.contains(agent_name) {
                        if anima::discovery::agent_exists(agent_name) {
                            if let Err(e) = store.add_participant(conv_name, agent_name) {
                                eprintln!(
                                    "\x1b[33mWarning: Could not add {} as participant: {}\x1b[0m",
                                    agent_name, e
                                );
                            } else {
                                participants.push(agent_name.clone());
                                println!(
                                    "\x1b[90m[@{} joined the conversation]\x1b[0m",
                                    agent_name
                                );
                            }
                        }
                    }
                }

                let expanded_mentions = expand_all_mention(&mentions, &participants);

                let mention_refs: Vec<&str> =
                    expanded_mentions.iter().map(|s| s.as_str()).collect();

                let user_msg_id = store.add_message(conv_name, "user", line, &mention_refs)?;

                last_seen_id.store(user_msg_id, Ordering::SeqCst);

                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as i64;
                let user_msg = anima::conversation::ConversationMessage {
                    id: user_msg_id,
                    conv_name: conv_name.to_string(),
                    from_agent: "user".to_string(),
                    content: line.to_string(),
                    mentions: expanded_mentions.clone(),
                    created_at: now,
                    expires_at: 0,
                    duration_ms: None,
                    tool_calls: None,
                    tokens_in: None,
                    tokens_out: None,
                    num_ctx: None,
                    triggered_by: None,
                    pinned: false,
                    prompt_eval_ns: None,
                    tool_call_id: None,
                    cached_tokens: None,
                    assistant_response: None,
                };
                // Overwrite the prompt line with the formatted message
                print!("\x1b[A\x1b[2K{}", format_message_display(&user_msg));

                // Notify mentioned agents (initial notifications only)
                // The daemon handles all follow-up @mention chains autonomously
                if !expanded_mentions.is_empty() {
                    let notify_results = notify_mentioned_agents_parallel(
                        &store,
                        conv_name,
                        user_msg_id,
                        &expanded_mentions,
                    )
                    .await;

                    for (agent, result) in &notify_results {
                        match result {
                            NotifyResult::Acknowledged => {
                                // Agent received notification, processing async.
                                // The background poller will display the response.
                            }
                            NotifyResult::Notified {
                                response_message_id,
                            } => {
                                // Synchronous response (backwards compat)
                                last_seen_id.store(*response_message_id, Ordering::SeqCst);
                                if let Ok(msgs) =
                                    store.get_messages(conv_name, Some(DEFAULT_CONTEXT_MESSAGES))
                                    && let Some(response_msg) =
                                        msgs.iter().find(|m| m.id == *response_message_id)
                                {
                                    println!(
                                        "\n\x1b[36m[{}]:\x1b[0m {}\n",
                                        agent, response_msg.content
                                    );
                                }
                            }
                            NotifyResult::Queued { notification_id: _ } => {
                                eprintln!(
                                    "\x1b[90m[@{} offline, notification queued]\x1b[0m",
                                    agent
                                );
                            }
                            NotifyResult::UnknownAgent => {
                                eprintln!(
                                    "\x1b[33m[@{} unknown agent - no such agent exists]\x1b[0m",
                                    agent
                                );
                            }
                            NotifyResult::Failed { reason } => {
                                eprintln!(
                                    "\x1b[33m[@{} notification failed: {}]\x1b[0m",
                                    agent, reason
                                );
                            }
                        }
                    }
                }
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
                println!("\n\x1b[33mInterrupted. Goodbye!\x1b[0m");
                break;
            }
            Err(rustyline::error::ReadlineError::Eof) => {
                println!("\n\x1b[33mGoodbye!\x1b[0m");
                break;
            }
            Err(e) => {
                eprintln!("\x1b[31mInput error: {}\x1b[0m", e);
                break;
            }
        }
    }

    shutdown.store(true, Ordering::SeqCst);
    let _ = poll_handle.await;

    Ok(())
}

// ---------------------------------------------------------------------------
// Agent status / listing
// ---------------------------------------------------------------------------

/// Show status of all agents (running/stopped).
async fn show_status() {
    use anima::socket_api::AgentState;

    let Some(entries) = list_agent_dirs() else {
        return;
    };

    println!(
        "\x1b[1m{:<20} {:<10} {:<10}\x1b[0m",
        "AGENT", "STATUS", "PID"
    );
    println!("{}", "-".repeat(42));

    for entry in entries {
        let name = entry.file_name().to_string_lossy().to_string();
        let agent_path = entry.path();
        let pid_path = agent_path.join("daemon.pid");

        let (status, pid) = if PidFile::is_running(&pid_path) {
            let pid = PidFile::read(&pid_path).unwrap_or(0);
            // Try to query the socket for idle/working state
            let socket_path = agent_path.join("agent.sock");
            let state_str = match UnixStream::connect(&socket_path).await {
                Ok(stream) => {
                    let mut api = SocketApi::new(stream);
                    if api.write_request(&Request::Status).await.is_ok() {
                        match tokio::time::timeout(
                            Duration::from_secs(2),
                            api.read_response(),
                        ).await {
                            Ok(Ok(Some(Response::Status { state, .. }))) => match state {
                                AgentState::Idle => "\x1b[32midle\x1b[0m",
                                AgentState::Working => "\x1b[33mworking\x1b[0m",
                            },
                            _ => "\x1b[32mrunning\x1b[0m",
                        }
                    } else {
                        "\x1b[32mrunning\x1b[0m"
                    }
                }
                Err(_) => "\x1b[32mrunning\x1b[0m",
            };
            (state_str, pid.to_string())
        } else {
            ("\x1b[90mstopped\x1b[0m", "-".to_string())
        };

        println!("{:<20} {:<19} {:<10}", name, status, pid);
    }
}

// ---------------------------------------------------------------------------
// Ask (one-shot query)
// ---------------------------------------------------------------------------

/// Query an agent with conversation persistence.
/// Starts daemon if not running, creates/uses conversation named after agent,
/// stores user message, streams response, and daemon stores agent response.
/// Format verbose output to stderr with color-coded prefixes.
fn format_verbose_output(kind: &str, data: &serde_json::Value) {
    match kind {
        "thinking" => {
            let text = data["content"].as_str().unwrap_or("");
            if !text.is_empty() {
                // Dim gray
                eprintln!("\x1b[2m[thinking] {}\x1b[0m", text);
            }
        }
        "usage" => {
            let tokens_in = data["tokens_in"].as_u64().unwrap_or(0);
            let tokens_out = data["tokens_out"].as_u64().unwrap_or(0);
            let cached = data["cached_tokens"].as_u64().unwrap_or(0);
            let duration_ms = data["duration_ms"].as_u64().unwrap_or(0);
            // Cyan
            let cached_str = if cached > 0 {
                format!(" ({} cached)", cached)
            } else {
                String::new()
            };
            eprintln!(
                "\x1b[36m[usage] {}in \u{2192} {}out{} in {}ms\x1b[0m",
                tokens_in, tokens_out, cached_str, duration_ms
            );
        }
        "tool_call" => {
            let name = data["name"].as_str().unwrap_or("?");
            let args = &data["arguments"];
            let summary = format_tool_params(args);
            // Yellow
            eprintln!("\x1b[33m[tool] {}: {}\x1b[0m", name, summary);
        }
        "tool_result" => {
            let name = data["name"].as_str().unwrap_or("?");
            let success = data["success"].as_bool().unwrap_or(true);
            let duration_ms = data["duration_ms"].as_u64().unwrap_or(0);
            let preview = data["preview"].as_str().unwrap_or("");
            let status = if success { "ok" } else { "error" };
            eprintln!(
                "\x1b[33m[tool-result] {} ({}) {}ms\x1b[0m",
                name, status, duration_ms
            );
            if !preview.is_empty() {
                eprintln!("{}", preview);
            }
        }
        "tools" => {
            let count = data["count"].as_u64().unwrap_or(0);
            let names = data["names"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
            eprintln!("\x1b[36m[tools] {} tools: {}\x1b[0m", count, names);
        }
        "user_prompt" => {
            let content = data["content"].as_str().unwrap_or("");
            if !content.is_empty() {
                eprintln!("\x1b[32m--- user prompt ---\x1b[0m");
                eprintln!("\x1b[32m{}\x1b[0m", content);
                eprintln!("\x1b[32m--- end prompt ---\x1b[0m");
            }
        }
        "state_frontmatter" => {
            let state = data["state"].as_str().unwrap_or("?");
            let wait = data["wait"].as_bool().unwrap_or(false);
            let tools = &data["tools"];
            let tools_str = match tools {
                serde_json::Value::Bool(b) => format!("{}", b),
                _ => "default".to_string(),
            };
            eprintln!(
                "\x1b[31m[frontmatter] state={} wait={} tools={}\x1b[0m",
                state, wait, tools_str
            );
        }
        "memory" => {
            let content = data["content"].as_str().unwrap_or("");
            if !content.is_empty() {
                // Magenta — show first 10 lines
                let lines: Vec<&str> = content.lines().take(10).collect();
                eprintln!("\x1b[35m[memory] {}\x1b[0m", lines.join("\n  "));
                let total = content.lines().count();
                if total > 10 {
                    eprintln!("\x1b[35m  ... ({} more lines)\x1b[0m", total - 10);
                }
            }
        }
        "retry" => {
            let attempt = data["attempt"].as_u64().unwrap_or(0);
            let delay_ms = data["delay_ms"].as_u64().unwrap_or(0);
            let operation = data["operation"].as_str().unwrap_or("unknown");
            // Red
            eprintln!(
                "\x1b[31m[retry] {} attempt {} (delay {}ms)\x1b[0m",
                operation, attempt, delay_ms
            );
        }
        _ => {
            // Gray fallback
            let json = serde_json::to_string(data).unwrap_or_default();
            eprintln!("\x1b[2m[verbose:{}] {}\x1b[0m", kind, json);
        }
    }
}

/// Stream agent response from a SocketApi connection.
/// Prints chunks to stdout, verbose/tool events to stderr.
async fn stream_agent_response(api: &mut SocketApi, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
    let mut had_chunks = false;
    let mut in_thinking = false;
    loop {
        match api
            .read_response()
            .await
            .map_err(|e| format!("Failed to read response: {}", e))?
        {
            Some(Response::Chunk { text }) => {
                // Detect <think> / </think> boundaries
                if text == "<think>" {
                    in_thinking = true;
                    if !had_chunks {
                        print!("\x1b[2m<think>\n");
                    } else {
                        print!("\x1b[0m\x1b[2m<think>\n");
                    }
                    had_chunks = true;
                    io::stdout().flush()?;
                    continue;
                }
                if text.starts_with("</think>") {
                    in_thinking = false;
                    print!("</think>\x1b[0m\x1b[33m");
                    let remainder = text.trim_start_matches("</think>");
                    if !remainder.is_empty() {
                        print!("{}", remainder);
                    }
                    io::stdout().flush()?;
                    continue;
                }

                // Detect python output blocks — render in cyan
                if text.contains("<python-output") {
                    if had_chunks {
                        print!("\x1b[0m");
                    }
                    print!("\x1b[36m{}\x1b[0m", text);
                    if !in_thinking {
                        print!("\x1b[33m");
                    } else {
                        print!("\x1b[2m");
                    }
                    had_chunks = true;
                    io::stdout().flush()?;
                    continue;
                }

                if !had_chunks {
                    if in_thinking {
                        print!("\x1b[2m");
                    } else {
                        print!("\x1b[33m");
                    }
                }
                had_chunks = true;
                print!("{}", text);
                io::stdout().flush()?;
            }
            Some(Response::ToolCall { tool, params }) => {
                if had_chunks {
                    eprintln!();
                    had_chunks = false;
                }
                let param_summary = format_tool_params(&params);
                eprintln!(" - [tool] {}: {}", tool, param_summary);
            }
            Some(Response::Verbose { kind, data }) => {
                if kind == "usage" {
                    // Always show usage in dim after response
                    let tokens_in = data["tokens_in"].as_u64().unwrap_or(0);
                    let tokens_out = data["tokens_out"].as_u64().unwrap_or(0);
                    let cached = data["cached_tokens"].as_u64().unwrap_or(0);
                    let duration_ms = data["duration_ms"].as_u64().unwrap_or(0);
                    let cached_str = if cached > 0 {
                        format!(", {} cached", cached)
                    } else {
                        String::new()
                    };
                    if had_chunks {
                        eprintln!();
                        had_chunks = false;
                    }
                    eprintln!(
                        "\x1b[2m  [{} in \u{2192} {} out{}, {}ms]\x1b[0m",
                        tokens_in, tokens_out, cached_str, duration_ms
                    );
                } else if verbose {
                    if had_chunks {
                        eprintln!();
                        had_chunks = false;
                    }
                    format_verbose_output(&kind, &data);
                }
            }
            Some(Response::Done) => {
                if had_chunks {
                    print!("\x1b[0m");
                }
                println!("\n");
                break;
            }
            Some(Response::Message { content }) => {
                println!("{}", content);
                break;
            }
            Some(Response::Error { message }) => {
                eprintln!("Error from agent: {}", message);
                std::process::exit(1);
            }
            None => {
                eprintln!("Connection closed unexpectedly");
                std::process::exit(1);
            }
            _ => {}
        }
    }
    Ok(())
}

async fn ask_agent(agent: &str, message: &str, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
    use anima::discovery;

    let agent_path = resolve_agent_path(agent);

    if !agent_path.exists() {
        return Err(format!("Agent '{}' not found at {}", agent, agent_path.display()).into());
    }

    let agent_dir = AgentDir::load(&agent_path)
        .map_err(|e| format!("Failed to load agent '{}': {}", agent, e))?;
    let agent_name = agent_dir.config.agent.name.clone();

    if !discovery::is_agent_running(&agent_name) {
        eprintln!("\x1b[33mStarting daemon for '{}'...\x1b[0m", agent_name);

        let exe =
            std::env::current_exe().map_err(|e| format!("Failed to get executable path: {}", e))?;

        Command::new(&exe)
            .args(["start", &agent_name])
            .spawn()
            .map_err(|e| format!("Failed to start daemon: {}", e))?;

        for _ in 0..20 {
            tokio::time::sleep(Duration::from_millis(100)).await;
            if discovery::is_agent_running(&agent_name) {
                break;
            }
        }

        if !discovery::is_agent_running(&agent_name) {
            return Err(format!("Daemon for '{}' failed to start", agent_name).into());
        }
    }

    let store = ConversationStore::init()?;
    let conv_name = agent_name.clone();

    if store.find_by_name(&conv_name)?.is_none() {
        store.create_conversation(Some(&conv_name), &["user", &agent_name])?;
    }

    store.add_message(&conv_name, "user", message, &[])?;

    let mut api = connect_to_agent(&agent_name).await?;

    api.write_request(&Request::Message {
        content: message.to_string(),
        conv_name: Some(conv_name),
        verbose,
        max_steps: None,
    })
    .await
    .map_err(|e| format!("Failed to send message: {}", e))?;

    stream_agent_response(&mut api, verbose).await
}

/// After a debug/step, fetch and display all messages created since `since_id`.
/// Shows tool results, recall injections, and agent messages that aren't visible in the stream.
fn dump_step_messages(conv_name: &str, since_id: i64) -> Result<(), Box<dyn std::error::Error>> {
    let store = ConversationStore::init()?;
    let messages = store.get_messages_filtered(conv_name, None, Some(since_id))?;
    if messages.is_empty() {
        return Ok(());
    }
    eprintln!("\x1b[2m--- step trace ({} messages) ---\x1b[0m", messages.len());
    for msg in &messages {
        let role = msg.from_agent.as_str();
        let display_content = if let Some(ref json) = msg.assistant_response {
            serde_json::from_str::<serde_json::Value>(json)
                .ok()
                .and_then(|v| v["content"].as_str().map(String::from))
                .unwrap_or_default()
        } else {
            msg.content.clone()
        };
        let preview = if display_content.is_empty() {
            "(empty)".to_string()
        } else {
            step_trace_preview(&display_content)
        };

        match role {
            "recall" => {
                // Dim gray for recall injections
                eprintln!("\x1b[2m  [recall] {}\x1b[0m", preview);
            }
            "tool" => {
                // Yellow for tool results
                let tc_id = msg.tool_call_id.as_deref().unwrap_or("");
                let id_suffix = if tc_id.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", tc_id)
                };
                eprintln!("\x1b[33m  [tool{}]\x1b[0m", id_suffix);
                eprintln!("{}", display_content);
            }
            _ => {
                // Cyan for agent messages
                let tc_summary = msg.tool_calls.as_ref().map(|tc| {
                    // Parse tool_calls JSON to show tool names
                    if let Ok(calls) = serde_json::from_str::<Vec<serde_json::Value>>(tc) {
                        let names: Vec<&str> = calls.iter()
                            .filter_map(|c| c["function"]["name"].as_str())
                            .collect();
                        if names.is_empty() { String::new() } else { format!(" -> [{}]", names.join(", ")) }
                    } else {
                        String::new()
                    }
                }).unwrap_or_default();
                eprintln!("\x1b[36m  [{}] {}{}\x1b[0m", role, preview, tc_summary);
            }
        }
    }
    eprintln!("\x1b[2m--- end trace ---\x1b[0m");
    Ok(())
}

/// Step-debug: fresh start. Restart daemon, delete & recreate conversation,
/// send user message, process one step, stream response.
async fn start_agent_debug(agent: &str, message: &str) -> Result<(), Box<dyn std::error::Error>> {
    use anima::discovery;

    let agent_path = resolve_agent_path(agent);

    if !agent_path.exists() {
        return Err(format!("Agent '{}' not found at {}", agent, agent_path.display()).into());
    }

    let agent_dir = AgentDir::load(&agent_path)
        .map_err(|e| format!("Failed to load agent '{}': {}", agent, e))?;
    let agent_name = agent_dir.config.agent.name.clone();

    // Stop daemon if running
    if discovery::is_agent_running(&agent_name) {
        eprintln!("\x1b[33mStopping daemon for '{}'...\x1b[0m", agent_name);
        stop_agent_impl(&agent_name, true, true).await?;
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    // Start daemon
    eprintln!("\x1b[33mStarting daemon for '{}'...\x1b[0m", agent_name);
    start_agent_impl(&agent_name, true)?;

    for _ in 0..20 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        if discovery::is_agent_running(&agent_name) {
            break;
        }
    }

    if !discovery::is_agent_running(&agent_name) {
        return Err(format!("Daemon for '{}' failed to start", agent_name).into());
    }

    // Delete and recreate conversation
    let store = ConversationStore::init()?;
    let conv_name = agent_name.clone();

    if store.find_by_name(&conv_name)?.is_some() {
        store.delete_conversation(&conv_name)?;
    }
    store.create_conversation(Some(&conv_name), &["user", &agent_name])?;

    // Store user message
    store.add_message(&conv_name, "user", message, &[])?;

    // Capture last message ID before the step
    let last_id = store.get_messages(&conv_name, Some(1))?
        .first().map(|m| m.id).unwrap_or(0);

    // Connect and send with max_steps=1
    let mut api = connect_to_agent(&agent_name).await?;

    api.write_request(&Request::Message {
        content: message.to_string(),
        conv_name: Some(conv_name.clone()),
        verbose: true,
        max_steps: Some(1),
    })
    .await
    .map_err(|e| format!("Failed to send message: {}", e))?;

    stream_agent_response(&mut api, true).await?;
    dump_step_messages(&conv_name, last_id)
}

/// Step-debug: continue from current conversation state.
/// Process one more step of the tool loop.
async fn step_agent(agent: &str) -> Result<(), Box<dyn std::error::Error>> {
    use anima::discovery;

    let agent_path = resolve_agent_path(agent);

    if !agent_path.exists() {
        return Err(format!("Agent '{}' not found at {}", agent, agent_path.display()).into());
    }

    let agent_dir = AgentDir::load(&agent_path)
        .map_err(|e| format!("Failed to load agent '{}': {}", agent, e))?;
    let agent_name = agent_dir.config.agent.name.clone();

    if !discovery::is_agent_running(&agent_name) {
        return Err(format!(
            "Agent '{}' is not running. Use `anima agent debug {} \"<message>\"` first.",
            agent_name, agent
        ).into());
    }

    // Capture last message ID before the step
    let conv_name = agent_name.clone();
    let store = ConversationStore::init()?;
    let last_id = store.get_messages(&conv_name, Some(1))?
        .first().map(|m| m.id).unwrap_or(0);

    let mut api = connect_to_agent(&agent_name).await?;

    api.write_request(&Request::Message {
        content: String::new(),
        conv_name: Some(conv_name.clone()),
        verbose: true,
        max_steps: Some(1),
    })
    .await
    .map_err(|e| format!("Failed to send message: {}", e))?;

    stream_agent_response(&mut api, true).await?;
    dump_step_messages(&conv_name, last_id)
}

// ---------------------------------------------------------------------------
// Daemon lifecycle (start / stop / restart)
// ---------------------------------------------------------------------------

/// Start an agent daemon in the background.
/// If `quiet` is true, suppresses output (used by batch operations).
fn start_agent_impl(agent: &str, quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    let agent_path = resolve_agent_path(agent);
    let pid_path = agent_path.join("daemon.pid");

    if !agent_path.exists() {
        return Err(format!("Agent '{}' not found at {}", agent, agent_path.display()).into());
    }

    if PidFile::is_running(&pid_path) {
        let pid = PidFile::read(&pid_path).unwrap_or(0);
        return Err(format!("Agent '{}' is already running (pid {})", agent, pid).into());
    }

    let exe =
        std::env::current_exe().map_err(|e| format!("Failed to get current executable: {}", e))?;

    let log_path = agent_path.join("agent.log");
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| format!("Failed to open log file: {}", e))?;
    let log_file_err = log_file
        .try_clone()
        .map_err(|e| format!("Failed to clone log file handle: {}", e))?;

    let child = Command::new(&exe)
        .arg("run")
        .arg(agent)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::from(log_file))
        .stderr(std::process::Stdio::from(log_file_err))
        .spawn()
        .map_err(|e| format!("Failed to spawn daemon: {}", e))?;

    let pid = child.id();
    if !quiet {
        println!("Started {} (pid {})", agent, pid);
    }

    Ok(())
}

/// Start an agent daemon in the background.
/// Supports glob patterns (*, ?) for matching multiple agents.
fn start_agent(agent: &str) -> Result<(), Box<dyn std::error::Error>> {
    use anima::discovery::match_agents;

    if agent == "all" {
        return start_all_agents();
    }

    if has_wildcards(agent) {
        let matches = match_agents(agent);

        if matches.is_empty() {
            return Err(format!("No agents match pattern: {}", agent).into());
        }

        let mut started = Vec::new();
        for name in &matches {
            match start_agent_impl(name, true) {
                Ok(()) => started.push(name.clone()),
                Err(e) => eprintln!("Failed to start {}: {}", name, e),
            }
        }

        if !started.is_empty() {
            println!("Started {} agent(s): {}", started.len(), started.join(", "));
        }

        return Ok(());
    }

    start_agent_impl(agent, false)
}

/// Start all configured agents that are not currently running.
fn start_all_agents() -> Result<(), Box<dyn std::error::Error>> {
    use anima::discovery::{is_agent_running, list_saved_agents};

    let all_agents = list_saved_agents();

    if all_agents.is_empty() {
        println!("No configured agents found");
        return Ok(());
    }

    let stopped_agents: Vec<_> = all_agents
        .into_iter()
        .filter(|name| !is_agent_running(name))
        .collect();

    if stopped_agents.is_empty() {
        println!("All agents already running");
        return Ok(());
    }

    let mut started = Vec::new();

    for agent in &stopped_agents {
        match start_agent_impl(agent, true) {
            Ok(()) => started.push(agent.clone()),
            Err(e) => eprintln!("Failed to start {}: {}", agent, e),
        }
    }

    if !started.is_empty() {
        println!("Started {} agent(s): {}", started.len(), started.join(", "));
    }

    Ok(())
}

/// Stop a running agent daemon.
/// - `quiet`: suppress output (used by batch operations)
/// - `force`: if true, SIGKILL without prompting on timeout; if false, prompt user
async fn stop_agent_impl(
    agent: &str,
    quiet: bool,
    force: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let agent_path = resolve_agent_path(agent);
    let pid_path = agent_path.join("daemon.pid");

    if !PidFile::is_running(&pid_path) {
        if !quiet {
            println!("Agent '{}' is not running", agent);
        }
        return Ok(());
    }

    let pid = PidFile::read(&pid_path).unwrap_or(0);

    // Try graceful shutdown via socket first
    match connect_to_agent(agent).await {
        Ok(mut api) => {
            if let Err(e) = api.write_request(&Request::Shutdown).await {
                if !quiet {
                    eprintln!("Warning: Failed to send shutdown request: {}", e);
                }
            } else {
                for _ in 0..10 {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    if !PidFile::is_running(&pid_path) {
                        if !quiet {
                            println!("Stopped {}", agent);
                        }
                        return Ok(());
                    }
                }
            }
        }
        Err(_) => {
            // Socket unavailable (daemon still initializing, or socket already cleaned up).
            // Fall through to SIGTERM, which handles this cleanly.
        }
    }

    // Fall back to SIGTERM
    unsafe {
        if libc::kill(pid as i32, libc::SIGTERM) == 0 {
            for _ in 0..30 {
                tokio::time::sleep(Duration::from_millis(100)).await;
                if !PidFile::is_running(&pid_path) {
                    if !quiet {
                        println!("Stopped {}", agent);
                    }
                    return Ok(());
                }
            }

            if !quiet {
                eprintln!("Warning: Agent did not stop within timeout");
            }

            let should_kill = if force {
                true
            } else if quiet {
                false
            } else {
                use std::io::{self, Write};
                print!("Stop timed out, do you want to kill {}? (y/N) ", agent);
                io::stdout().flush().ok();
                let mut input = String::new();
                if io::stdin().read_line(&mut input).is_ok() {
                    input.trim().eq_ignore_ascii_case("y")
                        || input.trim().eq_ignore_ascii_case("yes")
                } else {
                    false
                }
            };

            if should_kill {
                if libc::kill(pid as i32, libc::SIGKILL) == 0 {
                    for _ in 0..10 {
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        if !PidFile::is_running(&pid_path) {
                            break;
                        }
                    }
                    if !PidFile::is_running(&pid_path) {
                        let _ = std::fs::remove_file(&pid_path);
                    }
                    if !quiet {
                        println!("Killed {}", agent);
                    }
                } else if !quiet {
                    eprintln!("Warning: Failed to send SIGKILL to pid {}", pid);
                }
                return Ok(());
            }
            if !quiet {
                println!("Agent {} still running (pid {})", agent, pid);
            }
            return Ok(());
        } else if !quiet {
            eprintln!("Warning: Failed to send SIGTERM to pid {}", pid);
        }
    }

    if !quiet {
        println!("Stopped {}", agent);
    }
    Ok(())
}

/// Stop a running agent daemon.
/// Supports glob patterns (*, ?) for matching multiple agents.
async fn stop_agent(agent: &str) -> Result<(), Box<dyn std::error::Error>> {
    use anima::discovery::match_agents;

    if agent == "all" {
        return stop_all_agents().await;
    }

    if has_wildcards(agent) {
        let matches = match_agents(agent);

        if matches.is_empty() {
            return Err(format!("No agents match pattern: {}", agent).into());
        }

        let mut stopped = Vec::new();
        for name in &matches {
            match stop_agent_impl(name, true, true).await {
                Ok(()) => stopped.push(name.clone()),
                Err(e) => eprintln!("Failed to stop {}: {}", name, e),
            }
        }

        if !stopped.is_empty() {
            println!("Stopped {} agent(s): {}", stopped.len(), stopped.join(", "));
        }

        return Ok(());
    }

    stop_agent_impl(agent, false, false).await
}

/// Stop all running agent daemons.
async fn stop_all_agents() -> Result<(), Box<dyn std::error::Error>> {
    use anima::discovery::discover_running_agents;

    let running_agents = discover_running_agents();

    if running_agents.is_empty() {
        println!("No running agents to stop");
        return Ok(());
    }

    let mut stopped_count = 0;

    for agent in &running_agents {
        print!("Stopping {}... ", agent.name);
        io::stdout().flush()?;

        match stop_agent_impl(&agent.name, true, true).await {
            Ok(()) => {
                println!("stopped");
                stopped_count += 1;
            }
            Err(e) => {
                println!("failed: {}", e);
            }
        }
    }

    println!("\nStopped {} agent(s)", stopped_count);
    Ok(())
}

/// Stop and restart a single agent, printing progress. Returns true on success.
async fn restart_single_agent(name: &str) -> bool {
    print!("Restarting {}... ", name);
    let _ = io::stdout().flush();

    if let Err(e) = stop_agent_impl(name, true, true).await {
        println!("failed to stop: {}", e);
        return false;
    }

    tokio::time::sleep(Duration::from_millis(200)).await;

    match start_agent_impl(name, true) {
        Ok(()) => {
            let pid_path = resolve_agent_path(name).join("daemon.pid");
            let mut pid = 0u32;
            for _ in 0..20 {
                tokio::time::sleep(Duration::from_millis(50)).await;
                if let Ok(p) = PidFile::read(&pid_path)
                    && p > 0
                {
                    pid = p;
                    break;
                }
            }
            println!("done (pid {})", pid);
            true
        }
        Err(e) => {
            println!("failed to start: {}", e);
            false
        }
    }
}

/// Restart a running agent daemon (stop then start).
/// Supports glob patterns (*, ?) for matching multiple agents.
/// If agent is "all", restarts all currently running agents.
async fn restart_agent(agent: &str) -> Result<(), Box<dyn std::error::Error>> {
    use anima::discovery::match_agents;

    if agent == "all" {
        return restart_all_agents().await;
    }

    if has_wildcards(agent) {
        let matches = match_agents(agent);

        if matches.is_empty() {
            return Err(format!("No agents match pattern: {}", agent).into());
        }

        // Only restart agents that are currently running
        let running_matches: Vec<_> = matches
            .iter()
            .filter(|name| {
                let pid_path = resolve_agent_path(name).join("daemon.pid");
                PidFile::is_running(&pid_path)
            })
            .collect();

        if running_matches.is_empty() {
            println!("No running agents match pattern: {}", agent);
            return Ok(());
        }

        let mut restarted = Vec::new();
        for name in &running_matches {
            if restart_single_agent(name).await {
                restarted.push(name.to_string());
            }
        }

        println!(
            "Restarted {} agent{}",
            restarted.len(),
            if restarted.len() == 1 { "" } else { "s" }
        );

        return Ok(());
    }

    // Single agent (no wildcards)
    let agent_path = resolve_agent_path(agent);
    let pid_path = agent_path.join("daemon.pid");

    if !agent_path.exists() {
        return Err(format!("Agent '{}' not found at {}", agent, agent_path.display()).into());
    }

    let was_running = PidFile::is_running(&pid_path);

    if was_running {
        println!("Stopping {}...", agent);
        stop_agent(agent).await?;
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    println!("Starting {}...", agent);
    start_agent(agent)?;

    Ok(())
}

/// Restart all currently running agent daemons.
async fn restart_all_agents() -> Result<(), Box<dyn std::error::Error>> {
    use anima::discovery::discover_running_agents;

    let running_agents = discover_running_agents();

    if running_agents.is_empty() {
        println!("No running agents to restart");
        return Ok(());
    }

    let mut restarted_count = 0;

    for agent in &running_agents {
        if restart_single_agent(&agent.name).await {
            restarted_count += 1;
        }
    }

    println!(
        "Restarted {} agent{}",
        restarted_count,
        if restarted_count == 1 { "" } else { "s" }
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Daemon socket commands (clear, system, heartbeat)
// ---------------------------------------------------------------------------

/// Clear conversation history for a running agent daemon.
async fn clear_agent(agent: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut api = connect_to_agent(agent).await?;

    api.write_request(&Request::Clear)
        .await
        .map_err(|e| format!("Failed to send clear request: {}", e))?;

    match api
        .read_response()
        .await
        .map_err(|e| format!("Failed to read response: {}", e))?
    {
        Some(Response::Ok) => {
            println!("Cleared conversation for {}", agent);
        }
        Some(Response::Error { message }) => {
            return Err(format!("Error from agent: {}", message).into());
        }
        Some(other) => {
            return Err(format!("Unexpected response: {:?}", other).into());
        }
        None => {
            return Err("Connection closed unexpectedly".into());
        }
    }

    Ok(())
}

/// Show the system prompt for a running agent daemon.
async fn show_system_prompt(agent: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut api = connect_to_agent(agent).await?;

    api.write_request(&Request::System)
        .await
        .map_err(|e| format!("Failed to send system request: {}", e))?;

    match api
        .read_response()
        .await
        .map_err(|e| format!("Failed to read response: {}", e))?
    {
        Some(Response::System { system_prompt }) => {
            println!("{}", system_prompt);
        }
        Some(Response::Error { message }) => {
            return Err(format!("Error from agent: {}", message).into());
        }
        Some(other) => {
            return Err(format!("Unexpected response: {:?}", other).into());
        }
        None => {
            return Err("Connection closed unexpectedly".into());
        }
    }

    Ok(())
}

/// Trigger a heartbeat for a running agent daemon.
async fn trigger_heartbeat(agent: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut api = connect_to_agent(agent).await?;

    api.write_request(&Request::Heartbeat)
        .await
        .map_err(|e| format!("Failed to send heartbeat request: {}", e))?;

    match api
        .read_response()
        .await
        .map_err(|e| format!("Failed to read response: {}", e))?
    {
        Some(Response::HeartbeatTriggered) => {
            println!("Heartbeat triggered for {}", agent);
        }
        Some(Response::HeartbeatNotConfigured) => {
            println!(
                "Heartbeat not configured for {} (add [heartbeat] section to config.toml)",
                agent
            );
        }
        Some(Response::Error { message }) => {
            return Err(format!("Error from agent: {}", message).into());
        }
        Some(other) => {
            return Err(format!("Unexpected response: {:?}", other).into());
        }
        None => {
            return Err("Connection closed unexpectedly".into());
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Auth commands
// ---------------------------------------------------------------------------

async fn handle_login() -> Result<(), Box<dyn std::error::Error>> {
    let (url, verifier, expected_state) = anima::auth::build_auth_url();

    println!("Open this URL in your browser to log in:\n");
    println!("  {}\n", url);

    // Try to open the browser automatically
    if let Err(_) = Command::new("xdg-open")
        .arg(&url)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        // Also try macOS open
        let _ = Command::new("open")
            .arg(&url)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
    }

    print!("Paste the code from your browser: ");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let input = input.trim();

    // The callback returns code#state
    let (code, state) = if let Some(pos) = input.rfind('#') {
        (&input[..pos], &input[pos + 1..])
    } else {
        (input, expected_state.as_str())
    };

    let tokens = anima::auth::exchange_code(code, state, &expected_state, &verifier)
        .await
        .map_err(|e| e.to_string())?;
    anima::auth::save_tokens(&tokens).map_err(|e| e.to_string())?;

    let expires = chrono::DateTime::from_timestamp(tokens.expires_at, 0)
        .map(|dt| dt.format("%Y-%m-%d %H:%M UTC").to_string())
        .unwrap_or_else(|| "unknown".to_string());

    println!("Logged in successfully! Token expires: {}", expires);
    Ok(())
}

fn handle_whoami() {
    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        let masked = if key.len() > 8 {
            format!("{}...{}", &key[..4], &key[key.len() - 4..])
        } else {
            "****".to_string()
        };
        println!("Anthropic API key configured ({})", masked);
    }

    match anima::auth::load_tokens() {
        Some(tokens) => {
            let expires = chrono::DateTime::from_timestamp(tokens.expires_at, 0)
                .map(|dt| dt.format("%Y-%m-%d %H:%M UTC").to_string())
                .unwrap_or_else(|| "unknown".to_string());
            let now = chrono::Utc::now().timestamp();
            let status = if now < tokens.expires_at {
                "active"
            } else {
                "expired (will refresh on next use)"
            };
            println!(
                "Anthropic subscription: {} (expires: {})",
                status, expires
            );
        }
        None => {
            if std::env::var("ANTHROPIC_API_KEY").is_err() {
                println!("Not logged in. Run `anima login` or set ANTHROPIC_API_KEY.");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Chat subcommands
// ---------------------------------------------------------------------------

/// Handle chat subcommands.
async fn handle_chat_command(
    command: Option<ChatCommands>,
) -> Result<(), Box<dyn std::error::Error>> {
    let store = ConversationStore::init()?;

    match command {
        Some(ChatCommands::New { name }) => {
            let conv_name = store.create_conversation(name.as_deref(), &["user"])?;
            println!("Created conversation '\x1b[36m{}\x1b[0m'\n", conv_name);
            chat_with_conversation(&conv_name).await?;
        }

        Some(ChatCommands::Create { name }) => {
            let conv_name = store.create_conversation(name.as_deref(), &["user"])?;
            println!("Created conversation: {}", conv_name);
        }

        Some(ChatCommands::Join { name }) => {
            if store.find_by_name(&name)?.is_none() {
                return Err(format!(
                    "Conversation '{}' not found. Use 'anima chat create {}' to create it.",
                    name, name
                )
                .into());
            }

            chat_with_conversation(&name).await?;
        }

        Some(ChatCommands::Send { conv, message }) => {
            if store.find_by_name(&conv)?.is_none() {
                return Err(format!("Conversation '{}' not found", conv).into());
            }

            let mentions = parse_mentions(&message);

            let parts = store.get_participants(&conv)?;
            let participants: Vec<String> = parts
                .iter()
                .filter(|p| p.agent != "user")
                .map(|p| p.agent.clone())
                .collect();

            for agent_name in &mentions {
                if agent_name != "all"
                    && !participants.contains(agent_name)
                    && anima::discovery::agent_exists(agent_name)
                    && let Err(e) = store.add_participant(&conv, agent_name)
                {
                    eprintln!(
                        "Warning: Could not add {} as participant: {}",
                        agent_name, e
                    );
                }
            }

            let updated_parts = store.get_participants(&conv)?;
            let updated_participants: Vec<String> = updated_parts
                .iter()
                .filter(|p| p.agent != "user")
                .map(|p| p.agent.clone())
                .collect();
            let expanded_mentions = expand_all_mention(&mentions, &updated_participants);

            let mention_refs: Vec<&str> = expanded_mentions.iter().map(|s| s.as_str()).collect();

            let user_msg_id = store.add_message(&conv, "user", &message, &mention_refs)?;

            if !expanded_mentions.is_empty() {
                let results = notify_mentioned_agents_parallel(
                    &store,
                    &conv,
                    user_msg_id,
                    &expanded_mentions,
                )
                .await;

                let mut notified = Vec::new();
                let mut queued = Vec::new();
                let mut failed = Vec::new();

                for (agent, result) in results {
                    match result {
                        NotifyResult::Acknowledged | NotifyResult::Notified { .. } => {
                            notified.push(agent)
                        }
                        NotifyResult::Queued { .. } => queued.push(agent),
                        NotifyResult::UnknownAgent => failed.push(format!("{} (unknown)", agent)),
                        NotifyResult::Failed { reason } => {
                            failed.push(format!("{} ({})", agent, reason))
                        }
                    }
                }

                if !notified.is_empty() {
                    println!(
                        "Sent message to '{}', notified: {}",
                        conv,
                        notified.join(", ")
                    );
                }
                if !queued.is_empty() {
                    println!("Queued for offline agents: {}", queued.join(", "));
                }
                if !failed.is_empty() {
                    eprintln!("Failed to notify: {}", failed.join(", "));
                }
            } else {
                println!("Sent message to '{}' (no @mentions)", conv);
            }
        }

        Some(ChatCommands::View {
            conv,
            limit,
            since,
            json,
            pretty,
        }) => {
            if store.find_by_name(&conv)?.is_none() {
                return Err(format!("Conversation '{}' not found", conv).into());
            }

            let messages = store.get_messages_filtered(&conv, limit, since)?;

            if json || pretty {
                let json_messages: Vec<serde_json::Value> = messages
                    .iter()
                    .map(|msg| {
                        let tool_calls_val = msg.tool_calls.as_ref()
                            .and_then(|tc| serde_json::from_str::<serde_json::Value>(tc).ok());
                        serde_json::json!({
                            "id": msg.id,
                            "conv_name": msg.conv_name,
                            "from_agent": msg.from_agent,
                            "content": msg.content,
                            "mentions": msg.mentions,
                            "created_at": msg.created_at,
                            "expires_at": msg.expires_at,
                            "duration_ms": msg.duration_ms,
                            "tool_calls": tool_calls_val,
                            "tokens_in": msg.tokens_in,
                            "tokens_out": msg.tokens_out,
                            "num_ctx": msg.num_ctx,
                            "triggered_by": msg.triggered_by,
                            "pinned": msg.pinned,
                            "prompt_eval_ns": msg.prompt_eval_ns,
                        })
                    })
                    .collect();
                if pretty {
                    let output = serde_json::to_string_pretty(&json_messages)?;
                    let mut result = String::with_capacity(output.len());
                    for line in output.lines() {
                        let trimmed = line.trim_start();
                        if trimmed.starts_with("\"content\": \"") {
                            result.push_str(&line.replace("\\n", "\n").replace("\\t", "\t"));
                        } else {
                            result.push_str(line);
                        }
                        result.push('\n');
                    }
                    print!("{}", result.trim_end());
                    println!();
                } else {
                    println!("{}", serde_json::to_string_pretty(&json_messages)?);
                }
            } else {
                let messages: Vec<_> = messages
                    .into_iter()
                    .filter(|m| m.from_agent != "tool" && m.from_agent != "recall")
                    .collect();
                for msg in &messages {
                    print!("{}", format_message_display(msg));
                }
            }
        }

        Some(ChatCommands::Pause { conv, force }) => {
            let matches = store.match_conversations(&conv)?;

            if matches.is_empty() {
                return Err(format!("No conversations match pattern: {}", conv).into());
            }

            if !force
                && (matches.len() > 1 || has_wildcards(&conv))
                && !confirm_action("pause", &matches)?
            {
                println!("Aborted.");
                return Ok(());
            }

            for conv_name in &matches {
                let _ = store.set_paused(conv_name, true)?;
                println!("Paused conversation '\x1b[36m{}\x1b[0m'", conv_name);
            }
        }

        Some(ChatCommands::Stop { conv, force }) => {
            let matches = store.match_conversations(&conv)?;

            if matches.is_empty() {
                return Err(format!("No conversations match pattern: {}", conv).into());
            }

            if !force
                && (matches.len() > 1 || has_wildcards(&conv))
                && !confirm_action("stop", &matches)?
            {
                println!("Aborted.");
                return Ok(());
            }

            for conv_name in &matches {
                let _ = store.set_paused(conv_name, false)?;
                println!(
                    "Stopped conversation '\x1b[36m{}\x1b[0m' (catchup discarded)",
                    conv_name
                );
            }
        }

        Some(ChatCommands::Resume { conv, force }) => {
            let matches = store.match_conversations(&conv)?;

            if matches.is_empty() {
                return Err(format!("No conversations match pattern: {}", conv).into());
            }

            if !force
                && (matches.len() > 1 || has_wildcards(&conv))
                && !confirm_action("resume", &matches)?
            {
                println!("Aborted.");
                return Ok(());
            }

            for conv_name in &matches {
                let paused_at_msg_id = store.set_paused(conv_name, false)?;
                let catchup_count =
                    process_catchup(&store, conv_name, paused_at_msg_id).await;

                if catchup_count > 0 {
                    println!(
                        "Resumed conversation '\x1b[36m{}\x1b[0m' ({} pending item(s) processed)",
                        conv_name, catchup_count
                    );
                } else {
                    println!("Resumed conversation '\x1b[36m{}\x1b[0m'", conv_name);
                }
            }
        }

        Some(ChatCommands::Delete { name, force }) => {
            let matches = store.match_conversations(&name)?;

            if matches.is_empty() {
                return Err(format!("No conversations match pattern: {}", name).into());
            }

            if !force
                && (matches.len() > 1 || has_wildcards(&name))
                && !confirm_action("delete", &matches)?
            {
                println!("Aborted.");
                return Ok(());
            }

            for conv_name in &matches {
                store.delete_conversation(conv_name)?;
                println!("Deleted conversation '\x1b[36m{}\x1b[0m'", conv_name);
            }
        }

        Some(ChatCommands::Clear { conv, force }) => {
            let matches = store.match_conversations(&conv)?;

            if matches.is_empty() {
                return Err(format!("No conversations match pattern: {}", conv).into());
            }

            if !force
                && (matches.len() > 1 || has_wildcards(&conv))
                && !confirm_action("clear", &matches)?
            {
                println!("Aborted.");
                return Ok(());
            }

            for conv_name in &matches {
                let deleted = store.clear_messages(conv_name)?;
                println!(
                    "Cleared {} messages from '\x1b[36m{}\x1b[0m'",
                    deleted, conv_name
                );
            }
        }

        Some(ChatCommands::Pin { conv, id }) => {
            store.pin_message(&conv, id, true)?;
            println!("Pinned message \x1b[36m{}\x1b[0m in '\x1b[36m{}\x1b[0m'", id, conv);
        }

        Some(ChatCommands::Unpin { conv, id }) => {
            store.pin_message(&conv, id, false)?;
            println!("Unpinned message \x1b[36m{}\x1b[0m in '\x1b[36m{}\x1b[0m'", id, conv);
        }

        Some(ChatCommands::Cleanup) => {
            let (messages_deleted, convs_deleted) = store.cleanup_expired()?;
            println!("Cleanup complete:");
            println!("  - {} expired messages deleted", messages_deleted);
            println!("  - {} empty conversations deleted", convs_deleted);
        }

        None => {
            let conversations = store.list_conversations()?;

            if conversations.is_empty() {
                println!("\x1b[33mNo conversations found.\x1b[0m");
                println!("Create one with: \x1b[36manima chat create\x1b[0m");
                return Ok(());
            }

            println!(
                "\x1b[1m{:<30} {:>6}   {:<10} PARTICIPANTS\x1b[0m",
                "NAME", "MSGS", "UPDATED"
            );
            println!("{}", "-".repeat(80));

            for conv in conversations {
                let participants = store.get_participants(&conv.name)?;
                let agents: Vec<_> = participants.iter().map(|p| p.agent.as_str()).collect();

                let msg_count = store.get_message_count(&conv.name)?;
                let updated = format_relative_time(conv.updated_at);

                let name_display = if conv.is_paused() {
                    format!("{} \x1b[33m⏸\x1b[0m", conv.name)
                } else {
                    conv.name.clone()
                };

                println!(
                    "{:<30} {:>6}   {:<10} {}",
                    name_display,
                    msg_count,
                    updated,
                    agents.join(", ")
                );
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Memory subcommands
// ---------------------------------------------------------------------------

/// Handle memory subcommands.
async fn handle_memory_command(command: MemoryCommands) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        MemoryCommands::List { agent, limit } => {
            let store = open_memory_store(&agent)?;
            let memories = store.list_all(Some(limit))?;

            if memories.is_empty() {
                println!("No memories found for agent '{}'", agent);
                return Ok(());
            }

            println!(
                "\x1b[1m{:<6} {:<12} {:<10} CONTENT\x1b[0m",
                "ID", "CREATED", "SOURCE"
            );
            println!("{}", "─".repeat(110));

            for m in memories {
                let age = format_age(m.created_at);
                let content_display: String = m.content.chars().take(80).collect();
                let content_display = if m.content.chars().count() > 80 {
                    format!("{}...", content_display)
                } else {
                    content_display
                };
                let content_display = content_display.replace('\n', " ");

                println!(
                    "{:<6} {:<12} {:<10} {}",
                    m.id, age, m.source, content_display
                );
            }
        }

        MemoryCommands::Search {
            agent,
            query,
            limit,
        } => {
            let store = open_memory_store(&agent)?;

            let embedding_result = get_query_embedding(&query).await;

            let results = match embedding_result {
                Ok(embedding) =>
                {
                    #[allow(deprecated)]
                    store.recall_with_embedding(&query, limit, Some(&embedding))?
                }
                Err(_) => {
                    eprintln!(
                        "\x1b[33mWarning: Could not generate embedding, falling back to text search\x1b[0m"
                    );
                    let all = store.list_all(None)?;
                    let query_lower = query.to_lowercase();
                    all.into_iter()
                        .filter(|m| m.content.to_lowercase().contains(&query_lower))
                        .take(limit)
                        .map(|m| (m, 1.0))
                        .collect()
                }
            };

            if results.is_empty() {
                println!(
                    "No memories matching '{}' found for agent '{}'",
                    query, agent
                );
                return Ok(());
            }

            println!(
                "\x1b[1m{:<6} {:<8} {:<12} {:<10} CONTENT\x1b[0m",
                "ID", "SCORE", "CREATED", "SOURCE"
            );
            println!("{}", "─".repeat(80));

            for (m, score) in results {
                let age = format_age(m.created_at);
                let content_display: String = m.content.chars().take(35).collect();
                let content_display = if m.content.chars().count() > 35 {
                    format!("{}...", content_display)
                } else {
                    content_display
                };
                let content_display = content_display.replace('\n', " ");

                println!(
                    "{:<6} {:<8.3} {:<12} {:<10} {}",
                    m.id, score, age, m.source, content_display
                );
            }
        }

        MemoryCommands::Delete { agent, id } => {
            let store = open_memory_store(&agent)?;
            let deleted = store.delete(id)?;

            if deleted {
                println!("Deleted memory #{} for agent '{}'", id, agent);
            } else {
                println!("Memory #{} not found for agent '{}'", id, agent);
            }
        }

        MemoryCommands::Clear { agent, force } => {
            let store = open_memory_store(&agent)?;
            let count = store.count()?;

            if count == 0 {
                println!("No memories to clear for agent '{}'", agent);
                return Ok(());
            }

            if !force {
                print!(
                    "This will delete {} memories for '{}'. Continue? [y/N] ",
                    count, agent
                );
                io::stdout().flush()?;

                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                if !input.trim().eq_ignore_ascii_case("y") {
                    println!("Aborted.");
                    return Ok(());
                }
            }

            let deleted = store.clear_all()?;
            println!("Cleared {} memories for agent '{}'", deleted, agent);
        }

        MemoryCommands::Count { agent } => {
            let store = open_memory_store(&agent)?;
            let count = store.count()?;
            println!("Agent '{}' has {} memories", agent, count);
        }

        MemoryCommands::Show { agent, id } => {
            let store = open_memory_store(&agent)?;

            match store.get(id)? {
                Some(entry) => {
                    let age = format_age(entry.created_at);
                    println!("\x1b[1mMemory #{}\x1b[0m", entry.id);
                    println!("─────────────────────────────────────────");
                    println!("\x1b[90mCreated:\x1b[0m    {}", age);
                    println!("\x1b[90mSource:\x1b[0m     {}", entry.source);
                    println!("\x1b[90mImportance:\x1b[0m {:.2}", entry.importance);
                    println!("\x1b[90mContent:\x1b[0m");
                    println!("{}", entry.content);
                }
                None => {
                    return Err(format!("Memory #{} not found for agent '{}'", id, agent).into());
                }
            }
        }

        MemoryCommands::Add {
            agent,
            content,
            importance,
        } => {
            let agent_path = resolve_agent_path(&agent);
            let memory_path = agent_path.join("memory.db");

            // Create memory.db if it doesn't exist (unlike other commands that require it)
            let store = SemanticMemoryStore::open(&memory_path, &agent)?;

            let importance = importance.clamp(0.0, 1.0);

            match store.save(&content, importance, "cli")? {
                anima::memory::SaveResult::New(id) => {
                    println!("Created memory #{} for agent '{}'", id, agent);
                }
                anima::memory::SaveResult::Reinforced(id, old_imp, new_imp) => {
                    println!(
                        "Reinforced existing memory #{} (importance: {:.2} → {:.2})",
                        id, old_imp, new_imp
                    );
                }
            }
        }

        MemoryCommands::Replace { agent, id, content } => {
            let store = open_memory_store(&agent)?;

            if store.update_content(id, &content)? {
                println!("Updated memory #{} for agent '{}'", id, agent);
            } else {
                return Err(format!("Memory #{} not found for agent '{}'", id, agent).into());
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Run / Create / List / Task
// ---------------------------------------------------------------------------

/// Scaffold a new agent directory (delegates to shared function in agent_dir).
fn create_agent(name: &str, path: Option<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    anima::agent_dir::create_agent(name, path)?;
    Ok(())
}

/// List all agents in ~/.anima/agents/
fn list_agents() {
    let Some(entries) = list_agent_dirs() else {
        return;
    };

    println!("\x1b[1mAgents in ~/.anima/agents/:\x1b[0m");
    for entry in entries {
        let name = entry.file_name().to_string_lossy().to_string();

        let info = match AgentDir::load(entry.path()) {
            Ok(agent_dir) => match agent_dir.resolve_llm_config() {
                Ok(resolved) => format!(" ({}/{})", resolved.provider, resolved.model),
                Err(_) => " (config error)".to_string(),
            },
            Err(_) => " (config error)".to_string(),
        };

        println!("  \x1b[36m{}\x1b[0m{}", name, info);
    }
}

/// Run an agent with a single task (non-interactive, from config file).
/// Uses the unified run_tool_loop with an ephemeral DB conversation.
async fn run_agent_task(
    config_path: &str,
    task: &str,
    stream: bool,
    verbose_cli: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use anima::daemon::{AgentLogger, ToolExecutionContext, run_tool_loop, spawn_log_forwarder};
    use tokio::sync::Mutex;

    let config = AgentConfig::from_file(config_path)?;

    let verbose = verbose_cli || config.observe.verbose;
    let observer = Arc::new(ConsoleObserver::new(verbose));

    let llm: Arc<dyn LLM> = match config.llm.provider.as_str() {
        "openai" => {
            let api_key = std::env::var("OPENAI_API_KEY")
                .map_err(|_| "OPENAI_API_KEY environment variable not set")?;
            let mut client = OpenAIClient::new(api_key).with_model(&config.llm.model);
            if let Some(base_url) = &config.llm.base_url {
                client = client.with_base_url(base_url);
            }
            Arc::new(client)
        }
        "anthropic" => {
            let mut client = if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
                AnthropicClient::new(api_key)
            } else if let Some(token) = anima::auth::get_valid_token().await.map_err(|e| e.to_string())? {
                AnthropicClient::with_bearer(token)
            } else {
                return Err("No Anthropic API key or subscription login found. Run `anima login` or set ANTHROPIC_API_KEY.".into());
            };
            client = client.with_model(&config.llm.model);
            if let Some(base_url) = &config.llm.base_url {
                client = client.with_base_url(base_url);
            }
            if let Some(mt) = config.llm.max_tokens {
                client = client.with_max_tokens(mt);
            }
            Arc::new(client)
        }
        "claude-code" => Arc::new(anima::ClaudeCodeClient::new(&config.llm.model)),
        other => return Err(format!("Unknown LLM provider: {}", other).into()),
    };

    let memory: Box<dyn anima::Memory> = match config.memory.backend.as_str() {
        "sqlite" => {
            let path = config.memory.path.as_deref().unwrap_or("anima.db");
            Box::new(SqliteMemory::open(path, &config.agent.name)?)
        }
        "in_memory" => Box::new(InMemoryStore::new()),
        _ => Box::new(InMemoryStore::new()),
    };

    let mut runtime = Runtime::new();
    let mut agent = runtime.spawn_agent(config.agent.name.clone()).await;

    for tool_name in &config.tools.enabled {
        match tool_name.as_str() {
            "add" => agent.register_tool(Arc::new(AddTool)),
            "echo" => agent.register_tool(Arc::new(EchoTool)),
            "read_file" => agent.register_tool(Arc::new(ReadFileTool)),
            "peek_file" => agent.register_tool(Arc::new(PeekFileTool)),
            "write_file" => agent.register_tool(Arc::new(WriteFileTool)),
            "http" => agent.register_tool(Arc::new(HttpTool::new())),
            "shell" => agent.register_tool(Arc::new(ShellTool::new())),
            "copy_lines" => agent.register_tool(Arc::new(CopyLinesTool)),
            "edit_file" => agent.register_tool(Arc::new(EditFileTool)),
            "list_files" => agent.register_tool(Arc::new(ListFilesTool)),
            "safe_shell" => agent.register_tool(Arc::new(SafeShellTool::new())),
            unknown => eprintln!("Warning: Unknown tool '{}', skipping", unknown),
        }
    }

    agent = agent.with_llm(llm);
    agent = agent.with_memory(memory);
    agent = agent.with_observer(observer);

    let agent_name = config.agent.name.clone();

    // Wrap agent in Arc<Mutex<>> for run_tool_loop
    let agent = Arc::new(Mutex::new(agent));

    // Create ephemeral conversation in the shared DB
    let store = ConversationStore::init()?;
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let conv_name = format!("task-{}-{}", agent_name, timestamp);

    store.create_conversation(Some(&conv_name), &["user", &agent_name])?;
    store.add_message(&conv_name, "user", task, &[])?;

    // Create logger to a temp file (task mode has no agent directory)
    let log_dir = std::env::temp_dir();
    let log_path = log_dir.join(format!("anima-task-{}.log", timestamp));
    let logger = Arc::new(
        AgentLogger::from_path(&log_path, &agent_name)
            .unwrap_or_else(|_| AgentLogger::from_path(&std::env::temp_dir().join("anima-task.log"), &agent_name).expect("Failed to create logger"))
    );

    let shutdown = Arc::new(tokio::sync::Notify::new());
    let system_prompt = config.agent.system_prompt;
    let allowed_tools: Option<Vec<String>> = Some(config.tools.enabled.clone());

    // Build tool execution context
    let tool_context = ToolExecutionContext {
        agent_name: agent_name.clone(),
        task_store: None,
        conv_id: Some(conv_name.clone()),
        semantic_memory_store: None,
        embedding_client: None,
        allowed_tools: allowed_tools.clone(),
        logger: Some(logger.clone()),
        agent_dir: None,
        agent: Some(Arc::clone(&agent)),
        system_prompt: system_prompt.clone(),
        tool_registry: None,
        use_native_tools: true,
        num_ctx: None,
        shutdown: Some(Arc::clone(&shutdown)),
        dedup_lazy: false,
        subtask_depth: 0,
    };

    let (log_tx, log_fwd_handle) = spawn_log_forwarder(logger.clone());
    let start_time = std::time::Instant::now();
    let max_steps = config.think.max_steps;

    // Streaming: create token channel if streaming is requested
    let token_tx = if stream {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(100);
        tokio::spawn(async move {
            while let Some(token) = rx.recv().await {
                print!("{}", token);
                let _ = io::stdout().flush();
            }
        });
        Some(tx)
    } else {
        None
    };

    // Use <conversation> wrapping for the task content
    let user_content = format!("<conversation>\n{}\n</conversation>", task);

    let loop_result = run_tool_loop(
        &user_content,
        task,
        vec![], // no prior conversation history
        &agent,
        &agent_name,
        &system_prompt,
        None, // external_tools: None — think_single_step will use agent's registered tools
        true, // use_native_tools
        max_steps,
        None, // num_ctx
        &None, // no tool_registry
        &tool_context,
        &None, // no semantic_memory
        &None, // no embedding_client
        &conv_name,
        token_tx,
        None, // no cancellation
        None, // no response deadline
        start_time,
        &shutdown,
        &logger,
        log_tx.clone(),
        false, // dedup_lazy
        None, // no state_dir
        None, // no initial_state
        None, // no verbose_tx
    ).await;

    drop(log_tx);
    let _ = log_fwd_handle.await;

    // Clean up ephemeral conversation
    let _ = store.delete_conversation(&conv_name);

    // Print final response (if not already streamed)
    if !stream {
        println!("{}", loop_result.response);
    } else {
        println!(); // newline after streamed output
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_agents_dir() {
        let path = agents_dir();
        assert!(path.ends_with(".anima/agents"));
    }

    #[test]
    fn test_resolve_agent_path_name() {
        // A simple name should resolve to ~/.anima/agents/<name>
        let path = resolve_agent_path("myagent");
        assert!(path.ends_with(".anima/agents/myagent"));
    }

    #[test]
    fn test_resolve_agent_path_absolute() {
        // An absolute path should be used directly
        let path = resolve_agent_path("/some/absolute/path");
        assert_eq!(path, PathBuf::from("/some/absolute/path"));
    }

    #[test]
    fn test_resolve_agent_path_relative() {
        // A relative path with / should be used directly
        let path = resolve_agent_path("./myagent");
        assert_eq!(path, PathBuf::from("./myagent"));

        let path = resolve_agent_path("some/nested/path");
        assert_eq!(path, PathBuf::from("some/nested/path"));
    }

    #[test]
    fn test_create_agent_success() {
        let dir = tempdir().unwrap();
        let agent_path = dir.path().join("test-agent");

        let result = create_agent("test-agent", Some(agent_path.clone()));
        assert!(result.is_ok());

        // Check files were created
        assert!(agent_path.join("config.toml").exists());
        assert!(agent_path.join("system.md").exists());
        assert!(agent_path.join("recall.md").exists());

        // Check config.toml content
        let config_content = std::fs::read_to_string(agent_path.join("config.toml")).unwrap();
        assert!(config_content.contains("name = \"test-agent\""));
        assert!(config_content.contains("[llm]"));
        assert!(config_content.contains("[memory]"));
        assert!(config_content.contains("recall_file = \"recall.md\""));

        // Check system.md content
        let system_content = std::fs::read_to_string(agent_path.join("system.md")).unwrap();
        assert!(system_content.contains("# test-agent"));
        assert!(system_content.contains("You are test-agent"));

        // Check recall.md content
        let recall_content = std::fs::read_to_string(agent_path.join("recall.md")).unwrap();
        assert!(recall_content.contains("# Recall"));
        assert!(recall_content.contains("How Conversations Work"));
        assert!(recall_content.contains("Never @mention yourself"));
    }

    #[test]
    fn test_create_agent_already_exists() {
        let dir = tempdir().unwrap();
        let agent_path = dir.path().join("existing-agent");

        // Create the directory first
        std::fs::create_dir_all(&agent_path).unwrap();

        let result = create_agent("existing-agent", Some(agent_path));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }

    #[test]
    fn test_resolve_agent_socket() {
        let socket_path = resolve_agent_socket("myagent");
        assert!(socket_path.ends_with(".anima/agents/myagent/agent.sock"));
    }

    #[test]
    fn test_resolve_agent_socket_with_path() {
        let socket_path = resolve_agent_socket("/custom/path/myagent");
        assert_eq!(
            socket_path,
            PathBuf::from("/custom/path/myagent/agent.sock")
        );
    }

    #[test]
    fn test_status_with_running_agent() {
        // Create a temp agent directory with a PID file for current process
        let dir = tempdir().unwrap();
        let agent_path = dir.path().join("test-agent");
        std::fs::create_dir_all(&agent_path).unwrap();

        // Write a minimal config.toml
        let config_content = r#"
[agent]
name = "test-agent"

[llm]
provider = "openai"
model = "gpt-4"
api_key = "sk-test"
"#;
        std::fs::write(agent_path.join("config.toml"), config_content).unwrap();

        // Write a PID file with current process ID (so is_running returns true)
        let pid = std::process::id();
        std::fs::write(agent_path.join("daemon.pid"), pid.to_string()).unwrap();

        // Verify PidFile::is_running works
        assert!(PidFile::is_running(agent_path.join("daemon.pid")));
    }

    #[test]
    fn test_status_with_stopped_agent() {
        // Create a temp agent directory without a PID file
        let dir = tempdir().unwrap();
        let agent_path = dir.path().join("stopped-agent");
        std::fs::create_dir_all(&agent_path).unwrap();

        // Write a minimal config.toml
        let config_content = r#"
[agent]
name = "stopped-agent"

[llm]
provider = "openai"
model = "gpt-4"
api_key = "sk-test"
"#;
        std::fs::write(agent_path.join("config.toml"), config_content).unwrap();

        // No PID file means agent is stopped
        assert!(!PidFile::is_running(agent_path.join("daemon.pid")));
    }

    #[test]
    fn test_status_with_stale_pid_file() {
        // Create a temp agent directory with a PID file for a non-existent process
        let dir = tempdir().unwrap();
        let agent_path = dir.path().join("stale-agent");
        std::fs::create_dir_all(&agent_path).unwrap();

        // Write a minimal config.toml
        let config_content = r#"
[agent]
name = "stale-agent"

[llm]
provider = "openai"
model = "gpt-4"
api_key = "sk-test"
"#;
        std::fs::write(agent_path.join("config.toml"), config_content).unwrap();

        // Write a PID file with a very high PID that likely doesn't exist
        std::fs::write(agent_path.join("daemon.pid"), "999999999").unwrap();

        // Should return false for non-existent process
        assert!(!PidFile::is_running(agent_path.join("daemon.pid")));
    }
}
