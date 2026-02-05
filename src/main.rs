use anima::{
    Runtime, OpenAIClient, AnthropicClient, ThinkOptions, AutoMemoryConfig, ReflectionConfig,
    InMemoryStore, SqliteMemory, LLM, ConversationStore,
    parse_mentions, expand_all_mention, notify_mentioned_agents_parallel, NotifyResult,
    SemanticMemoryStore, format_age,
};
use anima::agent_dir::AgentDir;
use anima::config::AgentConfig;
use anima::daemon::PidFile;
use anima::observe::ConsoleObserver;
use anima::repl::Repl;
use anima::socket_api::{SocketApi, Request, Response};
use anima::tools::{AddTool, EchoTool, ReadFileTool, WriteFileTool, HttpTool, ShellTool};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::io::{self, Write};
use std::time::Duration;
use tokio::net::UnixStream;

/// Format tool params for display in a brief, readable way.
fn format_tool_params(params: &serde_json::Value) -> String {
    // For shell commands, show the command
    if let Some(cmd) = params.get("command").and_then(|c| c.as_str()) {
        return cmd.to_string();
    }
    // For file operations, show the path
    if let Some(path) = params.get("path").and_then(|p| p.as_str()) {
        return path.to_string();
    }
    // For HTTP requests, show method and URL
    if let Some(url) = params.get("url").and_then(|u| u.as_str()) {
        let method = params.get("method").and_then(|m| m.as_str()).unwrap_or("GET");
        return format!("{} {}", method, url);
    }
    // Fallback: compact JSON, truncated
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
    
    /// Enable debug logging to ~/.anima/repl.log (for REPL/run commands)
    #[arg(long, global = true)]
    log: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Run an agent from a directory or by name, starting an interactive REPL
    Run {
        /// Agent name (from ~/.anima/agents/) or path to agent directory
        agent: String,
        /// Run as a daemon (headless, no REPL)
        #[arg(long)]
        daemon: bool,
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
    /// Interactive REPL for exploring anima (default if no command given)
    Repl,
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
    },
    /// Start an agent daemon in the background. Supports glob patterns (*, ?).
    Start {
        /// Agent name or glob pattern (from ~/.anima/agents/)
        agent: String,
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
    /// Manage agent memories
    Memory {
        #[command(subcommand)]
        command: MemoryCommands,
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
        /// Output raw pipe-delimited format for scripts (ID|TIMESTAMP|FROM|CONTENT)
        #[arg(long)]
        raw: bool,
    },
    /// Pause a conversation (notifications will be queued). Supports glob patterns (*, ?).
    Pause {
        /// Name or pattern of the conversation(s) to pause
        conv: String,
        /// Skip confirmation prompt for multiple matches
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
    /// Run cleanup to delete expired messages and empty conversations
    Cleanup,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let command = cli.command.unwrap_or(Commands::Repl);

    match command {
        Commands::Run { agent, daemon } => {
            if daemon {
                if let Err(e) = anima::daemon::run_daemon(&agent).await {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            } else {
                if let Err(e) = run_agent_dir(&agent).await {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
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
        Commands::Task { config, task, stream, verbose } => {
            if let Err(e) = run_agent_task(&config, &task, stream, verbose).await {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Repl => {
            let mut repl = Repl::new().with_logging(cli.log);
            if let Err(e) = repl.run().await {
                eprintln!("REPL error: {}", e);
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
            show_status();
        }
        Commands::Ask { agent, message } => {
            if let Err(e) = ask_agent(&agent, &message).await {
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
        Commands::Memory { command } => {
            if let Err(e) = handle_memory_command(command).await {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
    }
}

/// Get the agents directory path (~/.anima/agents/)
fn agents_dir() -> PathBuf {
    dirs::home_dir()
        .expect("Could not determine home directory")
        .join(".anima")
        .join("agents")
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
        ).into());
    }

    let stream = UnixStream::connect(&socket_path).await
        .map_err(|e| format!("Failed to connect to agent '{}': {}", agent, e))?;

    Ok(SocketApi::new(stream))
}


/// Default number of context messages to load from conversation history.
const DEFAULT_CONTEXT_MESSAGES: usize = 20;

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
async fn chat_with_conversation(
    conv_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize conversation store
    let store = ConversationStore::init()?;

    // Load current participants
    let parts = store.get_participants(conv_name)?;
    let mut participants: Vec<String> = parts.iter()
        .filter(|p| p.agent != "user")
        .map(|p| p.agent.clone())
        .collect();

    // Track the last message ID we've displayed
    let last_seen_id = Arc::new(AtomicI64::new(0));

    // Get current last message ID so we don't display old messages
    if let Ok(msgs) = store.get_messages(conv_name, Some(1)) {
        if let Some(last) = msgs.last() {
            last_seen_id.store(last.id, Ordering::SeqCst);
        }
    }

    // Print connection info
    println!("\x1b[32m✓ Joined conversation '{}'\x1b[0m", conv_name);
    if participants.is_empty() {
        println!("\x1b[90mNo agents yet - use @mentions to invite them\x1b[0m");
    } else {
        println!("\x1b[90mParticipants: {}\x1b[0m", participants.join(", "));
    }

    // Display recent message history for context
    if let Ok(recent_msgs) = store.get_messages(conv_name, Some(25)) {
        if !recent_msgs.is_empty() {
            println!("\x1b[90m--- Recent messages ---\x1b[0m");
            for msg in &recent_msgs {
                // Skip tool messages
                if msg.from_agent == "tool" {
                    continue;
                }
                let color = if msg.from_agent == "user" { "33" } else { "36" }; // yellow for user, cyan for agents
                let datetime = format_timestamp_pretty(msg.created_at);
                // Show duration and context usage for agent messages if available
                let duration_str = if msg.from_agent != "user" {
                    msg.duration_ms.map(|d| format!(" \x1b[90m•\x1b[0m \x1b[33m{}\x1b[0m", format_duration_ms(d))).unwrap_or_default()
                } else {
                    String::new()
                };
                let ctx_str = if msg.from_agent != "user" {
                    format_context_usage(msg)
                } else {
                    String::new()
                };
                println!(
                    "\x1b[90m[{}] {}\x1b[0m \x1b[90m•\x1b[0m \x1b[{}m{}\x1b[0m{}{}",
                    msg.id, datetime, color, msg.from_agent, duration_str, ctx_str
                );
                println!("{}", msg.content);
                println!(); // Blank line between messages
            }

            // Update last_seen_id to the most recent message
            if let Some(last) = recent_msgs.last() {
                last_seen_id.store(last.id, Ordering::SeqCst);
            }
        }
    }

    println!("Type your messages. Press Ctrl+D or Ctrl+C to exit.\n");

    // Spawn background message polling task
    let poll_conv_name = conv_name.to_string();
    let poll_last_seen = last_seen_id.clone();
    let shutdown = Arc::new(AtomicBool::new(false));
    let poll_shutdown = shutdown.clone();

    let poll_handle = tokio::spawn(async move {
        // Create a separate store connection for the polling task
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
            if let Ok(msgs) = poll_store.get_messages_filtered(&poll_conv_name, None, Some(last_id)) {
                for msg in msgs {
                    // Only display messages from agents (not from "user" or "tool")
                    if msg.from_agent != "user" && msg.from_agent != "tool" {
                        // Print the agent message on a new line
                        print!("\r\x1b[K"); // Clear current line (in case user was typing)
                        let datetime = format_timestamp_pretty(msg.created_at);
                        // Show duration and context usage if available
                        let duration_str = msg.duration_ms.map(|d| format!(" \x1b[90m•\x1b[0m \x1b[33m{}\x1b[0m", format_duration_ms(d))).unwrap_or_default();
                        let ctx_str = format_context_usage(&msg);
                        println!(
                            "\x1b[90m[{}] {}\x1b[0m \x1b[90m•\x1b[0m \x1b[36m{}\x1b[0m{}{}",
                            msg.id, datetime, msg.from_agent, duration_str, ctx_str
                        );
                        println!("{}", msg.content);
                        println!(); // Blank line between messages
                        print!("\x1b[36m#{}\x1b[90m›\x1b[0m ", poll_conv_name); // Re-print prompt
                        let _ = io::stdout().flush();
                    }
                    // Update last seen ID regardless of sender
                    poll_last_seen.store(msg.id, Ordering::SeqCst);
                }
            }
        }
    });

    // Use rustyline for interactive input
    let mut rl = rustyline::DefaultEditor::new()?;
    let prompt = format!("\x1b[36m#{}\x1b[90m›\x1b[0m ", conv_name);

    loop {
        match rl.readline(&prompt) {
            Ok(line) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                // Add to history
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
                                Ok(_) => println!("\x1b[90mConversation paused. Notifications will be queued.\x1b[0m"),
                                Err(e) => eprintln!("\x1b[31mFailed to pause: {}\x1b[0m", e),
                            }
                            continue;
                        }
                        "/resume" => {
                            match store.set_paused(conv_name, false) {
                                Ok(paused_at_msg_id) => {
                                    let mut catchup_count = 0;

                                    // Process catchup items (tool calls and mentions skipped during pause)
                                    if let Some(paused_at) = paused_at_msg_id {
                                        if let Ok(catchup_msgs) = store.get_catchup_messages(conv_name, paused_at) {
                                            let catchup_items = ConversationStore::build_catchup_items(&catchup_msgs);
                                            for item in &catchup_items {
                                                // Process @mentions
                                                if !item.mentions.is_empty() {
                                                    let _ = notify_mentioned_agents_parallel(
                                                        &store,
                                                        conv_name,
                                                        item.message.id,
                                                        &item.mentions
                                                    ).await;
                                                    catchup_count += item.mentions.len();
                                                }
                                                // Note: Tool calls in catchup items will be handled
                                                // by the agent when it processes the follow-up notification
                                            }
                                        }
                                    }

                                    // Also check for pending notifications (for agents that weren't running)
                                    match store.get_pending_notifications_for_conversation(conv_name) {
                                        Ok(pending) => {
                                            let pending_count = pending.len();
                                            if pending_count > 0 {
                                                for notification in &pending {
                                                    let mentions = vec![notification.agent.clone()];
                                                    let _ = notify_mentioned_agents_parallel(
                                                        &store,
                                                        conv_name,
                                                        notification.message_id,
                                                        &mentions
                                                    ).await;
                                                    let _ = store.delete_pending_notification(notification.id);
                                                }
                                                catchup_count += pending_count;
                                            }
                                        }
                                        Err(_) => {}
                                    }

                                    if catchup_count > 0 {
                                        println!("\x1b[90mConversation resumed. {} pending item(s) processed.\x1b[0m", catchup_count);
                                    } else {
                                        println!("\x1b[90mConversation resumed.\x1b[0m");
                                    }
                                }
                                Err(e) => eprintln!("\x1b[31mFailed to resume: {}\x1b[0m", e),
                            }
                            continue;
                        }
                        "/help" => {
                            println!("\x1b[90mCommands: /clear, /pause, /resume, /quit, /exit, /help\x1b[0m");
                            continue;
                        }
                        _ => {
                            eprintln!("\x1b[33mUnknown command. Type /help for available commands.\x1b[0m");
                            continue;
                        }
                    }
                }

                // Parse @mentions from user message
                let mentions = parse_mentions(line);

                // Require @mentions to send messages
                if mentions.is_empty() {
                    eprintln!("\x1b[33m⚠ Use @mentions to notify agents (e.g., @arya what do you think?)\x1b[0m");
                    continue;
                }

                // @mention-as-invite: Add new mentions as participants
                for agent_name in &mentions {
                    if agent_name != "all" && !participants.contains(agent_name) {
                        // Only add if agent actually exists
                        if anima::discovery::agent_exists(agent_name) {
                            if let Err(e) = store.add_participant(conv_name, agent_name) {
                                eprintln!("\x1b[33mWarning: Could not add {} as participant: {}\x1b[0m", agent_name, e);
                            } else {
                                participants.push(agent_name.clone());
                                println!("\x1b[90m[@{} joined the conversation]\x1b[0m", agent_name);
                            }
                        }
                    }
                }

                // Expand @all to all participants (excluding "user")
                let expanded_mentions = expand_all_mention(&mentions, &participants);

                let mention_refs: Vec<&str> = expanded_mentions.iter().map(|s| s.as_str()).collect();

                // Store user message in conversation with mentions
                let user_msg_id = store.add_message(conv_name, "user", line, &mention_refs)?;

                // Update last_seen_id to the user's message to avoid re-displaying it
                last_seen_id.store(user_msg_id, Ordering::SeqCst);

                // Blank line before agent response
                println!();

                // Notify mentioned agents (initial notifications only)
                // The daemon now handles all follow-up @mention chains autonomously
                if !expanded_mentions.is_empty() {
                    let notify_results = notify_mentioned_agents_parallel(&store, conv_name, user_msg_id, &expanded_mentions).await;

                    // Display results for each initially notified agent
                    for (agent, result) in &notify_results {
                        match result {
                            NotifyResult::Acknowledged => {
                                // Fire-and-forget: agent received notification, processing async
                                // The background poller will display the response when it arrives
                                // (no message needed - "joined the conversation" already shown)
                            }
                            NotifyResult::Notified { response_message_id } => {
                                // Synchronous response (backwards compat)
                                // Update last_seen_id so poller doesn't re-display
                                last_seen_id.store(*response_message_id, Ordering::SeqCst);
                                if let Ok(msgs) = store.get_messages(conv_name, Some(DEFAULT_CONTEXT_MESSAGES)) {
                                    if let Some(response_msg) = msgs.iter().find(|m| m.id == *response_message_id) {
                                        println!("\n\x1b[36m[{}]:\x1b[0m {}\n", agent, response_msg.content);
                                    }
                                }
                            }
                            NotifyResult::Queued { notification_id: _ } => {
                                eprintln!("\x1b[90m[@{} offline, notification queued]\x1b[0m", agent);
                            }
                            NotifyResult::UnknownAgent => {
                                eprintln!("\x1b[33m[@{} unknown agent - no such agent exists]\x1b[0m", agent);
                            }
                            NotifyResult::Failed { reason } => {
                                eprintln!("\x1b[33m[@{} notification failed: {}]\x1b[0m", agent, reason);
                            }
                        }
                    }

                    // Note: Any @mentions in agent responses are now forwarded autonomously
                    // by the daemon. The CLI no longer tracks or displays follow-up chains.
                    // The background poller will display new messages as they arrive.
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

    // Signal shutdown and wait for the polling task to finish
    shutdown.store(true, Ordering::SeqCst);
    let _ = poll_handle.await;

    Ok(())
}

/// Show status of all agents (running/stopped).
fn show_status() {
    let agents_path = agents_dir();

    if !agents_path.exists() {
        println!("\x1b[33mNo agents directory found at {}\x1b[0m", agents_path.display());
        println!("Create an agent with: \x1b[36manima create <name>\x1b[0m");
        return;
    }

    let entries: Vec<_> = match std::fs::read_dir(&agents_path) {
        Ok(dir) => dir
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .filter(|e| e.path().join("config.toml").exists())
            .collect(),
        Err(e) => {
            eprintln!("\x1b[31mCould not read agents directory: {}\x1b[0m", e);
            return;
        }
    };

    if entries.is_empty() {
        println!("\x1b[33mNo agents found in {}\x1b[0m", agents_path.display());
        println!("Create an agent with: \x1b[36manima create <name>\x1b[0m");
        return;
    }

    // Print header
    println!("\x1b[1m{:<20} {:<10} {:<10}\x1b[0m", "AGENT", "STATUS", "PID");
    println!("{}", "-".repeat(42));

    for entry in entries {
        let name = entry.file_name().to_string_lossy().to_string();
        let agent_path = entry.path();
        let pid_path = agent_path.join("daemon.pid");

        let (status, pid) = if PidFile::is_running(&pid_path) {
            let pid = PidFile::read(&pid_path).unwrap_or(0);
            ("\x1b[32mrunning\x1b[0m", pid.to_string())
        } else {
            ("\x1b[90mstopped\x1b[0m", "-".to_string())
        };

        println!("{:<20} {:<19} {:<10}", name, status, pid);
    }
}

/// Query an agent with conversation persistence.
/// Starts daemon if not running, creates/uses conversation named after agent,
/// stores user message, streams response, and daemon stores agent response.
async fn ask_agent(agent: &str, message: &str) -> Result<(), Box<dyn std::error::Error>> {
    use anima::discovery;
    use std::process::Command;

    let agent_path = resolve_agent_path(agent);

    // Verify agent exists
    if !agent_path.exists() {
        return Err(format!("Agent '{}' not found at {}", agent, agent_path.display()).into());
    }

    // Load agent directory to get canonical name
    let agent_dir = AgentDir::load(&agent_path)
        .map_err(|e| format!("Failed to load agent '{}': {}", agent, e))?;
    let agent_name = agent_dir.config.agent.name.clone();

    // Start daemon if not running
    if !discovery::is_agent_running(&agent_name) {
        eprintln!("\x1b[33mStarting daemon for '{}'...\x1b[0m", agent_name);

        let exe = std::env::current_exe()
            .map_err(|e| format!("Failed to get executable path: {}", e))?;

        Command::new(&exe)
            .args(["start", &agent_name])
            .spawn()
            .map_err(|e| format!("Failed to start daemon: {}", e))?;

        // Wait for daemon to start (up to 2 seconds)
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

    // Initialize conversation store and create/get conversation named after agent
    let store = ConversationStore::init()?;
    let conv_name = agent_name.clone();

    // Create conversation if it doesn't exist
    if store.find_by_name(&conv_name)?.is_none() {
        store.create_conversation(Some(&conv_name), &["user", &agent_name])?;
    }

    // Store user message in conversation
    store.add_message(&conv_name, "user", message, &[])?;

    // Connect to daemon
    let mut api = connect_to_agent(&agent_name).await?;

    // Send message with conversation name for persistence
    api.write_request(&Request::Message {
        content: message.to_string(),
        conv_name: Some(conv_name),
    }).await.map_err(|e| format!("Failed to send message: {}", e))?;

    // Read responses (streaming)
    loop {
        match api.read_response().await.map_err(|e| format!("Failed to read response: {}", e))? {
            Some(Response::Chunk { text }) => {
                // Streaming: print tokens as they arrive
                print!("{}", text);
                io::stdout().flush()?;
            }
            Some(Response::ToolCall { tool, params }) => {
                // Show tool call
                let param_summary = format_tool_params(&params);
                eprintln!(" - [tool] {}: {}", tool, param_summary);
            }
            Some(Response::Done) => {
                // Streaming complete
                println!();  // Final newline
                break;
            }
            Some(Response::Message { content }) => {
                // Fallback for non-streaming responses
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
            _ => {
                // Ignore other response types
            }
        }
    }

    Ok(())
}

/// Start an agent daemon in the background.
/// If `quiet` is true, suppresses output (used by restart_all_agents).
fn start_agent_impl(agent: &str, quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    let agent_path = resolve_agent_path(agent);
    let pid_path = agent_path.join("daemon.pid");

    // Check if agent directory exists
    if !agent_path.exists() {
        return Err(format!("Agent '{}' not found at {}", agent, agent_path.display()).into());
    }

    // Check if already running
    if PidFile::is_running(&pid_path) {
        let pid = PidFile::read(&pid_path).unwrap_or(0);
        return Err(format!("Agent '{}' is already running (pid {})", agent, pid).into());
    }

    // Get path to current executable
    let exe = std::env::current_exe()
        .map_err(|e| format!("Failed to get current executable: {}", e))?;

    // Open log file for stdout/stderr redirection
    let log_path = agent_path.join("agent.log");
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| format!("Failed to open log file: {}", e))?;
    let log_file_err = log_file.try_clone()
        .map_err(|e| format!("Failed to clone log file handle: {}", e))?;

    // Spawn daemon process in background
    // Use `run --daemon` for the actual daemon logic
    let child = Command::new(&exe)
        .arg("run")
        .arg(agent)
        .arg("--daemon")
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

    // Handle "all" case - start all stopped agents
    if agent == "all" {
        return start_all_agents();
    }

    // Check for glob pattern
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
    use anima::discovery::{list_saved_agents, is_agent_running};

    let all_agents = list_saved_agents();

    if all_agents.is_empty() {
        println!("No configured agents found");
        return Ok(());
    }

    // Filter to only stopped agents
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
/// If `quiet` is true, suppresses output (used by restart_all_agents).
async fn stop_agent_impl(agent: &str, quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    let agent_path = resolve_agent_path(agent);
    let pid_path = agent_path.join("daemon.pid");

    // Check if running
    if !PidFile::is_running(&pid_path) {
        if !quiet {
            println!("Agent '{}' is not running", agent);
        }
        return Ok(());
    }

    let pid = PidFile::read(&pid_path).unwrap_or(0);

    // Try to send shutdown via socket first (graceful shutdown)
    match connect_to_agent(agent).await {
        Ok(mut api) => {
            // Send shutdown request
            if let Err(e) = api.write_request(&Request::Shutdown).await {
                if !quiet {
                    eprintln!("Warning: Failed to send shutdown request: {}", e);
                }
            } else {
                // Wait briefly for graceful shutdown
                for _ in 0..10 {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    if !PidFile::is_running(&pid_path) {
                        if !quiet {
                            println!("Stopped {}", agent);
                        }
                        return Ok(());
                    }
                }
            }
        }
        Err(e) => {
            if !quiet {
                eprintln!("Warning: Could not connect to agent socket: {}", e);
            }
        }
    }

    // If socket shutdown didn't work, send SIGTERM
    unsafe {
        if libc::kill(pid as i32, libc::SIGTERM) == 0 {
            // Wait for process to terminate
            for _ in 0..20 {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
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
        } else {
            if !quiet {
                eprintln!("Warning: Failed to send SIGTERM to pid {}", pid);
            }
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

    // Handle "all" case - stop all running agents
    if agent == "all" {
        return stop_all_agents().await;
    }

    // Check for glob pattern
    if has_wildcards(agent) {
        let matches = match_agents(agent);

        if matches.is_empty() {
            return Err(format!("No agents match pattern: {}", agent).into());
        }

        let mut stopped = Vec::new();
        for name in &matches {
            match stop_agent_impl(name, true).await {
                Ok(()) => stopped.push(name.clone()),
                Err(e) => eprintln!("Failed to stop {}: {}", name, e),
            }
        }

        if !stopped.is_empty() {
            println!("Stopped {} agent(s): {}", stopped.len(), stopped.join(", "));
        }

        return Ok(());
    }

    stop_agent_impl(agent, false).await
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

        match stop_agent_impl(&agent.name, true).await {
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

/// Restart a running agent daemon (stop then start).
/// Supports glob patterns (*, ?) for matching multiple agents.
/// If agent is "all", restarts all currently running agents.
async fn restart_agent(agent: &str) -> Result<(), Box<dyn std::error::Error>> {
    use anima::discovery::match_agents;

    // Handle "all" case - restart all running agents
    if agent == "all" {
        return restart_all_agents().await;
    }

    // Check for glob pattern
    if has_wildcards(agent) {
        let matches = match_agents(agent);

        if matches.is_empty() {
            return Err(format!("No agents match pattern: {}", agent).into());
        }

        // Filter to only running agents (restart shouldn't start stopped agents)
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
            print!("Restarting {}... ", name);
            io::stdout().flush()?;

            // Stop the agent (quiet mode)
            if let Err(e) = stop_agent_impl(name, true).await {
                println!("failed to stop: {}", e);
                continue;
            }

            // Brief wait to ensure clean shutdown
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;

            // Start the agent (quiet mode)
            match start_agent_impl(name, true) {
                Ok(()) => {
                    // Wait for daemon.pid to be written by the daemon process
                    let pid_path = resolve_agent_path(name).join("daemon.pid");
                    let mut pid = 0u32;
                    for _ in 0..20 {
                        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                        if let Ok(p) = PidFile::read(&pid_path) {
                            if p > 0 {
                                pid = p;
                                break;
                            }
                        }
                    }
                    println!("done (pid {})", pid);
                    restarted.push(name.to_string());
                }
                Err(e) => {
                    println!("failed to start: {}", e);
                }
            }
        }

        println!("Restarted {} agent{}", restarted.len(), if restarted.len() == 1 { "" } else { "s" });

        return Ok(());
    }

    // Single agent (no wildcards)
    let agent_path = resolve_agent_path(agent);
    let pid_path = agent_path.join("daemon.pid");

    // Check if agent directory exists
    if !agent_path.exists() {
        return Err(format!("Agent '{}' not found at {}", agent, agent_path.display()).into());
    }

    // Check if currently running
    let was_running = PidFile::is_running(&pid_path);

    if was_running {
        println!("Stopping {}...", agent);
        stop_agent(agent).await?;
        // Brief wait to ensure clean shutdown
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    println!("Starting {}...", agent);
    start_agent(agent)?;

    Ok(())
}

/// Restart all currently running agent daemons.
async fn restart_all_agents() -> Result<(), Box<dyn std::error::Error>> {
    use anima::discovery::discover_running_agents;

    // Get all running agents
    let running_agents = discover_running_agents();

    if running_agents.is_empty() {
        println!("No running agents to restart");
        return Ok(());
    }

    let mut restarted_count = 0;

    for agent in &running_agents {
        print!("Restarting {}... ", agent.name);
        io::stdout().flush()?;

        // Stop the agent (quiet mode - no output)
        if let Err(e) = stop_agent_impl(&agent.name, true).await {
            println!("failed to stop: {}", e);
            continue;
        }

        // Brief wait to ensure clean shutdown
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        // Start the agent (quiet mode - no output)
        match start_agent_impl(&agent.name, true) {
            Ok(()) => {
                // Wait for daemon.pid to be written by the daemon process
                let pid_path = resolve_agent_path(&agent.name).join("daemon.pid");
                let mut pid = 0u32;
                for _ in 0..20 {
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                    if let Ok(p) = PidFile::read(&pid_path) {
                        if p > 0 {
                            pid = p;
                            break;
                        }
                    }
                }
                println!("done (pid {})", pid);
                restarted_count += 1;
            }
            Err(e) => {
                println!("failed to start: {}", e);
            }
        }
    }

    println!("Restarted {} agent{}", restarted_count, if restarted_count == 1 { "" } else { "s" });

    Ok(())
}

/// Clear conversation history for a running agent daemon.
async fn clear_agent(agent: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut api = connect_to_agent(agent).await?;

    // Send clear request
    api.write_request(&Request::Clear).await
        .map_err(|e| format!("Failed to send clear request: {}", e))?;

    // Read the response
    match api.read_response().await.map_err(|e| format!("Failed to read response: {}", e))? {
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

    // Send system request
    api.write_request(&Request::System).await
        .map_err(|e| format!("Failed to send system request: {}", e))?;

    // Read the response
    match api.read_response().await.map_err(|e| format!("Failed to read response: {}", e))? {
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

    // Send heartbeat request
    api.write_request(&Request::Heartbeat).await
        .map_err(|e| format!("Failed to send heartbeat request: {}", e))?;

    // Read the response
    match api.read_response().await.map_err(|e| format!("Failed to read response: {}", e))? {
        Some(Response::HeartbeatTriggered) => {
            println!("Heartbeat triggered for {}", agent);
        }
        Some(Response::HeartbeatNotConfigured) => {
            println!("Heartbeat not configured for {} (add [heartbeat] section to config.toml)", agent);
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

/// Handle chat subcommands.
async fn handle_chat_command(command: Option<ChatCommands>) -> Result<(), Box<dyn std::error::Error>> {
    let store = ConversationStore::init()?;

    match command {
        // `anima chat new` or `anima chat new <name>` - create new chat and enter it
        Some(ChatCommands::New { name }) => {
            // Create conversation with "user" as participant (agents join via @mention)
            let conv_name = store.create_conversation(name.as_deref(), &["user"])?;
            println!("Created conversation '\x1b[36m{}\x1b[0m'\n", conv_name);

            // Enter the new chat
            chat_with_conversation(&conv_name).await?;
        }

        // `anima chat create` or `anima chat create <name>` - create conversation without entering interactive mode
        Some(ChatCommands::Create { name }) => {
            // Create conversation with "user" as participant (agents join via @mention)
            let conv_name = store.create_conversation(name.as_deref(), &["user"])?;
            println!("Created conversation: {}", conv_name);
        }

        // `anima chat join <name>` (or `anima chat open <name>`) - join existing chat by name
        Some(ChatCommands::Join { name }) => {
            // Check if conversation exists
            if store.find_by_name(&name)?.is_none() {
                return Err(format!("Conversation '{}' not found. Use 'anima chat create {}' to create it.", name, name).into());
            }

            chat_with_conversation(&name).await?;
        }

        // `anima chat send <conv> "message"` - fire-and-forget message injection
        Some(ChatCommands::Send { conv, message }) => {
            // Check if conversation exists
            if store.find_by_name(&conv)?.is_none() {
                return Err(format!("Conversation '{}' not found", conv).into());
            }

            // Parse @mentions from message
            let mentions = parse_mentions(&message);

            // Get current participants
            let parts = store.get_participants(&conv)?;
            let participants: Vec<String> = parts.iter()
                .filter(|p| p.agent != "user")
                .map(|p| p.agent.clone())
                .collect();

            // Add mentioned agents as participants if they exist and not already in
            for agent_name in &mentions {
                if agent_name != "all" && !participants.contains(agent_name) {
                    if anima::discovery::agent_exists(agent_name) {
                        if let Err(e) = store.add_participant(&conv, agent_name) {
                            eprintln!("Warning: Could not add {} as participant: {}", agent_name, e);
                        }
                    }
                }
            }

            // Expand @all to all participants
            let updated_parts = store.get_participants(&conv)?;
            let updated_participants: Vec<String> = updated_parts.iter()
                .filter(|p| p.agent != "user")
                .map(|p| p.agent.clone())
                .collect();
            let expanded_mentions = expand_all_mention(&mentions, &updated_participants);

            let mention_refs: Vec<&str> = expanded_mentions.iter().map(|s| s.as_str()).collect();

            // Store user message in conversation
            let user_msg_id = store.add_message(&conv, "user", &message, &mention_refs)?;

            // Notify mentioned agents - wait for acknowledgments but not responses
            // The daemon processes notifications asynchronously after acknowledging
            if !expanded_mentions.is_empty() {
                let results = notify_mentioned_agents_parallel(&store, &conv, user_msg_id, &expanded_mentions).await;

                // Report results
                let mut notified = Vec::new();
                let mut queued = Vec::new();
                let mut failed = Vec::new();

                for (agent, result) in results {
                    match result {
                        NotifyResult::Acknowledged | NotifyResult::Notified { .. } => notified.push(agent),
                        NotifyResult::Queued { .. } => queued.push(agent),
                        NotifyResult::UnknownAgent => failed.push(format!("{} (unknown)", agent)),
                        NotifyResult::Failed { reason } => failed.push(format!("{} ({})", agent, reason)),
                    }
                }

                if !notified.is_empty() {
                    println!("Sent message to '{}', notified: {}", conv, notified.join(", "));
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

        // `anima chat view <conv>` - view messages (pretty by default, --raw for scripts)
        Some(ChatCommands::View { conv, limit, since, raw }) => {
            // Check if conversation exists
            if store.find_by_name(&conv)?.is_none() {
                return Err(format!("Conversation '{}' not found", conv).into());
            }

            // Get messages with filtering
            let messages = store.get_messages_filtered(&conv, limit, since)?;

            // Filter out tool messages from display (they clutter the chat)
            let messages: Vec<_> = messages.into_iter().filter(|m| m.from_agent != "tool").collect();

            if raw {
                // Raw format: ID|TIMESTAMP|FROM|CONTENT (one line per message, escaped)
                for msg in messages {
                    let escaped_content = msg.content
                        .replace('\\', "\\\\")
                        .replace('\n', "\\n")
                        .replace('|', "\\|");
                    println!("{}|{}|{}|{}", msg.id, msg.created_at, msg.from_agent, escaped_content);
                }
            } else {
                // Pretty format (human-readable)
                for msg in &messages {
                    // Format timestamp: YYYY-MM-DD HH:MM
                    let datetime = format_timestamp_pretty(msg.created_at);

                    // Show duration and context usage for agent messages if available
                    let duration_str = if msg.from_agent != "user" && msg.from_agent != "tool" {
                        msg.duration_ms.map(|d| format!(" \x1b[90m•\x1b[0m \x1b[33m{}\x1b[0m", format_duration_ms(d))).unwrap_or_default()
                    } else {
                        String::new()
                    };
                    let ctx_str = if msg.from_agent != "user" && msg.from_agent != "tool" {
                        format_context_usage(msg)
                    } else {
                        String::new()
                    };

                    // [ID] YYYY-MM-DD HH:MM • agent_name • duration • X% ctx
                    // ID and timestamp in dim gray, agent name in cyan, duration in yellow, ctx in magenta
                    println!(
                        "\x1b[90m[{}] {}\x1b[0m \x1b[90m•\x1b[0m \x1b[36m{}\x1b[0m{}{}",
                        msg.id, datetime, msg.from_agent, duration_str, ctx_str
                    );
                    // Content in normal color
                    println!("{}", msg.content);
                    println!(); // Blank line between messages
                }
            }
        }

        // `anima chat pause <conv>` - pause notifications for a conversation (supports glob patterns)
        Some(ChatCommands::Pause { conv, force }) => {
            let matches = store.match_conversations(&conv)?;

            if matches.is_empty() {
                return Err(format!("No conversations match pattern: {}", conv).into());
            }

            // Ask for confirmation if multiple matches or using wildcards
            if !force && (matches.len() > 1 || has_wildcards(&conv)) {
                if !confirm_action("pause", &matches)? {
                    println!("Aborted.");
                    return Ok(());
                }
            }

            for conv_name in &matches {
                let _ = store.set_paused(conv_name, true)?;
                println!("Paused conversation '\x1b[36m{}\x1b[0m'", conv_name);
            }
        }

        // `anima chat resume <conv>` - resume a paused conversation (supports glob patterns)
        Some(ChatCommands::Resume { conv, force }) => {
            let matches = store.match_conversations(&conv)?;

            if matches.is_empty() {
                return Err(format!("No conversations match pattern: {}", conv).into());
            }

            // Ask for confirmation if multiple matches or using wildcards
            if !force && (matches.len() > 1 || has_wildcards(&conv)) {
                if !confirm_action("resume", &matches)? {
                    println!("Aborted.");
                    return Ok(());
                }
            }

            for conv_name in &matches {
                // Resume the conversation and get the paused_at_msg_id
                let paused_at_msg_id = store.set_paused(conv_name, false)?;
                let mut catchup_count = 0;

                // Process catchup items (tool calls and mentions skipped during pause)
                if let Some(paused_at) = paused_at_msg_id {
                    if let Ok(catchup_msgs) = store.get_catchup_messages(conv_name, paused_at) {
                        let catchup_items = ConversationStore::build_catchup_items(&catchup_msgs);
                        for item in &catchup_items {
                            // Process @mentions
                            if !item.mentions.is_empty() {
                                let _ = notify_mentioned_agents_parallel(
                                    &store,
                                    conv_name,
                                    item.message.id,
                                    &item.mentions
                                ).await;
                                catchup_count += item.mentions.len();
                            }
                            // Note: Tool calls in catchup items will be handled
                            // by the agent when it processes the follow-up notification
                        }
                    }
                }

                // Also check for pending notifications (for agents that weren't running)
                let pending = store.get_pending_notifications_for_conversation(conv_name)?;
                let pending_count = pending.len();

                if pending_count > 0 {
                    // Process pending notifications
                    for notification in &pending {
                        let mentions = vec![notification.agent.clone()];
                        let _ = notify_mentioned_agents_parallel(
                            &store,
                            conv_name,
                            notification.message_id,
                            &mentions
                        ).await;

                        // Clear this notification after processing
                        let _ = store.delete_pending_notification(notification.id);
                    }
                    catchup_count += pending_count;
                }

                if catchup_count > 0 {
                    println!("Resumed conversation '\x1b[36m{}\x1b[0m' ({} pending item(s) processed)", conv_name, catchup_count);
                } else {
                    println!("Resumed conversation '\x1b[36m{}\x1b[0m'", conv_name);
                }
            }
        }

        // `anima chat delete <name>` - delete a conversation (supports glob patterns)
        Some(ChatCommands::Delete { name, force }) => {
            let matches = store.match_conversations(&name)?;

            if matches.is_empty() {
                return Err(format!("No conversations match pattern: {}", name).into());
            }

            // Ask for confirmation if multiple matches or using wildcards
            if !force && (matches.len() > 1 || has_wildcards(&name)) {
                if !confirm_action("delete", &matches)? {
                    println!("Aborted.");
                    return Ok(());
                }
            }

            for conv_name in &matches {
                store.delete_conversation(conv_name)?;
                println!("Deleted conversation '\x1b[36m{}\x1b[0m'", conv_name);
            }
        }

        // `anima chat clear <conv>` - clear all messages from a conversation (supports glob patterns)
        Some(ChatCommands::Clear { conv, force }) => {
            let matches = store.match_conversations(&conv)?;

            if matches.is_empty() {
                return Err(format!("No conversations match pattern: {}", conv).into());
            }

            // Ask for confirmation if multiple matches or using wildcards
            if !force && (matches.len() > 1 || has_wildcards(&conv)) {
                if !confirm_action("clear", &matches)? {
                    println!("Aborted.");
                    return Ok(());
                }
            }

            for conv_name in &matches {
                let deleted = store.clear_messages(conv_name)?;
                println!("Cleared {} messages from '\x1b[36m{}\x1b[0m'", deleted, conv_name);
            }
        }

        // `anima chat cleanup` - run cleanup_expired
        Some(ChatCommands::Cleanup) => {
            let (messages_deleted, convs_deleted) = store.cleanup_expired()?;
            println!("Cleanup complete:");
            println!("  - {} expired messages deleted", messages_deleted);
            println!("  - {} empty conversations deleted", convs_deleted);
        }

        // `anima chat` - list all conversations
        None => {
            let conversations = store.list_conversations()?;

            if conversations.is_empty() {
                println!("\x1b[33mNo conversations found.\x1b[0m");
                println!("Create one with: \x1b[36manima chat create\x1b[0m");
                return Ok(());
            }

            // Column order: NAME, MSGS, UPDATED, PARTICIPANTS (last so it can overflow)
            println!("\x1b[1m{:<30} {:>6}   {:<10} {}\x1b[0m", "NAME", "MSGS", "UPDATED", "PARTICIPANTS");
            println!("{}", "-".repeat(80));

            for conv in conversations {
                let participants = store.get_participants(&conv.name)?;
                let agents: Vec<_> = participants.iter().map(|p| p.agent.as_str()).collect();

                // Get message count
                let msg_count = store.get_message_count(&conv.name)?;

                // Format updated_at as relative time
                let updated = format_relative_time(conv.updated_at);

                // Show paused indicator
                let name_display = if conv.paused {
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

/// Handle memory subcommands.
async fn handle_memory_command(command: MemoryCommands) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        MemoryCommands::List { agent, limit } => {
            let agent_path = resolve_agent_path(&agent);
            let memory_path = agent_path.join("memory.db");

            if !memory_path.exists() {
                return Err(format!("No memory database found for agent '{}' at {}", agent, memory_path.display()).into());
            }

            let store = SemanticMemoryStore::open(&memory_path, &agent)?;
            let memories = store.list_all(Some(limit))?;

            if memories.is_empty() {
                println!("No memories found for agent '{}'", agent);
                return Ok(());
            }

            // Print header
            println!("\x1b[1m{:<6} {:<12} {:<10} {}\x1b[0m", "ID", "CREATED", "SOURCE", "CONTENT");
            println!("{}", "─".repeat(110));

            for m in memories {
                let age = format_age(m.created_at);
                // Truncate content to ~80 chars for display
                let content_display: String = m.content.chars().take(80).collect();
                let content_display = if m.content.chars().count() > 80 {
                    format!("{}...", content_display)
                } else {
                    content_display
                };
                // Replace newlines with spaces for single-line display
                let content_display = content_display.replace('\n', " ");

                println!("{:<6} {:<12} {:<10} {}", m.id, age, m.source, content_display);
            }
        }

        MemoryCommands::Search { agent, query, limit } => {
            let agent_path = resolve_agent_path(&agent);
            let memory_path = agent_path.join("memory.db");

            if !memory_path.exists() {
                return Err(format!("No memory database found for agent '{}' at {}", agent, memory_path.display()).into());
            }

            let store = SemanticMemoryStore::open(&memory_path, &agent)?;

            // For semantic search, we need embeddings. Check if agent has embedding config.
            // Try to get embedding from Ollama if available
            let embedding_result = get_query_embedding(&query).await;

            let results = match embedding_result {
                Ok(embedding) => {
                    #[allow(deprecated)]
                    store.recall_with_embedding(&query, limit, Some(&embedding))?
                }
                Err(_) => {
                    // Fall back to listing all and doing simple text search
                    eprintln!("\x1b[33mWarning: Could not generate embedding, falling back to text search\x1b[0m");
                    let all = store.list_all(None)?;
                    let query_lower = query.to_lowercase();
                    all.into_iter()
                        .filter(|m| m.content.to_lowercase().contains(&query_lower))
                        .take(limit)
                        .map(|m| (m, 1.0)) // Dummy score for text matches
                        .collect()
                }
            };

            if results.is_empty() {
                println!("No memories matching '{}' found for agent '{}'", query, agent);
                return Ok(());
            }

            // Print header
            println!("\x1b[1m{:<6} {:<8} {:<12} {:<10} {}\x1b[0m", "ID", "SCORE", "CREATED", "SOURCE", "CONTENT");
            println!("{}", "─".repeat(80));

            for (m, score) in results {
                let age = format_age(m.created_at);
                // Truncate content to ~35 chars for display
                let content_display: String = m.content.chars().take(35).collect();
                let content_display = if m.content.chars().count() > 35 {
                    format!("{}...", content_display)
                } else {
                    content_display
                };
                let content_display = content_display.replace('\n', " ");

                println!("{:<6} {:<8.3} {:<12} {:<10} {}", m.id, score, age, m.source, content_display);
            }
        }

        MemoryCommands::Delete { agent, id } => {
            let agent_path = resolve_agent_path(&agent);
            let memory_path = agent_path.join("memory.db");

            if !memory_path.exists() {
                return Err(format!("No memory database found for agent '{}' at {}", agent, memory_path.display()).into());
            }

            let store = SemanticMemoryStore::open(&memory_path, &agent)?;
            let deleted = store.delete(id)?;

            if deleted {
                println!("Deleted memory #{} for agent '{}'", id, agent);
            } else {
                println!("Memory #{} not found for agent '{}'", id, agent);
            }
        }

        MemoryCommands::Clear { agent, force } => {
            let agent_path = resolve_agent_path(&agent);
            let memory_path = agent_path.join("memory.db");

            if !memory_path.exists() {
                return Err(format!("No memory database found for agent '{}' at {}", agent, memory_path.display()).into());
            }

            let store = SemanticMemoryStore::open(&memory_path, &agent)?;
            let count = store.count()?;

            if count == 0 {
                println!("No memories to clear for agent '{}'", agent);
                return Ok(());
            }

            // Confirm unless --force
            if !force {
                print!("This will delete {} memories for '{}'. Continue? [y/N] ", count, agent);
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
            let agent_path = resolve_agent_path(&agent);
            let memory_path = agent_path.join("memory.db");

            if !memory_path.exists() {
                return Err(format!("No memory database found for agent '{}' at {}", agent, memory_path.display()).into());
            }

            let store = SemanticMemoryStore::open(&memory_path, &agent)?;
            let count = store.count()?;

            println!("Agent '{}' has {} memories", agent, count);
        }

        MemoryCommands::Show { agent, id } => {
            let agent_path = resolve_agent_path(&agent);
            let memory_path = agent_path.join("memory.db");

            if !memory_path.exists() {
                return Err(format!("No memory database found for agent '{}' at {}", agent, memory_path.display()).into());
            }

            let store = SemanticMemoryStore::open(&memory_path, &agent)?;
            
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

        MemoryCommands::Add { agent, content, importance } => {
            let agent_path = resolve_agent_path(&agent);
            let memory_path = agent_path.join("memory.db");

            // Create memory.db if it doesn't exist
            let store = SemanticMemoryStore::open(&memory_path, &agent)?;
            
            // Clamp importance to valid range
            let importance = importance.clamp(0.0, 1.0);
            
            match store.save(&content, importance, "cli")? {
                anima::memory::SaveResult::New(id) => {
                    println!("Created memory #{} for agent '{}'", id, agent);
                }
                anima::memory::SaveResult::Reinforced(id, old_imp, new_imp) => {
                    println!("Reinforced existing memory #{} (importance: {:.2} → {:.2})", id, old_imp, new_imp);
                }
            }
        }

        MemoryCommands::Replace { agent, id, content } => {
            let agent_path = resolve_agent_path(&agent);
            let memory_path = agent_path.join("memory.db");

            if !memory_path.exists() {
                return Err(format!("No memory database found for agent '{}' at {}", agent, memory_path.display()).into());
            }

            let store = SemanticMemoryStore::open(&memory_path, &agent)?;
            
            if store.update_content(id, &content)? {
                println!("Updated memory #{} for agent '{}'", id, agent);
            } else {
                return Err(format!("Memory #{} not found for agent '{}'", id, agent).into());
            }
        }
    }

    Ok(())
}

/// Try to get an embedding for a query string using Ollama.
async fn get_query_embedding(query: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use anima::EmbeddingClient;

    // Try to use the default embedding model (nomic-embed-text is common)
    let client = EmbeddingClient::new("nomic-embed-text", None);
    let embedding = client.embed(query).await?;
    Ok(embedding)
}

/// Format a Unix timestamp as "YYYY-MM-DD HH:MM" for pretty display.
fn format_timestamp_pretty(timestamp: i64) -> String {
    use std::time::{Duration, UNIX_EPOCH};
    let datetime = UNIX_EPOCH + Duration::from_secs(timestamp as u64);
    let datetime: chrono::DateTime<chrono::Local> = datetime.into();
    datetime.format("%Y-%m-%d %H:%M").to_string()
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
/// Returns formatted string like "ctx 2k/32k (26%)" or empty string if data unavailable.
fn format_context_usage(msg: &anima::conversation::ConversationMessage) -> String {
    match (msg.tokens_in, msg.tokens_out, msg.num_ctx) {
        (Some(tokens_in), Some(tokens_out), Some(num_ctx)) if num_ctx > 0 => {
            let total_tokens = tokens_in + tokens_out;
            let percentage = (total_tokens as f64 / num_ctx as f64 * 100.0) as u32;
            let used_str = format_tokens_short(total_tokens);
            let ctx_str = format_tokens_short(num_ctx);
            format!(" \x1b[90m•\x1b[0m \x1b[35m{}/{} ({}%)\x1b[0m", used_str, ctx_str, percentage)
        }
        _ => String::new(),
    }
}

/// Format token count as short string (e.g., 2048 -> "2k", 32768 -> "33k", 500 -> "500")
fn format_tokens_short(tokens: i64) -> String {
    if tokens >= 1000 {
        let k = (tokens + 500) / 1000; // round to nearest k
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

/// Run an agent from a directory and start an interactive REPL
/// In daemon mode, this starts the daemon (if needed) and connects via REPL.
async fn run_agent_dir(agent: &str) -> Result<(), Box<dyn std::error::Error>> {
    let agent_path = resolve_agent_path(agent);

    // Load the agent directory to verify it exists
    let agent_dir = AgentDir::load(&agent_path)?;
    let agent_name = agent_dir.config.agent.name.clone();

    // Start a REPL connected to this agent (will start daemon if needed)
    let mut repl = Repl::with_agent(agent_name.clone()).await;
    println!("\x1b[32m✓ Connected to agent '{}'\x1b[0m", agent_name);

    repl.run().await?;
    Ok(())
}

/// Scaffold a new agent directory (delegates to shared function in agent_dir)
fn create_agent(name: &str, path: Option<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    anima::agent_dir::create_agent(name, path)?;
    Ok(())
}

/// List all agents in ~/.anima/agents/
fn list_agents() {
    let agents_path = agents_dir();

    if !agents_path.exists() {
        println!("\x1b[33mNo agents directory found at {}\x1b[0m", agents_path.display());
        println!("Create an agent with: \x1b[36manima create <name>\x1b[0m");
        return;
    }

    let entries: Vec<_> = match std::fs::read_dir(&agents_path) {
        Ok(dir) => dir
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .filter(|e| e.path().join("config.toml").exists())
            .collect(),
        Err(e) => {
            eprintln!("\x1b[31mCould not read agents directory: {}\x1b[0m", e);
            return;
        }
    };

    if entries.is_empty() {
        println!("\x1b[33mNo agents found in {}\x1b[0m", agents_path.display());
        println!("Create an agent with: \x1b[36manima create <name>\x1b[0m");
        return;
    }

    println!("\x1b[1mAgents in ~/.anima/agents/:\x1b[0m");
    for entry in entries {
        let name = entry.file_name().to_string_lossy().to_string();

        // Try to load config to get more info
        let info = match AgentDir::load(entry.path()) {
            Ok(agent_dir) => {
                match agent_dir.resolve_llm_config() {
                    Ok(resolved) => format!(" ({}/{})", resolved.provider, resolved.model),
                    Err(_) => " (config error)".to_string(),
                }
            }
            Err(_) => " (config error)".to_string(),
        };

        println!("  \x1b[36m{}\x1b[0m{}", name, info);
    }
}

/// Run an agent with a single task (non-interactive, from config file)
async fn run_agent_task(config_path: &str, task: &str, stream: bool, verbose_cli: bool) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load config
    let config = AgentConfig::from_file(config_path)?;

    // 2. Create observer (CLI flag overrides config)
    let verbose = verbose_cli || config.observe.verbose;
    let observer = Arc::new(ConsoleObserver::new(verbose));

    // 2. Create LLM from config (check env for API key)
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
            let api_key = std::env::var("ANTHROPIC_API_KEY")
                .map_err(|_| "ANTHROPIC_API_KEY environment variable not set")?;
            let mut client = AnthropicClient::new(api_key).with_model(&config.llm.model);
            if let Some(base_url) = &config.llm.base_url {
                client = client.with_base_url(base_url);
            }
            Arc::new(client)
        }
        other => return Err(format!("Unknown LLM provider: {}", other).into()),
    };

    // 3. Create memory from config
    let memory: Box<dyn anima::Memory> = match config.memory.backend.as_str() {
        "sqlite" => {
            let path = config.memory.path.as_deref().unwrap_or("anima.db");
            Box::new(SqliteMemory::open(path, &config.agent.name)?)
        }
        "in_memory" | _ => Box::new(InMemoryStore::new()),
    };

    // 4. Create agent, register enabled tools
    let mut runtime = Runtime::new();
    let mut agent = runtime.spawn_agent(config.agent.name.clone()).await;

    // Register enabled tools
    for tool_name in &config.tools.enabled {
        match tool_name.as_str() {
            "add" => agent.register_tool(Arc::new(AddTool)),
            "echo" => agent.register_tool(Arc::new(EchoTool)),
            "read_file" => agent.register_tool(Arc::new(ReadFileTool)),
            "write_file" => agent.register_tool(Arc::new(WriteFileTool)),
            "http" => agent.register_tool(Arc::new(HttpTool::new())),
            "shell" => agent.register_tool(Arc::new(ShellTool::new())),
            unknown => eprintln!("Warning: Unknown tool '{}', skipping", unknown),
        }
    }

    agent = agent.with_llm(llm);
    agent = agent.with_memory(memory);
    agent = agent.with_observer(observer);

    // 5. Build ThinkOptions from config
    let auto_memory = if config.think.auto_memory {
        Some(AutoMemoryConfig {
            max_entries: config.think.max_memory_entries,
            include_recent: true,
            key_prefixes: vec![],
        })
    } else {
        None
    };

    let reflection = if config.think.reflection {
        Some(ReflectionConfig::default())
    } else {
        None
    };

    let options = ThinkOptions {
        max_iterations: config.think.max_iterations,
        system_prompt: config.agent.system_prompt,
        always_prompt: None, // Not supported in task mode yet
        auto_memory,
        reflection,
        stream,
        retry_policy: Some(config.retry.to_policy()),
        conversation_history: None,
        external_tools: None, // One-shot mode uses registered tools
    };

    // 6. Run agent with streaming or non-streaming based on flag
    if stream {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(100);

        // Spawn a task to print tokens as they arrive
        let print_task = tokio::spawn(async move {
            while let Some(token) = rx.recv().await {
                print!("{}", token);
                // Flush stdout to ensure real-time display
                let _ = io::stdout().flush();
            }
            println!(); // Final newline after streaming completes
        });

        // Run the streaming agent
        let result = agent.think_streaming_with_options(task, options, tx).await?;

        // Wait for print task to complete
        let _ = print_task.await;

        // Note: result contains the full response, already printed via streaming
        let _ = result;
    } else {
        let result = agent.think_with_options(task, options).await?;
        println!("{}", result.response);
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
        assert!(agent_path.join("always.md").exists());

        // Check config.toml content
        let config_content = std::fs::read_to_string(agent_path.join("config.toml")).unwrap();
        assert!(config_content.contains("name = \"test-agent\""));
        assert!(config_content.contains("[llm]"));
        assert!(config_content.contains("[memory]"));
        assert!(config_content.contains("always_file = \"always.md\""));

        // Check system.md content
        let system_content = std::fs::read_to_string(agent_path.join("system.md")).unwrap();
        assert!(system_content.contains("# test-agent"));
        assert!(system_content.contains("You are test-agent"));

        // Check always.md content
        let always_content = std::fs::read_to_string(agent_path.join("always.md")).unwrap();
        assert!(always_content.contains("# Always"));
        assert!(always_content.contains("How Conversations Work"));
        assert!(always_content.contains("Never @mention yourself"));
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
        assert_eq!(socket_path, PathBuf::from("/custom/path/myagent/agent.sock"));
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
