use anima::{
    Runtime, OpenAIClient, AnthropicClient, OllamaClient, ThinkOptions, AutoMemoryConfig, ReflectionConfig,
    InMemoryStore, SqliteMemory, LLM, ConversationStore, canonical_1to1_id, canonical_group_id,
    parse_mentions, expand_all_mention, notify_mentioned_agents_parallel, NotifyResult,
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
use std::io::{self, Write};
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
    /// Send a message to a running agent daemon
    Send {
        /// Agent name (from ~/.anima/agents/)
        agent: String,
        /// Message to send
        message: String,
    },
    /// Start an interactive chat session with agent(s), using persistent conversation storage
    Chat {
        /// Agent name(s) (from ~/.anima/agents/). Multiple agents create a group conversation.
        #[arg(required_unless_present = "conv")]
        agents: Vec<String>,
        /// Use an existing conversation by name/id instead of creating implicit conversation
        #[arg(long)]
        conv: Option<String>,
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
    /// Start an agent daemon in the background
    Start {
        /// Agent name (from ~/.anima/agents/) or path to agent directory
        agent: String,
    },
    /// Stop a running agent daemon
    Stop {
        /// Agent name (from ~/.anima/agents/) or path to agent directory
        agent: String,
    },
    /// Clear conversation history for a running agent daemon
    Clear {
        /// Agent name (from ~/.anima/agents/) or path to agent directory
        agent: String,
    },
    /// Restart a running agent daemon (stop then start)
    Restart {
        /// Agent name (from ~/.anima/agents/) or path to agent directory
        agent: String,
    },
    /// Show the system prompt for a running agent daemon
    System {
        /// Agent name (from ~/.anima/agents/) or path to agent directory
        agent: String,
    },
    /// Manage conversations
    Conv {
        #[command(subcommand)]
        command: ConvCommands,
    },
}

#[derive(Subcommand)]
enum ConvCommands {
    /// Create a new conversation with participants
    New {
        /// Name for the conversation
        name: String,
        /// Comma-separated list of agent names to include
        #[arg(long = "with")]
        participants: String,
    },
    /// List all conversations
    List,
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
        Commands::Send { agent, message } => {
            if let Err(e) = send_message(&agent, &message).await {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Chat { agents, conv } => {
            if let Err(e) = chat_with_conversation(&agents, conv.as_deref()).await {
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
        Commands::Conv { command } => {
            if let Err(e) = handle_conv_command(command) {
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

/// Send a single message to a running agent daemon.
async fn send_message(agent: &str, message: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut api = connect_to_agent(agent).await?;

    // Send the message
    api.write_request(&Request::Message {
        content: message.to_string(),
    }).await.map_err(|e| format!("Failed to send message: {}", e))?;

    // Read responses (may be streaming or non-streaming)
    loop {
        match api.read_response().await.map_err(|e| format!("Failed to read response: {}", e))? {
            Some(Response::Chunk { text }) => {
                // Streaming: print tokens as they arrive
                print!("{}", text);
                io::stdout().flush()?;
            }
            Some(Response::ToolCall { tool, params }) => {
                // Show tool call: " - [tool] safe_shell: ls -la ~/dev"
                let param_summary = format_tool_params(&params);
                eprintln!(" - [tool] {}: {}", tool, param_summary);
            }
            Some(Response::Done) => {
                // Streaming complete
                println!();  // Final newline
                break;
            }
            Some(Response::Message { content }) => {
                // Fallback for non-streaming responses (JSON-block mode)
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

/// Default number of context messages to load from conversation history.
const DEFAULT_CONTEXT_MESSAGES: usize = 20;

/// Start an interactive chat session with agents using persistent conversation storage.
///
/// - With a single agent: creates/reuses implicit "1:1:{agent}:user" conversation
///   - Messages are sent directly to the agent
/// - With multiple agents: creates/reuses implicit "group:{sorted_names}" conversation
///   - Messages are ONLY sent to @mentioned agents
///   - If no @mentions, user is warned
/// - With --conv flag: uses the specified existing conversation
async fn chat_with_conversation(
    agents: &[String],
    conv_name: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize conversation store
    let store = ConversationStore::init()?;

    // Determine if this is a group chat (multiple agents)
    let is_group_chat = agents.len() > 1;

    // Determine conversation ID and participants
    let (conv_id, participants) = if let Some(name) = conv_name {
        // Explicit conversation - must exist
        let conv = store.find_by_name(name)?
            .ok_or_else(|| format!("Conversation '{}' not found", name))?;
        let parts = store.get_participants(&conv.id)?;
        let agent_names: Vec<String> = parts.iter()
            .filter(|p| p.agent != "user")
            .map(|p| p.agent.clone())
            .collect();
        (conv.id, agent_names)
    } else if agents.len() == 1 {
        // 1:1 conversation
        let agent = &agents[0];
        let id = canonical_1to1_id(agent);
        let parts: Vec<&str> = vec![agent.as_str(), "user"];
        // Get or create the conversation
        store.get_or_create_conversation(&id, &parts)?;
        (id, vec![agent.clone()])
    } else {
        // Group conversation
        let agent_refs: Vec<&str> = agents.iter().map(|s| s.as_str()).collect();
        let id = canonical_group_id(&agent_refs);
        let mut parts = agent_refs.clone();
        if !parts.contains(&"user") {
            parts.push("user");
        }
        // Get or create the conversation
        store.get_or_create_conversation(&id, &parts)?;
        (id, agents.to_vec())
    };

    // Print connection info
    if is_group_chat {
        println!("\x1b[32m✓ Group chat with: {}\x1b[0m", participants.join(", "));
        println!("\x1b[90mConversation: {}\x1b[0m", conv_id);
        println!("\x1b[33mNote: In group chat, use @mentions to notify agents (e.g., @arya @gendry)\x1b[0m");
    } else {
        println!("\x1b[32m✓ Chat with '{}'\x1b[0m", participants[0]);
        println!("\x1b[90mConversation: {}\x1b[0m", conv_id);
    }
    println!("Type your messages. Press Ctrl+D or Ctrl+C to exit.\n");

    // Use rustyline for interactive input
    let mut rl = rustyline::DefaultEditor::new()?;

    // Prompt format differs for group vs 1:1
    let prompt = if is_group_chat {
        "\x1b[36myou>\x1b[0m ".to_string()
    } else {
        format!("\x1b[36m{}>\x1b[0m ", participants[0])
    };

    loop {
        match rl.readline(&prompt) {
            Ok(line) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                // Add to history
                let _ = rl.add_history_entry(line);

                // Parse @mentions from user message
                let mentions = parse_mentions(line);

                // In group chat, require @mentions
                if is_group_chat && mentions.is_empty() {
                    eprintln!("\x1b[33m⚠ In group chat, you must @mention agents to notify them.\x1b[0m");
                    eprintln!("\x1b[33m  Example: @{} what do you think?\x1b[0m", participants[0]);
                    continue;
                }

                // Expand @all to all participants (excluding "user")
                let expanded_mentions = expand_all_mention(&mentions, &participants);

                let mention_refs: Vec<&str> = expanded_mentions.iter().map(|s| s.as_str()).collect();

                // Store user message in conversation with mentions
                let user_msg_id = store.add_message(&conv_id, "user", line, &mention_refs)?;

                // Determine which agents to notify
                let agents_to_notify: Vec<String> = if is_group_chat {
                    // Group chat: only notify @mentioned agents (with @all already expanded)
                    expanded_mentions.clone()
                } else {
                    // 1:1 chat: always notify the single agent
                    participants.clone()
                };

                // Notify agents in parallel and collect responses
                if !agents_to_notify.is_empty() {
                    let notify_results = notify_mentioned_agents_parallel(&store, &conv_id, user_msg_id, &agents_to_notify).await;

                    // Display results for each agent and check for @mentions in their responses
                    let mut agent_mentions_to_notify: Vec<(i64, Vec<String>)> = Vec::new();

                    for (agent, result) in &notify_results {
                        match result {
                            NotifyResult::Notified { response_message_id } => {
                                // Fetch and display the agent's response
                                if let Ok(msgs) = store.get_messages(&conv_id, Some(DEFAULT_CONTEXT_MESSAGES)) {
                                    if let Some(response_msg) = msgs.iter().find(|m| m.id == *response_message_id) {
                                        println!("\n\x1b[36m[{}]:\x1b[0m {}\n", agent, response_msg.content);

                                        // Parse @mentions from the agent's response
                                        let response_mentions = parse_mentions(&response_msg.content);
                                        // Filter out self and "user"
                                        let filtered_mentions: Vec<String> = response_mentions
                                            .into_iter()
                                            .filter(|m| m != agent && m != "user")
                                            .collect();

                                        if !filtered_mentions.is_empty() {
                                            agent_mentions_to_notify.push((*response_message_id, filtered_mentions));
                                        }
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

                    // Notify agents mentioned in agent responses (agent-to-agent communication)
                    for (msg_id, mentions) in agent_mentions_to_notify {
                        let followup_results = notify_mentioned_agents_parallel(&store, &conv_id, msg_id, &mentions).await;

                        for (agent, result) in &followup_results {
                            match result {
                                NotifyResult::Notified { response_message_id } => {
                                    if let Ok(msgs) = store.get_messages(&conv_id, Some(DEFAULT_CONTEXT_MESSAGES)) {
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

/// One-shot query to an agent without daemon.
/// Loads the agent config, creates LLM, sends message, prints response.
async fn ask_agent(agent: &str, message: &str) -> Result<(), Box<dyn std::error::Error>> {
    use anima::{ChatMessage, LLM};

    let agent_path = resolve_agent_path(agent);

    // Load the agent directory
    let agent_dir = AgentDir::load(&agent_path)
        .map_err(|e| format!("Failed to load agent '{}': {}", agent, e))?;

    // Load persona if configured
    let persona = agent_dir.load_persona()
        .map_err(|e| format!("Failed to load persona: {}", e))?;

    // Load always content if configured
    let always = agent_dir.load_always()
        .map_err(|e| format!("Failed to load always: {}", e))?;

    // Resolve LLM config (loads model file if specified, applies overrides)
    let llm_config = agent_dir.resolve_llm_config()
        .map_err(|e| format!("Failed to resolve LLM config: {}", e))?;

    // Get API key using resolved config
    let api_key = AgentDir::api_key_for_config(&llm_config)
        .map_err(|e| format!("Failed to get API key: {}", e))?;

    // Create LLM from resolved config
    let llm: Arc<dyn LLM> = match llm_config.provider.as_str() {
        "openai" => {
            let key = api_key.ok_or("OpenAI API key not configured")?;
            let mut client = OpenAIClient::new(key).with_model(&llm_config.model);
            if let Some(ref base_url) = llm_config.base_url {
                client = client.with_base_url(base_url);
            }
            Arc::new(client)
        }
        "anthropic" => {
            let key = api_key.ok_or("Anthropic API key not configured")?;
            let mut client = AnthropicClient::new(key).with_model(&llm_config.model);
            if let Some(ref base_url) = llm_config.base_url {
                client = client.with_base_url(base_url);
            }
            Arc::new(client)
        }
        "ollama" => {
            let mut client = OllamaClient::new()
                .with_model(&llm_config.model)
                .with_thinking(llm_config.thinking)
                .with_num_ctx(llm_config.num_ctx);
            if let Some(ref base_url) = llm_config.base_url {
                client = client.with_base_url(base_url);
            }
            Arc::new(client)
        }
        other => return Err(format!("Unsupported LLM provider: {}", other).into()),
    };

    // Build messages
    let mut messages = Vec::new();

    // Add system prompt from persona if available
    if let Some(persona_content) = persona {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: Some(persona_content),
            tool_call_id: None,
            tool_calls: None,
        });
    }

    // Inject always content as system message just before user message (recency bias)
    if let Some(always_content) = always {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: Some(always_content),
            tool_call_id: None,
            tool_calls: None,
        });
    }

    // Add user message
    messages.push(ChatMessage {
        role: "user".to_string(),
        content: Some(message.to_string()),
        tool_call_id: None,
        tool_calls: None,
    });

    // Call LLM directly (no tools for simple ask)
    let response = llm.chat_complete(messages, None).await
        .map_err(|e| format!("LLM error: {}", e))?;

    // Print the response
    if let Some(content) = response.content {
        println!("{}", content);
    }

    Ok(())
}

/// Start an agent daemon in the background.
fn start_agent(agent: &str) -> Result<(), Box<dyn std::error::Error>> {
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

    // Spawn daemon process in background
    // Use `run --daemon` for the actual daemon logic
    let child = Command::new(&exe)
        .arg("run")
        .arg(agent)
        .arg("--daemon")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("Failed to spawn daemon: {}", e))?;

    let pid = child.id();
    println!("Started {} (pid {})", agent, pid);

    Ok(())
}

/// Stop a running agent daemon.
async fn stop_agent(agent: &str) -> Result<(), Box<dyn std::error::Error>> {
    let agent_path = resolve_agent_path(agent);
    let pid_path = agent_path.join("daemon.pid");

    // Check if running
    if !PidFile::is_running(&pid_path) {
        println!("Agent '{}' is not running", agent);
        return Ok(());
    }

    let pid = PidFile::read(&pid_path).unwrap_or(0);

    // Try to send shutdown via socket first (graceful shutdown)
    match connect_to_agent(agent).await {
        Ok(mut api) => {
            // Send shutdown request
            if let Err(e) = api.write_request(&Request::Shutdown).await {
                eprintln!("Warning: Failed to send shutdown request: {}", e);
            } else {
                // Wait briefly for graceful shutdown
                for _ in 0..10 {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    if !PidFile::is_running(&pid_path) {
                        println!("Stopped {}", agent);
                        return Ok(());
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Warning: Could not connect to agent socket: {}", e);
        }
    }

    // If socket shutdown didn't work, send SIGTERM
    unsafe {
        if libc::kill(pid as i32, libc::SIGTERM) == 0 {
            // Wait for process to terminate
            for _ in 0..20 {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                if !PidFile::is_running(&pid_path) {
                    println!("Stopped {}", agent);
                    return Ok(());
                }
            }
            eprintln!("Warning: Agent did not stop within timeout");
        } else {
            eprintln!("Warning: Failed to send SIGTERM to pid {}", pid);
        }
    }

    println!("Stopped {}", agent);
    Ok(())
}

/// Restart a running agent daemon (stop then start).
async fn restart_agent(agent: &str) -> Result<(), Box<dyn std::error::Error>> {
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
        Some(Response::System { persona }) => {
            println!("{}", persona);
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

/// Handle conversation subcommands.
fn handle_conv_command(command: ConvCommands) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        ConvCommands::New { name, participants } => {
            let store = ConversationStore::init()?;

            // Parse comma-separated participants
            let agents: Vec<&str> = participants.split(',').map(|s| s.trim()).collect();

            if agents.is_empty() {
                return Err("At least one participant is required".into());
            }

            let id = store.create_conversation(Some(&name), &agents)?;
            println!("Created conversation '\x1b[36m{}\x1b[0m' (id: {})", name, id);
            println!("Participants: {}", agents.join(", "));
        }
        ConvCommands::List => {
            let store = ConversationStore::init()?;
            let conversations = store.list_conversations()?;

            if conversations.is_empty() {
                println!("\x1b[33mNo conversations found.\x1b[0m");
                println!("Create one with: \x1b[36manima conv new <name> --with <agents>\x1b[0m");
                return Ok(());
            }

            println!("\x1b[1m{:<20} {:<18} {}\x1b[0m", "NAME", "ID", "PARTICIPANTS");
            println!("{}", "-".repeat(60));

            for conv in conversations {
                let participants = store.get_participants(&conv.id)?;
                let agents: Vec<_> = participants.iter().map(|p| p.agent.as_str()).collect();
                let name = conv.name.unwrap_or_else(|| "(unnamed)".to_string());

                // Truncate ID for display
                let short_id = if conv.id.len() > 16 {
                    &conv.id[..16]
                } else {
                    &conv.id
                };

                println!(
                    "{:<20} {:<18} {}",
                    name,
                    short_id,
                    agents.join(", ")
                );
            }
        }
    }

    Ok(())
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
        assert!(agent_path.join("persona.md").exists());
        assert!(agent_path.join("always.md").exists());

        // Check config.toml content
        let config_content = std::fs::read_to_string(agent_path.join("config.toml")).unwrap();
        assert!(config_content.contains("name = \"test-agent\""));
        assert!(config_content.contains("[llm]"));
        assert!(config_content.contains("[memory]"));
        assert!(config_content.contains("always_file = \"always.md\""));

        // Check persona.md content
        let persona_content = std::fs::read_to_string(agent_path.join("persona.md")).unwrap();
        assert!(persona_content.contains("# test-agent"));
        assert!(persona_content.contains("You are test-agent"));

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
