//! REPL for interacting with agent daemons.
//!
//! In the daemon architecture, agents always run as separate processes.
//! The REPL connects to running daemons via Unix sockets.

use std::collections::HashMap;
use std::error::Error;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::{Editor, Helper};
use tokio::net::UnixStream;

use crate::agent::strip_thinking;
use crate::debug;
use crate::discovery::{self, RunningAgent};
use crate::socket_api::{SocketApi, Request, Response};

/// Parse @mentions from input text.
/// Pattern: `@([a-zA-Z][a-zA-Z0-9_-]*)`
/// Format elapsed duration as "Xm:Ys" or "X.Xs" for short durations.
fn format_elapsed(duration: std::time::Duration) -> String {
    let secs = duration.as_secs();
    if secs >= 60 {
        let mins = secs / 60;
        let remaining_secs = secs % 60;
        format!("{}m:{}s", mins, remaining_secs)
    } else {
        format!("{:.1}s", duration.as_secs_f64())
    }
}

/// Returns a Vec of mentioned agent names (without the @ prefix).
/// Special case: "@all" is returned as-is.
fn parse_mentions(input: &str) -> Vec<String> {
    let re = regex::Regex::new(r"@([a-zA-Z][a-zA-Z0-9_-]*)").unwrap();
    re.captures_iter(input)
        .map(|cap| cap[1].to_string())
        .collect()
}

/// Maximum number of entries to keep in the conversation log.
/// Prevents unbounded memory growth in long sessions.
const MAX_CONVERSATION_LOG: usize = 100;

const BANNER_ART: &str = r#"
    _          _
   / \   _ __ (_)_ __ ___   __ _
  / _ \ | '_ \| | '_ ` _ \ / _` |
 / ___ \| | | | | | | | | | (_| |
/_/   \_\_| |_|_|_| |_| |_|\__,_|
"#;

fn print_banner() {
    println!("{}", BANNER_ART);
    println!("Interactive REPL v{} (daemon mode) - Type '/help' for commands\n", env!("CARGO_PKG_VERSION"));
}

/// REPL helper for tab completion
struct ReplHelper {
    agent_names: Vec<String>,
}

impl ReplHelper {
    fn new() -> Self {
        ReplHelper {
            agent_names: Vec::new(),
        }
    }

    fn update_agents(&mut self, names: Vec<String>) {
        self.agent_names = names;
    }
}

impl Completer for ReplHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let mut completions = Vec::new();

        // Find the word being typed
        let word_start = line[..pos].rfind(' ').map(|i| i + 1).unwrap_or(0);
        let partial = &line[word_start..pos];

        // Slash commands (only complete at start of line or after /)
        if line.starts_with('/') {
            let cmd_partial = &line[1..pos]; // Skip the leading /
            let slash_commands = [
                "/start", "/load", "/stop", "/restart", "/create", "/status", "/list", "/clear",
                "/help", "/quit", "/exit"
            ];

            for cmd in &slash_commands {
                let without_slash = &cmd[1..]; // Compare without leading /
                if without_slash.starts_with(cmd_partial) {
                    completions.push(Pair {
                        display: cmd.to_string(),
                        replacement: cmd.to_string(),
                    });
                }
            }

            // Complete agent names after commands that use them
            if line.starts_with("/load ") || line.starts_with("/start ") ||
               line.starts_with("/stop ") || line.starts_with("/restart ") ||
               line.starts_with("/clear ") {
                for name in &self.agent_names {
                    if name.starts_with(partial) {
                        completions.push(Pair {
                            display: name.clone(),
                            replacement: name.clone(),
                        });
                    }
                }
            }
        } else {
            // @mentions for conversation
            if partial.starts_with('@') {
                let mention_partial = &partial[1..]; // Skip @

                // Add @all option
                if "all".starts_with(mention_partial) {
                    completions.push(Pair {
                        display: "@all".to_string(),
                        replacement: "@all".to_string(),
                    });
                }

                // Complete agent names
                for name in &self.agent_names {
                    if name.starts_with(mention_partial) {
                        completions.push(Pair {
                            display: format!("@{}", name),
                            replacement: format!("@{}", name),
                        });
                    }
                }
            }
        }

        Ok((word_start, completions))
    }
}

impl Hinter for ReplHelper {
    type Hint = String;

    fn hint(&self, _line: &str, _pos: usize, _ctx: &rustyline::Context<'_>) -> Option<String> {
        None
    }
}

impl Highlighter for ReplHelper {}
impl Validator for ReplHelper {}
impl Helper for ReplHelper {}

/// A single entry in the shared conversation log.
#[derive(Debug, Clone)]
struct ConversationEntry {
    /// Who sent the message: "user" or an agent name
    sender: String,
    /// The message content
    content: String,
}

/// Connection to a running agent daemon
#[derive(Debug, Clone)]
struct AgentConnection {
    /// Path to the Unix socket
    socket_path: PathBuf,
}

impl AgentConnection {
    fn from_running(agent: &RunningAgent) -> Self {
        Self {
            socket_path: agent.socket_path.clone(),
        }
    }

    /// Connect to the agent's daemon socket
    async fn connect(&self) -> Result<SocketApi, Box<dyn Error + Send + Sync>> {
        let stream = UnixStream::connect(&self.socket_path).await?;
        Ok(SocketApi::new(stream))
    }
}

pub struct Repl {
    /// Active connections to running daemons
    connections: HashMap<String, AgentConnection>,
    /// History file path
    history_file: Option<PathBuf>,
    /// Shared conversation log - all messages from user and agents
    conversation_log: Vec<ConversationEntry>,
    /// Per-agent cursor into conversation_log (last seen index)
    agent_cursors: HashMap<String, usize>,
}

impl Repl {
    pub fn new() -> Self {
        let history_file = dirs::home_dir().map(|h| h.join(".anima_history"));

        Repl {
            connections: HashMap::new(),
            history_file,
            conversation_log: Vec::new(),
            agent_cursors: HashMap::new(),
        }
    }

    /// Enable debug logging to ~/.anima/anima.log
    pub fn with_logging(self, enabled: bool) -> Self {
        if enabled {
            debug::enable();
            debug::log("=== REPL session started (daemon mode) ===");
        }
        self
    }

    /// Log a message to the conversation log.
    fn log_message(&mut self, sender: &str, content: &str) {
        self.conversation_log.push(ConversationEntry {
            sender: sender.to_string(),
            content: content.to_string(),
        });

        // Trim oldest entries if over limit
        if self.conversation_log.len() > MAX_CONVERSATION_LOG {
            let trim_count = self.conversation_log.len() - MAX_CONVERSATION_LOG;
            self.conversation_log.drain(0..trim_count);

            // Adjust all cursors (they now point to wrong indices)
            for cursor in self.agent_cursors.values_mut() {
                *cursor = cursor.saturating_sub(trim_count);
            }
            debug::log(&format!("LOG: trimmed {} entries, {} remain", trim_count, self.conversation_log.len()));
        }

        debug::log(&format!("LOG: {} entries, added from {}", self.conversation_log.len(), sender));
    }

    /// Get unseen conversation context for an agent and update their cursor.
    /// Returns formatted context string (each line is "sender: content").
    fn get_context_for_agent(&mut self, agent_name: &str) -> String {
        self.get_context_for_agent_up_to(agent_name, self.conversation_log.len(), true)
    }

    /// Get conversation context for an agent up to a specific log index.
    /// If update_cursor is true, updates the agent's cursor to the snapshot point.
    fn get_context_for_agent_up_to(&mut self, agent_name: &str, snapshot: usize, update_cursor: bool) -> String {
        let cursor = self.agent_cursors.get(agent_name).copied().unwrap_or(0);
        let end = snapshot.min(self.conversation_log.len());
        let unseen = &self.conversation_log[cursor..end];

        // Update cursor to snapshot point if requested
        if update_cursor {
            self.agent_cursors.insert(agent_name.to_string(), end);
        }

        // Format as multiline string
        unseen
            .iter()
            .map(|entry| {
                let escaped = entry.content
                    .replace('\\', "\\\\")
                    .replace('"', "\\\"")
                    .replace('\n', "\\n");
                format!("{{\"from\": \"{}\", \"text\": \"{}\"}}", entry.sender, escaped)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Create a REPL with a pre-loaded agent (ensures daemon is running and connects to it).
    pub async fn with_agent(name: String) -> Self {
        let history_file = dirs::home_dir().map(|h| h.join(".anima_history"));

        let mut repl = Repl {
            connections: HashMap::new(),
            history_file,
            conversation_log: Vec::new(),
            agent_cursors: HashMap::new(),
        };

        // Ensure the daemon is running and connect to it
        repl.ensure_daemon_running(&name).await;
        if let Some(agent) = discovery::get_running_agent(&name) {
            repl.connections.insert(name.clone(), AgentConnection::from_running(&agent));
        }

        repl
    }

    /// Ensure a daemon is running for the given agent name.
    /// Starts the daemon if it's not already running.
    async fn ensure_daemon_running(&self, name: &str) -> bool {
        if discovery::is_agent_running(name) {
            return true;
        }

        // Start the daemon in background
        println!("\x1b[33mStarting daemon for '{}'...\x1b[0m", name);

        // Get the path to the current executable
        let exe = match std::env::current_exe() {
            Ok(p) => p,
            Err(e) => {
                println!("\x1b[31mFailed to get executable path: {}\x1b[0m", e);
                return false;
            }
        };

        // Start daemon as background process
        match Command::new(&exe)
            .args(["start", name])
            .spawn()
        {
            Ok(_) => {
                // Wait a bit for daemon to start
                for _ in 0..20 {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    if discovery::is_agent_running(name) {
                        return true;
                    }
                }
                println!("\x1b[31mDaemon failed to start in time\x1b[0m");
                false
            }
            Err(e) => {
                println!("\x1b[31mFailed to start daemon: {}\x1b[0m", e);
                false
            }
        }
    }

    pub async fn run(&mut self) -> Result<(), Box<dyn Error>> {
        print_banner();

        let helper = ReplHelper::new();
        let mut rl: Editor<ReplHelper, _> = Editor::new()?;
        rl.set_helper(Some(helper));

        // Load history if available
        if let Some(ref path) = self.history_file {
            let _ = rl.load_history(path);
        }

        loop {
            // Update helper with current agent names (connected + running daemons)
            if let Some(h) = rl.helper_mut() {
                let mut names: Vec<String> = self.connections.keys().cloned().collect();
                // Also include discovered running agents not yet connected
                for agent in discovery::discover_running_agents() {
                    if !names.contains(&agent.name) {
                        names.push(agent.name);
                    }
                }
                h.update_agents(names);
            }

            let readline = rl.readline("\x1b[36manima>\x1b[0m ");
            match readline {
                Ok(line) => {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }

                    rl.add_history_entry(line)?;

                    if self.handle_input(line).await {
                        // handle_input returns true if we should exit
                        println!("\x1b[33mGoodbye!\x1b[0m");
                        break;
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    println!("\x1b[33m^C (use 'exit' or 'quit' to leave)\x1b[0m");
                    continue;
                }
                Err(ReadlineError::Eof) => {
                    println!("\x1b[33mGoodbye!\x1b[0m");
                    break;
                }
                Err(err) => {
                    eprintln!("\x1b[31mError: {:?}\x1b[0m", err);
                    break;
                }
            }
        }

        // Save history
        if let Some(ref path) = self.history_file {
            let _ = rl.save_history(path);
        }

        Ok(())
    }

    /// Handle input line. Returns true if we should exit the REPL.
    async fn handle_input(&mut self, input: &str) -> bool {
        debug::log(&format!("INPUT: {}", input));

        // Check for slash command prefix first
        if input.starts_with('/') {
            return self.handle_command(&input[1..]).await;
        }

        // Otherwise, treat as conversation with @mentions
        self.handle_conversation(input).await;
        false
    }

    /// Handle a slash command (input without the leading '/')
    /// Returns true if we should exit the REPL.
    async fn handle_command(&mut self, input: &str) -> bool {
        debug::log(&format!("CMD: /{}", input));

        // Parse command
        if input == "load" || input.starts_with("load ") {
            let name = if input == "load" { "" } else { &input[5..] };
            self.cmd_load(name).await;
        } else if input == "start" || input.starts_with("start ") {
            let name = if input == "start" { "" } else { &input[6..] };
            self.cmd_start(name).await;
        } else if input == "stop" || input.starts_with("stop ") {
            let name = if input == "stop" { "" } else { &input[5..] };
            self.cmd_stop(name).await;
        } else if input == "restart" || input.starts_with("restart ") {
            let name = if input == "restart" { "" } else { &input[8..] };
            self.cmd_restart(name).await;
        } else if input == "status" {
            self.cmd_status();
        } else if input == "list" {
            self.cmd_list();
        } else if input.starts_with("clear ") {
            self.cmd_clear(&input[6..]).await;
        } else if input == "clear" {
            // Clear with no args - if single connection, clear it
            if self.connections.len() == 1 {
                let name = self.connections.keys().next().unwrap().clone();
                self.cmd_clear(&name).await;
            } else {
                println!("\x1b[31mUsage: /clear <name>\x1b[0m");
            }
        } else if input == "create" || input.starts_with("create ") {
            let name = if input == "create" { "" } else { &input[7..] };
            self.cmd_create(name);
        } else if input == "help" {
            self.cmd_help();
        } else if input == "quit" || input == "exit" {
            return true;
        } else {
            println!("\x1b[31mUnknown command. Type '/help' for available commands.\x1b[0m");
        }
        false
    }

    /// Handle conversation input (non-command) with @mentions
    async fn handle_conversation(&mut self, input: &str) {
        let mentions = parse_mentions(input);

        // Check if this is an @all broadcast
        let is_broadcast = mentions.iter().any(|m| m == "all");

        // Determine which agents to send to
        let target_agents: Vec<String> = if is_broadcast {
            // @all means all connected agents
            self.connections.keys().cloned().collect()
        } else if !mentions.is_empty() {
            // Filter to only connected agent names (or auto-connect if running)
            let mut targets = Vec::new();
            for mention in &mentions {
                if self.connections.contains_key(mention) {
                    targets.push(mention.clone());
                } else if discovery::is_agent_running(mention) {
                    // Auto-connect to running agent
                    if let Some(agent) = discovery::get_running_agent(mention) {
                        self.connections.insert(mention.to_string(), AgentConnection::from_running(&agent));
                        println!("\x1b[32m✓ Connected to '{}'\x1b[0m", mention);
                        targets.push(mention.clone());
                    }
                } else {
                    println!("\x1b[31mAgent '{}' is not running\x1b[0m", mention);
                }
            }
            targets
        } else {
            // No mentions - use single agent if exactly one is connected
            if self.connections.len() == 1 {
                self.connections.keys().cloned().collect()
            } else if self.connections.is_empty() {
                println!("\x1b[31mNo agents connected. Use /start <name> to connect to an agent.\x1b[0m");
                return;
            } else {
                println!("\x1b[31mMultiple agents connected. Use @name to specify recipient.\x1b[0m");
                println!("\x1b[33mConnected agents: {}\x1b[0m", self.connections.keys().cloned().collect::<Vec<_>>().join(", "));
                return;
            }
        };

        if target_agents.is_empty() {
            println!("\x1b[31mNo valid agents to send to\x1b[0m");
            return;
        }

        // Log the user message to the conversation log
        self.log_message("user", input);

        // For @all broadcasts, use snapshot-based sending to ensure all agents
        // see the same context (no agent sees another's response from the same broadcast)
        if is_broadcast && target_agents.len() > 1 {
            self.send_broadcast_to_daemons(&target_agents).await;
        } else {
            // Send the message to each target agent normally
            for agent_name in target_agents {
                self.send_message_to_daemon(&agent_name).await;
            }
        }
    }

    /// Send a broadcast message to multiple daemons with snapshot isolation.
    /// All agents receive the same context snapshot - no agent sees another's
    /// response from the same broadcast.
    async fn send_broadcast_to_daemons(&mut self, agents: &[String]) {
        // Snapshot the current log position BEFORE sending to anyone
        let snapshot = self.conversation_log.len();
        debug::log(&format!("BROADCAST: snapshot at {} for {} agents", snapshot, agents.len()));

        // Collect contexts for all agents using the snapshot (don't update cursors yet)
        let mut contexts: HashMap<String, String> = HashMap::new();
        for agent_name in agents {
            let context = self.get_context_for_agent_up_to(agent_name, snapshot, false);
            if !context.is_empty() {
                contexts.insert(agent_name.clone(), context);
            }
        }

        // Send to all agents and collect responses (without logging yet)
        let mut responses: Vec<(String, String)> = Vec::new();
        for agent_name in agents {
            if let Some(context) = contexts.get(agent_name) {
                if let Some(response) = self.send_message_to_daemon_no_log(agent_name, context).await {
                    responses.push((agent_name.clone(), response));
                }
            }
        }

        // Now log all responses and update cursors
        for (agent_name, response) in &responses {
            self.log_message(agent_name, response);
        }

        // Update cursors to current log position (after all responses logged)
        let final_pos = self.conversation_log.len();
        for agent_name in agents {
            self.agent_cursors.insert(agent_name.clone(), final_pos);
        }

        // Handle @mention forwarding from responses
        for (agent_name, response) in responses {
            let mentions = parse_mentions(&response);
            for mention in mentions {
                // Never send back to sender
                if mention == agent_name {
                    continue;
                }

                // Handle @all in response
                if mention == "all" {
                    let others: Vec<String> = self.connections.keys()
                        .filter(|&name| name != &agent_name)
                        .cloned()
                        .collect();
                    for other in others {
                        debug::log(&format!("FORWARD (@all): {} (from {})", other, agent_name));
                        Box::pin(self.send_message_to_daemon_inner(&other, 1)).await;
                    }
                    continue;
                }

                // Auto-connect if needed
                if !self.connections.contains_key(&mention) {
                    if discovery::is_agent_running(&mention) {
                        if let Some(agent) = discovery::get_running_agent(&mention) {
                            self.connections.insert(mention.clone(), AgentConnection::from_running(&agent));
                            println!("\x1b[32m✓ Connected to '{}'\x1b[0m", mention);
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    }
                }

                debug::log(&format!("FORWARD: {} (from {})", mention, agent_name));
                Box::pin(self.send_message_to_daemon_inner(&mention, 1)).await;
            }
        }
    }

    /// Send a message to a daemon and return the response without logging it.
    /// Used for broadcast sends where we need to collect all responses before logging.
    async fn send_message_to_daemon_no_log(&self, agent_name: &str, context: &str) -> Option<String> {
        let connection = match self.connections.get(agent_name) {
            Some(c) => c.clone(),
            None => {
                println!("\x1b[31mAgent '{}' not connected\x1b[0m", agent_name);
                return None;
            }
        };

        print!("\x1b[33m[{}]\x1b[0m thinking...", agent_name);
        let _ = io::stdout().flush();
        let start = Instant::now();
        debug::log(&format!("BROADCAST CONV: {} <- context ({} chars)", agent_name, context.len()));

        // Connect to daemon
        let mut api = match connection.connect().await {
            Ok(api) => api,
            Err(e) => {
                println!("{}", format_elapsed(start.elapsed()));
                println!("\x1b[31mFailed to connect to '{}': {}\x1b[0m", agent_name, e);
                return None;
            }
        };

        // Send message request with conversation context
        let request = Request::Message { content: context.to_string(), conv_name: None };
        if let Err(e) = api.write_request(&request).await {
            println!("{}", format_elapsed(start.elapsed()));
            println!("\x1b[31mFailed to send message: {}\x1b[0m", e);
            return None;
        }

        // Read response
        match api.read_response().await {
            Ok(Some(Response::Message { content })) => {
                println!("{}", format_elapsed(start.elapsed()));
                let stripped = strip_thinking(&content);
                debug::log(&format!("BROADCAST RESPONSE (raw, {} chars): {}", content.len(),
                    if content.len() > 300 { format!("{}...", &content[..300]) } else { content.clone() }));
                debug::log(&format!("BROADCAST RESPONSE (stripped, {} chars): {}", stripped.len(),
                    if stripped.len() > 300 { format!("{}...", &stripped[..300]) } else { stripped.clone() }));

                println!("\x1b[33m[{}]\x1b[0m {}", agent_name, stripped);
                Some(stripped)
            }
            Ok(Some(Response::Error { message })) => {
                println!("\x1b[31m[{}] Error: {}\x1b[0m", agent_name, message);
                None
            }
            Ok(Some(_)) => {
                println!("\x1b[31mUnexpected response from '{}'\x1b[0m", agent_name);
                None
            }
            Ok(None) => {
                println!("\x1b[31mConnection closed by '{}'\x1b[0m", agent_name);
                None
            }
            Err(e) => {
                println!("\x1b[31mFailed to read response from '{}': {}\x1b[0m", agent_name, e);
                None
            }
        }
    }

    /// Send a conversation message to a daemon and display the response.
    /// Uses the shared conversation log to provide context.
    async fn send_message_to_daemon(&mut self, agent_name: &str) {
        self.send_message_to_daemon_inner(agent_name, 0).await;
    }

    /// Inner implementation that uses depth limit to prevent runaway loops.
    /// Gets unseen context from conversation_log and updates the agent's cursor.
    async fn send_message_to_daemon_inner(
        &mut self,
        agent_name: &str,
        depth: u32,
    ) {
        // Safety limit: stop after 15 hops to prevent runaway loops
        if depth > 15 {
            debug::log(&format!("DEPTH LIMIT: stopping at depth {} for {}", depth, agent_name));
            return;
        }
        let connection = match self.connections.get(agent_name) {
            Some(c) => c.clone(),
            None => {
                println!("\x1b[31mAgent '{}' not connected\x1b[0m", agent_name);
                return;
            }
        };

        // Get unseen conversation context for this agent
        let context = self.get_context_for_agent(agent_name);
        if context.is_empty() {
            debug::log(&format!("No new context for {}, skipping", agent_name));
            return;
        }

        print!("\x1b[33m[{}]\x1b[0m thinking...", agent_name);
        let _ = io::stdout().flush();
        let start = Instant::now();
        debug::log(&format!("CONV: {} <- context ({} chars)", agent_name, context.len()));

        // Connect to daemon
        let mut api = match connection.connect().await {
            Ok(api) => api,
            Err(e) => {
                println!("{}", format_elapsed(start.elapsed()));
                println!("\x1b[31mFailed to connect to '{}': {}\x1b[0m", agent_name, e);
                return;
            }
        };

        // Send message request with conversation context
        let request = Request::Message { content: context, conv_name: None };
        if let Err(e) = api.write_request(&request).await {
            println!("{}", format_elapsed(start.elapsed()));
            println!("\x1b[31mFailed to send message: {}\x1b[0m", e);
            return;
        }

        // Read response
        match api.read_response().await {
            Ok(Some(Response::Message { content })) => {
                println!("{}", format_elapsed(start.elapsed()));
                let stripped = strip_thinking(&content);
                debug::log(&format!("RESPONSE (raw, {} chars): {}", content.len(),
                    if content.len() > 300 { format!("{}...", &content[..300]) } else { content.clone() }));
                debug::log(&format!("RESPONSE (stripped, {} chars): {}", stripped.len(),
                    if stripped.len() > 300 { format!("{}...", &stripped[..300]) } else { stripped.clone() }));

                println!("\x1b[33m[{}]\x1b[0m {}", agent_name, stripped);

                // Log the agent's response to the conversation log
                self.log_message(agent_name, &stripped);

                // Check for @mentions in response and forward to mentioned agents
                let mentions = parse_mentions(&stripped);
                for mention in mentions {
                    // Never send back to sender (the only real rule)
                    if mention == agent_name {
                        continue;
                    }

                    // Handle @all: expand to all connected agents except sender
                    if mention == "all" {
                        let others: Vec<String> = self.connections.keys()
                            .filter(|&name| name != agent_name)
                            .cloned()
                            .collect();
                        for other in others {
                            debug::log(&format!("FORWARD (@all): {} (from {})", other, agent_name));
                            Box::pin(self.send_message_to_daemon_inner(&other, depth + 1)).await;
                        }
                        continue;
                    }

                    // Check if mentioned agent is connected or running
                    if !self.connections.contains_key(&mention) {
                        if discovery::is_agent_running(&mention) {
                            // Auto-connect to running agent
                            if let Some(agent) = discovery::get_running_agent(&mention) {
                                self.connections.insert(mention.clone(), AgentConnection::from_running(&agent));
                                println!("\x1b[32m✓ Connected to '{}'\x1b[0m", mention);
                            } else {
                                continue;
                            }
                        } else {
                            // Agent not running, skip
                            continue;
                        }
                    }

                    debug::log(&format!("FORWARD: {} (from {})", mention, agent_name));
                    Box::pin(self.send_message_to_daemon_inner(&mention, depth + 1)).await;
                }
            }
            Ok(Some(Response::Error { message })) => {
                println!("\x1b[31m[{}] Error: {}\x1b[0m", agent_name, message);
            }
            Ok(Some(_)) => {
                println!("\x1b[31mUnexpected response from '{}'\x1b[0m", agent_name);
            }
            Ok(None) => {
                println!("\x1b[31mConnection closed by '{}'\x1b[0m", agent_name);
            }
            Err(e) => {
                println!("\x1b[31mFailed to read response from '{}': {}\x1b[0m", agent_name, e);
            }
        }
    }

    /// Load/connect to an agent daemon. Starts the daemon if not running.
    async fn cmd_load(&mut self, name: &str) {
        let name = name.trim();
        if name.is_empty() {
            println!("\x1b[31mUsage: /load <name>\x1b[0m");
            return;
        }

        // Check if already connected
        if self.connections.contains_key(name) {
            println!("\x1b[33mAlready connected to '{}'\x1b[0m", name);
            return;
        }

        // Check if agent config exists
        let agent_exists = discovery::list_saved_agents().contains(&name.to_string());
        if !agent_exists {
            println!("\x1b[31mAgent '{}' not found in ~/.anima/agents/\x1b[0m", name);
            return;
        }

        // Ensure daemon is running
        if !self.ensure_daemon_running(name).await {
            println!("\x1b[31mFailed to start daemon for '{}'\x1b[0m", name);
            return;
        }

        // Connect
        if let Some(agent) = discovery::get_running_agent(name) {
            self.connections.insert(name.to_string(), AgentConnection::from_running(&agent));
            println!("\x1b[32m✓ Connected to '{}'\x1b[0m", name);
        } else {
            println!("\x1b[31mFailed to connect to '{}'\x1b[0m", name);
        }
    }

    /// Start an agent daemon (alias for load).
    async fn cmd_start(&mut self, name: &str) {
        let name = name.trim();
        if name.is_empty() {
            println!("\x1b[31mUsage: /start <name>\x1b[0m");
            return;
        }

        // Check if agent config exists
        let agent_exists = discovery::list_saved_agents().contains(&name.to_string());
        if !agent_exists {
            println!("\x1b[31mAgent '{}' not found in ~/.anima/agents/\x1b[0m", name);
            return;
        }

        // Check if already running
        if discovery::is_agent_running(name) {
            // Just connect if not already
            if !self.connections.contains_key(name) {
                if let Some(agent) = discovery::get_running_agent(name) {
                    self.connections.insert(name.to_string(), AgentConnection::from_running(&agent));
                    println!("\x1b[32m✓ Connected to already-running '{}'\x1b[0m", name);
                    return;
                }
            }
            println!("\x1b[33m'{}' is already running\x1b[0m", name);
            return;
        }

        // Start and connect
        self.cmd_load(name).await;
    }

    /// Restart an agent daemon (stop then start), reconnecting afterward.
    async fn cmd_restart(&mut self, name: &str) {
        let name = name.trim();
        if name.is_empty() {
            println!("\x1b[31mUsage: /restart <name>\x1b[0m");
            return;
        }

        // Check if agent config exists
        let agent_exists = discovery::list_saved_agents().contains(&name.to_string());
        if !agent_exists {
            println!("\x1b[31mAgent '{}' not found in ~/.anima/agents/\x1b[0m", name);
            return;
        }

        // Stop if running
        if discovery::is_agent_running(name) {
            println!("Stopping {}...", name);

            // Send shutdown request
            if let Some(socket_path) = discovery::agent_socket_path(name) {
                if let Ok(stream) = UnixStream::connect(&socket_path).await {
                    let mut api = SocketApi::new(stream);
                    let _ = api.write_request(&Request::Shutdown).await;
                    // Wait for response (brief)
                    let _ = api.read_response().await;
                }
            }

            // Remove from connections
            self.connections.remove(name);

            // Wait for daemon to stop
            for _ in 0..20 {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                if !discovery::is_agent_running(name) {
                    break;
                }
            }
        }

        // Brief wait to ensure clean shutdown
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        // Start and connect
        println!("Starting {}...", name);
        if !self.ensure_daemon_running(name).await {
            println!("\x1b[31mFailed to start daemon for '{}'\x1b[0m", name);
            return;
        }

        // Connect
        if let Some(agent) = discovery::get_running_agent(name) {
            self.connections.insert(name.to_string(), AgentConnection::from_running(&agent));
            println!("\x1b[32m✓ Connected to '{}'\x1b[0m", name);
        } else {
            println!("\x1b[31mFailed to connect to '{}'\x1b[0m", name);
        }
    }

    /// Stop an agent daemon.
    async fn cmd_stop(&mut self, name: &str) {
        let name = name.trim();
        if name.is_empty() {
            println!("\x1b[31mUsage: /stop <name>\x1b[0m");
            return;
        }

        // Check if running
        if !discovery::is_agent_running(name) {
            println!("\x1b[31m'{}' is not running\x1b[0m", name);
            return;
        }

        // Get connection (or create temporary one)
        let socket_path = match discovery::agent_socket_path(name) {
            Some(p) => p,
            None => {
                println!("\x1b[31mFailed to determine socket path for '{}'\x1b[0m", name);
                return;
            }
        };

        // Send shutdown request
        match UnixStream::connect(&socket_path).await {
            Ok(stream) => {
                let mut api = SocketApi::new(stream);
                if let Err(e) = api.write_request(&Request::Shutdown).await {
                    println!("\x1b[31mFailed to send shutdown: {}\x1b[0m", e);
                    return;
                }
                // Wait for response
                match api.read_response().await {
                    Ok(Some(Response::Ok)) => {
                        println!("\x1b[32m✓ Stopped '{}'\x1b[0m", name);
                    }
                    _ => {
                        println!("\x1b[33mShutdown sent to '{}'\x1b[0m", name);
                    }
                }
            }
            Err(e) => {
                println!("\x1b[31mFailed to connect to '{}': {}\x1b[0m", name, e);
            }
        }

        // Remove from connections
        self.connections.remove(name);
    }

    /// Show status of all agents.
    fn cmd_status(&self) {
        let running = discovery::discover_running_agents();
        let saved = discovery::list_saved_agents();

        if saved.is_empty() && running.is_empty() {
            println!("\x1b[33mNo agents found in ~/.anima/agents/\x1b[0m");
            return;
        }

        println!("\x1b[1mAgent Status:\x1b[0m");

        for name in &saved {
            let is_running = running.iter().any(|a| &a.name == name);
            let is_connected = self.connections.contains_key(name);

            let status = if is_running && is_connected {
                "\x1b[32m(running, connected)\x1b[0m"
            } else if is_running {
                "\x1b[32m(running)\x1b[0m"
            } else {
                "\x1b[33m(stopped)\x1b[0m"
            };

            println!("  \x1b[36m{}\x1b[0m {}", name, status);
        }

        // Show any running agents not in saved (shouldn't happen normally)
        for agent in &running {
            if !saved.contains(&agent.name) {
                let status = if self.connections.contains_key(&agent.name) {
                    "\x1b[32m(running, connected)\x1b[0m"
                } else {
                    "\x1b[32m(running)\x1b[0m"
                };
                println!("  \x1b[36m{}\x1b[0m {} (orphan?)", agent.name, status);
            }
        }
    }

    /// List connected agents.
    fn cmd_list(&self) {
        let running = discovery::discover_running_agents();

        if self.connections.is_empty() && running.is_empty() {
            println!("\x1b[33mNo agents connected or running. Use /start <name> to connect.\x1b[0m");
            return;
        }

        println!("\x1b[1mConnected:\x1b[0m");
        if self.connections.is_empty() {
            println!("  (none)");
        } else {
            for name in self.connections.keys() {
                println!("  \x1b[36m{}\x1b[0m", name);
            }
        }

        // Show running but not connected
        let not_connected: Vec<_> = running.iter()
            .filter(|a| !self.connections.contains_key(&a.name))
            .collect();

        if !not_connected.is_empty() {
            println!("\n\x1b[1mRunning (not connected):\x1b[0m");
            for agent in not_connected {
                println!("  \x1b[36m{}\x1b[0m", agent.name);
            }
        }
    }

    /// Clear conversation history for an agent.
    async fn cmd_clear(&self, name: &str) {
        let name = name.trim();

        let connection = match self.connections.get(name) {
            Some(c) => c,
            None => {
                // Try to connect if running
                if discovery::is_agent_running(name) {
                    if let Some(socket_path) = discovery::agent_socket_path(name) {
                        match UnixStream::connect(&socket_path).await {
                            Ok(stream) => {
                                let mut api = SocketApi::new(stream);
                                if let Err(e) = api.write_request(&Request::Clear).await {
                                    println!("\x1b[31mFailed to send clear: {}\x1b[0m", e);
                                    return;
                                }
                                match api.read_response().await {
                                    Ok(Some(Response::Ok)) => {
                                        println!("\x1b[32m✓ Cleared history for '{}'\x1b[0m", name);
                                    }
                                    _ => {
                                        println!("\x1b[31mFailed to clear history for '{}'\x1b[0m", name);
                                    }
                                }
                                return;
                            }
                            Err(e) => {
                                println!("\x1b[31mFailed to connect to '{}': {}\x1b[0m", name, e);
                                return;
                            }
                        }
                    }
                }
                println!("\x1b[31mAgent '{}' not connected\x1b[0m", name);
                return;
            }
        };

        // Send clear request
        let mut api = match connection.connect().await {
            Ok(api) => api,
            Err(e) => {
                println!("\x1b[31mFailed to connect to '{}': {}\x1b[0m", name, e);
                return;
            }
        };

        if let Err(e) = api.write_request(&Request::Clear).await {
            println!("\x1b[31mFailed to send clear: {}\x1b[0m", e);
            return;
        }

        match api.read_response().await {
            Ok(Some(Response::Ok)) => {
                println!("\x1b[32m✓ Cleared history for '{}'\x1b[0m", name);
            }
            _ => {
                println!("\x1b[31mFailed to clear history for '{}'\x1b[0m", name);
            }
        }
    }

    /// Create a new agent directory.
    fn cmd_create(&self, name: &str) {
        let name = name.trim();
        if name.is_empty() {
            println!("\x1b[31mUsage: /create <name>\x1b[0m");
            return;
        }

        match crate::agent_dir::create_agent(name, None) {
            Ok(()) => {}
            Err(e) => {
                println!("\x1b[31mFailed to create agent: {}\x1b[0m", e);
            }
        }
    }

    fn cmd_help(&self) {
        println!("\x1b[1mAnima REPL (Daemon Mode) - Slash Commands:\x1b[0m");
        println!();
        println!("  \x1b[36m/start <name>\x1b[0m");
        println!("      Start daemon (if needed) and connect to agent");
        println!();
        println!("  \x1b[36m/load <name>\x1b[0m");
        println!("      Alias for /start");
        println!();
        println!("  \x1b[36m/create <name>\x1b[0m");
        println!("      Create a new agent directory in ~/.anima/agents/");
        println!();
        println!("  \x1b[36m/stop <name>\x1b[0m");
        println!("      Stop an agent daemon");
        println!();
        println!("  \x1b[36m/restart <name>\x1b[0m");
        println!("      Restart an agent daemon (stop then start)");
        println!();
        println!("  \x1b[36m/status\x1b[0m");
        println!("      Show status of all agents");
        println!();
        println!("  \x1b[36m/list\x1b[0m");
        println!("      List connected and running agents");
        println!();
        println!("  \x1b[36m/clear [name]\x1b[0m");
        println!("      Clear conversation history");
        println!();
        println!("  \x1b[36m/help\x1b[0m");
        println!("      Show this help message");
        println!();
        println!("  \x1b[36m/quit\x1b[0m, \x1b[36m/exit\x1b[0m");
        println!("      Exit the REPL");
        println!();
        println!("\x1b[1mConversation (no slash):\x1b[0m");
        println!();
        println!("  \x1b[36m@name message\x1b[0m      Send message to specific agent");
        println!("  \x1b[36m@all message\x1b[0m       Send to all connected agents");
        println!("  \x1b[36mmessage\x1b[0m            Send to single connected agent (if only one)");
        println!();
        println!("\x1b[1mAgent Management:\x1b[0m");
        println!();
        println!("  Agents are stored in ~/.anima/agents/<name>/");
        println!("  Each agent runs as a separate daemon process.");
        println!("  Use 'anima start <name>' from command line to start daemons.");
    }
}

impl Default for Repl {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repl_helper_new() {
        let helper = ReplHelper::new();
        assert!(helper.agent_names.is_empty());
    }

    #[test]
    fn test_repl_helper_update_agents() {
        let mut helper = ReplHelper::new();
        helper.update_agents(vec!["alice".to_string(), "bob".to_string()]);
        assert_eq!(helper.agent_names.len(), 2);
        assert!(helper.agent_names.contains(&"alice".to_string()));
        assert!(helper.agent_names.contains(&"bob".to_string()));
    }

    #[test]
    fn test_repl_new() {
        let repl = Repl::new();
        assert!(repl.connections.is_empty());
    }

    #[test]
    fn test_repl_default() {
        let repl = Repl::default();
        assert!(repl.connections.is_empty());
    }

    #[test]
    fn test_cmd_help() {
        let repl = Repl::new();
        // Just verify it doesn't panic
        repl.cmd_help();
    }

    #[test]
    fn test_cmd_status_empty() {
        let repl = Repl::new();
        // Just verify it doesn't panic with no agents
        repl.cmd_status();
    }

    #[test]
    fn test_cmd_list_empty() {
        let repl = Repl::new();
        // Just verify it doesn't panic
        repl.cmd_list();
    }

    #[test]
    fn test_parse_mentions_single() {
        let mentions = parse_mentions("hello @arya how are you?");
        assert_eq!(mentions, vec!["arya"]);
    }

    #[test]
    fn test_parse_mentions_multiple() {
        let mentions = parse_mentions("@arya and @gendry let's chat");
        assert_eq!(mentions, vec!["arya", "gendry"]);
    }

    #[test]
    fn test_parse_mentions_all() {
        let mentions = parse_mentions("@all hello everyone");
        assert_eq!(mentions, vec!["all"]);
    }

    #[test]
    fn test_parse_mentions_none() {
        let mentions = parse_mentions("hello world");
        assert!(mentions.is_empty());
    }

    #[test]
    fn test_parse_mentions_with_hyphen_underscore() {
        let mentions = parse_mentions("@my-agent and @another_agent");
        assert_eq!(mentions, vec!["my-agent", "another_agent"]);
    }

    #[test]
    fn test_parse_mentions_must_start_with_letter() {
        // @123 should not be matched
        let mentions = parse_mentions("@123 is not valid");
        assert!(mentions.is_empty());
    }

    #[test]
    fn test_parse_mentions_at_start() {
        let mentions = parse_mentions("@arya hello");
        assert_eq!(mentions, vec!["arya"]);
    }

    #[test]
    fn test_parse_mentions_at_end() {
        let mentions = parse_mentions("hello @arya");
        assert_eq!(mentions, vec!["arya"]);
    }

    #[tokio::test]
    async fn test_handle_command_quit() {
        let mut repl = Repl::new();
        // /quit should return true (exit)
        assert!(repl.handle_command("quit").await);
        assert!(repl.handle_command("exit").await);
    }

    #[tokio::test]
    async fn test_handle_command_help() {
        let mut repl = Repl::new();
        // /help should not exit
        assert!(!repl.handle_command("help").await);
    }

    #[tokio::test]
    async fn test_handle_command_status() {
        let mut repl = Repl::new();
        // /status should not exit
        assert!(!repl.handle_command("status").await);
    }

    #[tokio::test]
    async fn test_handle_input_slash_command() {
        let mut repl = Repl::new();
        // /help is a command, should not exit
        assert!(!repl.handle_input("/help").await);
        // /quit should exit
        assert!(repl.handle_input("/quit").await);
    }

    #[tokio::test]
    async fn test_handle_conversation_no_agents() {
        let mut repl = Repl::new();
        // No agents connected, should print error but not panic
        repl.handle_conversation("hello").await;
    }

    #[tokio::test]
    async fn test_handle_conversation_unknown_mention() {
        let mut repl = Repl::new();
        // Mention unknown agent, should print error but not panic
        repl.handle_conversation("@unknown hello").await;
    }

    #[test]
    fn test_agent_connection_clone() {
        let conn = AgentConnection {
            socket_path: PathBuf::from("/tmp/test.sock"),
        };
        let cloned = conn.clone();
        assert_eq!(conn.socket_path, cloned.socket_path);
    }

    #[test]
    fn test_conversation_entry_clone() {
        let entry = ConversationEntry {
            sender: "user".to_string(),
            content: "hello".to_string(),
        };
        let cloned = entry.clone();
        assert_eq!(entry.sender, cloned.sender);
        assert_eq!(entry.content, cloned.content);
    }

    #[test]
    fn test_repl_new_initializes_conversation_log() {
        let repl = Repl::new();
        assert!(repl.conversation_log.is_empty());
        assert!(repl.agent_cursors.is_empty());
    }

    #[test]
    fn test_log_message() {
        let mut repl = Repl::new();
        repl.log_message("user", "hello @arya");
        repl.log_message("arya", "hey there!");

        assert_eq!(repl.conversation_log.len(), 2);
        assert_eq!(repl.conversation_log[0].sender, "user");
        assert_eq!(repl.conversation_log[0].content, "hello @arya");
        assert_eq!(repl.conversation_log[1].sender, "arya");
        assert_eq!(repl.conversation_log[1].content, "hey there!");
    }

    #[test]
    fn test_get_context_for_agent_first_time() {
        let mut repl = Repl::new();
        repl.log_message("user", "@arya hello");
        repl.log_message("arya", "hey there!");

        // First time getting context for gendry - should get full log
        let context = repl.get_context_for_agent("gendry");
        assert_eq!(context, "{\"from\": \"user\", \"text\": \"@arya hello\"}\n{\"from\": \"arya\", \"text\": \"hey there!\"}");
        assert_eq!(repl.agent_cursors.get("gendry"), Some(&2));
    }

    #[test]
    fn test_get_context_for_agent_incremental() {
        let mut repl = Repl::new();
        repl.log_message("user", "@arya hello");

        // arya gets the first message
        let context = repl.get_context_for_agent("arya");
        assert_eq!(context, "{\"from\": \"user\", \"text\": \"@arya hello\"}");
        assert_eq!(repl.agent_cursors.get("arya"), Some(&1));

        // Add more messages
        repl.log_message("arya", "hey there!");
        repl.log_message("user", "@arya ask gendry");

        // arya should only see the new messages
        let context = repl.get_context_for_agent("arya");
        assert_eq!(context, "{\"from\": \"arya\", \"text\": \"hey there!\"}\n{\"from\": \"user\", \"text\": \"@arya ask gendry\"}");
        assert_eq!(repl.agent_cursors.get("arya"), Some(&3));
    }

    #[test]
    fn test_get_context_for_agent_empty_when_up_to_date() {
        let mut repl = Repl::new();
        repl.log_message("user", "@arya hello");

        // arya gets the message
        let _ = repl.get_context_for_agent("arya");

        // No new messages, context should be empty
        let context = repl.get_context_for_agent("arya");
        assert!(context.is_empty());
    }

    #[test]
    fn test_get_context_full_flow() {
        let mut repl = Repl::new();

        // User: @arya hello
        repl.log_message("user", "@arya hello");

        // arya receives JSON formatted context
        let context = repl.get_context_for_agent("arya");
        assert_eq!(context, "{\"from\": \"user\", \"text\": \"@arya hello\"}");

        // arya responds: "hey there!"
        repl.log_message("arya", "hey there!");

        // User: @arya ask gendry
        repl.log_message("user", "@arya ask gendry");

        // arya receives only new messages
        let context = repl.get_context_for_agent("arya");
        assert_eq!(context, "{\"from\": \"arya\", \"text\": \"hey there!\"}\n{\"from\": \"user\", \"text\": \"@arya ask gendry\"}");

        // arya responds: "@gendry what do you think?"
        repl.log_message("arya", "@gendry what do you think?");

        // gendry has never received anything, gets full log
        let context = repl.get_context_for_agent("gendry");
        assert_eq!(
            context,
            "{\"from\": \"user\", \"text\": \"@arya hello\"}\n{\"from\": \"arya\", \"text\": \"hey there!\"}\n{\"from\": \"user\", \"text\": \"@arya ask gendry\"}\n{\"from\": \"arya\", \"text\": \"@gendry what do you think?\"}"
        );
    }

    #[test]
    fn test_get_context_for_agent_up_to_snapshot() {
        let mut repl = Repl::new();

        // Build up conversation
        repl.log_message("user", "@all hello everyone");
        let snapshot = repl.conversation_log.len(); // snapshot at 1

        // Get context for arya with snapshot, don't update cursor
        let arya_context = repl.get_context_for_agent_up_to("arya", snapshot, false);
        assert_eq!(arya_context, "{\"from\": \"user\", \"text\": \"@all hello everyone\"}");
        // Cursor should NOT be updated
        assert_eq!(repl.agent_cursors.get("arya"), None);

        // Get context for gendry with snapshot, don't update cursor
        let gendry_context = repl.get_context_for_agent_up_to("gendry", snapshot, false);
        assert_eq!(gendry_context, "{\"from\": \"user\", \"text\": \"@all hello everyone\"}");
        // Cursor should NOT be updated
        assert_eq!(repl.agent_cursors.get("gendry"), None);

        // Both agents should see exactly the same context
        assert_eq!(arya_context, gendry_context);
    }

    #[test]
    fn test_get_context_for_agent_up_to_with_cursor_update() {
        let mut repl = Repl::new();

        repl.log_message("user", "@all hello everyone");
        let snapshot = repl.conversation_log.len();

        // Get context with cursor update
        let context = repl.get_context_for_agent_up_to("arya", snapshot, true);
        assert_eq!(context, "{\"from\": \"user\", \"text\": \"@all hello everyone\"}");
        // Cursor SHOULD be updated
        assert_eq!(repl.agent_cursors.get("arya"), Some(&1));
    }

    #[test]
    fn test_broadcast_snapshot_isolation() {
        // Simulates @all broadcast behavior: all agents should see same context
        let mut repl = Repl::new();

        // User sends @all message
        repl.log_message("user", "@all what do you think?");
        let snapshot = repl.conversation_log.len();

        // Snapshot contexts for all agents BEFORE any responses
        let arya_context = repl.get_context_for_agent_up_to("arya", snapshot, false);
        let gendry_context = repl.get_context_for_agent_up_to("gendry", snapshot, false);

        // Simulate arya responding (this would normally happen via daemon)
        repl.log_message("arya", "I think it's great!");

        // Even after arya's response is logged, gendry's snapshot context
        // should NOT include arya's response
        assert_eq!(arya_context, "{\"from\": \"user\", \"text\": \"@all what do you think?\"}");
        assert_eq!(gendry_context, "{\"from\": \"user\", \"text\": \"@all what do you think?\"}");

        // Both got exactly the same context
        assert_eq!(arya_context, gendry_context);

        // Now if we get fresh context for gendry (without snapshot), it would include arya's response
        let gendry_fresh = repl.get_context_for_agent("gendry");
        assert_eq!(gendry_fresh, "{\"from\": \"user\", \"text\": \"@all what do you think?\"}\n{\"from\": \"arya\", \"text\": \"I think it's great!\"}");
    }

    #[test]
    fn test_conversation_log_trimming() {
        let mut repl = Repl::new();

        // Add MAX_CONVERSATION_LOG + 5 messages
        for i in 0..(MAX_CONVERSATION_LOG + 5) {
            repl.log_message("user", &format!("message {}", i));
        }

        // Should be trimmed to MAX_CONVERSATION_LOG
        assert_eq!(repl.conversation_log.len(), MAX_CONVERSATION_LOG);

        // First message should now be "message 5" (oldest 5 were trimmed)
        assert_eq!(repl.conversation_log[0].content, "message 5");

        // Last message should be the most recent
        assert_eq!(
            repl.conversation_log[MAX_CONVERSATION_LOG - 1].content,
            format!("message {}", MAX_CONVERSATION_LOG + 4)
        );
    }

    #[test]
    fn test_conversation_log_trimming_adjusts_cursors() {
        let mut repl = Repl::new();

        // Add some messages
        for i in 0..50 {
            repl.log_message("user", &format!("message {}", i));
        }

        // Set cursor for arya at position 30
        repl.agent_cursors.insert("arya".to_string(), 30);
        // Set cursor for gendry at position 10
        repl.agent_cursors.insert("gendry".to_string(), 10);

        // Add more messages to trigger trimming (need to exceed MAX_CONVERSATION_LOG)
        for i in 50..(MAX_CONVERSATION_LOG + 10) {
            repl.log_message("user", &format!("message {}", i));
        }

        // Log should be at MAX_CONVERSATION_LOG
        assert_eq!(repl.conversation_log.len(), MAX_CONVERSATION_LOG);

        // Cursors should be adjusted down by trim_count (10)
        // arya was at 30, now at 20
        assert_eq!(repl.agent_cursors.get("arya"), Some(&20));
        // gendry was at 10, now at 0 (saturating_sub prevents underflow)
        assert_eq!(repl.agent_cursors.get("gendry"), Some(&0));
    }
}
