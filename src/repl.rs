//! REPL for interacting with agent daemons.
//!
//! In the daemon architecture, agents always run as separate processes.
//! The REPL connects to running daemons via Unix sockets.

use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;
use std::process::Command;

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
/// Returns a Vec of mentioned agent names (without the @ prefix).
/// Special case: "@all" is returned as-is.
fn parse_mentions(input: &str) -> Vec<String> {
    let re = regex::Regex::new(r"@([a-zA-Z][a-zA-Z0-9_-]*)").unwrap();
    re.captures_iter(input)
        .map(|cap| cap[1].to_string())
        .collect()
}

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
}

impl Repl {
    pub fn new() -> Self {
        let history_file = dirs::home_dir().map(|h| h.join(".anima_history"));

        Repl {
            connections: HashMap::new(),
            history_file,
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

    /// Create a REPL with a pre-loaded agent (ensures daemon is running and connects to it).
    pub async fn with_agent(name: String) -> Self {
        let history_file = dirs::home_dir().map(|h| h.join(".anima_history"));

        let mut repl = Repl {
            connections: HashMap::new(),
            history_file,
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

        // Determine which agents to send to
        let target_agents: Vec<String> = if mentions.iter().any(|m| m == "all") {
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

        // Send the message to each target agent
        for agent_name in target_agents {
            self.send_message_to_daemon(&agent_name, input).await;
        }
    }

    /// Send a conversation message to a daemon and display the response.
    async fn send_message_to_daemon(&mut self, agent_name: &str, message: &str) {
        self.send_message_to_daemon_inner(agent_name, message, 0).await;
    }

    /// Inner implementation that uses depth limit to prevent runaway loops.
    async fn send_message_to_daemon_inner(
        &mut self,
        agent_name: &str,
        message: &str,
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

        // Format the message with [user] prefix for the agent
        let formatted_message = format!("[user] {}", message);

        println!("\x1b[33m[{}]\x1b[0m thinking...", agent_name);
        debug::log(&format!("CONV: {} -> {}", agent_name, message));

        // Connect to daemon
        let mut api = match connection.connect().await {
            Ok(api) => api,
            Err(e) => {
                println!("\x1b[31mFailed to connect to '{}': {}\x1b[0m", agent_name, e);
                return;
            }
        };

        // Send message request
        let request = Request::Message { content: formatted_message };
        if let Err(e) = api.write_request(&request).await {
            println!("\x1b[31mFailed to send message: {}\x1b[0m", e);
            return;
        }

        // Read response
        match api.read_response().await {
            Ok(Some(Response::Message { content })) => {
                let stripped = strip_thinking(&content);
                debug::log(&format!("RESPONSE (raw, {} chars): {}", content.len(),
                    if content.len() > 300 { format!("{}...", &content[..300]) } else { content.clone() }));
                debug::log(&format!("RESPONSE (stripped, {} chars): {}", stripped.len(),
                    if stripped.len() > 300 { format!("{}...", &stripped[..300]) } else { stripped.clone() }));

                println!("\x1b[33m[{}]\x1b[0m {}", agent_name, stripped);

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
                            let forwarded_message = format!("{}: {}", agent_name, stripped);
                            debug::log(&format!("FORWARD (@all): {} -> {} (from {})", other, forwarded_message, agent_name));
                            Box::pin(self.send_message_to_daemon_inner(&other, &forwarded_message, depth + 1)).await;
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

                    // Forward message with sender context
                    let forwarded_message = format!("{}: {}", agent_name, stripped);
                    debug::log(&format!("FORWARD: {} -> {} (from {})", mention, forwarded_message, agent_name));

                    Box::pin(self.send_message_to_daemon_inner(&mention, &forwarded_message, depth + 1)).await;
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
}
