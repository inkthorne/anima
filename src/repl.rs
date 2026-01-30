use std::collections::HashMap;
use std::error::Error;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;

use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::{Editor, Helper};

use crate::llm::{LLM, OpenAIClient, AnthropicClient, OllamaClient};
use crate::memory::{Memory, SqliteMemory, InMemoryStore};
use crate::observe::ConsoleObserver;
use crate::runtime::Runtime;
use crate::agent::{Agent, ThinkOptions};
use crate::tools::{AddTool, EchoTool, ReadFileTool, WriteFileTool, HttpTool, ShellTool, SendMessageTool, ListAgentsTool};

const BANNER: &str = r#"
    _          _
   / \   _ __ (_)_ __ ___   __ _
  / _ \ | '_ \| | '_ ` _ \ / _` |
 / ___ \| | | | | | | | | | (_| |
/_/   \_\_| |_|_|_| |_| |_|\__,_|

Interactive REPL v1.9 - Type 'help' for commands
"#;

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

        // Commands that can be completed
        let commands = ["agent create", "agent list", "agent remove", "memory", "set llm", "help", "exit", "quit"];

        // Find the word being typed
        let word_start = line[..pos].rfind(' ').map(|i| i + 1).unwrap_or(0);
        let partial = &line[word_start..pos];

        // Complete commands
        for cmd in &commands {
            if cmd.starts_with(partial) {
                completions.push(Pair {
                    display: cmd.to_string(),
                    replacement: cmd.to_string(),
                });
            }
        }

        // Complete agent names for commands that use them
        if line.starts_with("memory ") || line.starts_with("agent remove ") || line.contains(": ") {
            for name in &self.agent_names {
                if name.starts_with(partial) {
                    completions.push(Pair {
                        display: name.clone(),
                        replacement: name.clone(),
                    });
                }
            }
        }

        // For "<agent>:" syntax
        if !partial.is_empty() && !partial.contains(':') {
            for name in &self.agent_names {
                let with_colon = format!("{}: ", name);
                if with_colon.starts_with(partial) {
                    completions.push(Pair {
                        display: with_colon.clone(),
                        replacement: with_colon,
                    });
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

/// Agent entry stored in the REPL
struct ReplAgent {
    agent: Agent,
    llm_name: String,
}

pub struct Repl {
    runtime: Runtime,
    agents: HashMap<String, ReplAgent>,
    default_llm: Option<String>,
    history_file: Option<PathBuf>,
}

impl Repl {
    pub fn new() -> Self {
        // Set up history file in home directory
        let history_file = dirs::home_dir().map(|h| h.join(".anima_history"));

        Repl {
            runtime: Runtime::new(),
            agents: HashMap::new(),
            default_llm: None,
            history_file,
        }
    }

    pub async fn run(&mut self) -> Result<(), Box<dyn Error>> {
        println!("{}", BANNER);

        let helper = ReplHelper::new();
        let mut rl: Editor<ReplHelper, _> = Editor::new()?;
        rl.set_helper(Some(helper));

        // Load history if available
        if let Some(ref path) = self.history_file {
            let _ = rl.load_history(path);
        }

        loop {
            // Update helper with current agent names
            if let Some(h) = rl.helper_mut() {
                h.update_agents(self.agents.keys().cloned().collect());
            }

            let readline = rl.readline("\x1b[36manima>\x1b[0m ");
            match readline {
                Ok(line) => {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }

                    rl.add_history_entry(line)?;

                    if line == "exit" || line == "quit" {
                        println!("\x1b[33mGoodbye!\x1b[0m");
                        break;
                    }

                    self.handle_command(line).await;
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

    async fn handle_command(&mut self, input: &str) {
        // Parse command
        if input.starts_with("agent create ") {
            self.cmd_agent_create(&input[13..]).await;
        } else if input == "agent list" {
            self.cmd_agent_list();
        } else if input == "agent list-saved" {
            self.cmd_agent_list_saved();
        } else if input.starts_with("agent remove ") {
            self.cmd_agent_remove(&input[13..]).await;
        } else if input.starts_with("memory ") {
            self.cmd_memory(&input[7..]).await;
        } else if input.starts_with("set llm ") {
            self.cmd_set_llm(&input[8..]);
        } else if input == "help" {
            self.cmd_help();
        } else if let Some(pos) = input.find(": ") {
            // <agent>: <task> syntax
            let agent_name = &input[..pos];
            let task = &input[pos + 2..];

            // Check for "ask" syntax: <agent>: ask <other> "<question>"
            if task.starts_with("ask ") {
                self.cmd_ask(agent_name, &task[4..]).await;
            } else {
                self.cmd_task(agent_name, task).await;
            }
        } else {
            println!("\x1b[31mUnknown command. Type 'help' for available commands.\x1b[0m");
        }
    }

    async fn cmd_agent_create(&mut self, args: &str) {
        // Parse: <name> [--llm <provider/model>]
        let parts: Vec<&str> = args.split_whitespace().collect();
        if parts.is_empty() {
            println!("\x1b[31mUsage: agent create <name> [--llm <provider/model>]\x1b[0m");
            return;
        }

        let name = parts[0].to_string();

        // Check for duplicate
        if self.agents.contains_key(&name) {
            println!("\x1b[31mAgent '{}' already exists\x1b[0m", name);
            return;
        }

        // Parse --llm flag
        let llm_spec = if parts.len() >= 3 && parts[1] == "--llm" {
            Some(parts[2].to_string())
        } else {
            self.default_llm.clone()
        };

        let llm_name = llm_spec.clone().unwrap_or_else(|| "none".to_string());

        // Create the LLM client if specified
        let llm: Option<Arc<dyn LLM>> = if let Some(spec) = &llm_spec {
            match self.create_llm_from_spec(spec) {
                Ok(l) => Some(l),
                Err(e) => {
                    println!("\x1b[31mFailed to create LLM: {}\x1b[0m", e);
                    return;
                }
            }
        } else {
            None
        };

        // Create agent with runtime (registers with message router)
        let mut agent = self.runtime.spawn_agent(name.clone()).await;

        // Register default tools
        agent.register_tool(Arc::new(AddTool));
        agent.register_tool(Arc::new(EchoTool));
        agent.register_tool(Arc::new(ReadFileTool));
        agent.register_tool(Arc::new(WriteFileTool));
        agent.register_tool(Arc::new(HttpTool::new()));
        agent.register_tool(Arc::new(ShellTool::new()));

        // Register messaging tools
        let router = self.runtime.router().clone();
        agent.register_tool(Arc::new(SendMessageTool::new(router.clone(), name.clone())));
        agent.register_tool(Arc::new(ListAgentsTool::new(router)));

        // Add persistent memory (SQLite)
        let memory_dir = dirs::home_dir()
            .map(|h| h.join(".anima").join("memory"))
            .expect("Could not determine home directory");
        if let Err(e) = std::fs::create_dir_all(&memory_dir) {
            println!("\x1b[31mWarning: Could not create memory directory: {}\x1b[0m", e);
        }
        let memory_path = memory_dir.join(format!("{}.db", name));
        let memory: Box<dyn Memory> = match SqliteMemory::open(
            memory_path.to_str().unwrap(),
            &name,
        ) {
            Ok(m) => Box::new(m),
            Err(e) => {
                println!("\x1b[31mWarning: Could not open persistent memory: {}. Using in-memory.\x1b[0m", e);
                Box::new(InMemoryStore::new())
            }
        };
        agent = agent.with_memory(memory);

        // Add observer (non-verbose for cleaner REPL output)
        let observer = Arc::new(ConsoleObserver::new(false));
        agent = agent.with_observer(observer);

        // Add LLM if specified
        if let Some(l) = llm {
            agent = agent.with_llm(l);
        }

        self.agents.insert(name.clone(), ReplAgent {
            agent,
            llm_name: llm_name.clone(),
        });

        println!("\x1b[32m✓ Created agent '{}' (llm: {})\x1b[0m", name, llm_name);
    }

    fn create_llm_from_spec(&self, spec: &str) -> Result<Arc<dyn LLM>, Box<dyn Error>> {
        // Parse spec: "provider/model" or just "provider"
        let parts: Vec<&str> = spec.splitn(2, '/').collect();
        let provider = parts[0];
        let model = parts.get(1).copied();

        match provider {
            "openai" => {
                let api_key = std::env::var("OPENAI_API_KEY")
                    .map_err(|_| "OPENAI_API_KEY not set")?;
                let mut client = OpenAIClient::new(api_key);
                if let Some(m) = model {
                    client = client.with_model(m);
                }
                Ok(Arc::new(client))
            }
            "anthropic" => {
                let api_key = std::env::var("ANTHROPIC_API_KEY")
                    .map_err(|_| "ANTHROPIC_API_KEY not set")?;
                let mut client = AnthropicClient::new(api_key);
                if let Some(m) = model {
                    client = client.with_model(m);
                }
                Ok(Arc::new(client))
            }
            "ollama" => {
                let mut client = OllamaClient::new();
                if let Some(m) = model {
                    client = client.with_model(m);
                }
                Ok(Arc::new(client))
            }
            _ => Err(format!("Unknown provider: {} (use 'openai', 'anthropic', or 'ollama')", provider).into()),
        }
    }

    fn cmd_agent_list(&self) {
        if self.agents.is_empty() {
            println!("\x1b[33mNo agents created yet. Use 'agent create <name>' to create one.\x1b[0m");
            return;
        }

        println!("\x1b[1mAgents:\x1b[0m");
        for (name, entry) in &self.agents {
            println!("  \x1b[36m{}\x1b[0m (llm: {})", name, entry.llm_name);
        }
    }

    fn cmd_agent_list_saved(&self) {
        let memory_dir = match dirs::home_dir() {
            Some(h) => h.join(".anima").join("memory"),
            None => {
                println!("\x1b[31mCould not determine home directory\x1b[0m");
                return;
            }
        };

        if !memory_dir.exists() {
            println!("\x1b[33mNo saved agents yet. Memory directory doesn't exist.\x1b[0m");
            return;
        }

        let entries: Vec<_> = match std::fs::read_dir(&memory_dir) {
            Ok(dir) => dir
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().map(|ext| ext == "db").unwrap_or(false))
                .collect(),
            Err(e) => {
                println!("\x1b[31mCould not read memory directory: {}\x1b[0m", e);
                return;
            }
        };

        if entries.is_empty() {
            println!("\x1b[33mNo saved agents found.\x1b[0m");
            return;
        }

        println!("\x1b[1mSaved agents (with persistent memory):\x1b[0m");
        for entry in entries {
            let path = entry.path();
            let name = path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");
            let loaded = if self.agents.contains_key(name) { " (loaded)" } else { "" };
            println!("  \x1b[36m{}\x1b[0m{}", name, loaded);
        }
    }

    async fn cmd_agent_remove(&mut self, name: &str) {
        let name = name.trim();
        if self.agents.remove(name).is_some() {
            self.runtime.remove_agent(name).await;
            println!("\x1b[32m✓ Removed agent '{}'\x1b[0m", name);
        } else {
            println!("\x1b[31mAgent '{}' not found\x1b[0m", name);
        }
    }

    async fn cmd_task(&mut self, agent_name: &str, task: &str) {
        let agent_name = agent_name.trim();
        let task = task.trim();

        let entry = match self.agents.get_mut(agent_name) {
            Some(a) => a,
            None => {
                println!("\x1b[31mAgent '{}' not found. Create it with 'agent create {}'\x1b[0m", agent_name, agent_name);
                return;
            }
        };

        // Check if agent has an LLM
        if entry.llm_name == "none" {
            println!("\x1b[31mAgent '{}' has no LLM configured. Recreate with --llm flag.\x1b[0m", agent_name);
            return;
        }

        println!("\x1b[33m[{}]\x1b[0m thinking...", agent_name);

        // Use streaming for real-time output
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(100);

        let options = ThinkOptions {
            stream: true,
            ..Default::default()
        };

        // Spawn task to print tokens
        let print_task = tokio::spawn(async move {
            while let Some(token) = rx.recv().await {
                print!("{}", token);
                let _ = io::stdout().flush();
            }
            println!(); // Final newline
        });

        // Run the thinking
        match entry.agent.think_streaming_with_options(task, options, tx).await {
            Ok(_) => {
                // Wait for print task
                let _ = print_task.await;
            }
            Err(e) => {
                println!("\x1b[31mError: {}\x1b[0m", e);
            }
        }
    }

    async fn cmd_memory(&mut self, agent_name: &str) {
        let agent_name = agent_name.trim();

        let entry = match self.agents.get(&agent_name.to_string()) {
            Some(a) => a,
            None => {
                println!("\x1b[31mAgent '{}' not found\x1b[0m", agent_name);
                return;
            }
        };

        // We need to access memory through the agent's recall interface
        // Since we can't directly access the memory, we'll list what keys we know about
        // For now, show a message about memory inspection limitations

        // Use a workaround: try to recall some common keys
        println!("\x1b[1mMemory for '{}'\x1b[0m", agent_name);
        println!("\x1b[33m(Note: Memory inspection requires direct Memory trait access.)\x1b[0m");
        println!("\x1b[33mUse the agent to recall specific keys with a task.\x1b[0m");

        // Try to recall a few test keys
        let test_keys = ["task", "context", "history", "state"];
        let mut found_any = false;
        for key in &test_keys {
            if let Some(value) = entry.agent.recall(key).await {
                if !found_any {
                    println!("\n\x1b[1mStored values:\x1b[0m");
                    found_any = true;
                }
                println!("  \x1b[36m{}\x1b[0m: {}", key, value);
            }
        }

        if !found_any {
            println!("\x1b[33mNo memories stored yet.\x1b[0m");
        }
    }

    async fn cmd_ask(&mut self, from_agent: &str, args: &str) {
        let from_agent = from_agent.trim();

        // Parse: <other_agent> "<question>"
        // Find the target agent and question
        let parts: Vec<&str> = args.splitn(2, ' ').collect();
        if parts.len() < 2 {
            println!("\x1b[31mUsage: <agent>: ask <other_agent> \"<question>\"\x1b[0m");
            return;
        }

        let to_agent = parts[0].trim();
        let question = parts[1].trim().trim_matches('"');

        // Check both agents exist
        if !self.agents.contains_key(from_agent) {
            println!("\x1b[31mAgent '{}' not found\x1b[0m", from_agent);
            return;
        }
        if !self.agents.contains_key(to_agent) {
            println!("\x1b[31mAgent '{}' not found\x1b[0m", to_agent);
            return;
        }

        println!("\x1b[33m[{}]\x1b[0m asking \x1b[36m{}\x1b[0m: \"{}\"", from_agent, to_agent, question);

        // Send message from one agent to another
        let from_entry = self.agents.get(from_agent).unwrap();
        match from_entry.agent.send_message(to_agent, question).await {
            Ok(_) => {
                println!("\x1b[32m✓ Message sent to {}\x1b[0m", to_agent);
                println!("\x1b[33m(Send a task to '{}' and they'll see the message)\x1b[0m", to_agent);
            }
            Err(e) => {
                println!("\x1b[31mFailed to send message: {}\x1b[0m", e);
            }
        }
    }

    fn cmd_set_llm(&mut self, spec: &str) {
        let spec = spec.trim();
        if spec.is_empty() {
            println!("\x1b[31mUsage: set llm <provider/model>\x1b[0m");
            println!("  Examples: set llm openai/gpt-4o");
            println!("            set llm anthropic/claude-sonnet-4-20250514");
            return;
        }

        // Validate the spec
        let parts: Vec<&str> = spec.splitn(2, '/').collect();
        let provider = parts[0];
        if provider != "openai" && provider != "anthropic" && provider != "ollama" {
            println!("\x1b[31mUnknown provider '{}'. Use 'openai', 'anthropic', or 'ollama'.\x1b[0m", provider);
            return;
        }

        self.default_llm = Some(spec.to_string());
        println!("\x1b[32m✓ Default LLM set to '{}'\x1b[0m", spec);
    }

    fn cmd_help(&self) {
        println!("\x1b[1mAnima REPL Commands:\x1b[0m");
        println!();
        println!("  \x1b[36magent create <name> [--llm <provider/model>]\x1b[0m");
        println!("      Create a new agent (memory persists to ~/.anima/memory/)");
        println!();
        println!("  \x1b[36magent list\x1b[0m");
        println!("      List active agents in this session");
        println!();
        println!("  \x1b[36magent list-saved\x1b[0m");
        println!("      List agents with persistent memory (can be recreated)");
        println!();
        println!("  \x1b[36magent remove <name>\x1b[0m");
        println!("      Remove an agent from session (memory persists)");
        println!();
        println!("  \x1b[36m<agent>: <task>\x1b[0m");
        println!("      Send a task to an agent (streams output)");
        println!();
        println!("  \x1b[36mmemory <agent>\x1b[0m");
        println!("      Show an agent's memory");
        println!();
        println!("  \x1b[36m<agent>: ask <other> \"<question>\"\x1b[0m");
        println!("      Send a message from one agent to another");
        println!();
        println!("  \x1b[36mset llm <provider/model>\x1b[0m");
        println!("      Set default LLM for new agents");
        println!("      Examples: openai/gpt-4o, anthropic/claude-sonnet-4-20250514, ollama/llama3");
        println!("      Ollama: set OLLAMA_HOST env var (default: http://localhost:11434)");
        println!();
        println!("  \x1b[36mhelp\x1b[0m");
        println!("      Show this help message");
        println!();
        println!("  \x1b[36mexit\x1b[0m / \x1b[36mquit\x1b[0m");
        println!("      Exit the REPL");
        println!();
        println!("\x1b[33mNote: Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or OLLAMA_HOST environment variables.\x1b[0m");
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
        assert!(repl.agents.is_empty());
        assert!(repl.default_llm.is_none());
    }

    #[test]
    fn test_repl_default() {
        let repl = Repl::default();
        assert!(repl.agents.is_empty());
    }

    #[tokio::test]
    async fn test_cmd_agent_list_empty() {
        let repl = Repl::new();
        // This just verifies it doesn't panic
        repl.cmd_agent_list();
    }

    #[test]
    fn test_cmd_set_llm() {
        let mut repl = Repl::new();
        repl.cmd_set_llm("openai/gpt-4o");
        assert_eq!(repl.default_llm, Some("openai/gpt-4o".to_string()));
    }

    #[test]
    fn test_cmd_set_llm_invalid_provider() {
        let mut repl = Repl::new();
        repl.cmd_set_llm("invalid/model");
        // Should not set the default
        assert!(repl.default_llm.is_none());
    }

    #[test]
    fn test_cmd_help() {
        let repl = Repl::new();
        // Just verify it doesn't panic
        repl.cmd_help();
    }
}
