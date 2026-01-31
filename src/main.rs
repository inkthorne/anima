use anima::{
    Runtime, OpenAIClient, AnthropicClient, OllamaClient, ThinkOptions, AutoMemoryConfig, ReflectionConfig,
    InMemoryStore, SqliteMemory, LLM, Memory,
};
use anima::agent_dir::AgentDir;
use anima::config::AgentConfig;
use anima::observe::ConsoleObserver;
use anima::repl::Repl;
use anima::tools::{AddTool, EchoTool, ReadFileTool, WriteFileTool, HttpTool, ShellTool, SendMessageTool, ListAgentsTool};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;
use std::io::{self, Write};

#[derive(Parser)]
#[command(name = "anima", about = "The animating spirit - AI agent runtime")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run an agent from a directory or by name, starting an interactive REPL
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
    /// Interactive REPL for exploring anima (default if no command given)
    Repl,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let command = cli.command.unwrap_or(Commands::Repl);

    match command {
        Commands::Run { agent } => {
            if let Err(e) = run_agent_dir(&agent).await {
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
        Commands::Task { config, task, stream, verbose } => {
            if let Err(e) = run_agent_task(&config, &task, stream, verbose).await {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Repl => {
            let mut repl = Repl::new();
            if let Err(e) = repl.run().await {
                eprintln!("REPL error: {}", e);
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

/// Run an agent from a directory and start an interactive REPL
async fn run_agent_dir(agent: &str) -> Result<(), Box<dyn std::error::Error>> {
    let agent_path = resolve_agent_path(agent);

    // Load the agent directory
    let agent_dir = AgentDir::load(&agent_path)?;
    let agent_name = agent_dir.config.agent.name.clone();

    // Load persona if configured
    let persona = agent_dir.load_persona()?;

    // Get API key
    let api_key = agent_dir.api_key()?;

    // Create LLM from config
    let llm: Arc<dyn LLM> = match agent_dir.config.llm.provider.as_str() {
        "openai" => {
            let key = api_key.ok_or("OpenAI API key not configured")?;
            Arc::new(OpenAIClient::new(key).with_model(&agent_dir.config.llm.model))
        }
        "anthropic" => {
            let key = api_key.ok_or("Anthropic API key not configured")?;
            Arc::new(AnthropicClient::new(key).with_model(&agent_dir.config.llm.model))
        }
        "ollama" => {
            Arc::new(OllamaClient::new().with_model(&agent_dir.config.llm.model))
        }
        other => return Err(format!("Unsupported LLM provider: {}", other).into()),
    };

    // Create memory from config
    let memory: Box<dyn Memory> = if let Some(mem_path) = agent_dir.memory_path() {
        // Ensure parent directory exists
        if let Some(parent) = mem_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Box::new(SqliteMemory::open(
            mem_path.to_str().ok_or("Invalid memory path")?,
            &agent_name,
        )?)
    } else {
        Box::new(InMemoryStore::new())
    };

    // Create runtime and agent
    let mut runtime = Runtime::new();
    let mut agent = runtime.spawn_agent(agent_name.clone()).await;

    // Register tools
    agent.register_tool(Arc::new(AddTool));
    agent.register_tool(Arc::new(EchoTool));
    agent.register_tool(Arc::new(ReadFileTool));
    agent.register_tool(Arc::new(WriteFileTool));
    agent.register_tool(Arc::new(HttpTool::new()));
    agent.register_tool(Arc::new(ShellTool::new()));

    // Register messaging tools
    let router = runtime.router().clone();
    agent.register_tool(Arc::new(SendMessageTool::new(router.clone(), agent_name.clone())));
    agent.register_tool(Arc::new(ListAgentsTool::new(router)));

    // Apply LLM and memory
    agent = agent.with_llm(llm);
    agent = agent.with_memory(memory);

    // Add observer (non-verbose for cleaner REPL output)
    let observer = Arc::new(ConsoleObserver::new(false));
    agent = agent.with_observer(observer);

    // Start interactive REPL with the loaded agent
    let mut repl = Repl::with_agent(agent_name.clone(), agent, persona);
    println!("\x1b[32m✓ Loaded agent '{}' from {}\x1b[0m", agent_name, agent_path.display());

    repl.run().await?;
    Ok(())
}

/// Scaffold a new agent directory
fn create_agent(name: &str, path: Option<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    let agent_path = path.unwrap_or_else(|| agents_dir().join(name));

    // Check if directory already exists
    if agent_path.exists() {
        return Err(format!("Agent directory already exists: {}", agent_path.display()).into());
    }

    // Create the directory
    std::fs::create_dir_all(&agent_path)?;

    // Write config.toml template
    let config_content = format!(r#"[agent]
name = "{name}"
persona_file = "persona.md"

[llm]
provider = "anthropic"
model = "claude-sonnet-4-20250514"
api_key = "${{ANTHROPIC_API_KEY}}"

[memory]
path = "memory.db"

# Optional timer configuration
# [timer]
# enabled = true
# interval = "5m"
# message = "Heartbeat — check for anything interesting"
"#);
    std::fs::write(agent_path.join("config.toml"), config_content)?;

    // Write persona.md template
    let persona_content = format!(r#"# {name}

You are {name}, an AI agent running in the Anima runtime.

## Personality

Be helpful, concise, and focused on the task at hand.

## Capabilities

You have access to tools for:
- Reading and writing files
- Making HTTP requests
- Running shell commands
- Sending messages to other agents

## Guidelines

- Think step by step before acting
- Use tools when needed to accomplish tasks
- Be proactive about using your memory to track important information
"#);
    std::fs::write(agent_path.join("persona.md"), persona_content)?;

    println!("\x1b[32m✓ Created agent '{}' at {}\x1b[0m", name, agent_path.display());
    println!();
    println!("  Files created:");
    println!("    \x1b[36mconfig.toml\x1b[0m  — agent configuration");
    println!("    \x1b[36mpersona.md\x1b[0m   — system prompt / personality");
    println!();
    println!("  Next steps:");
    println!("    1. Edit config.toml to configure your LLM");
    println!("    2. Edit persona.md to define your agent's personality");
    println!("    3. Run with: \x1b[36manima run {}\x1b[0m", name);

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
                let provider = &agent_dir.config.llm.provider;
                let model = &agent_dir.config.llm.model;
                format!(" ({}/{})", provider, model)
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
        auto_memory,
        reflection,
        stream,
        retry_policy: Some(config.retry.to_policy()),
        conversation_history: None,
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
        println!("{}", result);
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

        // Check config.toml content
        let config_content = std::fs::read_to_string(agent_path.join("config.toml")).unwrap();
        assert!(config_content.contains("name = \"test-agent\""));
        assert!(config_content.contains("[llm]"));
        assert!(config_content.contains("[memory]"));

        // Check persona.md content
        let persona_content = std::fs::read_to_string(agent_path.join("persona.md")).unwrap();
        assert!(persona_content.contains("# test-agent"));
        assert!(persona_content.contains("You are test-agent"));
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
}
