use anima::{
    Runtime, OpenAIClient, AnthropicClient, ThinkOptions, AutoMemoryConfig, ReflectionConfig,
    InMemoryStore, SqliteMemory, LLM,
};
use anima::config::AgentConfig;
use anima::tools::{AddTool, EchoTool, ReadFileTool, WriteFileTool, HttpTool, ShellTool};
use clap::{Parser, Subcommand};
use std::sync::Arc;
use std::io::{self, Write};

#[derive(Parser)]
#[command(name = "anima", about = "The animating spirit - AI agent runtime")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run an agent with a task
    Run {
        /// Path to agent config file
        config: String,
        /// Task for the agent
        task: String,
        /// Enable streaming output
        #[arg(long)]
        stream: bool,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run { config, task, stream } => {
            if let Err(e) = run_agent(&config, &task, stream).await {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
    }
}

async fn run_agent(config_path: &str, task: &str, stream: bool) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load config
    let config = AgentConfig::from_file(config_path)?;

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
    let mut agent = runtime.spawn_agent(config.agent.name.clone());

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
