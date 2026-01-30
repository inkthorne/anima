use anima::{Runtime, OpenAIClient, ThinkOptions, AutoMemoryConfig};
use anima::tools::{AddTool, EchoTool};
use anima::InMemoryStore;
use std::sync::Arc;
use serde_json::json;

#[tokio::main]
async fn main() {
    println!("=== Anima v1.3 Demo: Auto-Memory ===\n");

    let llm = OpenAIClient::new("ollama")
        .with_base_url("http://100.67.222.97:11434/v1")
        .with_model("qwen3-coder-32k");

    let mut runtime = Runtime::new();
    let mut agent = runtime.spawn_agent("memory-agent".to_string());
    agent.register_tool(Arc::new(AddTool));
    agent.register_tool(Arc::new(EchoTool));
    agent = agent.with_llm(Arc::new(llm));

    // Attach memory
    let memory = Box::new(InMemoryStore::new());
    agent = agent.with_memory(memory);

    println!("Agent created with tools: add, echo");
    println!("Agent has memory attached\n");

    // Store some memories
    println!("--- Storing Memories ---");
    agent.remember("user:name", json!("Arya")).await.unwrap();
    agent.remember("user:role", json!("Lead Architect")).await.unwrap();
    agent.remember("project:name", json!("Anima")).await.unwrap();
    agent.remember("project:version", json!("1.3")).await.unwrap();
    println!("Stored: user:name = Arya");
    println!("Stored: user:role = Lead Architect");
    println!("Stored: project:name = Anima");
    println!("Stored: project:version = 1.3\n");

    // Demo 1: Without auto-memory
    println!("--- Without Auto-Memory ---");
    let task = "What do you know about the user?";
    println!("Task: {}", task);

    let options_no_memory = ThinkOptions::default();
    match agent.think_with_options(task, options_no_memory).await {
        Ok(response) => println!("Response: {}\n", response),
        Err(e) => println!("Error: {}\n", e),
    }

    // Demo 2: With auto-memory (all memories)
    println!("--- With Auto-Memory (All) ---");
    let task2 = "What do you know about the user and the project?";
    println!("Task: {}", task2);

    let options_with_memory = ThinkOptions {
        max_iterations: 10,
        system_prompt: Some("You are a helpful assistant.".to_string()),
        reflection: None,
        auto_memory: Some(AutoMemoryConfig::default()),
    };

    match agent.think_with_options(task2, options_with_memory).await {
        Ok(response) => println!("Response: {}\n", response),
        Err(e) => println!("Error: {}\n", e),
    }

    // Demo 3: With auto-memory (filtered by prefix)
    println!("--- With Auto-Memory (User Only) ---");
    let task3 = "Tell me about yourself based on what you remember.";
    println!("Task: {}", task3);

    let options_user_only = ThinkOptions {
        max_iterations: 10,
        system_prompt: None,
        reflection: None,
        auto_memory: Some(AutoMemoryConfig {
            max_entries: 10,
            include_recent: true,
            key_prefixes: vec!["user:".to_string()],
        }),
    };

    match agent.think_with_options(task3, options_user_only).await {
        Ok(response) => println!("Response: {}\n", response),
        Err(e) => println!("Error: {}\n", e),
    }

    println!("=== Demo Complete ===");
    println!("Auto-Memory enables:");
    println!("  • Automatic context injection from memory");
    println!("  • Prefix-based filtering (user:, project:, etc.)");
    println!("  • Configurable entry limits");
    println!("  • Recent-first or oldest-first ordering");
}
