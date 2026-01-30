use anima::{Runtime, OpenAIClient};
use anima::tools::{AddTool};

#[tokio::main]
async fn main() {
    println!("=== Anima v0.4 Demo ===");
    println!("Testing LLM-driven agent with tool usage");
    
    // Use Ollama on Mojave (OpenAI-compatible API)
    println!("Using Ollama on Mojave");
    let llm = OpenAIClient::new("ollama")  // Ollama ignores the API key
        .with_base_url("http://100.67.222.97:11434/v1")
        .with_model("qwen3-coder-32k");
    
    // Create runtime and agent
    let mut runtime = Runtime::new();
    let agent = runtime.spawn_agent("assistant".to_string());
    
    // Attach LLM to the agent
    let mut agent = agent.with_llm(Box::new(llm));
    
    // Register the add tool
    agent.register_tool(Box::new(AddTool));
    
    // Let the agent think!
    println!("Task: What is 5 + 3?");
    match agent.think("What is 5 + 3? Use the add tool.").await {
        Ok(response) => println!("Agent: {}", response),
        Err(e) => println!("Error: {}", e),
    }
    
    println!("\nDemo complete!");
}