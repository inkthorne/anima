use anima::{Runtime, OpenAIClient};
use anima::tools::AddTool;

#[tokio::main]
async fn main() {
    println!("=== Anima v0.5 Demo ===");
    println!("Testing multi-turn agentic loop\n");
    
    // Use Ollama on Mojave (OpenAI-compatible API)
    println!("Using Ollama on Mojave\n");
    let llm = OpenAIClient::new("ollama")
        .with_base_url("http://100.67.222.97:11434/v1")
        .with_model("qwen3-coder-32k");
    
    // Create runtime and agent
    let mut runtime = Runtime::new();
    let agent = runtime.spawn_agent("assistant".to_string());
    
    // Attach LLM to the agent
    let mut agent = agent.with_llm(Box::new(llm));
    
    // Register the add tool
    agent.register_tool(Box::new(AddTool));
    
    // Multi-step task requiring multiple tool calls
    let task = "First add 5 + 3. Then add 10 to that result. What's the final answer?";
    println!("Task: {}\n", task);
    
    match agent.think(task).await {
        Ok(response) => println!("Agent: {}", response),
        Err(e) => println!("Error: {}", e),
    }
    
    println!("\nDemo complete!");
}