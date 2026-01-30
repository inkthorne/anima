use anima::{Runtime, OpenAIClient, ThinkOptions, ReflectionConfig};
use anima::tools::{AddTool, EchoTool};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    println!("=== Anima v1.0 Demo: Self-Reflection ===\n");
    
    let llm = OpenAIClient::new("ollama")
        .with_base_url("http://100.67.222.97:11434/v1")
        .with_model("qwen3-coder-32k");
    
    let mut runtime = Runtime::new();
    let mut agent = runtime.spawn_agent("reflective-agent".to_string());
    agent.register_tool(Arc::new(AddTool));
    agent.register_tool(Arc::new(EchoTool));
    agent = agent.with_llm(Arc::new(llm));
    
    println!("Agent created with tools: add, echo\n");
    
    // Demo 1: Without reflection
    println!("--- Without Reflection ---");
    let task = "What is 15 + 27?";
    println!("Task: {}", task);
    
    let options_no_reflect = ThinkOptions::default();
    match agent.think_with_options(task, options_no_reflect).await {
        Ok(response) => println!("Response: {}\n", response),
        Err(e) => println!("Error: {}\n", e),
    }
    
    // Demo 2: With reflection
    println!("--- With Reflection ---");
    let task2 = "Calculate 15 + 27 and also 100 - 58. Report both results.";
    println!("Task: {}", task2);
    
    let options_with_reflect = ThinkOptions {
        max_iterations: 10,
        system_prompt: None,
        reflection: Some(ReflectionConfig {
            prompt: "Check: Did you calculate BOTH numbers? Did you report BOTH results clearly?".to_string(),
            max_revisions: 2,
        }),
    };
    
    match agent.think_with_options(task2, options_with_reflect).await {
        Ok(response) => println!("Response: {}\n", response),
        Err(e) => println!("Error: {}\n", e),
    }
    
    println!("=== Demo Complete ===");
    println!("Self-reflection enables:");
    println!("  • Think → Evaluate → Revise loops");
    println!("  • Configurable reflection prompts");
    println!("  • Bounded revision cycles");
    println!("  • More complete, accurate responses");
}
