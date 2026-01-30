use anima::{Runtime, OpenAIClient};
use anima::tools::{AddTool, EchoTool};
use anima::supervision::ChildConfig;

#[tokio::main]
async fn main() {
    println!("=== Anima v0.7 Demo: Agent Supervision ===");
    println!("This demo shows how a parent agent delegates tasks to child agents\n");
    
    let llm = OpenAIClient::new("ollama")
        .with_base_url("http://100.67.222.97:11434/v1")
        .with_model("qwen3-coder-32k");
    
    let mut runtime = Runtime::new();
    
    // Create parent agent with tools
    let mut parent = runtime.spawn_agent("parent".to_string());
    parent.register_tool(Box::new(AddTool));
    parent.register_tool(Box::new(EchoTool));
    parent = parent.with_llm(Box::new(llm));
    
    println!("Parent agent created: {}", parent.id);
    println!("Parent tools registered: add, echo\n");
    
    // Complex task that benefits from delegation
    let complex_task = "I need to calculate a few things and then summarize the results. 
                       First, add 5 and 3, then add 8 and 2, and finally echo 'Hello World' 
                       for a greeting. Combine all results into one final message.";
    
    println!("Parent task: {}", complex_task);
    
    // Parent starts by spawning children for subtasks
    println!("\n--- Spawning child agents ---");
    
    // Spawn child 1: Add 5 + 3
    let child_config1 = ChildConfig::new("Calculate 5 + 3");
    let child_id1 = parent.spawn_child(child_config1);
    println!("Child 1 spawned: {}", child_id1);
    
    // Spawn child 2: Add 8 + 2
    let child_config2 = ChildConfig::new("Calculate 8 + 2");
    let child_id2 = parent.spawn_child(child_config2);
    println!("Child 2 spawned: {}", child_id2);
    
    // Wait for first child to complete
    println!("\n--- Waiting for first child to complete ---");
    match parent.wait_for_child(&child_id1).await {
        Ok(result) => println!("Child 1 result: {}", result),
        Err(e) => println!("Child 1 failed: {}", e),
    }
    
    // Wait for second child to complete
    println!("\n--- Waiting for second child to complete ---");
    match parent.wait_for_child(&child_id2).await {
        Ok(result) => println!("Child 2 result: {}", result),
        Err(e) => println!("Child 2 failed: {}", e),
    }
    
    // Show that all children are completed
    println!("\n--- Final status check ---");
    println!("All children have completed their tasks.");
    
    // Parent can now use the child results in its own response
    println!("Parent using child results to form final response...");
    
    println!("\n=== Demo complete ===");
    println!("This demonstration shows the key supervision flow:");
    println!("1. Parent agent creates runtime and registers tools");
    println!("2. Parent delegates complex tasks by spawning child agents"); 
    println!("3. Parent waits for child results with wait_for_child()");
    println!("4. Parent uses child results in its own response");
}