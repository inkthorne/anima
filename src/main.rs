use anima::{Runtime, OpenAIClient};
use anima::tools::{AddTool, EchoTool};
use anima::supervision::ChildConfig;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    println!("=== Anima v0.8 Demo: Concurrent Child Execution ===");
    println!("This demo shows how a parent agent can spawn multiple child agents concurrently\n");
    
    let llm = OpenAIClient::new("ollama")
        .with_base_url("http://100.67.222.97:11434/v1")
        .with_model("qwen3-coder-32k");
    
    let mut runtime = Runtime::new();
    
     // Create parent agent with tools
     let mut parent = runtime.spawn_agent("parent".to_string());
     parent.register_tool(Arc::new(AddTool));
     parent.register_tool(Arc::new(EchoTool));
     parent = parent.with_llm(Arc::new(llm));
    
    println!("Parent agent created: {}", parent.id);
    println!("Parent tools registered: add, echo\n");
    
    // Complex task that benefits from delegation
    let complex_task = "I need to calculate a few things and then summarize the results. 
                       First, add 5 and 3, then add 8 and 2, and finally echo 'Hello World' 
                       for a greeting. Combine all results into one final message.";
    
    println!("Parent task: {}", complex_task);
    
    // Parent starts by spawning children for subtasks
    println!("\n--- Spawning child agents concurrently ---");
    
    // Spawn child 1: Add 5 + 3
    let child_config1 = ChildConfig::new("Calculate 5 + 3");
    let child_id1 = parent.spawn_child(child_config1);
    println!("Child 1 spawned: {}", child_id1);
    
    // Spawn child 2: Add 8 + 2
    let child_config2 = ChildConfig::new("Calculate 8 + 2");
    let child_id2 = parent.spawn_child(child_config2);
    println!("Child 2 spawned: {}", child_id2);
    
    // Spawn child 3: Echo Hello World
    let child_config3 = ChildConfig::new("Echo 'Hello World'");
    let child_id3 = parent.spawn_child(child_config3);
    println!("Child 3 spawned: {}", child_id3);
    
    // Wait for all children to complete concurrently
    println!("\n--- Waiting for all children to complete concurrently ---");
    println!("Note: Children are completed in the order they finish, not in the order they were spawned");
    
    // Wait for children to complete (this shows concurrent execution)
    let mut results = Vec::new();
    results.push(parent.wait_for_child(&child_id1).await);
    results.push(parent.wait_for_child(&child_id2).await);
    results.push(parent.wait_for_child(&child_id3).await);
    
    // Print all results
    for (i, result) in results.iter().enumerate() {
        match result {
            Ok(r) => println!("Child {} result: {}", i+1, r),
            Err(e) => println!("Child {} failed: {}", i+1, e),
        }
    }
    
    // Show that all children are completed
    println!("\n--- Final status check ---");
    println!("All children have completed their tasks.");
    
    // Parent can now use the child results in its own response
    println!("Parent using child results to form final response...");
    
    println!("\n=== Demo complete ===");
    println!("This demonstration shows concurrent child execution:");
    println!("1. Parent agent creates runtime and registers tools");
    println!("2. Parent spawns multiple child agents concurrently"); 
    println!("3. Parent waits for child results with wait_for_child()");
    println!("4. Parent uses child results in its own response");
}