use anima::{Memory, SqliteMemory};

#[tokio::main]
async fn main() {
    println!("=== Anima v0.9 Demo: Persistent Memory ===\n");
    
    let db_path = "/tmp/anima_demo.db";
    let agent_id = "demo-agent";
    
    // First run: Create memories
    {
        println!("--- First Session ---");
        let mut memory = SqliteMemory::open(db_path, agent_id)
            .expect("Failed to open database");
        
        // Store some memories
        memory.set("name", serde_json::json!("Arya")).await.unwrap();
        memory.set("count", serde_json::json!(42)).await.unwrap();
        memory.set("facts:rust", serde_json::json!("Rust is memory safe")).await.unwrap();
        memory.set("facts:anima", serde_json::json!("Anima means soul")).await.unwrap();
        
        println!("Stored memories: name, count, facts:rust, facts:anima");
        
        // Read them back
        if let Some(entry) = memory.get("name").await {
            println!("  name = {} (created: {})", entry.value, entry.created_at);
        }
        if let Some(entry) = memory.get("count").await {
            println!("  count = {} (created: {})", entry.value, entry.created_at);
        }
        
        // List keys with prefix
        let fact_keys = memory.list_keys(Some("facts:")).await;
        println!("  Keys starting with 'facts:': {:?}", fact_keys);
        
        println!("Session 1 complete. Memory closed.\n");
    }
    
    // Second run: Memories persist!
    {
        println!("--- Second Session (same agent) ---");
        let memory = SqliteMemory::open(db_path, agent_id)
            .expect("Failed to open database");
        
        // Memories still there!
        if let Some(entry) = memory.get("name").await {
            println!("  name = {} (still here!)", entry.value);
        }
        if let Some(entry) = memory.get("count").await {
            println!("  count = {} (persisted!)", entry.value);
        }
        
        let all_keys = memory.list_keys(None).await;
        println!("  All keys: {:?}", all_keys);
        
        println!("Session 2 complete. Agent remembers!\n");
    }
    
    // Third run: Different agent, different memories
    {
        println!("--- Third Session (different agent) ---");
        let other_agent = "other-agent";
        let mut memory = SqliteMemory::open(db_path, other_agent)
            .expect("Failed to open database");
        
        // This agent has no memories yet
        let keys = memory.list_keys(None).await;
        println!("  {} has keys: {:?}", other_agent, keys);
        
        // But can create its own
        memory.set("name", serde_json::json!("Other")).await.unwrap();
        println!("  Stored name = 'Other' for {}", other_agent);
        
        // Original agent's memories are isolated
        let original = SqliteMemory::open(db_path, agent_id)
            .expect("Failed to open database");
        if let Some(entry) = original.get("name").await {
            println!("  {} still has name = {}", agent_id, entry.value);
        }
        
        println!("Session 3 complete. Agents are isolated!\n");
    }
    
    // Episodic memory demo
    {
        println!("--- Episodic Memory Demo ---");
        let mut memory = SqliteMemory::open(db_path, agent_id)
            .expect("Failed to open database");
        
        // Add a new memory
        memory.set("recent", serde_json::json!("Just happened")).await.unwrap();
        
        // Query by time (last hour)
        let one_hour_ago = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() - 3600;
        
        let recent = memory.query_by_time(one_hour_ago, None).await.unwrap();
        println!("  Memories from last hour:");
        for (key, entry) in recent {
            println!("    {} = {} (at {})", key, entry.value, entry.updated_at);
        }
    }
    
    // Cleanup for demo reproducibility
    std::fs::remove_file(db_path).ok();
    
    println!("\n=== Demo Complete ===");
    println!("Persistent memory enables:");
    println!("  • Agent identity across sessions");
    println!("  • Isolated memory per agent");
    println!("  • Episodic queries (what happened when?)");
    println!("  • Memory is identity. 🧠");
}
