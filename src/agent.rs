use std::collections::HashMap;
use std::sync::Arc;

use crate::error::ToolError;
use crate::tool::Tool;
use crate::message::Message;
use crate::memory::{Memory, MemoryError};
use crate::llm::{LLM, ChatMessage, ToolSpec, LLMError};
use tokio::sync::mpsc;
use serde_json::Value;
use crate::supervision::{ChildHandle, ChildConfig, ChildStatus};
use tokio::sync::oneshot;

/// Options for the think() agentic loop
pub struct ThinkOptions {
    /// Maximum iterations before giving up (default: 10)
    pub max_iterations: usize,
    /// Optional system prompt to set agent behavior
    pub system_prompt: Option<String>,
    /// Optional reflection configuration for self-evaluation
    pub reflection: Option<ReflectionConfig>,
    /// Optional auto-memory configuration for injecting memories into context
    pub auto_memory: Option<AutoMemoryConfig>,
}

impl Default for ThinkOptions {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            system_prompt: None,
            reflection: None,
            auto_memory: None,
        }
    }
}

/// Configuration for self-reflection after generating a response
#[derive(Debug, Clone)]
pub struct ReflectionConfig {
    /// Prompt to use when asking the LLM to evaluate its response
    pub prompt: String,
    /// Maximum number of revision cycles allowed
    pub max_revisions: usize,
}

impl Default for ReflectionConfig {
    fn default() -> Self {
        Self {
            prompt: String::from("Evaluate your response. Is it complete and correct? If not, explain what needs to change."),
            max_revisions: 1,
        }
    }
}

/// Configuration for automatic memory injection during thinking
#[derive(Debug, Clone)]
pub struct AutoMemoryConfig {
    /// Maximum number of memory entries to include
    pub max_entries: usize,
    /// Include most recent memories
    pub include_recent: bool,
    /// Only include memories with keys matching these prefixes (empty = all)
    pub key_prefixes: Vec<String>,
}

impl Default for AutoMemoryConfig {
    fn default() -> Self {
        Self {
            max_entries: 10,
            include_recent: true,
            key_prefixes: vec![],
        }
    }
}

/// Result of a reflection evaluation
#[derive(Debug, Clone)]
pub struct ReflectionResult {
    /// Whether the response was accepted as-is
    pub accepted: bool,
    /// Feedback for revision if not accepted
    pub feedback: Option<String>,
}

pub struct Agent {
    pub id: String,
    tools: HashMap<String, Arc<dyn Tool>>,
    inbox: mpsc::Receiver<Message>,
    memory: Option<Box<dyn Memory>>,
    llm: Option<Arc<dyn LLM>>,
    pub children: HashMap<String, ChildHandle>,
}

impl Agent {
    pub fn new(id: String, inbox: mpsc::Receiver<Message>) -> Self {
        Agent {
            id,
            tools: HashMap::new(),
            inbox,
            memory: None,
            llm: None,
            children: HashMap::new(),
        }
    }

    pub fn register_tool(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub async fn call_tool(&self, name: &str, input: &str) -> Result<String, ToolError> {
        if let Some(tool) = self.tools.get(name) {
            let input_value: Value =
                serde_json::from_str(input).map_err(|e| ToolError::InvalidInput(e.to_string()))?;
            let result = (*tool).execute(input_value).await?;
            Ok(result.to_string())
        } else {
            Err(ToolError::ExecutionFailed(format!(
                "Tool '{}' not found",
                name
            )))
        }
    }

    pub fn with_memory(mut self, memory: Box<dyn Memory>) -> Self {
        self.memory = Some(memory);
        self
    }

    pub fn with_llm(mut self, llm: Arc<dyn LLM>) -> Self {
        self.llm = Some(llm);
        self
    }

    pub async fn remember(&mut self, key: &str, value: serde_json::Value) -> Result<(), MemoryError> {
        match &mut self.memory {
            Some(mem) => mem.set(key, value).await,
            None => Err(MemoryError::StorageError("No memory attached".to_string())),
        }
    }

    pub async fn recall(&self, key: &str) -> Option<serde_json::Value> {
        match &self.memory {
            Some(mem) => mem.get(key).await.map(|e| e.value),
            None => None,
        }
    }

pub async fn forget(&mut self, key: &str) -> bool {
          match &mut self.memory {
              Some(mem) => mem.delete(key).await,
              None => false,
          }
      }

    pub fn list_tools_for_llm(&self) -> Vec<ToolSpec> {
        self.tools.values().map(|t| ToolSpec {
            name: t.name().to_string(),
            description: t.description().to_string(),
            parameters: t.schema(),
        }).collect()
    }

    /// Build memory context string for auto-injection into thinking
    async fn build_memory_context(&self, config: &Option<AutoMemoryConfig>) -> Option<String> {
        let config = config.as_ref()?;
        let memory = self.memory.as_ref()?;

        // Get keys (filtered by prefixes if specified)
        let all_keys: Vec<String> = if config.key_prefixes.is_empty() {
            memory.list_keys(None).await
        } else {
            let mut keys = Vec::new();
            for prefix in &config.key_prefixes {
                keys.extend(memory.list_keys(Some(prefix)).await);
            }
            keys
        };

        if all_keys.is_empty() {
            return None;
        }

        // Get entries with their timestamps for sorting
        let mut entries: Vec<(String, crate::memory::MemoryEntry)> = Vec::new();
        for key in &all_keys {
            if let Some(entry) = memory.get(key).await {
                entries.push((key.clone(), entry));
            }
        }

        // Sort by updated_at (recent first if include_recent, oldest first otherwise)
        if config.include_recent {
            entries.sort_by(|a, b| b.1.updated_at.cmp(&a.1.updated_at));
        } else {
            entries.sort_by(|a, b| a.1.updated_at.cmp(&b.1.updated_at));
        }

        // Limit to max_entries
        entries.truncate(config.max_entries);

        if entries.is_empty() {
            return None;
        }

        // Format as context string
        let mut context = String::from("Your memories:\n");
        for (key, entry) in entries {
            // Format value - stringify JSON nicely
            let value_str = match &entry.value {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            context.push_str(&format!("- {}: {}\n", key, value_str));
        }

        Some(context)
    }

    pub async fn think_with_options(&mut self, task: &str, options: ThinkOptions) -> Result<String, crate::error::AgentError> {
        let llm = self.llm.as_ref().ok_or_else(||
            crate::error::AgentError::LlmError("No LLM attached".to_string()))?;

        let tools = self.list_tools_for_llm();

        // Build auto-memory context if configured
        let memory_context = self.build_memory_context(&options.auto_memory).await;

        // Combine memory context with system prompt
        let effective_system_prompt = match (&memory_context, &options.system_prompt) {
            (Some(mem), Some(sys)) => Some(format!("{}\n\n{}", mem, sys)),
            (Some(mem), None) => Some(mem.clone()),
            (None, Some(sys)) => Some(sys.clone()),
            (None, None) => None,
        };

        // Build initial messages
        let mut messages: Vec<ChatMessage> = Vec::new();

        // Optional system prompt (with memory context prepended)
        if let Some(system) = &effective_system_prompt {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: Some(system.clone()),
                tool_call_id: None,
                tool_calls: None,
            });
        }

        // User task
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: Some(task.to_string()),
            tool_call_id: None,
            tool_calls: None,
        });
        
        // Agentic loop
        for iteration in 0..options.max_iterations {
            let response = llm.chat_complete(messages.clone(), Some(tools.clone())).await
                .map_err(|e| crate::error::AgentError::LlmError(e.message))?;
            
            // If no tool calls, we have a final response
            if response.tool_calls.is_empty() {
                let final_response = response.content.unwrap_or_else(|| "No response".to_string());
                
                // Apply reflection if configured
                if let Some(ref config) = options.reflection {
                    return self.reflect_and_revise(
                        task,
                        &final_response,
                        config,
                        &options,
                    ).await;
                }
                
                return Ok(final_response);
            }
            
            // Add assistant message with tool calls
            messages.push(ChatMessage {
                role: "assistant".to_string(),
                content: response.content.clone(),
                tool_call_id: None,
                tool_calls: Some(response.tool_calls.clone()),
            });
            
            // Execute each tool and add results
            for tool_call in &response.tool_calls {
                let result = self.call_tool(&tool_call.name, &tool_call.arguments.to_string()).await
                    .unwrap_or_else(|e| format!("Error: {}", e));
                
                messages.push(ChatMessage {
                    role: "tool".to_string(),
                    content: Some(result),
                    tool_call_id: Some(tool_call.id.clone()),
                    tool_calls: None,
                });
            }
        }
        
        Err(crate::error::AgentError::MaxIterationsExceeded(options.max_iterations))
    }

     pub async fn think(&mut self, task: &str) -> Result<String, crate::error::AgentError> {
        self.think_with_options(task, ThinkOptions::default()).await
    }

    /// Reflect on a response and potentially revise it
    async fn reflect_and_revise(
        &mut self,
        original_task: &str,
        response: &str,
        config: &ReflectionConfig,
        options: &ThinkOptions,
    ) -> Result<String, crate::error::AgentError> {
        let mut current_response = response.to_string();
        
        for _revision in 0..config.max_revisions {
            // Clone LLM for this scope to avoid borrow issues
            let llm = self.llm.clone().ok_or_else(|| 
                crate::error::AgentError::LlmError("No LLM attached".to_string()))?;
            
            // Ask LLM to reflect on the response
            let reflection_prompt = format!(
                "{}\n\nOriginal task: {}\n\nResponse to evaluate:\n{}\n\nRespond with either:\n- ACCEPTED: if the response is complete and correct\n- REVISE: <feedback> if changes are needed",
                config.prompt, original_task, current_response
            );
            
            let reflection_messages = vec![ChatMessage {
                role: "user".to_string(),
                content: Some(reflection_prompt),
                tool_call_id: None,
                tool_calls: None,
            }];
            
            let reflection = llm.chat_complete(reflection_messages, None).await
                .map_err(|e| crate::error::AgentError::LlmError(e.message))?;
            
            let reflection_text = reflection.content.unwrap_or_default();
            
            // Parse reflection result
            if reflection_text.to_uppercase().starts_with("ACCEPTED") {
                return Ok(current_response);
            }
            
            // Extract feedback and revise
            let feedback = if reflection_text.to_uppercase().starts_with("REVISE:") {
                reflection_text[7..].trim().to_string()
            } else {
                reflection_text.clone()
            };
            
            // Generate revised response
            let revision_prompt = format!(
                "Original task: {}\n\nYour previous response:\n{}\n\nFeedback:\n{}\n\nPlease provide an improved response.",
                original_task, current_response, feedback
            );
            
            let revision_options = ThinkOptions {
                max_iterations: options.max_iterations,
                system_prompt: options.system_prompt.clone(),
                reflection: None, // Don't recurse
                auto_memory: options.auto_memory.clone(),
            };
            
            current_response = self.run_agentic_loop(&revision_prompt, &revision_options).await?;
        }
        
        // Return final response after max revisions
        Ok(current_response)
    }
    
    /// Core agentic loop extracted for reuse
    async fn run_agentic_loop(
        &mut self,
        task: &str,
        options: &ThinkOptions,
    ) -> Result<String, crate::error::AgentError> {
        let llm = self.llm.as_ref().ok_or_else(||
            crate::error::AgentError::LlmError("No LLM attached".to_string()))?;

        let tools = self.list_tools_for_llm();

        // Build auto-memory context if configured
        let memory_context = self.build_memory_context(&options.auto_memory).await;

        // Combine memory context with system prompt
        let effective_system_prompt = match (&memory_context, &options.system_prompt) {
            (Some(mem), Some(sys)) => Some(format!("{}\n\n{}", mem, sys)),
            (Some(mem), None) => Some(mem.clone()),
            (None, Some(sys)) => Some(sys.clone()),
            (None, None) => None,
        };

        let mut messages: Vec<ChatMessage> = Vec::new();

        if let Some(system) = &effective_system_prompt {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: Some(system.clone()),
                tool_call_id: None,
                tool_calls: None,
            });
        }

        messages.push(ChatMessage {
            role: "user".to_string(),
            content: Some(task.to_string()),
            tool_call_id: None,
            tool_calls: None,
        });
        
        for _iteration in 0..options.max_iterations {
            let response = llm.chat_complete(messages.clone(), Some(tools.clone())).await
                .map_err(|e| crate::error::AgentError::LlmError(e.message))?;
            
            if response.tool_calls.is_empty() {
                return Ok(response.content.unwrap_or_else(|| "No response".to_string()));
            }
            
            messages.push(ChatMessage {
                role: "assistant".to_string(),
                content: response.content.clone(),
                tool_call_id: None,
                tool_calls: Some(response.tool_calls.clone()),
            });
            
            for tool_call in &response.tool_calls {
                let result = self.call_tool(&tool_call.name, &tool_call.arguments.to_string()).await
                    .unwrap_or_else(|e| format!("Error: {}", e));
                
                messages.push(ChatMessage {
                    role: "tool".to_string(),
                    content: Some(result),
                    tool_call_id: Some(tool_call.id.clone()),
                    tool_calls: None,
                });
            }
        }
        
        Err(crate::error::AgentError::MaxIterationsExceeded(options.max_iterations))
    }

    /// Helper to create a child agent with cloned tools/LLM
    fn create_child_agent(parent: &Agent, child_id: String) -> Agent {
        let (_, rx) = tokio::sync::mpsc::channel(32);
        let mut child = Agent {
            id: child_id,
            tools: parent.tools.clone(),  // Arc clones are cheap
            inbox: rx,
            memory: None,
            llm: parent.llm.clone(),  // Arc clone
            children: std::collections::HashMap::new(),
        };
        child
    }

    /// Spawn a child agent for a subtask
    pub fn spawn_child(&mut self, config: ChildConfig) -> String {
        let child_id = format!("{}-child-{}", self.id, self.children.len());
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        
        // Create child agent with inherited tools/LLM
        let mut child = Self::create_child_agent(self, child_id.clone());
        let task = config.task.clone();
        
        // Spawn the child task in background
        tokio::spawn(async move {
            let result = child.think(&task).await;
            let _ = result_tx.send(result.map_err(|e| e.to_string()));
        });
        
        // Store handle for parent to wait on
        let handle = ChildHandle::new(child_id.clone(), config.task, result_rx);
        self.children.insert(child_id.clone(), handle);
        
        child_id
    }

    /// Wait for a specific child to complete
    pub async fn wait_for_child(&mut self, child_id: &str) -> Result<String, String> {
        let handle = self.children.get_mut(child_id)
            .ok_or_else(|| format!("Child {} not found", child_id))?;
        
        if let Some(rx) = handle.result_rx.take() {
            match rx.await {
                Ok(Ok(result)) => {
                    handle.status = ChildStatus::Completed(result.clone());
                    Ok(result)
                }
                Ok(Err(e)) => {
                    handle.status = ChildStatus::Failed(e.clone());
                    Err(e)
                }
                Err(_) => {
                    handle.status = ChildStatus::Failed("Channel closed".to_string());
                    Err("Channel closed".to_string())
                }
            }
        } else {
            match &handle.status {
                ChildStatus::Completed(r) => Ok(r.clone()),
                ChildStatus::Failed(e) => Err(e.clone()),
                ChildStatus::Running => Err("Already waiting".to_string()),
            }
        }
    }

    /// Non-blocking check of child status
    pub fn poll_child(&self, child_id: &str) -> Option<&ChildStatus> {
        self.children.get(child_id).map(|h| &h.status)
    }

    /// Wait for all children to complete
    pub async fn wait_for_all_children(&mut self) -> Vec<Result<String, String>> {
        let child_ids: Vec<String> = self.children.keys().cloned().collect();
        let mut results = Vec::new();
        for id in child_ids {
            results.push(self.wait_for_child(&id).await);
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{AddTool, EchoTool};
    use crate::memory::InMemoryStore;
    use serde_json::json;

    fn create_test_agent(id: &str) -> Agent {
        let (_tx, rx) = mpsc::channel(32);
        Agent::new(id.to_string(), rx)
    }

    // =========================================================================
    // Tool registration tests
    // =========================================================================

    #[test]
    fn test_agent_new() {
        let agent = create_test_agent("test-agent");
        assert_eq!(agent.id, "test-agent");
        assert!(agent.tools.is_empty());
        assert!(agent.memory.is_none());
        assert!(agent.llm.is_none());
        assert!(agent.children.is_empty());
    }

    #[test]
    fn test_register_tool() {
        let mut agent = create_test_agent("test-agent");
        agent.register_tool(Arc::new(AddTool));

        assert_eq!(agent.tools.len(), 1);
        assert!(agent.tools.contains_key("add"));
    }

    #[test]
    fn test_register_multiple_tools() {
        let mut agent = create_test_agent("test-agent");
        agent.register_tool(Arc::new(AddTool));
        agent.register_tool(Arc::new(EchoTool));

        assert_eq!(agent.tools.len(), 2);
        assert!(agent.tools.contains_key("add"));
        assert!(agent.tools.contains_key("echo"));
    }

    #[test]
    fn test_list_tools_for_llm() {
        let mut agent = create_test_agent("test-agent");
        agent.register_tool(Arc::new(AddTool));
        agent.register_tool(Arc::new(EchoTool));

        let specs = agent.list_tools_for_llm();
        assert_eq!(specs.len(), 2);

        let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"add"));
        assert!(names.contains(&"echo"));
    }

    // =========================================================================
    // Tool calling tests
    // =========================================================================

    #[tokio::test]
    async fn test_call_tool_success() {
        let mut agent = create_test_agent("test-agent");
        agent.register_tool(Arc::new(AddTool));

        let result = agent.call_tool("add", r#"{"a": 2, "b": 3}"#).await.unwrap();
        assert!(result.contains("5"));
    }

    #[tokio::test]
    async fn test_call_tool_not_found() {
        let agent = create_test_agent("test-agent");

        let result = agent.call_tool("nonexistent", "{}").await;
        assert!(matches!(result, Err(ToolError::ExecutionFailed(_))));
        if let Err(ToolError::ExecutionFailed(msg)) = result {
            assert!(msg.contains("not found"));
        }
    }

    #[tokio::test]
    async fn test_call_tool_invalid_json() {
        let mut agent = create_test_agent("test-agent");
        agent.register_tool(Arc::new(AddTool));

        let result = agent.call_tool("add", "not valid json").await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_call_echo_tool() {
        let mut agent = create_test_agent("test-agent");
        agent.register_tool(Arc::new(EchoTool));

        let result = agent.call_tool("echo", r#"{"message": "hello"}"#).await.unwrap();
        assert!(result.contains("hello"));
    }

    // =========================================================================
    // Memory tests
    // =========================================================================

    #[tokio::test]
    async fn test_agent_with_memory() {
        let agent = create_test_agent("test-agent");
        let memory = Box::new(InMemoryStore::new());
        let agent = agent.with_memory(memory);

        assert!(agent.memory.is_some());
    }

    #[tokio::test]
    async fn test_remember_and_recall() {
        let agent = create_test_agent("test-agent");
        let memory = Box::new(InMemoryStore::new());
        let mut agent = agent.with_memory(memory);

        agent.remember("key1", json!("value1")).await.unwrap();

        let value = agent.recall("key1").await;
        assert_eq!(value, Some(json!("value1")));
    }

    #[tokio::test]
    async fn test_recall_nonexistent() {
        let agent = create_test_agent("test-agent");
        let memory = Box::new(InMemoryStore::new());
        let agent = agent.with_memory(memory);

        let value = agent.recall("nonexistent").await;
        assert!(value.is_none());
    }

    #[tokio::test]
    async fn test_recall_without_memory() {
        let agent = create_test_agent("test-agent");
        let value = agent.recall("key").await;
        assert!(value.is_none());
    }

    #[tokio::test]
    async fn test_remember_without_memory() {
        let mut agent = create_test_agent("test-agent");
        let result = agent.remember("key", json!("value")).await;
        assert!(matches!(result, Err(MemoryError::StorageError(_))));
    }

    #[tokio::test]
    async fn test_forget() {
        let agent = create_test_agent("test-agent");
        let memory = Box::new(InMemoryStore::new());
        let mut agent = agent.with_memory(memory);

        agent.remember("key1", json!("value1")).await.unwrap();
        assert!(agent.forget("key1").await);
        assert!(agent.recall("key1").await.is_none());
    }

    #[tokio::test]
    async fn test_forget_nonexistent() {
        let agent = create_test_agent("test-agent");
        let memory = Box::new(InMemoryStore::new());
        let mut agent = agent.with_memory(memory);

        assert!(!agent.forget("nonexistent").await);
    }

    #[tokio::test]
    async fn test_forget_without_memory() {
        let mut agent = create_test_agent("test-agent");
        assert!(!agent.forget("key").await);
    }

    // =========================================================================
    // ThinkOptions tests
    // =========================================================================

    #[test]
    fn test_think_options_default() {
        let opts = ThinkOptions::default();
        assert_eq!(opts.max_iterations, 10);
        assert!(opts.system_prompt.is_none());
        assert!(opts.reflection.is_none());
    }

    #[test]
    fn test_reflection_config_default() {
        let config = ReflectionConfig::default();
        assert!(!config.prompt.is_empty());
        assert_eq!(config.max_revisions, 1);
    }

    // =========================================================================
    // AutoMemoryConfig tests
    // =========================================================================

    #[test]
    fn test_auto_memory_config_default() {
        let config = AutoMemoryConfig::default();
        assert_eq!(config.max_entries, 10);
        assert!(config.include_recent);
        assert!(config.key_prefixes.is_empty());
    }

    #[test]
    fn test_auto_memory_config_custom() {
        let config = AutoMemoryConfig {
            max_entries: 5,
            include_recent: false,
            key_prefixes: vec!["user:".to_string(), "task:".to_string()],
        };
        assert_eq!(config.max_entries, 5);
        assert!(!config.include_recent);
        assert_eq!(config.key_prefixes.len(), 2);
    }

    #[test]
    fn test_think_options_with_auto_memory() {
        let opts = ThinkOptions {
            max_iterations: 5,
            system_prompt: Some("Be helpful".to_string()),
            reflection: None,
            auto_memory: Some(AutoMemoryConfig::default()),
        };
        assert!(opts.auto_memory.is_some());
        assert_eq!(opts.auto_memory.as_ref().unwrap().max_entries, 10);
    }

    #[test]
    fn test_think_options_default_has_no_auto_memory() {
        let opts = ThinkOptions::default();
        assert!(opts.auto_memory.is_none());
    }

    #[tokio::test]
    async fn test_build_memory_context_no_config() {
        let agent = create_test_agent("test-agent");
        let memory = Box::new(InMemoryStore::new());
        let mut agent = agent.with_memory(memory);

        // Store some memories
        agent.remember("key1", json!("value1")).await.unwrap();

        // No config - should return None
        let context = agent.build_memory_context(&None).await;
        assert!(context.is_none());
    }

    #[tokio::test]
    async fn test_build_memory_context_no_memory() {
        let agent = create_test_agent("test-agent");

        // Agent has no memory attached
        let config = Some(AutoMemoryConfig::default());
        let context = agent.build_memory_context(&config).await;
        assert!(context.is_none());
    }

    #[tokio::test]
    async fn test_build_memory_context_empty_memory() {
        let agent = create_test_agent("test-agent");
        let memory = Box::new(InMemoryStore::new());
        let agent = agent.with_memory(memory);

        let config = Some(AutoMemoryConfig::default());
        let context = agent.build_memory_context(&config).await;
        assert!(context.is_none());
    }

    #[tokio::test]
    async fn test_build_memory_context_formats_correctly() {
        let agent = create_test_agent("test-agent");
        let memory = Box::new(InMemoryStore::new());
        let mut agent = agent.with_memory(memory);

        agent.remember("name", json!("Alice")).await.unwrap();
        agent.remember("role", json!("Engineer")).await.unwrap();

        let config = Some(AutoMemoryConfig::default());
        let context = agent.build_memory_context(&config).await;

        assert!(context.is_some());
        let ctx = context.unwrap();
        assert!(ctx.starts_with("Your memories:\n"));
        assert!(ctx.contains("name: Alice"));
        assert!(ctx.contains("role: Engineer"));
    }

    #[tokio::test]
    async fn test_build_memory_context_respects_max_entries() {
        let agent = create_test_agent("test-agent");
        let memory = Box::new(InMemoryStore::new());
        let mut agent = agent.with_memory(memory);

        // Create 5 memories
        for i in 0..5 {
            agent.remember(&format!("key{}", i), json!(i)).await.unwrap();
        }

        // Limit to 2 entries
        let config = Some(AutoMemoryConfig {
            max_entries: 2,
            include_recent: true,
            key_prefixes: vec![],
        });
        let context = agent.build_memory_context(&config).await.unwrap();

        // Count the entries (lines starting with "- ")
        let entry_count = context.lines().filter(|l| l.starts_with("- ")).count();
        assert_eq!(entry_count, 2);
    }

    #[tokio::test]
    async fn test_build_memory_context_filters_by_prefix() {
        let agent = create_test_agent("test-agent");
        let memory = Box::new(InMemoryStore::new());
        let mut agent = agent.with_memory(memory);

        agent.remember("user:name", json!("Bob")).await.unwrap();
        agent.remember("user:email", json!("bob@example.com")).await.unwrap();
        agent.remember("config:theme", json!("dark")).await.unwrap();

        // Only get user: prefixed keys
        let config = Some(AutoMemoryConfig {
            max_entries: 10,
            include_recent: true,
            key_prefixes: vec!["user:".to_string()],
        });
        let context = agent.build_memory_context(&config).await.unwrap();

        assert!(context.contains("user:name"));
        assert!(context.contains("user:email"));
        assert!(!context.contains("config:theme"));
    }

    // =========================================================================
    // Think without LLM (error case)
    // =========================================================================

    #[tokio::test]
    async fn test_think_without_llm() {
        let mut agent = create_test_agent("test-agent");
        let result = agent.think("do something").await;
        assert!(matches!(result, Err(crate::error::AgentError::LlmError(_))));
    }

    #[tokio::test]
    async fn test_think_with_options_without_llm() {
        let mut agent = create_test_agent("test-agent");
        let result = agent.think_with_options("do something", ThinkOptions::default()).await;
        assert!(matches!(result, Err(crate::error::AgentError::LlmError(_))));
    }

    // =========================================================================
    // Child agent tests
    // =========================================================================

    #[test]
    fn test_poll_child_nonexistent() {
        let agent = create_test_agent("test-agent");
        assert!(agent.poll_child("nonexistent").is_none());
    }

    #[tokio::test]
    async fn test_wait_for_child_nonexistent() {
        let mut agent = create_test_agent("test-agent");
        let result = agent.wait_for_child("nonexistent").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[tokio::test]
    async fn test_wait_for_all_children_empty() {
        let mut agent = create_test_agent("test-agent");
        let results = agent.wait_for_all_children().await;
        assert!(results.is_empty());
    }
}
