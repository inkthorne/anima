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
}

impl Default for ThinkOptions {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            system_prompt: None,
        }
    }
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

    pub async fn think_with_options(&mut self, task: &str, options: ThinkOptions) -> Result<String, crate::error::AgentError> {
        let llm = self.llm.as_ref().ok_or_else(|| 
            crate::error::AgentError::LlmError("No LLM attached".to_string()))?;
        
        let tools = self.list_tools_for_llm();
        
        // Build initial messages
        let mut messages: Vec<ChatMessage> = Vec::new();
        
        // Optional system prompt
        if let Some(system) = &options.system_prompt {
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
            
            // If no tool calls, we're done - return the response
            if response.tool_calls.is_empty() {
                return Ok(response.content.unwrap_or_else(|| "No response".to_string()));
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
