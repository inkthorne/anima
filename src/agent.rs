use std::collections::HashMap;

use crate::error::ToolError;
use crate::tool::{Tool, ToolInfo};
use crate::message::Message;
use crate::memory::{Memory, MemoryError};
use crate::llm::{LLM, ChatMessage, ToolSpec, LLMError};
use tokio::sync::mpsc;
use serde_json::Value;

pub struct Agent {
    pub id: String,
    tools: HashMap<String, Box<dyn Tool>>,
    inbox: mpsc::Receiver<Message>,
    memory: Option<Box<dyn Memory>>,
    llm: Option<Box<dyn LLM>>,
}

impl Agent {
    pub fn new(id: String, inbox: mpsc::Receiver<Message>) -> Self {
        Agent {
            id,
            tools: HashMap::new(),
            inbox,
            memory: None,
            llm: None,
        }
    }

    pub fn register_tool(&mut self, tool: Box<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub async fn call_tool(&self, name: &str, input: &str) -> Result<String, ToolError> {
        if let Some(tool) = self.tools.get(name) {
            let input_value: Value =
                serde_json::from_str(input).map_err(|e| ToolError::InvalidInput(e.to_string()))?;
            let result = tool.execute(input_value).await?;
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

    pub fn with_llm(mut self, llm: Box<dyn LLM>) -> Self {
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

    pub async fn think(&mut self, task: &str) -> Result<String, crate::error::AgentError> {
        let llm = self.llm.as_ref().ok_or_else(|| 
            crate::error::AgentError::LlmError("No LLM attached".to_string()))?;
        
        let tools = self.list_tools_for_llm();
        let messages = vec![
            ChatMessage { role: "user".to_string(), content: task.to_string(), tool_call_id: None }
        ];
        
        let response = llm.chat_complete(messages, Some(tools)).await
            .map_err(|e| crate::error::AgentError::LlmError(e.message))?;
        
        // If LLM wants to call tools, execute them
        if !response.tool_calls.is_empty() {
            for tool_call in &response.tool_calls {
                let result = self.call_tool(&tool_call.name, &tool_call.arguments.to_string()).await
                    .map_err(|e| crate::error::AgentError::ToolError(e))?;
                // For now, just return the tool result
                return Ok(format!("Tool \"{}\" returned: {}", tool_call.name, result));
            }
        }
        
        // Return LLM's text response
        Ok(response.content.unwrap_or_else(|| "No response".to_string()))
    }
}
