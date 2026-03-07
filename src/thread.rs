use std::sync::Arc;

use crate::agent_dir::{AgentDir, AgentDirError, create_llm_from_config};
use crate::llm::{ChatMessage, LLM, LLMError};

#[derive(Debug, thiserror::Error)]
pub enum ThreadError {
    #[error("Agent directory error: {0}")]
    AgentDir(#[from] AgentDirError),
    #[error("LLM error: {0}")]
    Llm(#[from] LLMError),
}

pub struct AnimaThread {
    agent_name: String,
    llm: Arc<dyn LLM>,
    system_prompt: Option<String>,
    history: Vec<ChatMessage>,
}

impl AnimaThread {
    /// Create from an agent directory.
    pub async fn from_agent_dir(agent_dir: &AgentDir) -> Result<Self, ThreadError> {
        let llm_config = agent_dir.resolve_llm_config()?;
        let api_key = AgentDir::api_key_for_config(&llm_config)?;
        let llm = create_llm_from_config(&llm_config, api_key).await?;
        let system_prompt = agent_dir.load_system()?;
        Ok(Self {
            agent_name: agent_dir.config.agent.name.clone(),
            llm,
            system_prompt,
            history: Vec::new(),
        })
    }

    /// Send a message and get a response.
    pub async fn send(&mut self, message: &str) -> Result<String, ThreadError> {
        let mut messages = Vec::new();
        if let Some(ref system) = self.system_prompt {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: Some(system.clone()),
                tool_call_id: None,
                tool_calls: None,
            });
        }
        messages.extend(self.history.clone());
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: Some(message.to_string()),
            tool_call_id: None,
            tool_calls: None,
        });

        let response = self.llm.chat_complete(messages, None).await?;
        let content = response.content.unwrap_or_default();

        self.history.push(ChatMessage {
            role: "user".to_string(),
            content: Some(message.to_string()),
            tool_call_id: None,
            tool_calls: None,
        });
        self.history.push(ChatMessage {
            role: "assistant".to_string(),
            content: Some(content.clone()),
            tool_call_id: None,
            tool_calls: None,
        });

        Ok(content)
    }

    pub fn agent_name(&self) -> &str {
        &self.agent_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_error_display() {
        let err = ThreadError::Llm(LLMError::permanent("test"));
        assert!(err.to_string().contains("LLM error"));
    }
}
