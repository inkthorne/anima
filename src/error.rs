use std::fmt;

#[derive(Debug, Clone)]
pub enum ToolError {
    InvalidInput(String),
    ExecutionFailed(String),
}

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            ToolError::ExecutionFailed(msg) => write!(f, "Execution failed: {}", msg),
        }
    }
}

impl std::error::Error for ToolError {}

#[derive(Debug, Clone)]
pub enum AgentError {
    ToolNotFound(String),
    ToolError(ToolError),
    ChannelClosed,
    LlmError(String),
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentError::ToolNotFound(name) => write!(f, "Tool not found: {}", name),
            AgentError::ToolError(e) => write!(f, "Tool error: {}", e),
            AgentError::ChannelClosed => write!(f, "Channel closed"),
            AgentError::LlmError(e) => write!(f, "LLM error: {}", e),
        }
    }
}

impl std::error::Error for AgentError {}

impl From<ToolError> for AgentError {
    fn from(e: ToolError) -> Self {
        AgentError::ToolError(e)
    }
}
