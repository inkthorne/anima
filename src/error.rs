use std::fmt;

/// Context for errors that occurred during retryable operations.
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Description of the operation that failed
    pub operation: String,
    /// Number of attempts made before final failure
    pub attempts: usize,
    /// The last error message
    pub last_error: String,
    /// Whether this error was classified as retryable
    pub retriable: bool,
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} failed after {} attempt(s): {}",
            self.operation, self.attempts, self.last_error
        )
    }
}

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
    MaxIterationsExceeded(usize),
    Cancelled,
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentError::ToolNotFound(name) => write!(f, "Tool not found: {}", name),
            AgentError::ToolError(e) => write!(f, "Tool error: {}", e),
            AgentError::ChannelClosed => write!(f, "Channel closed"),
            AgentError::LlmError(e) => write!(f, "LLM error: {}", e),
            AgentError::MaxIterationsExceeded(n) => write!(f, "Max iterations exceeded: {}", n),
            AgentError::Cancelled => write!(f, "Agent cancelled"),
        }
    }
}

impl std::error::Error for AgentError {}

impl From<ToolError> for AgentError {
    fn from(e: ToolError) -> Self {
        AgentError::ToolError(e)
    }
}
