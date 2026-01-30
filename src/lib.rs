pub mod agent;
pub mod config;
pub mod error;
pub mod memory;
pub mod message;
pub mod runtime;
pub mod tool;
pub mod tools;
pub mod llm;
pub mod supervision;

// Re-export main types for convenience
pub use agent::{Agent, ThinkOptions, ReflectionConfig, ReflectionResult, AutoMemoryConfig};
pub use error::{AgentError, ToolError};
pub use memory::{Memory, MemoryEntry, MemoryError, InMemoryStore, SqliteMemory};
pub use message::Message;
pub use runtime::Runtime;
pub use tool::{Tool, ToolInfo};
pub use llm::{LLM, ChatMessage, ToolSpec, ToolCall, LLMResponse, LLMError, OpenAIClient, AnthropicClient};
pub use supervision::{ChildHandle, ChildConfig, ChildStatus};
