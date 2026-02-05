pub mod agent;
pub mod agent_dir;
pub mod config;
pub mod conversation;
pub mod daemon;
pub mod debug;
pub mod discovery;
pub mod embedding;
pub mod error;
pub mod llm;
pub mod memory;
pub mod message;
pub mod messaging;
pub mod observe;
pub mod repl;
pub mod retry;
pub mod runtime;
pub mod socket_api;
pub mod supervision;
pub mod tool;
pub mod tool_registry;
pub mod tools;

// Re-export main types for convenience
pub use agent::{
    Agent, AutoMemoryConfig, ReflectionConfig, ReflectionResult, ThinkOptions, ThinkResult,
    ToolExecution,
};
pub use conversation::{
    Conversation, ConversationError, ConversationMessage, ConversationStore, NotifyResult,
    Participant, PendingNotification, expand_all_mention, generate_fun_name,
    notify_mentioned_agents, notify_mentioned_agents_fire_and_forget,
    notify_mentioned_agents_parallel, notify_mentioned_agents_parallel_owned, parse_mentions,
};
pub use embedding::{EmbeddingClient, EmbeddingError, cosine_similarity};
pub use error::{AgentError, ErrorContext, ToolError};
pub use llm::{
    AnthropicClient, ChatMessage, LLM, LLMError, LLMResponse, OllamaClient, OpenAIClient, ToolCall,
    ToolSpec, UsageInfo, strip_thinking_tags,
};
pub use memory::{
    InMemoryStore, Memory, MemoryEntry, MemoryError, SaveResult, SemanticMemoryEntry,
    SemanticMemoryStore, SqliteMemory, build_memory_injection, extract_remember_tags, format_age,
};
pub use message::Message;
pub use messaging::{AgentMailbox, AgentMessage, MessageRouter, MessagingError};
pub use observe::{
    ConsoleObserver, Event, MetricsCollector, MetricsSnapshot, MultiObserver, Observer,
};
pub use retry::{RetryPolicy, RetryResult, with_retry};
pub use runtime::Runtime;
pub use supervision::{ChildConfig, ChildHandle, ChildStatus};
pub use tool::{Tool, ToolInfo};
pub use tool_registry::{ToolDefinition, ToolRegistry, ToolRegistryError};
pub use tools::{ListAgentsTool, SendMessageTool};
