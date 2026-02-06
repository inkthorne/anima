use serde::Deserialize;

fn default_backend() -> String {
    "in_memory".to_string()
}

fn default_max_iter() -> usize {
    10
}

fn default_max_checkpoints() -> usize {
    5
}

fn default_mem_entries() -> usize {
    10
}

fn default_max_retries() -> usize {
    3
}

fn default_initial_delay_ms() -> u64 {
    100
}

fn default_max_delay_ms() -> u64 {
    5000
}

fn default_exponential_base() -> f64 {
    2.0
}

fn default_recall_limit() -> usize {
    5
}

fn default_history_limit() -> usize {
    20
}

fn default_min_importance() -> f64 {
    0.1
}

#[derive(Debug, Deserialize)]
pub struct AgentConfig {
    pub agent: AgentSection,
    pub llm: LlmSection,
    #[serde(default)]
    pub tools: ToolsSection,
    #[serde(default)]
    pub memory: MemorySection,
    #[serde(default)]
    pub semantic_memory: SemanticMemorySection,
    #[serde(default)]
    pub think: ThinkSection,
    #[serde(default)]
    pub retry: RetrySection,
    #[serde(default)]
    pub observe: ObserveSection,
    #[serde(default)]
    pub timer: Option<TimerSection>,
    #[serde(default)]
    pub heartbeat: HeartbeatConfig,
}

#[derive(Debug, Deserialize)]
pub struct AgentSection {
    pub name: String,
    #[serde(default)]
    pub system_prompt: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct LlmSection {
    pub provider: String, // "openai" or "anthropic"
    pub model: String,
    pub base_url: Option<String>,
    /// Enable thinking mode for Ollama models (default: None = false)
    #[serde(default)]
    pub thinking: Option<bool>,
    /// Enable tool support (default: true). Set to false for models that don't support tools.
    #[serde(default = "default_tools_enabled")]
    pub tools: bool,
    // API key from env: OPENAI_API_KEY or ANTHROPIC_API_KEY
}

fn default_tools_enabled() -> bool {
    true
}

#[derive(Debug, Deserialize, Default)]
pub struct ToolsSection {
    #[serde(default)]
    pub enabled: Vec<String>, // ["read_file", "shell", etc]
}

#[derive(Debug, Deserialize)]
pub struct MemorySection {
    #[serde(default = "default_backend")]
    pub backend: String, // "sqlite" or "in_memory"
    pub path: Option<String>,
}

impl Default for MemorySection {
    fn default() -> Self {
        Self {
            backend: default_backend(),
            path: None,
        }
    }
}

/// Configuration for the semantic memory system.
#[derive(Debug, Clone, Deserialize)]
pub struct SemanticMemorySection {
    /// Enable semantic memory (default: false)
    #[serde(default)]
    pub enabled: bool,
    /// Path to the memory database (relative to agent directory)
    #[serde(default = "default_semantic_memory_path")]
    pub path: String,
    /// Maximum number of memories to inject per turn
    #[serde(default = "default_recall_limit")]
    pub recall_limit: usize,
    /// Maximum number of conversation messages to include in context
    #[serde(default = "default_history_limit")]
    pub history_limit: usize,
    /// Minimum importance score to include (0.0-1.0)
    #[serde(default = "default_min_importance")]
    pub min_importance: f64,
}

fn default_semantic_memory_path() -> String {
    "memory.db".to_string()
}

impl Default for SemanticMemorySection {
    fn default() -> Self {
        Self {
            enabled: false,
            path: default_semantic_memory_path(),
            recall_limit: default_recall_limit(),
            history_limit: default_history_limit(),
            min_importance: default_min_importance(),
        }
    }
}

/// Configuration for timer-based triggers.
#[derive(Debug, Deserialize)]
pub struct TimerSection {
    /// Enable timer triggers
    #[serde(default)]
    pub enabled: bool,
    /// Timer interval (e.g., "5m", "1h")
    #[serde(default)]
    pub interval: String,
    /// Message to send on timer trigger
    #[serde(default)]
    pub message: Option<String>,
}

/// Configuration for heartbeat functionality.
/// Agents can wake up periodically, read their heartbeat.md, think, and log output.
#[derive(Debug, Deserialize, Default, Clone)]
pub struct HeartbeatConfig {
    /// Enable heartbeat triggers
    #[serde(default)]
    pub enabled: bool,
    /// Heartbeat interval (e.g., "30s", "5m", "1h", "2h30m")
    pub interval: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ThinkSection {
    #[serde(default = "default_max_iter")]
    pub max_iterations: usize,
    #[serde(default)]
    pub auto_memory: bool,
    #[serde(default = "default_mem_entries")]
    pub max_memory_entries: usize,
    #[serde(default)]
    pub reflection: bool,
    /// Tool calls per checkpoint window. None = checkpoints disabled (default).
    #[serde(default)]
    pub checkpoint_interval: Option<usize>,
    /// Maximum number of checkpoint restarts before giving up (default: 5).
    #[serde(default = "default_max_checkpoints")]
    pub max_checkpoints: usize,
}

impl Default for ThinkSection {
    fn default() -> Self {
        Self {
            max_iterations: default_max_iter(),
            auto_memory: false,
            max_memory_entries: default_mem_entries(),
            reflection: false,
            checkpoint_interval: None,
            max_checkpoints: default_max_checkpoints(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct RetrySection {
    /// Maximum number of retry attempts (0 = no retries)
    #[serde(default = "default_max_retries")]
    pub max_retries: usize,
    /// Initial delay before first retry in milliseconds
    #[serde(default = "default_initial_delay_ms")]
    pub initial_delay_ms: u64,
    /// Maximum delay cap in milliseconds
    #[serde(default = "default_max_delay_ms")]
    pub max_delay_ms: u64,
    /// Base for exponential backoff (typically 2.0)
    #[serde(default = "default_exponential_base")]
    pub exponential_base: f64,
}

impl Default for RetrySection {
    fn default() -> Self {
        Self {
            max_retries: default_max_retries(),
            initial_delay_ms: default_initial_delay_ms(),
            max_delay_ms: default_max_delay_ms(),
            exponential_base: default_exponential_base(),
        }
    }
}

impl RetrySection {
    /// Convert to a RetryPolicy
    pub fn to_policy(&self) -> crate::retry::RetryPolicy {
        crate::retry::RetryPolicy {
            max_retries: self.max_retries,
            initial_delay_ms: self.initial_delay_ms,
            max_delay_ms: self.max_delay_ms,
            exponential_base: self.exponential_base,
        }
    }
}

#[derive(Debug, Deserialize, Default)]
pub struct ObserveSection {
    /// Print all events when true, only errors/completions when false
    #[serde(default)]
    pub verbose: bool,
}

impl AgentConfig {
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&content)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_config() {
        let toml = r#"
[agent]
name = "test-agent"

[llm]
provider = "openai"
model = "gpt-4o"
"#;
        let config: AgentConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.agent.name, "test-agent");
        assert_eq!(config.llm.provider, "openai");
        assert_eq!(config.llm.model, "gpt-4o");
        assert!(config.tools.enabled.is_empty());
        assert_eq!(config.memory.backend, "in_memory");
        assert_eq!(config.think.max_iterations, 10);
        // Checkpoint defaults
        assert!(config.think.checkpoint_interval.is_none());
        assert_eq!(config.think.max_checkpoints, 5);
        // Retry defaults
        assert_eq!(config.retry.max_retries, 3);
        assert_eq!(config.retry.initial_delay_ms, 100);
        assert_eq!(config.retry.max_delay_ms, 5000);
        // Observe defaults
        assert!(!config.observe.verbose);
        // LLM tools default to enabled
        assert!(config.llm.tools);
        // Semantic memory defaults
        assert!(!config.semantic_memory.enabled);
        assert_eq!(config.semantic_memory.path, "memory.db");
        assert_eq!(config.semantic_memory.recall_limit, 5);
        assert_eq!(config.semantic_memory.history_limit, 20);
        assert!((config.semantic_memory.min_importance - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_full_config() {
        let toml = r#"
[agent]
name = "full-agent"
system_prompt = "You are helpful."

[llm]
provider = "anthropic"
model = "claude-sonnet-4-20250514"
base_url = "https://custom.api.com"

[tools]
enabled = ["echo", "add", "shell"]

[memory]
backend = "sqlite"
path = "/tmp/agent.db"

[think]
max_iterations = 5
auto_memory = true
max_memory_entries = 20
reflection = true

[retry]
max_retries = 5
initial_delay_ms = 200
max_delay_ms = 10000
exponential_base = 3.0

[observe]
verbose = true
"#;
        let config: AgentConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.agent.name, "full-agent");
        assert_eq!(
            config.agent.system_prompt,
            Some("You are helpful.".to_string())
        );
        assert_eq!(config.llm.provider, "anthropic");
        assert_eq!(
            config.llm.base_url,
            Some("https://custom.api.com".to_string())
        );
        assert_eq!(config.tools.enabled, vec!["echo", "add", "shell"]);
        assert_eq!(config.memory.backend, "sqlite");
        assert_eq!(config.memory.path, Some("/tmp/agent.db".to_string()));
        assert_eq!(config.think.max_iterations, 5);
        assert!(config.think.auto_memory);
        assert_eq!(config.think.max_memory_entries, 20);
        assert!(config.think.reflection);
        // Retry config
        assert_eq!(config.retry.max_retries, 5);
        assert_eq!(config.retry.initial_delay_ms, 200);
        assert_eq!(config.retry.max_delay_ms, 10000);
        // Observe config
        assert!(config.observe.verbose);
        assert!((config.retry.exponential_base - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_tools_disabled() {
        let toml = r#"
[agent]
name = "no-tools-agent"

[llm]
provider = "ollama"
model = "gemma3:27b"
tools = false
"#;
        let config: AgentConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.agent.name, "no-tools-agent");
        assert_eq!(config.llm.provider, "ollama");
        assert_eq!(config.llm.model, "gemma3:27b");
        assert!(!config.llm.tools);
    }

    #[test]
    fn test_parse_semantic_memory_config() {
        let toml = r#"
[agent]
name = "memory-agent"

[llm]
provider = "openai"
model = "gpt-4o"

[semantic_memory]
enabled = true
path = "semantic.db"
recall_limit = 10
history_limit = 50
min_importance = 0.2
"#;
        let config: AgentConfig = toml::from_str(toml).unwrap();
        assert!(config.semantic_memory.enabled);
        assert_eq!(config.semantic_memory.path, "semantic.db");
        assert_eq!(config.semantic_memory.recall_limit, 10);
        assert_eq!(config.semantic_memory.history_limit, 50);
        assert!((config.semantic_memory.min_importance - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_timer_config() {
        let toml = r#"
[agent]
name = "timer-agent"

[llm]
provider = "openai"
model = "gpt-4o"

[timer]
enabled = true
interval = "5m"
message = "heartbeat"
"#;
        let config: AgentConfig = toml::from_str(toml).unwrap();
        assert!(config.timer.is_some());
        let timer = config.timer.unwrap();
        assert!(timer.enabled);
        assert_eq!(timer.interval, "5m");
        assert_eq!(timer.message, Some("heartbeat".to_string()));
    }

    #[test]
    fn test_parse_heartbeat_config() {
        let toml = r#"
[agent]
name = "heartbeat-agent"

[llm]
provider = "openai"
model = "gpt-4o"

[heartbeat]
enabled = true
interval = "30m"
"#;
        let config: AgentConfig = toml::from_str(toml).unwrap();
        assert!(config.heartbeat.enabled);
        assert_eq!(config.heartbeat.interval, Some("30m".to_string()));
    }

    #[test]
    fn test_heartbeat_config_defaults() {
        let toml = r#"
[agent]
name = "no-heartbeat-agent"

[llm]
provider = "openai"
model = "gpt-4o"
"#;
        let config: AgentConfig = toml::from_str(toml).unwrap();
        assert!(!config.heartbeat.enabled);
        assert!(config.heartbeat.interval.is_none());
    }

    #[test]
    fn test_parse_checkpoint_config() {
        let toml = r#"
[agent]
name = "checkpoint-agent"

[llm]
provider = "ollama"
model = "qwen3-coder:30b"
tools = false

[think]
max_iterations = 100
checkpoint_interval = 20
max_checkpoints = 3
"#;
        let config: AgentConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.think.max_iterations, 100);
        assert_eq!(config.think.checkpoint_interval, Some(20));
        assert_eq!(config.think.max_checkpoints, 3);
    }
}
