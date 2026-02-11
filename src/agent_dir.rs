use std::collections::HashSet;
use std::path::{Path, PathBuf};

use dirs;
use regex::Regex;
use serde::Deserialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AgentDirError {
    #[error("Config file not found: {0}")]
    ConfigNotFound(PathBuf),
    #[error("Failed to parse config: {0}")]
    ParseError(#[from] toml::de::Error),
    #[error("Environment variable not set: {0}")]
    EnvVarNotSet(String),
    #[error("Failed to read file: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Unsupported LLM provider: {0}")]
    UnsupportedProvider(String),
    #[error("LLM error: {0}")]
    LlmError(String),
    #[error("Memory error: {0}")]
    MemoryError(String),
    #[error("Include file not found: {0}")]
    IncludeNotFound(PathBuf),
    #[error("Include cycle detected: {0}")]
    IncludeCycle(PathBuf),
    #[error("Model file not found: {0}")]
    ModelFileNotFound(PathBuf),
    #[error("Model file missing required field '{field}' in {path}")]
    ModelFileMissingField { field: String, path: PathBuf },
}

/// LLM configuration section in agent config.
/// Can either reference a shared model file via `model_file`, or specify config directly.
#[derive(Debug, Deserialize, Clone, Default)]
pub struct LlmSection {
    /// Reference to a shared model file in ~/.anima/models/{name}.toml
    pub model_file: Option<String>,
    /// LLM provider (openai, anthropic, ollama). Required if not using model_file.
    pub provider: Option<String>,
    /// Model name. Required if not using model_file.
    pub model: Option<String>,
    /// API key (can use ${ENV_VAR} syntax)
    pub api_key: Option<String>,
    /// Base URL for the API (optional, for custom endpoints)
    pub base_url: Option<String>,
    /// Enable thinking mode for Ollama models (default: None = false)
    #[serde(default)]
    pub thinking: Option<bool>,
    /// Enable tool support. Set to false for models that don't support tools.
    #[serde(default)]
    pub tools: Option<bool>,
    /// Context window size (num_ctx) for Ollama models
    #[serde(default)]
    pub num_ctx: Option<u32>,
    /// Model-specific recall text appended to agent recall prompt
    #[serde(default)]
    pub recall: Option<String>,
    /// Allowlist of tool names. If set, only these tools are available.
    /// If None, all tools are allowed.
    #[serde(default)]
    pub allowed_tools: Option<Vec<String>>,
    /// Maximum tokens for Anthropic API responses
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

/// Resolved LLM configuration after loading model file and applying overrides.
/// All required fields are guaranteed to be present.
#[derive(Debug, Clone)]
pub struct ResolvedLlmConfig {
    pub provider: String,
    pub model: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub thinking: Option<bool>,
    pub tools: bool,
    pub num_ctx: Option<u32>,
    /// Model-specific recall text appended to agent recall prompt
    pub recall: Option<String>,
    /// Allowlist of tool names. If set, only these tools are available.
    /// If None, all tools are allowed.
    pub allowed_tools: Option<Vec<String>>,
    /// Maximum tokens for Anthropic API responses
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct MemorySection {
    pub path: PathBuf,
}

#[derive(Debug, Deserialize)]
pub struct TimerSection {
    pub enabled: bool,
    pub interval: String,
    pub message: Option<String>,
}

fn default_recall_limit() -> usize {
    10
}

fn default_history_limit() -> usize {
    20
}

fn default_min_importance() -> f64 {
    0.1
}

fn default_semantic_memory_path() -> String {
    "memory.db".to_string()
}

fn default_conversation_recall_limit() -> usize {
    3
}

fn default_embedding_url() -> String {
    std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string())
}

/// Configuration for embedding-based semantic search.
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingConfig {
    /// Embedding provider (currently only "ollama" is supported)
    pub provider: String,
    /// Embedding model name (e.g., "nomic-embed-text")
    pub model: String,
    /// Base URL for the embedding API (optional, defaults to localhost)
    #[serde(default = "default_embedding_url")]
    pub url: String,
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
    /// Embedding configuration for semantic search
    #[serde(default)]
    pub embedding: Option<EmbeddingConfig>,
    /// Maximum number of recalled user messages from conversation history per turn
    #[serde(default = "default_conversation_recall_limit")]
    pub conversation_recall_limit: usize,
}

impl Default for SemanticMemorySection {
    fn default() -> Self {
        Self {
            enabled: false,
            path: default_semantic_memory_path(),
            recall_limit: default_recall_limit(),
            history_limit: default_history_limit(),
            min_importance: default_min_importance(),
            embedding: None,
            conversation_recall_limit: default_conversation_recall_limit(),
        }
    }
}

#[derive(Debug, Deserialize, Default)]
pub struct ThinkSection {
    /// Maximum iterations for tool call loops (default: 10)
    pub max_iterations: Option<usize>,
    /// Maximum wall-clock time for a single notify response (e.g. "10m", "1h").
    pub max_response_time: Option<String>,
    /// Shell command to run after file-modifying tools (write_file, edit_file) for build verification.
    /// Result is injected into LLM context as ephemeral feedback. Example: "cd /project && cargo check 2>&1"
    pub verify_command: Option<String>,
    /// Timeout in seconds for verify_command (default: 30)
    pub verify_timeout: Option<u64>,
}

fn default_mentions_enabled() -> bool {
    true
}

#[derive(Debug, Deserialize)]
pub struct AgentSection {
    pub name: String,
    pub description: Option<String>,
    pub system_file: Option<PathBuf>,
    pub recall_file: Option<PathBuf>,
    #[serde(default = "default_mentions_enabled")]
    pub mentions: bool,
}

#[derive(Debug, Deserialize)]
pub struct AgentDirConfig {
    pub agent: AgentSection,
    pub llm: LlmSection,
    pub memory: Option<MemorySection>,
    pub timer: Option<TimerSection>,
    #[serde(default)]
    pub semantic_memory: SemanticMemorySection,
    #[serde(default)]
    pub heartbeat: crate::config::HeartbeatConfig,
    #[serde(default)]
    pub think: ThinkSection,
}

#[derive(Debug)]
pub struct AgentDir {
    pub path: PathBuf,
    pub config: AgentDirConfig,
}

impl AgentDir {
    /// Load an agent directory from the given path.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, AgentDirError> {
        let path = path.as_ref().to_path_buf();
        let config_path = path.join("config.toml");

        if !config_path.exists() {
            return Err(AgentDirError::ConfigNotFound(config_path));
        }

        let content = std::fs::read_to_string(&config_path)?;
        let config: AgentDirConfig = toml::from_str(&content)?;
        Ok(Self { path, config })
    }

    /// Expand ${VAR} patterns in a string with environment variable values.
    /// Returns an error if any referenced variable is not set.
    pub fn expand_env_vars(s: &str) -> Result<String, AgentDirError> {
        let re = Regex::new(r"\$\{([^}]+)\}").unwrap();
        let mut result = s.to_string();

        for cap in re.captures_iter(s) {
            let var_name = &cap[1];
            let value = std::env::var(var_name)
                .map_err(|_| AgentDirError::EnvVarNotSet(var_name.to_string()))?;
            result = result.replace(&cap[0], &value);
        }

        Ok(result)
    }

    /// Load system prompt content from the configured system file.
    /// Returns Ok(None) if no system file is configured.
    /// Expands {{include:filename}} directives recursively.
    pub fn load_system(&self) -> Result<Option<String>, AgentDirError> {
        let Some(system_file) = &self.config.agent.system_file else {
            return Ok(None);
        };
        let full_path = self.path.join(system_file);
        self.load_and_expand(&full_path).map(Some)
    }

    /// Load recall content from the configured recall file (or default `recall.md`).
    /// Returns Ok(None) if no recall file exists.
    /// Expands {{include:filename}} directives recursively.
    pub fn load_recall(&self) -> Result<Option<String>, AgentDirError> {
        let recall_path = match &self.config.agent.recall_file {
            Some(recall_file) => self.path.join(recall_file),
            None => self.path.join("recall.md"),
        };

        if !recall_path.exists() {
            return Ok(None);
        }

        self.load_and_expand(&recall_path).map(Some)
    }

    /// Read a file and recursively expand all {{include:...}} directives.
    fn load_and_expand(&self, file_path: &Path) -> Result<String, AgentDirError> {
        let content = std::fs::read_to_string(file_path)?;
        let mut seen = HashSet::new();
        seen.insert(
            file_path
                .canonicalize()
                .unwrap_or(file_path.to_path_buf()),
        );
        self.expand_includes(&content, &mut seen)
    }

    /// Expand {{include:filename}} patterns in content.
    /// Paths are relative to the agent directory.
    /// Detects cycles by tracking seen files.
    fn expand_includes(
        &self,
        content: &str,
        seen: &mut HashSet<PathBuf>,
    ) -> Result<String, AgentDirError> {
        Self::expand_includes_with_base(content, &self.path, seen)
    }

    /// Expand {{include:filename}} patterns in content with a custom base path.
    /// Paths are relative to the given base_path.
    /// Detects cycles by tracking seen files.
    fn expand_includes_with_base(
        content: &str,
        base_path: &Path,
        seen: &mut HashSet<PathBuf>,
    ) -> Result<String, AgentDirError> {
        let re = Regex::new(r"\{\{include:([^}]+)\}\}").unwrap();
        let mut result = content.to_string();

        let matches: Vec<(String, String)> = re
            .captures_iter(content)
            .map(|cap| (cap[0].to_string(), cap[1].to_string()))
            .collect();

        for (full_match, filename) in matches {
            let include_path = base_path.join(filename.trim());
            let canonical = include_path
                .canonicalize()
                .map_err(|_| AgentDirError::IncludeNotFound(include_path.clone()))?;

            if seen.contains(&canonical) {
                return Err(AgentDirError::IncludeCycle(canonical));
            }

            seen.insert(canonical.clone());

            let include_content = std::fs::read_to_string(&canonical)
                .map_err(|_| AgentDirError::IncludeNotFound(include_path))?;

            let expanded_include =
                Self::expand_includes_with_base(&include_content, base_path, seen)?;

            result = result.replace(&full_match, &expanded_include);
        }

        Ok(result)
    }

    /// Get the absolute path for the memory database.
    /// Returns None if no memory section is configured.
    pub fn memory_path(&self) -> Option<PathBuf> {
        self.config.memory.as_ref().map(|m| self.path.join(&m.path))
    }

    /// Resolve the LLM configuration, loading model file if specified.
    pub fn resolve_llm_config(&self) -> Result<ResolvedLlmConfig, AgentDirError> {
        resolve_llm_config(&self.config.llm)
    }

    /// Get the API key, expanding environment variables if needed.
    /// Falls back to provider-specific env vars (OPENAI_API_KEY, ANTHROPIC_API_KEY).
    pub fn api_key(&self) -> Result<Option<String>, AgentDirError> {
        let resolved = self.resolve_llm_config()?;
        Self::api_key_for_config(&resolved)
    }

    /// Get the API key for a resolved LLM config.
    pub fn api_key_for_config(
        config: &ResolvedLlmConfig,
    ) -> Result<Option<String>, AgentDirError> {
        if let Some(key) = &config.api_key {
            return Self::expand_env_vars(key).map(Some);
        }

        // Fallback to provider-specific env var
        let env_var = match config.provider.as_str() {
            "openai" => "OPENAI_API_KEY",
            "anthropic" => "ANTHROPIC_API_KEY",
            _ => return Ok(None),
        };
        Ok(std::env::var(env_var).ok())
    }
}

/// Get the anima base directory path (~/.anima/)
fn anima_dir() -> PathBuf {
    dirs::home_dir()
        .expect("Could not determine home directory")
        .join(".anima")
}

/// Get the agents directory path (~/.anima/agents/)
pub fn agents_dir() -> PathBuf {
    anima_dir().join("agents")
}

/// Get the models directory path (~/.anima/models/)
pub fn models_dir() -> PathBuf {
    anima_dir().join("models")
}

/// Load a shared model definition file from ~/.anima/models/{name}.toml
fn load_model_file(name: &str) -> Result<LlmSection, AgentDirError> {
    let model_path = models_dir().join(format!("{}.toml", name));

    if !model_path.exists() {
        return Err(AgentDirError::ModelFileNotFound(model_path));
    }

    let content = std::fs::read_to_string(&model_path)?;
    let config: LlmSection = toml::from_str(&content)?;
    Ok(config)
}

/// Resolve LLM configuration by loading model file (if specified) and applying overrides.
/// Returns a ResolvedLlmConfig with all required fields guaranteed.
pub fn resolve_llm_config(llm_section: &LlmSection) -> Result<ResolvedLlmConfig, AgentDirError> {
    let base = match &llm_section.model_file {
        Some(model_file) => load_model_file(model_file)?,
        None => llm_section.clone(),
    };

    // Helper to build a "missing field" error pointing at the model file path
    let missing_field = |field: &str| AgentDirError::ModelFileMissingField {
        field: field.to_string(),
        path: llm_section
            .model_file
            .as_ref()
            .map(|n| models_dir().join(format!("{}.toml", n)))
            .unwrap_or_default(),
    };

    // Agent config overrides model file for all fields
    let provider = llm_section
        .provider
        .clone()
        .or(base.provider.clone())
        .ok_or_else(|| missing_field("provider"))?;
    let model = llm_section
        .model
        .clone()
        .or(base.model.clone())
        .ok_or_else(|| missing_field("model"))?;

    Ok(ResolvedLlmConfig {
        provider,
        model,
        api_key: llm_section.api_key.clone().or(base.api_key.clone()),
        base_url: llm_section.base_url.clone().or(base.base_url.clone()),
        thinking: llm_section.thinking.or(base.thinking),
        tools: llm_section.tools.or(base.tools).unwrap_or(true),
        num_ctx: llm_section.num_ctx.or(base.num_ctx),
        recall: llm_section.recall.clone().or(base.recall.clone()),
        allowed_tools: llm_section
            .allowed_tools
            .clone()
            .or(base.allowed_tools.clone()),
        max_tokens: llm_section.max_tokens.or(base.max_tokens),
    })
}

/// Scaffold a new agent directory with config.toml, system.md, and recall.md templates.
/// Creates the directory at the specified path, or defaults to ~/.anima/agents/<name>/.
/// Returns an error if the directory already exists.
pub fn create_agent(name: &str, path: Option<PathBuf>) -> Result<(), AgentDirError> {
    let agent_path = path.unwrap_or_else(|| agents_dir().join(name));

    if agent_path.exists() {
        return Err(AgentDirError::IoError(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!("Agent directory already exists: {}", agent_path.display()),
        )));
    }

    std::fs::create_dir_all(&agent_path)?;

    let config_content = format!(
        r#"[agent]
name = "{name}"
system_file = "system.md"
recall_file = "recall.md"
# mentions = true

[llm]
provider = "anthropic"
model = "claude-sonnet-4-20250514"
api_key = "${{ANTHROPIC_API_KEY}}"

[memory]
path = "memory.db"

# Optional timer configuration
# [timer]
# enabled = true
# interval = "5m"
# message = "Heartbeat — check for anything interesting"
"#
    );
    std::fs::write(agent_path.join("config.toml"), config_content)?;

    let system_content = format!(
        r#"# {name}

You are {name}, an AI agent running in the Anima runtime.

## Personality

Be helpful, concise, and focused on the task at hand.

## Capabilities

You have access to tools for:
- Reading and writing files
- Making HTTP requests
- Running shell commands
- Sending messages to other agents

## Guidelines

- Think step by step before acting
- Use tools when needed to accomplish tasks
- Be proactive about using your memory to track important information
"#
    );
    std::fs::write(agent_path.join("system.md"), system_content)?;

    let recall_content = r#"# Recall

## How Conversations Work

When someone talks to you, **just respond naturally** — no @mention needed.

The conversation history shows who said what. The `from` field tells you the speaker.

## Responding

- **To the user:** Just respond. No @mention.
- **To another agent:** Use @mention to reach them.

## Other Agents

You can reach other agents by @mentioning them.

**Rule:** Use @name to talk TO someone. Use Name to talk ABOUT someone.

Only @mentions actually reach the other agent.

**Never @mention yourself.**

## Response Style
- Respond naturally with your actual answer
- If you @mention someone, include your message to them
- Dont repeat what others just said — add your own perspective
"#;
    std::fs::write(agent_path.join("recall.md"), recall_content)?;

    println!(
        "\x1b[32m✓ Created agent '{}' at {}\x1b[0m",
        name,
        agent_path.display()
    );
    println!();
    println!("  Files created:");
    println!("    \x1b[36mconfig.toml\x1b[0m  — agent configuration");
    println!("    \x1b[36msystem.md\x1b[0m    — system prompt / personality");
    println!("    \x1b[36mrecall.md\x1b[0m    — recall content (injected each turn)");
    println!();
    println!("  Next steps:");
    println!("    1. Edit config.toml to configure your LLM");
    println!("    2. Edit system.md to define your agent's personality");
    println!("    3. Edit recall.md to add recall content");
    println!("    4. Run with: \x1b[36manima run {}\x1b[0m", name);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::fs;
    use tempfile::tempdir;

    /// Write a minimal agent config.toml to the given directory.
    /// `agent_extra` is appended to the [agent] section.
    /// `llm_extra` is appended to the [llm] section.
    fn write_config(dir: &Path, agent_extra: &str, llm_extra: &str) {
        let content = format!(
            "[agent]\nname = \"test\"\n{agent_extra}\n[llm]\nprovider = \"openai\"\nmodel = \"gpt-4\"\n{llm_extra}"
        );
        fs::write(dir.join("config.toml"), content).unwrap();
    }

    /// Run a closure with HOME temporarily set to `fake_home`.
    /// Restores the original HOME after the closure completes (or panics).
    fn with_fake_home<F: FnOnce()>(fake_home: &Path, f: F) {
        let original = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", fake_home) };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        match original {
            Some(h) => unsafe { std::env::set_var("HOME", h) },
            None => unsafe { std::env::remove_var("HOME") },
        }
        if let Err(e) = result {
            std::panic::resume_unwind(e);
        }
    }

    #[test]
    fn test_load_agent_dir() {
        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test-agent"
system_file = "system.md"

[llm]
provider = "openai"
model = "gpt-4"
api_key = "sk-test"

[memory]
path = "memory.db"

[timer]
enabled = true
interval = "5m"
message = "heartbeat"
"#;
        fs::write(dir.path().join("config.toml"), config_content).unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();

        assert_eq!(agent_dir.config.agent.name, "test-agent");
        assert_eq!(
            agent_dir.config.agent.system_file,
            Some(PathBuf::from("system.md"))
        );
        assert_eq!(agent_dir.config.llm.provider, Some("openai".to_string()));
        assert_eq!(agent_dir.config.llm.model, Some("gpt-4".to_string()));
        assert_eq!(agent_dir.config.llm.api_key, Some("sk-test".to_string()));
        assert_eq!(
            agent_dir.config.memory.as_ref().unwrap().path,
            PathBuf::from("memory.db")
        );
        let timer = agent_dir.config.timer.as_ref().unwrap();
        assert!(timer.enabled);
        assert_eq!(timer.interval, "5m");
        assert_eq!(timer.message, Some("heartbeat".to_string()));
    }

    #[test]
    fn test_config_not_found() {
        let dir = tempdir().unwrap();
        let result = AgentDir::load(dir.path());
        assert!(matches!(result, Err(AgentDirError::ConfigNotFound(_))));
    }

    #[test]
    fn test_expand_env_vars() {
        unsafe { std::env::set_var("TEST_ANIMA_VAR", "hello") };
        let result = AgentDir::expand_env_vars("prefix_${TEST_ANIMA_VAR}_suffix").unwrap();
        assert_eq!(result, "prefix_hello_suffix");
        unsafe { std::env::remove_var("TEST_ANIMA_VAR") };
    }

    #[test]
    fn test_expand_env_vars_missing() {
        let result = AgentDir::expand_env_vars("${DEFINITELY_NOT_SET_12345}");
        assert!(matches!(result, Err(AgentDirError::EnvVarNotSet(_))));
    }

    #[test]
    fn test_expand_env_vars_multiple() {
        unsafe {
            std::env::set_var("TEST_A", "aaa");
            std::env::set_var("TEST_B", "bbb");
        }
        let result = AgentDir::expand_env_vars("${TEST_A}/${TEST_B}").unwrap();
        assert_eq!(result, "aaa/bbb");
        unsafe {
            std::env::remove_var("TEST_A");
            std::env::remove_var("TEST_B");
        }
    }

    #[test]
    fn test_load_system() {
        let dir = tempdir().unwrap();
        write_config(dir.path(), "system_file = \"system.md\"\n", "");
        fs::write(dir.path().join("system.md"), "I am a helpful assistant.").unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        assert_eq!(
            agent_dir.load_system().unwrap(),
            Some("I am a helpful assistant.".to_string())
        );
    }

    #[test]
    fn test_load_system_none() {
        let dir = tempdir().unwrap();
        write_config(dir.path(), "", "");

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        assert_eq!(agent_dir.load_system().unwrap(), None);
    }

    #[test]
    fn test_memory_path() {
        let dir = tempdir().unwrap();
        write_config(dir.path(), "", "\n[memory]\npath = \"data/memory.db\"\n");

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        assert_eq!(
            agent_dir.memory_path().unwrap(),
            dir.path().join("data/memory.db")
        );
    }

    #[test]
    fn test_memory_path_none() {
        let dir = tempdir().unwrap();
        write_config(dir.path(), "", "");

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        assert!(agent_dir.memory_path().is_none());
    }

    #[test]
    fn test_api_key_literal() {
        let dir = tempdir().unwrap();
        write_config(dir.path(), "", "api_key = \"sk-literal-key\"\n");

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        assert_eq!(
            agent_dir.api_key().unwrap(),
            Some("sk-literal-key".to_string())
        );
    }

    #[test]
    fn test_api_key_env_expansion() {
        let dir = tempdir().unwrap();
        unsafe { std::env::set_var("TEST_API_KEY", "sk-from-env") };
        write_config(dir.path(), "", "api_key = \"${TEST_API_KEY}\"\n");

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        assert_eq!(
            agent_dir.api_key().unwrap(),
            Some("sk-from-env".to_string())
        );
        unsafe { std::env::remove_var("TEST_API_KEY") };
    }

    #[test]
    fn test_load_system_with_include() {
        let dir = tempdir().unwrap();
        write_config(dir.path(), "system_file = \"system.md\"\n", "");
        fs::write(
            dir.path().join("system.md"),
            "Header\n{{include:SOUL.md}}\nFooter",
        )
        .unwrap();
        fs::write(dir.path().join("SOUL.md"), "I am the soul.").unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        assert_eq!(
            agent_dir.load_system().unwrap().unwrap(),
            "Header\nI am the soul.\nFooter"
        );
    }

    #[test]
    fn test_load_system_with_multiple_includes() {
        let dir = tempdir().unwrap();
        write_config(dir.path(), "system_file = \"system.md\"\n", "");
        fs::write(
            dir.path().join("system.md"),
            "{{include:IDENTITY.md}}\n---\n{{include:USER.md}}",
        )
        .unwrap();
        fs::write(dir.path().join("IDENTITY.md"), "I am Arya.").unwrap();
        fs::write(dir.path().join("USER.md"), "My user is Alice.").unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        assert_eq!(
            agent_dir.load_system().unwrap().unwrap(),
            "I am Arya.\n---\nMy user is Alice."
        );
    }

    #[test]
    fn test_load_system_with_nested_includes() {
        let dir = tempdir().unwrap();
        write_config(dir.path(), "system_file = \"system.md\"\n", "");
        fs::write(dir.path().join("system.md"), "{{include:outer.md}}").unwrap();
        fs::write(
            dir.path().join("outer.md"),
            "Outer[{{include:inner.md}}]Outer",
        )
        .unwrap();
        fs::write(dir.path().join("inner.md"), "INNER").unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        assert_eq!(
            agent_dir.load_system().unwrap().unwrap(),
            "Outer[INNER]Outer"
        );
    }

    #[test]
    fn test_load_system_include_not_found() {
        let dir = tempdir().unwrap();
        write_config(dir.path(), "system_file = \"system.md\"\n", "");
        fs::write(dir.path().join("system.md"), "{{include:nonexistent.md}}").unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        assert!(matches!(
            agent_dir.load_system(),
            Err(AgentDirError::IncludeNotFound(_))
        ));
    }

    #[test]
    fn test_load_system_include_cycle_self() {
        let dir = tempdir().unwrap();
        write_config(dir.path(), "system_file = \"system.md\"\n", "");
        fs::write(dir.path().join("system.md"), "{{include:system.md}}").unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        assert!(matches!(
            agent_dir.load_system(),
            Err(AgentDirError::IncludeCycle(_))
        ));
    }

    #[test]
    fn test_load_system_include_cycle_indirect() {
        let dir = tempdir().unwrap();
        write_config(dir.path(), "system_file = \"system.md\"\n", "");
        fs::write(dir.path().join("system.md"), "{{include:a.md}}").unwrap();
        fs::write(dir.path().join("a.md"), "{{include:b.md}}").unwrap();
        fs::write(dir.path().join("b.md"), "{{include:system.md}}").unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        assert!(matches!(
            agent_dir.load_system(),
            Err(AgentDirError::IncludeCycle(_))
        ));
    }

    #[test]
    fn test_load_system_no_includes() {
        let dir = tempdir().unwrap();
        write_config(dir.path(), "system_file = \"system.md\"\n", "");
        fs::write(
            dir.path().join("system.md"),
            "Plain content with no includes.",
        )
        .unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        assert_eq!(
            agent_dir.load_system().unwrap().unwrap(),
            "Plain content with no includes."
        );
    }

    #[test]
    fn test_load_system_include_with_whitespace() {
        let dir = tempdir().unwrap();
        write_config(dir.path(), "system_file = \"system.md\"\n", "");
        fs::write(dir.path().join("system.md"), "{{include: SOUL.md }}").unwrap();
        fs::write(dir.path().join("SOUL.md"), "Soul content").unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        assert_eq!(
            agent_dir.load_system().unwrap().unwrap(),
            "Soul content"
        );
    }

    #[test]
    fn test_load_recall() {
        let dir = tempdir().unwrap();
        write_config(
            dir.path(),
            "system_file = \"system.md\"\nrecall_file = \"recall.md\"\n",
            "",
        );
        fs::write(dir.path().join("system.md"), "I am a helpful assistant.").unwrap();
        fs::write(dir.path().join("recall.md"), "Always be concise.").unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        assert_eq!(
            agent_dir.load_recall().unwrap(),
            Some("Always be concise.".to_string())
        );
    }

    #[test]
    #[serial]
    fn test_load_recall_none() {
        let fake_home = tempdir().unwrap();
        fs::create_dir_all(fake_home.path().join(".anima").join("agents")).unwrap();

        let dir = tempdir().unwrap();
        write_config(dir.path(), "", "");

        with_fake_home(fake_home.path(), || {
            let agent_dir = AgentDir::load(dir.path()).unwrap();
            assert_eq!(agent_dir.load_recall().unwrap(), None);
        });
    }

    #[test]
    #[serial]
    fn test_load_recall_file_missing() {
        let fake_home = tempdir().unwrap();
        fs::create_dir_all(fake_home.path().join(".anima").join("agents")).unwrap();

        let dir = tempdir().unwrap();
        write_config(dir.path(), "recall_file = \"recall.md\"\n", "");
        // Note: recall.md file is NOT created

        with_fake_home(fake_home.path(), || {
            let agent_dir = AgentDir::load(dir.path()).unwrap();
            assert_eq!(agent_dir.load_recall().unwrap(), None);
        });
    }

    #[test]
    fn test_load_recall_with_include() {
        let dir = tempdir().unwrap();
        write_config(dir.path(), "recall_file = \"recall.md\"\n", "");
        fs::write(
            dir.path().join("recall.md"),
            "Header\n{{include:RULES.md}}\nFooter",
        )
        .unwrap();
        fs::write(dir.path().join("RULES.md"), "Be helpful.").unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        assert_eq!(
            agent_dir.load_recall().unwrap().unwrap(),
            "Header\nBe helpful.\nFooter"
        );
    }

    // =========================================================================
    // Global fallback tests for recall.md
    // =========================================================================

    #[test]
    fn test_load_recall_agent_specific_explicit() {
        let dir = tempdir().unwrap();
        write_config(dir.path(), "recall_file = \"my_recall.md\"\n", "");
        fs::write(
            dir.path().join("my_recall.md"),
            "Agent-specific recall content",
        )
        .unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        assert_eq!(
            agent_dir.load_recall().unwrap(),
            Some("Agent-specific recall content".to_string())
        );
    }

    #[test]
    fn test_load_recall_agent_specific_default() {
        let dir = tempdir().unwrap();
        write_config(dir.path(), "", "");
        fs::write(dir.path().join("recall.md"), "Default agent recall content").unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        assert_eq!(
            agent_dir.load_recall().unwrap(),
            Some("Default agent recall content".to_string())
        );
    }

    #[test]
    #[serial]
    fn test_load_recall_agent_overrides_global() {
        let fake_home = tempdir().unwrap();
        let global_dir = fake_home.path().join(".anima").join("agents");
        fs::create_dir_all(&global_dir).unwrap();
        fs::write(global_dir.join("recall.md"), "Global recall content").unwrap();

        let agent_dir_path = tempdir().unwrap();
        write_config(agent_dir_path.path(), "", "");
        fs::write(
            agent_dir_path.path().join("recall.md"),
            "Agent-specific recall",
        )
        .unwrap();

        with_fake_home(fake_home.path(), || {
            let agent_dir = AgentDir::load(agent_dir_path.path()).unwrap();
            assert_eq!(
                agent_dir.load_recall().unwrap(),
                Some("Agent-specific recall".to_string())
            );
        });
    }

    #[test]
    #[serial]
    fn test_load_recall_no_global_fallback() {
        let fake_home = tempdir().unwrap();
        let global_dir = fake_home.path().join(".anima").join("agents");
        fs::create_dir_all(&global_dir).unwrap();
        fs::write(global_dir.join("recall.md"), "Global recall content").unwrap();

        let agent_dir_path = tempdir().unwrap();
        write_config(agent_dir_path.path(), "", "");

        with_fake_home(fake_home.path(), || {
            let agent_dir = AgentDir::load(agent_dir_path.path()).unwrap();
            assert_eq!(agent_dir.load_recall().unwrap(), None);
        });
    }

    #[test]
    #[serial]
    fn test_load_recall_neither_exists() {
        let fake_home = tempdir().unwrap();
        fs::create_dir_all(fake_home.path().join(".anima").join("agents")).unwrap();

        let agent_dir_path = tempdir().unwrap();
        write_config(agent_dir_path.path(), "", "");

        with_fake_home(fake_home.path(), || {
            let agent_dir = AgentDir::load(agent_dir_path.path()).unwrap();
            assert_eq!(agent_dir.load_recall().unwrap(), None);
        });
    }

    #[test]
    #[serial]
    fn test_load_recall_agent_includes_global_shared() {
        let fake_home = tempdir().unwrap();
        let global_dir = fake_home.path().join(".anima").join("agents");
        fs::create_dir_all(&global_dir).unwrap();
        fs::write(global_dir.join("shared-tools.md"), "Tool instructions here").unwrap();

        let agent_dir_path = tempdir().unwrap();
        write_config(agent_dir_path.path(), "", "");
        let global_path = global_dir.join("shared-tools.md");
        fs::write(
            agent_dir_path.path().join("recall.md"),
            format!(
                "Agent header\n{{{{include:{}}}}}\nAgent footer",
                global_path.display()
            ),
        )
        .unwrap();

        let agent_dir = AgentDir::load(agent_dir_path.path()).unwrap();
        assert_eq!(
            agent_dir.load_recall().unwrap(),
            Some("Agent header\nTool instructions here\nAgent footer".to_string())
        );
    }

    // =========================================================================
    // Shared model definitions tests
    // =========================================================================

    /// Helper: create a fake ~/.anima/models/ directory and write a model file.
    fn write_model_file(fake_home: &Path, name: &str, content: &str) -> PathBuf {
        let dir = fake_home.join(".anima").join("models");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(format!("{name}.toml")), content).unwrap();
        dir
    }

    #[test]
    #[serial]
    fn test_model_file_loading() {
        let fake_home = tempdir().unwrap();
        write_model_file(
            fake_home.path(),
            "test-model",
            "provider = \"ollama\"\nmodel = \"gemma3:27b\"\nnum_ctx = 32768\ntools = false\nthinking = true\n",
        );

        let agent_dir_path = tempdir().unwrap();
        let config_content = "[agent]\nname = \"test\"\n\n[llm]\nmodel_file = \"test-model\"\n";
        fs::write(
            agent_dir_path.path().join("config.toml"),
            config_content,
        )
        .unwrap();

        with_fake_home(fake_home.path(), || {
            let agent_dir = AgentDir::load(agent_dir_path.path()).unwrap();
            let resolved = agent_dir.resolve_llm_config().unwrap();

            assert_eq!(resolved.provider, "ollama");
            assert_eq!(resolved.model, "gemma3:27b");
            assert_eq!(resolved.num_ctx, Some(32768));
            assert_eq!(resolved.tools, false);
            assert_eq!(resolved.thinking, Some(true));
        });
    }

    #[test]
    #[serial]
    fn test_model_file_with_overrides() {
        let fake_home = tempdir().unwrap();
        write_model_file(
            fake_home.path(),
            "base-model",
            "provider = \"ollama\"\nmodel = \"gemma3:27b\"\nnum_ctx = 32768\ntools = false\nthinking = false\n",
        );

        let agent_dir_path = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"

[llm]
model_file = "base-model"
num_ctx = 65536
thinking = true
"#;
        fs::write(
            agent_dir_path.path().join("config.toml"),
            config_content,
        )
        .unwrap();

        with_fake_home(fake_home.path(), || {
            let agent_dir = AgentDir::load(agent_dir_path.path()).unwrap();
            let resolved = agent_dir.resolve_llm_config().unwrap();

            assert_eq!(resolved.provider, "ollama");
            assert_eq!(resolved.model, "gemma3:27b");
            assert_eq!(resolved.tools, false);
            assert_eq!(resolved.num_ctx, Some(65536));
            assert_eq!(resolved.thinking, Some(true));
        });
    }

    #[test]
    fn test_legacy_direct_config() {
        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"

[llm]
provider = "anthropic"
model = "claude-sonnet-4-20250514"
api_key = "sk-test"
tools = true
"#;
        fs::write(dir.path().join("config.toml"), config_content).unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        let resolved = agent_dir.resolve_llm_config().unwrap();

        assert_eq!(resolved.provider, "anthropic");
        assert_eq!(resolved.model, "claude-sonnet-4-20250514");
        assert_eq!(resolved.api_key, Some("sk-test".to_string()));
        assert_eq!(resolved.tools, true);
    }

    #[test]
    #[serial]
    fn test_model_file_not_found() {
        let fake_home = tempdir().unwrap();
        fs::create_dir_all(fake_home.path().join(".anima").join("models")).unwrap();

        let agent_dir_path = tempdir().unwrap();
        let config_content = "[agent]\nname = \"test\"\n\n[llm]\nmodel_file = \"nonexistent-model\"\n";
        fs::write(
            agent_dir_path.path().join("config.toml"),
            config_content,
        )
        .unwrap();

        with_fake_home(fake_home.path(), || {
            let agent_dir = AgentDir::load(agent_dir_path.path()).unwrap();
            assert!(matches!(
                agent_dir.resolve_llm_config(),
                Err(AgentDirError::ModelFileNotFound(_))
            ));
        });
    }

    #[test]
    #[serial]
    fn test_model_file_missing_provider() {
        let fake_home = tempdir().unwrap();
        write_model_file(fake_home.path(), "incomplete", "model = \"gemma3:27b\"\n");

        let agent_dir_path = tempdir().unwrap();
        let config_content =
            "[agent]\nname = \"test\"\n\n[llm]\nmodel_file = \"incomplete\"\n";
        fs::write(
            agent_dir_path.path().join("config.toml"),
            config_content,
        )
        .unwrap();

        with_fake_home(fake_home.path(), || {
            let agent_dir = AgentDir::load(agent_dir_path.path()).unwrap();
            assert!(matches!(
                agent_dir.resolve_llm_config(),
                Err(AgentDirError::ModelFileMissingField { .. })
            ));
        });
    }

    #[test]
    #[serial]
    fn test_partial_overrides() {
        let fake_home = tempdir().unwrap();
        write_model_file(
            fake_home.path(),
            "full-model",
            "provider = \"ollama\"\nmodel = \"gemma3:27b\"\nnum_ctx = 32768\ntools = false\nthinking = true\nbase_url = \"http://localhost:11434\"\n",
        );

        let agent_dir_path = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"

[llm]
model_file = "full-model"
num_ctx = 65536
"#;
        fs::write(
            agent_dir_path.path().join("config.toml"),
            config_content,
        )
        .unwrap();

        with_fake_home(fake_home.path(), || {
            let agent_dir = AgentDir::load(agent_dir_path.path()).unwrap();
            let resolved = agent_dir.resolve_llm_config().unwrap();

            assert_eq!(resolved.provider, "ollama");
            assert_eq!(resolved.model, "gemma3:27b");
            assert_eq!(resolved.tools, false);
            assert_eq!(resolved.thinking, Some(true));
            assert_eq!(
                resolved.base_url,
                Some("http://localhost:11434".to_string())
            );
            assert_eq!(resolved.num_ctx, Some(65536));
        });
    }

    #[test]
    fn test_tools_default_true() {
        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"

[llm]
provider = "anthropic"
model = "claude-sonnet-4-20250514"
"#;
        fs::write(dir.path().join("config.toml"), config_content).unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        assert_eq!(agent_dir.resolve_llm_config().unwrap().tools, true);
    }

    #[test]
    #[serial]
    fn test_override_provider_and_model() {
        let fake_home = tempdir().unwrap();
        write_model_file(
            fake_home.path(),
            "ollama-base",
            "provider = \"ollama\"\nmodel = \"gemma3:27b\"\ntools = false\n",
        );

        let agent_dir_path = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"

[llm]
model_file = "ollama-base"
provider = "anthropic"
model = "claude-sonnet-4-20250514"
api_key = "sk-test"
tools = true
"#;
        fs::write(
            agent_dir_path.path().join("config.toml"),
            config_content,
        )
        .unwrap();

        with_fake_home(fake_home.path(), || {
            let agent_dir = AgentDir::load(agent_dir_path.path()).unwrap();
            let resolved = agent_dir.resolve_llm_config().unwrap();

            assert_eq!(resolved.provider, "anthropic");
            assert_eq!(resolved.model, "claude-sonnet-4-20250514");
            assert_eq!(resolved.api_key, Some("sk-test".to_string()));
            assert_eq!(resolved.tools, true);
        });
    }

    // =========================================================================
    // create_agent tests
    // =========================================================================

    #[test]
    fn test_create_agent_success() {
        let dir = tempdir().unwrap();
        let agent_path = dir.path().join("test-agent");

        let result = create_agent("test-agent", Some(agent_path.clone()));
        assert!(result.is_ok());

        assert!(agent_path.join("config.toml").exists());
        assert!(agent_path.join("system.md").exists());
        assert!(agent_path.join("recall.md").exists());

        let config_content = fs::read_to_string(agent_path.join("config.toml")).unwrap();
        assert!(config_content.contains("name = \"test-agent\""));
        assert!(config_content.contains("[llm]"));
        assert!(config_content.contains("[memory]"));
        assert!(config_content.contains("recall_file = \"recall.md\""));

        let system_content = fs::read_to_string(agent_path.join("system.md")).unwrap();
        assert!(system_content.contains("# test-agent"));
        assert!(system_content.contains("You are test-agent"));

        let recall_content = fs::read_to_string(agent_path.join("recall.md")).unwrap();
        assert!(recall_content.contains("# Recall"));
        assert!(recall_content.contains("How Conversations Work"));
        assert!(recall_content.contains("Never @mention yourself"));
    }

    #[test]
    fn test_mentions_default_true() {
        let dir = tempdir().unwrap();
        write_config(dir.path(), "", "");

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        assert!(agent_dir.config.agent.mentions);
    }

    #[test]
    fn test_mentions_disabled() {
        let dir = tempdir().unwrap();
        write_config(dir.path(), "mentions = false\n", "");

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        assert!(!agent_dir.config.agent.mentions);
    }

    #[test]
    fn test_create_agent_already_exists() {
        let dir = tempdir().unwrap();
        let agent_path = dir.path().join("existing-agent");

        fs::create_dir_all(&agent_path).unwrap();

        let result = create_agent("existing-agent", Some(agent_path));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }
}
