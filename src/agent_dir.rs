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
    /// Model-specific always text appended to agent always prompt
    #[serde(default)]
    pub always: Option<String>,
    /// Allowlist of tool names. If set, only these tools are available.
    /// If None, all tools are allowed.
    #[serde(default)]
    pub allowed_tools: Option<Vec<String>>,
}

/// Resolved LLM configuration after loading model file and applying overrides.
/// This struct has all required fields guaranteed to be present.
#[derive(Debug, Clone)]
pub struct ResolvedLlmConfig {
    pub provider: String,
    pub model: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub thinking: Option<bool>,
    pub tools: bool,
    pub num_ctx: Option<u32>,
    /// Model-specific always text appended to agent always prompt
    pub always: Option<String>,
    /// Allowlist of tool names. If set, only these tools are available.
    /// If None, all tools are allowed.
    pub allowed_tools: Option<Vec<String>>,
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

fn default_min_importance() -> f64 {
    0.1
}

fn default_semantic_memory_path() -> String {
    "memory.db".to_string()
}

fn default_embedding_url() -> String {
    std::env::var("OLLAMA_HOST")
        .unwrap_or_else(|_| "http://localhost:11434".to_string())
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
    /// Minimum importance score to include (0.0-1.0)
    #[serde(default = "default_min_importance")]
    pub min_importance: f64,
    /// Embedding configuration for semantic search
    #[serde(default)]
    pub embedding: Option<EmbeddingConfig>,
}

impl Default for SemanticMemorySection {
    fn default() -> Self {
        Self {
            enabled: false,
            path: default_semantic_memory_path(),
            recall_limit: default_recall_limit(),
            min_importance: default_min_importance(),
            embedding: None,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct AgentSection {
    pub name: String,
    pub description: Option<String>,
    pub persona_file: Option<PathBuf>,
    pub always_file: Option<PathBuf>,
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

    /// Load persona content from the configured persona file.
    /// Returns Ok(None) if no persona file is configured.
    /// Expands {{include:filename}} directives recursively.
    pub fn load_persona(&self) -> Result<Option<String>, AgentDirError> {
        match &self.config.agent.persona_file {
            Some(persona_path) => {
                let full_path = self.path.join(persona_path);
                let content = std::fs::read_to_string(&full_path)?;
                let mut seen = HashSet::new();
                seen.insert(full_path.canonicalize().unwrap_or(full_path));
                let expanded = self.expand_includes(&content, &mut seen)?;
                Ok(Some(expanded))
            }
            None => Ok(None),
        }
    }

    /// Load always content from the configured always file.
    /// Returns Ok(None) if no always file exists.
    /// Expands {{include:filename}} directives recursively.
    ///
    /// Load agent's always.md file if it exists.
    /// Returns Ok(None) if no always.md exists in the agent directory.
    ///
    /// Global always.md files are available via `<!-- @include:path -->` directives
    /// if agents want to share common content.
    pub fn load_always(&self) -> Result<Option<String>, AgentDirError> {
        // Try agent-specific always file first
        let agent_always_path = match &self.config.agent.always_file {
            Some(always_path) => self.path.join(always_path),
            None => self.path.join("always.md"),
        };

        if agent_always_path.exists() {
            let content = std::fs::read_to_string(&agent_always_path)?;
            let mut seen = HashSet::new();
            seen.insert(agent_always_path.canonicalize().unwrap_or(agent_always_path));
            let expanded = self.expand_includes(&content, &mut seen)?;
            return Ok(Some(expanded));
        }

        // No agent-specific always.md — return None
        // (Global always.md is available via includes if agents want it)
        Ok(None)
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

        // Find all matches first to avoid borrowing issues
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

            // Recursively expand includes in the included content
            let expanded_include = Self::expand_includes_with_base(&include_content, base_path, seen)?;

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
    pub fn api_key_for_config(config: &ResolvedLlmConfig) -> Result<Option<String>, AgentDirError> {
        match &config.api_key {
            Some(key) => {
                // Expand env vars in the key (e.g., "${ANTHROPIC_API_KEY}")
                let expanded = Self::expand_env_vars(key)?;
                Ok(Some(expanded))
            }
            None => {
                // Fallback to provider-specific env var
                let env_var = match config.provider.as_str() {
                    "openai" => "OPENAI_API_KEY",
                    "anthropic" => "ANTHROPIC_API_KEY",
                    "ollama" => return Ok(None), // Ollama doesn't need an API key
                    _ => return Ok(None),
                };
                Ok(std::env::var(env_var).ok())
            }
        }
    }
}

/// Get the agents directory path (~/.anima/agents/)
pub fn agents_dir() -> PathBuf {
    dirs::home_dir()
        .expect("Could not determine home directory")
        .join(".anima")
        .join("agents")
}

/// Get the models directory path (~/.anima/models/)
pub fn models_dir() -> PathBuf {
    dirs::home_dir()
        .expect("Could not determine home directory")
        .join(".anima")
        .join("models")
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
    // If model_file is specified, load it and merge with overrides
    let base_config = if let Some(ref model_file) = llm_section.model_file {
        load_model_file(model_file)?
    } else {
        // No model file - use the section directly as base
        llm_section.clone()
    };

    // Apply overrides from agent config on top of base (model file or direct config)
    // For model_file case: agent's llm_section values override model file values
    // For direct config case: just use the values directly
    let provider = llm_section.provider.clone()
        .or(base_config.provider.clone())
        .ok_or_else(|| AgentDirError::ModelFileMissingField {
            field: "provider".to_string(),
            path: llm_section.model_file.as_ref()
                .map(|n| models_dir().join(format!("{}.toml", n)))
                .unwrap_or_default(),
        })?;

    let model = llm_section.model.clone()
        .or(base_config.model.clone())
        .ok_or_else(|| AgentDirError::ModelFileMissingField {
            field: "model".to_string(),
            path: llm_section.model_file.as_ref()
                .map(|n| models_dir().join(format!("{}.toml", n)))
                .unwrap_or_default(),
        })?;

    // For optional fields, agent config overrides model file
    let api_key = llm_section.api_key.clone().or(base_config.api_key.clone());
    let base_url = llm_section.base_url.clone().or(base_config.base_url.clone());
    let thinking = llm_section.thinking.or(base_config.thinking);
    let num_ctx = llm_section.num_ctx.or(base_config.num_ctx);
    let always = llm_section.always.clone().or(base_config.always.clone());
    let allowed_tools = llm_section.allowed_tools.clone().or(base_config.allowed_tools.clone());

    // tools: agent override takes precedence, then model file, then default true
    let tools = llm_section.tools
        .or(base_config.tools)
        .unwrap_or(true);

    Ok(ResolvedLlmConfig {
        provider,
        model,
        api_key,
        base_url,
        thinking,
        tools,
        num_ctx,
        always,
        allowed_tools,
    })
}

/// Scaffold a new agent directory with config.toml, persona.md, and always.md templates.
///
/// Creates the directory at the specified path, or defaults to ~/.anima/agents/<name>/.
/// Returns an error if the directory already exists.
pub fn create_agent(name: &str, path: Option<PathBuf>) -> Result<(), AgentDirError> {
    let agent_path = path.unwrap_or_else(|| agents_dir().join(name));

    // Check if directory already exists
    if agent_path.exists() {
        return Err(AgentDirError::IoError(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!("Agent directory already exists: {}", agent_path.display()),
        )));
    }

    // Create the directory
    std::fs::create_dir_all(&agent_path)?;

    // Write config.toml template
    let config_content = format!(r#"[agent]
name = "{name}"
persona_file = "persona.md"
always_file = "always.md"

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
"#);
    std::fs::write(agent_path.join("config.toml"), config_content)?;

    // Write persona.md template
    let persona_content = format!(r#"# {name}

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
"#);
    std::fs::write(agent_path.join("persona.md"), persona_content)?;

    // Write always.md template
    let always_content = r#"# Always

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
    std::fs::write(agent_path.join("always.md"), always_content)?;

    println!("\x1b[32m✓ Created agent '{}' at {}\x1b[0m", name, agent_path.display());
    println!();
    println!("  Files created:");
    println!("    \x1b[36mconfig.toml\x1b[0m  — agent configuration");
    println!("    \x1b[36mpersona.md\x1b[0m   — system prompt / personality");
    println!("    \x1b[36malways.md\x1b[0m    — persistent reminders (recency bias)");
    println!();
    println!("  Next steps:");
    println!("    1. Edit config.toml to configure your LLM");
    println!("    2. Edit persona.md to define your agent's personality");
    println!("    3. Edit always.md to add persistent reminders");
    println!("    4. Run with: \x1b[36manima run {}\x1b[0m", name);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_load_agent_dir() {
        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test-agent"
persona_file = "persona.md"

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
            agent_dir.config.agent.persona_file,
            Some(PathBuf::from("persona.md"))
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
    fn test_load_persona() {
        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"
persona_file = "persona.md"

[llm]
provider = "openai"
model = "gpt-4"
"#;
        fs::write(dir.path().join("config.toml"), config_content).unwrap();
        fs::write(dir.path().join("persona.md"), "I am a helpful assistant.").unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        let persona = agent_dir.load_persona().unwrap();
        assert_eq!(persona, Some("I am a helpful assistant.".to_string()));
    }

    #[test]
    fn test_load_persona_none() {
        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"

[llm]
provider = "openai"
model = "gpt-4"
"#;
        fs::write(dir.path().join("config.toml"), config_content).unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        let persona = agent_dir.load_persona().unwrap();
        assert_eq!(persona, None);
    }

    #[test]
    fn test_memory_path() {
        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"

[llm]
provider = "openai"
model = "gpt-4"

[memory]
path = "data/memory.db"
"#;
        fs::write(dir.path().join("config.toml"), config_content).unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        let mem_path = agent_dir.memory_path().unwrap();
        assert_eq!(mem_path, dir.path().join("data/memory.db"));
    }

    #[test]
    fn test_memory_path_none() {
        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"

[llm]
provider = "openai"
model = "gpt-4"
"#;
        fs::write(dir.path().join("config.toml"), config_content).unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        assert!(agent_dir.memory_path().is_none());
    }

    #[test]
    fn test_api_key_literal() {
        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"

[llm]
provider = "openai"
model = "gpt-4"
api_key = "sk-literal-key"
"#;
        fs::write(dir.path().join("config.toml"), config_content).unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        let key = agent_dir.api_key().unwrap();
        assert_eq!(key, Some("sk-literal-key".to_string()));
    }

    #[test]
    fn test_api_key_env_expansion() {
        let dir = tempdir().unwrap();
        unsafe { std::env::set_var("TEST_API_KEY", "sk-from-env") };
        let config_content = r#"
[agent]
name = "test"

[llm]
provider = "openai"
model = "gpt-4"
api_key = "${TEST_API_KEY}"
"#;
        fs::write(dir.path().join("config.toml"), config_content).unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        let key = agent_dir.api_key().unwrap();
        assert_eq!(key, Some("sk-from-env".to_string()));
        unsafe { std::env::remove_var("TEST_API_KEY") };
    }

    #[test]
    fn test_load_persona_with_include() {
        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"
persona_file = "persona.md"

[llm]
provider = "openai"
model = "gpt-4"
"#;
        fs::write(dir.path().join("config.toml"), config_content).unwrap();
        fs::write(
            dir.path().join("persona.md"),
            "Header\n{{include:SOUL.md}}\nFooter",
        )
        .unwrap();
        fs::write(dir.path().join("SOUL.md"), "I am the soul.").unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        let persona = agent_dir.load_persona().unwrap().unwrap();
        assert_eq!(persona, "Header\nI am the soul.\nFooter");
    }

    #[test]
    fn test_load_persona_with_multiple_includes() {
        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"
persona_file = "persona.md"

[llm]
provider = "openai"
model = "gpt-4"
"#;
        fs::write(dir.path().join("config.toml"), config_content).unwrap();
        fs::write(
            dir.path().join("persona.md"),
            "{{include:IDENTITY.md}}\n---\n{{include:USER.md}}",
        )
        .unwrap();
        fs::write(dir.path().join("IDENTITY.md"), "I am Arya.").unwrap();
        fs::write(dir.path().join("USER.md"), "My user is Alice.").unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        let persona = agent_dir.load_persona().unwrap().unwrap();
        assert_eq!(persona, "I am Arya.\n---\nMy user is Alice.");
    }

    #[test]
    fn test_load_persona_with_nested_includes() {
        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"
persona_file = "persona.md"

[llm]
provider = "openai"
model = "gpt-4"
"#;
        fs::write(dir.path().join("config.toml"), config_content).unwrap();
        fs::write(dir.path().join("persona.md"), "{{include:outer.md}}").unwrap();
        fs::write(
            dir.path().join("outer.md"),
            "Outer[{{include:inner.md}}]Outer",
        )
        .unwrap();
        fs::write(dir.path().join("inner.md"), "INNER").unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        let persona = agent_dir.load_persona().unwrap().unwrap();
        assert_eq!(persona, "Outer[INNER]Outer");
    }

    #[test]
    fn test_load_persona_include_not_found() {
        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"
persona_file = "persona.md"

[llm]
provider = "openai"
model = "gpt-4"
"#;
        fs::write(dir.path().join("config.toml"), config_content).unwrap();
        fs::write(
            dir.path().join("persona.md"),
            "{{include:nonexistent.md}}",
        )
        .unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        let result = agent_dir.load_persona();
        assert!(matches!(result, Err(AgentDirError::IncludeNotFound(_))));
    }

    #[test]
    fn test_load_persona_include_cycle_self() {
        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"
persona_file = "persona.md"

[llm]
provider = "openai"
model = "gpt-4"
"#;
        fs::write(dir.path().join("config.toml"), config_content).unwrap();
        fs::write(dir.path().join("persona.md"), "{{include:persona.md}}").unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        let result = agent_dir.load_persona();
        assert!(matches!(result, Err(AgentDirError::IncludeCycle(_))));
    }

    #[test]
    fn test_load_persona_include_cycle_indirect() {
        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"
persona_file = "persona.md"

[llm]
provider = "openai"
model = "gpt-4"
"#;
        fs::write(dir.path().join("config.toml"), config_content).unwrap();
        fs::write(dir.path().join("persona.md"), "{{include:a.md}}").unwrap();
        fs::write(dir.path().join("a.md"), "{{include:b.md}}").unwrap();
        fs::write(dir.path().join("b.md"), "{{include:persona.md}}").unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        let result = agent_dir.load_persona();
        assert!(matches!(result, Err(AgentDirError::IncludeCycle(_))));
    }

    #[test]
    fn test_load_persona_no_includes() {
        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"
persona_file = "persona.md"

[llm]
provider = "openai"
model = "gpt-4"
"#;
        fs::write(dir.path().join("config.toml"), config_content).unwrap();
        fs::write(dir.path().join("persona.md"), "Plain content with no includes.").unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        let persona = agent_dir.load_persona().unwrap().unwrap();
        assert_eq!(persona, "Plain content with no includes.");
    }

    #[test]
    fn test_load_persona_include_with_whitespace() {
        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"
persona_file = "persona.md"

[llm]
provider = "openai"
model = "gpt-4"
"#;
        fs::write(dir.path().join("config.toml"), config_content).unwrap();
        fs::write(dir.path().join("persona.md"), "{{include: SOUL.md }}").unwrap();
        fs::write(dir.path().join("SOUL.md"), "Soul content").unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        let persona = agent_dir.load_persona().unwrap().unwrap();
        assert_eq!(persona, "Soul content");
    }

    #[test]
    fn test_load_always() {
        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"
persona_file = "persona.md"
always_file = "always.md"

[llm]
provider = "openai"
model = "gpt-4"
"#;
        fs::write(dir.path().join("config.toml"), config_content).unwrap();
        fs::write(dir.path().join("persona.md"), "I am a helpful assistant.").unwrap();
        fs::write(dir.path().join("always.md"), "Always be concise.").unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        let always = agent_dir.load_always().unwrap();
        assert_eq!(always, Some("Always be concise.".to_string()));
    }

    #[test]
    #[serial]
    fn test_load_always_none() {
        // Create fake home directory WITHOUT global always.md
        let fake_home = tempdir().unwrap();
        let global_always_dir = fake_home.path().join(".anima").join("agents");
        fs::create_dir_all(&global_always_dir).unwrap();
        // NO always.md in global directory

        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"

[llm]
provider = "openai"
model = "gpt-4"
"#;
        fs::write(dir.path().join("config.toml"), config_content).unwrap();

        // Override HOME temporarily to avoid picking up real global always.md
        let original_home = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", fake_home.path()) };

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        let always = agent_dir.load_always().unwrap();

        // Restore HOME
        match original_home {
            Some(h) => unsafe { std::env::set_var("HOME", h) },
            None => unsafe { std::env::remove_var("HOME") },
        }

        assert_eq!(always, None);
    }

    #[test]
    #[serial]
    fn test_load_always_file_missing() {
        // Create fake home directory WITHOUT global always.md
        let fake_home = tempdir().unwrap();
        let global_always_dir = fake_home.path().join(".anima").join("agents");
        fs::create_dir_all(&global_always_dir).unwrap();
        // NO always.md in global directory

        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"
always_file = "always.md"

[llm]
provider = "openai"
model = "gpt-4"
"#;
        fs::write(dir.path().join("config.toml"), config_content).unwrap();
        // Note: always.md file is NOT created

        // Override HOME temporarily to avoid picking up real global always.md
        let original_home = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", fake_home.path()) };

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        // Should return None when file is missing (backward compatible)
        let always = agent_dir.load_always().unwrap();

        // Restore HOME
        match original_home {
            Some(h) => unsafe { std::env::set_var("HOME", h) },
            None => unsafe { std::env::remove_var("HOME") },
        }

        assert_eq!(always, None);
    }

    #[test]
    fn test_load_always_with_include() {
        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"
always_file = "always.md"

[llm]
provider = "openai"
model = "gpt-4"
"#;
        fs::write(dir.path().join("config.toml"), config_content).unwrap();
        fs::write(
            dir.path().join("always.md"),
            "Header\n{{include:RULES.md}}\nFooter",
        )
        .unwrap();
        fs::write(dir.path().join("RULES.md"), "Be helpful.").unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        let always = agent_dir.load_always().unwrap().unwrap();
        assert_eq!(always, "Header\nBe helpful.\nFooter");
    }

    // =========================================================================
    // Global fallback tests for always.md
    // =========================================================================

    /// Test: Agent-specific always.md is used when present (explicit config)
    #[test]
    fn test_load_always_agent_specific_explicit() {
        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"
always_file = "my_always.md"

[llm]
provider = "openai"
model = "gpt-4"
"#;
        fs::write(dir.path().join("config.toml"), config_content).unwrap();
        fs::write(dir.path().join("my_always.md"), "Agent-specific always content").unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        let always = agent_dir.load_always().unwrap();
        assert_eq!(always, Some("Agent-specific always content".to_string()));
    }

    /// Test: Default always.md in agent directory is used when no always_file configured
    #[test]
    fn test_load_always_agent_specific_default() {
        let dir = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"

[llm]
provider = "openai"
model = "gpt-4"
"#;
        fs::write(dir.path().join("config.toml"), config_content).unwrap();
        // Create always.md in agent directory (not configured explicitly)
        fs::write(dir.path().join("always.md"), "Default agent always content").unwrap();

        let agent_dir = AgentDir::load(dir.path()).unwrap();
        let always = agent_dir.load_always().unwrap();
        assert_eq!(always, Some("Default agent always content".to_string()));
    }

    /// Test: Agent-specific always.md overrides global (no merge)
    /// Note: This test creates a mock global always.md in a temp dir and temporarily
    /// overrides HOME. Since dirs::home_dir() uses the HOME env var on Unix, we can test this.
    #[test]
    #[serial]
    fn test_load_always_agent_overrides_global() {
        // Create fake home directory
        let fake_home = tempdir().unwrap();
        let global_always_dir = fake_home.path().join(".anima").join("agents");
        fs::create_dir_all(&global_always_dir).unwrap();
        fs::write(global_always_dir.join("always.md"), "Global always content").unwrap();

        // Create agent directory
        let agent_dir_path = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"

[llm]
provider = "openai"
model = "gpt-4"
"#;
        fs::write(agent_dir_path.path().join("config.toml"), config_content).unwrap();
        // Agent-specific always.md
        fs::write(agent_dir_path.path().join("always.md"), "Agent-specific always").unwrap();

        // Override HOME temporarily
        let original_home = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", fake_home.path()) };

        let agent_dir = AgentDir::load(agent_dir_path.path()).unwrap();
        let always = agent_dir.load_always().unwrap();

        // Restore HOME
        match original_home {
            Some(h) => unsafe { std::env::set_var("HOME", h) },
            None => unsafe { std::env::remove_var("HOME") },
        }

        // Agent-specific should be used, NOT global
        assert_eq!(always, Some("Agent-specific always".to_string()));
    }

    /// Test: Returns None when agent-specific doesn't exist (no global fallback)
    #[test]
    #[serial]
    fn test_load_always_no_global_fallback() {
        // Create fake home directory with global always.md
        let fake_home = tempdir().unwrap();
        let global_always_dir = fake_home.path().join(".anima").join("agents");
        fs::create_dir_all(&global_always_dir).unwrap();
        fs::write(global_always_dir.join("always.md"), "Global always content").unwrap();

        // Create agent directory WITHOUT always.md
        let agent_dir_path = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"

[llm]
provider = "openai"
model = "gpt-4"
"#;
        fs::write(agent_dir_path.path().join("config.toml"), config_content).unwrap();
        // NO always.md in agent directory

        // Override HOME temporarily
        let original_home = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", fake_home.path()) };

        let agent_dir = AgentDir::load(agent_dir_path.path()).unwrap();
        let always = agent_dir.load_always().unwrap();

        // Restore HOME
        match original_home {
            Some(h) => unsafe { std::env::set_var("HOME", h) },
            None => unsafe { std::env::remove_var("HOME") },
        }

        // Should return None — no global fallback
        assert_eq!(always, None);
    }

    /// Test: Returns None when neither agent-specific nor global exists
    #[test]
    #[serial]
    fn test_load_always_neither_exists() {
        // Create fake home directory WITHOUT global always.md
        let fake_home = tempdir().unwrap();
        let global_always_dir = fake_home.path().join(".anima").join("agents");
        fs::create_dir_all(&global_always_dir).unwrap();
        // NO always.md in global directory

        // Create agent directory WITHOUT always.md
        let agent_dir_path = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"

[llm]
provider = "openai"
model = "gpt-4"
"#;
        fs::write(agent_dir_path.path().join("config.toml"), config_content).unwrap();
        // NO always.md in agent directory

        // Override HOME temporarily
        let original_home = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", fake_home.path()) };

        let agent_dir = AgentDir::load(agent_dir_path.path()).unwrap();
        let always = agent_dir.load_always().unwrap();

        // Restore HOME
        match original_home {
            Some(h) => unsafe { std::env::set_var("HOME", h) },
            None => unsafe { std::env::remove_var("HOME") },
        }

        // Should return None when neither exists
        assert_eq!(always, None);
    }

    /// Test: Agent always.md can include shared files from global agents directory
    #[test]
    #[serial]
    fn test_load_always_agent_includes_global_shared() {
        // Create fake home directory with shared files
        let fake_home = tempdir().unwrap();
        let global_always_dir = fake_home.path().join(".anima").join("agents");
        fs::create_dir_all(&global_always_dir).unwrap();
        // Shared file that agents can include
        fs::write(global_always_dir.join("shared-tools.md"), "Tool instructions here").unwrap();

        // Create agent directory with always.md that includes the shared file
        let agent_dir_path = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"

[llm]
provider = "openai"
model = "gpt-4"
"#;
        fs::write(agent_dir_path.path().join("config.toml"), config_content).unwrap();
        // Agent always.md includes the global shared file
        let global_path = global_always_dir.join("shared-tools.md");
        fs::write(
            agent_dir_path.path().join("always.md"),
            format!("Agent header\n{{{{include:{}}}}}\nAgent footer", global_path.display()),
        )
        .unwrap();

        let agent_dir = AgentDir::load(agent_dir_path.path()).unwrap();
        let always = agent_dir.load_always().unwrap();

        // Includes should be expanded
        assert_eq!(always, Some("Agent header\nTool instructions here\nAgent footer".to_string()));
    }

    // =========================================================================
    // Shared model definitions tests
    // =========================================================================

    /// Test: Agent with model_file loads configuration from shared model file
    #[test]
    #[serial]
    fn test_model_file_loading() {
        // Create fake home directory with a model file
        let fake_home = tempdir().unwrap();
        let models_dir = fake_home.path().join(".anima").join("models");
        fs::create_dir_all(&models_dir).unwrap();
        fs::write(
            models_dir.join("test-model.toml"),
            r#"
provider = "ollama"
model = "gemma3:27b"
num_ctx = 32768
tools = false
thinking = true
"#,
        )
        .unwrap();

        // Create agent directory with model_file reference
        let agent_dir_path = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"

[llm]
model_file = "test-model"
"#;
        fs::write(agent_dir_path.path().join("config.toml"), config_content).unwrap();

        // Override HOME temporarily
        let original_home = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", fake_home.path()) };

        let agent_dir = AgentDir::load(agent_dir_path.path()).unwrap();
        let resolved = agent_dir.resolve_llm_config().unwrap();

        // Restore HOME
        match original_home {
            Some(h) => unsafe { std::env::set_var("HOME", h) },
            None => unsafe { std::env::remove_var("HOME") },
        }

        assert_eq!(resolved.provider, "ollama");
        assert_eq!(resolved.model, "gemma3:27b");
        assert_eq!(resolved.num_ctx, Some(32768));
        assert_eq!(resolved.tools, false);
        assert_eq!(resolved.thinking, Some(true));
    }

    /// Test: Agent overrides are applied on top of model file
    #[test]
    #[serial]
    fn test_model_file_with_overrides() {
        // Create fake home directory with a model file
        let fake_home = tempdir().unwrap();
        let models_dir = fake_home.path().join(".anima").join("models");
        fs::create_dir_all(&models_dir).unwrap();
        fs::write(
            models_dir.join("base-model.toml"),
            r#"
provider = "ollama"
model = "gemma3:27b"
num_ctx = 32768
tools = false
thinking = false
"#,
        )
        .unwrap();

        // Create agent directory with model_file reference AND overrides
        let agent_dir_path = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"

[llm]
model_file = "base-model"
num_ctx = 65536
thinking = true
"#;
        fs::write(agent_dir_path.path().join("config.toml"), config_content).unwrap();

        // Override HOME temporarily
        let original_home = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", fake_home.path()) };

        let agent_dir = AgentDir::load(agent_dir_path.path()).unwrap();
        let resolved = agent_dir.resolve_llm_config().unwrap();

        // Restore HOME
        match original_home {
            Some(h) => unsafe { std::env::set_var("HOME", h) },
            None => unsafe { std::env::remove_var("HOME") },
        }

        // Base values from model file
        assert_eq!(resolved.provider, "ollama");
        assert_eq!(resolved.model, "gemma3:27b");
        assert_eq!(resolved.tools, false);

        // Overridden values from agent config
        assert_eq!(resolved.num_ctx, Some(65536));
        assert_eq!(resolved.thinking, Some(true));
    }

    /// Test: Legacy config without model_file still works
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

    /// Test: Missing model file returns clear error
    #[test]
    #[serial]
    fn test_model_file_not_found() {
        // Create fake home directory WITHOUT the model file
        let fake_home = tempdir().unwrap();
        let models_dir = fake_home.path().join(".anima").join("models");
        fs::create_dir_all(&models_dir).unwrap();
        // NO test-model.toml file

        // Create agent directory with model_file reference
        let agent_dir_path = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"

[llm]
model_file = "nonexistent-model"
"#;
        fs::write(agent_dir_path.path().join("config.toml"), config_content).unwrap();

        // Override HOME temporarily
        let original_home = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", fake_home.path()) };

        let agent_dir = AgentDir::load(agent_dir_path.path()).unwrap();
        let result = agent_dir.resolve_llm_config();

        // Restore HOME
        match original_home {
            Some(h) => unsafe { std::env::set_var("HOME", h) },
            None => unsafe { std::env::remove_var("HOME") },
        }

        assert!(matches!(result, Err(AgentDirError::ModelFileNotFound(_))));
    }

    /// Test: Model file with missing required field returns clear error
    #[test]
    #[serial]
    fn test_model_file_missing_provider() {
        // Create fake home directory with incomplete model file
        let fake_home = tempdir().unwrap();
        let models_dir = fake_home.path().join(".anima").join("models");
        fs::create_dir_all(&models_dir).unwrap();
        fs::write(
            models_dir.join("incomplete.toml"),
            r#"
model = "gemma3:27b"
"#,
        )
        .unwrap();

        // Create agent directory with model_file reference
        let agent_dir_path = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"

[llm]
model_file = "incomplete"
"#;
        fs::write(agent_dir_path.path().join("config.toml"), config_content).unwrap();

        // Override HOME temporarily
        let original_home = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", fake_home.path()) };

        let agent_dir = AgentDir::load(agent_dir_path.path()).unwrap();
        let result = agent_dir.resolve_llm_config();

        // Restore HOME
        match original_home {
            Some(h) => unsafe { std::env::set_var("HOME", h) },
            None => unsafe { std::env::remove_var("HOME") },
        }

        assert!(matches!(result, Err(AgentDirError::ModelFileMissingField { .. })));
    }

    /// Test: Partial overrides - only specified fields are overridden
    #[test]
    #[serial]
    fn test_partial_overrides() {
        // Create fake home directory with a model file
        let fake_home = tempdir().unwrap();
        let models_dir = fake_home.path().join(".anima").join("models");
        fs::create_dir_all(&models_dir).unwrap();
        fs::write(
            models_dir.join("full-model.toml"),
            r#"
provider = "ollama"
model = "gemma3:27b"
num_ctx = 32768
tools = false
thinking = true
base_url = "http://localhost:11434"
"#,
        )
        .unwrap();

        // Create agent directory with only num_ctx override
        let agent_dir_path = tempdir().unwrap();
        let config_content = r#"
[agent]
name = "test"

[llm]
model_file = "full-model"
num_ctx = 65536
"#;
        fs::write(agent_dir_path.path().join("config.toml"), config_content).unwrap();

        // Override HOME temporarily
        let original_home = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", fake_home.path()) };

        let agent_dir = AgentDir::load(agent_dir_path.path()).unwrap();
        let resolved = agent_dir.resolve_llm_config().unwrap();

        // Restore HOME
        match original_home {
            Some(h) => unsafe { std::env::set_var("HOME", h) },
            None => unsafe { std::env::remove_var("HOME") },
        }

        // All fields from model file except num_ctx
        assert_eq!(resolved.provider, "ollama");
        assert_eq!(resolved.model, "gemma3:27b");
        assert_eq!(resolved.tools, false);
        assert_eq!(resolved.thinking, Some(true));
        assert_eq!(resolved.base_url, Some("http://localhost:11434".to_string()));

        // Only this field was overridden
        assert_eq!(resolved.num_ctx, Some(65536));
    }

    /// Test: tools defaults to true when not specified
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
        let resolved = agent_dir.resolve_llm_config().unwrap();

        assert_eq!(resolved.tools, true);
    }

    /// Test: Agent can override model to different provider
    #[test]
    #[serial]
    fn test_override_provider_and_model() {
        // Create fake home directory with an ollama model file
        let fake_home = tempdir().unwrap();
        let models_dir = fake_home.path().join(".anima").join("models");
        fs::create_dir_all(&models_dir).unwrap();
        fs::write(
            models_dir.join("ollama-base.toml"),
            r#"
provider = "ollama"
model = "gemma3:27b"
tools = false
"#,
        )
        .unwrap();

        // Create agent directory that overrides both provider and model
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
        fs::write(agent_dir_path.path().join("config.toml"), config_content).unwrap();

        // Override HOME temporarily
        let original_home = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", fake_home.path()) };

        let agent_dir = AgentDir::load(agent_dir_path.path()).unwrap();
        let resolved = agent_dir.resolve_llm_config().unwrap();

        // Restore HOME
        match original_home {
            Some(h) => unsafe { std::env::set_var("HOME", h) },
            None => unsafe { std::env::remove_var("HOME") },
        }

        // Everything is overridden
        assert_eq!(resolved.provider, "anthropic");
        assert_eq!(resolved.model, "claude-sonnet-4-20250514");
        assert_eq!(resolved.api_key, Some("sk-test".to_string()));
        assert_eq!(resolved.tools, true);
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

        // Check files were created
        assert!(agent_path.join("config.toml").exists());
        assert!(agent_path.join("persona.md").exists());
        assert!(agent_path.join("always.md").exists());

        // Check config.toml content
        let config_content = fs::read_to_string(agent_path.join("config.toml")).unwrap();
        assert!(config_content.contains("name = \"test-agent\""));
        assert!(config_content.contains("[llm]"));
        assert!(config_content.contains("[memory]"));
        assert!(config_content.contains("always_file = \"always.md\""));

        // Check persona.md content
        let persona_content = fs::read_to_string(agent_path.join("persona.md")).unwrap();
        assert!(persona_content.contains("# test-agent"));
        assert!(persona_content.contains("You are test-agent"));

        // Check always.md content
        let always_content = fs::read_to_string(agent_path.join("always.md")).unwrap();
        assert!(always_content.contains("# Always"));
        assert!(always_content.contains("How Conversations Work"));
        assert!(always_content.contains("Never @mention yourself"));
    }

    #[test]
    fn test_create_agent_already_exists() {
        let dir = tempdir().unwrap();
        let agent_path = dir.path().join("existing-agent");

        // Create the directory first
        fs::create_dir_all(&agent_path).unwrap();

        let result = create_agent("existing-agent", Some(agent_path));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }
}
