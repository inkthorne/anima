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
}

#[derive(Debug, Deserialize)]
pub struct LlmSection {
    pub provider: String,
    pub model: String,
    pub api_key: Option<String>,
    /// Enable thinking mode for Ollama models (default: None = false)
    #[serde(default)]
    pub thinking: Option<bool>,
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

#[derive(Debug, Deserialize)]
pub struct AgentSection {
    pub name: String,
    pub persona_file: Option<PathBuf>,
    pub always_file: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
pub struct AgentDirConfig {
    pub agent: AgentSection,
    pub llm: LlmSection,
    pub memory: Option<MemorySection>,
    pub timer: Option<TimerSection>,
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
    /// Resolution order:
    /// 1. Agent-specific: `<agent_dir>/always.md` (or configured always_file)
    /// 2. Global fallback: `~/.anima/agents/always.md`
    ///
    /// Agent-specific always.md completely overrides global (no merge).
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

        // Fall back to global always.md at ~/.anima/agents/always.md
        if let Some(global_agents_dir) = dirs::home_dir()
            .map(|h| h.join(".anima").join("agents"))
        {
            let global_always_path = global_agents_dir.join("always.md");
            if global_always_path.exists() {
                let content = std::fs::read_to_string(&global_always_path)?;
                let mut seen = HashSet::new();
                seen.insert(global_always_path.canonicalize().unwrap_or(global_always_path));
                // Use global agents directory as base path for includes
                let expanded = Self::expand_includes_with_base(&content, &global_agents_dir, &mut seen)?;
                return Ok(Some(expanded));
            }
        }

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

    /// Get the API key, expanding environment variables if needed.
    /// Falls back to provider-specific env vars (OPENAI_API_KEY, ANTHROPIC_API_KEY).
    pub fn api_key(&self) -> Result<Option<String>, AgentDirError> {
        match &self.config.llm.api_key {
            Some(key) => {
                // Expand env vars in the key (e.g., "${ANTHROPIC_API_KEY}")
                let expanded = Self::expand_env_vars(key)?;
                Ok(Some(expanded))
            }
            None => {
                // Fallback to provider-specific env var
                let env_var = match self.config.llm.provider.as_str() {
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

#[cfg(test)]
mod tests {
    use super::*;
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
        assert_eq!(agent_dir.config.llm.provider, "openai");
        assert_eq!(agent_dir.config.llm.model, "gpt-4");
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

    /// Test: Falls back to global when agent-specific doesn't exist
    #[test]
    fn test_load_always_global_fallback() {
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

        // Global should be used as fallback
        assert_eq!(always, Some("Global always content".to_string()));
    }

    /// Test: Returns None when neither agent-specific nor global exists
    #[test]
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

    /// Test: Global always.md supports include directives
    #[test]
    fn test_load_always_global_with_include() {
        // Create fake home directory with global always.md that uses includes
        let fake_home = tempdir().unwrap();
        let global_always_dir = fake_home.path().join(".anima").join("agents");
        fs::create_dir_all(&global_always_dir).unwrap();
        fs::write(
            global_always_dir.join("always.md"),
            "Global: {{include:rules.md}}",
        )
        .unwrap();
        fs::write(global_always_dir.join("rules.md"), "Be kind.").unwrap();

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

        // Includes should be expanded in global always.md
        assert_eq!(always, Some("Global: Be kind.".to_string()));
    }
}
