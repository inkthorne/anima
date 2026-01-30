use serde::Deserialize;

fn default_backend() -> String {
    "in_memory".to_string()
}

fn default_max_iter() -> usize {
    10
}

fn default_mem_entries() -> usize {
    10
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
    pub think: ThinkSection,
}

#[derive(Debug, Deserialize)]
pub struct AgentSection {
    pub name: String,
    #[serde(default)]
    pub system_prompt: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct LlmSection {
    pub provider: String,  // "openai" or "anthropic"
    pub model: String,
    pub base_url: Option<String>,
    // API key from env: OPENAI_API_KEY or ANTHROPIC_API_KEY
}

#[derive(Debug, Deserialize, Default)]
pub struct ToolsSection {
    #[serde(default)]
    pub enabled: Vec<String>,  // ["read_file", "shell", etc]
}

#[derive(Debug, Deserialize)]
pub struct MemorySection {
    #[serde(default = "default_backend")]
    pub backend: String,  // "sqlite" or "in_memory"
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
}

impl Default for ThinkSection {
    fn default() -> Self {
        Self {
            max_iterations: default_max_iter(),
            auto_memory: false,
            max_memory_entries: default_mem_entries(),
            reflection: false,
        }
    }
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
"#;
        let config: AgentConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.agent.name, "full-agent");
        assert_eq!(config.agent.system_prompt, Some("You are helpful.".to_string()));
        assert_eq!(config.llm.provider, "anthropic");
        assert_eq!(config.llm.base_url, Some("https://custom.api.com".to_string()));
        assert_eq!(config.tools.enabled, vec!["echo", "add", "shell"]);
        assert_eq!(config.memory.backend, "sqlite");
        assert_eq!(config.memory.path, Some("/tmp/agent.db".to_string()));
        assert_eq!(config.think.max_iterations, 5);
        assert!(config.think.auto_memory);
        assert_eq!(config.think.max_memory_entries, 20);
        assert!(config.think.reflection);
    }
}
