use std::path::{Path, PathBuf};

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct LlmSection {
    pub provider: String,
    pub model: String,
    pub api_key: Option<String>,
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
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let path = path.as_ref().to_path_buf();
        let config_path = path.join("config.toml");
        let content = std::fs::read_to_string(&config_path)?;
        let config: AgentDirConfig = toml::from_str(&content)?;
        Ok(Self { path, config })
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
}
