use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use regex::Regex;

use crate::daemon::AgentLogger;
use crate::llm::{ChatMessage, LLM, LLMError};

/// A model-driven pipeline that uses `<stage>` tags to load files on demand.
///
/// The model receives `instructions.md` (with `{{input}}` replaced) and can
/// emit `<stage>filename</stage>` tags to load files from the pipeline directory.
/// The loop continues until a response contains no `<stage>` tags.
#[derive(Debug, Clone)]
pub struct Pipeline {
    /// Content of `instructions.md` (with `{{input}}` placeholder)
    pub instructions: String,
    /// Path to the `pipeline/` directory
    pub pipeline_dir: PathBuf,
}

impl Pipeline {
    /// Load pipeline from `agent_dir/pipeline/instructions.md`.
    /// Returns `None` if the file doesn't exist.
    pub fn load(agent_dir: &Path) -> Option<Self> {
        let pipeline_dir = agent_dir.join("pipeline");
        let instructions_path = pipeline_dir.join("instructions.md");
        let instructions = std::fs::read_to_string(&instructions_path).ok()?;
        Some(Pipeline {
            instructions,
            pipeline_dir,
        })
    }

    /// Execute the pipeline by sending instructions to the LLM and looping
    /// on `<stage>` tags until the model produces a final response.
    pub async fn execute(
        &self,
        input: &str,
        llm: &Arc<dyn LLM>,
        logger: &AgentLogger,
        agent_dir: Option<&Path>,
        conv_name: Option<&str>,
    ) -> Result<String, LLMError> {
        let mut vars: HashMap<String, String> = HashMap::new();
        vars.insert("input".to_string(), input.to_string());

        let prompt = expand_vars(&self.instructions, &vars);

        let mut messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(prompt),
            tool_calls: None,
            tool_call_id: None,
        }];

        let mut iteration = 0;

        let model = llm.model_name().to_string();

        loop {
            let turn_n = agent_dir.and_then(|dir| {
                crate::debug::dump_request(
                    dir,
                    conv_name.unwrap_or("pipeline"),
                    &model,
                    &None,
                    &messages,
                )
            });

            let response = llm.chat_complete(messages.clone(), None).await?;

            if let Some(dir) = agent_dir {
                crate::debug::dump_response(
                    dir,
                    conv_name.unwrap_or("pipeline"),
                    turn_n,
                    &response,
                );
            }

            let content = response.content.unwrap_or_default();

            let (cleaned, stages) = extract_stage_tags(&content);

            // Extract XML vars from assistant output for use in later stages
            let new_vars = extract_xml_vars(&content);
            vars.extend(new_vars);

            if stages.is_empty() {
                // If the LLM failed to categorize on the first iteration,
                // fall back to default.md so it still gets proper instructions.
                if iteration == 0 {
                    let default_path = self.pipeline_dir.join("default.md");
                    if let Ok(default_content) = std::fs::read_to_string(&default_path) {
                        let expanded = expand_vars(&default_content, &vars);
                        logger.log("[pipeline] No <stage> tags on iteration 0, falling back to default.md");
                        messages.push(ChatMessage {
                            role: "assistant".to_string(),
                            content: Some(content),
                            tool_calls: None,
                            tool_call_id: None,
                        });
                        messages.push(ChatMessage {
                            role: "user".to_string(),
                            content: Some(expanded),
                            tool_calls: None,
                            tool_call_id: None,
                        });
                        iteration += 1;
                        continue;
                    }
                }

                logger.log(&format!(
                    "[pipeline] Final output after {} iterations: {}",
                    iteration,
                    truncate(&cleaned, 200)
                ));
                return Ok(cleaned);
            }

            logger.log(&format!(
                "[pipeline] Iteration {}: reading {:?}",
                iteration, stages
            ));

            // Push the assistant's response (with tags) into history
            messages.push(ChatMessage {
                role: "assistant".to_string(),
                content: Some(content),
                tool_calls: None,
                tool_call_id: None,
            });

            // Load requested files and expand variables
            let file_contents: Vec<String> = stages
                .iter()
                .map(|filename| {
                    let path = self.pipeline_dir.join(filename.trim());
                    match std::fs::read_to_string(&path) {
                        Ok(content) => expand_vars(&content, &vars),
                        Err(_) => format!("Error: file '{}' not found", filename.trim()),
                    }
                })
                .collect();

            let combined = file_contents.join("\n---\n");

            messages.push(ChatMessage {
                role: "user".to_string(),
                content: Some(combined),
                tool_calls: None,
                tool_call_id: None,
            });

            iteration += 1;
        }
    }
}

/// Extract `<stage>filename</stage>` tags from content.
/// Returns the cleaned content (tags stripped) and the list of filenames.
pub fn extract_stage_tags(content: &str) -> (String, Vec<String>) {
    let re = Regex::new(r"<stage>(.*?)</stage>").unwrap();
    let filenames: Vec<String> = re
        .captures_iter(content)
        .map(|cap| cap[1].to_string())
        .collect();
    let cleaned = re.replace_all(content, "").to_string();
    (cleaned, filenames)
}

/// Extract all `<tagname>content</tagname>` pairs from assistant output,
/// excluding `<stage>` tags (which are handled separately).
fn extract_xml_vars(content: &str) -> HashMap<String, String> {
    let open_re = Regex::new(r"<([a-zA-Z_][a-zA-Z0-9_]*)>").unwrap();
    let mut vars = HashMap::new();
    for cap in open_re.captures_iter(content) {
        let tag = &cap[1];
        if tag == "stage" {
            continue;
        }
        let close_tag = format!("</{}>", tag);
        let open_end = cap.get(0).unwrap().end();
        if let Some(close_pos) = content[open_end..].find(&close_tag) {
            let value = &content[open_end..open_end + close_pos];
            vars.insert(tag.to_string(), value.to_string());
        }
    }
    vars
}

/// Replace `{{varname}}` placeholders in text using the provided variable map.
fn expand_vars(text: &str, vars: &HashMap<String, String>) -> String {
    let re = Regex::new(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}").unwrap();
    re.replace_all(text, |caps: &regex::Captures| {
        let key = &caps[1];
        vars.get(key).cloned().unwrap_or_else(|| caps[0].to_string())
    })
    .to_string()
}

/// Truncate a string for logging, appending "..." if truncated.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_load_no_directory() {
        let tmp = TempDir::new().unwrap();
        assert!(Pipeline::load(tmp.path()).is_none());
    }

    #[test]
    fn test_load_no_instructions_md() {
        let tmp = TempDir::new().unwrap();
        let pipeline_dir = tmp.path().join("pipeline");
        fs::create_dir(&pipeline_dir).unwrap();
        fs::write(pipeline_dir.join("other.md"), "not instructions").unwrap();
        assert!(Pipeline::load(tmp.path()).is_none());
    }

    #[test]
    fn test_load_with_instructions() {
        let tmp = TempDir::new().unwrap();
        let pipeline_dir = tmp.path().join("pipeline");
        fs::create_dir(&pipeline_dir).unwrap();
        fs::write(
            pipeline_dir.join("instructions.md"),
            "Process: {{input}}",
        )
        .unwrap();

        let pipeline = Pipeline::load(tmp.path()).unwrap();
        assert_eq!(pipeline.instructions, "Process: {{input}}");
        assert_eq!(pipeline.pipeline_dir, pipeline_dir);
    }

    #[test]
    fn test_extract_stage_tags_none() {
        let (cleaned, stages) = extract_stage_tags("Hello world, no tags here.");
        assert_eq!(cleaned, "Hello world, no tags here.");
        assert!(stages.is_empty());
    }

    #[test]
    fn test_extract_stage_tags_single() {
        let (_, stages) = extract_stage_tags("Let me check <stage>questions.md</stage> for guidance.");
        assert_eq!(stages, vec!["questions.md"]);
    }

    #[test]
    fn test_extract_stage_tags_multiple() {
        let (_, stages) = extract_stage_tags(
            "Reading <stage>a.md</stage> and <stage>b.md</stage> now.",
        );
        assert_eq!(stages, vec!["a.md", "b.md"]);
    }

    #[test]
    fn test_extract_stage_tags_strips_from_content() {
        let (cleaned, _) =
            extract_stage_tags("Before <stage>file.md</stage> after");
        assert_eq!(cleaned, "Before  after");
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 5), "hello...");
    }

    #[test]
    fn test_extract_xml_vars_basic() {
        let vars = extract_xml_vars("<message>hello</message>");
        assert_eq!(vars.get("message").unwrap(), "hello");
    }

    #[test]
    fn test_extract_xml_vars_excludes_stage() {
        let vars = extract_xml_vars("<stage>foo</stage><message>bar</message>");
        assert!(!vars.contains_key("stage"));
        assert_eq!(vars.get("message").unwrap(), "bar");
    }

    #[test]
    fn test_extract_xml_vars_multiple() {
        let vars = extract_xml_vars("<intent>greet</intent> some text <tone>friendly</tone>");
        assert_eq!(vars.len(), 2);
        assert_eq!(vars.get("intent").unwrap(), "greet");
        assert_eq!(vars.get("tone").unwrap(), "friendly");
    }

    #[test]
    fn test_extract_xml_vars_none() {
        let vars = extract_xml_vars("No XML tags here at all.");
        assert!(vars.is_empty());
    }

    #[test]
    fn test_expand_vars() {
        let mut vars = HashMap::new();
        vars.insert("input".to_string(), "hello world".to_string());
        vars.insert("message".to_string(), "greetings".to_string());

        assert_eq!(expand_vars("User said: {{input}}", &vars), "User said: hello world");
        assert_eq!(expand_vars("Msg: {{message}}", &vars), "Msg: greetings");
        assert_eq!(expand_vars("{{unknown}}", &vars), "{{unknown}}");
        assert_eq!(expand_vars("no vars here", &vars), "no vars here");
    }
}
