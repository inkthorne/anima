use std::path::{Path, PathBuf};
use std::sync::Arc;

use regex::Regex;

use crate::daemon::AgentLogger;
use crate::llm::{ChatMessage, LLM, LLMError};

/// A model-driven pipeline that uses `<read>` tags to load files on demand.
///
/// The model receives `instructions.md` (with `{{input}}` replaced) and can
/// emit `<read>filename</read>` tags to load files from the pipeline directory.
/// The loop continues until a response contains no `<read>` tags.
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
    /// on `<read>` tags until the model produces a final response.
    pub async fn execute(
        &self,
        input: &str,
        llm: &Arc<dyn LLM>,
        logger: &AgentLogger,
    ) -> Result<String, LLMError> {
        let prompt = self.instructions.replace("{{input}}", input);

        let mut messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(prompt),
            tool_calls: None,
            tool_call_id: None,
        }];

        let mut iteration = 0;

        loop {
            let response = llm.chat_complete(messages.clone(), None).await?;
            let content = response.content.unwrap_or_default();

            let (cleaned, reads) = extract_read_tags(&content);

            if reads.is_empty() {
                // If the LLM failed to categorize on the first iteration,
                // fall back to default.md so it still gets proper instructions.
                if iteration == 0 {
                    let default_path = self.pipeline_dir.join("default.md");
                    if let Ok(default_content) = std::fs::read_to_string(&default_path) {
                        logger.log("[pipeline] No <read> tags on iteration 0, falling back to default.md");
                        messages.push(ChatMessage {
                            role: "assistant".to_string(),
                            content: Some(content),
                            tool_calls: None,
                            tool_call_id: None,
                        });
                        messages.push(ChatMessage {
                            role: "user".to_string(),
                            content: Some(default_content),
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
                iteration, reads
            ));

            // Push the assistant's response (with tags) into history
            messages.push(ChatMessage {
                role: "assistant".to_string(),
                content: Some(content),
                tool_calls: None,
                tool_call_id: None,
            });

            // Load requested files and send back as user message
            let file_contents: Vec<String> = reads
                .iter()
                .map(|filename| {
                    let path = self.pipeline_dir.join(filename.trim());
                    match std::fs::read_to_string(&path) {
                        Ok(content) => content,
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

/// Extract `<read>filename</read>` tags from content.
/// Returns the cleaned content (tags stripped) and the list of filenames.
pub fn extract_read_tags(content: &str) -> (String, Vec<String>) {
    let re = Regex::new(r"<read>(.*?)</read>").unwrap();
    let filenames: Vec<String> = re
        .captures_iter(content)
        .map(|cap| cap[1].to_string())
        .collect();
    let cleaned = re.replace_all(content, "").to_string();
    (cleaned, filenames)
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
    fn test_extract_read_tags_none() {
        let (cleaned, reads) = extract_read_tags("Hello world, no tags here.");
        assert_eq!(cleaned, "Hello world, no tags here.");
        assert!(reads.is_empty());
    }

    #[test]
    fn test_extract_read_tags_single() {
        let (_, reads) = extract_read_tags("Let me check <read>questions.md</read> for guidance.");
        assert_eq!(reads, vec!["questions.md"]);
    }

    #[test]
    fn test_extract_read_tags_multiple() {
        let (_, reads) = extract_read_tags(
            "Reading <read>a.md</read> and <read>b.md</read> now.",
        );
        assert_eq!(reads, vec!["a.md", "b.md"]);
    }

    #[test]
    fn test_extract_read_tags_strips_from_content() {
        let (cleaned, _) =
            extract_read_tags("Before <read>file.md</read> after");
        assert_eq!(cleaned, "Before  after");
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 5), "hello...");
    }
}
