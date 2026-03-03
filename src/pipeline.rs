use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use regex::Regex;

use crate::daemon::AgentLogger;
use crate::llm::{ChatMessage, LLM, LLMError};

/// Result of a pipeline execution.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// The final response text (with tags stripped).
    pub response: String,
    /// Whether the pipeline emitted a `<handoff/>` tag, indicating the daemon
    /// should enter the tool loop with `response` as context.
    pub handoff: bool,
}

/// A model-driven pipeline that uses `<next-state>` tags to load files on demand.
///
/// The model receives `instructions.md` (with `{{input}}` replaced) and can
/// emit `<next-state>filename</next-state>` tags to load files from the pipeline directory.
/// The loop continues until a response contains no `<next-state>` tags.
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
    ) -> Result<PipelineResult, LLMError> {
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

            let (cleaned, stages) = extract_state_tags(&content);

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
                        logger.log("[pipeline] No <next-state> tags on iteration 0, falling back to default.md");
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

                let (cleaned, handoff) = extract_handoff_tag(&cleaned);
                logger.log(&format!(
                    "[pipeline] Final output after {} iterations (handoff={}): {}",
                    iteration,
                    handoff,
                    truncate(&cleaned, 200)
                ));
                return Ok(PipelineResult {
                    response: cleaned,
                    handoff,
                });
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

/// Extract `<next-state>filename</next-state>` tags from content.
/// Returns the cleaned content (tags stripped) and the list of filenames.
pub fn extract_state_tags(content: &str) -> (String, Vec<String>) {
    let re = Regex::new(r"(?s)<next-state>(.*?)</next-state>").unwrap();
    let filenames: Vec<String> = re
        .captures_iter(content)
        .map(|cap| cap[1].trim().to_string())
        .collect();
    let cleaned = re.replace_all(content, "").to_string();
    (cleaned, filenames)
}

/// Detect and strip `<handoff/>` (or `<handoff>`) tag from content.
/// Returns the cleaned content and whether a handoff tag was found.
pub fn extract_handoff_tag(content: &str) -> (String, bool) {
    let re = Regex::new(r"(?s)<handoff\s*/?>").unwrap();
    let has_handoff = re.is_match(content);
    let cleaned = re.replace_all(content, "").trim().to_string();
    (cleaned, has_handoff)
}

/// Extract all `<tagname>content</tagname>` pairs from assistant output,
/// excluding `<next-state>` and `<python>` tags (which are handled separately).
pub fn extract_xml_vars(content: &str) -> HashMap<String, String> {
    let open_re = Regex::new(r"<([a-zA-Z_][a-zA-Z0-9_]*)>").unwrap();
    let mut vars = HashMap::new();
    for cap in open_re.captures_iter(content) {
        let tag = &cap[1];
        if tag == "next-state" || tag == "python" {
            continue;
        }
        let close_tag = format!("</{}>", tag);
        let open_end = cap.get(0).unwrap().end();
        if let Some(close_pos) = content[open_end..].find(&close_tag) {
            let value = &content[open_end..open_end + close_pos];
            vars.insert(tag.to_string(), value.trim().to_string());
        }
    }
    vars
}

/// Extract variables from a `<set-vars>...</set-vars>` wrapper tag.
/// Only parses child elements within the wrapper. Returns empty map if no wrapper found.
pub fn extract_set_vars(content: &str) -> HashMap<String, String> {
    let re = Regex::new(r"(?s)<set-vars>(.*?)</set-vars>").unwrap();
    let mut vars = HashMap::new();
    if let Some(cap) = re.captures(content) {
        let inner = &cap[1];
        // Use the same approach as extract_xml_vars: find opening tags, then their closing tags
        let open_re = Regex::new(r"<([a-zA-Z_][a-zA-Z0-9_]*)>").unwrap();
        for child_cap in open_re.captures_iter(inner) {
            let tag = &child_cap[1];
            let close_tag = format!("</{}>", tag);
            let open_end = child_cap.get(0).unwrap().end();
            if let Some(close_pos) = inner[open_end..].find(&close_tag) {
                let value = &inner[open_end..open_end + close_pos];
                vars.insert(tag.to_string(), value.trim().to_string());
            }
        }
    }
    vars
}

/// Detect and strip `<clear-vars />` (or `<clear-vars/>`) tag from content.
/// Returns the cleaned content and whether a clear-vars tag was found.
pub fn extract_clear_vars_tag(content: &str) -> (String, bool) {
    let re = Regex::new(r"(?s)\s*<clear-vars\s*/?>").unwrap();
    let has_clear = re.is_match(content);
    let cleaned = re.replace_all(content, "").trim().to_string();
    (cleaned, has_clear)
}

/// Extract variables from `<set-vars>` wrapper AND strip the wrapper from content.
pub fn extract_and_strip_set_vars(content: &str) -> (String, HashMap<String, String>) {
    let vars = extract_set_vars(content);
    let re = Regex::new(r"(?s)\s*<set-vars>.*?</set-vars>").unwrap();
    let cleaned = re.replace_all(content, "").trim().to_string();
    (cleaned, vars)
}

/// Peek at the `state` key inside a `<set-vars>` block without stripping.
pub fn peek_state_from_set_vars(content: &str) -> Option<String> {
    let vars = extract_set_vars(content);
    vars.get("state").cloned()
}

/// Extract XML vars from content AND return cleaned content with var tags stripped.
/// Like `extract_xml_vars` but also removes the matched tags from the content.
pub fn extract_and_strip_xml_vars(content: &str) -> (String, HashMap<String, String>) {
    let vars = extract_xml_vars(content);
    let mut cleaned = content.to_string();
    for (tag, _) in &vars {
        let pattern = format!("<{}>", tag);
        let close = format!("</{}>", tag);
        if let Some(open_pos) = cleaned.find(&pattern) {
            if let Some(close_pos) = cleaned[open_pos..].find(&close) {
                let end = open_pos + close_pos + close.len();
                cleaned = format!("{}{}", &cleaned[..open_pos], &cleaned[end..]);
            }
        }
    }
    (cleaned.trim().to_string(), vars)
}

/// Parsed YAML frontmatter from a state file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StateFrontmatter {
    /// Whether this state waits for user input before executing.
    pub wait: bool,
    /// Whether tools are enabled for this state.
    /// `None` = not specified (all tools, backward compatible).
    /// `Some(false)` = no tools sent to LLM.
    /// `Some(true)` = tools enabled (same as default).
    pub tools: Option<bool>,
    /// Default next state when the LLM doesn't emit a `<state>` tag.
    /// If set, auto-transitions to this state instead of keeping the current state.
    pub default_next: Option<String>,
    /// Valid transition targets from this state.
    /// Used in error messages to show reachable states instead of all states.
    pub transitions: Option<Vec<String>>,
}

/// Parse YAML frontmatter from a state file.
/// Returns the frontmatter properties and the body content (without frontmatter).
/// Frontmatter is delimited by `---` fences at the start of the file.
pub fn parse_state_frontmatter(content: &str) -> (StateFrontmatter, &str) {
    let trimmed = content.trim_start();
    if !trimmed.starts_with("---") {
        return (StateFrontmatter::default(), content);
    }
    // Find closing --- (must be on its own line after the opening ---)
    let after_open = &trimmed[3..];
    let after_open = after_open.strip_prefix('\n').unwrap_or(after_open);
    if let Some(end_pos) = after_open.find("\n---") {
        let frontmatter_str = &after_open[..end_pos];
        let body_start = end_pos + 4; // skip "\n---"
        let body = after_open[body_start..].strip_prefix('\n').unwrap_or(&after_open[body_start..]);

        let mut fm = StateFrontmatter::default();
        for line in frontmatter_str.lines() {
            let line = line.trim();
            if let Some(val) = line.strip_prefix("wait:") {
                fm.wait = val.trim() == "true";
            }
            if let Some(val) = line.strip_prefix("tools:") {
                fm.tools = Some(val.trim() == "true");
            }
            if let Some(val) = line.strip_prefix("default_next:") {
                let val = val.trim();
                if !val.is_empty() {
                    fm.default_next = Some(val.to_string());
                }
            }
            if let Some(val) = line.strip_prefix("transitions:") {
                let val = val.trim();
                if !val.is_empty() {
                    fm.transitions = Some(
                        val.split(',')
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect(),
                    );
                }
            }
        }

        (fm, body)
    } else {
        (StateFrontmatter::default(), content)
    }
}

/// Replace `{{varname}}` placeholders in text using the provided variable map.
pub fn expand_vars(text: &str, vars: &HashMap<String, String>) -> String {
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
    fn test_extract_state_tags_none() {
        let (cleaned, stages) = extract_state_tags("Hello world, no tags here.");
        assert_eq!(cleaned, "Hello world, no tags here.");
        assert!(stages.is_empty());
    }

    #[test]
    fn test_extract_state_tags_single() {
        let (_, stages) = extract_state_tags("Let me check <next-state>questions.md</next-state> for guidance.");
        assert_eq!(stages, vec!["questions.md"]);
    }

    #[test]
    fn test_extract_state_tags_multiple() {
        let (_, stages) = extract_state_tags(
            "Reading <next-state>a.md</next-state> and <next-state>b.md</next-state> now.",
        );
        assert_eq!(stages, vec!["a.md", "b.md"]);
    }

    #[test]
    fn test_extract_state_tags_strips_from_content() {
        let (cleaned, _) =
            extract_state_tags("Before <next-state>file.md</next-state> after");
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
    fn test_extract_xml_vars_excludes_next_state() {
        let vars = extract_xml_vars("<next-state>foo</next-state><message>bar</message>");
        assert!(!vars.contains_key("next-state"));
        assert_eq!(vars.get("message").unwrap(), "bar");
    }

    #[test]
    fn test_extract_xml_vars_excludes_python() {
        let vars = extract_xml_vars("<python>print('hello')</python><message>done</message>");
        assert!(!vars.contains_key("python"));
        assert_eq!(vars.get("message").unwrap(), "done");
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
    fn test_extract_state_tags_multiline() {
        let (_, stages) = extract_state_tags("<next-state>\nfile.md\n</next-state>");
        assert_eq!(stages, vec!["file.md"]);
    }

    #[test]
    fn test_extract_xml_vars_multiline() {
        let vars = extract_xml_vars("<message>\nhello\n</message>");
        assert_eq!(vars.get("message").unwrap(), "hello");
    }

    #[test]
    fn test_extract_handoff_self_closing() {
        let (cleaned, handoff) = extract_handoff_tag("Do this plan.\n\n<handoff/>");
        assert!(handoff);
        assert_eq!(cleaned, "Do this plan.");
    }

    #[test]
    fn test_extract_handoff_open_tag() {
        let (cleaned, handoff) = extract_handoff_tag("Execute now <handoff>");
        assert!(handoff);
        assert_eq!(cleaned, "Execute now");
    }

    #[test]
    fn test_extract_handoff_with_space() {
        let (_, handoff) = extract_handoff_tag("Plan ready <handoff />");
        assert!(handoff);
    }

    #[test]
    fn test_extract_handoff_none() {
        let (cleaned, handoff) = extract_handoff_tag("Just a normal response.");
        assert!(!handoff);
        assert_eq!(cleaned, "Just a normal response.");
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

    #[test]
    fn test_extract_and_strip_xml_vars_basic() {
        let (cleaned, vars) = extract_and_strip_xml_vars("<messages>hello</messages>");
        assert_eq!(vars.get("messages").unwrap(), "hello");
        assert_eq!(cleaned, "");
    }

    #[test]
    fn test_extract_and_strip_xml_vars_with_surrounding_text() {
        let (cleaned, vars) = extract_and_strip_xml_vars("before <tone>friendly</tone> after");
        assert_eq!(vars.get("tone").unwrap(), "friendly");
        assert_eq!(cleaned, "before  after");
    }

    #[test]
    fn test_extract_and_strip_xml_vars_multiple() {
        let (cleaned, vars) = extract_and_strip_xml_vars("<intent>greet</intent><tone>warm</tone>");
        assert_eq!(vars.len(), 2);
        assert_eq!(cleaned, "");
    }

    #[test]
    fn test_extract_and_strip_xml_vars_none() {
        let (cleaned, vars) = extract_and_strip_xml_vars("plain text");
        assert!(vars.is_empty());
        assert_eq!(cleaned, "plain text");
    }

    #[test]
    fn test_extract_and_strip_xml_vars_preserves_next_state() {
        // next-state tags are excluded by extract_xml_vars, so they should remain
        let (cleaned, vars) = extract_and_strip_xml_vars("<next-state>foo.md</next-state><messages>hi</messages>");
        assert!(!vars.contains_key("next-state"));
        assert_eq!(vars.get("messages").unwrap(), "hi");
        assert_eq!(cleaned, "<next-state>foo.md</next-state>");
    }

    #[test]
    fn test_extract_and_strip_xml_vars_preserves_python() {
        // python tags are excluded by extract_xml_vars, so they should remain in cleaned content
        let (cleaned, vars) = extract_and_strip_xml_vars(
            "Here's the code:\n\n<python>\nprint('hello')\n</python>\n\n<tone>helpful</tone>",
        );
        assert!(!vars.contains_key("python"));
        assert_eq!(vars.get("tone").unwrap(), "helpful");
        assert!(cleaned.contains("<python>"));
        assert!(cleaned.contains("print('hello')"));
        assert!(cleaned.contains("</python>"));
    }

    // --- Frontmatter parsing tests ---

    #[test]
    fn test_parse_frontmatter_none() {
        let (fm, body) = parse_state_frontmatter("Just a normal template.");
        assert!(!fm.wait);
        assert_eq!(body, "Just a normal template.");
    }

    #[test]
    fn test_parse_frontmatter_wait_true() {
        let content = "---\nwait: true\n---\nPrompt text here.";
        let (fm, body) = parse_state_frontmatter(content);
        assert!(fm.wait);
        assert_eq!(body, "Prompt text here.");
    }

    #[test]
    fn test_parse_frontmatter_wait_false() {
        let content = "---\nwait: false\n---\nAuto-execute prompt.";
        let (fm, body) = parse_state_frontmatter(content);
        assert!(!fm.wait);
        assert_eq!(body, "Auto-execute prompt.");
    }

    #[test]
    fn test_parse_frontmatter_no_wait_key() {
        let content = "---\nother: value\n---\nPrompt.";
        let (fm, body) = parse_state_frontmatter(content);
        assert!(!fm.wait); // defaults to false
        assert_eq!(body, "Prompt.");
    }

    #[test]
    fn test_parse_frontmatter_unclosed() {
        let content = "---\nwait: true\nNo closing fence.";
        let (fm, body) = parse_state_frontmatter(content);
        assert!(!fm.wait); // no valid frontmatter
        assert_eq!(body, content);
    }

    #[test]
    fn test_parse_frontmatter_multiline_body() {
        let content = "---\nwait: true\n---\nLine 1\nLine 2\nLine 3";
        let (fm, body) = parse_state_frontmatter(content);
        assert!(fm.wait);
        assert_eq!(body, "Line 1\nLine 2\nLine 3");
    }

    // --- <set-vars> extraction tests ---

    #[test]
    fn test_extract_set_vars_basic() {
        let vars = extract_set_vars("<set-vars>\n  <color>yellow</color>\n  <size>large</size>\n</set-vars>");
        assert_eq!(vars.len(), 2);
        assert_eq!(vars.get("color").unwrap(), "yellow");
        assert_eq!(vars.get("size").unwrap(), "large");
    }

    #[test]
    fn test_extract_set_vars_none() {
        let vars = extract_set_vars("No set-vars here.");
        assert!(vars.is_empty());
    }

    #[test]
    fn test_extract_set_vars_ignores_bare_xml() {
        // Bare XML tags outside <set-vars> should NOT be captured
        let vars = extract_set_vars("<message>hello</message>");
        assert!(vars.is_empty());
    }

    #[test]
    fn test_extract_set_vars_with_surrounding_text() {
        let content = "Some response text.\n\n<set-vars>\n  <topic>rust</topic>\n</set-vars>\n<next-state>final</next-state>";
        let vars = extract_set_vars(content);
        assert_eq!(vars.len(), 1);
        assert_eq!(vars.get("topic").unwrap(), "rust");
    }

    #[test]
    fn test_extract_and_strip_set_vars_basic() {
        let (cleaned, vars) = extract_and_strip_set_vars(
            "Response text.\n\n<set-vars>\n  <color>blue</color>\n</set-vars>"
        );
        assert_eq!(vars.get("color").unwrap(), "blue");
        assert_eq!(cleaned, "Response text.");
    }

    #[test]
    fn test_extract_and_strip_set_vars_preserves_other_tags() {
        let (cleaned, vars) = extract_and_strip_set_vars(
            "Text <message>hi</message>\n<set-vars>\n  <x>1</x>\n</set-vars>"
        );
        assert_eq!(vars.get("x").unwrap(), "1");
        assert!(cleaned.contains("<message>hi</message>"));
    }

    #[test]
    fn test_extract_and_strip_set_vars_none() {
        let (cleaned, vars) = extract_and_strip_set_vars("plain text");
        assert!(vars.is_empty());
        assert_eq!(cleaned, "plain text");
    }

    #[test]
    fn test_extract_set_vars_multiline_value() {
        let content = "<set-vars>\n  <code>fn main() {\n    println!(\"hello\");\n}</code>\n</set-vars>";
        let vars = extract_set_vars(content);
        assert_eq!(vars.get("code").unwrap(), "fn main() {\n    println!(\"hello\");\n}");
    }

    // --- peek_state_from_set_vars tests ---

    #[test]
    fn test_peek_state_from_set_vars_found() {
        let content = "Some text\n<set-vars>\n  <state>plan</state>\n  <task>do stuff</task>\n</set-vars>";
        assert_eq!(peek_state_from_set_vars(content), Some("plan".to_string()));
    }

    #[test]
    fn test_peek_state_from_set_vars_none() {
        let content = "No set-vars here.";
        assert_eq!(peek_state_from_set_vars(content), None);
    }

    #[test]
    fn test_peek_state_from_set_vars_no_state_key() {
        let content = "<set-vars>\n  <task>do stuff</task>\n</set-vars>";
        assert_eq!(peek_state_from_set_vars(content), None);
    }

    #[test]
    fn test_peek_state_from_set_vars_round_trip() {
        // Ensure state is preserved when extract_and_strip_set_vars processes the same content
        let content = "Response text.\n\n<set-vars>\n  <state>review</state>\n  <topic>rust</topic>\n</set-vars>";
        let peeked = peek_state_from_set_vars(content);
        assert_eq!(peeked, Some("review".to_string()));

        let (stripped, vars) = extract_and_strip_set_vars(content);
        assert_eq!(vars.get("state").cloned(), Some("review".to_string()));
        assert_eq!(vars.get("topic").cloned(), Some("rust".to_string()));
        assert_eq!(stripped, "Response text.");
    }

    #[test]
    fn test_extract_clear_vars_tag_self_closing() {
        let (cleaned, found) = extract_clear_vars_tag("Some response.\n\n<clear-vars />");
        assert!(found);
        assert_eq!(cleaned, "Some response.");
    }

    #[test]
    fn test_extract_clear_vars_tag_no_space() {
        let (cleaned, found) = extract_clear_vars_tag("Reset now.\n<clear-vars/>");
        assert!(found);
        assert_eq!(cleaned, "Reset now.");
    }

    #[test]
    fn test_extract_clear_vars_tag_not_present() {
        let (cleaned, found) = extract_clear_vars_tag("Just a normal response.");
        assert!(!found);
        assert_eq!(cleaned, "Just a normal response.");
    }

    #[test]
    fn test_extract_clear_vars_tag_with_set_vars() {
        let content = "Response.\n<clear-vars />\n<set-vars>\n  <x>1</x>\n</set-vars>";
        let (cleaned, found) = extract_clear_vars_tag(content);
        assert!(found);
        assert!(cleaned.contains("<set-vars>"));
        // After clear-vars stripped, set-vars should still be present for later extraction
        let (final_cleaned, vars) = extract_and_strip_set_vars(&cleaned);
        assert_eq!(vars.get("x").unwrap(), "1");
        assert_eq!(final_cleaned, "Response.");
    }

    // --- tools frontmatter tests ---

    #[test]
    fn test_parse_frontmatter_tools_false() {
        let content = "---\ntools: false\n---\nRoute the request.";
        let (fm, body) = parse_state_frontmatter(content);
        assert_eq!(fm.tools, Some(false));
        assert_eq!(body, "Route the request.");
    }

    #[test]
    fn test_parse_frontmatter_tools_true() {
        let content = "---\ntools: true\n---\nDo work.";
        let (fm, body) = parse_state_frontmatter(content);
        assert_eq!(fm.tools, Some(true));
        assert_eq!(body, "Do work.");
    }

    #[test]
    fn test_parse_frontmatter_wait_and_tools() {
        let content = "---\nwait: true\ntools: false\n---\nWait without tools.";
        let (fm, body) = parse_state_frontmatter(content);
        assert!(fm.wait);
        assert_eq!(fm.tools, Some(false));
        assert_eq!(body, "Wait without tools.");
    }

    #[test]
    fn test_parse_frontmatter_tools_not_specified() {
        let content = "---\nwait: true\n---\nJust wait.";
        let (fm, body) = parse_state_frontmatter(content);
        assert!(fm.wait);
        assert_eq!(fm.tools, None);
        assert_eq!(body, "Just wait.");
    }

    // --- default_next frontmatter tests ---

    #[test]
    fn test_parse_frontmatter_default_next() {
        let content = "---\ndefault_next: tests\n---\nWrite a plan.";
        let (fm, body) = parse_state_frontmatter(content);
        assert_eq!(fm.default_next, Some("tests".to_string()));
        assert_eq!(body, "Write a plan.");
    }

    #[test]
    fn test_parse_frontmatter_default_next_with_wait() {
        let content = "---\nwait: false\ndefault_next: implement\n---\nWrite tests.";
        let (fm, body) = parse_state_frontmatter(content);
        assert!(!fm.wait);
        assert_eq!(fm.default_next, Some("implement".to_string()));
        assert_eq!(body, "Write tests.");
    }

    #[test]
    fn test_parse_frontmatter_default_next_not_specified() {
        let content = "---\nwait: true\n---\nWait for input.";
        let (fm, _) = parse_state_frontmatter(content);
        assert_eq!(fm.default_next, None);
    }

    #[test]
    fn test_parse_frontmatter_default_next_empty() {
        let content = "---\ndefault_next: \n---\nPrompt.";
        let (fm, _) = parse_state_frontmatter(content);
        assert_eq!(fm.default_next, None);
    }

    #[test]
    fn test_parse_frontmatter_all_keys() {
        let content = "---\nwait: true\ntools: false\ndefault_next: review\ntransitions: tests, implement, done\n---\nDo work.";
        let (fm, body) = parse_state_frontmatter(content);
        assert!(fm.wait);
        assert_eq!(fm.tools, Some(false));
        assert_eq!(fm.default_next, Some("review".to_string()));
        assert_eq!(fm.transitions, Some(vec!["tests".to_string(), "implement".to_string(), "done".to_string()]));
        assert_eq!(body, "Do work.");
    }

    // --- transitions frontmatter tests ---

    #[test]
    fn test_parse_frontmatter_transitions() {
        let content = "---\ntransitions: tests, implement, done\n---\nWrite a plan.";
        let (fm, body) = parse_state_frontmatter(content);
        assert_eq!(fm.transitions, Some(vec!["tests".to_string(), "implement".to_string(), "done".to_string()]));
        assert_eq!(body, "Write a plan.");
    }

    #[test]
    fn test_parse_frontmatter_transitions_not_specified() {
        let content = "---\nwait: true\n---\nWait.";
        let (fm, _) = parse_state_frontmatter(content);
        assert_eq!(fm.transitions, None);
    }

    #[test]
    fn test_parse_frontmatter_transitions_empty() {
        let content = "---\ntransitions: \n---\nPrompt.";
        let (fm, _) = parse_state_frontmatter(content);
        assert_eq!(fm.transitions, None);
    }

    #[test]
    fn test_parse_frontmatter_transitions_single() {
        let content = "---\ntransitions: done\n---\nFinish up.";
        let (fm, _) = parse_state_frontmatter(content);
        assert_eq!(fm.transitions, Some(vec!["done".to_string()]));
    }
}
