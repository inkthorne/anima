use crate::error::ToolError;
use crate::tool::Tool;
use async_trait::async_trait;
use serde_json::Value;

/// Daemon-aware tool that dispatches a task to an agent and waits for the result.
/// Creates a task conversation, posts the task with origin metadata, notifies the child agent,
/// then polls until the child produces a final response (or timeout).
pub struct DaemonStartTaskTool {
    agent_name: String,
    conv_id: Option<String>,
}

impl DaemonStartTaskTool {
    pub fn new(agent_name: String, conv_id: Option<String>) -> Self {
        DaemonStartTaskTool {
            agent_name,
            conv_id,
        }
    }
}

impl std::fmt::Debug for DaemonStartTaskTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DaemonStartTaskTool")
            .field("agent_name", &self.agent_name)
            .finish()
    }
}

#[async_trait]
impl Tool for DaemonStartTaskTool {
    fn name(&self) -> &str {
        "start_task"
    }

    fn description(&self) -> &str {
        "Dispatch a task to an agent and wait for the result. The agent works in a separate conversation; this tool blocks until the agent finishes or goes idle (120s inactivity). At the 30-minute checkpoint, returns a progress report — use wait_task to continue waiting or stop_task to cancel."
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "Name of the agent to delegate to"
                },
                "task": {
                    "type": "string",
                    "description": "Complete task instructions. The agent works in a fresh conversation with NO prior context — include everything needed: full file paths, project directory, data structures, edge cases, and success criteria."
                }
            },
            "required": ["agent", "task"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        use crate::conversation::ConversationStore;
        use crate::discovery;
        use crate::socket_api::{Request, SocketApi};
        use rand::RngExt;
        use tokio::net::UnixStream;

        let agent = input
            .get("agent")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("agent is required".to_string()))?;

        let task = input
            .get("task")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("task is required".to_string()))?;

        // Validate agent exists
        if !discovery::agent_exists(agent) {
            return Err(ToolError::ExecutionFailed(format!(
                "Agent '{}' does not exist",
                agent
            )));
        }

        // Start agent if not running
        if !discovery::is_agent_running(agent) {
            discovery::start_agent_daemon(agent).map_err(|e| {
                ToolError::ExecutionFailed(format!("Failed to start agent '{}': {}", agent, e))
            })?;
        }

        // Generate short random ID for task conversation
        let short_id = {
            let mut rng = rand::rng();
            let id_bytes: [u8; 4] = rng.random();
            id_bytes.iter().map(|b| format!("{:02x}", b)).collect::<String>()
        };
        let conv_name = format!("task:{}:{}:{}", self.agent_name, agent, short_id);

        // Create task conversation
        let store = ConversationStore::init().map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to open conversation store: {}", e))
        })?;

        store
            .create_conversation(
                Some(&conv_name),
                &[&self.agent_name, agent],
            )
            .map_err(|e| {
                ToolError::ExecutionFailed(format!("Failed to create task conversation: {}", e))
            })?;

        // Build task message with origin metadata prefix
        let origin_conv = self.conv_id.as_deref().unwrap_or("unknown");
        let task_content = format!(
            "[task origin={} by={}]\n{}",
            origin_conv, self.agent_name, task
        );

        // Post task message and pin it so it stays in context
        let message_id = store
            .add_message(&conv_name, &self.agent_name, &task_content, &[agent])
            .map_err(|e| {
                ToolError::ExecutionFailed(format!("Failed to post task message: {}", e))
            })?;

        store.pin_message(&conv_name, message_id, true).map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to pin task message: {}", e))
        })?;

        // Connect to child agent with retry (handles both fresh-start and already-running)
        let socket_path = discovery::agents_dir().join(agent).join("agent.sock");
        let mut stream_opt = None;
        for _ in 0..50 {
            match UnixStream::connect(&socket_path).await {
                Ok(s) => {
                    stream_opt = Some(s);
                    break;
                }
                Err(_) => {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            }
        }
        let stream = stream_opt.ok_or_else(|| {
            ToolError::ExecutionFailed(format!(
                "Failed to connect to agent '{}' after retries",
                agent
            ))
        })?;

        let mut api = SocketApi::new(stream);
        let request = Request::Notify {
            conv_id: conv_name.clone(),
            message_id,
            depth: 0,
        };

        api.write_request(&request).await.map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to notify agent '{}': {}", agent, e))
        })?;

        // Read acknowledgement (dispatch confirmed)
        let _response = api.read_response().await.ok();

        poll_task_conversation(&self.agent_name, &conv_name, agent).await
    }
}

/// Daemon-aware tool that continues waiting for a running task's result.
/// Use after a start_task returns 'running' status at the 30-minute checkpoint.
pub struct DaemonWaitTaskTool {
    agent_name: String,
}

impl DaemonWaitTaskTool {
    pub fn new(agent_name: String) -> Self {
        DaemonWaitTaskTool { agent_name }
    }
}

impl std::fmt::Debug for DaemonWaitTaskTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DaemonWaitTaskTool")
            .field("agent_name", &self.agent_name)
            .finish()
    }
}

#[async_trait]
impl Tool for DaemonWaitTaskTool {
    fn name(&self) -> &str {
        "wait_task"
    }

    fn description(&self) -> &str {
        "Continue waiting for a running task's result. Use after a task returns 'running' status at the 30-minute checkpoint."
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "task_conv": {
                    "type": "string",
                    "description": "The task conversation ID from a previous 'running' result"
                }
            },
            "required": ["task_conv"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let task_conv = input
            .get("task_conv")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("task_conv is required".to_string()))?;

        let agent = extract_child_agent(task_conv).ok_or_else(|| {
            ToolError::InvalidInput(format!(
                "Cannot extract agent name from task_conv '{}' (expected task:parent:child:id)",
                task_conv
            ))
        })?;

        poll_task_conversation(&self.agent_name, task_conv, &agent).await
    }
}

/// Daemon-aware tool that stops a running task and signals the agent to stop working.
pub struct DaemonStopTaskTool {
    agent_name: String,
}

impl DaemonStopTaskTool {
    pub fn new(agent_name: String) -> Self {
        DaemonStopTaskTool { agent_name }
    }
}

impl std::fmt::Debug for DaemonStopTaskTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DaemonStopTaskTool")
            .field("agent_name", &self.agent_name)
            .finish()
    }
}

#[async_trait]
impl Tool for DaemonStopTaskTool {
    fn name(&self) -> &str {
        "stop_task"
    }

    fn description(&self) -> &str {
        "Stop a running task and signal the agent to stop working. Use when you no longer need the result of a delegated task."
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "task_conv": {
                    "type": "string",
                    "description": "The task conversation ID to cancel"
                }
            },
            "required": ["task_conv"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        use crate::conversation::ConversationStore;

        let task_conv = input
            .get("task_conv")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("task_conv is required".to_string()))?;

        let agent = extract_child_agent(task_conv).ok_or_else(|| {
            ToolError::InvalidInput(format!(
                "Cannot extract agent name from task_conv '{}' (expected task:parent:child:id)",
                task_conv
            ))
        })?;

        let store = ConversationStore::init().map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to open store: {}", e))
        })?;

        // Post cancel message so the agent knows why it was stopped
        store
            .add_message(
                task_conv,
                &self.agent_name,
                &format!("[task cancelled by {}]", self.agent_name),
                &[&agent],
            )
            .map_err(|e| {
                ToolError::ExecutionFailed(format!(
                    "Failed to post cancel message: {}",
                    e
                ))
            })?;

        // Pause conversation — triggers the existing cancel flag in handle_notify
        store.set_paused(task_conv, true).map_err(|e| {
            ToolError::ExecutionFailed(format!(
                "Failed to pause task conversation: {}",
                e
            ))
        })?;

        Ok(serde_json::json!({
            "status": "cancelled",
            "task_conv": task_conv,
            "agent": agent,
        }))
    }
}

/// Poll a task conversation until the child produces a final response, goes idle, or hits
/// the soft max (30m). On soft max, returns a "running" status with recent activity instead
/// of a hard timeout.
async fn poll_task_conversation(
    parent_agent: &str,
    conv_name: &str,
    agent: &str,
) -> Result<Value, ToolError> {
    use crate::conversation::ConversationStore;
    use crate::discovery;
    use crate::socket_api::{Request, SocketApi};
    use tokio::net::UnixStream;

    const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_secs(1);
    const INACTIVITY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);
    const SOFT_MAX: std::time::Duration = std::time::Duration::from_secs(1800);

    let socket_path = discovery::agents_dir().join(agent).join("agent.sock");
    let start = tokio::time::Instant::now();
    let mut last_activity = start;
    let mut last_seen_id: i64 = 0;

    loop {
        tokio::time::sleep(POLL_INTERVAL).await;

        let now = tokio::time::Instant::now();

        // Open a fresh store each poll to see latest writes
        let poll_store = ConversationStore::init().map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to open store for polling: {}", e))
        })?;

        let messages = poll_store.get_messages(conv_name, None).map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to poll task conversation: {}", e))
        })?;

        // Track activity: any new message resets the inactivity timer
        let mut new_messages_this_tick = false;
        if let Some(max_id) = messages.iter().map(|m| m.id).max() {
            if max_id > last_seen_id {
                last_seen_id = max_id;
                last_activity = now;
                new_messages_this_tick = true;
            }
        }

        // Iterate in reverse — look for the latest final response
        for msg in messages.iter().rev() {
            // Skip tool results, recall, and our own messages
            if msg.from_agent == "tool"
                || msg.from_agent == "recall"
                || msg.from_agent == parent_agent
            {
                continue;
            }
            // Must be an LLM response (has duration) with no pending tool calls
            if msg.duration_ms.is_some() && msg.tool_calls.is_none() {
                return Ok(serde_json::json!({
                    "status": "completed",
                    "task_conv": conv_name,
                    "agent": msg.from_agent,
                    "result": msg.content,
                }));
            }
            // If we hit a message with tool_calls, the agent is still working — stop scanning
            if msg.duration_ms.is_some() && msg.tool_calls.is_some() {
                break;
            }
        }

        // Soft max checkpoint (30 min) — return progress report instead of hard timeout
        if now.duration_since(start) >= SOFT_MAX {
            let recent_activity: Vec<String> = messages
                .iter()
                .rev()
                .filter(|m| {
                    m.from_agent != "tool"
                        && m.from_agent != "recall"
                        && m.from_agent != parent_agent
                })
                .take(3)
                .map(|m| {
                    let content = if m.content.len() > 200 {
                        format!("{}...", &m.content[..200])
                    } else {
                        m.content.clone()
                    };
                    format!("[{}] {}", m.from_agent, content)
                })
                .collect();

            let elapsed_minutes = now.duration_since(start).as_secs() / 60;
            return Ok(serde_json::json!({
                "status": "running",
                "task_conv": conv_name,
                "agent": agent,
                "elapsed_minutes": elapsed_minutes,
                "message_count": messages.len(),
                "recent_activity": recent_activity,
            }));
        }

        // If no new DB messages this tick, check if child agent is still working
        // via socket status. If working, reset inactivity timer.
        if !new_messages_this_tick {
            if let Ok(stream) = UnixStream::connect(&socket_path).await {
                let mut status_api = SocketApi::new(stream);
                if status_api.write_request(&Request::Status).await.is_ok() {
                    if let Ok(Some(resp)) = tokio::time::timeout(
                        std::time::Duration::from_secs(5),
                        status_api.read_response(),
                    )
                    .await
                    .unwrap_or(Ok(None))
                    {
                        if let crate::socket_api::Response::Status { state, .. } = resp {
                            if state == crate::socket_api::AgentState::Working {
                                last_activity = now;
                            }
                        }
                    }
                }
            }
        }

        // Inactivity timeout (120s since last new message or working state)
        if now.duration_since(last_activity) >= INACTIVITY_TIMEOUT {
            return Ok(serde_json::json!({
                "status": "timeout",
                "task_conv": conv_name,
                "agent": agent,
                "message": "No new messages for 120s"
            }));
        }
    }
}

/// Extract the child agent name from a task conversation name.
/// Convention: `task:parent:child:id` — returns the child segment (index 2).
pub fn extract_child_agent(task_conv: &str) -> Option<String> {
    let parts: Vec<&str> = task_conv.split(':').collect();
    if parts.len() >= 4 && parts[0] == "task" {
        Some(parts[2].to_string())
    } else {
        None
    }
}

/// Parse task origin metadata from a pinned task message content.
/// Expected format on the first line: `[task origin=<conv_id> by=<agent_name>]`
/// Returns `(origin_conv_id, origin_agent_name)` if found.
pub fn parse_task_origin(content: &str) -> Option<(String, String)> {
    let first_line = content.lines().next()?;
    // Match [task origin=<value> by=<value>]
    let rest = first_line.strip_prefix("[task origin=")?;
    let by_pos = rest.find(" by=")?;
    let origin = &rest[..by_pos];
    let after_by = &rest[by_pos + 4..]; // skip " by="
    let end_bracket = after_by.find(']')?;
    let agent = &after_by[..end_bracket];
    if origin.is_empty() || agent.is_empty() {
        return None;
    }
    Some((origin.to_string(), agent.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parse_task_origin_valid() {
        let content = "[task origin=chat-123 by=arya]\nRefactor the auth module";
        let (origin, agent) = parse_task_origin(content).unwrap();
        assert_eq!(origin, "chat-123");
        assert_eq!(agent, "arya");
    }

    #[test]
    fn test_parse_task_origin_no_body() {
        let content = "[task origin=my-conv by=dash]";
        let (origin, agent) = parse_task_origin(content).unwrap();
        assert_eq!(origin, "my-conv");
        assert_eq!(agent, "dash");
    }

    #[test]
    fn test_parse_task_origin_complex_names() {
        let content = "[task origin=task:arya:dash:abc1 by=arya]\nNested task";
        let (origin, agent) = parse_task_origin(content).unwrap();
        assert_eq!(origin, "task:arya:dash:abc1");
        assert_eq!(agent, "arya");
    }

    #[test]
    fn test_parse_task_origin_missing_prefix() {
        assert!(parse_task_origin("Just some text").is_none());
    }

    #[test]
    fn test_parse_task_origin_malformed() {
        assert!(parse_task_origin("[task origin= by=]").is_none());
        assert!(parse_task_origin("[task origin=x]").is_none());
        assert!(parse_task_origin("").is_none());
    }

    #[tokio::test]
    async fn test_start_task_tool_name() {
        let tool = DaemonStartTaskTool::new("parent".to_string(), Some("conv-1".to_string()));
        assert_eq!(tool.name(), "start_task");
    }

    #[tokio::test]
    async fn test_start_task_schema() {
        let tool = DaemonStartTaskTool::new("parent".to_string(), None);
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["agent"].is_object());
        assert!(schema["properties"]["task"].is_object());
        // task_conv and abort should not be present
        assert!(schema["properties"].get("task_conv").is_none());
        assert!(schema["properties"].get("abort").is_none());
        // required should be ["agent", "task"]
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("agent")));
        assert!(required.contains(&json!("task")));
    }

    #[tokio::test]
    async fn test_start_task_missing_agent() {
        let tool = DaemonStartTaskTool::new("parent".to_string(), None);
        let result = tool.execute(json!({"task": "do something"})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_start_task_missing_task() {
        let tool = DaemonStartTaskTool::new("parent".to_string(), None);
        let result = tool.execute(json!({"agent": "test-agent"})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_start_task_nonexistent_agent() {
        let tool = DaemonStartTaskTool::new("parent".to_string(), Some("conv-1".to_string()));
        let result = tool
            .execute(json!({"agent": "nonexistent-xyz-999", "task": "hello"}))
            .await;
        assert!(matches!(result, Err(ToolError::ExecutionFailed(_))));
    }

    #[tokio::test]
    async fn test_wait_task_schema() {
        let tool = DaemonWaitTaskTool::new("parent".to_string());
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["task_conv"].is_object());
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("task_conv")));
    }

    #[tokio::test]
    async fn test_wait_task_bad_task_conv() {
        let tool = DaemonWaitTaskTool::new("parent".to_string());
        let result = tool.execute(json!({"task_conv": "bad-format"})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_stop_task_schema() {
        let tool = DaemonStopTaskTool::new("parent".to_string());
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["task_conv"].is_object());
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("task_conv")));
    }

    #[tokio::test]
    async fn test_stop_task_bad_task_conv() {
        let tool = DaemonStopTaskTool::new("parent".to_string());
        let result = tool
            .execute(json!({"task_conv": "bad-format"}))
            .await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_stop_task_returns_cancelled() {
        use crate::conversation::ConversationStore;

        let store = ConversationStore::init().unwrap();
        let conv = format!("task:parent:child:{}", std::process::id());
        store
            .create_conversation(Some(&conv), &["parent", "child"])
            .unwrap();

        let tool = DaemonStopTaskTool::new("parent".to_string());
        let result = tool
            .execute(json!({"task_conv": conv}))
            .await
            .unwrap();

        assert_eq!(result["status"], "cancelled");
        assert_eq!(result["task_conv"], conv);
        assert_eq!(result["agent"], "child");

        // Verify cancel message was posted
        let messages = store.get_messages(&conv, None).unwrap();
        assert!(messages
            .iter()
            .any(|m| m.content.contains("[task cancelled by parent]")));

        // Verify conversation is paused
        assert!(store.is_paused(&conv).unwrap());

        store.delete_conversation(&conv).unwrap();
    }

    #[test]
    fn test_poll_detects_completed_response() {
        use crate::conversation::ConversationStore;

        let store = ConversationStore::init().unwrap();
        let conv = format!("task:poll-complete-{}", std::process::id());
        store.create_conversation(Some(&conv), &["parent", "child"]).unwrap();

        // Post task message from parent
        store.add_message(&conv, "parent", "[task origin=c1 by=parent]\nDo stuff", &["child"]).unwrap();

        // Child responds with duration_ms (final LLM response, no tool_calls)
        store.add_message_with_tool_calls(&conv, "child", "Task done!", &[], Some(1200), None).unwrap();

        // Simulate poll: scan messages in reverse for completed response
        let messages = store.get_messages(&conv, None).unwrap();
        let found = messages.iter().rev().find(|msg| {
            msg.from_agent != "parent"
                && msg.from_agent != "tool"
                && msg.from_agent != "recall"
                && msg.duration_ms.is_some()
                && msg.tool_calls.is_none()
        });
        assert!(found.is_some());
        assert_eq!(found.unwrap().content, "Task done!");
        assert_eq!(found.unwrap().from_agent, "child");

        store.delete_conversation(&conv).unwrap();
    }

    #[test]
    fn test_poll_skips_mid_loop_response() {
        use crate::conversation::ConversationStore;

        let store = ConversationStore::init().unwrap();
        let conv = format!("task:poll-midloop-{}", std::process::id());
        store.create_conversation(Some(&conv), &["parent", "child"]).unwrap();

        store.add_message(&conv, "parent", "[task origin=c1 by=parent]\nDo stuff", &["child"]).unwrap();

        // Child response with tool_calls — still working
        store.add_message_with_tool_calls(
            &conv, "child", "Let me read that file...",
            &[], Some(800), Some(r#"[{"id":"tc1","function":{"name":"read_file"}}]"#),
        ).unwrap();

        // Tool result
        store.add_native_tool_result(&conv, "tc1", "file contents here", "child").unwrap();

        // No final response yet — should not find a completed message
        let messages = store.get_messages(&conv, None).unwrap();
        let found = messages.iter().rev().find(|msg| {
            msg.from_agent != "parent"
                && msg.from_agent != "tool"
                && msg.from_agent != "recall"
                && msg.duration_ms.is_some()
                && msg.tool_calls.is_none()
        });
        assert!(found.is_none());

        store.delete_conversation(&conv).unwrap();
    }

    #[test]
    fn test_extract_child_agent_valid() {
        assert_eq!(
            extract_child_agent("task:gendry:dash:abc1"),
            Some("dash".to_string())
        );
    }

    #[test]
    fn test_extract_child_agent_complex_id() {
        assert_eq!(
            extract_child_agent("task:arya:gendry:deadbeef"),
            Some("gendry".to_string())
        );
    }

    #[test]
    fn test_extract_child_agent_missing_prefix() {
        assert_eq!(extract_child_agent("chat:foo:bar:baz"), None);
    }

    #[test]
    fn test_extract_child_agent_too_few_parts() {
        assert_eq!(extract_child_agent("task:parent:child"), None);
        assert_eq!(extract_child_agent("task:parent"), None);
        assert_eq!(extract_child_agent("task"), None);
    }

    #[test]
    fn test_poll_finds_final_after_tool_loop() {
        use crate::conversation::ConversationStore;

        let store = ConversationStore::init().unwrap();
        let conv = format!("task:poll-final-{}", std::process::id());
        store.create_conversation(Some(&conv), &["parent", "child"]).unwrap();

        store.add_message(&conv, "parent", "[task origin=c1 by=parent]\nDo stuff", &["child"]).unwrap();

        // Mid-loop response with tool calls
        store.add_message_with_tool_calls(
            &conv, "child", "Reading file...",
            &[], Some(500), Some(r#"[{"id":"tc1","function":{"name":"read_file"}}]"#),
        ).unwrap();

        // Tool result
        store.add_native_tool_result(&conv, "tc1", "contents", "child").unwrap();

        // Final response — no tool_calls
        store.add_message_with_tool_calls(
            &conv, "child", "All done, here is the result.",
            &[], Some(1500), None,
        ).unwrap();

        let messages = store.get_messages(&conv, None).unwrap();
        let found = messages.iter().rev().find(|msg| {
            msg.from_agent != "parent"
                && msg.from_agent != "tool"
                && msg.from_agent != "recall"
                && msg.duration_ms.is_some()
                && msg.tool_calls.is_none()
        });
        assert!(found.is_some());
        assert_eq!(found.unwrap().content, "All done, here is the result.");

        store.delete_conversation(&conv).unwrap();
    }
}
