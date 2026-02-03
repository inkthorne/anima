# Claude Code Tool Spec

## Overview

A tool that lets Anima agents delegate coding tasks to Claude Code, run in background, and get notified on completion. Designed for Gendry to be the "coding lead" who manages Claude Code on behalf of other agents.

---

## Design

### Workflow

```
Arya: "@gendry implement heartbeat feature per this spec: [spec]"
  ↓
Gendry: *uses claude_code tool*
  ↓
Tool: launches Claude Code in background, returns task ID
  ↓
Gendry: "Started task abc123, I'll let you know when it's done"
  ↓
[Claude Code runs... 5-30 minutes]
  ↓
Tool: detects completion, stores result, notifies Gendry
  ↓
Gendry: *checks result, verifies build*
  ↓
Gendry: "@arya done! Implementation complete, tests passing. [summary]"
```

### Key Principles

1. **Fire-and-forget launch** — Tool returns immediately with task ID
2. **Background execution** — Claude Code runs in separate process
3. **Completion notification** — Agent gets notified when done (via pending notification or next think)
4. **Result persistence** — Output stored for review

---

## Tool Definition

```toml
# In tools.toml
[tools.claude_code]
name = "claude_code"
description = "Delegate a coding task to Claude Code. Returns immediately with task ID. You'll be notified when complete."
parameters = [
    { name = "task", type = "string", description = "The coding task/prompt for Claude Code", required = true },
    { name = "workdir", type = "string", description = "Working directory (default: current project)", required = false },
]
```

---

## Implementation

### Task Storage

Store running/completed tasks in SQLite (new table or extend conversations.db):

```sql
CREATE TABLE claude_code_tasks (
    id TEXT PRIMARY KEY,           -- task ID (uuid)
    agent TEXT NOT NULL,           -- agent who launched it
    task_prompt TEXT NOT NULL,     -- the prompt sent to Claude Code
    workdir TEXT,                  -- working directory
    status TEXT NOT NULL,          -- 'running', 'completed', 'failed'
    pid INTEGER,                   -- process ID
    started_at INTEGER NOT NULL,
    completed_at INTEGER,
    exit_code INTEGER,
    output TEXT,                   -- captured output/summary
    result_summary TEXT            -- parsed summary for agent
);
```

### Tool Execution (tools/claude_code.rs)

```rust
pub async fn execute(params: Value, context: &ToolContext) -> Result<String> {
    let task = params["task"].as_str().ok_or("missing task")?;
    let workdir = params["workdir"].as_str().unwrap_or(".");
    
    // Generate task ID
    let task_id = uuid::Uuid::new_v4().to_string()[..8].to_string();
    
    // Build command
    let script = expand_path("~/clawd/scripts/run-claude.sh");
    let cmd = format!(
        "cd {} && nohup {} '{}' > /tmp/claude-{}.log 2>&1 & echo $!",
        workdir, script, escape_quotes(task), task_id
    );
    
    // Launch background process
    let output = Command::new("sh").arg("-c").arg(&cmd).output().await?;
    let pid: i32 = String::from_utf8_lossy(&output.stdout).trim().parse()?;
    
    // Store task record
    store_task(&task_id, context.agent_name, task, workdir, pid).await?;
    
    // Spawn completion watcher
    tokio::spawn(watch_task_completion(task_id.clone(), pid, context.agent_name.clone()));
    
    Ok(format!("Task {} started (pid {}). I'll notify you when complete.", task_id, pid))
}

async fn watch_task_completion(task_id: String, pid: i32, agent: String) {
    // Poll for process completion
    loop {
        tokio::time::sleep(Duration::from_secs(10)).await;
        
        if !process_running(pid) {
            // Process finished - read log, update status, notify agent
            let log = fs::read_to_string(format!("/tmp/claude-{}.log", task_id)).unwrap_or_default();
            let summary = extract_summary(&log);
            
            update_task_status(&task_id, "completed", &summary).await;
            
            // Notify agent via pending notification or conversation message
            notify_agent_task_complete(&agent, &task_id, &summary).await;
            break;
        }
    }
}
```

### Completion Notification

**Decision: Post to `claude-code-<agent>` conversation with @mention**

When task completes:
1. Get or create `claude-code-<agent>` conversation (e.g., `claude-code-gendry`)
2. Post message: `@<agent> Task <id> completed (exit <code>).\n\nOutput:\n<summary>`
3. Send `Notify` to the agent (existing mechanism)
4. Agent wakes up, sees the event in conversation

Benefits:
- Uses existing @mention + Notify infrastructure
- Persistent log (`anima chat view claude-code-gendry`)
- Focused context: only Claude Code events, useful for learning patterns
- Follows `<feature>-<agent>` pattern (like `heartbeat-<agent>`)

---

## Agent Workflow (Gendry)

Gendry's persona should include:

```markdown
## Claude Code Expert

You have access to the `claude_code` tool for delegating implementation work.

### When to use it:
- Complex multi-file changes
- Tasks that need exploration/iteration
- Anything that would take more than a few edits

### Workflow:
1. Receive task from @arya or others
2. Review/clarify the spec if needed
3. Use `claude_code` tool with clear prompt
4. Wait for completion notification
5. Verify: check build, run tests
6. Report back: summarize what was done, any issues

### Prompt tips:
- Include file paths when known
- Specify build/test commands
- Be specific about expected behavior
```

---

## CLI Commands (Optional)

```bash
anima tasks                     # List all tasks
anima tasks <id>                # Show task details/output
anima tasks cancel <id>         # Kill running task
```

---

## Open Questions

1. **Output capture** — Full log or just summary? Full logs can be huge.
   - Recommendation: Store last N lines + any error output

2. **Timeout** — Should tasks have a max runtime?
   - Recommendation: 1 hour default, configurable

3. **Multiple concurrent tasks** — Allow or serialize?
   - Recommendation: Allow, but warn if >2 running

4. **Which agents can use it** — All or allowlisted?
   - Recommendation: Allowlist in agent config (Gendry only initially)

---

## Phases

### Phase 1: Basic Tool
- Task storage (SQLite)
- Launch Claude Code in background
- Poll for completion
- Inject result into next think

### Phase 2: Polish
- CLI commands for task management
- Better output summarization
- Timeout handling

### Phase 3: Integration
- Gendry persona updates
- Test full workflow (Arya → Gendry → Claude Code → Gendry → Arya)
