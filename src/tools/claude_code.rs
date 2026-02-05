//! Claude Code Tool - Delegate coding tasks to Claude Code.
//!
//! This tool allows agents to delegate complex coding tasks to Claude Code,
//! running them in the background and notifying the agent when complete.

use async_trait::async_trait;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;
use tokio::process::Command;
use tokio::sync::Mutex;

use crate::error::ToolError;
use crate::tool::Tool;
use serde_json::Value;

/// Errors that can occur during task operations.
#[derive(Debug, Error)]
pub enum TaskError {
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Task not found: {0}")]
    NotFound(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Process error: {0}")]
    Process(String),
}

/// Status of a Claude Code task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskStatus {
    Running,
    Completed,
    Failed,
}

impl std::fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskStatus::Running => write!(f, "running"),
            TaskStatus::Completed => write!(f, "completed"),
            TaskStatus::Failed => write!(f, "failed"),
        }
    }
}

impl std::str::FromStr for TaskStatus {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "running" => Ok(TaskStatus::Running),
            "completed" => Ok(TaskStatus::Completed),
            "failed" => Ok(TaskStatus::Failed),
            _ => Err(format!("Unknown status: {}", s)),
        }
    }
}

/// A Claude Code task record.
#[derive(Debug, Clone)]
pub struct ClaudeCodeTask {
    pub id: String,
    pub agent: String,
    pub conv_id: Option<String>,
    pub task_prompt: String,
    pub workdir: String,
    pub status: TaskStatus,
    pub pid: Option<i32>,
    pub started_at: i64,
    pub completed_at: Option<i64>,
    pub exit_code: Option<i32>,
    pub output_summary: Option<String>,
}

/// SQLite-backed task store for Claude Code tasks.
pub struct TaskStore {
    conn: Connection,
}

impl TaskStore {
    /// Initialize the task store, creating the database if needed.
    /// Database is located at `~/.anima/conversations.db` (reusing existing DB).
    pub fn init() -> Result<Self, TaskError> {
        let db_path = Self::db_path()?;

        // Ensure parent directory exists
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(&db_path)?;
        let store = Self { conn };
        store.create_schema()?;
        Ok(store)
    }

    /// Open the task store from a specific path (for testing).
    pub fn open(path: &std::path::Path) -> Result<Self, TaskError> {
        let conn = Connection::open(path)?;
        let store = Self { conn };
        store.create_schema()?;
        Ok(store)
    }

    /// Get the default database path.
    fn db_path() -> Result<PathBuf, TaskError> {
        let home = dirs::home_dir().ok_or_else(|| {
            TaskError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Could not determine home directory",
            ))
        })?;
        Ok(home.join(".anima").join("conversations.db"))
    }

    /// Create the database schema if it doesn't exist.
    fn create_schema(&self) -> Result<(), TaskError> {
        self.conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS claude_code_tasks (
                id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                conv_id TEXT,
                task_prompt TEXT NOT NULL,
                workdir TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'running',
                pid INTEGER,
                started_at INTEGER NOT NULL,
                completed_at INTEGER,
                exit_code INTEGER,
                output_summary TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_tasks_agent ON claude_code_tasks(agent);
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON claude_code_tasks(status);
            "#,
        )?;

        // Migration: add conv_id column if it doesn't exist (for existing databases)
        let _ = self
            .conn
            .execute("ALTER TABLE claude_code_tasks ADD COLUMN conv_id TEXT", []);

        Ok(())
    }

    /// Create a new task record.
    pub fn create_task(
        &self,
        id: &str,
        agent: &str,
        conv_id: Option<&str>,
        task_prompt: &str,
        workdir: &str,
        pid: i32,
    ) -> Result<(), TaskError> {
        let now = current_timestamp();

        self.conn.execute(
            "INSERT INTO claude_code_tasks (id, agent, conv_id, task_prompt, workdir, status, pid, started_at)
             VALUES (?1, ?2, ?3, ?4, ?5, 'running', ?6, ?7)",
            params![id, agent, conv_id, task_prompt, workdir, pid, now],
        )?;

        Ok(())
    }

    /// Get a task by ID.
    pub fn get_task(&self, id: &str) -> Result<Option<ClaudeCodeTask>, TaskError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, agent, conv_id, task_prompt, workdir, status, pid, started_at, completed_at, exit_code, output_summary
             FROM claude_code_tasks WHERE id = ?1",
        )?;

        let mut rows = stmt.query(params![id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(task_from_row(row)?))
        } else {
            Ok(None)
        }
    }

    /// Get all running tasks.
    pub fn get_running_tasks(&self) -> Result<Vec<ClaudeCodeTask>, TaskError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, agent, conv_id, task_prompt, workdir, status, pid, started_at, completed_at, exit_code, output_summary
             FROM claude_code_tasks WHERE status = 'running' ORDER BY started_at",
        )?;

        let tasks = stmt
            .query_map([], task_from_row)?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(tasks)
    }

    /// Get all tasks for an agent.
    pub fn get_tasks_for_agent(&self, agent: &str) -> Result<Vec<ClaudeCodeTask>, TaskError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, agent, conv_id, task_prompt, workdir, status, pid, started_at, completed_at, exit_code, output_summary
             FROM claude_code_tasks WHERE agent = ?1 ORDER BY started_at DESC",
        )?;

        let tasks = stmt
            .query_map(params![agent], task_from_row)?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(tasks)
    }

    /// Update task status to completed or failed.
    pub fn complete_task(
        &self,
        id: &str,
        status: TaskStatus,
        exit_code: i32,
        output_summary: &str,
    ) -> Result<(), TaskError> {
        let now = current_timestamp();

        self.conn.execute(
            "UPDATE claude_code_tasks SET status = ?1, completed_at = ?2, exit_code = ?3, output_summary = ?4
             WHERE id = ?5",
            params![status.to_string(), now, exit_code, output_summary, id],
        )?;

        Ok(())
    }

    /// Delete a task by ID.
    pub fn delete_task(&self, id: &str) -> Result<(), TaskError> {
        self.conn
            .execute("DELETE FROM claude_code_tasks WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// List all tasks.
    pub fn list_tasks(&self) -> Result<Vec<ClaudeCodeTask>, TaskError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, agent, conv_id, task_prompt, workdir, status, pid, started_at, completed_at, exit_code, output_summary
             FROM claude_code_tasks ORDER BY started_at DESC",
        )?;

        let tasks = stmt
            .query_map([], task_from_row)?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(tasks)
    }
}

/// Helper to construct a task from a database row.
fn task_from_row(row: &rusqlite::Row) -> rusqlite::Result<ClaudeCodeTask> {
    let status_str: String = row.get(5)?;
    let status = status_str.parse().unwrap_or(TaskStatus::Running);

    Ok(ClaudeCodeTask {
        id: row.get(0)?,
        agent: row.get(1)?,
        conv_id: row.get(2)?,
        task_prompt: row.get(3)?,
        workdir: row.get(4)?,
        status,
        pid: row.get(6)?,
        started_at: row.get(7)?,
        completed_at: row.get(8)?,
        exit_code: row.get(9)?,
        output_summary: row.get(10)?,
    })
}

/// Get current timestamp in seconds since UNIX epoch.
fn current_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

/// Generate a short task ID (8 hex characters).
fn generate_task_id() -> String {
    use rand::Rng;
    let mut rng = rand::rng();
    let id: u32 = rng.random();
    format!("{:08x}", id)
}

/// Escape single quotes for shell commands.
fn escape_shell_arg(s: &str) -> String {
    // Replace single quotes with '\'' (end quote, escaped quote, start quote)
    s.replace('\'', "'\\''")
}

/// Expand ~ to home directory.
fn expand_home(path: &str) -> PathBuf {
    if path.starts_with("~/")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(&path[2..]);
    }
    PathBuf::from(path)
}

/// Check if a process with the given PID is still running.
pub fn is_process_running(pid: i32) -> bool {
    unsafe { libc::kill(pid, 0) == 0 }
}

/// Claude Code tool for delegating coding tasks.
pub struct ClaudeCodeTool {
    agent_name: String,
    task_store: Arc<Mutex<TaskStore>>,
    conv_id: Option<String>,
}

impl ClaudeCodeTool {
    /// Create a new Claude Code tool for the given agent.
    pub fn new(agent_name: String, task_store: Arc<Mutex<TaskStore>>) -> Self {
        Self {
            agent_name,
            task_store,
            conv_id: None,
        }
    }

    /// Create a new Claude Code tool with a conversation ID for notifications.
    pub fn with_conv_id(
        agent_name: String,
        task_store: Arc<Mutex<TaskStore>>,
        conv_id: Option<String>,
    ) -> Self {
        Self {
            agent_name,
            task_store,
            conv_id,
        }
    }
}

impl std::fmt::Debug for ClaudeCodeTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClaudeCodeTool")
            .field("agent_name", &self.agent_name)
            .finish()
    }
}

#[async_trait]
impl Tool for ClaudeCodeTool {
    fn name(&self) -> &str {
        "claude_code"
    }

    fn description(&self) -> &str {
        "Delegate a coding task to Claude Code. Runs in background - you will be notified when complete."
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The coding task/prompt for Claude Code"
                },
                "workdir": {
                    "type": "string",
                    "description": "Working directory (default: home directory)"
                }
            },
            "required": ["task"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let task = input.get("task").and_then(|v| v.as_str()).ok_or_else(|| {
            ToolError::InvalidInput("Missing or invalid 'task' field".to_string())
        })?;

        let workdir = input
            .get("workdir")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                dirs::home_dir()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|| ".".to_string())
            });

        // Generate task ID
        let task_id = generate_task_id();

        // Build command to run Claude Code in background
        let script_path = expand_home("~/clawd/scripts/run-claude.sh");
        let log_path = format!("/tmp/claude-{}.log", task_id);
        let escaped_task = escape_shell_arg(task);

        // Command: cd <workdir> && nohup <script> '<task>' > <log> 2>&1 & echo $!
        let cmd = format!(
            "cd '{}' && nohup '{}' '{}' > '{}' 2>&1 & echo $!",
            escape_shell_arg(&workdir),
            script_path.to_string_lossy(),
            escaped_task,
            log_path
        );

        // Launch the background process
        let output = Command::new("sh")
            .arg("-c")
            .arg(&cmd)
            .output()
            .await
            .map_err(|e| {
                ToolError::ExecutionFailed(format!("Failed to launch Claude Code: {}", e))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ToolError::ExecutionFailed(format!(
                "Failed to launch Claude Code: {}",
                stderr
            )));
        }

        // Parse PID from output
        let stdout = String::from_utf8_lossy(&output.stdout);
        let pid: i32 = stdout.trim().parse().map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to parse PID '{}': {}", stdout.trim(), e))
        })?;

        // Store task record
        {
            let store = self.task_store.lock().await;
            store
                .create_task(
                    &task_id,
                    &self.agent_name,
                    self.conv_id.as_deref(),
                    task,
                    &workdir,
                    pid,
                )
                .map_err(|e| ToolError::ExecutionFailed(format!("Failed to store task: {}", e)))?;
        }

        Ok(serde_json::json!({
            "task_id": task_id,
            "pid": pid,
            "message": format!("Task {} started (pid {}). You will be notified when complete.", task_id, pid)
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_task_status_roundtrip() {
        assert_eq!(TaskStatus::Running.to_string(), "running");
        assert_eq!(TaskStatus::Completed.to_string(), "completed");
        assert_eq!(TaskStatus::Failed.to_string(), "failed");

        assert_eq!(
            "running".parse::<TaskStatus>().unwrap(),
            TaskStatus::Running
        );
        assert_eq!(
            "completed".parse::<TaskStatus>().unwrap(),
            TaskStatus::Completed
        );
        assert_eq!("failed".parse::<TaskStatus>().unwrap(), TaskStatus::Failed);
    }

    #[test]
    fn test_escape_shell_arg() {
        assert_eq!(escape_shell_arg("hello"), "hello");
        assert_eq!(escape_shell_arg("it's a test"), "it'\\''s a test");
        assert_eq!(escape_shell_arg(""), "");
    }

    #[test]
    fn test_generate_task_id() {
        let id = generate_task_id();
        assert_eq!(id.len(), 8);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_task_store_crud() {
        let temp_file = NamedTempFile::new().unwrap();
        let store = TaskStore::open(temp_file.path()).unwrap();

        // Create a task with conv_id
        store
            .create_task(
                "test123",
                "gendry",
                Some("test-conv"),
                "Fix the bug",
                "/home/test",
                12345,
            )
            .unwrap();

        // Get the task
        let task = store.get_task("test123").unwrap().unwrap();
        assert_eq!(task.id, "test123");
        assert_eq!(task.agent, "gendry");
        assert_eq!(task.conv_id, Some("test-conv".to_string()));
        assert_eq!(task.task_prompt, "Fix the bug");
        assert_eq!(task.workdir, "/home/test");
        assert_eq!(task.status, TaskStatus::Running);
        assert_eq!(task.pid, Some(12345));
        assert!(task.completed_at.is_none());

        // Get running tasks
        let running = store.get_running_tasks().unwrap();
        assert_eq!(running.len(), 1);

        // Complete the task
        store
            .complete_task("test123", TaskStatus::Completed, 0, "All done!")
            .unwrap();

        // Verify completion
        let task = store.get_task("test123").unwrap().unwrap();
        assert_eq!(task.status, TaskStatus::Completed);
        assert_eq!(task.exit_code, Some(0));
        assert_eq!(task.output_summary, Some("All done!".to_string()));
        assert!(task.completed_at.is_some());

        // Running tasks should now be empty
        let running = store.get_running_tasks().unwrap();
        assert!(running.is_empty());

        // Delete the task
        store.delete_task("test123").unwrap();
        assert!(store.get_task("test123").unwrap().is_none());
    }

    #[test]
    fn test_get_tasks_for_agent() {
        let temp_file = NamedTempFile::new().unwrap();
        let store = TaskStore::open(temp_file.path()).unwrap();

        store
            .create_task("task1", "gendry", None, "Task 1", "/home", 1001)
            .unwrap();
        store
            .create_task("task2", "arya", Some("conv-a"), "Task 2", "/home", 1002)
            .unwrap();
        store
            .create_task("task3", "gendry", Some("conv-b"), "Task 3", "/home", 1003)
            .unwrap();

        let gendry_tasks = store.get_tasks_for_agent("gendry").unwrap();
        assert_eq!(gendry_tasks.len(), 2);

        let arya_tasks = store.get_tasks_for_agent("arya").unwrap();
        assert_eq!(arya_tasks.len(), 1);
        assert_eq!(arya_tasks[0].conv_id, Some("conv-a".to_string()));
    }
}
