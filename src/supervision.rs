use tokio::sync::oneshot;

/// Status of a child agent
#[derive(Debug, Clone)]
pub enum ChildStatus {
    Running,
    Completed(String),
    Failed(String),
}

/// Configuration for spawning a child agent
#[derive(Default)]
pub struct ChildConfig {
    pub task: String,
    pub inherit_llm: bool,
    pub inherit_memory: bool,
}

impl ChildConfig {
    pub fn new(task: impl Into<String>) -> Self {
        Self {
            task: task.into(),
            inherit_llm: true,
            inherit_memory: false,
        }
    }
}

/// Handle to a spawned child agent
pub struct ChildHandle {
    pub child_id: String,
    pub task: String,
    pub status: ChildStatus,
    pub result_rx: Option<oneshot::Receiver<Result<String, String>>>,
}

impl ChildHandle {
    pub fn new(
        child_id: String,
        task: String,
        result_rx: oneshot::Receiver<Result<String, String>>,
    ) -> Self {
        Self {
            child_id,
            task,
            status: ChildStatus::Running,
            result_rx: Some(result_rx),
        }
    }
}
