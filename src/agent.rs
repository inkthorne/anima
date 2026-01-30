use std::collections::HashMap;

use crate::error::ToolError;
use crate::tool::{Tool, ToolInfo};
use crate::message::Message;
use tokio::sync::mpsc;
use serde_json::Value;

pub struct Agent {
    pub id: String,
    tools: HashMap<String, Box<dyn Tool>>,
    inbox: mpsc::Receiver<Message>,
}

impl Agent {
    pub fn new(id: String, inbox: mpsc::Receiver<Message>) -> Self {
        Agent {
            id,
            tools: HashMap::new(),
            inbox,
        }
    }

    pub fn register_tool(&mut self, tool: Box<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub async fn call_tool(&self, name: &str, input: &str) -> Result<String, ToolError> {
        if let Some(tool) = self.tools.get(name) {
            let input_value: Value =
                serde_json::from_str(input).map_err(|e| ToolError::InvalidInput(e.to_string()))?;
            let result = tool.execute(input_value).await?;
            Ok(result.to_string())
        } else {
            Err(ToolError::ExecutionFailed(format!(
                "Tool '{}' not found",
                name
            )))
        }
    }

    pub async fn recv(&mut self) -> Option<Message> {
        self.inbox.recv().await
    }
}
