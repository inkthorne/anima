use std::collections::HashMap;

use crate::error::ToolError;
use crate::tool::{Tool, ToolInfo};
use serde_json::Value;

pub struct Agent {
    pub id: String,
    tools: HashMap<String, Box<dyn Tool>>,
}

impl Agent {
    pub fn new(id: String) -> Self {
        Agent {
            id,
            tools: HashMap::new(),
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

    pub fn list_tools(&self) -> Vec<ToolInfo> {
        self.tools
            .values()
            .map(|tool| ToolInfo {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
            })
            .collect()
    }
}
