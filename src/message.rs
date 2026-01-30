use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub from: String,
    pub to: String,
    pub payload: Value,
}

impl Message {
    pub fn new(from: impl Into<String>, to: impl Into<String>, payload: Value) -> Self {
        Message {
            from: from.into(),
            to: to.into(),
            payload,
        }
    }
}
