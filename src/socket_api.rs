//! Unix socket API for daemon communication.
//!
//! This module provides a simple JSON protocol over Unix sockets for
//! communicating with running agent daemons.

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

/// Request types for the socket API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    /// Send a message to the agent and get a response (from user/REPL).
    Message {
        content: String,
    },
    /// Incoming message from another agent (inter-daemon communication).
    IncomingMessage {
        /// The sender agent's name
        from: String,
        /// The message content
        content: String,
    },
    /// Notify agent about a new message in a conversation (multi-agent communication).
    /// The agent should fetch context from the conversation store and respond.
    Notify {
        /// The conversation ID
        conv_id: String,
        /// The message ID that triggered this notification
        message_id: i64,
        /// Depth of the @mention chain (for preventing infinite loops)
        /// Defaults to 0 for backwards compatibility
        #[serde(default)]
        depth: u32,
    },
    /// Get the current status of the daemon.
    Status,
    /// Request a graceful shutdown.
    Shutdown,
    /// Clear the agent's conversation history.
    Clear,
    /// List all agents visible to this daemon.
    ListAgents,
    /// Get the system prompt (persona) for this agent.
    System,
    /// Manually trigger a heartbeat.
    Heartbeat,
}

/// Response types for the socket API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Response {
    /// Response to a message request.
    Message {
        content: String,
    },
    /// Response to a status request.
    Status {
        running: bool,
        history_len: usize,
    },
    /// Response to a list_agents request.
    Agents {
        agents: Vec<String>,
    },
    /// Response to a system prompt request.
    System {
        persona: String,
    },
    /// Generic OK response.
    Ok,
    /// Error response.
    Error {
        message: String,
    },
    /// Streaming: partial text chunk.
    Chunk {
        text: String,
    },
    /// Streaming: tool is being executed.
    ToolCall {
        tool: String,
        params: serde_json::Value,
    },
    /// Streaming: complete.
    Done,
    /// Response to a Notify request - agent has processed the notification.
    Notified {
        /// The message ID of the agent's response (stored in conversation)
        response_message_id: i64,
    },
    /// Immediate acknowledgment for Notify request (fire-and-forget).
    /// The daemon will process the notification asynchronously.
    NotifyReceived,
    /// Response to a heartbeat request.
    HeartbeatTriggered,
    /// Response when heartbeat is not configured for the agent.
    HeartbeatNotConfigured,
}

/// Socket API handler for reading and writing protocol messages.
pub struct SocketApi {
    reader: BufReader<tokio::io::ReadHalf<UnixStream>>,
    writer: tokio::io::WriteHalf<UnixStream>,
}

impl SocketApi {
    /// Create a new SocketApi from a UnixStream.
    pub fn new(stream: UnixStream) -> Self {
        let (reader, writer) = tokio::io::split(stream);
        Self {
            reader: BufReader::new(reader),
            writer,
        }
    }

    /// Read a request from the socket.
    /// Returns None if the connection is closed.
    pub async fn read_request(&mut self) -> Result<Option<Request>, SocketApiError> {
        let mut line = String::new();
        let bytes_read = self.reader.read_line(&mut line).await
            .map_err(SocketApiError::Io)?;

        if bytes_read == 0 {
            return Ok(None);
        }

        let request: Request = serde_json::from_str(line.trim())
            .map_err(SocketApiError::Json)?;

        Ok(Some(request))
    }

    /// Write a response to the socket.
    pub async fn write_response(&mut self, response: &Response) -> Result<(), SocketApiError> {
        let json = serde_json::to_string(response)
            .map_err(SocketApiError::Json)?;

        self.writer.write_all(json.as_bytes()).await
            .map_err(SocketApiError::Io)?;
        self.writer.write_all(b"\n").await
            .map_err(SocketApiError::Io)?;
        self.writer.flush().await
            .map_err(SocketApiError::Io)?;

        Ok(())
    }

    /// Write a request to the socket (client-side).
    pub async fn write_request(&mut self, request: &Request) -> Result<(), SocketApiError> {
        let json = serde_json::to_string(request)
            .map_err(SocketApiError::Json)?;

        self.writer.write_all(json.as_bytes()).await
            .map_err(SocketApiError::Io)?;
        self.writer.write_all(b"\n").await
            .map_err(SocketApiError::Io)?;
        self.writer.flush().await
            .map_err(SocketApiError::Io)?;

        Ok(())
    }

    /// Read a response from the socket (client-side).
    /// Returns None if the connection is closed.
    pub async fn read_response(&mut self) -> Result<Option<Response>, SocketApiError> {
        let mut line = String::new();
        let bytes_read = self.reader.read_line(&mut line).await
            .map_err(SocketApiError::Io)?;

        if bytes_read == 0 {
            return Ok(None);
        }

        let response: Response = serde_json::from_str(line.trim())
            .map_err(SocketApiError::Json)?;

        Ok(Some(response))
    }
}

/// Errors that can occur in the socket API.
#[derive(Debug)]
pub enum SocketApiError {
    Io(std::io::Error),
    Json(serde_json::Error),
}

impl std::fmt::Display for SocketApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SocketApiError::Io(e) => write!(f, "IO error: {}", e),
            SocketApiError::Json(e) => write!(f, "JSON error: {}", e),
        }
    }
}

impl std::error::Error for SocketApiError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SocketApiError::Io(e) => Some(e),
            SocketApiError::Json(e) => Some(e),
        }
    }
}

// Make SocketApiError Send + Sync for use with tokio
unsafe impl Send for SocketApiError {}
unsafe impl Sync for SocketApiError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_message_serialization() {
        let request = Request::Message {
            content: "Hello, agent!".to_string(),
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"type\":\"message\""));
        assert!(json.contains("\"content\":\"Hello, agent!\""));

        // Deserialize back
        let parsed: Request = serde_json::from_str(&json).unwrap();
        match parsed {
            Request::Message { content } => assert_eq!(content, "Hello, agent!"),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_request_status_serialization() {
        let request = Request::Status;
        let json = serde_json::to_string(&request).unwrap();
        assert_eq!(json, r#"{"type":"status"}"#);

        let parsed: Request = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, Request::Status));
    }

    #[test]
    fn test_request_shutdown_serialization() {
        let request = Request::Shutdown;
        let json = serde_json::to_string(&request).unwrap();
        assert_eq!(json, r#"{"type":"shutdown"}"#);

        let parsed: Request = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, Request::Shutdown));
    }

    #[test]
    fn test_request_clear_serialization() {
        let request = Request::Clear;
        let json = serde_json::to_string(&request).unwrap();
        assert_eq!(json, r#"{"type":"clear"}"#);

        let parsed: Request = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, Request::Clear));
    }

    #[test]
    fn test_response_message_serialization() {
        let response = Response::Message {
            content: "Hello back!".to_string(),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"type\":\"message\""));
        assert!(json.contains("\"content\":\"Hello back!\""));

        let parsed: Response = serde_json::from_str(&json).unwrap();
        match parsed {
            Response::Message { content } => assert_eq!(content, "Hello back!"),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_response_status_serialization() {
        let response = Response::Status {
            running: true,
            history_len: 5,
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"type\":\"status\""));
        assert!(json.contains("\"running\":true"));
        assert!(json.contains("\"history_len\":5"));

        let parsed: Response = serde_json::from_str(&json).unwrap();
        match parsed {
            Response::Status { running, history_len } => {
                assert!(running);
                assert_eq!(history_len, 5);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_response_ok_serialization() {
        let response = Response::Ok;
        let json = serde_json::to_string(&response).unwrap();
        assert_eq!(json, r#"{"type":"ok"}"#);

        let parsed: Response = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, Response::Ok));
    }

    #[test]
    fn test_response_error_serialization() {
        let response = Response::Error {
            message: "Something went wrong".to_string(),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"type\":\"error\""));
        assert!(json.contains("\"message\":\"Something went wrong\""));

        let parsed: Response = serde_json::from_str(&json).unwrap();
        match parsed {
            Response::Error { message } => assert_eq!(message, "Something went wrong"),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_socket_api_error_display() {
        let io_error = SocketApiError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        ));
        assert!(io_error.to_string().contains("IO error"));

        let json_error = SocketApiError::Json(
            serde_json::from_str::<Request>("invalid json").unwrap_err()
        );
        assert!(json_error.to_string().contains("JSON error"));
    }

    #[test]
    fn test_request_incoming_message_serialization() {
        let request = Request::IncomingMessage {
            from: "agent-1".to_string(),
            content: "Hello from agent-1!".to_string(),
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"type\":\"incoming_message\""));
        assert!(json.contains("\"from\":\"agent-1\""));
        assert!(json.contains("\"content\":\"Hello from agent-1!\""));

        let parsed: Request = serde_json::from_str(&json).unwrap();
        match parsed {
            Request::IncomingMessage { from, content } => {
                assert_eq!(from, "agent-1");
                assert_eq!(content, "Hello from agent-1!");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_request_list_agents_serialization() {
        let request = Request::ListAgents;
        let json = serde_json::to_string(&request).unwrap();
        assert_eq!(json, r#"{"type":"list_agents"}"#);

        let parsed: Request = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, Request::ListAgents));
    }

    #[test]
    fn test_response_agents_serialization() {
        let response = Response::Agents {
            agents: vec!["agent-1".to_string(), "agent-2".to_string()],
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"type\":\"agents\""));
        assert!(json.contains("\"agents\""));
        assert!(json.contains("agent-1"));
        assert!(json.contains("agent-2"));

        let parsed: Response = serde_json::from_str(&json).unwrap();
        match parsed {
            Response::Agents { agents } => {
                assert_eq!(agents.len(), 2);
                assert!(agents.contains(&"agent-1".to_string()));
                assert!(agents.contains(&"agent-2".to_string()));
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_request_system_serialization() {
        let request = Request::System;
        let json = serde_json::to_string(&request).unwrap();
        assert_eq!(json, r#"{"type":"system"}"#);

        let parsed: Request = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, Request::System));
    }

    #[test]
    fn test_response_system_serialization() {
        let response = Response::System {
            persona: "You are a helpful assistant.".to_string(),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"type\":\"system\""));
        assert!(json.contains("\"persona\":\"You are a helpful assistant.\""));

        let parsed: Response = serde_json::from_str(&json).unwrap();
        match parsed {
            Response::System { persona } => {
                assert_eq!(persona, "You are a helpful assistant.");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_response_chunk_serialization() {
        let response = Response::Chunk {
            text: "Hello".to_string(),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"type\":\"chunk\""));
        assert!(json.contains("\"text\":\"Hello\""));

        let parsed: Response = serde_json::from_str(&json).unwrap();
        match parsed {
            Response::Chunk { text } => {
                assert_eq!(text, "Hello");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_response_done_serialization() {
        let response = Response::Done;
        let json = serde_json::to_string(&response).unwrap();
        assert_eq!(json, r#"{"type":"done"}"#);

        let parsed: Response = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, Response::Done));
    }

    #[test]
    fn test_response_tool_call_serialization() {
        let response = Response::ToolCall {
            tool: "safe_shell".to_string(),
            params: serde_json::json!({"command": "ls -la"}),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"type\":\"tool_call\""));
        assert!(json.contains("\"tool\":\"safe_shell\""));
        assert!(json.contains("\"command\":\"ls -la\""));

        let parsed: Response = serde_json::from_str(&json).unwrap();
        match parsed {
            Response::ToolCall { tool, params } => {
                assert_eq!(tool, "safe_shell");
                assert_eq!(params.get("command").and_then(|c| c.as_str()), Some("ls -la"));
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_request_notify_serialization() {
        let request = Request::Notify {
            conv_id: "1:1:arya:user".to_string(),
            message_id: 42,
            depth: 5,
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"type\":\"notify\""));
        assert!(json.contains("\"conv_id\":\"1:1:arya:user\""));
        assert!(json.contains("\"message_id\":42"));
        assert!(json.contains("\"depth\":5"));

        let parsed: Request = serde_json::from_str(&json).unwrap();
        match parsed {
            Request::Notify { conv_id, message_id, depth } => {
                assert_eq!(conv_id, "1:1:arya:user");
                assert_eq!(message_id, 42);
                assert_eq!(depth, 5);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_request_notify_default_depth() {
        // Test backwards compatibility: depth should default to 0 if not provided
        let json = r#"{"type":"notify","conv_id":"test-conv","message_id":10}"#;
        let parsed: Request = serde_json::from_str(json).unwrap();
        match parsed {
            Request::Notify { conv_id, message_id, depth } => {
                assert_eq!(conv_id, "test-conv");
                assert_eq!(message_id, 10);
                assert_eq!(depth, 0);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_response_notified_serialization() {
        let response = Response::Notified {
            response_message_id: 123,
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"type\":\"notified\""));
        assert!(json.contains("\"response_message_id\":123"));

        let parsed: Response = serde_json::from_str(&json).unwrap();
        match parsed {
            Response::Notified { response_message_id } => {
                assert_eq!(response_message_id, 123);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_response_notify_received_serialization() {
        let response = Response::NotifyReceived;
        let json = serde_json::to_string(&response).unwrap();
        assert_eq!(json, r#"{"type":"notify_received"}"#);

        let parsed: Response = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, Response::NotifyReceived));
    }

    #[test]
    fn test_request_heartbeat_serialization() {
        let request = Request::Heartbeat;
        let json = serde_json::to_string(&request).unwrap();
        assert_eq!(json, r#"{"type":"heartbeat"}"#);

        let parsed: Request = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, Request::Heartbeat));
    }

    #[test]
    fn test_response_heartbeat_triggered_serialization() {
        let response = Response::HeartbeatTriggered;
        let json = serde_json::to_string(&response).unwrap();
        assert_eq!(json, r#"{"type":"heartbeat_triggered"}"#);

        let parsed: Response = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, Response::HeartbeatTriggered));
    }

    #[test]
    fn test_response_heartbeat_not_configured_serialization() {
        let response = Response::HeartbeatNotConfigured;
        let json = serde_json::to_string(&response).unwrap();
        assert_eq!(json, r#"{"type":"heartbeat_not_configured"}"#);

        let parsed: Response = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, Response::HeartbeatNotConfigured));
    }
}
