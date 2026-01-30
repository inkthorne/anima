use std::collections::HashMap;
use std::fmt;
use tokio::sync::{mpsc, oneshot};
use serde::{Serialize, Deserialize};

/// Error types for messaging operations
#[derive(Debug, Clone)]
pub enum MessagingError {
    /// The recipient agent was not found
    AgentNotFound(String),
    /// The message channel is closed
    ChannelClosed,
    /// Timeout waiting for reply
    Timeout,
    /// The agent is not registered with a router
    NotRegistered,
}

impl fmt::Display for MessagingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MessagingError::AgentNotFound(id) => write!(f, "Agent not found: {}", id),
            MessagingError::ChannelClosed => write!(f, "Message channel closed"),
            MessagingError::Timeout => write!(f, "Timeout waiting for reply"),
            MessagingError::NotRegistered => write!(f, "Agent not registered with router"),
        }
    }
}

impl std::error::Error for MessagingError {}

/// A message between agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    /// Sender agent ID
    pub from: String,
    /// Recipient agent ID
    pub to: String,
    /// Message content
    pub content: String,
    /// For request-response pattern: unique ID to correlate replies
    pub reply_to: Option<String>,
}

impl AgentMessage {
    /// Create a new message
    pub fn new(from: impl Into<String>, to: impl Into<String>, content: impl Into<String>) -> Self {
        AgentMessage {
            from: from.into(),
            to: to.into(),
            content: content.into(),
            reply_to: None,
        }
    }

    /// Create a reply to this message
    pub fn reply(&self, content: impl Into<String>) -> Self {
        AgentMessage {
            from: self.to.clone(),
            to: self.from.clone(),
            content: content.into(),
            reply_to: self.reply_to.clone(),
        }
    }
}

/// Handle for sending messages to an agent
#[derive(Clone)]
pub struct AgentMailbox {
    /// The agent ID this mailbox belongs to
    pub agent_id: String,
    sender: mpsc::Sender<AgentMessage>,
}

impl AgentMailbox {
    /// Create a new mailbox
    pub fn new(agent_id: String, sender: mpsc::Sender<AgentMessage>) -> Self {
        AgentMailbox { agent_id, sender }
    }

    /// Send a message to this agent (fire and forget)
    pub async fn send(&self, from: &str, content: impl Into<String>) -> Result<(), MessagingError> {
        let msg = AgentMessage::new(from, &self.agent_id, content);
        self.sender
            .send(msg)
            .await
            .map_err(|_| MessagingError::ChannelClosed)
    }

    /// Send the full message to this agent
    pub async fn send_message(&self, msg: AgentMessage) -> Result<(), MessagingError> {
        self.sender
            .send(msg)
            .await
            .map_err(|_| MessagingError::ChannelClosed)
    }
}

/// Message router - knows about all agents and routes messages between them
pub struct MessageRouter {
    /// Map of agent ID to their message sender
    agents: HashMap<String, mpsc::Sender<AgentMessage>>,
    /// Pending reply channels for request-response pattern
    pending_replies: HashMap<String, oneshot::Sender<AgentMessage>>,
    /// Counter for generating unique reply IDs
    reply_counter: u64,
}

impl MessageRouter {
    /// Create a new message router
    pub fn new() -> Self {
        MessageRouter {
            agents: HashMap::new(),
            pending_replies: HashMap::new(),
            reply_counter: 0,
        }
    }

    /// Register an agent with the router
    /// Returns the receiver for incoming messages
    pub fn register(&mut self, agent_id: &str) -> mpsc::Receiver<AgentMessage> {
        let (tx, rx) = mpsc::channel(32);
        self.agents.insert(agent_id.to_string(), tx);
        rx
    }

    /// Unregister an agent from the router
    pub fn unregister(&mut self, agent_id: &str) {
        self.agents.remove(agent_id);
    }

    /// Get a mailbox for sending messages to an agent
    pub fn get_mailbox(&self, agent_id: &str) -> Option<AgentMailbox> {
        self.agents.get(agent_id).map(|sender| {
            AgentMailbox::new(agent_id.to_string(), sender.clone())
        })
    }

    /// Send a message to an agent
    pub async fn send(&self, msg: AgentMessage) -> Result<(), MessagingError> {
        if let Some(sender) = self.agents.get(&msg.to) {
            sender
                .send(msg)
                .await
                .map_err(|_| MessagingError::ChannelClosed)
        } else {
            Err(MessagingError::AgentNotFound(msg.to.clone()))
        }
    }

    /// Generate a unique reply ID
    pub fn generate_reply_id(&mut self) -> String {
        self.reply_counter += 1;
        format!("reply-{}", self.reply_counter)
    }

    /// Register a pending reply channel
    pub fn register_reply(&mut self, reply_id: String, tx: oneshot::Sender<AgentMessage>) {
        self.pending_replies.insert(reply_id, tx);
    }

    /// Complete a pending reply (called when a reply message is received)
    pub fn complete_reply(&mut self, reply_id: &str, msg: AgentMessage) -> bool {
        if let Some(tx) = self.pending_replies.remove(reply_id) {
            tx.send(msg).is_ok()
        } else {
            false
        }
    }

    /// List all registered agent IDs
    pub fn list_agents(&self) -> Vec<String> {
        self.agents.keys().cloned().collect()
    }

    /// Check if an agent is registered
    pub fn has_agent(&self, agent_id: &str) -> bool {
        self.agents.contains_key(agent_id)
    }

    /// Get the number of registered agents
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }
}

impl Default for MessageRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_message_new() {
        let msg = AgentMessage::new("agent-1", "agent-2", "hello");
        assert_eq!(msg.from, "agent-1");
        assert_eq!(msg.to, "agent-2");
        assert_eq!(msg.content, "hello");
        assert!(msg.reply_to.is_none());
    }

    #[test]
    fn test_agent_message_reply() {
        let msg = AgentMessage {
            from: "agent-1".to_string(),
            to: "agent-2".to_string(),
            content: "hello".to_string(),
            reply_to: Some("req-123".to_string()),
        };
        let reply = msg.reply("hi back");
        assert_eq!(reply.from, "agent-2");
        assert_eq!(reply.to, "agent-1");
        assert_eq!(reply.content, "hi back");
        assert_eq!(reply.reply_to, Some("req-123".to_string()));
    }

    #[test]
    fn test_message_router_new() {
        let router = MessageRouter::new();
        assert_eq!(router.agent_count(), 0);
        assert!(router.list_agents().is_empty());
    }

    #[test]
    fn test_message_router_register() {
        let mut router = MessageRouter::new();
        let _rx = router.register("agent-1");

        assert!(router.has_agent("agent-1"));
        assert_eq!(router.agent_count(), 1);
        assert_eq!(router.list_agents(), vec!["agent-1"]);
    }

    #[test]
    fn test_message_router_unregister() {
        let mut router = MessageRouter::new();
        let _rx = router.register("agent-1");
        router.unregister("agent-1");

        assert!(!router.has_agent("agent-1"));
        assert_eq!(router.agent_count(), 0);
    }

    #[test]
    fn test_message_router_get_mailbox() {
        let mut router = MessageRouter::new();
        let _rx = router.register("agent-1");

        let mailbox = router.get_mailbox("agent-1");
        assert!(mailbox.is_some());
        assert_eq!(mailbox.unwrap().agent_id, "agent-1");

        let no_mailbox = router.get_mailbox("nonexistent");
        assert!(no_mailbox.is_none());
    }

    #[test]
    fn test_generate_reply_id() {
        let mut router = MessageRouter::new();
        let id1 = router.generate_reply_id();
        let id2 = router.generate_reply_id();

        assert!(id1.starts_with("reply-"));
        assert!(id2.starts_with("reply-"));
        assert_ne!(id1, id2);
    }

    #[tokio::test]
    async fn test_message_router_send() {
        let mut router = MessageRouter::new();
        let mut rx = router.register("agent-1");

        let msg = AgentMessage::new("agent-2", "agent-1", "hello");
        router.send(msg).await.unwrap();

        let received = rx.recv().await.unwrap();
        assert_eq!(received.from, "agent-2");
        assert_eq!(received.to, "agent-1");
        assert_eq!(received.content, "hello");
    }

    #[tokio::test]
    async fn test_message_router_send_not_found() {
        let router = MessageRouter::new();
        let msg = AgentMessage::new("agent-1", "nonexistent", "hello");

        let result = router.send(msg).await;
        assert!(matches!(result, Err(MessagingError::AgentNotFound(_))));
    }

    #[tokio::test]
    async fn test_mailbox_send() {
        let mut router = MessageRouter::new();
        let mut rx = router.register("agent-1");

        let mailbox = router.get_mailbox("agent-1").unwrap();
        mailbox.send("agent-2", "hello").await.unwrap();

        let received = rx.recv().await.unwrap();
        assert_eq!(received.from, "agent-2");
        assert_eq!(received.content, "hello");
    }

    #[test]
    fn test_register_and_complete_reply() {
        let mut router = MessageRouter::new();
        let (tx, mut rx) = oneshot::channel();

        let reply_id = router.generate_reply_id();
        router.register_reply(reply_id.clone(), tx);

        let reply_msg = AgentMessage::new("agent-1", "agent-2", "response");
        let completed = router.complete_reply(&reply_id, reply_msg.clone());
        assert!(completed);

        // The reply should be received
        let received = rx.try_recv().unwrap();
        assert_eq!(received.content, "response");
    }

    #[test]
    fn test_complete_reply_not_found() {
        let mut router = MessageRouter::new();
        let msg = AgentMessage::new("agent-1", "agent-2", "response");

        let completed = router.complete_reply("nonexistent", msg);
        assert!(!completed);
    }

    #[test]
    fn test_messaging_error_display() {
        let err = MessagingError::AgentNotFound("agent-1".to_string());
        assert!(err.to_string().contains("agent-1"));

        let err = MessagingError::ChannelClosed;
        assert!(err.to_string().contains("closed"));

        let err = MessagingError::Timeout;
        assert!(err.to_string().contains("Timeout"));

        let err = MessagingError::NotRegistered;
        assert!(err.to_string().contains("not registered"));
    }
}
