use crate::agent::Agent;
use crate::message::Message;
use crate::memory::Memory;
use std::collections::HashMap;
use tokio::sync::mpsc;

pub struct Runtime {
    agents: HashMap<String, Agent>,
    senders: HashMap<String, mpsc::Sender<Message>>,
}

impl Runtime {
    pub fn new() -> Self {
        Runtime {
            agents: HashMap::new(),
            senders: HashMap::new(),
        }
    }

    /// Spawn a new agent with the given ID
    pub fn spawn_agent(&mut self, id: String) -> Agent {
        let (tx, rx) = mpsc::channel(32);
        let agent = Agent::new(id.clone(), rx);
        self.agents.insert(id.clone(), agent);
        self.agents.remove(&id).unwrap()
    }

    /// Spawn a new agent with the given ID and memory
    pub fn spawn_agent_with_memory(&mut self, id: String, memory: Box<dyn Memory>) -> &mut Agent {
        let (tx, rx) = mpsc::channel(32);
        let agent = Agent::new(id.clone(), rx).with_memory(memory);
        self.agents.insert(id.clone(), agent);
        self.senders.insert(id.clone(), tx);
        self.agents.get_mut(&id).unwrap()
    }

    /// Get an immutable reference to an agent by ID
    pub fn get_agent(&self, id: &str) -> Option<&Agent> {
        self.agents.get(id)
    }

    /// Get a mutable reference to an agent by ID
    pub fn get_agent_mut(&mut self, id: &str) -> Option<&mut Agent> {
        self.agents.get_mut(id)
    }

    /// Remove an agent by ID and return it if it existed
    pub fn remove_agent(&mut self, id: &str) -> Option<Agent> {
        self.agents.remove(id)
    }

    /// Check if an agent exists with the given ID
    pub fn has_agent(&self, id: &str) -> bool {
        self.agents.contains_key(id)
    }

    /// Get a list of all agent IDs
    pub fn agent_ids(&self) -> Vec<String> {
        self.agents.keys().cloned().collect()
    }

    /// Get the number of agents
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    /// Send a message to an agent
    pub async fn send_message(&self, msg: Message) -> Result<(), String> {
        if let Some(sender) = self.senders.get(&msg.to) {
            sender.send(msg).await.map_err(|e| e.to_string())
        } else {
            Err(format!("Agent not found: {}", msg.to))
        }
    }
}
