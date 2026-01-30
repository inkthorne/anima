use crate::agent::Agent;
use crate::message::Message;
use crate::memory::Memory;
use crate::messaging::MessageRouter;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::Mutex;

pub struct Runtime {
    agents: HashMap<String, Agent>,
    senders: HashMap<String, mpsc::Sender<Message>>,
    parent_map: HashMap<String, String>,
    /// Message router for agent-to-agent communication
    router: Arc<Mutex<MessageRouter>>,
}

impl Runtime {
    pub fn new() -> Self {
        Runtime {
            agents: HashMap::new(),
            senders: HashMap::new(),
            parent_map: HashMap::new(),
            router: Arc::new(Mutex::new(MessageRouter::new())),
        }
    }

    /// Get a reference to the message router
    pub fn router(&self) -> &Arc<Mutex<MessageRouter>> {
        &self.router
    }

    /// Spawn a new agent with the given ID (auto-registers with message router)
    pub fn spawn_agent(&mut self, id: String) -> Agent {
        let (tx, rx) = mpsc::channel(32);

        // Register with message router
        let message_rx = {
            let mut router = self.router.blocking_lock();
            router.register(&id)
        };

        let agent = Agent::new(id.clone(), rx)
            .with_router_and_rx(self.router.clone(), message_rx);
        self.agents.insert(id.clone(), agent);
        self.agents.remove(&id).unwrap()
    }

    /// Spawn a new agent with the given ID and memory (auto-registers with message router)
    pub fn spawn_agent_with_memory(&mut self, id: String, memory: Box<dyn Memory>) -> &mut Agent {
        let (tx, rx) = mpsc::channel::<Message>(32);

        // Register with message router
        let message_rx = {
            let mut router = self.router.blocking_lock();
            router.register(&id)
        };

        let agent = Agent::new(id.clone(), rx)
            .with_memory(memory)
            .with_router_and_rx(self.router.clone(), message_rx);
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
        // Unregister from message router
        {
            let mut router = self.router.blocking_lock();
            router.unregister(id);
        }
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

    /// Get the parent of an agent
    pub fn get_parent(&self, agent_id: &str) -> Option<&str> {
        self.parent_map.get(agent_id).map(|s| s.as_str())
    }

    /// Get the children of an agent
    pub fn get_children(&self, agent_id: &str) -> Vec<&str> {
        self.parent_map.iter()
            .filter(|(_, parent)| *parent == agent_id)
            .map(|(child, _)| child.as_str())
            .collect()
    }

    /// Run a child agent task asynchronously
    // pub async fn run_child_task(
    //     agent: &mut crate::agent::Agent,
    //     task: &str,
    //     result_tx: tokio::sync::oneshot::Sender<Result<String, String>>,
    // ) {
    //     let result = agent.think(task).await;
    //     let _ = result_tx.send(result.map_err(|e| e.to_string()));
    // }

    /// Terminate a specific child agent
    pub fn terminate_child(&mut self, child_id: &str) {
        // Unregister from message router
        {
            let mut router = self.router.blocking_lock();
            router.unregister(child_id);
        }
        self.agents.remove(child_id);
        self.parent_map.remove(child_id);
    }

    /// Terminate all children of an agent (recursive)
    pub fn terminate_children(&mut self, parent_id: &str) {
        let children: Vec<String> = self.parent_map
            .iter()
            .filter(|(_, p)| *p == parent_id)
            .map(|(c, _)| c.clone())
            .collect();
        
        for child_id in children {
            self.terminate_children(&child_id); // Recursive
            self.terminate_child(&child_id);
        }
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
