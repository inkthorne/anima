use crate::agent::Agent;
use std::collections::HashMap;

pub struct Runtime {
    agents: HashMap<String, Agent>,
}

impl Runtime {
    pub fn new() -> Self {
        Runtime {
            agents: HashMap::new(),
        }
    }

    /// Spawn a new agent with the given ID
    pub fn spawn_agent(&mut self, id: String) -> &mut Agent {
        // Create the agent with the provided ID, then insert it
        let agent = Agent::new(id.clone());
        self.agents.insert(id.clone(), agent);
        // Get the mutable reference by using the same key (String is Copy for the HashMap lookup)
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
}
