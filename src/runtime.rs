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

    pub fn spawn_agent(&mut self, id: String) -> &mut Agent {
        let agent = Agent::new(id.clone());
        self.agents.insert(id, agent);
        self.agents.get_mut(&id).unwrap()
    }

    pub fn get_agent(&self, id: &str) -> Option<&Agent> {
        self.agents.get(id)
    }

    pub fn get_agent_mut(&mut self, id: &str) -> Option<&mut Agent> {
        self.agents.get_mut(id)
    }
}
