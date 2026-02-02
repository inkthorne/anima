# Task: Semantic Memory System

Build: `cargo build`
Test: `cargo test`

## Overview

Implement a semantic memory system for anima agents. Memories are automatically injected into context before each turn (recall) and can be saved via `[REMEMBER: ...]` tags in agent output (save). Works for all models — no tool support required.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     DAEMON MESSAGE LOOP                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  BEFORE turn:                                           │
│    1. Embed incoming message (or use keywords)          │
│    2. Query memory.db for relevant memories             │
│    3. Score: relevance × recency × importance           │
│    4. Inject top N as system context                    │
│                                                         │
│  AFTER turn:                                            │
│    1. Parse agent output for [REMEMBER: ...] tags       │
│    2. Store flagged memories with high importance       │
│    3. Strip tags from output before returning           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Schema (memory.db)

```sql
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at INTEGER NOT NULL,        -- unix timestamp
    content TEXT NOT NULL,              -- the actual memory
    importance REAL DEFAULT 0.5,        -- 0.0-1.0, higher = more important
    source TEXT DEFAULT 'auto',         -- 'explicit', 'auto', 'conversation'
    keywords TEXT,                      -- comma-separated for simple search
    access_count INTEGER DEFAULT 0,     -- how often recalled
    last_accessed INTEGER               -- last recall timestamp
);

CREATE INDEX IF NOT EXISTS idx_memories_keywords ON memories(keywords);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
```

**Note:** Embeddings (BLOB column + vector search) can be added later. Start with keyword-based search for simplicity.

## New Module: `src/memory.rs`

```rust
pub struct Memory {
    pub id: i64,
    pub created_at: i64,
    pub content: String,
    pub importance: f64,
    pub source: String,
    pub keywords: Option<String>,
    pub access_count: i64,
    pub last_accessed: Option<i64>,
}

pub struct MemoryStore {
    conn: Connection,
}

impl MemoryStore {
    /// Open or create memory.db at the given path
    pub fn open(path: &Path) -> Result<Self>;
    
    /// Save a new memory
    pub fn save(&self, content: &str, importance: f64, source: &str) -> Result<i64>;
    
    /// Recall relevant memories for a query
    /// Returns top N memories scored by relevance × recency × importance
    pub fn recall(&self, query: &str, limit: usize) -> Result<Vec<Memory>>;
    
    /// Update access stats when a memory is used
    fn mark_accessed(&self, id: i64) -> Result<()>;
}
```

## Recall Algorithm

```rust
fn recall(&self, query: &str, limit: usize) -> Result<Vec<Memory>> {
    // 1. Extract keywords from query (simple: split on whitespace, filter stopwords)
    let keywords = extract_keywords(query);
    
    // 2. Query memories that match any keyword
    let candidates = self.search_by_keywords(&keywords)?;
    
    // 3. Score each candidate
    let now = unix_timestamp();
    let mut scored: Vec<(Memory, f64)> = candidates
        .into_iter()
        .map(|m| {
            let relevance = keyword_overlap(&keywords, &m.keywords);
            let recency = recency_score(m.created_at, now);  // e.g., 1.0 for today, decays over time
            let score = relevance * recency * m.importance;
            (m, score)
        })
        .collect();
    
    // 4. Sort by score descending, take top N
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.truncate(limit);
    
    // 5. Mark accessed
    for (m, _) in &scored {
        self.mark_accessed(m.id)?;
    }
    
    Ok(scored.into_iter().map(|(m, _)| m).collect())
}

fn recency_score(created_at: i64, now: i64) -> f64 {
    let age_hours = (now - created_at) as f64 / 3600.0;
    // Exponential decay: half-life of ~7 days (168 hours)
    0.5_f64.powf(age_hours / 168.0)
}
```

## Memory Injection (in daemon.rs)

Before sending a message to the agent:

```rust
fn inject_memories(agent: &Agent, user_message: &str) -> String {
    let memories = agent.memory_store.recall(user_message, 5)?;
    
    if memories.is_empty() {
        return String::new();
    }
    
    let mut injection = String::from("[Relevant memories]\n");
    for m in memories {
        let age = format_age(m.created_at);  // "2h ago", "yesterday", etc.
        let flag = if m.importance > 0.8 { " ⭐" } else { "" };
        injection.push_str(&format!("- ({age}) {}{flag}\n", m.content));
    }
    injection.push('\n');
    
    injection
}
```

Inject this as a system message right before the user message, or prepend to the always.md content.

## Memory Extraction (in daemon.rs)

After receiving agent output:

```rust
fn extract_and_store_memories(agent: &Agent, output: &str) -> String {
    let re = Regex::new(r"\[REMEMBER:\s*(.+?)\]").unwrap();
    let mut cleaned = output.to_string();
    
    for cap in re.captures_iter(output) {
        let memory_content = cap[1].trim();
        agent.memory_store.save(memory_content, 0.9, "explicit")?;  // High importance
        
        // Remove the tag from output
        cleaned = cleaned.replace(&cap[0], "");
    }
    
    cleaned.trim().to_string()
}
```

## Integration Points

1. **Agent struct** — add `memory_store: MemoryStore` field
2. **Agent::new()** — initialize MemoryStore from agent directory
3. **Daemon message handler** — call `inject_memories()` before LLM call
4. **Daemon response handler** — call `extract_and_store_memories()` after LLM response

## Config (config.toml)

```toml
[memory]
enabled = true
path = "memory.db"          # relative to agent directory
recall_limit = 5            # max memories to inject
min_importance = 0.1        # filter out low-importance memories
```

## Test Cases

1. **Save and recall** — save a memory, recall it with matching query
2. **Recency scoring** — recent memories rank higher than old ones
3. **Importance scoring** — explicit saves (0.9) rank higher than auto (0.5)
4. **Tag extraction** — `[REMEMBER: X]` is parsed and stripped
5. **Multiple tags** — multiple `[REMEMBER: ...]` in one response
6. **No memories** — empty injection when nothing matches
7. **Keyword extraction** — stopwords filtered, meaningful terms kept

## Future Enhancements (Not Now)

- [ ] Embedding-based semantic search (add `embedding BLOB` column)
- [ ] Auto-capture from conversations (extract facts/decisions)
- [ ] Memory consolidation (merge similar memories)
- [ ] Forgetting (prune old low-importance memories)

## Files to Create/Modify

- [x] `src/memory.rs` — added SemanticMemoryStore, SemanticMemoryEntry, scoring functions
- [x] `src/lib.rs` — added exports for semantic memory types
- [x] `src/agent_dir.rs` — added SemanticMemorySection config
- [x] `src/daemon.rs` — inject/extract memories in message loop
- [x] `src/config.rs` — added SemanticMemorySection (for reference)
- [x] `Cargo.toml` — regex already present
