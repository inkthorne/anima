# Task: Embedding-Based Semantic Memory

**Goal:** Replace keyword-based recall with embedding-based semantic search using Ollama.

**Build:** `cargo build --release && cargo test`

## Config Changes

`~/.anima/agents/<name>/config.toml`:
```toml
[semantic_memory]
enabled = true

[semantic_memory.embedding]
provider = "ollama"
model = "nomic-embed-text"
url = "http://localhost:11434"  # optional, defaults to localhost
```

## Schema Changes

Add to `semantic_memories` table:
```sql
ALTER TABLE semantic_memories ADD COLUMN embedding BLOB;
```

Add metadata table for tracking embedding model:
```sql
CREATE TABLE IF NOT EXISTS memory_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
-- Store: ("embedding_model", "nomic-embed-text")
```

## New Module: `src/embedding.rs`

```rust
pub struct EmbeddingClient {
    url: String,
    model: String,
}

impl EmbeddingClient {
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;
    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError>;
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32;
```

Ollama API: `POST /api/embeddings { "model": "...", "prompt": "..." }`

## Changes to `src/memory.rs`

**SemanticMemoryStore:**
- `save()` — Accept optional embedding, store in BLOB column
- `recall()` — If embeddings enabled, load all embeddings, compute cosine similarity, rank and return top N
- `backfill_embeddings()` — For migration: find rows with NULL embedding, compute and update
- `get_embedding_model()` / `set_embedding_model()` — Track current model in metadata
- Remove keyword-based recall code path entirely

## Changes to `src/daemon.rs`

On startup (if semantic_memory.embedding configured):
1. Create `EmbeddingClient` from config
2. Check `memory_meta` for stored model vs config model
3. If mismatch or NULL embeddings exist → backfill all
4. Store client in daemon state

On message handling:
- Before save: `let embedding = client.embed(&memory_text).await?`
- Pass embedding to `store.save()`
- On recall: `let query_embedding = client.embed(&query).await?`
- Pass to `store.recall()`

## Changes to `src/agent_dir.rs`

Add config structs:
```rust
pub struct EmbeddingConfig {
    pub provider: String,  // "ollama" for now
    pub model: String,
    pub url: Option<String>,
}
```

Parse from `[semantic_memory.embedding]` section.

## Migration Flow

```
Daemon starts
→ semantic_memory.embedding configured?
  → No: semantic memory disabled, skip
  → Yes: 
    → Check memory_meta.embedding_model
    → If different from config OR any NULL embeddings:
      → Log "Backfilling embeddings..."
      → For each memory: embed(content), UPDATE embedding
      → Update memory_meta.embedding_model
    → Continue with embedded recall
```

## Files to Modify

| File | Changes |
|------|---------|
| `src/embedding.rs` | **NEW** — EmbeddingClient, cosine_similarity |
| `src/memory.rs` | Add embedding column handling, remove keyword recall |
| `src/daemon.rs` | Integrate embedding client, backfill logic |
| `src/agent_dir.rs` | Parse EmbeddingConfig |
| `src/lib.rs` | Export new module |

## Checklist

- [x] Create `src/embedding.rs` with EmbeddingClient and cosine_similarity
- [x] Update `src/agent_dir.rs` with EmbeddingConfig parsing
- [x] Update `src/memory.rs` schema and recall logic
- [x] Update `src/daemon.rs` to integrate embeddings
- [x] Update `src/lib.rs` exports
- [x] Remove keyword-based recall code
- [x] Add tests for embedding module
- [x] Verify build passes
- [x] Verify tests pass
