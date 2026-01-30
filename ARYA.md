# Task: v0.9 Persistent Memory (SQLite)

Build: `cargo build`

## Steps
- [x] 1. [Cargo.toml] Add rusqlite dependency
- [x] 2. [memory.rs] Create SqliteMemory struct with connection
- [x] 3. [memory.rs] Implement open() and schema creation  
- [x] 4. [memory.rs] Implement Memory trait (get/set/delete/list_keys)
- [x] 5. [memory.rs] Add query_by_time() for episodic memory
- [x] 6. [lib.rs] Update exports
- [x] 7. [main.rs] Write demo
- [x] 8. [memory.rs] Fix i64/u64 conversion for SQLite compatibility

## Status: COMPLETE ✅

Demo output confirms:
- Persistence across sessions
- Agent isolation (different agents, different memories)
- Episodic queries (query_by_time works)

Ready for review.
