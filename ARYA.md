# Task: Implement anima v0.1 core skeleton

Build: `cargo build`

## Steps

- [x] 1. [Cargo.toml] Add serde and serde_json dependencies
- [x] 2. [src/lib.rs] Create library root with module declarations
- [x] 3. [src/error.rs] Define ToolError enum (wrote manually - Qwen hallucinated)
- [x] 4. [src/tool.rs] Define Tool trait
- [x] 5. [src/agent.rs] Define Agent struct with tool registry (Qwen did it, but broke lib.rs - fixed)
- [x] 6. [src/runtime.rs] Define Runtime struct with agent management
- [ ] 7. [src/tools/mod.rs] Create tools module with re-exports (Qwen made wrong stub)
- [ ] 8. [src/tools/echo.rs] Implement Echo tool
- [ ] 9. [src/tools/add.rs] Implement Add tool
- [ ] 10. [src/main.rs] Demo: create runtime, spawn agent, register tools, call them

## Plan

### Architecture

```
src/
├── lib.rs          # pub mod declarations, re-exports
├── error.rs        # ToolError enum
├── tool.rs         # Tool trait
├── agent.rs        # Agent struct
├── runtime.rs      # Runtime struct
├── tools/
│   ├── mod.rs      # pub mod echo; pub mod add;
│   ├── echo.rs     # Echo tool
│   └── add.rs      # Add tool
└── main.rs         # Demo binary
```

### Key Types

```rust
// error.rs
pub enum ToolError {
    InvalidInput(String),
    ExecutionFailed(String),
}

// tool.rs
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn execute(&self, input: Value) -> Result<Value, ToolError>;
}

// agent.rs
pub struct Agent {
    pub id: String,
    tools: HashMap<String, Box<dyn Tool>>,
}

// runtime.rs
pub struct Runtime {
    agents: HashMap<String, Agent>,
}
```

### Dependencies

```toml
[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```
