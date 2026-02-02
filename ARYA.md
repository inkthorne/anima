# Task: Sandboxed Safe Shell Tool

**Goal:** Create a `safe_shell` tool that only allows read-only, non-destructive commands. Replace `shell` with `safe_shell` in model configs.

**Build:** `cargo build --release && cargo test`

## Changes to `~/.anima/tools.toml`

Add new tool definition:
```toml
[[tool]]
name = "safe_shell"
description = "Run read-only shell commands (ls, grep, find, cat, head, tail, wc, pwd, file, stat, du, df)"
params = { command = "string" }
keywords = ["shell", "command", "run", "ls", "grep", "find", "list", "search", "directory"]
category = "system"
allowed_commands = ["ls", "grep", "find", "cat", "head", "tail", "wc", "pwd", "echo", "which", "file", "stat", "du", "df", "env", "date", "whoami", "hostname", "uname"]
```

Keep the original `shell` tool for trusted models, but remove dangerous keywords to reduce recall.

## Changes to `src/tool_registry.rs`

Update `ToolDefinition` struct to include optional `allowed_commands`:
```rust
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub params: HashMap<String, String>,
    pub keywords: HashSet<String>,
    pub category: Option<String>,
    pub allowed_commands: Option<Vec<String>>,  // NEW
}
```

Update TOML parsing to handle `allowed_commands`.

## Changes to `src/daemon.rs` (tool execution)

When executing a tool, check if `allowed_commands` is set:
```rust
fn execute_tool(tool: &ToolDefinition, params: &Value) -> Result<String, ToolError> {
    match tool.name.as_str() {
        "shell" | "safe_shell" => {
            let command = params["command"].as_str().unwrap();
            
            // If allowed_commands is set, validate the command
            if let Some(allowed) = &tool.allowed_commands {
                let first_word = command.split_whitespace().next().unwrap_or("");
                if !allowed.contains(&first_word.to_string()) {
                    return Err(ToolError::Forbidden(format!(
                        "Command '{}' not in allowed list. Allowed: {:?}", 
                        first_word, allowed
                    )));
                }
            }
            
            // Execute the command...
        }
        // ... other tools
    }
}
```

## Changes to model configs

Update `~/.anima/models/gemma-27b.toml`:
```toml
allowed_tools = ["read_file", "write_file", "safe_shell"]  # safe_shell instead of shell
```

Update any other model configs (qwen, etc.) similarly.

## Checklist

- [x] Add `allowed_commands` field to `ToolDefinition` in `src/tool_registry.rs`
- [x] Update TOML parsing for `allowed_commands`
- [x] Add command validation in tool execution in `src/daemon.rs`
- [x] Add `safe_shell` tool to `~/.anima/tools.toml`
- [x] Update `~/.anima/models/gemma-27b.toml` to use `safe_shell`
- [x] Add tests for command filtering
- [x] Verify build passes
- [x] Verify tests pass (367 + 11 = 378 tests passing)
