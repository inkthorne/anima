# Streaming Response Spec

## Problem

Currently, `anima send` and `anima chat` block silently while the agent processes. Users see nothing during:
- LLM generation (30-60+ seconds with Qwen)
- Tool execution loops
- Multi-turn agentic reasoning

This feels broken and may cause socket timeouts.

## Goal

Stream responses to the client in real-time:
- LLM tokens appear as they're generated
- Tool calls are announced as they happen
- Client stays alive (no idle timeout)

## Current Protocol

```
Client                    Daemon
  |                         |
  |-- Request::Message -->  |
  |                         | (silence for 30-60s)
  |<-- Response::Message -- |
  |                         |
```

## Proposed Protocol

```
Client                    Daemon
  |                         |
  |-- Request::Message -->  |
  |                         |
  |<-- Response::ToolStart -| (tool about to run)
  |<-- Response::ToolEnd ---|  (tool finished)
  |<-- Response::Chunk -----| (LLM tokens)
  |<-- Response::Chunk -----|
  |<-- Response::ToolStart -| (another tool)
  |<-- Response::ToolEnd ---|
  |<-- Response::Chunk -----|
  |<-- Response::Done ------| (stream complete)
  |                         |
```

## New Response Variants

```rust
pub enum Response {
    // Existing
    Message { content: String },
    Status { running: bool, history_len: usize },
    Agents { agents: Vec<String> },
    System { persona: String },
    Ok,
    Error { message: String },
    
    // New: Streaming
    Chunk { text: String },
    ToolStart { tool: String, params: serde_json::Value },
    ToolEnd { tool: String, success: bool, summary: Option<String> },
    Done { },
}
```

## Client Behavior

```rust
// In send_message() / chat loop
loop {
    match api.read_response().await? {
        Response::Chunk { text } => {
            print!("{}", text);  // No newline, tokens stream
            stdout().flush()?;
        }
        Response::ToolStart { tool, params } => {
            // Print on its own line, dimmed
            eprintln!("\x1b[2m-> {} | {}\x1b[0m", tool, format_params(&params));
        }
        Response::ToolEnd { tool, success, summary } => {
            // Optional: show result summary
            if let Some(s) = summary {
                eprintln!("\x1b[2m   <- {}\x1b[0m", truncate(&s, 60));
            }
        }
        Response::Done { } => {
            println!();  // Final newline
            break;
        }
        Response::Error { message } => {
            eprintln!("Error: {}", message);
            break;
        }
        _ => {}
    }
}
```

## Example Output

```
$ anima send arya "what do you want to work on?"
-> safe_shell | {"command": "ls -la ~/dev/anima"}
   <- total 120\ndrwxrwxr-x 8 chrip...
-> safe_shell | {"command": "cat ARYA.md"}
   <- # Task: Runtime Context...
I've been reviewing the project. I think we should focus on...
(tokens stream in real-time here)
...what do you think?
$
```

## Implementation Changes

### 1. socket_api.rs
- Add `Chunk`, `ToolStart`, `ToolEnd`, `Done` to Response enum

### 2. daemon.rs - Streaming Infrastructure

Replace `think_with_options` with `think_streaming_with_options`:

```rust
// Create channel for streaming events
let (token_tx, mut token_rx) = mpsc::channel::<StreamEvent>(100);

// Spawn the thinking task
let think_handle = tokio::spawn(async move {
    agent.think_streaming_with_options(&content, options, token_tx).await
});

// Forward events to socket as they arrive
while let Some(event) = token_rx.recv().await {
    match event {
        StreamEvent::Token(text) => {
            api.write_response(&Response::Chunk { text }).await?;
        }
        StreamEvent::ToolStart { tool, params } => {
            api.write_response(&Response::ToolStart { tool, params }).await?;
        }
        StreamEvent::ToolEnd { tool, success, summary } => {
            api.write_response(&Response::ToolEnd { tool, success, summary }).await?;
        }
    }
}

// Wait for completion
let result = think_handle.await??;
api.write_response(&Response::Done {}).await?;
```

### 3. agent.rs - Tool Event Callbacks

The streaming path needs to emit tool events. Currently `think_streaming_with_options_inner` has the token channel but doesn't emit tool events.

Add tool events to the channel:

```rust
pub enum StreamEvent {
    Token(String),
    ToolStart { tool: String, params: serde_json::Value },
    ToolEnd { tool: String, success: bool, summary: Option<String> },
}
```

In the tool execution loop:
```rust
// Before tool execution
token_tx.send(StreamEvent::ToolStart { 
    tool: tool_call.name.clone(),
    params: tool_call.arguments.clone(),
}).await;

// After tool execution
token_tx.send(StreamEvent::ToolEnd {
    tool: tool_call.name.clone(),
    success: result.is_ok(),
    summary: Some(truncate_result(&result)),
}).await;
```

### 4. main.rs - Client Updates

- `send_message()` - Handle streaming responses
- `chat()` loop - Handle streaming responses
- Add helper for formatting tool params

## Backward Compatibility

The non-streaming `Response::Message` can remain for:
- `Request::Status`
- `Request::System`
- `Request::ListAgents`
- Simple cases where streaming isn't needed

Only `Request::Message` and `Request::IncomingMessage` use the streaming path.

## Open Questions

1. **JSON-block tool mode**: Currently JSON-block tools are parsed by the daemon, not the agent. Need to emit tool events from daemon.rs for this path too.

2. **Native tool mode**: Tool events come from agent.rs. May need different integration points.

3. **Timer triggers**: Should timer-triggered thinking also stream? Probably not critical.

4. **Memory injection**: Should show "recalling memories..." event? Nice to have but not essential.

## Phases

**Phase 1**: LLM token streaming only (get the infrastructure in place)
**Phase 2**: Tool events (ToolStart/ToolEnd)
**Phase 3**: Polish (colors, formatting, memory events)
